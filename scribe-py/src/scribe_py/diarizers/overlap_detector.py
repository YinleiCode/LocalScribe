"""Optional overlapped-speech detection using Senko's segmentation model.

Senko 0.1.0 uses pyannote segmentation 3.0 only as a binary VAD and discards
the model's powerset output.  The same output also contains three two-speaker
classes.  This module preserves those frame probabilities and turns them into
overlap intervals without changing the existing diarization path.

All model imports are lazy.  ``detect_overlaps`` is deliberately fail-open:
missing models, unsupported audio, or inference errors produce an unavailable
empty result so callers can keep the current diarization unchanged.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_RATE = 16_000
CHUNK_DURATION = 10.0
# A 10-second segmentation window only needs 50% overlap for stable boundary
# coverage. The former 1-second step repeated almost identical CoreML work ten
# times and made diarization look like ASR had slowed down.
CHUNK_STEP = 5.0
FRAME_START = 0.0
FRAME_DURATION = 0.0619375
FRAME_STEP = 0.016875
POWERSET_CLASS_CARDINALITIES = np.asarray((0, 1, 1, 1, 2, 2, 2), dtype=np.int8)

__all__ = [
    "detect_overlaps",
    "filter_contaminated_windows",
    "map_overlap_to_segments",
    "overlap_ratio_for_window",
    "partition_contaminated_windows",
    "scores_to_overlap_intervals",
]


def _empty_result(error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "backend": "none",
        "overlap_intervals": [],
        "frame_confidence": {
            "frame_start": FRAME_START,
            "frame_duration": FRAME_DURATION,
            "frame_step": FRAME_STEP,
            "scores": [],
        },
        "stats": {"overlap_seconds": 0.0, "interval_count": 0},
    }
    if error:
        result["error"] = error
    return result


def _load_audio(audio_source: Any) -> np.ndarray:
    if isinstance(audio_source, np.ndarray):
        samples = audio_source
    else:
        import soundfile as sf

        samples, sample_rate = sf.read(
            str(Path(audio_source).expanduser()),
            dtype="float32",
            always_2d=False,
        )
        if int(sample_rate) != SAMPLE_RATE:
            raise ValueError(f"overlap detector expects 16kHz audio, got {sample_rate}Hz")

    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError("overlap detector expects mono audio")
    if not np.issubdtype(samples.dtype, np.floating):
        if np.issubdtype(samples.dtype, np.signedinteger):
            scale = float(1 << (samples.dtype.itemsize * 8 - 1))
            samples = samples.astype(np.float32) / scale
        elif samples.dtype == np.uint8:
            samples = (samples.astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported audio dtype: {samples.dtype}")
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if not np.all(np.isfinite(samples)):
        raise ValueError("audio samples must be finite")
    return samples


def _chunk_starts(num_samples: int, window_size: int, step_size: int) -> list[int]:
    if num_samples <= window_size:
        return [0]
    starts = list(range(0, num_samples - window_size + 1, step_size))
    last_start = num_samples - window_size
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _powerset_overlap_probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"expected 2-D segmentation output, got shape {values.shape}")
    if values.shape[1] != len(POWERSET_CLASS_CARDINALITIES):
        raise ValueError(
            "unsupported segmentation powerset output: "
            f"expected {len(POWERSET_CLASS_CARDINALITIES)} classes, got {values.shape[1]}"
        )

    # Works for logits, log-softmax output, and already normalized probabilities.
    if np.all(values >= 0.0) and np.allclose(values.sum(axis=1), 1.0, atol=1e-3):
        probabilities = values
    else:
        shifted = values - np.max(values, axis=1, keepdims=True)
        exp_values = np.exp(np.clip(shifted, -80.0, 0.0))
        probabilities = exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)
    return probabilities[:, POWERSET_CLASS_CARDINALITIES >= 2].sum(axis=1).astype(np.float32)


def _aggregate_chunk_scores(
    chunk_scores: Sequence[np.ndarray],
    chunk_starts: Sequence[int],
    total_samples: int,
) -> np.ndarray:
    if len(chunk_scores) != len(chunk_starts):
        raise ValueError("chunk scores and chunk starts must have the same length")
    total_duration = total_samples / SAMPLE_RATE
    num_frames = max(0, int(math.floor(total_duration / FRAME_STEP)) + 1)
    if num_frames == 0:
        return np.empty((0,), dtype=np.float32)

    sums = np.zeros(num_frames, dtype=np.float64)
    weights = np.zeros(num_frames, dtype=np.float64)
    for raw_scores, start_sample in zip(chunk_scores, chunk_starts):
        scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
        if scores.size == 0:
            continue
        start_frame = int(np.rint((start_sample / SAMPLE_RATE) / FRAME_STEP))
        end_frame = min(num_frames, start_frame + scores.size)
        if end_frame <= start_frame:
            continue
        local = scores[: end_frame - start_frame]
        window = np.hamming(scores.size)[: local.size]
        target = slice(start_frame, end_frame)
        sums[target] += local * window
        weights[target] += window

    aggregated = np.divide(sums, weights, out=np.zeros_like(sums), where=weights > 1e-12)
    return np.clip(aggregated, 0.0, 1.0).astype(np.float32)


def _scores_from_coreml(
    samples: np.ndarray,
    model_path: Path,
    *,
    model: Any = None,
) -> np.ndarray:
    if model is None:
        import coremltools as ct

        compiled_model = getattr(ct.models, "CompiledMLModel", None)
        if compiled_model is None:
            raise RuntimeError("coremltools does not provide CompiledMLModel")
        model = compiled_model(str(model_path))

    window_size = int(round(CHUNK_DURATION * SAMPLE_RATE))
    step_size = int(round(CHUNK_STEP * SAMPLE_RATE))
    starts = _chunk_starts(samples.size, window_size, step_size)
    outputs: list[np.ndarray] = []
    for start in starts:
        chunk = samples[start : start + window_size]
        if chunk.size < window_size:
            chunk = np.pad(chunk, (0, window_size - chunk.size))
        prediction = model.predict({"audio": chunk.reshape(1, 1, -1).astype(np.float32, copy=False)})
        raw = prediction.get("segments")
        if raw is None:
            if len(prediction) != 1:
                raise ValueError(f"cannot identify segmentation output: {list(prediction)}")
            raw = next(iter(prediction.values()))
        outputs.append(_powerset_overlap_probabilities(np.asarray(raw)))
    return _aggregate_chunk_scores(outputs, starts, samples.size)


def _scores_from_torch_backend(samples: np.ndarray, backend: Any) -> np.ndarray:
    required = ("torch", "model", "mapping", "_iter_chunks", "device")
    if not all(hasattr(backend, name) for name in required):
        raise TypeError("Senko VAD backend does not expose frame-level segmentation")

    chunks: list[np.ndarray] = []
    with backend.torch.inference_mode():
        for batch in backend._iter_chunks(samples):
            logits = backend.model(batch.to(backend.device))
            probabilities = backend.torch.softmax(logits, dim=-1)
            cardinalities = backend.mapping.sum(dim=-1)
            overlap = probabilities[..., cardinalities >= 2].sum(dim=-1)
            chunks.extend(overlap.detach().cpu().numpy())

    window_size = int(round(float(backend.chunk_duration) * SAMPLE_RATE))
    step_size = int(round(float(backend.chunk_step) * SAMPLE_RATE))
    starts = _chunk_starts(samples.size, window_size, step_size)
    return _aggregate_chunk_scores(chunks, starts, samples.size)


def _resolve_coreml_model_path(senko_diarizer: Any = None, model_path: str | Path | None = None) -> Path:
    if model_path is not None:
        path = Path(model_path).expanduser()
    else:
        paths = getattr(senko_diarizer, "model_paths", None)
        path = Path(getattr(paths, "pyannote_segmentation_coreml_model_path", ""))
        if not str(path) or str(path) == ".":
            from senko import config

            paths = config.resolve_model_paths(
                required_fields=("pyannote_segmentation_coreml_model_path",),
            )
            path = Path(paths.pyannote_segmentation_coreml_model_path)
    if not path.exists():
        raise FileNotFoundError(f"bundled Senko segmentation model not found: {path}")
    return path


def scores_to_overlap_intervals(
    scores: Sequence[float] | np.ndarray,
    *,
    onset: float = 0.55,
    offset: float = 0.45,
    min_duration: float = 0.12,
    min_gap: float = 0.08,
    total_duration: float | None = None,
) -> list[dict[str, float]]:
    """Convert frame overlap confidence to hysteresis-smoothed intervals."""
    values = np.clip(np.asarray(scores, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if values.size == 0:
        return []
    if not 0.0 <= offset <= onset <= 1.0:
        raise ValueError("expected 0 <= offset <= onset <= 1")

    centers = FRAME_START + 0.5 * FRAME_DURATION + np.arange(values.size) * FRAME_STEP
    active_runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, score in enumerate(values):
        if run_start is None and score >= onset:
            run_start = index
        elif run_start is not None and score < offset:
            active_runs.append((run_start, index - 1))
            run_start = None
    if run_start is not None:
        active_runs.append((run_start, values.size - 1))

    raw: list[tuple[float, float]] = []
    for first, last in active_runs:
        start = max(0.0, float(centers[first] - 0.5 * FRAME_DURATION))
        end = float(centers[last] + 0.5 * FRAME_DURATION)
        if total_duration is not None:
            end = min(float(total_duration), end)
        if end - start >= min_duration:
            raw.append((start, end))

    merged: list[tuple[float, float]] = []
    for start, end in raw:
        if merged and start - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    intervals: list[dict[str, float]] = []
    for start, end in merged:
        mask = (centers + 0.5 * FRAME_DURATION > start) & (centers - 0.5 * FRAME_DURATION < end)
        selected = values[mask]
        intervals.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "confidence": round(float(selected.mean()) if selected.size else 0.0, 4),
            "max_confidence": round(float(selected.max()) if selected.size else 0.0, 4),
        })
    return intervals


def detect_overlaps(
    audio_source: Any,
    *,
    senko_diarizer: Any = None,
    model_path: str | Path | None = None,
    coreml_model: Any = None,
    onset: float = 0.55,
    offset: float = 0.45,
    min_duration: float = 0.12,
    min_gap: float = 0.08,
) -> dict[str, Any]:
    """Detect true simultaneous speech, safely degrading to an empty result."""
    try:
        samples = _load_audio(audio_source)
        if samples.size == 0:
            result = _empty_result()
            result.update({"available": True, "backend": "empty_audio"})
            return result

        backend = getattr(senko_diarizer, "vad_backend", None)
        if backend is not None and all(
            hasattr(backend, name) for name in ("torch", "model", "mapping", "_iter_chunks", "device")
        ):
            scores = _scores_from_torch_backend(samples, backend)
            backend_name = "senko_segmentation_torch"
        else:
            resolved_path = _resolve_coreml_model_path(senko_diarizer, model_path)
            scores = _scores_from_coreml(samples, resolved_path, model=coreml_model)
            backend_name = "senko_segmentation_coreml"

        total_duration = samples.size / SAMPLE_RATE
        intervals = scores_to_overlap_intervals(
            scores,
            onset=onset,
            offset=offset,
            min_duration=min_duration,
            min_gap=min_gap,
            total_duration=total_duration,
        )
        overlap_seconds = _union_duration(intervals)
        return {
            "available": True,
            "backend": backend_name,
            "overlap_intervals": intervals,
            "frame_confidence": {
                "frame_start": FRAME_START,
                "frame_duration": FRAME_DURATION,
                "frame_step": FRAME_STEP,
                "scores": [round(float(score), 5) for score in scores],
            },
            "stats": {
                "model": "pyannote/segmentation-3.0 (Senko bundled)",
                "audio_seconds": round(total_duration, 3),
                "overlap_seconds": round(overlap_seconds, 3),
                "overlap_ratio": round(overlap_seconds / max(total_duration, 1e-9), 5),
                "interval_count": len(intervals),
                "frame_count": int(scores.size),
                "onset": onset,
                "offset": offset,
                "chunk_seconds": CHUNK_DURATION,
                "chunk_step_seconds": CHUNK_STEP,
            },
        }
    except Exception as exc:
        return _empty_result(f"{type(exc).__name__}: {exc}")


def _bounds(item: Any) -> tuple[float, float]:
    if isinstance(item, Mapping):
        return float(item.get("start", 0.0)), float(item.get("end", item.get("start", 0.0)))
    if hasattr(item, "start") and hasattr(item, "end"):
        return float(item.start), float(item.end)
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
        return float(item[0]), float(item[1])
    raise TypeError(f"cannot read time bounds from {type(item).__name__}")


def _merged_intersections(
    start: float,
    end: float,
    intervals: Sequence[Any],
    *,
    padding: float = 0.0,
) -> list[tuple[float, float]]:
    intersections: list[tuple[float, float]] = []
    for interval in intervals:
        overlap_start, overlap_end = _bounds(interval)
        overlap_start -= max(0.0, padding)
        overlap_end += max(0.0, padding)
        left = max(start, overlap_start)
        right = min(end, overlap_end)
        if right > left:
            intersections.append((left, right))
    intersections.sort()
    merged: list[tuple[float, float]] = []
    for left, right in intersections:
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def overlap_ratio_for_window(window: Any, overlap_intervals: Sequence[Any], *, padding: float = 0.0) -> float:
    """Return union overlap duration divided by window duration."""
    start, end = _bounds(window)
    duration = max(0.0, end - start)
    if duration == 0.0:
        return 0.0
    overlap = sum(right - left for left, right in _merged_intersections(
        start,
        end,
        overlap_intervals,
        padding=padding,
    ))
    return min(1.0, max(0.0, overlap / duration))


def _union_duration(intervals: Sequence[Any]) -> float:
    if not intervals:
        return 0.0
    starts = [start for start, _ in map(_bounds, intervals)]
    ends = [end for _, end in map(_bounds, intervals)]
    return sum(right - left for left, right in _merged_intersections(
        min(starts),
        max(ends),
        intervals,
    ))


def map_overlap_to_segments(
    segments: Sequence[Mapping[str, Any]],
    overlap_intervals: Sequence[Any],
    *,
    risk_ratio: float = 0.08,
    min_overlap_seconds: float = 0.12,
) -> list[dict[str, Any]]:
    """Copy segments and add overlap ratio/risk without changing text or timing."""
    mapped: list[dict[str, Any]] = []
    for segment in segments:
        item = copy.deepcopy(dict(segment))
        start, end = _bounds(item)
        duration = max(0.0, end - start)
        intersections = _merged_intersections(start, end, overlap_intervals)
        overlap_seconds = sum(right - left for left, right in intersections)
        ratio = overlap_seconds / duration if duration > 0.0 else 0.0
        risk = overlap_seconds >= min_overlap_seconds and ratio >= risk_ratio
        item["overlap_ratio"] = round(min(1.0, max(0.0, ratio)), 4)
        item["speaker_overlap_risk"] = bool(item.get("speaker_overlap_risk")) or risk
        mapped.append(item)
    return mapped


def filter_contaminated_windows(
    windows: Sequence[Any],
    overlap_intervals: Sequence[Any],
    *,
    max_overlap_ratio: float = 0.02,
    padding: float = 0.08,
) -> list[Any]:
    """Drop overlap-contaminated embedding/enrollment windows, preserving type/order."""
    if not 0.0 <= max_overlap_ratio <= 1.0:
        raise ValueError("max_overlap_ratio must be between 0 and 1")
    return [
        window
        for window in windows
        if overlap_ratio_for_window(window, overlap_intervals, padding=padding) <= max_overlap_ratio
    ]


def partition_contaminated_windows(
    windows: Sequence[Any],
    overlap_intervals: Sequence[Any],
    *,
    max_overlap_ratio: float = 0.02,
    padding: float = 0.08,
) -> tuple[list[Any], list[Any]]:
    """Return (clean, rejected) windows for audit-friendly integration."""
    clean: list[Any] = []
    rejected: list[Any] = []
    for window in windows:
        ratio = overlap_ratio_for_window(window, overlap_intervals, padding=padding)
        target = clean if ratio <= max_overlap_ratio else rejected
        target.append(window)
    return clean, rejected
