"""FunASR/SenseVoice transcriber.

This backend is used as the Chinese-first ASR path.  In packaged builds the
ModelScope cache is bundled under Resources/modelscope/hub so the App can run
the same backend as the command-line benchmark instead of falling back to MLX.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock
from typing import Any

from .hotwords import build_hotword_string
from .text_normalizer import normalize_segments
from .sensevoice_recovery import (
    analyze_recovery_candidate,
    decide_recovery_attempts,
    group_failure_windows,
    local_reference_from_segments,
    normalize_recovery_text,
)
from .transcriber_base import ProgressCallback, Transcriber
from .types import Segment, TranscribeOptions

DEFAULT_MODEL = "paraformer-zh"
SENSEVOICE_MODEL = "iic/SenseVoiceSmall"
FSMN_VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
PARAFORMER_MODELSCOPE_REPO = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
QWEN3_RECOVERY_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
_MODELSCOPE_ALIASES = {
    DEFAULT_MODEL: PARAFORMER_MODELSCOPE_REPO,
    "fsmn-vad": FSMN_VAD_MODEL,
}
_FUNASR_RNG_LOCK = Lock()
_SILERO_VAD_LOCK = Lock()
_SILERO_VAD_MODEL: Any | None = None


def _cached_huggingface_snapshot(model_id: str) -> tuple[Path | None, dict[str, Any], str | None]:
    metadata: dict[str, Any] = {}
    try:
        from huggingface_hub import try_to_load_from_cache

        config_path = try_to_load_from_cache(model_id, "config.json")
        if not isinstance(config_path, str):
            return None, metadata, "model_config_not_cached"
        config = Path(config_path)
        snapshot = config.parent
        weight_files = sorted(snapshot.glob("*.safetensors"))
        index_path = snapshot / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            expected_weights = sorted(set((index.get("weight_map") or {}).values()))
            if not expected_weights or any(not (snapshot / name).is_file() for name in expected_weights):
                return None, metadata, "model_weight_index_incomplete"
            weight_files = [snapshot / name for name in expected_weights]
        if not weight_files or any(path.stat().st_size <= 0 for path in weight_files):
            return None, metadata, "model_weights_not_cached"
        manifest = [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "blob": path.resolve().name if path.is_symlink() else hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in weight_files
        ]
        metadata = {
            "model_revision": snapshot.name,
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "weights_manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "weight_files": len(weight_files),
        }
        return snapshot, metadata, None
    except Exception as exc:
        return None, metadata, f"model_snapshot_invalid:{type(exc).__name__}"


def _is_sensevoice(model_id: str) -> bool:
    return "sensevoice" in model_id.lower()


def _modelscope_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for name in ("LOCALSCRIBE_MODELSCOPE_CACHE", "MODELSCOPE_CACHE"):
        raw = os.environ.get(name)
        if raw:
            roots.append(Path(raw).expanduser())
    roots.append(Path.home() / ".cache" / "modelscope" / "hub")

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _modelscope_repo_id(model_id: str) -> str:
    return _MODELSCOPE_ALIASES.get(model_id, model_id)


def modelscope_cache_candidates(model_id: str) -> list[Path]:
    """Return likely local ModelScope cache dirs for a FunASR model id."""
    repo = _modelscope_repo_id(model_id)
    parts = [p for p in repo.split("/") if p]
    if not parts:
        return []
    suffix = Path(*parts)
    candidates: list[Path] = []
    for root in _modelscope_cache_roots():
        candidates.append(root / "models" / suffix)
        if root.name == "models":
            candidates.append(root / suffix)
    return candidates


def _model_dir_is_usable(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.yaml").exists() or (path / "configuration.json").exists()
    return has_config and (path / "model.pt").exists()


def resolve_model_path(model_id: str) -> str:
    """Resolve a ModelScope id or FunASR alias to a bundled/local model path."""
    if Path(model_id).expanduser().exists():
        return model_id
    for candidate in modelscope_cache_candidates(model_id):
        if _model_dir_is_usable(candidate):
            return str(candidate)
    return model_id


def model_cached(model_id: str) -> bool:
    return any(_model_dir_is_usable(path) for path in modelscope_cache_candidates(model_id))


def _clean_text(text: str, *, sensevoice: bool = False) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if sensevoice:
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess

            text = rich_transcription_postprocess(text)
        except Exception:
            # Keep the raw text if the postprocess helper is unavailable.  This
            # happens with some FunASR/SenseVoice package combinations.
            pass
    # Paraformer without the optional punctuation model may return character
    # separated Chinese text, e.g. "我 们 的 产 品".  Remove only CJK-to-CJK
    # spacing so English words like "OK ok" remain readable.
    cjk = r"\u3400-\u4dbf\u4e00-\u9fff"
    text = re.sub(fr"([{cjk}])\s+(?=[{cjk}])", r"\1", text)
    text = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", text)
    text = re.sub(r"([，。！？；：、])\s+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw not in {"0", "false", "False", "no", "NO"}


def _sensevoice_local_recovery_mode() -> tuple[str, str, str | None]:
    requested = (os.environ.get("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE") or "off").strip().lower()
    if requested in {"off", "audit", "merge"}:
        return requested, requested, None
    return "off", requested, f"invalid_mode:{requested or '<empty>'};fallback_off"


def _sensevoice_local_recovery_provider() -> tuple[str, str | None]:
    provider = (os.environ.get("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_PROVIDER") or "off").strip().lower()
    if provider in {"off", "qwen3"}:
        return provider, None
    return "off", f"invalid_provider:{provider or '<empty>'};fallback_off"


def _segments_text_sha256(segments: list[Segment]) -> str:
    compact = re.sub(r"\s+", "", "".join(segment.text or "" for segment in segments))
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def _recovery_snapshot(
    segments: list[Segment],
    attempted_ranges: list[tuple[float, float]],
    recognized_ranges: list[tuple[float, float]],
    failed_ranges: list[tuple[float, float]],
) -> dict[str, Any]:
    attempted = sorted([round(start, 3), round(end, 3)] for start, end in attempted_ranges)
    recognized = sorted([round(start, 3), round(end, 3)] for start, end in recognized_ranges)
    failed = sorted([round(start, 3), round(end, 3)] for start, end in failed_ranges)

    def ranges_sha256(ranges: list[list[float]]) -> str:
        return hashlib.sha256(
            json.dumps(ranges, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    combined = sorted(recognized + failed)
    has_overlap = any(
        combined[index][0] < combined[index - 1][1] - 0.001
        for index in range(1, len(combined))
    )
    return {
        "segment_count": len(segments),
        "text_sha256": _segments_text_sha256(segments),
        "covered_count": len(recognized),
        "failed_count": len(failed),
        "attempted_ranges": attempted,
        "recognized_ranges": recognized,
        "failed_ranges": failed,
        "attempted_partition_sha256": ranges_sha256(attempted),
        "recognized_partition_sha256": ranges_sha256(recognized),
        "failed_partition_sha256": ranges_sha256(failed),
        "partition_valid": attempted == combined and not has_overlap,
    }


def _generate_quietly(model: Any, kwargs: dict[str, Any]) -> Any:
    """Run FunASR without leaking tqdm/progress noise into CLI JSON/App logs."""
    if _env_flag("LOCALSCRIBE_FUNASR_VERBOSE", False):
        return model.generate(**kwargs)
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        return model.generate(**kwargs)


def _generate_reproducibly(model: Any, kwargs: dict[str, Any]) -> Any:
    """Keep FunASR inference stable without leaking caller RNG state.

    FunASR's ``WavFrontend`` defaults to non-zero dither during inference. Its
    random noise can change CTC argmax results on otherwise identical audio.
    Some model utilities also use Python or NumPy randomness, so all three RNG
    families are scoped to the inference call and restored afterwards.
    """
    try:
        import torch
    except Exception:
        return _generate_quietly(model, kwargs)

    seed = _env_int("LOCALSCRIBE_FUNASR_INFERENCE_SEED", 0) & ((1 << 63) - 1)
    # Torch documents fork_rng as non-thread-safe. FunASR models are already
    # used serially by the App pipeline, and this lock also protects CLI users
    # that invoke multiple transcribers in one process.
    with _FUNASR_RNG_LOCK:
        python_state = random.getstate()
        numpy_module: Any | None = None
        numpy_state: Any | None = None
        try:
            import numpy as np

            numpy_module = np
            numpy_state = np.random.get_state()
        except Exception:
            pass

        try:
            random.seed(seed)
            if numpy_module is not None:
                numpy_module.random.seed(seed & 0xFFFFFFFF)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                return _generate_quietly(model, kwargs)
        finally:
            random.setstate(python_state)
            if numpy_module is not None and numpy_state is not None:
                numpy_module.random.set_state(numpy_state)


def _to_seconds(value: Any, *, assume_ms: bool = False) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    # FunASR sentence_info/timestamp values are milliseconds.  Keep auto-detect
    # for defensive parsing, but callers that read FunASR timestamps should pass
    # assume_ms=True so a 110 ms start does not become 110 seconds.
    if assume_ms or v > 1000:
        return v / 1000.0
    return v


def _segments_from_sentence_info(items: Any, *, sensevoice: bool) -> list[Segment]:
    if not isinstance(items, list):
        return []
    segments: list[Segment] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_text = item.get("text") or item.get("sentence") or item.get("raw_text") or ""
        text = _clean_text(str(raw_text), sensevoice=sensevoice)
        if not text:
            continue
        start = _to_seconds(item.get("start", 0.0), assume_ms=True)
        end = _to_seconds(item.get("end", start), assume_ms=True)
        speaker = item.get("spk")
        segments.append(
            Segment(
                start=start,
                end=max(end, start),
                text=text,
                speaker=f"SPEAKER_{speaker}" if speaker is not None else None,
            )
        )
    return segments


def _sentence_break_indexes(text: str, *, max_chars: int = 80) -> list[int]:
    indexes: list[int] = []
    last = 0
    for i, ch in enumerate(text, start=1):
        if ch in "。！？!?；;\n" or (i - last) >= max_chars:
            indexes.append(i)
            last = i
    if not indexes or indexes[-1] < len(text):
        indexes.append(len(text))
    return indexes


def _cue_break_indexes(text: str, *, max_chars: int = 12) -> list[int]:
    indexes: list[int] = []
    last = 0
    for i, ch in enumerate(text, start=1):
        span = i - last
        if (ch in "，。！？；：、,.!?;:" and span >= 3) or span >= max_chars:
            indexes.append(i)
            last = i
    if not indexes or indexes[-1] < len(text):
        indexes.append(len(text))
    return indexes


def _timestamp_pair(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and value:
        start = _to_seconds(value[0], assume_ms=True)
        end = _to_seconds(value[1] if len(value) > 1 else value[0], assume_ms=True)
    else:
        start = _to_seconds(value, assume_ms=True)
        end = start
    if end < start:
        end = start
    return start, end


def _char_time_pairs_from_timestamps(text: str, timestamps: list[Any]) -> list[tuple[float, float]]:
    pairs = [_timestamp_pair(ts) for ts in timestamps]
    if len(pairs) >= len(text):
        return pairs[:len(text)]

    timed_chars = sum(
        1 for ch in text if re.match(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]", ch)
    )
    if timed_chars == 0 or len(pairs) < timed_chars:
        return []

    out: list[tuple[float, float]] = []
    pair_idx = 0
    last_pair = pairs[0]
    for ch in text:
        if re.match(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]", ch):
            last_pair = pairs[pair_idx]
            pair_idx += 1
            out.append(last_pair)
        else:
            out.append((last_pair[1], last_pair[1]))
    return out


def _build_sync_cues_from_char_times(
    text: str,
    char_times: list[tuple[float, float]],
    *,
    segment_start: float,
    segment_end: float,
    max_chars: int = 12,
) -> list[dict[str, Any]]:
    if not text or len(char_times) < len(text):
        return []
    cues: list[dict[str, Any]] = []
    start_idx = 0
    break_indexes = set(_cue_break_indexes(text, max_chars=max_chars))
    previous_meaningful_index: int | None = None
    for index, char in enumerate(text):
        if not re.match(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]", char):
            continue
        if previous_meaningful_index is not None:
            previous_end = float(char_times[previous_meaningful_index][1])
            current_start = float(char_times[index][0])
            if current_start - previous_end >= 1.5:
                break_indexes.add(index)
        previous_meaningful_index = index
    for end_idx in sorted(index for index in break_indexes if 0 < index <= len(text)):
        cue_text = text[start_idx:end_idx].strip()
        if not cue_text:
            start_idx = end_idx
            continue
        indexed = [
            char_times[idx]
            for idx in range(start_idx, min(end_idx, len(char_times)))
            if idx < len(text) and re.match(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]", text[idx])
        ]
        if indexed:
            start = min(pair[0] for pair in indexed)
            end = max(pair[1] for pair in indexed)
        else:
            start, end = segment_start, segment_end
        start = max(segment_start, start)
        end = min(segment_end, max(end, start))
        cues.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": cue_text,
        })
        start_idx = end_idx

    if not cues:
        return []
    compacted: list[dict[str, Any]] = []
    for cue in cues:
        cue_text = str(cue["text"])
        has_text = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]", cue_text))
        if compacted and not has_text:
            compacted[-1]["text"] = f"{compacted[-1]['text']}{cue_text}"
            compacted[-1]["end"] = cue["end"]
            continue
        compacted.append(cue)
    cues = compacted
    if not cues:
        return []
    cues[0]["start"] = round(segment_start, 3)
    cues[-1]["end"] = round(max(segment_end, cues[-1]["end"]), 3)
    for idx in range(1, len(cues)):
        prev_end = float(cues[idx - 1]["end"])
        cur_start = float(cues[idx]["start"])
        if cur_start < prev_end:
            midpoint = round((cur_start + prev_end) / 2.0, 3)
            cues[idx - 1]["end"] = midpoint
            cues[idx]["start"] = midpoint
    return _compact_nonpositive_sync_cues(
        cues,
        segment_start=segment_start,
        segment_end=segment_end,
    )


def _compact_nonpositive_sync_cues(
    cues: list[dict[str, Any]],
    *,
    segment_start: float,
    segment_end: float,
) -> list[dict[str, Any]]:
    """Merge cues whose timing cannot support a trustworthy text boundary.

    A collapsed or overlapping boundary contains no acoustic evidence for a
    finer cursor split.  Keep the transcript text and segment geometry, but
    merge the affected text and mark the resulting cue as unreliable so the
    UI does not present fabricated precision.
    """
    compacted: list[dict[str, Any]] = []
    pending_prefix = ""
    pending_start: float | None = None
    for raw_cue in cues:
        cue = dict(raw_cue)
        cue_text = str(cue.get("text") or "")
        if not cue_text:
            continue
        start = max(segment_start, min(segment_end, float(cue.get("start", segment_start))))
        end = max(segment_start, min(segment_end, float(cue.get("end", start))))
        cue["start"] = round(start, 3)
        cue["end"] = round(end, 3)
        if float(cue["end"]) <= float(cue["start"]):
            if compacted:
                compacted[-1]["text"] = f"{compacted[-1]['text']}{cue_text}"
                compacted[-1]["reliable"] = False
            else:
                pending_prefix += cue_text
                pending_start = start if pending_start is None else min(pending_start, start)
            continue
        if pending_prefix:
            cue["text"] = f"{pending_prefix}{cue_text}"
            cue["start"] = round(min(float(cue["start"]), pending_start or segment_start), 3)
            cue["reliable"] = False
            pending_prefix = ""
            pending_start = None
        if compacted and float(cue["start"]) < float(compacted[-1]["end"]) - 0.001:
            compacted[-1]["text"] = f"{compacted[-1]['text']}{cue['text']}"
            compacted[-1]["end"] = round(
                max(float(compacted[-1]["end"]), float(cue["end"])),
                3,
            )
            compacted[-1]["reliable"] = False
            continue
        compacted.append(cue)

    if pending_prefix:
        if compacted:
            compacted[-1]["text"] = f"{compacted[-1]['text']}{pending_prefix}"
            compacted[-1]["reliable"] = False
        elif segment_end > segment_start:
            compacted.append({
                "start": round(segment_start, 3),
                "end": round(segment_end, 3),
                "text": pending_prefix,
                "reliable": False,
            })
    return [
        cue
        for cue in compacted
        if cue.get("text") and float(cue["end"]) > float(cue["start"])
    ]


def _repair_nonpositive_sync_cues_preserving_segments(
    segments: list[Segment],
) -> tuple[list[Segment], dict[str, Any]]:
    geometry_before = [(segment.start, segment.end, segment.text) for segment in segments]
    repaired: list[Segment] = []
    zero_before = 0
    zero_after = 0
    overlaps_before = 0
    overlaps_after = 0
    repaired_segments = 0

    def overlap_count(cues: list[dict[str, Any]]) -> int:
        count = 0
        previous_end: float | None = None
        for cue in cues:
            try:
                start = float(cue.get("start"))
                end = float(cue.get("end"))
            except (TypeError, ValueError):
                continue
            if previous_end is not None and start < previous_end - 0.001:
                count += 1
            previous_end = end
        return count

    for segment in segments:
        cues = list(segment.sync_cues or [])
        if not cues:
            repaired.append(segment)
            continue
        zero_before += sum(
            1
            for cue in cues
            if float(cue.get("end", segment.start)) <= float(cue.get("start", segment.start))
        )
        overlaps_before += overlap_count(cues)
        compacted = _compact_nonpositive_sync_cues(
            cues,
            segment_start=segment.start,
            segment_end=segment.end,
        )
        if compacted != cues:
            repaired_segments += 1
            segment = replace(segment, sync_cues=compacted or None)
        zero_after += sum(
            1
            for cue in compacted
            if float(cue.get("end", segment.start)) <= float(cue.get("start", segment.start))
        )
        overlaps_after += overlap_count(compacted)
        repaired.append(segment)

    geometry_preserved = geometry_before == [
        (segment.start, segment.end, segment.text) for segment in repaired
    ]
    if not geometry_preserved:
        raise RuntimeError("sync cue duration repair changed transcript geometry")
    return repaired, {
        "zero_duration_cues_before": zero_before,
        "zero_duration_cues_after": zero_after,
        "overlapping_cues_before": overlaps_before,
        "overlapping_cues_after": overlaps_after,
        "duration_repaired_segments": repaired_segments,
        "geometry_preserved": geometry_preserved,
    }


def _guard_unreliable_sync_cues(
    segments: list[Segment],
    speech_ranges: list[tuple[float, float]],
    *,
    vad_status: str,
) -> tuple[list[Segment], dict[str, Any]]:
    """Disable automatic highlighting where timing has no speech support."""
    geometry_before = [(segment.start, segment.end, segment.text, segment.speaker) for segment in segments]
    repaired, structural = _repair_nonpositive_sync_cues_preserving_segments(segments)
    stats: dict[str, Any] = {
        "enabled": vad_status == "ok",
        "vad_status": vad_status,
        "unreliable_cues": 0,
        "unreliable_segments": 0,
        "items": [],
        **structural,
    }
    if vad_status != "ok":
        stats["reason"] = "vad_evidence_unavailable"
        return repaired, stats

    speech = _normalize_intervals(speech_ranges)
    if not speech:
        stats["enabled"] = False
        stats["reason"] = "vad_detected_no_speech"
        return repaired, stats

    output: list[Segment] = []
    for segment_index, segment in enumerate(repaired):
        cues = list(segment.sync_cues or [])
        if not cues:
            output.append(segment)
            continue
        updated: list[dict[str, Any]] = []
        segment_changed = False
        for cue_index, raw_cue in enumerate(cues):
            cue = dict(raw_cue)
            try:
                start = float(cue.get("start"))
                end = float(cue.get("end"))
            except (TypeError, ValueError):
                updated.append(cue)
                continue
            duration = max(0.0, end - start)
            overlap_s = sum(
                max(0.0, min(end, speech_end) - max(start, speech_start))
                for speech_start, speech_end in speech
                if speech_end > start and speech_start < end
            )
            speech_ratio = overlap_s / duration if duration > 0 else 0.0
            text_chars = len(normalize_recovery_text(str(cue.get("text") or "")))
            chars_per_s = text_chars / duration if duration > 0 else float("inf")
            no_speech_support = duration >= 0.5 and overlap_s <= 0.08 and speech_ratio <= 0.08
            impossible_speed = duration >= 0.05 and text_chars >= 4 and chars_per_s > 12.0
            inherited_unreliable = cue.get("reliable") is False
            if inherited_unreliable or no_speech_support or impossible_speed:
                cue["reliable"] = False
                segment_changed = True
                stats["unreliable_cues"] += 1
                if len(stats["items"]) < 40:
                    reasons = []
                    if inherited_unreliable:
                        reasons.append("collapsed_or_overlapping_boundary")
                    if no_speech_support:
                        reasons.append("no_vad_speech_support")
                    if impossible_speed:
                        reasons.append("impossible_text_speed")
                    stats["items"].append({
                        "segment_index": segment_index,
                        "cue_index": cue_index,
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "speech_overlap_s": round(overlap_s, 3),
                        "speech_ratio": round(speech_ratio, 4),
                        "chars_per_s": round(chars_per_s, 3),
                        "reasons": reasons,
                    })
            updated.append(cue)
        if segment_changed:
            stats["unreliable_segments"] += 1
            segment = replace(segment, sync_cues=updated)
        output.append(segment)

    stats["reason"] = (
        "unreliable_cues_suppressed"
        if stats["unreliable_cues"]
        else "all_cues_have_timing_support"
    )
    stats["geometry_preserved"] = geometry_before == [
        (segment.start, segment.end, segment.text, segment.speaker)
        for segment in output
    ]
    if not stats["geometry_preserved"]:
        raise RuntimeError("sync cue reliability guard changed transcript geometry")
    return output, stats


def _char_time_pairs_from_points(
    text: str,
    points: list[float],
    *,
    segment_start: float,
    segment_end: float,
) -> list[tuple[float, float]]:
    normalized_count = sum(
        1 for ch in text if re.match(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]", ch)
    )
    if not text or normalized_count == 0 or len(points) < normalized_count:
        return []

    pairs_by_normalized: list[tuple[float, float]] = []
    for idx in range(normalized_count):
        if normalized_count == 1:
            start, end = segment_start, segment_end
        else:
            previous_gap = points[idx] - points[idx - 1] if idx > 0 else 0.0
            next_gap = points[idx + 1] - points[idx] if idx + 1 < normalized_count else 0.0
            start = (
                segment_start
                if idx == 0
                else (
                    (points[idx - 1] + points[idx]) / 2.0
                    if previous_gap < 1.5
                    else max(segment_start, points[idx] - 0.15)
                )
            )
            end = (
                segment_end
                if idx == normalized_count - 1
                else (
                    (points[idx] + points[idx + 1]) / 2.0
                    if next_gap < 1.5
                    else min(segment_end, max(points[idx] + 0.15, start + 0.12))
                )
            )
        pairs_by_normalized.append((max(segment_start, start), min(segment_end, max(end, start))))

    out: list[tuple[float, float]] = []
    normalized_idx = 0
    last_pair = (segment_start, segment_start)
    for ch in text:
        if re.match(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]", ch):
            last_pair = pairs_by_normalized[normalized_idx]
            normalized_idx += 1
            out.append(last_pair)
        else:
            out.append((last_pair[1], last_pair[1]))
    return out


def _split_text_without_timestamps(text: str) -> list[Segment]:
    chunks: list[Segment] = []
    start = 0
    for end in _sentence_break_indexes(text, max_chars=60):
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Segment(start=0.0, end=0.0, text=chunk))
        start = end
    return chunks or [Segment(start=0.0, end=0.0, text=text)]


def _split_text_with_coarse_timing(text: str, start: float, end: float) -> list[Segment]:
    chunks = _split_text_without_timestamps(text)
    duration = max(end - start, 0.0)
    if not chunks or duration <= 0:
        return chunks
    total_chars = sum(max(len(s.text), 1) for s in chunks)
    cursor = start
    fixed: list[Segment] = []
    for chunk in chunks:
        share = duration * (max(len(chunk.text), 1) / total_chars)
        fixed.append(Segment(start=cursor, end=cursor + share, text=chunk.text))
        cursor += share
    return fixed


def _segments_from_text_and_timestamps(text: str, timestamps: Any, *, sensevoice: bool) -> list[Segment]:
    text = _clean_text(text, sensevoice=sensevoice)
    if not text:
        return []
    if not isinstance(timestamps, list) or not timestamps:
        return _split_text_without_timestamps(text)

    # Paraformer commonly returns one [start_ms, end_ms] timestamp per emitted
    # Chinese character. Punctuation may be inserted later and therefore have no
    # timestamp; align timestamps to non-punctuation characters instead of
    # falling back to coarse text distribution.
    char_times = _char_time_pairs_from_timestamps(text, timestamps)
    if not char_times:
        start = _to_seconds(timestamps[0][0], assume_ms=True) if isinstance(timestamps[0], (list, tuple)) else 0.0
        last = timestamps[-1]
        end = _to_seconds(last[1] if isinstance(last, (list, tuple)) and len(last) > 1 else last, assume_ms=True)
        return _split_text_with_coarse_timing(text, start, max(end, start))

    segments: list[Segment] = []
    start_idx = 0
    for end_idx in _sentence_break_indexes(text):
        chunk = text[start_idx:end_idx].strip()
        if not chunk:
            start_idx = end_idx
            continue
        meaningful_times = [
            char_times[idx]
            for idx in range(start_idx, min(end_idx, len(char_times)))
            if idx < len(text) and re.match(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]", text[idx])
        ]
        if meaningful_times:
            start = min(pair[0] for pair in meaningful_times)
            end = max(pair[1] for pair in meaningful_times)
        else:
            start = char_times[start_idx][0]
            end = char_times[min(end_idx - 1, len(char_times) - 1)][1]
        segment_start = start
        segment_end = max(end, start)
        sync_cues = _build_sync_cues_from_char_times(
            chunk,
            char_times[start_idx:end_idx],
            segment_start=segment_start,
            segment_end=segment_end,
        )
        segments.append(Segment(start=segment_start, end=segment_end, text=chunk, sync_cues=sync_cues or None))
        start_idx = end_idx
    return segments


def _coarsen_zero_timing(segments: list[Segment], duration: float) -> list[Segment]:
    """Distribute coarse timings when the model output has text but no timecodes."""
    if not segments or any(s.end > s.start for s in segments) or duration <= 0:
        return segments
    total_chars = sum(max(len(s.text), 1) for s in segments)
    cursor = 0.0
    fixed: list[Segment] = []
    for seg in segments:
        share = duration * (max(len(seg.text), 1) / total_chars)
        fixed.append(Segment(start=cursor, end=cursor + share, text=seg.text, speaker=seg.speaker, sync_cues=seg.sync_cues))
        cursor += share
    return fixed


def _cue_normalized_times(segment: Segment) -> list[float]:
    times: list[float] = []
    for cue in segment.sync_cues or []:
        text = str(cue.get("text") or "")
        chars = [ch for ch in text if re.match(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]", ch)]
        if not chars:
            continue
        start = float(cue.get("start", segment.start))
        end = float(cue.get("end", start))
        if len(chars) <= 1 or end <= start:
            times.append((start + end) / 2.0)
            continue
        for idx, _ch in enumerate(chars):
            times.append(start + (end - start) * (idx / (len(chars) - 1)))
    return times


def _timing_stream(segments: list[Segment]) -> tuple[str, list[float], list[tuple[int, int]]]:
    chars: list[str] = []
    times: list[float] = []
    ranges: list[tuple[int, int]] = []
    for seg in segments:
        seg_start = len(chars)
        text = seg.text or ""
        normalized_chars = [ch.lower() for ch in text if re.match(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]", ch)]
        count = len(normalized_chars)
        emitted = 0
        cue_times = _cue_normalized_times(seg)
        for ch in text:
            if not re.match(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]", ch):
                continue
            chars.append(ch.lower())
            if emitted < len(cue_times):
                times.append(cue_times[emitted])
            elif count <= 1 or seg.end <= seg.start:
                times.append((seg.start + seg.end) / 2.0)
            else:
                times.append(seg.start + (seg.end - seg.start) * (emitted / (count - 1)))
            emitted += 1
        ranges.append((seg_start, len(chars)))
    return "".join(chars), times, ranges


def _map_text_to_anchor_times(
    source_text: str,
    anchor_text: str,
    anchor_times: list[float],
    *,
    lower_bound: float,
    upper_bound: float,
    source_ranges: list[tuple[int, int]] | None = None,
) -> tuple[list[float], int, int]:
    """Map every source character to a monotonic point on the anchor timeline.

    Equal text keeps its acoustic timestamp. Interior differences are
    interpolated between reliable neighbours. Edge differences intentionally
    keep the legacy mapping here and are handled by the narrowly scoped
    impossible-speed guard in ``_repair_collapsed_edge_timing_ranges``.
    """
    if not source_text or not anchor_text or not anchor_times:
        return [], 0, 0

    lower = min(float(lower_bound), float(anchor_times[0]))
    upper = max(float(upper_bound), float(anchor_times[-1]), lower)
    matcher = SequenceMatcher(None, source_text, anchor_text, autojunk=False)
    source_times: list[float | None] = [None] * len(source_text)
    source_kinds = [""] * len(source_text)
    equal_chars = 0
    estimated_chars = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        source_count = i2 - i1
        if tag == "equal":
            equal_chars += source_count
            for offset in range(source_count):
                anchor_index = j1 + offset
                if 0 <= anchor_index < len(anchor_times):
                    source_times[i1 + offset] = float(anchor_times[anchor_index])
            continue
        for source_index in range(i1, i2):
            source_kinds[source_index] = tag
        if tag == "replace" and source_count > 0 and j2 > j1:
            replacement_times = [
                float(value)
                for value in anchor_times[j1:min(j2, len(anchor_times))]
            ]
            if replacement_times:
                replacement_start = min(replacement_times)
                replacement_end = max(replacement_times)
                for offset in range(source_count):
                    ratio = (offset + 0.5) / source_count
                    source_times[i1 + offset] = replacement_start + (
                        replacement_end - replacement_start
                    ) * ratio
                estimated_chars += source_count
        # Source-only spans are filled below. Equal characters retain their
        # accepted acoustic timestamps.
    if not any(value is not None for value in source_times):
        return [], equal_chars, estimated_chars

    index = 0
    while index < len(source_times):
        if source_times[index] is not None:
            index += 1
            continue
        run_start = index
        while index < len(source_times) and source_times[index] is None:
            index += 1
        run_end = index
        left = float(source_times[run_start - 1]) if run_start > 0 else lower
        right = float(source_times[run_end]) if run_end < len(source_times) else upper
        right = max(left, right)
        run_length = run_end - run_start
        containing_range = next(
            (
                (range_start, range_end)
                for range_start, range_end in (source_ranges or [])
                if range_start <= run_start and run_end <= range_end
            ),
            None,
        )
        at_segment_start = bool(containing_range and run_start == containing_range[0])
        at_segment_end = bool(containing_range and run_end == containing_range[1])
        long_acoustic_gap = right - left > max(3.0, run_length / 1.5)
        source_only = all(
            kind == "delete" for kind in source_kinds[run_start:run_end]
        )
        intersecting_ranges = [
            (
                max(run_start, range_start),
                min(run_end, range_end),
                range_start,
                range_end,
            )
            for range_start, range_end in (source_ranges or [])
            if max(run_start, range_start) < min(run_end, range_end)
        ]
        if long_acoustic_gap and source_only and len(intersecting_ranges) > 1:
            right_pieces = [
                (piece_start, piece_end)
                for piece_start, piece_end, range_start, range_end in intersecting_ranges
                if piece_start == range_start and piece_end < range_end
            ]
            right_indexes = {
                source_index
                for piece_start, piece_end in right_pieces
                for source_index in range(piece_start, piece_end)
            }
            left_indexes = [
                source_index
                for source_index in range(run_start, run_end)
                if source_index not in right_indexes
            ]
            ordered_right_indexes = sorted(right_indexes)
            left_duration = min(2.0, max(0.2, len(left_indexes) / 4.0))
            right_duration = min(2.0, max(0.2, len(ordered_right_indexes) / 4.0))
            for position, source_index in enumerate(left_indexes):
                ratio = (position + 1) / (len(left_indexes) + 1)
                source_times[source_index] = left + min(right - left, left_duration) * ratio
            right_start = max(left, right - right_duration)
            for position, source_index in enumerate(ordered_right_indexes):
                ratio = (position + 1) / (len(ordered_right_indexes) + 1)
                source_times[source_index] = right_start + (right - right_start) * ratio
            estimated_chars += run_length
            continue
        attach_to_right = (
            long_acoustic_gap and source_only and at_segment_start and not at_segment_end
        )
        attach_to_left = (
            long_acoustic_gap and source_only and at_segment_end and not at_segment_start
        )
        compact_duration = min(2.0, max(0.2, run_length / 4.0))
        for offset in range(run_length):
            if run_start == 0:
                source_times[run_start + offset] = right
            elif run_end == len(source_times):
                source_times[run_start + offset] = left
            elif attach_to_right:
                compact_start = max(left, right - compact_duration)
                ratio = (offset + 1) / (run_length + 1)
                source_times[run_start + offset] = compact_start + (
                    right - compact_start
                ) * ratio
            elif attach_to_left:
                compact_end = min(right, left + compact_duration)
                ratio = (offset + 1) / (run_length + 1)
                source_times[run_start + offset] = left + (
                    compact_end - left
                ) * ratio
            else:
                ratio = (offset + 1) / (run_length + 1)
                source_times[run_start + offset] = left + (right - left) * ratio
        estimated_chars += run_length

    mapped_times = [float(value) for value in source_times if value is not None]
    if len(mapped_times) != len(source_text):
        return [], equal_chars, estimated_chars
    for index in range(1, len(mapped_times)):
        mapped_times[index] = max(mapped_times[index], mapped_times[index - 1])
    return mapped_times, equal_chars, estimated_chars


def _repair_collapsed_edge_timing_ranges(
    times: list[float],
    ranges: list[tuple[int, int]],
    *,
    lower_bound: float,
    upper_bound: float,
    max_chars_per_second: float = 20.0,
) -> tuple[list[float], list[str]]:
    """Reflow impossible edge speeds when an anchor omits opening/closing text.

    This is deliberately limited to the outermost source segments. Interior
    timestamps remain acoustic-anchor driven, while a collapsed edge borrows
    only from its adjacent segment and keeps that pair's outer boundary.
    """
    if len(times) < 2 or len(ranges) < 2 or max_chars_per_second <= 0:
        return times, []
    repaired = list(times)
    repaired_edges: list[str] = []

    def collapsed(start: int, end: int) -> bool:
        count = end - start
        if count < 3:
            return False
        span = max(0.0, repaired[end - 1] - repaired[start])
        return span + 1e-6 < count / max_chars_per_second

    first_start, first_end = ranges[0]
    next_start, next_end = ranges[1]
    if (
        collapsed(first_start, first_end)
        and next_end > next_start
        and repaired[next_end - 1] > lower_bound
    ):
        first_count = first_end - first_start
        next_count = next_end - next_start
        pair_end = repaired[next_end - 1]
        pair_duration = pair_end - lower_bound
        if pair_duration >= (first_count + next_count) / max_chars_per_second:
            boundary = lower_bound + pair_duration * first_count / (first_count + next_count)
            for offset in range(first_count):
                repaired[first_start + offset] = lower_bound + (
                    boundary - lower_bound
                ) * ((offset + 0.5) / first_count)
            old_start = repaired[next_start]
            old_end = repaired[next_end - 1]
            old_span = old_end - old_start
            for offset, index in enumerate(range(next_start, next_end)):
                ratio = (
                    (repaired[index] - old_start) / old_span
                    if old_span > 1e-6
                    else (offset + 0.5) / next_count
                )
                repaired[index] = boundary + (pair_end - boundary) * ratio
            repaired_edges.append("leading")

    previous_start, previous_end = ranges[-2]
    last_start, last_end = ranges[-1]
    if (
        collapsed(last_start, last_end)
        and previous_end > previous_start
        and upper_bound > repaired[previous_start]
    ):
        previous_count = previous_end - previous_start
        last_count = last_end - last_start
        pair_start = repaired[previous_start]
        pair_duration = upper_bound - pair_start
        if pair_duration >= (previous_count + last_count) / max_chars_per_second:
            boundary = pair_start + pair_duration * previous_count / (previous_count + last_count)
            old_start = repaired[previous_start]
            old_end = repaired[previous_end - 1]
            old_span = old_end - old_start
            for offset, index in enumerate(range(previous_start, previous_end)):
                ratio = (
                    (repaired[index] - old_start) / old_span
                    if old_span > 1e-6
                    else (offset + 0.5) / previous_count
                )
                repaired[index] = pair_start + (boundary - pair_start) * ratio
            for offset in range(last_count):
                repaired[last_start + offset] = boundary + (
                    upper_bound - boundary
                ) * ((offset + 0.5) / last_count)
            repaired_edges.append("trailing")

    for index in range(1, len(repaired)):
        repaired[index] = max(repaired[index], repaired[index - 1])
    return repaired, repaired_edges


def _align_segments_to_timing_anchor(
    source_segments: list[Segment],
    anchor_segments: list[Segment],
    *,
    min_equal_ratio: float = 0.55,
) -> tuple[list[Segment], dict[str, Any]]:
    """Keep source text while borrowing wall-clock timings from an anchor pass.

    SenseVoice full-audio decoding is usually the better Chinese text path, but
    it often returns no sentence timestamps.  A chunked VAD pass gives much
    better wall-clock anchors but can slightly change wording.  This aligns the
    full-audio text stream to the chunked text stream and transfers timings only
    when the two streams are similar enough.
    """
    source_text, _source_times, source_ranges = _timing_stream(source_segments)
    anchor_text, anchor_times, _anchor_ranges = _timing_stream(anchor_segments)
    if not source_text or not anchor_text or not anchor_times:
        return [], {
            "timing_alignment_ok": False,
            "timing_alignment_reason": "empty_source_or_anchor",
            "source_chars": len(source_text),
            "anchor_chars": len(anchor_text),
            "equal_char_ratio": 0.0,
        }

    matcher = SequenceMatcher(None, source_text, anchor_text, autojunk=False)
    equal_chars = sum(i2 - i1 for tag, i1, i2, _j1, _j2 in matcher.get_opcodes() if tag == "equal")

    equal_ratio = equal_chars / max(1, len(source_text))
    stats: dict[str, Any] = {
        "timing_alignment_ok": equal_ratio >= min_equal_ratio,
        "source_chars": len(source_text),
        "anchor_chars": len(anchor_text),
        "equal_char_ratio": round(equal_ratio, 4),
        "min_equal_ratio": min_equal_ratio,
    }
    if equal_ratio < min_equal_ratio:
        stats["timing_alignment_reason"] = "source_anchor_text_too_different"
        return [], stats

    anchor_start = min((segment.start for segment in anchor_segments), default=float(anchor_times[0]))
    anchor_end = max((segment.end for segment in anchor_segments), default=float(anchor_times[-1]))
    source_times, _equal_chars, estimated_chars = _map_text_to_anchor_times(
        source_text,
        anchor_text,
        anchor_times,
        lower_bound=anchor_start,
        upper_bound=anchor_end,
        source_ranges=source_ranges,
    )
    stats["estimated_timing_chars"] = estimated_chars
    if not source_times:
        stats["timing_alignment_ok"] = False
        stats["timing_alignment_reason"] = "no_equal_char_timing"
        return [], stats
    source_times, repaired_edges = _repair_collapsed_edge_timing_ranges(
        source_times,
        source_ranges,
        lower_bound=anchor_start,
        upper_bound=anchor_end,
    )
    stats["repaired_collapsed_edges"] = repaired_edges

    aligned: list[Segment] = []
    previous_end = 0.0
    for seg, (start_idx, end_idx) in zip(source_segments, source_ranges):
        seg_times: list[float] = []
        if end_idx > start_idx:
            seg_times = [
                float(value)
                for value in source_times[start_idx:end_idx]
            ]
            if seg_times:
                start = min(seg_times)
                end = max(seg_times)
                pad = 0.12
                if end > start:
                    pad = max(0.08, min(0.45, (end - start) / max(1, len(seg_times)) * 0.8))
                start = max(0.0, start - pad)
                end += pad
            else:
                start, end = seg.start, seg.end
        else:
            start, end = seg.start, seg.end

        if start < previous_end:
            start = previous_end
        if end <= start:
            end = start + max(0.5, min(6.0, len(seg.text or "") / 4.5))
        sync_cues = []
        if end_idx > start_idx and seg_times:
            char_pairs = _char_time_pairs_from_points(
                seg.text or "",
                seg_times,
                segment_start=start,
                segment_end=end,
            )
            sync_cues = _build_sync_cues_from_char_times(
                seg.text or "",
                char_pairs,
                segment_start=start,
                segment_end=end,
            )

        aligned.append(
            Segment(
                start=round(start, 3),
                end=round(end, 3),
                text=seg.text,
                original_text=seg.original_text,
                speaker=seg.speaker,
                sync_cues=sync_cues or None,
            )
        )
        previous_end = aligned[-1].end

    return aligned, stats


def _sync_cue_text_key(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _realign_sync_cues_preserving_segments(
    segments: list[Segment],
    timing_anchor: list[Segment],
    *,
    min_equal_ratio: float = 0.5,
) -> tuple[list[Segment], dict[str, Any]]:
    """Repair cue text after normalization without changing transcript geometry."""
    geometry_before = [(segment.start, segment.end, segment.text) for segment in segments]
    segments, duration_repair = _repair_nonpositive_sync_cues_preserving_segments(segments)
    target_text, _target_times, target_ranges = _timing_stream(segments)
    anchor_text, anchor_times, _anchor_ranges = _timing_stream(timing_anchor)
    mismatched = [
        index
        for index, segment in enumerate(segments)
        if segment.sync_cues
        and _sync_cue_text_key("".join(str(cue.get("text") or "") for cue in segment.sync_cues))
        != _sync_cue_text_key(segment.text)
    ]
    stats: dict[str, Any] = {
        "enabled": True,
        "input_segments": len(segments),
        "mismatched_segments_before": len(mismatched),
        "repaired_segments": 0,
        "equal_char_ratio": 0.0,
        "geometry_preserved": True,
        **duration_repair,
    }
    if not mismatched:
        stats["reason"] = "cue_text_already_aligned"
        return segments, stats
    if not target_text or not anchor_text or not anchor_times:
        stats["reason"] = "missing_timing_anchor"
        return segments, stats

    matcher = SequenceMatcher(None, target_text, anchor_text, autojunk=False)
    equal_chars = sum(i2 - i1 for tag, i1, i2, _j1, _j2 in matcher.get_opcodes() if tag == "equal")

    equal_ratio = equal_chars / max(1, len(target_text))
    stats["equal_char_ratio"] = round(equal_ratio, 4)
    if equal_ratio < min_equal_ratio:
        stats["reason"] = "normalized_text_too_different_from_timing_anchor"
        return segments, stats

    anchor_start = min((segment.start for segment in timing_anchor), default=float(anchor_times[0]))
    anchor_end = max((segment.end for segment in timing_anchor), default=float(anchor_times[-1]))
    target_times, _equal_chars, estimated_chars = _map_text_to_anchor_times(
        target_text,
        anchor_text,
        anchor_times,
        lower_bound=anchor_start,
        upper_bound=anchor_end,
        source_ranges=target_ranges,
    )
    stats["estimated_timing_chars"] = estimated_chars
    if not target_times:
        stats["reason"] = "no_mapped_character_times"
        return segments, stats

    mismatch_set = set(mismatched)
    repaired: list[Segment] = []
    for segment_index, (segment, (start_index, end_index)) in enumerate(zip(segments, target_ranges)):
        if segment_index not in mismatch_set or end_index <= start_index:
            repaired.append(segment)
            continue
        points = [
            min(segment.end, max(segment.start, float(value)))
            for value in target_times[start_index:end_index]
        ]
        if len(points) != end_index - start_index:
            repaired.append(segment)
            continue
        for index in range(1, len(points)):
            points[index] = max(points[index], points[index - 1])
        char_pairs = _char_time_pairs_from_points(
            segment.text,
            points,
            segment_start=segment.start,
            segment_end=segment.end,
        )
        cues = _build_sync_cues_from_char_times(
            segment.text,
            char_pairs,
            segment_start=segment.start,
            segment_end=segment.end,
        )
        if not cues or _sync_cue_text_key("".join(str(cue.get("text") or "") for cue in cues)) != _sync_cue_text_key(segment.text):
            repaired.append(segment)
            continue
        repaired.append(replace(segment, sync_cues=cues))
        stats["repaired_segments"] += 1

    stats["mismatched_segments_after"] = sum(
        1
        for segment in repaired
        if segment.sync_cues
        and _sync_cue_text_key("".join(str(cue.get("text") or "") for cue in segment.sync_cues))
        != _sync_cue_text_key(segment.text)
    )
    stats["geometry_preserved"] = (
        geometry_before == [(segment.start, segment.end, segment.text) for segment in repaired]
    )
    if not stats["geometry_preserved"]:
        raise RuntimeError("sync cue repair changed transcript geometry")
    stats["reason"] = "repaired" if stats["repaired_segments"] else "no_segment_repaired"
    return repaired, stats


def _items_have_timing(items: list[Any]) -> bool:
    for item in items:
        if not isinstance(item, dict):
            continue
        sentence_info = item.get("sentence_info")
        if isinstance(sentence_info, list):
            for sentence in sentence_info:
                if not isinstance(sentence, dict):
                    continue
                start = _to_seconds(sentence.get("start", 0.0), assume_ms=True)
                end = _to_seconds(sentence.get("end", start), assume_ms=True)
                if end > start:
                    return True
        timestamps = item.get("timestamp") or item.get("timestamps")
        if isinstance(timestamps, list):
            for timestamp in timestamps:
                start, end = _timestamp_pair(timestamp)
                if end > start:
                    return True
    return False


def _paraformer_anchor_quality(
    items: list[Any],
    segments: list[Segment],
) -> dict[str, Any]:
    """Reject a single coarse span masquerading as phrase-level timing."""
    raw_items = [item for item in items if isinstance(item, dict)]
    timed_sentence_count = 0
    timestamp_item_count = 0
    for item in raw_items:
        sentence_info = item.get("sentence_info")
        if isinstance(sentence_info, list):
            timed_sentence_count += sum(
                1
                for sentence in sentence_info
                if isinstance(sentence, dict)
                and _to_seconds(sentence.get("end", 0.0), assume_ms=True)
                > _to_seconds(sentence.get("start", 0.0), assume_ms=True)
            )
        timestamps = item.get("timestamp") or item.get("timestamps")
        if isinstance(timestamps, list) and timestamps:
            timestamp_item_count += 1

    sync_cue_segments = sum(1 for segment in segments if segment.sync_cues)
    if sync_cue_segments:
        precision = "character"
        usable = True
        reason = "character_timestamps"
    elif timed_sentence_count:
        precision = "sentence"
        usable = True
        reason = "sentence_timestamps"
    elif len(raw_items) >= 2 and len(segments) >= 2:
        precision = "item"
        usable = True
        reason = "multiple_timed_items"
    elif timestamp_item_count:
        precision = "coarse"
        usable = False
        reason = "single_item_coarse_timestamp_projection"
    else:
        precision = "none"
        usable = False
        reason = "no_usable_timing_detail"

    return {
        "usable": usable,
        "precision": precision,
        "reason": reason,
        "raw_items": len(raw_items),
        "timed_sentence_count": timed_sentence_count,
        "timestamp_item_count": timestamp_item_count,
        "sync_cue_segments": sync_cue_segments,
    }


def _paraformer_timing_preflight(
    segments: list[Segment],
    duration: float,
) -> dict[str, Any]:
    """Decide whether a cached Paraformer anchor should run before chunk ASR.

    Low transcript density on a medium-length recording is a strong signal that
    full-context SenseVoice and short wall-clock chunks will disagree. In that
    case running every chunk before falling back to Paraformer wastes most of
    the processing time. The decision is deliberately content-agnostic and can
    be disabled or tuned through environment variables.
    """
    enabled = _env_flag("LOCALSCRIBE_SENSEVOICE_PARAFORMER_PREFLIGHT", True)
    min_duration_s = max(
        0.0,
        _env_float("LOCALSCRIBE_SENSEVOICE_PARAFORMER_PREFLIGHT_MIN_S", 300.0),
    )
    max_duration_s = max(
        min_duration_s,
        _env_float("LOCALSCRIBE_SENSEVOICE_PARAFORMER_PREFLIGHT_MAX_S", 1200.0),
    )
    max_chars_per_s = max(
        0.1,
        _env_float("LOCALSCRIBE_SENSEVOICE_PARAFORMER_PREFLIGHT_MAX_CPS", 3.0),
    )
    normalized_text = normalize_recovery_text("".join(segment.text or "" for segment in segments))
    chars = len(normalized_text)
    chars_per_s = chars / duration if duration > 0 else 0.0

    selected = bool(
        enabled
        and chars > 0
        and min_duration_s <= duration <= max_duration_s
        and chars_per_s < max_chars_per_s
    )
    if not enabled:
        reason = "disabled"
    elif duration <= 0 or chars <= 0:
        reason = "missing_duration_or_text"
    elif duration < min_duration_s:
        reason = "recording_too_short"
    elif duration > max_duration_s:
        reason = "recording_too_long"
    elif chars_per_s >= max_chars_per_s:
        reason = "source_text_density_normal"
    else:
        reason = "low_source_text_density"
    return {
        "enabled": enabled,
        "selected": selected,
        "reason": reason,
        "duration_s": round(duration, 3),
        "source_chars": chars,
        "source_chars_per_s": round(chars_per_s, 4),
        "min_duration_s": min_duration_s,
        "max_duration_s": max_duration_s,
        "max_chars_per_s": max_chars_per_s,
    }


def _segments_from_generate_items(items: list[Any], *, sensevoice: bool) -> tuple[list[Segment], str | None]:
    segments: list[Segment] = []
    detected_language: str | None = None
    for item in items:
        if not isinstance(item, dict):
            continue
        detected_language = detected_language or item.get("language")
        sent_segments = _segments_from_sentence_info(item.get("sentence_info"), sensevoice=sensevoice)
        if sent_segments:
            segments.extend(sent_segments)
            continue
        text = item.get("text") or item.get("raw_text") or ""
        segments.extend(
            _segments_from_text_and_timestamps(
                str(text),
                item.get("timestamp") or item.get("timestamps"),
                sensevoice=sensevoice,
            )
        )
    return segments, detected_language


def _merge_speech_ranges(
    ranges: list[tuple[float, float]],
    *,
    max_chunk_s: float = 5.0,
    max_gap_s: float = 0.75,
) -> list[tuple[float, float]]:
    max_chunk_s = max(0.5, max_chunk_s)
    pieces: list[tuple[float, float]] = []
    for raw_start, raw_end in sorted(ranges):
        start = max(0.0, float(raw_start))
        end = float(raw_end)
        while end - start > max_chunk_s:
            pieces.append((start, start + max_chunk_s))
            start += max_chunk_s
        if end > start:
            pieces.append((start, end))

    chunks: list[tuple[float, float]] = []
    cur_start: float | None = None
    cur_end: float | None = None
    for start, end in pieces:
        if cur_start is None or cur_end is None:
            cur_start, cur_end = start, end
            continue
        if start - cur_end <= max_gap_s and end - cur_start <= max_chunk_s:
            cur_end = end
            continue
        chunks.append((cur_start, cur_end))
        cur_start, cur_end = start, end
    if cur_start is not None and cur_end is not None:
        chunks.append((cur_start, cur_end))
    return chunks


def _normalize_intervals(
    ranges: list[tuple[float, float]],
    *,
    duration: float = 0.0,
    collar_s: float = 0.0,
) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for raw_start, raw_end in ranges:
        try:
            start = max(0.0, float(raw_start) - collar_s)
            end = float(raw_end) + collar_s
        except (TypeError, ValueError):
            continue
        if duration > 0:
            start = min(start, duration)
            end = min(end, duration)
        if end > start:
            cleaned.append((start, end))
    cleaned.sort()
    merged: list[tuple[float, float]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _subtract_intervals(
    source: list[tuple[float, float]],
    covered: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    uncovered: list[tuple[float, float]] = []
    covered_index = 0
    for source_start, source_end in source:
        cursor = source_start
        while covered_index < len(covered) and covered[covered_index][1] <= source_start:
            covered_index += 1
        index = covered_index
        while index < len(covered) and covered[index][0] < source_end:
            cover_start, cover_end = covered[index]
            if cover_start > cursor:
                uncovered.append((cursor, min(cover_start, source_end)))
            cursor = max(cursor, cover_end)
            if cursor >= source_end:
                break
            index += 1
        if cursor < source_end:
            uncovered.append((cursor, source_end))
    return [(start, end) for start, end in uncovered if end > start]


def _speech_coverage_diagnostics(
    speech_ranges: list[tuple[float, float]],
    segments: list[Segment],
    *,
    duration: float = 0.0,
    collar_s: float = 0.5,
    vad_status: str = "ok",
    vad_reason: str = "ok",
    max_details: int = 20,
) -> dict[str, Any]:
    settings = {
        "segment_collar_s": round(max(0.0, collar_s), 3),
        "min_speech_coverage_ratio": _env_float("LOCALSCRIBE_SENSEVOICE_MIN_SPEECH_COVERAGE_RATIO", 0.99),
        "max_uncovered_speech_s": _env_float("LOCALSCRIBE_SENSEVOICE_MAX_UNCOVERED_SPEECH_S", 3.0),
        "max_edge_uncovered_speech_s": _env_float(
            "LOCALSCRIBE_SENSEVOICE_MAX_EDGE_UNCOVERED_SPEECH_S", 1.0
        ),
    }
    diagnostics: dict[str, Any] = {
        "status": vad_status,
        "reason": vad_reason,
        "speech_ranges": 0,
        "speech_duration_s": 0.0,
        "covered_speech_s": 0.0,
        "uncovered_speech_s": 0.0,
        "speech_coverage_ratio": None,
        "max_uncovered_speech_s": 0.0,
        "leading_uncovered_speech_s": 0.0,
        "trailing_uncovered_speech_s": 0.0,
        "speech_intervals": [],
        "covered_intervals": [],
        "uncovered_speech_ranges": [],
        "uncovered_speech_ranges_truncated": False,
        "settings": settings,
    }
    if vad_status not in {"ok", "no_speech"}:
        return diagnostics

    normalized_speech = _normalize_intervals(speech_ranges, duration=duration)
    if not normalized_speech:
        diagnostics.update({"status": "no_speech", "reason": vad_reason or "vad_detected_no_speech"})
        return diagnostics

    normalized_segments = _normalize_intervals(
        [(segment.start, segment.end) for segment in segments if segment.end > segment.start],
        duration=duration,
        collar_s=max(0.0, collar_s),
    )
    uncovered = _subtract_intervals(normalized_speech, normalized_segments)
    speech_duration = sum(end - start for start, end in normalized_speech)
    uncovered_duration = sum(end - start for start, end in uncovered)
    covered_duration = max(0.0, speech_duration - uncovered_duration)
    first_speech_start = normalized_speech[0][0]
    last_speech_end = normalized_speech[-1][1]
    leading = (
        uncovered[0][1] - uncovered[0][0]
        if uncovered and abs(uncovered[0][0] - first_speech_start) <= 0.001
        else 0.0
    )
    trailing = (
        uncovered[-1][1] - uncovered[-1][0]
        if uncovered and abs(uncovered[-1][1] - last_speech_end) <= 0.001
        else 0.0
    )
    details = [
        {"start": round(start, 3), "end": round(end, 3), "duration": round(end - start, 3)}
        for start, end in uncovered[:max_details]
    ]
    diagnostics.update({
        "status": "ok",
        "reason": "compared_vad_speech_with_transcript_segments",
        "speech_ranges": len(normalized_speech),
        "speech_duration_s": round(speech_duration, 3),
        "covered_speech_s": round(covered_duration, 3),
        "uncovered_speech_s": round(uncovered_duration, 3),
        "speech_coverage_ratio": round(covered_duration / speech_duration, 6),
        "max_uncovered_speech_s": round(max((end - start for start, end in uncovered), default=0.0), 3),
        "leading_uncovered_speech_s": round(leading, 3),
        "trailing_uncovered_speech_s": round(trailing, 3),
        "speech_intervals": [
            {"start": round(start, 3), "end": round(end, 3)}
            for start, end in normalized_speech
        ],
        "covered_intervals": [
            {"start": round(start, 3), "end": round(end, 3)}
            for start, end in normalized_segments
        ],
        "uncovered_speech_ranges": details,
        "uncovered_speech_ranges_truncated": len(uncovered) > max_details,
    })
    return diagnostics


def _suppress_vad_unsupported_segments(
    segments: list[Segment],
    speech_ranges: list[tuple[float, float]],
    *,
    vad_status: str,
) -> tuple[list[Segment], dict[str, Any]]:
    """Mark only lexical-empty or extremely sparse text outside detected speech.

    SenseVoice may emit punctuation or a few characters for hold music. Silero
    VAD is used only as corroborating evidence: normal-density text is never
    changed. Suspect text is replaced by a non-speech marker while every source
    segment and sync-cue boundary remains intact for playback synchronization.
    """
    stats: dict[str, Any] = {
        "mode": "vad_unsupported_text_guard_v1",
        "enabled": vad_status == "ok",
        "applied": False,
        "vad_status": vad_status,
        "input_segments": len(segments),
        "output_segments": len(segments),
        "suppressed_segments": 0,
        "suppressed_seconds": 0.0,
        "items": [],
        "uses_recording_name": False,
        "uses_fixed_transcript_phrases": False,
        "segment_geometry_preserved": True,
        "sync_cue_boundaries_preserved": True,
        "reason": "",
    }
    if vad_status != "ok":
        stats["reason"] = "vad_evidence_unavailable"
        return list(segments), stats

    normalized_speech = _normalize_intervals(speech_ranges)
    if not normalized_speech:
        stats["enabled"] = False
        stats["reason"] = "vad_detected_no_speech"
        return list(segments), stats

    output: list[Segment] = []
    suppressed_seconds = 0.0
    suppressed_count = 0
    for index, segment in enumerate(segments):
        start = float(segment.start)
        end = float(segment.end)
        duration = max(0.0, end - start)
        lexical = normalize_recovery_text(segment.text or "")
        overlap_s = sum(
            max(0.0, min(end, speech_end) - max(start, speech_start))
            for speech_start, speech_end in normalized_speech
            if speech_end > start and speech_start < end
        )
        punctuation_only = not lexical
        long_sparse_text = duration >= 8.0 and len(lexical) <= 6
        unsupported = overlap_s <= 0.12
        # A standalone punctuation segment carries no transcript content. Keep
        # its geometry, but suppress it independently of VAD overlap so timing
        # realignment cannot change the exported text hash.
        if punctuation_only or (unsupported and long_sparse_text):
            suppressed_seconds += duration
            suppressed_count += 1
            if len(stats["items"]) < 40:
                stats["items"].append({
                    "index": index,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration_s": round(duration, 3),
                    "text": segment.text,
                    "lexical_chars": len(lexical),
                    "vad_overlap_s": round(overlap_s, 3),
                    "reason": (
                        "standalone_punctuation"
                        if punctuation_only
                        else "long_sparse_text_without_detected_speech"
                    ),
                })
            marker = "" if punctuation_only else "（非语音）"
            cues = None
            if segment.sync_cues:
                cues = []
                for cue_index, cue in enumerate(segment.sync_cues):
                    cues.append({
                        **cue,
                        "text": marker if cue_index == 0 else "",
                    })
            output.append(replace(
                segment,
                text=marker,
                original_text=segment.original_text or segment.text,
                sync_cues=cues,
            ))
            continue
        output.append(segment)

    stats["output_segments"] = len(output)
    stats["suppressed_segments"] = suppressed_count
    stats["suppressed_seconds"] = round(suppressed_seconds, 3)
    stats["applied"] = bool(stats["suppressed_segments"])
    stats["reason"] = "unsupported_text_suppressed" if stats["applied"] else "no_unsupported_text"
    return output, stats


class FunASRTranscriber(Transcriber):
    backend = "funasr"

    def __init__(self, device: str | None = None, backend_name: str = "funasr"):
        self.device = device or os.environ.get("LOCALSCRIBE_FUNASR_DEVICE", "cpu")
        self.backend = backend_name
        self._model = None
        self._loaded_model_id: str | None = None
        self._speech_ranges_cache_path: str | None = None
        self._speech_ranges_cache: list[tuple[float, float]] = []
        self._speech_ranges_status = "unavailable"
        self._speech_ranges_reason = "not_run"
        self._wallclock_attempted_ranges: list[tuple[float, float]] = []
        self._wallclock_recognized_ranges: list[tuple[float, float]] = []
        self._wallclock_failed_ranges: list[tuple[float, float]] = []
        self._wallclock_failure_reasons: list[dict[str, Any]] = []
        self._strict_coverage_windows: list[dict[str, Any]] = []
        self._strong_asr_detector_segments: list[Segment] = []
        self._strong_asr_detector_source = ""
        self._strong_asr_detector_is_paraformer = False

    def _remember_strong_asr_detector(
        self,
        segments: list[Segment],
        *,
        source: str,
        is_paraformer: bool,
    ) -> None:
        """Keep an already-generated timing transcript for optional ASR review."""
        if not segments:
            return
        if self._strong_asr_detector_is_paraformer and not is_paraformer:
            return
        self._strong_asr_detector_segments = list(segments)
        self._strong_asr_detector_source = source
        self._strong_asr_detector_is_paraformer = is_paraformer

    def strong_asr_detector_snapshot(self) -> tuple[list[Segment], str, bool]:
        return (
            list(self._strong_asr_detector_segments),
            self._strong_asr_detector_source,
            self._strong_asr_detector_is_paraformer,
        )

    def _load(self, model_id: str):
        try:
            from funasr import AutoModel
        except Exception as exc:
            raise RuntimeError(
                "FunASR backend is not installed. Install it in the dev env with: "
                "python -m pip install funasr modelscope"
            ) from exc

        if self._model is not None and self._loaded_model_id == model_id:
            return self._model

        sensevoice = _is_sensevoice(model_id)
        resolved_model_id = resolve_model_path(model_id)
        resolved_vad_model = resolve_model_path("fsmn-vad")
        kwargs: dict[str, Any] = {
            "model": resolved_model_id,
            "vad_model": resolved_vad_model,
            "device": self.device,
            "disable_update": True,
        }
        if sensevoice:
            kwargs["vad_kwargs"] = {
                "max_single_segment_time": _env_int("LOCALSCRIBE_SENSEVOICE_VAD_MAX_MS", 30000)
            }
        else:
            kwargs["vad_kwargs"] = {
                "max_single_segment_time": _env_int("LOCALSCRIBE_FUNASR_VAD_MAX_MS", 60000)
            }
            # ct-punc currently pulls a very large punctuation model.  Keep it
            # opt-in for benchmark runs so a quick Chinese ASR trial does not
            # silently download another ~1GB dependency.
            if _env_flag("LOCALSCRIBE_FUNASR_PUNC", False):
                kwargs["punc_model"] = "ct-punc"

        self._model = AutoModel(**kwargs)
        self._loaded_model_id = model_id
        self._resolved_model_id = resolved_model_id
        self._resolved_vad_model = resolved_vad_model
        return self._model

    def _generate(self, model: Any, audio: Path, options: TranscribeOptions, *, sensevoice: bool):
        hotword = build_hotword_string(options.hotwords) or (options.initial_prompt or "").strip()
        if sensevoice:
            kwargs: dict[str, Any] = {
                "input": str(audio),
                "cache": {},
                "language": options.language or "auto",
                "use_itn": True,
                "batch_size_s": _env_int("LOCALSCRIBE_SENSEVOICE_BATCH_SIZE_S", 60),
                "merge_vad": _env_flag("LOCALSCRIBE_SENSEVOICE_MERGE_VAD", True),
                "merge_length_s": _env_float("LOCALSCRIBE_SENSEVOICE_MERGE_LENGTH_S", 15),
            }
            if hotword:
                kwargs["hotword"] = hotword
        else:
            kwargs = {
                "input": str(audio),
                "batch_size_s": _env_int("LOCALSCRIBE_FUNASR_BATCH_SIZE_S", 300),
            }
            if hotword:
                kwargs["hotword"] = hotword

        try:
            return _generate_reproducibly(model, kwargs)
        except TypeError:
            # Some FunASR releases expose narrower generate signatures.  Retry
            # with the safest minimum before surfacing a hard error.
            kwargs.pop("hotword", None)
            kwargs.pop("cache", None)
            kwargs.pop("merge_vad", None)
            kwargs.pop("merge_length_s", None)
            return _generate_reproducibly(model, kwargs)

    def _speech_ranges(self, audio: Path) -> list[tuple[float, float]]:
        cache_key = str(audio.resolve())
        if self._speech_ranges_cache_path == cache_key:
            return list(self._speech_ranges_cache)

        ranges: list[tuple[float, float]] = []
        try:
            import soundfile as sf
            import torch
            from silero_vad import get_speech_timestamps, load_silero_vad
        except Exception:
            self._store_speech_ranges(cache_key, ranges, "unavailable", "vad_dependencies_unavailable")
            return []

        try:
            info = sf.info(str(audio))
            sample_rate = int(info.samplerate)
            total_frames = int(info.frames)
        except Exception:
            self._store_speech_ranges(cache_key, ranges, "unavailable", "audio_read_failed")
            return []
        if sample_rate != 16000:
            self._store_speech_ranges(
                cache_key, ranges, "unavailable", f"unsupported_sample_rate:{sample_rate}"
            )
            return []

        chunk_s = min(1800.0, max(60.0, _env_float("LOCALSCRIBE_SENSEVOICE_VAD_STREAM_CHUNK_S", 600.0)))
        overlap_s = min(5.0, max(0.5, _env_float("LOCALSCRIBE_SENSEVOICE_VAD_STREAM_OVERLAP_S", 2.0)))
        chunk_frames = max(1, int(round(chunk_s * sample_rate)))
        overlap_frames = min(chunk_frames - 1, int(round(overlap_s * sample_rate)))
        step_frames = max(1, chunk_frames - overlap_frames)

        try:
            global _SILERO_VAD_MODEL
            with _SILERO_VAD_LOCK:
                if _SILERO_VAD_MODEL is None:
                    _SILERO_VAD_MODEL = load_silero_vad()
                vad_model = _SILERO_VAD_MODEL
                for chunk_start in range(0, total_frames, step_frames):
                    chunk_end = min(total_frames, chunk_start + chunk_frames)
                    samples, read_rate = sf.read(
                        str(audio),
                        dtype="float32",
                        start=chunk_start,
                        stop=chunk_end,
                        always_2d=True,
                    )
                    if int(read_rate) != sample_rate:
                        raise ValueError("vad_stream_sample_rate_mismatch")
                    if samples.shape[1] > 1:
                        samples = samples.mean(axis=1)
                    else:
                        samples = samples[:, 0]
                    timestamps = get_speech_timestamps(
                        torch.from_numpy(samples),
                        vad_model,
                        sampling_rate=16000,
                        threshold=_env_float("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD_THRESHOLD", 0.35),
                        min_speech_duration_ms=_env_int("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_MIN_SPEECH_MS", 120),
                        min_silence_duration_ms=_env_int("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_MIN_SILENCE_MS", 120),
                        speech_pad_ms=_env_int("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_PAD_MS", 120),
                    )
                    offset_s = chunk_start / float(sample_rate)
                    ranges.extend(
                        (
                            offset_s + float(item["start"]) / sample_rate,
                            offset_s + float(item["end"]) / sample_rate,
                        )
                        for item in timestamps
                        if item.get("end", 0) > item.get("start", 0)
                    )
                    if chunk_end >= total_frames:
                        break
        except Exception:
            self._store_speech_ranges(cache_key, [], "unavailable", "silero_vad_failed")
            return []

        ranges = _normalize_intervals(ranges, duration=total_frames / float(sample_rate))
        status = "ok" if ranges else "no_speech"
        reason = "vad_detected_speech" if ranges else "vad_detected_no_speech"
        self._store_speech_ranges(cache_key, ranges, status, reason)
        return list(ranges)

    def _store_speech_ranges(
        self,
        cache_key: str,
        ranges: list[tuple[float, float]],
        status: str,
        reason: str,
    ) -> None:
        self._speech_ranges_cache_path = cache_key
        self._speech_ranges_cache = list(ranges)
        self._speech_ranges_status = status
        self._speech_ranges_reason = reason

    def _load_local_recovery_provider(
        self,
        provider_name: str,
    ) -> tuple[Any | None, dict[str, Any], str | None]:
        metadata: dict[str, Any] = {
            "provider_id": "qwen3-independent",
            "provider_kind": "independent_asr",
            "model_id": QWEN3_RECOVERY_MODEL,
            "model_family": "qwen3_asr",
        }
        if provider_name != "qwen3":
            return None, metadata, "provider_disabled"
        try:
            snapshot, snapshot_metadata, snapshot_error = _cached_huggingface_snapshot(QWEN3_RECOVERY_MODEL)
            metadata.update(snapshot_metadata)
            if snapshot is None:
                return None, metadata, f"qwen3_{snapshot_error or 'model_not_cached'}"
            from .transcriber_qwen3 import Qwen3ASRTranscriber

            provider = Qwen3ASRTranscriber()
            provider._load(str(snapshot))
            provider._local_recovery_model_path = str(snapshot)
            return provider, metadata, None
        except Exception as exc:
            return None, metadata, f"qwen3_provider_failed:{type(exc).__name__}"

    def _run_local_recovery_provider(
        self,
        provider: Any,
        chunk_path: Path,
        options: TranscribeOptions,
    ) -> tuple[str, dict[str, Any]]:
        provider_segments, _language = provider._run(
            chunk_path,
            TranscribeOptions(
                model_id=str(provider._local_recovery_model_path),
                language=options.language or "zh",
                audio_preprocess="off",
            ),
            None,
        )
        return "".join(segment.text or "" for segment in provider_segments), dict(
            getattr(provider, "last_filter_stats", {}) or {}
        )

    def _run_sensevoice_wallclock_vad(
        self,
        model: Any,
        audio: Path,
        options: TranscribeOptions,
        on_progress: ProgressCallback | None,
    ) -> tuple[list[Segment], str | None, dict[str, Any]]:
        self._wallclock_attempted_ranges = []
        self._wallclock_recognized_ranges = []
        self._wallclock_failed_ranges = []
        self._wallclock_failure_reasons = []
        ranges = self._speech_ranges(audio)
        configured_max_chunk_s = _env_float(
            "LOCALSCRIBE_SENSEVOICE_WALLCLOCK_MAX_CHUNK_S",
            25.0,
        )
        effective_max_chunk_s = max(0.5, configured_max_chunk_s)
        chunks = _merge_speech_ranges(
            ranges,
            max_chunk_s=effective_max_chunk_s,
            max_gap_s=_env_float("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_MAX_GAP_S", 0.75),
        )
        if not chunks:
            return [], None, {
                "timing_mode": "wallclock_vad_failed",
                "timing_reliable": False,
                "timing_reason": "SenseVoice 无时间戳，且本地 VAD 分块失败",
            }

        segments: list[Segment] = []
        detected_language: str | None = None
        recognized_ranges: list[tuple[float, float]] = []
        failed_ranges: list[tuple[float, float]] = []
        failure_reasons: list[dict[str, Any]] = []
        self._wallclock_attempted_ranges = list(chunks)
        with tempfile.TemporaryDirectory(prefix="localscribe-sensevoice-vad-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            for idx, (start, end) in enumerate(chunks):
                if on_progress and (idx == 0 or idx % 8 == 0 or idx + 1 == len(chunks)):
                    on_progress({
                        "stage": "sensevoice_wallclock_chunk",
                        "current": idx + 1,
                        "total": len(chunks),
                    })
                chunk_path = tmp_root / f"chunk_{idx:04d}.wav"
                try:
                    import soundfile as sf

                    data, sr = sf.read(str(audio), dtype="float32", start=int(start * 16000), stop=int(end * 16000))
                    sf.write(str(chunk_path), data, sr)
                except Exception:
                    failed_ranges.append((start, end))
                    failure_reasons.append({"start": start, "end": end, "reason": "chunk_audio_write_failed"})
                    continue
                try:
                    chunk_res = self._generate(model, chunk_path, options, sensevoice=True)
                except Exception:
                    failed_ranges.append((start, end))
                    failure_reasons.append({"start": start, "end": end, "reason": "chunk_inference_failed"})
                    continue
                if not isinstance(chunk_res, list):
                    chunk_res = [chunk_res]
                text_parts: list[str] = []
                for item in chunk_res:
                    if not isinstance(item, dict):
                        continue
                    detected_language = detected_language or item.get("language")
                    raw_text = item.get("text") or item.get("raw_text") or ""
                    text = _clean_text(str(raw_text), sensevoice=True)
                    if text:
                        text_parts.append(text)
                text = " ".join(text_parts).strip()
                if not text:
                    failed_ranges.append((start, end))
                    failure_reasons.append({"start": start, "end": end, "reason": "empty_transcript"})
                    continue
                speech_duration = sum(
                    max(0.0, min(end, speech_end) - max(start, speech_start))
                    for speech_start, speech_end in ranges
                )
                normalized_chars = len(normalize_recovery_text(text))
                min_chars_per_s = _env_float("LOCALSCRIBE_SENSEVOICE_COVERAGE_MIN_CHARS_PER_S", 0.75)
                chars_per_s = normalized_chars / max(speech_duration, 0.001)
                if normalized_chars == 0 or chars_per_s < min_chars_per_s:
                    failed_ranges.append((start, end))
                    failure_reasons.append({
                        "start": start,
                        "end": end,
                        "reason": "low_text_density",
                        "speech_duration_s": round(speech_duration, 3),
                        "normalized_chars": normalized_chars,
                        "chars_per_s": round(chars_per_s, 3),
                    })
                else:
                    recognized_ranges.append((start, end))
                    segments.extend(_split_text_with_coarse_timing(text, start, end))

        self._wallclock_recognized_ranges = recognized_ranges
        self._wallclock_failed_ranges = failed_ranges
        self._wallclock_failure_reasons = failure_reasons
        return segments, detected_language or options.language, {
            "timing_mode": "wallclock_vad_chunks",
            "timing_reliable": True,
            "timing_reason": "SenseVoice 未返回句级时间戳，已改用本地 VAD 真实音频时间分块回填",
            "wallclock_vad_ranges": len(ranges),
            "wallclock_vad_chunks": len(chunks),
            "wallclock_recognized_chunks": len(recognized_ranges),
            "wallclock_failed_chunks": len(failed_ranges),
            "wallclock_low_density_chunks": sum(
                1 for item in failure_reasons if item.get("reason") == "low_text_density"
            ),
            "wallclock_min_chars_per_s": _env_float("LOCALSCRIBE_SENSEVOICE_COVERAGE_MIN_CHARS_PER_S", 0.75),
            "wallclock_strict_coverage": False,
            "wallclock_anchor_max_chunk_s": effective_max_chunk_s,
            "wallclock_max_chunk_s": effective_max_chunk_s,
        }

    def _strict_coverage_manifest(self) -> dict[str, Any]:
        partition = _recovery_snapshot(
            [],
            self._wallclock_attempted_ranges,
            self._wallclock_recognized_ranges,
            self._wallclock_failed_ranges,
        )
        return {
            "coverage_schema_version": 2,
            "strict_probe_windows": [dict(item) for item in self._strict_coverage_windows],
            "strict_probe_windows_truncated": False,
            "strict_partition": {
                key: partition[key]
                for key in (
                    "attempted_ranges",
                    "recognized_ranges",
                    "failed_ranges",
                    "attempted_partition_sha256",
                    "recognized_partition_sha256",
                    "failed_partition_sha256",
                    "covered_count",
                    "failed_count",
                    "partition_valid",
                )
            },
        }

    def _run_sensevoice_strict_coverage_probe(
        self,
        model: Any,
        audio: Path,
        options: TranscribeOptions,
        on_progress: ProgressCallback | None,
        *,
        duration: float,
    ) -> dict[str, Any]:
        ranges = self._speech_ranges(audio)
        requested_max_chunk_s = _env_float(
            "LOCALSCRIBE_SENSEVOICE_STRICT_COVERAGE_MAX_CHUNK_S",
            1.5,
        )
        core_max_chunk_s = min(1.5, max(0.5, requested_max_chunk_s))
        requested_pad_s = _env_float(
            "LOCALSCRIBE_SENSEVOICE_STRICT_COVERAGE_CONTEXT_PAD_S",
            0.5,
        )
        context_pad_s = min(1.0, max(0.0, requested_pad_s))
        chunks = _merge_speech_ranges(
            ranges,
            max_chunk_s=core_max_chunk_s,
            max_gap_s=_env_float("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_MAX_GAP_S", 0.75),
        )
        self._wallclock_attempted_ranges = list(chunks)
        self._wallclock_recognized_ranges = []
        self._wallclock_failed_ranges = []
        self._wallclock_failure_reasons = []
        self._strict_coverage_windows = []
        if not chunks:
            return {
                "ok": False,
                "reason": "strict_probe_has_no_windows",
                "strict_core_max_chunk_s": core_max_chunk_s,
                "strict_decode_context_pad_s": context_pad_s,
            }

        import soundfile as sf

        info = sf.info(str(audio))
        sample_rate = int(info.samplerate)
        audio_frames = int(info.frames)
        available_duration = duration if duration > 0 else audio_frames / float(sample_rate)
        min_chars_per_s = _env_float("LOCALSCRIBE_SENSEVOICE_COVERAGE_MIN_CHARS_PER_S", 0.75)
        detected_language: str | None = None
        with tempfile.TemporaryDirectory(prefix="localscribe-sensevoice-strict-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            for idx, (core_start, core_end) in enumerate(chunks):
                if on_progress and (idx == 0 or idx % 16 == 0 or idx + 1 == len(chunks)):
                    on_progress({
                        "stage": "sensevoice_strict_coverage_probe",
                        "current": idx + 1,
                        "total": len(chunks),
                    })
                decode_start = max(0.0, core_start - context_pad_s)
                decode_end = min(available_duration, core_end + context_pad_s)
                window = {
                    "core_start": round(core_start, 3),
                    "core_end": round(core_end, 3),
                    "decode_start": round(decode_start, 3),
                    "decode_end": round(decode_end, 3),
                    "status": "failed",
                    "reason": "unknown",
                    "recognition_source": "strict_probe",
                }
                speech_duration = sum(
                    max(0.0, min(core_end, speech_end) - max(core_start, speech_start))
                    for speech_start, speech_end in ranges
                )
                window["speech_duration_s"] = round(speech_duration, 3)
                def decode_slice(slice_start: float, slice_end: float, suffix: str) -> str:
                    chunk_path = tmp_root / f"strict_{idx:04d}_{suffix}.wav"
                    start_frame = max(0, int(round(slice_start * sample_rate)))
                    end_frame = min(audio_frames, int(round(slice_end * sample_rate)))
                    data, read_rate = sf.read(
                        str(audio),
                        dtype="float32",
                        start=start_frame,
                        stop=end_frame,
                    )
                    sf.write(str(chunk_path), data, read_rate)
                    chunk_res = self._generate(model, chunk_path, options, sensevoice=True)
                    items = chunk_res if isinstance(chunk_res, list) else [chunk_res]
                    text_parts: list[str] = []
                    nonlocal detected_language
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        detected_language = detected_language or item.get("language")
                        raw_text = item.get("text") or item.get("raw_text") or ""
                        cleaned = _clean_text(str(raw_text), sensevoice=True)
                        if cleaned:
                            text_parts.append(cleaned)
                    return " ".join(text_parts).strip()

                try:
                    core_text = decode_slice(core_start, core_end, "core")
                    core_chars = len(normalize_recovery_text(core_text))
                    core_chars_per_s = core_chars / max(speech_duration, 0.001)
                    window["core_normalized_chars"] = core_chars
                    window["core_chars_per_s"] = round(core_chars_per_s, 3)
                    window["decode_start"] = round(core_start, 3)
                    window["decode_end"] = round(core_end, 3)
                    if not core_text or core_chars == 0:
                        window["reason"] = "empty_core_transcript"
                    elif core_chars_per_s >= min_chars_per_s:
                        window["normalized_chars"] = core_chars
                        window["chars_per_s"] = round(core_chars_per_s, 3)
                        window["status"] = "recognized"
                        window["reason"] = "core_density_ok"
                    elif context_pad_s > 0 and (decode_start < core_start or decode_end > core_end):
                        padded_text = decode_slice(decode_start, decode_end, "context")
                        padded_chars = len(normalize_recovery_text(padded_text))
                        padded_chars_per_s = padded_chars / max(speech_duration, 0.001)
                        window["decode_start"] = round(decode_start, 3)
                        window["decode_end"] = round(decode_end, 3)
                        window["normalized_chars"] = padded_chars
                        window["chars_per_s"] = round(padded_chars_per_s, 3)
                        if padded_chars > 0 and padded_chars_per_s >= min_chars_per_s:
                            window["status"] = "recognized"
                            window["reason"] = "context_density_ok_with_core_evidence"
                        else:
                            window["reason"] = "low_text_density"
                    else:
                        window["normalized_chars"] = core_chars
                        window["chars_per_s"] = round(core_chars_per_s, 3)
                        window["reason"] = "low_text_density"
                except Exception as exc:
                    window["reason"] = f"strict_probe_failed:{type(exc).__name__}"

                self._strict_coverage_windows.append(window)
                core = (core_start, core_end)
                if window["status"] == "recognized":
                    self._wallclock_recognized_ranges.append(core)
                else:
                    self._wallclock_failed_ranges.append(core)
                    self._wallclock_failure_reasons.append({
                        "start": core_start,
                        "end": core_end,
                        "reason": window["reason"],
                        "decode_start": window.get("decode_start"),
                        "decode_end": window.get("decode_end"),
                        "speech_duration_s": window.get("speech_duration_s"),
                        "normalized_chars": window.get("normalized_chars", 0),
                        "chars_per_s": window.get("chars_per_s", 0.0),
                    })

        return {
            "ok": True,
            "reason": "ok",
            "detected_language": detected_language,
            "strict_core_max_chunk_s_requested": requested_max_chunk_s,
            "strict_core_max_chunk_s": core_max_chunk_s,
            "strict_decode_context_pad_s_requested": requested_pad_s,
            "strict_decode_context_pad_s": context_pad_s,
            **self._strict_coverage_manifest(),
        }

    def _run_sensevoice_local_recovery(
        self,
        model: Any,
        audio: Path,
        options: TranscribeOptions,
        segments: list[Segment],
        *,
        duration: float,
        on_progress: ProgressCallback | None,
        normalization_error: str | None = None,
        normalization_language: str | None = None,
        normalization_profile: str | None = None,
        normalization_context_segments: list[Segment] | None = None,
    ) -> tuple[list[Segment], dict[str, Any]]:
        mode, requested_mode, mode_diagnostic = _sensevoice_local_recovery_mode()
        provider_name, provider_diagnostic = _sensevoice_local_recovery_provider()
        if provider_diagnostic:
            mode_diagnostic = provider_diagnostic
        if mode == "merge" and provider_name != "qwen3":
            mode = "audit"
            mode_diagnostic = "merge_requires_qwen3_independent_provider"
        if mode == "merge" and normalization_error:
            mode = "audit"
            mode_diagnostic = "merge_disabled_text_normalization_failed"

        pending_windows: list[dict[str, Any]] = []
        for start, end in self._wallclock_failed_ranges:
            reason = next(
                (
                    dict(item)
                    for item in self._wallclock_failure_reasons
                    if abs(float(item.get("start", -1.0)) - start) <= 0.001
                    and abs(float(item.get("end", -1.0)) - end) <= 0.001
                ),
                {"start": start, "end": end, "reason": "unknown"},
            )
            reason["start"] = start
            reason["end"] = end
            pending_windows.append(reason)
        groups = [
            {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "windows": [dict(item)],
            }
            for item in pending_windows
        ]

        before = _recovery_snapshot(
            segments,
            self._wallclock_attempted_ranges,
            self._wallclock_recognized_ranges,
            self._wallclock_failed_ranges,
        )
        stats: dict[str, Any] = {
            "mode": mode,
            "requested_mode": requested_mode,
            "diagnostic": mode_diagnostic,
            "pending_windows": len(pending_windows),
            "pending_groups": len(groups),
            "attempts": 0,
            "matched_existing": 0,
            "inserted": 0,
            "rejected": 0,
            "error": 0,
            "before": before,
            "after": dict(before),
            "details": [],
            "details_truncated": False,
            "provider": {
                "requested": provider_name,
                "available": False,
                "error": None,
            },
            "text_normalization": {
                "language": normalization_language or options.language or "zh",
                "profile": normalization_profile,
                "error": normalization_error or None,
            },
        }
        if mode == "off" or not groups:
            return segments, stats

        recovery_provider: Any | None = None
        provider_metadata: dict[str, Any] = {}
        provider_error: str | None = None
        if provider_name == "qwen3":
            recovery_provider, provider_metadata, provider_error = self._load_local_recovery_provider(provider_name)
            stats["provider"] = {
                "requested": provider_name,
                "available": recovery_provider is not None,
                "error": provider_error,
                **provider_metadata,
            }
            if mode == "merge" and recovery_provider is None:
                mode = "audit"
                stats["mode"] = mode
                stats["diagnostic"] = f"merge_downgraded:{provider_error or 'provider_unavailable'}"

        sample_rate = 0
        audio_frames = 0
        audio_error: str | None = None
        try:
            import soundfile as sf

            audio_info = sf.info(str(audio))
            sample_rate = int(audio_info.samplerate)
            audio_frames = int(audio_info.frames)
            if sample_rate != 16000:
                audio_error = f"unsupported_sample_rate:{sample_rate}"
            elif duration <= 0:
                duration = audio_frames / float(sample_rate)
        except Exception as exc:
            sf = None
            audio_error = f"audio_read_failed:{type(exc).__name__}"

        def same_range(left: tuple[float, float], right: tuple[float, float]) -> bool:
            return abs(left[0] - right[0]) <= 0.001 and abs(left[1] - right[1]) <= 0.001

        def move_group_to_recognized(group: dict[str, Any]) -> None:
            moved = [(float(item["start"]), float(item["end"])) for item in group["windows"]]
            self._wallclock_failed_ranges = [
                item for item in self._wallclock_failed_ranges
                if not any(same_range(item, accepted) for accepted in moved)
            ]
            self._wallclock_failure_reasons = [
                item for item in self._wallclock_failure_reasons
                if not any(
                    abs(float(item.get("start", -1.0)) - accepted[0]) <= 0.001
                    and abs(float(item.get("end", -1.0)) - accepted[1]) <= 0.001
                    for accepted in moved
                )
            ]
            for accepted in moved:
                if not any(same_range(item, accepted) for item in self._wallclock_recognized_ranges):
                    self._wallclock_recognized_ranges.append(accepted)
                for window in self._strict_coverage_windows:
                    core = (
                        float(window.get("core_start", -1.0)),
                        float(window.get("core_end", -1.0)),
                    )
                    if same_range(core, accepted):
                        window["status"] = "recognized"
                        window["reason"] = "local_recovery_confirmed"
                        window["recognition_source"] = "local_recovery"
            self._wallclock_recognized_ranges.sort()

        with tempfile.TemporaryDirectory(prefix="localscribe-sensevoice-recovery-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            for group_index, group in enumerate(groups):
                if on_progress:
                    on_progress({
                        "stage": "sensevoice_local_recovery",
                        "current": group_index + 1,
                        "total": len(groups),
                    })
                group_start = float(group["start"])
                group_end = float(group["end"])
                context = local_reference_from_segments(segments, group_start, group_end)
                min_required_chars = max(
                    2,
                    int(math.ceil(max(0.0, group_end - group_start) * _env_float(
                        "LOCALSCRIBE_SENSEVOICE_COVERAGE_MIN_CHARS_PER_S", 0.75
                    ))),
                )
                attempts: list[dict[str, Any]] = []
                attempted_framings: set[tuple[str, str]] = set()

                def run_framing(
                    framing: str,
                    pad_s: float,
                    *,
                    provider: Any | None = None,
                    metadata: dict[str, Any] | None = None,
                    unavailable_error: str | None = None,
                ) -> None:
                    evidence_metadata = metadata or {
                        "provider_id": "sensevoice-primary",
                        "provider_kind": "primary_asr",
                        "model_id": options.model_id or SENSEVOICE_MODEL,
                        "model_family": "sensevoice",
                    }
                    attempt_key = (str(evidence_metadata.get("provider_id") or ""), framing)
                    if attempt_key in attempted_framings:
                        return
                    attempted_framings.add(attempt_key)
                    stats["attempts"] += 1
                    candidate_kwargs = {
                        "provider_id": str(evidence_metadata.get("provider_id") or ""),
                        "provider_kind": str(evidence_metadata.get("provider_kind") or ""),
                        "model_id": str(evidence_metadata.get("model_id") or ""),
                        "model_family": str(evidence_metadata.get("model_family") or ""),
                    }
                    if unavailable_error:
                        attempts.append(analyze_recovery_candidate(
                            "",
                            framing=framing,
                            pad_s=pad_s,
                            left=context["left"],
                            right=context["right"],
                            local_reference=context["reference"],
                            error=unavailable_error,
                            min_required_chars=min_required_chars,
                            **candidate_kwargs,
                        ))
                        return
                    if audio_error or sf is None or sample_rate <= 0:
                        attempts.append(analyze_recovery_candidate(
                            "",
                            framing=framing,
                            pad_s=pad_s,
                            left=context["left"],
                            right=context["right"],
                            local_reference=context["reference"],
                            error=audio_error or "audio_unavailable",
                            min_required_chars=min_required_chars,
                            **candidate_kwargs,
                        ))
                        return

                    available_duration = audio_frames / float(sample_rate)
                    slice_start = max(0.0, group_start - pad_s)
                    slice_end = min(available_duration, group_end + pad_s)
                    if slice_end <= slice_start:
                        attempt = analyze_recovery_candidate(
                            "",
                            framing=framing,
                            pad_s=pad_s,
                            left=context["left"],
                            right=context["right"],
                            local_reference=context["reference"],
                            error="empty_audio_slice",
                            min_required_chars=min_required_chars,
                            **candidate_kwargs,
                        )
                        attempt.update({"slice_start": slice_start, "slice_end": slice_end})
                        attempts.append(attempt)
                        return

                    provider_slug = str(evidence_metadata.get("provider_id") or "provider").replace("-", "_")
                    chunk_path = tmp_root / f"group_{group_index:04d}_{provider_slug}_{framing.replace('.', '_')}.wav"
                    start_sample = max(0, int(round(slice_start * sample_rate)))
                    end_sample = min(audio_frames, int(round(slice_end * sample_rate)))
                    slice_sha256 = ""
                    provider_stats: dict[str, Any] = {}
                    try:
                        slice_data, read_rate = sf.read(
                            str(audio),
                            dtype="float32",
                            start=start_sample,
                            stop=end_sample,
                            always_2d=True,
                        )
                        if int(read_rate) != sample_rate:
                            raise ValueError("slice_sample_rate_mismatch")
                        if slice_data.shape[1] > 1:
                            slice_data = slice_data.mean(axis=1)
                        else:
                            slice_data = slice_data[:, 0]
                        slice_sha256 = hashlib.sha256(slice_data.tobytes()).hexdigest()
                        sf.write(str(chunk_path), slice_data, sample_rate)
                        if provider is None:
                            raw_result = self._generate(model, chunk_path, options, sensevoice=True)
                            raw_items = raw_result if isinstance(raw_result, list) else [raw_result]
                            raw_parts: list[str] = []
                            for item in raw_items:
                                if isinstance(item, dict):
                                    raw_text = item.get("text") or item.get("raw_text") or ""
                                else:
                                    raw_text = item if isinstance(item, str) else ""
                                if raw_text:
                                    raw_parts.append(str(raw_text))
                            raw = " ".join(raw_parts).strip()
                        else:
                            raw, provider_stats = self._run_local_recovery_provider(
                                provider, chunk_path, options
                            )
                        attempt = analyze_recovery_candidate(
                            raw,
                            framing=framing,
                            pad_s=pad_s,
                            left=context["left"],
                            right=context["right"],
                            local_reference=context["reference"],
                            min_required_chars=min_required_chars,
                            hallucination_risk=bool(provider_stats.get("has_hallucination_risk")),
                            **candidate_kwargs,
                        )
                    except Exception as exc:
                        attempt = analyze_recovery_candidate(
                            "",
                            framing=framing,
                            pad_s=pad_s,
                            left=context["left"],
                            right=context["right"],
                            local_reference=context["reference"],
                            error=f"inference_failed:{type(exc).__name__}",
                            min_required_chars=min_required_chars,
                            **candidate_kwargs,
                        )
                    attempt.update({
                        "slice_start": round(slice_start, 3),
                        "slice_end": round(slice_end, 3),
                        "slice_sha256": slice_sha256,
                        "model_revision": evidence_metadata.get("model_revision"),
                        "config_sha256": evidence_metadata.get("config_sha256"),
                        "weights_manifest_sha256": evidence_metadata.get("weights_manifest_sha256"),
                    })
                    attempt["evidence_sha256"] = hashlib.sha256(
                        json.dumps(
                            {
                                "provider_id": attempt.get("provider_id"),
                                "model_id": attempt.get("model_id"),
                                "model_revision": attempt.get("model_revision"),
                                "config_sha256": attempt.get("config_sha256"),
                                "weights_manifest_sha256": attempt.get("weights_manifest_sha256"),
                                "slice_sha256": slice_sha256,
                                "raw": attempt.get("raw"),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    attempts.append(attempt)

                run_framing("exact", 0.0)
                run_framing("pad0.5", 0.5)
                decision = decide_recovery_attempts(attempts)
                qwen_framing = "pad1.0"
                qwen_pad_s = 1.0
                if not decision.get("primary_consensus"):
                    run_framing("pad1.0", 1.0)
                    decision = decide_recovery_attempts(attempts)
                if not decision.get("primary_consensus"):
                    run_framing("pad2.0", 2.0)
                    decision = decide_recovery_attempts(attempts)
                    qwen_framing = "pad2.0"
                    qwen_pad_s = 2.0
                primary_has_candidate = any(
                    attempt.get("provider_kind") == "primary_asr"
                    and attempt.get("status") in {"matched_existing", "valid"}
                    for attempt in attempts
                )
                if provider_name == "qwen3" and (
                    mode == "audit"
                    or bool(decision.get("primary_consensus"))
                    or primary_has_candidate
                ):
                    run_framing(
                        qwen_framing,
                        qwen_pad_s,
                        provider=recovery_provider,
                        metadata=provider_metadata,
                        unavailable_error=provider_error if recovery_provider is None else None,
                    )
                    decision = decide_recovery_attempts(attempts)

                evidence_decision = str(decision["decision"])
                decision_name = evidence_decision
                inserted_raw_text = str(decision.get("inserted_text") or "")
                inserted_text = ""
                insertion_normalization: dict[str, Any] | None = None
                normalization_rejection_reason: str | None = None
                if decision_name == "insert_accepted":
                    try:
                        raw_context = [
                            Segment(
                                start=segment.start,
                                end=segment.end,
                                text=segment.original_text or segment.text,
                                original_text=segment.original_text,
                                speaker=segment.speaker,
                                sync_cues=segment.sync_cues,
                            )
                            for segment in (normalization_context_segments or segments)
                        ]
                        raw_context.append(Segment(
                            start=group_start,
                            end=group_end,
                            text=inserted_raw_text,
                            original_text=inserted_raw_text,
                        ))
                        raw_context.sort(key=lambda segment: (segment.start, segment.end))
                        normalized_context, insertion_normalization = normalize_segments(
                            raw_context,
                            language=normalization_language or options.language or "zh",
                            profile=normalization_profile,
                        )
                        normalized_insertions = [
                            segment
                            for segment in normalized_context
                            if abs(segment.start - group_start) <= 0.001
                            and abs(segment.end - group_end) <= 0.001
                        ]
                        if len(normalized_insertions) != 1:
                            raise ValueError("recovery_normalizer_changed_candidate_identity")
                        inserted_text = normalized_insertions[0].text
                        if len(normalize_recovery_text(inserted_text)) < 2:
                            raise ValueError("recovery_normalizer_removed_candidate")
                    except Exception as exc:
                        decision_name = "rejected"
                        inserted_text = ""
                        normalization_rejection_reason = f"{type(exc).__name__}:{exc}"
                        insertion_normalization = {
                            "mode": "local_text_normalizer",
                            "error": normalization_rejection_reason,
                        }
                counter = {
                    "matched_existing": "matched_existing",
                    "insert_accepted": "inserted",
                    "rejected": "rejected",
                    "error": "error",
                }[decision_name]
                stats[counter] += 1

                if mode == "merge" and decision_name in {"matched_existing", "insert_accepted"}:
                    if decision_name == "insert_accepted" and len(normalize_recovery_text(inserted_text)) >= 2:
                        segments.append(Segment(
                            start=group_start,
                            end=group_end,
                            text=inserted_text,
                            original_text=inserted_raw_text,
                        ))
                        segments.sort(key=lambda segment: (segment.start, segment.end))
                    move_group_to_recognized(group)

                stats["details"].append({
                        "start": round(group_start, 3),
                        "end": round(group_end, 3),
                        "window_count": len(group["windows"]),
                        "original_failures": group["windows"],
                        "left_context": context["left"],
                        "right_context": context["right"],
                        "overlapping_context": context["overlapping"],
                        "local_reference": context["reference"],
                        "min_required_chars": min_required_chars,
                        "attempts": attempts,
                        "evidence_decision": evidence_decision,
                        "decision": decision_name,
                        "normalization_rejection_reason": normalization_rejection_reason,
                        "consensus": str(decision.get("consensus") or ""),
                        "evidence_framings": list(decision.get("evidence_framings") or []),
                        "evidence_providers": list(decision.get("evidence_providers") or []),
                        "evidence_models": list(decision.get("evidence_models") or []),
                        "evidence_ids": list(decision.get("evidence_ids") or []),
                        "primary_status": str(decision.get("primary_status") or ""),
                        "primary_consensus": str(decision.get("primary_consensus") or ""),
                        "primary_evidence_framings": list(decision.get("primary_evidence_framings") or []),
                        "inserted_raw_text": inserted_raw_text,
                        "inserted_text": inserted_text,
                        "insertion_normalization": insertion_normalization,
                })

        stats["after"] = _recovery_snapshot(
            segments,
            self._wallclock_attempted_ranges,
            self._wallclock_recognized_ranges,
            self._wallclock_failed_ranges,
        )
        if not stats["after"]["partition_valid"]:
            stats["diagnostic"] = "wallclock_partition_invariant_failed"
        return segments, stats

    def _finalize_transcription_segments(self, segments: list[Segment]) -> None:
        coverage = (getattr(self, "last_filter_stats", {}) or {}).get("speech_coverage")
        if not isinstance(coverage, dict):
            return
        recovery = coverage.get("local_recovery")
        if not isinstance(recovery, dict):
            return

        before_snapshot = recovery.get("before") if isinstance(recovery.get("before"), dict) else {}
        before_attempted = [
            (float(item[0]), float(item[1]))
            for item in before_snapshot.get("attempted_ranges", [])
            if isinstance(item, list) and len(item) == 2
        ]
        before_recognized = [
            (float(item[0]), float(item[1]))
            for item in before_snapshot.get("recognized_ranges", [])
            if isinstance(item, list) and len(item) == 2
        ]
        before_failed = [
            (float(item[0]), float(item[1]))
            for item in before_snapshot.get("failed_ranges", [])
            if isinstance(item, list) and len(item) == 2
        ]

        base_segments = list(segments)
        if recovery.get("mode") == "merge":
            for detail in recovery.get("details") or []:
                if not isinstance(detail, dict) or detail.get("decision") != "insert_accepted":
                    continue
                start = float(detail.get("start", -1.0))
                end = float(detail.get("end", -1.0))
                inserted_text = str(detail.get("inserted_text") or "")
                inserted_raw_text = str(detail.get("inserted_raw_text") or "")
                match_index = next(
                    (
                        index
                        for index, segment in enumerate(base_segments)
                        if abs(segment.start - start) <= 0.001
                        and abs(segment.end - end) <= 0.001
                        and str(segment.text or "") == inserted_text
                        and str(segment.original_text or "") == inserted_raw_text
                    ),
                    None,
                )
                if match_index is None:
                    recovery["diagnostic"] = "post_normalization_inserted_segment_unresolved"
                    continue
                detail["inserted_text"] = base_segments[match_index].text
                base_segments.pop(match_index)

        context_segments = base_segments if recovery.get("mode") == "merge" else segments
        for detail in recovery.get("details") or []:
            if not isinstance(detail, dict):
                continue
            context = local_reference_from_segments(
                context_segments,
                float(detail.get("start", 0.0)),
                float(detail.get("end", 0.0)),
            )
            detail.update({
                "left_context": context["left"],
                "right_context": context["right"],
                "overlapping_context": context["overlapping"],
                "local_reference": context["reference"],
            })

        if recovery.get("mode") in {"off", "audit"}:
            current_snapshot = _recovery_snapshot(
                segments,
                self._wallclock_attempted_ranges,
                self._wallclock_recognized_ranges,
                self._wallclock_failed_ranges,
            )
            expected_snapshot = (
                recovery.get("after") if isinstance(recovery.get("after"), dict) else {}
            )
            partition_fields = (
                "attempted_ranges",
                "recognized_ranges",
                "failed_ranges",
                "attempted_partition_sha256",
                "recognized_partition_sha256",
                "failed_partition_sha256",
                "partition_valid",
            )
            partition_changed = any(
                expected_snapshot.get(field) != current_snapshot.get(field)
                for field in partition_fields
            )
            recovery["post_normalization"] = current_snapshot
            recovery["normalization_changed_segments"] = (
                expected_snapshot.get("segment_count") != current_snapshot.get("segment_count")
                or expected_snapshot.get("text_sha256") != current_snapshot.get("text_sha256")
            )
            recovery["partition_preserved_after_normalization"] = not partition_changed
            if partition_changed:
                # Auditing must never discard a completed transcript. Surface
                # the invariant failure for review and preserve primary output.
                recovery["diagnostic"] = "audit_local_recovery_partition_changed_after_snapshot"
            return

        recovery["before"] = _recovery_snapshot(
            base_segments,
            before_attempted,
            before_recognized,
            before_failed,
        )
        recovery["after"] = _recovery_snapshot(
            segments,
            self._wallclock_attempted_ranges,
            self._wallclock_recognized_ranges,
            self._wallclock_failed_ranges,
        )

    def _run_paraformer_timing_anchor(
        self,
        audio: Path,
        options: TranscribeOptions,
        on_progress: ProgressCallback | None,
        *,
        force_cached: bool = False,
    ) -> tuple[list[Segment], str | None, dict[str, Any]]:
        if not force_cached and not _env_flag("LOCALSCRIBE_SENSEVOICE_PARAFORMER_ANCHOR", False):
            return [], None, {
                "paraformer_anchor_enabled": False,
                "paraformer_anchor_reason": "disabled",
            }
        if not model_cached(DEFAULT_MODEL) and not _env_flag("LOCALSCRIBE_ALLOW_MODEL_DOWNLOAD", False):
            return [], None, {
                "paraformer_anchor_enabled": True,
                "paraformer_anchor_ok": False,
                "paraformer_anchor_reason": "paraformer_model_not_cached",
            }
        if on_progress:
            on_progress({"stage": "paraformer_timing_anchor", "backend": self.backend, "model": DEFAULT_MODEL})
        preserved_state = {
            name: getattr(self, name, None)
            for name in ("_model", "_loaded_model_id", "_resolved_model_id", "_resolved_vad_model")
        }
        try:
            anchor_model = self._load(DEFAULT_MODEL)
            anchor_options = TranscribeOptions(
                language=options.language,
                model_id=DEFAULT_MODEL,
                initial_prompt=options.initial_prompt,
                hotwords=options.hotwords,
                audio_preprocess=options.audio_preprocess,
            )
            anchor_raw = self._generate(anchor_model, audio, anchor_options, sensevoice=False)
        except Exception as exc:
            return [], None, {
                "paraformer_anchor_enabled": True,
                "paraformer_anchor_ok": False,
                "paraformer_anchor_reason": f"anchor_failed:{type(exc).__name__}",
            }
        finally:
            for name, value in preserved_state.items():
                setattr(self, name, value)
        anchor_items = anchor_raw if isinstance(anchor_raw, list) else [anchor_raw]
        anchor_segments, anchor_language = _segments_from_generate_items(anchor_items, sensevoice=False)
        has_timing = _items_have_timing(anchor_items)
        anchor_quality = _paraformer_anchor_quality(anchor_items, anchor_segments)
        ok = bool(anchor_segments and has_timing and anchor_quality["usable"])
        if ok:
            self._remember_strong_asr_detector(
                anchor_segments,
                source=(
                    "paraformer_recovery_timing_anchor"
                    if force_cached
                    else "paraformer_timing_anchor"
                ),
                is_paraformer=True,
            )
        return (anchor_segments if ok else []), anchor_language, {
            "paraformer_anchor_enabled": True,
            "paraformer_anchor_mode": "recovery" if force_cached else "explicit",
            "paraformer_anchor_ok": ok,
            "paraformer_anchor_segments": len(anchor_segments),
            "paraformer_anchor_has_timing": bool(has_timing),
            "paraformer_anchor_timing_precision": anchor_quality["precision"],
            "paraformer_anchor_quality": anchor_quality,
            "paraformer_anchor_reason": (
                "ok"
                if ok
                else (
                    str(anchor_quality["reason"])
                    if has_timing
                    else "no_timestamp_anchor"
                )
            ),
        }

    def _run(
        self,
        audio: Path,
        options: TranscribeOptions,
        on_progress: ProgressCallback | None,
    ) -> tuple[list[Segment], str | None]:
        model_id = options.model_id or DEFAULT_MODEL
        sensevoice = _is_sensevoice(model_id)
        coverage_enabled = sensevoice and _env_flag("LOCALSCRIBE_SENSEVOICE_SPEECH_COVERAGE", True)
        self._speech_ranges_cache_path = None
        self._speech_ranges_cache = []
        self._speech_ranges_status = "unavailable"
        self._speech_ranges_reason = "not_run"
        self._wallclock_attempted_ranges = []
        self._wallclock_recognized_ranges = []
        self._wallclock_failed_ranges = []
        self._wallclock_failure_reasons = []
        self._strict_coverage_windows = []
        self._strong_asr_detector_segments = []
        self._strong_asr_detector_source = ""
        self._strong_asr_detector_is_paraformer = False
        speech_ranges = self._speech_ranges(audio) if coverage_enabled else []
        if on_progress:
            on_progress({"stage": "loading_model", "backend": self.backend, "model": model_id})
        model = self._load(model_id)

        if on_progress:
            on_progress({"stage": "transcribing", "backend": self.backend, "model": model_id})

        segments: list[Segment] = []
        detected_language: str | None = None
        used_wallclock_fallback = False
        used_timing_alignment = False
        timing_stats: dict[str, Any] = {}
        res: list[Any] = []

        timing_align_requested = (
            bool(options.timing_align)
            if options.timing_align is not None
            else _env_flag("LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN", False)
        )
        timing_align_enabled = sensevoice and timing_align_requested
        explicit_wallclock_enabled = sensevoice and _env_flag("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD", False)
        wallclock_enabled = explicit_wallclock_enabled or timing_align_enabled
        tried_wallclock_first = (
            explicit_wallclock_enabled
            and _env_flag("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD_FIRST", False)
        )
        if tried_wallclock_first:
            wallclock_segments, wallclock_language, timing_stats = self._run_sensevoice_wallclock_vad(
                model,
                audio,
                options,
                on_progress,
            )
            if wallclock_segments:
                self._remember_strong_asr_detector(
                    wallclock_segments,
                    source="sensevoice_wallclock_anchor",
                    is_paraformer=False,
                )
                segments = wallclock_segments
                detected_language = wallclock_language
                used_wallclock_fallback = True

        if not used_wallclock_fallback:
            res_raw = self._generate(model, audio, options, sensevoice=sensevoice)
            if not isinstance(res_raw, list):
                res = [res_raw]
            else:
                res = res_raw

        source_has_timing = _items_have_timing(res)

        if not used_wallclock_fallback:
            timing_stats = {
                "timing_mode": "model_timestamps" if source_has_timing else "coarse_text_distribution",
                "timing_reliable": bool(source_has_timing),
                "timing_reason": (
                    "模型返回了句级/字级时间戳"
                    if source_has_timing
                    else "模型未返回时间戳，只能按文本长度粗略分配时间"
                ),
            }

        if not used_wallclock_fallback:
            segments, parsed_language = _segments_from_generate_items(res, sensevoice=sensevoice)
            detected_language = detected_language or parsed_language

        try:
            from .audio import probe_audio

            duration = float(probe_audio(audio).get("duration") or 0.0)
        except Exception:
            duration = 0.0

        paraformer_preflight = _paraformer_timing_preflight(segments, duration)
        paraformer_env = os.environ.get("LOCALSCRIBE_SENSEVOICE_PARAFORMER_ANCHOR")
        prefer_cached_paraformer = bool(
            timing_align_enabled
            and paraformer_env is None
            and paraformer_preflight.get("selected")
            and model_cached(DEFAULT_MODEL)
        )
        paraformer_attempted = False

        if (
            wallclock_enabled
            and not used_wallclock_fallback
            and not source_has_timing
            and not tried_wallclock_first
        ):
            anchor_stats: dict[str, Any] = {
                "paraformer_preflight": paraformer_preflight,
            }
            if timing_align_enabled and segments:
                explicit_paraformer = _env_flag(
                    "LOCALSCRIBE_SENSEVOICE_PARAFORMER_ANCHOR",
                    False,
                )
                if prefer_cached_paraformer or explicit_paraformer:
                    paraformer_attempted = True
                    anchor_segments, anchor_language, generated_anchor_stats = (
                        self._run_paraformer_timing_anchor(
                            audio,
                            options,
                            on_progress,
                            force_cached=prefer_cached_paraformer,
                        )
                    )
                    anchor_stats.update(generated_anchor_stats)
                else:
                    anchor_segments, anchor_language = [], None
                    anchor_stats.update({
                        "paraformer_anchor_enabled": False,
                        "paraformer_anchor_reason": "disabled",
                    })
                if anchor_segments:
                    aligned_segments, align_stats = _align_segments_to_timing_anchor(
                        segments,
                        anchor_segments,
                        min_equal_ratio=_env_float("LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN_MIN_RATIO", 0.55),
                    )
                    if aligned_segments:
                        segments = aligned_segments
                        detected_language = detected_language or anchor_language
                        used_timing_alignment = True
                        timing_stats = {
                            **timing_stats,
                            **anchor_stats,
                            **align_stats,
                            "timing_mode": (
                                "aligned_to_paraformer_recovery_anchor"
                                if prefer_cached_paraformer
                                else "aligned_to_paraformer_timestamp_anchor"
                            ),
                            "timing_reliable": True,
                            "timing_reason": (
                                "正文密度预检显示分块时间锚点风险较高，已直接使用本地 Paraformer 字级时间轴恢复对齐"
                                if prefer_cached_paraformer
                                else (
                                    "SenseVoice 文本未返回时间戳，已用本地 Paraformer 字级时间轴"
                                    "做文本对齐，生成飞书式短语同步锚点"
                                )
                            ),
                        }
                    else:
                        timing_stats = {**timing_stats, **anchor_stats, **align_stats}
                else:
                    timing_stats = {**timing_stats, **anchor_stats}

            if not used_timing_alignment:
                wallclock_segments, wallclock_language, wallclock_stats = self._run_sensevoice_wallclock_vad(
                    model,
                    audio,
                    options,
                    on_progress,
                )
                self._remember_strong_asr_detector(
                    wallclock_segments,
                    source="sensevoice_wallclock_anchor",
                    is_paraformer=False,
                )
            else:
                wallclock_segments, wallclock_language, wallclock_stats = [], None, {}

            if not used_timing_alignment and timing_align_enabled and segments:
                if wallclock_segments:
                    aligned_segments, align_stats = _align_segments_to_timing_anchor(
                        segments,
                        wallclock_segments,
                        min_equal_ratio=_env_float(
                            "LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN_MIN_RATIO",
                            0.55,
                        ),
                    )
                else:
                    aligned_segments = []
                    align_stats = {
                        "timing_alignment_ok": False,
                        "timing_alignment_reason": "wallclock_anchor_empty",
                        "equal_char_ratio": 0.0,
                    }
                paraformer_recovery_stats: dict[str, Any] = {}
                paraformer_recovery_language: str | None = None
                if (
                    not aligned_segments
                    and not paraformer_attempted
                    and paraformer_env is None
                    and model_cached(DEFAULT_MODEL)
                ):
                    paraformer_segments, paraformer_recovery_language, paraformer_anchor_stats = (
                        self._run_paraformer_timing_anchor(
                            audio,
                            options,
                            on_progress,
                            force_cached=True,
                        )
                    )
                    paraformer_recovery_stats = {
                        "wallclock_alignment_ok": False,
                        "wallclock_equal_char_ratio": align_stats.get("equal_char_ratio"),
                        "wallclock_alignment_reason": align_stats.get("timing_alignment_reason"),
                        **paraformer_anchor_stats,
                    }
                    if paraformer_segments:
                        aligned_segments, paraformer_align_stats = _align_segments_to_timing_anchor(
                            segments,
                            paraformer_segments,
                            min_equal_ratio=_env_float(
                                "LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN_MIN_RATIO",
                                0.55,
                            ),
                        )
                        paraformer_recovery_stats = {
                            **paraformer_recovery_stats,
                            **paraformer_align_stats,
                            "timing_alignment_reason": paraformer_align_stats.get(
                                "timing_alignment_reason"
                            ),
                        }
                if aligned_segments:
                    segments = aligned_segments
                    detected_language = (
                        detected_language
                        or paraformer_recovery_language
                        or wallclock_language
                    )
                    used_timing_alignment = True
                    recovered_with_paraformer = bool(
                        paraformer_recovery_stats.get("timing_alignment_ok")
                    )
                    timing_stats = {
                        **wallclock_stats,
                        **align_stats,
                        **anchor_stats,
                        **paraformer_recovery_stats,
                        "paraformer_preflight": paraformer_preflight,
                        "timing_mode": (
                            "aligned_to_paraformer_recovery_anchor"
                            if recovered_with_paraformer
                            else "aligned_to_wallclock_anchor"
                        ),
                        "timing_reliable": True,
                        "timing_reason": (
                            "VAD/SenseVoice 时间锚点不足，已自动使用本地 Paraformer 字级时间轴恢复对齐"
                            if recovered_with_paraformer
                            else (
                                "SenseVoice 全段文字未返回句级时间戳，已用本地 VAD/SenseVoice 分块结果"
                                "做字符级时间锚点重对齐"
                            )
                        ),
                    }
                elif explicit_wallclock_enabled and wallclock_segments:
                    segments = wallclock_segments
                    detected_language = wallclock_language
                    used_wallclock_fallback = True
                    timing_stats = {**wallclock_stats, **align_stats, **anchor_stats}
                else:
                    timing_stats = {
                        **timing_stats,
                        **wallclock_stats,
                        **align_stats,
                        **anchor_stats,
                        "paraformer_preflight": paraformer_preflight,
                        "timing_mode": "coarse_text_distribution",
                        "timing_reliable": False,
                        "timing_reason": "时间锚点文本相似度不足，回退为按文本长度粗略分配时间",
                    }
            elif explicit_wallclock_enabled and wallclock_segments:
                segments = wallclock_segments
                detected_language = wallclock_language
                used_wallclock_fallback = True
                timing_stats = wallclock_stats
            else:
                timing_stats = {
                    **timing_stats,
                    **wallclock_stats,
                    "timing_mode": timing_stats.get("timing_mode") or "coarse_text_distribution",
                    "timing_reliable": bool(timing_stats.get("timing_reliable")),
                }

        if not used_wallclock_fallback and not used_timing_alignment:
            segments = _coarsen_zero_timing(segments, duration)

        local_recovery_stats: dict[str, Any] | None = None
        strict_coverage_requested = coverage_enabled and _env_flag(
            "LOCALSCRIBE_SENSEVOICE_STRICT_COVERAGE", False
        )
        strict_probe_stats: dict[str, Any] = {}
        if strict_coverage_requested and self._speech_ranges_status == "ok":
            try:
                strict_probe_stats = self._run_sensevoice_strict_coverage_probe(
                    model,
                    audio,
                    options,
                    on_progress,
                    duration=duration,
                )
            except Exception as exc:
                self._wallclock_attempted_ranges = []
                self._wallclock_recognized_ranges = []
                self._wallclock_failed_ranges = []
                self._wallclock_failure_reasons = []
                self._strict_coverage_windows = []
                strict_probe_stats = {
                    "ok": False,
                    "reason": f"strict_probe_failed:{type(exc).__name__}",
                }

        if coverage_enabled:
            if self._speech_ranges_status == "no_speech":
                coverage_segments = []
                coverage_basis = "silero_vad_no_speech"
                coverage_status = "no_speech"
                coverage_reason = self._speech_ranges_reason
                coverage_collar_s = 0.0
            elif strict_coverage_requested:
                coverage_segments = [
                    Segment(start=start, end=end, text="recognized")
                    for start, end in self._wallclock_recognized_ranges
                ]
                if strict_probe_stats.get("ok") and self._wallclock_attempted_ranges:
                    coverage_basis = "wallclock_strict_windows"
                    coverage_status = self._speech_ranges_status
                    coverage_reason = self._speech_ranges_reason
                else:
                    coverage_basis = "unavailable"
                    coverage_status = "unavailable"
                    coverage_reason = str(strict_probe_stats.get("reason") or "strict_probe_unavailable")
                coverage_collar_s = 0.0
            elif self._wallclock_attempted_ranges:
                coverage_segments = [
                    Segment(start=start, end=end, text="recognized")
                    for start, end in self._wallclock_recognized_ranges
                ]
                coverage_basis = "wallclock_recognized_chunks"
                coverage_status = self._speech_ranges_status
                coverage_reason = self._speech_ranges_reason
                coverage_collar_s = 0.0
            elif source_has_timing and bool(timing_stats.get("timing_reliable")):
                coverage_segments = segments
                coverage_basis = "model_timestamps"
                coverage_status = self._speech_ranges_status
                coverage_reason = self._speech_ranges_reason
                coverage_collar_s = _env_float("LOCALSCRIBE_SENSEVOICE_SPEECH_COVERAGE_COLLAR_S", 0.5)
            else:
                coverage_segments = []
                coverage_basis = "unavailable"
                coverage_status = "unavailable"
                coverage_reason = "recognition_coverage_evidence_unavailable"
                coverage_collar_s = 0.0
            speech_coverage = _speech_coverage_diagnostics(
                speech_ranges,
                coverage_segments,
                duration=duration,
                collar_s=coverage_collar_s,
                vad_status=coverage_status,
                vad_reason=coverage_reason,
            )
            speech_coverage.update({
                "basis": coverage_basis,
                "wallclock_attempted_chunks": len(self._wallclock_attempted_ranges),
                "wallclock_recognized_chunks": len(self._wallclock_recognized_ranges),
                "wallclock_failed_chunks": len(self._wallclock_failed_ranges),
                "wallclock_failed_ranges": [
                    {"start": round(start, 3), "end": round(end, 3)}
                    for start, end in self._wallclock_failed_ranges
                ],
                "wallclock_failure_reasons": list(self._wallclock_failure_reasons),
                "wallclock_failure_details_truncated": False,
                "wallclock_min_chars_per_s": _env_float(
                    "LOCALSCRIBE_SENSEVOICE_COVERAGE_MIN_CHARS_PER_S", 0.75
                ),
                "wallclock_strict_coverage": _env_flag(
                    "LOCALSCRIBE_SENSEVOICE_STRICT_COVERAGE", False
                ),
                "wallclock_max_chunk_s": (
                    strict_probe_stats.get("strict_core_max_chunk_s")
                    if strict_coverage_requested
                    else timing_stats.get("wallclock_max_chunk_s")
                ),
            })
            if strict_coverage_requested:
                speech_coverage.update({
                    key: value
                    for key, value in strict_probe_stats.items()
                    if key not in {"ok", "reason", "detected_language"}
                })
                speech_coverage.update(self._strict_coverage_manifest())
        else:
            speech_coverage = _speech_coverage_diagnostics(
                [],
                [],
                duration=duration,
                vad_status="disabled",
                vad_reason="speech_coverage_disabled",
            )
            speech_coverage["basis"] = "disabled"
        if local_recovery_stats is not None:
            speech_coverage["local_recovery"] = local_recovery_stats

        if on_progress:
            on_progress({"stage": "done", "segments": len(segments)})
        self.last_filter_stats = {
            "mode": "funasr",
            "raw_items": len(res),
            "model_family": "sensevoice" if sensevoice else "paraformer",
            **timing_stats,
            "speech_coverage": speech_coverage,
            "settings": {
                "punc_model": "ct-punc" if (not sensevoice and _env_flag("LOCALSCRIBE_FUNASR_PUNC", False)) else None,
                "sensevoice_merge_vad": _env_flag("LOCALSCRIBE_SENSEVOICE_MERGE_VAD", True) if sensevoice else None,
                "sensevoice_merge_length_s": _env_float("LOCALSCRIBE_SENSEVOICE_MERGE_LENGTH_S", 15) if sensevoice else None,
                "sensevoice_wallclock_vad": _env_flag("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD", False) if sensevoice else None,
                "sensevoice_wallclock_vad_first": _env_flag("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD_FIRST", False) if sensevoice else None,
                "sensevoice_timing_align": timing_align_requested if sensevoice else None,
                "sensevoice_paraformer_anchor": (
                    (
                        _env_flag("LOCALSCRIBE_SENSEVOICE_PARAFORMER_ANCHOR", False)
                        or bool(timing_stats.get("paraformer_anchor_enabled"))
                    )
                    if sensevoice
                    else None
                ),
                "sensevoice_timing_align_min_ratio": _env_float("LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN_MIN_RATIO", 0.55) if sensevoice else None,
                "sensevoice_speech_coverage": coverage_enabled if sensevoice else None,
                "sensevoice_strict_coverage": strict_coverage_requested if sensevoice else None,
                "sensevoice_strict_coverage_max_chunk_s": strict_probe_stats.get("strict_core_max_chunk_s") if sensevoice else None,
                "sensevoice_strict_coverage_context_pad_s": strict_probe_stats.get("strict_decode_context_pad_s") if sensevoice else None,
                "sensevoice_speech_coverage_collar_s": _env_float("LOCALSCRIBE_SENSEVOICE_SPEECH_COVERAGE_COLLAR_S", 0.5) if sensevoice else None,
                "sensevoice_local_recovery_mode": _sensevoice_local_recovery_mode()[0] if sensevoice else None,
                "sensevoice_local_recovery_provider": _sensevoice_local_recovery_provider()[0] if sensevoice else None,
                "hotwords": len(options.hotwords),
                "sensevoice_hotword": bool(options.hotwords or options.initial_prompt) if sensevoice else None,
                "funasr_inference_seed": _env_int("LOCALSCRIBE_FUNASR_INFERENCE_SEED", 0),
                "resolved_model_path": getattr(self, "_resolved_model_id", model_id),
                "resolved_vad_model_path": getattr(self, "_resolved_vad_model", "fsmn-vad"),
            },
        }
        return segments, detected_language or options.language

    def _post_normalize_transcription(
        self,
        segments: list[Segment],
        audio: Path,
        options: TranscribeOptions,
        on_progress: ProgressCallback | None,
    ) -> list[Segment]:
        if not _is_sensevoice(options.model_id or DEFAULT_MODEL):
            return segments
        try:
            from .audio import probe_audio

            duration = float(probe_audio(audio).get("duration") or 0.0)
        except Exception:
            duration = 0.0
        segments, recovery = self._run_sensevoice_local_recovery(
            self._model,
            audio,
            options,
            segments,
            duration=duration,
            on_progress=on_progress,
            normalization_error=getattr(self, "_text_normalization_error", "") or None,
            normalization_language=getattr(self, "_text_normalization_language", None),
            normalization_profile=getattr(self, "_text_normalization_profile", None),
            normalization_context_segments=list(
                getattr(self, "_text_normalization_context_segments", []) or []
            ),
        )
        segments, non_speech_suppression = _suppress_vad_unsupported_segments(
            segments,
            self._speech_ranges_cache,
            vad_status=self._speech_ranges_status,
        )
        self.last_filter_stats["non_speech_suppression"] = non_speech_suppression
        segments, sync_cue_realign = _realign_sync_cues_preserving_segments(
            segments,
            list(getattr(self, "_text_normalization_context_segments", []) or []),
        )
        self.last_filter_stats["sync_cue_realign"] = sync_cue_realign
        segments, sync_cue_guard = _guard_unreliable_sync_cues(
            segments,
            self._speech_ranges_cache,
            vad_status=self._speech_ranges_status,
        )
        self.last_filter_stats["sync_cue_guard"] = sync_cue_guard
        coverage = (self.last_filter_stats or {}).get("speech_coverage")
        if isinstance(coverage, dict) and self._wallclock_attempted_ranges:
            coverage_segments = [
                Segment(start=start, end=end, text="recognized")
                for start, end in self._wallclock_recognized_ranges
            ]
            updated = _speech_coverage_diagnostics(
                self._speech_ranges_cache,
                coverage_segments,
                duration=duration,
                collar_s=0.0,
                vad_status=self._speech_ranges_status,
                vad_reason=self._speech_ranges_reason,
            )
            updated.update({
                "basis": coverage.get("basis"),
                "wallclock_attempted_chunks": len(self._wallclock_attempted_ranges),
                "wallclock_recognized_chunks": len(self._wallclock_recognized_ranges),
                "wallclock_failed_chunks": len(self._wallclock_failed_ranges),
                "wallclock_failed_ranges": [
                    {"start": round(start, 3), "end": round(end, 3)}
                    for start, end in self._wallclock_failed_ranges
                ],
                "wallclock_failure_reasons": list(self._wallclock_failure_reasons),
                "wallclock_failure_details_truncated": False,
                "wallclock_min_chars_per_s": coverage.get("wallclock_min_chars_per_s"),
                "wallclock_strict_coverage": coverage.get("wallclock_strict_coverage"),
                "wallclock_max_chunk_s": coverage.get("wallclock_max_chunk_s"),
            })
            if coverage.get("basis") == "wallclock_strict_windows":
                updated.update({
                    key: value
                    for key, value in coverage.items()
                    if key.startswith("strict_") or key == "coverage_schema_version"
                })
                updated.update(self._strict_coverage_manifest())
            coverage.clear()
            coverage.update(updated)
        if isinstance(coverage, dict):
            coverage["local_recovery"] = recovery
        settings = (self.last_filter_stats or {}).get("settings")
        if isinstance(settings, dict):
            settings["sensevoice_local_recovery_mode"] = recovery.get("mode")
            settings["sensevoice_local_recovery_provider"] = (recovery.get("provider") or {}).get("requested")
        return segments
