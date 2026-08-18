"""Senko + CAM++ 中文说话人分离(替换原 resemblyzer + KMeans)。

为什么换:
- CAM++ 在中文上 DER ~13%(AISHELL-4),resemblyzer 主英文训练效果差
- senko 用 CoreML 加速 → 96 分钟音频 ~47 秒搞定(原方案分钟级)
- 输出 192 维声纹中心(原 256 维 resemblyzer),声纹库需要重新上传样本

为什么不用 senko 默认 umap_hdbscan:
- macOS 上 libomp 冲突会死锁(senko + scikit-learn / hdbscan 各自带 libomp)
- 强制走 spectral 即可,慢一点但稳定。这里所有长度的音频都走 spectral
"""
from __future__ import annotations

import atexit
import math
import os
import re
import subprocess
import tempfile
import wave
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# 缓解 macOS libomp 冲突 —— senko / sklearn / numpy 各自带 libomp 副本同时加载会卡死
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# 声纹库匹配阈值(senko 192d centroid 上的 cosine)
# 0.875 是 senko 推荐的同录音同人阈值,跨录音同人通常 0.70-0.85,这里取中段
MATCH_THRESHOLD = 0.75
MATCH_REVIEW_THRESHOLD = 0.70
MATCH_MARGIN = 0.04

# Cue-level matching compares CAM++ embeddings from the same recording.  Keep
# automatic reassignment stricter than the review band so mixed boundary cues
# and short acknowledgements do not become hard speaker labels.
CUE_MATCH_THRESHOLD = 0.72
CUE_MATCH_REVIEW_THRESHOLD = 0.64
CUE_MATCH_MARGIN = 0.06
CUE_MATCH_DISTINCT_THRESHOLD = 0.67
CUE_MATCH_DISTINCT_MARGIN = 0.12
CUE_MATCH_DISTINCT_COVERAGE_RATIO = 0.50
CUE_MATCH_MIN_COVERAGE_SECONDS = 0.30
CUE_MATCH_MIN_COVERAGE_RATIO = 0.15
CUE_MATCH_MAX_OVERLAP_RATIO = 0.08
STABLE_CENTROID_MIN_WINDOWS = 3
STABLE_CENTROID_MIN_SIMILARITY = 0.66
STABLE_CENTROID_MIN_MARGIN = 0.04
EXACT_CUE_POST_CHANGE_CONTEXT_SECONDS = 1.5
EXACT_CUE_MAX_EXTRA_POST_CHANGE_CUES = 2
# Senko's native arbitrary-window Fbank path has crashed on real 0.56-second
# cues even when extracted individually. Its regular 1.5-second diarization
# windows are stable, so exact-cue verification uses the same minimum length.
EXACT_CUE_MIN_DURATION_SECONDS = 1.50

SR = 16_000

# Keep acoustic count estimation bounded.  Speaker embeddings have already been
# extracted at this point; this cap only limits the spectral eigendecomposition.
ACOUSTIC_COUNT_MIN_SPEAKERS = 2
ACOUSTIC_COUNT_MAX_SPEAKERS = 8
ACOUSTIC_COUNT_SAMPLE_LIMIT = 512
ACOUSTIC_COUNT_MIN_EMBEDDINGS = 8
ACOUSTIC_COUNT_MAX_OVERLAP_RATIO = 0.02

# A manually selected count remains authoritative unless the same-recording
# CAM++ graph provides substantially stronger evidence for another count.  The
# three gates keep this conservative: medium confidence alone is not enough;
# the winning eigengap must also have both a useful absolute lead and a large
# proportional lead over the requested count.
SPEAKER_COUNT_GUARD_MIN_CONFIDENCE = 0.45
SPEAKER_COUNT_GUARD_MIN_RECOMMENDED_GAP = 0.15
SPEAKER_COUNT_GUARD_MIN_GAP_ADVANTAGE = 0.15
SPEAKER_COUNT_GUARD_MAX_REQUESTED_GAP_RATIO = 0.55

# Overlap detection answers "is more than one person speaking?".  The
# exclusive CAM++ timeline then supplies a conservative second-speaker
# candidate without changing the transcript's primary speaker assignment.
OVERLAP_SECONDARY_OSD_CONFIDENCE = 0.55
OVERLAP_SECONDARY_CONTEXT_SECONDS = 12.0
OVERLAP_SECONDARY_WINDOW_SUPPORT = 0.05
OVERLAP_SECONDARY_CONTEXT_WEIGHT = 0.35
OVERLAP_SECONDARY_MIN_SECONDS = 0.12


@dataclass
class DiarizedSegment:
    start: float
    end: float
    text: str
    speaker: str  # "三修" / "SPEAKER_A" 等
    speaker_confidence: float | None = None
    speaker_votes: dict[str, float] | None = None
    voice_pitch_hz: float | None = None
    voice_pitch_confidence: float | None = None
    voice_band: str | None = None
    speaker_subsegments: list[dict] | None = None
    speaker_change_points: list[float] | None = None
    speaker_overlap_risk: bool | None = None
    overlap_ratio: float | None = None
    speaker_overlap_confidence: float | None = None
    speaker_overlap_candidates: list[dict] | None = None
    speaker_cue_embeddings: list[dict] | None = None


@dataclass
class DiarizationResult:
    segments: list[DiarizedSegment]
    speakers: list[str]
    cluster_count: int
    matched_profiles: dict[str, str]  # display_name → real name (匹配过的)
    stats: dict


@dataclass
class _SenkoEmbeddingContext:
    vad_segments: list
    subsegments: list[tuple[float, float]]
    embeddings: np.ndarray
    timing_stats: dict
    subsegment_pitch_hz: np.ndarray
    subsegment_pitch_confidence: np.ndarray
    overlap_intervals: list[dict] = field(default_factory=list)
    overlap_stats: dict = field(default_factory=dict)
    overlap_available: bool = False
    subsegment_overlap_ratios: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )
    runtime_backend: str = "unknown"
    runtime_fallback_reason: str | None = None
    runtime_vad_fallback_reason: str | None = None
    runtime_embedding_fallback_reason: str | None = None
    analysis_wav: Path | None = None
    spectral_workspaces: dict[tuple, dict] = field(default_factory=dict)
    cluster_results: dict[tuple, dict] = field(default_factory=dict)


_SENKO_EMBEDDING_CACHE: dict[tuple[str, int, int, str], _SenkoEmbeddingContext] = {}
_SENKO_EXACT_CUE_EMBEDDING_CACHE: dict[tuple, np.ndarray] = {}

_DIARIZATION_TEMP_PREFIX = "localscribe-diarization-"
_STALE_DIARIZATION_WAVS_CLEANED = False
_SENKO_COREML_FALLBACK_REASON: str | None = None

SENKO_COREML_RUNTIME = "coreml_pyannote_campp"
SENKO_HYBRID_RUNTIME = "coreml_campp_torch_pyannote"
SENKO_COREML_SILERO_RUNTIME = "coreml_campp_silero"
SENKO_CPU_PYANNOTE_RUNTIME = "cpu_pyannote_campp"
SENKO_CPU_RUNTIME = "cpu_silero_campp"


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        # Keep the file when process state cannot be determined safely.
        return True
    return True


def _cleanup_stale_diarization_wavs(temp_dir: Path | None = None) -> dict[str, int]:
    """Remove only WAVs left by dead LocalScribe diarization processes."""
    root = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
    pattern = re.compile(
        rf"^{re.escape(_DIARIZATION_TEMP_PREFIX)}(?P<pid>\d+)-.+\.wav$"
    )
    removed_files = 0
    removed_bytes = 0
    try:
        paths = list(root.glob(f"{_DIARIZATION_TEMP_PREFIX}*.wav"))
    except OSError:
        paths = []
    for path in paths:
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        owner_pid = int(match.group("pid"))
        if owner_pid == os.getpid() or _process_is_running(owner_pid):
            continue
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError:
            continue
        removed_files += 1
        removed_bytes += int(size)
    return {"removed_files": removed_files, "removed_bytes": removed_bytes}


def _cleanup_stale_diarization_wavs_once() -> None:
    global _STALE_DIARIZATION_WAVS_CLEANED
    if _STALE_DIARIZATION_WAVS_CLEANED:
        return
    _STALE_DIARIZATION_WAVS_CLEANED = True
    _cleanup_stale_diarization_wavs()


def _remove_cached_analysis_wav(ctx: _SenkoEmbeddingContext) -> None:
    path = ctx.analysis_wav
    ctx.analysis_wav = None
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        pass


def _clear_senko_embedding_cache() -> None:
    for ctx in _SENKO_EMBEDDING_CACHE.values():
        _remove_cached_analysis_wav(ctx)
    _SENKO_EMBEDDING_CACHE.clear()
    _SENKO_EXACT_CUE_EMBEDDING_CACHE.clear()


atexit.register(_clear_senko_embedding_cache)


def _estimate_acoustic_speaker_count(
    embeddings: np.ndarray,
    *,
    min_speakers: int = ACOUSTIC_COUNT_MIN_SPEAKERS,
    max_speakers: int = ACOUSTIC_COUNT_MAX_SPEAKERS,
    sample_limit: int = ACOUSTIC_COUNT_SAMPLE_LIMIT,
) -> dict:
    """Estimate speaker count from existing CAM++ embeddings via eigengap.

    The graph construction mirrors Senko's spectral path: cosine affinity,
    sparse nearest-neighbour pruning, symmetrization, then an unnormalized graph
    Laplacian.  This function never extracts audio features and never mutates the
    embeddings or the clusterer's fixed speaker-count setting.
    """
    values = np.asarray(embeddings, dtype=np.float64)
    bounded_min = min(
        ACOUSTIC_COUNT_MAX_SPEAKERS,
        max(ACOUSTIC_COUNT_MIN_SPEAKERS, int(min_speakers)),
    )
    bounded_max = min(ACOUSTIC_COUNT_MAX_SPEAKERS, max(bounded_min, int(max_speakers)))
    diagnostics = {
        "available": False,
        "method": "campp_cosine_spectral_eigengap",
        "recommended_n_speakers": bounded_min,
        "confidence": 0.0,
        "confidence_level": "low",
        "eigengap_score": None,
        "relative_eigengap": None,
        "gap_dominance": None,
        "embedding_distance_p90": None,
        "min_speakers": bounded_min,
        "max_speakers": bounded_max,
        "input_embeddings": int(values.shape[0]) if values.ndim >= 1 else 0,
        "sampled_embeddings": 0,
        "eigenvalues": [],
        "eigengaps": {},
        "relative_eigengaps": {},
    }
    if values.ndim != 2 or values.shape[0] < ACOUSTIC_COUNT_MIN_EMBEDDINGS or values.shape[1] == 0:
        diagnostics["reason"] = "insufficient_clean_embeddings"
        return diagnostics

    finite_mask = np.all(np.isfinite(values), axis=1)
    values = values[finite_mask]
    norms = np.linalg.norm(values, axis=1)
    values = values[norms > 1e-9]
    if len(values) < ACOUSTIC_COUNT_MIN_EMBEDDINGS:
        diagnostics["reason"] = "insufficient_finite_embeddings"
        return diagnostics
    values = values / np.linalg.norm(values, axis=1, keepdims=True)

    limit = max(ACOUSTIC_COUNT_MIN_EMBEDDINGS, int(sample_limit))
    if len(values) > limit:
        sample_indices = np.linspace(0, len(values) - 1, limit).astype(int)
        values = values[sample_indices]
    diagnostics["sampled_embeddings"] = int(len(values))

    max_candidate = min(bounded_max, len(values) - 1)
    if max_candidate < bounded_min:
        diagnostics["reason"] = "insufficient_embeddings_for_bounds"
        return diagnostics

    affinity = np.clip(values @ values.T, 0.0, 1.0)
    off_diagonal = ~np.eye(len(values), dtype=bool)
    pairwise_distances = 1.0 - affinity[off_diagonal]
    embedding_distance_p90 = float(np.quantile(pairwise_distances, 0.90))
    np.fill_diagonal(affinity, 0.0)

    # Senko keeps roughly the strongest 2% per row, with at least six peers.
    # The same rule keeps the estimate aligned with the clustering engine while
    # avoiding a dense graph that hides small, well-separated speaker groups.
    neighbour_count = min(len(values) - 1, max(6, int(np.ceil(len(values) * 0.02))))
    if neighbour_count < len(values) - 1:
        keep_indices = np.argpartition(affinity, -neighbour_count, axis=1)[:, -neighbour_count:]
        rows = np.arange(len(values))[:, None]
        sparse_affinity = np.zeros_like(affinity)
        sparse_affinity[rows, keep_indices] = affinity[rows, keep_indices]
        affinity = sparse_affinity
    affinity = 0.5 * (affinity + affinity.T)

    degrees = affinity.sum(axis=1)
    if np.count_nonzero(degrees > 1e-9) < ACOUSTIC_COUNT_MIN_EMBEDDINGS:
        diagnostics["reason"] = "degenerate_affinity_graph"
        return diagnostics
    laplacian = np.diag(degrees) - affinity
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    required = max_candidate + 1
    leading = eigenvalues[:required]

    gaps = {
        count: float(leading[count] - leading[count - 1])
        for count in range(bounded_min, max_candidate + 1)
    }
    relative_gaps = {
        count: gap / max(float(leading[count]), 1e-9)
        for count, gap in gaps.items()
    }
    best_count, best_gap = max(gaps.items(), key=lambda item: (item[1], -item[0]))
    other_gaps = [gap for count, gap in gaps.items() if count != best_count]
    second_gap = max(other_gaps, default=0.0)
    upper_eigenvalue = float(leading[best_count])
    relative_gap = best_gap / max(upper_eigenvalue, 1e-9)
    dominance = max(0.0, best_gap - second_gap) / max(best_gap, 1e-9)
    # A sparse graph can create an artificial gap when all vectors are nearly
    # identical and nearest-neighbour ties are arbitrary.  Calibrate the gap by
    # global CAM++ spread so acoustically inseparable data stays low confidence.
    spread_factor = float(np.clip(embedding_distance_p90 / 0.12, 0.0, 1.0))
    confidence = float(
        np.sqrt(np.clip(relative_gap, 0.0, 1.0) * np.clip(dominance, 0.0, 1.0))
        * spread_factor
    )
    if confidence >= 0.70:
        confidence_level = "high"
    elif confidence >= 0.40:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    diagnostics.update({
        "available": True,
        "recommended_n_speakers": int(best_count),
        "confidence": round(confidence, 4),
        "confidence_level": confidence_level,
        "eigengap_score": round(float(best_gap), 6),
        "relative_eigengap": round(float(relative_gap), 6),
        "gap_dominance": round(float(dominance), 6),
        "embedding_distance_p90": round(embedding_distance_p90, 6),
        "eigenvalues": [round(float(value), 6) for value in leading],
        "eigengaps": {str(count): round(float(gap), 6) for count, gap in gaps.items()},
        "relative_eigengaps": {
            str(count): round(float(score), 6)
            for count, score in relative_gaps.items()
        },
        "reason": "ok" if confidence_level != "low" else "ambiguous_eigengap",
    })
    return diagnostics


def _resolve_requested_speaker_count(requested_n_speakers: int, acoustic_count: dict) -> dict:
    """Choose a clustering count without blindly enforcing a weak request.

    A user request is preserved whenever the model is unavailable, uncertain,
    or reasonably close to the best acoustic candidate.  It is overridden only
    when all guard thresholds independently show that forcing it would create
    acoustically unsupported clusters.
    """
    requested = int(requested_n_speakers or 0)
    available = bool(acoustic_count.get("available"))
    recommended = (
        int(acoustic_count.get("recommended_n_speakers") or 0)
        if available else 0
    )
    confidence = float(acoustic_count.get("confidence") or 0.0)
    eigengaps = acoustic_count.get("eigengaps") or {}

    recommended_gap = _finite_float(eigengaps.get(str(recommended)))
    if recommended_gap is None:
        recommended_gap = _finite_float(acoustic_count.get("eigengap_score"))
    requested_gap = _finite_float(eigengaps.get(str(requested))) if requested > 0 else None
    gap_advantage = (
        recommended_gap - requested_gap
        if recommended_gap is not None and requested_gap is not None
        else None
    )
    requested_gap_ratio = (
        requested_gap / recommended_gap
        if recommended_gap is not None
        and requested_gap is not None
        and recommended_gap > 1e-9
        else None
    )

    decision = "requested_preserved"
    reason = "requested_count_supported"
    oracle_count = requested
    guard_applied = False

    if requested <= 0:
        decision = "automatic"
        reason = "no_explicit_count_requested"
        oracle_count = 0
    elif not available or recommended <= 0:
        reason = "model_count_unavailable"
    elif requested == recommended:
        reason = "requested_matches_model_recommendation"
    elif recommended > requested:
        # An explicit count is a hard upper bound. Acoustic eigengaps are more
        # prone to over-splitting channels/noise than to proving a new person;
        # only the user or automatic 2-8 sweep may increase the requested count.
        reason = "higher_model_count_does_not_override_explicit_request"
    elif confidence < SPEAKER_COUNT_GUARD_MIN_CONFIDENCE:
        reason = "model_confidence_too_low"
    elif recommended_gap is None or requested_gap is None:
        reason = "candidate_eigengap_unavailable"
    elif recommended_gap < SPEAKER_COUNT_GUARD_MIN_RECOMMENDED_GAP:
        reason = "recommended_eigengap_too_weak"
    elif gap_advantage is None or gap_advantage < SPEAKER_COUNT_GUARD_MIN_GAP_ADVANTAGE:
        reason = "candidate_evidence_close"
    elif (
        requested_gap_ratio is None
        or requested_gap_ratio > SPEAKER_COUNT_GUARD_MAX_REQUESTED_GAP_RATIO
    ):
        reason = "candidate_evidence_close"
    else:
        decision = "model_override"
        reason = "model_evidence_strongly_rejects_requested_count"
        oracle_count = recommended
        guard_applied = True

    selected_count = oracle_count if oracle_count > 0 else (recommended or None)
    return {
        "decision": decision,
        "reason": reason,
        "guard_applied": guard_applied,
        "requested_n_speakers": requested,
        "selected_n_speakers": selected_count,
        "oracle_n_speakers": oracle_count,
        "recommended_n_speakers": recommended or None,
        "model_confidence": round(confidence, 4),
        "recommended_eigengap": _rounded_or_none(recommended_gap),
        "requested_eigengap": _rounded_or_none(requested_gap),
        "eigengap_advantage": _rounded_or_none(gap_advantage),
        "requested_gap_ratio": _rounded_or_none(requested_gap_ratio),
        "thresholds": {
            "min_confidence": SPEAKER_COUNT_GUARD_MIN_CONFIDENCE,
            "min_recommended_eigengap": SPEAKER_COUNT_GUARD_MIN_RECOMMENDED_GAP,
            "min_eigengap_advantage": SPEAKER_COUNT_GUARD_MIN_GAP_ADVANTAGE,
            "max_requested_gap_ratio": SPEAKER_COUNT_GUARD_MAX_REQUESTED_GAP_RATIO,
        },
    }


def _finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rounded_or_none(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def _ordered_senko_speakers(raw_segments: list[dict], centroids: dict) -> list[str]:
    speakers: list[str] = []
    for raw in sorted(raw_segments, key=lambda item: float(item.get("start", 0.0))):
        speaker_id = str(raw.get("speaker") or "")
        if speaker_id and speaker_id not in speakers:
            speakers.append(speaker_id)
    for speaker_id in sorted(centroids):
        if speaker_id not in speakers:
            speakers.append(speaker_id)
    return speakers


def _speaker_overlap_candidates_for_window(
    segment_start: float,
    segment_end: float,
    *,
    primary_timeline: Sequence[dict],
    subsegments: Sequence[tuple[float, float]],
    sub_labels: Sequence[int],
    speaker_ids: Sequence[str],
    overlap_intervals: Sequence[dict],
    label_map: dict[str, str],
    osd_confidence_threshold: float = OVERLAP_SECONDARY_OSD_CONFIDENCE,
    context_seconds: float = OVERLAP_SECONDARY_CONTEXT_SECONDS,
    min_window_support: float = OVERLAP_SECONDARY_WINDOW_SUPPORT,
    context_weight: float = OVERLAP_SECONDARY_CONTEXT_WEIGHT,
    min_duration: float = OVERLAP_SECONDARY_MIN_SECONDS,
) -> list[dict]:
    """Return conservative second-speaker evidence for OSD-positive intervals.

    OSD establishes that speech overlaps.  It does not identify who the second
    person is, so attribution combines CAM++ window support inside the overlap
    with proximity to that speaker's nearest stable exclusive turn.  The
    result is metadata only; callers keep the segment's primary speaker.
    """
    if (
        segment_end <= segment_start
        or not primary_timeline
        or not overlap_intervals
        or not speaker_ids
        or len(subsegments) != len(sub_labels)
    ):
        return []

    clean_primary: list[tuple[float, float, str]] = []
    for item in primary_timeline:
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (AttributeError, TypeError, ValueError):
            continue
        speaker = str(item.get("speaker") or "")
        if speaker and end > start:
            clean_primary.append((start, end, speaker))
    if not clean_primary:
        return []

    clean_windows: list[tuple[float, float, str]] = []
    for window, raw_label in zip(subsegments, sub_labels):
        try:
            start, end = float(window[0]), float(window[1])
            label_index = int(raw_label)
        except (IndexError, TypeError, ValueError):
            continue
        if end <= start or not 0 <= label_index < len(speaker_ids):
            continue
        clean_windows.append((start, end, str(speaker_ids[label_index])))

    decay_seconds = max(0.001, float(context_seconds) / 4.0)
    candidates: list[dict] = []
    for overlap_item in overlap_intervals:
        try:
            overlap_start = max(segment_start, float(overlap_item.get("start")))
            overlap_end = min(segment_end, float(overlap_item.get("end")))
            osd_confidence = float(
                overlap_item.get("confidence", overlap_item.get("max_confidence", 0.0))
                or 0.0
            )
        except (AttributeError, TypeError, ValueError):
            continue
        if osd_confidence < osd_confidence_threshold or overlap_end <= overlap_start:
            continue

        for turn_start, turn_end, primary_speaker in clean_primary:
            start = max(overlap_start, turn_start)
            end = min(overlap_end, turn_end)
            if end - start < min_duration:
                continue

            window_support: dict[str, float] = defaultdict(float)
            total_window_support = 0.0
            for window_start, window_end, window_speaker in clean_windows:
                duration = max(0.0, min(end, window_end) - max(start, window_start))
                if duration <= 0.0:
                    continue
                window_support[window_speaker] += duration
                total_window_support += duration

            ranked: list[tuple[float, float, float, str]] = []
            for candidate_speaker in speaker_ids:
                candidate_speaker = str(candidate_speaker)
                if candidate_speaker == primary_speaker:
                    continue
                window_ratio = (
                    float(window_support.get(candidate_speaker, 0.0)) / total_window_support
                    if total_window_support > 0.0 else 0.0
                )
                nearest_distance = min(
                    (
                        max(0.0, candidate_start - end, start - candidate_end)
                        for candidate_start, candidate_end, speaker in clean_primary
                        if speaker == candidate_speaker
                    ),
                    default=float("inf"),
                )
                context_score = (
                    math.exp(-nearest_distance / decay_seconds)
                    if nearest_distance <= context_seconds else 0.0
                )
                if max(window_ratio, context_score) < min_window_support:
                    continue
                score = window_ratio + context_weight * context_score
                ranked.append((score, window_ratio, context_score, candidate_speaker))
            if not ranked:
                continue

            score, window_ratio, context_score, secondary_speaker = max(
                ranked,
                key=lambda item: (item[0], item[1], item[2], item[3]),
            )
            primary_display = label_map.get(primary_speaker, primary_speaker)
            secondary_display = label_map.get(secondary_speaker, secondary_speaker)
            if primary_display == secondary_display:
                continue
            candidates.append({
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "primary_speaker": primary_display,
                "secondary_speaker": secondary_display,
                "confidence": round(float(osd_confidence), 4),
                "window_ratio": round(float(window_ratio), 4),
                "context_score": round(float(context_score), 4),
                "candidate_score": round(float(score), 4),
                "source": "osd_campp_context_v1",
            })

    return sorted(
        candidates,
        key=lambda item: (
            float(item["start"]),
            float(item["end"]),
            str(item["primary_speaker"]),
            str(item["secondary_speaker"]),
        ),
    )


def _ffmpeg_to_16k_wav(audio: Path) -> Path:
    """转码任意音频到 16kHz mono pcm_s16le 临时 wav(senko 要求的输入格式)。"""
    audio = audio.expanduser()
    if not audio.exists():
        raise FileNotFoundError(
            f"源音频文件不存在，无法运行说话人分离：{audio}。"
            "如果这是微信/聊天软件拖拽出来的临时文件，请重新导入原音频，"
            "或从 transcripts 历史库中打开已保存任务后再运行。"
        )

    _cleanup_stale_diarization_wavs_once()
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"{_DIARIZATION_TEMP_PREFIX}{os.getpid()}-",
        suffix=".wav",
        delete=False,
    )
    tmp_path = tmp.name
    tmp.close()
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(audio), "-ac", "1", "-ar", str(SR),
        "-acodec", "pcm_s16le", tmp_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        stderr = (e.stderr or "").strip()
        detail = f"；ffmpeg stderr: {stderr}" if stderr else ""
        raise RuntimeError(
            f"ffmpeg 转码失败，无法运行说话人分离：{audio}"
            f"；exit={e.returncode}{detail}"
        ) from e
    return Path(tmp_path)


def _load_pcm16_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        frames = int(wf.getnframes())
        raw = wf.readframes(frames)
    if sampwidth != 2:
        return np.empty((0,), dtype=np.float32)
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data


def _estimate_pitch_hz(samples: np.ndarray, sr: int = SR) -> tuple[float | None, float]:
    """Estimate a coarse speech pitch for guardrails, not for user-facing gender labels."""
    if samples.size < int(sr * 0.18):
        return None, 0.0

    max_samples = int(sr * 1.6)
    if samples.size > max_samples:
        center = samples.size // 2
        half = max_samples // 2
        samples = samples[max(0, center - half):center + half]

    samples = np.asarray(samples, dtype=np.float32)
    samples = samples - float(np.mean(samples))
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak < 0.004:
        return None, 0.0

    frame_len = int(sr * 0.05)
    step = int(sr * 0.08)
    min_lag = max(1, int(sr / 360.0))
    max_lag = min(frame_len - 2, int(sr / 55.0))
    if max_lag <= min_lag:
        return None, 0.0

    pitches: list[float] = []
    periodicities: list[float] = []
    frame_count = 0
    window = np.hanning(frame_len).astype(np.float32)
    for start in range(0, max(1, samples.size - frame_len + 1), step):
        if frame_count >= 6:
            break
        frame = samples[start:start + frame_len]
        if frame.size < frame_len:
            break
        frame_count += 1
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms < 0.006:
            continue
        frame = (frame - float(np.mean(frame))) * window
        corr = np.correlate(frame, frame, mode="full")[frame_len - 1:]
        if corr.size <= max_lag or float(corr[0]) <= 1e-8:
            continue
        search = corr[min_lag:max_lag + 1]
        best_rel = int(np.argmax(search))
        best_lag = min_lag + best_rel
        periodicity = float(search[best_rel] / (corr[0] + 1e-9))
        if periodicity < 0.28:
            continue
        if 1 <= best_lag < corr.size - 1:
            y0, y1, y2 = float(corr[best_lag - 1]), float(corr[best_lag]), float(corr[best_lag + 1])
            denom = (y0 - 2.0 * y1 + y2)
            if abs(denom) > 1e-9:
                best_lag = best_lag + 0.5 * (y0 - y2) / denom
        pitch = sr / float(best_lag)
        if 55.0 <= pitch <= 360.0:
            pitches.append(float(pitch))
            periodicities.append(periodicity)

    if not pitches:
        return None, 0.0
    confidence = min(
        1.0,
        (len(pitches) / max(3.0, float(frame_count)))
        * (float(np.mean(periodicities)) / 0.55),
    )
    if confidence < 0.18:
        return None, round(confidence, 3)
    return float(np.median(np.asarray(pitches, dtype=np.float32))), round(confidence, 3)


def _estimate_subsegment_pitch(wav: Path, subsegments: Sequence[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    if not subsegments:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    try:
        audio = _load_pcm16_wav(wav)
    except Exception:
        audio = np.empty((0,), dtype=np.float32)
    pitches = np.full((len(subsegments),), np.nan, dtype=np.float32)
    confidences = np.zeros((len(subsegments),), dtype=np.float32)
    if audio.size == 0:
        return pitches, confidences
    for idx, (start, end) in enumerate(subsegments):
        s = max(0, int(float(start) * SR))
        e = min(audio.size, int(float(end) * SR))
        pitch, confidence = _estimate_pitch_hz(audio[s:e], SR)
        if pitch is not None:
            pitches[idx] = float(pitch)
            confidences[idx] = float(confidence)
    return pitches, confidences


def _voice_band_from_pitch(pitch_hz: float | None, confidence: float | None) -> str:
    try:
        pitch = float(pitch_hz)
        conf = float(confidence)
    except (TypeError, ValueError):
        return "unknown"
    if pitch <= 0 or conf < 0.25:
        return "unknown"
    if pitch < 155.0:
        return "low"
    if pitch > 185.0:
        return "high"
    return "mid"


def _weighted_median(values: list[tuple[float, float]]) -> float | None:
    clean = sorted((float(v), max(0.0, float(w))) for v, w in values if w > 0)
    if not clean:
        return None
    total = sum(w for _, w in clean)
    cursor = 0.0
    for value, weight in clean:
        cursor += weight
        if cursor >= total / 2.0:
            return value
    return clean[-1][0]


def _embedding_cache_key(audio: Path, accurate: bool | None = None) -> tuple[str, int, int, str]:
    stat = audio.expanduser().stat()
    return (
        str(audio.expanduser().resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        str(accurate),
    )


def cached_analysis_wav(audio: Path, accurate: bool | None = None) -> Path | None:
    """Return Senko's already-decoded 16k PCM source for this recording."""
    try:
        ctx = _SENKO_EMBEDDING_CACHE.get(_embedding_cache_key(audio, accurate))
    except OSError:
        return None
    path = ctx.analysis_wav if ctx is not None else None
    return path if path is not None and path.is_file() else None


def _configure_senko_runtime_cache(senko_config) -> Path | None:
    cache_dir = os.environ.get("LOCALSCRIBE_SENKO_CACHE_DIR", "").strip()
    if not cache_dir:
        return None
    cache_root = Path(cache_dir).expanduser()
    # Senko normally compiles CoreML beside its packaged model assets. That
    # mutates Contents/Resources after signing, so bundled builds redirect
    # only the generated cache while keeping model inputs read-only.
    senko_config.ModelPaths.cache_base_dir = property(lambda _self: cache_root)
    return cache_root


def _read_silero_pcm_wav(audio_path: str | os.PathLike[str]):
    """Load our guaranteed 16k mono PCM WAV without torchaudio/torchcodec."""
    import torch

    with wave.open(str(audio_path), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        if channels != 1 or sample_width != 2 or sample_rate != SR:
            raise ValueError(
                "Silero fallback expects 16kHz mono 16-bit PCM WAV, got "
                f"{sample_rate}Hz/{channels}ch/{sample_width * 8}bit"
            )
        frames = reader.readframes(reader.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(np.ascontiguousarray(samples))


def _make_senko_cpu_diarizer(senko, senko_config):
    """Force Senko's packaged CAM++ model onto its portable CPU path."""
    import torch
    import senko.diarizer as senko_diarizer_module

    # Senko 0.1.0 hard-codes every macOS process to CoreML at import time.  Its
    # CPU implementation is bundled and works on macOS, but needs these two
    # module globals enabled before constructing the Diarizer.
    senko_config.DARWIN = False
    senko_diarizer_module.torch = torch
    diarizer = senko.Diarizer(
        device="cpu",
        vad="silero",
        clustering="cpu",
        warmup=False,
        quiet=True,
    )
    # silero-vad's current read_audio delegates to torchaudio, whose 2.11
    # release requires torchcodec. LocalScribe already supplies the exact PCM
    # format Senko expects, so using a small deterministic reader avoids an
    # unnecessary runtime dependency in the signed offline App.
    diarizer.read_audio_silero = _read_silero_pcm_wav
    try:
        from senko.vad_local_pyannote import LocalSegmentationVADCuda

        vad_model_paths = senko_config.resolve_model_paths(
            required_fields=("pyannote_segmentation_senko_model_path",),
        )
        diarizer.vad_backend = LocalSegmentationVADCuda(
            checkpoint_path=vad_model_paths.pyannote_segmentation_senko_model_path,
            torch_device=torch.device("cpu"),
        )
        diarizer.vad_model_type = "pyannote"
        uses_torch_pyannote = True
        diarizer._localscribe_vad_fallback_reason = None
    except Exception as exc:
        # Silero remains a last-resort path when the packaged PyTorch
        # segmentation checkpoint cannot load. Never fail the whole recording
        # solely because both accelerated and high-fidelity VAD are unavailable.
        uses_torch_pyannote = False
        diarizer._localscribe_vad_fallback_reason = f"{type(exc).__name__}: {exc}"

    try:
        coreml_embeddings = diarizer._load_coreml_embeddings_model()
        diarizer.embeddings_model = coreml_embeddings
        diarizer.coreml_fixed_frames = 150
        diarizer.coreml_batch_size = 16
        diarizer.device = "coreml"
        diarizer.torch_device = None
        diarizer._localscribe_runtime_backend = (
            SENKO_HYBRID_RUNTIME
            if uses_torch_pyannote
            else SENKO_COREML_SILERO_RUNTIME
        )
        diarizer._localscribe_embedding_fallback_reason = None
    except Exception as exc:
        diarizer._localscribe_runtime_backend = (
            SENKO_CPU_PYANNOTE_RUNTIME
            if uses_torch_pyannote
            else SENKO_CPU_RUNTIME
        )
        diarizer._localscribe_embedding_fallback_reason = f"{type(exc).__name__}: {exc}"
    diarizer._localscribe_uses_torch_pyannote = uses_torch_pyannote
    return diarizer


def _make_diarizer():
    """Build Senko, retaining CoreML unless initialization actually fails."""
    global _SENKO_COREML_FALLBACK_REASON

    import senko
    from senko import config as senko_config

    _configure_senko_runtime_cache(senko_config)

    fallback_reason = _SENKO_COREML_FALLBACK_REASON
    if bool(getattr(senko_config, "DARWIN", False)) and fallback_reason is None:
        try:
            d = senko.Diarizer(device="auto", warmup=False, quiet=True)
            runtime_backend = SENKO_COREML_RUNTIME
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            _SENKO_COREML_FALLBACK_REASON = fallback_reason
            d = _make_senko_cpu_diarizer(senko, senko_config)
            runtime_backend = str(d._localscribe_runtime_backend)
    elif fallback_reason is not None:
        d = _make_senko_cpu_diarizer(senko, senko_config)
        runtime_backend = str(d._localscribe_runtime_backend)
    else:
        d = senko.Diarizer(device="auto", warmup=False, quiet=True)
        runtime_backend = (
            f"{getattr(d, 'device', 'auto')}_"
            f"{getattr(d, 'vad_model_type', 'auto')}_campp"
        )

    d._localscribe_runtime_backend = runtime_backend
    d._localscribe_fallback_reason = fallback_reason
    # 强制所有音频长度都用 spectral —— umap_hdbscan 在 macOS libomp 上死锁
    d.umap_hdbscan_cluster = d.spectral_cluster
    return d


def _perform_cpu_pyannote_analysis(wav: Path, diarizer) -> tuple[list, dict]:
    """Share one PyTorch segmentation pass between VAD and overlap detection."""
    from senko.vad_local_pyannote.audio import load_audio_source
    from senko.vad_local_pyannote.postprocess import (
        aggregate_sliding_scores,
        powerset_logits_to_speech,
        scores_to_segments,
    )

    from .overlap_detector import scores_to_overlap_intervals

    backend = diarizer.vad_backend
    waveform = load_audio_source(wav)
    if waveform.size == 0:
        return [], {
            "available": True,
            "backend": "empty_audio",
            "overlap_intervals": [],
            "stats": {"overlap_seconds": 0.0, "interval_count": 0},
        }

    speech_batches: list[np.ndarray] = []
    overlap_batches: list[np.ndarray] = []
    with backend.torch.inference_mode():
        for batch in backend._iter_chunks(waveform):
            logits = backend.model(batch.to(backend.device))
            speech_batches.append(powerset_logits_to_speech(logits, backend.mapping))
            probabilities = backend.torch.softmax(logits, dim=-1)
            cardinalities = backend.mapping.sum(dim=-1)
            overlap = probabilities[..., cardinalities >= 2].sum(dim=-1, keepdim=True)
            overlap_batches.append(overlap.detach().cpu().numpy())

    if not speech_batches:
        return [], {
            "available": True,
            "backend": "senko_segmentation_torch_shared_pass",
            "overlap_intervals": [],
            "stats": {"overlap_seconds": 0.0, "interval_count": 0},
        }

    total_duration = len(waveform) / backend.sample_rate
    aggregation_args = {
        "chunk_duration": backend.chunk_duration,
        "chunk_step": backend.chunk_step,
        "frame_start": backend.frame_start,
        "frame_duration": backend.frame_duration,
        "frame_step": backend.frame_step,
        "total_duration": total_duration,
        "warm_up": backend.warm_up,
    }
    speech_scores = aggregate_sliding_scores(
        np.vstack(speech_batches),
        **aggregation_args,
    )
    overlap_scores = aggregate_sliding_scores(
        np.vstack(overlap_batches),
        **aggregation_args,
    )
    parameters = backend.parameters
    vad_segments = scores_to_segments(
        speech_scores[:, 0],
        frame_start=backend.frame_start,
        frame_duration=backend.frame_duration,
        frame_step=backend.frame_step,
        onset=parameters.onset,
        offset=parameters.offset,
        min_duration_on=parameters.min_duration_on,
        min_duration_off=parameters.min_duration_off,
    )
    intervals = scores_to_overlap_intervals(
        overlap_scores[:, 0],
        total_duration=total_duration,
    )
    overlap_seconds = sum(
        max(0.0, float(item["end"]) - float(item["start"]))
        for item in intervals
    )
    return vad_segments, {
        "available": True,
        "backend": "senko_segmentation_torch_shared_pass",
        "overlap_intervals": intervals,
        "stats": {
            "model": "pyannote/segmentation-3.0 (Senko bundled PyTorch)",
            "audio_seconds": round(total_duration, 3),
            "overlap_seconds": round(overlap_seconds, 3),
            "overlap_ratio": round(overlap_seconds / max(total_duration, 1e-9), 5),
            "interval_count": len(intervals),
            "frame_count": int(len(overlap_scores)),
            "shared_vad_osd_pass": True,
        },
    }


def _set_oracle_speaker_count(d, n_speakers: int) -> None:
    """Pass the guarded fixed speaker count to Senko clustering.

    Senko's public API does not expose this per-call, but its spectral cluster
    object supports `oracle_num`. Without this, fixed 3-person mode can still
    auto-detect 4+ tiny clusters.
    """
    if n_speakers <= 0:
        return
    for clusterer in (getattr(d, "spectral_cluster", None), getattr(d, "umap_hdbscan_cluster", None)):
        inner = getattr(clusterer, "cluster", None)
        if inner is not None and hasattr(inner, "k"):
            inner.k = int(n_speakers)


def _extract_senko_embeddings(audio: Path, on_progress=None, accurate: bool | None = None) -> _SenkoEmbeddingContext:
    """Run ffmpeg/VAD/CAM++ once and cache reusable short-window embeddings."""
    key = _embedding_cache_key(audio, accurate)
    cached = _SENKO_EMBEDDING_CACHE.get(key)
    if cached is not None:
        if on_progress:
            on_progress({
                "stage": "diarize_cached_embeddings",
                "engine": "senko",
                "subsegments": int(len(cached.subsegments)),
            })
        return cached

    if on_progress:
        on_progress({"stage": "diarize_load_audio"})
    wav = _ffmpeg_to_16k_wav(audio)
    keep_wav = False
    try:
        from senko.utils import suppress_stdout_stderr

        if on_progress:
            on_progress({"stage": "diarize_init"})
        d = _make_diarizer()
        runtime_backend = str(getattr(d, "_localscribe_runtime_backend", "unknown"))
        runtime_fallback_reason = getattr(d, "_localscribe_fallback_reason", None)
        runtime_vad_fallback_reason = getattr(
            d,
            "_localscribe_vad_fallback_reason",
            None,
        )
        runtime_embedding_fallback_reason = getattr(
            d,
            "_localscribe_embedding_fallback_reason",
            None,
        )
        overlap_result = None

        if on_progress:
            on_progress({"stage": "diarize_extract_embeddings"})
        with suppress_stdout_stderr():
            if bool(getattr(d, "_localscribe_uses_torch_pyannote", False)):
                vad_segments, overlap_result = _perform_cpu_pyannote_analysis(wav, d)
            else:
                vad_segments = d._perform_vad(str(wav))
            if not vad_segments:
                ctx = _SenkoEmbeddingContext(
                    vad_segments=[],
                    subsegments=[],
                    embeddings=np.empty((0, 192), dtype=np.float32),
                    timing_stats=getattr(d, "_timing_stats", {}),
                    subsegment_pitch_hz=np.empty((0,), dtype=np.float32),
                    subsegment_pitch_confidence=np.empty((0,), dtype=np.float32),
                    runtime_backend=runtime_backend,
                    runtime_fallback_reason=runtime_fallback_reason,
                    runtime_vad_fallback_reason=runtime_vad_fallback_reason,
                    runtime_embedding_fallback_reason=runtime_embedding_fallback_reason,
                    analysis_wav=wav,
                )
            else:
                subsegments = d._generate_subsegments(vad_segments, accurate)
                features_flat, frames_per_subsegment, subsegment_offsets, feature_dim = d._extract_fbank_features(
                    str(wav),
                    subsegments,
                )
                subsegment_offsets = [int(offset) for offset in subsegment_offsets]
                embeddings = d._generate_embeddings(
                    features_flat,
                    frames_per_subsegment,
                    subsegment_offsets,
                    feature_dim,
                )
                subsegment_pitch_hz, subsegment_pitch_confidence = _estimate_subsegment_pitch(wav, subsegments)
                if on_progress:
                    on_progress({"stage": "diarize_detect_overlap"})
                from .overlap_detector import detect_overlaps, overlap_ratio_for_window

                if overlap_result is not None:
                    pass
                elif runtime_backend == SENKO_CPU_RUNTIME:
                    overlap_result = {
                        "available": False,
                        "backend": "none",
                        "overlap_intervals": [],
                        "stats": {
                            "overlap_seconds": 0.0,
                            "interval_count": 0,
                            "reason": "CoreML OSD unavailable during CPU/Silero fallback",
                            "runtime_backend": runtime_backend,
                        },
                    }
                else:
                    overlap_result = detect_overlaps(wav, senko_diarizer=d)
                overlap_intervals = list(overlap_result.get("overlap_intervals") or [])
                subsegment_overlap_ratios = np.asarray([
                    overlap_ratio_for_window(window, overlap_intervals, padding=0.04)
                    for window in subsegments
                ], dtype=np.float32)
                ctx = _SenkoEmbeddingContext(
                    vad_segments=vad_segments,
                    subsegments=subsegments,
                    embeddings=np.asarray(embeddings, dtype=np.float32),
                    timing_stats=getattr(d, "_timing_stats", {}),
                    subsegment_pitch_hz=subsegment_pitch_hz,
                    subsegment_pitch_confidence=subsegment_pitch_confidence,
                    runtime_backend=runtime_backend,
                    runtime_fallback_reason=runtime_fallback_reason,
                    runtime_vad_fallback_reason=runtime_vad_fallback_reason,
                    runtime_embedding_fallback_reason=runtime_embedding_fallback_reason,
                    overlap_intervals=overlap_intervals,
                    overlap_stats=dict(overlap_result.get("stats") or {}),
                    overlap_available=bool(overlap_result.get("available")),
                    subsegment_overlap_ratios=subsegment_overlap_ratios,
                    analysis_wav=wav,
                )
            keep_wav = True
    finally:
        if not keep_wav:
            try:
                os.unlink(wav)
            except OSError:
                pass

    _clear_senko_embedding_cache()
    _SENKO_EMBEDDING_CACHE[key] = ctx
    return ctx


def _estimate_context_speaker_count(ctx: _SenkoEmbeddingContext) -> dict:
    """Estimate count from clean context embeddings and retain filter audit."""
    embeddings = np.asarray(ctx.embeddings, dtype=np.float32)
    count_embeddings = embeddings
    overlap_filtered = 0
    overlap_filter_applied = False
    overlap_ratios = np.asarray(ctx.subsegment_overlap_ratios, dtype=np.float32)
    if ctx.overlap_available and len(overlap_ratios) == len(embeddings):
        clean_mask = overlap_ratios <= ACOUSTIC_COUNT_MAX_OVERLAP_RATIO
        count_embeddings = embeddings[clean_mask]
        overlap_filtered = int(len(embeddings) - len(count_embeddings))
        overlap_filter_applied = overlap_filtered > 0

    acoustic_count = _estimate_acoustic_speaker_count(count_embeddings)
    acoustic_count.update({
        "total_embeddings": int(len(embeddings)),
        "clean_embeddings": int(len(count_embeddings)),
        "overlap_filtered_embeddings": overlap_filtered,
        "overlap_filter_applied": overlap_filter_applied,
        "overlap_filter_threshold": ACOUSTIC_COUNT_MAX_OVERLAP_RATIO,
    })
    return acoustic_count


def _cached_spectral_labels(
    d,
    ctx: _SenkoEmbeddingContext,
    embeddings: np.ndarray,
    *,
    oracle_count: int,
    workspace_key: tuple,
) -> tuple[np.ndarray | None, bool]:
    """Reuse Senko's expensive affinity eigendecomposition across 2-8 candidates."""
    common = getattr(d, "spectral_cluster", None)
    inner = getattr(common, "cluster", None)
    required_inner = (
        "get_sim_mat",
        "p_pruning",
        "get_laplacian",
        "cluster_embs",
    )
    required_common = ("filter_minor_cluster", "min_cluster_size")
    if (
        oracle_count <= 0
        or not all(hasattr(inner, name) for name in required_inner)
        or not all(hasattr(common, name) for name in required_common)
    ):
        return None, False

    values = np.asarray(embeddings)
    if values.ndim != 2 or values.shape[0] == 0 or oracle_count > values.shape[0]:
        return None, False
    cluster_line = int(getattr(common, "cluster_line", 0) or 0)
    if values.shape[0] < cluster_line:
        return np.ones(values.shape[0], dtype=int), False

    cache_key = (*workspace_key, int(values.shape[1]))
    workspace = ctx.spectral_workspaces.get(cache_key)
    workspace_cache_hit = workspace is not None
    if workspace is None:
        try:
            similarity = inner.get_sim_mat(values)
            pruned = inner.p_pruning(similarity, None)
            symmetric = 0.5 * (pruned + pruned.T)
            laplacian = inner.get_laplacian(symmetric)
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
            return None, False
        max_components = min(
            ACOUSTIC_COUNT_MAX_SPEAKERS,
            int(eigenvectors.shape[1]),
        )
        workspace = {
            "basis": np.asarray(
                eigenvectors[:, :max_components],
                dtype=np.float64,
            ).copy(),
            "eigenvalues": np.asarray(
                eigenvalues[: max_components + 1],
                dtype=np.float64,
            ).copy(),
        }
        ctx.spectral_workspaces[cache_key] = workspace

    basis = np.asarray(workspace.get("basis"))
    if basis.ndim != 2 or oracle_count > basis.shape[1]:
        return None, workspace_cache_hit
    try:
        labels = np.asarray(
            inner.cluster_embs(basis[:, :oracle_count], oracle_count),
            dtype=int,
        )
        labels = common.filter_minor_cluster(
            labels,
            values,
            int(common.min_cluster_size),
        )
        merge_threshold = getattr(common, "mer_cos", None)
        if merge_threshold is not None:
            labels = common.merge_by_cos(labels, values, merge_threshold)
    except (AttributeError, TypeError, ValueError):
        return None, workspace_cache_hit
    return np.asarray(labels, dtype=int), workspace_cache_hit


def _cluster_senko_embeddings(
    d,
    ctx: _SenkoEmbeddingContext,
    *,
    acoustic_count: dict | None = None,
) -> tuple[dict, list[tuple[float, float]], np.ndarray, np.ndarray]:
    """Cluster cached senko embeddings with the current oracle speaker count."""
    vad_segments = ctx.vad_segments
    if not vad_segments:
        return {
            "raw_segments": [],
            "raw_speakers_detected": 0,
            "merged_speakers_detected": 0,
            "merged_segments": [],
            "speaker_centroids": {},
            "timing_stats": ctx.timing_stats or getattr(d, "_timing_stats", {}),
            "vad": [],
            "acoustic_speaker_count": dict(
                acoustic_count or _estimate_context_speaker_count(ctx)
            ),
        }, [], np.empty((0, 192), dtype=np.float32), np.empty((0,), dtype=int)

    subsegments = ctx.subsegments
    embeddings = ctx.embeddings
    cluster_embeddings = embeddings
    cluster_subsegments = subsegments
    filtered_overlap_windows = 0
    overlap_ratios = np.asarray(ctx.subsegment_overlap_ratios, dtype=np.float32)
    overlap_filter_enabled = str(
        os.environ.get("LOCALSCRIBE_OVERLAP_CLUSTER_FILTER", "1")
    ).strip().lower() not in {"0", "false", "no", "off"}
    if overlap_filter_enabled and ctx.overlap_available and len(overlap_ratios) == len(subsegments):
        clean_mask = overlap_ratios <= 0.02
        clean_count = int(np.count_nonzero(clean_mask))
        oracle_count = 0
        inner = getattr(getattr(d, "spectral_cluster", None), "cluster", None)
        try:
            oracle_count = int(getattr(inner, "k", 0) or 0)
        except (TypeError, ValueError):
            oracle_count = 0
        min_clean = max(8, oracle_count * 3)
        if clean_count >= min_clean and clean_count >= int(len(subsegments) * 0.35):
            cluster_embeddings = embeddings[clean_mask]
            cluster_subsegments = [window for window, keep in zip(subsegments, clean_mask) if bool(keep)]
            filtered_overlap_windows = len(subsegments) - clean_count

    oracle_count = 0
    inner = getattr(getattr(d, "spectral_cluster", None), "cluster", None)
    try:
        oracle_count = int(getattr(inner, "k", 0) or 0)
    except (TypeError, ValueError):
        oracle_count = 0
    cluster_key = (
        oracle_count,
        bool(overlap_filter_enabled),
        int(filtered_overlap_windows),
        int(len(cluster_embeddings)),
    )
    cached_cluster = ctx.cluster_results.get(cluster_key)
    cluster_cache_hit = cached_cluster is not None
    spectral_workspace_cache_hit = False
    if cached_cluster is not None:
        raw_segments = cached_cluster["raw_segments"]
        merged_segments = cached_cluster["merged_segments"]
        centroids = cached_cluster["centroids"]
    else:
        labels, spectral_workspace_cache_hit = _cached_spectral_labels(
            d,
            ctx,
            cluster_embeddings,
            oracle_count=oracle_count,
            workspace_key=cluster_key[1:],
        )
        if labels is None:
            raw_segments, merged_segments, centroids = d._perform_clustering(
                cluster_embeddings,
                cluster_subsegments,
            )
        else:
            original_spectral = d.spectral_cluster
            original_long_audio = d.umap_hdbscan_cluster

            class _FixedLabels:
                def __call__(self, _values):
                    return np.asarray(labels, dtype=int).copy()

            fixed_labels = _FixedLabels()
            try:
                d.spectral_cluster = fixed_labels
                d.umap_hdbscan_cluster = fixed_labels
                raw_segments, merged_segments, centroids = d._perform_clustering(
                    cluster_embeddings,
                    cluster_subsegments,
                )
            finally:
                d.spectral_cluster = original_spectral
                d.umap_hdbscan_cluster = original_long_audio
        ctx.cluster_results[cluster_key] = {
            "raw_segments": raw_segments,
            "merged_segments": merged_segments,
            "centroids": centroids,
        }

    acoustic_count = dict(acoustic_count or _estimate_context_speaker_count(ctx))

    speaker_to_idx = {speaker: idx for idx, speaker in enumerate(sorted(centroids))}
    sub_labels = np.zeros(len(subsegments), dtype=int)
    centroid_ids = sorted(centroids)
    if centroid_ids and len(embeddings):
        embedding_mat = np.asarray(embeddings, dtype=np.float32)
        embedding_mat = embedding_mat / (np.linalg.norm(embedding_mat, axis=1, keepdims=True) + 1e-9)
        centroid_mat = np.stack([np.asarray(centroids[speaker], dtype=np.float32) for speaker in centroid_ids])
        centroid_mat = centroid_mat / (np.linalg.norm(centroid_mat, axis=1, keepdims=True) + 1e-9)
        sub_labels = np.argmax(embedding_mat @ centroid_mat.T, axis=1).astype(int)
    else:
        for idx, (start, end) in enumerate(subsegments):
            best_spk = None
            best_overlap = 0.0
            for rs in raw_segments:
                overlap = max(0.0, min(float(rs["end"]), end) - max(float(rs["start"]), start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_spk = rs["speaker"]
            if best_spk in speaker_to_idx:
                sub_labels[idx] = speaker_to_idx[best_spk]

    timing = dict(ctx.timing_stats or {})
    timing.update(getattr(d, "_timing_stats", {}) or {})
    timing["total_time"] = round(sum(float(v) for v in timing.values()), 2) if timing else 0.0
    result = {
        "raw_segments": raw_segments,
        "raw_speakers_detected": len(set(segment["speaker"] for segment in raw_segments)),
        "merged_speakers_detected": len(set(segment["speaker"] for segment in merged_segments)),
        "merged_segments": merged_segments,
        "speaker_centroids": centroids,
        "timing_stats": timing,
        "vad": vad_segments,
        "overlap_filtered_subsegments": filtered_overlap_windows,
        "overlap_filter_enabled": overlap_filter_enabled,
        "cluster_result_cache_hit": cluster_cache_hit,
        "spectral_workspace_cache_hit": spectral_workspace_cache_hit,
        "acoustic_speaker_count": acoustic_count,
    }
    return result, subsegments, embeddings, sub_labels


def extract_voice_embedding(audio: Path) -> list[float]:
    """从样本音频提取 192 维 L2 归一化声纹向量。

    用 senko 跑一遍 diarization,取出现时长最长的 speaker 的 centroid。
    假设上传的样本以目标说话人为主。返回长度 192。
    """
    wav = _ffmpeg_to_16k_wav(audio)
    try:
        from senko.utils import suppress_stdout_stderr

        d = _make_diarizer()
        res = d.diarize(str(wav))
        centroids = res.get('speaker_centroids', {})
        if not centroids:
            raise ValueError("senko 未提取到任何声纹中心 — 音频可能无人声 / 太短")

        # 取主导说话人(说话总时长最长)
        dur = defaultdict(float)
        for s in res.get('raw_segments', []):
            dur[s['speaker']] += s['end'] - s['start']
        dominant = (
            max(dur.items(), key=lambda x: x[1])[0]
            if dur
            else next(iter(centroids))
        )
        emb = np.asarray(centroids[dominant], dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb.astype(float).tolist()
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass


def _overlapping_subsegment_indices(
    subsegment_starts: np.ndarray,
    subsegment_ends: np.ndarray,
    start: float,
    end: float,
) -> range:
    if not len(subsegment_starts) or not len(subsegment_ends):
        return range(0, 0)
    left = int(np.searchsorted(subsegment_ends, float(start), side="right"))
    right = int(np.searchsorted(subsegment_starts, float(end), side="left"))
    return range(max(0, left), min(len(subsegment_starts), right))


def _merge_time_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(
        ((float(start), float(end)) for start, end in intervals if float(end) > float(start)),
        key=lambda item: (item[0], item[1]),
    ):
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _interval_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    return float(sum(end - start for start, end in _merge_time_intervals(intervals)))


def _weighted_window_embedding(
    embeddings: np.ndarray,
    subsegment_starts: np.ndarray,
    subsegment_ends: np.ndarray,
    start: float,
    end: float,
    allowed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float, list[tuple[float, float]]]:
    """Return an embedding plus unique wall-clock voice coverage for a window."""
    vecs: list[np.ndarray] = []
    weights: list[float] = []
    covered_intervals: list[tuple[float, float]] = []
    for idx in _overlapping_subsegment_indices(subsegment_starts, subsegment_ends, start, end):
        if allowed_mask is not None and len(allowed_mask) == len(embeddings) and not bool(allowed_mask[idx]):
            continue
        ss = float(subsegment_starts[idx])
        ee = float(subsegment_ends[idx])
        overlap_start = max(ss, float(start))
        overlap_end = min(ee, float(end))
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap <= 0:
            continue
        vec = np.asarray(embeddings[idx], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-9:
            continue
        vecs.append(vec / norm)
        weights.append(overlap)
        covered_intervals.append((overlap_start, overlap_end))
    if not vecs:
        return None, 0.0, []
    weight_arr = np.asarray(weights, dtype=np.float32)
    weight_mass = float(weight_arr.sum())
    weight_arr = weight_arr / (weight_mass + 1e-9)
    emb = np.sum(np.stack(vecs) * weight_arr[:, None], axis=0)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    merged_intervals = _merge_time_intervals(covered_intervals)
    return emb.astype(np.float32), _interval_seconds(merged_intervals), merged_intervals


def _window_embedding_vectors(
    embeddings: np.ndarray,
    subsegment_starts: np.ndarray,
    subsegment_ends: np.ndarray,
    start: float,
    end: float,
    *,
    min_overlap: float = 0.18,
    allowed_mask: np.ndarray | None = None,
) -> list[tuple[np.ndarray, float, int]]:
    vectors: list[tuple[np.ndarray, float, int]] = []
    for idx in _overlapping_subsegment_indices(subsegment_starts, subsegment_ends, start, end):
        if allowed_mask is not None and len(allowed_mask) == len(embeddings) and not bool(allowed_mask[idx]):
            continue
        ss = float(subsegment_starts[idx])
        ee = float(subsegment_ends[idx])
        overlap = max(0.0, min(ee, float(end)) - max(ss, float(start)))
        if overlap < min_overlap:
            continue
        vec = np.asarray(embeddings[idx], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-9:
            continue
        vectors.append((vec / norm, overlap, idx))
    return vectors


def _embedding_consistency(vectors: Sequence[np.ndarray]) -> dict:
    if len(vectors) < 2:
        return {
            "vector_count": len(vectors),
            "pair_count": 0,
            "median_similarity": None,
            "p10_similarity": None,
            "min_similarity": None,
            "stable": len(vectors) == 1,
        }
    mat = np.stack([np.asarray(v, dtype=np.float32) for v in vectors])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims: list[float] = []
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[0]):
            sims.append(float(mat[i] @ mat[j]))
    arr = np.asarray(sims, dtype=np.float32)
    return {
        "vector_count": int(mat.shape[0]),
        "pair_count": int(arr.size),
        "median_similarity": round(float(np.median(arr)), 4),
        "p10_similarity": round(float(np.percentile(arr, 10)), 4),
        "min_similarity": round(float(np.min(arr)), 4),
        "stable": True,
    }


def preflight_voiceprint_anchor_candidates(
    audio: Path,
    segments: Sequence[dict],
    *,
    min_anchor_seconds: float = 1.2,
    min_quality_vectors: int = 2,
    min_anchor_consistency: float = 0.68,
    on_progress=None,
) -> dict:
    """Find transcript segments that can safely seed a voiceprint anchor.

    The UI used to rank candidates from diarization metadata alone.  That is
    useful for speed, but cannot prove that the underlying CAM++ sliding
    windows are internally consistent.  This read-only preflight reuses the
    exact gates enforced by ``reidentify_with_voice_anchors`` so users do not
    get a promising-looking candidate that is rejected only after clicking
    reidentify.  It never changes transcript text, timing, cues, or labels.
    """
    ctx = _extract_senko_embeddings(audio, on_progress=on_progress)
    embeddings = np.asarray(ctx.embeddings, dtype=np.float32)
    if len(embeddings) == 0:
        return {
            "candidates": [],
            "stats": {
                "mode": "voiceprint_anchor_preflight",
                "checked_segments": 0,
                "eligible_segments": 0,
                "rejected_segments": 0,
                "reason": "音频里没有可用人声声纹",
            },
        }
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    subsegment_starts = np.asarray([float(start) for start, _ in ctx.subsegments], dtype=np.float32)
    subsegment_ends = np.asarray([float(end) for _, end in ctx.subsegments], dtype=np.float32)
    overlap_ratios = np.asarray(ctx.subsegment_overlap_ratios, dtype=np.float32)
    clean_embedding_mask = (
        overlap_ratios <= 0.02
        if ctx.overlap_available and len(overlap_ratios) == len(embeddings)
        else np.ones(len(embeddings), dtype=bool)
    )
    from .overlap_detector import overlap_ratio_for_window

    if on_progress:
        on_progress({"stage": "voiceprint_anchor_preflight", "segments": len(segments)})

    candidates: list[dict] = []
    rejected = 0
    for index, raw_segment in enumerate(segments):
        segment = dict(raw_segment)
        speaker = str(segment.get("speaker") or "").strip()
        try:
            start = float(segment.get("start"))
            end = float(segment.get("end"))
        except (TypeError, ValueError):
            rejected += 1
            continue
        if not speaker or end <= start:
            rejected += 1
            continue
        overlap_ratio = overlap_ratio_for_window(
            (start, end),
            ctx.overlap_intervals,
        ) if ctx.overlap_available else 0.0
        if overlap_ratio > 0.02:
            rejected += 1
            continue
        emb, covered, _covered_intervals = _weighted_window_embedding(
            embeddings,
            subsegment_starts,
            subsegment_ends,
            start,
            end,
            allowed_mask=clean_embedding_mask,
        )
        if emb is None or covered < min_anchor_seconds:
            rejected += 1
            continue
        vector_rows = _window_embedding_vectors(
            embeddings,
            subsegment_starts,
            subsegment_ends,
            start,
            end,
            allowed_mask=clean_embedding_mask,
        )
        quality = _embedding_consistency([vector for vector, _overlap, _idx in vector_rows])
        if int(quality.get("vector_count") or 0) < min_quality_vectors:
            rejected += 1
            continue
        median_similarity = quality.get("median_similarity")
        if median_similarity is not None and float(median_similarity) < min_anchor_consistency:
            rejected += 1
            continue
        candidates.append({
            "index": index,
            "speaker": speaker,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "text": str(segment.get("text") or ""),
            "covered_seconds": round(covered, 3),
            "quality": quality,
            "reason": "已通过 CAM++ 段内一致性和重叠语音预检",
        })

    candidates.sort(
        key=lambda item: (
            -float((item.get("quality") or {}).get("median_similarity") or -1.0),
            -float(item.get("covered_seconds") or 0.0),
            int(item.get("index") or 0),
        ),
    )
    return {
        "candidates": candidates,
        "stats": {
            "mode": "voiceprint_anchor_preflight",
            "checked_segments": len(segments),
            "eligible_segments": len(candidates),
            "rejected_segments": rejected,
            "min_anchor_seconds": round(float(min_anchor_seconds), 3),
            "min_quality_vectors": int(min_quality_vectors),
            "min_anchor_consistency": round(float(min_anchor_consistency), 3),
            "reason": (
                f"已按 CAM++ 声纹质量门禁筛出 {len(candidates)} 个可用锚点"
                if candidates else "没有通过声纹质量门禁的候选，请手动选择更清晰的单人发言"
            ),
        },
    }


def _build_stable_speaker_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroid_ids: Sequence[str],
    centroids: dict,
    overlap_ratios: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray, dict]:
    """Build conservative per-speaker centroids from clean CAM++ windows.

    Senko's clustering centroids can include boundary or overlapping windows.
    Cue verification instead uses only windows that clearly prefer their own
    cluster and are not marked as overlapping speech.
    """
    values = np.asarray(embeddings, dtype=np.float32)
    label_values = np.asarray(labels, dtype=int)
    raw_ids = [str(speaker) for speaker in centroid_ids]
    diagnostics = {
        "available": False,
        "method": "campp_clean_cluster_centroids",
        "speaker_count": 0,
        "speakers": {},
    }
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[0] != label_values.shape[0]
        or not raw_ids
    ):
        diagnostics["reason"] = "missing_embeddings_or_labels"
        return [], np.empty((0, values.shape[1] if values.ndim == 2 else 0), dtype=np.float32), diagnostics

    norms = np.linalg.norm(values, axis=1)
    finite_mask = np.all(np.isfinite(values), axis=1) & (norms > 1e-9)
    normalized = values / (norms[:, None] + 1e-9)

    original_rows: list[np.ndarray] = []
    valid_raw_ids: list[str] = []
    for speaker_id in raw_ids:
        centroid = np.asarray(centroids.get(speaker_id), dtype=np.float32)
        if centroid.shape != (values.shape[1],) or not np.all(np.isfinite(centroid)):
            continue
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm <= 1e-9:
            continue
        valid_raw_ids.append(speaker_id)
        original_rows.append(centroid / centroid_norm)
    if len(valid_raw_ids) != len(raw_ids):
        diagnostics["reason"] = "invalid_cluster_centroids"
        return [], np.empty((0, values.shape[1]), dtype=np.float32), diagnostics

    original_mat = np.stack(original_rows)
    score_mat = normalized @ original_mat.T
    if overlap_ratios is not None and len(overlap_ratios) == len(values):
        overlap_values = np.asarray(overlap_ratios, dtype=np.float32)
        clean_mask = np.isfinite(overlap_values) & (overlap_values <= ACOUSTIC_COUNT_MAX_OVERLAP_RATIO)
    else:
        clean_mask = np.ones((len(values),), dtype=bool)

    stable_ids: list[str] = []
    stable_rows: list[np.ndarray] = []
    for speaker_index, speaker_id in enumerate(raw_ids):
        candidate_indices = np.flatnonzero(
            finite_mask
            & clean_mask
            & (label_values == speaker_index)
        )
        speaker_stats = {
            "input_windows": int(np.count_nonzero(label_values == speaker_index)),
            "clean_windows": int(len(candidate_indices)),
            "trusted_windows": 0,
            "available": False,
        }
        if len(candidate_indices) < STABLE_CENTROID_MIN_WINDOWS:
            speaker_stats["reason"] = "insufficient_clean_windows"
            diagnostics["speakers"][speaker_id] = speaker_stats
            continue

        candidate_scores = score_mat[candidate_indices]
        own_scores = candidate_scores[:, speaker_index]
        if candidate_scores.shape[1] > 1:
            alternatives = candidate_scores.copy()
            alternatives[:, speaker_index] = -np.inf
            second_scores = np.max(alternatives, axis=1)
            margins = own_scores - second_scores
        else:
            margins = own_scores
        trusted_mask = (
            (own_scores >= STABLE_CENTROID_MIN_SIMILARITY)
            & (margins >= STABLE_CENTROID_MIN_MARGIN)
        )
        trusted_indices = candidate_indices[trusted_mask]
        speaker_stats["trusted_windows"] = int(len(trusted_indices))
        if len(trusted_indices) < STABLE_CENTROID_MIN_WINDOWS:
            speaker_stats["reason"] = "insufficient_distinct_windows"
            diagnostics["speakers"][speaker_id] = speaker_stats
            continue

        stable = np.mean(normalized[trusted_indices], axis=0)
        stable_norm = float(np.linalg.norm(stable))
        if stable_norm <= 1e-9:
            speaker_stats["reason"] = "degenerate_centroid"
            diagnostics["speakers"][speaker_id] = speaker_stats
            continue
        stable = stable / stable_norm
        internal_scores = normalized[trusted_indices] @ stable
        speaker_stats.update({
            "available": True,
            "median_similarity": round(float(np.median(internal_scores)), 4),
            "p10_similarity": round(float(np.percentile(internal_scores, 10)), 4),
            "mean_cluster_margin": round(float(np.mean(margins[trusted_mask])), 4),
            "reason": "ok",
        })
        diagnostics["speakers"][speaker_id] = speaker_stats
        stable_ids.append(speaker_id)
        stable_rows.append(stable.astype(np.float32))

    diagnostics["available"] = bool(stable_rows)
    diagnostics["speaker_count"] = len(stable_rows)
    diagnostics["reason"] = "ok" if stable_rows else "no_stable_speaker_centroids"
    matrix = (
        np.stack(stable_rows)
        if stable_rows
        else np.empty((0, values.shape[1]), dtype=np.float32)
    )
    return stable_ids, matrix, diagnostics


def _extract_exact_cue_embeddings(
    audio: Path,
    diarizer,
    windows: Sequence[tuple[float, float]],
    on_progress=None,
) -> np.ndarray:
    """Extract CAM++ embeddings from exact cue audio instead of sliding windows."""
    if not windows:
        return np.empty((0, 192), dtype=np.float32)
    normalized_windows = tuple(
        (round(float(start), 6), round(float(end), 6))
        for start, end in windows
    )
    key = (*_embedding_cache_key(audio), normalized_windows)
    cached = _SENKO_EXACT_CUE_EMBEDDING_CACHE.get(key)
    if cached is not None:
        return cached

    if on_progress:
        on_progress({"stage": "diarize_extract_exact_cue_embeddings", "cues": len(windows)})
    wav = _ffmpeg_to_16k_wav(audio)
    try:
        from senko.utils import suppress_stdout_stderr

        values: list[np.ndarray] = []
        with suppress_stdout_stderr():
            # libfbank_extractor batches arbitrary windows on native threads.
            # Mixed cue lengths can corrupt its shared output buffer and crash
            # the whole Python sidecar with SIGBUS/SIGSEGV. Exact cues are few,
            # so extract them one at a time; the main fixed 1.5-second diarizer
            # batch remains unchanged.
            for window in windows:
                features_flat, frames, offsets, feature_dim = diarizer._extract_fbank_features(
                    str(wav),
                    [window],
                )
                # Senko's CoreML path uses values as slice indices but its
                # arbitrary-window extractor can return uint64 arrays.
                window_values = diarizer._generate_embeddings(
                    features_flat,
                    [int(value) for value in frames],
                    [int(value) for value in offsets],
                    feature_dim,
                )
                window_array = np.asarray(window_values, dtype=np.float32)
                if window_array.ndim != 2 or window_array.shape[0] != 1:
                    raise ValueError(
                        f"exact cue embedding count mismatch for one window: {window_array.shape}"
                    )
                values.append(window_array[0])
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass

    embeddings = np.asarray(values, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(windows):
        raise ValueError(
            f"exact cue embedding count mismatch: expected {len(windows)}, got {embeddings.shape}"
        )
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    _SENKO_EXACT_CUE_EMBEDDING_CACHE.clear()
    _SENKO_EXACT_CUE_EMBEDDING_CACHE[key] = embeddings
    return embeddings


def _is_safe_exact_cue_window(start: float, end: float) -> bool:
    """Keep undersized windows out of Senko's native Fbank extractor."""
    return float(end) - float(start) >= EXACT_CUE_MIN_DURATION_SECONDS


def _normalized_exact_cue_window(start: float, end: float) -> tuple[float, float] | None:
    """Return a fixed 1.5-second window centered inside a long sync cue."""
    start = float(start)
    end = float(end)
    if not _is_safe_exact_cue_window(start, end):
        return None
    midpoint = (start + end) / 2.0
    half = EXACT_CUE_MIN_DURATION_SECONDS / 2.0
    normalized_start = max(start, min(end - EXACT_CUE_MIN_DURATION_SECONDS, midpoint - half))
    return normalized_start, normalized_start + EXACT_CUE_MIN_DURATION_SECONDS


def _exact_cue_positions_near_changes(
    bounded_cues: Sequence[tuple[int, float, float]],
    change_points: Sequence[float],
) -> set[int]:
    """Select cue positions close enough to a change for exact embeddings."""
    selected: set[int] = set()
    for change_point in change_points:
        containing_position = next(
            (
                position
                for position, (_, cue_start, cue_end) in enumerate(bounded_cues)
                if cue_start <= change_point <= cue_end
            ),
            None,
        )
        if containing_position is None and bounded_cues:
            containing_position = min(
                range(len(bounded_cues)),
                key=lambda position: abs(
                    (bounded_cues[position][1] + bounded_cues[position][2]) / 2.0
                    - change_point
                ),
            )
        if containing_position is None:
            continue
        for position in range(
            max(0, containing_position - 1),
            min(len(bounded_cues), containing_position + 2),
        ):
            selected.add(position)

        # A 1.5-second sliding embedding can drag one speaker beyond the next
        # cue. Include up to two additional short cues while they remain inside
        # that acoustic context instead of extracting every cue in the file.
        extra_positions = range(
            containing_position + 2,
            min(
                len(bounded_cues),
                containing_position + 2 + EXACT_CUE_MAX_EXTRA_POST_CHANGE_CUES,
            ),
        )
        for position in extra_positions:
            if bounded_cues[position][1] - change_point > EXACT_CUE_POST_CHANGE_CONTEXT_SECONDS:
                break
            selected.add(position)
    return selected


def _speaker_cue_embedding_evidence(
    segment: dict,
    embeddings: np.ndarray,
    subsegment_starts: np.ndarray,
    subsegment_ends: np.ndarray,
    stable_ids: Sequence[str],
    stable_centroids: np.ndarray,
    label_map: dict[str, str],
    *,
    overlap_ratios: np.ndarray | None = None,
    overlap_intervals: Sequence[dict] = (),
    overlap_available: bool = False,
    exact_embeddings: dict[int, np.ndarray] | None = None,
) -> list[dict]:
    """Score immutable ASR sync cues against stable same-recording centroids."""
    raw_cues = segment.get("sync_cues")
    centroid_mat = np.asarray(stable_centroids, dtype=np.float32)
    if (
        not isinstance(raw_cues, list)
        or not raw_cues
        or not stable_ids
        or centroid_mat.ndim != 2
        or centroid_mat.shape[0] != len(stable_ids)
    ):
        return []

    allowed_mask = None
    if overlap_ratios is not None and len(overlap_ratios) == len(embeddings):
        overlap_values = np.asarray(overlap_ratios, dtype=np.float32)
        allowed_mask = np.isfinite(overlap_values) & (overlap_values <= ACOUSTIC_COUNT_MAX_OVERLAP_RATIO)

    from .overlap_detector import overlap_ratio_for_window

    rows: list[dict] = []
    segment_start = float(segment.get("start", 0.0))
    segment_end = float(segment.get("end", segment_start))
    for cue_index, cue in enumerate(raw_cues):
        if not isinstance(cue, dict):
            continue
        try:
            start = max(segment_start, float(cue.get("start")))
            end = min(segment_end, float(cue.get("end")))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue

        weighted_embedding, coverage_seconds, _ = _weighted_window_embedding(
            embeddings,
            subsegment_starts,
            subsegment_ends,
            start,
            end,
            allowed_mask=allowed_mask,
        )
        exact_embedding = (exact_embeddings or {}).get(cue_index)
        embedding_scope = "sliding_window_weighted"
        if exact_embedding is not None:
            exact_value = np.asarray(exact_embedding, dtype=np.float32)
            exact_norm = float(np.linalg.norm(exact_value))
            embedding = exact_value / exact_norm if exact_norm > 1e-9 else weighted_embedding
            if exact_norm > 1e-9:
                embedding_scope = "exact_sync_cue"
        else:
            embedding = weighted_embedding
        if embedding is None:
            continue
        scores = np.asarray(embedding @ centroid_mat.T, dtype=np.float32)
        order = np.argsort(scores)[::-1]
        best_index = int(order[0])
        second_index = int(order[1]) if len(order) > 1 else None
        best_score = float(scores[best_index])
        second_score = float(scores[second_index]) if second_index is not None else -1.0
        margin = best_score - second_score if second_index is not None else best_score
        duration = end - start
        coverage_ratio = min(1.0, coverage_seconds / max(duration, 1e-9))
        overlap_ratio = (
            float(overlap_ratio_for_window((start, end), list(overlap_intervals)))
            if overlap_available else 0.0
        )
        enough_voice = (
            coverage_seconds >= CUE_MATCH_MIN_COVERAGE_SECONDS
            and coverage_ratio >= CUE_MATCH_MIN_COVERAGE_RATIO
        )
        low_overlap = overlap_ratio < CUE_MATCH_MAX_OVERLAP_RATIO
        strongly_distinct = (
            best_score >= CUE_MATCH_DISTINCT_THRESHOLD
            and margin >= CUE_MATCH_DISTINCT_MARGIN
            and coverage_ratio >= CUE_MATCH_DISTINCT_COVERAGE_RATIO
        )
        exact_strongly_distinct = (
            embedding_scope == "exact_sync_cue"
            and duration >= CUE_MATCH_MIN_COVERAGE_SECONDS
            and low_overlap
            and best_score >= CUE_MATCH_DISTINCT_THRESHOLD
            and margin >= CUE_MATCH_DISTINCT_MARGIN
        )
        if (
            exact_strongly_distinct
            or (
                enough_voice
                and low_overlap
                and (
                    (best_score >= CUE_MATCH_THRESHOLD and margin >= CUE_MATCH_MARGIN)
                    or strongly_distinct
                )
            )
        ):
            decision = "assign"
        elif enough_voice and best_score >= CUE_MATCH_REVIEW_THRESHOLD:
            decision = "review"
        else:
            decision = "insufficient"

        best_raw_id = str(stable_ids[best_index])
        row = {
            "cue_index": int(cue_index),
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": label_map.get(best_raw_id, best_raw_id),
            "score": round(best_score, 4),
            "second_score": round(second_score, 4) if second_index is not None else None,
            "margin": round(margin, 4),
            "voice_coverage_seconds": round(float(coverage_seconds), 3),
            "voice_coverage_ratio": round(float(coverage_ratio), 4),
            "overlap_ratio": round(float(overlap_ratio), 4),
            "decision": decision,
            "source": "campp_sync_cue_embedding",
            "embedding_scope": embedding_scope,
        }
        if second_index is not None:
            second_raw_id = str(stable_ids[second_index])
            row["second_speaker"] = label_map.get(second_raw_id, second_raw_id)
        rows.append(row)
    return rows


def _normalize_profile_embeddings(profile: dict) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    raw_vectors = profile.get("embeddings")
    if isinstance(raw_vectors, list):
        for item in raw_vectors:
            arr = np.asarray(item or [], dtype=np.float32)
            if arr.ndim == 1 and arr.size > 0:
                norm = float(np.linalg.norm(arr))
                if norm > 1e-9:
                    vectors.append(arr / norm)
    if not vectors:
        arr = np.asarray(profile.get("embedding") or [], dtype=np.float32)
        if arr.ndim == 1 and arr.size > 0:
            norm = float(np.linalg.norm(arr))
            if norm > 1e-9:
                vectors.append(arr / norm)
    return vectors


def _global_profile_assignment(scores: dict[str, dict[str, float]], min_score: float) -> dict[str, str]:
    """Maximum-score one-to-one assignment for meeting-sized profile sets."""
    speakers = sorted(scores)
    profiles = sorted({profile for row in scores.values() for profile in row})
    if not speakers or not profiles:
        return {}
    profile_index = {profile: idx for idx, profile in enumerate(profiles)}
    states: dict[int, tuple[float, tuple[tuple[str, str], ...]]] = {0: (0.0, ())}
    for speaker in speakers:
        next_states = dict(states)
        for mask, (total, pairs) in states.items():
            for profile, score in scores.get(speaker, {}).items():
                if float(score) < float(min_score):
                    continue
                bit = 1 << profile_index[profile]
                if mask & bit:
                    continue
                candidate = (total + float(score), pairs + ((speaker, profile),))
                current = next_states.get(mask | bit)
                if current is None or candidate[0] > current[0] + 1e-12:
                    next_states[mask | bit] = candidate
        states = next_states
    best = max(states.values(), key=lambda item: (item[0], len(item[1])))
    return dict(best[1])


def reidentify_with_voice_anchors(
    audio: Path,
    segments: Sequence[dict],
    anchors: Sequence[dict],
    *,
    threshold: float = 0.78,
    review_threshold: float = 0.70,
    margin: float = 0.05,
    min_anchor_seconds: float = 1.2,
    min_profile_seconds: float = 10.0,
    min_segment_seconds: float = 0.35,
    min_quality_vectors: int = 3,
    min_anchor_consistency: float = 0.68,
    min_profile_consistency: float = 0.70,
    require_enrollment_quality: bool = True,
    on_progress=None,
) -> dict:
    """Re-label transcript segments from user-confirmed voice anchors.

    This is the local "mark one segment, fix many" workflow.  It deliberately
    does not recluster the whole meeting: existing transcript timing/text stays
    fixed, and only high-confidence voiceprint matches are relabeled.
    """
    if not anchors:
        raise ValueError("至少需要 1 个已确认说话人片段")

    ctx = _extract_senko_embeddings(audio, on_progress=on_progress)
    embeddings = np.asarray(ctx.embeddings, dtype=np.float32)
    if len(embeddings) == 0:
        raise ValueError("音频里没有可用人声声纹，无法重识别说话人")
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    subsegment_starts = np.asarray([float(s) for s, _ in ctx.subsegments], dtype=np.float32)
    subsegment_ends = np.asarray([float(e) for _, e in ctx.subsegments], dtype=np.float32)
    overlap_ratios = np.asarray(ctx.subsegment_overlap_ratios, dtype=np.float32)
    clean_embedding_mask = (
        overlap_ratios <= 0.02
        if ctx.overlap_available and len(overlap_ratios) == len(embeddings)
        else np.ones(len(embeddings), dtype=bool)
    )
    from .overlap_detector import overlap_ratio_for_window

    if on_progress:
        on_progress({"stage": "voiceprint_build_anchors", "anchors": len(anchors)})

    grouped: dict[
        str,
        list[tuple[
            np.ndarray,
            float,
            dict,
            list[tuple[np.ndarray, float, int]],
            dict,
            list[tuple[float, float]],
        ]],
    ] = defaultdict(list)
    rejected_anchors: list[dict] = []
    seen_anchor_ranges: set[tuple[str, int, int]] = set()
    accepted_anchor_ranges: list[tuple[str, float, float]] = []
    accepted_subsegment_labels: dict[int, str] = {}
    for raw_anchor in anchors:
        label = str(
            raw_anchor.get("speaker")
            or raw_anchor.get("name")
            or raw_anchor.get("label")
            or ""
        ).strip()
        if not label:
            rejected_anchors.append({"reason": "missing_label", "anchor": raw_anchor})
            continue
        if require_enrollment_quality and re.fullmatch(r"SPEAKER_.+", label, re.IGNORECASE):
            rejected_anchors.append({
                "reason": "generic_profile_name",
                "speaker": label,
                "anchor": raw_anchor,
            })
            continue
        try:
            start = float(raw_anchor.get("start"))
            end = float(raw_anchor.get("end"))
        except (TypeError, ValueError):
            rejected_anchors.append({"reason": "bad_time", "anchor": raw_anchor})
            continue
        if end <= start:
            rejected_anchors.append({"reason": "bad_time", "anchor": raw_anchor})
            continue
        anchor_key = (label, round(start * 1000), round(end * 1000))
        if anchor_key in seen_anchor_ranges:
            rejected_anchors.append({
                "reason": "duplicate_anchor",
                "speaker": label,
                "start": start,
                "end": end,
            })
            continue
        conflicting_label = next((
            accepted_label
            for accepted_label, accepted_start, accepted_end in accepted_anchor_ranges
            if accepted_label != label
            and min(end, accepted_end) - max(start, accepted_start) > 0.05
        ), None)
        if conflicting_label is not None:
            rejected_anchors.append({
                "reason": "conflicting_anchor_label",
                "speaker": label,
                "conflicts_with": conflicting_label,
                "start": start,
                "end": end,
            })
            continue
        anchor_overlap_ratio = overlap_ratio_for_window(
            (start, end),
            ctx.overlap_intervals,
        ) if ctx.overlap_available else 0.0
        if anchor_overlap_ratio > 0.02:
            rejected_anchors.append({
                "reason": "overlapping_speech",
                "speaker": label,
                "start": start,
                "end": end,
                "overlap_ratio": round(float(anchor_overlap_ratio), 4),
            })
            continue
        emb, covered, covered_intervals = _weighted_window_embedding(
            embeddings,
            subsegment_starts,
            subsegment_ends,
            start,
            end,
            allowed_mask=clean_embedding_mask,
        )
        if emb is None or covered < min_anchor_seconds:
            rejected_anchors.append({
                "reason": "too_little_speech",
                "speaker": label,
                "start": start,
                "end": end,
                "covered_s": round(covered, 3),
            })
            continue
        vector_rows = _window_embedding_vectors(
            embeddings,
            subsegment_starts,
            subsegment_ends,
            start,
            end,
            allowed_mask=clean_embedding_mask,
        )
        conflicting_subsegments = sorted({
            subsegment_index
            for _vector, _overlap, subsegment_index in vector_rows
            if accepted_subsegment_labels.get(subsegment_index) not in {None, label}
        })
        if conflicting_subsegments:
            rejected_anchors.append({
                "reason": "conflicting_anchor_voice_window",
                "speaker": label,
                "conflicts_with": sorted({
                    accepted_subsegment_labels[subsegment_index]
                    for subsegment_index in conflicting_subsegments
                }),
                "start": start,
                "end": end,
                "subsegment_indices": conflicting_subsegments[:20],
            })
            continue
        vector_list = [vec for vec, _, _ in vector_rows]
        quality = _embedding_consistency(vector_list)
        median_similarity = quality.get("median_similarity")
        if median_similarity is not None and float(median_similarity) < min_anchor_consistency:
            rejected_anchors.append({
                "reason": "mixed_voice_anchor",
                "speaker": label,
                "start": start,
                "end": end,
                "covered_s": round(covered, 3),
                "quality": quality,
            })
            continue
        seen_anchor_ranges.add(anchor_key)
        accepted_anchor_ranges.append((label, start, end))
        for _vector, _overlap, subsegment_index in vector_rows:
            accepted_subsegment_labels.setdefault(subsegment_index, label)
        grouped[label].append((
            emb,
            covered,
            raw_anchor,
            vector_rows,
            quality,
            covered_intervals,
        ))

    profiles: list[dict] = []
    rejected_profiles: list[dict] = []
    for label, items in grouped.items():
        sample_intervals = _merge_time_intervals(
            interval
            for item in items
            for interval in item[5]
        )
        sample_seconds = _interval_seconds(sample_intervals)
        profile_vectors: list[np.ndarray] = []
        seen_subsegments: set[int] = set()
        for item in items:
            for vector, _overlap, subsegment_index in item[3]:
                if subsegment_index in seen_subsegments:
                    continue
                seen_subsegments.add(subsegment_index)
                profile_vectors.append(np.asarray(vector, dtype=np.float32))
        profile_quality = _embedding_consistency(profile_vectors)
        median_similarity = profile_quality.get("median_similarity")
        enrollment_ready = True
        enrollment_reasons: list[str] = []
        if sample_seconds < min_profile_seconds:
            enrollment_ready = False
            enrollment_reasons.append("too_short_enrollment")
        if len(profile_vectors) < min_quality_vectors:
            enrollment_ready = False
            enrollment_reasons.append("too_few_quality_vectors")
        if median_similarity is not None and float(median_similarity) < min_profile_consistency:
            enrollment_ready = False
            enrollment_reasons.append("mixed_voice_profile")
        if require_enrollment_quality and "too_short_enrollment" in enrollment_reasons:
            rejected_profiles.append({
                "reason": "too_short_enrollment",
                "speaker": label,
                "sample_seconds": round(sample_seconds, 3),
                "min_profile_seconds": round(float(min_profile_seconds), 3),
                "anchor_count": len(items),
            })
            continue
        if require_enrollment_quality and "too_few_quality_vectors" in enrollment_reasons:
            rejected_profiles.append({
                "reason": "too_few_quality_vectors",
                "speaker": label,
                "vector_count": len(profile_vectors),
                "min_quality_vectors": int(min_quality_vectors),
                "sample_seconds": round(sample_seconds, 3),
            })
            continue
        if require_enrollment_quality and "mixed_voice_profile" in enrollment_reasons:
            rejected_profiles.append({
                "reason": "mixed_voice_profile",
                "speaker": label,
                "sample_seconds": round(sample_seconds, 3),
                "quality": profile_quality,
            })
            continue
        profile_matrix = np.stack(profile_vectors)
        emb = np.mean(profile_matrix, axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        enroll_embeddings = [
            vector / (np.linalg.norm(vector) + 1e-9)
            for vector in profile_vectors
        ]
        profiles.append({
            "name": label,
            "embedding": emb.astype(float).tolist(),
            "embeddings": [vec.astype(float).tolist() for vec in enroll_embeddings],
            "dims": int(emb.shape[0]),
            "anchor_count": len(items),
            "sample_seconds": round(sample_seconds, 3),
            "sample_seconds_basis": "unique_clean_wallclock",
            "quality": profile_quality,
            "enrollment_source": "user_confirmed_anchors",
            "enrollment_ready": enrollment_ready,
            "enrollment_reasons": enrollment_reasons,
        })

    if not profiles:
        rejected = [*rejected_profiles, *rejected_anchors]
        details = "; ".join(
            f"{item.get('speaker')}:{item.get('reason')}" for item in rejected[:3]
        )
        suffix = f"；{details}" if details else ""
        raise ValueError(f"已选片段无法通过声纹注册质量闸，请累计选择 10 秒以上同一人的清晰片段{suffix}")

    profile_names = [str(p["name"]) for p in profiles]
    profile_mat = np.asarray([p["embedding"] for p in profiles], dtype=np.float32)
    profile_mat = profile_mat / (np.linalg.norm(profile_mat, axis=1, keepdims=True) + 1e-9)

    if on_progress:
        on_progress({"stage": "voiceprint_reidentify_segments", "profiles": len(profiles)})

    out_segments: list[dict] = []
    changed = 0
    matched = 0
    reviewed = 0
    skipped_no_voice = 0
    score_rows: list[dict] = []

    for idx, raw_seg in enumerate(segments):
        seg = dict(raw_seg)
        try:
            start = float(seg.get("start"))
            end = float(seg.get("end"))
        except (TypeError, ValueError):
            out_segments.append(seg)
            continue
        emb, covered, _covered_intervals = _weighted_window_embedding(
            embeddings,
            subsegment_starts,
            subsegment_ends,
            start,
            end,
            allowed_mask=clean_embedding_mask,
        )
        if emb is None or covered < min_segment_seconds:
            skipped_no_voice += 1
            out_segments.append(seg)
            continue

        sims = emb @ profile_mat.T
        order = np.argsort(sims)[::-1]
        best_idx = int(order[0])
        best_score = float(sims[best_idx])
        second_score = float(sims[int(order[1])]) if len(order) > 1 else -1.0
        score_margin = best_score - second_score if len(order) > 1 else best_score
        target = profile_names[best_idx]
        current = str(seg.get("speaker") or "")
        high_confidence = best_score >= threshold and (len(profile_names) == 1 or score_margin >= margin)
        review_confidence = best_score >= review_threshold
        segment_overlap_ratio = overlap_ratio_for_window(
            (start, end),
            ctx.overlap_intervals,
        ) if ctx.overlap_available else 0.0
        if segment_overlap_ratio >= 0.08:
            high_confidence = False
            review_confidence = True

        score_rows.append({
            "index": idx,
            "start": start,
            "end": end,
            "speaker": current,
            "target": target,
            "score": round(best_score, 4),
            "margin": round(score_margin, 4),
            "covered_s": round(covered, 3),
            "changed": bool(high_confidence and current != target),
            "overlap_ratio": round(float(segment_overlap_ratio), 4),
        })

        if high_confidence:
            matched += 1
            seg["speaker_voiceprint_score"] = round(best_score, 4)
            seg["speaker_voiceprint_anchor"] = target
            seg["speaker_voiceprint_reidentified"] = True
            seg["speaker_assignment_review"] = False
            seg["speaker_voiceprint_review"] = False
            seg["speaker_review_reason"] = "已按声纹锚点回扫确认"
            if current != target:
                seg["original_speaker"] = current
                seg["speaker"] = target
                changed += 1
        elif review_confidence and current != target:
            reviewed += 1
            seg["speaker_assignment_review"] = True
            seg["speaker_voiceprint_review"] = True
            seg["speaker_voiceprint_reidentified"] = False
            seg["speaker_voiceprint_score"] = round(best_score, 4)
            seg["speaker_voiceprint_anchor"] = target
            reason = str(seg.get("speaker_review_reason") or "")
            if segment_overlap_ratio >= 0.08:
                addition = f"检测到重叠语音({segment_overlap_ratio:.0%})，未自动按声纹改派"
            else:
                addition = f"声纹锚点接近 {target}({best_score:.2f})，但未达自动改派阈值"
            if addition not in reason.split("；"):
                seg["speaker_review_reason"] = f"{reason}；{addition}" if reason else addition
        out_segments.append(seg)

    speakers: list[str] = []
    for seg in out_segments:
        speaker = str(seg.get("speaker") or "")
        if speaker and speaker not in speakers:
            speakers.append(speaker)

    return {
        "segments": out_segments,
        "speakers": speakers,
        "profiles": profiles,
        "stats": {
            "mode": "voiceprint_anchor_reidentify",
            "engine": "senko",
            "embedding_dim": int(profile_mat.shape[1]),
            "anchor_count": int(sum(p.get("anchor_count", 0) for p in profiles)),
            "profile_count": len(profiles),
            "rejected_anchor_count": len(rejected_anchors),
            "rejected_profile_count": len(rejected_profiles),
            "rejected_anchors": rejected_anchors,
            "rejected_profiles": rejected_profiles,
            "threshold": round(float(threshold), 3),
            "review_threshold": round(float(review_threshold), 3),
            "margin": round(float(margin), 3),
            "min_anchor_seconds": round(float(min_anchor_seconds), 3),
            "min_profile_seconds": round(float(min_profile_seconds), 3),
            "min_quality_vectors": int(min_quality_vectors),
            "min_anchor_consistency": round(float(min_anchor_consistency), 3),
            "min_profile_consistency": round(float(min_profile_consistency), 3),
            "require_enrollment_quality": bool(require_enrollment_quality),
            "segment_count": len(out_segments),
            "matched_segments": matched,
            "changed_segments": changed,
            "review_segments": reviewed,
            "skipped_no_voice_segments": skipped_no_voice,
            "score_rows": score_rows[:80],
            "reason": (
                f"声纹锚点回扫: 高置信改派 {changed} 段，"
                f"待确认 {reviewed} 段，注册 {len(profiles)} 人"
            ),
        },
    }


def diarize(
    audio: Path,
    segments: Sequence[dict],
    n_speakers: int = 0,  # 0=自动；显式人数在声学证据明显反对时受保护
    profiles: Iterable[dict] | None = None,
    on_progress=None,
) -> DiarizationResult:
    """对 whisper 给出的 segments 打 speaker 标签。

    流程:
      1. ffmpeg 转 16k mono wav
      2. senko 跑 VAD + Fbank + CAM++ embeddings + spectral clustering
      3. 给每个 whisper segment 找重叠最多的 senko speaker
      4. (可选)用 senko 192d centroid vs profiles 匹配真实姓名
    """
    profiles = list(profiles or [])

    ctx = _extract_senko_embeddings(audio, on_progress=on_progress)
    acoustic_count = _estimate_context_speaker_count(ctx)
    speaker_count_guard = _resolve_requested_speaker_count(
        int(n_speakers),
        acoustic_count,
    )
    d = _make_diarizer()
    _set_oracle_speaker_count(d, int(speaker_count_guard["oracle_n_speakers"]))

    if on_progress:
        on_progress({"stage": "diarize_run"})
    from senko.utils import suppress_stdout_stderr

    with suppress_stdout_stderr():
        senko_res, subsegments, sub_embeddings, sub_labels = _cluster_senko_embeddings(
            d,
            ctx,
            acoustic_count=acoustic_count,
        )

    raw_segs = senko_res.get('raw_segments') or []
    centroids = senko_res.get('speaker_centroids') or {}

    if on_progress:
        on_progress({
            "stage": "diarize_cluster",
            "speakers": senko_res.get('raw_speakers_detected', len(centroids)),
        })

    # ---- 把 ASR segment → senko speaker ----
    # 优先使用段内短声纹窗的多数归属。长段或多人连续说话时,把整段 embedding
    # 平均后再找 centroid 容易被男女声/重叠发言拉偏；短窗多数投票更接近
    # diarization 的真实边界。若没有有效短窗覆盖,再回退到 centroid embedding。
    centroid_ids = sorted(centroids)
    centroid_mat = []
    for speaker_id in centroid_ids:
        cent = np.asarray(centroids[speaker_id], dtype=np.float32)
        cent = cent / (np.linalg.norm(cent) + 1e-9)
        centroid_mat.append(cent)
    centroid_mat_np = np.stack(centroid_mat) if centroid_mat else np.empty((0, 192), dtype=np.float32)
    sub_embeddings = np.asarray(sub_embeddings, dtype=np.float32)
    if len(sub_embeddings):
        sub_embeddings = sub_embeddings / (np.linalg.norm(sub_embeddings, axis=1, keepdims=True) + 1e-9)
    subsegment_pitch_hz = np.asarray(ctx.subsegment_pitch_hz, dtype=np.float32)
    subsegment_pitch_confidence = np.asarray(ctx.subsegment_pitch_confidence, dtype=np.float32)
    subsegment_starts = np.asarray([float(s) for s, _ in subsegments], dtype=np.float32)
    subsegment_ends = np.asarray([float(e) for _, e in subsegments], dtype=np.float32)
    from .overlap_detector import overlap_ratio_for_window

    overlap_intervals = list(ctx.overlap_intervals or [])

    def overlapping_subsegment_indices(s: float, e: float) -> range:
        if not len(subsegments):
            return range(0, 0)
        left = int(np.searchsorted(subsegment_ends, float(s), side="right"))
        right = int(np.searchsorted(subsegment_starts, float(e), side="left"))
        return range(max(0, left), min(len(subsegments), right))

    def pitch_for_window(s: float, e: float) -> tuple[float | None, float, str]:
        if (
            not len(subsegments)
            or len(subsegment_pitch_hz) != len(subsegments)
            or len(subsegment_pitch_confidence) != len(subsegments)
        ):
            return None, 0.0, "unknown"
        pitch_values: list[tuple[float, float]] = []
        for idx in overlapping_subsegment_indices(s, e):
            ss = float(subsegment_starts[idx])
            ee = float(subsegment_ends[idx])
            overlap = max(0.0, min(float(ee), e) - max(float(ss), s))
            if overlap <= 0:
                continue
            pitch = float(subsegment_pitch_hz[idx])
            confidence = float(subsegment_pitch_confidence[idx])
            if not np.isfinite(pitch) or pitch <= 0.0 or confidence <= 0.0:
                continue
            weight = overlap * confidence
            pitch_values.append((pitch, weight))
        pitch = _weighted_median(pitch_values)
        if pitch is None:
            return None, 0.0, "unknown"
        covered = sum(weight for _, weight in pitch_values)
        speech_span = max(0.5, e - s)
        confidence = min(1.0, covered / speech_span)
        return float(pitch), round(float(confidence), 3), _voice_band_from_pitch(pitch, confidence)

    def speaker_timeline_for_window(s: float, e: float) -> list[dict]:
        timeline: list[dict] = []
        if not len(subsegments) or len(sub_labels) != len(subsegments) or not centroid_ids:
            return timeline
        for idx in overlapping_subsegment_indices(s, e):
            ss = float(subsegment_starts[idx])
            ee = float(subsegment_ends[idx])
            overlap_start = max(float(ss), s)
            overlap_end = min(float(ee), e)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap <= 0:
                continue
            label_idx = int(sub_labels[idx])
            if not 0 <= label_idx < len(centroid_ids):
                continue
            timeline.append({
                "start": overlap_start,
                "end": overlap_end,
                "speaker": centroid_ids[label_idx],
                "duration": overlap,
            })
        return timeline

    def label_for_window(s: float, e: float) -> tuple[str, float | None, dict[str, float], list[dict]]:
        if len(subsegments) and len(sub_labels) == len(subsegments) and centroid_ids:
            overlap_by_speaker: dict[str, float] = defaultdict(float)
            total_overlap = 0.0
            timeline = speaker_timeline_for_window(s, e)
            for item in timeline:
                overlap = float(item["duration"])
                overlap_by_speaker[str(item["speaker"])] += overlap
                total_overlap += overlap
            if overlap_by_speaker and total_overlap > 0:
                best_speaker, best_overlap = max(overlap_by_speaker.items(), key=lambda item: item[1])
                purity = best_overlap / max(total_overlap, 1e-9)
                # A strict threshold would discard exactly the hard cases we
                # care about. Use majority when present, but require at least a
                # small amount of covered speech so silence-only tails do not win.
                if purity >= 0.45 or best_overlap >= min(1.2, max(e - s, 0.0) * 0.35):
                    return best_speaker, float(purity), dict(overlap_by_speaker), timeline

        if len(subsegments) and len(centroid_mat_np):
            weights = []
            vecs = []
            for idx in overlapping_subsegment_indices(s, e):
                ss = float(subsegment_starts[idx])
                ee = float(subsegment_ends[idx])
                overlap = max(0.0, min(float(ee), e) - max(float(ss), s))
                if overlap > 0:
                    weights.append(overlap)
                    vecs.append(sub_embeddings[idx])
            if vecs:
                weight_arr = np.asarray(weights, dtype=np.float32)
                weight_arr = weight_arr / (float(weight_arr.sum()) + 1e-9)
                seg_emb = np.sum(np.stack(vecs) * weight_arr[:, None], axis=0)
                seg_emb = seg_emb / (np.linalg.norm(seg_emb) + 1e-9)
                speaker_idx = int(np.argmax(seg_emb @ centroid_mat_np.T))
                return centroid_ids[speaker_idx], None, {}, []

        if raw_segs:
            best_spk = None
            best_overlap = 0.0
            for rs in raw_segs:
                ov = max(0.0, min(rs['end'], e) - max(rs['start'], s))
                if ov > best_overlap:
                    best_overlap = ov
                    best_spk = rs['speaker']
            if best_spk is not None:
                return best_spk, None, {}, []
            mid = (s + e) / 2
            return min(
                raw_segs,
                key=lambda r: min(abs(mid - r['start']), abs(mid - r['end'])),
            )['speaker'], None, {}, []
        return "SPEAKER_01", None, {}, []

    # ---- 声纹库匹配:senko 192d centroid vs profiles ----
    matched: dict[str, str] = {}  # senko speaker id -> real name
    profile_match_review: list[dict] = []
    if profiles and centroids:
        prof_vectors_by_name: dict[str, list[np.ndarray]] = defaultdict(list)
        skipped_dim = 0
        for p in profiles:
            vectors = _normalize_profile_embeddings(dict(p))
            vectors = [vec for vec in vectors if vec.shape == (192,)]
            if not vectors:
                # 旧 resemblyzer 的 256 维 profile 不兼容 —— 用户需重新上传
                skipped_dim += 1
                continue
            name = str(p.get('name') or 'SPEAKER')
            prof_vectors_by_name[name].extend(vectors)
        prof_pairs: list[tuple[str, np.ndarray]] = []
        for name, vectors in sorted(prof_vectors_by_name.items()):
            emb = np.mean(np.stack(vectors), axis=0)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            prof_pairs.append((name, emb))

        score_matrix: dict[str, dict[str, float]] = {}
        for spk_id, cent in centroids.items():
            cent_arr = np.asarray(cent, dtype=np.float32)
            cent_arr = cent_arr / (np.linalg.norm(cent_arr) + 1e-9)
            score_matrix[spk_id] = {
                name: float(cent_arr @ emb)
                for name, emb in prof_pairs
            }
        assignment = _global_profile_assignment(score_matrix, MATCH_REVIEW_THRESHOLD)
        for spk_id, row in score_matrix.items():
            ordered = sorted(row.items(), key=lambda item: item[1], reverse=True)
            if not ordered:
                continue
            best_name, best_sim = ordered[0]
            second_sim = ordered[1][1] if len(ordered) > 1 else -1.0
            score_margin = best_sim - second_sim if len(ordered) > 1 else best_sim
            assigned_name = assignment.get(spk_id)
            if (
                assigned_name == best_name
                and best_sim >= MATCH_THRESHOLD
                and (len(ordered) == 1 or score_margin >= MATCH_MARGIN)
            ):
                matched[spk_id] = best_name
            elif best_sim >= MATCH_REVIEW_THRESHOLD:
                profile_match_review.append({
                    "speaker": spk_id,
                    "candidate": best_name,
                    "score": round(float(best_sim), 4),
                    "margin": round(float(score_margin), 4),
                    "assigned_candidate": assigned_name,
                    "reason": "声纹相似但未通过一对一高置信门禁",
                })

        if on_progress and skipped_dim:
            on_progress({
                "stage": "diarize_profile_skipped",
                "reason": "old 256d profiles incompatible — please re-upload voice samples",
                "skipped": skipped_dim,
            })

    if on_progress:
        on_progress({"stage": "diarize_assign", "matched": matched})

    # ---- 命名:senko SPEAKER_01/02 → SPEAKER_A/B(更友好)+ 应用匹配 ----
    # 用首次出现顺序命名,不要用聚类 id 排序。这样 A/B/C/D 对用户来说稳定可解释:
    # 第一个开口的人=A,第二个=B,后续依次类推。
    senko_speakers = _ordered_senko_speakers(raw_segs, centroids)
    label_map: dict[str, str] = {}
    reserved_display_names = set(matched.values())
    next_letter_idx = 0
    for sk in senko_speakers:
        if sk in matched:
            label_map[sk] = matched[sk]
        else:
            candidate = f"SPEAKER_{chr(ord('A') + next_letter_idx)}"
            while candidate in reserved_display_names:
                next_letter_idx += 1
                candidate = f"SPEAKER_{chr(ord('A') + next_letter_idx)}"
            label_map[sk] = candidate
            reserved_display_names.add(candidate)
            next_letter_idx += 1

    stable_centroid_ids, stable_centroid_mat, stable_centroid_stats = _build_stable_speaker_centroids(
        sub_embeddings,
        sub_labels,
        centroid_ids,
        centroids,
        overlap_ratios=np.asarray(ctx.subsegment_overlap_ratios, dtype=np.float32),
    )
    stable_centroid_stats["display_speakers"] = [
        label_map.get(speaker_id, speaker_id)
        for speaker_id in stable_centroid_ids
    ]

    exact_window_refs: list[tuple[int, int]] = []
    exact_windows: list[tuple[float, float]] = []
    for segment_index, segment in enumerate(segments):
        try:
            segment_start = float(segment.get("start", 0.0))
            segment_end = float(segment.get("end", segment_start))
        except (TypeError, ValueError):
            continue
        sync_cues = segment.get("sync_cues")
        if not isinstance(sync_cues, list) or not sync_cues:
            continue
        raw_timeline = speaker_timeline_for_window(segment_start, segment_end)
        change_points = [
            float(item["start"])
            for position, item in enumerate(raw_timeline[1:], start=1)
            if item.get("speaker") != raw_timeline[position - 1].get("speaker")
        ]
        if not change_points:
            continue
        bounded_cues: list[tuple[int, float, float]] = []
        for cue_index, cue in enumerate(sync_cues):
            if not isinstance(cue, dict):
                continue
            try:
                cue_start = max(segment_start, float(cue.get("start")))
                cue_end = min(segment_end, float(cue.get("end")))
            except (TypeError, ValueError):
                continue
            if cue_end <= cue_start:
                continue
            bounded_cues.append((cue_index, cue_start, cue_end))
        selected_cue_indices = _exact_cue_positions_near_changes(
            bounded_cues,
            change_points,
        )
        for position in sorted(selected_cue_indices):
            cue_index, cue_start, cue_end = bounded_cues[position]
            safe_window = _normalized_exact_cue_window(cue_start, cue_end)
            if safe_window is None:
                continue
            exact_window_refs.append((segment_index, cue_index))
            exact_windows.append(safe_window)

    exact_by_segment: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    exact_embedding_error = ""
    if stable_centroid_ids and exact_windows:
        try:
            exact_values = _extract_exact_cue_embeddings(
                audio,
                d,
                exact_windows,
                on_progress=on_progress,
            )
            for (segment_index, cue_index), vector in zip(exact_window_refs, exact_values):
                exact_by_segment[segment_index][cue_index] = vector
        except Exception as exc:
            exact_embedding_error = f"{type(exc).__name__}: {exc}"
    stable_centroid_stats.update({
        "exact_cue_embeddings": sum(len(rows) for rows in exact_by_segment.values()),
        "exact_cue_embedding_available": bool(exact_by_segment),
        "exact_cue_embedding_error": exact_embedding_error,
    })

    # ---- 给每个 whisper segment 打标签 ----
    out_segs: list[DiarizedSegment] = []
    speakers_seen: list[str] = []
    speaker_pitch_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    overlap_candidate_count = 0
    for segment_index, seg in enumerate(segments):
        s, e = float(seg['start']), float(seg['end'])
        senko_spk, confidence, raw_votes, raw_timeline = label_for_window(s, e)
        pitch_hz, pitch_confidence, voice_band = pitch_for_window(s, e)
        spk = label_map.get(senko_spk, "SPEAKER_A")
        speaker_votes = {
            label_map.get(raw_speaker, raw_speaker): round(float(duration), 3)
            for raw_speaker, duration in raw_votes.items()
        }
        mapped_timeline = []
        previous_speaker = None
        for item in raw_timeline:
            mapped_speaker = label_map.get(str(item["speaker"]), str(item["speaker"]))
            start = round(float(item["start"]), 3)
            end = round(float(item["end"]), 3)
            if (
                mapped_timeline
                and mapped_timeline[-1]["speaker"] == mapped_speaker
                and abs(float(mapped_timeline[-1]["end"]) - start) <= 0.05
            ):
                mapped_timeline[-1]["end"] = end
                mapped_timeline[-1]["duration"] = round(
                    float(mapped_timeline[-1]["duration"]) + float(item["duration"]),
                    3,
                )
            else:
                mapped_timeline.append({
                    "start": start,
                    "end": end,
                    "speaker": mapped_speaker,
                    "duration": round(float(item["duration"]), 3),
                })
            previous_speaker = mapped_speaker
        change_points = [
            float(item["start"])
            for pos, item in enumerate(mapped_timeline[1:], start=1)
            if item.get("speaker") != mapped_timeline[pos - 1].get("speaker")
        ]
        secondary_duration = 0.0
        if speaker_votes:
            ordered_votes = sorted(speaker_votes.values(), reverse=True)
            if len(ordered_votes) >= 2:
                secondary_duration = float(ordered_votes[1])
        overlap_risk = (
            len([v for v in speaker_votes.values() if float(v) >= 0.8]) >= 2
            and max(speaker_votes.values()) / max(0.001, sum(float(v) for v in speaker_votes.values())) < 0.72
            and secondary_duration >= 1.2
        )
        true_overlap_ratio = (
            overlap_ratio_for_window((s, e), overlap_intervals)
            if ctx.overlap_available else 0.0
        )
        true_overlap_seconds = true_overlap_ratio * max(0.0, e - s)
        true_overlap_risk = true_overlap_ratio >= 0.08 and true_overlap_seconds >= 0.12
        overlap_confidence = None
        if true_overlap_risk:
            confidences = [
                float(item.get("max_confidence", item.get("confidence", 0.0)) or 0.0)
                for item in overlap_intervals
                if max(0.0, min(float(item.get("end", 0.0)), e) - max(float(item.get("start", 0.0)), s)) > 0.0
            ]
            if confidences:
                overlap_confidence = max(confidences)
        if spk not in speakers_seen:
            speakers_seen.append(spk)
        if pitch_hz is not None and pitch_confidence >= 0.25:
            speaker_pitch_values[spk].append((float(pitch_hz), max(0.05, float(pitch_confidence)) * max(0.1, e - s)))
        overlap_candidates = _speaker_overlap_candidates_for_window(
            s,
            e,
            primary_timeline=raw_segs,
            subsegments=subsegments,
            sub_labels=sub_labels,
            speaker_ids=centroid_ids,
            overlap_intervals=overlap_intervals if ctx.overlap_available else [],
            label_map=label_map,
        )
        overlap_candidate_count += len(overlap_candidates)
        cue_embedding_evidence = _speaker_cue_embedding_evidence(
            dict(seg),
            sub_embeddings,
            subsegment_starts,
            subsegment_ends,
            stable_centroid_ids,
            stable_centroid_mat,
            label_map,
            overlap_ratios=np.asarray(ctx.subsegment_overlap_ratios, dtype=np.float32),
            overlap_intervals=overlap_intervals,
            overlap_available=bool(ctx.overlap_available),
            exact_embeddings=exact_by_segment.get(segment_index),
        )
        out_segs.append(DiarizedSegment(
            start=s, end=e,
            text=str(seg.get('text') or ''),
            speaker=spk,
            speaker_confidence=round(float(confidence), 3) if confidence is not None else None,
            speaker_votes=speaker_votes or None,
            voice_pitch_hz=round(float(pitch_hz), 1) if pitch_hz is not None else None,
            voice_pitch_confidence=round(float(pitch_confidence), 3) if pitch_hz is not None else None,
            voice_band=voice_band,
            speaker_subsegments=mapped_timeline or None,
            speaker_change_points=[round(float(t), 3) for t in change_points] or None,
            speaker_overlap_risk=bool(overlap_risk or true_overlap_risk) or None,
            overlap_ratio=round(float(true_overlap_ratio), 4) if ctx.overlap_available else None,
            speaker_overlap_confidence=(
                round(float(overlap_confidence), 4)
                if overlap_confidence is not None else None
            ),
            speaker_overlap_candidates=overlap_candidates or None,
            speaker_cue_embeddings=cue_embedding_evidence or None,
        ))

    speaker_voice_summary: dict[str, dict] = {}
    for speaker, values in speaker_pitch_values.items():
        pitch = _weighted_median(values)
        if pitch is None:
            continue
        total_weight = sum(weight for _, weight in values)
        confidence = min(1.0, total_weight / 18.0)
        speaker_voice_summary[speaker] = {
            "pitch_hz": round(float(pitch), 1),
            "pitch_confidence": round(float(confidence), 3),
            "voice_band": _voice_band_from_pitch(pitch, confidence),
            "samples": len(values),
        }

    acoustic_count = dict(senko_res.get("acoustic_speaker_count") or {})
    model_recommended_n_speakers = (
        int(acoustic_count["recommended_n_speakers"])
        if acoustic_count.get("available")
        else None
    )
    requested_count = int(n_speakers or 0)
    model_selected_n_speakers = speaker_count_guard["selected_n_speakers"]
    relative_eigengaps = acoustic_count.get("relative_eigengaps") or {}
    model_recommended_score = acoustic_count.get('relative_eigengap')
    model_selected_score = relative_eigengaps.get(str(model_selected_n_speakers))
    if model_selected_score is None and model_selected_n_speakers == model_recommended_n_speakers:
        model_selected_score = model_recommended_score

    return DiarizationResult(
        segments=out_segs,
        speakers=speakers_seen,
        cluster_count=len(senko_speakers),
        # 公开的 matched_profiles 用最终展示名(SPEAKER_A 或真名)→ 真名
        # 这样 UI 端展示一致
        matched_profiles={label_map[sk]: matched[sk] for sk in matched if sk in label_map},
        stats={
            'engine': 'senko',
            'runtime_backend': ctx.runtime_backend,
            'fallback_reason': ctx.runtime_fallback_reason,
            'vad_fallback_reason': ctx.runtime_vad_fallback_reason,
            'embedding_fallback_reason': ctx.runtime_embedding_fallback_reason,
            'embedding_dim': 192,
            'senko_raw_segments': len(raw_segs),
            'senko_subsegments': len(subsegments),
            'overlap_detection': {
                'available': bool(ctx.overlap_available),
                'interval_count': len(overlap_intervals),
                'filtered_subsegments': int(senko_res.get('overlap_filtered_subsegments', 0)),
                'cluster_filter_enabled': bool(senko_res.get('overlap_filter_enabled', False)),
                **dict(ctx.overlap_stats or {}),
            },
            'overlap_second_speaker': {
                'available': bool(ctx.overlap_available and centroid_ids and raw_segs),
                'method': 'osd_campp_context_v1',
                'candidate_count': overlap_candidate_count,
                'osd_confidence_threshold': OVERLAP_SECONDARY_OSD_CONFIDENCE,
                'context_seconds': OVERLAP_SECONDARY_CONTEXT_SECONDS,
                'window_support_threshold': OVERLAP_SECONDARY_WINDOW_SUPPORT,
                'context_weight': OVERLAP_SECONDARY_CONTEXT_WEIGHT,
                'minimum_interval_seconds': OVERLAP_SECONDARY_MIN_SECONDS,
                'primary_speaker_unchanged': True,
            },
            'assignment': 'subsegment_majority',
            'cluster_result_cache_hit': bool(senko_res.get('cluster_result_cache_hit', False)),
            'spectral_workspace_cache_hit': bool(senko_res.get('spectral_workspace_cache_hit', False)),
            'senko_speakers_detected': int(senko_res.get('raw_speakers_detected', 0)),
            'senko_speakers_merged': int(senko_res.get('merged_speakers_detected', 0)),
            'segment_count': len(out_segs),
            'matched_profile_count': len(matched),
            'profile_match_mode': 'global_one_to_one_high_confidence',
            'profile_match_review': profile_match_review,
            'speaker_voice_summary': speaker_voice_summary,
            'cue_embedding_verification': stable_centroid_stats,
            'model_recommended_n_speakers': model_recommended_n_speakers,
            'model_recommended_score': model_recommended_score,
            'model_selected_n_speakers': model_selected_n_speakers,
            'model_selected_score': model_selected_score,
            'requested_n_speakers': requested_count,
            'speaker_count_guard': speaker_count_guard,
            'model_recommended_confidence': acoustic_count.get('confidence', 0.0),
            'model_recommended_confidence_level': acoustic_count.get('confidence_level', 'low'),
            'model_recommended_diagnostics': acoustic_count,
            'timing': senko_res.get('timing_stats', {}),
            'duration_s': float(raw_segs[-1]['end']) if raw_segs else 0.0,
        },
    )
