"""Conservative CAM++ repair for long transcript cues missed by primary VAD.

The primary diarizer remains authoritative. This module only revisits a sync
cue when neither the Senko sliding timeline nor its cue embedding produced any
evidence. It never changes transcript text, timestamps, sync cues, or segment
geometry, and it fails closed when the optional local model is unavailable.
"""
from __future__ import annotations

import math
import os
import subprocess
import threading
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ..core.audio import find_ffmpeg


SAMPLE_RATE = 16_000
MODEL_REPOSITORIES = (
    "damo/speech_campplus_sv_zh-cn_16k-common",
    "iic/speech_campplus_sv_zh-cn_16k-common",
)

# These gates are intentionally narrower than regular cue projection. They are
# fixed from the frozen continuous-evaluation arena, not from recording names.
MIN_TARGET_DURATION_SECONDS = 4.0
MIN_ACCEPT_SCORE = 0.50
MIN_ACCEPT_MARGIN = 0.20
MIN_WINDOW_AGREEMENT = 1.0
MIN_ANCHOR_DURATION_SECONDS = 1.5
MIN_ANCHORS_PER_SPEAKER = 3
MAX_ANCHORS_PER_SPEAKER = 24
MAX_TARGET_WINDOWS = 3
MAX_OVERLAP_RATIO = 0.08
PROTECTED_SEGMENT_FLAGS = (
    "speaker_calibrated",
    "speaker_voiceprint_reidentified",
    "voice_band_repaired",
    "continuity_repaired",
    "speaker_handoff_voice_guard_repaired",
)


_PIPELINE_CACHE: dict[str, Any] = {}
_PIPELINE_LOCK = threading.Lock()


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _interval_key(start: Any, end: Any) -> tuple[float, float]:
    return round(_finite_float(start), 3), round(_finite_float(end), 3)


def _union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def _speaker_cue_index(segment: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    sync_cues = segment.get("sync_cues")
    if not isinstance(sync_cues, list):
        return {}
    sync_keys = {
        _interval_key(cue.get("start"), cue.get("end")): index
        for index, cue in enumerate(sync_cues)
        if isinstance(cue, dict)
    }
    indexed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in segment.get("speaker_cues") or []:
        if not isinstance(row, dict):
            continue
        cue_index = row.get("cue_index")
        try:
            parsed_index = int(cue_index)
        except (TypeError, ValueError):
            parsed_index = sync_keys.get(_interval_key(row.get("start"), row.get("end")), -1)
        if 0 <= parsed_index < len(sync_cues):
            indexed[parsed_index].append(row)
    return dict(indexed)


def _window_votes(
    segment: dict[str, Any], start: float, end: float
) -> dict[str, float]:
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in segment.get("speaker_subsegments") or []:
        if not isinstance(row, dict):
            continue
        speaker = str(row.get("speaker") or "")
        row_start = _finite_float(row.get("start"))
        row_end = _finite_float(row.get("end"), row_start)
        overlap_start = max(start, row_start)
        overlap_end = min(end, row_end)
        if speaker and overlap_end > overlap_start:
            intervals[speaker].append((overlap_start, overlap_end))
    return {
        speaker: _union_duration(values)
        for speaker, values in intervals.items()
    }


def _embedding_evidence(segment: dict[str, Any], cue_index: int) -> dict[str, Any] | None:
    return next(
        (
            dict(row)
            for row in segment.get("speaker_cue_embeddings") or []
            if isinstance(row, dict)
            and int(_finite_float(row.get("cue_index"), -1.0)) == cue_index
        ),
        None,
    )


def _collect_cues(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    protected_indexes = {
        int(row["index"])
        for row in (candidate.get("human_annotation_reuse") or {}).get("rows") or []
        if isinstance(row, dict) and isinstance(row.get("index"), int)
    }
    for segment_index, segment in enumerate(candidate.get("segments") or []):
        if not isinstance(segment, dict):
            continue
        if segment_index in protected_indexes or any(
            segment.get(flag) for flag in PROTECTED_SEGMENT_FLAGS
        ):
            continue
        sync_cues = segment.get("sync_cues")
        if not isinstance(sync_cues, list) or not sync_cues:
            continue
        projected = _speaker_cue_index(segment)
        segment_speaker = str(segment.get("speaker") or "")
        for cue_index, cue in enumerate(sync_cues):
            if not isinstance(cue, dict):
                continue
            start = _finite_float(cue.get("start"))
            end = _finite_float(cue.get("end"), start)
            if end <= start:
                continue
            projected_rows = projected.get(cue_index) or []
            current = (
                str(projected_rows[0].get("speaker") or "")
                if len(projected_rows) == 1
                else segment_speaker
            )
            if not current:
                continue
            votes = _window_votes(segment, start, end)
            total = sum(votes.values())
            ordered = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
            window_speaker = ordered[0][0] if ordered else ""
            window_purity = ordered[0][1] / total if total > 0.0 else 0.0
            window_coverage = total / max(end - start, 1e-9)
            direct = _embedding_evidence(segment, cue_index)
            direct_assign = bool(
                direct
                and direct.get("decision") == "assign"
                and str(direct.get("speaker") or "") == current
                and _finite_float(direct.get("score")) >= 0.70
                and _finite_float(direct.get("margin")) >= 0.06
            )
            window_assign = bool(
                window_speaker == current
                and window_purity >= 0.85
                and window_coverage >= 0.50
            )
            overlap_ratio = _finite_float(segment.get("overlap_ratio"))
            cues.append({
                "segment_index": segment_index,
                "cue_index": cue_index,
                "start": start,
                "end": end,
                "duration": end - start,
                "text": str(cue.get("text") or ""),
                "current": current,
                "missing": not votes and direct is None,
                "trusted": direct_assign or window_assign,
                "writable": not projected_rows or len(projected_rows) == 1,
                "direct": direct,
                "window_purity": window_purity,
                "window_coverage": window_coverage,
                "segment_confidence": _finite_float(
                    segment.get("speaker_confidence"), 0.0
                ),
                "overlap_ratio": overlap_ratio,
                "overlap_risk": bool(segment.get("speaker_overlap_risk")),
            })
    return cues


def _anchor_rank(cue: dict[str, Any]) -> tuple[float, ...]:
    direct = cue.get("direct") or {}
    direct_assign = float(
        direct.get("decision") == "assign"
        and str(direct.get("speaker") or "") == cue.get("current")
    )
    return (
        direct_assign,
        _finite_float(direct.get("score")),
        _finite_float(direct.get("margin")),
        _finite_float(cue.get("window_purity")),
        _finite_float(cue.get("window_coverage")),
        _finite_float(cue.get("segment_confidence")),
        min(_finite_float(cue.get("duration")), 8.0),
    )


def _select_anchors(cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cue in cues:
        if (
            cue.get("trusted")
            and not cue.get("overlap_risk")
            and _finite_float(cue.get("overlap_ratio")) < MAX_OVERLAP_RATIO
            and _finite_float(cue.get("duration")) >= MIN_ANCHOR_DURATION_SECONDS
            and cue.get("current")
        ):
            grouped[str(cue["current"])].append(cue)

    selected: list[dict[str, Any]] = []
    for speaker, rows in sorted(grouped.items()):
        speaker_rows: list[dict[str, Any]] = []
        for cue in sorted(rows, key=_anchor_rank, reverse=True):
            midpoint = (_finite_float(cue["start"]) + _finite_float(cue["end"])) / 2.0
            if any(
                abs(
                    midpoint
                    - (_finite_float(existing["start"]) + _finite_float(existing["end"]))
                    / 2.0
                )
                < 3.0
                for existing in speaker_rows
            ):
                continue
            speaker_rows.append(cue)
            if len(speaker_rows) >= MAX_ANCHORS_PER_SPEAKER:
                break
        if len(speaker_rows) >= MIN_ANCHORS_PER_SPEAKER:
            selected.extend({**cue, "anchor_speaker": speaker} for cue in speaker_rows)
    return selected


def _model_roots() -> list[Path]:
    roots: list[Path] = []
    for name in ("LOCALSCRIBE_MODELSCOPE_CACHE", "MODELSCOPE_CACHE"):
        raw = os.environ.get(name)
        if raw:
            base = Path(raw).expanduser()
            roots.extend((base, base / "models"))
    resources = os.environ.get("LOCALSCRIBE_RESOURCES")
    if resources:
        roots.append(Path(resources).expanduser() / "modelscope/hub/models")
    roots.append(Path.home() / ".cache/modelscope/hub/models")
    deduplicated: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved.name == "hub":
            resolved = resolved / "models"
        if resolved not in deduplicated:
            deduplicated.append(resolved)
    return deduplicated


def resolve_local_model_path() -> Path:
    for root in _model_roots():
        for repository in MODEL_REPOSITORIES:
            candidate = root / repository
            if (
                candidate.is_dir()
                and (candidate / "campplus_cn_common.bin").is_file()
                and ((candidate / "config.yaml").is_file() or (candidate / "configuration.json").is_file())
            ):
                return candidate.resolve()
    searched = [str(root / repository) for root in _model_roots() for repository in MODEL_REPOSITORIES]
    raise FileNotFoundError(
        "local CAM++ speaker-verification model is unavailable; searched: "
        + ", ".join(searched)
    )


def _get_pipeline(model_path: Path):
    key = str(model_path.resolve())
    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        return cached
    os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from modelscope.pipelines import pipeline

    value = pipeline(task="speaker-verification", model=key)
    _PIPELINE_CACHE[key] = value
    return value


def _decode_audio_16k(audio: Path) -> np.ndarray:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required for CAM++ exact embedding fallback")
    decoded = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    samples = np.frombuffer(decoded.stdout, dtype="<f4").copy()
    if samples.size < SAMPLE_RATE:
        raise RuntimeError("decoded audio is shorter than one second")
    return samples


def _window_samples(
    samples: np.ndarray, start: float, end: float
) -> np.ndarray | None:
    left = max(0, int(round(start * SAMPLE_RATE)))
    right = min(len(samples), int(round(end * SAMPLE_RATE)))
    if right - left < int(1.45 * SAMPLE_RATE):
        return None
    return np.asarray(samples[left:right], dtype=np.float32)


def _rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    centered = values - float(np.mean(values))
    return float(np.sqrt(np.mean(np.square(centered), dtype=np.float64)))


def _energy_windows(
    samples: np.ndarray, start: float, end: float, *, limit: int
) -> list[dict[str, Any]]:
    duration = end - start
    if duration < 1.5:
        return []
    window_duration = min(3.0, duration)
    if duration <= window_duration + 1e-6:
        starts = [start]
    else:
        count = max(1, int(math.floor((duration - window_duration) / 0.5)) + 1)
        starts = [start + index * 0.5 for index in range(count)]
        final_start = end - window_duration
        if not starts or abs(starts[-1] - final_start) > 0.05:
            starts.append(final_start)

    candidates: list[dict[str, Any]] = []
    for window_start in starts:
        window_end = min(end, window_start + window_duration)
        values = _window_samples(samples, window_start, window_end)
        if values is not None:
            candidates.append({
                "start": window_start,
                "end": window_end,
                "samples": values,
                "rms": _rms(values),
            })

    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: -_finite_float(row["rms"])):
        midpoint = (_finite_float(candidate["start"]) + _finite_float(candidate["end"])) / 2.0
        if any(
            abs(
                midpoint
                - (_finite_float(existing["start"]) + _finite_float(existing["end"])) / 2.0
            )
            < 1.0
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _robust_prototype(values: list[np.ndarray]) -> tuple[np.ndarray | None, dict[str, Any]]:
    if len(values) < MIN_ANCHORS_PER_SPEAKER:
        return None, {"available": False, "reason": "insufficient_anchors"}
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return None, {"available": False, "reason": "invalid_anchor_embeddings"}
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    similarities = matrix @ matrix.T
    centrality = np.median(similarities, axis=1)
    medoid_index = int(np.argmax(centrality))
    keep_count = max(MIN_ANCHORS_PER_SPEAKER, int(math.ceil(len(matrix) * 0.60)))
    keep_indices = np.argsort(similarities[medoid_index])[::-1][:keep_count]
    retained = matrix[keep_indices]
    prototype = np.mean(retained, axis=0)
    prototype = prototype / (np.linalg.norm(prototype) + 1e-9)
    retained_scores = retained @ prototype
    return prototype.astype(np.float32), {
        "available": True,
        "input_anchors": int(len(matrix)),
        "retained_anchors": int(len(retained)),
        "medoid_similarity": round(float(centrality[medoid_index]), 4),
        "retained_median_similarity": round(float(np.median(retained_scores)), 4),
        "retained_min_similarity": round(float(np.min(retained_scores)), 4),
    }


def _run_embeddings(pipeline: Any, waveforms: list[np.ndarray]) -> np.ndarray:
    result = pipeline(waveforms, output_emb=True)
    embeddings = np.asarray(result["embs"], dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]
    if embeddings.ndim != 2 or len(embeddings) != len(waveforms):
        raise RuntimeError("CAM++ embedding count mismatch")
    return embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)


def _materialize_speaker_cues(
    segment: dict[str, Any], cue_speakers: dict[int, str]
) -> list[dict[str, Any]] | None:
    sync_cues = segment.get("sync_cues")
    if not isinstance(sync_cues, list) or not sync_cues:
        return None
    existing = segment.get("speaker_cues")
    if isinstance(existing, list) and existing:
        indexed = _speaker_cue_index(segment)
        if any(len(rows) != 1 for rows in indexed.values()) or len(indexed) != len(sync_cues):
            return None
        return [dict(indexed[index][0]) for index in range(len(sync_cues))]

    default_speaker = str(segment.get("speaker") or "")
    if not default_speaker:
        return None
    rows: list[dict[str, Any]] = []
    for cue_index, cue in enumerate(sync_cues):
        if not isinstance(cue, dict):
            return None
        rows.append({
            "cue_index": cue_index,
            "start": cue.get("start"),
            "end": cue.get("end"),
            "text": str(cue.get("text") or ""),
            "speaker": cue_speakers.get(cue_index, default_speaker),
            "confidence": _finite_float(segment.get("speaker_confidence"), 0.5),
            "source": "segment_assignment",
        })
    return rows


def _apply_proposals(
    candidate: dict[str, Any], proposals: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = {
        **candidate,
        "segments": [deepcopy(segment) for segment in candidate.get("segments") or []],
        "stats": dict(candidate.get("stats") or {}),
    }
    accepted = [
        proposal
        for proposal in proposals
        if proposal["target"] != proposal["current"]
        and _finite_float(proposal["duration"]) >= MIN_TARGET_DURATION_SECONDS
        and _finite_float(proposal["score"]) >= MIN_ACCEPT_SCORE
        and _finite_float(proposal["margin"]) >= MIN_ACCEPT_MARGIN
        and _finite_float(proposal["window_agreement"]) >= MIN_WINDOW_AGREEMENT
    ]
    by_segment: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for proposal in accepted:
        by_segment[int(proposal["segment_index"])].append(proposal)

    applied: list[dict[str, Any]] = []
    for segment_index, segment_proposals in by_segment.items():
        if not 0 <= segment_index < len(output["segments"]):
            continue
        segment = output["segments"][segment_index]
        current_by_cue = {
            int(proposal["cue_index"]): str(proposal["current"])
            for proposal in segment_proposals
        }
        speaker_cues = _materialize_speaker_cues(segment, current_by_cue)
        if speaker_cues is None:
            continue
        for proposal in segment_proposals:
            cue_index = int(proposal["cue_index"])
            if not 0 <= cue_index < len(speaker_cues):
                continue
            row = dict(speaker_cues[cue_index])
            row.update({
                "speaker": proposal["target"],
                "confidence": round(_finite_float(proposal["score"]), 3),
                "source": "campp_exact_missing_evidence",
            })
            speaker_cues[cue_index] = row
            applied.append(proposal)
        if segment_proposals:
            segment["speaker_cues"] = speaker_cues
            segment["speaker_cue_mode"] = "campp_exact_missing_evidence"
    return output, applied


def repair_missing_evidence_cues(
    audio: Path,
    candidate: dict[str, Any],
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Return a copied candidate with only high-confidence missing cues repaired."""
    started = time.perf_counter()
    base = {
        **candidate,
        "segments": [deepcopy(segment) for segment in candidate.get("segments") or []],
        "stats": dict(candidate.get("stats") or {}),
    }
    diagnostics: dict[str, Any] = {
        "available": False,
        "applied": False,
        "method": "campp_exact_missing_evidence_v1",
        "frozen_transcript_geometry": True,
        "min_target_duration_seconds": MIN_TARGET_DURATION_SECONDS,
        "min_score": MIN_ACCEPT_SCORE,
        "min_margin": MIN_ACCEPT_MARGIN,
        "min_window_agreement": MIN_WINDOW_AGREEMENT,
        "candidate_cues": 0,
        "changed_cues": 0,
        "changed_seconds": 0.0,
    }

    try:
        audio = Path(audio).expanduser().resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        primary_engine = str((base.get("stats") or {}).get("engine") or "").strip().lower()
        if primary_engine not in {"senko", "campp", "cam++"}:
            diagnostics.update({
                "available": True,
                "reason": "primary_engine_not_campp",
                "primary_engine": primary_engine or "unknown",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            })
            base["stats"]["exact_embedding_fallback"] = diagnostics
            return base
        cues = _collect_cues(base)
        targets = [
            cue
            for cue in cues
            if cue.get("missing")
            and cue.get("writable")
            and _finite_float(cue.get("duration")) >= MIN_TARGET_DURATION_SECONDS
        ]
        diagnostics["candidate_cues"] = len(targets)
        if not targets:
            diagnostics.update({
                "available": True,
                "reason": "no_long_missing_evidence_cues",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            })
            base["stats"]["exact_embedding_fallback"] = diagnostics
            return base

        anchors = _select_anchors(cues)
        anchor_counts: dict[str, int] = defaultdict(int)
        for anchor in anchors:
            anchor_counts[str(anchor["anchor_speaker"])] += 1
        diagnostics["anchor_counts"] = dict(sorted(anchor_counts.items()))
        if len(anchor_counts) < 2:
            diagnostics.update({
                "available": True,
                "reason": "insufficient_trusted_speaker_anchors",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            })
            base["stats"]["exact_embedding_fallback"] = diagnostics
            return base

        model_path = resolve_local_model_path()
        if on_progress:
            on_progress({
                "stage": "diarize_exact_missing_evidence",
                "targets": len(targets),
                "speakers": len(anchor_counts),
            })
        samples = _decode_audio_16k(audio)
        descriptors: list[dict[str, Any]] = []
        waveforms: list[np.ndarray] = []
        for anchor in anchors:
            for window in _energy_windows(
                samples,
                _finite_float(anchor["start"]),
                _finite_float(anchor["end"]),
                limit=1,
            ):
                descriptors.append({
                    "kind": "anchor",
                    "speaker": anchor["anchor_speaker"],
                })
                waveforms.append(window["samples"])
        for target in targets:
            for window in _energy_windows(
                samples,
                _finite_float(target["start"]),
                _finite_float(target["end"]),
                limit=MAX_TARGET_WINDOWS,
            ):
                descriptors.append({
                    "kind": "target",
                    "segment_index": target["segment_index"],
                    "cue_index": target["cue_index"],
                    "start": window["start"],
                    "end": window["end"],
                    "rms": window["rms"],
                })
                waveforms.append(window["samples"])
        if not waveforms:
            raise RuntimeError("no valid CAM++ windows")

        with _PIPELINE_LOCK:
            pipeline = _get_pipeline(model_path)
            embeddings = _run_embeddings(pipeline, waveforms)

        anchor_values: dict[str, list[np.ndarray]] = defaultdict(list)
        target_values: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for descriptor, embedding in zip(descriptors, embeddings):
            if descriptor["kind"] == "anchor":
                anchor_values[str(descriptor["speaker"])].append(embedding)
            else:
                target_values[(
                    int(descriptor["segment_index"]),
                    int(descriptor["cue_index"]),
                )].append({**descriptor, "embedding": embedding})

        speaker_ids: list[str] = []
        prototypes: list[np.ndarray] = []
        prototype_stats: dict[str, Any] = {}
        for speaker, values in sorted(anchor_values.items()):
            prototype, stats = _robust_prototype(values)
            prototype_stats[speaker] = stats
            if prototype is not None:
                speaker_ids.append(speaker)
                prototypes.append(prototype)
        diagnostics["prototype_stats"] = prototype_stats
        if len(prototypes) < 2:
            diagnostics.update({
                "available": True,
                "reason": "insufficient_valid_speaker_prototypes",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            })
            base["stats"]["exact_embedding_fallback"] = diagnostics
            return base

        prototype_matrix = np.stack(prototypes)
        target_index = {
            (int(target["segment_index"]), int(target["cue_index"])): target
            for target in targets
        }
        proposals: list[dict[str, Any]] = []
        for key, windows in target_values.items():
            target = target_index.get(key)
            if target is None or not windows:
                continue
            votes: dict[str, int] = defaultdict(int)
            scored: list[dict[str, Any]] = []
            for window in windows:
                scores = np.asarray(window["embedding"] @ prototype_matrix.T)
                order = np.argsort(scores)[::-1]
                best_index = int(order[0])
                second_index = int(order[1])
                speaker = speaker_ids[best_index]
                votes[speaker] += 1
                scored.append({
                    "speaker": speaker,
                    "score": float(scores[best_index]),
                    "margin": float(scores[best_index] - scores[second_index]),
                    "start": _finite_float(window["start"]),
                    "end": _finite_float(window["end"]),
                    "rms": _finite_float(window["rms"]),
                })
            selected_speaker = sorted(
                votes,
                key=lambda speaker: (
                    -votes[speaker],
                    -max(row["score"] for row in scored if row["speaker"] == speaker),
                    speaker,
                ),
            )[0]
            agreeing = [row for row in scored if row["speaker"] == selected_speaker]
            best = max(agreeing, key=lambda row: (row["score"] + row["margin"], row["rms"]))
            proposals.append({
                **target,
                "target": selected_speaker,
                "score": best["score"],
                "margin": best["margin"],
                "window_agreement": len(agreeing) / len(scored),
                "window_count": len(scored),
                "best_window": best,
            })

        repaired, applied = _apply_proposals(base, proposals)
        diagnostics.update({
            "available": True,
            "applied": bool(applied),
            "reason": "high_confidence_cues_repaired" if applied else "no_proposal_passed_gates",
            "model": str(model_path),
            "proposal_count": len(proposals),
            "changed_cues": len(applied),
            "changed_seconds": round(sum(_finite_float(row["duration"]) for row in applied), 3),
            "changes": [
                {
                    "segment_index": int(row["segment_index"]),
                    "cue_index": int(row["cue_index"]),
                    "start": round(_finite_float(row["start"]), 3),
                    "end": round(_finite_float(row["end"]), 3),
                    "from": row["current"],
                    "to": row["target"],
                    "score": round(_finite_float(row["score"]), 4),
                    "margin": round(_finite_float(row["margin"]), 4),
                    "window_agreement": round(_finite_float(row["window_agreement"]), 4),
                }
                for row in applied
            ],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        })
        repaired["stats"]["exact_embedding_fallback"] = diagnostics
        return repaired
    except Exception as exc:
        diagnostics.update({
            "available": False,
            "applied": False,
            "reason": "fallback_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        })
        base["stats"]["exact_embedding_fallback"] = diagnostics
        return base


__all__ = [
    "repair_missing_evidence_cues",
    "resolve_local_model_path",
]
