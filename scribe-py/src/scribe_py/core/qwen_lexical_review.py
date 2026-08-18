"""Qwen3-ASR lexical review with a frozen SenseVoice timeline.

This module is intentionally isolated from the default transcription pipeline.
Qwen may replace text only after it aligns to the primary transcript; segment,
speaker, and sync-cue timing always remain owned by the primary result.
"""
from __future__ import annotations

import tempfile
import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable

from .strong_asr import (
    QWEN_MODEL,
    _extract_clip,
    _qwen_model_cached,
    _replace_cue_text_preserving_times,
    _split_text_by_weights,
    consensus_rewrite,
    normalized_text,
    text_similarity,
    timeline_fingerprint,
)
from .transcriber_funasr import _align_segments_to_timing_anchor
from .transcriber_qwen3 import Qwen3ASRTranscriber
from .types import Segment, TranscribeOptions


ProgressCallback = Callable[[dict[str, Any]], None]

_MAX_WINDOW_SECONDS = 90.0
_MIN_ALIGNMENT_RATIO = 0.65
_MIN_LENGTH_RATIO = 0.65
_MAX_LENGTH_RATIO = 1.35
_MAX_ESTIMATED_TIMING_RATIO = 0.40
# Local window selection only. The outer high-noise route remains stricter in
# strong_asr so normal recordings do not load Qwen.
_REFERENCE_DISAGREEMENT_RATIO = 0.90
_REFERENCE_MIN_MISMATCH_CHARS = 6
_REFERENCE_PARTIAL_LENGTH_BALANCE = 0.65
_REFERENCE_SHORT_DISAGREEMENT_RATIO = 0.55
_REFERENCE_BOUNDARY_SHIFTS_SECONDS = (-1.0, 0.0, 1.0)
_PARAFORMER_CONSENSUS_CONTEXT_SECONDS = 2.0
_PARAFORMER_SEGMENT_MIN_SIMILARITY = 0.65
_LEADING_PUNCTUATION_RE = re.compile(r"^([，。！？；：、,.!?;:]+)")


@dataclass(frozen=True)
class _FrozenCue:
    segment_index: int
    cue_index: int | None
    start: float
    end: float


@dataclass(frozen=True)
class _ReviewWindow:
    start: float
    end: float
    segment_indexes: tuple[int, ...]


@dataclass(frozen=True)
class _AlignedFragment:
    order: int
    start: float
    end: float
    text: str


def _frozen_cues(primary_segments: list[Segment]) -> list[_FrozenCue]:
    cues: list[_FrozenCue] = []
    for segment_index, segment in enumerate(primary_segments):
        if segment.sync_cues:
            for cue_index, cue in enumerate(segment.sync_cues):
                cues.append(
                    _FrozenCue(
                        segment_index=segment_index,
                        cue_index=cue_index,
                        start=float(cue.get("start", segment.start)),
                        end=float(cue.get("end", segment.end)),
                    )
                )
        else:
            cues.append(
                _FrozenCue(
                    segment_index=segment_index,
                    cue_index=None,
                    start=float(segment.start),
                    end=float(segment.end),
                )
            )
    return cues


def _overlap_seconds(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _distance_to_cue(start: float, end: float, cue: _FrozenCue) -> float:
    if _overlap_seconds(start, end, cue.start, cue.end) > 0:
        return 0.0
    if end <= cue.start:
        return cue.start - end
    return start - cue.end


def _equal_text_chars(left: str, right: str) -> int:
    left_normalized = normalized_text(left)
    right_normalized = normalized_text(right)
    if not left_normalized or not right_normalized:
        return 0
    return sum(
        left_end - left_start
        for operation, left_start, left_end, _right_start, _right_end in SequenceMatcher(
            None,
            left_normalized,
            right_normalized,
            autojunk=False,
        ).get_opcodes()
        if operation == "equal"
    )


def _frozen_metadata_matches(before: list[Segment], after: list[Segment]) -> bool:
    if len(before) != len(after):
        return False
    for primary, projected in zip(before, after):
        if (
            float(primary.start) != float(projected.start)
            or float(primary.end) != float(projected.end)
            or primary.speaker != projected.speaker
        ):
            return False
        primary_cues = list(primary.sync_cues or [])
        projected_cues = list(projected.sync_cues or [])
        if len(primary_cues) != len(projected_cues):
            return False
        for primary_cue, projected_cue in zip(primary_cues, projected_cues):
            if (
                float(primary_cue.get("start", 0.0))
                != float(projected_cue.get("start", 0.0))
                or float(primary_cue.get("end", 0.0))
                != float(projected_cue.get("end", 0.0))
            ):
                return False
    return True


def _aligned_fragments(source: list[Segment]) -> list[_AlignedFragment]:
    fragments: list[_AlignedFragment] = []
    order = 0
    for segment in source:
        segment_text = segment.text or ""
        cues = [cue for cue in (segment.sync_cues or []) if str(cue.get("text") or "")]
        cue_text = "".join(str(cue.get("text") or "") for cue in cues)
        if cues and normalized_text(cue_text) == normalized_text(segment_text):
            for cue in cues:
                fragments.append(
                    _AlignedFragment(
                        order=order,
                        start=float(cue.get("start", segment.start)),
                        end=float(cue.get("end", segment.end)),
                        text=str(cue.get("text") or ""),
                    )
                )
                order += 1
            continue
        if segment_text:
            fragments.append(
                _AlignedFragment(
                    order=order,
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment_text,
                )
            )
            order += 1
    return fragments


def project_aligned_text_to_frozen_timeline(
    primary_segments: Iterable[Segment],
    aligned_source_segments: Iterable[Segment],
) -> list[Segment]:
    """Project aligned source text onto primary cues without moving the timeline.

    Source segments that overlap several cues are split in proportion to their
    exact overlap durations. A source segment with no overlap is assigned to the
    nearest cue. Any projection that loses or reorders normalized source text is
    rejected with ``ValueError``.
    """
    primary = list(primary_segments)
    source = list(aligned_source_segments)
    before_fingerprint = timeline_fingerprint(primary)
    source_text = "".join(segment.text or "" for segment in source)
    source_normalized = normalized_text(source_text)
    targets = _frozen_cues(primary)
    target_original_text = {
        (segment_index, cue_index): str(cue.get("text") or "")
        for segment_index, segment in enumerate(primary)
        for cue_index, cue in enumerate(segment.sync_cues or [])
    }
    target_original_text.update({
        (segment_index, None): segment.text or ""
        for segment_index, segment in enumerate(primary)
        if not segment.sync_cues
    })

    if source_normalized and not targets:
        raise ValueError("source_text_has_no_frozen_cue_target")

    fragments: list[list[tuple[int, int, str]]] = [[] for _target in targets]
    for source_fragment in _aligned_fragments(source):
        text = source_fragment.text
        start = source_fragment.start
        end = max(source_fragment.end, start)
        overlaps = [
            (target_index, _overlap_seconds(start, end, target.start, target.end))
            for target_index, target in enumerate(targets)
        ]
        overlaps = [(target_index, overlap) for target_index, overlap in overlaps if overlap > 0]
        if not overlaps:
            target_index = min(
                range(len(targets)),
                key=lambda index: (_distance_to_cue(start, end, targets[index]), index),
            )
            fragments[target_index].append((source_fragment.order, 0, text))
            continue

        text_scores = [
            (
                target_index,
                _equal_text_chars(
                    text,
                    target_original_text.get(
                        (targets[target_index].segment_index, targets[target_index].cue_index),
                        "",
                    ),
                ),
                overlap,
            )
            for target_index, overlap in overlaps
        ]
        best_text_target, best_text_score, _best_text_overlap = max(
            text_scores,
            key=lambda item: (item[1], item[2], -item[0]),
        )
        min_anchor_chars = max(2, int(round(len(normalized_text(text)) * 0.20)))
        if best_text_score >= min_anchor_chars:
            fragments[best_text_target].append((source_fragment.order, 0, text))
            continue

        covered_duration = max(sum(overlap for _index, overlap in overlaps), 1e-6)
        largest_target, largest_overlap = max(overlaps, key=lambda item: (item[1], -item[0]))
        if largest_overlap / covered_duration >= 0.70:
            fragments[largest_target].append((source_fragment.order, 0, text))
            continue

        weights = [max(1, int(round(overlap * 1_000_000))) for _index, overlap in overlaps]
        parts = _split_text_by_weights(text, weights)
        if len(parts) != len(overlaps):
            raise ValueError("source_text_split_failed")
        for part_index, ((target_index, _overlap), part) in enumerate(zip(overlaps, parts)):
            fragments[target_index].append((source_fragment.order, part_index, part))

    projected = list(primary)
    cue_texts: dict[tuple[int, int | None], str] = {}
    for target, target_fragments in zip(targets, fragments):
        cue_texts[(target.segment_index, target.cue_index)] = "".join(
            text for _source_index, _part_index, text in sorted(target_fragments)
        )
    for target_index in range(1, len(targets)):
        current = targets[target_index]
        previous = targets[target_index - 1]
        current_key = (current.segment_index, current.cue_index)
        previous_key = (previous.segment_index, previous.cue_index)
        current_text = cue_texts.get(current_key, "")
        match = _LEADING_PUNCTUATION_RE.match(current_text)
        if not match:
            continue
        punctuation = match.group(1)
        cue_texts[previous_key] = cue_texts.get(previous_key, "") + punctuation
        cue_texts[current_key] = current_text[len(punctuation) :]

    for segment_index, current in enumerate(primary):
        if current.sync_cues:
            updated_cues = [
                {
                    **cue,
                    "text": cue_texts.get((segment_index, cue_index), ""),
                }
                for cue_index, cue in enumerate(current.sync_cues)
            ]
            projected_text = "".join(str(cue.get("text") or "") for cue in updated_cues)
        else:
            updated_cues = current.sync_cues
            projected_text = cue_texts.get((segment_index, None), "")
        projected[segment_index] = replace(
            current,
            text=projected_text,
            original_text=(current.original_text or current.text) if projected_text != current.text else current.original_text,
            sync_cues=updated_cues,
        )

    projected_normalized = normalized_text("".join(segment.text or "" for segment in projected))
    if projected_normalized != source_normalized:
        raise ValueError("projected_text_does_not_match_source")
    if timeline_fingerprint(projected) != before_fingerprint:
        raise ValueError("timeline_fingerprint_changed")
    if not _frozen_metadata_matches(primary, projected):
        raise ValueError("frozen_timeline_or_speaker_changed")
    return projected


def _build_review_windows(
    primary_segments: list[Segment],
    *,
    max_seconds: float = _MAX_WINDOW_SECONDS,
) -> list[_ReviewWindow]:
    windows: list[_ReviewWindow] = []
    indexes: list[int] = []
    window_start = 0.0
    window_end = 0.0
    for index, segment in enumerate(primary_segments):
        segment_start = float(segment.start)
        segment_end = max(float(segment.end), segment_start)
        if indexes and segment_end - window_start > max_seconds:
            windows.append(_ReviewWindow(window_start, window_end, tuple(indexes)))
            indexes = []
        if not indexes:
            window_start = segment_start
            window_end = segment_end
        else:
            window_end = max(window_end, segment_end)
        indexes.append(index)
    if indexes:
        windows.append(_ReviewWindow(window_start, window_end, tuple(indexes)))
    return windows


def _window_changed(before: list[Segment], after: list[Segment]) -> bool:
    if [segment.text for segment in before] != [segment.text for segment in after]:
        return True
    return [
        [str(cue.get("text") or "") for cue in segment.sync_cues or []]
        for segment in before
    ] != [
        [str(cue.get("text") or "") for cue in segment.sync_cues or []]
        for segment in after
    ]


def _apply_bounded_paraformer_consensus(
    primary_segments: list[Segment],
    projected_segments: list[Segment],
    reference_segments: list[Segment],
) -> tuple[list[Segment], list[dict[str, Any]]]:
    """Keep only Qwen lexical changes independently confirmed by Paraformer."""
    output = list(primary_segments)
    accepted_changes: list[dict[str, Any]] = []
    for segment_index, (primary, projected) in enumerate(
        zip(primary_segments, projected_segments)
    ):
        if not _window_changed([primary], [projected]):
            continue
        if text_similarity(primary.text or "", projected.text or "") < _PARAFORMER_SEGMENT_MIN_SIMILARITY:
            continue
        detector_text = _text_in_interval(
            reference_segments,
            max(0.0, float(primary.start) - _PARAFORMER_CONSENSUS_CONTEXT_SECONDS),
            float(primary.end) + _PARAFORMER_CONSENSUS_CONTEXT_SECONDS,
        )
        corrected_text, changes = consensus_rewrite(
            primary.text or "",
            projected.text or "",
            projected.text or "",
            detector_text=detector_text,
        )
        if not changes or corrected_text == primary.text:
            continue
        output[segment_index] = replace(
            primary,
            text=corrected_text,
            original_text=primary.original_text or primary.text,
            sync_cues=_replace_cue_text_preserving_times(
                primary.sync_cues,
                corrected_text,
            ),
        )
        accepted_changes.append(
            {
                "segment_index": segment_index,
                "start": round(float(primary.start), 3),
                "end": round(float(primary.end), 3),
                "changes": changes,
            }
        )
    return output, accepted_changes


def _text_in_interval(segments: Iterable[Segment], start: float, end: float) -> str:
    pieces: list[str] = []
    for segment in segments:
        text = segment.text or ""
        if not text:
            continue
        segment_start = float(segment.start)
        segment_end = max(float(segment.end), segment_start)
        overlap = _overlap_seconds(start, end, segment_start, segment_end)
        if overlap <= 0:
            continue
        duration = segment_end - segment_start
        if duration <= 1e-6 or overlap >= duration - 1e-6:
            pieces.append(text)
            continue
        left_ratio = max(0.0, min(1.0, (start - segment_start) / duration))
        right_ratio = max(0.0, min(1.0, (end - segment_start) / duration))
        left = max(0, min(len(text), int(len(text) * left_ratio)))
        right = max(left + 1, min(len(text), int(len(text) * right_ratio + 0.999999)))
        pieces.append(text[left:right])
    return "".join(pieces)


def _reference_selection_diagnostic(
    window: _ReviewWindow,
    anchors: list[Segment],
    reference_segments: list[Segment],
) -> dict[str, Any]:
    anchor_normalized = normalized_text("".join(segment.text or "" for segment in anchors))
    anchor_chars = len(anchor_normalized)
    candidates: list[dict[str, Any]] = []
    for shift_seconds in _REFERENCE_BOUNDARY_SHIFTS_SECONDS:
        reference_normalized = normalized_text(
            _text_in_interval(
                reference_segments,
                max(0.0, window.start + shift_seconds),
                max(0.0, window.end + shift_seconds),
            )
        )
        reference_chars = len(reference_normalized)
        equal_chars = sum(
            left_end - left_start
            for operation, left_start, left_end, _right_start, _right_end in SequenceMatcher(
                None,
                anchor_normalized,
                reference_normalized,
                autojunk=False,
            ).get_opcodes()
            if operation == "equal"
        )
        total_chars = anchor_chars + reference_chars
        similarity = (2.0 * equal_chars / total_chars) if total_chars else 1.0
        maximum_chars = max(anchor_chars, reference_chars)
        length_balance = (
            min(anchor_chars, reference_chars) / maximum_chars if maximum_chars else 1.0
        )
        mismatch_chars = max(anchor_chars - equal_chars, reference_chars - equal_chars)
        candidates.append({
            "reference_chars": reference_chars,
            "reference_similarity": similarity,
            "reference_mismatch_chars": mismatch_chars,
            "reference_length_balance": length_balance,
            "reference_shift_seconds": shift_seconds,
        })
    best = max(
        candidates,
        key=lambda item: (
            item["reference_similarity"],
            item["reference_length_balance"],
            -abs(item["reference_shift_seconds"]),
        ),
    )
    reference_chars = int(best["reference_chars"])
    similarity = float(best["reference_similarity"])
    mismatch_chars = int(best["reference_mismatch_chars"])
    length_balance = float(best["reference_length_balance"])
    maximum_chars = max(anchor_chars, reference_chars)

    diagnostic: dict[str, Any] = {
        "reference_anchor_chars": anchor_chars,
        "reference_chars": reference_chars,
        "reference_similarity": round(similarity, 4),
        "reference_mismatch_chars": mismatch_chars,
        "reference_length_balance": round(length_balance, 4),
        "reference_shift_seconds": float(best["reference_shift_seconds"]),
    }
    if anchor_chars >= 4 and not reference_chars:
        selection_reason = "reference_text_missing"
        selected = True
    elif anchor_chars >= 12 and length_balance < _REFERENCE_PARTIAL_LENGTH_BALANCE:
        selection_reason = "reference_text_partial"
        selected = True
    elif (
        maximum_chars >= 12
        and similarity < _REFERENCE_DISAGREEMENT_RATIO
        and mismatch_chars >= _REFERENCE_MIN_MISMATCH_CHARS
    ):
        selection_reason = "reference_text_disagreement"
        selected = True
    elif (
        4 <= maximum_chars < 12
        and similarity < _REFERENCE_SHORT_DISAGREEMENT_RATIO
        and mismatch_chars >= 4
    ):
        selection_reason = "reference_short_strong_disagreement"
        selected = True
    else:
        selection_reason = (
            "insufficient_primary_text" if anchor_chars < 4 else "reference_text_agreement"
        )
        selected = False
    diagnostic.update({
        "selected": selected,
        "selection_reason": selection_reason,
        "reason": "pending_review" if selected else selection_reason,
    })
    return diagnostic


def run_qwen_lexical_review(
    audio: Path | str,
    primary_segments: Iterable[Segment],
    *,
    language: str | None = "zh",
    on_progress: ProgressCallback | None = None,
    qwen_model: str = QWEN_MODEL,
    review_reference_segments: Iterable[Segment] | None = None,
    review_reference_source: str = "",
    review_reference_is_paraformer: bool = False,
) -> tuple[list[Segment], dict[str, Any]]:
    """Review primary text with one Qwen instance and a frozen primary timeline."""
    primary = list(primary_segments)
    output = list(primary)
    windows = _build_review_windows(primary)
    reference_segments = list(review_reference_segments or [])
    before_fingerprint = timeline_fingerprint(primary)
    stats: dict[str, Any] = {
        "mode": "qwen_lexical_frozen_timeline",
        "enabled": False,
        "applied": False,
        "reason": "",
        "qwen_model": qwen_model,
        "window_count": len(windows),
        "candidate_window_count": 0,
        "selected_window_count": 0,
        "skipped_window_count": 0,
        "qwen_invoked_window_count": 0,
        "selected_by_reason": {},
        "max_window_seconds": _MAX_WINDOW_SECONDS,
        "accepted": 0,
        "rejected": 0,
        "changed": 0,
        "timeline_fingerprint_before": before_fingerprint,
        "timeline_fingerprint_after": before_fingerprint,
        "timeline_preserved": True,
        "window_diagnostics": [],
        "alignment_min_equal_ratio": _MIN_ALIGNMENT_RATIO,
        "length_ratio_bounds": [_MIN_LENGTH_RATIO, _MAX_LENGTH_RATIO],
        "max_estimated_timing_ratio": _MAX_ESTIMATED_TIMING_RATIO,
        "reference_available": bool(reference_segments),
        "reference_source": str(review_reference_source or ""),
        "reference_is_paraformer": bool(review_reference_is_paraformer),
        "reference_disagreement_ratio": _REFERENCE_DISAGREEMENT_RATIO,
        "application_mode": (
            "paraformer_qwen_bounded_consensus"
            if review_reference_is_paraformer
            else "qwen_frozen_window_projection"
        ),
        "bounded_consensus_segment_count": 0,
        "bounded_consensus_change_count": 0,
    }
    if not primary:
        stats["reason"] = "empty_primary_segments"
        return output, stats

    window_entries: list[tuple[_ReviewWindow, dict[str, Any]]] = []
    for window_index, window in enumerate(windows):
        diagnostic: dict[str, Any] = {
            "window": window_index,
            "start": round(window.start, 3),
            "end": round(window.end, 3),
            "segment_indexes": list(window.segment_indexes),
            "accepted": False,
            "changed": False,
            "selected": True,
            "reason": "pending_review",
        }
        if reference_segments:
            anchors = [primary[index] for index in window.segment_indexes]
            diagnostic.update(
                _reference_selection_diagnostic(window, anchors, reference_segments)
            )
        window_entries.append((window, diagnostic))

    candidates = [entry for entry in window_entries if bool(entry[1].get("selected"))]
    stats["candidate_window_count"] = len(candidates)
    stats["selected_window_count"] = len(candidates)
    stats["skipped_window_count"] = len(windows) - len(candidates)
    selected_by_reason: dict[str, int] = {}
    for _window, diagnostic in candidates:
        reason = str(diagnostic.get("selection_reason") or "anchor_snapshot_missing")
        selected_by_reason[reason] = selected_by_reason.get(reason, 0) + 1
    stats["selected_by_reason"] = selected_by_reason
    stats["window_diagnostics"] = [diagnostic for _window, diagnostic in window_entries]
    if not candidates:
        stats["reason"] = "no_suspect_windows"
        return output, stats

    audio_path = Path(audio).expanduser().resolve()
    if not audio_path.exists():
        stats["reason"] = "audio_not_found"
        stats["rejected"] = len(candidates)
        for _window, diagnostic in candidates:
            diagnostic["reason"] = "audio_not_found"
        return output, stats
    if not _qwen_model_cached(qwen_model):
        stats["reason"] = "qwen_model_not_cached"
        stats["rejected"] = len(candidates)
        for _window, diagnostic in candidates:
            diagnostic["reason"] = "qwen_model_not_cached"
        return output, stats

    stats["enabled"] = True
    qwen = Qwen3ASRTranscriber()
    with tempfile.TemporaryDirectory(prefix="localscribe-qwen-lexical-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for candidate_index, (window, diagnostic) in enumerate(candidates):
            window_index = int(diagnostic["window"])
            if on_progress:
                on_progress(
                    {
                        "stage": "qwen_lexical_review",
                        "current": candidate_index + 1,
                        "total": len(candidates),
                    }
                )
            try:
                stats["qwen_invoked_window_count"] += 1
                clip = tmp_root / f"window_{window_index:04d}.wav"
                _extract_clip(audio_path, window.start, window.end, clip)
                result = qwen.transcribe(
                    clip,
                    TranscribeOptions(
                        language=language or "zh",
                        model_id=qwen_model,
                        audio_preprocess="off",
                    ),
                )
                qwen_stats = dict(result.filter_stats or {})
                source_segments = list(result.segments or [])
                anchors = [primary[index] for index in window.segment_indexes]
                source_chars = len(normalized_text("".join(segment.text or "" for segment in source_segments)))
                anchor_chars = len(normalized_text("".join(segment.text or "" for segment in anchors)))
                length_ratio = source_chars / max(anchor_chars, 1)
                diagnostic.update(
                    {
                        "source_chars": source_chars,
                        "anchor_chars": anchor_chars,
                        "length_ratio": round(length_ratio, 4),
                        "hallucination_risk": bool(qwen_stats.get("has_hallucination_risk")),
                    }
                )
                if not source_chars:
                    raise ValueError("empty_qwen_text")
                if diagnostic["hallucination_risk"]:
                    raise ValueError("qwen_hallucination_risk")
                if not (_MIN_LENGTH_RATIO <= length_ratio <= _MAX_LENGTH_RATIO):
                    raise ValueError("length_ratio_out_of_range")

                aligned, align_stats = _align_segments_to_timing_anchor(
                    source_segments,
                    anchors,
                    min_equal_ratio=_MIN_ALIGNMENT_RATIO,
                )
                diagnostic["alignment"] = align_stats
                if not aligned or not bool(align_stats.get("timing_alignment_ok")):
                    raise ValueError("alignment_rejected")
                aligned_source_chars = int(align_stats.get("source_chars") or source_chars)
                estimated_timing_chars = int(align_stats.get("estimated_timing_chars") or 0)
                estimated_timing_ratio = estimated_timing_chars / max(aligned_source_chars, 1)
                diagnostic["estimated_timing_ratio"] = round(estimated_timing_ratio, 4)
                if estimated_timing_ratio > _MAX_ESTIMATED_TIMING_RATIO:
                    raise ValueError("estimated_timing_ratio_too_high")

                projected = project_aligned_text_to_frozen_timeline(anchors, aligned)
                diagnostic["projection_changed"] = _window_changed(anchors, projected)
                if review_reference_is_paraformer:
                    projected, consensus_segments = _apply_bounded_paraformer_consensus(
                        anchors,
                        projected,
                        reference_segments,
                    )
                    consensus_change_count = sum(
                        len(item.get("changes") or []) for item in consensus_segments
                    )
                    diagnostic["bounded_consensus_segments"] = consensus_segments
                    diagnostic["bounded_consensus_segment_count"] = len(consensus_segments)
                    diagnostic["bounded_consensus_change_count"] = consensus_change_count
                    stats["bounded_consensus_segment_count"] += len(consensus_segments)
                    stats["bounded_consensus_change_count"] += consensus_change_count
                changed = _window_changed(anchors, projected)
                for local_index, segment_index in enumerate(window.segment_indexes):
                    output[segment_index] = projected[local_index]
                diagnostic.update({"accepted": True, "changed": changed, "reason": "accepted"})
                stats["accepted"] += 1
                if changed:
                    stats["changed"] += 1
            except Exception as exc:  # Each failed window keeps the primary text.
                diagnostic["reason"] = str(exc) or type(exc).__name__
                diagnostic["error_type"] = type(exc).__name__
                stats["rejected"] += 1

    after_fingerprint = timeline_fingerprint(output)
    stats["timeline_fingerprint_after"] = after_fingerprint
    stats["timeline_preserved"] = after_fingerprint == before_fingerprint and _frozen_metadata_matches(primary, output)
    if not stats["timeline_preserved"]:
        output = primary
        stats.update(
            {
                "applied": False,
                "reason": "timeline_guard_rejected_all_changes",
                "accepted": 0,
                "rejected": len(candidates),
                "changed": 0,
                "timeline_fingerprint_after": before_fingerprint,
                "timeline_preserved": True,
            }
        )
        return output, stats

    stats["applied"] = stats["changed"] > 0
    stats["reason"] = "review_completed" if stats["accepted"] else "all_windows_rejected"
    return output, stats
