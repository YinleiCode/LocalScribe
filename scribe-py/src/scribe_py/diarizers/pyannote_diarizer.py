"""Optional pyannote.audio speaker diarization baseline.

This module is deliberately optional.  LocalScribe's default app path remains
Senko/CAM++; pyannote is used as an industry baseline or high-accuracy mode
when the user has installed pyannote.audio and accepted/downloaded the model.
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class DiarizedSegment:
    start: float
    end: float
    text: str
    speaker: str
    speaker_confidence: float | None = None
    speaker_votes: dict[str, float] | None = None
    speaker_subsegments: list[dict] | None = None
    speaker_change_points: list[float] | None = None
    speaker_overlap_risk: bool | None = None
    speaker_overlap_ratio: float | None = None


@dataclass
class DiarizationResult:
    segments: list[DiarizedSegment]
    speakers: list[str]
    cluster_count: int
    matched_profiles: dict[str, str]
    stats: dict


def _load_pipeline():
    try:
        from pyannote.audio import Pipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed. Install it only for baseline/"
            "high-accuracy diarization, e.g. `pip install pyannote.audio`, "
            "and set HF_TOKEN if the selected model requires Hugging Face access."
        ) from exc
    return Pipeline


def _friendly_label(raw_label: str, label_map: dict[str, str]) -> str:
    if raw_label not in label_map:
        label_map[raw_label] = f"SPEAKER_{chr(ord('A') + len(label_map))}"
    return label_map[raw_label]


def _turns_from_annotation(annotation) -> list[dict]:
    if annotation is None:
        return []
    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })
    turns.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    return turns


def _segment_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _timeline_for_segment(turns: list[dict], start: float, end: float, label_map: dict[str, str]) -> list[dict]:
    timeline = []
    for turn in turns:
        overlap = _segment_overlap(start, end, float(turn["start"]), float(turn["end"]))
        if overlap <= 0:
            continue
        speaker = _friendly_label(str(turn["speaker"]), label_map)
        item = {
            "start": round(max(start, float(turn["start"])), 3),
            "end": round(min(end, float(turn["end"])), 3),
            "speaker": speaker,
            "duration": round(overlap, 3),
        }
        if timeline and timeline[-1]["speaker"] == speaker and abs(float(timeline[-1]["end"]) - item["start"]) <= 0.05:
            timeline[-1]["end"] = item["end"]
            timeline[-1]["duration"] = round(float(timeline[-1]["duration"]) + overlap, 3)
        else:
            timeline.append(item)
    return timeline


def _timeline_overlap_duration(timeline: list[dict]) -> float:
    """Measure time covered by at least two distinct speakers."""
    boundaries = sorted({
        float(item[key])
        for item in timeline
        for key in ("start", "end")
    })
    duration = 0.0
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        active = {
            str(item.get("speaker") or "")
            for item in timeline
            if float(item.get("start", 0.0)) <= midpoint < float(item.get("end", 0.0))
        }
        active.discard("")
        if len(active) >= 2:
            duration += right - left
    return duration


def diarize(
    audio: Path,
    segments: Sequence[dict],
    n_speakers: int = 0,
    profiles: Iterable[dict] | None = None,
    on_progress=None,
) -> DiarizationResult:
    del profiles
    Pipeline = _load_pipeline()
    model_id = os.environ.get(
        "LOCALSCRIBE_PYANNOTE_MODEL",
        "pyannote/speaker-diarization-community-1",
    )
    if on_progress:
        on_progress({"stage": "diarize_init", "engine": "pyannote", "model": model_id})

    try:
        load_kwargs = {}
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if token:
            load_kwargs["token"] = token
        pipeline = Pipeline.from_pretrained(model_id, **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "failed to load pyannote baseline model. Make sure the model is "
            "available locally or HF_TOKEN has access to it: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    kwargs: dict[str, int] = {}
    if int(n_speakers or 0) > 0:
        kwargs["num_speakers"] = int(n_speakers)

    if on_progress:
        on_progress({"stage": "diarize_run", "engine": "pyannote", **kwargs})
    pipeline_output = pipeline(str(audio.expanduser()), **kwargs)
    # community-1 returns a DiarizeOutput object.  Older pyannote versions
    # returned Annotation directly, so keep both paths working.  The exclusive
    # timeline is designed for transcript attribution; the regular timeline is
    # retained to measure genuine simultaneous speech.
    annotation = getattr(pipeline_output, "speaker_diarization", pipeline_output)
    exclusive_annotation = getattr(pipeline_output, "exclusive_speaker_diarization", None)
    raw_turns = _turns_from_annotation(annotation)
    assignment_turns = _turns_from_annotation(exclusive_annotation) or raw_turns

    label_map: dict[str, str] = {}
    out: list[DiarizedSegment] = []
    seen: list[str] = []
    overlap_risk_count = 0
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        timeline = _timeline_for_segment(raw_turns, start, end, label_map)
        assignment_timeline = _timeline_for_segment(assignment_turns, start, end, label_map)
        votes: dict[str, float] = defaultdict(float)
        for item in assignment_timeline:
            votes[str(item["speaker"])] += float(item["duration"])
        if votes:
            speaker, best = max(votes.items(), key=lambda item: item[1])
            total = sum(votes.values())
            confidence = best / max(1e-9, total)
        else:
            speaker = "SPEAKER_A"
            confidence = None
            total = 0.0
        if speaker not in seen:
            seen.append(speaker)
        change_points = [
            float(item["start"])
            for idx, item in enumerate(assignment_timeline[1:], start=1)
            if item.get("speaker") != assignment_timeline[idx - 1].get("speaker")
        ]
        overlap_duration = _timeline_overlap_duration(timeline)
        segment_duration = max(0.0, end - start)
        overlap_ratio = overlap_duration / max(segment_duration, 1e-9)
        overlap_risk = overlap_duration >= 0.20 and overlap_ratio >= 0.03
        if overlap_risk:
            overlap_risk_count += 1
        out.append(DiarizedSegment(
            start=start,
            end=end,
            text=str(seg.get("text") or ""),
            speaker=speaker,
            speaker_confidence=round(float(confidence), 3) if confidence is not None else None,
            speaker_votes={k: round(v, 3) for k, v in votes.items()} or None,
            speaker_subsegments=timeline or None,
            speaker_change_points=[round(float(t), 3) for t in change_points] or None,
            speaker_overlap_risk=bool(overlap_risk) or None,
            speaker_overlap_ratio=round(float(overlap_ratio), 4) if overlap_duration > 0 else None,
        ))

    return DiarizationResult(
        segments=out,
        speakers=seen,
        cluster_count=len(seen),
        matched_profiles={},
        stats={
            "engine": "pyannote",
            "model": model_id,
            "raw_turns": len(raw_turns),
            "exclusive_turns": len(assignment_turns),
            "segment_count": len(out),
            "speaker_overlap_risk_count": overlap_risk_count,
            "assignment": "pyannote_turn_overlap",
        },
    )
