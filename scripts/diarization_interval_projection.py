#!/usr/bin/env python3
"""Project interval diarization output onto frozen LocalScribe speech cues.

This is an evaluation-only adapter. It never edits transcript text, segment
geometry, or sync cues; it only replaces speaker labels in a copied prediction
file so candidate diarization engines can be scored on the same ASR timeline.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _candidate_intervals(path: Path) -> list[tuple[float, float, str]]:
    data = _read_object(path)
    rows = data.get("text")
    if not isinstance(rows, list):
        raise ValueError("candidate JSON must contain a text interval array")
    result: list[tuple[float, float, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) < 3:
            raise ValueError(f"invalid candidate interval at index {index}")
        start, end = float(row[0]), float(row[1])
        if start < 0 or end <= start:
            raise ValueError(f"invalid candidate interval at index {index}: {row}")
        result.append((start, end, f"CANDIDATE_{row[2]}"))
    result.sort(key=lambda item: (item[0], item[1], item[2]))
    return result


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _nearest_speaker(
    start: float,
    end: float,
    intervals: list[tuple[float, float, str]],
) -> tuple[str, float]:
    midpoint = (start + end) / 2.0
    nearest = min(
        intervals,
        key=lambda item: (
            0.0
            if item[0] <= midpoint <= item[1]
            else min(abs(midpoint - item[0]), abs(midpoint - item[1])),
            item[0],
            item[1],
            item[2],
        ),
    )
    distance = (
        0.0
        if nearest[0] <= midpoint <= nearest[1]
        else min(abs(midpoint - nearest[0]), abs(midpoint - nearest[1]))
    )
    return nearest[2], distance


def project_speaker(
    start: float,
    end: float,
    intervals: list[tuple[float, float, str]],
) -> tuple[str, float, float]:
    overlap_by_speaker: Counter[str] = Counter()
    for interval_start, interval_end, speaker in intervals:
        if interval_start >= end:
            break
        if interval_end <= start:
            continue
        overlap_by_speaker[speaker] += _overlap(start, end, interval_start, interval_end)
    if overlap_by_speaker:
        speaker, overlap = max(
            overlap_by_speaker.items(),
            key=lambda item: (item[1], item[0]),
        )
        return speaker, overlap, 0.0
    speaker, distance = _nearest_speaker(start, end, intervals)
    return speaker, 0.0, distance


def project_prediction(
    base_prediction_path: Path,
    candidate_path: Path,
    uri: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prediction = _read_object(base_prediction_path)
    segments = prediction.get("segments")
    if not isinstance(segments, list):
        raise ValueError("base prediction must contain a segments array")
    intervals = _candidate_intervals(candidate_path)
    if not intervals:
        raise ValueError("candidate interval list is empty")

    projected_count = 0
    no_overlap_count = 0
    no_overlap_duration = 0.0
    max_nearest_distance = 0.0
    speaker_counts: Counter[str] = Counter()
    for row in segments:
        if not isinstance(row, dict) or str(row.get("uri") or "") != uri:
            continue
        start, end = float(row["start"]), float(row["end"])
        speaker, overlap, distance = project_speaker(start, end, intervals)
        row["speaker"] = speaker
        projected_count += 1
        speaker_counts[speaker] += 1
        if overlap <= 0.0:
            no_overlap_count += 1
            no_overlap_duration += end - start
            max_nearest_distance = max(max_nearest_distance, distance)

    if projected_count == 0:
        raise ValueError(f"base prediction contains no segments for URI {uri!r}")
    stats = {
        "uri": uri,
        "projected_cue_count": projected_count,
        "candidate_interval_count": len(intervals),
        "candidate_speaker_count": len({row[2] for row in intervals}),
        "projected_speaker_cue_counts": dict(sorted(speaker_counts.items())),
        "no_overlap_cue_count": no_overlap_count,
        "no_overlap_cue_duration_s": round(no_overlap_duration, 6),
        "max_nearest_interval_distance_s": round(max_nearest_distance, 6),
    }
    prediction["candidate_projection"] = stats
    return prediction, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将候选分人区间投影到冻结的 LocalScribe speech cues"
    )
    parser.add_argument("--base-prediction", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    prediction, stats = project_prediction(
        args.base_prediction.expanduser().resolve(),
        args.candidate.expanduser().resolve(),
        args.uri,
    )
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(out), **stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
