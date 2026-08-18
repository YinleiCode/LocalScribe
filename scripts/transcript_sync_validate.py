#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def _compact(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def validate_transcript(path: Path) -> tuple[dict[str, Any], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(segments, list):
        return {"file": str(path), "segments": 0}, ["root.segments must be a list"]

    errors: list[str] = []
    cue_count = 0
    segments_with_cues = 0
    cue_text_mismatches = 0
    zero_duration_cues = 0
    overlapping_cues = 0
    previous_segment_start = -math.inf
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            errors.append(f"segment {segment_index}: expected object")
            continue
        start = _number(segment.get("start"))
        end = _number(segment.get("end"))
        if start is None or end is None or end < start:
            errors.append(f"segment {segment_index}: invalid segment bounds")
            continue
        if start < previous_segment_start:
            errors.append(f"segment {segment_index}: start moved backwards")
        previous_segment_start = start

        cues = segment.get("sync_cues") or []
        if not isinstance(cues, list):
            errors.append(f"segment {segment_index}: sync_cues must be a list")
            continue
        if not cues:
            continue
        segments_with_cues += 1
        cue_count += len(cues)
        if _compact("".join(str(cue.get("text") or "") for cue in cues if isinstance(cue, dict))) != _compact(segment.get("text")):
            cue_text_mismatches += 1
            errors.append(f"segment {segment_index}: cue text does not reconstruct segment text")

        previous_cue_start = start
        previous_cue_end = start
        for cue_index, cue in enumerate(cues):
            if not isinstance(cue, dict):
                errors.append(f"segment {segment_index} cue {cue_index}: expected object")
                continue
            cue_start = _number(cue.get("start"))
            cue_end = _number(cue.get("end"))
            if cue_start is None or cue_end is None:
                errors.append(f"segment {segment_index} cue {cue_index}: invalid bounds")
                continue
            if cue_end <= cue_start:
                zero_duration_cues += 1
                errors.append(f"segment {segment_index} cue {cue_index}: zero duration")
            if cue_start < start - 0.001 or cue_end > end + 0.001 or cue_end < cue_start:
                errors.append(f"segment {segment_index} cue {cue_index}: outside segment bounds")
            if cue_start < previous_cue_start - 0.001 or cue_end < previous_cue_end - 0.001:
                errors.append(f"segment {segment_index} cue {cue_index}: cue time moved backwards")
            if cue_index > 0 and cue_start < previous_cue_end - 0.001:
                overlapping_cues += 1
                errors.append(f"segment {segment_index} cue {cue_index}: overlaps previous cue")
            previous_cue_start = cue_start
            previous_cue_end = cue_end

    summary = {
        "file": str(path),
        "segments": len(segments),
        "segments_with_cues": segments_with_cues,
        "segments_without_cues": len(segments) - segments_with_cues,
        "cue_count": cue_count,
        "cue_text_mismatches": cue_text_mismatches,
        "zero_duration_cues": zero_duration_cues,
        "overlapping_cues": overlapping_cues,
        "errors": len(errors),
        "status": "PASS" if not errors else "FAIL",
    }
    return summary, errors


def compare_geometry(current_path: Path, baseline_path: Path) -> list[str]:
    current = json.loads(current_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current_geometry = [
        (segment.get("start"), segment.get("end"), segment.get("text"))
        for segment in current.get("segments") or []
    ]
    baseline_geometry = [
        (segment.get("start"), segment.get("end"), segment.get("text"))
        for segment in baseline.get("segments") or []
    ]
    return [] if current_geometry == baseline_geometry else ["transcript geometry differs from baseline"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate LocalScribe transcript playback sync metadata")
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary, errors = validate_transcript(args.transcript)
    if args.baseline:
        errors.extend(compare_geometry(args.transcript, args.baseline))
        summary["baseline"] = str(args.baseline)
        summary["geometry_preserved"] = not any("geometry" in error for error in errors)
    summary["errors"] = len(errors)
    summary["status"] = "PASS" if not errors else "FAIL"

    if args.json:
        print(json.dumps({"summary": summary, "details": errors}, ensure_ascii=False, indent=2))
    else:
        print(f"{summary['status']}: {args.transcript}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
