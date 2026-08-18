#!/usr/bin/env python3
"""Fail closed when ASR text or cursor timing regresses from a baseline.

The gate is read-only. It compares transcript text and validates the cursor
timeline without importing the LocalScribe runtime, so it can run in CI or
against exported transcript JSON files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


_TIME_EPSILON_S = 1e-9


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalized_text(text: str) -> str:
    """Normalize compatibility characters and ignore layout-only whitespace."""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(character for character in normalized if not character.isspace())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _failure(code: str, message: str, location: str | None = None) -> dict[str, str]:
    failure = {"code": code, "message": message}
    if location:
        failure["location"] = location
    return failure


def _diagnostic_key(diagnostic: dict[str, str]) -> tuple[str, str]:
    return diagnostic["code"], diagnostic.get("location", "")


def _partition_candidate_diagnostics(
    baseline_diagnostics: list[dict[str, str]],
    candidate_diagnostics: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Separate inherited baseline defects from newly introduced defects."""
    remaining_baseline = Counter(_diagnostic_key(item) for item in baseline_diagnostics)
    inherited: list[dict[str, str]] = []
    regressions: list[dict[str, str]] = []
    for diagnostic in candidate_diagnostics:
        key = _diagnostic_key(diagnostic)
        if remaining_baseline[key] > 0:
            remaining_baseline[key] -= 1
            inherited.append(diagnostic)
        else:
            regressions.append(diagnostic)
    return inherited, regressions


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"file not found: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read transcript JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"transcript root must be an object: {path}")
    return data


def inspect_transcript(
    data: dict[str, Any],
    *,
    path: Path,
    short_cue_threshold_s: float,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    failures: list[dict[str, str]] = []
    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        failures.append(_failure("segments_invalid", "root.segments must be a non-empty list"))
        raw_segments = []

    text_parts: list[str] = []
    segment_geometry: list[list[Any]] = []
    cue_count = 0
    short_cue_count = 0
    zero_duration_cue_count = 0
    overlapping_cue_count = 0
    cue_text_mismatch_count = 0
    previous_segment_start = -math.inf
    previous_segment_end = -math.inf
    previous_cue_start = -math.inf
    previous_cue_end = -math.inf

    for segment_index, segment in enumerate(raw_segments):
        segment_location = f"segments[{segment_index}]"
        if not isinstance(segment, dict):
            failures.append(_failure("segment_invalid", "segment must be an object", segment_location))
            continue

        raw_text = segment.get("text")
        if not isinstance(raw_text, str):
            failures.append(_failure("segment_text_invalid", "segment.text must be a string", segment_location))
            segment_text = "" if raw_text is None else str(raw_text)
        else:
            segment_text = raw_text
        text_parts.append(segment_text)

        segment_start = _finite_number(segment.get("start"))
        segment_end = _finite_number(segment.get("end"))
        segment_geometry_row: list[Any] = [segment_start, segment_end, segment_text, []]
        segment_geometry.append(segment_geometry_row)
        if segment_start is None or segment_end is None:
            failures.append(
                _failure("segment_time_invalid", "segment start/end must be finite numbers", segment_location)
            )
        else:
            if segment_start < 0 or segment_end < 0:
                failures.append(_failure("segment_time_negative", "segment time cannot be negative", segment_location))
            if segment_end + _TIME_EPSILON_S < segment_start:
                failures.append(
                    _failure("segment_duration_negative", "segment end precedes start", segment_location)
                )
            if segment_start + _TIME_EPSILON_S < previous_segment_start:
                failures.append(
                    _failure("segment_start_non_monotonic", "segment start moved backwards", segment_location)
                )
            if segment_end + _TIME_EPSILON_S < previous_segment_end:
                failures.append(
                    _failure("segment_end_non_monotonic", "segment end moved backwards", segment_location)
                )
            previous_segment_start = segment_start
            previous_segment_end = segment_end

        raw_cues = segment.get("sync_cues")
        if not isinstance(raw_cues, list):
            failures.append(_failure("sync_cues_invalid", "segment.sync_cues must be a list", segment_location))
            raw_cues = []

        cue_text_parts: list[str] = []
        for cue_index, cue in enumerate(raw_cues):
            cue_location = f"{segment_location}.sync_cues[{cue_index}]"
            cue_count += 1
            if not isinstance(cue, dict):
                failures.append(_failure("cue_invalid", "cue must be an object", cue_location))
                continue

            cue_text = cue.get("text")
            if not isinstance(cue_text, str):
                failures.append(_failure("cue_text_invalid", "cue.text must be a string", cue_location))
                cue_text = "" if cue_text is None else str(cue_text)
            cue_text_parts.append(cue_text)

            cue_start = _finite_number(cue.get("start"))
            cue_end = _finite_number(cue.get("end"))
            segment_geometry_row[3].append([cue_start, cue_end, cue_text])
            if cue_start is None or cue_end is None:
                failures.append(_failure("cue_time_invalid", "cue start/end must be finite numbers", cue_location))
                continue
            if cue_start < 0 or cue_end < 0:
                failures.append(_failure("cue_time_negative", "cue time cannot be negative", cue_location))
            if cue_end + _TIME_EPSILON_S < cue_start:
                failures.append(_failure("cue_duration_negative", "cue end precedes start", cue_location))
            elif cue_end <= cue_start + _TIME_EPSILON_S:
                zero_duration_cue_count += 1
                failures.append(_failure("cue_zero_duration", "cue duration must be greater than zero", cue_location))
            elif cue_end - cue_start < short_cue_threshold_s:
                short_cue_count += 1

            if cue_start + _TIME_EPSILON_S < previous_cue_start:
                failures.append(_failure("cue_start_non_monotonic", "cue start moved backwards", cue_location))
            if cue_end + _TIME_EPSILON_S < previous_cue_end:
                failures.append(_failure("cue_end_non_monotonic", "cue end moved backwards", cue_location))
            if cue_index > 0 and cue_start + _TIME_EPSILON_S < previous_cue_end:
                overlapping_cue_count += 1
                failures.append(_failure("cue_overlap", "cue overlaps the previous cue", cue_location))
            previous_cue_start = cue_start
            previous_cue_end = cue_end

            if segment_start is not None and cue_start + _TIME_EPSILON_S < segment_start:
                failures.append(_failure("cue_before_segment", "cue starts before its segment", cue_location))
            if segment_end is not None and cue_end > segment_end + _TIME_EPSILON_S:
                failures.append(_failure("cue_after_segment", "cue ends after its segment", cue_location))

        reconstructed_text = "".join(cue_text_parts)
        if _normalized_text(reconstructed_text) != _normalized_text(segment_text):
            cue_text_mismatch_count += 1
            failures.append(
                _failure(
                    "cue_text_mismatch",
                    "sync cue text does not reconstruct segment text",
                    segment_location,
                )
            )

    transcript_text = "".join(text_parts)
    normalized_transcript_text = _normalized_text(transcript_text)
    geometry_payload = json.dumps(segment_geometry, ensure_ascii=False, separators=(",", ":"))
    short_cue_ratio = short_cue_count / cue_count if cue_count else 0.0
    summary: dict[str, Any] = {
        "path": str(path),
        "segment_count": len(raw_segments),
        "cue_count": cue_count,
        "text_chars": len(transcript_text),
        "text_sha256": _sha256(transcript_text),
        "normalized_text_chars": len(normalized_transcript_text),
        "normalized_text_sha256": _sha256(normalized_transcript_text),
        "segment_geometry_sha256": _sha256(geometry_payload),
        "cue_text_mismatches": cue_text_mismatch_count,
        "zero_duration_cues": zero_duration_cue_count,
        "overlapping_cues": overlapping_cue_count,
        "short_cues": short_cue_count,
        "short_cue_ratio": short_cue_ratio,
        "short_cue_threshold_ms": short_cue_threshold_s * 1000.0,
    }
    return summary, failures


def evaluate_gate(
    baseline_path: Path,
    candidate_path: Path,
    *,
    text_mode: str = "exact",
    short_cue_threshold_ms: float = 150.0,
    max_short_cues: int | None = None,
    max_short_cue_ratio: float | None = None,
    require_segment_geometry: bool = False,
) -> dict[str, Any]:
    if text_mode not in {"exact", "normalized"}:
        raise ValueError("text_mode must be 'exact' or 'normalized'")
    if short_cue_threshold_ms < 0:
        raise ValueError("short_cue_threshold_ms must be non-negative")
    if max_short_cues is not None and max_short_cues < 0:
        raise ValueError("max_short_cues must be non-negative")
    if max_short_cue_ratio is not None and not 0.0 <= max_short_cue_ratio <= 1.0:
        raise ValueError("max_short_cue_ratio must be between 0 and 1")

    baseline_path = baseline_path.expanduser().resolve()
    candidate_path = candidate_path.expanduser().resolve()
    threshold_s = short_cue_threshold_ms / 1000.0
    baseline, baseline_failures = inspect_transcript(
        _load_json(baseline_path), path=baseline_path, short_cue_threshold_s=threshold_s
    )
    candidate, candidate_failures = inspect_transcript(
        _load_json(candidate_path), path=candidate_path, short_cue_threshold_s=threshold_s
    )

    inherited_diagnostics, candidate_regressions = _partition_candidate_diagnostics(
        baseline_failures,
        candidate_failures,
    )
    failures: list[dict[str, str]] = []
    for failure in candidate_regressions:
        failures.append({**failure, "code": f"candidate_{failure['code']}"})

    hash_key = "text_sha256" if text_mode == "exact" else "normalized_text_sha256"
    if baseline[hash_key] != candidate[hash_key]:
        failures.append(
            _failure(
                "transcript_text_changed",
                f"candidate {text_mode} transcript text differs from baseline",
            )
        )
    if (
        require_segment_geometry
        and baseline["segment_geometry_sha256"] != candidate["segment_geometry_sha256"]
    ):
        failures.append(
            _failure(
                "segment_geometry_changed",
                "candidate segment or sync-cue text/timing/order differs from baseline",
            )
        )

    baseline_short_budget = baseline["short_cues"] + baseline["zero_duration_cues"]
    baseline_short_ratio_budget = baseline_short_budget / baseline["cue_count"] if baseline["cue_count"] else 0.0
    effective_max_short_cues = baseline_short_budget if max_short_cues is None else max_short_cues
    effective_max_short_ratio = (
        baseline_short_ratio_budget if max_short_cue_ratio is None else max_short_cue_ratio
    )
    if candidate["short_cues"] > effective_max_short_cues:
        failures.append(
            _failure(
                "short_cue_count_exceeded",
                f"candidate has {candidate['short_cues']} short cues; limit is {effective_max_short_cues}",
            )
        )
    if candidate["short_cue_ratio"] > effective_max_short_ratio + _TIME_EPSILON_S:
        failures.append(
            _failure(
                "short_cue_ratio_exceeded",
                f"candidate short-cue ratio is {candidate['short_cue_ratio']:.6f}; "
                f"limit is {effective_max_short_ratio:.6f}",
            )
        )

    return {
        "ok": not failures,
        "text_mode": text_mode,
        "compared_hash": hash_key,
        "require_segment_geometry": require_segment_geometry,
        "limits": {
            "short_cue_threshold_ms": short_cue_threshold_ms,
            "max_short_cues": effective_max_short_cues,
            "max_short_cue_ratio": effective_max_short_ratio,
            "short_cue_limits_source": {
                "count": "baseline" if max_short_cues is None else "explicit",
                "ratio": "baseline" if max_short_cue_ratio is None else "explicit",
            },
        },
        "baseline": baseline,
        "baseline_diagnostics": baseline_failures,
        "candidate": candidate,
        "candidate_diagnostics": candidate_failures,
        "inherited_diagnostics": inherited_diagnostics,
        "failures": failures,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify that candidate ASR text and cursor timing remain inside a frozen baseline gate."
    )
    parser.add_argument("baseline", type=Path, help="known-good transcript JSON")
    parser.add_argument("candidate", type=Path, help="candidate transcript JSON")
    parser.add_argument(
        "--text-mode",
        choices=("exact", "normalized"),
        default="exact",
        help="exact hashes raw concatenated segment text; normalized additionally applies NFKC and removes whitespace",
    )
    parser.add_argument(
        "--short-cue-threshold-ms",
        type=float,
        default=150.0,
        help="positive cues shorter than this are counted as very short (default: 150)",
    )
    parser.add_argument(
        "--max-short-cues",
        type=int,
        default=None,
        help="maximum very-short cue count (default: baseline count)",
    )
    parser.add_argument(
        "--max-short-cue-ratio",
        type=float,
        default=None,
        help="maximum very-short cue ratio in [0,1] (default: baseline ratio)",
    )
    parser.add_argument(
        "--require-segment-geometry",
        action="store_true",
        help="also require segment and sync-cue text/timing/order to match the baseline exactly",
    )
    parser.add_argument("--output", type=Path, help="optionally write the JSON result to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = evaluate_gate(
            args.baseline,
            args.candidate,
            text_mode=args.text_mode,
            short_cue_threshold_ms=args.short_cue_threshold_ms,
            max_short_cues=args.max_short_cues,
            max_short_cue_ratio=args.max_short_cue_ratio,
            require_segment_geometry=args.require_segment_geometry,
        )
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
