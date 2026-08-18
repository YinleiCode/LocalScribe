#!/usr/bin/env python3
"""Score diarization predictions against sparse human-reviewed time ranges."""
from __future__ import annotations

import argparse
import difflib
import functools
import json
import re
from pathlib import Path
from typing import Any


TIME_RE = re.compile(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)")


def _seconds(value: str) -> float:
    match = TIME_RE.search(value.strip())
    if not match:
        raise ValueError(f"无法解析时间: {value}")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def _annotation_bounds(value: str) -> tuple[float, float]:
    parts = re.split(r"\s*[-–—]\s*", str(value).strip(), maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"无法解析标注时间段: {value}")
    return _seconds(parts[0]), _seconds(parts[1])


def _labels(value: Any) -> set[str]:
    labels = set()
    for part in re.split(r"[/,，、\s]+", str(value or "").strip()):
        token = part.strip().upper().removeprefix("SPEAKER_")
        if token:
            labels.add(token)
    return labels


def _normalized_speaker_label(value: Any) -> str:
    return str(value or "").strip().upper().removeprefix("SPEAKER_")


def _prediction_label(
    segment: dict[str, Any],
    start: float,
    end: float,
) -> tuple[str, str]:
    speaker_cues = segment.get("speaker_cues")
    if isinstance(speaker_cues, list):
        overlap_by_speaker: dict[str, float] = {}
        for cue in speaker_cues:
            if not isinstance(cue, dict):
                continue
            speaker = _normalized_speaker_label(cue.get("speaker"))
            try:
                cue_start = float(cue.get("start"))
                cue_end = float(cue.get("end"))
            except (TypeError, ValueError):
                continue
            overlap = max(0.0, min(end, cue_end) - max(start, cue_start))
            if speaker and overlap > 0:
                overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + overlap
        if overlap_by_speaker:
            speaker = max(
                overlap_by_speaker,
                key=lambda label: (overlap_by_speaker[label], label),
            )
            return speaker, "speaker_cue"
    return _normalized_speaker_label(segment.get("speaker")), "segment"


def _normalized_text(value: Any) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:'\"“”‘’()（）\[\]【】]+", "", str(value or "")).lower()


def _best_segment(
    segments: list[dict[str, Any]],
    start: float,
    end: float,
    annotation_text: str,
) -> tuple[int, dict[str, Any], str]:
    normalized_annotation = _normalized_text(annotation_text)
    if normalized_annotation:
        exact = [
            (idx, segment)
            for idx, segment in enumerate(segments)
            if _normalized_text(segment.get("text")) == normalized_annotation
        ]
        if len(exact) == 1:
            return exact[0][0], exact[0][1], "text_exact"
        candidates: list[tuple[float, int, dict[str, Any]]] = []
        for idx, segment in enumerate(segments):
            normalized_segment = _normalized_text(segment.get("text"))
            if not normalized_segment:
                continue
            similarity = difflib.SequenceMatcher(None, normalized_annotation, normalized_segment).ratio()
            if similarity >= 0.88:
                candidates.append((similarity, idx, segment))
        if candidates:
            similarity, idx, segment = max(candidates, key=lambda item: item[0])
            return idx, segment, f"text_fuzzy:{similarity:.3f}"

    best_idx = -1
    best: dict[str, Any] | None = None
    best_key = (-1.0, float("-inf"))
    midpoint = (start + end) / 2
    for idx, segment in enumerate(segments):
        seg_start = float(segment.get("start") or 0.0)
        seg_end = float(segment.get("end") or seg_start)
        overlap = max(0.0, min(end, seg_end) - max(start, seg_start))
        distance = abs(((seg_start + seg_end) / 2) - midpoint)
        key = (overlap, -distance)
        if key > best_key:
            best_idx, best, best_key = idx, segment, key
    if best is None or (best_key[0] <= 0 and -best_key[1] > 1.0):
        raise ValueError(f"预测结果找不到对应片段: {start:.3f}-{end:.3f}")
    return best_idx, best, "time_overlap"


def _maximum_agreement_mapping(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, str | None], dict[str, dict[str, int]], int, int]:
    """Return a deterministic maximum-weight one-to-one speaker mapping.

    Each reviewed row contributes one vote from its predicted label to every
    acceptable expected label. Only positive-evidence edges are considered,
    and an expected label can be assigned to at most one predicted label.
    """
    if not rows:
        return {}, {}, 0, 0

    predicted_labels = sorted({str(row["raw_predicted"]) for row in rows if row.get("raw_predicted")})
    expected_labels = sorted({label for row in rows for label in row.get("expected", [])})
    agreement = {
        predicted: {
            expected: sum(
                1
                for row in rows
                if row.get("raw_predicted") == predicted and expected in row.get("expected", [])
            )
            for expected in expected_labels
        }
        for predicted in predicted_labels
    }

    @functools.lru_cache(maxsize=None)
    def solve(index: int, used_expected_mask: int) -> tuple[int, int, tuple[str | None, ...]]:
        if index >= len(predicted_labels):
            return 0, 1, ()

        predicted = predicted_labels[index]
        options: list[tuple[str | None, int, int]] = [(None, used_expected_mask, 0)]
        for expected_index, expected in enumerate(expected_labels):
            expected_bit = 1 << expected_index
            weight = agreement[predicted][expected]
            if weight > 0 and not used_expected_mask & expected_bit:
                options.append((expected, used_expected_mask | expected_bit, weight))

        best_score = -1
        optimal_count = 0
        best_assignment: tuple[str | None, ...] | None = None
        expected_rank = {label: rank for rank, label in enumerate(expected_labels)}

        def assignment_key(assignment: tuple[str | None, ...]) -> tuple[int, ...]:
            unmapped_rank = len(expected_labels)
            return tuple(expected_rank.get(label, unmapped_rank) for label in assignment)

        for expected, next_mask, weight in options:
            suffix_score, suffix_count, suffix_assignment = solve(index + 1, next_mask)
            score = weight + suffix_score
            assignment = (expected, *suffix_assignment)
            if score > best_score:
                best_score = score
                optimal_count = suffix_count
                best_assignment = assignment
            elif score == best_score:
                optimal_count += suffix_count
                if best_assignment is None or assignment_key(assignment) < assignment_key(best_assignment):
                    best_assignment = assignment

        assert best_assignment is not None
        return best_score, optimal_count, best_assignment

    best_score, optimal_count, assignment = solve(0, 0)
    mapping = dict(zip(predicted_labels, assignment, strict=True))
    return mapping, agreement, best_score, optimal_count


def _mapping_status(
    rows: list[dict[str, Any]],
    mapping: dict[str, str | None],
    optimal_solution_count: int,
) -> tuple[str, bool, str]:
    expected_labels = sorted({label for row in rows for label in row.get("expected", [])})
    if not rows:
        return "no_reviewed_rows", False, "No reviewed rows were available; no mapping or accuracy was derived."
    if len(expected_labels) == 1:
        return (
            "underdetermined_single_expected_label",
            False,
            "Only one expected human label was reviewed; mapped accuracy cannot establish multi-speaker separation.",
        )
    if optimal_solution_count > 1:
        return (
            "underdetermined_multiple_optima",
            False,
            "Multiple one-to-one mappings have equal agreement; a deterministic lexical tie-break was used.",
        )
    unmapped = sorted(predicted for predicted, expected in mapping.items() if expected is None)
    if unmapped:
        return (
            "underdetermined_unmapped_predicted_labels",
            False,
            "Some reviewed predicted labels have no one-to-one expected-label assignment.",
        )
    return "determined", True, "The reviewed rows identify a unique maximum-agreement one-to-one mapping."


def score(prediction_path: Path, annotation_path: Path) -> dict[str, Any]:
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    segments = prediction.get("segments") if isinstance(prediction, dict) else prediction
    if not isinstance(segments, list) or not isinstance(annotations, list):
        raise ValueError("预测需包含 segments 数组，人工标注需为数组")

    rows = []
    for annotation in annotations:
        start, end = _annotation_bounds(str(annotation.get("时间") or annotation.get("time") or ""))
        expected = _labels(annotation.get("你的标注") or annotation.get("label") or annotation.get("speaker"))
        annotation_text = str(annotation.get("文本") or annotation.get("text") or "")
        index, segment, alignment = _best_segment(segments, start, end, annotation_text)
        predicted, prediction_resolution = _prediction_label(segment, start, end)
        rows.append({
            "annotation_index": annotation.get("序号"),
            "prediction_index": index,
            "start": start,
            "end": end,
            "expected": sorted(expected),
            "predicted": predicted,
            "raw_predicted": predicted,
            "alignment": alignment,
            "prediction_resolution": prediction_resolution,
            "text": str(segment.get("text") or ""),
        })

    mapping, agreement, mapping_objective, optimal_solution_count = _maximum_agreement_mapping(rows)
    raw_correct = 0
    mapped_correct = 0
    for row in rows:
        raw_predicted = row["raw_predicted"]
        mapped_predicted = mapping.get(raw_predicted)
        raw_ok = raw_predicted in row["expected"]
        mapped_ok = mapped_predicted is not None and mapped_predicted in row["expected"]
        raw_correct += int(raw_ok)
        mapped_correct += int(mapped_ok)
        row["mapped_predicted"] = mapped_predicted
        row["raw_correct"] = raw_ok
        row["mapped_correct"] = mapped_ok
        row["correct"] = mapped_ok

    mapping_status, mapping_is_determined, mapping_note = _mapping_status(
        rows, mapping, optimal_solution_count
    )
    raw_accuracy = round(raw_correct / len(rows), 4) if rows else None
    mapped_accuracy = round(mapped_correct / len(rows), 4) if rows else None
    return {
        "prediction": str(prediction_path),
        "annotations": str(annotation_path),
        "total": len(rows),
        "raw_correct": raw_correct,
        "raw_accuracy": raw_accuracy,
        "mapped_correct": mapped_correct,
        "mapped_accuracy": mapped_accuracy,
        "correct": mapped_correct,
        "accuracy": mapped_accuracy,
        "mapping_method": "maximum_agreement_one_to_one",
        "mapping_calibration_rows": len(rows),
        "mapping": mapping,
        "agreement_matrix": agreement,
        "mapping_objective_correct": mapping_objective,
        "mapping_optimal_solution_count": optimal_solution_count,
        "mapping_status": mapping_status,
        "mapping_is_determined": mapping_is_determined,
        "mapping_note": mapping_note,
        "observed_predicted_labels": sorted(mapping),
        "observed_expected_labels": sorted({label for row in rows for label in row["expected"]}),
        "unmapped_predicted_labels": sorted(
            predicted for predicted, expected in mapping.items() if expected is None
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="按稀疏人工标注时间段评估说话人标签")
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = score(args.prediction.expanduser().resolve(), args.annotations.expanduser().resolve())
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        out = args.out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
