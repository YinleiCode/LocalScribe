#!/usr/bin/env python3
"""Validate completed long-form ASR outputs without changing transcripts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_TXT_LINE_RE = re.compile(
    r"^\[(?P<start>\d{2}:\d{2}:\d{2}\.\d{3}) - (?P<end>\d{2}:\d{2}:\d{2}\.\d{3})\]\s?(?P<text>.*)$"
)
_SRT_TIME_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3}) --> (?P<end>\d{2}:\d{2}:\d{2},\d{3})$"
)


@dataclass
class ExportCheck:
    path: str
    present: bool
    entries: int
    text_matches: bool | None
    timing_valid: bool | None
    errors: list[str]


@dataclass
class CaseResult:
    label: str
    transcript: str
    ok: bool
    errors: list[str]
    warnings: list[str]
    metrics: dict[str, Any]
    exports: dict[str, ExportCheck]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_labeled_path(value: str, option: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"{option} must be LABEL=PATH, got: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise SystemExit(f"{option} label is empty: {value}")
    if not path.is_file():
        raise SystemExit(f"{option} file not found: {path}")
    return label, path


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalized_intervals(raw: Any) -> tuple[list[tuple[float, float]], list[str]]:
    errors: list[str] = []
    intervals: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        return [], ["coverage intervals must be a list"]
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append(f"coverage interval {index} must be an object")
            continue
        start = _finite_number(item.get("start"))
        end = _finite_number(item.get("end"))
        if start is None or end is None or start < 0 or end <= start:
            errors.append(f"coverage interval {index} is invalid")
            continue
        intervals.append((start, end))
    intervals.sort()
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 0.001:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged, errors


def _subtract_coverage_intervals(
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


def _compact_text(texts: list[str]) -> str:
    return re.sub(r"\s+", "", "".join(texts))


def _clock_seconds(value: str) -> float | None:
    normalized = value.replace(",", ".")
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})", normalized)
    if not match:
        return None
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    if minutes >= 60 or seconds >= 60:
        return None
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def _expected_texts(segments: list[dict[str, Any]]) -> list[str]:
    return [str(segment.get("text") or "").strip() for segment in segments if str(segment.get("text") or "").strip()]


def _strip_speaker_prefix(text: str) -> str:
    return re.sub(r"^\[(?:SPEAKER_[^\]]+|说话人[^\]]*)\]\s*", "", text.strip())


def _check_txt(path: Path, expected_texts: list[str]) -> ExportCheck:
    if not path.is_file():
        return ExportCheck(str(path), False, 0, None, None, [])
    errors: list[str] = []
    entries: list[tuple[float, float, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        match = _TXT_LINE_RE.match(line)
        if not match:
            errors.append(f"line {line_number}: invalid TXT row")
            continue
        start = _clock_seconds(match.group("start"))
        end = _clock_seconds(match.group("end"))
        if start is None or end is None or end < start:
            errors.append(f"line {line_number}: invalid TXT timestamp")
            continue
        entries.append((start, end, _strip_speaker_prefix(match.group("text"))))
    texts = [entry[2] for entry in entries]
    return ExportCheck(
        str(path),
        True,
        len(entries),
        texts == expected_texts,
        not any("timestamp" in error for error in errors),
        errors + ([] if texts == expected_texts else ["TXT text differs from transcript JSON"]),
    )


def _check_srt(path: Path, expected_texts: list[str]) -> ExportCheck:
    if not path.is_file():
        return ExportCheck(str(path), False, 0, None, None, [])
    errors: list[str] = []
    blocks = [block for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip()) if block.strip()]
    texts: list[str] = []
    previous_end = 0.0
    for block_index, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        if len(lines) < 3:
            errors.append(f"block {block_index}: incomplete SRT block")
            continue
        try:
            sequence = int(lines[0].strip())
        except ValueError:
            errors.append(f"block {block_index}: invalid sequence")
            continue
        if sequence != block_index:
            errors.append(f"block {block_index}: expected sequence {block_index}, got {sequence}")
        match = _SRT_TIME_RE.match(lines[1].strip())
        if not match:
            errors.append(f"block {block_index}: invalid SRT timestamp row")
            continue
        start = _clock_seconds(match.group("start"))
        end = _clock_seconds(match.group("end"))
        if start is None or end is None or end < start:
            errors.append(f"block {block_index}: invalid SRT timestamp")
            continue
        if start + 0.001 < previous_end:
            errors.append(f"block {block_index}: SRT timestamp is not monotonic")
        previous_end = max(previous_end, end)
        texts.append(_strip_speaker_prefix("\n".join(lines[2:])))
    return ExportCheck(
        str(path),
        True,
        len(blocks),
        texts == expected_texts,
        not any("timestamp" in error for error in errors),
        errors + ([] if texts == expected_texts else ["SRT text differs from transcript JSON"]),
    )


def _validate_segments(segments: list[Any], duration: float) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    valid_segments: list[dict[str, Any]] = []
    gaps: list[float] = []
    overlaps: list[float] = []
    previous_start = 0.0
    previous_end = 0.0
    cue_count = 0

    if not segments:
        return ["transcript has no segments"], [], {"segments": 0, "chars": 0}

    for index, raw_segment in enumerate(segments):
        if not isinstance(raw_segment, dict):
            errors.append(f"segment {index}: expected object")
            continue
        text = str(raw_segment.get("text") or "").strip()
        if not text:
            errors.append(f"segment {index}: empty text")
        start = _finite_number(raw_segment.get("start"))
        end = _finite_number(raw_segment.get("end"))
        if start is None or end is None:
            errors.append(f"segment {index}: timestamp is not finite")
            continue
        if start < 0 or end < 0:
            errors.append(f"segment {index}: negative timestamp")
        if end < start:
            errors.append(f"segment {index}: end precedes start")
        if index and start + 0.001 < previous_start:
            errors.append(f"segment {index}: start timestamp moved backwards")
        if index and end + 0.001 < previous_end:
            errors.append(f"segment {index}: end timestamp moved backwards")
        if index:
            delta = start - previous_end
            if delta >= 0:
                gaps.append(delta)
            else:
                overlaps.append(-delta)
                if -delta > 0.05:
                    errors.append(f"segment {index}: overlaps previous segment by {-delta:.3f}s")
        if duration > 0 and end > duration + 1.0:
            errors.append(f"segment {index}: end {end:.3f}s exceeds duration {duration:.3f}s")

        previous_cue_start = start
        previous_cue_end = start
        sync_cues = raw_segment.get("sync_cues") or []
        if not isinstance(sync_cues, list):
            errors.append(f"segment {index}: sync_cues must be a list")
            sync_cues = []
        for cue_index, cue in enumerate(sync_cues):
            cue_count += 1
            if not isinstance(cue, dict):
                errors.append(f"segment {index} cue {cue_index}: expected object")
                continue
            cue_text = str(cue.get("text") or "").strip()
            cue_start = _finite_number(cue.get("start"))
            cue_end = _finite_number(cue.get("end"))
            if not cue_text:
                errors.append(f"segment {index} cue {cue_index}: empty text")
            if cue_start is None or cue_end is None:
                errors.append(f"segment {index} cue {cue_index}: timestamp is not finite")
                continue
            if cue_start < start - 0.05 or cue_end > end + 0.05:
                errors.append(f"segment {index} cue {cue_index}: timestamp outside segment")
            if cue_end < cue_start:
                errors.append(f"segment {index} cue {cue_index}: end precedes start")
            if cue_start + 0.001 < previous_cue_start or cue_end + 0.001 < previous_cue_end:
                errors.append(f"segment {index} cue {cue_index}: timestamp moved backwards")
            if cue_index > 0 and cue_start + 0.001 < previous_cue_end:
                errors.append(f"segment {index} cue {cue_index}: overlaps previous cue")
            previous_cue_start = cue_start
            previous_cue_end = cue_end

        valid_segments.append(raw_segment)
        previous_start = start
        previous_end = end

    texts = _expected_texts(valid_segments)
    compact = _compact_text(texts)
    first_start = _finite_number(valid_segments[0].get("start")) if valid_segments else None
    last_end = _finite_number(valid_segments[-1].get("end")) if valid_segments else None
    if first_start is not None and first_start > 10:
        warnings.append(f"first segment starts at {first_start:.3f}s")
    if duration > 0 and last_end is not None and duration - last_end > 30:
        warnings.append(f"trailing uncovered audio is {duration - last_end:.3f}s")

    metrics = {
        "segments": len(valid_segments),
        "chars": len(compact),
        "md5": hashlib.md5(compact.encode("utf-8")).hexdigest(),
        "sha256": hashlib.sha256(compact.encode("utf-8")).hexdigest(),
        "sync_cues": cue_count,
        "first_start_s": first_start,
        "last_end_s": last_end,
        "leading_uncovered_s": first_start if first_start is not None else None,
        "trailing_uncovered_s": max(0.0, duration - last_end) if duration > 0 and last_end is not None else None,
        "max_gap_s": max(gaps, default=0.0),
        "overlap_count": len(overlaps),
        "max_overlap_s": max(overlaps, default=0.0),
    }
    return errors, warnings, metrics


def _validate_speech_coverage(
    raw_coverage: Any,
    *,
    required: bool,
    min_ratio: float,
    max_uncovered_s: float,
    max_edge_uncovered_s: float,
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    metrics = {
        "speech_coverage_status": "missing",
        "speech_coverage_reason": "missing",
        "speech_coverage_basis": "missing",
        "speech_duration_s": 0.0,
        "covered_speech_s": 0.0,
        "uncovered_speech_s": 0.0,
        "speech_coverage_ratio": None,
        "max_uncovered_speech_s": 0.0,
        "leading_uncovered_speech_s": 0.0,
        "trailing_uncovered_speech_s": 0.0,
        "wallclock_attempted_chunks": 0,
        "wallclock_recognized_chunks": 0,
        "wallclock_failed_chunks": 0,
        "wallclock_max_chunk_s": None,
        "speech_coverage_required": required,
        "speech_coverage_min_ratio": min_ratio,
        "speech_coverage_max_uncovered_s": max_uncovered_s,
        "speech_coverage_max_edge_uncovered_s": max_edge_uncovered_s,
    }
    if not isinstance(raw_coverage, dict):
        message = "speech coverage diagnostics are missing"
        (errors if required else warnings).append(message)
        return errors, warnings, metrics

    status = str(raw_coverage.get("status") or "missing")
    reason = str(raw_coverage.get("reason") or "")
    basis = str(raw_coverage.get("basis") or "missing")
    ratio = _finite_number(raw_coverage.get("speech_coverage_ratio"))
    speech_duration = _finite_number(raw_coverage.get("speech_duration_s")) or 0.0
    covered_duration = _finite_number(raw_coverage.get("covered_speech_s")) or 0.0
    uncovered_duration = _finite_number(raw_coverage.get("uncovered_speech_s")) or 0.0
    maximum = _finite_number(raw_coverage.get("max_uncovered_speech_s")) or 0.0
    leading = _finite_number(raw_coverage.get("leading_uncovered_speech_s")) or 0.0
    trailing = _finite_number(raw_coverage.get("trailing_uncovered_speech_s")) or 0.0
    attempted = int(raw_coverage.get("wallclock_attempted_chunks") or 0)
    recognized = int(raw_coverage.get("wallclock_recognized_chunks") or 0)
    failed = int(raw_coverage.get("wallclock_failed_chunks") or 0)
    wallclock_max_chunk_s = _finite_number(raw_coverage.get("wallclock_max_chunk_s"))
    metrics.update({
        "speech_coverage_status": status,
        "speech_coverage_reason": reason,
        "speech_coverage_basis": basis,
        "speech_duration_s": speech_duration,
        "covered_speech_s": covered_duration,
        "uncovered_speech_s": uncovered_duration,
        "speech_coverage_ratio": ratio,
        "max_uncovered_speech_s": maximum,
        "leading_uncovered_speech_s": leading,
        "trailing_uncovered_speech_s": trailing,
        "wallclock_attempted_chunks": attempted,
        "wallclock_recognized_chunks": recognized,
        "wallclock_failed_chunks": failed,
        "wallclock_max_chunk_s": wallclock_max_chunk_s,
    })

    if status == "no_speech":
        violations = []
        if required and basis != "silero_vad_no_speech":
            violations.append(f"no-speech coverage basis is invalid: {basis}")
        if speech_duration != 0 or covered_duration != 0 or uncovered_duration != 0:
            violations.append("no-speech coverage contains non-zero durations")
        (errors if required else warnings).extend(violations)
        return errors, warnings, metrics
    if status != "ok":
        message = f"speech coverage diagnostics unavailable: {status} ({reason or 'no reason'})"
        (errors if required else warnings).append(message)
        return errors, warnings, metrics

    violations: list[str] = []
    strict_recognized_ranges: list[list[float]] = []
    strict_schema2_active = False
    if required and basis != "wallclock_strict_windows":
        violations.append(f"speech coverage basis is not strict recognition evidence: {basis}")
    if ratio is None or not 0 <= ratio <= 1:
        violations.append("speech coverage ratio must be between 0 and 1")
    if min(speech_duration, covered_duration, uncovered_duration, maximum, leading, trailing) < 0:
        violations.append("speech coverage durations must be non-negative")
    if abs((covered_duration + uncovered_duration) - speech_duration) > 0.05:
        violations.append("covered plus uncovered speech does not equal speech duration")

    if basis in {"wallclock_recognized_chunks", "wallclock_strict_windows"}:
        if attempted <= 0:
            violations.append("wallclock coverage has no attempted chunks")
        if min(recognized, failed) < 0:
            violations.append("wallclock chunk counts must be non-negative")
        if recognized + failed != attempted:
            violations.append("wallclock recognized plus failed chunks does not equal attempted chunks")
    if basis == "wallclock_strict_windows":
        if wallclock_max_chunk_s is None or wallclock_max_chunk_s <= 0 or wallclock_max_chunk_s > 1.5:
            violations.append("strict coverage window must be present and no greater than 1.5s")
        schema_version = int(raw_coverage.get("coverage_schema_version") or 0)
        if schema_version >= 2:
            strict_schema2_active = True
            windows = raw_coverage.get("strict_probe_windows")
            if not isinstance(windows, list) or not windows:
                violations.append("strict coverage schema 2 windows are missing")
                windows = []
            if raw_coverage.get("strict_probe_windows_truncated") is not False:
                violations.append("strict coverage schema 2 windows must not be truncated")
            strict_core_max = _finite_number(raw_coverage.get("strict_core_max_chunk_s"))
            strict_pad = _finite_number(raw_coverage.get("strict_decode_context_pad_s"))
            if strict_core_max is None or strict_core_max <= 0 or strict_core_max > 1.5:
                violations.append("strict core maximum must be present and no greater than 1.5s")
                strict_core_max = 1.5
            if strict_pad is None or strict_pad < 0 or strict_pad > 1.0:
                violations.append("strict decode context padding must be between 0 and 1s")
                strict_pad = 0.0
            attempted_ranges: list[list[float]] = []
            failed_ranges: list[list[float]] = []
            for index, window in enumerate(windows):
                if not isinstance(window, dict):
                    violations.append(f"strict coverage window {index} must be an object")
                    continue
                core_start = _finite_number(window.get("core_start"))
                core_end = _finite_number(window.get("core_end"))
                decode_start = _finite_number(window.get("decode_start"))
                decode_end = _finite_number(window.get("decode_end"))
                if None in {core_start, core_end, decode_start, decode_end}:
                    violations.append(f"strict coverage window {index} boundaries are invalid")
                    continue
                assert core_start is not None and core_end is not None
                assert decode_start is not None and decode_end is not None
                if core_start < 0 or core_end <= core_start:
                    violations.append(f"strict coverage window {index} core is invalid")
                    continue
                if core_end - core_start > strict_core_max + 0.001:
                    violations.append(f"strict coverage window {index} exceeds 1.5s core limit")
                if decode_start > core_start + 0.001 or decode_end < core_end - 0.001:
                    violations.append(f"strict coverage window {index} decode range does not contain core")
                if core_start - decode_start > strict_pad + 0.001 or decode_end - core_end > strict_pad + 0.001:
                    violations.append(f"strict coverage window {index} exceeds declared decode padding")
                core = [round(core_start, 3), round(core_end, 3)]
                attempted_ranges.append(core)
                if window.get("status") == "recognized":
                    strict_recognized_ranges.append(core)
                elif window.get("status") == "failed":
                    failed_ranges.append(core)
                else:
                    violations.append(f"strict coverage window {index} status is invalid")
            if attempted_ranges != sorted(attempted_ranges) or _recovery_ranges_overlap(attempted_ranges):
                violations.append("strict coverage core windows overlap or are out of order")
            if len(attempted_ranges) != attempted:
                violations.append("strict coverage window count differs from attempted chunks")
            if len(strict_recognized_ranges) != recognized or len(failed_ranges) != failed:
                violations.append("strict coverage window statuses differ from chunk counts")
            partition = raw_coverage.get("strict_partition")
            if not isinstance(partition, dict):
                violations.append("strict coverage partition manifest is missing")
            else:
                expected_ranges = {
                    "attempted_ranges": attempted_ranges,
                    "recognized_ranges": strict_recognized_ranges,
                    "failed_ranges": failed_ranges,
                }
                for field, expected in expected_ranges.items():
                    actual = _recovery_canonical_ranges(
                        partition.get(field),
                        label=f"strict coverage {field}",
                        errors=violations,
                    )
                    if actual != expected:
                        violations.append(f"strict coverage partition {field} differs from windows")
                    hash_field = field.replace("_ranges", "_partition_sha256")
                    if partition.get(hash_field) != _recovery_ranges_sha256(expected):
                        violations.append(f"strict coverage partition {hash_field} is invalid")
                if int(partition.get("covered_count") or 0) != len(strict_recognized_ranges):
                    violations.append("strict coverage partition covered_count is invalid")
                if int(partition.get("failed_count") or 0) != len(failed_ranges):
                    violations.append("strict coverage partition failed_count is invalid")
                expected_valid = sorted(strict_recognized_ranges + failed_ranges) == attempted_ranges
                if bool(partition.get("partition_valid")) != expected_valid or not expected_valid:
                    violations.append("strict coverage partition is not closed")
        elif required:
            violations.append("strict coverage schema 2 evidence is required")

    speech_intervals, speech_interval_errors = _normalized_intervals(raw_coverage.get("speech_intervals"))
    covered_intervals, covered_interval_errors = _normalized_intervals(raw_coverage.get("covered_intervals"))
    violations.extend(speech_interval_errors)
    violations.extend(covered_interval_errors)
    if strict_schema2_active:
        normalized_strict_recognized, strict_range_errors = _normalized_intervals(
            [{"start": start, "end": end} for start, end in strict_recognized_ranges]
        )
        violations.extend(strict_range_errors)
        if normalized_strict_recognized != covered_intervals:
            violations.append("strict recognized core ranges differ from covered intervals")
    if not speech_interval_errors and speech_intervals:
        computed_speech = sum(end - start for start, end in speech_intervals)
        computed_uncovered_ranges = _subtract_coverage_intervals(speech_intervals, covered_intervals)
        computed_uncovered = sum(end - start for start, end in computed_uncovered_ranges)
        computed_covered = max(0.0, computed_speech - computed_uncovered)
        computed_ratio = computed_covered / computed_speech
        computed_max = max((end - start for start, end in computed_uncovered_ranges), default=0.0)
        computed_leading = (
            computed_uncovered_ranges[0][1] - computed_uncovered_ranges[0][0]
            if computed_uncovered_ranges
            and abs(computed_uncovered_ranges[0][0] - speech_intervals[0][0]) <= 0.001
            else 0.0
        )
        computed_trailing = (
            computed_uncovered_ranges[-1][1] - computed_uncovered_ranges[-1][0]
            if computed_uncovered_ranges
            and abs(computed_uncovered_ranges[-1][1] - speech_intervals[-1][1]) <= 0.001
            else 0.0
        )
        if abs(computed_speech - speech_duration) > 0.05:
            violations.append("reported speech duration differs from interval evidence")
        if abs(computed_uncovered - uncovered_duration) > 0.05:
            violations.append("reported uncovered speech differs from interval evidence")
        if ratio is not None and abs(computed_ratio - ratio) > 0.001:
            violations.append("reported speech coverage ratio differs from interval evidence")
        if abs(computed_max - maximum) > 0.05:
            violations.append("reported maximum uncovered speech differs from interval evidence")
        maximum = computed_max
        metrics["max_uncovered_speech_s"] = computed_max
        if abs(computed_leading - leading) > 0.05:
            violations.append("reported leading uncovered speech differs from interval evidence")
        if abs(computed_trailing - trailing) > 0.05:
            violations.append("reported trailing uncovered speech differs from interval evidence")
        leading = computed_leading
        trailing = computed_trailing
        metrics["leading_uncovered_speech_s"] = computed_leading
        metrics["trailing_uncovered_speech_s"] = computed_trailing
    elif required:
        violations.append("speech coverage interval evidence is empty")

    if ratio is not None and ratio < min_ratio:
        violations.append(f"speech coverage ratio {ratio:.4f} is below {min_ratio:.4f}")
    if maximum + 0.0005 >= max_uncovered_s:
        violations.append(f"maximum uncovered speech {maximum:.3f}s reaches {max_uncovered_s:.3f}s limit")
    if leading > max_edge_uncovered_s:
        violations.append(f"leading uncovered speech {leading:.3f}s exceeds {max_edge_uncovered_s:.3f}s")
    if trailing > max_edge_uncovered_s:
        violations.append(f"trailing uncovered speech {trailing:.3f}s exceeds {max_edge_uncovered_s:.3f}s")
    (errors if required else warnings).extend(violations)
    return errors, warnings, metrics


def _recovery_body(text: Any) -> str:
    body = unicodedata.normalize("NFKC", str(text or ""))
    body = re.sub(r"<\|[^<>]*?\|>", "", body)
    body = re.sub(r"</?[^<>]{1,40}>", "", body)
    body = re.sub(
        r"[\[【（(]\s*(?:music|speech|noise|laughter|laugh|applause|cough|breath|"
        r"音乐|语音|噪音|杂音|笑声|鼓掌|咳嗽|呼吸)\s*[\]】）)]",
        "",
        body,
        flags=re.IGNORECASE,
    )
    return body.strip()


def _recovery_normalized(text: Any) -> str:
    normalized = _recovery_body(text).lower()
    return "".join(ch for ch in normalized if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]", ch))


def _recovery_overlap(left: str, right: str, *, limit: int | None = None) -> int:
    upper = min(len(left), len(right), limit if limit is not None else max(len(left), len(right)))
    for size in range(upper, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def _recovery_local_reference_from_segments(
    segments: list[dict[str, Any]],
    start: float,
    end: float,
    *,
    context_s: float = 15.0,
    max_chars: int = 500,
) -> dict[str, str]:
    ordered = sorted(
        segments,
        key=lambda item: (float(item.get("start") or 0.0), float(item.get("end") or 0.0)),
    )
    left_parts: list[str] = []
    right_parts: list[str] = []
    overlapping_parts: list[str] = []
    for segment in ordered:
        seg_start = _finite_number(segment.get("start"))
        seg_end = _finite_number(segment.get("end"))
        body = _recovery_body(segment.get("text"))
        if seg_start is None or seg_end is None or not body:
            continue
        if seg_end <= start and start - seg_end <= context_s:
            left_parts.append(body)
        elif seg_start >= end and seg_start - end <= context_s:
            right_parts.append(body)
        elif seg_start < end and seg_end > start:
            overlapping_parts.append(body)
    left = "".join(left_parts)[-max_chars:]
    right = "".join(right_parts)[:max_chars]
    overlapping = "".join(overlapping_parts)[:max_chars]
    return {
        "left": left,
        "right": right,
        "overlapping": overlapping,
        "reference": overlapping,
    }


def _recovery_canonical_ranges(
    raw: Any,
    *,
    label: str,
    errors: list[str],
) -> list[list[float]]:
    if not isinstance(raw, list):
        errors.append(f"local recovery {label} ranges must be a list")
        return []
    ranges: list[list[float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            errors.append(f"local recovery {label} range {index} is invalid")
            continue
        start = _finite_number(item[0])
        end = _finite_number(item[1])
        if start is None or end is None or start < 0 or end <= start:
            errors.append(f"local recovery {label} range {index} is invalid")
            continue
        ranges.append([round(start, 3), round(end, 3)])
    return sorted(ranges)


def _recovery_ranges_sha256(ranges: list[list[float]]) -> str:
    return hashlib.sha256(
        json.dumps(ranges, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _recovery_ranges_overlap(ranges: list[list[float]]) -> bool:
    ordered = sorted(ranges)
    return any(
        ordered[index][0] < ordered[index - 1][1] - 0.001
        for index in range(1, len(ordered))
    )


def _recovery_deduplicated(raw: Any, left: Any, right: Any) -> tuple[str, str]:
    normalized = _recovery_normalized(raw)
    left_normalized = _recovery_normalized(left)
    right_normalized = _recovery_normalized(right)
    left_overlap = _recovery_overlap(left_normalized, normalized)
    right_overlap = _recovery_overlap(normalized, right_normalized, limit=max(0, len(normalized) - left_overlap))
    residual_end = len(normalized) - right_overlap
    residual = normalized[left_overlap:residual_end] if residual_end > left_overlap else ""
    return normalized, residual


def _recovery_repeated(normalized: str) -> bool:
    text = _recovery_normalized(normalized)
    if len(text) < 3:
        return False
    for unit_len in range(1, min(12, len(text) // 3) + 1):
        if len(text) % unit_len == 0 and text == text[:unit_len] * (len(text) // unit_len):
            return True
    return False


def _recovery_decision_from_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    def stable_primary(status: str, field: str) -> tuple[str, list[dict[str, Any]]]:
        grouped: dict[str, dict[str, dict[str, Any]]] = {}
        for item in attempts:
            if (
                item.get("verified_status") != status
                or item.get("verified_provider_kind") != "primary_asr"
                or bool(item.get("hallucination_risk"))
            ):
                continue
            value = str(item.get(field) or "")
            framing = str(item.get("framing") or "")
            if value and framing:
                grouped.setdefault(value, {})[framing] = item
        candidates: list[tuple[str, list[dict[str, Any]]]] = []
        for value, by_framing in grouped.items():
            items = list(by_framing.values())
            hashes = {str(item.get("slice_sha256") or "") for item in items}
            if len(items) >= 2 and "" not in hashes and len(hashes) >= 2:
                candidates.append((value, items))
        if not candidates:
            return "", []
        candidates.sort(key=lambda item: (-len(item[1]), item[0]))
        return candidates[0]

    def independent(status: str, field: str, consensus: str) -> list[dict[str, Any]]:
        primary_families = {
            str(item.get("verified_model_family") or "")
            for item in attempts
            if item.get("verified_provider_kind") == "primary_asr"
        }
        evidence: dict[str, dict[str, Any]] = {}
        for item in attempts:
            provider_id = str(item.get("verified_provider_id") or "")
            model_family = str(item.get("verified_model_family") or "")
            if (
                item.get("verified_status") == status
                and item.get("verified_provider_kind") == "independent_asr"
                and str(item.get(field) or "") == consensus
                and provider_id == "qwen3-independent"
                and model_family == "qwen3_asr"
                and str(item.get("verified_model_id") or "") == "mlx-community/Qwen3-ASR-1.7B-8bit"
                and bool(item.get("slice_sha256"))
                and bool(item.get("model_revision"))
                and bool(item.get("config_sha256"))
                and bool(item.get("weights_manifest_sha256"))
                and model_family not in primary_families
                and not bool(item.get("hallucination_risk"))
            ):
                evidence[provider_id] = item
        return list(evidence.values())

    def single_primary_existing() -> tuple[str, list[dict[str, Any]]]:
        for primary in attempts:
            if (
                primary.get("verified_status") != "matched_existing"
                or primary.get("verified_provider_kind") != "primary_asr"
                or bool(primary.get("hallucination_risk"))
            ):
                continue
            consensus = str(primary.get("verified_normalized") or "")
            if not consensus or not primary.get("slice_sha256"):
                continue
            independent_items = independent(
                "matched_existing",
                "verified_normalized",
                consensus,
            )
            if independent_items:
                return consensus, [primary, *independent_items]
        return "", []

    def evidence_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "evidence_framings": sorted({str(item.get("framing") or "") for item in items}),
            "evidence_providers": sorted({str(item.get("verified_provider_id") or "") for item in items}),
            "evidence_models": sorted({str(item.get("verified_model_id") or "") for item in items}),
            "evidence_ids": sorted(
                "|".join((
                    str(item.get("verified_provider_id") or ""),
                    str(item.get("framing") or ""),
                    str(item.get("slice_sha256") or ""),
                ))
                for item in items
            ),
        }

    primary_matched, matched_items = stable_primary("matched_existing", "verified_normalized")
    primary_residual, valid_items = stable_primary("valid", "verified_residual")
    primary_status = "matched_existing" if primary_matched else "valid" if primary_residual else ""
    primary_consensus = primary_matched or primary_residual
    primary_items = matched_items or valid_items
    base = {
        "primary_status": primary_status,
        "primary_consensus": primary_consensus,
        "primary_evidence_framings": sorted(str(item.get("framing") or "") for item in primary_items),
    }
    if not primary_matched:
        cross_model_matched, cross_model_items = single_primary_existing()
        if cross_model_matched:
            return {
                **base,
                "primary_status": "matched_existing",
                "primary_consensus": cross_model_matched,
                "primary_evidence_framings": sorted({
                    str(item.get("framing") or "")
                    for item in cross_model_items
                    if item.get("verified_provider_kind") == "primary_asr"
                }),
                "decision": "matched_existing",
                "consensus": cross_model_matched,
                **evidence_payload(cross_model_items),
            }
    if primary_matched:
        independent_items = independent("matched_existing", "verified_normalized", primary_matched)
        if independent_items:
            return {
                **base,
                "decision": "matched_existing",
                "consensus": primary_matched,
                **evidence_payload(matched_items + independent_items),
            }
    if primary_residual:
        independent_items = independent("valid", "verified_residual", primary_residual)
        if independent_items:
            return {
                **base,
                "decision": "insert_accepted",
                "consensus": primary_residual,
                "inserted_raw_text": str(valid_items[0].get("verified_residual_text") or ""),
                **evidence_payload(valid_items + independent_items),
            }
    return {
        **base,
        "decision": "error" if attempts and all(item.get("verified_status") == "error" for item in attempts) else "rejected",
        "consensus": "",
        "evidence_framings": [],
        "evidence_providers": [],
        "evidence_models": [],
        "evidence_ids": [],
        "inserted_raw_text": "",
    }


def _validate_local_recovery(
    raw_recovery: Any,
    *,
    segments: list[Any],
    raw_coverage: Any,
    audio_path: Path | None = None,
    preprocess_mode: str = "adaptive",
    audio_standardization: dict[str, Any] | None = None,
    audio_quality: dict[str, Any] | None = None,
    normalization_language: str | None = None,
    normalization_profile: str | None = None,
    min_chars_per_s: float = 0.75,
    enforce_audio_evidence: bool = False,
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {
        "local_recovery_mode": "missing",
        "local_recovery_pending_windows": 0,
        "local_recovery_pending_groups": 0,
        "local_recovery_attempts": 0,
        "local_recovery_matched_existing": 0,
        "local_recovery_inserted": 0,
        "local_recovery_rejected": 0,
        "local_recovery_error": 0,
    }
    if raw_recovery is None:
        return errors, warnings, metrics
    if not isinstance(raw_recovery, dict):
        return ["local recovery diagnostics must be an object"], warnings, metrics

    mode = str(raw_recovery.get("mode") or "")
    metrics["local_recovery_mode"] = mode
    if mode not in {"off", "audit", "merge"}:
        errors.append(f"local recovery mode is invalid: {mode or 'missing'}")
    provider = raw_recovery.get("provider")
    if provider is not None and not isinstance(provider, dict):
        errors.append("local recovery provider must be an object")
        provider = {}
    provider = provider if isinstance(provider, dict) else {}
    text_normalization = raw_recovery.get("text_normalization")
    if not isinstance(text_normalization, dict):
        text_normalization = {}
        if mode == "merge":
            errors.append("merge local recovery text normalization manifest is missing")
    if mode == "merge":
        expected_provider = {
            "requested": "qwen3",
            "provider_id": "qwen3-independent",
            "provider_kind": "independent_asr",
            "model_id": "mlx-community/Qwen3-ASR-1.7B-8bit",
            "model_family": "qwen3_asr",
        }
        for field, expected in expected_provider.items():
            if provider.get(field) != expected:
                errors.append(f"merge local recovery provider {field} is invalid")
        if provider.get("available") is not True or provider.get("error") not in (None, ""):
            errors.append("merge local recovery independent provider is unavailable")
        if (
            not provider.get("model_revision")
            or not provider.get("config_sha256")
            or not provider.get("weights_manifest_sha256")
        ):
            errors.append("merge local recovery provider identity is incomplete")
        if text_normalization.get("error"):
            errors.append("merge local recovery text normalization failed")
        if normalization_language is not None and text_normalization.get("language") != normalization_language:
            errors.append("merge local recovery normalization language differs from transcript")
        if text_normalization.get("profile") != normalization_profile:
            errors.append("merge local recovery normalization profile differs from transcript")

    count_names = ("pending_windows", "pending_groups", "attempts", "matched_existing", "inserted", "rejected", "error")
    counts: dict[str, int] = {}
    for name in count_names:
        value = raw_recovery.get(name, 0)
        if isinstance(value, bool):
            errors.append(f"local recovery {name} must be an integer")
            counts[name] = 0
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            errors.append(f"local recovery {name} must be an integer")
            number = 0
        if number < 0:
            errors.append(f"local recovery {name} must be non-negative")
        counts[name] = max(0, number)
        metrics[f"local_recovery_{name}"] = max(0, number)
    decision_total = counts["matched_existing"] + counts["inserted"] + counts["rejected"] + counts["error"]
    if mode in {"audit", "merge"} and decision_total != counts["pending_groups"]:
        errors.append("local recovery decision counts do not equal pending groups")
    if mode == "off":
        if counts["attempts"] != 0 or decision_total != 0:
            errors.append("off local recovery must not contain attempts or decisions")
        if raw_recovery.get("details") not in (None, []):
            errors.append("off local recovery must not contain details")

    before = raw_recovery.get("before")
    after = raw_recovery.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        errors.append("local recovery before/after snapshots are missing")
        before = before if isinstance(before, dict) else {}
        after = after if isinstance(after, dict) else {}

    def validate_snapshot(snapshot: dict[str, Any], label: str) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        required_fields = (
            "segment_count",
            "text_sha256",
            "covered_count",
            "failed_count",
            "attempted_ranges",
            "recognized_ranges",
            "failed_ranges",
            "attempted_partition_sha256",
            "recognized_partition_sha256",
            "failed_partition_sha256",
            "partition_valid",
        )
        for field in required_fields:
            if field not in snapshot:
                errors.append(f"local recovery {label} snapshot missing {field}")
        attempted_ranges = _recovery_canonical_ranges(
            snapshot.get("attempted_ranges"), label=f"{label} attempted", errors=errors
        )
        recognized_ranges = _recovery_canonical_ranges(
            snapshot.get("recognized_ranges"), label=f"{label} recognized", errors=errors
        )
        failed_ranges = _recovery_canonical_ranges(
            snapshot.get("failed_ranges"), label=f"{label} failed", errors=errors
        )
        if int(snapshot.get("covered_count") or 0) != len(recognized_ranges):
            errors.append(f"local recovery {label} covered count differs from ranges")
        if int(snapshot.get("failed_count") or 0) != len(failed_ranges):
            errors.append(f"local recovery {label} failed count differs from ranges")
        combined_ranges = sorted(recognized_ranges + failed_ranges)
        if attempted_ranges != combined_ranges:
            errors.append(f"local recovery {label} attempted partition is not closed")
        if _recovery_ranges_overlap(attempted_ranges):
            errors.append(f"local recovery {label} attempted ranges overlap")
        if _recovery_ranges_overlap(recognized_ranges):
            errors.append(f"local recovery {label} recognized ranges overlap")
        if _recovery_ranges_overlap(failed_ranges):
            errors.append(f"local recovery {label} failed ranges overlap")
        if _recovery_ranges_overlap(combined_ranges):
            errors.append(f"local recovery {label} recognized and failed ranges overlap")
        expected_hashes = {
            "attempted_partition_sha256": _recovery_ranges_sha256(attempted_ranges),
            "recognized_partition_sha256": _recovery_ranges_sha256(recognized_ranges),
            "failed_partition_sha256": _recovery_ranges_sha256(failed_ranges),
        }
        for field, expected in expected_hashes.items():
            if snapshot.get(field) != expected:
                errors.append(f"local recovery {label} {field} is invalid")
        computed_partition_valid = (
            attempted_ranges == combined_ranges
            and not _recovery_ranges_overlap(combined_ranges)
        )
        if snapshot.get("partition_valid") is not computed_partition_valid:
            errors.append(f"local recovery {label} partition_valid is invalid")
        return attempted_ranges, recognized_ranges, failed_ranges

    before_attempted, before_recognized, before_failed = validate_snapshot(before, "before")
    after_attempted, after_recognized, after_failed = validate_snapshot(after, "after")
    snapshot_fields = ("segment_count", "text_sha256", "covered_count", "failed_count")
    actual_segments = [dict(item) for item in segments if isinstance(item, dict)]
    actual_text_sha256 = hashlib.sha256(
        _compact_text(_expected_texts(actual_segments)).encode("utf-8")
    ).hexdigest()
    if mode in {"off", "audit"}:
        for field in snapshot_fields:
            if before.get(field) != after.get(field):
                errors.append(f"{mode} local recovery changed {field}")
        if before_attempted != after_attempted or before_recognized != after_recognized or before_failed != after_failed:
            errors.append(f"{mode} local recovery changed range partitions")
    if after.get("segment_count") != len(actual_segments):
        errors.append(f"{mode or 'local'} local recovery segment count differs from transcript")
    if after.get("text_sha256") != actual_text_sha256:
        errors.append(f"{mode or 'local'} local recovery text hash differs from transcript")

    coverage = raw_coverage if isinstance(raw_coverage, dict) else {}
    if after:
        if "wallclock_recognized_chunks" in coverage and after.get("covered_count") != int(coverage.get("wallclock_recognized_chunks") or 0):
            errors.append("local recovery after covered count differs from coverage")
        if "wallclock_failed_chunks" in coverage and after.get("failed_count") != int(coverage.get("wallclock_failed_chunks") or 0):
            errors.append("local recovery after failed count differs from coverage")


    details = raw_recovery.get("details")
    if not isinstance(details, list):
        errors.append("local recovery details must be a list")
        details = []
    base_segments = [dict(item) for item in actual_segments]
    if mode == "merge":
        for detail_index, detail in enumerate(details):
            if not isinstance(detail, dict) or detail.get("decision") != "insert_accepted":
                continue
            start = _finite_number(detail.get("start"))
            end = _finite_number(detail.get("end"))
            inserted_raw_text = str(detail.get("inserted_raw_text") or "")
            inserted_text = str(detail.get("inserted_text") or "")
            matching_indexes = [
                index
                for index, item in enumerate(base_segments)
                if start is not None
                and end is not None
                and abs(float(item.get("start") or 0.0) - start) <= 0.001
                and abs(float(item.get("end") or 0.0) - end) <= 0.001
                and str(item.get("text") or "") == inserted_text
                and str(item.get("original_text") or "") == inserted_raw_text
            ]
            if len(matching_indexes) != 1:
                errors.append(f"local recovery detail {detail_index} inserted segment is missing or duplicated")
            elif matching_indexes:
                base_segments.pop(matching_indexes[0])

    base_text_sha256 = hashlib.sha256(
        _compact_text(_expected_texts(base_segments)).encode("utf-8")
    ).hexdigest()
    if before.get("segment_count") != len(base_segments):
        errors.append("merge local recovery before segment count differs from reconstructed transcript")
    if before.get("text_sha256") != base_text_sha256:
        errors.append("merge local recovery before text hash differs from reconstructed transcript")
    if mode == "merge" and (
        int(after.get("segment_count") or 0) - int(before.get("segment_count") or 0)
        != counts["inserted"]
    ):
        errors.append("merge local recovery segment-count delta differs from inserted count")

    evidence_tempdir: tempfile.TemporaryDirectory[str] | None = None
    evidence_audio: Path | None = None
    evidence_sf: Any | None = None
    evidence_sample_rate = 0
    evidence_frames = 0
    attempts_claimed = any(
        isinstance(item, dict) and any(
            isinstance(attempt, dict) and attempt.get("status") != "error"
            for attempt in item.get("attempts") or []
        )
        for item in details
    )
    if enforce_audio_evidence and attempts_claimed:
        try:
            if audio_path is None or not audio_path.is_file():
                raise FileNotFoundError(str(audio_path or ""))
            import soundfile as sf
            from scribe_py.core.audio import standardize_audio_for_asr

            evidence_sf = sf

            evidence_tempdir = tempfile.TemporaryDirectory(prefix="localscribe-recovery-acceptance-")
            evidence_audio, rebuilt_stats = standardize_audio_for_asr(
                audio_path,
                Path(evidence_tempdir.name),
                audio_quality=audio_quality or None,
                mode=preprocess_mode or "adaptive",
            )
            expected_standardization = audio_standardization or {}
            expected_hash = str(expected_standardization.get("standardized_sha256") or "")
            if not expected_hash or _file_sha256(evidence_audio) != expected_hash:
                errors.append("local recovery standardized audio hash is invalid")
            for field in ("mode", "applied", "applied_filters", "fallback"):
                if field in expected_standardization and rebuilt_stats.get(field) != expected_standardization.get(field):
                    errors.append(f"local recovery standardized audio {field} differs from runtime manifest")
            evidence_info = sf.info(str(evidence_audio))
            evidence_sample_rate = int(evidence_info.samplerate)
            evidence_frames = int(evidence_info.frames)
            if evidence_sample_rate != 16000:
                raise ValueError(f"unsupported evidence sample rate: {evidence_sample_rate}")
        except Exception as exc:
            errors.append(f"local recovery audio evidence unavailable: {type(exc).__name__}")
            evidence_audio = None

    detailed_counts = {"matched_existing": 0, "inserted": 0, "rejected": 0, "error": 0}
    accepted_windows = 0
    detailed_failure_ranges: list[list[float]] = []
    for detail_index, detail in enumerate(details):
        if not isinstance(detail, dict):
            errors.append(f"local recovery detail {detail_index} must be an object")
            continue
        start = _finite_number(detail.get("start"))
        end = _finite_number(detail.get("end"))
        if start is None or end is None or start < 0 or end <= start:
            errors.append(f"local recovery detail {detail_index} has invalid range")
        decision = str(detail.get("decision") or "")
        decision_counter = {
            "matched_existing": "matched_existing",
            "insert_accepted": "inserted",
            "rejected": "rejected",
            "error": "error",
        }.get(decision)
        if decision_counter is None:
            errors.append(f"local recovery detail {detail_index} has invalid decision")
        else:
            detailed_counts[decision_counter] += 1

        original_failures = detail.get("original_failures")
        original_ranges: list[list[float]] = []
        if not isinstance(original_failures, list):
            errors.append(f"local recovery detail {detail_index} original_failures must be a list")
            original_failures = []
        for failure_index, failure in enumerate(original_failures):
            if not isinstance(failure, dict):
                errors.append(f"local recovery detail {detail_index} failure {failure_index} must be an object")
                continue
            failure_start = _finite_number(failure.get("start"))
            failure_end = _finite_number(failure.get("end"))
            if failure_start is None or failure_end is None or failure_start < 0 or failure_end <= failure_start:
                errors.append(f"local recovery detail {detail_index} failure {failure_index} has invalid range")
                continue
            original_ranges.append([round(failure_start, 3), round(failure_end, 3)])
        original_ranges.sort()
        if int(detail.get("window_count") or 0) != len(original_ranges):
            errors.append(f"local recovery detail {detail_index} window_count differs from original failures")
        detailed_failure_ranges.extend(original_ranges)

        computed_context = (
            _recovery_local_reference_from_segments(base_segments, start, end)
            if start is not None and end is not None
            else {"left": "", "right": "", "overlapping": "", "reference": ""}
        )
        for field, expected in (
            ("left_context", computed_context["left"]),
            ("right_context", computed_context["right"]),
            ("overlapping_context", computed_context["overlapping"]),
            ("local_reference", computed_context["reference"]),
        ):
            if str(detail.get(field) or "") != expected:
                errors.append(f"local recovery detail {detail_index} {field} is not derived from transcript")
        left_context = computed_context["left"]
        right_context = computed_context["right"]
        local_reference = computed_context["reference"]
        attempts = detail.get("attempts")
        if not isinstance(attempts, list):
            errors.append(f"local recovery detail {detail_index} attempts must be a list")
            attempts = []
        seen_attempts: set[tuple[str, str]] = set()
        attempts_by_evidence_id: dict[str, dict[str, Any]] = {}
        verified_attempts: list[dict[str, Any]] = []
        expected_min_required = max(
            2,
            int(math.ceil(max(0.0, (end or 0.0) - (start or 0.0)) * min_chars_per_s)),
        )
        detail_min_required = max(2, int(detail.get("min_required_chars") or 2))
        if detail_min_required != expected_min_required:
            errors.append(f"local recovery detail {detail_index} density threshold is not trusted")
        detail_min_required = expected_min_required
        expected_pads = {"exact": 0.0, "pad0.5": 0.5, "pad1.0": 1.0, "pad2.0": 2.0}
        for attempt_index, attempt in enumerate(attempts):
            if not isinstance(attempt, dict):
                errors.append(f"local recovery detail {detail_index} attempt {attempt_index} must be an object")
                continue
            framing = str(attempt.get("framing") or "")
            if framing not in expected_pads:
                errors.append(f"local recovery detail {detail_index} has invalid framing: {framing}")
            provider_id = str(attempt.get("provider_id") or "sensevoice-primary")
            provider_kind = str(attempt.get("provider_kind") or "primary_asr")
            model_id = str(attempt.get("model_id") or "iic/SenseVoiceSmall")
            model_family = str(attempt.get("model_family") or "sensevoice")
            attempt_key = (provider_id, framing)
            if attempt_key in seen_attempts:
                errors.append(f"local recovery detail {detail_index} repeats provider framing {provider_id}:{framing}")
            seen_attempts.add(attempt_key)
            if provider_kind == "primary_asr":
                if provider_id != "sensevoice-primary" or model_family != "sensevoice" or "sensevoice" not in model_id.lower():
                    errors.append(f"local recovery detail {detail_index} primary provider identity is invalid")
            elif provider_kind == "independent_asr":
                if (
                    provider_id != "qwen3-independent"
                    or model_family != "qwen3_asr"
                    or model_id != "mlx-community/Qwen3-ASR-1.7B-8bit"
                ):
                    errors.append(f"local recovery detail {detail_index} independent provider identity is invalid")
                if (
                    not attempt.get("model_revision")
                    or not attempt.get("config_sha256")
                    or not attempt.get("weights_manifest_sha256")
                ):
                    errors.append(f"local recovery detail {detail_index} independent provider fingerprint is missing")
            else:
                errors.append(f"local recovery detail {detail_index} provider kind is invalid")
            pad_s = _finite_number(attempt.get("pad_s"))
            if framing in expected_pads and (pad_s is None or abs(pad_s - expected_pads[framing]) > 0.001):
                errors.append(f"local recovery detail {detail_index} framing {framing} has invalid pad")
            if (
                enforce_audio_evidence
                and attempt.get("status") != "error"
                and framing in expected_pads
                and start is not None
                and end is not None
                and evidence_audio is not None
                and evidence_sf is not None
                and evidence_sample_rate > 0
            ):
                available_duration = evidence_frames / float(evidence_sample_rate)
                expected_start = max(0.0, start - expected_pads[framing])
                expected_end = min(available_duration, end + expected_pads[framing])
                reported_start = _finite_number(attempt.get("slice_start"))
                reported_end = _finite_number(attempt.get("slice_end"))
                if reported_start is None or abs(reported_start - expected_start) > 0.001:
                    errors.append(f"local recovery detail {detail_index} framing {framing} slice_start is invalid")
                if reported_end is None or abs(reported_end - expected_end) > 0.001:
                    errors.append(f"local recovery detail {detail_index} framing {framing} slice_end is invalid")
                start_frame = max(0, int(round(expected_start * evidence_sample_rate)))
                end_frame = min(evidence_frames, int(round(expected_end * evidence_sample_rate)))
                try:
                    slice_data, read_rate = evidence_sf.read(
                        str(evidence_audio),
                        dtype="float32",
                        start=start_frame,
                        stop=end_frame,
                        always_2d=True,
                    )
                    if int(read_rate) != evidence_sample_rate:
                        raise ValueError("slice_sample_rate_mismatch")
                    if slice_data.shape[1] > 1:
                        slice_data = slice_data.mean(axis=1)
                    else:
                        slice_data = slice_data[:, 0]
                    expected_slice_hash = hashlib.sha256(slice_data.tobytes()).hexdigest()
                    if attempt.get("slice_sha256") != expected_slice_hash:
                        errors.append(f"local recovery detail {detail_index} framing {framing} slice_sha256 is invalid")
                except Exception as exc:
                    errors.append(
                        f"local recovery detail {detail_index} framing {framing} audio evidence failed: {type(exc).__name__}"
                    )
            for field in ("raw", "normalized", "residual"):
                if field not in attempt:
                    errors.append(f"local recovery detail {detail_index} attempt missing {field}")
            evidence_id = "|".join((provider_id, framing, str(attempt.get("slice_sha256") or "")))
            attempts_by_evidence_id[evidence_id] = attempt
            if attempt.get("evidence_sha256"):
                expected_evidence_hash = hashlib.sha256(
                    json.dumps(
                        {
                            "provider_id": provider_id,
                            "model_id": model_id,
                            "model_revision": attempt.get("model_revision"),
                            "config_sha256": attempt.get("config_sha256"),
                            "weights_manifest_sha256": attempt.get("weights_manifest_sha256"),
                            "slice_sha256": str(attempt.get("slice_sha256") or ""),
                            "raw": attempt.get("raw"),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                if attempt.get("evidence_sha256") != expected_evidence_hash:
                    errors.append(f"local recovery detail {detail_index} attempt evidence hash is invalid")
            verified = dict(attempt)
            verified.update({
                "verified_provider_id": provider_id,
                "verified_provider_kind": provider_kind,
                "verified_model_id": model_id,
                "verified_model_family": model_family,
            })
            if attempt.get("status") == "error":
                if not attempt.get("error"):
                    errors.append(f"local recovery detail {detail_index} error attempt lacks error evidence")
                verified.update({
                    "verified_status": "error",
                    "verified_normalized": "",
                    "verified_residual": "",
                })
            else:
                from scribe_py.core.sensevoice_recovery import deduplicate_candidate

                deduplicated = deduplicate_candidate(
                    str(attempt.get("raw") or ""), left_context, right_context
                )
                computed_normalized = str(deduplicated.get("normalized") or "")
                computed_residual = str(deduplicated.get("residual_normalized") or "")
                computed_residual_text = str(deduplicated.get("residual_text") or "")
                if str(attempt.get("normalized") or "") != computed_normalized:
                    errors.append(f"local recovery detail {detail_index} attempt normalized text is forged")
                reported_required_chars = max(2, int(attempt.get("min_required_chars") or 2))
                if reported_required_chars != detail_min_required:
                    errors.append(f"local recovery detail {detail_index} attempt density threshold differs")
                required_chars = detail_min_required
                reference_normalized = _recovery_normalized(local_reference)
                if not computed_normalized:
                    verified_status = "rejected"
                elif computed_normalized in reference_normalized:
                    verified_status = "matched_existing"
                elif reference_normalized:
                    verified_status = "rejected"
                elif (
                    len(computed_normalized) < required_chars
                    or _recovery_repeated(computed_normalized)
                    or bool(attempt.get("hallucination_risk"))
                    or len(computed_residual) < required_chars
                    or _recovery_repeated(computed_residual)
                ):
                    verified_status = "rejected"
                else:
                    verified_status = "valid"
                expected_residual = "" if verified_status == "matched_existing" else computed_residual
                if str(attempt.get("residual") or "") != expected_residual:
                    errors.append(f"local recovery detail {detail_index} attempt residual is forged")
                expected_residual_text = "" if verified_status == "matched_existing" else computed_residual_text
                if str(attempt.get("residual_text") or "") != expected_residual_text:
                    errors.append(f"local recovery detail {detail_index} attempt residual_text is forged")
                if str(attempt.get("status") or "") != verified_status:
                    errors.append(f"local recovery detail {detail_index} attempt status is forged")
                verified.update({
                    "verified_status": verified_status,
                    "verified_normalized": computed_normalized,
                    "verified_residual": expected_residual,
                    "verified_residual_text": expected_residual_text,
                })
            verified_attempts.append(verified)

        inserted_raw_text = str(detail.get("inserted_raw_text") or "")
        inserted_text = str(detail.get("inserted_text") or "")
        evidence_fields: dict[str, list[str]] = {}
        for field in ("evidence_framings", "evidence_providers", "evidence_models", "evidence_ids"):
            raw_evidence = detail.get(field) or []
            if not isinstance(raw_evidence, list):
                errors.append(f"local recovery detail {detail_index} {field} must be a list")
                raw_evidence = []
            evidence_fields[field] = [str(item) for item in raw_evidence]
        consensus = _recovery_normalized(detail.get("consensus"))
        verified_decision = _recovery_decision_from_attempts(verified_attempts)
        evidence_decision = str(detail.get("evidence_decision") or decision)
        if evidence_decision != verified_decision["decision"]:
            errors.append(f"local recovery detail {detail_index} evidence decision is not supported by attempts")
        if consensus != verified_decision["consensus"]:
            errors.append(f"local recovery detail {detail_index} consensus is not supported by attempts")
        for field in ("evidence_framings", "evidence_providers", "evidence_models", "evidence_ids"):
            if sorted(set(evidence_fields[field])) != verified_decision[field]:
                errors.append(f"local recovery detail {detail_index} {field} are not supported by attempts")
        for field in ("primary_status", "primary_consensus", "primary_evidence_framings"):
            reported = detail.get(field) or ([] if field.endswith("framings") else "")
            expected = verified_decision[field]
            if (sorted(str(item) for item in reported) if isinstance(reported, list) else str(reported)) != expected:
                errors.append(f"local recovery detail {detail_index} {field} is not supported by attempts")

        expected_inserted_text = ""
        normalization_succeeded = False
        if verified_decision["decision"] == "insert_accepted":
            canonical_raw_text = str(verified_decision.get("inserted_raw_text") or "")
            try:
                from scribe_py.core.text_normalizer import normalize_segments
                from scribe_py.core.types import Segment

                raw_context = [
                    Segment(
                        start=float(item.get("start") or 0.0),
                        end=float(item.get("end") or 0.0),
                        text=str(item.get("original_text") or item.get("text") or ""),
                        original_text=item.get("original_text"),
                    )
                    for item in base_segments
                ]
                raw_context.append(Segment(
                    start=float(start or 0.0),
                    end=float(end or 0.0),
                    text=canonical_raw_text,
                    original_text=canonical_raw_text,
                ))
                raw_context.sort(key=lambda segment: (segment.start, segment.end))
                normalized_context, _normalization_stats = normalize_segments(
                    raw_context,
                    language=normalization_language or str(text_normalization.get("language") or "zh"),
                    profile=normalization_profile,
                )
                normalized_insertions = [
                    segment
                    for segment in normalized_context
                    if abs(segment.start - float(start or 0.0)) <= 0.001
                    and abs(segment.end - float(end or 0.0)) <= 0.001
                ]
                if len(normalized_insertions) == 1:
                    expected_inserted_text = normalized_insertions[0].text
                    normalization_succeeded = len(_recovery_normalized(expected_inserted_text)) >= 2
            except Exception:
                normalization_succeeded = False

        expected_final_decision = verified_decision["decision"]
        if verified_decision["decision"] == "insert_accepted" and not normalization_succeeded:
            expected_final_decision = "rejected"
        if decision != expected_final_decision:
            errors.append(f"local recovery detail {detail_index} final decision disagrees with normalization gate")

        if decision == "matched_existing":
            accepted_windows += int(detail.get("window_count") or 0)
            if inserted_text:
                errors.append(f"local recovery detail {detail_index} matched_existing inserted text must be empty")
            if inserted_raw_text:
                errors.append(f"local recovery detail {detail_index} matched_existing inserted raw text must be empty")
        elif verified_decision["decision"] == "insert_accepted":
            if len(consensus) < 2:
                errors.append(f"local recovery detail {detail_index} insertion consensus is too short")
            if inserted_raw_text != verified_decision.get("inserted_raw_text"):
                errors.append(f"local recovery detail {detail_index} inserted raw text differs from canonical evidence")
            if decision == "insert_accepted":
                accepted_windows += int(detail.get("window_count") or 0)
                if len(_recovery_normalized(inserted_raw_text)) < 2:
                    errors.append(f"local recovery detail {detail_index} inserted raw text is too short")
                if len(_recovery_normalized(inserted_text)) < 2:
                    errors.append(f"local recovery detail {detail_index} inserted text is too short")
                if _recovery_normalized(inserted_raw_text) != consensus:
                    errors.append(f"local recovery detail {detail_index} inserted raw text differs from consensus")
                if inserted_text != expected_inserted_text:
                    errors.append(f"local recovery detail {detail_index} inserted text is not trusted-normalizer output")
            else:
                if inserted_text:
                    errors.append(f"local recovery detail {detail_index} normalization-rejected text must be empty")
                if not detail.get("normalization_rejection_reason"):
                    errors.append(f"local recovery detail {detail_index} normalization rejection reason is missing")
        if evidence_decision in {"matched_existing", "insert_accepted"}:
            providers = set(evidence_fields["evidence_providers"])
            models = set(evidence_fields["evidence_models"])
            if providers != {"sensevoice-primary", "qwen3-independent"}:
                errors.append(f"local recovery detail {detail_index} lacks required independent providers")
            if not any("SenseVoice" in model for model in models) or "mlx-community/Qwen3-ASR-1.7B-8bit" not in models:
                errors.append(f"local recovery detail {detail_index} lacks required independent models")
            evidence_attempts = [attempts_by_evidence_id.get(item) for item in evidence_fields["evidence_ids"]]
            if any(item is None for item in evidence_attempts):
                errors.append(f"local recovery detail {detail_index} contains unknown evidence ids")
            primary_evidence = [item for item in evidence_attempts if item and item.get("provider_kind") == "primary_asr"]
            independent_evidence = [item for item in evidence_attempts if item and item.get("provider_kind") == "independent_asr"]
            primary_hashes = {str(item.get("slice_sha256") or "") for item in primary_evidence}
            primary_evidence_ok = (
                len(primary_evidence) >= 1
                and "" not in primary_hashes
                if evidence_decision == "matched_existing"
                else len(primary_evidence) >= 2 and "" not in primary_hashes and len(primary_hashes) >= 2
            )
            if not primary_evidence_ok:
                errors.append(f"local recovery detail {detail_index} primary evidence is not stable across slices")
            if len(independent_evidence) < 1:
                errors.append(f"local recovery detail {detail_index} lacks independent ASR evidence")
            for attempt in evidence_attempts:
                if attempt and not attempt.get("evidence_sha256"):
                    errors.append(f"local recovery detail {detail_index} evidence attempt lacks evidence hash")
            for attempt in independent_evidence:
                if (
                    attempt.get("model_revision") != provider.get("model_revision")
                    or attempt.get("config_sha256") != provider.get("config_sha256")
                    or attempt.get("weights_manifest_sha256") != provider.get("weights_manifest_sha256")
                ):
                    errors.append(f"local recovery detail {detail_index} independent model fingerprint differs from provider")
            expected_status = "matched_existing" if evidence_decision == "matched_existing" else "valid"
            for attempt in evidence_attempts:
                if not attempt or attempt.get("status") != expected_status:
                    errors.append(f"local recovery detail {detail_index} contains unsupported evidence attempt")
                    continue
                value = attempt.get("normalized") if evidence_decision == "matched_existing" else attempt.get("residual")
                if _recovery_normalized(value) != consensus:
                    errors.append(f"local recovery detail {detail_index} evidence texts are inconsistent")
    details_truncated = bool(raw_recovery.get("details_truncated"))
    if details_truncated:
        errors.append("local recovery machine-verifiable details must not be truncated")
    if mode == "off":
        if details:
            errors.append("off local recovery must not contain details")
    elif len(details) != counts["pending_groups"]:
        errors.append("local recovery details do not equal pending groups")

    if mode in {"audit", "merge"}:
        if sorted(detailed_failure_ranges) != before_failed:
            errors.append("local recovery original failures do not equal before failed partition")
        if len(detailed_failure_ranges) != counts["pending_windows"]:
            errors.append("local recovery original failure count differs from pending windows")
    for name, value in detailed_counts.items():
        if value != counts[name]:
            errors.append(f"local recovery {name} count differs from details")
    if mode == "merge" and before and after:
        if int(after.get("segment_count") or 0) - int(before.get("segment_count") or 0) != counts["inserted"]:
            errors.append("local recovery inserted count does not match segment-count delta")
        if int(after.get("covered_count") or 0) - int(before.get("covered_count") or 0) != accepted_windows:
            errors.append("local recovery accepted windows do not match covered-count delta")
        if int(before.get("failed_count") or 0) - int(after.get("failed_count") or 0) != accepted_windows:
            errors.append("local recovery accepted windows do not match failed-count delta")
    if sum(len(item.get("attempts") or []) for item in details if isinstance(item, dict)) != counts["attempts"]:
        errors.append("local recovery attempt count differs from detailed attempts")
    if evidence_tempdir is not None:
        evidence_tempdir.cleanup()

    return errors, warnings, metrics


def validate_case(
    label: str,
    transcript_path: Path,
    *,
    require_exports: bool = False,
    require_speech_coverage: bool = False,
    min_speech_coverage_ratio: float = 0.99,
    max_uncovered_speech_seconds: float = 3.0,
    max_edge_uncovered_speech_seconds: float = 1.0,
    min_recovery_chars_per_second: float = 0.75,
) -> CaseResult:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        data = _read_json(transcript_path)
    except Exception as exc:
        return CaseResult(label, str(transcript_path), False, [f"cannot read transcript JSON: {exc}"], [], {}, {})
    if not isinstance(data, dict):
        return CaseResult(label, str(transcript_path), False, ["transcript JSON must be an object"], [], {}, {})

    duration = _finite_number(data.get("duration") or data.get("duration_s")) or 0.0
    segment_errors, segment_warnings, metrics = _validate_segments(list(data.get("segments") or []), duration)
    errors.extend(segment_errors)
    warnings.extend(segment_warnings)
    filter_stats = data.get("filter_stats") or {}
    settings = filter_stats.get("settings") or {}
    coverage_errors, coverage_warnings, coverage_metrics = _validate_speech_coverage(
        filter_stats.get("speech_coverage"),
        required=require_speech_coverage,
        min_ratio=min_speech_coverage_ratio,
        max_uncovered_s=max_uncovered_speech_seconds,
        max_edge_uncovered_s=max_edge_uncovered_speech_seconds,
    )
    errors.extend(coverage_errors)
    warnings.extend(coverage_warnings)
    recovery_audio_path = Path(str(data.get("audio") or "")).expanduser()
    if recovery_audio_path and not recovery_audio_path.is_absolute():
        recovery_audio_path = (transcript_path.parent / recovery_audio_path).resolve()
    recovery_errors, recovery_warnings, recovery_metrics = _validate_local_recovery(
        (filter_stats.get("speech_coverage") or {}).get("local_recovery")
        if isinstance(filter_stats.get("speech_coverage"), dict)
        else None,
        segments=list(data.get("segments") or []),
        raw_coverage=filter_stats.get("speech_coverage"),
        audio_path=recovery_audio_path,
        preprocess_mode=str(((filter_stats.get("audio_standardization") or {}).get("mode") or "adaptive")),
        audio_standardization=filter_stats.get("audio_standardization") or {},
        audio_quality=filter_stats.get("audio_quality") or {},
        normalization_language=str(data.get("language") or ""),
        normalization_profile=(filter_stats.get("text_normalization") or {}).get("profile"),
        min_chars_per_s=min_recovery_chars_per_second,
        enforce_audio_evidence=True,
    )
    errors.extend(recovery_errors)
    warnings.extend(recovery_warnings)
    if require_speech_coverage and coverage_metrics.get("speech_coverage_status") == "no_speech":
        if metrics.get("chars", 0) > 0:
            errors.append("speech coverage reports no_speech but transcript contains text")
        else:
            errors = [
                error
                for error in errors
                if error not in {"segments is empty", "transcript has no segments"}
            ]
    metrics.update({
        "audio": str(data.get("audio") or ""),
        "duration_s": duration,
        "transcribe_seconds": _finite_number(data.get("transcribe_seconds")) or 0.0,
        "rtf": _finite_number(data.get("rtf")) or 0.0,
        "backend": str(data.get("backend") or ""),
        "model_id": str(data.get("model_id") or ""),
        "preprocess_mode": str(((filter_stats.get("audio_standardization") or {}).get("mode") or "")),
        "timing_align": settings.get("sensevoice_timing_align"),
        "timing_mode": str(filter_stats.get("timing_mode") or ""),
        "timing_reliable": filter_stats.get("timing_reliable"),
        "timing_alignment_ok": filter_stats.get("timing_alignment_ok"),
        "timing_equal_char_ratio": filter_stats.get("equal_char_ratio"),
        "wallclock_vad_ranges": filter_stats.get("wallclock_vad_ranges"),
        "wallclock_vad_chunks": filter_stats.get("wallclock_vad_chunks"),
        **coverage_metrics,
        **recovery_metrics,
    })
    if not metrics["backend"]:
        errors.append("backend is missing")
    if not metrics["model_id"]:
        errors.append("model_id is missing")
    if metrics["timing_align"] is True and metrics["timing_reliable"] is False:
        errors.append("precise timing was requested but the result is marked unreliable")
    if metrics["timing_align"] is True and metrics["timing_alignment_ok"] is False:
        errors.append("precise timing alignment failed")

    expected_texts = _expected_texts([segment for segment in data.get("segments") or [] if isinstance(segment, dict)])
    txt_check = _check_txt(transcript_path.with_suffix(".txt"), expected_texts)
    srt_check = _check_srt(transcript_path.with_suffix(".srt"), expected_texts)
    exports = {"txt": txt_check, "srt": srt_check}
    for name, check in exports.items():
        if require_exports and not check.present:
            errors.append(f"{name.upper()} export is missing")
        errors.extend(f"{name.upper()}: {error}" for error in check.errors)

    return CaseResult(label, str(transcript_path), not errors, errors, warnings, metrics, exports)


def _export_dict(check: ExportCheck) -> dict[str, Any]:
    return {
        "path": check.path,
        "present": check.present,
        "entries": check.entries,
        "text_matches": check.text_matches,
        "timing_valid": check.timing_valid,
        "errors": check.errors,
    }


def _case_dict(case: CaseResult) -> dict[str, Any]:
    return {
        "label": case.label,
        "transcript": case.transcript,
        "ok": case.ok,
        "errors": case.errors,
        "warnings": case.warnings,
        "metrics": case.metrics,
        "exports": {name: _export_dict(check) for name, check in case.exports.items()},
    }


def render_markdown(cases: list[CaseResult]) -> str:
    lines = [
        "# ASR 长音频验收报告\n\n",
        f"- 录音数：{len(cases)}\n",
        f"- 通过：{sum(case.ok for case in cases)}\n",
        f"- 失败：{sum(not case.ok for case in cases)}\n\n",
        "| 录音 | 状态 | 时长秒 | 段数 | 字符数 | RTF | 后端 | 模型 | 语音覆盖率 | 最大漏语音秒 | 首/尾漏语音秒 | 最大间隔秒 | 重叠数 | TXT | SRT |\n",
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|\n",
    ]
    for case in cases:
        metrics = case.metrics
        exports = case.exports
        lines.append(
            f"| {case.label} | {'PASS' if case.ok else 'FAIL'} | {float(metrics.get('duration_s') or 0):.2f} | "
            f"{metrics.get('segments', 0)} | {metrics.get('chars', 0)} | {float(metrics.get('rtf') or 0):.3f} | "
            f"{metrics.get('backend', '')} | {metrics.get('model_id', '')} | "
            f"{float(metrics.get('speech_coverage_ratio') or 0):.4f} | "
            f"{float(metrics.get('max_uncovered_speech_s') or 0):.3f} | "
            f"{float(metrics.get('leading_uncovered_speech_s') or 0):.3f}/{float(metrics.get('trailing_uncovered_speech_s') or 0):.3f} | "
            f"{float(metrics.get('max_gap_s') or 0):.3f} | {metrics.get('overlap_count', 0)} | "
            f"{'PASS' if exports.get('txt') and exports['txt'].present and not exports['txt'].errors else '缺失' if not exports.get('txt') or not exports['txt'].present else 'FAIL'} | "
            f"{'PASS' if exports.get('srt') and exports['srt'].present and not exports['srt'].errors else '缺失' if not exports.get('srt') or not exports['srt'].present else 'FAIL'} |\n"
        )
    for case in cases:
        lines.append(f"\n## {case.label}\n\n")
        lines.append(f"- JSON：`{case.transcript}`\n")
        lines.append(f"- 文本 MD5：`{case.metrics.get('md5', '')}`\n")
        lines.append(f"- 文本 SHA256：`{case.metrics.get('sha256', '')}`\n")
        lines.append(
            f"- 有效语音覆盖：状态 `{case.metrics.get('speech_coverage_status', 'missing')}`，"
            f"证据 `{case.metrics.get('speech_coverage_basis', 'missing')}`，"
            f"覆盖率 `{float(case.metrics.get('speech_coverage_ratio') or 0):.4f}`，"
            f"最大未覆盖 `{float(case.metrics.get('max_uncovered_speech_s') or 0):.3f}s`，"
            f"开头/结尾 `{float(case.metrics.get('leading_uncovered_speech_s') or 0):.3f}s/"
            f"{float(case.metrics.get('trailing_uncovered_speech_s') or 0):.3f}s`，"
            f"分块 `{case.metrics.get('wallclock_recognized_chunks', 0)}/"
            f"{case.metrics.get('wallclock_attempted_chunks', 0)}` 成功"
            f"（失败 `{case.metrics.get('wallclock_failed_chunks', 0)}`）\n"
        )
        if case.errors:
            lines.append("- 错误：\n")
            lines.extend(f"  - {error}\n" for error in case.errors)
        if case.warnings:
            lines.append("- 提醒：\n")
            lines.extend(f"  - {warning}\n" for warning in case.warnings)
        if not case.errors and not case.warnings:
            lines.append("- 未发现结构、时间戳或导出一致性问题。\n")
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查已完成的长音频 ASR JSON、时间戳和导出一致性")
    parser.add_argument("--case", action="append", required=True, help="LABEL=transcript.json，可重复")
    parser.add_argument("--out-dir", type=Path, default=Path("output/asr_longform_acceptance_latest"))
    parser.add_argument("--require-exports", action="store_true", help="TXT/SRT 缺失时判定失败")
    parser.add_argument(
        "--require-speech-coverage",
        action="store_true",
        help="缺少有效语音覆盖诊断或覆盖指标不达标时判定失败",
    )
    parser.add_argument("--min-speech-coverage-ratio", type=float, default=0.99)
    parser.add_argument("--max-uncovered-speech-seconds", type=float, default=3.0)
    parser.add_argument("--max-edge-uncovered-speech-seconds", type=float, default=1.0)
    parser.add_argument("--min-recovery-chars-per-second", type=float, default=0.75)
    args = parser.parse_args(argv)
    if not 0 <= args.min_speech_coverage_ratio <= 1:
        parser.error("--min-speech-coverage-ratio must be between 0 and 1")
    if args.max_uncovered_speech_seconds < 0 or args.max_edge_uncovered_speech_seconds < 0:
        parser.error("speech coverage duration thresholds must be non-negative")
    if args.min_recovery_chars_per_second <= 0:
        parser.error("--min-recovery-chars-per-second must be positive")

    cases = [
        validate_case(
            label,
            path,
            require_exports=args.require_exports,
            require_speech_coverage=args.require_speech_coverage,
            min_speech_coverage_ratio=args.min_speech_coverage_ratio,
            max_uncovered_speech_seconds=args.max_uncovered_speech_seconds,
            max_edge_uncovered_speech_seconds=args.max_edge_uncovered_speech_seconds,
            min_recovery_chars_per_second=args.min_recovery_chars_per_second,
        )
        for label, path in (_parse_labeled_path(value, "--case") for value in args.case)
    ]
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "asr_longform_acceptance.json"
    markdown_path = out_dir / "asr_longform_acceptance.md"
    payload = {"ok": all(case.ok for case in cases), "cases": [_case_dict(case) for case in cases]}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(cases), encoding="utf-8")
    print(json.dumps({
        "ok": payload["ok"],
        "checked": [case.label for case in cases],
        "json": str(json_path),
        "markdown": str(markdown_path),
    }, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
