#!/usr/bin/env python3
"""Evaluate speaker diarization output against human reference annotations.

The implementation is dependency-free and supports RTTM plus LocalScribe-like
JSON.  Metrics are computed on exact time boundaries and therefore preserve
overlapping speech instead of reducing the recording to one label per segment.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EPSILON = 1e-9


@dataclass(frozen=True)
class Segment:
    uri: str
    start: float
    end: float
    speaker: str


@dataclass
class RecordingMetrics:
    uri: str
    reference_speaker_time_s: float
    miss_s: float
    false_alarm_s: float
    confusion_s: float
    der: float | None
    jer: float | None
    reference_speaker_count: int
    prediction_speaker_count: int
    speaker_count_error: int
    speaker_count_absolute_error: int
    speaker_mapping: dict[str, str]
    jer_speaker_mapping: dict[str, str]
    per_reference_speaker_jer: dict[str, float]

    @property
    def diarization_error_s(self) -> float:
        return self.miss_s + self.false_alarm_s + self.confusion_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "reference_speaker_time_s": self.reference_speaker_time_s,
            "miss_s": self.miss_s,
            "false_alarm_s": self.false_alarm_s,
            "confusion_s": self.confusion_s,
            "diarization_error_s": self.diarization_error_s,
            "der": self.der,
            "jer": self.jer,
            "reference_speaker_count": self.reference_speaker_count,
            "prediction_speaker_count": self.prediction_speaker_count,
            "speaker_count_error": self.speaker_count_error,
            "speaker_count_absolute_error": self.speaker_count_absolute_error,
            "speaker_mapping": self.speaker_mapping,
            "jer_speaker_mapping": self.jer_speaker_mapping,
            "per_reference_speaker_jer": self.per_reference_speaker_jer,
        }


@dataclass
class EvaluationResult:
    recordings: list[RecordingMetrics]

    def aggregate(self) -> dict[str, Any]:
        reference_time = sum(item.reference_speaker_time_s for item in self.recordings)
        miss = sum(item.miss_s for item in self.recordings)
        false_alarm = sum(item.false_alarm_s for item in self.recordings)
        confusion = sum(item.confusion_s for item in self.recordings)
        diarization_error = miss + false_alarm + confusion
        jer_values = [
            value
            for item in self.recordings
            for value in item.per_reference_speaker_jer.values()
        ]
        return {
            "recording_count": len(self.recordings),
            "reference_speaker_time_s": reference_time,
            "miss_s": miss,
            "false_alarm_s": false_alarm,
            "confusion_s": confusion,
            "diarization_error_s": diarization_error,
            "der": diarization_error / reference_time if reference_time > EPSILON else None,
            "miss_rate": miss / reference_time if reference_time > EPSILON else None,
            "false_alarm_rate": false_alarm / reference_time if reference_time > EPSILON else None,
            "confusion_rate": confusion / reference_time if reference_time > EPSILON else None,
            "jer": sum(jer_values) / len(jer_values) if jer_values else None,
            "reference_speaker_count": sum(
                item.reference_speaker_count for item in self.recordings
            ),
            "prediction_speaker_count": sum(
                item.prediction_speaker_count for item in self.recordings
            ),
            "speaker_count_error": sum(item.speaker_count_error for item in self.recordings),
            "speaker_count_absolute_error": sum(
                item.speaker_count_absolute_error for item in self.recordings
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "aggregate": self.aggregate(),
            "recordings": [item.as_dict() for item in self.recordings],
        }


def _as_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 不是有效数字: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} 必须是有限数字: {value!r}")
    return result


def _clean_uri(value: Any, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    if "/" in raw or "\\" in raw:
        return Path(raw).stem or fallback
    return raw


def _speaker_values(row: dict[str, Any]) -> list[str]:
    raw = row.get("speaker")
    if raw is None:
        raw = row.get("speaker_id", row.get("label"))
    if raw is None:
        raw = row.get("speakers", row.get("speaker_ids"))
    values = raw if isinstance(raw, list) else [raw]
    return [str(value).strip() for value in values if str(value or "").strip()]


def _json_segment_rows(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)], {}
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是对象或片段数组")
    for key in ("segments", "diarization_segments", "speaker_segments"):
        rows = data.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)], data
    diarization = data.get("diarization")
    if isinstance(diarization, dict) and isinstance(diarization.get("segments"), list):
        return [row for row in diarization["segments"] if isinstance(row, dict)], data
    raise ValueError("JSON 中未找到 segments/diarization_segments/speaker_segments")


def load_json(path: Path) -> dict[str, list[Segment]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    rows, root = _json_segment_rows(data)
    fallback_uri = path.stem
    root_uri = _clean_uri(
        root.get("uri")
        or root.get("recording_id")
        or root.get("case")
        or root.get("audio")
        or root.get("source_audio")
        or root.get("audio_path"),
        fallback_uri,
    )
    recordings: dict[str, list[Segment]] = defaultdict(list)
    for index, row in enumerate(rows):
        start = _as_float(row.get("start"), f"segments[{index}].start")
        if row.get("end") is not None:
            end = _as_float(row.get("end"), f"segments[{index}].end")
        elif row.get("duration") is not None:
            end = start + _as_float(row.get("duration"), f"segments[{index}].duration")
        else:
            raise ValueError(f"segments[{index}] 缺少 end 或 duration")
        if start < 0 or end <= start:
            raise ValueError(f"segments[{index}] 时间范围无效: {start} - {end}")
        speakers = _speaker_values(row)
        if not speakers:
            continue
        uri = _clean_uri(
            row.get("uri") or row.get("recording_id") or row.get("file"),
            root_uri,
        )
        for speaker in speakers:
            recordings[uri].append(Segment(uri, start, end, speaker))
    return _sorted_recordings(recordings)


def load_rttm(path: Path) -> dict[str, list[Segment]]:
    recordings: dict[str, list[Segment]] = defaultdict(list)
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8 or fields[0].upper() != "SPEAKER":
            raise ValueError(f"RTTM 第 {line_number} 行格式无效: {raw_line}")
        uri = fields[1]
        start = _as_float(fields[3], f"RTTM 第 {line_number} 行 start")
        duration = _as_float(fields[4], f"RTTM 第 {line_number} 行 duration")
        speaker = fields[7].strip()
        if start < 0 or duration <= 0 or not speaker:
            raise ValueError(f"RTTM 第 {line_number} 行包含无效时间或说话人")
        recordings[uri].append(Segment(uri, start, start + duration, speaker))
    return _sorted_recordings(recordings)


def _sorted_recordings(
    recordings: dict[str, list[Segment]],
) -> dict[str, list[Segment]]:
    return {
        uri: sorted(segments, key=lambda item: (item.start, item.end, item.speaker))
        for uri, segments in sorted(recordings.items())
    }


def load_annotations(path: Path) -> dict[str, list[Segment]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"标注文件不存在: {resolved}")
    if resolved.suffix.lower() == ".rttm":
        return load_rttm(resolved)
    if resolved.suffix.lower() == ".json":
        return load_json(resolved)
    raise ValueError(f"不支持的标注格式: {resolved.suffix}，仅支持 .rttm 和 .json")


def _event_table(segments: Iterable[Segment]) -> dict[float, Counter[str]]:
    events: dict[float, Counter[str]] = defaultdict(Counter)
    for segment in segments:
        events[segment.start][segment.speaker] += 1
        events[segment.end][segment.speaker] -= 1
    return events


def _active_sets(
    reference: list[Segment], prediction: list[Segment]
) -> list[tuple[float, set[str], set[str]]]:
    ref_events = _event_table(reference)
    pred_events = _event_table(prediction)
    boundaries = sorted(set(ref_events) | set(pred_events))
    ref_active: Counter[str] = Counter()
    pred_active: Counter[str] = Counter()
    intervals: list[tuple[float, set[str], set[str]]] = []
    for index, boundary in enumerate(boundaries[:-1]):
        ref_active.update(ref_events.get(boundary, {}))
        pred_active.update(pred_events.get(boundary, {}))
        ref_active += Counter()
        pred_active += Counter()
        duration = boundaries[index + 1] - boundary
        if duration > EPSILON:
            intervals.append((duration, set(ref_active), set(pred_active)))
    return intervals


def _maximum_weight_assignment(
    rows: list[str],
    columns: list[str],
    weights: dict[tuple[str, str], float],
) -> dict[str, str]:
    """Return a maximum-weight one-to-one row -> column assignment."""
    if not rows or not columns:
        return {}
    size = max(len(rows), len(columns))
    max_weight = max(weights.values(), default=0.0)
    costs = [[max_weight] * size for _ in range(size)]
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            costs[row_index][column_index] = max_weight - weights.get((row, column), 0.0)

    # Hungarian algorithm for a square minimization matrix.
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        min_values = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = costs[row0 - 1][column - 1] - u[row0] - v[column]
                if current < min_values[column]:
                    min_values[column] = current
                    way[column] = column0
                if min_values[column] < delta:
                    delta = min_values[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_values[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment: dict[str, str] = {}
    for column_index in range(1, size + 1):
        row_index = p[column_index]
        if row_index <= len(rows) and column_index <= len(columns):
            row = rows[row_index - 1]
            column = columns[column_index - 1]
            if weights.get((row, column), 0.0) > EPSILON:
                assignment[row] = column
    return assignment


def evaluate_recording(
    uri: str,
    reference: list[Segment],
    prediction: list[Segment],
) -> RecordingMetrics:
    intervals = _active_sets(reference, prediction)
    reference_speakers = sorted({segment.speaker for segment in reference})
    prediction_speakers = sorted({segment.speaker for segment in prediction})
    overlap: dict[tuple[str, str], float] = defaultdict(float)
    reference_duration: Counter[str] = Counter()
    prediction_duration: Counter[str] = Counter()
    reference_speaker_time = 0.0
    for duration, ref_active, pred_active in intervals:
        reference_speaker_time += len(ref_active) * duration
        for speaker in ref_active:
            reference_duration[speaker] += duration
        for speaker in pred_active:
            prediction_duration[speaker] += duration
        for predicted in pred_active:
            for expected in ref_active:
                overlap[(predicted, expected)] += duration

    speaker_mapping = _maximum_weight_assignment(
        prediction_speakers,
        reference_speakers,
        overlap,
    )
    miss = 0.0
    false_alarm = 0.0
    confusion = 0.0
    for duration, ref_active, pred_active in intervals:
        ref_count = len(ref_active)
        pred_count = len(pred_active)
        correct = sum(
            1
            for predicted in pred_active
            if speaker_mapping.get(predicted) in ref_active
        )
        miss += max(0, ref_count - pred_count) * duration
        false_alarm += max(0, pred_count - ref_count) * duration
        confusion += (min(ref_count, pred_count) - correct) * duration

    # DIHARD JER uses the same globally optimal co-occurrence mapping as DER,
    # then gives every reference speaker equal weight regardless of duration.
    ref_to_pred = {expected: predicted for predicted, expected in speaker_mapping.items()}
    per_speaker_jer: dict[str, float] = {}
    for expected in reference_speakers:
        predicted = ref_to_pred.get(expected)
        if predicted is None:
            per_speaker_jer[expected] = 1.0
            continue
        intersection = overlap.get((predicted, expected), 0.0)
        union = reference_duration[expected] + prediction_duration[predicted] - intersection
        per_speaker_jer[expected] = 1.0 - (intersection / union if union > EPSILON else 0.0)
    jer_mapping = dict(speaker_mapping)
    jer = (
        sum(per_speaker_jer.values()) / len(per_speaker_jer)
        if per_speaker_jer
        else None
    )
    der = (
        (miss + false_alarm + confusion) / reference_speaker_time
        if reference_speaker_time > EPSILON
        else None
    )
    count_error = len(prediction_speakers) - len(reference_speakers)
    return RecordingMetrics(
        uri=uri,
        reference_speaker_time_s=reference_speaker_time,
        miss_s=miss,
        false_alarm_s=false_alarm,
        confusion_s=confusion,
        der=der,
        jer=jer,
        reference_speaker_count=len(reference_speakers),
        prediction_speaker_count=len(prediction_speakers),
        speaker_count_error=count_error,
        speaker_count_absolute_error=abs(count_error),
        speaker_mapping=speaker_mapping,
        jer_speaker_mapping=jer_mapping,
        per_reference_speaker_jer=per_speaker_jer,
    )


def evaluate(
    reference: dict[str, list[Segment]],
    prediction: dict[str, list[Segment]],
) -> EvaluationResult:
    # A single JSON transcript commonly has a different filename from its gold
    # RTTM.  There is no ambiguity in this case, so align the sole recordings.
    if len(reference) == 1 and len(prediction) == 1:
        ref_uri = next(iter(reference))
        pred_uri = next(iter(prediction))
        if ref_uri != pred_uri:
            prediction = {ref_uri: next(iter(prediction.values()))}
    uris = sorted(set(reference) | set(prediction))
    return EvaluationResult([
        evaluate_recording(uri, reference.get(uri, []), prediction.get(uri, []))
        for uri in uris
    ])


def evaluate_files(gold: Path, prediction: Path) -> EvaluationResult:
    return evaluate(load_annotations(gold), load_annotations(prediction))


def _round(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def _percent(value: float | None) -> str:
    return "无参考语音，无法计算" if value is None else f"{value:.2%}"


def chinese_report(
    result: EvaluationResult,
    gold_path: Path,
    prediction_path: Path,
) -> dict[str, Any]:
    aggregate = result.aggregate()
    return {
        "状态": "成功",
        "指标口径": {
            "DER": "(漏检 + 误检 + 说话人混淆) / 人工真值说话人总时长",
            "JER": "每位真值说话人与其最优匹配预测说话人的 Jaccard 错误率均值",
            "重叠语音": "保留并按同时活跃的说话人数累计说话人时长",
            "映射": "每条录音独立执行全局一对一最优说话人映射",
        },
        "输入": {
            "人工真值": str(gold_path),
            "预测结果": str(prediction_path),
        },
        "汇总": {
            "录音数": aggregate["recording_count"],
            "人工真值说话人总时长秒": _round(aggregate["reference_speaker_time_s"]),
            "DER": _round(aggregate["der"]),
            "DER百分比": _percent(aggregate["der"]),
            "漏检秒": _round(aggregate["miss_s"]),
            "漏检率": _round(aggregate["miss_rate"]),
            "误检秒": _round(aggregate["false_alarm_s"]),
            "误检率": _round(aggregate["false_alarm_rate"]),
            "说话人混淆秒": _round(aggregate["confusion_s"]),
            "说话人混淆率": _round(aggregate["confusion_rate"]),
            "JER": _round(aggregate["jer"]),
            "JER百分比": _percent(aggregate["jer"]),
            "真值说话人数合计": aggregate["reference_speaker_count"],
            "预测说话人数合计": aggregate["prediction_speaker_count"],
            "说话人数有符号误差合计": aggregate["speaker_count_error"],
            "说话人数绝对误差合计": aggregate["speaker_count_absolute_error"],
        },
        "逐录音": [
            {
                "录音": item.uri,
                "人工真值说话人时长秒": _round(item.reference_speaker_time_s),
                "DER": _round(item.der),
                "DER百分比": _percent(item.der),
                "漏检秒": _round(item.miss_s),
                "误检秒": _round(item.false_alarm_s),
                "说话人混淆秒": _round(item.confusion_s),
                "JER": _round(item.jer),
                "JER百分比": _percent(item.jer),
                "真值说话人数": item.reference_speaker_count,
                "预测说话人数": item.prediction_speaker_count,
                "说话人数误差": item.speaker_count_error,
                "说话人数绝对误差": item.speaker_count_absolute_error,
                "DER最优说话人映射_预测到真值": item.speaker_mapping,
                "JER最优说话人映射_预测到真值": item.jer_speaker_mapping,
                "逐真值说话人JER": {
                    speaker: _round(value)
                    for speaker, value in item.per_reference_speaker_jer.items()
                },
            }
            for item in result.recordings
        ],
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["汇总"]
    lines = [
        "# 说话人分离真实评测报告",
        "",
        f"- 人工真值: `{report['输入']['人工真值']}`",
        f"- 预测结果: `{report['输入']['预测结果']}`",
        f"- DER: **{summary['DER百分比']}**",
        f"- JER: **{summary['JER百分比']}**",
        f"- DER 分解: 漏检 {summary['漏检秒']:.3f}s / 误检 {summary['误检秒']:.3f}s / 说话人混淆 {summary['说话人混淆秒']:.3f}s",
        f"- 说话人数绝对误差合计: {summary['说话人数绝对误差合计']}",
        "",
        "> DER 以人工真值说话人总时长为分母；重叠语音按同时活跃人数累计。说话人名称不同不直接算错，先执行全局最优一对一映射。",
        "",
        "| 录音 | DER | 漏检秒 | 误检秒 | 混淆秒 | JER | 真值人数 | 预测人数 | 人数误差 | DER 最优映射（预测→真值） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["逐录音"]:
        mapping = " / ".join(
            f"{predicted}→{expected}"
            for predicted, expected in sorted(item["DER最优说话人映射_预测到真值"].items())
        ) or "无"
        lines.append(
            f"| {str(item['录音']).replace('|', '/')} | {item['DER百分比']} | "
            f"{item['漏检秒']:.3f} | {item['误检秒']:.3f} | {item['说话人混淆秒']:.3f} | "
            f"{item['JER百分比']} | {item['真值说话人数']} | {item['预测说话人数']} | "
            f"{item['说话人数误差']:+d} | {mapping.replace('|', '/')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="计算真实说话人分离 DER/JER 指标")
    parser.add_argument("--gold", required=True, type=Path, help="人工真值 RTTM 或 JSON")
    parser.add_argument(
        "--prediction", "--pred", required=True, type=Path, help="预测结果 RTTM 或 JSON"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("diarization_metrics_report"), help="报告输出目录"
    )
    args = parser.parse_args()

    gold_path = args.gold.expanduser().resolve()
    prediction_path = args.prediction.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    result = evaluate_files(gold_path, prediction_path)
    report = chinese_report(result, gold_path, prediction_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "diarization_metrics.json"
    markdown_path = out_dir / "diarization_metrics.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(markdown_path, report)
    aggregate = result.aggregate()
    print(json.dumps({
        "ok": True,
        "json": str(json_path),
        "markdown": str(markdown_path),
        "DER": _round(aggregate["der"]),
        "JER": _round(aggregate["jer"]),
        "speaker_count_absolute_error": aggregate["speaker_count_absolute_error"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
