#!/usr/bin/env python3
"""Score exported continuous blind annotations against hidden predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from diarization_metrics import (  # noqa: E402
    Segment,
    _active_sets,
    chinese_report,
    evaluate,
    load_annotations,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_inputs(
    gold_path: Path,
    prediction_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    gold = _read_json(gold_path)
    prediction = _read_json(prediction_path)
    manifest = _read_json(manifest_path)
    pack_ids = {str(item.get("pack_id") or "") for item in (gold, prediction, manifest)}
    if len(pack_ids) != 1 or not next(iter(pack_ids)):
        raise ValueError("人工真值、隐藏预测与清单的 pack_id 不一致")
    if gold.get("kind") != "continuous_diarization_gold":
        raise ValueError("不是连续盲标页面导出的人工真值")
    expected_ids = {str(item.get("id") or "") for item in manifest.get("items") or []}
    completed_ids = {
        str(item.get("id") or "")
        for item in gold.get("items") or []
        if isinstance(item, dict) and item.get("complete") and item.get("markers")
    }
    if expected_ids != completed_ids:
        missing = sorted(expected_ids - completed_ids)
        raise ValueError(f"仍有连续盲标未完成: {missing}")
    if not gold.get("segments"):
        raise ValueError("人工真值没有可评分的说话人片段")
    return gold, prediction, manifest


def overlap_metrics(reference, prediction, recording_metrics) -> dict[str, Any]:
    by_uri = {item.uri: item for item in recording_metrics}
    overlap_wallclock = 0.0
    reference_speaker_time = 0.0
    miss = 0.0
    false_alarm = 0.0
    confusion = 0.0
    recordings_with_overlap: set[str] = set()
    for uri in sorted(set(reference) | set(prediction)):
        mapping = by_uri[uri].speaker_mapping
        for duration, ref_active, pred_active in _active_sets(
            reference.get(uri, []), prediction.get(uri, [])
        ):
            if len(ref_active) < 2:
                continue
            recordings_with_overlap.add(uri)
            overlap_wallclock += duration
            reference_speaker_time += len(ref_active) * duration
            correct = sum(
                1 for predicted in pred_active if mapping.get(predicted) in ref_active
            )
            miss += max(0, len(ref_active) - len(pred_active)) * duration
            false_alarm += max(0, len(pred_active) - len(ref_active)) * duration
            confusion += (min(len(ref_active), len(pred_active)) - correct) * duration
    error = miss + false_alarm + confusion
    return {
        "含重叠录音数": len(recordings_with_overlap),
        "人工真值重叠时长秒": round(overlap_wallclock, 6),
        "人工真值重叠说话人总时长秒": round(reference_speaker_time, 6),
        "重叠漏检秒": round(miss, 6),
        "重叠误检秒": round(false_alarm, 6),
        "重叠说话人混淆秒": round(confusion, 6),
        "重叠总错误秒": round(error, 6),
        "重叠语音错误率": round(error / reference_speaker_time, 6)
        if reference_speaker_time > 0
        else None,
    }


def annotation_coverage(gold: dict[str, Any]) -> tuple[dict[str, tuple[float, float]], float]:
    coverage: dict[str, tuple[float, float]] = {}
    omitted_leading_seconds = 0.0
    for item in gold.get("items") or []:
        if not isinstance(item, dict):
            continue
        markers = [row for row in item.get("markers") or [] if isinstance(row, dict)]
        if not markers:
            continue
        window_start = float(item.get("window_start") or 0.0)
        window_end = float(item.get("window_end") or 0.0)
        first_marker = min(float(row.get("time") or 0.0) for row in markers)
        start = max(window_start, first_marker)
        if window_end > start:
            uri = str(item.get("uri") or "").strip()
            if uri:
                coverage[uri] = (start, window_end)
                omitted_leading_seconds += max(0.0, start - window_start)
    return coverage, omitted_leading_seconds


def crop_recordings(recordings, coverage: dict[str, tuple[float, float]]):
    cropped: dict[str, list[Segment]] = {}
    for uri, segments in recordings.items():
        if uri not in coverage:
            continue
        coverage_start, coverage_end = coverage[uri]
        rows: list[Segment] = []
        for segment in segments:
            start = max(segment.start, coverage_start)
            end = min(segment.end, coverage_end)
            if end > start:
                rows.append(Segment(uri, start, end, segment.speaker))
        cropped[uri] = rows
    return cropped


def _percent(value: Any) -> str:
    return "无人工重叠真值" if value is None else f"{float(value):.2%}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["汇总"]
    overlap = report["重叠语音"]
    lines = [
        "# 连续说话人分离 DER/JER 验收报告",
        "",
        f"- 连续验收窗口: **{report['连续验收时长分钟']:.1f} 分钟**",
        f"- 有效人工覆盖: **{report['有效人工覆盖分钟']:.1f} 分钟**",
        f"- DER: **{summary['DER百分比']}**",
        f"- JER: **{summary['JER百分比']}**",
        f"- 重叠语音错误率: **{_percent(overlap['重叠语音错误率'])}**",
        f"- 说话人数绝对误差合计: **{summary['说话人数绝对误差合计']}**",
        "",
        "> 本报告只评价冻结 ASR 语音区间上的说话人归属；文字、段落时间和播放光标不参与修改。",
        "",
        "| 录音 | DER | JER | 真值人数 | 预测人数 | 人数误差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["逐录音"]:
        lines.append(
            f"| {str(item['录音']).replace('|', '/')} | {item['DER百分比']} | "
            f"{item['JER百分比']} | {item['真值说话人数']} | "
            f"{item['预测说话人数']} | {item['说话人数误差']:+d} |"
        )
    lines.extend([
        "",
        "## 重叠语音",
        "",
        f"- 人工真值重叠时长: {overlap['人工真值重叠时长秒']:.3f}s",
        f"- 重叠漏检: {overlap['重叠漏检秒']:.3f}s",
        f"- 重叠误检: {overlap['重叠误检秒']:.3f}s",
        f"- 重叠混淆: {overlap['重叠说话人混淆秒']:.3f}s",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_pack(gold_path: Path, pack_dir: Path, out_dir: Path) -> dict[str, Any]:
    gold_path = gold_path.expanduser().resolve()
    pack_dir = pack_dir.expanduser().resolve()
    prediction_path = pack_dir / "scoring" / "当前通用分人预测.json"
    manifest_path = pack_dir / "连续分人盲标清单.json"
    gold, _, manifest = validate_inputs(gold_path, prediction_path, manifest_path)
    reference = load_annotations(gold_path)
    prediction = load_annotations(prediction_path)
    coverage, omitted_leading_seconds = annotation_coverage(gold)
    reference = crop_recordings(reference, coverage)
    prediction = crop_recordings(prediction, coverage)
    result = evaluate(reference, prediction)
    report = chinese_report(result, gold_path, prediction_path)
    report["评测类型"] = "冻结ASR语音区间上的连续说话人归属"
    report["pack_id"] = manifest["pack_id"]
    report["连续验收时长分钟"] = round(sum(
        float(item["window_end"]) - float(item["window_start"])
        for item in manifest.get("items") or []
    ) / 60.0, 3)
    report["未标注开头秒"] = round(omitted_leading_seconds, 3)
    report["有效人工覆盖分钟"] = round(
        sum(end - start for start, end in coverage.values()) / 60.0,
        3,
    )
    report["重叠语音"] = overlap_metrics(reference, prediction, result.recordings)
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "连续分人DER_JER报告.json"
    markdown_path = out_dir / "连续分人DER_JER报告.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(markdown_path, report)
    return {
        "ok": True,
        "json": str(json_path),
        "markdown": str(markdown_path),
        "DER": report["汇总"]["DER"],
        "JER": report["汇总"]["JER"],
        "overlap_error_rate": report["重叠语音"]["重叠语音错误率"],
        "speaker_count_absolute_error": report["汇总"]["说话人数绝对误差合计"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="计算连续盲标 DER/JER 与重叠错误率")
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--pack-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(score_pack(args.gold, args.pack_dir, args.out), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
