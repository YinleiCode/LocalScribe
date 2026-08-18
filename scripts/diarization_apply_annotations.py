#!/usr/bin/env python3
"""Apply sparse human speaker annotations to a transcript copy.

The script never overwrites the source transcript.  It is a lightweight bridge
between manual review and a future in-app calibration workflow: sparse rows are
treated as trusted anchor labels, written into a calibrated copy, and scored
against the original output.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


VALID_LABEL_RE = re.compile(r"^(?:SPEAKER_)?([A-Z])$")


def _speaker_name(label: str) -> str:
    label = str(label or "").strip()
    match = VALID_LABEL_RE.match(label)
    if not match:
        return ""
    return f"SPEAKER_{match.group(1)}"


def _read_annotations(path: Path) -> dict[int, str]:
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        out: dict[int, str] = {}
        for row in rows:
            try:
                idx = int(row.get("序号", row.get("index")))
            except (TypeError, ValueError):
                continue
            label = _speaker_name(row.get("你的标注") or row.get("correct_speaker") or row.get("label") or "")
            if label:
                out[idx] = label
        return out

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        out: dict[int, str] = {}
        for row in rows:
            try:
                idx = int(row.get("序号") or row.get("index") or "")
            except (TypeError, ValueError):
                continue
            label = _speaker_name(
                row.get("你的标注")
                or row.get("正确speaker(只填这里)")
                or row.get("correct_speaker")
                or ""
            )
            if label:
                out[idx] = label
        return out

    return _read_markdown_annotations(path)


def _read_markdown_annotations(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or line.startswith("|---") or "序号" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        label = ""
        # Preferred: explicit correct-speaker column.
        if len(parts) >= 5:
            label = _speaker_name(parts[3])
        # Tolerate accidental edits in the current-speaker column, as happened
        # during manual review.  This keeps the workflow forgiving.
        if not label:
            label = _speaker_name(parts[2])
        if label:
            out[idx] = label
    return out


def _fmt_ts(seconds: float) -> str:
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes:02d}:{sec:05.2f}"


def _apply_annotations(transcript: dict[str, Any], annotations: dict[int, str]) -> dict[str, Any]:
    calibrated = json.loads(json.dumps(transcript, ensure_ascii=False))
    segments = calibrated.get("segments") or []
    changed = 0
    matched = 0
    rows = []
    for idx, correct_speaker in sorted(annotations.items()):
        if idx < 0 or idx >= len(segments):
            rows.append({
                "index": idx,
                "status": "out_of_range",
                "correct_speaker": correct_speaker,
            })
            continue
        seg = segments[idx]
        original_speaker = str(seg.get("speaker") or "")
        if original_speaker == correct_speaker:
            matched += 1
        else:
            changed += 1
            seg["original_speaker"] = original_speaker
            seg["speaker"] = correct_speaker
            seg["speaker_calibrated"] = True
            seg["speaker_calibration_source"] = "human_annotation"
        rows.append({
            "index": idx,
            "start": float(seg.get("start") or 0.0),
            "end": float(seg.get("end") or 0.0),
            "original_speaker": original_speaker,
            "correct_speaker": correct_speaker,
            "changed": original_speaker != correct_speaker,
            "text": str(seg.get("text") or ""),
        })
    stats = dict(calibrated.get("diarization_stats") or {})
    stats["human_annotation_count"] = len(annotations)
    stats["human_annotation_matched"] = matched
    stats["human_annotation_changed"] = changed
    stats["human_annotation_accuracy_before"] = (
        round(matched / len(annotations), 4)
        if annotations else None
    )
    stats["human_calibration_mode"] = "direct_sparse_anchor"
    calibrated["diarization_stats"] = stats
    calibrated["speaker_calibration"] = {
        "mode": "direct_sparse_anchor",
        "annotation_count": len(annotations),
        "matched_before": matched,
        "changed": changed,
        "accuracy_before": stats["human_annotation_accuracy_before"],
        "rows": rows,
    }
    return calibrated


def _write_txt(path: Path, data: dict[str, Any]) -> None:
    lines = []
    stats = data.get("speaker_calibration") or {}
    lines.append(f"# speaker calibration: {stats.get('mode', '')}")
    lines.append(f"# annotations={stats.get('annotation_count', 0)} changed={stats.get('changed', 0)}")
    lines.append("")
    for seg in data.get("segments") or []:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        speaker = str(seg.get("speaker") or "")
        marker = " *校准" if seg.get("speaker_calibrated") else ""
        lines.append(
            f"[{_fmt_ts(float(seg.get('start') or 0.0))} - {_fmt_ts(float(seg.get('end') or 0.0))}] "
            f"{speaker}{marker}: {text}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(path: Path, data: dict[str, Any]) -> None:
    cal = data.get("speaker_calibration") or {}
    rows = cal.get("rows") or []
    lines = [
        "# 说话人标注校准报告",
        "",
        f"- 标注数: {cal.get('annotation_count', 0)}",
        f"- 原始命中: {cal.get('matched_before', 0)}",
        f"- 已校准改动: {cal.get('changed', 0)}",
        f"- 校准前准确率: {cal.get('accuracy_before')}",
        "",
        "| 序号 | 时间 | 原speaker | 标注speaker | 是否改动 | 文本 |",
        "|---:|---|---|---|---|---|",
    ]
    for row in rows:
        if row.get("status") == "out_of_range":
            lines.append(f"| {row.get('index')} | 越界 |  | {row.get('correct_speaker')} | 否 |  |")
            continue
        text = str(row.get("text") or "").replace("|", "/")
        time_range = f"{_fmt_ts(float(row.get('start') or 0.0))} - {_fmt_ts(float(row.get('end') or 0.0))}"
        lines.append(
            f"| {row.get('index')} | {time_range} | {row.get('original_speaker')} | "
            f"{row.get('correct_speaker')} | {'是' if row.get('changed') else '否'} | {text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    transcript_path = args.transcript.expanduser().resolve()
    annotation_path = args.annotations.expanduser().resolve()
    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir else transcript_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    annotations = _read_annotations(annotation_path)
    calibrated = _apply_annotations(transcript, annotations)

    stem = transcript_path.stem
    json_path = out_dir / f"{stem}_speaker_calibrated.json"
    txt_path = out_dir / f"{stem}_speaker_calibrated.txt"
    report_path = out_dir / f"{stem}_speaker_calibration_report.md"
    json_path.write_text(json.dumps(calibrated, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_txt(txt_path, calibrated)
    _write_report(report_path, calibrated)

    print(json.dumps({
        "ok": True,
        "json": str(json_path),
        "txt": str(txt_path),
        "report": str(report_path),
        "annotation_count": len(annotations),
        "changed": calibrated["speaker_calibration"]["changed"],
        "accuracy_before": calibrated["speaker_calibration"]["accuracy_before"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
