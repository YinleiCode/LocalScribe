#!/usr/bin/env python3
"""Score complete App transcripts against source-timed human gold clips.

The direct gold workflow transcribes short clips in isolation. This scorer
checks whether the same words survived a complete App run. It extracts a small
timeline neighborhood from the full result and uses the best local alignment
so harmless segment/cue boundary differences do not inflate CER.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from asr_gold_score import edit_distance, normalize_for_cer  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def audio_key(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return str(Path(raw).expanduser().resolve())
    except OSError:
        return raw


def result_index(paths: list[Path]) -> dict[str, tuple[Path, dict[str, Any]]]:
    indexed: dict[str, tuple[Path, dict[str, Any]]] = {}
    basenames: dict[str, list[str]] = {}
    for path in paths:
        data = read_json(path)
        key = audio_key(data.get("audio"))
        if not key:
            raise ValueError(f"full result has no audio field: {path}")
        if key in indexed:
            raise ValueError(f"duplicate full result for audio: {key}")
        indexed[key] = (path, data)
        basenames.setdefault(Path(key).name, []).append(key)
    for basename, keys in basenames.items():
        if len(keys) == 1:
            indexed[f"basename:{basename}"] = indexed[keys[0]]
    return indexed


def find_result(
    item: dict[str, Any],
    indexed: dict[str, tuple[Path, dict[str, Any]]],
) -> tuple[Path, dict[str, Any]] | None:
    key = audio_key(item.get("audio"))
    if key and key in indexed:
        return indexed[key]
    if key:
        return indexed.get(f"basename:{Path(key).name}")
    return None


def _time_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def extract_timeline_text(
    result: dict[str, Any],
    *,
    start: float,
    end: float,
    pad_seconds: float = 4.0,
) -> tuple[str, dict[str, Any]]:
    """Extract ordered cue text near a source-time gold window."""
    lower = max(0.0, start - max(0.0, pad_seconds))
    upper = end + max(0.0, pad_seconds)
    parts: list[str] = []
    cue_parts = 0
    segment_parts = 0
    for segment in result.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        seg_start = _time_value(segment.get("start"))
        seg_end = _time_value(segment.get("end"))
        if seg_end <= lower or seg_start >= upper:
            continue
        selected_cues: list[str] = []
        for cue in segment.get("sync_cues") or []:
            if not isinstance(cue, dict):
                continue
            cue_start = _time_value(cue.get("start"))
            cue_end = _time_value(cue.get("end"))
            if cue_end > lower and cue_start < upper:
                text = str(cue.get("text") or "")
                if text:
                    selected_cues.append(text)
        if selected_cues:
            parts.extend(selected_cues)
            cue_parts += len(selected_cues)
            continue
        text = str(segment.get("text") or "")
        if text:
            parts.append(text)
            segment_parts += 1
    return "".join(parts), {
        "window_start": round(lower, 3),
        "window_end": round(upper, 3),
        "cue_parts": cue_parts,
        "segment_parts": segment_parts,
    }


def best_local_alignment(reference: str, candidate: str) -> dict[str, Any]:
    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(candidate)
    if not ref:
        return {
            "reference": ref,
            "candidate": hyp,
            "matched": "",
            "start": 0,
            "end": 0,
            "edit_distance": 0 if not hyp else len(hyp),
            "cer": None,
        }
    if not hyp:
        return {
            "reference": ref,
            "candidate": hyp,
            "matched": "",
            "start": 0,
            "end": 0,
            "edit_distance": len(ref),
            "cer": 1.0,
        }

    allowance = max(8, int(round(len(ref) * 0.4)))
    min_length = max(1, len(ref) - allowance)
    max_length = min(len(hyp), len(ref) + allowance)
    if len(hyp) < min_length:
        lengths = [len(hyp)]
    else:
        lengths = range(min_length, max_length + 1)

    best: tuple[int, int, int, int] | None = None
    for length in lengths:
        for start_index in range(0, len(hyp) - length + 1):
            end_index = start_index + length
            edits = edit_distance(ref, hyp[start_index:end_index])
            rank = (edits, abs(length - len(ref)), start_index, end_index)
            if best is None or rank < best:
                best = rank
    if best is None:
        best = (edit_distance(ref, hyp), abs(len(hyp) - len(ref)), 0, len(hyp))
    edits, _length_delta, start_index, end_index = best
    return {
        "reference": ref,
        "candidate": hyp,
        "matched": hyp[start_index:end_index],
        "start": start_index,
        "end": end_index,
        "edit_distance": edits,
        "cer": edits / len(ref),
    }


def score_full_results(
    gold: dict[str, Any],
    indexed: dict[str, tuple[Path, dict[str, Any]]],
    *,
    pad_seconds: float = 4.0,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_ref = 0
    total_edits = 0
    missing_results = 0
    for item in gold.get("items") or []:
        if not isinstance(item, dict):
            continue
        reference = str(item.get("correct_text") or "").strip()
        if not reference or str(item.get("decision") or "") == "unusable":
            continue
        match = find_result(item, indexed)
        if match is None:
            missing_results += 1
            rows.append({
                "id": str(item.get("id") or ""),
                "case_id": str(item.get("case_id") or ""),
                "status": "missing_full_result",
            })
            continue
        result_path, result = match
        start = _time_value(item.get("start"))
        end = start + max(0.0, _time_value(item.get("duration")))
        timeline_text, extraction = extract_timeline_text(
            result,
            start=start,
            end=end,
            pad_seconds=pad_seconds,
        )
        alignment = best_local_alignment(reference, timeline_text)
        total_ref += len(alignment["reference"])
        total_edits += int(alignment["edit_distance"])
        rows.append({
            "id": str(item.get("id") or ""),
            "case_id": str(item.get("case_id") or ""),
            "status": "scored",
            "start": start,
            "end": end,
            "result": str(result_path),
            "reference_text": reference,
            "timeline_text": timeline_text,
            "matched_text_normalized": alignment["matched"],
            "reference_normalized": alignment["reference"],
            "edit_distance": alignment["edit_distance"],
            "reference_chars": len(alignment["reference"]),
            "cer": alignment["cer"],
            "extraction": extraction,
        })
    return {
        "mode": "full_app_timeline_local_alignment",
        "pad_seconds": pad_seconds,
        "scored_rows": sum(row.get("status") == "scored" for row in rows),
        "missing_full_results": missing_results,
        "total_reference_chars": total_ref,
        "total_edit_distance": total_edits,
        "overall_cer": total_edits / total_ref if total_ref else None,
        "rows": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    cer = report.get("overall_cer")
    cer_text = "-" if cer is None else f"{float(cer) * 100:.2f}%"
    lines = [
        "# 完整 App 转录对人工真值评分\n\n",
        f"- 已评分片段: {report['scored_rows']}\n",
        f"- 缺少完整结果: {report['missing_full_results']}\n",
        f"- 参考字数: {report['total_reference_chars']}\n",
        f"- 编辑距离: {report['total_edit_distance']}\n",
        f"- micro-CER: {cer_text}\n",
        f"- 时间窗前后容差: {report['pad_seconds']:.1f}s\n\n",
        "| ID | 状态 | CER | 编辑距离 | 人工标准答案 | 完整 App 最佳局部匹配 |\n",
        "|---|---|---:|---:|---|---|\n",
    ]
    for row in report.get("rows") or []:
        row_cer = row.get("cer")
        row_cer_text = "-" if row_cer is None else f"{float(row_cer) * 100:.2f}%"
        reference = str(row.get("reference_text") or "").replace("\n", "<br>").replace("|", "/")
        matched = str(row.get("matched_text_normalized") or "").replace("|", "/")
        lines.append(
            f"| {row.get('id', '')} | {row.get('status', '')} | {row_cer_text} | "
            f"{row.get('edit_distance', '-')} | {reference} | {matched} |\n"
        )
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评分完整 App 转录在人工真值时间窗内的局部 CER")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pad-seconds", type=float, default=4.0)
    args = parser.parse_args(argv)

    gold_path = args.gold.expanduser().resolve()
    result_paths = [path.expanduser().resolve() for path in args.result]
    report = score_full_results(
        read_json(gold_path),
        result_index(result_paths),
        pad_seconds=max(0.0, args.pad_seconds),
    )
    report["gold"] = str(gold_path)
    report["results"] = [str(path) for path in result_paths]
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "完整App人工真值评分.json"
    markdown_path = out_dir / "完整App人工真值评分.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "ok": report["scored_rows"] > 0,
        "scored_rows": report["scored_rows"],
        "missing_full_results": report["missing_full_results"],
        "overall_cer": report["overall_cer"],
        "json": str(json_path),
        "markdown": str(markdown_path),
    }, ensure_ascii=False))
    return 0 if report["scored_rows"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
