#!/usr/bin/env python3
"""Score a LocalScribe diarization review pack without mutating source data."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERDICT_LABELS = {
    "": "待标注",
    "correct": "正确",
    "wrong_speaker": "人员错",
    "wrong_boundary": "切点错",
    "missed_split": "漏拆",
    "false_split": "误拆",
    "uncertain": "不确定",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def normalize_sequence(value: Any) -> str:
    text = str(value or "").upper().replace("SPEAKER_", "")
    compact = re.sub(r"\s+", "", text)
    if re.fullmatch(r"[A-H]{2,}", compact):
        tokens = list(compact)
    else:
        tokens = re.findall(r"[A-Z][A-Z0-9]*", text)
    return "->".join(tokens)


def _timeline_prediction(source: dict[str, Any]) -> str:
    tokens: list[str] = []
    for row in source.get("timeline") or []:
        if not isinstance(row, dict) or row.get("context"):
            continue
        normalized = normalize_sequence(row.get("speaker"))
        if normalized:
            tokens.extend(normalized.split("->"))
    return "->".join(tokens)


def _parse_override(value: str) -> tuple[str, dict[str, str]]:
    item_id, separator, payload = value.partition("=")
    if not separator or not item_id.strip() or not payload.strip():
        raise argparse.ArgumentTypeError("override must be ID=VERDICT[:SEQUENCE]")
    verdict, _, sequence = payload.partition(":")
    verdict = verdict.strip()
    if verdict not in VERDICT_LABELS:
        raise argparse.ArgumentTypeError(f"unsupported verdict: {verdict}")
    return item_id.strip(), {
        "verdict": verdict,
        "correct_speaker_sequence": sequence.strip(),
    }


def _parse_prediction(value: str) -> tuple[str, str]:
    item_id, separator, sequence = value.partition("=")
    normalized = normalize_sequence(sequence)
    if not separator or not item_id.strip() or not normalized:
        raise argparse.ArgumentTypeError("prediction must be ID=SEQUENCE")
    return item_id.strip(), normalized


def build_rows(
    manifest: dict[str, Any],
    annotations: dict[str, Any],
    *,
    overrides: dict[str, dict[str, str]] | None = None,
    predictions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if manifest.get("pack_id") != annotations.get("pack_id"):
        raise ValueError("annotation pack_id does not match manifest")
    manifest_items = {
        str(item.get("id") or ""): item
        for item in manifest.get("items") or []
        if isinstance(item, dict) and item.get("id")
    }
    annotation_items = {
        str(item.get("id") or ""): item
        for item in annotations.get("items") or []
        if isinstance(item, dict) and item.get("id")
    }
    if set(manifest_items) != set(annotation_items):
        missing = sorted(set(manifest_items) - set(annotation_items))
        extra = sorted(set(annotation_items) - set(manifest_items))
        raise ValueError(f"annotation item IDs do not match manifest: missing={missing}, extra={extra}")

    rows: list[dict[str, Any]] = []
    for item_id, source in manifest_items.items():
        annotation = dict(annotation_items[item_id])
        override = (overrides or {}).get(item_id)
        if override:
            annotation.update(override)
        verdict = str(annotation.get("verdict") or "")
        if verdict not in VERDICT_LABELS:
            raise ValueError(f"unsupported verdict for {item_id}: {verdict}")
        baseline_prediction = normalize_sequence(source.get("current_prediction"))
        baseline_detailed_prediction = _timeline_prediction(source) or baseline_prediction
        prediction_override = (predictions or {}).get(item_id)
        prediction = normalize_sequence(prediction_override) if prediction_override else baseline_detailed_prediction
        provided_gold = normalize_sequence(annotation.get("correct_speaker_sequence"))
        if verdict == "correct":
            gold = baseline_detailed_prediction
        elif verdict in {"", "uncertain"}:
            gold = ""
        else:
            gold = provided_gold
        scorable = bool(gold)
        rows.append({
            "id": item_id,
            "recording": str(source.get("recording") or ""),
            "category": str(source.get("category") or ""),
            "review_start": source.get("review_start"),
            "review_end": source.get("review_end"),
            "baseline_prediction": baseline_prediction,
            "baseline_detailed_prediction": baseline_detailed_prediction,
            "prediction": prediction,
            "verdict": verdict,
            "verdict_label": VERDICT_LABELS[verdict],
            "gold_sequence": gold,
            "scorable": scorable,
            "prediction_correct": bool(scorable and prediction == gold),
            "notes": str(annotation.get("notes") or ""),
            "override_applied": bool(override),
            "prediction_override_applied": bool(prediction_override),
        })
    return rows


def _group_metrics(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "")].append(row)
    output = []
    for label, items in sorted(groups.items()):
        scorable = [item for item in items if item["scorable"]]
        correct = sum(item["prediction_correct"] for item in scorable)
        output.append({
            key: label,
            "items": len(items),
            "scorable": len(scorable),
            "correct": correct,
            "errors": len(scorable) - correct,
            "accuracy": round(correct / len(scorable), 4) if scorable else None,
        })
    return output


def _group_strict_metrics(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "")].append(row)
    output = []
    for label, items in sorted(groups.items()):
        scorable = [item for item in items if item["strict_scorable"]]
        correct = sum(item["strict_correct"] for item in scorable)
        output.append({
            key: label,
            "items": len(items),
            "scorable": len(scorable),
            "correct": correct,
            "errors": len(scorable) - correct,
            "accuracy": round(correct / len(scorable), 4) if scorable else None,
        })
    return output


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_mode = any(row.get("prediction_override_applied") for row in rows)
    for row in rows:
        if candidate_mode:
            row["strict_scorable"] = bool(row["scorable"])
            row["strict_correct"] = bool(row["scorable"] and row["prediction_correct"])
        else:
            row["strict_scorable"] = row["verdict"] not in {"", "uncertain"}
            row["strict_correct"] = row["verdict"] == "correct"

    scorable = [row for row in rows if row["scorable"]]
    correct = sum(row["prediction_correct"] for row in scorable)
    strict_scorable = [row for row in rows if row["strict_scorable"]]
    strict_correct = sum(row["strict_correct"] for row in strict_scorable)
    completed = sum(bool(row["verdict"]) for row in rows)
    verdict_counts = Counter(row["verdict"] for row in rows)
    return {
        "items": len(rows),
        "completed": completed,
        "completion_rate": round(completed / len(rows), 4) if rows else None,
        "scorable": len(scorable),
        "correct": correct,
        "errors": len(scorable) - correct,
        "accuracy": round(correct / len(scorable), 4) if scorable else None,
        "strict_scorable": len(strict_scorable),
        "strict_correct": strict_correct,
        "strict_errors": len(strict_scorable) - strict_correct,
        "strict_accuracy": round(strict_correct / len(strict_scorable), 4) if strict_scorable else None,
        "verdict_counts": {
            VERDICT_LABELS[key]: count
            for key, count in sorted(verdict_counts.items())
        },
        "by_recording": _group_metrics(rows, "recording"),
        "by_category": _group_metrics(rows, "category"),
        "strict_by_recording": _group_strict_metrics(rows, "recording"),
        "strict_by_category": _group_strict_metrics(rows, "category"),
        "error_items": [row for row in rows if row["scorable"] and not row["prediction_correct"]],
        "unscored_items": [row for row in rows if not row["scorable"]],
        "strict_error_items": [row for row in rows if row["strict_scorable"] and not row["strict_correct"]],
        "strict_unscored_items": [row for row in rows if not row["strict_scorable"]],
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    accuracy = metrics.get("strict_accuracy")
    accuracy_text = f"{accuracy * 100:.1f}%" if accuracy is not None else "--"
    lines = [
        "# 通用分人人工验收报告\n\n",
        f"- 样本：{metrics['items']}\n",
        f"- 已完成：{metrics['completed']}\n",
        f"- 严格可计分：{metrics['strict_scorable']}\n",
        f"- 人工确认正确：{metrics['strict_correct']}\n",
        f"- 人工确认错误：{metrics['strict_errors']}\n",
        f"- 严格通过率：{accuracy_text}\n\n",
        "严格通过率会把漏拆、误拆、人员错和切点错全部计为失败；待标注和不确定项目不进入分母。\n\n",
        "## 按录音\n\n",
        "| 录音 | 样本 | 严格可计分 | 正确 | 错误 | 通过率 |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in metrics["strict_by_recording"]:
        value = row.get("accuracy")
        rendered = f"{value * 100:.1f}%" if value is not None else "--"
        lines.append(
            f"| {row['recording']} | {row['items']} | {row['scorable']} | {row['correct']} | {row['errors']} | {rendered} |\n"
        )
    lines.extend([
        "\n## 按场景\n\n",
        "| 场景 | 样本 | 严格可计分 | 正确 | 错误 | 通过率 |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ])
    for row in metrics["strict_by_category"]:
        value = row.get("accuracy")
        rendered = f"{value * 100:.1f}%" if value is not None else "--"
        lines.append(
            f"| {row['category']} | {row['items']} | {row['scorable']} | {row['correct']} | {row['errors']} | {rendered} |\n"
        )
    lines.extend([
        "\n## 错误样本\n\n",
        "| ID | 录音 | 场景 | 当前 | 人工真值 | 类型 | 备注 |\n",
        "|---|---|---|---|---|---|---|\n",
    ])
    for row in metrics["strict_error_items"]:
        note = str(row.get("notes") or "").replace("|", "\\|")
        lines.append(
            f"| {row['id']} | {row['recording']} | {row['category']} | {row['prediction']} | {row['gold_sequence'] or '未填写完整序列'} | {row['verdict_label']} | {note} |\n"
        )
    lines.extend([
        "\n## 精确序列补充指标\n\n",
        f"- 有完整人工序列：{metrics['scorable']}\n",
        f"- 序列完全正确：{metrics['correct']}\n",
        f"- 序列错误：{metrics['errors']}\n",
        f"- 精确序列准确率：{metrics['accuracy'] * 100:.1f}%\n" if metrics["accuracy"] is not None else "- 精确序列准确率：--\n",
    ])
    if metrics["strict_unscored_items"]:
        lines.extend(["\n## 不计分\n\n"])
        for row in metrics["strict_unscored_items"]:
            lines.append(f"- `{row['id']}` {row['recording']}：{row['verdict_label']}；{row['notes']}\n")
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="计算通用分人人工验收结果")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--override", action="append", type=_parse_override, default=[])
    parser.add_argument("--prediction", action="append", type=_parse_prediction, default=[])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    overrides = dict(args.override)
    predictions = dict(args.prediction)
    rows = build_rows(
        _read_json(args.manifest.expanduser().resolve()),
        _read_json(args.annotations.expanduser().resolve()),
        overrides=overrides,
        predictions=predictions,
    )
    report = {
        "schema_version": 1,
        "manifest": str(args.manifest.expanduser().resolve()),
        "annotations": str(args.annotations.expanduser().resolve()),
        "overrides": overrides,
        "candidate_predictions": predictions,
        "metrics": score_rows(rows),
        "rows": rows,
    }
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "通用分人人工验收结果.json"
    markdown_path = out_dir / "通用分人人工验收报告.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "json": str(json_path),
        "markdown": str(markdown_path),
        "metrics": report["metrics"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
