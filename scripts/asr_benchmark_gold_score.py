#!/usr/bin/env python3
"""Score every backend in an ASR benchmark against a shared gold template."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from asr_gold_score import edit_distance, normalize_for_cer  # noqa: E402


_ID_RE = re.compile(r"^(GOLD-\d+)")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _result_text(path: Path) -> str:
    data = _read_json(path)
    return "".join(
        str(segment.get("text") or "")
        for segment in data.get("segments") or []
        if isinstance(segment, dict)
    )


def score_benchmark(gold_path: Path, benchmark_dir: Path) -> dict[str, Any]:
    gold = _read_json(gold_path)
    gold_items = {
        str(item.get("id") or ""): item
        for item in gold.get("items") or []
        if isinstance(item, dict) and str(item.get("correct_text") or "").strip()
    }
    rows_by_backend: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[str] = []
    for item_id, item in gold_items.items():
        case_dirs = sorted(path for path in benchmark_dir.glob(f"{item_id}_*") if path.is_dir())
        if not case_dirs:
            missing.append(item_id)
            continue
        reference = str(item.get("correct_text") or "")
        ref = normalize_for_cer(reference)
        for result_path in sorted(case_dirs[0].glob("*/result.json")):
            backend = result_path.parent.name
            hypothesis = _result_text(result_path)
            hyp = normalize_for_cer(hypothesis)
            edits = edit_distance(ref, hyp)
            rows_by_backend[backend].append(
                {
                    "id": item_id,
                    "case": str(item.get("case") or ""),
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "reference_chars": len(ref),
                    "hypothesis_chars": len(hyp),
                    "edit_distance": edits,
                    "cer": (edits / len(ref)) if ref else None,
                    "result_json": str(result_path),
                }
            )

    summaries: list[dict[str, Any]] = []
    for backend, rows in rows_by_backend.items():
        ref_chars = sum(int(row["reference_chars"]) for row in rows)
        edits = sum(int(row["edit_distance"]) for row in rows)
        summaries.append(
            {
                "backend": backend,
                "scored_items": len(rows),
                "reference_chars": ref_chars,
                "edit_distance": edits,
                "cer": (edits / ref_chars) if ref_chars else None,
                "accuracy": (1.0 - edits / ref_chars) if ref_chars else None,
                "perfect_items": sum(1 for row in rows if row["edit_distance"] == 0),
            }
        )
    summaries.sort(key=lambda row: float(row["cer"] if row["cer"] is not None else 999.0))
    return {
        "mode": "asr_benchmark_gold_score",
        "gold": str(gold_path),
        "benchmark": str(benchmark_dir),
        "filled_gold_items": len(gold_items),
        "excluded_partial_items": [
            str(item.get("id") or "")
            for item in gold.get("items") or []
            if isinstance(item, dict)
            and item.get("decision") == "部分校正"
            and not str(item.get("correct_text") or "").strip()
        ],
        "missing_items": missing,
        "summaries": summaries,
        "results": dict(rows_by_backend),
    }


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ASR 三模型人工真值评分\n\n",
        f"- 严格人工标准段：{report['filled_gold_items']}\n",
        f"- 部分校正未计分：{len(report['excluded_partial_items'])}\n",
        f"- 缺失模型结果：{len(report['missing_items'])}\n",
        "- 口径：简体化后忽略标点和空白，按总参考字数计算 CER。\n\n",
        "| 排名 | 模型后端 | 评分段 | 参考字数 | 字符错误 | CER | 字符准确率 | 完全正确段 |\n",
        "|---:|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for rank, summary in enumerate(report.get("summaries") or [], start=1):
        lines.append(
            f"| {rank} | {summary['backend']} | {summary['scored_items']} | "
            f"{summary['reference_chars']} | {summary['edit_distance']} | {_pct(summary['cer'])} | "
            f"{_pct(summary['accuracy'])} | {summary['perfect_items']} |\n"
        )
    lines.extend(["\n## 各模型最差片段\n\n"])
    for summary in report.get("summaries") or []:
        backend = str(summary["backend"])
        rows = sorted(
            report.get("results", {}).get(backend, []),
            key=lambda row: (float(row.get("cer") or 0.0), int(row.get("edit_distance") or 0)),
            reverse=True,
        )[:5]
        lines.extend(
            [
                f"### {backend}\n\n",
                "| ID | CER | 参考文字 | 模型文字 |\n",
                "|---|---:|---|---|\n",
            ]
        )
        for row in rows:
            reference = str(row["reference"]).replace("\n", "<br>").replace("|", "\\|")
            hypothesis = str(row["hypothesis"]).replace("\n", "<br>").replace("|", "\\|")
            lines.append(f"| {row['id']} | {_pct(row['cer'])} | {reference} | {hypothesis} |\n")
        lines.append("\n")
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按人工标准答案评分 ASR benchmark 的全部模型")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    report = score_benchmark(args.gold.expanduser().resolve(), args.benchmark.expanduser().resolve())
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "三模型人工真值评分.json"
    md_path = out_dir / "三模型人工真值评分.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
