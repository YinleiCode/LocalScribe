#!/usr/bin/env python3
"""Score ASR output against manually corrected gold text.

This is a local, deterministic regression tool. It reads the manual correction
template produced by the ASR review workflow, skips rows without `correct_text`,
and writes Chinese Markdown/JSON reports with character error rate (CER).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_REPORT_NAME = "ASR回归评分报告"


@dataclass
class GoldScoreRow:
    id: str
    index: str
    time: str
    reasons: str
    current_text: str
    correct_text: str
    notes: str
    current_normalized: str
    correct_normalized: str
    ref_chars: int
    hyp_chars: int
    edit_distance: int
    cer: float | None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _load_gold_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f)), {"source_format": "csv"}

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)], {"source_format": "json"}
    if isinstance(data, dict):
        for key in ("items", "rows", "segments"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)], {
                    "source_format": "json",
                    "source_json": data.get("source_json", ""),
                    "row_key": key,
                }
    raise ValueError(f"unsupported gold template shape: {path}")


def _is_ignored_char(ch: str, *, ignore_punctuation: bool, ignore_whitespace: bool) -> bool:
    if ignore_whitespace and ch.isspace():
        return True
    category = unicodedata.category(ch)
    if ignore_whitespace and category.startswith("Z"):
        return True
    if ignore_punctuation and category.startswith("P"):
        return True
    return False


def normalize_for_cer(
    text: str,
    *,
    ignore_punctuation: bool = True,
    ignore_whitespace: bool = True,
) -> str:
    normalized = unicodedata.normalize("NFKC", _as_text(text))
    return "".join(
        ch
        for ch in normalized
        if not _is_ignored_char(
            ch,
            ignore_punctuation=ignore_punctuation,
            ignore_whitespace=ignore_whitespace,
        )
    )


def edit_distance(reference: str, hypothesis: str) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_ch in enumerate(reference, start=1):
        current = [i]
        for j, hyp_ch in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (0 if ref_ch == hyp_ch else 1)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def score_gold_rows(
    rows: list[dict[str, Any]],
    *,
    ignore_punctuation: bool = True,
    ignore_whitespace: bool = True,
) -> tuple[dict[str, Any], list[GoldScoreRow]]:
    scored: list[GoldScoreRow] = []
    skipped_empty_rows = 0
    empty_reference_rows = 0
    total_ref_chars = 0
    total_hyp_chars = 0
    total_edits = 0

    for raw in rows:
        correct_text = _as_text(raw.get("correct_text")).strip()
        if not correct_text:
            skipped_empty_rows += 1
            continue

        current_text = _as_text(raw.get("current_text"))
        ref = normalize_for_cer(
            correct_text,
            ignore_punctuation=ignore_punctuation,
            ignore_whitespace=ignore_whitespace,
        )
        hyp = normalize_for_cer(
            current_text,
            ignore_punctuation=ignore_punctuation,
            ignore_whitespace=ignore_whitespace,
        )
        edits = edit_distance(ref, hyp)
        ref_chars = len(ref)
        hyp_chars = len(hyp)
        cer = (edits / ref_chars) if ref_chars else (0.0 if edits == 0 else None)
        if ref_chars == 0:
            empty_reference_rows += 1
        total_ref_chars += ref_chars
        total_hyp_chars += hyp_chars
        total_edits += edits
        scored.append(
            GoldScoreRow(
                id=_as_text(raw.get("id")),
                index=_as_text(raw.get("index")),
                time=_as_text(raw.get("time")),
                reasons=_as_text(raw.get("reasons")),
                current_text=current_text,
                correct_text=correct_text,
                notes=_as_text(raw.get("notes")),
                current_normalized=hyp,
                correct_normalized=ref,
                ref_chars=ref_chars,
                hyp_chars=hyp_chars,
                edit_distance=edits,
                cer=cer,
            )
        )

    summary = {
        "total_rows": len(rows),
        "filled_gold_rows": len(scored),
        "skipped_empty_rows": skipped_empty_rows,
        "empty_reference_rows_after_normalization": empty_reference_rows,
        "total_reference_chars": total_ref_chars,
        "total_hypothesis_chars": total_hyp_chars,
        "total_edit_distance": total_edits,
        "overall_cer": (total_edits / total_ref_chars) if total_ref_chars else None,
        "ignore_punctuation": ignore_punctuation,
        "ignore_whitespace": ignore_whitespace,
    }
    return summary, scored


def _pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def _md_cell(value: Any) -> str:
    return _as_text(value).replace("\n", "<br>").replace("|", "\\|")


def _row_to_dict(row: GoldScoreRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "index": row.index,
        "time": row.time,
        "reasons": row.reasons,
        "current_text": row.current_text,
        "correct_text": row.correct_text,
        "notes": row.notes,
        "current_normalized": row.current_normalized,
        "correct_normalized": row.correct_normalized,
        "ref_chars": row.ref_chars,
        "hyp_chars": row.hyp_chars,
        "edit_distance": row.edit_distance,
        "cer": row.cer,
    }


def _worst_rows(rows: list[GoldScoreRow], limit: int) -> list[GoldScoreRow]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -1.0 if row.cer is None else row.cer,
            row.edit_distance,
            row.ref_chars,
        ),
        reverse=True,
    )
    if limit <= 0:
        return ordered
    return ordered[:limit]


def render_markdown_report(
    *,
    gold_path: Path,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    rows: list[GoldScoreRow],
    worst_limit: int = 10,
) -> str:
    lines = [
        "# ASR 人工标准答案回归评分\n\n",
        f"- 标准答案文件: `{gold_path}`\n",
        f"- 输入格式: `{metadata.get('source_format', '-')}`\n",
        f"- 总行数: {summary['total_rows']}\n",
        f"- 已填写标准答案: {summary['filled_gold_rows']}\n",
        f"- 跳过空标准答案: {summary['skipped_empty_rows']}\n",
        f"- 参与评分参考字数: {summary['total_reference_chars']}\n",
        f"- 总编辑距离: {summary['total_edit_distance']}\n",
        f"- 总体 CER: {_pct(summary['overall_cer'])}\n",
        f"- 评分口径: {'忽略标点' if summary['ignore_punctuation'] else '保留标点'}；{'忽略空白' if summary['ignore_whitespace'] else '保留空白'}\n",
        "\n",
    ]
    if summary["filled_gold_rows"] == 0:
        lines.extend(
            [
                "## 结论\n\n",
                "当前模板还没有填写 `correct_text`，所以本次只完成模板读取校验，没有可计算的 CER。人工补齐标准答案后重新运行本脚本即可得到回归评分。\n",
            ]
        )
        return "".join(lines)

    lines.extend(
        [
            "## 最差片段\n\n",
            "| ID | 时间 | CER | 编辑距离 | 参考字数 | 当前文本 | 人工标准答案 |\n",
            "|---|---|---:|---:|---:|---|---|\n",
        ]
    )
    for row in _worst_rows(rows, worst_limit):
        lines.append(
            "| "
            f"{_md_cell(row.id)} | "
            f"{_md_cell(row.time)} | "
            f"{_pct(row.cer)} | "
            f"{row.edit_distance} | "
            f"{row.ref_chars} | "
            f"{_md_cell(row.current_text)} | "
            f"{_md_cell(row.correct_text)} |\n"
        )

    lines.extend(
        [
            "\n## 全部已评分片段\n\n",
            "| ID | Index | 时间 | CER | 编辑距离 | 当前文本 | 人工标准答案 | 备注 |\n",
            "|---|---:|---|---:|---:|---|---|---|\n",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{_md_cell(row.id)} | "
            f"{_md_cell(row.index)} | "
            f"{_md_cell(row.time)} | "
            f"{_pct(row.cer)} | "
            f"{row.edit_distance} | "
            f"{_md_cell(row.current_text)} | "
            f"{_md_cell(row.correct_text)} | "
            f"{_md_cell(row.notes)} |\n"
        )
    return "".join(lines)


def write_reports(
    *,
    gold_path: Path,
    out_dir: Path,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    rows: list[GoldScoreRow],
    worst_limit: int = 10,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / f"{DEFAULT_REPORT_NAME}.json"
    md_out = out_dir / f"{DEFAULT_REPORT_NAME}.md"
    payload = {
        "source_gold_path": str(gold_path),
        "metadata": metadata,
        "summary": summary,
        "rows": [_row_to_dict(row) for row in rows],
        "worst_rows": [_row_to_dict(row) for row in _worst_rows(rows, worst_limit)],
    }
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_out.write_text(
        render_markdown_report(
            gold_path=gold_path,
            metadata=metadata,
            summary=summary,
            rows=rows,
            worst_limit=worst_limit,
        ),
        encoding="utf-8",
    )
    return json_out, md_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按人工标准答案计算 ASR CER 回归评分")
    parser.add_argument("gold_path", type=Path, help="人工标准答案模板 JSON/CSV")
    parser.add_argument("--out", type=Path, default=None, help="输出目录; 默认写到 gold_path 同级")
    parser.add_argument("--worst-limit", type=int, default=10, help="最差片段表最多展示多少行; 0 表示全部")
    parser.add_argument("--keep-punctuation", action="store_true", help="评分时保留标点")
    parser.add_argument("--keep-whitespace", action="store_true", help="评分时保留空白")
    args = parser.parse_args(argv)

    gold_path = args.gold_path.expanduser().resolve()
    if not gold_path.exists():
        raise SystemExit(f"gold template not found: {gold_path}")
    rows_raw, metadata = _load_gold_rows(gold_path)
    summary, rows = score_gold_rows(
        rows_raw,
        ignore_punctuation=not args.keep_punctuation,
        ignore_whitespace=not args.keep_whitespace,
    )
    out_dir = args.out.expanduser().resolve() if args.out else gold_path.parent
    json_out, md_out = write_reports(
        gold_path=gold_path,
        out_dir=out_dir,
        metadata=metadata,
        summary=summary,
        rows=rows,
        worst_limit=args.worst_limit,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "json": str(json_out),
                "md": str(md_out),
                "filled_gold_rows": summary["filled_gold_rows"],
                "skipped_empty_rows": summary["skipped_empty_rows"],
                "overall_cer": summary["overall_cer"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
