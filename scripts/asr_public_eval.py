#!/usr/bin/env python3
"""Prepare and score a small public Chinese ASR regression set.

The public set is deliberately independent from LocalScribe customer audio. It
is not a replacement for real meeting gold data, but it prevents model choices
from being made only against recordings that were previously hand-corrected.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PY_SRC = ROOT / "scribe-py" / "src"
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

from scribe_py.core.text_normalizer import simplify_chinese_value  # noqa: E402


DATASET = "AudioLLMs/aishell_1_zh_test"
CONFIG = "default"
SPLIT = "test"
FIRST_ROWS_URL = (
    "https://datasets-server.huggingface.co/first-rows"
    f"?dataset={urllib.parse.quote(DATASET, safe='')}"
    f"&config={CONFIG}&split={SPLIT}"
)


def normalize_cer_text(value: Any) -> str:
    text = simplify_chinese_value(str(value or ""))
    return "".join(ch.lower() for ch in text if ch.isalnum())


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
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_ch != hyp_ch),
                )
            )
        previous = current
    return previous[-1]


def _read_json_url(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "LocalScribe-ASR-Eval/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise RuntimeError("dataset server returned an unexpected response")
    return data


def _audio_url(row: dict[str, Any]) -> str:
    context = row.get("context") or []
    if not isinstance(context, list):
        return ""
    for item in context:
        if isinstance(item, dict) and item.get("src"):
            return str(item["src"])
    return ""


def prepare(out_dir: Path, limit: int) -> Path:
    payload = _read_json_url(FIRST_ROWS_URL)
    rows = list(payload.get("rows") or [])[: max(limit, 0)]
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
            continue
        row = item["row"]
        row_index = int(item.get("row_idx") or 0)
        source = _audio_url(row)
        answer = str(row.get("answer") or "").strip()
        if not source or not answer:
            continue
        audio_path = audio_dir / f"aishell_{row_index:04d}.wav"
        if not audio_path.exists() or audio_path.stat().st_size <= 44:
            request = urllib.request.Request(source, headers={"User-Agent": "LocalScribe-ASR-Eval/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                audio_path.write_bytes(response.read())
        manifest_rows.append(
            {
                "id": f"aishell_{row_index:04d}",
                "row_index": row_index,
                "audio": str(audio_path.resolve()),
                "reference": answer,
            }
        )

    manifest = {
        "mode": "public_asr_gold",
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "license_note": "Evaluation sample only; source audio is downloaded from the Hugging Face dataset server.",
        "count": len(manifest_rows),
        "items": manifest_rows,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _hypothesis_from_result(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return "".join(
        str(segment.get("text") or "")
        for segment in (data.get("segments") or [])
        if isinstance(segment, dict)
    )


def score(manifest_path: Path, benchmark_dir: Path, out_dir: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    items = [item for item in (manifest.get("items") or []) if isinstance(item, dict)]
    grouped: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for item in items:
        case_id = str(item.get("id") or "")
        case_dir = benchmark_dir / case_id
        result_paths = sorted(case_dir.glob("*/result.json"))
        if not result_paths:
            missing.append(case_id)
            continue
        reference = str(item.get("reference") or "")
        normalized_reference = normalize_cer_text(reference)
        for result_path in result_paths:
            backend = result_path.parent.name
            hypothesis = _hypothesis_from_result(result_path)
            normalized_hypothesis = normalize_cer_text(hypothesis)
            edits = edit_distance(normalized_reference, normalized_hypothesis)
            row = {
                "id": case_id,
                "backend": backend,
                "reference": reference,
                "hypothesis": hypothesis,
                "reference_chars": len(normalized_reference),
                "hypothesis_chars": len(normalized_hypothesis),
                "edit_distance": edits,
                "cer": (edits / len(normalized_reference)) if normalized_reference else None,
                "result_json": str(result_path),
            }
            grouped.setdefault(backend, []).append(row)

    summaries: list[dict[str, Any]] = []
    for backend, rows in sorted(grouped.items()):
        ref_chars = sum(int(row["reference_chars"]) for row in rows)
        edits = sum(int(row["edit_distance"]) for row in rows)
        summaries.append(
            {
                "backend": backend,
                "cases": len(rows),
                "reference_chars": ref_chars,
                "edit_distance": edits,
                "cer": (edits / ref_chars) if ref_chars else None,
            }
        )
    summaries.sort(key=lambda item: float(item["cer"] if item["cer"] is not None else 999.0))

    report = {
        "mode": "public_asr_gold_score",
        "manifest": str(manifest_path),
        "benchmark_dir": str(benchmark_dir),
        "missing_cases": missing,
        "summaries": summaries,
        "results": grouped,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "public_eval.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 公开中文 ASR 真值评分\n\n",
        f"- 数据集: `{manifest.get('dataset', '')}` / `{manifest.get('split', '')}`\n",
        f"- 固定样本数: {len(items)}\n",
        f"- 缺失结果: {len(missing)}\n",
        "- 口径: 简体化后忽略标点和空白，计算字符错误率 CER；越低越好。\n\n",
        "| 排名 | 后端 | 完成样本 | 参考字数 | 编辑距离 | CER |\n",
        "|---:|---|---:|---:|---:|---:|\n",
    ]
    for rank, summary in enumerate(summaries, start=1):
        cer = summary.get("cer")
        lines.append(
            f"| {rank} | {summary['backend']} | {summary['cases']} | "
            f"{summary['reference_chars']} | {summary['edit_distance']} | "
            f"{('-' if cer is None else f'{cer:.2%}')} |\n"
        )
    markdown_path = out_dir / "public_eval.md"
    markdown_path.write_text("".join(lines), encoding="utf-8")
    return markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or score a fixed public Chinese ASR set")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--out", required=True)
    prepare_parser.add_argument("--limit", type=int, default=20)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--manifest", required=True)
    score_parser.add_argument("--benchmark", required=True)
    score_parser.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        path = prepare(Path(args.out).expanduser().resolve(), args.limit)
    else:
        path = score(
            Path(args.manifest).expanduser().resolve(),
            Path(args.benchmark).expanduser().resolve(),
            Path(args.out).expanduser().resolve(),
        )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
