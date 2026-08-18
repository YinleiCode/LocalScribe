#!/usr/bin/env python3
"""Build a non-destructive ASR review pack from multi-model disagreement."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PY_SRC = ROOT / "scribe-py" / "src"
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

from scribe_py.core.strong_asr import normalized_text, text_similarity  # noqa: E402


_ID_RE = re.compile(r"^(GOLD-\d+)")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _result_text(path: Path) -> str:
    data = _read_json(path)
    return "\n".join(
        str(segment.get("text") or "").strip()
        for segment in data.get("segments") or []
        if isinstance(segment, dict) and str(segment.get("text") or "").strip()
    )


def _model_result(benchmark_dir: Path, item_id: str, backend: str) -> tuple[str, str]:
    case_dirs = sorted(path for path in benchmark_dir.glob(f"{item_id}_*") if path.is_dir())
    if not case_dirs:
        return "", ""
    result_path = case_dirs[0] / backend / "result.json"
    if not result_path.exists():
        return "", ""
    return _result_text(result_path), str(result_path)


def disagreement_metrics(primary: str, paraformer: str, qwen3: str) -> dict[str, float | bool]:
    primary_para = text_similarity(primary, paraformer)
    primary_qwen = text_similarity(primary, qwen3)
    para_qwen = text_similarity(paraformer, qwen3)
    primary_len = max(len(normalized_text(primary)), 1)
    para_len = len(normalized_text(paraformer))
    qwen_len = len(normalized_text(qwen3))
    length_penalty = min(
        1.0,
        max(abs(para_len - primary_len), abs(qwen_len - primary_len)) / primary_len,
    )
    primary_gap = 1.0 - ((primary_para + primary_qwen) / 2.0)
    auxiliary_consensus_gap = max(0.0, para_qwen - ((primary_para + primary_qwen) / 2.0))
    score = min(1.0, (0.62 * primary_gap) + (0.25 * auxiliary_consensus_gap) + (0.13 * length_penalty))
    return {
        "primary_paraformer_similarity": round(primary_para, 4),
        "primary_qwen3_similarity": round(primary_qwen, 4),
        "paraformer_qwen3_similarity": round(para_qwen, 4),
        "length_penalty": round(length_penalty, 4),
        "disagreement_score": round(score, 4),
        "auxiliary_consensus_against_primary": bool(
            para_qwen >= 0.78
            and ((primary_para + primary_qwen) / 2.0) <= 0.72
        ),
    }


def build_rows(template_path: Path, benchmark_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json(template_path)
    base_dir = template_path.parent
    rows: list[dict[str, Any]] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        if not _ID_RE.match(item_id):
            continue
        primary = str(item.get("current_text") or "").strip()
        paraformer, paraformer_result = _model_result(benchmark_dir, item_id, "funasr")
        qwen3, qwen3_result = _model_result(benchmark_dir, item_id, "qwen3")
        if not primary or not paraformer or not qwen3:
            continue
        clip_path = Path(str(item.get("clip_path") or ""))
        if not clip_path.is_absolute():
            clip_path = (base_dir / clip_path).resolve()
        metrics = disagreement_metrics(primary, paraformer, qwen3)
        rows.append(
            {
                "id": item_id,
                "case_id": str(item.get("case_id") or ""),
                "case": str(item.get("case") or ""),
                "start": float(item.get("start") or 0.0),
                "duration": float(item.get("duration") or 0.0),
                "clip_path": str(clip_path),
                "sensevoice": primary,
                "paraformer": paraformer,
                "qwen3": qwen3,
                "paraformer_result": paraformer_result,
                "qwen3_result": qwen3_result,
                **metrics,
            }
        )
    return sorted(rows, key=lambda row: (-float(row["disagreement_score"]), row["id"]))


def select_rows(rows: list[dict[str, Any]], *, limit: int, per_case: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    case_ids = sorted({str(row.get("case_id") or "") for row in rows})
    for case_id in case_ids:
        case_rows = [row for row in rows if str(row.get("case_id") or "") == case_id]
        for row in case_rows[: max(per_case, 0)]:
            if len(selected) >= limit:
                break
            selected.append(row)
            selected_ids.add(str(row["id"]))
    for row in rows:
        if len(selected) >= limit:
            break
        if str(row["id"]) in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(str(row["id"]))
    return sorted(selected, key=lambda row: (-float(row["disagreement_score"]), row["id"]))


def _cell(value: Any) -> str:
    return str(value or "").replace("\n", "<br>").replace("|", "\\|")


def materialize_review_clips(rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    materialized: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        item = dict(row)
        source = Path(str(row.get("clip_path") or ""))
        if source.exists():
            destination = clips_dir / f"{rank:02d}_{row['id']}{source.suffix.lower() or '.wav'}"
            shutil.copyfile(source, destination)
            item["review_clip_path"] = str(destination.resolve())
        else:
            item["review_clip_path"] = ""
        materialized.append(item)
    return materialized


def render_markdown(rows: list[dict[str, Any]], *, total: int) -> str:
    consensus_count = sum(bool(row["auxiliary_consensus_against_primary"]) for row in rows)
    lines = [
        "# ASR 模型分歧优先核对清单\n\n",
        f"- 全量片段：{total}\n",
        f"- 优先核对：{len(rows)}\n",
        f"- 两个辅助模型共同反对主模型：{consensus_count}\n",
        "- 说明：辅助文本只用于定位疑点，不会自动替换 SenseVoice 结果。\n",
        "- 操作：播放 Clip；完全正确填 `确认`，有错误时填写完整正确文字。\n\n",
        "| 排名 | ID | 录音 | 分歧分 | 辅助共识 | SenseVoice | Paraformer | Qwen3 | Clip |\n",
        "|---:|---|---|---:|---|---|---|---|---|\n",
    ]
    for rank, row in enumerate(rows, start=1):
        clip_path = row.get("review_clip_path") or row["clip_path"]
        lines.append(
            f"| {rank} | {row['id']} | {_cell(row['case'])} | {float(row['disagreement_score']):.3f} | "
            f"{'是' if row['auxiliary_consensus_against_primary'] else '否'} | {_cell(row['sensevoice'])} | "
            f"{_cell(row['paraformer'])} | {_cell(row['qwen3'])} | `{clip_path}` |\n"
        )
    return "".join(lines)


def build_review_template(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mode": "asr_disagreement_review",
        "policy": "辅助模型只提供疑点证据，不自动改写主转录",
        "items": [
            {
                **row,
                "decision": "",
                "correct_text": "",
                "notes": "",
            }
            for row in rows
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按多模型分歧生成非破坏性 ASR 人工核对包")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--per-case", type=int, default=2)
    args = parser.parse_args(argv)

    template = args.template.expanduser().resolve()
    benchmark = args.benchmark.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    rows = build_rows(template, benchmark)
    selected = select_rows(rows, limit=max(args.limit, 0), per_case=max(args.per_case, 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = materialize_review_clips(selected, out_dir)
    json_path = out_dir / "人工优先核对模板.json"
    md_path = out_dir / "人工优先核对清单.md"
    json_path.write_text(json.dumps(build_review_template(selected), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(selected, total=len(rows)), encoding="utf-8")
    print(json.dumps({"ok": True, "total": len(rows), "selected": len(selected), "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
