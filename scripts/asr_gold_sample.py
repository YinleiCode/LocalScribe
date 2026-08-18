#!/usr/bin/env python3
"""Sample ASR transcript rows for manual gold correction.

The script reads completed transcript JSON files and optional
`ASR质量检查.json` sidecars next to them.  It writes a deterministic JSON
template with strong ASR doubts, weak/ordinary doubts, and normal segments so
CER/WER scoring is not tuned only on known bad rows.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PY_SRC = ROOT / "scribe-py" / "src"
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

try:
    from scribe_py.core.asr_quality import is_strong_asr_review_item, select_asr_review_segments
except Exception:  # pragma: no cover - defensive fallback for running outside repo checkout
    _WEAK_REASON_TOKENS = ("重复词", "重复字", "只有标点", "长句缺少标点")
    _STRONG_REASON_TOKENS = ("ASR 混淆", "ASR 易混淆", "明显不通顺", "语义不顺", "词语断裂", "低文本密度", "密度异常")

    def is_strong_asr_review_item(item: dict[str, Any]) -> bool:
        reasons = [str(reason) for reason in (item.get("reasons") or [])]
        return any(
            any(token in reason for token in _STRONG_REASON_TOKENS)
            and not any(token in reason for token in _WEAK_REASON_TOKENS)
            for reason in reasons
        )

    def select_asr_review_segments(
        transcript_data: dict[str, Any],
        *,
        transcript_json: Path | None = None,
        scope: str = "all",
    ) -> dict[str, Any]:
        stats = (transcript_data.get("filter_stats") or {}).get("text_normalization") or {}
        items = list(stats.get("asr_review_segments") or [])
        items.extend(list(transcript_data.get("asr_review_segments") or []))
        if transcript_json is not None:
            sidecar = transcript_json.parent / "ASR质量检查.json"
            if sidecar.exists():
                data = json.loads(sidecar.read_text(encoding="utf-8-sig"))
                if data.get("mode") == "local_asr_quality":
                    items.extend(list((data.get("review") or {}).get("segments") or []))
        return {"scope": scope, "sources": ["transcript"], "segments": items}


DEFAULT_PER_CASE_STRONG = 10
DEFAULT_PER_CASE_WEAK = 5
DEFAULT_PER_CASE_NORMAL = 5


@dataclass(frozen=True)
class CaseInput:
    label: str
    path: Path
    data: dict[str, Any]
    segments: list[dict[str, Any]]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _parse_labeled_path(value: str, option: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"{option} must be CASE=PATH, got: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise SystemExit(f"{option} case name is empty: {value}")
    if not path.exists():
        raise SystemExit(f"{option} file not found: {path}")
    return label, path


def _load_case(value: str) -> CaseInput:
    label, path = _parse_labeled_path(value, "--case")
    data = _read_json(path)
    if not isinstance(data, dict):
        raise SystemExit(f"transcript JSON must be an object: {path}")
    segments = [seg for seg in (data.get("segments") or []) if isinstance(seg, dict)]
    return CaseInput(label=label, path=path, data=data, segments=segments)


def _segment_text(seg: dict[str, Any]) -> str:
    return str(seg.get("text") or seg.get("current_text") or seg.get("original_text") or "")


def _segment_start(seg: dict[str, Any]) -> float:
    return float(seg.get("start") or 0.0)


def _segment_end(seg: dict[str, Any]) -> float:
    return float(seg.get("end") or 0.0)


def _clean_reasons(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_reasons: list[Any] = [value]
    else:
        raw_reasons = list(value or [])
    reasons: list[str] = []
    for raw in raw_reasons:
        reason = str(raw).strip()
        if reason and reason not in reasons:
            reasons.append(reason)
    return reasons


def _review_key(item: dict[str, Any]) -> tuple[str, int | tuple[float, float, str]]:
    try:
        return ("index", int(item.get("index")))
    except Exception:
        return (
            "time_text",
            (
                round(float(item.get("start") or 0.0), 3),
                round(float(item.get("end") or 0.0), 3),
                str(item.get("text") or item.get("current_text") or item.get("original_text") or ""),
            ),
        )


def _merged_review_segments(case: CaseInput) -> list[dict[str, Any]]:
    selection = select_asr_review_segments(case.data, transcript_json=case.path, scope="all")
    merged: dict[tuple[str, int | tuple[float, float, str]], dict[str, Any]] = {}
    for raw in selection.get("segments") or []:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["reasons"] = _clean_reasons(item.get("reasons"))
        key = _review_key(item)
        current = merged.get(key)
        if current is None:
            merged[key] = item
            continue
        for field in ("index", "start", "end", "text", "current_text", "original_text"):
            if current.get(field) in (None, "") and item.get(field) not in (None, ""):
                current[field] = item[field]
        reasons = list(current.get("reasons") or [])
        for reason in item.get("reasons") or []:
            if reason not in reasons:
                reasons.append(reason)
        current["reasons"] = reasons

    return sorted(
        merged.values(),
        key=lambda item: (
            _segment_start(item),
            int(item.get("index") if item.get("index") is not None else 1_000_000),
        ),
    )


def _review_index(item: dict[str, Any], segment_count: int) -> int | None:
    try:
        index = int(item.get("index"))
    except Exception:
        return None
    if 0 <= index < segment_count:
        return index
    return None


def _row_from_segment(
    *,
    case: CaseInput,
    index: int,
    sample_type: str,
    reasons: list[str],
) -> dict[str, Any]:
    seg = case.segments[index]
    return {
        "case": case.label,
        "index": index,
        "start": _segment_start(seg),
        "end": _segment_end(seg),
        "current_text": _segment_text(seg),
        "reasons": reasons,
        "sample_type": sample_type,
        "correct_text": "",
    }


def _sample_pool(pool: list[int], limit: int, rng: random.Random) -> list[int]:
    if limit <= 0 or not pool:
        return []
    if len(pool) <= limit:
        return sorted(pool)
    return sorted(rng.sample(pool, limit))


def sample_case(
    case: CaseInput,
    *,
    per_case_strong: int = DEFAULT_PER_CASE_STRONG,
    per_case_weak: int = DEFAULT_PER_CASE_WEAK,
    per_case_normal: int = DEFAULT_PER_CASE_NORMAL,
    rng: random.Random | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = rng or random.Random(0)
    review_items = _merged_review_segments(case)
    reasons_by_index: dict[int, list[str]] = {}
    strong_pool: list[int] = []
    weak_pool: list[int] = []

    for item in review_items:
        index = _review_index(item, len(case.segments))
        if index is None:
            continue
        reasons = _clean_reasons(item.get("reasons"))
        existing = reasons_by_index.setdefault(index, [])
        for reason in reasons:
            if reason not in existing:
                existing.append(reason)
        if is_strong_asr_review_item(item):
            if index not in strong_pool:
                strong_pool.append(index)
        elif index not in weak_pool:
            weak_pool.append(index)

    weak_pool = [idx for idx in weak_pool if idx not in set(strong_pool)]
    flagged = set(strong_pool) | set(weak_pool)
    normal_pool = [
        idx
        for idx, seg in enumerate(case.segments)
        if idx not in flagged and _segment_text(seg).strip()
    ]

    selected_types: dict[int, str] = {}
    for index in _sample_pool(strong_pool, per_case_strong, rng):
        selected_types[index] = "strong"
    for index in _sample_pool(weak_pool, per_case_weak, rng):
        selected_types[index] = "weak"
    for index in _sample_pool(normal_pool, per_case_normal, rng):
        selected_types[index] = "normal"

    rows = [
        _row_from_segment(
            case=case,
            index=index,
            sample_type=selected_types[index],
            reasons=[] if selected_types[index] == "normal" else reasons_by_index.get(index, []),
        )
        for index in sorted(selected_types)
    ]
    summary = {
        "case": case.label,
        "transcript": str(case.path),
        "quality_sidecar": str(case.path.parent / "ASR质量检查.json")
        if (case.path.parent / "ASR质量检查.json").exists()
        else "",
        "segment_count": len(case.segments),
        "strong_candidate_count": len(strong_pool),
        "weak_candidate_count": len(weak_pool),
        "normal_candidate_count": len(normal_pool),
        "sampled_strong_count": sum(1 for row in rows if row["sample_type"] == "strong"),
        "sampled_weak_count": sum(1 for row in rows if row["sample_type"] == "weak"),
        "sampled_normal_count": sum(1 for row in rows if row["sample_type"] == "normal"),
    }
    return rows, summary


def build_template(
    case_args: list[str],
    *,
    per_case_strong: int = DEFAULT_PER_CASE_STRONG,
    per_case_weak: int = DEFAULT_PER_CASE_WEAK,
    per_case_normal: int = DEFAULT_PER_CASE_NORMAL,
    seed: int = 0,
) -> dict[str, Any]:
    rng = random.Random(seed)
    cases = [_load_case(value) for value in case_args]
    items: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for case in cases:
        rows, summary = sample_case(
            case,
            per_case_strong=per_case_strong,
            per_case_weak=per_case_weak,
            per_case_normal=per_case_normal,
            rng=rng,
        )
        items.extend(rows)
        summaries.append(summary)
    return {
        "template": "ASR人工校对gold抽样模板",
        "seed": seed,
        "per_case": {
            "strong": per_case_strong,
            "weak": per_case_weak,
            "normal": per_case_normal,
        },
        "cases": summaries,
        "items": items,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从转写 JSON 和 ASR质量检查.json 抽样生成 ASR 人工校对 gold 模板")
    parser.add_argument("--case", action="append", required=True, help="录音名=转写JSON路径; 可重复")
    parser.add_argument("--out", type=Path, required=True, help="输出 gold 抽样模板 JSON 路径")
    parser.add_argument("--per-case-strong", type=int, default=DEFAULT_PER_CASE_STRONG, help="每条录音抽样强疑点数量")
    parser.add_argument("--per-case-weak", type=int, default=DEFAULT_PER_CASE_WEAK, help="每条录音抽样普通/弱疑点数量")
    parser.add_argument("--per-case-normal", type=int, default=DEFAULT_PER_CASE_NORMAL, help="每条录音抽样正常段数量")
    parser.add_argument("--seed", type=int, default=0, help="随机种子,用于可复现抽样")
    args = parser.parse_args(argv)

    payload = build_template(
        args.case,
        per_case_strong=max(args.per_case_strong, 0),
        per_case_weak=max(args.per_case_weak, 0),
        per_case_normal=max(args.per_case_normal, 0),
        seed=args.seed,
    )
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(out),
                "cases": len(payload["cases"]),
                "items": len(payload["items"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
