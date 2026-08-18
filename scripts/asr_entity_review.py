#!/usr/bin/env python3
"""Export and safely apply ASR entity consistency review decisions.

The workflow is deliberately conservative:

1. export: read a transcript JSON and its ASR quality sidecar, then write a
   Chinese review checklist plus an editable decisions JSON.
2. apply: read the decisions JSON and produce a new transcript JSON. The source
   transcript is never modified unless a caller explicitly writes the output
   over it outside this script.

Entity drift candidates are review-only by default. They can be fixed only with
explicit per-segment occurrence replacements, because replacing a generic word
such as "男人" globally would be unsafe.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PY_SRC = ROOT / "scribe-py" / "src"
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

from scribe_py.core.asr_quality import build_asr_quality_report, write_asr_quality_reports  # noqa: E402
from scribe_py.core.types import Segment  # noqa: E402


DEFAULT_OUT_NAME = "实体一致性核对"
APPLY_MARKER = "entity_consistency_review"
APPLICABLE_KINDS = {"phonetic_entity", "orthographic_term"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt_ts(seconds: float) -> str:
    ms = int(round(float(seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _md(value: Any) -> str:
    return str(value or "").replace("\n", "<br>").replace("|", "\\|")


def _load_quality(transcript_json: Path, transcript_data: dict[str, Any], quality_json: Path | None = None) -> dict[str, Any]:
    candidates: list[Path] = []
    if quality_json is not None:
        candidates.append(quality_json)
    candidates.append(transcript_json.parent / "ASR质量检查.json")
    for path in candidates:
        if path.exists():
            data = _read_json(path)
            if isinstance(data, dict):
                return data
    quality = transcript_data.get("asr_quality")
    return quality if isinstance(quality, dict) else {}


def _term_consistency_candidates(quality: dict[str, Any]) -> list[dict[str, Any]]:
    term_consistency = quality.get("term_consistency") or {}
    return [item for item in (term_consistency.get("candidates") or []) if isinstance(item, dict)]


def _kind_label(kind: str) -> str:
    return {
        "phonetic_entity": "同音实体",
        "entity_drift": "实体漂移",
        "orthographic_term": "字形相近",
    }.get(kind or "", kind or "-")


def _candidate_terms(candidate: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    raw_terms = candidate.get("terms") or []
    for raw in raw_terms:
        term = str(raw or "").strip()
        if term and term not in terms:
            terms.append(term)
    if terms:
        return terms
    for variant in candidate.get("variants") or []:
        if not isinstance(variant, dict):
            continue
        term = str(variant.get("text") or "").strip()
        if term and term not in terms:
            terms.append(term)
    return terms


def _variant_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants = candidate.get("variants") or []
    if variants:
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            rows.append(
                {
                    "text": str(variant.get("text") or ""),
                    "count": int(variant.get("count") or 0),
                    "contexts": list(variant.get("contexts") or [])[:8],
                }
            )
        return rows
    return [{"text": term, "count": 0, "contexts": []} for term in _candidate_terms(candidate)]


def _context_rows(candidate: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, float, str]] = set()
    for raw in candidate.get("contexts") or []:
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("index"))
        except Exception:
            index = -1
        start = float(raw.get("start") or 0)
        text = str(raw.get("text") or "")
        key = (index, round(start, 3), text)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "index": index,
                "start": start,
                "end": float(raw.get("end") or 0),
                "text": text,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _decision_from_candidate(candidate: dict[str, Any], *, context_limit: int) -> dict[str, Any]:
    kind = str(candidate.get("kind") or "orthographic_term")
    terms = _candidate_terms(candidate)
    can_global_unify = kind in APPLICABLE_KINDS
    occurrence_replacements = []
    if kind == "entity_drift":
        for context in _context_rows(candidate, limit=context_limit):
            occurrence_replacements.append(
                {
                    "index": context["index"],
                    "from": "",
                    "to": "",
                    "note": "如确认本段某个普通词应为实体名,填 from/to 并把 action 改为 replace_occurrences。",
                }
            )
    return {
        "id": str(candidate.get("id") or ""),
        "kind": kind,
        "action": "skip",
        "canonical_text": "",
        "replace_terms": terms if can_global_unify else [],
        "allow_global_unify": can_global_unify,
        "phonetic_key": str(candidate.get("phonetic_key") or ""),
        "terms": terms,
        "occurrence_replacements": occurrence_replacements,
        "note": "可选 action: skip / unify / replace_occurrences。unify 只适用于 allow_global_unify=true 的候选。",
    }


def build_review_payload(
    transcript_json: Path,
    *,
    quality_json: Path | None = None,
    context_limit: int = 8,
) -> dict[str, Any]:
    transcript_data = _read_json(transcript_json)
    if not isinstance(transcript_data, dict):
        raise SystemExit(f"transcript JSON must be an object: {transcript_json}")
    quality = _load_quality(transcript_json, transcript_data, quality_json)
    candidates = _term_consistency_candidates(quality)
    review_candidates: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for candidate in candidates:
        item = {
            "id": str(candidate.get("id") or ""),
            "kind": str(candidate.get("kind") or "orthographic_term"),
            "kind_label": _kind_label(str(candidate.get("kind") or "orthographic_term")),
            "action": str(candidate.get("action") or "review"),
            "confidence": float(candidate.get("confidence") or 0),
            "phonetic_key": str(candidate.get("phonetic_key") or ""),
            "terms": _candidate_terms(candidate),
            "suggested_canonical": candidate.get("suggested_canonical"),
            "total_count": int(candidate.get("total_count") or 0),
            "reason": str(candidate.get("reason") or ""),
            "variants": _variant_rows(candidate),
            "contexts": _context_rows(candidate, limit=context_limit),
        }
        review_candidates.append(item)
        decisions.append(_decision_from_candidate(candidate, context_limit=context_limit))
    return {
        "template": "ASR实体一致性核对清单",
        "source_json": str(transcript_json),
        "quality_json": str(quality_json or transcript_json.parent / "ASR质量检查.json"),
        "candidate_count": len(review_candidates),
        "instructions": [
            "默认 action=skip,不会改任何文本。",
            "同音实体/字形相近候选: 如确认同一实体,把 action 改为 unify,并填写 canonical_text。",
            "实体漂移候选: 不支持全局统一; 如确认某一段需要修正,把 action 改为 replace_occurrences,并填写 occurrence_replacements 里的 index/from/to。",
            "apply 会生成新的转录 JSON,不会覆盖原文件。",
        ],
        "candidates": review_candidates,
        "decisions": decisions,
    }


def render_review_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ASR 实体一致性核对清单\n\n",
        f"- 转录 JSON: `{payload.get('source_json', '')}`\n",
        f"- 质量报告 JSON: `{payload.get('quality_json', '')}`\n",
        f"- 候选组数: {payload.get('candidate_count', 0)}\n",
        "- 说明: 本清单只用于人工确认；默认不改原文。确认后用 `action=unify` 或 `action=replace_occurrences` 生成新转录。\n\n",
        "## 操作口径\n\n",
    ]
    for instruction in payload.get("instructions") or []:
        lines.append(f"- {instruction}\n")
    lines.extend(
        [
            "\n",
            "## 候选总表\n\n",
            "| ID | 类型 | 置信度 | 读音键 | 候选词 | 建议主写法 | 出现次数 | 原因 |\n",
            "|---|---|---:|---|---|---|---:|---|\n",
        ]
    )
    for candidate in payload.get("candidates") or []:
        terms = "、".join(str(x) for x in candidate.get("terms") or [])
        lines.append(
            f"| {_md(candidate.get('id'))} | {_md(candidate.get('kind_label'))} | "
            f"{float(candidate.get('confidence') or 0):.3f} | {_md(candidate.get('phonetic_key') or '-')} | "
            f"{_md(terms)} | {_md(candidate.get('suggested_canonical') or '-')} | "
            f"{int(candidate.get('total_count') or 0)} | {_md(candidate.get('reason'))} |\n"
        )
    for candidate in payload.get("candidates") or []:
        lines.extend(
            [
                "\n",
                f"## {candidate.get('id')} {candidate.get('kind_label')}\n\n",
                f"- 候选词: {'、'.join(str(x) for x in candidate.get('terms') or [])}\n",
                f"- 读音键: `{candidate.get('phonetic_key') or '-'}`\n",
                f"- 原因: {candidate.get('reason') or ''}\n\n",
                "| 变体 | 出现次数 |\n",
                "|---|---:|\n",
            ]
        )
        for variant in candidate.get("variants") or []:
            lines.append(f"| {_md(variant.get('text'))} | {int(variant.get('count') or 0)} |\n")
        contexts = candidate.get("contexts") or []
        if contexts:
            lines.extend(["\n", "| 时间 | 段号 | 上下文 |\n", "|---|---:|---|\n"])
            for context in contexts:
                ts = f"{_fmt_ts(float(context.get('start') or 0))}-{_fmt_ts(float(context.get('end') or 0))}"
                lines.append(f"| {ts} | {int(context.get('index') or 0)} | {_md(context.get('text'))} |\n")
    return "".join(lines)


def export_review(
    transcript_json: Path,
    *,
    quality_json: Path | None = None,
    out_dir: Path | None = None,
    context_limit: int = 8,
) -> dict[str, str]:
    payload = build_review_payload(transcript_json, quality_json=quality_json, context_limit=context_limit)
    target_dir = out_dir or transcript_json.parent / DEFAULT_OUT_NAME
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "实体一致性核对清单.json"
    md_path = target_dir / "实体一致性核对清单.md"
    _write_json(json_path, payload)
    md_path.write_text(render_review_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path), "candidates": str(payload["candidate_count"])}


def _segments_from_transcript(data: dict[str, Any]) -> list[Segment]:
    return [
        Segment(
            start=float(seg.get("start") or 0),
            end=float(seg.get("end") or 0),
            text=str(seg.get("text") or ""),
            original_text=seg.get("original_text"),
            speaker=seg.get("speaker"),
        )
        for seg in data.get("segments") or []
        if isinstance(seg, dict)
    ]


def _build_quality_for_transcript(data: dict[str, Any]) -> dict[str, Any]:
    fs = data.get("filter_stats") or {}
    return build_asr_quality_report(
        _segments_from_transcript(data),
        text_normalization=fs.get("text_normalization") or {},
        audio_quality=fs.get("audio_quality") or {},
        audio_preprocessing=fs.get("audio_standardization") or {},
        backend=str(data.get("backend") or ""),
        model_id=str(data.get("model_id") or ""),
        duration=float(data.get("duration") or 0),
        transcribe_seconds=float(data.get("transcribe_seconds") or 0),
        rtf=float(data.get("rtf") or 0),
    )


def _decision_by_id(review_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("id") or ""): item
        for item in review_payload.get("decisions") or []
        if isinstance(item, dict) and str(item.get("id") or "")
    }


def _candidate_by_id(review_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("id") or ""): item
        for item in review_payload.get("candidates") or []
        if isinstance(item, dict) and str(item.get("id") or "")
    }


def _validate_replace_terms(terms: list[str], canonical: str) -> list[str]:
    cleaned: list[str] = []
    for raw in terms:
        term = str(raw or "").strip()
        if not term or term == canonical:
            continue
        if len(term) < 2:
            continue
        if term not in cleaned:
            cleaned.append(term)
    return cleaned


def _apply_global_unify(
    segments: list[dict[str, Any]],
    *,
    terms: list[str],
    canonical: str,
    candidate_id: str,
) -> dict[str, Any]:
    applied_segments = 0
    replacements = 0
    touched: list[dict[str, Any]] = []
    for index, seg in enumerate(segments):
        text = str(seg.get("text") or "")
        before = text
        segment_replacements = 0
        for term in terms:
            count = text.count(term)
            if count <= 0:
                continue
            text = text.replace(term, canonical)
            segment_replacements += count
        if text == before:
            continue
        seg["text"] = text
        applied_segments += 1
        replacements += segment_replacements
        touched.append(
            {
                "candidate_id": candidate_id,
                "index": index,
                "before": before,
                "after": text,
                "replacement_count": segment_replacements,
            }
        )
    return {
        "segments": applied_segments,
        "replacements": replacements,
        "touched": touched,
    }


def _apply_occurrence_replacements(
    segments: list[dict[str, Any]],
    *,
    replacements: list[dict[str, Any]],
    candidate_id: str,
) -> dict[str, Any]:
    applied_segments = 0
    replacement_count = 0
    touched: list[dict[str, Any]] = []
    for item in replacements:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except Exception:
            continue
        if index < 0 or index >= len(segments):
            continue
        old = str(item.get("from") or "").strip()
        new = str(item.get("to") or "").strip()
        if not old or not new or old == new:
            continue
        seg = segments[index]
        before = str(seg.get("text") or "")
        count = before.count(old)
        if count <= 0:
            continue
        after = before.replace(old, new)
        seg["text"] = after
        applied_segments += 1
        replacement_count += count
        touched.append(
            {
                "candidate_id": candidate_id,
                "index": index,
                "before": before,
                "after": after,
                "from": old,
                "to": new,
                "replacement_count": count,
            }
        )
    return {"segments": applied_segments, "replacements": replacement_count, "touched": touched}


def apply_review(
    transcript_json: Path,
    review_json: Path,
    *,
    out_json: Path | None = None,
    write_quality: bool = True,
) -> dict[str, Any]:
    data = _read_json(transcript_json)
    if not isinstance(data, dict):
        raise SystemExit(f"transcript JSON must be an object: {transcript_json}")
    review_payload = _read_json(review_json)
    if not isinstance(review_payload, dict):
        raise SystemExit(f"review JSON must be an object: {review_json}")
    segments = [dict(seg) for seg in (data.get("segments") or []) if isinstance(seg, dict)]
    data["segments"] = segments

    decisions = _decision_by_id(review_payload)
    candidates = _candidate_by_id(review_payload)
    all_touched: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    decision_summaries: list[dict[str, Any]] = []
    for candidate_id, decision in decisions.items():
        action = str(decision.get("action") or "skip").strip()
        if action in {"", "skip", "review"}:
            continue
        candidate = candidates.get(candidate_id, {})
        kind = str(decision.get("kind") or candidate.get("kind") or "")
        if action == "unify":
            if kind not in APPLICABLE_KINDS or not bool(decision.get("allow_global_unify", kind in APPLICABLE_KINDS)):
                skipped.append({"id": candidate_id, "reason": "该候选不允许全局统一，请使用 replace_occurrences。"})
                continue
            canonical = str(decision.get("canonical_text") or "").strip()
            if not canonical:
                skipped.append({"id": candidate_id, "reason": "canonical_text 为空。"})
                continue
            terms = _validate_replace_terms(list(decision.get("replace_terms") or candidate.get("terms") or []), canonical)
            if not terms:
                skipped.append({"id": candidate_id, "reason": "replace_terms 为空或只包含标准写法。"})
                continue
            result = _apply_global_unify(segments, terms=terms, canonical=canonical, candidate_id=candidate_id)
        elif action == "replace_occurrences":
            result = _apply_occurrence_replacements(
                segments,
                replacements=[item for item in (decision.get("occurrence_replacements") or []) if isinstance(item, dict)],
                candidate_id=candidate_id,
            )
        else:
            skipped.append({"id": candidate_id, "reason": f"未知 action: {action}"})
            continue
        all_touched.extend(result["touched"])
        decision_summaries.append(
            {
                "id": candidate_id,
                "action": action,
                "kind": kind,
                "segments": result["segments"],
                "replacements": result["replacements"],
            }
        )

    stats = data.setdefault("filter_stats", {})
    if not isinstance(stats, dict):
        stats = {}
        data["filter_stats"] = stats
    stats[APPLY_MARKER] = {
        "source_json": str(transcript_json),
        "review_json": str(review_json),
        "decision_count": len(decisions),
        "applied_decisions": decision_summaries,
        "skipped": skipped,
        "replacement_count": sum(int(item.get("replacements") or 0) for item in decision_summaries),
        "touched_segment_count": len({int(item["index"]) for item in all_touched}),
    }
    data["entity_consistency_review"] = {
        "source_json": str(transcript_json),
        "review_json": str(review_json),
        "touched": all_touched,
        "skipped": skipped,
    }

    output = out_json or transcript_json.with_name(f"{transcript_json.stem}_实体统一.json")
    output = output.expanduser().resolve()
    _write_json(output, data)
    quality_paths: dict[str, str] = {}
    if write_quality:
        quality = _build_quality_for_transcript(data)
        quality_paths = write_asr_quality_reports(output.parent, output.stem, quality)
    return {
        "json": str(output),
        "quality": quality_paths,
        "replacement_count": stats[APPLY_MARKER]["replacement_count"],
        "touched_segment_count": stats[APPLY_MARKER]["touched_segment_count"],
        "skipped": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ASR 同音/近音实体核对清单导出与安全应用")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="导出实体一致性核对清单和确认模板")
    export_parser.add_argument("json_path", type=Path, help="转录 JSON")
    export_parser.add_argument("--quality-json", type=Path, default=None, help="可选 ASR质量检查.json; 默认读取转录同级")
    export_parser.add_argument("--out", type=Path, default=None, help=f"输出目录; 默认转录同级 {DEFAULT_OUT_NAME}/")
    export_parser.add_argument("--context-limit", type=int, default=8, help="每组候选最多展示多少个上下文")

    apply_parser = subparsers.add_parser("apply", help="按确认模板生成新的实体统一转录 JSON")
    apply_parser.add_argument("json_path", type=Path, help="原始转录 JSON")
    apply_parser.add_argument("--review-json", type=Path, required=True, help="已人工确认的 实体一致性核对清单.json")
    apply_parser.add_argument("--out-json", type=Path, default=None, help="输出新转录 JSON; 默认写到原文件同级 *_实体统一.json")
    apply_parser.add_argument("--no-quality", action="store_true", help="不重建 ASR质量检查.json/md")

    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_review(
            args.json_path.expanduser().resolve(),
            quality_json=args.quality_json.expanduser().resolve() if args.quality_json else None,
            out_dir=args.out.expanduser().resolve() if args.out else None,
            context_limit=max(int(args.context_limit), 1),
        )
    else:
        result = apply_review(
            args.json_path.expanduser().resolve(),
            args.review_json.expanduser().resolve(),
            out_json=args.out_json.expanduser().resolve() if args.out_json else None,
            write_quality=not args.no_quality,
        )
    print(json.dumps({"ok": True, **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
