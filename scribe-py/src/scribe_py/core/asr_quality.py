"""Local ASR quality signals.

This is not a substitute for ground-truth CER/WER.  It gives every transcript a
repeatable local check: punctuation, simplified Chinese, flagged ASR risks, and
customer hotword coverage.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .term_consistency import find_term_consistency_candidates
from .types import Segment


_PUNCT_RE = re.compile(r"[，。！？；：、,.!?;:]")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LEXICAL_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]")
_ONLY_PUNCT_RE = re.compile(r"^[\s，。！？；：、,.!?;:…-]+$")
_REPEATED_WORD_RE = re.compile(r"([\u4e00-\u9fff]{2,4})\1")
_REPEATED_CHAR_RE = re.compile(r"([\u4e00-\u9fff])\1{3,}")
_TRAD_CHARS = set(
    "聖誕節當們將會現這個嗎為聽講認況裡讓與對說辦過還應該點樣實問題發後師愛氣團禱導衝憐憫處響協調緒數標準錄"
    "進學時狀麼讚來參開關係經體親專業張課萬無話長間從憑兩條質幫動"
)

_STRONG_SUSPICION_REASONS = {
    "空文本",
    "疑似明显语义不顺",
    "疑似常见 ASR 混淆",
    "疑似断句导致词语断裂",
    "长时间低文本密度",
    "语速密度异常偏高",
    "独立模型转写存在分歧",
}
_WEAK_REVIEW_REASONS = {
    "只有标点/空白",
    "疑似重复词",
    "疑似重复字",
    "疑似重复口头词/重复识别",
    "长句缺少标点",
}
_STRONG_REASON_KEYWORDS = (
    "ASR 混淆",
    "ASR 易混淆",
    "明显不通顺",
    "语义不顺",
    "词语断裂",
    "不自然口语",
    "相关混淆",
    "人名混淆",
    "已确认",
    "低文本密度",
    "密度异常",
    "已应用",
)


def _fmt_ts(seconds: float) -> str:
    ms = int(round(float(seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _hotword_candidates(term: str, text: str, *, limit: int = 3) -> list[dict[str, Any]]:
    """Find near matches for a missing hotword without adding a heavy NLP dep."""
    term = term.strip()
    if len(term) < 2:
        return []
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return []

    scored: list[tuple[float, str]] = []
    min_len = max(2, len(term) - 1)
    max_len = min(len(term) + 1, max(len(term), 8))
    for size in range(min_len, max_len + 1):
        if size > len(compact):
            continue
        for i in range(0, len(compact) - size + 1):
            candidate = compact[i : i + size]
            if not _CJK_RE.search(candidate):
                continue
            score = SequenceMatcher(None, term, candidate).ratio()
            if len(term) <= 2:
                threshold = 0.5
            else:
                threshold = 0.62
            if score >= threshold and candidate != term:
                scored.append((score, candidate))

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for score, candidate in sorted(scored, key=lambda item: item[0], reverse=True):
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append({"text": candidate, "similarity": round(score, 3)})
        if len(out) >= limit:
            break
    return out


def _traditional_hits(text: str) -> list[str]:
    """Detect traditional Chinese with zhconv when available, then fallback map.

    The fixed character set is only a guardrail.  Packaged builds must include
    zhconv, otherwise less common characters such as "進/學/狀" can slip through.
    """
    hits = {ch for ch in text if ch in _TRAD_CHARS}
    try:
        from .text_normalizer import _to_simplified

        simplified = _to_simplified(text)
        if len(simplified) == len(text):
            hits.update(
                ch
                for ch, converted in zip(text, simplified)
                if ch != converted and _CJK_RE.fullmatch(ch)
            )
        else:
            for idx, ch in enumerate(text):
                if not _CJK_RE.fullmatch(ch):
                    continue
                start = max(0, idx - 4)
                window = text[start:idx + 8]
                converted = _to_simplified(window)
                rel = idx - start
                if len(converted) == len(window) and rel < len(converted) and converted[rel] != ch:
                    hits.add(ch)
    except Exception:
        pass
    return sorted(hits)


def _segment_suspicion_reasons(seg: Segment) -> list[str]:
    text = seg.text or ""
    stripped = text.strip()
    reasons: list[str] = []
    chars = len(_CJK_RE.findall(text))
    duration = max(float(seg.end or 0) - float(seg.start or 0), 0.001)

    if not stripped:
        if seg.original_text and not _LEXICAL_RE.search(seg.original_text):
            return []
        return ["空文本"]
    if stripped == "（非语音）":
        return []
    if _ONLY_PUNCT_RE.fullmatch(stripped):
        return ["只有标点/空白"]

    if chars >= 10 and _REPEATED_WORD_RE.search(text):
        reasons.append("疑似重复词")
    if _REPEATED_CHAR_RE.search(text):
        reasons.append("疑似重复字")
    if duration >= 8 and chars <= 6:
        reasons.append("长时间低文本密度")
    if duration >= 6 and chars / duration >= 7.5:
        reasons.append("语速密度异常偏高")
    if chars >= 25 and not _PUNCT_RE.search(text):
        reasons.append("长句缺少标点")

    return reasons


def _merge_review_segments(
    segments: list[Segment],
    existing: list[dict[str, Any]],
    *,
    limit: int = 40,
) -> tuple[list[dict[str, Any]], int, int, int]:
    by_index: dict[int, dict[str, Any]] = {}
    fallback: list[dict[str, Any]] = []
    for item in existing:
        try:
            index = int(item.get("index"))
        except Exception:
            fallback.append(item)
            continue
        if 0 <= index < len(segments):
            by_index[index] = {
                **item,
                "index": index,
                "reasons": [str(x) for x in (item.get("reasons") or [])],
            }
        else:
            fallback.append(item)

    generic_count = 0
    strong_generic_count = 0
    for idx, seg in enumerate(segments):
        reasons = _segment_suspicion_reasons(seg)
        if idx > 0:
            prev = segments[idx - 1]
            if _has_suspicious_boundary_split(prev, seg):
                reasons.append("疑似断句导致词语断裂")
        if not reasons:
            continue
        generic_count += 1
        if any(reason in _STRONG_SUSPICION_REASONS for reason in reasons):
            strong_generic_count += 1
        current = by_index.get(idx)
        if current is None:
            by_index[idx] = {
                "index": idx,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "original_text": seg.original_text or seg.text,
                "reasons": reasons,
            }
            continue
        merged_reasons = list(current.get("reasons") or [])
        for reason in reasons:
            if reason not in merged_reasons:
                merged_reasons.append(reason)
        current["reasons"] = merged_reasons

    merged = sorted(by_index.values(), key=lambda item: (float(item.get("start") or 0), int(item.get("index") or 0)))
    if fallback:
        merged.extend(fallback)
    strong_merged_count = sum(1 for item in merged if is_strong_asr_review_item(item))
    return merged[:limit], generic_count, strong_generic_count, strong_merged_count


def _model_disagreement_review_segments(
    segments: list[Segment],
    model_review: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Project model disagreement windows onto primary timeline segments."""
    review = dict(model_review or {})
    diagnostics = list(review.get("window_diagnostics") or [])
    selected: dict[int, dict[str, Any]] = {}
    for diagnostic in diagnostics:
        try:
            start = float(diagnostic.get("start"))
            end = float(diagnostic.get("end"))
            similarity = float(diagnostic.get("primary_para_similarity"))
        except (TypeError, ValueError):
            continue
        qwen_eligible = bool(diagnostic.get("qwen_review_eligible"))
        unresolved_qwen_disagreement = (
            qwen_eligible and int(diagnostic.get("confirmed_change_count") or 0) == 0
        )
        qwen_unavailable = str(diagnostic.get("qwen_skipped_reason") or "") == "qwen_runtime_unavailable"
        audit_only = bool(diagnostic.get("audit_only"))
        if end <= start or not (
            unresolved_qwen_disagreement
            or audit_only
            or (qwen_unavailable and similarity < 0.75)
        ):
            continue
        for index, segment in enumerate(segments):
            overlap = min(float(segment.end), end) - max(float(segment.start), start)
            if overlap <= 0.0:
                continue
            if not (segment.text or "").strip() or (segment.text or "").strip() == "（非语音）":
                continue
            selected[index] = {
                "index": index,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "original_text": segment.original_text or segment.text,
                "reasons": ["独立模型转写存在分歧"],
                "model_disagreement": {
                    "primary_para_similarity": round(similarity, 4),
                    "qwen_review_eligible": qwen_eligible,
                    "qwen_skipped_reason": str(diagnostic.get("qwen_skipped_reason") or ""),
                },
            }
    return [selected[index] for index in sorted(selected)]


def _has_suspicious_boundary_split(prev: Segment, seg: Segment) -> bool:
    prev_text = (prev.text or "").strip()
    text = (seg.text or "").strip()
    if not prev_text or not text:
        return False
    if float(seg.start or 0) - float(prev.end or 0) > 0.8:
        return False
    if len(text) <= 8 and re.match(r"^[\u4e00-\u9fff][的地得之和与及在是了着过]", text):
        prev_body = prev_text.rstrip(_ENDING_PUNCT_FOR_BOUNDARY)
        if prev_body and _CJK_RE.search(prev_body[-1]):
            return True
    return False


_ENDING_PUNCT_FOR_BOUNDARY = "，。！？；：、,.!?;:…"


def is_strong_asr_review_reason(reason: str) -> bool:
    """Return whether a local ASR reason is worth second-pass verification."""
    reason = str(reason or "").strip()
    if not reason or reason in _WEAK_REVIEW_REASONS:
        return False
    if "重复" in reason and not any(token in reason for token in ("混淆", "语义", "不通顺")):
        return False
    if reason in _STRONG_SUSPICION_REASONS:
        return True
    return any(token in reason for token in _STRONG_REASON_KEYWORDS)


def is_strong_asr_review_item(item: dict[str, Any]) -> bool:
    reasons = [str(x) for x in (item.get("reasons") or [])]
    return any(is_strong_asr_review_reason(reason) for reason in reasons)


def _review_segments_from_quality_report(data: dict[str, Any]) -> list[dict[str, Any]]:
    if data.get("mode") == "local_asr_quality":
        return list((data.get("review") or {}).get("segments") or [])
    quality = data.get("asr_quality") or {}
    if isinstance(quality, dict):
        return list((quality.get("review") or {}).get("segments") or [])
    return []


def _review_segments_from_transcript(data: dict[str, Any]) -> list[dict[str, Any]]:
    stats = (data.get("filter_stats") or {}).get("text_normalization") or {}
    review = list(stats.get("asr_review_segments") or [])
    review.extend(list(data.get("asr_review_segments") or []))
    review.extend(_review_segments_from_quality_report(data))
    return review


def _sidecar_quality_segments(transcript_json: Path | None) -> tuple[list[dict[str, Any]], str | None]:
    if transcript_json is None:
        return [], None
    candidates = [
        transcript_json.parent / f"{transcript_json.stem}_ASR质量检查.json",
        transcript_json.parent / "ASR质量检查.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        return _review_segments_from_quality_report(data), path.name
    return [], None


def _review_item_key(item: dict[str, Any]) -> tuple[Any, ...]:
    index = item.get("index")
    if index is not None:
        try:
            return ("index", int(index))
        except Exception:
            pass
    return (
        "time_text",
        round(float(item.get("start") or 0), 3),
        round(float(item.get("end") or 0), 3),
        str(item.get("text") or item.get("original_text") or ""),
    )


def _clean_review_item(item: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    for raw in item.get("reasons") or []:
        reason = str(raw).strip()
        if reason and reason not in reasons:
            reasons.append(reason)
    cleaned = dict(item)
    cleaned["reasons"] = reasons
    return cleaned


def _merge_review_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in items:
        if not isinstance(raw, dict):
            continue
        item = _clean_review_item(raw)
        key = _review_item_key(item)
        current = merged.get(key)
        if current is None:
            merged[key] = item
            continue
        for field in ("index", "start", "end", "text", "original_text"):
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
            float(item.get("start") or 0),
            int(item.get("index") if item.get("index") is not None else 1_000_000),
        ),
    )


def select_asr_review_segments(
    transcript_data: dict[str, Any],
    *,
    transcript_json: Path | None = None,
    scope: str = "strong",
) -> dict[str, Any]:
    """Select ASR review segments from transcript data and optional sidecar QA.

    Default scope is intentionally conservative: only strong ASR doubts are
    returned for LLM/local recheck. Weak spot-check markers stay visible in the
    quality report but do not trigger extra processing by default.
    """
    normalized_scope = str(scope or "strong").strip().lower()
    if normalized_scope not in {"strong", "all"}:
        raise ValueError("scope must be 'strong' or 'all'")

    sources = ["transcript"]
    raw_items = _review_segments_from_transcript(transcript_data)
    sidecar_items, sidecar_name = _sidecar_quality_segments(transcript_json)
    if sidecar_name:
        sources.append(sidecar_name)
        raw_items.extend(sidecar_items)

    all_items = _merge_review_items(raw_items)
    strong_items = [item for item in all_items if is_strong_asr_review_item(item)]
    selected = strong_items if normalized_scope == "strong" else all_items
    weak_count = max(len(all_items) - len(strong_items), 0)
    return {
        "scope": normalized_scope,
        "sources": sources,
        "segments": selected,
        "total_segment_count": len(all_items),
        "strong_segment_count": len(strong_items),
        "weak_segment_count": weak_count,
        "skipped_weak_count": weak_count if normalized_scope == "strong" else 0,
    }


def build_asr_quality_report(
    segments: list[Segment],
    *,
    hotwords: list[str] | None = None,
    text_normalization: dict[str, Any] | None = None,
    model_review: dict[str, Any] | None = None,
    audio_quality: dict[str, Any] | None = None,
    audio_preprocessing: dict[str, Any] | None = None,
    backend: str = "",
    model_id: str = "",
    duration: float = 0.0,
    transcribe_seconds: float = 0.0,
    rtf: float = 0.0,
) -> dict[str, Any]:
    text = "\n".join(seg.text for seg in segments if seg.text)
    chars = len(re.sub(r"\s+", "", text))
    cjk_chars = len(_CJK_RE.findall(text))
    punctuation_segments = sum(1 for seg in segments if _PUNCT_RE.search(seg.text or ""))
    punctuation_ratio = round(punctuation_segments / len(segments), 4) if segments else 0.0
    traditional_hits = _traditional_hits(text)

    terms = []
    for term in hotwords or []:
        if not term or term in terms:
            continue
        terms.append(term)
    exact_hits = [term for term in terms if term in text]
    missing_terms = [term for term in terms if term not in text]
    near_misses = [
        {"term": term, "candidates": candidates}
        for term in missing_terms
        if (candidates := _hotword_candidates(term, text))
    ]
    hotword_coverage = round(len(exact_hits) / len(terms), 4) if terms else None

    review_segments = []
    review_count = 0
    if text_normalization:
        review_count = int(text_normalization.get("asr_review_segment_count") or 0)
        review_segments = list(text_normalization.get("asr_review_segments") or [])
    review_segments.extend(_model_disagreement_review_segments(segments, model_review))
    review_segments, generic_review_count, strong_generic_review_count, strong_review_count = _merge_review_segments(segments, review_segments)
    review_count = max(review_count, generic_review_count, len(review_segments))
    review_ratio = (review_count / len(segments)) if segments else 0.0
    strong_review_ratio = (strong_review_count / len(segments)) if segments else 0.0
    term_consistency = find_term_consistency_candidates(segments)
    term_candidate_count = int(term_consistency.get("candidate_count") or 0)

    risk_reasons: list[str] = []
    spot_check_reasons: list[str] = []
    audio_quality = audio_quality or {}
    audio_preprocessing = audio_preprocessing or {}
    audio_risk_level = str(audio_quality.get("risk_level") or "unknown")
    audio_risk_reasons = [str(x) for x in (audio_quality.get("risk_reasons") or [])]
    if not segments:
        risk_reasons.append("没有识别出有效文本")
    if audio_risk_level == "high":
        risk_reasons.append("音频质量高风险")
    elif audio_risk_level == "medium":
        risk_reasons.append("音频质量存在风险")
    if punctuation_ratio < 0.8 and len(segments) >= 5:
        risk_reasons.append("标点覆盖率偏低")
    if traditional_hits:
        risk_reasons.append("仍存在繁体字")
    if strong_generic_review_count:
        risk_reasons.append("存在疑似语义/异常片段")
    if generic_review_count > strong_generic_review_count:
        spot_check_reasons.append("存在重复/口语抽查片段")
    if strong_review_count >= 20 or (strong_review_count >= 3 and strong_review_ratio >= 0.25):
        risk_reasons.append("ASR 疑点段较多")
    elif strong_review_count >= 8 or (strong_review_count >= 2 and strong_review_ratio >= 0.12):
        risk_reasons.append("存在多处 ASR 疑点段")
    if terms and hotword_coverage is not None:
        if hotword_coverage < 0.5:
            risk_reasons.append("客户热词命中率偏低")
        elif missing_terms:
            risk_reasons.append("部分客户热词未命中")
    if term_candidate_count:
        spot_check_reasons.append("存在疑似同音/近音实体一致性候选")
    if audio_preprocessing.get("enabled") and not audio_preprocessing.get("applied"):
        risk_reasons.append("音频预处理未成功应用")

    if not risk_reasons:
        risk_level = "low"
        recommendation = "可进入纪要/分人流程；建议抽听少量关键片段。"
    elif any(reason in risk_reasons for reason in {"没有识别出有效文本", "客户热词命中率偏低", "ASR 疑点段较多", "音频质量高风险", "音频预处理未成功应用"}):
        risk_level = "high"
        recommendation = "先复核疑点段和热词缺失项，再进入纪要/分人流程。"
    else:
        risk_level = "medium"
        recommendation = "建议抽听疑点段；热词缺失如为重要人名/术语，需要补充词表后重跑。"

    industry_pipeline = _industry_pipeline_summary(
        hotword_count=len(terms),
        strong_review_count=strong_review_count,
        audio_quality=audio_quality,
        audio_preprocessing=audio_preprocessing,
    )

    return {
        "mode": "local_asr_quality",
        "backend": backend,
        "model_id": model_id,
        "duration_s": duration,
        "transcribe_seconds": transcribe_seconds,
        "rtf": rtf,
        "segments": len(segments),
        "chars": chars,
        "cjk_chars": cjk_chars,
        "punctuation_ratio": punctuation_ratio,
        "traditional_char_hits": traditional_hits,
        "hotwords": {
            "count": len(terms),
            "exact_hit_count": len(exact_hits),
            "coverage": hotword_coverage,
            "exact_hits": exact_hits,
            "missing_terms": missing_terms,
            "near_misses": near_misses,
        },
        "term_consistency": term_consistency,
        "audio_quality": audio_quality,
        "audio_preprocessing": audio_preprocessing,
        "industry_pipeline": industry_pipeline,
        "review": {
            "segment_count": review_count,
            "segment_ratio": round(review_ratio, 4),
            "generic_segment_count": generic_review_count,
            "strong_segment_count": strong_review_count,
            "strong_segment_ratio": round(strong_review_ratio, 4),
            "segments": review_segments,
        },
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "spot_check_reasons": spot_check_reasons,
        "recommendation": recommendation,
    }


def _industry_pipeline_summary(
    *,
    hotword_count: int,
    strong_review_count: int,
    audio_quality: dict[str, Any],
    audio_preprocessing: dict[str, Any],
) -> dict[str, Any]:
    preprocessing_filters = [str(x) for x in (audio_preprocessing.get("applied_filters") or [])]
    preprocessing_status = "未启用"
    if audio_preprocessing.get("enabled"):
        preprocessing_status = "已应用" if audio_preprocessing.get("applied") else "未成功"
    steps = [
        {
            "step": "音频质量门禁",
            "status": "已启用" if audio_quality else "未记录",
            "detail": f"风险={audio_quality.get('risk_level', 'unknown')}" if audio_quality else "未取得音频质量数据",
        },
        {
            "step": "音频标准化/响度均衡",
            "status": preprocessing_status,
            "detail": "、".join(preprocessing_filters) if preprocessing_filters else "未应用滤镜",
        },
        {
            "step": "客户热词/术语",
            "status": "已配置" if hotword_count else "未配置",
            "detail": f"{hotword_count} 个热词" if hotword_count else "无客户词表时只做通用识别",
        },
        {
            "step": "本地疑点二次核验入口",
            "status": "可用",
            "detail": f"{strong_review_count} 个强疑点可进入 review-asr / asr_local_recheck",
        },
        {
            "step": "人工 CER 回归评分",
            "status": "可用",
            "detail": "填写 correct_text 后用 scripts/asr_gold_score.py 量化准确率",
        },
    ]
    return {
        "source": "industry-inspired-local-pipeline",
        "principle": "先保护转文字准确率,再做分人和纪要; 本地规则默认不依赖大模型。",
        "steps": steps,
    }


def render_asr_quality_markdown(report: dict[str, Any]) -> str:
    hotwords = report.get("hotwords") or {}
    term_consistency = report.get("term_consistency") or {}
    review = report.get("review") or {}
    audio_quality = report.get("audio_quality") or {}
    audio_preprocessing = report.get("audio_preprocessing") or {}
    industry_pipeline = report.get("industry_pipeline") or {}
    lines = [
        "# ASR 质量检查\n\n",
        f"- 风险等级: {report.get('risk_level', '')}\n",
        f"- 建议: {report.get('recommendation', '')}\n",
        f"- 后端/模型: {report.get('backend', '')} / {report.get('model_id', '')}\n",
        f"- 时长: {float(report.get('duration_s') or 0):.1f}s\n",
        f"- 转录耗时: {float(report.get('transcribe_seconds') or 0):.1f}s\n",
        f"- RTF: {float(report.get('rtf') or 0):.3f}\n",
        f"- 段数/字数: {report.get('segments', 0)} / {report.get('chars', 0)}\n",
        f"- 标点覆盖率: {float(report.get('punctuation_ratio') or 0):.1%}\n",
        f"- 繁体字命中: {len(report.get('traditional_char_hits') or [])}\n",
    ]
    if audio_quality:
        audio_reasons = audio_quality.get("risk_reasons") or []
        lines.extend([
            f"- 音频质量风险: {audio_quality.get('risk_level', 'unknown')}\n",
            f"- 音频响度/峰值: {_number_or_dash(audio_quality.get('integrated_lufs'))} LUFS / {_number_or_dash(audio_quality.get('true_peak_dbfs'))} dBFS\n",
            f"- 静音占比: {_percent_or_dash(audio_quality.get('silence_ratio'))}\n",
        ])
        if audio_reasons:
            lines.append(f"- 音频质量原因: {'；'.join(str(x) for x in audio_reasons)}\n")
    if audio_preprocessing:
        filters = audio_preprocessing.get("applied_filters") or []
        skipped = audio_preprocessing.get("skipped_actions") or []
        lines.extend([
            f"- 音频预处理: {audio_preprocessing.get('mode', 'unknown')} / {'已应用' if audio_preprocessing.get('applied') else '未应用'}\n",
            f"- 预处理滤镜: {'、'.join(str(x) for x in filters) if filters else '-'}\n",
        ])
        if skipped:
            lines.append(f"- 预处理跳过项: {'；'.join(str(x) for x in skipped)}\n")
    spot_check_reasons = report.get("spot_check_reasons") or []
    if spot_check_reasons:
        lines.append(f"- 抽查提示: {'；'.join(str(x) for x in spot_check_reasons)}\n")
    steps = industry_pipeline.get("steps") or []
    if steps:
        lines.extend([
            "\n",
            "## 工业化转录链路\n\n",
            f"{industry_pipeline.get('principle', '')}\n\n",
            "| 环节 | 状态 | 说明 |\n",
            "|---|---|---|\n",
        ])
        for step in steps:
            lines.append(
                f"| {str(step.get('step', '')).replace('|', '\\|')} | "
                f"{str(step.get('status', '')).replace('|', '\\|')} | "
                f"{str(step.get('detail', '')).replace('|', '\\|')} |\n"
            )
    lines.extend([
        "\n",
        "## 热词命中\n\n",
        f"- 热词数: {hotwords.get('count', 0)}\n",
        f"- 精确命中: {hotwords.get('exact_hit_count', 0)}\n",
        f"- 命中率: {_percent_or_dash(hotwords.get('coverage'))}\n",
    ])
    missing = hotwords.get("missing_terms") or []
    if missing:
        lines.append(f"- 未命中: {'、'.join(missing)}\n")
    near_misses = hotwords.get("near_misses") or []
    if near_misses:
        lines.extend(["\n", "| 热词 | 疑似识别成 | 相似度 |\n", "|---|---|---:|\n"])
        for item in near_misses:
            term = str(item.get("term") or "")
            for cand in item.get("candidates") or []:
                lines.append(f"| {term} | {cand.get('text', '')} | {cand.get('similarity', '')} |\n")

    lines.extend([
        "\n",
        "## 同音/近音实体一致性\n\n",
        f"- 候选组数: {term_consistency.get('candidate_count', 0)}\n",
        "- 说明: 只标注相同/近似读音的实体写法或疑似实体偏离，建议用户确认标准写法；系统不自动替换原文。\n",
    ])
    candidates = term_consistency.get("candidates") or []
    if candidates:
        lines.extend(["\n", "| 类型 | 动作 | 置信度 | 候选词 | 读音键 | 建议主写法 | 出现次数 | 原因 |\n", "|---|---|---:|---|---|---|---:|---|\n"])
        for item in candidates[:20]:
            terms = "、".join(str(x) for x in (item.get("terms") or []))
            reason = str(item.get("reason") or "").replace("|", "\\|")
            lines.append(
                f"| {_term_candidate_kind_label(str(item.get('kind') or 'orthographic_term'))} | "
                f"{item.get('action', '')} | "
                f"{float(item.get('confidence') or 0):.3f} | "
                f"{terms.replace('|', '\\|')} | "
                f"{str(item.get('phonetic_key') or '-').replace('|', '\\|')} | "
                f"{str(item.get('suggested_canonical') or '-').replace('|', '\\|')} | "
                f"{int(item.get('total_count') or 0)} | "
                f"{reason} |\n"
            )
    else:
        lines.append("\n无明显同音/近音实体一致性候选。\n")

    lines.extend([
        "\n",
        "## 疑点段\n\n",
        f"- 本地疑点段数: {review.get('segment_count', 0)}\n",
        f"- 疑点段占比: {_percent_or_dash(review.get('segment_ratio'))}\n",
        f"- 强疑点段数: {review.get('strong_segment_count', 0)}\n",
        f"- 强疑点占比: {_percent_or_dash(review.get('strong_segment_ratio'))}\n",
    ])
    segments = review.get("segments") or []
    if segments:
        lines.extend(["\n", "| 时间 | 原因 | 文本 |\n", "|---|---|---|\n"])
        for seg in segments[:40]:
            ts = f"{_fmt_ts(float(seg.get('start') or 0))}-{_fmt_ts(float(seg.get('end') or 0))}"
            reasons = "；".join(str(x) for x in (seg.get("reasons") or []))
            text = str(seg.get("text") or "").replace("|", "\\|")
            lines.append(f"| {ts} | {reasons} | {text} |\n")
    else:
        lines.append("\n无本地疑点段。\n")
    return "".join(lines)


def _percent_or_dash(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1%}"


def _term_candidate_kind_label(kind: str) -> str:
    return {
        "phonetic_entity": "同音实体",
        "entity_drift": "实体漂移",
        "orthographic_term": "字形相近",
    }.get(kind, kind or "-")


def _number_or_dash(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}"


def write_asr_quality_reports(
    out_dir: Path,
    stem: str,
    report: dict[str, Any],
    *,
    per_transcript: bool = False,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{stem}_" if per_transcript else ""
    json_path = out_dir / f"{prefix}ASR质量检查.json"
    md_path = out_dir / f"{prefix}ASR质量检查.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_asr_quality_markdown(report), encoding="utf-8")
    return {"asr_quality_json": str(json_path), "asr_quality_md": str(md_path)}
