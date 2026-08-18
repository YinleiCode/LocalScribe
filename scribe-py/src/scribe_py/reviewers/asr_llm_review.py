"""LLM review for locally flagged ASR risk segments.

This module does not re-polish the whole transcript.  It sends only segments
flagged by the local normalizer plus small neighboring context, and asks the LLM
for auditable suggestions with confidence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from ..core.types import Segment


REVIEW_SYSTEM_PROMPT_ZH = """你是中文 ASR 转写质检员。你只做二次核验,不要润色全文。

任务:
1. 根据给出的疑点片段、前后文、可选项目词表,判断 ASR 文本是否可能听错。
2. 如果能从上下文高置信判断,给出建议修正;如果不确定,保留原文并标记 need_human_review=true。
3. 不要编造音频里没有的内容;不要扩写;不要总结;不要改说话人。
4. 必须输出简体中文,并保留每条 item 的 index/start/end。

置信度:
- high: 上下文明显支持,适合自动应用。
- medium: 很可能正确,建议人工看一眼。
- low: 无法确认,只标注不修改。

输出严格 JSON:
{
  "items": [
    {
      "index": int,
      "source_segment_index": int,
      "start": number,
      "end": number,
      "original_text": "原文",
      "suggested_text": "建议文本或原文",
      "confidence": "high|medium|low",
      "need_human_review": true|false,
      "reason": "一句话说明"
    }
  ]
}"""


@dataclass
class AsrReviewResult:
    suggestions: list[dict[str, Any]]
    changed_high_confidence: int
    model: str


def _segment_context(segments: list[Segment], index: int, window: int) -> list[dict[str, Any]]:
    start = max(0, index - window)
    end = min(len(segments), index + window + 1)
    return [
        {
            "index": i,
            "start": segments[i].start,
            "end": segments[i].end,
            "text": segments[i].text,
            **({"speaker": segments[i].speaker} if segments[i].speaker else {}),
        }
        for i in range(start, end)
    ]


def build_review_items(
    segments: list[Segment],
    review_segments: list[dict[str, Any]],
    *,
    limit: int = 40,
    context_window: int = 2,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw in review_segments[:limit]:
        try:
            index = int(raw.get("index"))
        except Exception:
            continue
        if index < 0 or index >= len(segments):
            continue
        seg = segments[index]
        items.append(
            {
                "index": index,
                "source_segment_index": index,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "original_text": raw.get("original_text") or seg.original_text or seg.text,
                "local_reasons": raw.get("reasons") or [],
                "context": _segment_context(segments, index, context_window),
            }
        )
    return items


def review_asr_segments(
    segments: list[Segment],
    review_segments: list[dict[str, Any]],
    *,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-v4-flash",
    project_glossary: str = "",
    context_hint: str = "",
    limit: int = 40,
    context_window: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> AsrReviewResult:
    items = build_review_items(
        segments,
        review_segments,
        limit=limit,
        context_window=context_window,
    )
    if not items:
        return AsrReviewResult(suggestions=[], changed_high_confidence=0, model=model)

    payload = {
        "context_hint": context_hint,
        "project_glossary": project_glossary,
        "items": items,
    }
    client = OpenAI(api_key=api_key, base_url=base_url)
    rsp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT_ZH},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
    )
    data = json.loads(rsp.choices[0].message.content or "{}")
    suggestions = data.get("items") or []
    if not isinstance(suggestions, list):
        suggestions = []

    by_index = {int(item["index"]): item for item in items if "index" in item}
    clean: list[dict[str, Any]] = []
    changed_high = 0
    for raw in suggestions:
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("index"))
        except Exception:
            continue
        src = by_index.get(index)
        if not src:
            continue
        original_text = str(raw.get("original_text") or src["text"])
        suggested_text = str(raw.get("suggested_text") or original_text).strip() or original_text
        confidence = str(raw.get("confidence") or "low").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        need_human_review = bool(raw.get("need_human_review", confidence != "high"))
        if confidence == "low":
            suggested_text = src["text"]
            need_human_review = True
        if confidence == "high" and suggested_text != src["text"]:
            changed_high += 1
        clean.append(
            {
                "index": index,
                "source_segment_index": int(src["source_segment_index"]),
                "start": float(raw.get("start") or src["start"]),
                "end": float(raw.get("end") or src["end"]),
                "original_text": original_text,
                "current_text": src["text"],
                "suggested_text": suggested_text,
                "confidence": confidence,
                "need_human_review": need_human_review,
                "reason": str(raw.get("reason") or "").strip(),
                "local_reasons": src.get("local_reasons") or [],
            }
        )
    return AsrReviewResult(suggestions=clean, changed_high_confidence=changed_high, model=model)
