from __future__ import annotations

import sys
import json
from pathlib import Path
from types import SimpleNamespace

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from scribe_py.core.types import Segment
from scribe_py.reviewers import asr_llm_review
from scribe_py.reviewers.asr_llm_review import build_review_items, review_asr_segments


def test_build_review_items_only_uses_locally_flagged_segments_with_context():
    segments = [
        Segment(start=0.0, end=1.0, text="前一句。"),
        Segment(start=1.0, end=2.0, text="管有关个地。"),
        Segment(start=2.0, end=3.0, text="后一句。"),
        Segment(start=3.0, end=4.0, text="不相关。"),
    ]
    review_segments = [
        {
            "index": 1,
            "start": 1.0,
            "end": 2.0,
            "text": "管有关个地。",
            "original_text": "管有关个地",
            "reasons": ["命中明显不通顺 ASR 片段"],
        }
    ]

    items = build_review_items(segments, review_segments, context_window=1)

    assert len(items) == 1
    assert items[0]["index"] == 1
    assert items[0]["text"] == "管有关个地。"
    assert items[0]["source_segment_index"] == 1
    assert items[0]["local_reasons"] == ["命中明显不通顺 ASR 片段"]
    assert [ctx["index"] for ctx in items[0]["context"]] == [0, 1, 2]


class _FakeCompletions:
    def __init__(self, response: dict, captured: dict):
        self.response = response
        self.captured = captured

    def create(self, **kwargs):
        self.captured["request"] = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(self.response, ensure_ascii=False))
                )
            ]
        )


class _FakeOpenAI:
    response: dict = {}
    captured: dict = {}

    def __init__(self, *, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(self.response, self.captured)
        )


def test_review_asr_segments_sends_only_selected_segments_and_context(monkeypatch):
    segments = [
        Segment(start=0.0, end=1.0, text="前一句。"),
        Segment(start=1.0, end=2.0, text="管有关个地。"),
        Segment(start=2.0, end=3.0, text="后一句。"),
        Segment(start=3.0, end=4.0, text="全文里未选中的内容。"),
    ]
    review_segments = [
        {"index": 1, "start": 1, "end": 2, "text": "管有关个地。", "reasons": ["疑似明显语义不顺"]}
    ]
    _FakeOpenAI.response = {
        "items": [
            {
                "index": 1,
                "start": 1,
                "end": 2,
                "original_text": "管有关个地。",
                "suggested_text": "管好有关各地。",
                "confidence": "high",
                "need_human_review": False,
                "reason": "上下文支持该修正",
            }
        ]
    }
    _FakeOpenAI.captured = {}
    monkeypatch.setattr(asr_llm_review, "OpenAI", _FakeOpenAI)

    result = review_asr_segments(
        segments,
        review_segments,
        api_key="test-key",
        context_window=1,
    )

    request = _FakeOpenAI.captured["request"]
    payload = json.loads(request["messages"][1]["content"])
    assert len(payload["items"]) == 1
    assert payload["items"][0]["index"] == 1
    assert [ctx["index"] for ctx in payload["items"][0]["context"]] == [0, 1, 2]
    assert "全文里未选中的内容" not in request["messages"][1]["content"]
    assert result.changed_high_confidence == 1
    assert result.suggestions == [
        {
            "index": 1,
            "source_segment_index": 1,
            "start": 1.0,
            "end": 2.0,
            "original_text": "管有关个地。",
            "current_text": "管有关个地。",
            "suggested_text": "管好有关各地。",
            "confidence": "high",
            "need_human_review": False,
            "reason": "上下文支持该修正",
            "local_reasons": ["疑似明显语义不顺"],
        }
    ]


def test_review_asr_segments_does_not_modify_low_confidence_suggestions(monkeypatch):
    segments = [
        Segment(start=0.0, end=1.0, text="前一句。"),
        Segment(start=1.0, end=2.0, text="疑点原文。"),
    ]
    review_segments = [
        {"index": 1, "start": 1, "end": 2, "text": "疑点原文。", "reasons": ["疑似明显语义不顺"]}
    ]
    _FakeOpenAI.response = {
        "items": [
            {
                "index": 1,
                "start": 1,
                "end": 2,
                "original_text": "疑点原文。",
                "suggested_text": "模型低置信乱改。",
                "confidence": "low",
                "need_human_review": False,
                "reason": "无法确认",
            }
        ]
    }
    _FakeOpenAI.captured = {}
    monkeypatch.setattr(asr_llm_review, "OpenAI", _FakeOpenAI)

    result = review_asr_segments(segments, review_segments, api_key="test-key")

    assert result.changed_high_confidence == 0
    assert result.suggestions[0]["suggested_text"] == "疑点原文。"
    assert result.suggestions[0]["confidence"] == "low"
    assert result.suggestions[0]["need_human_review"] is True
    assert result.suggestions[0]["reason"] == "无法确认"
    assert result.suggestions[0]["source_segment_index"] == 1
