from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scribe_py.core.selector import default_model_id, make_transcriber
from scribe_py.core.transcriber_qwen3 import (
    DEFAULT_MODEL,
    Qwen3ASRTranscriber,
    _language_name,
    _split_chunk_text,
)
from scribe_py.core.types import TranscribeOptions


def test_qwen3_backend_is_explicit_and_has_own_default_model():
    assert default_model_id("qwen3") == DEFAULT_MODEL
    assert isinstance(make_transcriber("qwen3"), Qwen3ASRTranscriber)


def test_qwen3_language_mapping_prefers_chinese_for_zh():
    assert _language_name("zh") == "Chinese"
    assert _language_name("yue") == "Cantonese"


def test_qwen3_coarse_chunk_is_split_without_changing_words():
    segments = _split_chunk_text("第一句。第二句有术语 Redis！", 10.0, 16.0)

    assert [item.text for item in segments] == ["第一句。", "第二句有术语 Redis！"]
    assert segments[0].start == 10.0
    assert segments[-1].end == 16.0


def test_qwen3_run_records_text_only_timing_limit(monkeypatch):
    class FakeModel:
        def generate(self, audio, **kwargs):
            assert audio == "/tmp/example.wav"
            assert kwargs["language"] == "Chinese"
            assert kwargs["repetition_penalty"] == 1.1
            assert kwargs["max_tokens"] >= 256
            return SimpleNamespace(
                text="第一句。第二句。",
                segments=[{"text": "第一句。第二句。", "start": 0.0, "end": 4.0}],
                prompt_tokens=10,
                generation_tokens=8,
                total_time=1.25,
            )

    transcriber = Qwen3ASRTranscriber()
    monkeypatch.setattr(transcriber, "_load", lambda model_id: FakeModel())
    segments, language = transcriber._run(
        Path("/tmp/example.wav"),
        TranscribeOptions(language="zh", model_id=DEFAULT_MODEL),
        None,
    )

    assert language == "zh"
    assert [item.text for item in segments] == ["第一句。", "第二句。"]
    assert transcriber.last_filter_stats["timing_reliable"] is False
    assert transcriber.last_filter_stats["generation_tokens"] == 8
    assert transcriber.last_filter_stats["has_hallucination_risk"] is False
