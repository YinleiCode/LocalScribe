from __future__ import annotations

from pathlib import Path

from scribe_py.core import qwen_lexical_review as review
from scribe_py.core.strong_asr import normalized_text, timeline_fingerprint
from scribe_py.core.types import Segment, TranscribeResult


def _segment(start: float, end: float, text: str, speaker: str) -> Segment:
    middle = (start + end) / 2
    return Segment(
        start=start,
        end=end,
        text=text,
        speaker=speaker,
        sync_cues=[
            {"start": start, "end": middle, "text": text[: max(1, len(text) // 2)]},
            {"start": middle, "end": end, "text": text[max(1, len(text) // 2) :]},
        ],
    )


def _result(text: str, *, hallucination: bool = False) -> TranscribeResult:
    return TranscribeResult(
        audio="clip.wav",
        language="zh",
        duration=5.0,
        transcribe_seconds=0.1,
        rtf=0.02,
        backend="qwen3",
        model_id="qwen-test",
        segments=[Segment(start=0.0, end=5.0, text=text)],
        filter_stats={"has_hallucination_risk": hallucination},
    )


def _prepare_run(monkeypatch, tmp_path: Path, fake_qwen_type):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(review, "_qwen_model_cached", lambda _model: True)
    monkeypatch.setattr(review, "Qwen3ASRTranscriber", fake_qwen_type)
    monkeypatch.setattr(
        review,
        "_extract_clip",
        lambda _audio, _start, _end, output: output.write_bytes(b"clip"),
    )
    return audio


def test_projection_preserves_complete_source_text_and_frozen_cue_timeline():
    primary = [
        _segment(0.0, 4.0, "原始第一段", "SPEAKER_A"),
        _segment(4.0, 8.0, "原始第二段", "SPEAKER_B"),
    ]
    source = [
        Segment(start=0.0, end=3.0, text="修正后的第一部分，"),
        Segment(start=3.0, end=8.0, text="以及完整的第二部分。"),
    ]
    before = timeline_fingerprint(primary)
    frozen = [
        (segment.start, segment.end, segment.speaker, [(cue["start"], cue["end"]) for cue in segment.sync_cues or []])
        for segment in primary
    ]

    output = review.project_aligned_text_to_frozen_timeline(primary, source)

    assert normalized_text("".join(segment.text for segment in output)) == normalized_text(
        "".join(segment.text for segment in source)
    )
    assert timeline_fingerprint(output) == before
    assert [
        (segment.start, segment.end, segment.speaker, [(cue["start"], cue["end"]) for cue in segment.sync_cues or []])
        for segment in output
    ] == frozen


def test_projection_assigns_non_overlapping_source_to_nearest_cue():
    primary = [
        Segment(
            start=0.0,
            end=7.0,
            text="原文",
            speaker="SPEAKER_A",
            sync_cues=[
                {"start": 0.0, "end": 2.0, "text": "原"},
                {"start": 5.0, "end": 7.0, "text": "文"},
            ],
        )
    ]
    source = [Segment(start=3.8, end=4.0, text="最近的文字")]

    output = review.project_aligned_text_to_frozen_timeline(primary, source)

    assert output[0].sync_cues[0]["text"] == ""
    assert output[0].sync_cues[1]["text"] == "最近的文字"
    assert normalized_text(output[0].text) == normalized_text(source[0].text)


def test_projection_keeps_aligned_short_cue_whole_across_small_boundary_sliver():
    primary = [
        Segment(
            start=0.0,
            end=4.1,
            text="原始材料",
            sync_cues=[{"start": 3.6, "end": 4.1, "text": "材料"}],
        ),
        Segment(
            start=4.1,
            end=6.0,
            text="下一段",
            sync_cues=[{"start": 4.1, "end": 6.0, "text": "下一段"}],
        ),
    ]
    source = [
        Segment(
            start=3.55,
            end=4.25,
            text="的材料。",
            sync_cues=[{"start": 3.55, "end": 4.25, "text": "的材料。"}],
        )
    ]

    output = review.project_aligned_text_to_frozen_timeline(primary, source)

    assert output[0].text == "的材料。"
    assert output[1].text == ""
    assert output[0].sync_cues[0]["start"] == 3.6
    assert output[0].sync_cues[0]["end"] == 4.1
    assert timeline_fingerprint(output) == timeline_fingerprint(primary)


def test_projection_moves_leading_punctuation_to_previous_frozen_cue():
    primary = [
        Segment(start=0.0, end=2.0, text="前文"),
        Segment(start=2.0, end=4.0, text="后文"),
    ]
    source = [
        Segment(start=0.0, end=2.0, text="前文"),
        Segment(start=2.0, end=4.0, text="，后文。"),
    ]

    output = review.project_aligned_text_to_frozen_timeline(primary, source)

    assert output[0].text == "前文，"
    assert output[1].text == "后文。"
    assert "".join(segment.text for segment in output) == "前文，后文。"
    assert timeline_fingerprint(output) == timeline_fingerprint(primary)


def test_paraformer_consensus_applies_supported_fix_and_rejects_qwen_only_rewrite():
    primary = [
        _segment(0.0, 10.0, "车辆发生了跨送之后报警。", "SPEAKER_A"),
        _segment(10.0, 20.0, "我们应该老来为伴互相照顾。", "SPEAKER_B"),
        _segment(20.0, 30.0, "他年纪大，我不想那么好，我。", "SPEAKER_C"),
    ]
    projected = [
        _segment(0.0, 10.0, "车辆发生了刮蹭之后报警。", "SPEAKER_A"),
        _segment(10.0, 20.0, "我们应该老来为难互相照顾。", "SPEAKER_B"),
        _segment(20.0, 30.0, "他们年纪大了不想跑，", "SPEAKER_C"),
    ]
    reference = [
        Segment(start=0.0, end=10.0, text="车辆发生了刮蹭之后报警。"),
        Segment(start=10.0, end=20.0, text="我们应该老来为伴互相照顾。"),
        Segment(start=20.0, end=30.0, text="他们年纪大了不想跑，"),
    ]
    before = timeline_fingerprint(primary)

    output, changes = review._apply_bounded_paraformer_consensus(
        primary,
        projected,
        reference,
    )

    assert output[0].text == "车辆发生了刮蹭之后报警。"
    assert output[1].text == "我们应该老来为伴互相照顾。"
    assert output[2].text == "他年纪大，我不想那么好，我。"
    assert len(changes) == 1
    assert changes[0]["changes"][0]["from"] == "跨送"
    assert changes[0]["changes"][0]["to"] == "刮蹭"
    assert timeline_fingerprint(output) == before
    assert output[0].speaker == "SPEAKER_A"
    assert output[1].speaker == "SPEAKER_B"
    assert output[2].speaker == "SPEAKER_C"


def test_run_rejects_low_similarity_and_hallucination_windows(monkeypatch, tmp_path: Path):
    primary = [
        _segment(0.0, 10.0, "这是第一段原始会议内容。", "SPEAKER_A"),
        _segment(200.0, 210.0, "这是第二段原始会议内容。", "SPEAKER_B"),
    ]

    class _Qwen:
        def __init__(self):
            self.calls = 0

        def transcribe(self, _audio, _options):
            self.calls += 1
            if self.calls == 1:
                return _result("完全不相关的天气预报文本。")
            return _result("这是第二段原始会议内容。", hallucination=True)

    audio = _prepare_run(monkeypatch, tmp_path, _Qwen)
    output, stats = review.run_qwen_lexical_review(audio, primary)

    assert output == primary
    assert stats["window_count"] == 2
    assert stats["accepted"] == 0
    assert stats["rejected"] == 2
    assert stats["changed"] == 0
    assert stats["timeline_preserved"] is True
    assert stats["window_diagnostics"][0]["reason"] == "alignment_rejected"
    assert stats["window_diagnostics"][1]["reason"] == "qwen_hallucination_risk"


def test_run_keeps_failed_window_and_accepts_other_window(monkeypatch, tmp_path: Path):
    primary = [
        _segment(0.0, 10.0, "会议开始以后请大家认真检察材料然后继续讨论。", "SPEAKER_A"),
        _segment(200.0, 210.0, "第二个窗口必须保留原始正文。", "SPEAKER_B"),
    ]
    instances = []

    class _Qwen:
        def __init__(self):
            self.calls = 0
            instances.append(self)

        def transcribe(self, _audio, _options):
            self.calls += 1
            if self.calls == 1:
                return _result("会议开始以后请大家认真检查材料然后继续讨论。")
            raise RuntimeError("second_window_failed")

    audio = _prepare_run(monkeypatch, tmp_path, _Qwen)
    before = timeline_fingerprint(primary)
    output, stats = review.run_qwen_lexical_review(audio, primary)

    assert len(instances) == 1
    assert instances[0].calls == 2
    assert "认真检查材料" in output[0].text
    assert output[1] == primary[1]
    assert output[0].speaker == "SPEAKER_A"
    assert timeline_fingerprint(output) == before
    assert stats["window_count"] == 2
    assert stats["accepted"] == 1
    assert stats["rejected"] == 1
    assert stats["changed"] == 1
    assert stats["timeline_preserved"] is True


def test_run_uses_cached_reference_to_review_only_disagreement_windows(monkeypatch, tmp_path: Path):
    primary = [
        _segment(0.0, 10.0, "第一窗口文字完全一致无需复核。", "SPEAKER_A"),
        _segment(200.0, 210.0, "第二窗口存在明显错误需要复核。", "SPEAKER_B"),
    ]
    reference = [
        Segment(start=0.0, end=10.0, text="第一窗口文字完全一致无需复核。"),
        Segment(start=200.0, end=210.0, text="第二窗口内容差异很大必须重新检查。"),
    ]
    instances = []

    class _Qwen:
        def __init__(self):
            self.calls = 0
            instances.append(self)

        def transcribe(self, _audio, _options):
            self.calls += 1
            return _result("第二窗口存在明显问题需要复核。")

    audio = _prepare_run(monkeypatch, tmp_path, _Qwen)
    output, stats = review.run_qwen_lexical_review(
        audio,
        primary,
        review_reference_segments=reference,
        review_reference_source="sensevoice_wallclock_anchor",
    )

    assert len(instances) == 1
    assert instances[0].calls == 1
    assert output[0] == primary[0]
    assert output[1].text == "第二窗口存在明显问题需要复核。"
    assert stats["window_count"] == 2
    assert stats["candidate_window_count"] == 1
    assert stats["skipped_window_count"] == 1
    assert stats["reference_available"] is True
    assert stats["reference_source"] == "sensevoice_wallclock_anchor"
    assert stats["window_diagnostics"][0]["selected"] is False
    assert stats["window_diagnostics"][0]["reason"] == "reference_text_agreement"
    assert stats["window_diagnostics"][1]["selected"] is True
    assert stats["window_diagnostics"][1]["selection_reason"] == "reference_text_disagreement"
    assert stats["window_diagnostics"][1]["reason"] == "accepted"


def test_run_reviews_window_when_cached_reference_has_no_text(monkeypatch, tmp_path: Path):
    primary = [_segment(0.0, 10.0, "缓存参考缺失的这一段正文仍然需要复核。", "SPEAKER_A")]
    reference = [Segment(start=100.0, end=110.0, text="其他时间的参考文字。")]

    class _Qwen:
        def transcribe(self, _audio, _options):
            return _result("缓存参考缺失的这一段正文仍需进行复核。")

    audio = _prepare_run(monkeypatch, tmp_path, _Qwen)
    output, stats = review.run_qwen_lexical_review(
        audio,
        primary,
        review_reference_segments=reference,
    )

    assert output[0].text == "缓存参考缺失的这一段正文仍需进行复核。"
    assert stats["candidate_window_count"] == 1
    assert stats["window_diagnostics"][0]["reference_chars"] == 0
    assert stats["window_diagnostics"][0]["selection_reason"] == "reference_text_missing"
    assert stats["window_diagnostics"][0]["reason"] == "accepted"


def test_reference_selection_tolerates_one_second_window_boundary_shift():
    primary = [Segment(start=10.0, end=20.0, text="边界偏移不应导致重复调用复核模型。")]
    reference = [Segment(start=9.0, end=19.0, text="边界偏移不应导致重复调用复核模型。")]
    window = review._ReviewWindow(start=10.0, end=20.0, segment_indexes=(0,))

    diagnostic = review._reference_selection_diagnostic(window, primary, reference)

    assert diagnostic["selected"] is False
    assert diagnostic["selection_reason"] == "reference_text_agreement"
    assert diagnostic["reference_shift_seconds"] == -1.0


def test_reference_selection_reviews_high_mismatch_window_near_ninety_percent_similarity():
    shared = "".join(chr(0x4E00 + index) for index in range(179))
    primary_text = shared + "".join(chr(0x5200 + index) for index in range(21))
    reference_text = shared + "".join(chr(0x6200 + index) for index in range(21))
    primary = [Segment(start=0.0, end=87.0, text=primary_text)]
    reference = [Segment(start=2.0, end=85.0, text=reference_text)]
    window = review._ReviewWindow(start=0.0, end=87.0, segment_indexes=(0,))

    diagnostic = review._reference_selection_diagnostic(window, primary, reference)

    assert diagnostic["reference_similarity"] == 0.895
    assert diagnostic["reference_mismatch_chars"] == 21
    assert diagnostic["selected"] is True
    assert diagnostic["selection_reason"] == "reference_text_disagreement"


def test_run_rejects_abnormal_length_ratio(monkeypatch, tmp_path: Path):
    primary = [_segment(0.0, 10.0, "这是简短的原始文本。", "SPEAKER_A")]

    class _Qwen:
        def transcribe(self, _audio, _options):
            return _result("这是非常非常长的模型输出，包含大量不应出现的额外内容和重复扩写。" * 3)

    audio = _prepare_run(monkeypatch, tmp_path, _Qwen)
    output, stats = review.run_qwen_lexical_review(audio, primary)

    assert output == primary
    assert stats["rejected"] == 1
    assert stats["window_diagnostics"][0]["reason"] == "length_ratio_out_of_range"


def test_run_rejects_excessive_estimated_timing_ratio(monkeypatch, tmp_path: Path):
    primary = [_segment(0.0, 10.0, "这是原始会议内容。", "SPEAKER_A")]

    class _Qwen:
        def transcribe(self, _audio, _options):
            return _result("这是修正会议内容。")

    audio = _prepare_run(monkeypatch, tmp_path, _Qwen)
    monkeypatch.setattr(
        review,
        "_align_segments_to_timing_anchor",
        lambda source, _anchors, min_equal_ratio: (
            [Segment(start=0.0, end=10.0, text=source[0].text)],
            {
                "timing_alignment_ok": True,
                "source_chars": 8,
                "estimated_timing_chars": 4,
                "min_equal_ratio": min_equal_ratio,
            },
        ),
    )

    output, stats = review.run_qwen_lexical_review(audio, primary)

    assert output == primary
    assert stats["rejected"] == 1
    assert stats["window_diagnostics"][0]["reason"] == "estimated_timing_ratio_too_high"


def test_run_preserves_everything_when_model_is_not_cached(monkeypatch, tmp_path: Path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    primary = [_segment(0.0, 10.0, "必须完整保留。", "SPEAKER_A")]
    monkeypatch.setattr(review, "_qwen_model_cached", lambda _model: False)
    monkeypatch.setattr(
        review,
        "Qwen3ASRTranscriber",
        lambda: (_ for _ in ()).throw(AssertionError("uncached model must not be loaded")),
    )

    output, stats = review.run_qwen_lexical_review(audio, primary)

    assert output == primary
    assert stats["enabled"] is False
    assert stats["accepted"] == 0
    assert stats["rejected"] == 1
    assert stats["changed"] == 0
    assert stats["reason"] == "qwen_model_not_cached"
    assert stats["timeline_preserved"] is True


def test_run_preserves_everything_when_audio_is_missing(monkeypatch, tmp_path: Path):
    primary = [_segment(0.0, 10.0, "音频缺失时保留。", "SPEAKER_A")]
    monkeypatch.setattr(
        review,
        "_qwen_model_cached",
        lambda _model: (_ for _ in ()).throw(AssertionError("missing audio must fail first")),
    )

    output, stats = review.run_qwen_lexical_review(tmp_path / "missing.wav", primary)

    assert output == primary
    assert stats["enabled"] is False
    assert stats["accepted"] == 0
    assert stats["rejected"] == 1
    assert stats["changed"] == 0
    assert stats["reason"] == "audio_not_found"
    assert stats["timeline_preserved"] is True
