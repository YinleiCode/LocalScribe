from __future__ import annotations

from scribe_py.core.strong_asr import (
    ReviewWindow,
    _overlap_text,
    apply_window_text,
    atomic_aligned_independent_consensus_rewrite,
    build_review_windows,
    consensus_rewrite,
    timeline_fingerprint,
)
from scribe_py.core.types import Segment


def test_consensus_rewrite_requires_paraformer_and_qwen_context_agreement():
    primary = "也就是说，因为我们流写分离这块需要改造。"
    paraformer = "也就是说，因为我们读写分离这块需要改造。"
    qwen = "就比如说，因为我们读写分离这块儿需要做改造。"

    corrected, changes = consensus_rewrite(
        primary, paraformer, qwen, detector_text=paraformer
    )

    assert "读写分离" in corrected
    assert "流写分离" not in corrected
    assert len(changes) == 1
    assert changes[0]["from"] == "流"
    assert changes[0]["to"] == "读"
    assert changes[0]["evidence"] == "sensevoice_redecode_paraformer_qwen_context_agreement"


def test_consensus_rewrite_keeps_primary_when_models_disagree():
    primary = "这个人平时挺好的，很自然就会不好了。"
    paraformer = "这个人平时脾气挺好的，怎么突然不好了。"
    qwen = "这个人平时脾气挺好的，突然间脾气不好了。"

    corrected, changes = consensus_rewrite(
        primary, paraformer, qwen, detector_text=paraformer
    )

    assert corrected == primary
    assert changes == []


def test_consensus_rewrite_requires_independent_detector_confirmation():
    primary = "会议开始以后请大家认真检察材料然后继续讨论。"
    redecode = "会议开始以后请大家认真检查材料然后继续讨论。"
    qwen = redecode
    detector = primary

    corrected, changes = consensus_rewrite(
        primary,
        redecode,
        qwen,
        detector_text=detector,
    )

    assert corrected == primary
    assert changes == []


def test_consensus_rewrite_rejects_qwen_hallucination_risk():
    primary = "我们流写分离这块需要改造。"
    paraformer = "我们读写分离这块需要改造。"
    qwen = "我们读写分离这块需要改造。" + "我" * 100

    corrected, changes = consensus_rewrite(
        primary,
        paraformer,
        qwen,
        detector_text=paraformer,
        qwen_hallucination_risk=True,
    )

    assert corrected == primary
    assert changes == []


def test_consensus_rewrite_does_not_change_conversational_fillers():
    primary = "还是我们的呃逻辑上层面。"
    paraformer = "还是我们的啊逻辑上层面。"
    qwen = "还是我们的啊逻辑上层面。"

    corrected, changes = consensus_rewrite(
        primary, paraformer, qwen, detector_text=paraformer
    )

    assert corrected == primary
    assert changes == []


def test_consensus_rewrite_does_not_change_conversational_particles():
    primary = "那个会议就是十点到十二点的吧？"
    paraformer = "那个会议就是十点到十二点的嘛？"
    qwen = "那个会议就是十点到十二点的嘛？"

    corrected, changes = consensus_rewrite(
        primary, paraformer, qwen, detector_text=paraformer
    )

    assert corrected == primary
    assert changes == []


def test_consensus_rewrite_does_not_change_numeric_rendering_only():
    primary = "会议时间是10点到12点。"
    paraformer = "会议时间是十点到十二点。"
    qwen = "会议时间是十点到十二点。"

    corrected, changes = consensus_rewrite(
        primary, paraformer, qwen, detector_text=paraformer
    )

    assert corrected == primary
    assert changes == []


def test_consensus_rewrite_does_not_guess_an_addressed_person_name():
    primary = "你先跟海宁对一下，之后再回复。"
    paraformer = "你先跟海林对一下，之后再回复。"
    qwen = "你先跟海林对一下，之后再回复。"

    corrected, changes = consensus_rewrite(
        primary, paraformer, qwen, detector_text=paraformer
    )

    assert corrected == primary
    assert changes == []


def test_consensus_rewrite_rejects_multichar_change_on_weak_window_agreement():
    primary = "好了我们更罗罗是做售前扩展。"
    paraformer = "好了我们更多的是做售前扩展。"
    qwen = "前面有杂音好了我们更多的是做拓展后面也有杂音。"

    corrected, changes = consensus_rewrite(
        primary,
        paraformer,
        qwen,
        detector_text=paraformer,
        candidate_qwen_similarity=0.4692,
        primary_qwen_similarity=0.4646,
        detector_qwen_similarity=0.4692,
    )

    assert corrected == primary
    assert changes == []


def test_consensus_rewrite_keeps_single_char_fix_on_lower_window_agreement():
    primary = "也就是说因为我们流写分离这块需要改造。"
    paraformer = "也就是说因为我们读写分离这块需要改造。"
    qwen = "因为我们读写分离这块需要做一些改造。"

    corrected, changes = consensus_rewrite(
        primary,
        paraformer,
        qwen,
        detector_text=paraformer,
        candidate_qwen_similarity=0.529,
        primary_qwen_similarity=0.4575,
        detector_qwen_similarity=0.529,
    )

    assert "读写分离" in corrected
    assert len(changes) == 1


def test_consensus_rewrite_can_keep_confirmed_spelling_without_candidate_insertion():
    primary = "嗯，就就是说因为我们流写分离这块也需要做一些工作的改造。"
    candidate = "嗯，就是说因为我们读写分离这块儿也需要做一些工作的改造。"

    corrected, changes = consensus_rewrite(
        primary,
        candidate,
        candidate,
        detector_text=candidate,
        allow_insertions=False,
    )

    assert corrected == "嗯，就就是说因为我们读写分离这块也需要做一些工作的改造。"
    assert changes == [
        {
            "from": "流",
            "to": "读",
            "normalized_from": "流",
            "normalized_to": "读",
            "left_context": "说因为我们",
            "right_context": "写分离这块",
            "evidence": "sensevoice_redecode_paraformer_qwen_context_agreement",
        }
    ]


def test_consensus_rewrite_keeps_multichar_fix_on_strong_window_agreement():
    primary = "所以大家真有不要着急。"
    paraformer = "所以大家千万不要着急。"
    qwen = "所以大家千万不要着急。"

    corrected, changes = consensus_rewrite(
        primary,
        paraformer,
        qwen,
        detector_text=paraformer,
        candidate_qwen_similarity=0.5981,
        primary_qwen_similarity=0.5696,
        detector_qwen_similarity=0.5981,
    )

    assert "千万不要" in corrected
    assert len(changes) == 1


def test_atomic_aligned_consensus_recovers_shared_character_from_wider_rewrite():
    primary = "过节以后，当我们即江的安排活动时，需要提前确认流程。"
    paraformer = "过节以后，当我们即将的安排活动时，需要提前确认流程。"
    qwen = "过节以后，当我们即将要安排活动时，需要提前确认流程。"

    corrected, changes = atomic_aligned_independent_consensus_rewrite(
        primary,
        paraformer,
        qwen,
    )

    assert "即将的安排" in corrected
    assert len(changes) == 1
    assert changes[0]["from"] == "江"
    assert changes[0]["to"] == "将"
    assert changes[0]["evidence"] == "paraformer_qwen_atomic_aligned_consensus"


def test_atomic_aligned_consensus_rejects_unknown_local_token():
    primary = "项目开始以后，当我们即沐的安排活动时，需要提前确认流程。"
    paraformer = "项目开始以后，当我们即洋的安排活动时，需要提前确认流程。"
    qwen = "项目开始以后，当我们即洋要安排活动时，需要提前确认流程。"

    corrected, changes = atomic_aligned_independent_consensus_rewrite(
        primary,
        paraformer,
        qwen,
    )

    assert corrected == primary
    assert changes == []


def test_consensus_rewrite_deletes_bounded_internal_extra_words_with_three_way_agreement():
    primary = "项目讨论以后把这些资源的规划确认清楚再继续执行。"
    confirmed = "项目讨论以后把这些规划确认清楚再继续执行。"

    corrected, changes = consensus_rewrite(
        primary,
        confirmed,
        confirmed,
        detector_text=confirmed,
    )

    assert corrected == confirmed
    assert len(changes) == 1
    assert changes[0]["operation"] == "delete"
    assert changes[0]["from"] == "资源的"
    assert changes[0]["to"] == ""
    assert changes[0]["evidence"].endswith("internal_indel_agreement")


def test_consensus_rewrite_inserts_bounded_internal_missing_words_with_three_way_agreement():
    primary = "项目讨论以后把这些规划确认清楚再继续执行。"
    confirmed = "项目讨论以后把这些资源规划确认清楚再继续执行。"

    corrected, changes = consensus_rewrite(
        primary,
        confirmed,
        confirmed,
        detector_text=confirmed,
    )

    assert corrected == confirmed
    assert len(changes) == 1
    assert changes[0]["operation"] == "insert"
    assert changes[0]["from"] == ""
    assert changes[0]["to"] == "资源"


def test_consensus_rewrite_rejects_internal_indel_without_both_independent_confirmers():
    primary = "项目讨论以后把这些资源的规划确认清楚再继续执行。"
    redecode = "项目讨论以后把这些规划确认清楚再继续执行。"

    corrected, changes = consensus_rewrite(
        primary,
        redecode,
        redecode,
        detector_text=primary,
    )

    assert corrected == primary
    assert changes == []


def test_consensus_rewrite_rejects_large_or_boundary_indels():
    cases = [
        (
            "项目讨论以后把这些规划确认清楚再继续执行。",
            "项目讨论以后把这些必要资源项规划确认清楚再继续执行。",
        ),
        ("资源的规划确认清楚以后再继续执行。", "规划确认清楚以后再继续执行。"),
    ]

    for primary, confirmed in cases:
        corrected, changes = consensus_rewrite(
            primary,
            confirmed,
            confirmed,
            detector_text=confirmed,
        )

        assert corrected == primary
        assert changes == []


def test_consensus_rewrite_protects_sensitive_or_ambiguous_internal_indels():
    cases = [
        ("会议以后第3项继续讨论。", "会议以后第项继续讨论。"),
        ("会议以后我们继续讨论。", "会议以后继续讨论。"),
        ("会议以后嗯继续讨论。", "会议以后继续讨论。"),
        ("会议以后要要继续讨论。", "会议以后要继续讨论。"),
        ("你先跟海宁对一下之后再回复。", "你先跟对一下之后再回复。"),
        ("你先跟对一下之后再回复。", "你先跟海宁对一下之后再回复。"),
    ]

    for primary, confirmed in cases:
        corrected, changes = consensus_rewrite(
            primary,
            confirmed,
            confirmed,
            detector_text=confirmed,
        )

        assert corrected == primary
        assert changes == []


def test_apply_window_text_preserves_segment_and_cue_timeline():
    segments = [
        Segment(
            start=0.0,
            end=3.0,
            text="我们流写分离。",
            sync_cues=[
                {"start": 0.0, "end": 1.5, "text": "我们流写"},
                {"start": 1.5, "end": 3.0, "text": "分离"},
            ],
        ),
        Segment(start=3.0, end=6.0, text="这块需要改造。"),
    ]
    before = timeline_fingerprint(segments)
    window = ReviewWindow(start=0.0, end=6.0, segment_indexes=(0, 1))

    updated = apply_window_text(segments, window, "我们读写分离。\n这块需要改造。")

    assert timeline_fingerprint(updated) == before
    assert updated[0].text == "我们读写分离。"
    assert updated[0].original_text == "我们流写分离。"
    assert [(cue["start"], cue["end"]) for cue in updated[0].sync_cues or []] == [
        (0.0, 1.5),
        (1.5, 3.0),
    ]


def test_build_review_windows_is_generic_and_duration_bounded():
    segments = [
        Segment(start=0, end=5, text="第一段。"),
        Segment(start=5, end=11, text="第二段。"),
        Segment(start=11, end=18, text="第三段。"),
        Segment(start=18, end=26, text="第四段。"),
    ]

    windows = build_review_windows(segments, max_seconds=20)

    assert [window.segment_indexes for window in windows] == [(0, 1, 2), (3,)]


def test_overlap_text_crops_coarse_secondary_segments_by_time():
    segments = [Segment(start=0, end=10, text="一二三四五六七八九十")]

    assert _overlap_text(segments, 5, 10) == "六七八九十"
