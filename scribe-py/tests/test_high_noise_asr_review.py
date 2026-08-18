from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scribe_py import ipc
from scribe_py.core.strong_asr import (
    ReviewWindow,
    aligned_independent_consensus_rewrite,
    _qwen_window_text,
    _transcribe_paraformer_for_review,
    _transcribe_paraformer_review_windows,
    consensus_rewrite,
    decide_auto_high_noise_review,
    independent_consensus_rewrite,
    phonetic_near_independent_consensus_rewrite,
    run_strong_asr_review,
    select_bounded_independent_review_windows,
    select_sparse_review_windows,
    select_standard_selective_review_windows,
    technical_consensus_rewrite,
)
from scribe_py.core.types import Segment, TranscribeResult


@pytest.fixture(autouse=True)
def _mock_qwen_runtime_for_unit_tests(monkeypatch):
    from scribe_py.core import strong_asr

    monkeypatch.setattr(strong_asr, "_qwen_runtime_available", lambda: (True, "unit_test"))


def test_qwen_review_text_is_clipped_to_the_exact_window():
    tiles = [
        (
            0,
            {
                "text": "前文目标前半",
                "segments": [
                    Segment(start=0.0, end=10.0, text="前文"),
                    Segment(start=10.0, end=20.0, text="目标前半"),
                ],
            },
        ),
        (
            1,
            {
                "text": "目标后半后文",
                "segments": [
                    Segment(start=20.0, end=30.0, text="目标后半"),
                    Segment(start=30.0, end=40.0, text="后文"),
                ],
            },
        ),
    ]

    assert _qwen_window_text(tiles, 10.0, 30.0) == "目标前半目标后半"


def test_standard_selective_review_uses_generic_anomalies_without_coverage_probes():
    segments = [
        Segment(start=0.0, end=8.0, text="正常的会议开场内容。"),
        Segment(start=8.0, end=18.0, text="后续有read这样的片段，同时出现成本成本。"),
        Segment(start=18.0, end=26.0, text="最后一段正常结束。"),
    ]

    selected, stats = select_standard_selective_review_windows(segments, max_windows=2)

    assert len(selected) == 1
    assert selected[0].start < 8.0 < selected[0].end
    assert stats["uses_recording_name"] is False
    assert stats["uses_fixed_transcript_phrases"] is False
    assert stats["uses_coverage_probes"] is False
    assert "inline_lowercase_latin_fragment" in stats["diagnostics"][0]["reasons"]


def test_standard_selective_review_does_not_treat_sparse_speech_as_an_error():
    selected, stats = select_standard_selective_review_windows([
        Segment(start=100.0, end=112.0, text="你说什么。"),
    ])

    assert selected == []
    assert stats["candidate_window_count"] == 0


def test_paraformer_and_qwen_can_fix_a_systematic_primary_term_error():
    primary = "然后里巴拉唑也先吃一年吧。"
    paraformer = "然后雷贝拉唑也先吃一年吧。"
    qwen = "医生说，然后雷贝拉唑也先吃一年吧。"

    corrected, changes = independent_consensus_rewrite(primary, paraformer, qwen)

    assert corrected == "然后雷贝拉唑也先吃一年吧。"
    assert changes == [
        {
            "from": "里巴",
            "to": "雷贝",
            "normalized_from": "里巴",
            "normalized_to": "雷贝",
            "left_context": "然后",
            "right_context": "拉唑也先吃",
            "evidence": "paraformer_qwen_independent_context_agreement",
        }
    ]


def test_consensus_preserves_uppercase_acronym_casing():
    primary = "材料必须跟BT对齐以后再继续。"
    candidate = "材料必须跟BD对齐以后再继续。"

    corrected, changes = independent_consensus_rewrite(primary, candidate, candidate)

    assert corrected == candidate
    assert changes[0]["from"] == "T"
    assert changes[0]["to"] == "D"
    assert changes[0]["normalized_to"] == "d"


def test_acronym_rewrite_rejects_another_expanded_variant_of_dominant_term():
    primary = "他们那个DSBD盘是怎么做的？"
    candidate = "他们那个BSBD盘是怎么做的？"
    global_context = "SBD盘支持三个挂载。SBD设备已经确认。继续检查SBD共享盘。"

    corrected, changes = aligned_independent_consensus_rewrite(
        primary,
        candidate,
        candidate,
        global_context_text=global_context,
    )

    assert corrected == primary
    assert changes == []


def test_short_acronym_rewrite_is_not_blocked_by_unrelated_dominant_term():
    primary = "材料必须跟BT对齐以后再继续。"
    candidate = "材料必须跟BD对齐以后再继续。"
    global_context = "SBD盘支持三个挂载。SBD设备已经确认。继续检查SBD共享盘。"

    corrected, changes = independent_consensus_rewrite(
        primary,
        candidate,
        candidate,
        global_context_text=global_context,
    )

    assert corrected == candidate
    assert changes[0]["to"] == "D"


def test_single_character_rewrite_between_repeated_boundaries_is_rejected():
    primary = "但是这种超这种模式也是支持的。"
    candidate = "但是这种方这种模式也是支持的。"

    corrected, changes = independent_consensus_rewrite(
        primary,
        candidate,
        candidate,
        global_context_text=primary,
    )

    assert corrected == primary
    assert changes == []


def test_repeated_character_rewrite_is_allowed_when_it_completes_a_known_word():
    primary = "现场出现了一些多多少美的状况。"
    candidate = "现场出现了一些多多少少的状况。"

    corrected, changes = independent_consensus_rewrite(
        primary,
        candidate,
        candidate,
        global_context_text=primary,
    )

    assert corrected == candidate
    assert changes[0]["from"] == "美"
    assert changes[0]["to"] == "少"


def test_two_independent_models_cannot_delete_primary_speech_without_redecode_confirmation():
    primary = "这件事我推动了那么久，我推动了半年。"
    omitted = "这件事我推动了那么，我推动了半年。"

    corrected, changes = independent_consensus_rewrite(
        primary,
        omitted,
        omitted,
        global_context_text=primary,
    )

    assert corrected == primary
    assert changes == []


def test_independent_models_can_complete_a_long_latin_fragment():
    primary = "如果是ma node发生切换，就执行命令。"
    candidate = "如果是master node发生切换，就执行命令。"

    corrected, changes = independent_consensus_rewrite(
        primary,
        candidate,
        candidate,
        global_context_text=primary,
    )

    assert corrected == primary
    assert changes == []

    corrected, changes = technical_consensus_rewrite(
        primary,
        candidate,
        candidate,
        global_context_text=primary,
    )

    assert corrected == candidate
    assert changes[0]["normalized_to"] == "master"


def test_paraformer_candidate_is_rejected_when_qwen_conflicts():
    primary = "然后里巴拉唑也先吃一年吧。"
    paraformer = "然后雷贝拉唑也先吃一年吧。"
    qwen = "医生说，然后奥美拉唑也先吃一年吧。"

    corrected, changes = independent_consensus_rewrite(primary, paraformer, qwen)

    assert corrected == primary
    assert changes == []


def test_exact_chinese_homophone_does_not_overwrite_primary_domain_term():
    primary = "右冠梗完的人可能血压偏低。"
    paraformer = "又灌梗完的人可能血压偏低。"
    qwen = "医生说，又灌梗完的人可能血压偏低。"

    corrected, changes = independent_consensus_rewrite(primary, paraformer, qwen)

    assert corrected == primary
    assert changes == []


def test_exact_single_character_homophone_does_not_create_fake_name_fix():
    primary = "我园林夕认错。"
    candidate = "我袁林夕认错。"

    corrected, changes = independent_consensus_rewrite(primary, candidate, candidate)

    assert corrected == primary
    assert changes == []


def test_multi_character_rewrite_cannot_change_pronoun_semantics():
    primary = "因为术后我这有点痛。"
    candidate = "因为术后不是有点痛。"

    corrected, changes = independent_consensus_rewrite(primary, candidate, candidate)

    assert corrected == primary
    assert changes == []


def test_potential_vocative_name_is_not_rewritten():
    primary = "所以英兰你知道吗，就是这个情况。"
    candidate = "所以英磊你知道吗，就是这个情况。"

    corrected, changes = independent_consensus_rewrite(primary, candidate, candidate)

    assert corrected == primary
    assert changes == []


def test_qwen_spelling_wins_when_paraformer_heard_a_closer_phonetic_variant():
    primary = "然后里巴拉作也先吃一年吧。"
    paraformer = "然后雷布拉唑也先吃一年吧。"
    qwen = "医生说，然后雷贝拉唑也先吃一年吧。"

    corrected, changes = phonetic_near_independent_consensus_rewrite(
        primary,
        paraformer,
        qwen,
    )

    assert corrected == "然后雷贝拉作也先吃一年吧。"
    assert changes[0]["evidence"] == "paraformer_qwen_phonetic_near_consensus"
    assert changes[0]["paraformer_variant"] == "雷布"


def test_phonetic_near_consensus_rejects_a_variant_not_closer_than_primary():
    primary = "右冠梗完的人可能血压偏低。"
    paraformer = "又灌梗完的人可能血压偏低。"
    qwen = "医生说，右管梗完的人可能血压偏低。"

    corrected, changes = phonetic_near_independent_consensus_rewrite(
        primary,
        paraformer,
        qwen,
    )

    assert corrected == primary
    assert changes == []


def test_phonetic_near_consensus_rejects_weak_three_syllable_agreement():
    primary = "这个回孕霜先吃着然后复查。"
    paraformer = "这个核心爽先吃着然后复查。"
    qwen = "医生说，这个和银双先吃着然后复查。"

    corrected, changes = phonetic_near_independent_consensus_rewrite(
        primary,
        paraformer,
        qwen,
    )

    assert corrected == primary
    assert changes == []


def test_exact_aligned_consensus_handles_adjacent_term_character_errors():
    primary = "然后里巴拉作也先吃一年吧。"
    paraformer = "然后雷布拉唑也先吃一年吧。"
    qwen = "医生说，然后雷贝拉唑也先吃一年吧。"

    corrected, changes = aligned_independent_consensus_rewrite(
        primary,
        paraformer,
        qwen,
    )

    assert corrected == "然后里巴拉唑也先吃一年吧。"
    assert changes == [
        {
            "from": "作",
            "to": "唑",
            "normalized_from": "作",
            "normalized_to": "唑",
            "left_context": "然后里巴拉",
            "right_context": "也先吃一年",
            "evidence": "paraformer_qwen_exact_aligned_consensus",
        }
    ]


def test_exact_aligned_consensus_protects_numbers_and_pronouns():
    primary = "你记一下编号1765，然后回复我。"
    candidate = "您记一下编号幺七六，然后回复我。"

    corrected, changes = aligned_independent_consensus_rewrite(
        primary,
        candidate,
        candidate,
    )

    assert corrected == primary
    assert changes == []


def test_two_character_phonetic_guess_requires_adjacent_exact_anchor():
    primary = "现在进入后续流程。"
    paraformer = "现在机泌后续流程。"
    qwen = "现在机密后续流程。"

    corrected, changes = phonetic_near_independent_consensus_rewrite(
        primary,
        paraformer,
        qwen,
    )

    assert corrected == primary
    assert changes == []


def test_two_character_phonetic_guess_can_use_existing_transcript_anchor():
    primary = "这属于内部进入文件。"
    paraformer = "这属于内部技术文件。"
    qwen = "这属于内部机密文件。"

    corrected, changes = phonetic_near_independent_consensus_rewrite(
        primary,
        paraformer,
        qwen,
        global_context_text="客户随后追问为什么是机密文件。",
    )

    assert corrected == "这属于内部机密文件。"
    assert changes[0]["evidence"] == "paraformer_qwen_phonetic_near_consensus"


def test_independent_consensus_preserves_discourse_connectors():
    primary = "如果交付不了，反正我就申请退款。"
    candidate = "了我交付不了，然后我就申请退款。"

    corrected, changes = independent_consensus_rewrite(
        primary,
        candidate,
        candidate,
    )

    assert corrected == primary
    assert changes == []


def test_aligned_single_character_change_cannot_override_repeated_source_term():
    primary = "我需要帮您上报看一下。"
    candidate = "我需要帮您上面看一下。"

    corrected, changes = aligned_independent_consensus_rewrite(
        primary,
        candidate,
        candidate,
        global_context_text="稍后会继续上报，这个问题已经上报。",
    )

    assert corrected == primary
    assert changes == []


def test_aligned_single_character_change_preserves_same_window_repetition():
    primary = "本身都是跨A级的，有些还跨了3个A级。"
    candidate = "本身都是跨A级的，有些还花了3个A级。"

    corrected, changes = aligned_independent_consensus_rewrite(
        primary,
        candidate,
        candidate,
    )

    assert corrected == primary
    assert changes == []


def test_independent_consensus_does_not_create_nearby_repeated_word():
    primary = "成本其实是最关键的机制，包括我们的周期。"
    candidate = "成本其实是最关键的其实，包括我们的周期。"

    corrected, changes = independent_consensus_rewrite(
        primary,
        candidate,
        candidate,
    )

    assert corrected == primary
    assert changes == []


def test_phonetic_consensus_does_not_create_nearby_repeated_word():
    primary = "成本其实是最关键的机制，包括我们的周期。"
    candidate = "成本其实是最关键的其实，包括我们的周期。"

    corrected, changes = phonetic_near_independent_consensus_rewrite(
        primary,
        candidate,
        candidate,
        global_context_text=primary,
    )

    assert corrected == primary
    assert changes == []


def test_aligned_consensus_does_not_create_adjacent_duplicate_character():
    primary = "先要以商品类险为中心为北基组。"
    candidate = "先要以商品类险为中心为为基组。"

    corrected, changes = aligned_independent_consensus_rewrite(
        primary,
        candidate,
        candidate,
    )

    assert corrected == primary
    assert changes == []


def test_technical_consensus_repairs_repeated_measurement_unit_conflict():
    primary = "客户知道0.7毫米，后面又说0.7毫秒，两个区域是0.7毫米。"
    paraformer = "客户知道零点七毫秒，后面又说零点七毫秒，两个区域是零点七毫秒。"
    qwen = "客户知道零点七毫秒，后面又说零点七毫秒，两个区域是零点七毫秒。"

    corrected, changes = technical_consensus_rewrite(
        primary,
        paraformer,
        qwen,
        global_context_text=primary,
    )

    assert corrected.count("0.7毫秒") == 3
    assert "毫米" not in corrected
    assert len(changes) == 2
    assert all(
        change["evidence"] == "paraformer_qwen_repeated_measurement_consensus"
        for change in changes
    )


def test_technical_consensus_preserves_valid_distinct_measurements():
    text = "间距是0.7毫米，延迟是0.7毫秒。"
    wrong_independent = "间距是零点七毫秒，延迟是零点七毫秒。"

    corrected, changes = technical_consensus_rewrite(
        text,
        wrong_independent,
        wrong_independent,
    )

    assert corrected == text
    assert changes == []

    short_primary = "这里写的是AZA区域。"
    short_qwen = "这里写的是 AZ 区域。"
    short_global = "AZ区域。AZ区域。AZ区域。这里写的是AZA区域。"
    corrected, changes = technical_consensus_rewrite(
        short_primary,
        short_primary,
        short_qwen,
        global_context_text=short_global,
    )

    assert corrected == short_primary
    assert changes == []


def test_technical_consensus_repairs_anchored_mixed_latin_corruption():
    primary = "我再敲一个1块销over的命令，然后继续。"
    paraformer = "我再敲一个 switch over 的命令，然后继续。"
    qwen = "我再敲一个 switch over 的命令，然后继续。"

    corrected, changes = technical_consensus_rewrite(primary, paraformer, qwen)

    assert corrected == "我再敲一个switchover的命令，然后继续。"
    assert changes == [
        {
            "from": "1块销over",
            "to": "switchover",
            "normalized_from": "1块销over",
            "normalized_to": "switchover",
            "left_context": "我再敲一个",
            "right_context": "的命令然后",
            "evidence": "paraformer_qwen_anchored_latin_consensus",
        }
    ]


def test_technical_consensus_requires_bilateral_latin_anchors():
    primary = "文档写了1块销over，但这里上下文不同。"
    paraformer = "另一个位置执行 switch over 的命令。"
    qwen = "再解释一下 switch over 的含义。"

    corrected, changes = technical_consensus_rewrite(primary, paraformer, qwen)

    assert corrected == primary
    assert changes == []


def test_technical_consensus_never_replaces_across_segment_newlines():
    primary = "我再敲一个1块销\nover的命令，然后继续。"
    paraformer = "我再敲一个 switch over 的命令，然后继续。"
    qwen = "我再敲一个 switch over 的命令，然后继续。"

    corrected, changes = technical_consensus_rewrite(primary, paraformer, qwen)

    assert corrected == primary
    assert changes == []


def test_technical_consensus_repairs_dominant_one_letter_acronym_noise():
    primary = "这里有SCBD盘，可以继续切换。"
    paraformer = "这里有 S B D 盘，可以继续切换。"
    qwen = "这里有 S B D 盘，可以继续切换。"
    global_context = "SBD盘已经配置。SBD设备有三个。检查SBD状态。这里有SCBD盘。"

    corrected, changes = technical_consensus_rewrite(
        primary,
        paraformer,
        qwen,
        global_context_text=global_context,
    )

    assert corrected == "这里有SBD盘，可以继续切换。"
    assert changes[0]["evidence"] == "paraformer_qwen_global_acronym_consistency"


def test_technical_consensus_requires_paraformer_acronym_confirmation():
    primary = "这里有SCBD盘，可以继续切换。"
    qwen = "这里有 S B D 盘，可以继续切换。"
    global_context = "SBD盘已经配置。SBD设备有三个。检查SBD状态。这里有SCBD盘。"

    corrected, changes = technical_consensus_rewrite(
        primary,
        primary,
        qwen,
        global_context_text=global_context,
    )

    assert corrected == primary
    assert changes == []


def test_technical_consensus_does_not_match_inside_a_longer_qwen_acronym():
    primary = "他们那个DSBD盘是怎么做的？"
    qwen = "他们那个B S B D盘是怎么做的？后面再说SBD盘怎么配置。"
    global_context = "SBD盘已配置。检查SBD设备。确认SBD状态。这里写了DSBD盘。"

    corrected, changes = technical_consensus_rewrite(
        primary,
        primary,
        qwen,
        global_context_text=global_context,
    )

    assert corrected == primary
    assert changes == []


def test_technical_consensus_preserves_numbers_fillers_and_valid_acronyms():
    text = "嗯，APP部署在AZ1和AZ2，AWS编号是10，不是十，也不要猜AZA。"
    global_context = text + "APP保持不变。AWS保持不变。AZ1和AZ2保持不变。AZ是区域。AZ是区域。"

    corrected, changes = technical_consensus_rewrite(
        text,
        text,
        text,
        global_context_text=global_context,
    )

    assert corrected == text
    assert changes == []


def test_model_release_never_touches_mps_cache_by_default(monkeypatch):
    from scribe_py.core import strong_asr

    calls = []
    fake_torch = SimpleNamespace(mps=SimpleNamespace(empty_cache=lambda: calls.append(True)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("LOCALSCRIBE_TORCH_MPS_EMPTY_CACHE", raising=False)

    strong_asr._release_model_memory()

    assert calls == []


def test_model_release_allows_explicitly_validated_mps_cache_opt_in(monkeypatch):
    from scribe_py.core import strong_asr

    calls = []
    fake_torch = SimpleNamespace(mps=SimpleNamespace(empty_cache=lambda: calls.append(True)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("LOCALSCRIBE_TORCH_MPS_EMPTY_CACHE", "1")

    strong_asr._release_model_memory()

    assert calls == [True]


def test_long_paraformer_review_is_tiled_and_projected_to_non_overlapping_cores(
    monkeypatch,
    tmp_path: Path,
):
    audio = tmp_path / "long.wav"
    audio.write_bytes(b"audio")
    calls = []

    class _Paraformer:
        def transcribe(self, audio_path, _options):
            calls.append(Path(audio_path).name)
            return TranscribeResult(
                audio=str(audio_path),
                language="zh",
                duration=62.0,
                transcribe_seconds=0.1,
                rtf=0.01,
                backend="funasr",
                model_id="paraformer-zh",
                segments=[Segment(start=0.0, end=62.0, text="这是分块复核文本")],
            )

    from scribe_py.core import strong_asr

    monkeypatch.setattr(
        strong_asr,
        "_extract_clip",
        lambda _audio, _start, _end, output: output.write_bytes(b"clip"),
    )

    segments, tile_count = _transcribe_paraformer_for_review(
        _Paraformer(),
        audio,
        language="zh",
        model_id="paraformer-zh",
        audio_end=121.0,
    )

    assert tile_count == 3
    assert len(calls) == 3
    assert [(segment.start, segment.end) for segment in segments] == [
        (0.0, 60.0),
        (60.0, 120.0),
        (120.0, 121.0),
    ]


def test_sparse_review_window_selection_is_structural_and_bounded():
    segments = [
        Segment(start=0.0, end=5.0, text="这是正常且完整的一段转写。"),
        Segment(start=10.0, end=22.0, text="你说什么。"),
        Segment(start=40.0, end=50.0, text="这也是正常密度的一段转写内容。"),
        Segment(start=70.0, end=80.0, text="啊啊啊啊。"),
    ]

    windows, stats = select_sparse_review_windows(segments, max_windows=2)

    assert len(windows) == 2
    assert [window.segment_indexes for window in windows] == [(1,), (3,)]
    assert stats["uses_recording_name"] is False
    assert stats["uses_fixed_transcript_phrases"] is False
    assert stats["selected_window_count"] == 2


def test_bounded_independent_review_covers_every_short_recording_window():
    segments = [
        Segment(
            start=float(index * 22),
            end=float(index * 22 + 8),
            text="这是正常密度且语义完整的一段通用转写内容。",
        )
        for index in range(5)
    ]

    windows, stats = select_bounded_independent_review_windows(
        segments,
        max_windows=12,
    )

    assert len(windows) == 5
    assert stats["coverage_mode"] == "full"
    assert stats["selected_window_count"] == 5
    assert stats["uses_recording_name"] is False
    assert stats["uses_fixed_transcript_phrases"] is False


def test_bounded_independent_review_fully_covers_recordings_under_twenty_minutes():
    segments = [
        Segment(
            start=float(index * 22),
            end=float(index * 22 + 8),
            text="这是正常密度且语义完整的一段通用转写内容。",
        )
        for index in range(14)
    ]

    windows, stats = select_bounded_independent_review_windows(
        segments,
        max_windows=12,
    )

    assert len(windows) == 14
    assert stats["coverage_mode"] == "full"
    assert stats["selected_window_count"] == 14
    assert stats["coverage_probe_count"] == 14
    assert windows[0].start < 1.0
    assert windows[-1].start > 250.0


def test_bounded_independent_review_samples_recordings_over_twenty_minutes():
    segments = [
        Segment(
            start=float(index * 30),
            end=float(index * 30 + 8),
            text="这是正常密度且语义完整的一段通用转写内容。",
        )
        for index in range(80)
    ]

    windows, stats = select_bounded_independent_review_windows(
        segments,
        max_windows=12,
    )

    assert len(windows) == 12
    assert stats["coverage_mode"] == "structural_and_stratified"
    assert stats["selected_window_count"] == 12
    assert stats["duration_s"] > 1200.0


def test_long_sparse_review_reserves_stratified_coverage_probes():
    segments = [
        Segment(
            start=float(index * 30),
            end=float(index * 30 + 8),
            text="这是正常密度且语义完整的一段通用转写内容。",
        )
        for index in range(80)
    ]
    segments[40] = Segment(start=1200.0, end=1208.0, text="啊啊啊啊。")

    windows, stats = select_sparse_review_windows(segments, max_windows=6)

    assert len(windows) == 6
    assert stats["structural_window_count"] == 1
    assert stats["coverage_probe_count"] == 5
    coverage = [
        item
        for item in stats["diagnostics"]
        if item["selected"] and item["selection_kind"] == "coverage_probe"
    ]
    assert len(coverage) == 5
    assert coverage[0]["start"] < 300.0
    assert coverage[-1]["start"] > 2100.0
    assert all("stratified_coverage_probe" in item["reasons"] for item in coverage)


def test_long_sparse_review_always_covers_beginning_and_end():
    segments = [
        Segment(
            start=float(index * 30),
            end=float(index * 30 + 8),
            text="这是正常密度且语义完整的一段通用转写内容。",
        )
        for index in range(80)
    ]

    windows, stats = select_sparse_review_windows(segments, max_windows=12)

    assert windows[0].segment_indexes == (0,)
    assert windows[1].segment_indexes == (1,)
    assert windows[-1].segment_indexes == (79,)
    selected = [item for item in stats["diagnostics"] if item["selected"]]
    assert any("recording_start_probe" in item["reasons"] for item in selected)
    assert any("recording_end_probe" in item["reasons"] for item in selected)


def test_sparse_review_uses_generic_lexical_fragmentation_only_for_selection():
    segments = [
        Segment(start=0.0, end=8.0, text="今天我们讨论项目上线安排和缓存同步方案。"),
        Segment(start=30.0, end=38.0, text="这段转写夹杂活务教费中事等词组显得破碎。"),
        Segment(start=60.0, end=68.0, text="大家还有什么意见都可以直接提出来。"),
    ]

    windows, stats = select_sparse_review_windows(segments, max_windows=1)

    assert windows[0].segment_indexes == (1,)
    diagnostic = next(item for item in stats["diagnostics"] if item["selected"])
    assert "generic_lexical_fragmentation" in diagnostic["reasons"]
    assert stats["uses_recording_name"] is False
    assert stats["uses_fixed_transcript_phrases"] is False


def test_sparse_paraformer_review_transcribes_only_selected_windows(monkeypatch, tmp_path: Path):
    audio = tmp_path / "long.wav"
    audio.write_bytes(b"audio")
    calls = []

    class _Paraformer:
        def transcribe(self, audio_path, _options):
            calls.append(Path(audio_path).name)
            return TranscribeResult(
                audio=str(audio_path),
                language="zh",
                duration=30.0,
                transcribe_seconds=0.1,
                rtf=0.01,
                backend="funasr",
                model_id="paraformer-zh",
                segments=[Segment(start=0.0, end=30.0, text="疑点窗口复核文本")],
            )

    from scribe_py.core import strong_asr

    monkeypatch.setattr(
        strong_asr,
        "_extract_clip",
        lambda _audio, _start, _end, output: output.write_bytes(b"clip"),
    )
    windows = [
        ReviewWindow(start=100.0, end=120.0, segment_indexes=(5,)),
        ReviewWindow(start=900.0, end=918.0, segment_indexes=(40,)),
    ]

    segments, clip_count = _transcribe_paraformer_review_windows(
        _Paraformer(),
        audio,
        windows,
        language="zh",
        model_id="paraformer-zh",
        audio_end=2400.0,
    )

    assert clip_count == 2
    assert len(calls) == 2
    assert [(segment.start, segment.end) for segment in segments] == [
        (100.0, 120.0),
        (900.0, 918.0),
    ]


def _decision(
    *,
    risk: str = "high",
    reasons=None,
    snr=2.79,
    backend="sensevoice",
    timing_reliable=False,
    alignment_similarity=0.29,
    noise_floor=None,
    true_peak=None,
    paraformer_reason=None,
    suppressed_segments=0,
):
    return decide_auto_high_noise_review(
        quality_mode="standard",
        backend=backend,
        model_id="iic/SenseVoiceSmall",
        audio_quality={
            "risk_level": risk,
            "risk_reasons": reasons if reasons is not None else ["信噪比过低", "背景噪声明显"],
            "estimated_snr_db": snr,
            "noise_floor_dbfs": noise_floor,
            "true_peak_dbfs": true_peak,
        },
        transcription_stats={
            "timing_reliable": timing_reliable,
            "equal_char_ratio": alignment_similarity,
            "timing_alignment_reason": "source_anchor_text_too_different",
            "paraformer_preflight": (
                {"reason": paraformer_reason} if paraformer_reason else {}
            ),
            "non_speech_suppression": {
                "suppressed_segments": suppressed_segments,
            },
        },
    )


def test_high_noise_review_is_advisory_by_default(monkeypatch):
    monkeypatch.delenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", raising=False)

    decision = _decision()

    assert decision["recommended"] is True
    assert decision["eligible"] is False
    assert decision["auto_run_enabled"] is False
    assert decision["auto_run_reason"] == "standard_mode_advisory_only"
    assert decision["reason"] == "high_noise_severe_decode_disagreement"
    assert decision["strategy"] == "local_strong_asr_consensus"
    assert _decision(risk="medium")["reason"] == "audio_risk_not_high"
    assert _decision(backend="funasr")["reason"] == "not_sensevoice_primary"
    assert _decision(reasons=["静音占比过高"], snr=None)["reason"] == "high_risk_without_noise_evidence"


def test_auto_review_skips_low_snr_meeting_when_existing_alignment_is_reliable(monkeypatch):
    monkeypatch.delenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", raising=False)

    decision = _decision(
        reasons=["信噪比过低", "背景噪声明显"],
        snr=3.02,
        timing_reliable=True,
        alignment_similarity=0.6667,
    )

    assert decision["recommended"] is False
    assert decision["eligible"] is False
    assert decision["auto_run_enabled"] is False
    assert decision["reason"] == "high_noise_without_severe_decode_disagreement"
    assert decision["decode_disagreement_evidence"] == []


def test_auto_review_routes_compound_extreme_acoustic_risk_to_frozen_qwen(monkeypatch):
    monkeypatch.delenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", raising=False)

    decision = _decision(
        reasons=["疑似峰值削波/爆音", "信噪比过低", "背景噪声明显"],
        snr=2.14,
        noise_floor=-31.26,
        true_peak=0.0,
        timing_reliable=True,
        alignment_similarity=0.8519,
    )

    assert decision["recommended"] is True
    assert decision["eligible"] is False
    assert decision["auto_run_enabled"] is False
    assert decision["strategy"] == "qwen_lexical_frozen_timeline"
    assert decision["reason"] == "high_noise_extreme_acoustic_and_lexical_risk"
    assert decision["extreme_acoustic_evidence"] == [
        "estimated_snr_at_or_below_3db",
        "noise_floor_above_-40dbfs",
        "true_peak_at_or_above_-0.2dbfs",
    ]
    assert decision["lexical_disagreement_evidence"] == [
        "alignment_similarity_below_0.88"
    ]


def test_auto_review_routes_very_low_snr_and_high_noise_floor_without_clipping(monkeypatch):
    monkeypatch.delenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", raising=False)

    decision = _decision(
        reasons=["整体音量过低", "信噪比过低", "背景噪声偏高"],
        snr=1.05,
        noise_floor=-37.62,
        true_peak=-15.2,
        timing_reliable=True,
        alignment_similarity=0.85,
    )

    assert decision["eligible"] is False
    assert decision["auto_run_enabled"] is False
    assert decision["strategy"] == "qwen_lexical_frozen_timeline"
    assert decision["extreme_acoustic_evidence"] == [
        "estimated_snr_at_or_below_3db",
        "noise_floor_above_-40dbfs",
    ]


def test_auto_review_does_not_route_on_single_extreme_snr_measurement(monkeypatch):
    monkeypatch.delenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", raising=False)

    decision = _decision(
        snr=2.14,
        noise_floor=None,
        true_peak=None,
        timing_reliable=True,
        alignment_similarity=0.8519,
    )

    assert decision["recommended"] is False
    assert decision["eligible"] is False
    assert decision["strategy"] == "none"
    assert decision["extreme_acoustic_evidence"] == []


def test_auto_review_routes_long_extreme_noise_to_sparse_review(monkeypatch):
    monkeypatch.delenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", raising=False)

    decision = _decision(
        reasons=["疑似峰值削波/爆音", "信噪比过低", "背景噪声明显"],
        snr=2.81,
        noise_floor=-29.9,
        true_peak=0.5,
        timing_reliable=True,
        alignment_similarity=0.8983,
        paraformer_reason="recording_too_long",
    )

    assert decision["recommended"] is True
    assert decision["eligible"] is False
    assert decision["auto_run_enabled"] is False
    assert decision["strategy"] == "sparse_independent_consensus"
    assert decision["reason"] == "high_noise_long_recording_sparse_independent_review"
    assert decision["extreme_acoustic_evidence"]
    assert decision["severe_acoustic_evidence"] == []
    assert decision["lexical_disagreement_evidence"] == []


def test_extreme_noise_same_model_agreement_still_requests_independent_review(monkeypatch):
    monkeypatch.delenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", raising=False)

    decision = _decision(
        reasons=["信噪比过低", "背景噪声明显"],
        snr=1.35,
        noise_floor=-23.38,
        true_peak=-0.4,
        timing_reliable=True,
        alignment_similarity=0.9038,
    )

    assert decision["recommended"] is True
    assert decision["eligible"] is False
    assert decision["auto_run_enabled"] is False
    assert decision["strategy"] == "bounded_independent_consensus"
    assert decision["reason"] == "high_noise_severe_acoustic_bounded_independent_review"


def test_high_noise_vad_unsupported_text_triggers_bounded_independent_review(monkeypatch):
    monkeypatch.delenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", raising=False)

    decision = _decision(
        reasons=["信噪比过低", "背景噪声明显"],
        snr=1.77,
        noise_floor=-24.53,
        true_peak=-2.0,
        timing_reliable=True,
        alignment_similarity=0.9569,
        suppressed_segments=3,
    )

    assert decision["eligible"] is False
    assert decision["strategy"] == "bounded_independent_consensus"
    assert decision["reason"] == "high_noise_structural_bounded_independent_review"
    assert decision["structural_disagreement_evidence"] == [
        "vad_unsupported_text_suppressed"
    ]


def test_extreme_noise_long_recording_runs_sparse_independent_review(monkeypatch):
    monkeypatch.delenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", raising=False)

    decision = _decision(
        reasons=["疑似峰值削波/爆音", "信噪比过低", "背景噪声明显"],
        snr=1.4,
        noise_floor=-20.09,
        true_peak=0.0,
        timing_reliable=True,
        alignment_similarity=0.9757,
        paraformer_reason="recording_too_long",
    )

    assert decision["recommended"] is True
    assert decision["eligible"] is False
    assert decision["auto_run_enabled"] is False
    assert decision["strategy"] == "sparse_independent_consensus"
    assert decision["reason"] == "high_noise_long_recording_sparse_independent_review"
    assert decision["paraformer_preflight_reason"] == "recording_too_long"
    assert decision["severe_acoustic_evidence"] == [
        "estimated_snr_at_or_below_1.5db",
        "noise_floor_above_-25dbfs",
        "true_peak_at_or_above_-0.2dbfs",
    ]


def test_legacy_environment_disable_switch_cannot_enable_standard_review(monkeypatch):
    monkeypatch.setenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", "0")

    decision = _decision(
        snr=2.14,
        noise_floor=-31.26,
        true_peak=0.0,
        timing_reliable=True,
        alignment_similarity=0.8519,
    )

    assert decision["recommended"] is True
    assert decision["eligible"] is False
    assert decision["strategy"] == "qwen_lexical_frozen_timeline"
    assert decision["auto_run_enabled"] is False
    assert decision["auto_run_reason"] == "standard_mode_advisory_only"


def test_auto_review_fails_closed_when_alignment_evidence_is_missing(monkeypatch):
    monkeypatch.delenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", raising=False)

    decision = _decision(timing_reliable=None, alignment_similarity=None)

    assert decision["recommended"] is False
    assert decision["eligible"] is False
    assert decision["reason"] == "high_noise_without_severe_decode_disagreement"


def test_legacy_environment_switch_cannot_enable_standard_review(monkeypatch):
    monkeypatch.setenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", "1")

    decision = _decision()

    assert decision["recommended"] is True
    assert decision["eligible"] is False
    assert decision["auto_run_enabled"] is False
    assert decision["auto_run_reason"] == "standard_mode_advisory_only"
    assert decision["strategy"] == "local_strong_asr_consensus"


def test_legacy_disable_switch_keeps_standard_review_advisory(monkeypatch):
    monkeypatch.setenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", "0")

    decision = _decision()

    assert decision["recommended"] is True
    assert decision["eligible"] is False
    assert decision["reason"] == "high_noise_severe_decode_disagreement"
    assert decision["auto_run_reason"] == "standard_mode_advisory_only"
    assert decision["strategy"] == "local_strong_asr_consensus"


def test_consensus_rewrite_rejects_even_single_char_change_on_weak_window_similarity():
    primary = "前文我们流写分离这块需要改造后文。"
    paraformer = "前文我们读写分离这块需要改造后文。"
    qwen = "大量无关内容前文我们读写分离这块需要改造后文大量无关内容。"

    corrected, changes = consensus_rewrite(
        primary,
        paraformer,
        qwen,
        detector_text=paraformer,
        candidate_qwen_similarity=0.30,
        primary_qwen_similarity=0.29,
        detector_qwen_similarity=0.30,
    )

    assert corrected == primary
    assert changes == []


def test_consensus_rewrite_applies_bounded_internal_change_with_double_evidence():
    primary = "会议开始以后请大家认真检察材料然后继续讨论。"
    confirmed = "会议开始以后请大家认真检查材料然后继续讨论。"

    corrected, changes = consensus_rewrite(
        primary, confirmed, confirmed, detector_text=confirmed
    )

    assert corrected == confirmed
    assert changes == [
        {
            "from": "察",
            "to": "查",
            "normalized_from": "察",
            "normalized_to": "查",
            "left_context": "大家认真检",
            "right_context": "材料然后继",
            "evidence": "sensevoice_redecode_paraformer_qwen_context_agreement",
        }
    ]


def test_consensus_rewrite_rejects_change_at_unanchored_text_boundary():
    primary = "流写分离这块需要改造。"
    paraformer = "读写分离这块需要改造。"
    qwen = "读写分离这块需要改造。"

    corrected, changes = consensus_rewrite(
        primary, paraformer, qwen, detector_text=paraformer
    )

    assert corrected == primary
    assert changes == []


def test_consensus_rewrite_does_not_rewrite_repeated_disfluency():
    primary = "今天吃完饭饭饭以后继续开会。"
    paraformer = "今天吃完晚晚饭以后继续开会。"
    qwen = "今天吃完晚晚饭以后继续开会。"

    corrected, changes = consensus_rewrite(
        primary, paraformer, qwen, detector_text=paraformer
    )

    assert corrected == primary
    assert changes == []


def test_consensus_rewrite_does_not_change_pronouns_or_structural_particles():
    for primary, secondary in [
        ("王小姐说她上午提交材料。", "王小姐说他上午提交材料。"),
        ("把医疗费都拿回去。", "把医疗费的拿回去。"),
    ]:
        corrected, changes = consensus_rewrite(
            primary, secondary, secondary, detector_text=secondary
        )

        assert corrected == primary
        assert changes == []


class _FakeTranscriber:
    def __init__(self, *, risk: str):
        self.risk = risk

    def transcribe(self, audio, options, on_progress=None):
        return TranscribeResult(
            audio=str(audio),
            language="zh",
            duration=5.0,
            transcribe_seconds=1.0,
            rtf=0.2,
            backend="sensevoice",
            model_id="iic/SenseVoiceSmall",
            segments=[
                Segment(
                    start=0.0,
                    end=5.0,
                    text="这是原始转写。",
                    sync_cues=[{"start": 0.0, "end": 5.0, "text": "这是原始转写。"}],
                )
            ],
            filter_stats={
                "timing_reliable": self.risk != "high",
                "equal_char_ratio": 0.29 if self.risk == "high" else 0.9,
                "timing_alignment_reason": (
                    "source_anchor_text_too_different" if self.risk == "high" else ""
                ),
                "audio_quality": {
                    "risk_level": self.risk,
                    "risk_reasons": ["信噪比过低"] if self.risk == "high" else [],
                    "estimated_snr_db": 2.79 if self.risk == "high" else 24.0,
                },
                "audio_standardization": {"path": str(audio)},
                "text_normalization": {},
            },
        )


def test_standard_mode_high_noise_preserves_text_without_starting_review(monkeypatch, tmp_path: Path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", "1")
    monkeypatch.setattr(ipc, "make_transcriber", lambda _backend: _FakeTranscriber(risk="high"))

    from scribe_py.core import strong_asr

    calls = []

    monkeypatch.setattr(
        strong_asr,
        "run_strong_asr_review",
        lambda *_args, **_kwargs: calls.append(True),
    )
    payload = ipc.handle_transcribe(
        {"audio": str(audio), "backend": "sensevoice", "asr_quality_mode": "standard"}
    )

    assert payload["segments"][0]["start"] == 0.0
    assert payload["segments"][0]["end"] == 5.0
    assert payload["filter_stats"]["asr_quality_mode"] == "standard"
    stats = payload["filter_stats"]["strong_asr"]
    assert calls == []
    assert stats["enabled"] is False
    assert stats["trigger"] == "none"
    assert stats["review_recommended"] is True
    assert stats["reason"] == "high_noise_severe_decode_disagreement"
    assert stats["auto_review_decision"]["eligible"] is False


def test_standard_mode_reviews_only_generic_high_signal_windows(monkeypatch, tmp_path: Path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")

    class _AnomalousTranscriber(_FakeTranscriber):
        def transcribe(self, audio_path, options, on_progress=None):
            result = super().transcribe(audio_path, options, on_progress)
            result.segments = [
                Segment(start=0.0, end=8.0, text="正常会议开场内容。"),
                Segment(start=8.0, end=16.0, text="出现read这样的异常碎片。"),
                Segment(start=16.0, end=24.0, text="正常会议结束内容。"),
            ]
            return result

    monkeypatch.setattr(
        ipc,
        "make_transcriber",
        lambda _backend: _AnomalousTranscriber(risk="high"),
    )
    from scribe_py.core import strong_asr

    captured = {}

    def fake_review(_audio, segments, **kwargs):
        captured.update(kwargs)
        return list(segments), {
            "mode": "local_strong_asr_consensus",
            "enabled": True,
            "applied": False,
            "timeline_preserved": True,
            "reason": "no_confirmed_consensus_changes",
        }

    monkeypatch.setattr(strong_asr, "run_strong_asr_review", fake_review)
    payload = ipc.handle_transcribe(
        {"audio": str(audio), "backend": "sensevoice", "asr_quality_mode": "standard"}
    )

    windows = captured["detector_windows"]
    assert len(windows) == 1
    assert windows[0].start < 8.0 < windows[0].end
    stats = payload["filter_stats"]["strong_asr"]
    assert stats["trigger"] == "standard_selective"
    assert stats["standard_review_selection"]["uses_recording_name"] is False
    assert stats["standard_review_selection"]["uses_fixed_transcript_phrases"] is False
    assert stats["standard_review_selection"]["uses_coverage_probes"] is False


def test_standard_mode_does_not_consume_cached_paraformer_anchor(monkeypatch, tmp_path: Path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    detector = [Segment(start=0.0, end=5.0, text="这是独立锚点。")]

    class _Transcriber(_FakeTranscriber):
        def strong_asr_detector_snapshot(self):
            return detector, "paraformer_recovery_timing_anchor", True

    monkeypatch.setenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", "1")
    monkeypatch.setattr(ipc, "make_transcriber", lambda _backend: _Transcriber(risk="high"))

    from scribe_py.core import strong_asr

    captured = {}

    monkeypatch.setattr(
        strong_asr,
        "run_strong_asr_review",
        lambda *_args, **kwargs: captured.update(kwargs),
    )

    ipc.handle_transcribe(
        {"audio": str(audio), "backend": "sensevoice", "asr_quality_mode": "standard"}
    )

    assert captured == {}


def test_cli_strong_mode_passes_cached_paraformer_detector(monkeypatch, tmp_path: Path):
    from scribe_py import __main__ as cli
    from scribe_py.core import strong_asr

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    detector = [Segment(start=0.0, end=5.0, text="这是独立锚点。")]
    result = _FakeTranscriber(risk="low").transcribe(audio, None)
    captured = {}

    def fake_review(_audio, segments, **kwargs):
        captured.update(kwargs)
        return list(segments), {
            "mode": "local_strong_asr_consensus",
            "enabled": True,
            "applied": False,
            "reason": "no_consensus_changes",
        }

    monkeypatch.setattr(strong_asr, "run_strong_asr_review", fake_review)

    cli._apply_quality_mode(
        result,
        audio,
        "strong",
        detector_segments=detector,
        detector_source="paraformer_timing_anchor",
    )

    assert captured["detector_segments"] == detector
    assert captured["detector_source"] == "paraformer_timing_anchor"


def test_standard_mode_high_noise_preserves_text_timeline_and_cursor_cues(
    monkeypatch,
    tmp_path: Path,
):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    monkeypatch.delenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", raising=False)
    monkeypatch.setattr(ipc, "make_transcriber", lambda _backend: _FakeTranscriber(risk="high"))

    from scribe_py.core import strong_asr

    calls = []

    monkeypatch.setattr(
        strong_asr,
        "run_strong_asr_review",
        lambda *_args, **_kwargs: calls.append(True),
    )

    payload = ipc.handle_transcribe(
        {"audio": str(audio), "backend": "sensevoice", "asr_quality_mode": "standard"}
    )

    stats = payload["filter_stats"]["strong_asr"]
    assert calls == []
    assert stats["enabled"] is False
    assert stats["review_recommended"] is True
    assert stats["trigger"] == "none"
    assert stats["reason"] == "high_noise_severe_decode_disagreement"
    assert stats["auto_review_decision"]["eligible"] is False
    assert stats["auto_review_decision"]["recommended"] is True
    assert stats["auto_review_decision"]["auto_run_reason"] == "standard_mode_advisory_only"
    assert payload["segments"][0]["start"] == 0.0
    assert payload["segments"][0]["end"] == 5.0
    assert payload["segments"][0]["sync_cues"] == [
        {"start": 0.0, "end": 5.0, "text": "这是原始转写。"}
    ]


def test_standard_mode_never_calls_a_review_that_could_change_timeline_or_cues(monkeypatch, tmp_path: Path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    monkeypatch.delenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", raising=False)
    monkeypatch.setattr(ipc, "make_transcriber", lambda _backend: _FakeTranscriber(risk="high"))

    from scribe_py.core import strong_asr

    monkeypatch.setattr(
        strong_asr,
        "run_strong_asr_review",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("standard mode must not run a second ASR review")
        ),
    )

    payload = ipc.handle_transcribe(
        {"audio": str(audio), "backend": "sensevoice", "asr_quality_mode": "standard"}
    )

    segment = payload["segments"][0]
    assert segment["start"] == 0.0
    assert segment["end"] == 5.0
    assert segment["text"] == "这是原始转写。"
    assert segment["sync_cues"] == [
        {"start": 0.0, "end": 5.0, "text": "这是原始转写。"}
    ]
    stats = payload["filter_stats"]["strong_asr"]
    assert stats["enabled"] is False
    assert stats["applied"] is False
    assert stats["trigger"] == "none"


def test_standard_mode_high_noise_never_builds_independent_review_reference(
    monkeypatch,
    tmp_path: Path,
):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")

    class _ExtremeTranscriber(_FakeTranscriber):
        def strong_asr_detector_snapshot(self):
            return (
                [Segment(start=0.0, end=5.0, text="这是缓存的分块解码。")],
                "sensevoice_wallclock_anchor",
                False,
            )

        def transcribe(self, audio_path, options, on_progress=None):
            result = super().transcribe(audio_path, options, on_progress)
            result.filter_stats.update({
                "timing_reliable": True,
                "equal_char_ratio": 0.8519,
                "timing_alignment_reason": "",
                "audio_quality": {
                    "risk_level": "high",
                    "risk_reasons": [
                        "疑似峰值削波/爆音",
                        "信噪比过低",
                        "背景噪声明显",
                    ],
                    "estimated_snr_db": 2.14,
                    "noise_floor_dbfs": -31.26,
                    "true_peak_dbfs": 0.0,
                },
            })
            return result

    monkeypatch.delenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", raising=False)
    monkeypatch.setattr(ipc, "make_transcriber", lambda _backend: _ExtremeTranscriber(risk="high"))

    from scribe_py.core import qwen_lexical_review, strong_asr

    qwen_calls = []
    strong_calls = []

    monkeypatch.setattr(
        qwen_lexical_review,
        "run_qwen_lexical_review",
        lambda *_args, **_kwargs: qwen_calls.append(True),
    )
    monkeypatch.setattr(
        strong_asr,
        "run_strong_asr_review",
        lambda *_args, **kwargs: strong_calls.append(kwargs),
    )

    payload = ipc.handle_transcribe(
        {"audio": str(audio), "backend": "sensevoice", "asr_quality_mode": "standard"}
    )

    stats = payload["filter_stats"]["strong_asr"]
    assert qwen_calls == []
    assert strong_calls == []
    assert stats["mode"] == "local_strong_asr_consensus"
    assert stats["enabled"] is False
    assert stats["applied"] is False
    assert stats["reason"] == "high_noise_extreme_acoustic_and_lexical_risk"
    assert stats["trigger"] == "none"
    assert stats["auto_review_decision"]["reason"] == (
        "high_noise_extreme_acoustic_and_lexical_risk"
    )
    assert payload["segments"][0]["start"] == 0.0
    assert payload["segments"][0]["end"] == 5.0


def test_standard_long_high_noise_is_advisory_without_a_sparse_second_decode(monkeypatch, tmp_path: Path):
    audio = tmp_path / "long.wav"
    audio.write_bytes(b"audio")

    class _LongExtremeTranscriber(_FakeTranscriber):
        def strong_asr_detector_snapshot(self):
            return ([], "", False)

        def transcribe(self, audio_path, options, on_progress=None):
            result = super().transcribe(audio_path, options, on_progress)
            result.duration = 1300.0
            result.segments = [
                Segment(
                    start=100.0,
                    end=112.0,
                    text="你说什么。",
                    sync_cues=[{"start": 100.0, "end": 112.0, "text": "你说什么。"}],
                )
            ]
            result.filter_stats.update({
                "timing_reliable": True,
                "equal_char_ratio": 0.9757,
                "timing_alignment_reason": "",
                "paraformer_preflight": {"reason": "recording_too_long"},
                "audio_quality": {
                    "risk_level": "high",
                    "risk_reasons": ["信噪比过低", "背景噪声明显"],
                    "estimated_snr_db": 1.4,
                    "noise_floor_dbfs": -20.09,
                    "true_peak_dbfs": 0.0,
                },
            })
            return result

    monkeypatch.delenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", raising=False)
    monkeypatch.setattr(
        ipc,
        "make_transcriber",
        lambda _backend: _LongExtremeTranscriber(risk="high"),
    )

    from scribe_py.core import strong_asr

    captured = {}
    monkeypatch.setattr(
        strong_asr,
        "run_strong_asr_review",
        lambda *_args, **kwargs: captured.update(kwargs),
    )

    payload = ipc.handle_transcribe(
        {"audio": str(audio), "backend": "sensevoice", "asr_quality_mode": "standard"}
    )

    stats = payload["filter_stats"]["strong_asr"]
    assert captured == {}
    assert stats["enabled"] is False
    assert stats["trigger"] == "none"
    assert stats["review_recommended"] is True
    assert stats["auto_review_decision"]["reason"] == (
        "high_noise_long_recording_sparse_independent_review"
    )


def test_standard_mode_high_noise_does_not_run_qwen_with_a_cached_paraformer_reference(
    monkeypatch,
    tmp_path: Path,
):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")

    class _ExtremeTranscriber(_FakeTranscriber):
        def strong_asr_detector_snapshot(self):
            return (
                [Segment(start=0.0, end=5.0, text="这是独立的分块解码。")],
                "paraformer_recovery_timing_anchor",
                True,
            )

        def transcribe(self, audio_path, options, on_progress=None):
            result = super().transcribe(audio_path, options, on_progress)
            result.filter_stats.update({
                "timing_reliable": True,
                "equal_char_ratio": 0.8519,
                "timing_alignment_reason": "",
                "audio_quality": {
                    "risk_level": "high",
                    "risk_reasons": ["信噪比过低", "背景噪声明显"],
                    "estimated_snr_db": 2.14,
                    "noise_floor_dbfs": -31.26,
                    "true_peak_dbfs": 0.0,
                },
            })
            return result

    monkeypatch.delenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", raising=False)
    monkeypatch.setattr(ipc, "make_transcriber", lambda _backend: _ExtremeTranscriber(risk="high"))

    from scribe_py.core import qwen_lexical_review

    captured = {}

    monkeypatch.setattr(
        qwen_lexical_review,
        "run_qwen_lexical_review",
        lambda *_args, **kwargs: captured.update(kwargs),
    )

    payload = ipc.handle_transcribe(
        {"audio": str(audio), "backend": "sensevoice", "asr_quality_mode": "standard"}
    )

    stats = payload["filter_stats"]["strong_asr"]
    assert stats["enabled"] is False
    assert stats["trigger"] == "none"
    assert captured == {}


def test_standard_mode_does_not_review_low_risk_audio(monkeypatch, tmp_path: Path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(ipc, "make_transcriber", lambda _backend: _FakeTranscriber(risk="low"))

    from scribe_py.core import strong_asr

    calls = []
    monkeypatch.setattr(
        strong_asr,
        "run_strong_asr_review",
        lambda *_args, **_kwargs: calls.append(True),
    )
    payload = ipc.handle_transcribe(
        {"audio": str(audio), "backend": "sensevoice", "asr_quality_mode": "standard"}
    )

    stats = payload["filter_stats"]["strong_asr"]
    assert stats["enabled"] is False
    assert stats["review_recommended"] is False
    assert stats["reason"] == "audio_risk_not_high"
    assert calls == []


def test_invalid_quality_mode_fails_closed_to_standard_advisory_policy(monkeypatch, tmp_path: Path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setenv("LOCALSCRIBE_AUTO_HIGH_NOISE_ASR_REVIEW", "1")
    monkeypatch.setattr(ipc, "make_transcriber", lambda _backend: _FakeTranscriber(risk="high"))

    from scribe_py.core import strong_asr

    calls = []

    monkeypatch.setattr(
        strong_asr,
        "run_strong_asr_review",
        lambda *_args, **_kwargs: calls.append(True),
    )
    payload = ipc.handle_transcribe(
        {"audio": str(audio), "backend": "sensevoice", "asr_quality_mode": "unexpected"}
    )

    assert payload["filter_stats"]["asr_quality_mode"] == "standard"
    stats = payload["filter_stats"]["strong_asr"]
    assert calls == []
    assert stats["enabled"] is False
    assert stats["trigger"] == "none"
    assert stats["review_recommended"] is True


def test_low_similarity_window_skips_expensive_qwen_and_never_replaces_primary(
    monkeypatch,
    tmp_path: Path,
):
    from scribe_py.core import strong_asr
    from scribe_py.core import transcriber_funasr, transcriber_qwen3

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    primary = [
        Segment(
            start=0.0,
            end=5.0,
            text="这是主模型保留的原始正文。",
            sync_cues=[{"start": 0.0, "end": 5.0, "text": "这是主模型保留的原始正文。"}],
        )
    ]
    qwen_calls = []
    paraformer_calls = []

    class _Paraformer:
        def __init__(self, *, backend_name):
            assert backend_name == "funasr"

        def transcribe(self, audio_path, options):
            paraformer_calls.append(Path(audio_path))
            return TranscribeResult(
                audio=str(audio_path),
                language="zh",
                duration=5.0,
                transcribe_seconds=0.1,
                rtf=0.02,
                backend="funasr",
                model_id="paraformer",
                segments=[Segment(start=0.0, end=5.0, text="完全不同的第二模型候选内容。")],
            )

    class _Qwen:
        def transcribe(self, audio_path, options):
            qwen_calls.append(Path(audio_path))
            return TranscribeResult(
                audio=str(audio_path),
                language="zh",
                duration=5.0,
                transcribe_seconds=0.1,
                rtf=0.02,
                backend="qwen3",
                model_id=options.model_id,
                segments=[Segment(start=0.0, end=5.0, text="完全不同的第二模型候选内容。")],
            )

    monkeypatch.setattr(strong_asr, "_qwen_model_cached", lambda _model: True)
    monkeypatch.setattr(transcriber_funasr, "model_cached", lambda _model: True)
    monkeypatch.setattr(transcriber_funasr, "FunASRTranscriber", _Paraformer)
    monkeypatch.setattr(transcriber_qwen3, "Qwen3ASRTranscriber", _Qwen)
    monkeypatch.setattr(
        strong_asr,
        "_extract_clip",
        lambda _audio, _start, _end, output: output.write_bytes(b"clip"),
    )
    monkeypatch.setattr(
        strong_asr,
        "consensus_rewrite",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("audit-only windows must not enter automatic rewrite")
        ),
    )

    output, stats = run_strong_asr_review(
        audio,
        primary,
        detector_segments=[Segment(start=0.0, end=5.0, text="同模型时间锚点。")],
        detector_source="sensevoice_wallclock_anchor",
    )

    assert qwen_calls == []
    assert len(paraformer_calls) == 1
    assert output == primary
    assert output[0].text == "这是主模型保留的原始正文。"
    assert output[0].sync_cues == primary[0].sync_cues
    assert stats["candidate_window_count"] == 1
    assert stats["structurally_reviewable_window_count"] == 0
    assert stats["primary_redecode_window_count"] == 0
    assert stats["qwen_tile_count"] == 0
    assert stats["qwen_skipped_window_count"] == 1
    assert stats["reviewed_windows"] == 1
    assert stats["audit_only_window_count"] == 1
    assert stats["replacement_count"] == 0
    assert stats["applied"] is False
    assert stats["timeline_preserved"] is True
    assert stats["detector_reuse_rejected"] is True
    assert stats["detector_reuse_rejected_reason"] == "detector_is_not_independent_paraformer"
    assert stats["reason"] == "audit_only_candidates_recorded"

    candidate = stats["audit_only_candidates"][0]
    assert candidate["reason"] == "primary_para_similarity_below_0.45"
    assert candidate["primary_para_similarity"] < 0.45
    assert candidate["primary"] == "这是主模型保留的原始正文。"
    assert candidate["paraformer"] == "完全不同的第二模型候选内容。"
    assert candidate["qwen"] == ""
    assert candidate["qwen_skipped_reason"] == "audit_only_window"

    diagnostic = stats["window_diagnostics"][0]
    assert diagnostic["audit_only"] is True
    assert diagnostic["audit_only_reason"] == "primary_para_similarity_below_0.45"
    assert diagnostic["auto_replace_allowed"] is False
    assert diagnostic["confirmed_change_count"] == 0
    assert diagnostic["candidates"] == {
        "primary": candidate["primary"],
        "primary_redecode": "",
        "paraformer": candidate["paraformer"],
        "qwen": candidate["qwen"],
    }
    assert diagnostic["qwen_review_eligible"] is False
    assert diagnostic["qwen_skipped_reason"] == "audit_only_window"


def test_qwen_runtime_failure_is_contained_after_independent_review(monkeypatch, tmp_path: Path):
    from scribe_py.core import strong_asr

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    primary = [Segment(start=0.0, end=5.0, text="主模型内容。")]
    detector = [Segment(start=0.0, end=5.0, text="第二模型不同内容。")]

    monkeypatch.setattr(strong_asr, "_qwen_model_cached", lambda _model: True)
    monkeypatch.setattr(
        strong_asr,
        "_qwen_runtime_available",
        lambda: (False, "mlx_probe_unavailable"),
    )

    output, stats = run_strong_asr_review(
        audio,
        primary,
        detector_segments=detector,
        detector_source="paraformer_review_pass",
    )

    assert output == primary
    assert stats["reason"] == "qwen_runtime_unavailable"
    assert stats["qwen_runtime_available"] is False
    assert stats["candidate_window_count"] == 1
    assert stats["window_diagnostics"][0]["qwen_skipped_reason"] == "qwen_runtime_unavailable"


def test_full_independent_coverage_uses_minute_tiles_not_one_decode_per_window(
    monkeypatch,
    tmp_path: Path,
):
    from scribe_py.core import strong_asr, transcriber_funasr

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    primary = [
        Segment(start=float(index * 22), end=float(index * 22 + 8), text="主模型内容。")
        for index in range(13)
    ]
    windows, selection = select_bounded_independent_review_windows(primary)
    calls: list[Path] = []

    class _Paraformer:
        def __init__(self, *, backend_name):
            assert backend_name == "funasr"

        def transcribe(self, audio_path, options):
            calls.append(Path(audio_path))
            return TranscribeResult(
                audio=str(audio_path),
                language="zh",
                duration=60.0,
                transcribe_seconds=0.1,
                rtf=0.01,
                backend="funasr",
                model_id=options.model_id,
                segments=[Segment(start=0.0, end=60.0, text="第二模型内容。")],
            )

    monkeypatch.setattr(strong_asr, "_qwen_model_cached", lambda _model: True)
    monkeypatch.setattr(
        strong_asr,
        "_qwen_runtime_available",
        lambda: (False, "mlx_probe_unavailable"),
    )
    monkeypatch.setattr(transcriber_funasr, "model_cached", lambda _model: True)
    monkeypatch.setattr(transcriber_funasr, "FunASRTranscriber", _Paraformer)
    monkeypatch.setattr(
        strong_asr,
        "_extract_clip",
        lambda _audio, _start, _end, output: output.write_bytes(b"clip"),
    )

    _output, stats = run_strong_asr_review(
        audio,
        primary,
        detector_windows=windows,
    )

    assert selection["coverage_mode"] == "full"
    assert len(windows) == 13
    assert stats["detector_scope"] == "full_recording"
    assert stats["paraformer_tile_count"] == len(calls) == 5


def test_protected_detector_delta_skips_redecode_and_qwen(monkeypatch, tmp_path: Path):
    from scribe_py.core import strong_asr
    from scribe_py.core import transcriber_qwen3

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    primary_text = "会议开始以后嗯请大家继续讨论并确认最终方案。"
    detector_text = "会议开始以后啊请大家继续讨论并确认最终方案。"
    primary = [Segment(start=0.0, end=5.0, text=primary_text)]
    detector = [Segment(start=0.0, end=5.0, text=detector_text)]

    class _UnexpectedQwen:
        def __init__(self):
            raise AssertionError("protected deltas must not start Qwen")

    monkeypatch.setattr(strong_asr, "_qwen_model_cached", lambda _model: True)
    monkeypatch.setattr(transcriber_qwen3, "Qwen3ASRTranscriber", _UnexpectedQwen)
    monkeypatch.setattr(
        strong_asr,
        "_transcribe_primary_redecode_windows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("protected deltas must not start primary re-decode")
        ),
    )

    output, stats = run_strong_asr_review(
        audio,
        primary,
        detector_segments=detector,
        detector_source="paraformer_timing_anchor",
    )

    assert output == primary
    assert stats["candidate_window_count"] == 1
    assert stats["structurally_reviewable_window_count"] == 0
    assert stats["primary_redecode_window_count"] == 0
    assert stats["qwen_tile_count"] == 0
    assert stats["qwen_skipped_window_count"] == 1
    assert stats["replacement_count"] == 0
    assert stats["timeline_preserved"] is True


def test_review_uses_primary_redecode_as_candidate_and_two_independent_confirmers(
    monkeypatch,
    tmp_path: Path,
):
    from scribe_py.core import strong_asr
    from scribe_py.core import transcriber_qwen3

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    primary_text = "会议开始以后请大家认真检察材料然后继续讨论。"
    confirmed_text = "会议开始以后请大家认真检查材料然后继续讨论。"
    primary = [
        Segment(
            start=0.0,
            end=5.0,
            text=primary_text,
            sync_cues=[{"start": 0.0, "end": 5.0, "text": primary_text}],
        )
    ]
    detector = [Segment(start=0.0, end=5.0, text=confirmed_text)]

    class _Qwen:
        def transcribe(self, audio_path, options):
            return TranscribeResult(
                audio=str(audio_path),
                language="zh",
                duration=5.0,
                transcribe_seconds=0.1,
                rtf=0.02,
                backend="qwen3",
                model_id=options.model_id,
                segments=[Segment(start=0.0, end=5.0, text=confirmed_text)],
            )

    monkeypatch.setattr(strong_asr, "_qwen_model_cached", lambda _model: True)
    monkeypatch.setattr(transcriber_qwen3, "Qwen3ASRTranscriber", _Qwen)
    monkeypatch.setattr(
        strong_asr,
        "_extract_clip",
        lambda _audio, _start, _end, output: output.write_bytes(b"clip"),
    )
    monkeypatch.setattr(
        strong_asr,
        "_transcribe_primary_redecode_windows",
        lambda _audio, windows, **_kwargs: {
            window_index: confirmed_text for window_index, _window in windows
        },
    )

    output, stats = run_strong_asr_review(
        audio,
        primary,
        detector_segments=detector,
        detector_source="paraformer_timing_anchor",
    )

    assert output[0].text == confirmed_text
    assert output[0].start == primary[0].start
    assert output[0].end == primary[0].end
    assert stats["detector_reused"] is True
    assert stats["primary_redecode_window_count"] == 1
    assert stats["replacement_count"] == 1
    assert stats["timeline_preserved"] is True
