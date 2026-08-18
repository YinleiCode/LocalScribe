from __future__ import annotations

import sys
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from scribe_py.core.asr_quality import (
    build_asr_quality_report,
    render_asr_quality_markdown,
    select_asr_review_segments,
    write_asr_quality_reports,
)
from scribe_py.core.types import Segment


def test_asr_quality_reports_hotword_coverage_and_near_miss():
    segments = [
        Segment(start=0, end=5, text="今天李慧和金子一起讨论青年团契。"),
        Segment(start=5, end=10, text="后面会继续安排服侍。"),
    ]

    report = build_asr_quality_report(
        segments,
        hotwords=["李会", "金子", "青年团契"],
        text_normalization={"asr_review_segment_count": 1, "asr_review_segments": [{"index": 0, "start": 0, "end": 5, "text": segments[0].text, "reasons": ["疑似人名混淆"]}]},
    )

    assert report["hotwords"]["coverage"] == 0.6667
    assert "李会" in report["hotwords"]["missing_terms"]
    assert report["hotwords"]["near_misses"][0]["term"] == "李会"
    assert report["risk_level"] == "medium"


def test_asr_quality_char_count_uses_final_non_whitespace_transcript():
    segments = [Segment(start=0, end=2, text="中文 ABC，123。")]

    report = build_asr_quality_report(segments)

    assert report["chars"] == len("中文ABC，123。")
    assert report["cjk_chars"] == 2


def test_asr_quality_markdown_is_chinese_and_includes_review_table():
    report = build_asr_quality_report(
        [Segment(start=1.2, end=3.4, text="没有标点")],
        hotwords=[],
        text_normalization={"asr_review_segment_count": 0, "asr_review_segments": []},
        audio_preprocessing={"enabled": True, "applied": True, "mode": "adaptive", "applied_filters": ["downmix_mono", "resample_16k"]},
    )

    markdown = render_asr_quality_markdown(report)

    assert "# ASR 质量检查" in markdown
    assert "工业化转录链路" in markdown
    assert "人工 CER 回归评分" in markdown
    assert "热词命中" in markdown
    assert "同音/近音实体一致性" in markdown
    assert "疑点段" in markdown


def test_asr_quality_includes_audio_quality_gate():
    report = build_asr_quality_report(
        [Segment(start=0, end=2, text="这是一句正常内容。")],
        hotwords=[],
        text_normalization={"asr_review_segment_count": 0, "asr_review_segments": []},
        audio_quality={
            "risk_level": "high",
            "integrated_lufs": -35.2,
            "true_peak_dbfs": -0.1,
            "silence_ratio": 0.52,
            "risk_reasons": ["整体音量过低", "疑似峰值削波/爆音", "静音占比过高"],
        },
        audio_preprocessing={
            "enabled": True,
            "applied": True,
            "mode": "adaptive",
            "applied_filters": ["downmix_mono", "resample_16k", "loudness_normalization"],
            "skipped_actions": ["不删除静音, 避免破坏字幕/分人时间轴"],
        },
    )

    markdown = render_asr_quality_markdown(report)

    assert report["risk_level"] == "high"
    assert "音频质量高风险" in report["risk_reasons"]
    assert report["audio_preprocessing"]["mode"] == "adaptive"
    assert report["industry_pipeline"]["steps"][1]["status"] == "已应用"
    assert "音频质量风险: high" in markdown
    assert "音频预处理: adaptive / 已应用" in markdown
    assert "整体音量过低" in markdown


def test_asr_quality_uses_review_ratio_for_short_recordings():
    segments = [
        Segment(start=0, end=1, text="第一句。"),
        Segment(start=1, end=2, text="有点矫正嗯。"),
        Segment(start=2, end=3, text="第二句。"),
        Segment(start=3, end=4, text="守的来做人性化。"),
        Segment(start=4, end=5, text="第三句。"),
    ]

    report = build_asr_quality_report(
        segments,
        text_normalization={
            "asr_review_segment_count": 2,
            "asr_review_segments": [
                {"index": 1, "start": 1, "end": 2, "text": segments[1].text, "reasons": ["命中已知 ASR 易混淆词"]},
                {"index": 3, "start": 3, "end": 4, "text": segments[3].text, "reasons": ["命中明显不通顺 ASR 片段"]},
            ],
        },
    )

    assert report["review"]["segment_ratio"] == 0.4
    assert report["risk_level"] == "medium"
    assert "存在多处 ASR 疑点段" in report["risk_reasons"]


def test_asr_quality_does_not_use_known_recording_phrases_as_generic_evidence():
    segments = [
        Segment(
            start=0,
            end=18.6,
            text="就骚谣我，咱那调查去，我绝对不承受，他真的不辱我，我觉他好极了，我要不老实，我不会得当等。",
        ),
        Segment(start=18.6, end=24.0, text="后面这句基本正常。"),
    ]

    report = build_asr_quality_report(
        segments,
        text_normalization={"asr_review_segment_count": 0, "asr_review_segments": []},
    )

    assert "存在疑似语义/异常片段" not in report["risk_reasons"]
    assert report["review"]["generic_segment_count"] == 0
    assert report["review"]["strong_segment_count"] == 0


def test_asr_quality_includes_term_consistency_candidates_without_rewriting():
    segments = [
        Segment(start=0, end=5, text="今天张敏负责接口评审，王工会一起参加。"),
        Segment(start=5, end=10, text="刚才张明提到接口评审的风险，需要后续确认。"),
        Segment(start=10, end=15, text="张敏说接口评审结束后再同步给产品。"),
        Segment(start=15, end=20, text="如果张明那边确认接口评审通过，我们就继续发布。"),
    ]
    before = "\n".join(seg.text for seg in segments)

    report = build_asr_quality_report(
        segments,
        text_normalization={"asr_review_segment_count": 0, "asr_review_segments": []},
    )
    markdown = render_asr_quality_markdown(report)
    after = "\n".join(seg.text for seg in segments)

    assert after == before
    assert report["term_consistency"]["candidate_count"] >= 1
    assert any({"张敏", "张明"}.issubset(set(item["terms"])) for item in report["term_consistency"]["candidates"])
    assert "存在疑似同音/近音实体一致性候选" in report["spot_check_reasons"]
    assert "同音/近音实体一致性" in markdown
    assert "张敏、张明" in markdown or "张明、张敏" in markdown


def test_asr_quality_renders_phonetic_entity_candidates():
    segments = [
        Segment(start=0, end=5, text="今天兰艺和金子一起在群里沟通服侍。"),
        Segment(start=5, end=10, text="蓝衣同学后来也跟金子确认了安排。"),
        Segment(start=10, end=15, text="兰意和金子都说这个安排可以。"),
        Segment(start=15, end=20, text="兰依同学最后补充说会继续服侍。"),
    ]

    report = build_asr_quality_report(
        segments,
        text_normalization={"asr_review_segment_count": 0, "asr_review_segments": []},
    )
    markdown = render_asr_quality_markdown(report)

    assert "存在疑似同音/近音实体一致性候选" in report["spot_check_reasons"]
    assert "同音实体" in markdown
    assert "lan-yi" in markdown
    assert "系统不自动替换原文" in markdown


def test_asr_quality_does_not_flag_fixed_collocations_without_model_evidence():
    segments = [
        Segment(start=0, end=5, text="但是我并不知道他跟张个月开工资。"),
        Segment(start=5, end=10, text="夫妻之傅之间，没那么多对与错。"),
        Segment(start=10, end=15, text="到您这个岁数叫品淡如水，对吧？"),
        Segment(start=15, end=20, text="这句正常，不应该被标强疑点。"),
    ]

    report = build_asr_quality_report(
        segments,
        text_normalization={"asr_review_segment_count": 0, "asr_review_segments": []},
    )

    assert report["review"]["strong_segment_count"] == 0
    assert "存在疑似语义/异常片段" not in report["risk_reasons"]


def test_asr_quality_flags_independent_model_disagreement_without_fixed_phrases():
    segments = [
        Segment(start=0, end=5, text="第一段转写内容。"),
        Segment(start=5, end=10, text="任意领域的新录音内容。"),
    ]

    report = build_asr_quality_report(
        segments,
        text_normalization={"asr_review_segment_count": 0, "asr_review_segments": []},
        model_review={
            "window_diagnostics": [
                {
                    "start": 5.0,
                    "end": 10.0,
                    "primary_para_similarity": 0.42,
                    "qwen_review_eligible": False,
                    "qwen_skipped_reason": "qwen_runtime_unavailable",
                }
            ]
        },
    )

    assert report["review"]["strong_segment_count"] == 1
    assert report["review"]["segments"][0]["index"] == 1
    assert report["review"]["segments"][0]["reasons"] == ["独立模型转写存在分歧"]


def test_asr_quality_does_not_flag_suppressed_non_speech_punctuation():
    report = build_asr_quality_report(
        [
            Segment(start=0, end=1, text="", original_text="🎼。"),
            Segment(start=1, end=10, text="（非语音）", original_text="🎼吧见。"),
        ],
        text_normalization={"asr_review_segment_count": 0, "asr_review_segments": []},
        model_review={
            "window_diagnostics": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "primary_para_similarity": 0.2,
                    "qwen_review_eligible": True,
                    "confirmed_change_count": 0,
                }
            ]
        },
    )

    assert report["review"]["generic_segment_count"] == 0
    assert report["review"]["strong_segment_count"] == 0


def test_asr_quality_flags_suspicious_boundary_word_split():
    segments = [
        Segment(
            start=0,
            end=8,
            text="老人他今天起诉离婚呢到了法庭，他也就是想了解一下这个公。",
        ),
        Segment(start=8, end=9, text="司的情况。"),
        Segment(start=9, end=12, text="这句正常。"),
    ]

    report = build_asr_quality_report(
        segments,
        text_normalization={"asr_review_segment_count": 0, "asr_review_segments": []},
    )

    assert report["review"]["strong_segment_count"] == 1
    flagged = report["review"]["segments"][0]
    assert flagged["index"] == 1
    assert "疑似断句导致词语断裂" in flagged["reasons"]


def test_asr_quality_flags_repeated_words_for_manual_spot_check():
    segments = [
        Segment(start=0, end=12, text="对对啊，明白，那我去然后其实这个这个架构的话，在AW上面有一个官方博客。"),
        Segment(start=12, end=20, text="这句没有明显重复。"),
    ]

    report = build_asr_quality_report(
        segments,
        text_normalization={"asr_review_segment_count": 0, "asr_review_segments": []},
    )

    assert report["risk_level"] == "low"
    assert report["review"]["segment_count"] == 1
    assert report["review"]["strong_segment_count"] == 0
    assert "存在重复/口语抽查片段" in report["spot_check_reasons"]
    assert "存在重复/口语抽查片段" not in report["risk_reasons"]
    assert "疑似重复词" in report["review"]["segments"][0]["reasons"]


def test_asr_quality_does_not_escalate_isolated_punctuation_segments():
    segments = [
        Segment(start=0, end=0.3, text="。"),
        Segment(start=0.3, end=8, text="这是一段正常的会议讨论内容。"),
    ]

    report = build_asr_quality_report(
        segments,
        text_normalization={"asr_review_segment_count": 0, "asr_review_segments": []},
    )

    assert report["risk_level"] == "low"
    assert report["review"]["segment_count"] == 1
    assert report["review"]["strong_segment_count"] == 0


def test_select_asr_review_segments_defaults_to_strong_only():
    data = {
        "filter_stats": {
            "text_normalization": {
                "asr_review_segments": [
                    {
                        "index": 0,
                        "start": 0,
                        "end": 2,
                        "text": "对对啊，这个这个要看。",
                        "reasons": ["疑似重复词"],
                    },
                    {
                        "index": 1,
                        "start": 2,
                        "end": 4,
                        "text": "说的直接您好养。",
                        "reasons": ["命中家庭/调解场景 ASR 混淆"],
                    },
                    {
                        "index": 2,
                        "start": 4,
                        "end": 5,
                        "text": "。",
                        "reasons": ["只有标点/空白"],
                    },
                ]
            }
        }
    }

    selection = select_asr_review_segments(data)

    assert selection["total_segment_count"] == 3
    assert selection["strong_segment_count"] == 1
    assert selection["weak_segment_count"] == 2
    assert selection["skipped_weak_count"] == 2
    assert [item["index"] for item in selection["segments"]] == [1]


def test_select_asr_review_segments_all_includes_weak_items():
    data = {
        "asr_quality": {
            "review": {
                "segments": [
                    {"index": 0, "start": 0, "end": 1, "text": "。", "reasons": ["只有标点/空白"]},
                    {"index": 1, "start": 1, "end": 2, "text": "管有关个地。", "reasons": ["疑似明显语义不顺"]},
                ]
            }
        }
    }

    selection = select_asr_review_segments(data, scope="all")

    assert selection["total_segment_count"] == 2
    assert selection["strong_segment_count"] == 1
    assert selection["skipped_weak_count"] == 0
    assert [item["index"] for item in selection["segments"]] == [0, 1]


def test_select_asr_review_segments_reads_sidecar_quality_report(tmp_path: Path):
    transcript = tmp_path / "demo.json"
    transcript.write_text('{"segments": []}', encoding="utf-8")
    sidecar = tmp_path / "ASR质量检查.json"
    sidecar.write_text(
        """
        {
          "mode": "local_asr_quality",
          "review": {
            "segments": [
              {"index": 3, "start": 3, "end": 4, "text": "不辱我。", "reasons": ["疑似明显语义不顺"]}
            ]
          }
        }
        """,
        encoding="utf-8",
    )

    selection = select_asr_review_segments({}, transcript_json=transcript)

    assert selection["sources"] == ["transcript", "ASR质量检查.json"]
    assert selection["strong_segment_count"] == 1
    assert selection["segments"][0]["index"] == 3


def test_per_transcript_quality_sidecar_avoids_flat_batch_overwrites(tmp_path: Path):
    report = {
        "mode": "local_asr_quality",
        "review": {
            "segments": [
                {"index": 2, "start": 2, "end": 3, "text": "不辱我。", "reasons": ["疑似明显语义不顺"]}
            ]
        },
    }
    paths = write_asr_quality_reports(tmp_path, "meeting_a", report, per_transcript=True)
    transcript = tmp_path / "meeting_a.json"
    transcript.write_text('{"segments": []}', encoding="utf-8")

    selection = select_asr_review_segments({}, transcript_json=transcript)

    assert Path(paths["asr_quality_json"]).name == "meeting_a_ASR质量检查.json"
    assert selection["sources"] == ["transcript", "meeting_a_ASR质量检查.json"]
    assert selection["strong_segment_count"] == 1
