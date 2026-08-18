from __future__ import annotations

import sys
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from scribe_py.core.term_consistency import (
    apply_first_mention_phonetic_consistency,
    find_term_consistency_candidates,
)
from scribe_py.core.text_normalizer import normalize_segments
from scribe_py.core.types import Segment


def test_term_consistency_finds_repeated_near_terms_in_shared_context():
    segments = [
        Segment(start=0, end=5, text="今天张敏负责接口评审，王工会一起参加。"),
        Segment(start=5, end=10, text="刚才张明提到接口评审的风险，需要后续确认。"),
        Segment(start=10, end=15, text="张敏说接口评审结束后再同步给产品。"),
        Segment(start=15, end=20, text="如果张明那边确认接口评审通过，我们就继续发布。"),
    ]

    report = find_term_consistency_candidates(segments)

    assert report["mode"] == "term_consistency"
    assert report["candidate_count"] >= 1
    candidate = next(item for item in report["candidates"] if {"张敏", "张明"}.issubset(set(item["terms"])))
    assert candidate["action"] in {"review", "maybe_unify"}
    assert 0.0 < candidate["confidence"] <= 1.0
    assert candidate["total_count"] == 4
    assert candidate["suggested_canonical"] in {"张敏", None}
    assert {variant["text"]: variant["count"] for variant in candidate["variants"]}["张敏"] == 2
    assert {variant["text"]: variant["count"] for variant in candidate["variants"]}["张明"] == 2
    assert {"index", "start", "end", "text"}.issubset(candidate["contexts"][0])
    assert candidate["contexts"][0]["text"] == segments[0].text


def test_term_consistency_is_conservative_for_single_noise_or_unrelated_context():
    segments = [
        Segment(start=0, end=3, text="张敏今天负责接口评审。"),
        Segment(start=3, end=6, text="王强在讨论预算流程。"),
        Segment(start=6, end=9, text="天气不错，会议很短。"),
    ]

    report = find_term_consistency_candidates(segments)

    assert report["candidate_count"] == 0
    assert report["candidates"] == []


def test_term_consistency_does_not_force_other_recording_names_to_standard3_answers():
    segments = [
        Segment(start=0, end=5, text="今天李慧主持项目例会，兰琪负责记录。"),
        Segment(start=5, end=10, text="李会后来补充了项目例会里的预算问题。"),
        Segment(start=10, end=15, text="兰琪说金子系统下周上线，李慧需要确认。"),
        Segment(start=15, end=20, text="李会和兰琪继续讨论金子系统的灰度计划。"),
    ]

    original_text = "\n".join(segment.text for segment in segments)
    report = find_term_consistency_candidates(segments)
    after_text = "\n".join(segment.text for segment in segments)

    assert after_text == original_text
    assert "兰艺" not in after_text
    assert report["candidate_count"] >= 1
    assert all(candidate.get("suggested_canonical") != "兰艺" for candidate in report["candidates"])
    li_candidate = next(item for item in report["candidates"] if {"李慧", "李会"}.issubset(set(item["terms"])))
    assert li_candidate["action"] in {"review", "maybe_unify"}
    assert li_candidate["suggested_canonical"] in {"李慧", None}


def test_term_consistency_filters_sliding_window_sentence_fragments():
    segments = [
        Segment(start=0, end=5, text="其实我们推荐这个口径，你要根据现场情况确认。"),
        Segment(start=5, end=10, text="其实我们推荐这个口径，你要先看客户的数据。"),
        Segment(start=10, end=15, text="实我们推荐这个口径你要同步给项目组。"),
        Segment(start=15, end=20, text="发生了切换以后，生了切换记录也要保存。"),
    ]

    report = find_term_consistency_candidates(segments)
    terms = {term for candidate in report["candidates"] for term in candidate["terms"]}

    assert "其实我们" not in terms
    assert "实我们推" not in terms
    assert "个口径你" not in terms
    assert "口径你要" not in terms
    assert "发生了切" not in terms
    assert "生了切换" not in terms


def test_term_consistency_drops_large_low_confidence_chains():
    segments = [
        Segment(start=0, end=5, text="业务其实包括很多数据，比如我们要考虑方案。"),
        Segment(start=5, end=10, text="其实业务包括成本和数据，也包括客户方案。"),
        Segment(start=10, end=15, text="我们讨论业务数据、客户成本、方案范围。"),
        Segment(start=15, end=20, text="这里还有业务目标、客户数据、成本方案。"),
    ]

    report = find_term_consistency_candidates(segments)

    assert all(len(candidate["terms"]) <= 6 for candidate in report["candidates"])
    all_terms = {term for candidate in report["candidates"] for term in candidate["terms"]}
    assert "业务" not in all_terms
    assert "其实" not in all_terms
    assert "包括" not in all_terms


def test_term_consistency_filters_generic_abstract_terms_and_verb_fragments():
    segments = [
        Segment(start=0, end=5, text="这个方案的方式需要再确认，数据库切换时要写数据库。"),
        Segment(start=5, end=10, text="这个方面的方案还没定，真正切换时不是写数据库这么简单。"),
        Segment(start=10, end=15, text="我们讨论方式和方案，也讨论个数据库连接怎么改。"),
        Segment(start=15, end=20, text="这个方式可能影响成本，那个方面也影响周期。"),
    ]

    report = find_term_consistency_candidates(segments)
    all_terms = {term for candidate in report["candidates"] for term in candidate["terms"]}

    assert "方式" not in all_terms
    assert "方案" not in all_terms
    assert "方面" not in all_terms
    assert "个数据库" not in all_terms
    assert "写数据库" not in all_terms


def test_term_consistency_filters_degree_adverb_sliding_fragments():
    segments = [
        Segment(start=0, end=5, text="这是非常之宝贵的提醒。"),
        Segment(start=5, end=10, text="这个关系非常好，大家都能理解。"),
        Segment(start=10, end=15, text="他讲得非常非常清楚。"),
        Segment(start=15, end=20, text="我们常常有时候也需要等一等。"),
    ]

    report = find_term_consistency_candidates(segments)
    all_terms = {term for candidate in report["candidates"] for term in candidate["terms"]}

    assert "常之" not in all_terms
    assert "常好" not in all_terms
    assert "常非" not in all_terms


def test_term_consistency_keeps_real_domain_name_variants():
    segments = [
        Segment(start=0, end=5, text="亚伯拉罕在这一段经文里回应神的呼召。"),
        Segment(start=5, end=10, text="后面亚巴拉罕继续跟随神，不只是遵守律法。"),
        Segment(start=10, end=15, text="亚伯拉罕的信心和律法之间有很重要的关系。"),
        Segment(start=15, end=20, text="如果把亚巴拉罕读错，讲章里的专名就不一致。"),
    ]

    report = find_term_consistency_candidates(segments)

    assert report["candidate_count"] >= 1
    assert any({"亚伯拉罕", "亚巴拉罕"}.issubset(set(item["terms"])) for item in report["candidates"])


def test_term_consistency_groups_homophone_entity_variants_without_rewriting():
    segments = [
        Segment(start=0, end=5, text="今天兰艺和金子一起在群里沟通服侍。"),
        Segment(start=5, end=10, text="蓝衣同学后来也跟金子确认了安排。"),
        Segment(start=10, end=15, text="兰意和金子都说这个安排可以。"),
        Segment(start=15, end=20, text="兰依同学最后补充说会继续服侍。"),
    ]
    before = "\n".join(segment.text for segment in segments)

    report = find_term_consistency_candidates(segments)
    after = "\n".join(segment.text for segment in segments)

    assert after == before
    candidate = next(
        item for item in report["candidates"] if {"兰艺", "蓝衣", "兰意", "兰依"}.issubset(set(item["terms"]))
    )
    assert candidate["kind"] == "phonetic_entity"
    assert candidate["action"] == "review"
    assert candidate["phonetic_key"] == "lan-yi"
    assert candidate["suggested_canonical"] is None
    assert "不自动替换" in candidate["reason"]


def test_term_consistency_flags_phonetic_entity_drift_near_anchor():
    segments = [
        Segment(start=0, end=5, text="今天兰艺和金子一起确认群里的安排。"),
        Segment(start=5, end=10, text="蓝一同学也跟金子说后面可以继续。"),
        Segment(start=10, end=15, text="因为对方是男人和金子，所以他不好意思。"),
        Segment(start=15, end=20, text="后来兰艺还是会继续参与服侍。"),
    ]

    report = find_term_consistency_candidates(segments)

    candidate = next(item for item in report["candidates"] if item.get("kind") == "entity_drift")
    assert "男人" in candidate["terms"]
    assert any(term in candidate["terms"] for term in {"兰艺", "蓝一"})
    assert candidate["action"] == "review"
    assert candidate["suggested_canonical"] is None
    assert "男人和金子" in candidate["reason"]


def test_term_consistency_does_not_force_homophone_to_standard3_spelling():
    segments = [
        Segment(start=0, end=5, text="蓝衣和金子先确认了安排。"),
        Segment(start=5, end=10, text="兰意同学后来跟金子说可以。"),
        Segment(start=10, end=15, text="兰依也在群里回复了金子。"),
    ]
    before = "\n".join(segment.text for segment in segments)

    report = find_term_consistency_candidates(segments)
    after = "\n".join(segment.text for segment in segments)

    assert after == before
    assert "兰艺" not in after
    assert all(candidate.get("suggested_canonical") != "兰艺" for candidate in report["candidates"])
    assert any(item.get("kind") == "phonetic_entity" for item in report["candidates"])


def test_term_consistency_keeps_more_than_six_homophone_variants_visible():
    variants = ["兰艺", "蓝衣", "兰意", "兰依", "蓝一", "蓝以", "兰逸"]
    segments = [
        Segment(start=i * 5, end=i * 5 + 5, text=f"{variant}和金子一起确认群里的安排。")
        for i, variant in enumerate(variants)
    ]

    report = find_term_consistency_candidates(segments)

    candidate = next(item for item in report["candidates"] if item.get("phonetic_key") == "lan-yi")
    assert set(variants).issubset(set(candidate["terms"]))


def test_first_mention_phonetic_consistency_uses_first_credible_spelling():
    segments = [
        Segment(start=0, end=5, text="今天李辉负责项目例会，王工一起参加。"),
        Segment(start=5, end=10, text="刚才李慧提到项目例会的风险，需要后续确认。"),
        Segment(start=10, end=15, text="后面理慧继续说明项目例会的预算问题。"),
        Segment(start=15, end=20, text="李慧说项目例会结束后再同步给产品。"),
    ]

    updated, stats = apply_first_mention_phonetic_consistency(segments)
    text = "\n".join(segment.text for segment in updated)

    assert stats["replacement_count"] == 3
    assert stats["groups"][0]["canonical"] == "李辉"
    assert text.count("李辉") == 4
    assert "李慧" not in text
    assert "理慧" not in text
    assert updated[1].original_text == segments[1].text


def test_first_mention_phonetic_consistency_keeps_explicit_distinct_people():
    segments = [
        Segment(start=0, end=5, text="今天李慧和李辉都参加项目例会。"),
        Segment(start=5, end=10, text="李慧负责预算，李辉负责合同。"),
        Segment(start=10, end=15, text="李慧和李辉后面继续同步。"),
    ]

    updated, stats = apply_first_mention_phonetic_consistency(segments)

    assert stats["replacement_count"] == 0
    assert stats["groups"] == []
    assert any(item["reason"] == "explicit_distinct_entity_pair" for item in stats["skipped"])
    assert [segment.text for segment in updated] == [segment.text for segment in segments]


def test_first_mention_phonetic_consistency_does_not_rewrite_sentence_fragments_as_names():
    segments = [
        Segment(start=0, end=5, text="这个其实可以完成平台部署。"),
        Segment(start=5, end=10, text="这种方式可以继续完成平台部署。"),
        Segment(start=10, end=15, text="同时可以安排科委完成平台验收。"),
        Segment(start=15, end=20, text="当时科委也要求完成平台验收。"),
        Segment(start=20, end=25, text="相当于做一个平台工具。"),
        Segment(start=25, end=30, text="相当于昨天已经讨论过平台工具。"),
        Segment(start=30, end=35, text="后来又相当于做了一次平台工具。"),
    ]
    original = [segment.text for segment in segments]

    updated, stats = apply_first_mention_phonetic_consistency(segments)

    assert [segment.text for segment in updated] == original
    assert stats["replacement_count"] == 0
    assert all(group.get("canonical") not in {"时可", "于昨"} for group in stats["groups"])


def test_first_mention_phonetic_consistency_uses_repeated_person_context_as_anchor():
    segments = [
        Segment(start=0, end=5, text="李慧负责接口评审。"),
        Segment(start=5, end=10, text="李慧说稍后继续同步。"),
        Segment(start=10, end=15, text="是理慧，接口评审刚才已经确认过。"),
    ]

    updated, stats = apply_first_mention_phonetic_consistency(segments)

    assert stats["replacement_count"] == 1
    assert "理慧" not in "\n".join(segment.text for segment in updated)


def test_real_name_anchor_does_not_absorb_embedded_sentence_fragments():
    segments = [
        Segment(start=0, end=5, text="时可老师负责平台验收。"),
        Segment(start=5, end=10, text="时可说下周继续推进。"),
        Segment(start=10, end=15, text="这种方式可以完成平台部署。"),
        Segment(start=15, end=20, text="这个其实可以继续完成平台部署。"),
        Segment(start=20, end=25, text="当时科委也参加了平台验收。"),
    ]
    original = [segment.text for segment in segments]

    updated, stats = apply_first_mention_phonetic_consistency(segments)

    assert [segment.text for segment in updated] == original
    assert stats["replacement_count"] == 0


def test_normalizer_keeps_first_mention_phonetic_variants_by_default():
    segments = [
        Segment(start=0, end=5, text="今天李辉负责项目例会，王工一起参加"),
        Segment(start=5, end=10, text="刚才李慧提到项目例会的风险"),
        Segment(start=10, end=15, text="后面理慧继续说明项目例会"),
    ]

    normalized, stats = normalize_segments(segments)
    text = "\n".join(segment.text for segment in normalized)
    consistency = stats["first_mention_phonetic_consistency"]

    assert consistency["enabled"] is False
    assert consistency["mode"] == "review_only"
    assert consistency["replacement_count"] == 0
    assert "李辉" in text
    assert "李慧" in text
    assert "理慧" in text


def test_normalizer_only_applies_first_mention_consistency_in_explicit_legacy_profile():
    segments = [
        Segment(start=0, end=5, text="今天李辉负责项目例会，王工一起参加"),
        Segment(start=5, end=10, text="刚才李慧提到项目例会的风险"),
        Segment(start=10, end=15, text="后面理慧继续说明项目例会"),
    ]

    normalized, stats = normalize_segments(segments, profile="legacy_general")
    text = "\n".join(segment.text for segment in normalized)
    consistency = stats["first_mention_phonetic_consistency"]

    assert consistency["enabled"] is True
    assert consistency["replacement_count"] == 2
    assert "李慧" not in text
    assert "理慧" not in text
