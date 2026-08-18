"""Conservative transcript term consistency discovery.

This module first looks for repeated short terms that have near spellings or
near pronunciations and appear in overlapping local contexts, then returns
review candidates for later QA/reporting.  A separate conservative helper can
also unify high-confidence same-pronunciation entity variants to the first
credible spelling already present in the transcript.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import replace
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Iterable

from .types import Segment

try:  # Packaged dependency; guarded so review reporting never breaks ASR.
    from pypinyin import lazy_pinyin as _lazy_pinyin
except Exception:  # pragma: no cover - defensive fallback for incomplete dev envs.
    _lazy_pinyin = None


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{2,}")
_ASCII_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_SPLIT_RE = re.compile(r"[\s，。！？；：、,.!?;:…()[\]（）“”‘’\"'`]+")
_MAX_CONTEXTS_PER_TERM = 12
_MAX_GROUP_CONTEXTS = 24
_MAX_VARIANTS_PER_CANDIDATE = 12

_FUNCTION_WORDS = {
    "一个",
    "一些",
    "一下",
    "这个",
    "那个",
    "这些",
    "那些",
    "因为",
    "所以",
    "但是",
    "然后",
    "如果",
    "就是",
    "也是",
    "不是",
    "没有",
    "我们",
    "你们",
    "他们",
    "大家",
    "今天",
    "明天",
    "现在",
    "这里",
    "那里",
    "可以",
    "可能",
    "觉得",
    "时候",
    "什么",
    "怎么",
    "这样",
    "一样",
    "还是",
    "或者",
    "比较",
    "需要",
    "问题",
    "事情",
    "东西",
    "进行",
    "来说",
    "的话",
}
_GENERIC_SHORT_TERMS = {
    "方式",
    "方案",
    "方面",
    "成本",
    "周期",
    "资源",
    "业务",
    "问题",
    "数据",
    "目标",
    "场景",
    "范围",
    "模式",
    "流程",
    "阶段",
    "系统",
    "平台",
    "项目",
}
_BOUNDARY_WORDS = _FUNCTION_WORDS | {
    "实际",
    "已经",
    "后来",
    "前面",
    "后面",
    "里面",
    "外面",
    "上面",
    "下面",
    "一下",
    "出来",
    "进去",
    "起来",
}
_FRAGMENT_PREFIX_CHARS = set("个写读查改做搞弄用换")
_DOMAIN_SUFFIXES = (
    "系统",
    "平台",
    "项目",
    "方案",
    "架构",
    "接口",
    "网络",
    "数据",
    "数据库",
    "团契",
    "教会",
    "牧师",
    "事工",
    "祷告",
    "婚姻",
)
_DOMAIN_CHARS = set("神耶稣督伯罕契祷牧会姊妹华为云数库接口架构系统项目方案网络婚姻")
_BAD_PREFIX = set("的一是在了着过和与及也就都很还再更最这那我你他她它")
_BAD_SUFFIX = set("的一是在了着过和与及也就都很还再吗呢吧啊呀")
_COMMON_SURNAMES = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹"
) | set("林蓝兰石东梁")
_COMPOUND_SURNAMES = ("欧阳", "司马", "上官", "诸葛", "东方", "夏侯", "皇甫", "尉迟", "公孙", "慕容", "司徒", "司空")
_NAME_LIKE_2_RE = re.compile(r"^[\u4e00-\u9fff]{2}$")
_NAME_LIKE_3_RE = re.compile(r"^[\u4e00-\u9fff]{3}$")
_CJK_ONLY_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")
_PAIR_CONTEXT_RE = re.compile(r"(?P<left>[\u3400-\u4dbf\u4e00-\u9fff]{2,6})(?P<connector>和|跟|与|及)(?P<right>[\u3400-\u4dbf\u4e00-\u9fff]{2,6})")
_PERSON_LEFT_CONTEXT_ENDINGS = (
    "请",
    "让",
    "叫",
    "找",
    "问",
    "跟",
    "和",
    "与",
    "及",
    "给",
    "向",
    "像",
    "由",
    "通过",
    "联系",
    "告诉",
    "听到",
    "提到",
)
_PERSON_RIGHT_CONTEXT_PREFIXES = (
    "老师",
    "同学",
    "同工",
    "姐",
    "姐妹",
    "姊妹",
    "弟兄",
    "牧师",
    "组长",
    "先生",
    "女士",
    "总",
    "和",
    "跟",
    "与",
    "及",
    "让",
    "说",
    "提到",
    "提了",
    "负责",
    "参加",
    "主持",
    "确认",
    "补充",
    "表达",
    "回复",
    "沟通",
    "联系",
    "打招呼",
    "继续",
    "后来",
    "曾经",
    "那边",
    "也在",
)
_NAME_LEFT_BOUNDARY_ENDINGS = _PERSON_LEFT_CONTEXT_ENDINGS + ("是", "把", "被", "为")
_NAME_RIGHT_BOUNDARY_PREFIXES = _PERSON_RIGHT_CONTEXT_PREFIXES + ("给", "有", "是", "在", "要", "会", "再")
_GENERIC_ENTITY_DRIFT_PARTNERS = {
    "男人",
    "女人",
    "人们",
    "别人",
    "对方",
    "大家",
    "我们",
    "你们",
    "他们",
    "这个",
    "那个",
}
_BAD_TERM_LEFT_CONTEXT_ENDINGS = ("非常", "特别", "常常", "很", "太", "更", "最")
_BAD_TERM_RIGHT_CONTEXT_PREFIXES = ("时候",)
@dataclass(frozen=True)
class _Occurrence:
    term: str
    index: int
    start: float
    end: float
    text: str
    left: str
    right: str
    context_terms: frozenset[str]


def find_term_consistency_candidates(
    segments: Iterable[Segment],
    *,
    min_total_occurrences: int = 4,
    min_variant_count: int = 2,
    min_occurrences_per_variant: int = 2,
) -> dict[str, Any]:
    """Find likely term/name consistency candidates without changing text.

    The detector is intentionally conservative.  A candidate group needs:
    - at least two distinct short variants,
    - enough total occurrences,
    - near spelling similarity,
    - and overlapping local context.
    """
    segment_list = list(segments)
    occurrences_by_term = _collect_occurrences(segment_list)
    repeated_terms = sorted(
        term
        for term, occurrences in occurrences_by_term.items()
        if len(occurrences) >= min_occurrences_per_variant
        and _can_form_high_value_group(term, occurrences)
    )
    phonetic_terms = _phonetic_entity_terms(occurrences_by_term)

    candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, tuple[str, ...]]] = set()
    for i, left in enumerate(repeated_terms):
        for right in repeated_terms[i + 1 :]:
            if len(left) != len(right):
                continue
            if not _terms_are_near(left, right):
                continue
            if not _has_context_overlap(occurrences_by_term[left], occurrences_by_term[right]):
                continue
            group_terms = sorted([left, right])
            if not _is_high_value_group(group_terms, occurrences_by_term):
                continue
            occurrences = [occ for term in group_terms for occ in occurrences_by_term[term]]
            if len(occurrences) < min_total_occurrences:
                continue
            confidence = _group_confidence(group_terms, occurrences_by_term)
            if confidence < 0.58:
                continue

            _append_candidate(
                candidates,
                seen_keys,
                group_terms,
                occurrences_by_term,
                confidence=confidence,
                kind="orthographic_term",
                reason="相近短词在重叠上下文中反复出现，建议人工确认是否为同一实体/术语。",
            )

    for key, group_terms in _build_phonetic_groups(phonetic_terms):
        group_terms = sorted(set(group_terms))
        if len(group_terms) < min_variant_count:
            continue
        if _looks_like_sliding_window_fragment(group_terms):
            continue
        occurrences = [occ for term in group_terms for occ in occurrences_by_term[term]]
        if len(occurrences) < max(3, min_total_occurrences - 1):
            continue
        if not _has_group_context_overlap(group_terms, occurrences_by_term) and not _has_repeated_entity_pair_context(group_terms, occurrences_by_term):
            continue
        if not _is_phonetic_entity_group(group_terms, occurrences_by_term):
            continue
        confidence = _phonetic_group_confidence(group_terms, occurrences_by_term)
        if confidence < 0.58:
            continue
        _append_candidate(
            candidates,
            seen_keys,
            group_terms,
            occurrences_by_term,
            confidence=confidence,
            kind="phonetic_entity",
            phonetic_key=key,
            action="review",
            reason="相同/近似读音的实体写法在上下文中反复出现，建议确认标准写法；系统不自动替换。",
            suggested_canonical=None,
        )

    for group in _find_entity_drift_groups(segment_list, occurrences_by_term):
        _append_candidate(
            candidates,
            seen_keys,
            group["terms"],
            occurrences_by_term,
            confidence=group["confidence"],
            kind="entity_drift",
            phonetic_key=group.get("phonetic_key"),
            action="review",
            reason=group["reason"],
            extra_occurrences=group.get("extra_occurrences") or [],
            suggested_canonical=None,
        )

    candidates.sort(
        key=lambda item: (
            _candidate_kind_rank(str(item.get("kind") or "")),
            -float(item["confidence"]),
            -int(item["total_count"]),
            item["terms"],
        )
    )
    for idx, candidate in enumerate(candidates, start=1):
        candidate["id"] = f"term-consistency-{idx}"

    return {
        "mode": "term_consistency",
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def apply_first_mention_phonetic_consistency(
    segments: Iterable[Segment],
    *,
    min_confidence: float = 0.74,
) -> tuple[list[Segment], dict[str, Any]]:
    """Unify same-pronunciation entity variants to the first credible spelling.

    This is intentionally different from a customer-specific profile.  It never
    chooses a hard-coded spelling such as "李会" or "兰艺"; it only looks at the
    spellings already produced in the current transcript.  For example, if the
    transcript first says "李辉" and later says "李慧"/"理慧" in matching entity
    contexts, later variants are rewritten to "李辉".
    """
    items = list(segments)
    stats: dict[str, Any] = {
        "mode": "first_mention_phonetic_consistency",
        "enabled": True,
        "candidate_count": 0,
        "applied_group_count": 0,
        "replacement_count": 0,
        "segments_changed": 0,
        "groups": [],
        "skipped": [],
    }
    if not items:
        return [], stats

    report = find_term_consistency_candidates(items)
    candidates = list(report.get("candidates") or [])
    stats["candidate_count"] = len(candidates)
    output = items
    locked_terms: set[str] = set()
    compact_original_texts = [_compact_context(seg.text or "") for seg in items]

    for candidate in candidates:
        if candidate.get("kind") != "phonetic_entity":
            continue
        confidence = float(candidate.get("confidence") or 0.0)
        if confidence < min_confidence:
            stats["skipped"].append({
                "id": candidate.get("id"),
                "reason": "confidence_below_threshold",
                "confidence": round(confidence, 3),
            })
            continue

        terms = [str(term) for term in (candidate.get("terms") or []) if isinstance(term, str) and term.strip()]
        terms = sorted(set(terms), key=lambda value: (-len(value), value))
        if len(terms) < 2:
            continue
        variant_counts = {
            str(item.get("text") or ""): int(item.get("count") or 0)
            for item in (candidate.get("variants") or [])
            if isinstance(item, dict)
        }
        person_context_counts = {
            str(item.get("text") or ""): int(item.get("person_context_count") or 0)
            for item in (candidate.get("variants") or [])
            if isinstance(item, dict)
        }
        name_boundary_counts = {
            str(item.get("text") or ""): int(item.get("name_boundary_count") or 0)
            for item in (candidate.get("variants") or [])
            if isinstance(item, dict)
        }
        name_like_terms = [term for term in terms if _is_name_like_term(term)]
        strong_name_terms = [
            term
            for term in name_like_terms
            if person_context_counts.get(term, 0) > 0
        ]
        strong_person_context_count = sum(person_context_counts.get(term, 0) for term in strong_name_terms)
        if not name_like_terms or strong_person_context_count < 2:
            stats["skipped"].append({
                "id": candidate.get("id"),
                "terms": terms,
                "reason": "insufficient_clear_person_name_variants",
            })
            continue
        # A common phrase can share the same pinyin as a person name, e.g.
        # "例会" and "李慧". Keep only clear names plus low-frequency noisy
        # spellings; repeated non-name terms remain review-only.
        terms = [
            term
            for term in terms
            if (
                term in strong_name_terms
                or name_boundary_counts.get(term, 0) > 0
            )
            and (term in name_like_terms or variant_counts.get(term, 0) <= 2)
        ]
        if len(terms) < 2:
            continue
        keys = {_phonetic_key(term) for term in terms}
        keys.discard(None)
        if len(keys) != 1:
            stats["skipped"].append({
                "id": candidate.get("id"),
                "terms": terms,
                "reason": "not_exact_same_phonetic_key",
            })
            continue
        if locked_terms.intersection(terms):
            stats["skipped"].append({
                "id": candidate.get("id"),
                "terms": terms,
                "reason": "overlaps_previous_group",
            })
            continue
        if _has_explicit_distinct_entity_pair(output, terms, compact_texts=compact_original_texts):
            stats["skipped"].append({
                "id": candidate.get("id"),
                "terms": terms,
                "reason": "explicit_distinct_entity_pair",
            })
            continue

        canonical = _first_credible_spelling(output, terms)
        if not canonical:
            stats["skipped"].append({
                "id": candidate.get("id"),
                "terms": terms,
                "reason": "no_credible_first_spelling",
            })
            continue

        replace_terms = [term for term in terms if term != canonical]
        if not replace_terms:
            continue
        output, replacement_count, changed_count = _replace_terms_in_segments(output, canonical, replace_terms)
        if replacement_count <= 0:
            continue

        locked_terms.update(terms)
        stats["applied_group_count"] += 1
        stats["replacement_count"] += replacement_count
        stats["segments_changed"] += changed_count
        stats["groups"].append({
            "id": candidate.get("id"),
            "phonetic_key": next(iter(keys)),
            "canonical": canonical,
            "terms": terms,
            "replaced_terms": replace_terms,
            "confidence": round(confidence, 3),
            "replacement_count": replacement_count,
            "segments_changed": changed_count,
            "reason": "同音实体按本转录首次可信写法统一",
        })

    return output, stats


def _candidate_kind_rank(kind: str) -> int:
    return {"phonetic_entity": 0, "entity_drift": 1, "orthographic_term": 2}.get(kind, 3)


def _append_candidate(
    candidates: list[dict[str, Any]],
    seen_keys: set[tuple[str, tuple[str, ...]]],
    group_terms: list[str],
    occurrences_by_term: dict[str, list[_Occurrence]],
    *,
    confidence: float,
    kind: str,
    reason: str,
    phonetic_key: str | None = None,
    action: str | None = None,
    extra_occurrences: list[_Occurrence] | None = None,
    suggested_canonical: str | None | object = ...,
) -> None:
    unique_terms = sorted(set(group_terms), key=lambda value: (-len(occurrences_by_term.get(value, [])), value))
    if len(unique_terms) < 2:
        return
    key = (kind, tuple(sorted(unique_terms)))
    if key in seen_keys:
        return
    seen_keys.add(key)
    variants = [
        {
            "text": term,
            "count": len(occurrences_by_term.get(term, [])),
            "person_context_count": sum(
                1
                for occurrence in occurrences_by_term.get(term, [])
                if _occurrence_has_strong_person_context(occurrence)
            ),
            "name_boundary_count": sum(
                1
                for occurrence in occurrences_by_term.get(term, [])
                if _occurrence_has_name_boundary(occurrence)
            ),
            "contexts": [_context_dict(occ) for occ in occurrences_by_term.get(term, [])[:_MAX_CONTEXTS_PER_TERM]],
        }
        for term in unique_terms[:_MAX_VARIANTS_PER_CANDIDATE]
    ]
    occurrences = [occ for term in unique_terms for occ in occurrences_by_term.get(term, [])]
    occurrences.extend(extra_occurrences or [])
    canonical = _suggest_canonical(variants) if suggested_canonical is ... else suggested_canonical
    item = {
        "id": f"term-consistency-{len(candidates) + 1}",
        "kind": kind,
        "action": action or ("maybe_unify" if confidence >= 0.72 else "review"),
        "confidence": round(float(confidence), 3),
        "terms": [item["text"] for item in variants],
        "suggested_canonical": canonical,
        "total_count": len(occurrences),
        "variants": variants,
        "contexts": [
            _context_dict(occ)
            for occ in sorted(occurrences, key=lambda item: (item.index, item.start))[:_MAX_GROUP_CONTEXTS]
        ],
        "reason": reason,
    }
    if phonetic_key:
        item["phonetic_key"] = phonetic_key
    candidates.append(item)


def _collect_occurrences(segments: list[Segment]) -> dict[str, list[_Occurrence]]:
    occurrences: dict[str, list[_Occurrence]] = {}
    for index, seg in enumerate(segments):
        text = seg.text or ""
        segment_terms = _segment_context_terms(text)
        for term, start, end in _extract_terms(text):
            left = _compact_context(text[max(0, start - 8) : start])
            right = _compact_context(text[end : end + 8])
            context_terms = frozenset(token for token in segment_terms if token != term)
            occurrence = _Occurrence(
                term=term,
                index=index,
                start=float(seg.start or 0),
                end=float(seg.end or 0),
                text=text,
                left=left,
                right=right,
                context_terms=context_terms,
            )
            if _is_low_value_occurrence(occurrence):
                continue
            occurrences.setdefault(term, []).append(occurrence)
    return {
        term: items
        for term, items in occurrences.items()
        if len(items) >= 1 and not _is_low_value_term(term)
    }


@lru_cache(maxsize=4096)
def _extract_terms(text: str) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for match in _CJK_RUN_RE.finditer(text):
        run = match.group(0)
        base = match.start()
        max_size = min(4, len(run))
        for size in range(2, max_size + 1):
            for offset in range(0, len(run) - size + 1):
                term = run[offset : offset + size]
                if _is_low_value_term(term):
                    continue
                out.append((term, base + offset, base + offset + size))
    for match in _ASCII_TERM_RE.finditer(text):
        term = match.group(0)
        if not _is_low_value_term(term):
            out.append((term, match.start(), match.end()))
    return out


@lru_cache(maxsize=4096)
def _segment_context_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for part in _SPLIT_RE.split(text or ""):
        if not part:
            continue
        if _CJK_RE.search(part):
            for size in (2, 3, 4):
                if len(part) < size:
                    continue
                for offset in range(0, len(part) - size + 1):
                    term = part[offset : offset + size]
                    if not _is_low_value_term(term):
                        terms.add(term)
        elif _ASCII_TERM_RE.fullmatch(part) and not _is_low_value_term(part):
            terms.add(part)
    return terms


@lru_cache(maxsize=65536)
def _is_low_value_term(term: str) -> bool:
    term = (term or "").strip()
    if len(term) < 2 or len(term) > 24:
        return True
    if term in _FUNCTION_WORDS or term in _GENERIC_SHORT_TERMS:
        return True
    if any(word in term for word in _BOUNDARY_WORDS):
        return True
    if len(term) >= 3 and term[0] in _FRAGMENT_PREFIX_CHARS:
        return True
    if _CJK_RE.fullmatch(term[0]) and term[0] in _BAD_PREFIX:
        return True
    if _CJK_RE.fullmatch(term[-1]) and term[-1] in _BAD_SUFFIX and not _allow_short_phonetic_name_suffix(term):
        return True
    if len(set(term)) == 1:
        return True
    return False


def _is_low_value_occurrence(occurrence: _Occurrence) -> bool:
    if occurrence.left.endswith("非") and occurrence.term.startswith("常"):
        return True
    if any(occurrence.left.endswith(prefix) for prefix in _BAD_TERM_LEFT_CONTEXT_ENDINGS):
        return True
    if any(occurrence.right.startswith(suffix) for suffix in _BAD_TERM_RIGHT_CONTEXT_PREFIXES):
        return True
    return False


def _allow_short_phonetic_name_suffix(term: str) -> bool:
    return bool(_NAME_LIKE_2_RE.fullmatch(term) and term[0] in _COMMON_SURNAMES and term[-1] == "一")


def _is_high_value_group(group_terms: list[str], occurrences_by_term: dict[str, list[_Occurrence]]) -> bool:
    if len(group_terms) < 2:
        return False
    if _looks_like_sliding_window_fragment(group_terms):
        return False
    if any(_is_low_value_term(term) for term in group_terms):
        return False
    if all(_is_name_like_term(term) for term in group_terms):
        return True
    if all(_is_domain_term(term) for term in group_terms):
        return True
    if any(_is_domain_term(term) for term in group_terms) and all(_is_domain_name_like(term) for term in group_terms):
        return True
    if all(_ASCII_TERM_RE.fullmatch(term) for term in group_terms):
        return True
    return False


def _can_form_high_value_group(term: str, occurrences: list[_Occurrence]) -> bool:
    if _ASCII_TERM_RE.fullmatch(term):
        return True
    if _is_domain_term(term) or _is_domain_name_like(term):
        return True
    if _is_name_like_term(term):
        return any(_occurrence_has_strong_person_context(occurrence) for occurrence in occurrences)
    return False


def _phonetic_entity_terms(occurrences_by_term: dict[str, list[_Occurrence]]) -> list[str]:
    strong_terms = {
        term
        for term, occurrences in occurrences_by_term.items()
        if _phonetic_key(term) and _is_entity_like_term_or_context(term, occurrences)
    }
    strong_name_keys: dict[str, set[str]] = {}
    person_contexts_by_key: dict[str, int] = {}
    for term in strong_terms:
        if not _is_name_like_term(term):
            continue
        key = _phonetic_key(term)
        if key:
            strong_name_keys.setdefault(key, set()).add(term)
            person_contexts_by_key[key] = person_contexts_by_key.get(key, 0) + sum(
                1
                for occurrence in occurrences_by_term.get(term, [])
                if _occurrence_has_strong_person_context(occurrence)
            )
    established_name_keys = {
        key
        for key, terms in strong_name_keys.items()
        if terms and person_contexts_by_key.get(key, 0) >= 2
    }
    if established_name_keys:
        strong_terms.update(
            term
            for term, occurrences in occurrences_by_term.items()
            if _phonetic_key(term) in established_name_keys
            and (
                _is_name_like_term(term)
                or (
                    len(occurrences) <= 2
                    and any(_occurrence_has_name_boundary(occurrence) for occurrence in occurrences)
                )
            )
        )
    return sorted(strong_terms)


def _build_phonetic_groups(terms: list[str]) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    terms_by_key: dict[str, list[str]] = {}
    for term in terms:
        key = _phonetic_key(term)
        if key:
            terms_by_key.setdefault(key, []).append(term)

    used_terms: set[str] = set()
    for key, same_key_terms in sorted(terms_by_key.items()):
        if len(same_key_terms) >= 2:
            groups.append((key, sorted(same_key_terms)))
            used_terms.update(same_key_terms)

    remaining = [term for term in terms if term not in used_terms]
    for i, left in enumerate(remaining):
        left_key = _phonetic_key(left)
        if not left_key:
            continue
        near_terms = [left]
        for right in remaining[i + 1 :]:
            right_key = _phonetic_key(right)
            if right_key and _phonetic_keys_are_near(left_key, right_key):
                near_terms.append(right)
        if len(near_terms) >= 2:
            groups.append((left_key, sorted(near_terms)))
    return groups


@lru_cache(maxsize=65536)
def _phonetic_key(term: str) -> str | None:
    if not term or not _CJK_ONLY_RE.fullmatch(term):
        return None
    if len(term) < 2 or len(term) > 4:
        return None
    if _lazy_pinyin is not None:
        try:
            syllables = [str(item).strip().lower() for item in _lazy_pinyin(term) if str(item).strip()]
        except Exception:
            syllables = []
        if len(syllables) == len(term) and all(_is_valid_pinyin_syllable(item) for item in syllables):
            return "-".join(syllables)

    return None


def _is_valid_pinyin_syllable(value: str) -> bool:
    return bool(value and re.fullmatch(r"[a-züv]+", value))


def _phonetic_keys_are_near(left: str, right: str) -> bool:
    if left == right:
        return True
    left_parts = left.split("-")
    right_parts = right.split("-")
    if len(left_parts) != len(right_parts):
        return False
    if not left_parts:
        return False
    equal_or_near = 0
    for left_part, right_part in zip(left_parts, right_parts):
        if left_part == right_part:
            equal_or_near += 1
            continue
        if SequenceMatcher(None, left_part, right_part).ratio() >= 0.67:
            equal_or_near += 1
    return equal_or_near == len(left_parts) and SequenceMatcher(None, left, right).ratio() >= 0.72


def _is_entity_like_term_or_context(term: str, occurrences: list[_Occurrence]) -> bool:
    if _is_domain_name_like(term):
        return True
    if _is_name_like_term(term):
        return any(_occurrence_has_strong_person_context(occurrence) for occurrence in occurrences)
    return False


def _occurrence_has_strong_person_context(occurrence: _Occurrence) -> bool:
    if any(occurrence.left.endswith(value) for value in _PERSON_LEFT_CONTEXT_ENDINGS):
        return True
    if any(occurrence.right.startswith(value) for value in _PERSON_RIGHT_CONTEXT_PREFIXES):
        return True
    return False


def _occurrence_has_name_boundary(occurrence: _Occurrence) -> bool:
    if not occurrence.left or not _CJK_RE.fullmatch(occurrence.left[-1]):
        return True
    if not occurrence.right or not _CJK_RE.fullmatch(occurrence.right[0]):
        return True
    if any(occurrence.left.endswith(value) for value in _NAME_LEFT_BOUNDARY_ENDINGS):
        return True
    if len(occurrence.left) >= 2 and _is_name_like_term(occurrence.left[-2:]):
        return True
    if len(occurrence.right) >= 2 and _is_name_like_term(occurrence.right[:2]):
        return True
    return any(occurrence.right.startswith(value) for value in _NAME_RIGHT_BOUNDARY_PREFIXES)


def _has_group_context_overlap(group_terms: list[str], occurrences_by_term: dict[str, list[_Occurrence]]) -> bool:
    for i, left in enumerate(group_terms):
        for right in group_terms[i + 1 :]:
            if _has_context_overlap(occurrences_by_term[left], occurrences_by_term[right]):
                return True
    return False


def _has_repeated_entity_pair_context(group_terms: list[str], occurrences_by_term: dict[str, list[_Occurrence]]) -> bool:
    partners: dict[str, set[str]] = {}
    for term in group_terms:
        for occurrence in occurrences_by_term.get(term, []):
            for left, right in _entity_pairs(occurrence.text):
                if term == left:
                    partners.setdefault(term, set()).add(right)
                elif term == right:
                    partners.setdefault(term, set()).add(left)
    for i, left in enumerate(group_terms):
        for right in group_terms[i + 1 :]:
            if partners.get(left, set()).intersection(partners.get(right, set())):
                return True
    return False


def _is_phonetic_entity_group(group_terms: list[str], occurrences_by_term: dict[str, list[_Occurrence]]) -> bool:
    if any(_is_low_value_term(term) for term in group_terms):
        return False
    entity_like_count = sum(1 for term in group_terms if _is_entity_like_term_or_context(term, occurrences_by_term.get(term, [])))
    person_context_count = sum(
        1
        for term in group_terms
        for occurrence in occurrences_by_term.get(term, [])
        if _is_name_like_term(term) and _occurrence_has_strong_person_context(occurrence)
    )
    if entity_like_count < 2 and person_context_count < 2:
        return False
    total = sum(len(occurrences_by_term.get(term, [])) for term in group_terms)
    if total < 3:
        return False
    return True


def _phonetic_group_confidence(group_terms: list[str], occurrences_by_term: dict[str, list[_Occurrence]]) -> float:
    context_bonus = 0.16 if _has_group_context_overlap(group_terms, occurrences_by_term) else 0.0
    pair_bonus = 0.12 if _has_repeated_entity_pair_context(group_terms, occurrences_by_term) else 0.0
    total = sum(len(occurrences_by_term.get(term, [])) for term in group_terms)
    repetition_bonus = min(total / 18, 0.14)
    same_key_bonus = 0.0
    keys = {_phonetic_key(term) for term in group_terms}
    keys.discard(None)
    if len(keys) == 1:
        same_key_bonus = 0.18
    return round(min(0.48 + context_bonus + pair_bonus + repetition_bonus + same_key_bonus, 0.94), 3)


def _entity_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for match in _PAIR_CONTEXT_RE.finditer(text or ""):
        left = _trim_pair_side(match.group("left"), side="left")
        right = _trim_pair_side(match.group("right"), side="right")
        if left and right and left != right:
            pairs.append((left, right))
    return pairs


def _trim_pair_side(value: str, *, side: str) -> str:
    value = re.sub(r"^[的地得之这个那个一些一个]+", "", value or "")
    value = re.sub(r"[的地得之这个那个一些一个]+$", "", value)
    value = re.sub(r"(老师|同学|同工|姊妹|姐妹|弟兄|牧师|组长)$", "", value)
    if len(value) < 2:
        return ""
    candidates: list[str] = []
    if side == "left":
        candidates.extend([value[-2:], value[-3:], value])
    else:
        candidates.extend([value[:2], value[:3], value])
    for candidate in candidates:
        if len(candidate) < 2:
            continue
        candidate = re.sub(r"^[是把被给到在从跟和与及]", "", candidate)
        candidate = re.sub(r"[是把被给到在从跟和与及]$", "", candidate)
        if len(candidate) >= 2 and not _is_low_value_term(candidate):
            return candidate
    return ""


def _find_entity_drift_groups(
    segments: list[Segment],
    occurrences_by_term: dict[str, list[_Occurrence]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    paired_entities_by_anchor = _paired_entities_by_anchor(segments, occurrences_by_term)
    if not paired_entities_by_anchor:
        return groups

    for index, seg in enumerate(segments):
        text = seg.text or ""
        for match in _PAIR_CONTEXT_RE.finditer(text):
            left = _trim_pair_side(match.group("left"), side="left")
            right = _trim_pair_side(match.group("right"), side="right")
            if not left or not right:
                continue
            for suspicious, anchor in ((left, right), (right, left)):
                if suspicious not in _GENERIC_ENTITY_DRIFT_PARTNERS:
                    continue
                related = _select_drift_related_terms(paired_entities_by_anchor.get(anchor, {}))
                if not related:
                    continue
                pseudo = _Occurrence(
                    term=suspicious,
                    index=index,
                    start=float(seg.start or 0),
                    end=float(seg.end or 0),
                    text=text,
                    left=_compact_context(text[max(0, match.start() - 8) : match.start()]),
                    right=_compact_context(text[match.end() : match.end() + 8]),
                    context_terms=frozenset({anchor}),
                )
                for term in sorted(related):
                    key = ("entity_drift", suspicious, term)
                    if key in seen:
                        continue
                    seen.add(key)
                    groups.append(
                        {
                            "terms": [term, suspicious],
                            "confidence": 0.72,
                            "phonetic_key": _phonetic_key(term),
                            "extra_occurrences": [pseudo],
                            "reason": f"“{suspicious}{match.group('connector')}{anchor}”像把人名/实体识别成普通词；建议人工核对该实体，不自动替换。",
                        }
                    )
    return groups


def _paired_entities_by_anchor(
    segments: list[Segment],
    occurrences_by_term: dict[str, list[_Occurrence]],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for seg in segments:
        for left, right in _entity_pairs(seg.text or ""):
            if left not in occurrences_by_term or right not in occurrences_by_term:
                continue
            left_items = occurrences_by_term[left]
            right_items = occurrences_by_term[right]
            left_entity = _phonetic_key(left) and _is_entity_like_term_or_context(left, left_items)
            right_entity = _phonetic_key(right) and _is_entity_like_term_or_context(right, right_items)
            if left_entity and right_entity:
                _add_anchor_pair(out, right, left)
                _add_anchor_pair(out, left, right)
            elif left_entity:
                _add_anchor_pair(out, right, left)
            elif right_entity:
                _add_anchor_pair(out, left, right)
    return out


def _add_anchor_pair(out: dict[str, dict[str, int]], anchor: str, entity: str) -> None:
    current = out.setdefault(anchor, {})
    current[entity] = current.get(entity, 0) + 1


def _select_drift_related_terms(counts: dict[str, int]) -> list[str]:
    if not counts:
        return []
    ranked = sorted(
        counts,
        key=lambda term: (
            -counts[term],
            -len(occurrences_key := _phonetic_key(term) or ""),
            occurrences_key,
            term,
        ),
    )
    best_count = counts[ranked[0]]
    selected = [term for term in ranked if counts[term] == best_count]
    if len(selected) > 2:
        selected = selected[:2]
    return selected


def _looks_like_sliding_window_fragment(group_terms: list[str]) -> bool:
    for i, left in enumerate(group_terms):
        for right in group_terms[i + 1 :]:
            if len(left) != len(right):
                continue
            if len(left) >= 3 and (left[1:] == right[:-1] or right[1:] == left[:-1]):
                return True
    return False


@lru_cache(maxsize=65536)
def _is_name_like_term(term: str) -> bool:
    if _NAME_LIKE_2_RE.fullmatch(term):
        return term[0] in _COMMON_SURNAMES and term not in _FUNCTION_WORDS
    if _NAME_LIKE_3_RE.fullmatch(term):
        if term[:2] in _BOUNDARY_WORDS or term[-2:] in _BOUNDARY_WORDS:
            return False
        if any(ch in term for ch in _DOMAIN_CHARS):
            return False
        return term[0] in _COMMON_SURNAMES or any(term.startswith(surname) for surname in _COMPOUND_SURNAMES)
    return False


@lru_cache(maxsize=65536)
def _is_domain_term(term: str) -> bool:
    if not _CJK_ONLY_RE.fullmatch(term):
        return False
    if len(term) < 3:
        return False
    if any(term.endswith(suffix) for suffix in _DOMAIN_SUFFIXES):
        return True
    return sum(1 for ch in term if ch in _DOMAIN_CHARS) >= 2


@lru_cache(maxsize=65536)
def _is_domain_name_like(term: str) -> bool:
    if not _CJK_ONLY_RE.fullmatch(term):
        return False
    if len(term) < 3 or len(term) > 5:
        return False
    if any(word in term for word in _BOUNDARY_WORDS):
        return False
    return any(ch in term for ch in _DOMAIN_CHARS)


def _terms_are_near(left: str, right: str) -> bool:
    if left == right:
        return False
    if _CJK_RE.fullmatch(left) and _CJK_RE.fullmatch(right):
        if len(left) == 2 and left[0] != right[0] and left[-1] != right[-1]:
            return False
    ratio = SequenceMatcher(None, left.lower(), right.lower()).ratio()
    if len(left) <= 2 and len(right) <= 2:
        return ratio >= 0.5
    return ratio >= 0.67


def _has_context_overlap(left_items: list[_Occurrence], right_items: list[_Occurrence]) -> bool:
    for left in left_items:
        for right in right_items:
            if left.index == right.index:
                return True
            if _side_context_overlap(left, right):
                return True
            if left.context_terms and right.context_terms and left.context_terms.intersection(right.context_terms):
                return True
    return False


def _side_context_overlap(left: _Occurrence, right: _Occurrence) -> bool:
    return bool(
        (left.left and right.left and (left.left.endswith(right.left[-2:]) or right.left.endswith(left.left[-2:])))
        or (left.right and right.right and (left.right.startswith(right.right[:2]) or right.right.startswith(left.right[:2])))
    )


def _group_confidence(group_terms: list[str], occurrences_by_term: dict[str, list[_Occurrence]]) -> float:
    pair_scores: list[float] = []
    for i, left in enumerate(group_terms):
        for right in group_terms[i + 1 :]:
            similarity = SequenceMatcher(None, left.lower(), right.lower()).ratio()
            context = 1.0 if _has_context_overlap(occurrences_by_term[left], occurrences_by_term[right]) else 0.0
            pair_scores.append((similarity * 0.7) + (context * 0.3))
    if not pair_scores:
        return 0.0
    total = sum(len(occurrences_by_term[term]) for term in group_terms)
    repetition_bonus = min(total / 10, 0.12)
    return round(min((sum(pair_scores) / len(pair_scores)) + repetition_bonus, 0.99), 3)


def _suggest_canonical(variants: list[dict[str, Any]]) -> str | None:
    if not variants:
        return None
    first_count = int(variants[0]["count"])
    if len(variants) > 1 and int(variants[1]["count"]) == first_count:
        return None
    return str(variants[0]["text"])


def _first_credible_spelling(segments: list[Segment], terms: list[str]) -> str | None:
    occurrences: list[tuple[int, int, str]] = []
    for index, seg in enumerate(segments):
        text = seg.text or ""
        for term in terms:
            for match in re.finditer(re.escape(term), text):
                occurrences.append((index, match.start(), term))
    if not occurrences:
        return None
    occurrences.sort(key=lambda item: (item[0], item[1], -len(item[2]), item[2]))

    # Prefer a real-looking person/domain spelling.  This avoids locking an
    # early noisy homophone such as "理慧" when "李慧/李辉" appears later.
    for _, _, term in occurrences:
        if _is_name_like_term(term) or _is_domain_name_like(term):
            return term
    return occurrences[0][2]


def _has_explicit_distinct_entity_pair(
    segments: list[Segment],
    terms: list[str],
    *,
    compact_texts: list[str] | None = None,
) -> bool:
    if len(terms) < 2:
        return False
    escaped = [re.escape(term) for term in terms]
    connector = r"(?:和|跟|与|及|、|，|,|/)"
    patterns = [
        re.compile(rf"{left}{connector}{right}|{right}{connector}{left}")
        for i, left in enumerate(escaped)
        for right in escaped[i + 1 :]
    ]
    texts = compact_texts if compact_texts is not None else [_compact_context(seg.text or "") for seg in segments]
    for compact in texts:
        if any(pattern.search(compact) for pattern in patterns):
            return True
    return False


def _replace_terms_in_segments(
    segments: list[Segment],
    canonical: str,
    replace_terms: list[str],
) -> tuple[list[Segment], int, int]:
    replace_terms = [term for term in sorted(set(replace_terms), key=lambda value: (-len(value), value)) if term]
    if not replace_terms:
        return segments, 0, 0
    pattern = re.compile("|".join(re.escape(term) for term in replace_terms))
    replacement_count = 0
    changed_count = 0
    out: list[Segment] = []

    for seg in segments:
        text = seg.text or ""
        local_count = 0

        def repl(match: re.Match[str]) -> str:
            nonlocal local_count
            local_count += 1
            return canonical

        new_text = pattern.sub(repl, text)
        if local_count:
            replacement_count += local_count
            changed_count += 1
            original_text = seg.original_text or text
            out.append(replace(seg, text=new_text, original_text=original_text if original_text != new_text else None))
        else:
            out.append(seg)
    return out, replacement_count, changed_count


def _context_dict(occurrence: _Occurrence) -> dict[str, Any]:
    return {
        "index": occurrence.index,
        "start": occurrence.start,
        "end": occurrence.end,
        "text": occurrence.text,
    }


@lru_cache(maxsize=32768)
def _compact_context(text: str) -> str:
    return re.sub(r"\s+", "", text or "")
