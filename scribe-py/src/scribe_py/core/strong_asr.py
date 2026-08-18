"""Conservative local multi-model review for Chinese ASR.

SenseVoice remains the timeline owner. A short-window SenseVoice re-decode may
propose a local text change, while Paraformer and Qwen3-ASR must independently
confirm the same change in nearby context. Only confirmed, bounded internal
substitutions, insertions, or deletions are applied; timestamps never change.
"""
from __future__ import annotations

import gc
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

from .types import Segment, TranscribeOptions


QWEN_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
_TEXT_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]")
_FILLER_TOKENS = {"啊", "呃", "嗯", "哦", "哎", "唉", "呀", "诶", "吧", "嘛", "呢", "哈", "啦", "咯", "呗"}
_PROTECTED_FUNCTION_TOKENS = {
    "他", "她", "它", "的", "地", "得", "都", "是", "有", "在", "了",
    "着", "过", "把", "被", "和", "与", "或", "就", "也", "还", "才",
    "又", "去", "来", "我", "你", "您", "我们", "你们", "他们", "她们",
    "它们", "咱们", "如果", "的话", "然后", "反正",
}
_PROTECTED_INDEL_TOKENS = _PROTECTED_FUNCTION_TOKENS | {
    "我",
    "你",
    "您",
    "我们",
    "你们",
    "他们",
    "她们",
    "它们",
    "咱们",
}
_PROTECTED_PRONOUN_TOKENS = {
    "我", "你", "您", "他", "她", "它",
    "我们", "你们", "他们", "她们", "它们", "咱们",
}
_ARABIC_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
_CHINESE_NUMBER_RE = re.compile(r"^[零〇一二三四五六七八九十百千万亿两]+$")
_ADJACENT_DUPLICATE_RE = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9])\1", re.IGNORECASE)
_ADDRESSED_ENTITY_PREFIXES = ("跟", "和", "找", "请", "让", "问", "由", "给", "向", "叫", "联系")
_ADDRESSED_ENTITY_PREFIX_RE = re.compile(
    rf"(?:{'|'.join(_ADDRESSED_ENTITY_PREFIXES)})[\u3400-\u4dbf\u4e00-\u9fff]{{1,3}}$"
)
_ADDRESSED_ENTITY_ACTIONS = (
    "对",
    "说",
    "讲",
    "问",
    "聊",
    "看",
    "确认",
    "沟通",
    "联系",
    "负责",
    "处理",
    "做",
)
_COMMON_SURNAME_CHARS = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费"
    "廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元顾孟平黄和穆"
    "萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴宋茅庞熊纪舒屈项祝董梁杜阮蓝闵"
    "席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯卢"
    "莫房裘缪干解应宗丁宣邓郁单杭洪包诸左石崔吉龚程邢裴陆荣翁荀羊甄家封"
    "芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁"
)
_MULTICHAR_CANDIDATE_QWEN_MIN = 0.54
_MULTICHAR_PRIMARY_QWEN_MIN = 0.48
_ANY_CHANGE_CANDIDATE_QWEN_MIN = 0.42
_ANY_CHANGE_PRIMARY_QWEN_MIN = 0.38
_ANY_CHANGE_DETECTOR_QWEN_MIN = 0.42
_PRIMARY_PARA_AUDIT_ONLY_BELOW = 0.45
_PRIMARY_PARA_REVIEW_BELOW = 0.985
_BOUNDARY_CONTEXT_MIN_CHARS = 2
_INDEL_BOUNDARY_CONTEXT_MIN_CHARS = 4
_AUTO_REVIEW_NOISE_REASON_MARKERS = ("信噪比", "背景噪声", "噪声底")
_AUTO_REVIEW_MAX_ALIGNMENT_SIMILARITY = 0.45
_AUTO_REVIEW_EXTREME_SNR_DB = 3.0
_AUTO_REVIEW_CLIPPED_SNR_DB = 5.0
_AUTO_REVIEW_EXTREME_NOISE_FLOOR_DBFS = -40.0
_AUTO_REVIEW_CLIPPING_PEAK_DBFS = -0.2
_AUTO_REVIEW_EXTREME_MAX_ALIGNMENT_SIMILARITY = 0.88
_AUTO_REVIEW_SEVERE_SNR_DB = 1.5
_AUTO_REVIEW_SEVERE_NOISE_FLOOR_DBFS = -25.0
_PARAFORMER_FULL_AUDIO_MAX_SECONDS = 90.0
_PARAFORMER_TILE_SECONDS = 60.0
_PARAFORMER_TILE_PADDING_SECONDS = 0.75
_INDEPENDENT_FULL_COVERAGE_MAX_SECONDS = 1200.0
_REDECODE_LEFT_CONTEXT_SECONDS = 2.5
_REDECODE_RIGHT_CONTEXT_SECONDS = 5.0
_DEFAULT_MAX_INDEL_CHARS = 4
_CJK_ONLY_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")
_MEASUREMENT_UNITS = (
    "千赫兹", "兆赫兹", "吉赫兹", "千字节", "兆字节", "吉字节", "太字节",
    "摄氏度", "毫秒", "微秒", "纳秒", "分钟", "小时", "毫米", "厘米",
    "分米", "公里", "千米", "毫升", "千克", "公斤", "毫克", "千瓦",
    "毫伏", "毫安", "赫兹", "比特", "字节", "秒", "天", "周", "米",
    "升", "克", "伏", "安", "瓦", "度",
)
_MEASUREMENT_UNIT_PATTERN = "|".join(
    re.escape(unit) for unit in sorted(_MEASUREMENT_UNITS, key=len, reverse=True)
)
_MEASUREMENT_RE = re.compile(
    rf"(?:\d+(?:\.\d+)?|[零〇一二三四五六七八九十百千万亿两点]+)"
    rf"(?P<unit>{_MEASUREMENT_UNIT_PATTERN})"
)
_LATIN_TERM_RE = re.compile(r"[a-z][a-z0-9]{4,23}")
_RAW_LATIN_TERM_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]{4,23})(?![A-Za-z0-9])"
)
_RAW_ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,8})(?![A-Za-z0-9])")
_CJK_TEXT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_CJK_TOKEN_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")
_LEXICAL_FRAGMENT_MIN_CHARS = 6
_LEXICAL_FRAGMENT_MIN_TOKENS = 3
_LEXICAL_FRAGMENT_MIN_RATIO = 0.08
_INLINE_LOWER_LATIN_RE = re.compile(r"(?<![A-Za-z0-9])[a-z][a-z0-9]{2,23}(?![A-Za-z0-9])")
_REPEATED_CJK_WORD_RE = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fff]{2,4})\1")


@dataclass(frozen=True)
class ReviewWindow:
    start: float
    end: float
    segment_indexes: tuple[int, ...]


@dataclass(frozen=True)
class _RewriteCandidate:
    operation: str
    source: str
    target: str
    raw_start: int
    raw_end: int
    left_context: str
    right_context: str


def decide_auto_high_noise_review(
    *,
    quality_mode: str,
    backend: str,
    model_id: str,
    audio_quality: dict[str, Any] | None,
    transcription_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess whether standard SenseVoice output should be reviewed manually.

    The standard customer path is deliberately transcript-preserving: acoustic
    risk can recommend an explicit high-quality review, but it can never start
    another ASR model or rewrite text by itself. This assessment never inspects
    a recording name, transcript phrase, or normalizer profile.
    """
    quality = dict(audio_quality or {})
    pipeline = dict(transcription_stats or {})
    risk_level = str(quality.get("risk_level") or "unknown").strip().lower()
    risk_reasons = [str(item) for item in (quality.get("risk_reasons") or [])]
    try:
        estimated_snr_db = float(quality["estimated_snr_db"])
    except (KeyError, TypeError, ValueError):
        estimated_snr_db = None
    try:
        noise_floor_dbfs = float(quality["noise_floor_dbfs"])
    except (KeyError, TypeError, ValueError):
        noise_floor_dbfs = None
    try:
        true_peak_dbfs = float(quality["true_peak_dbfs"])
    except (KeyError, TypeError, ValueError):
        true_peak_dbfs = None
    try:
        alignment_similarity = float(pipeline["equal_char_ratio"])
    except (KeyError, TypeError, ValueError):
        alignment_similarity = None
    timing_reliable_raw = pipeline.get("timing_reliable")
    timing_reliable = timing_reliable_raw if isinstance(timing_reliable_raw, bool) else None
    alignment_reason = str(pipeline.get("timing_alignment_reason") or "")
    paraformer_preflight = pipeline.get("paraformer_preflight") or {}
    if not isinstance(paraformer_preflight, dict):
        paraformer_preflight = {}
    non_speech_suppression = pipeline.get("non_speech_suppression") or {}
    if not isinstance(non_speech_suppression, dict):
        non_speech_suppression = {}

    decision: dict[str, Any] = {
        "eligible": False,
        "recommended": False,
        "auto_run_enabled": False,
        "auto_run_reason": "",
        "policy": "sensevoice_high_noise_advisory_v8",
        "strategy": "none",
        "reason": "",
        "quality_mode": str(quality_mode or "standard").strip().lower(),
        "backend": str(backend or ""),
        "model_id": str(model_id or ""),
        "audio_risk_level": risk_level,
        "estimated_snr_db": estimated_snr_db,
        "noise_floor_dbfs": noise_floor_dbfs,
        "true_peak_dbfs": true_peak_dbfs,
        "timing_reliable": timing_reliable,
        "alignment_similarity": alignment_similarity,
        "alignment_reason": alignment_reason,
        "noise_evidence": [],
        "extreme_acoustic_evidence": [],
        "severe_acoustic_evidence": [],
        "decode_disagreement_evidence": [],
        "lexical_disagreement_evidence": [],
        "structural_disagreement_evidence": [],
        "paraformer_preflight_reason": str(paraformer_preflight.get("reason") or ""),
    }
    if decision["quality_mode"] != "standard":
        decision["reason"] = "quality_mode_not_standard"
        return decision
    backend_key = decision["backend"].strip().lower()
    model_key = decision["model_id"].strip().lower()
    if backend_key != "sensevoice" or "sensevoice" not in model_key:
        decision["reason"] = "not_sensevoice_primary"
        return decision
    if risk_level != "high":
        decision["reason"] = "audio_risk_not_high"
        return decision

    noise_evidence: list[str] = []
    if estimated_snr_db is not None and estimated_snr_db < 10.0:
        noise_evidence.append("estimated_snr_below_10db")
    for reason in risk_reasons:
        if any(marker in reason for marker in _AUTO_REVIEW_NOISE_REASON_MARKERS):
            noise_evidence.append(reason)
    decision["noise_evidence"] = list(dict.fromkeys(noise_evidence))
    if not decision["noise_evidence"]:
        decision["reason"] = "high_risk_without_noise_evidence"
        return decision

    decode_evidence: list[str] = []
    if (
        timing_reliable is False
        and alignment_similarity is not None
        and alignment_similarity < _AUTO_REVIEW_MAX_ALIGNMENT_SIMILARITY
    ):
        decode_evidence.extend([
            "timing_alignment_unreliable",
            f"alignment_similarity_below_{_AUTO_REVIEW_MAX_ALIGNMENT_SIMILARITY:.2f}",
        ])
        if alignment_reason:
            decode_evidence.append(alignment_reason)
    decision["decode_disagreement_evidence"] = decode_evidence

    extreme_acoustic_evidence: list[str] = []
    if estimated_snr_db is not None:
        if (
            estimated_snr_db <= _AUTO_REVIEW_EXTREME_SNR_DB
            and noise_floor_dbfs is not None
            and noise_floor_dbfs > _AUTO_REVIEW_EXTREME_NOISE_FLOOR_DBFS
        ):
            extreme_acoustic_evidence.extend([
                f"estimated_snr_at_or_below_{_AUTO_REVIEW_EXTREME_SNR_DB:.0f}db",
                f"noise_floor_above_{_AUTO_REVIEW_EXTREME_NOISE_FLOOR_DBFS:.0f}dbfs",
            ])
        if (
            estimated_snr_db <= _AUTO_REVIEW_CLIPPED_SNR_DB
            and true_peak_dbfs is not None
            and true_peak_dbfs >= _AUTO_REVIEW_CLIPPING_PEAK_DBFS
        ):
            if not extreme_acoustic_evidence:
                extreme_acoustic_evidence.append(
                    f"estimated_snr_at_or_below_{_AUTO_REVIEW_CLIPPED_SNR_DB:.0f}db"
                )
            extreme_acoustic_evidence.append(
                f"true_peak_at_or_above_{_AUTO_REVIEW_CLIPPING_PEAK_DBFS:.1f}dbfs"
            )
    decision["extreme_acoustic_evidence"] = extreme_acoustic_evidence

    severe_acoustic_evidence: list[str] = []
    if (
        estimated_snr_db is not None
        and estimated_snr_db <= _AUTO_REVIEW_SEVERE_SNR_DB
        and noise_floor_dbfs is not None
        and noise_floor_dbfs > _AUTO_REVIEW_SEVERE_NOISE_FLOOR_DBFS
    ):
        severe_acoustic_evidence.extend([
            f"estimated_snr_at_or_below_{_AUTO_REVIEW_SEVERE_SNR_DB:.1f}db",
            f"noise_floor_above_{_AUTO_REVIEW_SEVERE_NOISE_FLOOR_DBFS:.0f}dbfs",
        ])
        if true_peak_dbfs is not None and true_peak_dbfs >= _AUTO_REVIEW_CLIPPING_PEAK_DBFS:
            severe_acoustic_evidence.append(
                f"true_peak_at_or_above_{_AUTO_REVIEW_CLIPPING_PEAK_DBFS:.1f}dbfs"
            )
    decision["severe_acoustic_evidence"] = severe_acoustic_evidence

    structural_disagreement_evidence: list[str] = []
    if int(non_speech_suppression.get("suppressed_segments") or 0) > 0:
        structural_disagreement_evidence.append("vad_unsupported_text_suppressed")
    decision["structural_disagreement_evidence"] = structural_disagreement_evidence

    lexical_disagreement_evidence: list[str] = []
    if (
        extreme_acoustic_evidence
        and alignment_similarity is not None
        and alignment_similarity < _AUTO_REVIEW_EXTREME_MAX_ALIGNMENT_SIMILARITY
    ):
        lexical_disagreement_evidence.append(
            "alignment_similarity_below_"
            f"{_AUTO_REVIEW_EXTREME_MAX_ALIGNMENT_SIMILARITY:.2f}"
        )
    decision["lexical_disagreement_evidence"] = lexical_disagreement_evidence
    qwen_extreme_risk = bool(extreme_acoustic_evidence and lexical_disagreement_evidence)
    if not decode_evidence and not qwen_extreme_risk:
        if extreme_acoustic_evidence:
            long_recording = (
                decision["paraformer_preflight_reason"] == "recording_too_long"
            )
            if (
                long_recording
                or severe_acoustic_evidence
                or structural_disagreement_evidence
            ):
                # Same-model timing agreement cannot validate lexical content
                # under severe noise. Keep the likely manual-review strategy in
                # the report, but do not start a secondary decode in standard
                # mode: that would make default text and runtime non-repeatable.
                decision["recommended"] = True
                decision["strategy"] = (
                    "sparse_independent_consensus"
                    if long_recording
                    else "bounded_independent_consensus"
                )
                decision["reason"] = (
                    "high_noise_long_recording_sparse_independent_review"
                    if long_recording
                    else (
                        "high_noise_severe_acoustic_bounded_independent_review"
                        if severe_acoustic_evidence
                        else "high_noise_structural_bounded_independent_review"
                    )
                )
                decision["auto_run_reason"] = "standard_mode_advisory_only"
                return decision
            # Same-model alignment agreement cannot prove that noisy lexical
            # content is correct. Surface the risk without paying for, or
            # trusting, an automatic rewrite when no independent decode is
            # available.
            decision["recommended"] = True
            decision["strategy"] = "independent_local_review_required"
            decision["reason"] = "high_noise_independent_review_required"
            decision["auto_run_reason"] = "independent_decode_evidence_unavailable"
            return decision
        decision["reason"] = "high_noise_without_severe_decode_disagreement"
        return decision

    decision["recommended"] = True
    if qwen_extreme_risk:
        decision["strategy"] = "qwen_lexical_frozen_timeline"
        eligible_reason = "high_noise_extreme_acoustic_and_lexical_risk"
    else:
        decision["strategy"] = "local_strong_asr_consensus"
        eligible_reason = "high_noise_severe_decode_disagreement"
    decision["auto_run_reason"] = "standard_mode_advisory_only"
    decision["reason"] = eligible_reason
    return decision


def _normalized_stream(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    raw_positions: list[int] = []
    for index, char in enumerate(text or ""):
        if _TEXT_CHAR_RE.fullmatch(char):
            chars.append(char.lower())
            raw_positions.append(index)
    return "".join(chars), raw_positions


def normalized_text(text: str) -> str:
    return _normalized_stream(text)[0]


def text_similarity(left: str, right: str) -> float:
    left_norm = normalized_text(left)
    right_norm = normalized_text(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio()


def _is_number_rendering_change(source: str, target: str) -> bool:
    """Reject numeric typography changes such as ``10`` -> ``十``.

    This review stage is for acoustic lexical errors.  Whether numbers render
    as Arabic digits or Chinese numerals is a presentation convention and must
    not create automatic customer-facing edits or inflate a CER comparison.
    """
    return bool(
        (_ARABIC_NUMBER_RE.fullmatch(source) and _CHINESE_NUMBER_RE.fullmatch(target))
        or (_CHINESE_NUMBER_RE.fullmatch(source) and _ARABIC_NUMBER_RE.fullmatch(target))
    )


def _is_addressed_entity_context(primary: str, start: int, end: int) -> bool:
    """Protect an addressed name or term from an unverified spelling guess."""
    before = primary[max(0, start - 6) : start]
    after = primary[end : end + 4]
    if (
        before
        and _CJK_ONLY_RE.fullmatch(before[-1:])
        and any(after.startswith(prefix) for prefix in ("你知道", "你看", "你说", "你觉得"))
    ):
        return True
    has_prefix = bool(_ADDRESSED_ENTITY_PREFIX_RE.search(before)) or any(
        before.endswith(prefix) for prefix in _ADDRESSED_ENTITY_PREFIXES
    )
    return has_prefix and any(
        after.startswith(action) for action in _ADDRESSED_ENTITY_ACTIONS
    )


def timeline_fingerprint(segments: Iterable[Segment]) -> str:
    values: list[str] = []
    for segment in segments:
        values.append(f"s:{float(segment.start):.6f}:{float(segment.end):.6f}")
        for cue in segment.sync_cues or []:
            values.append(
                f"c:{float(cue.get('start', 0.0)):.6f}:{float(cue.get('end', 0.0)):.6f}"
            )
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()


def build_review_windows(
    segments: Iterable[Segment],
    *,
    max_seconds: float = 20.0,
    max_gap_seconds: float = 1.5,
) -> list[ReviewWindow]:
    indexed = [
        (index, segment)
        for index, segment in enumerate(segments)
        if (segment.text or "").strip() and segment.end > segment.start
    ]
    windows: list[ReviewWindow] = []
    current: list[tuple[int, Segment]] = []

    def flush() -> None:
        if not current:
            return
        windows.append(
            ReviewWindow(
                start=max(0.0, float(current[0][1].start) - 0.12),
                end=float(current[-1][1].end) + 0.12,
                segment_indexes=tuple(index for index, _segment in current),
            )
        )
        current.clear()

    for index, segment in indexed:
        if not current:
            current.append((index, segment))
            continue
        first = current[0][1]
        previous = current[-1][1]
        would_span = float(segment.end) - float(first.start)
        gap = float(segment.start) - float(previous.end)
        if would_span > max_seconds or gap > max_gap_seconds:
            flush()
        current.append((index, segment))
    flush()
    return windows


@lru_cache(maxsize=4096)
def _generic_lexical_fragmentation(text: str) -> dict[str, Any]:
    """Score dictionary-unknown Chinese fragments without proposing a rewrite.

    The score only decides which expensive acoustic review windows to inspect.
    Unknown names and domain terms are therefore harmless: text still changes
    only after independent SenseVoice, Paraformer, and Qwen audio agreement.
    """
    compact = "".join(_CJK_TEXT_RE.findall(text or ""))
    if len(compact) < 12:
        return {
            "score": 0.0,
            "unknown_chars": 0,
            "unknown_tokens": 0,
            "unknown_ratio": 0.0,
            "available": True,
        }
    try:
        import jieba

        jieba.initialize()
        frequencies = jieba.dt.FREQ
        tokens = list(jieba.cut(compact, HMM=True))
    except Exception as exc:
        return {
            "score": 0.0,
            "unknown_chars": 0,
            "unknown_tokens": 0,
            "unknown_ratio": 0.0,
            "available": False,
            "reason": f"jieba_unavailable:{type(exc).__name__}",
        }

    unknown = [
        token
        for token in tokens
        if 2 <= len(token) <= 8
        and _CJK_TOKEN_RE.fullmatch(token)
        and int(frequencies.get(token) or 0) <= 0
    ]
    unknown_chars = sum(len(token) for token in unknown)
    ratio = unknown_chars / len(compact)
    qualifies = (
        unknown_chars >= _LEXICAL_FRAGMENT_MIN_CHARS
        and len(unknown) >= _LEXICAL_FRAGMENT_MIN_TOKENS
        and ratio >= _LEXICAL_FRAGMENT_MIN_RATIO
    )
    score = 0.0
    if qualifies:
        score = min(
            6.0,
            2.0 + ratio * 12.0 + max(0, len(unknown) - _LEXICAL_FRAGMENT_MIN_TOKENS) * 0.35,
        )
    return {
        "score": round(score, 4),
        "unknown_chars": unknown_chars,
        "unknown_tokens": len(unknown),
        "unknown_ratio": round(ratio, 4),
        "available": True,
    }


def select_sparse_review_windows(
    segments: Iterable[Segment],
    *,
    max_windows: int = 12,
) -> tuple[list[ReviewWindow], dict[str, Any]]:
    """Select structural risks plus bounded long-form coverage probes."""
    primary = list(segments)
    all_windows = build_review_windows(primary)
    scored: list[tuple[int, float, ReviewWindow, list[str]]] = []
    for window in all_windows:
        score = 0.0
        reasons: list[str] = []
        window_text: list[str] = []
        for index in window.segment_indexes:
            segment = primary[index]
            text = segment.text or ""
            window_text.append(text)
            chars = len(normalized_text(text))
            duration = max(float(segment.end) - float(segment.start), 0.001)
            density = chars / duration
            if duration >= 8.0 and chars <= 6:
                score += 6.0
                reasons.append("long_low_text_density")
            elif duration >= 15.0 and density < 1.2:
                score += 5.0
                reasons.append("low_text_density")
            if duration >= 6.0 and density >= 7.5:
                score += 5.0
                reasons.append("implausibly_high_text_density")
            if re.search(r"([\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9])\1{3,}", text, re.IGNORECASE):
                score += 3.0
                reasons.append("repeated_character_run")
            cues = list(segment.sync_cues or [])
            if cues and any(
                float(cue.get("end", segment.start)) <= float(cue.get("start", segment.start))
                for cue in cues
            ):
                score += 6.0
                reasons.append("invalid_sync_cue_duration")
        lexical = _generic_lexical_fragmentation("".join(window_text))
        if float(lexical.get("score") or 0.0) > 0.0:
            score += float(lexical["score"])
            reasons.append("generic_lexical_fragmentation")
        if score > 0:
            scored.append((len(scored), score, window, list(dict.fromkeys(reasons))))

    limit = max(1, int(max_windows))
    long_form = len(all_windows) >= max(24, limit * 4)
    coverage_budget = min(6, max(3, limit // 2)) if long_form and limit >= 4 else 0
    risk_budget = max(0, limit - coverage_budget)
    ranked = sorted(scored, key=lambda item: (-item[1], item[2].start, item[0]))[
        :risk_budget if coverage_budget else limit
    ]
    selected = [item[2] for item in ranked]
    selected_keys = {(window.start, window.end) for window in selected}
    coverage_keys: set[tuple[float, float]] = set()

    remaining_budget = limit - len(selected)
    if long_form and remaining_budget > 0 and all_windows:
        first_center = (all_windows[0].start + all_windows[0].end) / 2.0
        last_center = (all_windows[-1].start + all_windows[-1].end) / 2.0
        available = [
            window
            for window in all_windows
            if (window.start, window.end) not in selected_keys
        ]
        edge_candidates = [*all_windows[:2], *all_windows[-1:]]
        for chosen in edge_candidates:
            if remaining_budget <= 0 or chosen not in available:
                continue
            key = (chosen.start, chosen.end)
            selected.append(chosen)
            selected_keys.add(key)
            coverage_keys.add(key)
            available.remove(chosen)
            remaining_budget -= 1

        target_fractions = [
            (index + 1) / (remaining_budget + 1)
            for index in range(remaining_budget)
        ]
        for fraction in target_fractions:
            if not available:
                break
            target = first_center + ((last_center - first_center) * fraction)
            chosen = min(
                available,
                key=lambda window: (
                    abs(((window.start + window.end) / 2.0) - target),
                    window.start,
                ),
            )
            key = (chosen.start, chosen.end)
            selected.append(chosen)
            selected_keys.add(key)
            coverage_keys.add(key)
            available.remove(chosen)

    selected = sorted(selected, key=lambda window: window.start)
    selected_keys = {(window.start, window.end) for window in selected}
    scored_by_key = {
        (window.start, window.end): (score, reasons, window)
        for _order, score, window, reasons in scored
    }
    windows_by_key = {
        (window.start, window.end): window
        for window in all_windows
    }
    diagnostic_keys = set(scored_by_key) | coverage_keys
    diagnostics = []
    for key in sorted(diagnostic_keys):
        score, reasons, window = scored_by_key.get(
            key,
            (0.0, [], windows_by_key[key]),
        )
        diagnostic_reasons = list(reasons)
        if key in coverage_keys:
            diagnostic_reasons.append("stratified_coverage_probe")
            if window in all_windows[:2]:
                diagnostic_reasons.append("recording_start_probe")
            if window == all_windows[-1]:
                diagnostic_reasons.append("recording_end_probe")
        diagnostics.append({
            "start": round(window.start, 3),
            "end": round(window.end, 3),
            "segment_indexes": list(window.segment_indexes),
            "score": score,
            "reasons": list(dict.fromkeys(diagnostic_reasons)),
            "selected": key in selected_keys,
            "selection_kind": "coverage_probe" if key in coverage_keys else "structural_risk",
        })
    return selected, {
        "mode": "generic_lexical_structural_and_coverage_sparse_review_v3",
        "all_window_count": len(all_windows),
        "candidate_window_count": len(scored),
        "selected_window_count": len(selected),
        "structural_window_count": len(selected) - len(coverage_keys),
        "coverage_probe_count": len(coverage_keys),
        "max_windows": limit,
        "uses_recording_name": False,
        "uses_fixed_transcript_phrases": False,
        "diagnostics": diagnostics,
    }


def select_standard_selective_review_windows(
    segments: Iterable[Segment],
    *,
    max_windows: int = 3,
) -> tuple[list[ReviewWindow], dict[str, Any]]:
    """Pick only generic, high-signal windows for the standard ASR path.

    This deliberately has no coverage probes. Standard mode must not pay for a
    second pass just because the audio is noisy; it only asks independent models
    to inspect text that contains a generic structural anomaly. The candidates
    are never rewritten by this selector, so unknown names and terminology stay
    safe until the regular multi-model consensus gate confirms a change.
    """
    primary = list(segments)
    all_windows = build_review_windows(primary)
    scored: list[tuple[float, ReviewWindow, list[str]]] = []
    for window in all_windows:
        text = _window_text(primary, window.segment_indexes)
        score = 0.0
        reasons: list[str] = []
        has_explicit_lexical_anomaly = False

        inline_latin = _INLINE_LOWER_LATIN_RE.findall(text)
        if inline_latin:
            score += 5.0
            reasons.append("inline_lowercase_latin_fragment")
            has_explicit_lexical_anomaly = True
        if _REPEATED_CJK_WORD_RE.search(text):
            score += 2.0
            reasons.append("repeated_cjk_word")
            has_explicit_lexical_anomaly = True

        lexical = _generic_lexical_fragmentation(text)
        lexical_score = float(lexical.get("score") or 0.0)
        if lexical_score > 0.0:
            score += lexical_score
            reasons.append("generic_lexical_fragmentation")
            has_explicit_lexical_anomaly = True

        for index in window.segment_indexes:
            segment = primary[index]
            segment_text = segment.text or ""
            duration = max(float(segment.end) - float(segment.start), 0.001)
            chars = len(normalized_text(segment_text))
            density = chars / duration
            if duration >= 8.0 and chars <= 6:
                score += 4.0
                reasons.append("long_low_text_density")
            elif duration >= 6.0 and density >= 7.5:
                score += 4.0
                reasons.append("implausibly_high_text_density")
            if re.search(r"([\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9])\1{3,}", segment_text, re.IGNORECASE):
                score += 3.0
                reasons.append("repeated_character_run")
                has_explicit_lexical_anomaly = True

        # Sparse speech alone is not evidence of a recognition error. It may
        # simply be a long pause or a slow speaker, so only use density as a
        # ranking boost after a concrete lexical anomaly was observed.
        if score > 0.0 and has_explicit_lexical_anomaly:
            scored.append((score, window, list(dict.fromkeys(reasons))))

    limit = max(1, int(max_windows))
    ranked = sorted(scored, key=lambda item: (-item[0], item[1].start))[:limit]
    selected = [item[1] for item in ranked]
    diagnostics = [
        {
            "start": round(window.start, 3),
            "end": round(window.end, 3),
            "segment_indexes": list(window.segment_indexes),
            "score": round(score, 4),
            "reasons": reasons,
            "selected": True,
            "selection_kind": "generic_structural_risk",
        }
        for score, window, reasons in ranked
    ]
    return selected, {
        "mode": "generic_standard_selective_review_v1",
        "all_window_count": len(all_windows),
        "candidate_window_count": len(scored),
        "selected_window_count": len(selected),
        "max_windows": limit,
        "uses_recording_name": False,
        "uses_fixed_transcript_phrases": False,
        "uses_coverage_probes": False,
        "diagnostics": diagnostics,
    }


def select_bounded_independent_review_windows(
    segments: Iterable[Segment],
    *,
    max_windows: int = 12,
    full_coverage_max_seconds: float = _INDEPENDENT_FULL_COVERAGE_MAX_SECONDS,
) -> tuple[list[ReviewWindow], dict[str, Any]]:
    """Cover a short recording fully, or sample a long one structurally.

    The selector is intentionally independent of filenames, transcript phrases,
    and domain vocabulary. High-noise recordings up to 20 minutes get complete
    independent coverage; longer recordings reuse the bounded sparse selector.
    """
    primary = list(segments)
    all_windows = build_review_windows(primary)
    limit = max(1, int(max_windows))
    duration_s = max((float(segment.end) for segment in primary), default=0.0)
    cover_fully = duration_s <= max(0.0, float(full_coverage_max_seconds))
    if len(all_windows) > limit and not cover_fully:
        selected, stats = select_sparse_review_windows(primary, max_windows=limit)
        selected_keys = {(window.start, window.end) for window in selected}
        remaining = limit - len(selected)
        added_coverage: list[ReviewWindow] = []
        available = [
            window
            for window in all_windows
            if (window.start, window.end) not in selected_keys
        ]
        first_center = (all_windows[0].start + all_windows[0].end) / 2.0
        last_center = (all_windows[-1].start + all_windows[-1].end) / 2.0
        for index in range(remaining):
            if not available:
                break
            fraction = 0.5 if remaining == 1 else index / (remaining - 1)
            target = first_center + ((last_center - first_center) * fraction)
            chosen = min(
                available,
                key=lambda window: (
                    abs(((window.start + window.end) / 2.0) - target),
                    window.start,
                ),
            )
            selected.append(chosen)
            added_coverage.append(chosen)
            available.remove(chosen)
        selected = sorted(selected, key=lambda window: window.start)
        diagnostics = list(stats.get("diagnostics") or [])
        diagnostic_keys = {
            (float(item.get("start", 0.0)), float(item.get("end", 0.0)))
            for item in diagnostics
        }
        for window in added_coverage:
            key = (round(window.start, 3), round(window.end, 3))
            if key in diagnostic_keys:
                for item in diagnostics:
                    if (
                        float(item.get("start", 0.0)),
                        float(item.get("end", 0.0)),
                    ) == key:
                        item["selected"] = True
                        item["selection_kind"] = "coverage_probe"
                        reasons = list(item.get("reasons") or [])
                        if "bounded_stratified_coverage" not in reasons:
                            reasons.append("bounded_stratified_coverage")
                        item["reasons"] = reasons
                        break
                continue
            diagnostics.append({
                "start": key[0],
                "end": key[1],
                "segment_indexes": list(window.segment_indexes),
                "score": 0.0,
                "reasons": ["bounded_stratified_coverage"],
                "selected": True,
                "selection_kind": "coverage_probe",
            })
        coverage_count = int(stats.get("coverage_probe_count") or 0) + len(added_coverage)
        return selected, {
            **stats,
            "mode": "generic_bounded_independent_review_v1",
            "coverage_mode": "structural_and_stratified",
            "duration_s": round(duration_s, 3),
            "full_coverage_max_seconds": float(full_coverage_max_seconds),
            "selected_window_count": len(selected),
            "structural_window_count": max(0, len(selected) - coverage_count),
            "coverage_probe_count": coverage_count,
            "diagnostics": sorted(
                diagnostics,
                key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
            ),
        }

    diagnostics = [
        {
            "start": round(window.start, 3),
            "end": round(window.end, 3),
            "segment_indexes": list(window.segment_indexes),
            "score": 0.0,
            "reasons": ["bounded_full_coverage"],
            "selected": True,
            "selection_kind": "full_coverage",
        }
        for window in all_windows
    ]
    return all_windows, {
        "mode": "generic_bounded_independent_review_v1",
        "coverage_mode": "full",
        "duration_s": round(duration_s, 3),
        "full_coverage_max_seconds": float(full_coverage_max_seconds),
        "all_window_count": len(all_windows),
        "candidate_window_count": len(all_windows),
        "selected_window_count": len(all_windows),
        "structural_window_count": 0,
        "coverage_probe_count": len(all_windows),
        "max_windows": limit,
        "uses_recording_name": False,
        "uses_fixed_transcript_phrases": False,
        "diagnostics": diagnostics,
    }


def _confirmed_in_context(
    qwen_text: str,
    *,
    left_context: str,
    replacement: str,
    right_context: str,
    min_context_chars: int = _BOUNDARY_CONTEXT_MIN_CHARS,
) -> bool:
    qwen_norm = normalized_text(qwen_text)
    if not qwen_norm:
        return False
    max_context = min(5, len(left_context), len(right_context))
    for size in range(max_context, min_context_chars - 1, -1):
        pattern = left_context[-size:] + replacement + right_context[:size]
        if pattern in qwen_norm:
            return True
    return False


def _contains_number_token(text: str) -> bool:
    return bool(re.search(r"[0-9零〇一二三四五六七八九十百千万亿两]", text))


@lru_cache(maxsize=4096)
def _pinyin_stream(text: str) -> tuple[str, ...]:
    if not text or not _CJK_ONLY_RE.fullmatch(text):
        return ()
    try:
        from pypinyin import lazy_pinyin

        syllables = [str(item).strip().lower() for item in lazy_pinyin(text)]
    except Exception:
        return ()
    if len(syllables) != len(text) or not all(syllables):
        return ()
    return tuple(syllables)


@lru_cache(maxsize=4096)
def _pinyin_tone_stream(text: str) -> tuple[str, ...]:
    if not text or not _CJK_ONLY_RE.fullmatch(text):
        return ()
    try:
        from pypinyin import Style, lazy_pinyin

        syllables = [
            str(item).strip().lower()
            for item in lazy_pinyin(
                text,
                style=Style.TONE3,
                neutral_tone_with_five=True,
            )
        ]
    except Exception:
        return ()
    if len(syllables) != len(text) or not all(syllables):
        return ()
    return tuple(syllables)


def _is_exact_homophone_change(source: str, target: str) -> bool:
    if source == target or len(source) != len(target) or not (2 <= len(source) <= 6):
        return False
    source_pinyin = _pinyin_stream(source)
    target_pinyin = _pinyin_stream(target)
    return bool(source_pinyin and source_pinyin == target_pinyin)


def _is_probable_surname_rewrite(
    primary_norm: str,
    start: int,
    source: str,
    target: str,
) -> bool:
    if len(source) != 1 or len(target) != 1 or target not in _COMMON_SURNAME_CHARS:
        return False
    if _pinyin_tone_stream(source) != _pinyin_tone_stream(target):
        return False
    before = primary_norm[max(0, start - 1):start]
    after = primary_norm[start + 1:start + 3]
    return before in {"我", "你", "他", "她", "叫", "是"} and len(after) == 2


def _collect_rewrite_candidates(
    primary_text: str,
    primary_norm: str,
    primary_positions: list[int],
    candidate_norm: str,
    *,
    max_span_chars: int,
    max_indel_chars: int,
) -> list[_RewriteCandidate]:
    candidates: list[_RewriteCandidate] = []
    matcher = SequenceMatcher(None, primary_norm, candidate_norm, autojunk=False)
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        if operation not in {"replace", "insert", "delete"}:
            continue
        source = primary_norm[i1:i2]
        target = candidate_norm[j1:j2]
        if source == target or (not source and not target):
            continue
        if operation == "replace" and (not source or not target):
            continue

        changed_text = source + target
        if any(char in _FILLER_TOKENS for char in changed_text):
            continue
        protected_tokens = (
            _PROTECTED_FUNCTION_TOKENS
            if operation == "replace"
            else _PROTECTED_INDEL_TOKENS
        )
        if source in protected_tokens or target in protected_tokens:
            continue
        if _ADJACENT_DUPLICATE_RE.search(source) or _ADJACENT_DUPLICATE_RE.search(target):
            continue
        if operation == "replace":
            if any(
                token in source or token in target
                for token in _PROTECTED_PRONOUN_TOKENS
            ):
                continue
            if _is_number_rendering_change(source, target):
                continue
            if re.search(r"\d", source + target) or _CHINESE_NUMBER_RE.fullmatch(source):
                continue
            protected_characters = set("".join(_PROTECTED_FUNCTION_TOKENS))
            if changed_text and all(char in protected_characters for char in changed_text):
                continue
            # Blind ASR cannot safely choose between exact Chinese homophones.
            # Preserve the timeline owner's spelling for names and domain terms.
            if _is_exact_homophone_change(source, target):
                continue
            if _is_probable_surname_rewrite(primary_norm, i1, source, target):
                continue
            if len(source) > max_span_chars or len(target) > max_span_chars:
                continue
        else:
            changed_span = source or target
            if len(changed_span) > max_indel_chars or _contains_number_token(changed_span):
                continue

        left = primary_norm[max(0, i1 - 5) : i1]
        right = primary_norm[i2 : i2 + 5]
        if (
            len(left) < _BOUNDARY_CONTEXT_MIN_CHARS
            or len(right) < _BOUNDARY_CONTEXT_MIN_CHARS
        ):
            continue
        if operation != "replace":
            source_form = left[-1:] + source + right[:1]
            target_form = left[-1:] + target + right[:1]
            if _ADJACENT_DUPLICATE_RE.search(source_form) or _ADJACENT_DUPLICATE_RE.search(
                target_form
            ):
                continue
        if _is_addressed_entity_context(primary_norm, i1, i2):
            continue

        if i1 == i2:
            if i1 >= len(primary_positions):
                continue
            raw_start = primary_positions[i1]
            raw_end = raw_start
            previous_raw_end = primary_positions[i1 - 1] + 1
            separator = primary_text[previous_raw_end:raw_start]
            if separator.strip() or "\n" in separator:
                continue
        else:
            raw_start = primary_positions[i1]
            raw_end = primary_positions[i2 - 1] + 1
        if "\n" in primary_text[raw_start:raw_end]:
            continue

        candidates.append(
            _RewriteCandidate(
                operation=operation,
                source=source,
                target=target,
                raw_start=raw_start,
                raw_end=raw_end,
                left_context=left,
                right_context=right,
            )
        )
    return candidates


def _detector_confirms_rewrite_candidate(
    primary_text: str,
    primary_redecode_text: str,
    detector_text: str,
    *,
    max_span_chars: int = 10,
    max_indel_chars: int = _DEFAULT_MAX_INDEL_CHARS,
) -> bool:
    primary_norm, primary_positions = _normalized_stream(primary_text)
    candidate_norm = normalized_text(primary_redecode_text)
    if not primary_norm or not candidate_norm or not normalized_text(detector_text):
        return False
    for candidate in _collect_rewrite_candidates(
        primary_text,
        primary_norm,
        primary_positions,
        candidate_norm,
        max_span_chars=max_span_chars,
        max_indel_chars=max_indel_chars,
    ):
        min_context_chars = (
            _INDEL_BOUNDARY_CONTEXT_MIN_CHARS
            if candidate.operation != "replace"
            else _BOUNDARY_CONTEXT_MIN_CHARS
        )
        if not _confirmed_in_context(
            detector_text,
            left_context=candidate.left_context,
            replacement=candidate.target,
            right_context=candidate.right_context,
            min_context_chars=min_context_chars,
        ):
            continue
        if _confirmed_in_context(
            detector_text,
            left_context=candidate.left_context,
            replacement=candidate.source,
            right_context=candidate.right_context,
            min_context_chars=min_context_chars,
        ):
            continue
        return True
    return False


def _violates_local_consistency_guard(
    primary_text: str,
    primary_norm: str,
    candidate: _RewriteCandidate,
) -> bool:
    """Reject isolated rewrites that make one window internally inconsistent."""
    source = candidate.source
    target = candidate.target
    if not source or not target or source == target:
        return False
    if (
        len(source) == 1
        and primary_norm.count(source) >= 2
        and target not in primary_norm
    ):
        return True
    if len(target) == 1 and (
        candidate.left_context[-1:] == target
        or candidate.right_context[:1] == target
    ):
        if not _replacement_forms_known_token(primary_text, candidate):
            return True
    if len(source) == 1 and len(target) == 1:
        shared_boundary = max(
            (
                size
                for size in range(
                    2,
                    min(len(candidate.left_context), len(candidate.right_context), 5) + 1,
                )
                if candidate.left_context[-size:] == candidate.right_context[:size]
            ),
            default=0,
        )
        if shared_boundary:
            return True
    if len(target) > 1:
        before = normalized_text(
            primary_text[max(0, candidate.raw_start - 20):candidate.raw_start]
        )
        after = normalized_text(
            primary_text[candidate.raw_end:candidate.raw_end + 20]
        )
        if target in before or target in after:
            return True
    return False


def _replacement_forms_known_token(
    primary_text: str,
    candidate: _RewriteCandidate,
) -> bool:
    """Allow repeated characters only when they complete a known local word."""
    if (
        candidate.operation != "replace"
        or len(candidate.source) != 1
        or len(candidate.target) != 1
        or not _CJK_ONLY_RE.fullmatch(candidate.source + candidate.target)
    ):
        return False
    corrected = (
        primary_text[:candidate.raw_start]
        + candidate.target
        + primary_text[candidate.raw_end:]
    )
    compact = normalized_text(corrected)
    changed_offset = len(normalized_text(primary_text[:candidate.raw_start]))
    try:
        import jieba

        jieba.initialize()
        frequencies = jieba.dt.FREQ
        tokens = list(jieba.cut(compact, HMM=False))
    except Exception:
        return False

    offset = 0
    for token in tokens:
        end = offset + len(token)
        contains_change = offset <= changed_offset < end
        if (
            contains_change
            and 2 <= len(token) <= 8
            and _CJK_TOKEN_RE.fullmatch(token)
            and int(frequencies.get(token) or 0) >= 50
        ):
            return True
        offset = end
    return False


def _preserve_acronym_case(
    primary_text: str,
    raw_start: int,
    raw_end: int,
    target: str,
) -> str:
    """Keep replacements inside an all-uppercase ASCII token uppercase."""
    if not target or not re.fullmatch(r"[a-z]+", target):
        return target
    token_start = raw_start
    token_end = raw_end
    while token_start > 0 and primary_text[token_start - 1].isalpha() and primary_text[token_start - 1].isascii():
        token_start -= 1
    while token_end < len(primary_text) and primary_text[token_end].isalpha() and primary_text[token_end].isascii():
        token_end += 1
    token = primary_text[token_start:token_end]
    if len(token) >= 2 and re.fullmatch(r"[A-Z]+", token):
        return target.upper()
    return target


def _ascii_rewrite_is_consistent(
    primary_text: str,
    candidate: _RewriteCandidate,
    target: str,
    *,
    global_context_text: str = "",
) -> bool:
    """Reject partial Latin/acronym rewrites that conflict with the transcript.

    Independent ASR models often agree on the same extra acronym letter.  A
    local one-character rewrite can then turn one invalid acronym into another
    while a shorter spelling is already dominant across the recording.  Long
    Latin completions remain eligible when the primary already contains a
    meaningful fragment; unrelated Latin substitutions stay review-only.
    """
    changed = candidate.source + candidate.target
    if not re.search(r"[A-Za-z]", changed):
        return True
    if (
        candidate.operation == "insert"
        and candidate.raw_start > 0
        and primary_text[candidate.raw_start - 1].isspace()
    ):
        return False

    token_start = candidate.raw_start
    token_end = candidate.raw_end
    while (
        token_start > 0
        and primary_text[token_start - 1].isascii()
        and primary_text[token_start - 1].isalnum()
    ):
        token_start -= 1
    while (
        token_end < len(primary_text)
        and primary_text[token_end].isascii()
        and primary_text[token_end].isalnum()
    ):
        token_end += 1
    source_token = primary_text[token_start:token_end]
    relative_start = candidate.raw_start - token_start
    relative_end = candidate.raw_end - token_start
    target_token = (
        source_token[:relative_start] + target + source_token[relative_end:]
    )
    if not source_token or not target_token:
        return False

    source_is_acronym = bool(re.fullmatch(r"[A-Z]{2,8}", source_token))
    target_is_acronym = bool(re.fullmatch(r"[A-Z]{2,8}", target_token))
    if source_is_acronym or target_is_acronym:
        if not (source_is_acronym and target_is_acronym):
            return False
        global_acronyms = _acronym_counts(global_context_text)
        for canonical, count in global_acronyms.items():
            if count < 3 or len(canonical) + 1 != len(target_token):
                continue
            if any(
                target_token[:index] + target_token[index + 1:] == canonical
                for index in range(len(target_token))
            ):
                return False
        return True

    if not (
        re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,23}", source_token)
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9]{4,23}", target_token)
    ):
        return False
    source_lower = source_token.lower()
    target_lower = target_token.lower()
    return len(source_lower) >= 2 and source_lower in target_lower


def consensus_rewrite(
    primary_text: str,
    primary_redecode_text: str,
    qwen_text: str,
    *,
    detector_text: str,
    qwen_hallucination_risk: bool = False,
    max_span_chars: int = 10,
    max_indel_chars: int = _DEFAULT_MAX_INDEL_CHARS,
    candidate_qwen_similarity: float | None = None,
    primary_qwen_similarity: float | None = None,
    detector_qwen_similarity: float | None = None,
    global_context_text: str = "",
    allow_deletions: bool = True,
    allow_insertions: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply a bounded re-decode change only when two other models confirm it."""
    if qwen_hallucination_risk:
        return primary_text, []
    primary_norm, primary_positions = _normalized_stream(primary_text)
    candidate_norm = normalized_text(primary_redecode_text)
    detector_norm = normalized_text(detector_text)
    qwen_norm = normalized_text(qwen_text)
    if not primary_norm or not candidate_norm or not detector_norm or not qwen_norm:
        return primary_text, []

    primary_len = len(primary_norm)
    # The short re-decode deliberately includes acoustic context on both sides
    # to survive imperfect long-recording timestamps.
    if not (0.45 <= len(candidate_norm) / primary_len <= 2.25):
        return primary_text, []
    # Qwen receives extra acoustic context around the target window, so its
    # transcript may legitimately be longer. It is only used as contextual
    # evidence; replacements still come from the bounded primary re-decode.
    if not (0.20 <= len(qwen_norm) / primary_len <= 8.00):
        return primary_text, []

    candidate_qwen_similarity = (
        text_similarity(primary_redecode_text, qwen_text)
        if candidate_qwen_similarity is None
        else float(candidate_qwen_similarity)
    )
    primary_qwen_similarity = (
        text_similarity(primary_text, qwen_text)
        if primary_qwen_similarity is None
        else float(primary_qwen_similarity)
    )
    detector_qwen_similarity = (
        text_similarity(detector_text, qwen_text)
        if detector_qwen_similarity is None
        else float(detector_qwen_similarity)
    )
    if (
        candidate_qwen_similarity < _ANY_CHANGE_CANDIDATE_QWEN_MIN
        or primary_qwen_similarity < _ANY_CHANGE_PRIMARY_QWEN_MIN
        or detector_qwen_similarity < _ANY_CHANGE_DETECTOR_QWEN_MIN
    ):
        return primary_text, []

    replacements: list[tuple[int, int, str, dict[str, Any]]] = []
    for candidate in _collect_rewrite_candidates(
        primary_text,
        primary_norm,
        primary_positions,
        candidate_norm,
        max_span_chars=max_span_chars,
        max_indel_chars=max_indel_chars,
    ):
        source = candidate.source
        target = candidate.target
        if candidate.operation == "delete" and not allow_deletions:
            continue
        if candidate.operation == "insert" and not allow_insertions:
            continue
        if _violates_local_consistency_guard(
            primary_text,
            primary_norm,
            candidate,
        ):
            continue
        min_context_chars = (
            _INDEL_BOUNDARY_CONTEXT_MIN_CHARS
            if candidate.operation != "replace"
            else _BOUNDARY_CONTEXT_MIN_CHARS
        )
        if max(len(source), len(target)) > 1 and (
            (
                candidate_qwen_similarity is not None
                and candidate_qwen_similarity < _MULTICHAR_CANDIDATE_QWEN_MIN
            )
            or (
                primary_qwen_similarity is not None
                and primary_qwen_similarity < _MULTICHAR_PRIMARY_QWEN_MIN
            )
        ):
            continue
        if not _confirmed_in_context(
            qwen_text,
            left_context=candidate.left_context,
            replacement=target,
            right_context=candidate.right_context,
            min_context_chars=min_context_chars,
        ):
            continue
        if not _confirmed_in_context(
            detector_text,
            left_context=candidate.left_context,
            replacement=target,
            right_context=candidate.right_context,
            min_context_chars=min_context_chars,
        ):
            continue
        # Do not apply a change if Qwen also explicitly confirms the original
        # reading in the same context.
        if _confirmed_in_context(
            qwen_text,
            left_context=candidate.left_context,
            replacement=source,
            right_context=candidate.right_context,
            min_context_chars=min_context_chars,
        ):
            continue
        if _confirmed_in_context(
            detector_text,
            left_context=candidate.left_context,
            replacement=source,
            right_context=candidate.right_context,
            min_context_chars=min_context_chars,
        ):
            continue
        display_target = _preserve_acronym_case(
            primary_text,
            candidate.raw_start,
            candidate.raw_end,
            target,
        )
        if not _ascii_rewrite_is_consistent(
            primary_text,
            candidate,
            display_target,
            global_context_text=global_context_text,
        ):
            continue
        detail: dict[str, Any] = {
            "from": primary_text[candidate.raw_start:candidate.raw_end],
            "to": display_target,
            "normalized_from": source,
            "normalized_to": target,
            "left_context": candidate.left_context,
            "right_context": candidate.right_context,
            "evidence": "sensevoice_redecode_paraformer_qwen_context_agreement",
        }
        if candidate.operation != "replace":
            detail["operation"] = candidate.operation
            detail["evidence"] = (
                "sensevoice_redecode_paraformer_qwen_internal_indel_agreement"
            )
        replacements.append(
            (
                candidate.raw_start,
                candidate.raw_end,
                display_target,
                detail,
            )
        )

    corrected = primary_text
    changes: list[dict[str, Any]] = []
    for raw_start, raw_end, target, detail in reversed(replacements):
        corrected = corrected[:raw_start] + target + corrected[raw_end:]
        changes.append(detail)
    changes.reverse()
    return corrected, changes


def independent_consensus_rewrite(
    primary_text: str,
    paraformer_text: str,
    qwen_text: str,
    *,
    qwen_hallucination_risk: bool = False,
    max_span_chars: int = 10,
    max_indel_chars: int = _DEFAULT_MAX_INDEL_CHARS,
    global_context_text: str = "",
    allow_insertions: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply only bounded Paraformer changes independently confirmed by Qwen."""
    corrected, changes = consensus_rewrite(
        primary_text,
        paraformer_text,
        qwen_text,
        detector_text=paraformer_text,
        qwen_hallucination_risk=qwen_hallucination_risk,
        max_span_chars=max_span_chars,
        max_indel_chars=max_indel_chars,
        global_context_text=global_context_text,
        # Two independent models can share the same omission. Deletions are
        # therefore reserved for the three-way path above, where a fresh
        # primary re-decode also confirms that the source audio omits the text.
        allow_deletions=False,
        allow_insertions=allow_insertions,
    )
    for change in changes:
        operation = str(change.get("operation") or "replace")
        change["evidence"] = (
            "paraformer_qwen_independent_internal_indel_agreement"
            if operation != "replace"
            else "paraformer_qwen_independent_context_agreement"
        )
    return corrected, changes


def aligned_independent_consensus_rewrite(
    primary_text: str,
    paraformer_text: str,
    qwen_text: str,
    *,
    qwen_hallucination_risk: bool = False,
    max_span_chars: int = 6,
    global_context_text: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Accept exact independent replacements aligned to the same source span."""
    if qwen_hallucination_risk:
        return primary_text, []
    primary_norm, primary_positions = _normalized_stream(primary_text)
    paraformer_norm = normalized_text(paraformer_text)
    qwen_norm = normalized_text(qwen_text)
    if not primary_norm or not paraformer_norm or not qwen_norm:
        return primary_text, []
    if text_similarity(primary_text, qwen_text) < _ANY_CHANGE_PRIMARY_QWEN_MIN:
        return primary_text, []
    if text_similarity(paraformer_text, qwen_text) < _ANY_CHANGE_DETECTOR_QWEN_MIN:
        return primary_text, []

    def replacements_for(candidate_text: str) -> list[_RewriteCandidate]:
        return [
            item
            for item in _collect_rewrite_candidates(
                primary_text,
                primary_norm,
                primary_positions,
                candidate_text,
                max_span_chars=max_span_chars,
                max_indel_chars=0,
            )
            if item.operation == "replace"
        ]

    paraformer_candidates = replacements_for(paraformer_norm)
    qwen_candidates = replacements_for(qwen_norm)
    paraformer_by_span = {
        (candidate.raw_start, candidate.raw_end, candidate.target): candidate
        for candidate in paraformer_candidates
    }
    replacements: list[tuple[int, int, str, dict[str, Any]]] = []
    global_context_norm = normalized_text(global_context_text)
    for candidate in qwen_candidates:
        match = paraformer_by_span.get(
            (candidate.raw_start, candidate.raw_end, candidate.target)
        )
        if match is None:
            continue
        if _violates_local_consistency_guard(
            primary_text,
            primary_norm,
            candidate,
        ):
            continue
        if len(candidate.source) == 1 and global_context_norm:
            source_anchors = {
                candidate.left_context[-1:] + candidate.source,
                candidate.source + candidate.right_context[:1],
            }
            if any(
                len(anchor) >= 2 and global_context_norm.count(anchor) >= 2
                for anchor in source_anchors
            ):
                continue
        display_target = _preserve_acronym_case(
            primary_text,
            candidate.raw_start,
            candidate.raw_end,
            candidate.target,
        )
        if not _ascii_rewrite_is_consistent(
            primary_text,
            candidate,
            display_target,
            global_context_text=global_context_text,
        ):
            continue
        replacements.append((
            candidate.raw_start,
            candidate.raw_end,
            display_target,
            {
                "from": primary_text[candidate.raw_start:candidate.raw_end],
                "to": display_target,
                "normalized_from": candidate.source,
                "normalized_to": candidate.target,
                "left_context": candidate.left_context,
                "right_context": candidate.right_context,
                "evidence": "paraformer_qwen_exact_aligned_consensus",
            },
        ))

    corrected = primary_text
    changes: list[dict[str, Any]] = []
    for raw_start, raw_end, target, detail in sorted(
        replacements,
        key=lambda item: (item[0], item[1]),
        reverse=True,
    ):
        corrected = corrected[:raw_start] + target + corrected[raw_end:]
        changes.append(detail)
    changes.reverse()
    return corrected, changes


def atomic_aligned_independent_consensus_rewrite(
    primary_text: str,
    paraformer_text: str,
    qwen_text: str,
    *,
    qwen_hallucination_risk: bool = False,
    max_span_chars: int = 6,
    global_context_text: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Recover a shared character edit from differently sized replacements.

    Independent decoders can agree on one character while disagreeing on a
    neighboring insertion.  SequenceMatcher then emits, for example, a
    one-character replacement for one decoder and a two-character replacement
    for the other.  Keep only the position and target character confirmed by
    both decoders, and require the result to form a known local token.
    """
    if qwen_hallucination_risk:
        return primary_text, []
    primary_norm, primary_positions = _normalized_stream(primary_text)
    paraformer_norm = normalized_text(paraformer_text)
    qwen_norm = normalized_text(qwen_text)
    if not primary_norm or not paraformer_norm or not qwen_norm:
        return primary_text, []
    if text_similarity(primary_text, qwen_text) < _ANY_CHANGE_PRIMARY_QWEN_MIN:
        return primary_text, []
    if text_similarity(paraformer_text, qwen_text) < _ANY_CHANGE_DETECTOR_QWEN_MIN:
        return primary_text, []

    def replacements_for(candidate_text: str) -> list[_RewriteCandidate]:
        return [
            item
            for item in _collect_rewrite_candidates(
                primary_text,
                primary_norm,
                primary_positions,
                candidate_text,
                max_span_chars=max_span_chars,
                max_indel_chars=0,
            )
            if item.operation == "replace"
            and len(item.source) == len(item.target)
        ]

    def atomic_edits(
        candidates: list[_RewriteCandidate],
    ) -> dict[tuple[int, int, str], tuple[_RewriteCandidate, _RewriteCandidate]]:
        edits: dict[
            tuple[int, int, str],
            tuple[_RewriteCandidate, _RewriteCandidate],
        ] = {}
        for parent in candidates:
            raw_source = primary_text[parent.raw_start:parent.raw_end]
            if raw_source != parent.source or len(raw_source) != len(parent.target):
                continue
            for offset, (source_char, target_char) in enumerate(
                zip(parent.source, parent.target)
            ):
                if source_char == target_char:
                    continue
                raw_start = parent.raw_start + offset
                atom = _RewriteCandidate(
                    operation="replace",
                    source=source_char,
                    target=target_char,
                    raw_start=raw_start,
                    raw_end=raw_start + 1,
                    left_context=(parent.left_context + parent.source[:offset])[-5:],
                    right_context=(parent.source[offset + 1:] + parent.right_context)[:5],
                )
                edits[(atom.raw_start, atom.raw_end, atom.target)] = (atom, parent)
        return edits

    paraformer_atoms = atomic_edits(replacements_for(paraformer_norm))
    qwen_atoms = atomic_edits(replacements_for(qwen_norm))
    global_context_norm = normalized_text(global_context_text)
    replacements: list[tuple[int, int, str, dict[str, Any]]] = []
    for key in sorted(set(paraformer_atoms) & set(qwen_atoms)):
        paraformer_atom, paraformer_parent = paraformer_atoms[key]
        qwen_atom, qwen_parent = qwen_atoms[key]
        if (
            paraformer_parent.raw_start == qwen_parent.raw_start
            and paraformer_parent.raw_end == qwen_parent.raw_end
            and paraformer_parent.target == qwen_parent.target
        ):
            # Exact replacements are handled by the stricter existing path.
            continue
        if not _CJK_ONLY_RE.fullmatch(
            paraformer_atom.source + paraformer_atom.target
        ):
            continue
        if _violates_local_consistency_guard(
            primary_text,
            primary_norm,
            paraformer_atom,
        ):
            continue
        if not _replacement_forms_known_token(primary_text, paraformer_atom):
            continue
        if global_context_norm:
            source_anchors = {
                paraformer_atom.left_context[-1:] + paraformer_atom.source,
                paraformer_atom.source + paraformer_atom.right_context[:1],
            }
            if any(
                len(anchor) >= 2 and global_context_norm.count(anchor) >= 2
                for anchor in source_anchors
            ):
                continue
        replacements.append((
            paraformer_atom.raw_start,
            paraformer_atom.raw_end,
            paraformer_atom.target,
            {
                "from": paraformer_atom.source,
                "to": paraformer_atom.target,
                "normalized_from": paraformer_atom.source,
                "normalized_to": paraformer_atom.target,
                "left_context": paraformer_atom.left_context,
                "right_context": paraformer_atom.right_context,
                "paraformer_parent": paraformer_parent.target,
                "qwen_parent": qwen_parent.target,
                "evidence": "paraformer_qwen_atomic_aligned_consensus",
            },
        ))

    corrected = primary_text
    changes: list[dict[str, Any]] = []
    for raw_start, raw_end, target, detail in sorted(
        replacements,
        key=lambda item: (item[0], item[1]),
        reverse=True,
    ):
        corrected = corrected[:raw_start] + target + corrected[raw_end:]
        changes.append(detail)
    changes.reverse()
    return corrected, changes


def phonetic_near_independent_consensus_rewrite(
    primary_text: str,
    paraformer_text: str,
    qwen_text: str,
    *,
    qwen_hallucination_risk: bool = False,
    max_span_chars: int = 6,
    global_context_text: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Use Qwen spelling only when Paraformer heard a closer phonetic variant."""
    if qwen_hallucination_risk:
        return primary_text, []
    primary_norm, primary_positions = _normalized_stream(primary_text)
    paraformer_norm = normalized_text(paraformer_text)
    qwen_norm = normalized_text(qwen_text)
    if not primary_norm or not paraformer_norm or not qwen_norm:
        return primary_text, []
    if text_similarity(primary_text, qwen_text) < _ANY_CHANGE_PRIMARY_QWEN_MIN:
        return primary_text, []
    if text_similarity(paraformer_text, qwen_text) < _ANY_CHANGE_DETECTOR_QWEN_MIN:
        return primary_text, []

    paraformer_candidates = [
        item
        for item in _collect_rewrite_candidates(
            primary_text,
            primary_norm,
            primary_positions,
            paraformer_norm,
            max_span_chars=max_span_chars,
            max_indel_chars=0,
        )
        if item.operation == "replace"
    ]
    qwen_candidates = [
        item
        for item in _collect_rewrite_candidates(
            primary_text,
            primary_norm,
            primary_positions,
            qwen_norm,
            max_span_chars=max_span_chars,
            max_indel_chars=0,
        )
        if item.operation == "replace"
    ]
    paraformer_exact = {
        (item.raw_start, item.raw_end, item.target)
        for item in paraformer_candidates
    }
    exact_anchor_spans = [
        (item.raw_start, item.raw_end)
        for item in qwen_candidates
        if (item.raw_start, item.raw_end, item.target) in paraformer_exact
    ]

    replacements: list[tuple[int, int, str, dict[str, Any]]] = []
    for candidate in qwen_candidates:
        if _violates_local_consistency_guard(
            primary_text,
            primary_norm,
            candidate,
        ):
            continue
        if not (2 <= len(candidate.source) <= max_span_chars):
            continue
        if not (2 <= len(candidate.target) <= max_span_chars):
            continue
        if len(candidate.target) == 2:
            has_adjacent_exact_anchor = any(
                (
                    max(
                        anchor_start - candidate.raw_end,
                        candidate.raw_start - anchor_end,
                        0,
                    )
                    <= 1
                    and (anchor_start, anchor_end)
                    != (candidate.raw_start, candidate.raw_end)
                )
                for anchor_start, anchor_end in exact_anchor_spans
            )
            has_global_text_anchor = candidate.target in normalized_text(global_context_text)
            if not has_adjacent_exact_anchor and not has_global_text_anchor:
                continue
        target_pinyin = _pinyin_stream(candidate.target)
        source_pinyin = _pinyin_stream(candidate.source)
        if not target_pinyin or not source_pinyin:
            continue
        matches: list[tuple[float, float, _RewriteCandidate]] = []
        for paraformer_candidate in paraformer_candidates:
            if (
                paraformer_candidate.raw_start != candidate.raw_start
                or paraformer_candidate.raw_end != candidate.raw_end
            ):
                continue
            paraformer_pinyin = _pinyin_stream(paraformer_candidate.target)
            if not paraformer_pinyin:
                continue
            target_similarity = SequenceMatcher(
                None, target_pinyin, paraformer_pinyin, autojunk=False
            ).ratio()
            source_similarity = SequenceMatcher(
                None, source_pinyin, paraformer_pinyin, autojunk=False
            ).ratio()
            minimum_target_similarity = 0.5 if len(candidate.target) == 2 else 0.75
            if (
                target_similarity < minimum_target_similarity
                or target_similarity < source_similarity + 0.25
            ):
                continue
            matches.append((target_similarity, source_similarity, paraformer_candidate))
        if not matches:
            continue
        target_similarity, source_similarity, paraformer_candidate = max(
            matches,
            key=lambda item: (item[0], -item[1]),
        )
        # Both independent decoders aligned a replacement to the exact same
        # source span. Requiring unchanged immediate neighbors here would miss
        # adjacent character corrections inside one technical term.
        display_target = _preserve_acronym_case(
            primary_text,
            candidate.raw_start,
            candidate.raw_end,
            candidate.target,
        )
        replacements.append((
            candidate.raw_start,
            candidate.raw_end,
            display_target,
            {
                "from": primary_text[candidate.raw_start:candidate.raw_end],
                "to": display_target,
                "normalized_from": candidate.source,
                "normalized_to": candidate.target,
                "paraformer_variant": paraformer_candidate.target,
                "left_context": candidate.left_context,
                "right_context": candidate.right_context,
                "target_paraformer_phonetic_similarity": round(target_similarity, 4),
                "source_paraformer_phonetic_similarity": round(source_similarity, 4),
                "evidence": "paraformer_qwen_phonetic_near_consensus",
            },
        ))

    corrected = primary_text
    changes: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for raw_start, raw_end, target, detail in sorted(
        replacements,
        key=lambda item: (item[0], item[1]),
        reverse=True,
    ):
        if any(raw_start < end and raw_end > start for start, end in occupied):
            continue
        corrected = corrected[:raw_start] + target + corrected[raw_end:]
        occupied.append((raw_start, raw_end))
        changes.append(detail)
    changes.reverse()
    return corrected, changes


def _measurement_records(text: str) -> list[tuple[str, str, int, int]]:
    normalized, positions = _normalized_stream(text)
    records: list[tuple[str, str, int, int]] = []
    for match in _MEASUREMENT_RE.finditer(normalized):
        start, end = match.span("unit")
        if start >= len(positions) or end <= 0:
            continue
        records.append((
            normalized[match.start():start],
            match.group("unit"),
            positions[start],
            positions[end - 1] + 1,
        ))
    return records


def _measurement_mentions(text: str) -> list[tuple[str, int, int]]:
    return [
        (unit, raw_start, raw_end)
        for _number, unit, raw_start, raw_end in _measurement_records(text)
    ]


def _measurement_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit, _start, _end in _measurement_mentions(text):
        counts[unit] = counts.get(unit, 0) + 1
    return counts


def _technical_review_signal(primary_text: str, detector_text: str) -> bool:
    """Select a bounded Qwen pass for repeated, conflicting measurement units."""
    primary_records = _measurement_records(primary_text)
    detector_counts = _measurement_counts(detector_text)
    if not detector_counts:
        return False
    number_counts: dict[str, int] = {}
    for number, _unit, _start, _end in primary_records:
        number_counts[number] = number_counts.get(number, 0) + 1
    repeated_numbers = {
        number for number, count in number_counts.items() if count >= 3
    }
    repeated_units = {
        unit
        for number, unit, _start, _end in primary_records
        if number in repeated_numbers
    }
    if len(repeated_units) < 2:
        return False
    for source in repeated_units:
        for target in detector_counts:
            if (
                len(source) >= 2
                and len(source) == len(target)
                and source[0] == target[0]
                and sum(left != right for left, right in zip(source, target)) == 1
            ):
                return True
    return False


def _common_suffix(left: str, right: str, *, limit: int = 5) -> str:
    length = 0
    for offset in range(1, min(len(left), len(right), limit) + 1):
        if left[-offset:] != right[-offset:]:
            break
        length = offset
    return left[-length:] if length else ""


def _common_prefix(left: str, right: str, *, limit: int = 5) -> str:
    length = 0
    for offset in range(min(len(left), len(right), limit)):
        if left[offset] != right[offset]:
            break
        length = offset + 1
    return left[:length]


def _acronym_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in _RAW_ACRONYM_RE.finditer(text or ""):
        token = match.group(1)
        counts[token] = counts.get(token, 0) + 1
    return counts


def _raw_latin_terms(text: str) -> set[str]:
    return {
        match.group(1).lower()
        for match in _RAW_LATIN_TERM_RE.finditer(text or "")
    }


def technical_consensus_rewrite(
    primary_text: str,
    paraformer_text: str,
    qwen_text: str,
    *,
    qwen_hallucination_risk: bool = False,
    global_context_text: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Repair constrained technical tokens without replacing a whole window.

    The pass handles only repeated measurement-unit conflicts, independently
    shared Latin terms anchored on both sides, and one-letter acronym insertion
    noise backed by a dominant transcript-wide spelling. It never changes the
    ASR timeline and deliberately leaves ambiguous vocabulary for review.
    """
    if qwen_hallucination_risk:
        return primary_text, []
    primary_norm, primary_positions = _normalized_stream(primary_text)
    paraformer_norm = normalized_text(paraformer_text)
    qwen_norm = normalized_text(qwen_text)
    if not primary_norm or not paraformer_norm or not qwen_norm:
        return primary_text, []

    replacements: list[tuple[int, int, str, dict[str, Any]]] = []

    # A unit is changed only when both independent models consistently heard
    # the same unit for every repeated measurement in the primary window.
    primary_records = _measurement_records(primary_text)
    paraformer_unit_counts = _measurement_counts(paraformer_text)
    qwen_unit_counts = _measurement_counts(qwen_text)
    records_by_number: dict[str, list[tuple[str, int, int]]] = {}
    for number, unit, raw_start, raw_end in primary_records:
        records_by_number.setdefault(number, []).append((unit, raw_start, raw_end))
    for number_records in records_by_number.values():
        if len(number_records) < 3:
            continue
        primary_unit_counts: dict[str, int] = {}
        for unit, _raw_start, _raw_end in number_records:
            primary_unit_counts[unit] = primary_unit_counts.get(unit, 0) + 1
        for target in set(paraformer_unit_counts) & set(qwen_unit_counts):
            if (
                primary_unit_counts.get(target, 0) < 1
                or paraformer_unit_counts[target] < len(number_records)
                or qwen_unit_counts[target] < len(number_records)
            ):
                continue
            for source, source_count in primary_unit_counts.items():
                if source == target or source_count < 1:
                    continue
                if paraformer_unit_counts.get(source, 0) or qwen_unit_counts.get(source, 0):
                    continue
                if (
                    len(source) < 2
                    or len(source) != len(target)
                    or source[0] != target[0]
                    or sum(left != right for left, right in zip(source, target)) != 1
                ):
                    continue
                for unit, raw_start, raw_end in number_records:
                    if unit != source:
                        continue
                    replacements.append((
                        raw_start,
                        raw_end,
                        target,
                        {
                            "from": primary_text[raw_start:raw_end],
                            "to": target,
                            "normalized_from": source,
                            "normalized_to": target,
                            "evidence": "paraformer_qwen_repeated_measurement_consensus",
                        },
                    ))

    # Long Latin terms are eligible only when both independent models contain
    # the exact token and agree on at least two normalized characters on each
    # side. The primary gap must already contain a >=3-character token fragment.
    shared_raw_latin_terms = _raw_latin_terms(paraformer_text) & _raw_latin_terms(qwen_text)
    shared_normalized_latin_terms = set(_LATIN_TERM_RE.findall(paraformer_norm)) & set(
        _LATIN_TERM_RE.findall(qwen_norm)
    )
    shared_latin_terms = shared_raw_latin_terms | shared_normalized_latin_terms
    for target in sorted(shared_latin_terms, key=len, reverse=True):
        para_occurrences = [
            match.start() for match in re.finditer(re.escape(target), paraformer_norm)
        ]
        qwen_occurrences = [
            match.start() for match in re.finditer(re.escape(target), qwen_norm)
        ]
        for para_start in para_occurrences:
            for qwen_start in qwen_occurrences:
                left_anchor = _common_suffix(
                    paraformer_norm[max(0, para_start - 5):para_start],
                    qwen_norm[max(0, qwen_start - 5):qwen_start],
                )
                right_anchor = _common_prefix(
                    paraformer_norm[para_start + len(target):para_start + len(target) + 5],
                    qwen_norm[qwen_start + len(target):qwen_start + len(target) + 5],
                )
                if len(left_anchor) < 2 or len(right_anchor) < 2:
                    continue
                if not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", left_anchor + right_anchor):
                    continue
                gap_pattern = re.compile(
                    re.escape(left_anchor)
                    + rf"(?P<gap>.{{1,{len(target) + 4}}}?)"
                    + re.escape(right_anchor)
                )
                for match in gap_pattern.finditer(primary_norm):
                    gap = match.group("gap")
                    if gap == target:
                        continue
                    if (
                        target not in shared_raw_latin_terms
                        and not _CJK_TEXT_RE.search(gap)
                    ):
                        continue
                    fragments = re.findall(r"[a-z0-9]+", gap)
                    if not any(
                        (
                            len(fragment) >= 3
                            and (fragment in target or target in fragment)
                        )
                        or (
                            len(target) >= 6
                            and len(fragment) >= 2
                            and (
                                target.startswith(fragment)
                                or target.endswith(fragment)
                            )
                        )
                        for fragment in fragments
                    ):
                        continue
                    norm_start, norm_end = match.span("gap")
                    raw_start = primary_positions[norm_start]
                    raw_end = primary_positions[norm_end - 1] + 1
                    if "\n" in primary_text[raw_start:raw_end]:
                        continue
                    replacements.append((
                        raw_start,
                        raw_end,
                        target,
                        {
                            "from": primary_text[raw_start:raw_end],
                            "to": target,
                            "normalized_from": gap,
                            "normalized_to": target,
                            "left_context": left_anchor,
                            "right_context": right_anchor,
                            "evidence": "paraformer_qwen_anchored_latin_consensus",
                        },
                    ))

    # Acronym cleanup is transcript-consistency based: both independent models
    # must confirm the shorter spelling in the same local context, and that
    # spelling must dominate the one-letter-expanded primary token.
    global_acronyms = _acronym_counts(global_context_text)
    primary_acronyms = _acronym_counts(primary_text)
    raw_to_norm = {raw: index for index, raw in enumerate(primary_positions)}
    for match in _RAW_ACRONYM_RE.finditer(primary_text):
        source = match.group(1)
        source_norm = source.lower()
        source_norm_start = raw_to_norm.get(match.start(1))
        if source_norm_start is None:
            continue
        source_norm_end = source_norm_start + len(source_norm)
        left_context = primary_norm[max(0, source_norm_start - 2):source_norm_start]
        right_context = primary_norm[source_norm_end:source_norm_end + 2]
        for target, target_count in global_acronyms.items():
            target_norm = target.lower()
            if len(target_norm) < 3:
                continue
            if len(target_norm) + 1 != len(source_norm):
                continue
            if not any(
                source_norm[:index] + source_norm[index + 1:] == target_norm
                for index in range(len(source_norm))
            ):
                continue
            source_count = max(global_acronyms.get(source, 0), primary_acronyms.get(source, 0))
            if target_count < max(3, source_count * 3):
                continue
            right_anchor = right_context[:2]
            left_anchor = left_context[-2:]
            anchored_in_qwen = bool(
                (
                    right_anchor
                    and re.search(
                        rf"(?<![a-z0-9]){re.escape(target_norm + right_anchor)}",
                        qwen_norm,
                    )
                )
                or (
                    left_anchor
                    and re.search(
                        rf"{re.escape(left_anchor + target_norm)}(?![a-z0-9])",
                        qwen_norm,
                    )
                )
            )
            anchored_in_paraformer = bool(
                (
                    right_anchor
                    and re.search(
                        rf"(?<![a-z0-9]){re.escape(target_norm + right_anchor)}",
                        paraformer_norm,
                    )
                )
                or (
                    left_anchor
                    and re.search(
                        rf"{re.escape(left_anchor + target_norm)}(?![a-z0-9])",
                        paraformer_norm,
                    )
                )
            )
            if not anchored_in_qwen or not anchored_in_paraformer:
                continue
            replacements.append((
                match.start(1),
                match.end(1),
                target,
                {
                    "from": source,
                    "to": target,
                    "normalized_from": source_norm,
                    "normalized_to": target_norm,
                    "left_context": left_context,
                    "right_context": right_context,
                    "global_target_count": target_count,
                    "global_source_count": source_count,
                    "evidence": "paraformer_qwen_global_acronym_consistency",
                },
            ))

    corrected = primary_text
    changes: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for raw_start, raw_end, target, detail in sorted(
        replacements,
        key=lambda item: (item[0], item[1]),
        reverse=True,
    ):
        if any(raw_start < end and raw_end > start for start, end in occupied):
            continue
        corrected = corrected[:raw_start] + target + corrected[raw_end:]
        occupied.append((raw_start, raw_end))
        changes.append(detail)
    changes.reverse()
    return corrected, changes


def _split_text_by_weights(text: str, weights: list[int]) -> list[str]:
    if not weights:
        return []
    if len(weights) == 1:
        return [text]
    total = sum(max(weight, 1) for weight in weights)
    normalized, positions = _normalized_stream(text)
    if not normalized or not positions:
        return [text] + [""] * (len(weights) - 1)
    parts: list[str] = []
    raw_cursor = 0
    cumulative = 0
    for weight in weights[:-1]:
        cumulative += max(weight, 1)
        target_index = min(len(positions) - 1, max(0, round(len(positions) * cumulative / total) - 1))
        raw_cut = positions[target_index] + 1
        while raw_cut < len(text) and text[raw_cut] in "，。！？；：、,.!?;: ":
            raw_cut += 1
        parts.append(text[raw_cursor:raw_cut].strip())
        raw_cursor = raw_cut
    parts.append(text[raw_cursor:].strip())
    return parts


def _replace_cue_text_preserving_times(
    cues: list[dict[str, Any]] | None,
    corrected_text: str,
) -> list[dict[str, Any]] | None:
    if not cues:
        return cues
    weights = [max(len(normalized_text(str(cue.get("text") or ""))), 1) for cue in cues]
    parts = _split_text_by_weights(corrected_text, weights)
    return [
        {**cue, "text": parts[index] if index < len(parts) else ""}
        for index, cue in enumerate(cues)
    ]


def apply_window_text(
    segments: list[Segment],
    window: ReviewWindow,
    corrected_window_text: str,
) -> list[Segment]:
    original_parts = [segments[index].text for index in window.segment_indexes]
    corrected_parts = corrected_window_text.split("\n")
    if len(corrected_parts) != len(original_parts):
        return segments
    output = list(segments)
    for index, corrected in zip(window.segment_indexes, corrected_parts):
        current = output[index]
        if corrected == current.text:
            continue
        output[index] = replace(
            current,
            text=corrected,
            original_text=current.original_text or current.text,
            sync_cues=_replace_cue_text_preserving_times(current.sync_cues, corrected),
        )
    return output


def _window_text(segments: list[Segment], indexes: tuple[int, ...]) -> str:
    return "\n".join(segments[index].text or "" for index in indexes)


def _slice_text_by_time_ratio(text: str, start_ratio: float, end_ratio: float) -> str:
    normalized, positions = _normalized_stream(text)
    if not normalized or not positions:
        return text
    start_ratio = min(max(start_ratio, 0.0), 1.0)
    end_ratio = min(max(end_ratio, start_ratio), 1.0)
    start_index = min(len(positions) - 1, int(len(positions) * start_ratio))
    end_index = min(len(positions), max(start_index + 1, round(len(positions) * end_ratio)))
    raw_start = positions[start_index]
    raw_end = positions[end_index - 1] + 1
    return text[raw_start:raw_end]


def _overlap_text(segments: list[Segment], start: float, end: float) -> str:
    parts: list[str] = []
    for segment in segments:
        seg_start = float(segment.start)
        seg_end = float(segment.end)
        overlap_start = max(seg_start, start)
        overlap_end = min(seg_end, end)
        if overlap_end <= overlap_start:
            continue
        duration = seg_end - seg_start
        if duration <= 0 or (overlap_start <= seg_start and overlap_end >= seg_end):
            parts.append(segment.text or "")
            continue
        parts.append(
            _slice_text_by_time_ratio(
                segment.text or "",
                (overlap_start - seg_start) / duration,
                (overlap_end - seg_start) / duration,
            )
        )
    return "".join(parts)


def _offset_segments(segments: list[Segment], offset: float) -> list[Segment]:
    """Move clip-relative candidate timestamps onto the source timeline."""
    shifted: list[Segment] = []
    for segment in segments:
        shifted.append(
            replace(
                segment,
                start=float(segment.start) + offset,
                end=float(segment.end) + offset,
            )
        )
    return shifted


def _qwen_window_text(
    matching_tiles: list[tuple[int, dict[str, Any]]],
    start: float,
    end: float,
) -> str:
    """Return only Qwen text overlapping a review window.

    Qwen runs on reusable 20-second tiles. Concatenating whole tile transcripts
    polluted a short review window with up to 40 seconds of neighbouring text,
    which depressed model agreement and hid otherwise safe corrections.
    """
    parts: list[str] = []
    for _tile_index, tile in matching_tiles:
        tile_segments = list(tile.get("segments") or [])
        if tile_segments:
            parts.append(_overlap_text(tile_segments, start, end))
        else:
            parts.append(str(tile.get("text") or ""))
    return "".join(parts)


def _qwen_model_cached(model_id: str = QWEN_MODEL) -> bool:
    if os.environ.get("LOCALSCRIBE_ALLOW_MODEL_DOWNLOAD", "").strip().lower() in {"1", "true", "yes"}:
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        config = try_to_load_from_cache(model_id, "config.json")
        if not isinstance(config, str):
            return False
        snapshot = Path(config).parent
        return any(snapshot.glob("*.safetensors"))
    except Exception:
        return False


def _qwen_runtime_available() -> tuple[bool, str]:
    """Probe MLX in a child process so a native Metal abort cannot kill ASR."""
    override = os.environ.get("LOCALSCRIBE_QWEN_RUNTIME", "").strip().lower()
    if override in {"0", "false", "no", "off", "disabled"}:
        return False, "disabled_by_environment"
    if override in {"1", "true", "yes", "on", "force"}:
        return True, "forced_by_environment"
    if getattr(sys, "frozen", False):
        # The packaged sidecar cannot execute ``-c`` as a Python interpreter.
        # It is launched by the macOS GUI process where Metal is available.
        return True, "packaged_macos_runtime"

    probe = (
        "import mlx.core as mx; "
        "value=mx.array([1.0]); "
        "mx.eval(value); "
        "print(float(value[0]))"
    )
    try:
        process = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"mlx_probe_failed:{type(exc).__name__}"
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "mlx_probe_failed").strip()
        return False, f"mlx_probe_unavailable:{detail[-240:]}"
    return True, "mlx_probe_ok"


def _extract_clip(audio: Path, start: float, end: float, output: Path) -> None:
    from .audio import find_ffmpeg

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is unavailable")
    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            f"{max(start, 0.0):.3f}",
            "-t",
            f"{max(end - start, 0.1):.3f}",
            "-i",
            str(audio),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0 or not output.exists() or output.stat().st_size <= 44:
        raise RuntimeError((process.stderr or "failed to extract ASR review clip").strip())


def _transcribe_paraformer_for_review(
    paraformer: Any,
    audio_path: Path,
    *,
    language: str | None,
    model_id: str,
    audio_end: float,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[Segment], int]:
    """Transcribe long review audio in bounded, timeline-safe chunks."""
    options = TranscribeOptions(
        language=language,
        model_id=model_id,
        audio_preprocess="off",
    )
    if audio_end <= _PARAFORMER_FULL_AUDIO_MAX_SECONDS:
        result = paraformer.transcribe(audio_path, options)
        return list(result.segments), 1

    tile_count = max(
        1,
        int((audio_end + _PARAFORMER_TILE_SECONDS - 0.001) // _PARAFORMER_TILE_SECONDS),
    )
    output: list[Segment] = []
    with tempfile.TemporaryDirectory(prefix="localscribe-paraformer-tiles-") as tmp:
        tmp_root = Path(tmp)
        for tile_index in range(tile_count):
            core_start = tile_index * _PARAFORMER_TILE_SECONDS
            core_end = min(audio_end, core_start + _PARAFORMER_TILE_SECONDS)
            clip_start = max(0.0, core_start - _PARAFORMER_TILE_PADDING_SECONDS)
            clip_end = min(audio_end, core_end + _PARAFORMER_TILE_PADDING_SECONDS)
            clip = tmp_root / f"paraformer_{tile_index:04d}.wav"
            _extract_clip(audio_path, clip_start, clip_end, clip)
            if on_progress:
                on_progress({
                    "stage": "strong_asr_paraformer",
                    "current": tile_index,
                    "total": tile_count,
                    "preview": f"高质量复核:第二模型 {tile_index + 1}/{tile_count}",
                })
            result = paraformer.transcribe(clip, options)
            for segment in _offset_segments(list(result.segments), clip_start):
                start = max(float(segment.start), core_start)
                end = min(float(segment.end), core_end)
                if end <= start:
                    continue
                text = _overlap_text([segment], start, end)
                if normalized_text(text):
                    output.append(Segment(start=start, end=end, text=text))
            if on_progress:
                on_progress({
                    "stage": "strong_asr_paraformer",
                    "current": tile_index + 1,
                    "total": tile_count,
                    "preview": f"高质量复核:第二模型 {tile_index + 1}/{tile_count} 完成",
                })
    return output, tile_count


def _transcribe_paraformer_review_windows(
    paraformer: Any,
    audio_path: Path,
    windows: Iterable[ReviewWindow],
    *,
    language: str | None,
    model_id: str,
    audio_end: float,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[Segment], int]:
    """Run Paraformer only on selected risk windows and project to source time."""
    selected = list(windows)
    options = TranscribeOptions(
        language=language,
        model_id=model_id,
        audio_preprocess="off",
    )
    output: list[Segment] = []
    with tempfile.TemporaryDirectory(prefix="localscribe-paraformer-windows-") as tmp:
        tmp_root = Path(tmp)
        for position, window in enumerate(selected):
            core_start = max(0.0, float(window.start))
            core_end = min(audio_end, float(window.end))
            if core_end <= core_start:
                continue
            clip_start = max(0.0, core_start - _PARAFORMER_TILE_PADDING_SECONDS)
            clip_end = min(audio_end, core_end + _PARAFORMER_TILE_PADDING_SECONDS)
            clip = tmp_root / f"paraformer_window_{position:04d}.wav"
            _extract_clip(audio_path, clip_start, clip_end, clip)
            if on_progress:
                on_progress({
                    "stage": "strong_asr_paraformer",
                    "current": position,
                    "total": len(selected),
                    "preview": f"高质量复核:疑点窗口 {position + 1}/{len(selected)}",
                })
            result = paraformer.transcribe(clip, options)
            for segment in _offset_segments(list(result.segments), clip_start):
                start = max(float(segment.start), core_start)
                end = min(float(segment.end), core_end)
                if end <= start:
                    continue
                text = _overlap_text([segment], start, end)
                if normalized_text(text):
                    output.append(Segment(start=start, end=end, text=text))
            if on_progress:
                on_progress({
                    "stage": "strong_asr_paraformer",
                    "current": position + 1,
                    "total": len(selected),
                    "preview": f"高质量复核:疑点窗口 {position + 1}/{len(selected)} 完成",
                })
    return output, len(selected)


def _review_evidence_bounds(
    window: ReviewWindow,
    *,
    audio_end: float,
) -> tuple[float, float]:
    return (
        max(0.0, float(window.start) - _REDECODE_LEFT_CONTEXT_SECONDS),
        min(audio_end, float(window.end) + _REDECODE_RIGHT_CONTEXT_SECONDS),
    )


def _transcribe_primary_redecode_windows(
    audio_path: Path,
    windows: list[tuple[int, ReviewWindow]],
    *,
    language: str | None,
    audio_end: float,
    on_progress: ProgressCallback | None = None,
) -> dict[int, str]:
    """Re-decode only disputed windows with the primary SenseVoice model."""
    from .transcriber_funasr import SENSEVOICE_MODEL, FunASRTranscriber, model_cached

    if not model_cached(SENSEVOICE_MODEL):
        raise RuntimeError("sensevoice_model_not_cached")

    transcriber = FunASRTranscriber(backend_name="sensevoice")
    output: dict[int, str] = {}
    with tempfile.TemporaryDirectory(prefix="localscribe-sensevoice-redecode-") as tmp:
        tmp_root = Path(tmp)
        for position, (window_index, window) in enumerate(windows):
            clip_start, clip_end = _review_evidence_bounds(window, audio_end=audio_end)
            clip = tmp_root / f"window_{window_index:04d}.wav"
            _extract_clip(audio_path, clip_start, clip_end, clip)
            if on_progress:
                on_progress(
                    {
                        "stage": "strong_asr_primary_redecode",
                        "current": position + 1,
                        "total": len(windows),
                        "preview": f"高质量复核:主模型短窗 {position + 1}/{len(windows)}",
                    }
                )
            result = transcriber.transcribe(
                clip,
                TranscribeOptions(
                    language=language,
                    model_id=SENSEVOICE_MODEL,
                    timing_align=False,
                    audio_preprocess="off",
                ),
            )
            output[window_index] = "".join(segment.text or "" for segment in result.segments)
    del transcriber
    _release_model_memory()
    return output


def _release_model_memory() -> None:
    gc.collect()
    # Some macOS/PyTorch builds abort the interpreter inside the native MPS
    # cache API, so Python exception handling cannot recover. Object deletion
    # plus GC is the stable default; allow cache flushing only in environments
    # where it has been explicitly validated.
    opt_in = os.environ.get("LOCALSCRIBE_TORCH_MPS_EMPTY_CACHE", "0").strip().lower()
    if opt_in not in {"1", "true", "yes", "on"}:
        return
    try:
        import torch

        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


ProgressCallback = Callable[[dict[str, Any]], None]


def run_strong_asr_review(
    audio: Path | str,
    primary_segments: list[Segment],
    *,
    qwen_audio: Path | str | None = None,
    detector_segments: list[Segment] | None = None,
    detector_source: str = "",
    detector_windows: Iterable[ReviewWindow] | None = None,
    language: str | None = "zh",
    on_progress: ProgressCallback | None = None,
    qwen_model: str = QWEN_MODEL,
    allow_length_changing_edits: bool = True,
) -> tuple[list[Segment], dict[str, Any]]:
    """Run the optional local consensus pass without changing timeline data."""
    before_fingerprint = timeline_fingerprint(primary_segments)
    sparse_windows = list(detector_windows) if detector_windows is not None else None
    stats: dict[str, Any] = {
        "mode": "local_strong_asr_consensus",
        "enabled": True,
        "applied": False,
        "primary_segments": len(primary_segments),
        "window_count": 0,
        "candidate_window_count": 0,
        "structurally_reviewable_window_count": 0,
        "technical_review_window_count": 0,
        "primary_redecode_skipped_window_count": 0,
        "primary_redecode_detector_agreement_count": 0,
        "detector_confirmed_window_count": 0,
        "qwen_skipped_window_count": 0,
        "audit_only_window_count": 0,
        "audit_only_candidates": [],
        "paraformer_tile_count": 0,
        "detector_reused": False,
        "detector_source": detector_source or "paraformer_review_pass",
        "detector_scope": "selected_windows" if sparse_windows is not None else "full_recording",
        "primary_redecode_window_count": 0,
        "qwen_tile_count": 0,
        "reviewed_windows": 0,
        "changed_windows": 0,
        "replacement_count": 0,
        "changes": [],
        "window_diagnostics": [],
        "timeline_fingerprint_before": before_fingerprint,
        "timeline_fingerprint_after": before_fingerprint,
        "timeline_preserved": True,
        "qwen_model": qwen_model,
        "qwen_runtime_available": None,
        "qwen_runtime_reason": "",
        "primary_timeline_owner": "input_primary_segments",
        "selection_policy": "bounded_detector_delta_plus_constrained_technical_consensus_v3",
        "independent_confirmation_count": 2,
        "similarity_gate": {
            "primary_para_window_min": _PRIMARY_PARA_AUDIT_ONLY_BELOW,
            "primary_para_window_max": _PRIMARY_PARA_REVIEW_BELOW,
            "primary_para_audit_only_below": _PRIMARY_PARA_AUDIT_ONLY_BELOW,
            "any_change_candidate_qwen_min": _ANY_CHANGE_CANDIDATE_QWEN_MIN,
            "any_change_primary_qwen_min": _ANY_CHANGE_PRIMARY_QWEN_MIN,
            "any_change_detector_qwen_min": _ANY_CHANGE_DETECTOR_QWEN_MIN,
            "multichar_candidate_qwen_min": _MULTICHAR_CANDIDATE_QWEN_MIN,
            "multichar_primary_qwen_min": _MULTICHAR_PRIMARY_QWEN_MIN,
        },
        "boundary_context_min_chars": _BOUNDARY_CONTEXT_MIN_CHARS,
        "internal_indel_boundary_context_min_chars": _INDEL_BOUNDARY_CONTEXT_MIN_CHARS,
        "max_internal_indel_chars": _DEFAULT_MAX_INDEL_CHARS,
        "allow_length_changing_edits": bool(allow_length_changing_edits),
        "reason": "",
    }
    if not primary_segments:
        stats["reason"] = "empty_primary_segments"
        return primary_segments, stats
    if sparse_windows is not None and not sparse_windows:
        stats["reason"] = "no_sparse_review_windows"
        return primary_segments, stats
    if not _qwen_model_cached(qwen_model):
        stats["reason"] = "qwen_model_not_cached"
        return primary_segments, stats
    qwen_runtime_available, qwen_runtime_reason = _qwen_runtime_available()
    stats["qwen_runtime_available"] = qwen_runtime_available
    stats["qwen_runtime_reason"] = qwen_runtime_reason

    audio_path = Path(audio).expanduser().resolve()
    if not audio_path.exists():
        stats["reason"] = "audio_not_found"
        return primary_segments, stats
    qwen_audio_path = Path(qwen_audio or audio_path).expanduser().resolve()
    if not qwen_audio_path.exists():
        qwen_audio_path = audio_path
    stats["paraformer_audio"] = str(audio_path)
    stats["qwen_audio"] = str(qwen_audio_path)
    stats["independent_audio_views"] = qwen_audio_path != audio_path

    audio_end = max((float(segment.end) for segment in primary_segments), default=0.0)
    all_windows = build_review_windows(primary_segments)
    all_window_keys = {
        (round(window.start, 6), round(window.end, 6), window.segment_indexes)
        for window in all_windows
    }
    sparse_window_keys = {
        (round(window.start, 6), round(window.end, 6), window.segment_indexes)
        for window in (sparse_windows or [])
    }
    detector_covers_full_recording = sparse_windows is None or (
        bool(all_windows) and sparse_window_keys == all_window_keys
    )
    stats["detector_scope"] = (
        "full_recording" if detector_covers_full_recording else "selected_windows"
    )
    para_segments = list(detector_segments or [])
    if para_segments and "paraformer" not in str(detector_source or "").lower():
        stats["detector_reuse_rejected"] = True
        stats["detector_reuse_rejected_reason"] = "detector_is_not_independent_paraformer"
        stats["detector_source"] = "paraformer_review_pass"
        para_segments = []
    if para_segments:
        stats["detector_reused"] = True
        stats["detector_source"] = detector_source or "existing_timing_anchor"
    else:
        try:
            from .transcriber_funasr import DEFAULT_MODEL as PARAFORMER_MODEL
            from .transcriber_funasr import FunASRTranscriber, model_cached

            if not model_cached(PARAFORMER_MODEL):
                stats["reason"] = "paraformer_model_not_cached"
                return primary_segments, stats
            if on_progress:
                on_progress({"stage": "strong_asr_paraformer", "preview": "高质量复核:生成分歧检测转写"})
            paraformer = FunASRTranscriber(backend_name="funasr")
            if detector_covers_full_recording:
                para_segments, paraformer_tile_count = _transcribe_paraformer_for_review(
                    paraformer,
                    audio_path,
                    language=language,
                    model_id=PARAFORMER_MODEL,
                    audio_end=audio_end,
                    on_progress=on_progress,
                )
            else:
                para_segments, paraformer_tile_count = _transcribe_paraformer_review_windows(
                    paraformer,
                    audio_path,
                    sparse_windows,
                    language=language,
                    model_id=PARAFORMER_MODEL,
                    audio_end=audio_end,
                    on_progress=on_progress,
                )
            stats["paraformer_tile_count"] = paraformer_tile_count
            del paraformer
            _release_model_memory()
        except Exception as exc:
            stats["reason"] = f"paraformer_failed:{type(exc).__name__}"
            stats["error"] = str(exc)
            return primary_segments, stats

    windows = sparse_windows if sparse_windows is not None else all_windows
    stats["all_window_count"] = len(all_windows)
    stats["window_count"] = len(windows)
    output = list(primary_segments)
    primary_global_context = "".join(segment.text or "" for segment in output)
    try:
        from .transcriber_qwen3 import Qwen3ASRTranscriber

        with tempfile.TemporaryDirectory(prefix="localscribe-strong-asr-") as tmp:
            tmp_root = Path(tmp)
            candidate_windows: list[tuple[int, ReviewWindow, str, str, float, bool]] = []
            structurally_reviewable_window_indexes: set[int] = set()
            technical_review_window_indexes: set[int] = set()
            tile_seconds = 20.0
            for window_index, window in enumerate(windows):
                primary_text = _window_text(output, window.segment_indexes)
                para_text = _overlap_text(para_segments, window.start, window.end)
                primary_para_similarity = text_similarity(primary_text, para_text)
                if primary_para_similarity >= _PRIMARY_PARA_REVIEW_BELOW:
                    continue
                audit_only = primary_para_similarity < _PRIMARY_PARA_AUDIT_ONLY_BELOW
                candidate_windows.append(
                    (
                        window_index,
                        window,
                        primary_text,
                        para_text,
                        primary_para_similarity,
                        audit_only,
                    )
                )
                if not audit_only and _detector_confirms_rewrite_candidate(
                    primary_text,
                    para_text,
                    para_text,
                ):
                    structurally_reviewable_window_indexes.add(window_index)
                elif not audit_only and _technical_review_signal(primary_text, para_text):
                    technical_review_window_indexes.add(window_index)

            stats["candidate_window_count"] = len(candidate_windows)
            stats["structurally_reviewable_window_count"] = len(
                structurally_reviewable_window_indexes
            )
            stats["technical_review_window_count"] = len(
                technical_review_window_indexes
            )
            if not qwen_runtime_available:
                stats["reviewed_windows"] = len(candidate_windows)
                stats["qwen_skipped_window_count"] = len(candidate_windows)
                for (
                    window_index,
                    window,
                    primary_text,
                    para_text,
                    primary_para_similarity,
                    audit_only,
                ) in candidate_windows[:100]:
                    stats["window_diagnostics"].append({
                        "window": window_index,
                        "start": round(window.start, 3),
                        "end": round(window.end, 3),
                        "primary_para_similarity": round(primary_para_similarity, 4),
                        "audit_only": audit_only,
                        "qwen_review_eligible": False,
                        "qwen_skipped_reason": "qwen_runtime_unavailable",
                        "auto_replace_allowed": False,
                        "candidates": {
                            "primary": primary_text,
                            "primary_redecode": "",
                            "paraformer": para_text,
                            "qwen": "",
                        },
                        "confirmed_change_count": 0,
                    })
                stats["reason"] = "qwen_runtime_unavailable"
                return primary_segments, stats
            rewrite_windows = [
                (window_index, window)
                for window_index, window, _primary, _detector, _similarity, audit_only in candidate_windows
                if not audit_only and window_index in structurally_reviewable_window_indexes
            ]
            stats["primary_redecode_skipped_window_count"] = (
                len(candidate_windows) - len(rewrite_windows)
            )
            primary_redecode = _transcribe_primary_redecode_windows(
                audio_path,
                rewrite_windows,
                language=language,
                audio_end=audio_end,
                on_progress=on_progress,
            ) if rewrite_windows else {}
            stats["primary_redecode_window_count"] = len(primary_redecode)

            qwen_eligible_window_indexes: set[int] = set()
            primary_redecode_detector_agreement_indexes: set[int] = set()
            needed_tile_indexes: set[int] = set()
            for (
                window_index,
                window,
                primary_text,
                _para_text,
                _primary_para_similarity,
                audit_only,
            ) in candidate_windows:
                if audit_only:
                    continue
                if window_index in technical_review_window_indexes:
                    evidence_start, evidence_end = _review_evidence_bounds(
                        window,
                        audio_end=audio_end,
                    )
                    first_tile = max(0, int(evidence_start // tile_seconds))
                    last_tile = max(
                        first_tile,
                        int(max(evidence_end - 0.001, 0.0) // tile_seconds),
                    )
                    needed_tile_indexes.update(range(first_tile, last_tile + 1))
                if window_index not in structurally_reviewable_window_indexes:
                    continue
                evidence_start, evidence_end = _review_evidence_bounds(
                    window,
                    audio_end=audio_end,
                )
                detector_evidence_text = _overlap_text(
                    para_segments,
                    evidence_start,
                    evidence_end,
                )
                if _detector_confirms_rewrite_candidate(
                    primary_text,
                    primary_redecode.get(window_index, ""),
                    detector_evidence_text,
                ):
                    primary_redecode_detector_agreement_indexes.add(window_index)
                # Qwen arbitrates every bounded SenseVoice/Paraformer
                # disagreement. Requiring the same SenseVoice model to agree
                # with itself first hides systematic lexical errors.
                qwen_eligible_window_indexes.add(window_index)
                first_tile = max(0, int(evidence_start // tile_seconds))
                last_tile = max(
                    first_tile,
                    int(max(evidence_end - 0.001, 0.0) // tile_seconds),
                )
                needed_tile_indexes.update(range(first_tile, last_tile + 1))
            stats["detector_confirmed_window_count"] = len(qwen_eligible_window_indexes)
            stats["primary_redecode_detector_agreement_count"] = len(
                primary_redecode_detector_agreement_indexes
            )
            stats["qwen_skipped_window_count"] = (
                len(candidate_windows)
                - len(qwen_eligible_window_indexes | technical_review_window_indexes)
            )

            qwen_tiles: dict[int, dict[str, Any]] = {}
            ordered_tiles = [
                tile_index
                for tile_index in sorted(needed_tile_indexes)
                if tile_index * tile_seconds < audio_end
            ]
            stats["qwen_tile_count"] = len(ordered_tiles)
            qwen = Qwen3ASRTranscriber() if ordered_tiles else None
            for tile_position, tile_index in enumerate(ordered_tiles):
                tile_start = tile_index * tile_seconds
                tile_end = min(audio_end, tile_start + tile_seconds)
                if tile_end <= tile_start:
                    continue
                clip = tmp_root / f"tile_{tile_index:04d}.wav"
                _extract_clip(qwen_audio_path, tile_start, tile_end, clip)
                if on_progress:
                    on_progress(
                        {
                            "stage": "strong_asr_qwen",
                            "current": tile_position + 1,
                            "total": len(ordered_tiles),
                            "preview": f"高质量复核 {tile_position + 1}/{len(ordered_tiles)}",
                        }
                    )
                qwen_result = qwen.transcribe(
                    clip,
                    TranscribeOptions(
                        language=language,
                        model_id=qwen_model,
                        audio_preprocess="off",
                    ),
                )
                qwen_text = "".join(segment.text or "" for segment in qwen_result.segments)
                qwen_segments = _offset_segments(qwen_result.segments, tile_start)
                qwen_stats = qwen_result.filter_stats or {}
                qwen_tiles[tile_index] = {
                    "start": tile_start,
                    "end": tile_end,
                    "text": qwen_text,
                    "segments": qwen_segments,
                    "hallucination_risk": bool(qwen_stats.get("has_hallucination_risk")),
                }

            for (
                window_index,
                window,
                primary_text,
                para_text,
                primary_para_similarity,
                audit_only,
            ) in candidate_windows:
                evidence_start, evidence_end = _review_evidence_bounds(
                    window,
                    audio_end=audio_end,
                )
                matching_tiles = [
                    (tile_index, tile)
                    for tile_index, tile in sorted(qwen_tiles.items())
                    if min(float(tile["end"]), evidence_end)
                    - max(float(tile["start"]), evidence_start)
                    > 0.0
                ]
                qwen_text = _qwen_window_text(
                    matching_tiles,
                    evidence_start,
                    evidence_end,
                )
                detector_evidence_text = _overlap_text(
                    para_segments,
                    evidence_start,
                    evidence_end,
                )
                primary_redecode_text = primary_redecode.get(window_index, "")
                hallucination_risk = any(
                    bool(tile["hallucination_risk"]) for _tile_index, tile in matching_tiles
                )
                stats["reviewed_windows"] += 1
                candidate_qwen_similarity = text_similarity(primary_redecode_text, qwen_text)
                primary_qwen_similarity = text_similarity(primary_text, qwen_text)
                detector_qwen_similarity = text_similarity(detector_evidence_text, qwen_text)
                audit_only_reason = (
                    "primary_para_similarity_below_0.45" if audit_only else None
                )
                qwen_skip_reason = None
                if audit_only:
                    qwen_skip_reason = "audit_only_window"
                elif (
                    window_index not in structurally_reviewable_window_indexes
                    and window_index not in technical_review_window_indexes
                ):
                    qwen_skip_reason = "no_bounded_detector_delta"
                elif window_index not in qwen_eligible_window_indexes:
                    if window_index not in technical_review_window_indexes:
                        qwen_skip_reason = "detector_did_not_confirm_primary_redecode"
                consensus_path = "none"
                if audit_only:
                    # Severe disagreement is useful review evidence but too
                    # ambiguous for an unattended edit, even if Qwen agrees
                    # with one candidate. Keep all candidates for inspection
                    # and do not enter the automatic rewrite path.
                    corrected, changes = primary_text, []
                    stats["audit_only_window_count"] += 1
                    if len(stats["audit_only_candidates"]) < 100:
                        stats["audit_only_candidates"].append(
                            {
                                "window": window_index,
                                "start": round(window.start, 3),
                                "end": round(window.end, 3),
                                "reason": audit_only_reason,
                                "primary_para_similarity": round(primary_para_similarity, 4),
                                "primary": primary_text,
                                "paraformer": para_text,
                                "primary_redecode": primary_redecode_text,
                                "qwen": qwen_text,
                                "qwen_skipped_reason": qwen_skip_reason,
                                "qwen_hallucination_risk": hallucination_risk,
                            }
                        )
                elif (
                    window_index not in qwen_eligible_window_indexes
                    and window_index not in technical_review_window_indexes
                ):
                    corrected, changes = primary_text, []
                elif window_index in technical_review_window_indexes:
                    corrected, changes = primary_text, []
                else:
                    corrected, changes = consensus_rewrite(
                        primary_text,
                        primary_redecode_text,
                        qwen_text,
                        detector_text=detector_evidence_text,
                        qwen_hallucination_risk=hallucination_risk,
                        candidate_qwen_similarity=candidate_qwen_similarity,
                        primary_qwen_similarity=primary_qwen_similarity,
                        detector_qwen_similarity=detector_qwen_similarity,
                        global_context_text=primary_global_context,
                        allow_deletions=allow_length_changing_edits,
                        allow_insertions=allow_length_changing_edits,
                    )
                    consensus_path = "sensevoice_redecode_paraformer_qwen"
                    if not changes:
                        corrected, changes = independent_consensus_rewrite(
                            primary_text,
                            para_text,
                            qwen_text,
                            qwen_hallucination_risk=hallucination_risk,
                            global_context_text=primary_global_context,
                            allow_insertions=allow_length_changing_edits,
                        )
                        if changes:
                            consensus_path = "paraformer_qwen_independent"
                    if not changes:
                        corrected, changes = phonetic_near_independent_consensus_rewrite(
                            primary_text,
                            para_text,
                            qwen_text,
                            qwen_hallucination_risk=hallucination_risk,
                            global_context_text=primary_global_context,
                        )
                        if changes:
                            consensus_path = "paraformer_qwen_phonetic_near"
                    length_preserving = all(
                        len(str(item.get("normalized_from") or ""))
                        == len(str(item.get("normalized_to") or ""))
                        for item in changes
                    )
                    if not changes or length_preserving:
                        aligned_corrected, aligned_changes = (
                            aligned_independent_consensus_rewrite(
                                corrected,
                                para_text,
                                qwen_text,
                                qwen_hallucination_risk=hallucination_risk,
                                global_context_text=primary_global_context,
                            )
                        )
                        if aligned_changes:
                            had_changes = bool(changes)
                            corrected = aligned_corrected
                            changes.extend(aligned_changes)
                            consensus_path = (
                                f"{consensus_path}+exact_aligned"
                                if had_changes
                                else "paraformer_qwen_exact_aligned"
                            )
                        atomic_corrected, atomic_changes = (
                            atomic_aligned_independent_consensus_rewrite(
                                corrected,
                                para_text,
                                qwen_text,
                                qwen_hallucination_risk=hallucination_risk,
                                global_context_text=primary_global_context,
                            )
                        )
                        if atomic_changes:
                            had_changes = bool(changes)
                            corrected = atomic_corrected
                            changes.extend(atomic_changes)
                            consensus_path = (
                                f"{consensus_path}+atomic_aligned"
                                if had_changes
                                else "paraformer_qwen_atomic_aligned"
                            )
                if (
                    not audit_only
                    and window_index
                    in (qwen_eligible_window_indexes | technical_review_window_indexes)
                ):
                    technical_corrected, technical_changes = technical_consensus_rewrite(
                        corrected,
                        para_text,
                        qwen_text,
                        qwen_hallucination_risk=hallucination_risk,
                        global_context_text=primary_global_context,
                    )
                    if technical_changes:
                        had_changes = bool(changes)
                        corrected = technical_corrected
                        changes.extend(technical_changes)
                        consensus_path = (
                            f"{consensus_path}+technical_consensus"
                            if had_changes
                            else "technical_consensus"
                        )
                if (
                    audit_only
                    or window_index
                    not in (qwen_eligible_window_indexes | technical_review_window_indexes)
                ):
                    consensus_path = "none"
                if len(stats["window_diagnostics"]) < 100:
                    stats["window_diagnostics"].append(
                        {
                            "window": window_index,
                            "start": round(window.start, 3),
                            "end": round(window.end, 3),
                            "qwen_tiles": [tile_index for tile_index, _tile in matching_tiles],
                            "evidence_start": round(evidence_start, 3),
                            "evidence_end": round(evidence_end, 3),
                            "primary_para_similarity": round(primary_para_similarity, 4),
                            "candidate_qwen_similarity": round(candidate_qwen_similarity, 4),
                            "primary_qwen_similarity": round(primary_qwen_similarity, 4),
                            "detector_qwen_similarity": round(detector_qwen_similarity, 4),
                            "qwen_hallucination_risk": hallucination_risk,
                            "audit_only": audit_only,
                            "audit_only_reason": audit_only_reason,
                            "qwen_review_eligible": window_index
                            in (qwen_eligible_window_indexes | technical_review_window_indexes),
                            "technical_review_only": window_index
                            in technical_review_window_indexes,
                            "qwen_skipped_reason": qwen_skip_reason,
                            "auto_replace_allowed": window_index
                            in (qwen_eligible_window_indexes | technical_review_window_indexes),
                            "consensus_path": consensus_path,
                            "candidates": {
                                "primary": primary_text,
                                "primary_redecode": primary_redecode_text,
                                "paraformer": para_text,
                                "qwen": qwen_text,
                            },
                            "confirmed_change_count": len(changes),
                        }
                    )
                if not changes:
                    continue
                candidate = apply_window_text(output, window, corrected)
                if candidate == output:
                    continue
                if timeline_fingerprint(candidate) != before_fingerprint:
                    continue
                output = candidate
                stats["changed_windows"] += 1
                stats["replacement_count"] += len(changes)
                if len(stats["changes"]) < 100:
                    stats["changes"].append(
                        {
                            "window": window_index,
                            "start": round(window.start, 3),
                            "end": round(window.end, 3),
                            "primary_para_similarity": round(primary_para_similarity, 4),
                            "items": changes,
                        }
                    )
        if qwen is not None:
            del qwen
        _release_model_memory()
    except Exception as exc:
        stats["reason"] = f"qwen_failed:{type(exc).__name__}"
        stats["error"] = str(exc)
        return primary_segments, stats

    after_fingerprint = timeline_fingerprint(output)
    stats["timeline_fingerprint_after"] = after_fingerprint
    stats["timeline_preserved"] = before_fingerprint == after_fingerprint
    if not stats["timeline_preserved"]:
        stats["reason"] = "timeline_changed_guard_rejected"
        return primary_segments, stats
    stats["applied"] = bool(stats["replacement_count"])
    if stats["applied"]:
        stats["reason"] = "consensus_changes_applied"
    elif stats["audit_only_window_count"]:
        stats["reason"] = "audit_only_candidates_recorded"
    else:
        stats["reason"] = "no_confirmed_consensus_changes"
    return output, stats
