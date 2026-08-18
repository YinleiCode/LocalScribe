"""Pure helpers for conservative SenseVoice failed-window recovery.

The helpers in this module intentionally use exact normalized evidence only.
They do not perform fuzzy matching or model inference.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Iterable


_VALID_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]")
_SENSEVOICE_TAG_RE = re.compile(r"<\|[^<>]*?\|>")
_XML_TAG_RE = re.compile(r"</?[^<>]{1,40}>")
_EVENT_LABEL_RE = re.compile(
    r"[\[【（(]\s*(?:music|speech|noise|laughter|laugh|applause|cough|breath|"
    r"音乐|语音|噪音|杂音|笑声|鼓掌|咳嗽|呼吸)\s*[\]】）)]",
    re.IGNORECASE,
)
_EDGE_NOISE_RE = re.compile(r"^[\s\W_]+|[\s\W_]+$", re.UNICODE)


def extract_local_body(text: str) -> str:
    """Remove SenseVoice/event labels while preserving readable body text."""
    body = unicodedata.normalize("NFKC", str(text or ""))
    body = _SENSEVOICE_TAG_RE.sub("", body)
    body = _XML_TAG_RE.sub("", body)
    body = _EVENT_LABEL_RE.sub("", body)
    return body.strip()


def normalize_recovery_text(text: str) -> str:
    """NFKC/lowercase text and retain only Chinese, ASCII letters, and digits."""
    body = extract_local_body(text).lower()
    return "".join(ch for ch in body if _VALID_CHAR_RE.fullmatch(ch))


def is_repeated_hallucination(normalized: str) -> bool:
    """Reject exact short-unit repetition commonly produced by failed decoding."""
    text = normalize_recovery_text(normalized)
    if len(text) < 3:
        return False
    for unit_len in range(1, min(12, len(text) // 3) + 1):
        if len(text) % unit_len:
            continue
        repeats = len(text) // unit_len
        if repeats >= 3 and text == text[:unit_len] * repeats:
            return True
    return False


def _window_dict(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        item = dict(raw)
        start = item.get("start")
        end = item.get("end")
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        start, end = raw[0], raw[1]
        item = {"start": start, "end": end, "reason": raw[2] if len(raw) > 2 else "unknown"}
    else:
        return None
    try:
        item["start"] = float(start)
        item["end"] = float(end)
    except (TypeError, ValueError):
        return None
    if item["end"] <= item["start"]:
        return None
    item.setdefault("reason", "unknown")
    return item


def group_failure_windows(
    windows: Iterable[Any],
    *,
    max_gap_s: float = 0.05,
) -> list[dict[str, Any]]:
    """Group failure windows whose wall-clock gap is no greater than 50 ms."""
    cleaned = [item for raw in windows if (item := _window_dict(raw)) is not None]
    cleaned.sort(key=lambda item: (item["start"], item["end"]))
    groups: list[dict[str, Any]] = []
    for item in cleaned:
        if groups and item["start"] - groups[-1]["end"] <= max_gap_s + 1e-9:
            groups[-1]["end"] = max(groups[-1]["end"], item["end"])
            groups[-1]["windows"].append(item)
        else:
            groups.append({"start": item["start"], "end": item["end"], "windows": [item]})
    return groups


def _segment_value(segment: Any, name: str, default: Any) -> Any:
    if isinstance(segment, dict):
        return segment.get(name, default)
    return getattr(segment, name, default)


def local_reference_from_segments(
    segments: Iterable[Any],
    start: float,
    end: float,
    *,
    context_s: float = 15.0,
    max_chars: int = 500,
) -> dict[str, str]:
    """Extract nearby left/right text plus overlapping text for exact lookup."""
    ordered = sorted(
        segments,
        key=lambda segment: (
            float(_segment_value(segment, "start", 0.0) or 0.0),
            float(_segment_value(segment, "end", 0.0) or 0.0),
        ),
    )
    left_parts: list[str] = []
    right_parts: list[str] = []
    overlapping_parts: list[str] = []
    for segment in ordered:
        seg_start = float(_segment_value(segment, "start", 0.0) or 0.0)
        seg_end = float(_segment_value(segment, "end", seg_start) or seg_start)
        text = extract_local_body(str(_segment_value(segment, "text", "") or ""))
        if not text:
            continue
        if seg_end <= start and start - seg_end <= context_s:
            left_parts.append(text)
        elif seg_start >= end and seg_start - end <= context_s:
            right_parts.append(text)
        elif seg_start < end and seg_end > start:
            overlapping_parts.append(text)
    left = "".join(left_parts)[-max_chars:]
    right = "".join(right_parts)[:max_chars]
    overlapping = "".join(overlapping_parts)[:max_chars]
    return {
        "left": left,
        "right": right,
        "overlapping": overlapping,
        "reference": overlapping,
    }


def _longest_suffix_prefix(left: str, right: str, *, limit: int | None = None) -> int:
    upper = min(len(left), len(right), limit if limit is not None else max(len(left), len(right)))
    for size in range(upper, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def _normalized_positions(body: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    positions: list[int] = []
    for index, ch in enumerate(body):
        normalized = unicodedata.normalize("NFKC", ch).lower()
        for normalized_ch in normalized:
            if _VALID_CHAR_RE.fullmatch(normalized_ch):
                normalized_chars.append(normalized_ch)
                positions.append(index)
    return "".join(normalized_chars), positions


def deduplicate_candidate(candidate: str, left: str, right: str) -> dict[str, Any]:
    """Remove exact left-suffix/prefix and candidate-suffix/right-prefix overlap."""
    body = extract_local_body(candidate)
    normalized, positions = _normalized_positions(body)
    left_normalized = normalize_recovery_text(left)
    right_normalized = normalize_recovery_text(right)
    left_overlap = _longest_suffix_prefix(left_normalized, normalized)
    remaining = len(normalized) - left_overlap
    right_overlap = _longest_suffix_prefix(normalized, right_normalized, limit=max(0, remaining))
    residual_start = left_overlap
    residual_end = len(normalized) - right_overlap
    residual_normalized = normalized[residual_start:residual_end] if residual_end > residual_start else ""
    if residual_normalized and positions:
        raw_start = positions[residual_start]
        raw_end = positions[residual_end - 1] + 1
        residual_text = _EDGE_NOISE_RE.sub("", body[raw_start:raw_end])
    else:
        residual_text = ""
    return {
        "body": body,
        "normalized": normalized,
        "left_overlap_chars": left_overlap,
        "right_overlap_chars": right_overlap,
        "residual_text": residual_text,
        "residual_normalized": residual_normalized,
    }


def analyze_recovery_candidate(
    raw: str,
    *,
    framing: str,
    pad_s: float,
    left: str,
    right: str,
    local_reference: str,
    error: str | None = None,
    min_required_chars: int = 2,
    provider_id: str = "sensevoice-primary",
    provider_kind: str = "primary_asr",
    model_id: str = "iic/SenseVoiceSmall",
    model_family: str = "sensevoice",
    hallucination_risk: bool = False,
) -> dict[str, Any]:
    """Build one serializable, exact-evidence recovery attempt."""
    attempt: dict[str, Any] = {
        "framing": framing,
        "pad_s": float(pad_s),
        "raw": str(raw or ""),
        "normalized": "",
        "residual": "",
        "residual_text": "",
        "status": "error" if error else "rejected",
        "min_required_chars": max(2, int(min_required_chars)),
        "provider_id": str(provider_id or ""),
        "provider_kind": str(provider_kind or ""),
        "model_id": str(model_id or ""),
        "model_family": str(model_family or ""),
        "hallucination_risk": bool(hallucination_risk),
    }
    if error:
        attempt["error"] = error
        return attempt

    deduped = deduplicate_candidate(raw, left, right)
    attempt.update({
        "body": deduped["body"],
        "normalized": deduped["normalized"],
        "residual": deduped["residual_normalized"],
        "residual_text": deduped["residual_text"],
        "left_overlap_chars": deduped["left_overlap_chars"],
        "right_overlap_chars": deduped["right_overlap_chars"],
    })
    normalized = deduped["normalized"]
    if not normalized:
        attempt["rejection_reason"] = "empty_candidate"
        return attempt
    required_chars = max(2, int(min_required_chars))
    reference_normalized = normalize_recovery_text(local_reference)
    if normalized in reference_normalized:
        attempt["status"] = "matched_existing"
        attempt["residual"] = ""
        attempt["residual_text"] = ""
        return attempt
    if reference_normalized:
        attempt["rejection_reason"] = "overlapping_reference_unresolved"
        return attempt

    if len(normalized) < required_chars:
        attempt["rejection_reason"] = "single_character" if len(normalized) < 2 else "low_text_density"
        return attempt
    if is_repeated_hallucination(normalized):
        attempt["rejection_reason"] = "repeated_hallucination"
        return attempt
    if hallucination_risk:
        attempt["rejection_reason"] = "provider_hallucination_risk"
        return attempt

    residual = deduped["residual_normalized"]
    if len(residual) < required_chars:
        attempt["rejection_reason"] = "residual_too_short" if len(residual) < 2 else "low_text_density"
        return attempt
    if is_repeated_hallucination(residual):
        attempt["rejection_reason"] = "repeated_hallucination"
        return attempt
    attempt["status"] = "valid"
    return attempt


def _evidence_key(item: dict[str, Any]) -> str:
    return "|".join((
        str(item.get("provider_id") or ""),
        str(item.get("framing") or ""),
        str(item.get("slice_sha256") or ""),
    ))


def _stable_primary_evidence(
    attempts: list[dict[str, Any]],
    *,
    status: str,
    value_field: str,
) -> tuple[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for item in attempts:
        if (
            item.get("status") != status
            or item.get("provider_kind") != "primary_asr"
            or bool(item.get("hallucination_risk"))
        ):
            continue
        value = normalize_recovery_text(str(item.get(value_field) or ""))
        framing = str(item.get("framing") or "")
        if value and framing:
            grouped.setdefault(value, {})[framing] = item
    candidates: list[tuple[str, list[dict[str, Any]]]] = []
    for value, by_framing in grouped.items():
        items = list(by_framing.values())
        slice_hashes = {str(item.get("slice_sha256") or "") for item in items}
        if len(items) >= 2 and "" not in slice_hashes and len(slice_hashes) >= 2:
            candidates.append((value, items))
    if not candidates:
        return "", []
    candidates.sort(key=lambda item: (-len(item[1]), item[0]))
    return candidates[0]


def _independent_evidence(
    attempts: list[dict[str, Any]],
    *,
    status: str,
    value_field: str,
    consensus: str,
) -> list[dict[str, Any]]:
    primary_families = {
        str(item.get("model_family") or "")
        for item in attempts
        if item.get("provider_kind") == "primary_asr"
    }
    evidence: dict[str, dict[str, Any]] = {}
    for item in attempts:
        if item.get("status") != status or item.get("provider_kind") != "independent_asr":
            continue
        provider_id = str(item.get("provider_id") or "")
        model_family = str(item.get("model_family") or "")
        model_id = str(item.get("model_id") or "")
        slice_hash = str(item.get("slice_sha256") or "")
        value = normalize_recovery_text(str(item.get(value_field) or ""))
        if (
            provider_id != "qwen3-independent"
            or model_family != "qwen3_asr"
            or model_id != "mlx-community/Qwen3-ASR-1.7B-8bit"
            or not slice_hash
            or not item.get("model_revision")
            or not item.get("config_sha256")
            or not item.get("weights_manifest_sha256")
            or model_family in primary_families
            or value != consensus
            or bool(item.get("hallucination_risk"))
        ):
            continue
        evidence[provider_id] = item
    return list(evidence.values())


def _single_primary_existing_consensus(
    attempts: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Allow one primary framing only when an independent model confirms existing text.

    This path can only prove that text is already present. It must never authorize insertion.
    """
    for primary in attempts:
        if primary.get("status") != "matched_existing" or primary.get("provider_kind") != "primary_asr":
            continue
        consensus = normalize_recovery_text(str(primary.get("normalized") or ""))
        if not consensus or not primary.get("slice_sha256"):
            continue
        independent = _independent_evidence(
            attempts,
            status="matched_existing",
            value_field="normalized",
            consensus=consensus,
        )
        if independent:
            return consensus, [primary, *independent]
    return "", []


def decide_recovery_attempts(attempts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Require stable primary evidence for insertion; existing text may use cross-model proof."""
    attempts_list = [dict(item) for item in attempts]
    primary_matched, matched_primary_items = _stable_primary_evidence(
        attempts_list, status="matched_existing", value_field="normalized"
    )
    primary_residual, valid_primary_items = _stable_primary_evidence(
        attempts_list, status="valid", value_field="residual"
    )
    primary_status = "matched_existing" if primary_matched else "valid" if primary_residual else ""
    primary_consensus = primary_matched or primary_residual
    primary_items = matched_primary_items or valid_primary_items
    base = {
        "primary_status": primary_status,
        "primary_consensus": primary_consensus,
        "primary_evidence_framings": sorted(str(item.get("framing") or "") for item in primary_items),
    }

    if not primary_matched:
        cross_model_matched, cross_model_items = _single_primary_existing_consensus(attempts_list)
        if cross_model_matched:
            return {
                **base,
                "primary_status": "matched_existing",
                "primary_consensus": cross_model_matched,
                "primary_evidence_framings": sorted({
                    str(item.get("framing") or "")
                    for item in cross_model_items
                    if item.get("provider_kind") == "primary_asr"
                }),
                "decision": "matched_existing",
                "consensus": cross_model_matched,
                "evidence_framings": sorted({str(item.get("framing") or "") for item in cross_model_items}),
                "evidence_providers": sorted({str(item.get("provider_id") or "") for item in cross_model_items}),
                "evidence_models": sorted({str(item.get("model_id") or "") for item in cross_model_items}),
                "evidence_ids": sorted(_evidence_key(item) for item in cross_model_items),
                "inserted_text": "",
            }

    if primary_matched:
        independent = _independent_evidence(
            attempts_list,
            status="matched_existing",
            value_field="normalized",
            consensus=primary_matched,
        )
        if independent:
            evidence_items = matched_primary_items + independent
            return {
                **base,
                "decision": "matched_existing",
                "consensus": primary_matched,
                "evidence_framings": sorted({str(item.get("framing") or "") for item in evidence_items}),
                "evidence_providers": sorted({str(item.get("provider_id") or "") for item in evidence_items}),
                "evidence_models": sorted({str(item.get("model_id") or "") for item in evidence_items}),
                "evidence_ids": sorted(_evidence_key(item) for item in evidence_items),
                "inserted_text": "",
            }

    if primary_residual:
        independent = _independent_evidence(
            attempts_list,
            status="valid",
            value_field="residual",
            consensus=primary_residual,
        )
        if independent:
            evidence_items = valid_primary_items + independent
            ordered_attempt = valid_primary_items[0]
            inserted_text = str(ordered_attempt.get("residual_text") or "").strip()
            if len(normalize_recovery_text(inserted_text)) >= 2:
                return {
                    **base,
                    "decision": "insert_accepted",
                    "consensus": primary_residual,
                    "evidence_framings": sorted({str(item.get("framing") or "") for item in evidence_items}),
                    "evidence_providers": sorted({str(item.get("provider_id") or "") for item in evidence_items}),
                    "evidence_models": sorted({str(item.get("model_id") or "") for item in evidence_items}),
                    "evidence_ids": sorted(_evidence_key(item) for item in evidence_items),
                    "inserted_text": inserted_text,
                }

    if attempts_list and all(item.get("status") == "error" for item in attempts_list):
        decision = "error"
    else:
        decision = "rejected"
    return {
        **base,
        "decision": decision,
        "consensus": "",
        "evidence_framings": [],
        "evidence_providers": [],
        "evidence_models": [],
        "evidence_ids": [],
        "inserted_text": "",
    }


def decide_asymmetric_context_evidence(
    *,
    core_start: float,
    core_end: float,
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
    left_text: str,
    right_text: str,
    left_slice_sha256: str,
    right_slice_sha256: str,
    local_reference: str,
    speech_duration_s: float,
    min_chars_per_s: float = 0.75,
) -> dict[str, Any]:
    left_normalized = normalize_recovery_text(left_text)
    right_normalized = normalize_recovery_text(right_text)
    reference_normalized = normalize_recovery_text(local_reference)
    base = {
        "decision": "rejected",
        "consensus": "",
        "core_start": round(float(core_start), 3),
        "core_end": round(float(core_end), 3),
        "left_start": round(float(left_start), 3),
        "left_end": round(float(left_end), 3),
        "right_start": round(float(right_start), 3),
        "right_end": round(float(right_end), 3),
        "left_normalized": left_normalized,
        "right_normalized": right_normalized,
        "left_slice_sha256": left_slice_sha256,
        "right_slice_sha256": right_slice_sha256,
        "local_reference": local_reference,
        "recognition_source": "asymmetric_context_consensus",
    }
    if core_end <= core_start:
        return {**base, "rejection_reason": "invalid_core"}
    intersection_start = max(left_start, right_start)
    intersection_end = min(left_end, right_end)
    if abs(intersection_start - core_start) > 0.001 or abs(intersection_end - core_end) > 0.001:
        return {**base, "rejection_reason": "framing_intersection_differs_from_core"}
    if not left_slice_sha256 or not right_slice_sha256 or left_slice_sha256 == right_slice_sha256:
        return {**base, "rejection_reason": "non_independent_slices"}
    if len(left_normalized) < 2 or len(right_normalized) < 2:
        return {**base, "rejection_reason": "insufficient_text"}
    if is_repeated_hallucination(left_normalized) or is_repeated_hallucination(right_normalized):
        return {**base, "rejection_reason": "repeated_hallucination"}
    if left_normalized != right_normalized:
        return {**base, "rejection_reason": "asymmetric_text_disagreement"}
    chars_per_s = len(left_normalized) / max(float(speech_duration_s), 0.001)
    if chars_per_s < float(min_chars_per_s):
        return {**base, "rejection_reason": "low_text_density", "chars_per_s": round(chars_per_s, 3)}
    if not reference_normalized or left_normalized not in reference_normalized:
        return {**base, "rejection_reason": "local_reference_mismatch", "chars_per_s": round(chars_per_s, 3)}
    accepted = {
        **base,
        "decision": "matched_existing",
        "consensus": left_normalized,
        "rejection_reason": "",
        "chars_per_s": round(chars_per_s, 3),
    }
    accepted["evidence_sha256"] = hashlib.sha256(
        json.dumps(accepted, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return accepted


def build_anchor_character_ownership(
    *,
    final_text: str,
    anchor_chunks: list[dict[str, Any]],
    strict_windows: list[dict[str, Any]],
    min_equal_run_chars: int = 6,
    max_unique_context_chars: int = 16,
    boundary_guard_s: float = 0.12,
    max_support_s: float = 0.75,
    min_chars_per_s: float = 0.75,
) -> dict[str, Any]:
    final_stream = normalize_recovery_text(final_text)
    anchor_stream_parts: list[str] = []
    anchor_meta: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(anchor_chunks):
        if chunk.get("status") != "recognized":
            continue
        normalized = normalize_recovery_text(str(chunk.get("text") or ""))
        if not normalized:
            continue
        start = float(chunk["start"]); end = float(chunk["end"])
        width = (end - start) / len(normalized)
        for char_index, char in enumerate(normalized):
            anchor_stream_parts.append(char)
            anchor_meta.append({
                "chunk_index": chunk_index,
                "char_index": char_index,
                "support_start": start + char_index * width,
                "support_end": start + (char_index + 1) * width,
            })
    anchor_stream = "".join(anchor_stream_parts)
    matcher = SequenceMatcher(None, final_stream, anchor_stream, autojunk=False)
    eligible: dict[int, dict[str, Any]] = {}
    equal_chars = 0
    for block in matcher.get_matching_blocks():
        if block.size < min_equal_run_chars:
            continue
        block_text = final_stream[block.a:block.a + block.size]
        if any(
            block_text[index:index + min_equal_run_chars] in block_text[index + 1:]
            for index in range(max(0, len(block_text) - min_equal_run_chars + 1))
        ):
            continue
        equal_chars += block.size
        for offset in range(block.size):
            final_index = block.a + offset
            anchor_index = block.b + offset
            unique = False
            for size in range(min_equal_run_chars, max_unique_context_chars + 1):
                left = max(block.a, final_index - size // 2)
                right = min(block.a + block.size, left + size)
                left = max(block.a, right - size)
                key = final_stream[left:right]
                if len(key) >= min_equal_run_chars and final_stream.count(key) == 1 and anchor_stream.count(key) == 1:
                    unique = True
                    break
            if unique:
                eligible[final_index] = {"anchor_index": anchor_index, **anchor_meta[anchor_index]}
    used_final_indices: set[int] = set()
    claims: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for window in sorted(strict_windows, key=lambda item: (float(item["core_start"]), float(item["core_end"]))):
        start=float(window["core_start"]); end=float(window["core_end"])
        speech=float(window.get("speech_duration_s") or end-start)
        candidates=[]
        for final_index, item in eligible.items():
            if final_index in used_final_indices:
                continue
            support_start=float(item["support_start"]); support_end=float(item["support_end"])
            if support_end-support_start > max_support_s:
                continue
            if support_start >= start+boundary_guard_s and support_end <= end-boundary_guard_s:
                candidates.append((final_index,item))
        candidates.sort()
        best=[]; current=[]
        for pair in candidates:
            if current and pair[0] != current[-1][0]+1:
                if len(current)>len(best): best=current
                current=[]
            current.append(pair)
        if len(current)>len(best): best=current
        required=max(2, int(speech*min_chars_per_s + 0.999999))
        if len(best) < required:
            rejected.append({"core_start":start,"core_end":end,"reason":"insufficient_owned_chars","owned_chars":len(best),"required_chars":required})
            continue
        indices=[pair[0] for pair in best]; used_final_indices.update(indices)
        claim={"core_start":start,"core_end":end,"owned_chars":len(indices),"final_char_start":indices[0],"final_char_end":indices[-1]+1,"text":final_stream[indices[0]:indices[-1]+1],"chars_per_s":round(len(indices)/max(speech,0.001),3)}
        claim["evidence_sha256"]=hashlib.sha256(json.dumps(claim,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        claims.append(claim)
    payload={"final_normalized_chars":len(final_stream),"anchor_normalized_chars":len(anchor_stream),"equal_char_ratio":round(equal_chars/max(len(final_stream),1),6),"claims":claims,"rejected":rejected}
    payload["claims_sha256"]=hashlib.sha256(json.dumps(claims,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return payload


def parse_paraformer_native_timestamps(
    items: list[dict[str, Any]],
    *,
    duration_s: float,
) -> dict[str, Any]:
    units_out: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    previous_start_ms = 0
    duration_ms = int(round(float(duration_s) * 1000))
    for item_index, item in enumerate(items):
        raw_text = str(item.get("text") or "").strip()
        timestamps = item.get("timestamp") or item.get("timestamps")
        if not raw_text or not isinstance(timestamps, list):
            rejected.append({"item_index": item_index, "reason": "missing_text_or_timestamps"})
            continue
        raw_units = raw_text.split()
        if len(raw_units) == 1 and len(timestamps) > 1:
            normalized_chars = list(normalize_recovery_text(raw_text))
            if len(normalized_chars) == len(timestamps):
                raw_units = normalized_chars
        if len(raw_units) != len(timestamps):
            rejected.append({"item_index": item_index, "reason": "unit_timestamp_count_mismatch", "units": len(raw_units), "timestamps": len(timestamps)})
            continue
        parsed_item: list[dict[str, Any]] = []
        valid = True
        local_previous_start = previous_start_ms
        for unit_index, (unit, pair) in enumerate(zip(raw_units, timestamps)):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                valid = False; reason = "invalid_timestamp_pair"; break
            start_raw, end_raw = pair
            if isinstance(start_raw, bool) or isinstance(end_raw, bool):
                valid = False; reason = "invalid_timestamp_number"; break
            try:
                start_value=float(start_raw); end_value=float(end_raw)
            except (TypeError, ValueError):
                valid = False; reason = "invalid_timestamp_number"; break
            if not math.isfinite(start_value) or not math.isfinite(end_value):
                valid = False; reason = "non_finite_timestamp"; break
            if not start_value.is_integer() or not end_value.is_integer():
                valid = False; reason = "timestamp_not_integer_ms"; break
            start_ms=int(start_value); end_ms=int(end_value)
            if start_ms < 0 or end_ms <= start_ms or end_ms > duration_ms or start_ms < local_previous_start:
                valid = False; reason = "invalid_or_non_monotonic_timestamp"; break
            normalized=normalize_recovery_text(str(unit))
            parsed_item.append({"native_id":f"item-{item_index}-unit-{unit_index}","item_index":item_index,"unit_index":unit_index,"raw_unit":str(unit),"normalized":normalized,"native_character":len(normalized)==1,"start_ms":start_ms,"end_ms":end_ms})
            local_previous_start=start_ms
        if not valid:
            rejected.append({"item_index": item_index, "reason": reason})
            continue
        units_out.extend(parsed_item); previous_start_ms=local_previous_start
    payload={"units":units_out,"rejected_items":rejected,"native_character_count":sum(1 for item in units_out if item["native_character"]),"coarse_fallback_count":0,"interpolated_char_count":0}
    payload["evidence_sha256"]=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return payload


def build_paraformer_native_ownership(
    *,
    final_text: str,
    native_units: list[dict[str, Any]],
    strict_windows: list[dict[str, Any]],
    min_equal_run_chars: int = 6,
    max_unique_context_chars: int = 16,
    max_interval_ms: int = 750,
    min_chars_per_s: float = 0.75,
) -> dict[str, Any]:
    final_stream=normalize_recovery_text(final_text)
    native_stream=""; native_meta: list[dict[str,Any]]=[]
    for unit in native_units:
        normalized=str(unit.get("normalized") or "")
        for offset,char in enumerate(normalized):
            native_stream+=char; native_meta.append({**unit,"native_character":bool(unit.get("native_character")) and len(normalized)==1,"unit_char_offset":offset})
    matcher=SequenceMatcher(None,final_stream,native_stream,autojunk=False)
    eligible: dict[int,dict[str,Any]]={}; equal_chars=0
    for block in matcher.get_matching_blocks():
        if block.size < min_equal_run_chars: continue
        for offset in range(block.size):
            fi=block.a+offset; ni=block.b+offset; unique=False
            for size in range(min_equal_run_chars,max_unique_context_chars+1):
                left=max(block.a,fi-size//2); right=min(block.a+block.size,left+size); left=max(block.a,right-size); key=final_stream[left:right]
                if len(key)>=min_equal_run_chars and final_stream.count(key)==1 and native_stream.count(key)==1: unique=True; break
            if unique:
                equal_chars+=1; eligible[fi]={"native_index":ni,**native_meta[ni]}
    used_final:set[int]=set(); used_native:set[int]=set(); claims=[]; rejected=[]
    for window_index,window in enumerate(sorted(strict_windows,key=lambda x:(float(x["core_start"]),float(x["core_end"])))):
        start_ms=int(round(float(window["core_start"])*1000)); end_ms=int(round(float(window["core_end"])*1000)); speech=float(window.get("speech_duration_s") or (end_ms-start_ms)/1000)
        candidates=[]
        for fi,meta in eligible.items():
            ni=int(meta["native_index"]); s=int(meta.get("start_ms",-1)); e=int(meta.get("end_ms",-1))
            if fi in used_final or ni in used_native or not meta.get("native_character"): continue
            if e-s<=0 or e-s>max_interval_ms or s<start_ms or e>end_ms: continue
            candidates.append((fi,ni,meta))
        candidates.sort(); runs=[]; current=[]
        for pair in candidates:
            if current and (pair[0]!=current[-1][0]+1 or pair[1]!=current[-1][1]+1): runs.append(current); current=[]
            current.append(pair)
        if current:runs.append(current)
        runs.sort(key=len,reverse=True); best=runs[0] if runs else []; required=max(1,math.ceil(speech*min_chars_per_s))
        if len(best)<required or (len(runs)>1 and len(runs[1])==len(best)):
            rejected.append({"window_index":window_index,"core_start":start_ms/1000,"core_end":end_ms/1000,"reason":"insufficient_or_ambiguous_native_chars","owned_chars":len(best),"required_chars":required}); continue
        final_indices=[x[0] for x in best]; native_indices=[x[1] for x in best]; used_final.update(final_indices); used_native.update(native_indices)
        claim={"window_index":window_index,"core_start":start_ms/1000,"core_end":end_ms/1000,"speech_duration_s":speech,"owned_chars":len(best),"required_chars":required,"chars_per_s":round(len(best)/max(speech,0.001),3),"final_char_start":final_indices[0],"final_char_end":final_indices[-1]+1,"text":final_stream[final_indices[0]:final_indices[-1]+1],"native_ids":[x[2]["native_id"] for x in best],"padding_credit_chars":0,"mapping_kind":"exact_equal_unique_context","timestamp_kind":"paraformer_native_ms_interval"}
        claim["evidence_sha256"]=hashlib.sha256(json.dumps(claim,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest(); claims.append(claim)
    payload={"final_normalized_chars":len(final_stream),"native_normalized_chars":len(native_stream),"equal_char_ratio":round(equal_chars/max(len(final_stream),1),6),"claims":claims,"rejected":rejected,"interpolated_char_count":0,"padding_credit_chars":0}
    payload["claims_sha256"]=hashlib.sha256(json.dumps(claims,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest(); return payload
