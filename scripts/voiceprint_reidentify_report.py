#!/usr/bin/env python3
"""Compare transcript speaker labels before and after voiceprint reidentification.

The report is intentionally independent from the diarization backend.  It reads
two transcript JSON files, preserves their text/timestamps, and only describes
speaker-label changes.  The assignment helper is also standalone so the main
pipeline can later reuse the same globally unique speaker-to-profile mapping.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


RISK_RANK = {"low": 0, "medium": 1, "high": 2, "unknown": 1}
RISK_ZH = {"low": "低", "medium": "中", "high": "高", "unknown": "未知"}


def global_one_to_one_assignment(
    scores: Mapping[str, Mapping[str, float]],
    *,
    min_score: float | None = None,
) -> dict[str, str]:
    """Return the maximum-score globally unique speaker-to-profile mapping.

    Each speaker and profile can occur at most once.  A speaker may remain
    unassigned, which is preferable to accepting a score below ``min_score``.
    The exact dynamic-programming solver is practical for meeting-sized inputs
    (normally 2-8 profiles) and has no scipy dependency.
    """

    speakers = sorted(str(speaker) for speaker in scores)
    profiles = sorted({str(profile) for row in scores.values() for profile in row})
    if not speakers or not profiles:
        return {}

    threshold = -math.inf if min_score is None else float(min_score)
    profile_index = {profile: idx for idx, profile in enumerate(profiles)}
    # mask -> (total score, tuple of (speaker, profile)); an empty assignment is
    # always valid so low-confidence rows can remain unknown.
    states: dict[int, tuple[float, tuple[tuple[str, str], ...]]] = {0: (0.0, ())}

    for speaker in speakers:
        next_states = dict(states)
        row = scores.get(speaker) or {}
        for mask, (total, pairs) in states.items():
            for profile in profiles:
                raw_score = row.get(profile)
                if raw_score is None:
                    continue
                score = float(raw_score)
                if not math.isfinite(score) or score < threshold:
                    continue
                bit = 1 << profile_index[profile]
                if mask & bit:
                    continue
                candidate = (total + score, pairs + ((speaker, profile),))
                current = next_states.get(mask | bit)
                if current is None or _assignment_is_better(candidate, current):
                    next_states[mask | bit] = candidate
        states = next_states

    best: tuple[float, tuple[tuple[str, str], ...]] | None = None
    for candidate in states.values():
        if best is None or _assignment_is_better(candidate, best):
            best = candidate
    assert best is not None
    return dict(best[1])


def _assignment_is_better(
    candidate: tuple[float, tuple[tuple[str, str], ...]],
    current: tuple[float, tuple[tuple[str, str], ...]],
) -> bool:
    if not math.isclose(candidate[0], current[0], rel_tol=0.0, abs_tol=1e-12):
        return candidate[0] > current[0]
    if len(candidate[1]) != len(current[1]):
        return len(candidate[1]) > len(current[1])
    return candidate[1] < current[1]


def _segments(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("segments") or []
    return [row for row in rows if isinstance(row, dict)]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _speaker(segment: Mapping[str, Any]) -> str:
    return str(segment.get("speaker") or "未标注")


def _distribution(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    total_seconds = 0.0
    for segment in segments:
        speaker = _speaker(segment)
        duration = max(0.0, _safe_float(segment.get("end")) - _safe_float(segment.get("start")))
        text = str(segment.get("text") or "").strip()
        row = totals.setdefault(speaker, {"speaker": speaker, "segments": 0, "seconds": 0.0, "chars": 0})
        row["segments"] += 1
        row["seconds"] += duration
        row["chars"] += len(text)
        total_seconds += duration
    result = []
    for speaker in sorted(totals):
        row = dict(totals[speaker])
        row["seconds"] = round(float(row["seconds"]), 3)
        row["share"] = round(float(row["seconds"]) / total_seconds, 4) if total_seconds else 0.0
        result.append(row)
    return result


def _paired_segments(
    original: list[dict[str, Any]],
    reidentified: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    return [(idx, original[idx], reidentified[idx]) for idx in range(min(len(original), len(reidentified)))]


def _changed_rows(
    original: list[dict[str, Any]],
    reidentified: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for idx, before, after in _paired_segments(original, reidentified):
        before_speaker = _speaker(before)
        after_speaker = _speaker(after)
        if before_speaker == after_speaker:
            continue
        rows.append({
            "index": idx,
            "start": _safe_float(after.get("start"), _safe_float(before.get("start"))),
            "end": _safe_float(after.get("end"), _safe_float(before.get("end"))),
            "before_speaker": before_speaker,
            "after_speaker": after_speaker,
            "score": after.get("speaker_voiceprint_score"),
            "anchor": str(after.get("speaker_voiceprint_anchor") or ""),
            "explicit_reidentified": bool(after.get("speaker_voiceprint_reidentified")),
            "text": str(after.get("text") or before.get("text") or "").strip(),
        })
    return rows


def _review_rows(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for idx, segment in enumerate(segments):
        if not (segment.get("speaker_voiceprint_review") or segment.get("speaker_assignment_review")):
            continue
        rows.append({
            "index": idx,
            "start": _safe_float(segment.get("start")),
            "end": _safe_float(segment.get("end")),
            "speaker": _speaker(segment),
            "candidate": str(segment.get("speaker_voiceprint_anchor") or ""),
            "score": segment.get("speaker_voiceprint_score"),
            "reason": str(segment.get("speaker_review_reason") or ""),
            "text": str(segment.get("text") or "").strip(),
        })
    return rows


def _voiceprint_section(data: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("voiceprint_reidentify", "speaker_voiceprint", "reidentify"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return data


def _profile_quality(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    section = _voiceprint_section(data)
    candidates = (
        section.get("profiles"),
        data.get("voiceprint_profiles"),
        data.get("speaker_voiceprint_profiles"),
    )
    profiles = next((value for value in candidates if isinstance(value, list)), [])
    rows = []
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        quality = profile.get("quality") if isinstance(profile.get("quality"), dict) else {}
        rows.append({
            "name": str(profile.get("name") or profile.get("speaker") or "未命名"),
            "anchor_count": int(_safe_float(profile.get("anchor_count"))),
            "sample_seconds": round(_safe_float(profile.get("sample_seconds")), 3),
            "enrollment_ready": profile.get("enrollment_ready"),
            "enrollment_reasons": [str(value) for value in (profile.get("enrollment_reasons") or [])],
            "median_similarity": quality.get("median_similarity"),
            "min_similarity": quality.get("min_similarity"),
            "vector_count": quality.get("vector_count"),
        })
    return rows


def _explicit_diarization_risk(data: Mapping[str, Any]) -> str | None:
    section = _voiceprint_section(data)
    for candidate in (
        section.get("stats"),
        data.get("diarization_stats"),
        data.get("speaker_stats"),
    ):
        if not isinstance(candidate, dict):
            continue
        value = str(candidate.get("risk_level") or "").lower()
        if value in RISK_RANK:
            return value
    return None


def _risk_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    segments = _segments(data)
    reviews = _review_rows(segments)
    unknown = sum(1 for segment in segments if _speaker(segment) in {"未标注", "UNKNOWN", "待定"})
    profiles = _profile_quality(data)
    rejected_profiles = sum(1 for profile in profiles if profile.get("enrollment_ready") is False)
    total = len(segments)
    review_rate = len(reviews) / total if total else 0.0
    unknown_rate = unknown / total if total else 0.0
    explicit = _explicit_diarization_risk(data)
    if explicit:
        level = explicit
        source = "diarization_stats"
    elif review_rate >= 0.20 or unknown_rate >= 0.10 or rejected_profiles:
        level = "high"
        source = "calculated"
    elif review_rate >= 0.05 or unknown:
        level = "medium"
        source = "calculated"
    else:
        level = "low"
        source = "calculated"
    return {
        "level": level,
        "source": source,
        "review_segments": len(reviews),
        "review_rate": round(review_rate, 4),
        "unknown_segments": unknown,
        "unknown_rate": round(unknown_rate, 4),
        "rejected_profiles": rejected_profiles,
    }


def _risk_change(before: dict[str, Any], after: dict[str, Any]) -> str:
    before_rank = RISK_RANK.get(str(before.get("level")), 1)
    after_rank = RISK_RANK.get(str(after.get("level")), 1)
    if after_rank < before_rank:
        return "降低"
    if after_rank > before_rank:
        return "升高"
    before_rate = float(before.get("review_rate") or 0.0)
    after_rate = float(after.get("review_rate") or 0.0)
    if after_rate < before_rate:
        return "同级但待确认减少"
    if after_rate > before_rate:
        return "同级但待确认增加"
    return "不变"


def build_report(original: Mapping[str, Any], reidentified: Mapping[str, Any]) -> dict[str, Any]:
    original_segments = _segments(original)
    reidentified_segments = _segments(reidentified)
    before_risk = _risk_summary(original)
    after_risk = _risk_summary(reidentified)
    changed = _changed_rows(original_segments, reidentified_segments)
    reviews = _review_rows(reidentified_segments)
    changed_pairs = Counter((row["before_speaker"], row["after_speaker"]) for row in changed)
    return {
        "summary": {
            "original_segment_count": len(original_segments),
            "reidentified_segment_count": len(reidentified_segments),
            "segment_count_equal": len(original_segments) == len(reidentified_segments),
            "changed_segment_count": len(changed),
            "review_segment_count": len(reviews),
            "risk_change": _risk_change(before_risk, after_risk),
        },
        "distribution": {
            "before": _distribution(original_segments),
            "after": _distribution(reidentified_segments),
        },
        "changed_segments": changed,
        "change_pairs": [
            {"before_speaker": before, "after_speaker": after, "segments": count}
            for (before, after), count in sorted(changed_pairs.items())
        ],
        "review_segments": reviews,
        "profile_quality": _profile_quality(reidentified),
        "risk": {"before": before_risk, "after": after_risk, "change": _risk_change(before_risk, after_risk)},
    }


def _fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"


def _md_escape(value: Any) -> str:
    return str(value if value is not None else "-").replace("|", "/").replace("\n", " ")


def _format_score(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    risk = report["risk"]
    lines = [
        "# 声纹重识别前后对比报告",
        "",
        "## 总览",
        "",
        f"- 原始/重识别后段数: {summary['original_segment_count']} / {summary['reidentified_segment_count']}",
        f"- 改派段: {summary['changed_segment_count']}",
        f"- 待确认段: {summary['review_segment_count']}",
        f"- 分人风险: {RISK_ZH[risk['before']['level']]} -> {RISK_ZH[risk['after']['level']]}（{risk['change']}）",
    ]
    if not summary["segment_count_equal"]:
        lines.append("- 警告: 前后段数不同，报告只按相同序号范围比较。")

    lines.extend(["", "## 每人分布", "", "| 阶段 | 说话人 | 段数 | 时长(秒) | 时长占比 | 字数 |", "|---|---|---:|---:|---:|---:|"])
    for stage, label in (("before", "重识别前"), ("after", "重识别后")):
        for row in report["distribution"][stage]:
            lines.append(
                f"| {label} | {_md_escape(row['speaker'])} | {row['segments']} | {row['seconds']:.3f} | "
                f"{row['share']:.2%} | {row['chars']} |"
            )

    lines.extend(["", "## 改派段", "", "| 序号 | 时间 | 原说话人 | 新说话人 | 分数 | 锚点 | 文本 |", "|---:|---|---|---|---:|---|---|"])
    for row in report["changed_segments"]:
        lines.append(
            f"| {row['index']} | {_fmt_time(row['start'])} - {_fmt_time(row['end'])} | "
            f"{_md_escape(row['before_speaker'])} | {_md_escape(row['after_speaker'])} | "
            f"{_format_score(row['score'])} | {_md_escape(row['anchor'])} | {_md_escape(row['text'])} |"
        )
    if not report["changed_segments"]:
        lines.append("| - | - | - | - | - | - | 无改派 |")

    lines.extend(["", "## 待确认段", "", "| 序号 | 时间 | 当前说话人 | 候选身份 | 分数 | 原因 | 文本 |", "|---:|---|---|---|---:|---|---|"])
    for row in report["review_segments"]:
        lines.append(
            f"| {row['index']} | {_fmt_time(row['start'])} - {_fmt_time(row['end'])} | "
            f"{_md_escape(row['speaker'])} | {_md_escape(row['candidate'])} | {_format_score(row['score'])} | "
            f"{_md_escape(row['reason'])} | {_md_escape(row['text'])} |"
        )
    if not report["review_segments"]:
        lines.append("| - | - | - | - | - | 无 | 无待确认段 |")

    lines.extend(["", "## 声纹锚点质量", "", "| 身份 | 锚点数 | 有效语音(秒) | 可注册 | 中位相似度 | 最低相似度 | 质量说明 |", "|---|---:|---:|---|---:|---:|---|"])
    for row in report["profile_quality"]:
        ready = "是" if row["enrollment_ready"] is True else ("否" if row["enrollment_ready"] is False else "未知")
        lines.append(
            f"| {_md_escape(row['name'])} | {row['anchor_count']} | {row['sample_seconds']:.3f} | {ready} | "
            f"{_format_score(row['median_similarity'])} | {_format_score(row['min_similarity'])} | "
            f"{_md_escape(', '.join(row['enrollment_reasons']) or '通过')} |"
        )
    if not report["profile_quality"]:
        lines.append("| - | 0 | 0 | 未知 | - | - | 结果中没有声纹档案质量数据 |")

    lines.extend([
        "",
        "## 风险变化",
        "",
        "| 阶段 | 风险 | 待确认段 | 待确认率 | 未知段 | 未通过注册质量闸 | 风险来源 |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for key, label in (("before", "重识别前"), ("after", "重识别后")):
        row = risk[key]
        lines.append(
            f"| {label} | {RISK_ZH[row['level']]} | {row['review_segments']} | {row['review_rate']:.2%} | "
            f"{row['unknown_segments']} | {row['rejected_profiles']} | {_md_escape(row['source'])} |"
        )
    lines.append(f"\n结论: 分人风险{risk['change']}。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tsv(path: Path, report: Mapping[str, Any]) -> None:
    fields = [
        "记录类型", "阶段", "序号/身份", "开始秒", "结束秒", "原说话人", "新说话人/候选",
        "段数", "时长秒", "占比", "字数", "分数", "可注册", "质量/原因", "文本",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for stage, label in (("before", "重识别前"), ("after", "重识别后")):
            for row in report["distribution"][stage]:
                writer.writerow({"记录类型": "每人分布", "阶段": label, "序号/身份": row["speaker"], "段数": row["segments"], "时长秒": row["seconds"], "占比": row["share"], "字数": row["chars"]})
        for row in report["changed_segments"]:
            writer.writerow({"记录类型": "改派段", "阶段": "重识别后", "序号/身份": row["index"], "开始秒": row["start"], "结束秒": row["end"], "原说话人": row["before_speaker"], "新说话人/候选": row["after_speaker"], "分数": row["score"], "文本": row["text"]})
        for row in report["review_segments"]:
            writer.writerow({"记录类型": "待确认段", "阶段": "重识别后", "序号/身份": row["index"], "开始秒": row["start"], "结束秒": row["end"], "原说话人": row["speaker"], "新说话人/候选": row["candidate"], "分数": row["score"], "质量/原因": row["reason"], "文本": row["text"]})
        for row in report["profile_quality"]:
            writer.writerow({"记录类型": "锚点质量", "阶段": "重识别后", "序号/身份": row["name"], "段数": row["anchor_count"], "时长秒": row["sample_seconds"], "分数": row["median_similarity"], "可注册": row["enrollment_ready"], "质量/原因": ",".join(row["enrollment_reasons"]) or "通过"})
        for key, label in (("before", "重识别前"), ("after", "重识别后")):
            row = report["risk"][key]
            writer.writerow({"记录类型": "风险", "阶段": label, "序号/身份": RISK_ZH[row["level"]], "段数": row["review_segments"], "占比": row["review_rate"], "质量/原因": row["source"]})


def main() -> int:
    parser = argparse.ArgumentParser(description="生成声纹重识别前后中文对比报告")
    parser.add_argument("--original", required=True, type=Path, help="重识别前转录 JSON")
    parser.add_argument("--reidentified", required=True, type=Path, help="重识别后转录 JSON")
    parser.add_argument("--out-dir", type=Path, default=None, help="输出目录，默认使用重识别后文件目录")
    parser.add_argument("--prefix", default="voiceprint_reidentify_report", help="输出文件名前缀")
    args = parser.parse_args()

    original_path = args.original.expanduser().resolve()
    reidentified_path = args.reidentified.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else reidentified_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    original = json.loads(original_path.read_text(encoding="utf-8"))
    reidentified = json.loads(reidentified_path.read_text(encoding="utf-8"))
    report = build_report(original, reidentified)
    report["inputs"] = {"original": str(original_path), "reidentified": str(reidentified_path)}

    json_path = out_dir / f"{args.prefix}.json"
    md_path = out_dir / f"{args.prefix}.md"
    tsv_path = out_dir / f"{args.prefix}.tsv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, report)
    write_tsv(tsv_path, report)
    print(json.dumps({
        "ok": True,
        "json": str(json_path),
        "markdown": str(md_path),
        "tsv": str(tsv_path),
        "changed_segments": report["summary"]["changed_segment_count"],
        "review_segments": report["summary"]["review_segment_count"],
        "risk_change": report["summary"]["risk_change"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
