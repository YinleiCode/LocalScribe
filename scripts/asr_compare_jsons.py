#!/usr/bin/env python3
"""Compare existing ASR JSON outputs without rerunning transcription.

This is intended for ASR regression work: give it several transcript JSON files
for the same recording and it writes a Chinese summary table plus optional
phrase checks. It does not judge correctness without ground truth; it makes
differences visible and marks whether a version used a recording-specific
normalizer profile.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_PUNCT_RE = re.compile(r"[，。！？；：、,.!?;:]")
_TRAD_CHARS = set("聖誕節當們將會現這個嗎為聽講認況裡讓與對說辦過還應該點樣實問題發後師愛氣團禱導衝憐憫處響協調緒數標準錄")


@dataclass
class Case:
    label: str
    path: Path
    data: dict[str, Any]
    segments: list[dict[str, Any]]
    text: str


@dataclass
class Check:
    label: str
    start: float | None
    end: float | None
    expected: list[str]
    suspicious: list[str]


def _parse_case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"--case must be LABEL=PATH, got: {value}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"--case label is empty: {value}")
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"case file not found: {p}")
    return label, p


def _parse_check(value: str) -> Check:
    if "=" not in value:
        raise SystemExit(f"--check must be LABEL[@START-END]=expected1,expected2::bad1,bad2, got: {value}")
    left, right = value.split("=", 1)
    label = left.strip()
    start = end = None
    if "@" in label:
        label, window = label.rsplit("@", 1)
        if "-" not in window:
            raise SystemExit(f"check time window must be START-END seconds, got: {window}")
        raw_start, raw_end = window.split("-", 1)
        start = float(raw_start)
        end = float(raw_end)
    expected_raw, suspicious_raw = (right.split("::", 1) + [""])[:2] if "::" in right else (right, "")
    expected = [x.strip() for x in expected_raw.split(",") if x.strip()]
    suspicious = [x.strip() for x in suspicious_raw.split(",") if x.strip()]
    return Check(label=label.strip(), start=start, end=end, expected=expected, suspicious=suspicious)


def _load_case(value: str) -> Case:
    label, path = _parse_case(value)
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = list(data.get("segments") or [])
    text = "\n".join(str(seg.get("text") or "") for seg in segments)
    return Case(label=label, path=path, data=data, segments=segments, text=text)


def _profile(case: Case) -> str:
    stats = (case.data.get("filter_stats") or {}).get("text_normalization") or {}
    return str(stats.get("profile") or "通用/无专用 profile")


def _review_count(case: Case) -> int:
    quality = case.data.get("asr_quality") or {}
    review = quality.get("review") or {}
    if "segment_count" in review:
        return int(review.get("segment_count") or 0)
    stats = (case.data.get("filter_stats") or {}).get("text_normalization") or {}
    return int(stats.get("asr_review_segment_count") or 0)


def _punctuation_ratio(case: Case) -> float:
    quality = case.data.get("asr_quality") or {}
    if "punctuation_ratio" in quality:
        return float(quality.get("punctuation_ratio") or 0)
    if not case.segments:
        return 0.0
    return sum(1 for seg in case.segments if _PUNCT_RE.search(str(seg.get("text") or ""))) / len(case.segments)


def _speaker_count(case: Case) -> int:
    speakers = {str(seg.get("speaker")) for seg in case.segments if seg.get("speaker")}
    return len(speakers)


def _text_for_check(case: Case, check: Check) -> str:
    if check.start is None or check.end is None:
        return case.text
    parts: list[str] = []
    for seg in case.segments:
        try:
            start = float(seg.get("start") or 0)
            end = float(seg.get("end") or start)
        except Exception:
            continue
        if end >= check.start and start <= check.end:
            parts.append(str(seg.get("text") or ""))
    return "\n".join(parts)


def _snippet(text: str, terms: list[str]) -> str:
    compact = text.replace("\n", " ")
    hit_at = None
    for term in terms:
        pos = compact.find(term)
        if pos >= 0 and (hit_at is None or pos < hit_at):
            hit_at = pos
    if hit_at is None:
        return compact[:80]
    start = max(0, hit_at - 28)
    end = min(len(compact), hit_at + 72)
    return compact[start:end]


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _case_rows(cases: list[Case]) -> list[dict[str, Any]]:
    baseline = cases[0] if cases else None
    rows: list[dict[str, Any]] = []
    for case in cases:
        cjk_chars = len(_CJK_RE.findall(case.text))
        rows.append(
            {
                "版本": case.label,
                "JSON": str(case.path),
                "后端": case.data.get("backend", ""),
                "模型": case.data.get("model_id", ""),
                "profile": _profile(case),
                "段数": len(case.segments),
                "中文字数": cjk_chars,
                "标点覆盖率": f"{_punctuation_ratio(case):.1%}",
                "繁体字数": sum(1 for ch in case.text if ch in _TRAD_CHARS),
                "疑点段数": _review_count(case),
                "说话人列数": _speaker_count(case),
                "音频时长秒": f"{float(case.data.get('duration') or 0):.1f}",
                "转录耗时秒": f"{float(case.data.get('transcribe_seconds') or 0):.1f}",
                "RTF": f"{float(case.data.get('rtf') or 0):.3f}",
                "与第一版相似度": "-" if baseline is case else f"{SequenceMatcher(None, baseline.text, case.text).ratio():.4f}",
            }
        )
    return rows


def _render_markdown(cases: list[Case], checks: list[Check], rows: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        "# ASR 转文字对比报告\n\n",
        "说明: 本报告只比较已有转录 JSON, 不重新转录; 如果 profile 不是空, 表示该版本使用了录音专用纠错规则, 不能直接视为通用 ASR 能力。\n\n",
        "## 版本概览\n\n",
        "| 版本 | 后端 | 模型 | profile | 段数 | 中文字数 | 标点覆盖率 | 繁体字数 | 疑点段数 | 说话人列数 | 转录耗时秒 | RTF | 与第一版相似度 |\n",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in rows:
        lines.append(
            "| {版本} | {后端} | `{模型}` | {profile} | {段数} | {中文字数} | {标点覆盖率} | {繁体字数} | {疑点段数} | {说话人列数} | {转录耗时秒} | {RTF} | {与第一版相似度} |\n".format(
                **{k: str(v).replace("|", "\\|") for k, v in row.items()}
            )
        )

    if checks:
        lines.extend(
            [
                "\n## 关键片段检查\n\n",
                "| 片段 | 版本 | 正向词命中 | 疑似错词命中 | 摘录 |\n",
                "|---|---|---:|---:|---|\n",
            ]
        )
        for check in checks:
            for case in cases:
                text = _text_for_check(case, check)
                expected_hits = [term for term in check.expected if term in text]
                suspicious_hits = [term for term in check.suspicious if term in text]
                snippet_terms = expected_hits or suspicious_hits or check.expected or check.suspicious
                snippet = _snippet(text, snippet_terms).replace("|", "\\|")
                label = check.label
                if check.start is not None and check.end is not None:
                    label = f"{label} ({check.start:.1f}-{check.end:.1f}s)"
                lines.append(
                    f"| {label} | {case.label} | {len(expected_hits)}/{len(check.expected)} | "
                    f"{len(suspicious_hits)}/{len(check.suspicious)} | {snippet} |\n"
                )

    lines.extend(["\n## JSON 路径\n\n"])
    for case in cases:
        lines.append(f"- {case.label}: `{case.path}`\n")
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="比较多个已有 ASR JSON 输出并生成中文报告")
    parser.add_argument("--case", action="append", required=True, help="版本名=转录JSON路径; 可重复")
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        help="关键片段检查,格式: 名称[@开始秒-结束秒]=正向词1,正向词2::疑似错词1,疑似错词2",
    )
    parser.add_argument("--out", required=True, help="输出 markdown 路径")
    args = parser.parse_args()

    cases = [_load_case(value) for value in args.case]
    checks = [_parse_check(value) for value in args.check]
    rows = _case_rows(cases)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_markdown(cases, checks, rows), encoding="utf-8")
    _write_tsv(out_path.with_suffix(".tsv"), rows)
    print(json.dumps({"ok": True, "markdown": str(out_path), "tsv": str(out_path.with_suffix(".tsv"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
