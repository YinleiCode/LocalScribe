#!/usr/bin/env python3
"""Generate a generic multi-recording ASR regression Markdown report.

This script reads completed transcript JSON files and optional gold annotation
files. It does not rerun ASR or touch frontend/core ASR code.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from asr_gold_score import edit_distance, normalize_for_cer  # noqa: E402


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_PUNCT_RE = re.compile(r"[，。！？；：、,.!?;:]")
_TRAD_CHARS = set("聖誕節當們將會現這個嗎為聽講認況裡讓與對說辦過還應該點樣實問題發後師愛氣團禱導衝憐憫處響協調緒數標準錄")


@dataclass
class GoldScore:
    filled_rows: int = 0
    ref_chars: int = 0
    char_edits: int = 0
    ref_words: int = 0
    word_edits: int = 0

    @property
    def cer(self) -> float | None:
        return self.char_edits / self.ref_chars if self.ref_chars else None

    @property
    def wer(self) -> float | None:
        return self.word_edits / self.ref_words if self.ref_words else None


@dataclass
class Case:
    label: str
    path: Path
    data: dict[str, Any]
    quality: dict[str, Any]
    segments: list[dict[str, Any]]
    text: str
    gold_path: Path | None
    score: GoldScore | None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _md(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def _parse_labeled_path(value: str, option: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"{option} must be LABEL=PATH, got: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise SystemExit(f"{option} label is empty: {value}")
    if not path.exists():
        raise SystemExit(f"{option} file not found: {path}")
    return label, path


def _load_quality(transcript_path: Path, transcript_data: dict[str, Any]) -> dict[str, Any]:
    sidecar = transcript_path.parent / "ASR质量检查.json"
    if sidecar.exists():
        try:
            data = _read_json(sidecar)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    quality = transcript_data.get("asr_quality")
    return quality if isinstance(quality, dict) else {}


def _segments(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [seg for seg in data.get("segments") or [] if isinstance(seg, dict)]


def _text(segments: list[dict[str, Any]]) -> str:
    return "\n".join(str(seg.get("text") or "") for seg in segments)


def _profile(data: dict[str, Any]) -> str:
    stats = (data.get("filter_stats") or {}).get("text_normalization") or {}
    profile = stats.get("profile") or data.get("normalizer_profile") or ""
    return str(profile or "").strip()


def _review_count(data: dict[str, Any], quality: dict[str, Any]) -> int:
    review = quality.get("review") or {}
    if "segment_count" in review:
        return int(review.get("segment_count") or 0)
    stats = (data.get("filter_stats") or {}).get("text_normalization") or {}
    return int(stats.get("asr_review_segment_count") or 0)


def _strong_review_count(data: dict[str, Any], quality: dict[str, Any]) -> int:
    review = quality.get("review") or {}
    if "strong_segment_count" in review:
        return int(review.get("strong_segment_count") or 0)
    return _review_count(data, quality)


def _term_consistency_count(quality: dict[str, Any]) -> int:
    term_consistency = quality.get("term_consistency") or {}
    return int(term_consistency.get("candidate_count") or 0)


def _punctuation_ratio(segments: list[dict[str, Any]], quality: dict[str, Any]) -> float:
    if "punctuation_ratio" in quality:
        return float(quality.get("punctuation_ratio") or 0)
    if not segments:
        return 0.0
    return sum(1 for seg in segments if _PUNCT_RE.search(str(seg.get("text") or ""))) / len(segments)


def _traditional_count(text: str, quality: dict[str, Any]) -> int:
    hits = quality.get("traditional_char_hits")
    if isinstance(hits, list):
        return len(hits)
    return sum(1 for ch in text if ch in _TRAD_CHARS)


def _hotword_missing(quality: dict[str, Any]) -> list[str]:
    hotwords = quality.get("hotwords") or {}
    missing = hotwords.get("missing_terms") or []
    return [str(term) for term in missing if str(term).strip()]


def _load_gold_rows(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("items", "rows", "segments"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"unsupported gold annotation shape: {path}")


def _word_tokens(text: str) -> list[str]:
    normalized = normalize_for_cer(text, ignore_punctuation=True, ignore_whitespace=False).strip()
    if not normalized:
        return []
    if re.search(r"\s", normalized):
        return [token for token in re.split(r"\s+", normalized) if token]

    tokens: list[str] = []
    current = ""
    for ch in normalized:
        if _CJK_RE.fullmatch(ch):
            if current:
                tokens.append(current)
                current = ""
            tokens.append(ch)
        elif ch.strip():
            current += ch.lower()
    if current:
        tokens.append(current)
    return tokens


def _sequence_edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (0 if ref_token == hyp_token else 1)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def _hypothesis_for_gold_row(row: dict[str, Any], segments: list[dict[str, Any]]) -> str:
    try:
        index = int(row.get("index"))
    except Exception:
        index = -1
    if 0 <= index < len(segments):
        return str(segments[index].get("text") or "")
    return str(row.get("current_text") or row.get("text") or "")


def score_gold(segments: list[dict[str, Any]], gold_path: Path) -> GoldScore:
    score = GoldScore()
    for row in _load_gold_rows(gold_path):
        reference = str(row.get("correct_text") or row.get("gold_text") or row.get("reference_text") or "").strip()
        if not reference:
            continue
        hypothesis = _hypothesis_for_gold_row(row, segments)
        ref_chars = normalize_for_cer(reference)
        hyp_chars = normalize_for_cer(hypothesis)
        ref_words = _word_tokens(reference)
        hyp_words = _word_tokens(hypothesis)

        score.filled_rows += 1
        score.ref_chars += len(ref_chars)
        score.char_edits += edit_distance(ref_chars, hyp_chars)
        score.ref_words += len(ref_words)
        score.word_edits += _sequence_edit_distance(ref_words, hyp_words)
    return score


def _find_gold(label: str, transcript_path: Path, explicit: dict[str, Path], gold_dir: Path | None) -> Path | None:
    if label in explicit:
        return explicit[label]
    if gold_dir is None:
        return None
    candidates = [
        gold_dir / f"{label}.json",
        gold_dir / f"{transcript_path.stem}.json",
        gold_dir / f"{label}_gold.json",
        gold_dir / f"{transcript_path.stem}_gold.json",
        gold_dir / f"{label}.gold.json",
        gold_dir / f"{transcript_path.stem}.gold.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_cases(case_args: list[str], gold_args: list[str], gold_dir: Path | None) -> list[Case]:
    explicit_gold = dict(_parse_labeled_path(value, "--gold") for value in gold_args)
    cases: list[Case] = []
    for value in case_args:
        label, path = _parse_labeled_path(value, "--case")
        data = _read_json(path)
        if not isinstance(data, dict):
            raise SystemExit(f"transcript JSON must be an object: {path}")
        segments = _segments(data)
        text = _text(segments)
        quality = _load_quality(path, data)
        gold_path = _find_gold(label, path, explicit_gold, gold_dir)
        score = score_gold(segments, gold_path) if gold_path else None
        cases.append(
            Case(
                label=label,
                path=path,
                data=data,
                quality=quality,
                segments=segments,
                text=text,
                gold_path=gold_path,
                score=score,
            )
        )
    return cases


def _summary_row(case: Case) -> dict[str, str]:
    profile = _profile(case.data)
    missing_hotwords = _hotword_missing(case.quality)
    score = case.score
    return {
        "录音": case.label,
        "段数": str(len(case.segments)),
        "字数": str(len(_CJK_RE.findall(case.text))),
        "本地疑点数": str(_review_count(case.data, case.quality)),
        "强疑点数": str(_strong_review_count(case.data, case.quality)),
        "实体候选": str(_term_consistency_count(case.quality)),
        "繁体数": str(_traditional_count(case.text, case.quality)),
        "标点率": _pct(_punctuation_ratio(case.segments, case.quality)),
        "热词缺失": "、".join(missing_hotwords) if missing_hotwords else "无",
        "CER": _pct(score.cer, 2) if score else "-",
        "WER": _pct(score.wer, 2) if score else "-",
        "profile": profile or "通用",
        "结论提醒": "使用专用 profile，不能视为纯通用能力" if profile else "可按通用能力观察",
    }


def render_markdown(cases: list[Case]) -> str:
    profiled = [case for case in cases if _profile(case.data)]
    scored_count = sum(1 for case in cases if case.score)
    total_segments = sum(len(case.segments) for case in cases)
    total_chars = sum(len(_CJK_RE.findall(case.text)) for case in cases)
    total_reviews = sum(_review_count(case.data, case.quality) for case in cases)
    total_strong_reviews = sum(_strong_review_count(case.data, case.quality) for case in cases)
    total_term_candidates = sum(_term_consistency_count(case.quality) for case in cases)

    lines: list[str] = [
        "# ASR 通用回归评测汇总\n\n",
        "本报告汇总多个已完成转写 JSON，不重新转写；用于观察通用 ASR 在真实录音上的稳定回归表现。\n\n",
        "## 总览\n\n",
        "| 指标 | 数值 |\n",
        "|---|---:|\n",
        f"| 录音数 | {len(cases)} |\n",
        f"| 已接入 gold 的录音数 | {scored_count} |\n",
        f"| 总段数 | {total_segments} |\n",
        f"| 总字数 | {total_chars} |\n",
        f"| 总本地疑点数 | {total_reviews} |\n",
        f"| 总强疑点数 | {total_strong_reviews} |\n",
        f"| 总同音/近音实体一致性候选数 | {total_term_candidates} |\n",
        "\n",
    ]
    if profiled:
        names = "、".join(case.label for case in profiled)
        lines.extend(
            [
                "## profile 提醒\n\n",
                f"{names} 使用了非空 profile。这类结果包含录音专用纠错规则，不能直接视为纯通用 ASR 能力。\n\n",
            ]
        )

    lines.extend(
        [
            "## 逐条录音指标\n\n",
            "| 录音 | 段数 | 字数 | 本地疑点数 | 强疑点数 | 实体候选 | 繁体数 | 标点率 | 热词缺失 | CER | WER | profile | 结论提醒 |\n",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|\n",
        ]
    )
    for case in cases:
        row = _summary_row(case)
        lines.append(
            "| {录音} | {段数} | {字数} | {本地疑点数} | {强疑点数} | {实体候选} | {繁体数} | {标点率} | {热词缺失} | {CER} | {WER} | {profile} | {结论提醒} |\n".format(
                **{key: _md(value) for key, value in row.items()}
            )
        )

    lines.extend(["\n## gold 评分口径\n\n"])
    if scored_count:
        lines.append(
            "CER/WER 仅对已提供 gold 标注且 `correct_text` 非空的行计算；优先按 gold 行里的 `index` 读取对应转写段文本，找不到时使用 gold 行内 `current_text`。\n\n"
        )
        lines.extend(
            [
                "| 录音 | gold 文件 | 有效标注行 | 参考字数 | 字符编辑数 | 参考词数 | 词编辑数 |\n",
                "|---|---|---:|---:|---:|---:|---:|\n",
            ]
        )
        for case in cases:
            if not case.score:
                continue
            lines.append(
                f"| {_md(case.label)} | `{_md(case.gold_path)}` | {case.score.filled_rows} | {case.score.ref_chars} | {case.score.char_edits} | {case.score.ref_words} | {case.score.word_edits} |\n"
            )
    else:
        lines.append("本次未提供 gold 标注，因此只输出本地质量指标，不计算 CER/WER。\n")

    lines.extend(["\n## 输入 JSON\n\n"])
    for case in cases:
        lines.append(f"- {_md(case.label)}: `{case.path}`\n")
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成多录音 ASR 通用回归 Markdown 汇总")
    parser.add_argument("--case", action="append", required=True, help="录音名=转写JSON路径; 可重复")
    parser.add_argument("--gold", action="append", default=[], help="录音名=gold JSON路径; 可重复")
    parser.add_argument("--gold-dir", type=Path, default=None, help="gold JSON 目录; 按录音名或转写文件名自动匹配")
    parser.add_argument("--out", type=Path, default=Path("ASR通用回归评测汇总.md"), help="输出 Markdown 路径")
    args = parser.parse_args(argv)

    gold_dir = args.gold_dir.expanduser().resolve() if args.gold_dir else None
    if gold_dir and not gold_dir.exists():
        raise SystemExit(f"gold dir not found: {gold_dir}")
    cases = load_cases(args.case, args.gold, gold_dir)
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(cases), encoding="utf-8")
    print(json.dumps({"ok": True, "markdown": str(out), "cases": len(cases)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
