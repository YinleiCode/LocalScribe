#!/usr/bin/env python3
"""Lock LocalScribe ASR output against known-good transcript baselines.

This script is intentionally read-only. It compares transcript JSON files
against frozen text hashes and key ASR settings so diarization/UI work cannot
silently regress transcription.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASELINES: dict[str, dict[str, Any]] = {
    "标准录音 3": {
        "aliases": ["录音3", "录音 3", "standard3", "std3"],
        "segments": 478,
        "chars": 13616,
        "md5": "ec271c4d06f12110be1fb32d0af2046e",
        "sha256": "003720e14e5a12bf1f139d15b4d9ea6cff31674ffd866da48a02adb1d31a578f",
        "backend": "sensevoice",
        "model_id": "iic/SenseVoiceSmall",
        "preprocess_mode": "adaptive",
        "applied_filters": ["downmix_mono", "resample_16k", "pcm_s16le"],
        "timing_align": True,
    },
    "标准录音 10": {
        "aliases": ["录音10", "录音 10", "standard10", "std10"],
        "segments": 140,
        "chars": 3391,
        "md5": "5a817eed85da7c472fe9b54450091a42",
        "sha256": "db7805f0be1a855531577d316119d07ccbc05bb3386a7ae5c229ef97578c44f2",
        "backend": "sensevoice",
        "model_id": "iic/SenseVoiceSmall",
        "preprocess_mode": "adaptive",
        "applied_filters": ["downmix_mono", "resample_16k", "pcm_s16le", "loudness_normalization"],
        "timing_align": True,
    },
}


@dataclass
class Case:
    label: str
    canonical: str
    path: Path
    data: dict[str, Any]
    compact_text: str
    summary: dict[str, Any]
    failures: list[str]


def _compact_text(segments: list[dict[str, Any]]) -> str:
    text = "".join(str(segment.get("text") or "") for segment in segments)
    return re.sub(r"\s+", "", text)


def _baseline_alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for canonical, baseline in DEFAULT_BASELINES.items():
        aliases[canonical.lower()] = canonical
        for alias in baseline.get("aliases") or []:
            aliases[str(alias).lower()] = canonical
    return aliases


def _canonical_label(label: str) -> str:
    cleaned = label.strip()
    aliases = _baseline_alias_map()
    return aliases.get(cleaned.lower(), cleaned)


def _parse_case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"--case must be LABEL=PATH, got: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise SystemExit(f"--case label is empty: {value}")
    if not path.exists():
        raise SystemExit(f"--case file not found: {path}")
    return label, path


def _default_case_values() -> list[str]:
    root = Path.home() / "Library" / "Application Support" / "LocalScribe" / "transcripts"
    def resolve_case(canonical: str) -> Path:
        preferred = root / canonical / f"{canonical}.json"
        if preferred.is_file():
            return preferred
        candidates = [
            path
            for path in root.glob(f"{canonical}*/{canonical}.json")
            if path.is_file()
        ]
        if candidates:
            return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        return preferred

    return [
        f"标准录音 3={resolve_case('标准录音 3')}",
        f"标准录音 10={resolve_case('标准录音 10')}",
    ]


def _summarize(data: dict[str, Any], compact: str) -> dict[str, Any]:
    segments = list(data.get("segments") or [])
    filter_stats = data.get("filter_stats") or {}
    audio = filter_stats.get("audio_standardization") or {}
    settings = filter_stats.get("settings") or {}
    text_norm = filter_stats.get("text_normalization") or {}
    first_mention = text_norm.get("first_mention_phonetic_consistency") or {}
    return {
        "segments": len(segments),
        "chars": len(compact),
        "md5": hashlib.md5(compact.encode("utf-8")).hexdigest(),
        "sha256": hashlib.sha256(compact.encode("utf-8")).hexdigest(),
        "backend": data.get("backend") or "",
        "model_id": data.get("model_id") or "",
        "duration": float(data.get("duration") or data.get("duration_s") or 0.0),
        "transcribe_seconds": float(data.get("transcribe_seconds") or 0.0),
        "rtf": float(data.get("rtf") or 0.0),
        "preprocess_mode": audio.get("mode") or "",
        "applied_filters": list(audio.get("applied_filters") or []),
        # The standardized PCM is the exact acoustic input seen by ASR.  Keep
        # it in the baseline so a different ffmpeg/resampler cannot masquerade
        # as a model or text-normalization regression.
        "standardized_audio_sha256": str(audio.get("standardized_sha256") or ""),
        "timing_align": settings.get("sensevoice_timing_align"),
        "timing_mode": filter_stats.get("timing_mode") or "",
        "lexical_rewrites_enabled": text_norm.get("lexical_rewrites_enabled"),
        "funasr_inference_seed": settings.get("funasr_inference_seed"),
        "textnorm_segments_changed": text_norm.get("segments_changed"),
        "textnorm_safe_replacements": text_norm.get("safe_replacements"),
        "textnorm_first_mention_replacements": first_mention.get("replacement_count"),
    }


def _check_summary(canonical: str, summary: dict[str, Any], strict_filters: bool) -> list[str]:
    baseline = DEFAULT_BASELINES.get(canonical)
    if not baseline:
        return [f"unknown_baseline:{canonical}"]

    failures: list[str] = []
    for key in ["segments", "chars", "md5", "sha256", "backend", "model_id", "preprocess_mode"]:
        if summary.get(key) != baseline.get(key):
            failures.append(f"{key}: expected {baseline.get(key)!r}, got {summary.get(key)!r}")
    if strict_filters and summary.get("applied_filters") != baseline.get("applied_filters"):
        failures.append(
            f"applied_filters: expected {baseline.get('applied_filters')!r}, "
            f"got {summary.get('applied_filters')!r}"
        )
    expected_audio_sha = str(baseline.get("standardized_audio_sha256") or "")
    if expected_audio_sha and summary.get("standardized_audio_sha256") != expected_audio_sha:
        failures.append(
            "standardized_audio_sha256: "
            f"expected {expected_audio_sha!r}, got {summary.get('standardized_audio_sha256')!r}"
        )
    if baseline.get("timing_align") is not None and summary.get("timing_align") != baseline.get("timing_align"):
        failures.append(f"timing_align: expected {baseline.get('timing_align')!r}, got {summary.get('timing_align')!r}")
    for key in ["timing_mode", "lexical_rewrites_enabled", "funasr_inference_seed", "textnorm_safe_replacements"]:
        if key in baseline and summary.get(key) != baseline.get(key):
            failures.append(f"{key}: expected {baseline.get(key)!r}, got {summary.get(key)!r}")
    return failures


def _load_case(value: str, strict_filters: bool) -> Case:
    label, path = _parse_case(value)
    canonical = _canonical_label(label)
    data = json.loads(path.read_text(encoding="utf-8"))
    compact = _compact_text(list(data.get("segments") or []))
    summary = _summarize(data, compact)
    failures = _check_summary(canonical, summary, strict_filters)
    return Case(
        label=label,
        canonical=canonical,
        path=path,
        data=data,
        compact_text=compact,
        summary=summary,
        failures=failures,
    )


def _rows(cases: list[Case]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        s = case.summary
        rows.append({
            "录音": case.canonical,
            "状态": "PASS" if not case.failures else "FAIL",
            "JSON": str(case.path),
            "段数": s["segments"],
            "字数": s["chars"],
            "MD5": s["md5"],
            "后端": s["backend"],
            "模型": s["model_id"],
            "预处理": s["preprocess_mode"],
            "滤镜": ",".join(s["applied_filters"]),
            "标准化音频SHA256": s["standardized_audio_sha256"],
            "精准时间轴": s["timing_align"],
            "转录耗时秒": f"{s['transcribe_seconds']:.2f}",
            "RTF": f"{s['rtf']:.3f}",
            "失败原因": "；".join(case.failures),
        })
    return rows


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ASR 转录基线锁\n",
        "\n",
        "说明: 本报告只比较已有转录 JSON, 不重新转录, 不修改文件。只要状态为 FAIL, 后续分人/打包工作应停止。\n",
        "\n",
    ]
    headers = ["录音", "状态", "段数", "字数", "MD5", "预处理", "滤镜", "标准化音频SHA256", "转录耗时秒", "RTF", "失败原因"]
    lines.append("| " + " | ".join(headers) + " |\n")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|\n")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")).replace("|", "/") for h in headers) + " |\n")
    lines.append("\n## JSON 路径\n\n")
    for row in rows:
        lines.append(f"- {row['录音']}: `{row['JSON']}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="检查录音 3/10 的转录结果是否仍等于冻结基线")
    parser.add_argument("--case", action="append", default=[], help="LABEL=transcript.json; 不传则检查本机最新标准录音 3/10")
    parser.add_argument("--baseline-file", type=Path, default=None, help="可选版本化基线 JSON；不传时使用脚本内历史基线")
    parser.add_argument("--out-dir", type=Path, default=Path("output/asr_baseline_lock_latest"))
    parser.add_argument("--no-strict-filters", action="store_true", help="只锁文本 hash，不强制滤镜列表完全一致")
    args = parser.parse_args()

    if args.baseline_file:
        baseline_path = args.baseline_file.expanduser().resolve()
        payload = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
        baselines = payload.get("baselines") if isinstance(payload, dict) else None
        if not isinstance(baselines, dict) or not baselines:
            raise SystemExit(f"baseline file must contain a non-empty 'baselines' object: {baseline_path}")
        DEFAULT_BASELINES.clear()
        DEFAULT_BASELINES.update(baselines)

    case_values = args.case or _default_case_values()
    cases = [_load_case(value, strict_filters=not args.no_strict_filters) for value in case_values]
    rows = _rows(cases)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "asr_baseline_lock.md"
    tsv_path = out_dir / "asr_baseline_lock.tsv"
    json_path = out_dir / "asr_baseline_lock.json"
    _write_markdown(md_path, rows)
    _write_tsv(tsv_path, rows)
    ok = all(not case.failures for case in cases)
    json_path.write_text(json.dumps({
        "ok": ok,
        "rows": rows,
        "baselines": DEFAULT_BASELINES,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "ok": ok,
        "markdown": str(md_path),
        "tsv": str(tsv_path),
        "json": str(json_path),
        "checked": [case.canonical for case in cases],
    }, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
