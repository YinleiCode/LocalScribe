#!/usr/bin/env python3
"""Run several ASR backends on the same audio and write comparable outputs."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PY_SRC = ROOT / "scribe-py" / "src"
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

from scribe_py.core.asr_quality import build_asr_quality_report, write_asr_quality_reports  # noqa: E402
from scribe_py.core.selector import default_model_id, make_transcriber  # noqa: E402
from scribe_py.core.types import Segment, TranscribeOptions, TranscribeResult  # noqa: E402


def fmt_ts(seconds: float, comma: bool = False) -> str:
    millis = int(round(max(float(seconds), 0.0) * 1000))
    h, rem = divmod(millis, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    sep = "," if comma else "."
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def safe_name(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name).strip()
    return name or "audio"


def parse_model_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--model must be BACKEND=MODEL, got: {value}")
        backend, model = value.split("=", 1)
        backend = backend.strip()
        model = model.strip()
        if not backend or not model:
            raise SystemExit(f"--model must be BACKEND=MODEL, got: {value}")
        overrides[backend] = model
    return overrides


def segment_dict(seg: Segment) -> dict[str, Any]:
    return seg.to_dict()


def write_text(path: Path, result: TranscribeResult) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {Path(result.audio).name}\n")
        f.write(
            f"# backend={result.backend} model={result.model_id} "
            f"duration={result.duration:.1f}s segments={len(result.segments)} "
            f"rtf={result.rtf:.3f}\n\n"
        )
        for seg in result.segments:
            text = seg.text.strip()
            if text:
                f.write(f"[{fmt_ts(seg.start)} - {fmt_ts(seg.end)}] {text}\n")


def write_srt(path: Path, segments: list[Segment]) -> None:
    with path.open("w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            f.write(f"{idx}\n")
            f.write(f"{fmt_ts(seg.start, comma=True)} --> {fmt_ts(seg.end, comma=True)}\n")
            f.write(f"{text}\n\n")
            idx += 1


def write_json(path: Path, result: TranscribeResult) -> None:
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def build_quality_report(result: TranscribeResult) -> dict[str, Any]:
    filter_stats = result.filter_stats or {}
    return build_asr_quality_report(
        result.segments,
        text_normalization=filter_stats.get("text_normalization") or {},
        audio_quality=filter_stats.get("audio_quality") or {},
        audio_preprocessing=filter_stats.get("audio_standardization") or {},
        backend=result.backend,
        model_id=result.model_id,
        duration=result.duration,
        transcribe_seconds=result.transcribe_seconds,
        rtf=result.rtf,
    )


def read_prompt(prompt: str, hotwords_file: str) -> str:
    parts: list[str] = []
    if prompt.strip():
        parts.append(prompt.strip())
    if hotwords_file:
        hotwords_path = Path(hotwords_file).expanduser().resolve()
        if not hotwords_path.exists():
            raise SystemExit(f"hotwords file not found: {hotwords_path}")
        hotwords = hotwords_path.read_text(encoding="utf-8").strip()
        if hotwords:
            parts.append(hotwords)
    return "\n".join(parts)


def progress_logger(path: Path):
    def _log(event: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    return _log


def run_one(
    audio: Path,
    backend: str,
    model_id: str,
    out_dir: Path,
    *,
    transcriber: Any | None = None,
    language: str,
    prompt: str,
    word_timestamps: bool,
    normalizer_profile: str,
) -> dict[str, Any]:
    backend_dir = out_dir / safe_name(backend)
    backend_dir.mkdir(parents=True, exist_ok=True)
    progress_path = backend_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    row: dict[str, Any] = {
        "audio": audio.name,
        "backend": backend,
        "model": model_id,
        "status": "error",
        "segments": 0,
        "chars": 0,
        "duration_s": "",
        "cost_s": "",
        "rtf": "",
        "language": "",
        "risk_level": "",
        "strong_review_count": 0,
        "review_count": 0,
        "punctuation_ratio": "",
        "traditional_count": 0,
        "term_candidate_count": 0,
        "hotword_coverage": "",
        "quality_json": "",
        "out_dir": str(backend_dir),
        "error": "",
    }
    try:
        transcriber = transcriber or make_transcriber(backend)
        options = TranscribeOptions(
            language=language or None,
            model_id=model_id,
            initial_prompt=prompt,
            word_timestamps=word_timestamps,
            normalizer_profile=normalizer_profile or None,
        )
        result = transcriber.transcribe(audio, options, on_progress=progress_logger(progress_path))
        write_text(backend_dir / "transcript.txt", result)
        write_srt(backend_dir / "transcript.srt", result.segments)
        write_json(backend_dir / "result.json", result)
        quality = build_quality_report(result)
        write_asr_quality_reports(backend_dir, "transcript", quality)

        text = "".join(s.text.strip() for s in result.segments)
        review = quality.get("review") or {}
        hotwords = quality.get("hotwords") or {}
        term_consistency = quality.get("term_consistency") or {}
        row.update(
            {
                "status": "ok",
                "segments": len(result.segments),
                "chars": len(text),
                "duration_s": f"{result.duration:.2f}",
                "cost_s": f"{result.transcribe_seconds:.2f}",
                "rtf": f"{result.rtf:.4f}",
                "language": result.language or "",
                "risk_level": quality.get("risk_level", ""),
                "strong_review_count": int(review.get("strong_segment_count") or 0),
                "review_count": int(review.get("segment_count") or 0),
                "punctuation_ratio": f"{float(quality.get('punctuation_ratio') or 0):.4f}",
                "traditional_count": len(quality.get("traditional_char_hits") or []),
                "term_candidate_count": int(term_consistency.get("candidate_count") or 0),
                "hotword_coverage": "" if hotwords.get("coverage") is None else f"{float(hotwords.get('coverage') or 0):.4f}",
                "quality_json": str(backend_dir / "ASR质量检查.json"),
            }
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        row["error"] = message
        (backend_dir / "error.txt").write_text(
            message + "\n\n" + traceback.format_exc(),
            encoding="utf-8",
        )
        (backend_dir / "error.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return row


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except Exception:
        return default


def _risk_rank(value: Any) -> int:
    return {"low": 0, "medium": 1, "unknown": 2, "high": 3}.get(str(value or "unknown"), 2)


def recommendation_score(row: dict[str, Any]) -> tuple[int, int, int, int, float, float]:
    """Conservative sort key for comparing completed backend outputs.

    This is not a correctness proof.  It ranks local risk signals first, then
    uses speed only as a tie-breaker.
    """
    if row.get("status") != "ok":
        return (999, 999, 999, 999, 999.0, 999.0)
    punctuation_penalty = int(round((1.0 - _as_float(row.get("punctuation_ratio"), 0.0)) * 1000))
    return (
        _risk_rank(row.get("risk_level")),
        _as_int(row.get("strong_review_count")),
        _as_int(row.get("review_count")),
        (_as_int(row.get("traditional_count")) * 10) + punctuation_penalty,
        _as_float(row.get("rtf"), 999.0),
        _as_float(row.get("cost_s"), 999.0),
    )


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get("status") == "ok"]
    ranked = sorted(candidates, key=recommendation_score)
    ranked_rows = []
    for rank, row in enumerate(ranked, start=1):
        ranked_rows.append(
            {
                "rank": rank,
                "audio": row.get("audio", ""),
                "backend": row.get("backend", ""),
                "model": row.get("model", ""),
                "risk_level": row.get("risk_level", ""),
                "strong_review_count": _as_int(row.get("strong_review_count")),
                "review_count": _as_int(row.get("review_count")),
                "punctuation_ratio": _as_float(row.get("punctuation_ratio")),
                "traditional_count": _as_int(row.get("traditional_count")),
                "rtf": _as_float(row.get("rtf"), 0.0),
                "score_key": list(recommendation_score(row)),
                "out_dir": row.get("out_dir", ""),
            }
        )
    return ranked_rows


def build_recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audio_names = sorted({str(row.get("audio") or "") for row in rows})
    per_audio: list[dict[str, Any]] = []
    for audio in audio_names:
        audio_rows = [row for row in rows if str(row.get("audio") or "") == audio]
        ranked_rows = _rank_rows(audio_rows)
        best = ranked_rows[0] if ranked_rows else None
        per_audio.append(
            {
                "audio": audio,
                "recommended_backend": best.get("backend") if best else "",
                "recommended_model": best.get("model") if best else "",
                "ranked": ranked_rows,
                "failed": [
                    {
                        "audio": row.get("audio", ""),
                        "backend": row.get("backend", ""),
                        "model": row.get("model", ""),
                        "error": row.get("error", ""),
                    }
                    for row in audio_rows
                    if row.get("status") != "ok"
                ],
            }
        )

    first = per_audio[0] if len(per_audio) == 1 else {}
    return {
        "mode": "asr_backend_recommendation",
        "recommended_backend": first.get("recommended_backend", ""),
        "recommended_model": first.get("recommended_model", ""),
        "reason": (
            "优先选择本地风险等级更低、强疑点更少、标点/简体更稳定的结果；速度只作为并列时的兜底条件。"
            if any(item.get("recommended_backend") for item in per_audio)
            else "没有成功完成的后端结果，无法推荐。"
        ),
        "recommendations": per_audio,
    }


def write_recommendation(out_root: Path, rows: list[dict[str, Any]]) -> None:
    recommendation = build_recommendation(rows)
    (out_root / "recommendation.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# ASR 后端推荐\n\n",
        f"- 推荐后端: {recommendation.get('recommended_backend') or '见逐录音推荐'}\n",
        f"- 推荐模型: `{recommendation.get('recommended_model') or '见逐录音推荐'}`\n",
        f"- 推荐口径: {recommendation.get('reason')}\n\n",
        "| 排名 | 录音 | 后端 | 模型 | 风险 | 强疑点 | 本地疑点 | 标点率 | 繁体数 | RTF | 输出目录 |\n",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---|\n",
    ]
    failed: list[dict[str, Any]] = []
    for group in recommendation.get("recommendations") or []:
        for item in group.get("ranked") or []:
            lines.append(
                "| {rank} | {audio} | {backend} | `{model}` | {risk_level} | {strong_review_count} | "
                "{review_count} | {punctuation_ratio:.1%} | {traditional_count} | {rtf:.4f} | `{out_dir}` |\n".format(
                    **item
                )
            )
        failed.extend(group.get("failed") or [])
    if failed:
        lines.extend(["\n## 失败后端\n\n", "| 录音 | 后端 | 模型 | 错误 |\n", "|---|---|---|---|\n"])
        for item in failed:
            error = str(item.get("error") or "").replace("|", "\\|")
            lines.append(f"| {item.get('audio', '')} | {item.get('backend', '')} | `{item.get('model', '')}` | {error} |\n")
    (out_root / "recommendation.md").write_text("".join(lines), encoding="utf-8")


def write_summary(out_root: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "audio",
        "backend",
        "model",
        "status",
        "segments",
        "chars",
        "duration_s",
        "cost_s",
        "rtf",
        "language",
        "risk_level",
        "strong_review_count",
        "review_count",
        "punctuation_ratio",
        "traditional_count",
        "term_candidate_count",
        "hotword_coverage",
        "quality_json",
        "out_dir",
        "error",
    ]
    with (out_root / "summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    header = "| 录音 | 后端 | 模型 | 状态 | 段数 | 字数 | 风险 | 强疑点 | 本地疑点 | 标点率 | 繁体数 | 耗时(s) | RTF | 错误 |\n"
    sep = "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|\n"
    lines = [header, sep]
    for row in rows:
        err = str(row.get("error") or "").replace("|", "\\|")
        lines.append(
            "| {audio} | {backend} | `{model}` | {status} | {segments} | {chars} | "
            "{risk_level} | {strong_review_count} | {review_count} | {punctuation_ratio} | "
            "{traditional_count} | {cost_s} | {rtf} | {error} |\n".format(
                **{**row, "error": err}
            )
        )
    (out_root / "summary.md").write_text("".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "ASR 横向对比:同一录音分别跑 mlx / ct2 / funasr / sensevoice / qwen3, "
            "输出统一 json/txt/srt 和中文汇总表。"
        )
    )
    ap.add_argument("audios", nargs="+", help="输入音频文件")
    ap.add_argument(
        "--backends",
        nargs="+",
        default=["mlx", "ct2", "sensevoice"],
        help="要对比的后端,可选 mlx ct2 sensevoice funasr qwen3",
    )
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        help="模型覆盖,格式 BACKEND=MODEL,可重复。例: --model funasr=paraformer-zh",
    )
    ap.add_argument("--out", default="", help="输出目录,默认 output/asr_compare_<时间>")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--prompt", default="", help="ASR initial_prompt / FunASR hotword")
    ap.add_argument("--hotwords-file", default="", help="热词文件,每行一个或多个词")
    ap.add_argument("--word-timestamps", action="store_true")
    ap.add_argument(
        "--normalizer-profile",
        default="",
        help="录音专用已确认纠错 profile。默认空=只做通用清理和疑点标注；标准录音3可显式传 standard3",
    )
    args = ap.parse_args()

    audios = [Path(p).expanduser().resolve() for p in args.audios]
    missing = [str(p) for p in audios if not p.exists()]
    if missing:
        raise SystemExit("audio not found:\n" + "\n".join(missing))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out).expanduser().resolve() if args.out else ROOT / "output" / f"asr_compare_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    model_overrides = parse_model_overrides(args.model)
    prompt = read_prompt(args.prompt, args.hotwords_file)
    rows: list[dict[str, Any]] = []
    transcriber_cache: dict[tuple[str, str], Any] = {}

    print(f"[asr-benchmark] out={out_root}")
    for audio in audios:
        audio_dir = out_root / safe_name(audio.stem)
        audio_dir.mkdir(parents=True, exist_ok=True)
        print(f"[audio] {audio}")
        for backend in args.backends:
            model_id = model_overrides.get(backend) or default_model_id(backend)
            print(f"  [run] backend={backend} model={model_id}")
            cache_key = (backend, model_id)
            transcriber = transcriber_cache.get(cache_key)
            if transcriber is None:
                transcriber = make_transcriber(backend)
                transcriber_cache[cache_key] = transcriber
            row = run_one(
                audio,
                backend,
                model_id,
                audio_dir,
                transcriber=transcriber,
                language=args.language,
                prompt=prompt,
                word_timestamps=args.word_timestamps,
                normalizer_profile=args.normalizer_profile,
            )
            rows.append(row)
            if row["status"] != "ok":
                transcriber_cache.pop(cache_key, None)
            if row["status"] == "ok":
                print(
                    f"    ok: {row['segments']} segs, {row['chars']} chars, "
                    f"rtf={row['rtf']}"
                )
            else:
                print(f"    error: {row['error']}")

    write_summary(out_root, rows)
    write_recommendation(out_root, rows)
    print(f"[summary] {out_root / 'summary.md'}")
    print(f"[recommendation] {out_root / 'recommendation.md'}")
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
