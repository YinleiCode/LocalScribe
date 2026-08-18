#!/usr/bin/env python3
"""Preflight ASR audio preprocessing choices on short clips.

This is a conservative local workflow for high-risk recordings: sample a few
short windows, run the same ASR backend with several preprocessing modes, then
recommend the mode with lower local risk before a full transcription.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PY_SRC = ROOT / "scribe-py" / "src"
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

from scribe_py.core.asr_quality import build_asr_quality_report, write_asr_quality_reports  # noqa: E402
from scribe_py.core.audio import analyze_audio_quality_for_asr, probe_audio  # noqa: E402
from scribe_py.core.selector import default_model_id, make_transcriber  # noqa: E402
from scribe_py.core.types import TranscribeOptions, TranscribeResult  # noqa: E402


RISK_RANK = {"low": 0, "medium": 1, "unknown": 2, "high": 3}
DEFAULT_MODES = ["standard", "adaptive", "enhance"]


def safe_name(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]+", "_", name).strip() or "audio"


def fmt_ts(seconds: float) -> str:
    ms = int(round(max(float(seconds), 0.0) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ffmpeg() -> str:
    value = shutil.which("ffmpeg")
    if not value:
        raise SystemExit("ffmpeg not found in PATH")
    return value


def extract_clip(audio: Path, out: Path, start: float, duration: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            f"{max(start, 0):.3f}",
            "-t",
            f"{max(duration, 0.1):.3f}",
            "-i",
            str(audio),
            "-vn",
            "-acodec",
            "copy",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not out.exists() or out.stat().st_size <= 0:
        out.unlink(missing_ok=True)
        proc = subprocess.run(
            [
                ffmpeg(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-ss",
                f"{max(start, 0):.3f}",
                "-t",
                f"{max(duration, 0.1):.3f}",
                "-i",
                str(audio),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(out.with_suffix(".wav")),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        out = out.with_suffix(".wav")
    if proc.returncode != 0 or not out.exists() or out.stat().st_size <= 44:
        raise RuntimeError((proc.stderr or "ffmpeg produced no clip").strip())


def choose_windows(duration: float, *, clip_seconds: float, max_clips: int) -> list[tuple[float, float]]:
    if duration <= 0:
        return [(0.0, clip_seconds)]
    if duration <= clip_seconds:
        return [(0.0, duration)]
    max_clips = max(1, max_clips)
    if max_clips == 1:
        return [(max((duration - clip_seconds) / 2, 0.0), clip_seconds)]
    anchors = [0.12, 0.5, 0.82]
    if max_clips == 2:
        anchors = [0.2, 0.72]
    anchors = anchors[:max_clips]
    windows: list[tuple[float, float]] = []
    seen: set[int] = set()
    for anchor in anchors:
        start = min(max(duration * anchor - (clip_seconds / 2), 0.0), max(duration - clip_seconds, 0.0))
        key = int(round(start))
        if key in seen:
            continue
        seen.add(key)
        windows.append((start, min(clip_seconds, duration - start)))
    return windows or [(0.0, min(clip_seconds, duration))]


def text_chars(result: TranscribeResult) -> int:
    return sum(len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", seg.text or "")) for seg in result.segments)


def run_sample(
    clip: Path,
    *,
    backend: str,
    model_id: str,
    mode: str,
    language: str,
) -> tuple[dict[str, Any], TranscribeResult | None]:
    started = time.time()
    row: dict[str, Any] = {
        "clip": str(clip),
        "mode": mode,
        "backend": backend,
        "model": model_id,
        "status": "error",
        "risk_level": "unknown",
        "review_count": 0,
        "strong_review_count": 0,
        "term_candidate_count": 0,
        "chars": 0,
        "segments": 0,
        "punctuation_ratio": 0.0,
        "rtf": 999.0,
        "cost_seconds": 0.0,
        "error": "",
    }
    try:
        transcriber = make_transcriber(backend)
        result = transcriber.transcribe(
            clip,
            TranscribeOptions(
                language=language or "zh",
                model_id=model_id,
                normalizer_profile=None,
                audio_preprocess=mode,
            ),
        )
        quality = build_asr_quality_report(
            result.segments,
            text_normalization=(result.filter_stats or {}).get("text_normalization") or {},
            audio_quality=(result.filter_stats or {}).get("audio_quality") or {},
            audio_preprocessing=(result.filter_stats or {}).get("audio_standardization") or {},
            backend=result.backend,
            model_id=result.model_id,
            duration=result.duration,
            transcribe_seconds=result.transcribe_seconds,
            rtf=result.rtf,
        )
        review = quality.get("review") or {}
        terms = quality.get("term_consistency") or {}
        row.update(
            {
                "status": "ok",
                "risk_level": quality.get("risk_level", "unknown"),
                "review_count": int(review.get("segment_count") or 0),
                "strong_review_count": int(review.get("strong_segment_count") or 0),
                "term_candidate_count": int(terms.get("candidate_count") or 0),
                "chars": text_chars(result),
                "segments": len(result.segments),
                "punctuation_ratio": float(quality.get("punctuation_ratio") or 0.0),
                "rtf": float(result.rtf or 0.0),
                "quality": quality,
            }
        )
        return row, result
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row, None
    finally:
        row["cost_seconds"] = time.time() - started


def score_row(row: dict[str, Any]) -> tuple[int, int, int, int, int, float]:
    if row.get("status") != "ok":
        return (999, 999, 999, 999, 999, 999.0)
    chars = int(row.get("chars") or 0)
    low_text_penalty = 1 if chars < 10 else 0
    punctuation_penalty = int(round((1.0 - float(row.get("punctuation_ratio") or 0.0)) * 100))
    return (
        RISK_RANK.get(str(row.get("risk_level") or "unknown"), 2),
        int(row.get("strong_review_count") or 0),
        int(row.get("review_count") or 0),
        int(row.get("term_candidate_count") or 0),
        low_text_penalty + punctuation_penalty,
        float(row.get("rtf") or 999.0),
    )


def aggregate_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return {
            "mode": rows[0].get("mode") if rows else "",
            "status": "error",
            "score_key": [999, 999, 999, 999, 999, 999.0],
            "error_count": len(rows),
        }
    risk = max(RISK_RANK.get(str(row.get("risk_level") or "unknown"), 2) for row in ok_rows)
    total_strong = sum(int(row.get("strong_review_count") or 0) for row in ok_rows)
    total_review = sum(int(row.get("review_count") or 0) for row in ok_rows)
    total_terms = sum(int(row.get("term_candidate_count") or 0) for row in ok_rows)
    total_chars = sum(int(row.get("chars") or 0) for row in ok_rows)
    avg_rtf = sum(float(row.get("rtf") or 0.0) for row in ok_rows) / len(ok_rows)
    return {
        "mode": ok_rows[0].get("mode"),
        "status": "ok",
        "risk_level": next((name for name, rank in RISK_RANK.items() if rank == risk), "unknown"),
        "strong_review_count": total_strong,
        "review_count": total_review,
        "term_candidate_count": total_terms,
        "chars": total_chars,
        "avg_rtf": avg_rtf,
        "ok_count": len(ok_rows),
        "error_count": len(rows) - len(ok_rows),
        "score_key": [risk, total_strong, total_review, total_terms, 0 if total_chars >= 10 else 1, avg_rtf],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ASR 预检择优报告\n\n",
        f"- 源音频: `{payload['audio']}`\n",
        f"- 后端/模型: {payload['backend']} / `{payload['model']}`\n",
        f"- 音频时长: {float(payload['duration_s']):.1f}s\n",
        f"- 音频质量: {payload['audio_quality'].get('risk_level', 'unknown')}\n",
        f"- 音频质量原因: {'；'.join(payload['audio_quality'].get('risk_reasons') or []) or '-'}\n",
        f"- 推荐预处理: `{payload['recommended_mode']}`\n\n",
        "## 模式汇总\n\n",
        "| 预处理 | 风险 | 强疑点 | 疑点 | 实体候选 | 字数 | 平均RTF | 成功/失败 | 排序键 |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for row in payload["mode_summary"]:
        lines.append(
            f"| {row.get('mode')} | {row.get('risk_level', '-')} | "
            f"{row.get('strong_review_count', 0)} | {row.get('review_count', 0)} | "
            f"{row.get('term_candidate_count', 0)} | {row.get('chars', 0)} | "
            f"{float(row.get('avg_rtf') or 0):.3f} | {row.get('ok_count', 0)}/{row.get('error_count', 0)} | "
            f"`{row.get('score_key')}` |\n"
        )
    lines.extend(
        [
            "\n## 样本明细\n\n",
            "| Clip | 预处理 | 状态 | 风险 | 强疑点 | 疑点 | 实体候选 | 字数 | RTF | 错误 |\n",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|\n",
        ]
    )
    for row in payload["sample_rows"]:
        error = str(row.get("error") or "").replace("|", "\\|")
        lines.append(
            f"| `{Path(row.get('clip', '')).name}` | {row.get('mode')} | {row.get('status')} | "
            f"{row.get('risk_level')} | {row.get('strong_review_count')} | {row.get('review_count')} | "
            f"{row.get('term_candidate_count')} | {row.get('chars')} | {float(row.get('rtf') or 0):.3f} | {error} |\n"
        )
    command = payload.get("recommended_command") or ""
    if command:
        lines.extend(["\n## 推荐全量命令\n\n", "```bash\n", command, "\n```\n"])
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="高风险录音 ASR 预检: 小样本比较预处理模式并推荐全量参数")
    parser.add_argument("audio", help="源音频")
    parser.add_argument("--backend", default="sensevoice", choices=["sensevoice", "funasr", "mlx", "ct2"])
    parser.add_argument("--model", default="")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help="逗号分隔: off,standard,adaptive,enhance")
    parser.add_argument("--clip-seconds", type=float, default=35.0)
    parser.add_argument("--max-clips", type=int, default=3)
    parser.add_argument("--out", default="", help="输出目录")
    args = parser.parse_args()

    audio = Path(args.audio).expanduser().resolve()
    if not audio.exists():
        raise SystemExit(f"audio not found: {audio}")
    model_id = args.model or default_model_id(args.backend)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out).expanduser().resolve() if args.out else ROOT / "output" / f"asr_preflight_{safe_name(audio.stem)}_{timestamp}"
    clips_dir = out_dir / "clips"
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    probe = probe_audio(audio)
    duration = float(probe.get("duration") or 0.0)
    audio_quality = analyze_audio_quality_for_asr(audio)
    windows = choose_windows(duration, clip_seconds=args.clip_seconds, max_clips=args.max_clips)
    modes = [part.strip() for part in args.modes.replace("，", ",").split(",") if part.strip()]
    if not modes:
        modes = DEFAULT_MODES

    clip_paths: list[dict[str, Any]] = []
    for idx, (start, clip_duration) in enumerate(windows, start=1):
        clip = clips_dir / f"clip_{idx:02d}_{int(start):06d}_{int(start + clip_duration):06d}.m4a"
        extract_clip(audio, clip, start, clip_duration)
        if not clip.exists():
            wav = clip.with_suffix(".wav")
            if wav.exists():
                clip = wav
        clip_paths.append({"index": idx, "start": start, "duration": clip_duration, "path": clip})

    sample_rows: list[dict[str, Any]] = []
    for mode in modes:
        for item in clip_paths:
            row, result = run_sample(
                item["path"],
                backend=args.backend,
                model_id=model_id,
                mode=mode,
                language=args.language,
            )
            row.update({"clip_index": item["index"], "clip_start": item["start"], "clip_duration": item["duration"]})
            sample_rows.append({k: v for k, v in row.items() if k != "quality"})
            if result is not None:
                run_dir = runs_dir / safe_name(mode) / f"clip_{item['index']:02d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "result.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
                quality = row.get("quality") or {}
                write_asr_quality_reports(run_dir, "ASR质量检查", quality)

    mode_summary = []
    for mode in modes:
        mode_summary.append(aggregate_mode([row for row in sample_rows if row.get("mode") == mode]))
    mode_summary.sort(key=lambda row: tuple(row.get("score_key") or [999]))
    recommended = str(mode_summary[0].get("mode") or "adaptive") if mode_summary else "adaptive"
    full_out = ROOT / "output" / f"asr_full_{safe_name(audio.stem)}_{timestamp}"
    recommended_command = (
        f"PYTHONPATH=scribe-py/src .venv/bin/python -m scribe_py transcribe "
        f"{json.dumps(str(audio), ensure_ascii=False)} --out {json.dumps(str(full_out), ensure_ascii=False)} "
        f"--backend {args.backend} --model {json.dumps(model_id, ensure_ascii=False)} "
        f"--audio-preprocess {recommended} --formats txt,srt,json,md"
    )

    payload = {
        "mode": "asr_preflight_select",
        "audio": str(audio),
        "backend": args.backend,
        "model": model_id,
        "duration_s": duration,
        "probe": probe,
        "audio_quality": audio_quality,
        "windows": [
            {"index": item["index"], "start": item["start"], "end": item["start"] + item["duration"], "path": str(item["path"])}
            for item in clip_paths
        ],
        "sample_rows": sample_rows,
        "mode_summary": mode_summary,
        "recommended_mode": recommended,
        "recommended_command": recommended_command,
    }
    json_out = out_dir / "预检推荐.json"
    md_out = out_dir / "预检推荐.md"
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"ok": True, "recommended_mode": recommended, "json": str(json_out), "md": str(md_out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
