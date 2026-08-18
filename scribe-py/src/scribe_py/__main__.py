"""LocalScribe CLI — 设计为 AI 编码工具可直接驱动。

模式:
  - `python -m scribe_py pipeline AUDIO`                 → 一条龙:转录 + 校对 + 排版
  - `python -m scribe_py transcribe AUDIO`               → 仅转录
  - `python -m scribe_py correct TRANSCRIPT.json`        → LLM 字级校对
  - `python -m scribe_py polish TRANSCRIPT.json`         → LLM 整篇排版
  - `python -m scribe_py ls`                             → 列出 transcripts/ 历史库
  - `python -m scribe_py check-model`                    → 检查模型缓存
  - `python -m scribe_py probe-audio AUDIO`              → 探测音频元数据
  - `python -m scribe_py ipc`                            → JSON-RPC sidecar 模式

所有命令支持 `--json`,输出严格 JSON 到 stdout(便于 AI 工具解析),
日志走 stderr。错误时 exit code 非零。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .core.audio import analyze_audio_quality_for_asr, build_asr_preprocess_plan, probe_audio
from .core.asr_quality import build_asr_quality_report, select_asr_review_segments, write_asr_quality_reports
from .core.hotwords import load_hotword_terms
from .core.selector import default_model_id, make_transcriber
from .core.text_normalizer import simplify_chinese_value
from .core.types import Segment, TranscribeOptions
from .correctors.openai_compatible import OpenAICompatibleCorrector
from .exporters import json_export, md, srt, txt
from .exporters._common import fmt_ts
from .polishers.article_polisher import ArticlePolisher
from .reviewers.asr_llm_review import review_asr_segments

# Track --json globally so helpers can suppress human output.
_json_mode = False


def _emit_json(obj: dict) -> None:
    """JSON output mode:仅一行 JSON 到 stdout(便于 AI 工具解析)。"""
    sys.stdout.write(json.dumps(simplify_chinese_value(obj), ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    """Human log → stderr (stdout 留给 JSON / 实际结果)。"""
    if not _json_mode:
        sys.stdout.write(msg + "\n")
    else:
        sys.stderr.write(msg + "\n")


def _segments_from_json(json_path: Path) -> tuple[list[Segment], dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = [
        Segment(
            start=float(s["start"]),
            end=float(s["end"]),
            text=s["text"],
            original_text=s.get("original_text"),
            speaker=s.get("speaker"),
            sync_cues=s.get("sync_cues") or None,
        )
        for s in data["segments"]
    ]
    return segments, data


def _read_api_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("ERROR: API key required. Pass --api-key or set $DEEPSEEK_API_KEY / $OPENAI_API_KEY.")
    return key


def _resolve_out_dir(args: argparse.Namespace, audio_stem: str) -> Path:
    """Determine the per-file output directory.

    If `--out DIR` looks like a parent (no per-file subdir requested), create
    `DIR/<stem>/` so this command's outputs share a folder with future stages.
    """
    out_root = Path(args.out)
    return out_root / audio_stem if not args.flat else out_root


def _load_cli_hotwords(args: argparse.Namespace) -> list[str]:
    return load_hotword_terms(
        inline=getattr(args, "hotwords", "") or "",
        file_path=getattr(args, "hotwords_file", "") or None,
    )


def _apply_quality_mode(
    result: Any,
    audio: Path,
    quality_mode: str,
    *,
    detector_segments: list[Segment] | None = None,
    detector_source: str = "",
) -> Any:
    """Optionally apply the same conservative local review used by the App.

    The primary ASR transcript remains authoritative for segmentation and time
    alignment.  Strong mode can only make a bounded text substitution after
    Paraformer and Qwen independently agree; it never replaces the timeline.
    """
    mode = (quality_mode or "standard").strip().lower()
    if mode != "strong":
        result.filter_stats = {
            **(result.filter_stats or {}),
            "asr_quality_mode": "standard",
            "strong_asr": {
                "mode": "local_strong_asr_consensus",
                "enabled": False,
                "applied": False,
                "reason": "standard_mode",
            },
        }
        return result

    import gc

    started = time.monotonic()
    audio_stats = (result.filter_stats or {}).get("audio_standardization") or {}
    review_audio = Path(str(audio_stats.get("path") or audio))
    if not review_audio.exists():
        review_audio = audio
    try:
        from .core.strong_asr import run_strong_asr_review

        reviewed_segments, stats = run_strong_asr_review(
            review_audio,
            result.segments,
            qwen_audio=audio,
            detector_segments=detector_segments,
            detector_source=detector_source,
            language=result.language or "zh",
        )
        result.segments = reviewed_segments
    except Exception as exc:  # Keep the baseline transcript on any review failure.
        stats = {
            "mode": "local_strong_asr_consensus",
            "enabled": True,
            "applied": False,
            "reason": f"strong_review_failed:{type(exc).__name__}",
            "error": str(exc),
        }
    finally:
        gc.collect()

    cost_seconds = time.monotonic() - started
    stats["cost_seconds"] = round(cost_seconds, 3)
    result.transcribe_seconds += cost_seconds
    result.rtf = result.transcribe_seconds / result.duration if result.duration else 0.0
    result.filter_stats = {
        **(result.filter_stats or {}),
        "asr_quality_mode": "strong",
        "strong_asr": stats,
    }
    return result


# ============================================================================
# Subcommands
# ============================================================================

def cmd_transcribe(args: argparse.Namespace) -> int:
    audio = Path(args.audio).expanduser().resolve()
    if not audio.exists():
        _log(f"ERROR: audio not found: {audio}")
        return 2
    stem = audio.stem
    out_dir = _resolve_out_dir(args, stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    transcriber = make_transcriber(args.backend)
    hotwords = _load_cli_hotwords(args)
    options = TranscribeOptions(
        language=args.language,
        model_id=args.model or default_model_id(args.backend),
        initial_prompt=args.prompt or "",
        hotwords=hotwords,
        normalizer_profile=args.normalizer_profile or None,
        audio_preprocess=args.audio_preprocess,
    )
    _log(f"[transcribe] {audio.name} backend={transcriber.backend} hotwords={len(hotwords)}")
    t0 = time.time()
    result = transcriber.transcribe(audio, options)
    detector_segments: list[Segment] = []
    detector_source = ""
    if args.quality_mode == "strong":
        detector_snapshot = getattr(transcriber, "strong_asr_detector_snapshot", None)
        if callable(detector_snapshot):
            cached_segments, cached_source, cached_is_paraformer = detector_snapshot()
            if cached_is_paraformer:
                detector_segments = list(cached_segments)
                detector_source = str(cached_source or "paraformer_timing_anchor")
        # Free the baseline model before loading the two review models.
        del transcriber
    result = _apply_quality_mode(
        result,
        audio,
        args.quality_mode,
        detector_segments=detector_segments,
        detector_source=detector_source,
    )
    elapsed = time.time() - t0
    _log(f"[done] {len(result.segments)} segments cost={elapsed:.1f}s rtf={result.rtf:.3f}")

    # Always write txt + srt + json (the canonical 3 — AI tools can pick what they need)
    paths = {
        "txt": str(txt.write(
            out_dir / f"{stem}.txt",
            result.segments,
            header=f"{audio.name}\nbackend={result.backend} duration={result.duration:.1f}s",
        )),
        "srt": str(srt.write(out_dir / f"{stem}.srt", result.segments)),
        "json": str(json_export.write(out_dir / f"{stem}.json", result)),
    }
    if "md" in args.formats.split(","):
        paths["md"] = str(md.write(out_dir / f"{stem}.md", result.segments, title=audio.name))
    quality_report = simplify_chinese_value(build_asr_quality_report(
        result.segments,
        hotwords=hotwords,
        text_normalization=(result.filter_stats or {}).get("text_normalization") or {},
        audio_quality=(result.filter_stats or {}).get("audio_quality") or {},
        audio_preprocessing=(result.filter_stats or {}).get("audio_standardization") or {},
        backend=result.backend,
        model_id=result.model_id,
        duration=result.duration,
        transcribe_seconds=result.transcribe_seconds,
        rtf=result.rtf,
    ))
    paths.update(write_asr_quality_reports(
        out_dir,
        stem,
        quality_report,
        per_transcript=bool(args.flat),
    ))

    if _json_mode:
        _emit_json({
            "ok": True,
            "stage": "transcribe",
            "audio": str(audio),
            "out_dir": str(out_dir),
            "files": paths,
            "stats": {
                "segments": len(result.segments),
                "duration_s": result.duration,
                "transcribe_seconds": result.transcribe_seconds,
                "rtf": result.rtf,
                "backend": result.backend,
                "model": result.model_id,
                "language": result.language,
                "hotwords": len(hotwords),
                "audio_preprocess": args.audio_preprocess,
                "quality_mode": args.quality_mode,
                "asr_risk_level": quality_report["risk_level"],
            },
        })
    else:
        _log("[output]")
        for k, p in paths.items():
            _log(f"  {k}: {p}")
    return 0


def cmd_correct(args: argparse.Namespace) -> int:
    src = Path(args.json_path).expanduser().resolve()
    if not src.exists():
        _log(f"ERROR: transcript json not found: {src}")
        return 2
    segments, _src_data = _segments_from_json(src)
    api_key = _read_api_key(args.api_key)
    corrector = OpenAICompatibleCorrector(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        mode=args.mode,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        use_glossary=not args.no_glossary,
    )
    _log(f"[correct] {len(segments)} segs · {args.model} · {args.mode} · concurrency={args.concurrency}")
    t0 = time.time()
    out = corrector.correct(segments, context_hint=args.context or "")
    elapsed = time.time() - t0
    changed = sum(1 for s in out if s.original_text and s.text != s.original_text)
    _log(f"[done] changed={changed}/{len(out)} cost={elapsed:.1f}s")

    stem = src.stem.replace("_corrected", "")
    out_dir = src.parent

    # Build diff text
    diff_lines = [f"# diff: {changed} changes / {len(out)} segments\n"]
    for s in out:
        if s.original_text and s.text != s.original_text:
            diff_lines.append(f"[{fmt_ts(s.start)}]\n  - {s.original_text}\n  + {s.text}\n")

    paths = {
        "txt": str(txt.write(out_dir / f"{stem}_corrected.txt", out, header=f"{stem} (corrected by {args.model})")),
        "srt": str(srt.write(out_dir / f"{stem}_corrected.srt", out)),
        "json": str(out_dir / f"{stem}_corrected.json"),
        "diff": str(out_dir / f"{stem}_diff.txt"),
    }
    Path(paths["json"]).write_text(
        json.dumps(simplify_chinese_value({
            "audio": stem,
            "corrected_by": args.model,
            "changed": changed,
            "total": len(out),
            "glossary": corrector.last_glossary,
            "segments": [s.to_dict() for s in out],
        }), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(paths["diff"]).write_text(simplify_chinese_value("\n".join(diff_lines)), encoding="utf-8")

    if _json_mode:
        _emit_json({
            "ok": True,
            "stage": "correct",
            "input_json": str(src),
            "files": paths,
            "stats": {
                "changed": changed,
                "total": len(out),
                "model": corrector.model,
                "mode": corrector.mode,
                "glossary_count": len(corrector.last_glossary),
                "cost_seconds": elapsed,
                "cancelled": corrector.last_cancelled,
            },
        })
    else:
        _log("[output]")
        for k, p in paths.items():
            _log(f"  {k}: {p}")
    return 0


def cmd_polish(args: argparse.Namespace) -> int:
    src = Path(args.json_path).expanduser().resolve()
    if not src.exists():
        _log(f"ERROR: json not found: {src}")
        return 2
    segments, _ = _segments_from_json(src)
    api_key = _read_api_key(args.api_key)
    polisher = ArticlePolisher(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
    )
    _log(f"[polish] {len(segments)} segs → {args.model}")
    t0 = time.time()
    out = polisher.polish(segments)
    elapsed = time.time() - t0
    text = out["text"]
    truncated = out.get("truncated", False)
    _log(f"[done] {len(text)} chars cost={elapsed:.1f}s" + (" ⚠ TRUNCATED" if truncated else ""))

    stem = src.stem.replace("_corrected", "")
    out_path = src.parent / f"{stem}_完整版.txt"
    out_path.write_text(
        f"# {stem} — 完整文字稿\n# 排版 {args.model}\n\n{text}\n",
        encoding="utf-8",
    )

    if _json_mode:
        _emit_json({
            "ok": True,
            "stage": "polish",
            "input_json": str(src),
            "file": str(out_path),
            "stats": {
                "char_count": len(text),
                "truncated": truncated,
                "finish_reason": out.get("finish_reason"),
                "input_chars": out.get("input_chars"),
                "model": polisher.model,
                "cost_seconds": elapsed,
            },
        })
    else:
        _log(f"[output] {out_path}")
        if truncated:
            _log("⚠ WARNING: output was truncated (hit max_tokens). Increase --max-tokens.")
    return 1 if truncated else 0


def _write_asr_review_markdown(path: Path, suggestions: list[dict[str, Any]]) -> None:
    lines = [
        "# ASR 疑点二次核验\n\n",
        "| 时间 | 置信度 | 需要人工 | 当前文本 | 建议文本 | 原因 |\n",
        "|---|---|---:|---|---|---|\n",
    ]
    for item in suggestions:
        ts = f"{fmt_ts(float(item.get('start', 0)))}-{fmt_ts(float(item.get('end', 0)))}"
        current = str(item.get("current_text") or "").replace("|", "\\|")
        suggested = str(item.get("suggested_text") or "").replace("|", "\\|")
        reason = str(item.get("reason") or "").replace("|", "\\|")
        need = "是" if item.get("need_human_review") else "否"
        lines.append(f"| {ts} | {item.get('confidence', '')} | {need} | {current} | {suggested} | {reason} |\n")
    path.write_text("".join(lines), encoding="utf-8")


def cmd_review_asr(args: argparse.Namespace) -> int:
    src = Path(args.json_path).expanduser().resolve()
    if not src.exists():
        _log(f"ERROR: transcript json not found: {src}")
        return 2
    segments, data = _segments_from_json(src)
    review_selection = select_asr_review_segments(data, transcript_json=src, scope=args.review_scope)
    review_segments = list(review_selection.get("segments") or [])
    if not review_segments:
        _log(
            "[review-asr] no "
            f"{args.review_scope} local ASR review segments found "
            f"(total={review_selection.get('total_segment_count', 0)}, "
            f"strong={review_selection.get('strong_segment_count', 0)}, "
            f"weak={review_selection.get('weak_segment_count', 0)})"
        )
        if _json_mode:
            _emit_json({
                "ok": True,
                "stage": "review-asr",
                "input_json": str(src),
                "review_scope": args.review_scope,
                "selection": review_selection,
                "suggestions": [],
                "changed_high_confidence": 0,
            })
        return 0

    glossary = ""
    if args.glossary_file:
        glossary_path = Path(args.glossary_file).expanduser().resolve()
        if not glossary_path.exists():
            _log(f"ERROR: glossary file not found: {glossary_path}")
            return 2
        glossary = glossary_path.read_text(encoding="utf-8")

    api_key = _read_api_key(args.api_key)
    _log(
        f"[review-asr] scope={args.review_scope} · selected={len(review_segments)} "
        f"· total={review_selection.get('total_segment_count', 0)} "
        f"· strong={review_selection.get('strong_segment_count', 0)} "
        f"· skipped_weak={review_selection.get('skipped_weak_count', 0)} "
        f"· limit={args.limit} · model={args.model}"
    )
    t0 = time.time()
    result = review_asr_segments(
        segments,
        review_segments,
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        project_glossary=glossary,
        context_hint=args.context or "",
        limit=args.limit,
        context_window=args.context_window,
        max_tokens=args.max_tokens,
    )
    elapsed = time.time() - t0
    out_dir = Path(args.out).expanduser().resolve() if args.out else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    json_path = out_dir / f"{stem}_asr_review.json"
    md_path = out_dir / f"{stem}_asr_review.md"
    payload = {
        "input_json": str(src),
        "model": result.model,
        "review_scope": args.review_scope,
        "review_selection": {
            "sources": review_selection.get("sources") or [],
            "total_segment_count": review_selection.get("total_segment_count", 0),
            "strong_segment_count": review_selection.get("strong_segment_count", 0),
            "weak_segment_count": review_selection.get("weak_segment_count", 0),
            "skipped_weak_count": review_selection.get("skipped_weak_count", 0),
        },
        "local_review_segments": len(review_segments),
        "reviewed_segments": len(result.suggestions),
        "changed_high_confidence": result.changed_high_confidence,
        "suggestions": result.suggestions,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_asr_review_markdown(md_path, result.suggestions)
    _log(f"[done] reviewed={len(result.suggestions)} high_conf_changed={result.changed_high_confidence} cost={elapsed:.1f}s")

    if _json_mode:
        _emit_json({
            "ok": True,
            "stage": "review-asr",
            "input_json": str(src),
            "files": {"json": str(json_path), "md": str(md_path)},
            "stats": {
                "local_review_segments": len(review_segments),
                "local_review_total_segments": review_selection.get("total_segment_count", 0),
                "local_review_strong_segments": review_selection.get("strong_segment_count", 0),
                "local_review_skipped_weak_segments": review_selection.get("skipped_weak_count", 0),
                "reviewed_segments": len(result.suggestions),
                "changed_high_confidence": result.changed_high_confidence,
                "model": result.model,
                "cost_seconds": elapsed,
            },
        })
    else:
        _log("[output]")
        _log(f"  json: {json_path}")
        _log(f"  md:   {md_path}")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """一条龙:transcribe → (correct) → (polish)。AI 工具最常用的入口。"""
    audio = Path(args.audio).expanduser().resolve()
    if not audio.exists():
        _log(f"ERROR: audio not found: {audio}")
        return 2
    stem = audio.stem
    out_dir = _resolve_out_dir(args, stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()
    final = {"ok": True, "stage": "pipeline", "audio": str(audio), "out_dir": str(out_dir), "stages": {}}

    # ---- Stage 1: transcribe ----
    transcriber = make_transcriber(args.backend)
    hotwords = _load_cli_hotwords(args)
    options = TranscribeOptions(
        language=args.language,
        model_id=args.model or default_model_id(args.backend),
        initial_prompt=args.prompt or "",
        hotwords=hotwords,
        normalizer_profile=args.normalizer_profile or None,
        audio_preprocess=args.audio_preprocess,
    )
    _log(f"[1/3 transcribe] {audio.name} hotwords={len(hotwords)}")
    t0 = time.time()
    result = transcriber.transcribe(audio, options)
    t_elapsed = time.time() - t0
    raw_paths = {
        "txt": str(txt.write(out_dir / f"{stem}.txt", result.segments, header=audio.name)),
        "srt": str(srt.write(out_dir / f"{stem}.srt", result.segments)),
        "json": str(json_export.write(out_dir / f"{stem}.json", result)),
    }
    quality_report = build_asr_quality_report(
        result.segments,
        hotwords=hotwords,
        text_normalization=(result.filter_stats or {}).get("text_normalization") or {},
        audio_quality=(result.filter_stats or {}).get("audio_quality") or {},
        audio_preprocessing=(result.filter_stats or {}).get("audio_standardization") or {},
        backend=result.backend,
        model_id=result.model_id,
        duration=result.duration,
        transcribe_seconds=result.transcribe_seconds,
        rtf=result.rtf,
    )
    raw_paths.update(write_asr_quality_reports(out_dir, stem, quality_report))
    _log(f"      {len(result.segments)} segs · {t_elapsed:.1f}s · RTF {result.rtf:.3f}")
    final["stages"]["transcribe"] = {
        "files": raw_paths,
        "segments": len(result.segments),
        "duration_s": result.duration,
        "cost_seconds": t_elapsed,
        "rtf": result.rtf,
        "backend": result.backend,
        "hotwords": len(hotwords),
        "audio_preprocess": args.audio_preprocess,
        "asr_risk_level": quality_report["risk_level"],
    }

    # If user asked transcribe-only or LLM key not set + no flag, stop here.
    has_key = bool(os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or args.api_key)
    if args.transcribe_only or not has_key:
        if not has_key and not args.transcribe_only:
            _log("[skip correct/polish] no API key in env (DEEPSEEK_API_KEY / OPENAI_API_KEY)")
        if _json_mode:
            final["total_cost_seconds"] = time.time() - overall_t0
            _emit_json(final)
        return 0

    # ---- Stage 2: correct ----
    api_key = _read_api_key(args.api_key)
    corrector = OpenAICompatibleCorrector(
        api_key=api_key,
        base_url=args.base_url,
        model=args.llm_model,
        mode=args.mode,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        use_glossary=not args.no_glossary,
    )
    _log(f"[2/3 correct] {args.llm_model} · concurrency={args.concurrency}")
    t0 = time.time()
    corr_segs = corrector.correct(result.segments, context_hint=args.context or "")
    c_elapsed = time.time() - t0
    changed = sum(1 for s in corr_segs if s.original_text and s.text != s.original_text)
    _log(f"      changed {changed}/{len(corr_segs)} · {c_elapsed:.1f}s")

    corr_paths = {
        "txt": str(txt.write(out_dir / f"{stem}_corrected.txt", corr_segs)),
        "srt": str(srt.write(out_dir / f"{stem}_corrected.srt", corr_segs)),
        "json": str(out_dir / f"{stem}_corrected.json"),
    }
    Path(corr_paths["json"]).write_text(
        json.dumps(simplify_chinese_value({
            "audio": stem,
            "corrected_by": args.llm_model,
            "changed": changed,
            "total": len(corr_segs),
            "glossary": corrector.last_glossary,
            "segments": [s.to_dict() for s in corr_segs],
        }), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final["stages"]["correct"] = {
        "files": corr_paths,
        "changed": changed,
        "total": len(corr_segs),
        "glossary_count": len(corrector.last_glossary),
        "cost_seconds": c_elapsed,
    }

    if args.no_polish:
        if _json_mode:
            final["total_cost_seconds"] = time.time() - overall_t0
            _emit_json(final)
        return 0

    # ---- Stage 3: polish ----
    polisher = ArticlePolisher(
        api_key=api_key,
        base_url=args.base_url,
        model=args.llm_model,
        max_tokens=args.max_tokens,
    )
    _log(f"[3/3 polish] {args.llm_model}")
    t0 = time.time()
    pol = polisher.polish(corr_segs)
    p_elapsed = time.time() - t0
    truncated = pol.get("truncated", False)
    text = pol["text"]
    _log(f"      {len(text)} chars · {p_elapsed:.1f}s" + (" ⚠ TRUNCATED" if truncated else ""))
    article_path = out_dir / f"{stem}_完整版.txt"
    article_path.write_text(
        f"# {stem} — 完整文字稿\n# 排版 {args.llm_model}\n\n{text}\n",
        encoding="utf-8",
    )
    final["stages"]["polish"] = {
        "file": str(article_path),
        "char_count": len(text),
        "truncated": truncated,
        "finish_reason": pol.get("finish_reason"),
        "cost_seconds": p_elapsed,
    }

    final["total_cost_seconds"] = time.time() - overall_t0
    final["ok"] = not truncated  # ok=False if article was truncated

    if _json_mode:
        _emit_json(final)
    else:
        _log("\n[pipeline done]")
        _log(f"  raw:       {raw_paths['txt']}")
        _log(f"  corrected: {corr_paths['txt']}")
        _log(f"  article:   {article_path}")
        if truncated:
            _log("  ⚠ article truncated; raise --max-tokens")
    return 1 if truncated else 0


def cmd_check_model(args: argparse.Namespace) -> int:
    from . import ipc
    info = ipc.handle_check_model({"backend": args.backend, "model_id": args.model})
    if _json_mode:
        _emit_json({"ok": True, **info})
    else:
        _log(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def cmd_probe_audio(args: argparse.Namespace) -> int:
    audio = Path(args.audio).expanduser().resolve()
    if not audio.exists():
        _log(f"ERROR: audio not found: {audio}")
        return 2
    info = probe_audio(audio)
    quality = analyze_audio_quality_for_asr(audio)
    payload = {
        "audio": str(audio),
        **info,
        "asr_quality_gate": quality,
        "asr_preprocess_plan": build_asr_preprocess_plan(quality),
    }
    if _json_mode:
        _emit_json({"ok": True, **payload})
    else:
        _log(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_ls(args: argparse.Namespace) -> int:
    """列出 transcripts/ 历史库(每个子目录一个任务)。"""
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        _log(f"transcripts dir not found: {root}")
        return 0
    entries = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        meta_path = sub / "task.json"
        meta: dict[str, Any] = {"stem": sub.name, "path": str(sub)}
        if meta_path.exists():
            try:
                meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                pass
        entries.append(meta)
    if _json_mode:
        _emit_json({"ok": True, "root": str(root), "entries": entries})
    else:
        for e in entries:
            line = e.get("stem", "?")
            extras = []
            if "duration" in e: extras.append(f"{int(e['duration']/60)}min")
            if "segments" in e: extras.append(f"{e['segments']} segs")
            if e.get("has_corrected"): extras.append("✓校对")
            if e.get("has_polished"): extras.append("✓排版")
            _log(f"  {line:40s}  {' · '.join(extras)}")
        if not entries:
            _log("(empty)")
    return 0


def _articles_root() -> Path:
    """Find articles/ relative to project root (LocalScribe/).

    与 Rust 端 `library::project_root` 判定保持一致:同时存在
    package.json + scribe-py/ 才算 LocalScribe 项目根。这样 src-tauri/
    或子目录都不会被误判为根。
    """
    cur = Path.cwd().resolve()
    for p in [cur, *cur.parents]:
        if (p / "package.json").exists() and (p / "scribe-py").exists():
            return p / "articles"
    # Last-resort: hardcoded path for personal-use builds
    fallback = Path("/Users/apple/gitCommit/SwarmPathAI/LocalScribe/articles")
    if fallback.parent.exists():
        return fallback
    return cur / "articles"


def cmd_articles_ls(args: argparse.Namespace) -> int:
    root = _articles_root()
    if not root.exists():
        if _json_mode:
            _emit_json({"ok": True, "root": str(root), "articles": []})
        else:
            _log("(empty — articles/ 目录不存在)")
        return 0
    items: list[dict] = []
    for p in sorted(root.glob("*.md")):
        meta = _parse_frontmatter(p)
        meta["path"] = str(p)
        meta["filename"] = p.name
        items.append(meta)
    items.sort(key=lambda m: m.get("modified_at", ""), reverse=True)
    if _json_mode:
        _emit_json({"ok": True, "root": str(root), "articles": items})
    else:
        for m in items:
            tags = m.get("tags", [])
            tag_str = " · " + ", ".join(tags) if tags else ""
            _log(f"  {m.get('title', m['filename']):40s}  {m.get('char_count', 0)} 字{tag_str}")
            _log(f"    {m['path']}")
        if not items:
            _log("(empty)")
    return 0


def cmd_articles_show(args: argparse.Namespace) -> int:
    root = _articles_root()
    # Accept either a filename or a title (we'll try both)
    target = args.target
    candidate = root / (target if target.endswith(".md") else f"{target}.md")
    if not candidate.exists():
        # try title match
        for p in root.glob("*.md"):
            meta = _parse_frontmatter(p)
            if meta.get("title", "").strip() == target.strip():
                candidate = p
                break
    if not candidate.exists():
        _log(f"ERROR: article not found: {target}")
        return 2
    content = candidate.read_text(encoding="utf-8")
    if _json_mode:
        _emit_json({"ok": True, "path": str(candidate), "content": content})
    else:
        sys.stdout.write(content)
    return 0


def cmd_articles_dir(args: argparse.Namespace) -> int:
    root = _articles_root()
    if _json_mode:
        _emit_json({"ok": True, "path": str(root), "exists": root.exists()})
    else:
        _log(str(root))
    return 0


def _parse_frontmatter(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    out: dict = {}
    if raw.startswith("---\n"):
        end = raw.find("\n---\n", 4)
        if end != -1:
            for line in raw[4:end].splitlines():
                if ":" not in line:
                    continue
                k, _, v = line.partition(":")
                v = v.strip().strip('"').strip("'")
                k = k.strip()
                if k == "tags":
                    inner = v.lstrip("[").rstrip("]")
                    out["tags"] = [t.strip().strip('"').strip("'") for t in inner.split(",") if t.strip()]
                elif k in ("duration_seconds", "char_count"):
                    try:
                        out[k] = float(v) if "." in v else int(v)
                    except Exception:  # noqa: BLE001
                        out[k] = v
                else:
                    out[k] = v
    try:
        mt = path.stat().st_mtime
        import datetime
        out["modified_at"] = (
            datetime.datetime.fromtimestamp(mt, datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        )
    except Exception:  # noqa: BLE001
        pass
    return out


def cmd_ipc(_args: argparse.Namespace) -> int:
    from . import ipc as ipc_mod
    ipc_mod.run()
    return 0


# ============================================================================
# Argument parser
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    # Common flags inherited by every subcommand (so --json works in any position).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="结果以 JSON 输出到 stdout(便于 AI 工具解析)")

    p = argparse.ArgumentParser(
        prog="scribe-py",
        description=(
            "LocalScribe CLI — 离线录音转文字 + 可选 LLM 校对/排版。\n"
            "对 AI 编码工具友好:所有命令支持 --json,日志走 stderr。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- pipeline ----
    pl = sub.add_parser("pipeline", parents=[common], help="一条龙:转录 → 校对 → 排版(推荐 AI 工具入口)")
    pl.add_argument("audio", help="输入音频文件路径")
    pl.add_argument("--out", default="transcripts", help="输出根目录,默认 transcripts/")
    pl.add_argument("--flat", action="store_true", help="所有产物直接放 --out 下,不建子目录")
    pl.add_argument("--backend", default="auto", choices=["auto", "mlx", "ct2", "funasr", "sensevoice"])
    pl.add_argument("--model", default="", help="Whisper model id 覆盖")
    pl.add_argument("--language", default="zh")
    pl.add_argument("--prompt", default="", help="Whisper initial_prompt")
    pl.add_argument("--hotwords", default="", help="ASR 热词/客户词表,逗号或换行分隔")
    pl.add_argument("--hotwords-file", default="", help="ASR 热词/客户词表文件,支持纯文本或 JSON")
    pl.add_argument("--normalizer-profile", default="", help="录音专用已确认纠错 profile；默认空=仅通用清理/疑点标注")
    pl.add_argument("--audio-preprocess", default="adaptive", choices=["off", "standard", "adaptive", "ai_denoise", "enhance"], help="音频预处理: adaptive=格式标准化+噪声/低音量时自动轻增强; ai_denoise=DeepFilterNet AI 降噪; enhance=强降噪+响度均衡")
    # LLM stages
    pl.add_argument("--api-key", default="", help="LLM API key(也可读 $DEEPSEEK_API_KEY)")
    pl.add_argument("--base-url", default="https://api.deepseek.com")
    pl.add_argument("--llm-model", default="deepseek-v4-flash")
    pl.add_argument("--mode", default="medium", choices=["light", "medium", "heavy"])
    pl.add_argument("--batch-size", type=int, default=20)
    pl.add_argument("--concurrency", type=int, default=5)
    pl.add_argument("--no-glossary", action="store_true")
    pl.add_argument("--context", default="")
    pl.add_argument("--max-tokens", type=int, default=384000, help="排版输出 token 上限,默认 384K")
    pl.add_argument("--transcribe-only", action="store_true", help="仅跑转录,不做 LLM 校对/排版")
    pl.add_argument("--no-polish", action="store_true", help="跑转录 + 校对,不做排版")
    pl.set_defaults(func=cmd_pipeline)

    # ---- transcribe ----
    t = sub.add_parser("transcribe", parents=[common], help="仅转录")
    t.add_argument("audio")
    t.add_argument("--out", default="transcripts")
    t.add_argument("--flat", action="store_true")
    t.add_argument("--backend", default="auto", choices=["auto", "mlx", "ct2", "funasr", "sensevoice"])
    t.add_argument("--model", default="")
    t.add_argument("--language", default="zh")
    t.add_argument("--prompt", default="")
    t.add_argument("--hotwords", default="", help="ASR 热词/客户词表,逗号或换行分隔")
    t.add_argument("--hotwords-file", default="", help="ASR 热词/客户词表文件,支持纯文本或 JSON")
    t.add_argument("--normalizer-profile", default="", help="录音专用已确认纠错 profile；默认空=仅通用清理/疑点标注")
    t.add_argument("--audio-preprocess", default="adaptive", choices=["off", "standard", "adaptive", "ai_denoise", "enhance"], help="音频预处理: adaptive=格式标准化+噪声/低音量时自动轻增强; ai_denoise=DeepFilterNet AI 降噪; enhance=强降噪+响度均衡")
    t.add_argument("--quality-mode", default="standard", choices=["standard", "strong"], help="strong=仅对本地三模型一致的局部错字复核，较慢")
    t.add_argument("--formats", default="txt,srt,json", help="逗号分隔: txt,srt,json,md")
    t.set_defaults(func=cmd_transcribe)

    # ---- correct ----
    c = sub.add_parser("correct", parents=[common], help="LLM 字级校对(对已有 transcribe json)")
    c.add_argument("json_path")
    c.add_argument("--api-key", default="")
    c.add_argument("--base-url", default="https://api.deepseek.com")
    c.add_argument("--model", default="deepseek-v4-flash")
    c.add_argument("--mode", default="medium", choices=["light", "medium", "heavy"])
    c.add_argument("--batch-size", type=int, default=20)
    c.add_argument("--concurrency", type=int, default=5)
    c.add_argument("--no-glossary", action="store_true")
    c.add_argument("--context", default="")
    c.set_defaults(func=cmd_correct)

    # ---- review-asr ----
    rv = sub.add_parser("review-asr", parents=[common], help="只把本地标出的 ASR 疑点交给 LLM 二次核验")
    rv.add_argument("json_path")
    rv.add_argument("--api-key", default="")
    rv.add_argument("--base-url", default="https://api.deepseek.com")
    rv.add_argument("--model", default="deepseek-v4-flash")
    rv.add_argument("--out", default="", help="输出目录,默认与 transcript json 同目录")
    rv.add_argument("--context", default="", help="项目/录音上下文提示")
    rv.add_argument("--glossary-file", default="", help="项目词表/确认人名文件,纯文本即可")
    rv.add_argument("--review-scope", default="strong", choices=["strong", "all"], help="默认只核验强疑点; all 会包含弱抽查段")
    rv.add_argument("--limit", type=int, default=40, help="最多核验多少条本地疑点")
    rv.add_argument("--context-window", type=int, default=2, help="每条疑点带前后多少段上下文")
    rv.add_argument("--max-tokens", type=int, default=4096)
    rv.set_defaults(func=cmd_review_asr)

    # ---- polish ----
    pp = sub.add_parser("polish", parents=[common], help="LLM 整篇排版(对 transcribe json 或 corrected json)")
    pp.add_argument("json_path")
    pp.add_argument("--api-key", default="")
    pp.add_argument("--base-url", default="https://api.deepseek.com")
    pp.add_argument("--model", default="deepseek-v4-flash")
    pp.add_argument("--max-tokens", type=int, default=384000)
    pp.set_defaults(func=cmd_polish)

    # ---- ls ----
    lsp = sub.add_parser("ls", parents=[common], help="列出 transcripts/ 历史库")
    lsp.add_argument("--root", default="transcripts")
    lsp.set_defaults(func=cmd_ls)

    # ---- check-model ----
    cm = sub.add_parser("check-model", parents=[common], help="检查 Whisper 模型缓存")
    cm.add_argument("--backend", default="auto", choices=["auto", "mlx", "ct2", "funasr", "sensevoice"])
    cm.add_argument("--model", default="")
    cm.set_defaults(func=cmd_check_model)

    # ---- probe-audio ----
    pa = sub.add_parser("probe-audio", parents=[common], help="ffprobe 元数据")
    pa.add_argument("audio")
    pa.set_defaults(func=cmd_probe_audio)

    # ---- articles (文章库) ----
    art = sub.add_parser("articles", parents=[common], help="文章库管理(AI agent 友好)")
    art_sub = art.add_subparsers(dest="art_cmd", required=True)
    art_ls = art_sub.add_parser("ls", parents=[common], help="列出所有已文章")
    art_ls.set_defaults(func=cmd_articles_ls)
    art_show = art_sub.add_parser("show", parents=[common], help="读取一篇文章(按文件名或标题)")
    art_show.add_argument("target", help="文件名(如 项目纪要.md)或标题")
    art_show.set_defaults(func=cmd_articles_show)
    art_dir = art_sub.add_parser("dir", parents=[common], help="返回 articles/ 目录绝对路径")
    art_dir.set_defaults(func=cmd_articles_dir)

    # ---- ipc (Tauri sidecar internal) ----
    sub.add_parser("ipc", parents=[common], help="JSON-RPC sidecar 模式(供 GUI 调用)").set_defaults(func=cmd_ipc)

    return p


def main(argv: list[str] | None = None) -> None:
    global _json_mode
    args = build_parser().parse_args(argv)
    _json_mode = bool(getattr(args, "json", False))
    rc = args.func(args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
