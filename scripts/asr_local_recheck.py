#!/usr/bin/env python3
"""Locally recheck ASR review segments with another ASR backend.

The script is deliberately non-destructive: it never edits the source
transcript. It extracts short audio clips around locally flagged ASR risk
segments, runs a local backend on each clip, and writes a Chinese report with
the original text and candidate re-transcriptions.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PY_SRC = ROOT / "scribe-py" / "src"
DEFAULT_JSON = ROOT / "output" / "asr_test_generic_20260630" / "标准录音 2" / "标准录音 2.json"
DEFAULT_OUT_NAME = "ASR本地疑点复核"
ALLOWED_BACKENDS = {"sensevoice", "funasr", "qwen3", "mlx", "ct2"}
SEVERE_LOW_SIMILARITY = 0.45
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

from scribe_py.core.asr_quality import select_asr_review_segments  # noqa: E402
from scribe_py.core.selector import default_model_id, make_transcriber  # noqa: E402
from scribe_py.core.strong_asr import text_similarity  # noqa: E402
from scribe_py.core.types import TranscribeOptions  # noqa: E402


@dataclass
class RecheckCandidate:
    backend: str
    model: str
    candidate_text: str
    candidate_segments: int
    error: str = ""
    current_similarity: float | None = None


@dataclass
class RecheckItem:
    index: int
    start: float
    end: float
    clip_start: float
    clip_end: float
    original_text: str
    current_text: str
    reasons: list[str]
    candidate_text: str
    candidate_segments: int
    backend: str
    model: str
    clip_path: Path
    error: str = ""
    candidates: list[RecheckCandidate] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)


def _fmt_ts(seconds: float) -> str:
    ms = int(round(float(seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _read_segments(json_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return list(data.get("segments") or []), data


def _review_segments(
    data: dict[str, Any],
    *,
    transcript_json: Path | None = None,
    scope: str = "strong",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = select_asr_review_segments(data, transcript_json=transcript_json, scope=scope)
    return list(selection.get("segments") or []), selection


def _existing_audio_path(raw_path: str, transcript_json: Path) -> Path | None:
    raw_path = str(raw_path or "").strip()
    if not raw_path:
        return None
    expanded = Path(raw_path).expanduser()
    candidates: list[Path]
    if expanded.is_absolute():
        candidates = [expanded, transcript_json.parent / expanded.name]
    else:
        candidates = [
            transcript_json.parent / expanded,
            Path.cwd() / expanded,
            ROOT / expanded,
        ]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved.exists():
            return resolved
    return None


def _audio_field_candidates(data: dict[str, Any]) -> list[str]:
    keys = ("audio", "audio_path", "source_audio", "source_path", "input", "file", "path")
    values: list[str] = []
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip() and value not in values:
            values.append(value)
    return values


def _source_audio(args_audio: str, data: dict[str, Any], transcript_json: Path) -> Path:
    if args_audio:
        path = _existing_audio_path(args_audio, transcript_json)
        if path:
            return path
        raise SystemExit(
            "audio not found: "
            f"{args_audio}\n"
            "请确认源音频存在,或使用 --audio 传入绝对路径/相对路径。"
        )

    checked = _audio_field_candidates(data)
    for raw_path in checked:
        path = _existing_audio_path(raw_path, transcript_json)
        if path:
            return path

    hint = "、".join(checked) if checked else "JSON 中没有 audio/audio_path/source_audio 字段"
    raise SystemExit(
        "audio not found from transcript JSON.\n"
        f"- 转录 JSON: {transcript_json}\n"
        f"- 已检查音频字段: {hint}\n"
        "- 解决方式: 使用 --audio /path/to/source-audio.mp3 指定源音频。"
    )


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found in PATH")
    return ffmpeg


def _extract_clip(audio: Path, out: Path, start: float, end: float, *, timeout: float = 20.0) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = max(end - start, 0.05)
    try:
        proc = subprocess.run(
            [
                _ffmpeg(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(audio),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg clip extraction timed out after {timeout:.1f}s") from exc
    if proc.returncode != 0 or not out.exists() or out.stat().st_size <= 44:
        raise RuntimeError((proc.stderr or "ffmpeg produced no clip").strip())


def _neighbor_text(segments: list[dict[str, Any]], index: int, window: int) -> str:
    start = max(0, index - window)
    end = min(len(segments), index + window + 1)
    return "\n".join(str(segments[i].get("text") or "") for i in range(start, end))


def _run_local_asr(
    clip: Path,
    *,
    backend: str,
    model: str,
    language: str,
    hotwords: list[str],
    context: str,
    transcriber: Any | None = None,
) -> tuple[str, int]:
    transcriber = transcriber or make_transcriber(backend)
    result = transcriber.transcribe(
        clip,
        TranscribeOptions(
            language=language or "zh",
            model_id=model or default_model_id(backend),
            hotwords=hotwords,
            initial_prompt=context,
            normalizer_profile=None,
        ),
    )
    text = "".join(seg.text.strip() for seg in result.segments if seg.text.strip())
    return text, len(result.segments)


def _parse_backend_list(raw: str) -> list[str]:
    backends: list[str] = []
    for part in str(raw or "").replace("，", ",").split(","):
        backend = part.strip()
        if not backend:
            continue
        if backend not in ALLOWED_BACKENDS:
            raise SystemExit(f"unsupported backend in --compare-backends: {backend}")
        if backend not in backends:
            backends.append(backend)
    return backends


def _selected_backends(backend: str, compare_backends: str) -> list[str]:
    backends = _parse_backend_list(compare_backends) if compare_backends else []
    if not backends:
        backends = [backend]
    return backends


def _model_overrides(raw: str, backends: list[str]) -> dict[str, str]:
    value = str(raw or "").strip()
    if not value:
        return {}
    if "=" not in value:
        if len(backends) > 1:
            raise SystemExit("--model 在多后端复核时请使用 BACKEND=MODEL,可用逗号分隔多个覆盖。")
        return {backends[0]: value}
    overrides: dict[str, str] = {}
    for part in value.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--model must be BACKEND=MODEL, got: {part}")
        backend, model = part.split("=", 1)
        backend = backend.strip()
        model = model.strip()
        if backend not in ALLOWED_BACKENDS or not model:
            raise SystemExit(f"--model must be BACKEND=MODEL, got: {part}")
        overrides[backend] = model
    return overrides


def _candidate_to_dict(candidate: RecheckCandidate) -> dict[str, Any]:
    return {
        "backend": candidate.backend,
        "model": candidate.model,
        "candidate_text": candidate.candidate_text,
        "candidate_segments": candidate.candidate_segments,
        "error": candidate.error,
        "current_similarity": candidate.current_similarity,
    }


def _candidate_for_backend(
    candidates: list[RecheckCandidate], backend: str
) -> RecheckCandidate | None:
    return next((candidate for candidate in candidates if candidate.backend == backend), None)


def _usable_candidate_text(candidate: RecheckCandidate | None) -> str:
    if candidate is None or candidate.error:
        return ""
    return candidate.candidate_text.strip()


def _similarity(left: str, right: str) -> float | None:
    if not left.strip() or not right.strip():
        return None
    return round(text_similarity(left, right), 4)


def _build_audit(
    current_text: str,
    candidates: list[RecheckCandidate],
    *,
    severe_threshold: float = SEVERE_LOW_SIMILARITY,
) -> dict[str, Any]:
    paraformer = _candidate_for_backend(candidates, "funasr")
    qwen3 = _candidate_for_backend(candidates, "qwen3")
    paraformer_text = _usable_candidate_text(paraformer)
    qwen3_text = _usable_candidate_text(qwen3)
    similarities = {
        "primary_paraformer": _similarity(current_text, paraformer_text),
        "primary_qwen3": _similarity(current_text, qwen3_text),
        "paraformer_qwen3": _similarity(paraformer_text, qwen3_text),
    }
    low_similarity_pairs = [
        pair
        for pair, similarity in similarities.items()
        if similarity is not None and similarity < severe_threshold
    ]
    missing_candidates = [
        label
        for label, candidate_text in (
            ("paraformer", paraformer_text),
            ("qwen3", qwen3_text),
        )
        if not candidate_text
    ]

    if low_similarity_pairs:
        status = "severe_low_similarity"
        recommendation = (
            "严重低相似度：仅作审计并优先人工复听；禁止用任何候选自动替换 primary/current 正文。"
        )
    elif missing_candidates:
        status = "candidate_incomplete"
        recommendation = (
            "候选不完整：保持 primary/current 正文，补跑缺失后端或人工复听；禁止自动替换。"
        )
    elif (
        similarities["paraformer_qwen3"] is not None
        and similarities["paraformer_qwen3"] >= 0.78
        and (
            (similarities["primary_paraformer"] or 0.0)
            + (similarities["primary_qwen3"] or 0.0)
        )
        / 2.0
        <= 0.72
    ):
        status = "auxiliary_consensus_against_primary"
        recommendation = (
            "Paraformer 与 Qwen3 较一致但不同于 primary/current：优先人工复听；禁止自动替换。"
        )
    elif all(
        similarity is not None and similarity >= 0.85
        for similarity in similarities.values()
    ):
        status = "consistent"
        recommendation = "三方文本高度一致，可降低人工复听优先级；正文仍不自动替换。"
    else:
        status = "review"
        recommendation = "三方存在一般分歧：保留 primary/current，结合音频人工确认；禁止自动替换。"

    return {
        "mode": "audit_only",
        "status": status,
        "auto_replace_allowed": False,
        "severe_low_similarity": bool(low_similarity_pairs),
        "severe_low_similarity_threshold": severe_threshold,
        "low_similarity_pairs": low_similarity_pairs,
        "missing_candidates": missing_candidates,
        "primary_current": current_text,
        "paraformer_candidate": paraformer_text,
        "qwen3_candidate": qwen3_text,
        "similarities": similarities,
        "recommendation": recommendation,
    }


def _load_hotwords(path: str) -> list[str]:
    if not path:
        return []
    from scribe_py.core.hotwords import load_hotword_terms

    return load_hotword_terms(file_path=path)


def _md_cell(value: Any) -> str:
    return str(value or "").replace("\n", "<br>").replace("|", "\\|")


def _display_candidate(candidate: RecheckCandidate | None) -> str:
    if candidate is None:
        return "未运行"
    result = f"ERROR: {candidate.error}" if candidate.error else (candidate.candidate_text or "空结果")
    return _md_cell(f"{candidate.backend}/{candidate.model}: {result}")


def _display_similarity(value: float | None) -> str:
    return "未计算" if value is None else f"{value:.3f}"


def _render_report(
    items: list[RecheckItem],
    source_json: Path,
    source_audio: Path | None,
    selection: dict[str, Any] | None = None,
) -> str:
    selection = selection or {}
    total = selection.get("total_segment_count", len(items))
    strong = selection.get("strong_segment_count", len(items))
    weak = selection.get("weak_segment_count", 0)
    skipped_weak = selection.get("skipped_weak_count", 0)
    lines = [
        "# ASR 本地疑点复核\n\n",
        f"- 转录 JSON: `{source_json}`\n",
        f"- 源音频: `{source_audio or '未解析/未使用'}`\n",
        f"- 复核范围: `{selection.get('scope', 'strong')}`\n",
        f"- 本地疑点总数: {total}；强疑点数: {strong}；弱抽查数: {weak}；跳过弱抽查数: {skipped_weak}\n",
        f"- 复核条数: {len(items)}\n",
        "- 模式: `audit_only`;候选文本只供参考,作为本地复听证据之一,不会自动替换原转录（primary/current 正文）。\n",
        f"- 严重低相似度阈值: `< {SEVERE_LOW_SIMILARITY:.2f}`;命中后必须人工复听。\n\n",
        "| 时间 | 原因 | Primary / Current | Paraformer 候选 | Qwen3 候选 | 其他本地候选 | 三方相似度 | 审计建议 | Clip |\n",
        "|---|---|---|---|---|---|---|---|---|\n",
    ]
    for item in items:
        ts = f"{_fmt_ts(item.start)}-{_fmt_ts(item.end)}"
        reasons = _md_cell("；".join(item.reasons))
        current = _md_cell(item.current_text)
        candidates = item.candidates or [
            RecheckCandidate(
                backend=item.backend,
                model=item.model,
                candidate_text=item.candidate_text,
                candidate_segments=item.candidate_segments,
                error=item.error,
            )
        ]
        paraformer = _candidate_for_backend(candidates, "funasr")
        qwen3 = _candidate_for_backend(candidates, "qwen3")
        other_candidates = [candidate for candidate in candidates if candidate.backend not in {"funasr", "qwen3"}]
        others = "<br>".join(_display_candidate(candidate) for candidate in other_candidates) or "无"
        audit = item.audit or _build_audit(item.current_text, candidates)
        similarities = audit["similarities"]
        similarity_text = "<br>".join(
            [
                f"current↔paraformer: {_display_similarity(similarities['primary_paraformer'])}",
                f"current↔qwen3: {_display_similarity(similarities['primary_qwen3'])}",
                f"paraformer↔qwen3: {_display_similarity(similarities['paraformer_qwen3'])}",
            ]
        )
        recommendation = _md_cell(f"{audit['status']}: {audit['recommendation']}")
        lines.append(
            f"| {ts} | {reasons} | {current} | {_display_candidate(paraformer)} | "
            f"{_display_candidate(qwen3)} | {others} | {similarity_text} | {recommendation} | `{item.clip_path}` |\n"
        )
    return "".join(lines)


def _selected_review_items(review: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return review
    return review[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="本地复核 ASR 疑点段,不修改原转录")
    parser.add_argument("json_path", nargs="?", default=str(DEFAULT_JSON), help=f"转录 JSON; 默认 {DEFAULT_JSON}")
    parser.add_argument("--audio", default="", help="源音频; 默认读取 JSON audio 字段")
    parser.add_argument(
        "--backend",
        default="funasr",
        choices=["sensevoice", "funasr", "qwen3", "mlx", "ct2"],
    )
    parser.add_argument(
        "--compare-backends",
        default="",
        help="多本地后端交叉复听,逗号分隔;三方审计使用 funasr,qwen3",
    )
    parser.add_argument("--model", default="", help="复听模型; 单后端可填模型名,多后端用 BACKEND=MODEL 逗号分隔")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--limit", type=int, default=0, help="最多复核多少个疑点段; 0 表示不限制")
    parser.add_argument("--pad", type=float, default=1.2, help="疑点段前后补多少秒")
    parser.add_argument("--context-window", type=int, default=1, help="作为 prompt 的前后文段数")
    parser.add_argument("--hotwords-file", default="", help="可选热词文件")
    parser.add_argument("--review-scope", default="strong", choices=["strong", "all"], help="默认只复核强疑点; all 会包含弱抽查段")
    parser.add_argument("--dry-run", action="store_true", help="只生成待复核清单和中文报告,不切音频也不跑本地 ASR")
    parser.add_argument("--out", default="", help=f"输出目录; 默认写到转录 JSON 同级 {DEFAULT_OUT_NAME}/")
    args = parser.parse_args()

    json_path = Path(args.json_path).expanduser().resolve()
    if not json_path.exists():
        raise SystemExit(f"transcript json not found: {json_path}")
    segments, data = _read_segments(json_path)
    review, selection = _review_segments(data, transcript_json=json_path, scope=args.review_scope)
    review = _selected_review_items(review, args.limit)
    audio = None if args.dry_run else _source_audio(args.audio, data, json_path)
    out_dir = Path(args.out).expanduser().resolve() if args.out else json_path.parent / DEFAULT_OUT_NAME
    clips_dir = out_dir / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    hotwords = [] if args.dry_run else _load_hotwords(args.hotwords_file)
    backends = _selected_backends(args.backend, args.compare_backends)
    model_overrides = _model_overrides(args.model, backends)
    backend_models = [(backend, model_overrides.get(backend) or default_model_id(backend)) for backend in backends]
    backend = backend_models[0][0]
    model = backend_models[0][1]
    transcriber_cache: dict[tuple[str, str], Any] = {}

    items: list[RecheckItem] = []
    started = time.time()
    for raw in review:
        try:
            index = int(raw.get("index"))
        except Exception:
            continue
        if index < 0 or index >= len(segments):
            continue
        seg = segments[index]
        start = float(raw.get("start") if raw.get("start") is not None else seg.get("start") or 0)
        end = float(raw.get("end") if raw.get("end") is not None else seg.get("end") or start)
        clip_start = max(0.0, start - max(args.pad, 0))
        clip_end = max(end + max(args.pad, 0), clip_start + 0.05)
        clip = clips_dir / f"review_{index:04d}_{int(round(clip_start * 1000)):010d}_{int(round(clip_end * 1000)):010d}.wav"
        current_text = str(raw.get("text") or seg.get("text") or "")
        original_text = str(raw.get("original_text") or seg.get("original_text") or current_text)
        reasons = [str(x) for x in (raw.get("reasons") or [])]
        candidate_text = ""
        candidate_segments = 0
        error = ""
        candidates: list[RecheckCandidate] = []
        if args.dry_run:
            error = "dry-run: 未切音频,未执行本地 ASR"
            candidates = [
                RecheckCandidate(
                    backend=candidate_backend,
                    model=candidate_model,
                    candidate_text="",
                    candidate_segments=0,
                    error=error,
                    current_similarity=None,
                )
                for candidate_backend, candidate_model in backend_models
            ]
        else:
            try:
                assert audio is not None
                _extract_clip(audio, clip, clip_start, clip_end)
                context = _neighbor_text(segments, index, args.context_window)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            if not error:
                for candidate_backend, candidate_model in backend_models:
                    candidate_error = ""
                    candidate_result_text = ""
                    candidate_result_segments = 0
                    try:
                        cache_key = (candidate_backend, candidate_model)
                        transcriber = transcriber_cache.get(cache_key)
                        if transcriber is None:
                            transcriber = make_transcriber(candidate_backend)
                            transcriber_cache[cache_key] = transcriber
                        candidate_result_text, candidate_result_segments = _run_local_asr(
                            clip,
                            backend=candidate_backend,
                            model=candidate_model,
                            language=args.language,
                            hotwords=hotwords,
                            context=context,
                            transcriber=transcriber,
                        )
                    except Exception as exc:  # noqa: BLE001
                        candidate_error = f"{type(exc).__name__}: {exc}"
                    candidates.append(
                        RecheckCandidate(
                            backend=candidate_backend,
                            model=candidate_model,
                            candidate_text=candidate_result_text,
                            candidate_segments=candidate_result_segments,
                            error=candidate_error,
                            current_similarity=_similarity(current_text, candidate_result_text)
                            if not candidate_error
                            else None,
                        )
                    )
                first = candidates[0] if candidates else None
                if first:
                    candidate_text = first.candidate_text
                    candidate_segments = first.candidate_segments
                    error = first.error
        audit = _build_audit(current_text, candidates)
        items.append(
            RecheckItem(
                index=index,
                start=start,
                end=end,
                clip_start=clip_start,
                clip_end=clip_end,
                original_text=original_text,
                current_text=current_text,
                reasons=reasons,
                candidate_text=candidate_text,
                candidate_segments=candidate_segments,
                backend=backend,
                model=model,
                clip_path=clip,
                error=error,
                candidates=candidates,
                audit=audit,
            )
        )

    payload = {
        "mode": "audit_only",
        "policy": "候选只用于审计；不会修改源转录，严重低相似度禁止自动替换正文。",
        "auto_replace_allowed": False,
        "severe_low_similarity_threshold": SEVERE_LOW_SIMILARITY,
        "source_json": str(json_path),
        "source_audio": str(audio) if audio else "",
        "backend": backend,
        "model": model,
        "compare_backends": [
            {"backend": candidate_backend, "model": candidate_model}
            for candidate_backend, candidate_model in backend_models
        ],
        "review_scope": args.review_scope,
        "review_selection": {
            "sources": selection.get("sources") or [],
            "total_segment_count": selection.get("total_segment_count", 0),
            "strong_segment_count": selection.get("strong_segment_count", 0),
            "weak_segment_count": selection.get("weak_segment_count", 0),
            "skipped_weak_count": selection.get("skipped_weak_count", 0),
        },
        "limit": args.limit,
        "dry_run": args.dry_run,
        "pad": args.pad,
        "cost_seconds": time.time() - started,
        "items": [
            {
                "index": item.index,
                "start": item.start,
                "end": item.end,
                "clip_start": item.clip_start,
                "clip_end": item.clip_end,
                "original_text": item.original_text,
                "current_text": item.current_text,
                "reasons": item.reasons,
                "candidate_text": item.candidate_text,
                "candidate_segments": item.candidate_segments,
                "backend": item.backend,
                "model": item.model,
                "clip_path": str(item.clip_path),
                "error": item.error,
                "candidates": [_candidate_to_dict(candidate) for candidate in item.candidates],
                "primary_current": item.current_text,
                "paraformer_candidate": (
                    _candidate_to_dict(candidate)
                    if (candidate := _candidate_for_backend(item.candidates, "funasr"))
                    else None
                ),
                "qwen3_candidate": (
                    _candidate_to_dict(candidate)
                    if (candidate := _candidate_for_backend(item.candidates, "qwen3"))
                    else None
                ),
                "similarities": item.audit.get("similarities") or {},
                "audit": item.audit,
            }
            for item in items
        ],
    }
    json_out = out_dir / "ASR本地疑点复核.json"
    md_out = out_dir / "ASR本地疑点复核.md"
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_out.write_text(_render_report(items, json_path, audio, selection), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_out), "md": str(md_out), "items": len(items)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
