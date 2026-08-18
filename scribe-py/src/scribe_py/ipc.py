"""JSON-RPC 2.0 over stdin/stdout. One JSON per line.

Architecture (since v0.2):
  - **reader thread** reads stdin and pushes parsed requests onto a queue
  - **main loop** processes requests:
    * Control methods (correct_pause / correct_resume / correct_cancel) are handled
      synchronously on the main thread — they just toggle flags on the global CONTROL.
    * Long-running methods (transcribe / correct / polish / ...) are submitted to
      a worker thread pool so the main loop keeps reading control commands while
      a correction is in flight.

Request:
  {"id": 1, "method": "transcribe", "params": {...}}

Response:
  {"id": 1, "result": {...}}
  {"id": 1, "error": {"code": -32000, "message": "..."}}

Progress notification (no id, server → client only):
  {"event": "progress", "method": "transcribe", "data": {...}}
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import traceback
import csv
import difflib
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .control import CONTROL
from .core.audio import analyze_audio_quality_for_asr, find_ffmpeg, find_ffprobe, probe_audio
from .core.asr_quality import build_asr_quality_report
from .core.hotwords import parse_hotword_terms
from .core.selector import (
    backend_unavailable_reason,
    default_backend,
    default_model_id,
    is_apple_silicon,
    make_transcriber,
    resolve_backend,
)
from .core.text_normalizer import _SIMPLIFY_SKIP_KEYS, simplify_chinese_value
from .core.types import Segment, TranscribeOptions, TranscribeResult
from .correctors.openai_compatible import OpenAICompatibleCorrector
from .polishers.article_polisher import ArticlePolisher
from .translators.article_translator import ArticleTranslator

# stdout writes need to be atomic across threads.
_emit_lock = threading.Lock()
_asr_job_lock = threading.Lock()
_correction_job_lock = threading.Lock()


def _emit(obj: dict) -> None:
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with _emit_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def _segments_from_dicts(items: list[dict]) -> list[Segment]:
    return [
        Segment(
            start=float(s["start"]),
            end=float(s["end"]),
            text=s["text"],
            original_text=s.get("original_text"),
            speaker=s.get("speaker") or None,
            sync_cues=s.get("sync_cues") or None,
        )
        for s in items
    ]


_DIARIZATION_SEGMENT_KEYS = frozenset({
    "speaker",
    "speaker_confidence",
    "speaker_votes",
    "speaker_subsegments",
    "speaker_change_points",
    "speaker_cues",
    "speaker_cue_embeddings",
    "speaker_cue_review",
    "speaker_cue_mode",
    "speaker_cue_split",
    "speaker_overlap_risk",
    "overlap_ratio",
    "speaker_overlap_confidence",
    "speaker_overlap_candidates",
    "speaker_resegmented",
    "speaker_resegmentation_review",
    "speaker_handoff_review",
    "speaker_handoff_voice_guard_repaired",
    "voice_pitch_hz",
    "voice_pitch_confidence",
    "voice_band",
    "speaker_handoff_split",
    "speaker_handoff_bridge",
    "speaker_handoff_relabel",
    "speaker_handoff_text_review",
    "speaker_assignment_review",
    "speaker_review_reason",
    "speaker_split_from_index",
    "voice_band_repaired",
    "continuity_repaired",
    "original_speaker",
    "speaker_voiceprint_reidentified",
    "speaker_voiceprint_review",
    "speaker_voiceprint_score",
    "speaker_voiceprint_anchor",
    "speaker_calibrated",
    "speaker_calibration_source",
    "voice_line_refined",
    "voice_line_review",
    "speaker_overlap_ratio",
})


def _segments_with_diarization_speakers(source_segments: list[dict], diarized_segments: list[Any]) -> list[dict]:
    """Copy ASR segments verbatim and attach only diarization speaker labels."""
    out: list[dict] = []
    for idx, src in enumerate(source_segments):
        seg = {
            key: deepcopy(value)
            for key, value in dict(src).items()
            if key not in _DIARIZATION_SEGMENT_KEYS
        }
        if idx < len(diarized_segments):
            diarized = diarized_segments[idx]
            speaker = getattr(diarized, "speaker", None)
            speaker_confidence = getattr(diarized, "speaker_confidence", None)
            speaker_votes = getattr(diarized, "speaker_votes", None)
            speaker_subsegments = getattr(diarized, "speaker_subsegments", None)
            speaker_cue_embeddings = getattr(diarized, "speaker_cue_embeddings", None)
            speaker_change_points = getattr(diarized, "speaker_change_points", None)
            speaker_overlap_risk = getattr(diarized, "speaker_overlap_risk", None)
            overlap_ratio = getattr(diarized, "overlap_ratio", None)
            if overlap_ratio is None:
                overlap_ratio = getattr(diarized, "speaker_overlap_ratio", None)
            speaker_overlap_confidence = getattr(diarized, "speaker_overlap_confidence", None)
            speaker_overlap_candidates = getattr(diarized, "speaker_overlap_candidates", None)
            voice_pitch_hz = getattr(diarized, "voice_pitch_hz", None)
            voice_pitch_confidence = getattr(diarized, "voice_pitch_confidence", None)
            voice_band = getattr(diarized, "voice_band", None)
            if isinstance(diarized, dict):
                if speaker is None:
                    speaker = diarized.get("speaker")
                if speaker_confidence is None:
                    speaker_confidence = diarized.get("speaker_confidence")
                if speaker_votes is None:
                    speaker_votes = diarized.get("speaker_votes")
                if speaker_subsegments is None:
                    speaker_subsegments = diarized.get("speaker_subsegments")
                if speaker_cue_embeddings is None:
                    speaker_cue_embeddings = diarized.get("speaker_cue_embeddings")
                if speaker_change_points is None:
                    speaker_change_points = diarized.get("speaker_change_points")
                if speaker_overlap_risk is None:
                    speaker_overlap_risk = diarized.get("speaker_overlap_risk")
                if overlap_ratio is None:
                    overlap_ratio = diarized.get("overlap_ratio")
                if overlap_ratio is None:
                    overlap_ratio = diarized.get("speaker_overlap_ratio")
                if speaker_overlap_confidence is None:
                    speaker_overlap_confidence = diarized.get("speaker_overlap_confidence")
                if speaker_overlap_candidates is None:
                    speaker_overlap_candidates = diarized.get("speaker_overlap_candidates")
                if voice_pitch_hz is None:
                    voice_pitch_hz = diarized.get("voice_pitch_hz")
                if voice_pitch_confidence is None:
                    voice_pitch_confidence = diarized.get("voice_pitch_confidence")
                if voice_band is None:
                    voice_band = diarized.get("voice_band")
            if speaker:
                seg["speaker"] = str(speaker)
            if speaker_confidence is not None:
                try:
                    seg["speaker_confidence"] = round(float(speaker_confidence), 3)
                except (TypeError, ValueError):
                    pass
            if isinstance(speaker_votes, dict) and speaker_votes:
                seg["speaker_votes"] = {
                    str(k): round(float(v), 3)
                    for k, v in speaker_votes.items()
                    if v is not None
                }
            if isinstance(speaker_subsegments, list) and speaker_subsegments:
                cleaned_subsegments = []
                for item in speaker_subsegments:
                    if not isinstance(item, dict):
                        continue
                    try:
                        cleaned_subsegments.append({
                            "start": round(float(item.get("start")), 3),
                            "end": round(float(item.get("end")), 3),
                            "speaker": str(item.get("speaker") or ""),
                            "duration": round(float(item.get("duration") or 0.0), 3),
                        })
                    except (TypeError, ValueError):
                        continue
                if cleaned_subsegments:
                    seg["speaker_subsegments"] = cleaned_subsegments
            if isinstance(speaker_cue_embeddings, list) and speaker_cue_embeddings:
                cleaned_cue_embeddings = []
                for item in speaker_cue_embeddings:
                    if not isinstance(item, dict):
                        continue
                    try:
                        cleaned = {
                            "cue_index": int(item.get("cue_index")),
                            "start": round(float(item.get("start")), 3),
                            "end": round(float(item.get("end")), 3),
                            "speaker": str(item.get("speaker") or ""),
                            "score": round(float(item.get("score")), 4),
                            "margin": round(float(item.get("margin")), 4),
                            "voice_coverage_seconds": round(float(item.get("voice_coverage_seconds") or 0.0), 3),
                            "voice_coverage_ratio": round(float(item.get("voice_coverage_ratio") or 0.0), 4),
                            "overlap_ratio": round(float(item.get("overlap_ratio") or 0.0), 4),
                            "decision": str(item.get("decision") or "insufficient"),
                            "source": str(item.get("source") or "campp_sync_cue_embedding"),
                            "embedding_scope": str(item.get("embedding_scope") or "sliding_window_weighted"),
                        }
                        if item.get("second_score") is not None:
                            cleaned["second_score"] = round(float(item.get("second_score")), 4)
                        if item.get("second_speaker"):
                            cleaned["second_speaker"] = str(item.get("second_speaker"))
                        cleaned_cue_embeddings.append(cleaned)
                    except (TypeError, ValueError):
                        continue
                if cleaned_cue_embeddings:
                    seg["speaker_cue_embeddings"] = cleaned_cue_embeddings
            if isinstance(speaker_change_points, list) and speaker_change_points:
                cleaned_points = []
                for point in speaker_change_points:
                    try:
                        cleaned_points.append(round(float(point), 3))
                    except (TypeError, ValueError):
                        continue
                if cleaned_points:
                    seg["speaker_change_points"] = cleaned_points
            if speaker_overlap_risk:
                seg["speaker_overlap_risk"] = True
            if overlap_ratio is not None:
                try:
                    seg["overlap_ratio"] = round(float(overlap_ratio), 4)
                except (TypeError, ValueError):
                    pass
            if speaker_overlap_confidence is not None:
                try:
                    seg["speaker_overlap_confidence"] = round(float(speaker_overlap_confidence), 4)
                except (TypeError, ValueError):
                    pass
            if isinstance(speaker_overlap_candidates, list) and speaker_overlap_candidates:
                cleaned_overlap_candidates = []
                for item in speaker_overlap_candidates:
                    if not isinstance(item, dict):
                        continue
                    try:
                        start = round(float(item.get("start")), 3)
                        end = round(float(item.get("end")), 3)
                        primary_speaker = str(item.get("primary_speaker") or "")
                        secondary_speaker = str(item.get("secondary_speaker") or "")
                        if end <= start or not primary_speaker or not secondary_speaker:
                            continue
                        cleaned_overlap_candidates.append({
                            "start": start,
                            "end": end,
                            "primary_speaker": primary_speaker,
                            "secondary_speaker": secondary_speaker,
                            "confidence": round(float(item.get("confidence") or 0.0), 4),
                            "window_ratio": round(float(item.get("window_ratio") or 0.0), 4),
                            "context_score": round(float(item.get("context_score") or 0.0), 4),
                            "candidate_score": round(float(item.get("candidate_score") or 0.0), 4),
                            "source": str(item.get("source") or "osd_campp_context_v1"),
                        })
                    except (TypeError, ValueError):
                        continue
                if cleaned_overlap_candidates:
                    seg["speaker_overlap_candidates"] = cleaned_overlap_candidates
            if isinstance(diarized, dict):
                if diarized.get("speaker_assignment_review"):
                    seg["speaker_assignment_review"] = True
                if diarized.get("speaker_review_reason"):
                    seg["speaker_review_reason"] = str(diarized.get("speaker_review_reason") or "")
            if voice_pitch_hz is not None:
                try:
                    seg["voice_pitch_hz"] = round(float(voice_pitch_hz), 1)
                except (TypeError, ValueError):
                    pass
            if voice_pitch_confidence is not None:
                try:
                    seg["voice_pitch_confidence"] = round(float(voice_pitch_confidence), 3)
                except (TypeError, ValueError):
                    pass
            if voice_band:
                seg["voice_band"] = str(voice_band)
        out.append(seg)
    return out


def _simplify_diarization_response(value):
    return simplify_chinese_value(
        value,
        skip_keys={*_SIMPLIFY_SKIP_KEYS, "segments"},
    )


def _has_complete_speaker_assignment(source_segments: list[dict], output_segments: list[dict]) -> bool:
    return _preserves_transcript_partition(source_segments, output_segments) and all(
        str(segment.get("speaker") or "").strip()
        for segment in output_segments
    )


def _preserves_transcript_geometry(source_segments: list[dict], output_segments: list[dict]) -> bool:
    return len(source_segments) == len(output_segments) and all(
        source.get("text") == output.get("text")
        and source.get("start") == output.get("start")
        and source.get("end") == output.get("end")
        and source.get("sync_cues") == output.get("sync_cues")
        for source, output in zip(source_segments, output_segments)
    )


def _finalize_speaker_metadata_only(
    source_segments: list[dict], candidate_segments: list[dict]
) -> list[dict] | None:
    """Rebuild output from frozen ASR rows and a strict speaker-metadata whitelist."""
    if not _preserves_transcript_geometry(source_segments, candidate_segments):
        return None

    output: list[dict] = []
    for source, candidate in zip(source_segments, candidate_segments):
        row = {
            key: deepcopy(value)
            for key, value in source.items()
            if key not in _DIARIZATION_SEGMENT_KEYS
        }
        for key in _DIARIZATION_SEGMENT_KEYS:
            if key in candidate:
                row[key] = deepcopy(candidate[key])
        output.append(row)
    return output


def _preserves_transcript_partition(source_segments: list[dict], output_segments: list[dict]) -> bool:
    """Allow cue-boundary splits while freezing text, cue data, and coverage."""
    if not source_segments:
        return not output_segments
    if not output_segments:
        return False

    output_index = 0
    tolerance = 1e-6
    for source in source_segments:
        try:
            source_start = float(source.get("start"))
            source_end = float(source.get("end"))
        except (TypeError, ValueError):
            return False
        if source_end < source_start or output_index >= len(output_segments):
            return False

        pieces: list[dict] = []
        expected_start = source_start
        while output_index < len(output_segments):
            piece = output_segments[output_index]
            try:
                piece_start = float(piece.get("start"))
                piece_end = float(piece.get("end"))
            except (TypeError, ValueError):
                return False
            if (
                piece_end <= piece_start
                or abs(piece_start - expected_start) > tolerance
                or piece_end > source_end + tolerance
            ):
                return False
            pieces.append(piece)
            output_index += 1
            expected_start = piece_end
            if abs(piece_end - source_end) <= tolerance:
                break
        if not pieces or abs(expected_start - source_end) > tolerance:
            return False
        if "".join(str(piece.get("text") or "") for piece in pieces) != str(source.get("text") or ""):
            return False

        source_cues = source.get("sync_cues")
        if len(pieces) > 1 and (not isinstance(source_cues, list) or not source_cues):
            return False
        output_cues: list[dict] = []
        for piece in pieces:
            raw = piece.get("sync_cues")
            if isinstance(raw, list):
                if len(pieces) > 1:
                    if not raw:
                        return False
                    try:
                        first_cue_start = float(raw[0].get("start"))
                        last_cue_end = float(raw[-1].get("end"))
                    except (AttributeError, TypeError, ValueError):
                        return False
                    if (
                        abs(first_cue_start - float(piece.get("start"))) > tolerance
                        or abs(last_cue_end - float(piece.get("end"))) > tolerance
                    ):
                        return False
                output_cues.extend(raw)
            elif raw is not None:
                return False
        if source_cues is None:
            if output_cues:
                return False
        elif output_cues != source_cues:
            return False

    return output_index == len(output_segments)


def _make_progress(method: str):
    def cb(data: dict) -> None:
        _emit({"event": "progress", "method": method, "data": data})
    return cb


RISK_RANK = {"low": 0, "medium": 1, "unknown": 2, "high": 3}
SENSEVOICE_MODEL_ID = "iic/SenseVoiceSmall"
_PYIN_PITCH_CACHE: dict[tuple[str, int, int, float, float], tuple[float | None, float, str]] = {}


def _compatible_asr_model_id(backend: str, model_id: str | None) -> str:
    """Keep persisted old MLX model ids from leaking into FunASR/SenseVoice."""
    selected = model_id or default_model_id(backend)
    if backend in {"sensevoice", "funasr"} and selected.startswith("mlx-community/"):
        return default_model_id(backend)
    return selected


def _asr_text_chars(result: TranscribeResult) -> int:
    return sum(len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", seg.text or "")) for seg in result.segments)


def _choose_preflight_windows(duration: float, *, clip_seconds: float, max_clips: int) -> list[tuple[float, float]]:
    if duration <= 0:
        return [(0.0, clip_seconds)]
    if duration <= clip_seconds:
        return [(0.0, duration)]
    max_clips = max(1, max_clips)
    anchors = [0.2, 0.72] if max_clips <= 2 else [0.12, 0.5, 0.82]
    windows: list[tuple[float, float]] = []
    seen: set[int] = set()
    for anchor in anchors[:max_clips]:
        start = min(max(duration * anchor - (clip_seconds / 2), 0.0), max(duration - clip_seconds, 0.0))
        key = int(round(start))
        if key in seen:
            continue
        seen.add(key)
        windows.append((start, min(clip_seconds, duration - start)))
    return windows or [(0.0, min(clip_seconds, duration))]


def _extract_preflight_clip(audio: Path, out: Path, start: float, duration: float) -> Path:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            ffmpeg,
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
    if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
        return out
    out.unlink(missing_ok=True)
    wav = out.with_suffix(".wav")
    proc = subprocess.run(
        [
            ffmpeg,
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
            str(wav),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not wav.exists() or wav.stat().st_size <= 44:
        raise RuntimeError((proc.stderr or "ffmpeg produced no clip").strip())
    return wav


def _preflight_sample_row(
    *,
    transcriber: Any,
    clip: Path,
    backend: str,
    model_id: str,
    mode: str,
    language: str,
    hotwords: list[str],
) -> dict[str, Any]:
    import time

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
        result = transcriber.transcribe(
            clip,
            TranscribeOptions(
                language=language or "zh",
                model_id=model_id,
                hotwords=hotwords,
                audio_preprocess=mode,
            ),
        )
        quality = build_asr_quality_report(
            result.segments,
            hotwords=hotwords,
            text_normalization=(result.filter_stats or {}).get("text_normalization") or {},
            model_review=(result.filter_stats or {}).get("strong_asr") or {},
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
                "preprocess_mode": ((result.filter_stats or {}).get("audio_standardization") or {}).get("mode") or mode,
                "preprocess_fallback": bool(((result.filter_stats or {}).get("audio_standardization") or {}).get("fallback_applied")),
                "preprocess_filters": ((result.filter_stats or {}).get("audio_standardization") or {}).get("applied_filters") or [],
                "review_count": int(review.get("segment_count") or 0),
                "strong_review_count": int(review.get("strong_segment_count") or 0),
                "term_candidate_count": int(terms.get("candidate_count") or 0),
                "chars": _asr_text_chars(result),
                "segments": len(result.segments),
                "punctuation_ratio": float(quality.get("punctuation_ratio") or 0.0),
                "rtf": float(result.rtf or 0.0),
            }
        )
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["cost_seconds"] = time.time() - started
    return row


def _aggregate_preflight_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return {
            "mode": rows[0].get("mode") if rows else "",
            "status": "error",
            "score_key": [999, 999, 999, 999, 999, 999.0],
            "ok_count": 0,
            "error_count": len(rows),
        }
    risk = max(RISK_RANK.get(str(row.get("risk_level") or "unknown"), 2) for row in ok_rows)
    total_strong = sum(int(row.get("strong_review_count") or 0) for row in ok_rows)
    total_review = sum(int(row.get("review_count") or 0) for row in ok_rows)
    total_terms = sum(int(row.get("term_candidate_count") or 0) for row in ok_rows)
    total_chars = sum(int(row.get("chars") or 0) for row in ok_rows)
    avg_rtf = sum(float(row.get("rtf") or 0.0) for row in ok_rows) / len(ok_rows)
    avg_punctuation = sum(float(row.get("punctuation_ratio") or 0.0) for row in ok_rows) / len(ok_rows)
    fallback_count = sum(1 for row in ok_rows if row.get("preprocess_fallback"))
    applied_filters = sorted({str(item) for row in ok_rows for item in (row.get("preprocess_filters") or [])})
    punctuation_penalty = int(round((1.0 - avg_punctuation) * 100))
    score_key = [risk, total_strong, total_review, total_terms, 0 if total_chars >= 10 else 1, punctuation_penalty, avg_rtf]
    return {
        "mode": ok_rows[0].get("mode"),
        "status": "ok",
        "risk_level": next((name for name, rank in RISK_RANK.items() if rank == risk), "unknown"),
        "strong_review_count": total_strong,
        "review_count": total_review,
        "term_candidate_count": total_terms,
        "chars": total_chars,
        "avg_punctuation_ratio": round(avg_punctuation, 4),
        "avg_rtf": avg_rtf,
        "preprocess_fallback": fallback_count > 0,
        "preprocess_fallback_count": fallback_count,
        "preprocess_filters": applied_filters,
        "ok_count": len(ok_rows),
        "error_count": len(rows) - len(ok_rows),
        "score_key": score_key,
    }


def _preflight_risk_rank(row: dict[str, Any]) -> int:
    return RISK_RANK.get(str(row.get("risk_level") or "unknown"), 2)


def _audio_needs_destructive_enhancement(audio_quality: dict[str, Any] | None) -> bool:
    """Return whether the quality gate found a reason to allow denoise/bandpass.

    Clipping and channel count are real risks, but local denoise cannot recover
    clipping and may damage already-clear speech.  Keep automatic enhancement
    for objective low-loudness/noise-style cases only.
    """
    quality = audio_quality or {}
    reasons = " ".join(str(x) for x in (quality.get("risk_reasons") or []))
    loudness = quality.get("integrated_lufs")
    snr = quality.get("estimated_snr_db")
    noise_floor = quality.get("noise_floor_dbfs")
    if loudness is not None and float(loudness) < -30:
        return True
    try:
        if snr is not None and float(snr) < 16:
            return True
        if noise_floor is not None and snr is not None and float(noise_floor) > -40 and float(snr) < 20:
            return True
    except (TypeError, ValueError):
        pass
    return any(token in reasons for token in ("整体音量过低", "背景噪声", "信噪比", "噪声"))


def _select_preflight_recommendation(
    mode_summary: list[dict[str, Any]],
    *,
    fallback_mode: str,
    audio_quality: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], str]:
    """Pick a preprocessing mode with ASR-first conservative guardrails."""
    if not mode_summary:
        return fallback_mode, {}, "预检没有有效样本，使用通用安全模式。"

    sorted_summary = sorted(mode_summary, key=lambda row: tuple(row.get("score_key") or [999]))
    by_mode = {str(row.get("mode") or ""): row for row in mode_summary}
    fallback = by_mode.get(fallback_mode)
    best = sorted_summary[0]
    best_mode = str(best.get("mode") or fallback_mode)

    if best_mode == fallback_mode:
        return best_mode, best, (
            f"预检比较后使用 {best_mode}：强疑点 {best.get('strong_review_count', 0)}、"
            f"疑点 {best.get('review_count', 0)}。"
        )

    if not fallback or fallback.get("status") != "ok":
        return best_mode, best, f"{fallback_mode} 预检不可用，使用样本表现最好的 {best_mode}。"

    # Stronger denoise modes are useful for genuinely bad audio, but can damage
    # clear conversational speech.  Do not let weak review-count differences
    # override the safe mode.
    if best_mode in {"ai_denoise", "enhance"}:
        audio_need = _audio_needs_destructive_enhancement(audio_quality)
        risk_better = _preflight_risk_rank(best) < _preflight_risk_rank(fallback)
        strong_delta = int(fallback.get("strong_review_count") or 0) - int(best.get("strong_review_count") or 0)
        review_delta = int(fallback.get("review_count") or 0) - int(best.get("review_count") or 0)
        required_review_delta = max(3, int(round(int(fallback.get("review_count") or 0) * 0.35)))
        chars_ok = int(best.get("chars") or 0) >= int((fallback.get("chars") or 0) * 0.95)
        punctuation_ok = float(best.get("avg_punctuation_ratio") or 0.0) >= float(fallback.get("avg_punctuation_ratio") or 0.0) - 0.02
        terms_ok = int(best.get("term_candidate_count") or 0) <= int(fallback.get("term_candidate_count") or 0)
        decisive_text_gain = strong_delta >= 2 and review_delta >= required_review_delta
        actually_applied = not bool(best.get("preprocess_fallback"))

        if actually_applied and audio_need and chars_ok and punctuation_ok and terms_ok and (risk_better or decisive_text_gain):
            return best_mode, best, (
                f"音频存在低音量/噪声类风险，且 {best_mode} 样本强疑点减少 {strong_delta}、"
                f"疑点减少 {review_delta}，允许自动使用 {best_mode}。"
            )

        guard_reason = (
            f"预检样本中 {best_mode} 排名靠前，但未达到保守增强阈值；"
            f"{fallback_mode} 强疑点 {fallback.get('strong_review_count', 0)}、"
            f"疑点 {fallback.get('review_count', 0)}，{best_mode} 强疑点 {best.get('strong_review_count', 0)}、"
            f"疑点 {best.get('review_count', 0)}。为优先保证转文字准确率，使用 {fallback_mode}。"
        )
        return fallback_mode, fallback, guard_reason

    return best_mode, best, (
        f"预检比较后使用 {best_mode}：强疑点 {best.get('strong_review_count', 0)}、"
        f"疑点 {best.get('review_count', 0)}。"
    )


def _load_saved_diarization_settings() -> dict:
    candidates = [
        Path.home() / "Library/Application Support/ai.swarmpath.localscribe/settings.json",
        Path.home() / "Library/Application Support/LocalScribe/settings.json",
    ]
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        diar = data.get("diarization")
        if isinstance(diar, dict):
            return diar
    return {}


def _existing_audio_path(value: str | Path) -> Path:
    audio = Path(value).expanduser()
    if audio.is_file():
        return audio
    raise FileNotFoundError(
        f"源音频文件已失效，无法运行说话人分离：{audio}。"
        "如果这是从微信临时目录拖入的历史任务，请重新导入原始录音；"
        "新版会在转录后自动保存一份稳定音频副本。"
    )


_HUMAN_SPEAKER_LABEL_RE = re.compile(r"^(?:SPEAKER_)?([A-Z])$")


def _normalize_human_speaker_label(value: Any) -> str:
    raw = str(value or "").strip().upper()
    match = _HUMAN_SPEAKER_LABEL_RE.match(raw)
    if not match:
        return ""
    return f"SPEAKER_{match.group(1)}"


def _annotation_time_midpoint_seconds(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    match = re.search(
        r"(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*[-–—]\s*(\d{1,2}):(\d{1,2}(?:\.\d+)?)",
        raw,
    )
    if not match:
        return None
    start = int(match.group(1)) * 60 + float(match.group(2))
    end = int(match.group(3)) * 60 + float(match.group(4))
    return (start + end) / 2.0


def _annotation_label_from_row(row: dict[str, Any], previous_label: str = "") -> str:
    for key in (
        "你的标注",
        "正确speaker(只填这里)",
        "正确speaker(你填A/B/C/D或同上一人)",
        "正确speaker",
        "correct_speaker",
        "label",
    ):
        raw = str(row.get(key) or "").strip()
        if raw in {"同上一人", "同上", "上一人", "same", "same speaker", "previous"}:
            return previous_label
        label = _normalize_human_speaker_label(raw)
        if label:
            return label
    return ""


def _normalized_annotation_text(value: Any) -> str:
    return re.sub(
        r"[\s，。！？、；：,.!?;:'\"“”‘’()（）\[\]【】]+",
        "",
        str(value or ""),
    ).lower()


def _read_human_speaker_annotation_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        if path.suffix.lower() == ".json":
            raw_rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw_rows, list):
                return []
            iterable = [row for row in raw_rows if isinstance(row, dict)]
        elif path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                iterable = list(csv.DictReader(f))
        else:
            return []
    except Exception:
        return []

    previous_label = ""
    for row in iterable:
        label = _annotation_label_from_row(row, previous_label)
        if not label:
            continue
        previous_label = label
        midpoint = _annotation_time_midpoint_seconds(row.get("时间") or row.get("time"))
        try:
            index = int(row.get("序号", row.get("index")))
        except (TypeError, ValueError):
            index = None
        rows.append({
            "index": index,
            "midpoint": midpoint,
            "speaker": label,
            "text": str(
                row.get("文本")
                or row.get("当前文本")
                or row.get("text")
                or ""
            ).strip(),
            "source": str(path),
        })
    return rows


def _stem_from_audio_path(audio: Path) -> str:
    parent = audio.parent
    if parent.name == "audio" and parent.parent.name:
        return parent.parent.name
    return audio.stem


def _find_human_speaker_annotation_files(audio: Path) -> list[Path]:
    stem = _stem_from_audio_path(audio)
    roots = [
        Path.home() / "Library/Application Support/LocalScribe/transcripts",
        Path.cwd() / "transcripts",
    ]
    patterns = (
        "*标注*.json",
        "*标注*.csv",
        "*annotation*.json",
        "*annotation*.csv",
    )
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for folder in root.glob(f"{stem}*"):
            if not folder.is_dir():
                continue
            for pattern in patterns:
                for path in folder.glob(pattern):
                    if path in seen or "只需标注" in path.name:
                        continue
                    seen.add(path)
                    out.append(path)
    return sorted(out, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def _apply_historical_human_speaker_annotations(candidate: dict, audio: Path) -> dict:
    if str(os.environ.get("LOCALSCRIBE_REUSE_HUMAN_ANNOTATIONS", "0")).strip().lower() not in {
        "1", "true", "yes", "on",
    }:
        return candidate
    files = _find_human_speaker_annotation_files(audio)
    if not files:
        return candidate
    rows: list[dict[str, Any]] = []
    for path in files[:12]:
        rows.extend(_read_human_speaker_annotation_rows(path))
    if not rows:
        return candidate

    corrected = {
        **candidate,
        "segments": [dict(seg) for seg in candidate.get("segments") or []],
        "stats": dict(candidate.get("stats") or {}),
    }
    segments = corrected["segments"]
    changed = 0
    matched = 0
    applied_rows = []
    used_indexes: set[int] = set()
    for row in rows:
        idx = None
        midpoint = row.get("midpoint")
        annotation_text = _normalized_annotation_text(row.get("text"))
        alignment = ""
        if annotation_text:
            exact_matches = [
                pos for pos, seg in enumerate(segments)
                if pos not in used_indexes
                and _normalized_annotation_text(seg.get("text")) == annotation_text
            ]
            if exact_matches:
                idx = min(
                    exact_matches,
                    key=lambda pos: abs(
                        ((float(segments[pos].get("start") or 0.0) + float(segments[pos].get("end") or 0.0)) / 2.0)
                        - float(midpoint or 0.0)
                    ),
                )
                alignment = "text_exact"
            else:
                fuzzy_matches: list[tuple[float, int]] = []
                for pos, seg in enumerate(segments):
                    if pos in used_indexes:
                        continue
                    current_text = _normalized_annotation_text(seg.get("text"))
                    if not current_text:
                        continue
                    similarity = difflib.SequenceMatcher(None, annotation_text, current_text).ratio()
                    if similarity >= 0.88:
                        fuzzy_matches.append((similarity, pos))
                if fuzzy_matches:
                    fuzzy_matches.sort(reverse=True)
                    best_similarity, best_pos = fuzzy_matches[0]
                    runner_up = fuzzy_matches[1][0] if len(fuzzy_matches) > 1 else 0.0
                    if best_similarity - runner_up >= 0.03:
                        idx = best_pos
                        alignment = f"text_fuzzy:{best_similarity:.3f}"
        # Text-bearing annotations must never fall back to a stale timestamp or
        # index. Historical ASR revisions can shift the timeline by many seconds.
        elif str(os.environ.get("LOCALSCRIBE_ALLOW_UNTEXTED_ANNOTATIONS", "")).lower() in {"1", "true", "yes"}:
            if midpoint is not None:
                idx = next(
                    (
                        pos for pos, seg in enumerate(segments)
                        if pos not in used_indexes
                        and float(seg.get("start") or 0.0) - 0.35 <= float(midpoint) <= float(seg.get("end") or 0.0) + 0.35
                    ),
                    None,
                )
            if idx is None:
                idx = row.get("index")
            alignment = "legacy_time_or_index"
        if idx is None or idx in used_indexes or idx < 0 or idx >= len(segments):
            continue
        label = str(row.get("speaker") or "")
        if not label:
            continue
        used_indexes.add(int(idx))
        seg = segments[int(idx)]
        previous = str(seg.get("speaker") or "")
        if previous == label:
            matched += 1
        else:
            changed += 1
            seg["original_speaker"] = previous
            seg["speaker"] = label
            seg["speaker_calibrated"] = True
            seg["speaker_calibration_source"] = "historical_human_annotation"
            seg["speaker_assignment_review"] = False
            seg["speaker_review_reason"] = "已按历史人工标注校准"
        applied_rows.append({
            "index": int(idx),
            "original_speaker": previous,
            "correct_speaker": label,
            "changed": previous != label,
            "alignment": alignment,
            "source": row.get("source"),
        })

    if not applied_rows:
        return candidate
    corrected["summary"] = _speaker_summary(segments)
    stats = dict(corrected.get("stats") or {})
    stats["historical_human_annotation_count"] = len(applied_rows)
    stats["historical_human_annotation_matched"] = matched
    stats["historical_human_annotation_changed"] = changed
    stats["historical_human_annotation_sources"] = sorted({str(row.get("source")) for row in applied_rows if row.get("source")})
    corrected["stats"] = stats
    corrected["human_annotation_reuse"] = {
        "mode": "historical_sparse_anchor",
        "annotation_count": len(applied_rows),
        "matched_before": matched,
        "changed": changed,
        "rows": applied_rows,
    }
    return corrected


# ---- method handlers ----

def handle_check_model(params: dict) -> dict:
    """检查模型是否就绪。

    解析顺序与 transcriber_mlx._resolve_model_path 保持一致:
      1. LOCALSCRIBE_MODEL_DIR (环境变量)
      2. <project_root>/models/<basename>/weights.safetensors
      3. HF cache: ~/.cache/huggingface/hub/models--<org>--<name>/
    返回字段额外暴露 `expected_local_path`,供前端引导用户放置文件。
    """
    backend = params.get("backend", "auto")
    backend = resolve_backend(backend)
    if backend == "auto":
        backend = default_backend()
    model_id = _compatible_asr_model_id(backend, params.get("model_id"))
    basename = model_id.rsplit("/", 1)[-1]

    if backend in {"funasr", "sensevoice"}:
        from .core.transcriber_funasr import modelscope_cache_candidates

        candidates = modelscope_cache_candidates(model_id)
        for cache_dir in candidates:
            if cache_dir.exists():
                return {
                    "backend": backend, "model_id": model_id,
                    "exists": True, "source": "modelscope_cache",
                    "path": str(cache_dir),
                    "expected_local_path": str(cache_dir),
                }
        return {
            "backend": backend, "model_id": model_id,
            "exists": False, "source": None,
            "path": None,
            "expected_local_path": str(candidates[0]) if candidates else None,
        }

    # ---- 0. 打包后的 .app 资源目录 (Rust 注入 LOCALSCRIBE_RESOURCES) ----
    res = os.environ.get("LOCALSCRIBE_RESOURCES")
    if res:
        bundled = Path(res) / "models" / basename
        if (bundled / "weights.safetensors").exists():
            return {
                "backend": backend, "model_id": model_id,
                "exists": True, "source": "bundle",
                "path": str(bundled),
                "expected_local_path": str(bundled),
            }

    # ---- 1. 环境变量 ----
    env_dir = os.environ.get("LOCALSCRIBE_MODEL_DIR")
    if env_dir:
        p = Path(env_dir).expanduser()
        candidate = p if (p / "weights.safetensors").exists() else (p / basename)
        if (candidate / "weights.safetensors").exists():
            return {
                "backend": backend, "model_id": model_id,
                "exists": True, "source": "env",
                "path": str(candidate),
                "expected_local_path": str(candidate),
            }

    # ---- 2. 项目内 models/ (dev 模式) ----
    here = Path(__file__).resolve()
    project_models = None
    for ancestor in here.parents:
        if (ancestor / "scribe-py").is_dir() and (ancestor / "package.json").is_file():
            project_models = ancestor / "models" / basename
            break
    if project_models and (project_models / "weights.safetensors").exists():
        return {
            "backend": backend, "model_id": model_id,
            "exists": True, "source": "project",
            "path": str(project_models),
            "expected_local_path": str(project_models),
        }

    # ---- 3. HF cache ----
    repo_dir = model_id.replace("/", "--")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_dir}"
    if cache_dir.exists():
        return {
            "backend": backend, "model_id": model_id,
            "exists": True, "source": "hf_cache",
            "path": str(cache_dir),
            "expected_local_path": str(project_models) if project_models else None,
        }

    return {
        "backend": backend, "model_id": model_id,
        "exists": False, "source": None,
        "path": None,
        "expected_local_path": str(project_models) if project_models else None,
    }


def handle_probe_audio(params: dict) -> dict:
    audio = params["audio"]
    info = probe_audio(audio)
    return {
        "audio": audio,
        "ffmpeg": find_ffmpeg(),
        "ffprobe": find_ffprobe(),
        **info,
    }


def handle_environment(params: dict) -> dict:
    backend = default_backend()
    from .diarizers import available_engines

    diarization_engines_available = available_engines()
    if "senko" in diarization_engines_available:
        diarization_engine = "senko"
        diarization_fallback_reason = None
    else:
        diarization_engine = "resemblyzer"
        diarization_fallback_reason = "senko module is not installed"
    return {
        "apple_silicon": is_apple_silicon(),
        "default_backend": backend,
        "ffmpeg": find_ffmpeg(),
        "ffprobe": find_ffprobe(),
        "default_model_id": default_model_id(backend),
        "diarization_engine": diarization_engine,
        "diarization_engines_available": diarization_engines_available,
        "diarization_fallback_reason": diarization_fallback_reason,
    }


def _app_normalizer_profile(_params: dict) -> None:
    """Keep customer App transcription independent of benchmark profiles."""
    return None


def handle_transcribe(params: dict) -> dict:
    audio = params["audio"]
    requested_backend = params.get("backend", "auto")
    fallback_reason = backend_unavailable_reason(requested_backend)
    backend = resolve_backend(requested_backend)
    if backend == "auto":
        backend = default_backend()
        model_id = default_model_id(backend)
    else:
        model_id = _compatible_asr_model_id(backend, params.get("model_id"))
    hotwords = parse_hotword_terms(params.get("hotwords", ""))
    options = TranscribeOptions(
        language=params.get("language", "zh"),
        model_id=model_id,
        initial_prompt=params.get("initial_prompt", ""),
        hotwords=hotwords,
        word_timestamps=bool(params.get("word_timestamps", False)),
        timing_align=params.get("timing_align") if isinstance(params.get("timing_align"), bool) else None,
        normalizer_profile=_app_normalizer_profile(params),
        audio_preprocess=params.get("audio_preprocess") or "adaptive",
    )
    transcriber = make_transcriber(backend)
    result = transcriber.transcribe(audio, options, on_progress=_make_progress("transcribe"))
    quality_mode = str(params.get("asr_quality_mode") or "standard").strip().lower()
    if quality_mode not in {"standard", "strong"}:
        quality_mode = "standard"
    from .core.strong_asr import decide_auto_high_noise_review

    auto_review = decide_auto_high_noise_review(
        quality_mode=quality_mode,
        backend=result.backend,
        model_id=result.model_id,
        audio_quality=(result.filter_stats or {}).get("audio_quality") or {},
        transcription_stats=result.filter_stats or {},
    )
    # Standard mode remains SenseVoice-first. Only a few generic structural
    # anomalies in a high-noise recording can trigger bounded local consensus;
    # there are no coverage probes or recording-specific phrase rules here.
    standard_review_windows = []
    standard_review_selection = {}
    if quality_mode == "standard" and bool(auto_review.get("recommended")):
        from .core.strong_asr import select_standard_selective_review_windows

        standard_review_windows, standard_review_selection = (
            select_standard_selective_review_windows(result.segments, max_windows=3)
        )
    should_review = quality_mode == "strong" or bool(standard_review_windows)
    if should_review:
        import gc
        import time

        strong_started = time.monotonic()
        detector_segments: list[Segment] = []
        detector_source = ""
        review_reference_segments: list[Segment] = []
        review_reference_source = ""
        review_reference_is_paraformer = False
        detector_snapshot = getattr(transcriber, "strong_asr_detector_snapshot", None)
        if quality_mode == "strong" and callable(detector_snapshot):
            cached_segments, cached_source, cached_is_paraformer = detector_snapshot()
            review_reference_segments = list(cached_segments)
            review_reference_source = str(cached_source or "")
            review_reference_is_paraformer = bool(cached_is_paraformer)
            # Only Paraformer is an independent confirmation model. A cached
            # SenseVoice wallclock transcript remains useful for playback
            # alignment, but must not count as separate ASR evidence.
            if cached_is_paraformer:
                detector_segments = list(cached_segments)
                detector_source = str(cached_source or "paraformer_timing_anchor")
        del transcriber
        gc.collect()
        audio_stats = (result.filter_stats or {}).get("audio_standardization") or {}
        review_audio = Path(str(audio_stats.get("path") or audio))
        if not review_audio.exists():
            review_audio = Path(audio)
        try:
            from .core.strong_asr import timeline_fingerprint

            original_segments = list(result.segments)
            original_timeline = timeline_fingerprint(original_segments)
            from .core.strong_asr import run_strong_asr_review

            reviewed_segments, strong_stats = run_strong_asr_review(
                review_audio,
                original_segments,
                qwen_audio=Path(audio),
                detector_segments=detector_segments,
                detector_source=detector_source,
                detector_windows=(
                    standard_review_windows if quality_mode == "standard" else None
                ),
                language=result.language or options.language,
                on_progress=_make_progress("transcribe"),
                # Standard mode is a conservative customer-default path:
                # local review may correct an agreed spelling, but never add
                # or remove text whose timing cannot be independently owned.
                allow_length_changing_edits=quality_mode != "standard",
            )
            strong_stats = dict(strong_stats or {})
            if timeline_fingerprint(reviewed_segments) != original_timeline:
                strong_stats.update({
                    "applied": False,
                    "timeline_guard_rejected": True,
                    "reviewer_reason": str(strong_stats.get("reason") or ""),
                    "reason": "timeline_guard_rejected_review",
                    "replacement_count": 0,
                    "changes": [],
                })
                result.segments = original_segments
            else:
                strong_stats["timeline_guard_rejected"] = False
                result.segments = reviewed_segments
        except Exception as exc:
            strong_stats = {
                "mode": "local_strong_asr_consensus",
                "enabled": True,
                "applied": False,
                "reason": f"strong_review_failed:{type(exc).__name__}",
                "error": str(exc),
            }
        strong_stats["trigger"] = (
            "standard_selective" if quality_mode == "standard" else "manual_strong"
        )
        if standard_review_selection:
            strong_stats["standard_review_selection"] = standard_review_selection
        strong_stats["review_recommended"] = bool(auto_review.get("recommended"))
        strong_stats["review_strategy"] = "local_strong_asr_consensus"
        strong_stats["auto_review_decision"] = auto_review
        strong_seconds = time.monotonic() - strong_started
        strong_stats["cost_seconds"] = round(strong_seconds, 3)
        result.transcribe_seconds += strong_seconds
        result.rtf = result.transcribe_seconds / result.duration if result.duration else 0.0
        result.filter_stats = {
            **(result.filter_stats or {}),
            "asr_quality_mode": quality_mode,
            "strong_asr": strong_stats,
        }
    else:
        result.filter_stats = {
            **(result.filter_stats or {}),
            "asr_quality_mode": "standard",
            "strong_asr": {
                "mode": "local_strong_asr_consensus",
                "enabled": False,
                "applied": False,
                "reason": str(auto_review.get("reason") or "standard_mode"),
                "trigger": "none",
                "review_recommended": bool(auto_review.get("recommended")),
                "auto_review_decision": auto_review,
                "standard_review_selection": standard_review_selection,
            },
        }
    payload = result.to_dict()
    if fallback_reason:
        payload["requested_backend"] = requested_backend
        filter_stats = dict(payload.get("filter_stats") or {})
        filter_stats["backend_fallback_reason"] = fallback_reason
        payload["filter_stats"] = filter_stats
    payload["asr_quality"] = build_asr_quality_report(
        result.segments,
        hotwords=hotwords,
        text_normalization=(result.filter_stats or {}).get("text_normalization") or {},
        model_review=(result.filter_stats or {}).get("strong_asr") or {},
        audio_quality=(result.filter_stats or {}).get("audio_quality") or {},
        audio_preprocessing=(result.filter_stats or {}).get("audio_standardization") or {},
        backend=result.backend,
        model_id=result.model_id,
        duration=result.duration,
        transcribe_seconds=result.transcribe_seconds,
        rtf=result.rtf,
    )
    return simplify_chinese_value(payload)


def handle_asr_preflight_select(params: dict) -> dict:
    """Recommend an ASR audio preprocessing mode before full transcription.

    Keep the App path conservative: cheap audio quality gate first; only high-risk
    recordings pay for short ASR samples.  The output is advisory and the caller
    can safely fall back to `adaptive` if this method fails.
    """
    audio = Path(params["audio"]).expanduser()
    if not audio.is_file():
        raise FileNotFoundError(f"audio not found: {audio}")
    requested_backend = params.get("backend", "auto")
    backend = resolve_backend(requested_backend)
    if backend == "auto":
        backend = default_backend()
        model_id = default_model_id(backend)
    else:
        model_id = _compatible_asr_model_id(backend, params.get("model_id"))
    language = params.get("language", "zh")
    hotwords = parse_hotword_terms(params.get("hotwords", ""))
    preferred = str(params.get("preferred_mode") or "adaptive").strip().lower()
    if preferred in {"off", "standard", "adaptive", "ai_denoise", "enhance"}:
        fallback_mode = preferred
    else:
        fallback_mode = "adaptive"

    probe = {
        "audio": str(audio),
        "ffmpeg": find_ffmpeg(),
        "ffprobe": find_ffprobe(),
        **probe_audio(audio),
    }
    duration = float(probe.get("duration") or 0.0)
    audio_quality = analyze_audio_quality_for_asr(audio)
    audio_risk = str(audio_quality.get("risk_level") or "unknown")
    max_clips = max(1, min(3, int(params.get("max_clips") or 2)))
    clip_seconds = max(8.0, min(35.0, float(params.get("clip_seconds") or 25.0)))
    force = bool(params.get("force", False))

    if not force and audio_risk in {"low", "medium"} and not _audio_needs_destructive_enhancement(audio_quality):
        reason = "音频质量门禁未发现高风险，本次不做额外抽样，直接使用通用安全模式。"
        return {
            "mode": "asr_preflight_select",
            "audio": str(audio),
            "backend": backend,
            "model": model_id,
            "duration_s": duration,
            "probe": probe,
            "audio_quality": audio_quality,
            "skipped": True,
            "skip_reason": reason,
            "windows": [],
            "sample_rows": [],
            "mode_summary": [
                {
                    "mode": fallback_mode,
                    "status": "skipped",
                    "risk_level": audio_risk,
                    "reason": reason,
                }
            ],
            "recommended_mode": fallback_mode,
            "recommendation_reason": reason,
        }

    requested_modes = params.get("modes")
    if isinstance(requested_modes, str):
        modes = [part.strip().lower() for part in requested_modes.replace("，", ",").split(",") if part.strip()]
    elif isinstance(requested_modes, list):
        modes = [str(part).strip().lower() for part in requested_modes if str(part).strip()]
    else:
        modes = ["adaptive", "ai_denoise", "enhance"]
    modes = [mode for mode in modes if mode in {"off", "standard", "adaptive", "ai_denoise", "enhance"}]
    if not modes:
        modes = ["adaptive", "ai_denoise", "enhance"]
    if fallback_mode in {"standard", "adaptive", "ai_denoise", "enhance"} and fallback_mode not in modes:
        modes.insert(0, fallback_mode)

    windows = _choose_preflight_windows(duration, clip_seconds=clip_seconds, max_clips=max_clips)
    progress = _make_progress("asr_preflight_select")
    sample_rows: list[dict[str, Any]] = []
    window_payload: list[dict[str, Any]] = []
    transcriber = make_transcriber(backend)
    with tempfile.TemporaryDirectory(prefix="localscribe-preflight-") as tmp_dir:
        clips_dir = Path(tmp_dir)
        clips: list[dict[str, Any]] = []
        for idx, (start, clip_duration) in enumerate(windows, start=1):
            clip = _extract_preflight_clip(
                audio,
                clips_dir / f"clip_{idx:02d}_{int(start):06d}_{int(start + clip_duration):06d}.m4a",
                start,
                clip_duration,
            )
            item = {"index": idx, "start": start, "end": start + clip_duration, "duration": clip_duration, "path": str(clip)}
            clips.append({**item, "path_obj": clip})
            window_payload.append(item)

        total = max(len(modes) * len(clips), 1)
        current = 0
        for mode in modes:
            for item in clips:
                current += 1
                progress({
                    "current": current - 1,
                    "total": total,
                    "preview": f"预检 {mode} · 样本 {item['index']}/{len(clips)}",
                })
                row = _preflight_sample_row(
                    transcriber=transcriber,
                    clip=item["path_obj"],
                    backend=backend,
                    model_id=model_id,
                    mode=mode,
                    language=language,
                    hotwords=hotwords,
                )
                row.update({
                    "clip_index": item["index"],
                    "clip_start": item["start"],
                    "clip_duration": item["duration"],
                })
                sample_rows.append(row)
                progress({
                    "current": current,
                    "total": total,
                    "preview": f"预检 {mode} 完成 · 风险 {row.get('risk_level', 'unknown')}",
                })

    mode_summary = [_aggregate_preflight_mode([row for row in sample_rows if row.get("mode") == mode]) for mode in modes]
    mode_summary.sort(key=lambda row: tuple(row.get("score_key") or [999]))
    recommended, best, reason = _select_preflight_recommendation(
        mode_summary,
        fallback_mode=fallback_mode,
        audio_quality=audio_quality,
    )
    return {
        "mode": "asr_preflight_select",
        "audio": str(audio),
        "backend": backend,
        "model": model_id,
        "duration_s": duration,
        "probe": probe,
        "audio_quality": audio_quality,
        "skipped": False,
        "skip_reason": "",
        "windows": window_payload,
        "sample_rows": sample_rows,
        "mode_summary": mode_summary,
        "recommended_mode": recommended,
        "recommendation_reason": reason,
    }


def handle_correct(params: dict) -> dict:
    # Reset shared control before starting a fresh run.
    CONTROL.reset()
    segments = _segments_from_dicts(params["segments"])
    corrector = OpenAICompatibleCorrector(
        api_key=params["api_key"],
        base_url=params.get("base_url", "https://api.deepseek.com"),
        model=params.get("model", "deepseek-v4-flash"),
        mode=params.get("mode", "medium"),
        batch_size=int(params.get("batch_size", 30)),
        temperature=float(params.get("temperature", 0.1)),
        max_tokens=int(params.get("max_tokens", 8192)),
        top_p=float(params.get("top_p", 1.0)),
        frequency_penalty=float(params.get("frequency_penalty", 0.0)),
        presence_penalty=float(params.get("presence_penalty", 0.0)),
        use_glossary=bool(params.get("use_glossary", True)),
        concurrency=int(params.get("concurrency", 15)),
        language=params.get("language"),
    )
    out = corrector.correct(
        segments,
        context_hint=params.get("context_hint", ""),
        on_progress=_make_progress("correct"),
        control=CONTROL,
    )
    changed = sum(1 for s in out if s.text != (s.original_text or s.text))
    return simplify_chinese_value({
        "segments": [s.to_dict() for s in out],
        "changed": changed,
        "total": len(out),
        "model": corrector.model,
        "mode": corrector.mode,
        "glossary": corrector.last_glossary,
        "cancelled": corrector.last_cancelled,
        "concurrency": corrector.concurrency,
    })


def handle_polish(params: dict) -> dict:
    segments = _segments_from_dicts(params["segments"])
    polisher = ArticlePolisher(
        api_key=params["api_key"],
        base_url=params.get("base_url", "https://api.deepseek.com"),
        model=params.get("model", "deepseek-v4-flash"),
        temperature=float(params.get("temperature", 0.3)),
        max_tokens=int(params.get("max_tokens", 384000)),
        top_p=float(params.get("top_p", 1.0)),
        frequency_penalty=float(params.get("frequency_penalty", 0.0)),
        presence_penalty=float(params.get("presence_penalty", 0.0)),
    )
    out = polisher.polish(segments, on_progress=_make_progress("polish"))
    text = out.get("text", "")
    return simplify_chinese_value({
        "text": text,
        "model": polisher.model,
        "char_count": len(text),
        "finish_reason": out.get("finish_reason", "stop"),
        "truncated": out.get("truncated", False),
        "input_chars": out.get("input_chars", 0),
        "chunks": out.get("chunks", 1),
        "mode": out.get("mode", "monologue"),  # "monologue" | "dialogue"
    })


def handle_translate_article(params: dict) -> dict:
    """Translate article text to target language.

    Args:
        params: dict with keys:
            - text: article text to translate
            - source_language: source language (optional, for reference)
            - target_language: target language code ("zh", "en", "ja", "ko")
            - glossary: optional glossary from correction phase
            - api_key: LLM API key
            - base_url: API base URL
            - model: model name
            - temperature, max_tokens, top_p, frequency_penalty, presence_penalty

    Returns:
        dict with translated text and metadata
    """
    translator = ArticleTranslator(
        api_key=params["api_key"],
        base_url=params.get("base_url", "https://api.deepseek.com"),
        model=params.get("model", "deepseek-v4-flash"),
        temperature=float(params.get("temperature", 0.3)),
        max_tokens=int(params.get("max_tokens", 384000)),
        top_p=float(params.get("top_p", 1.0)),
        frequency_penalty=float(params.get("frequency_penalty", 0.0)),
        presence_penalty=float(params.get("presence_penalty", 0.0)),
    )
    result = translator.translate(
        text=params["text"],
        source_language=params.get("source_language"),
        target_language=params["target_language"],
        glossary=params.get("glossary"),
    )
    return result


# ---- control methods (instant, run on main thread) ----

def handle_correct_pause(_params: dict) -> dict:
    CONTROL.request_pause()
    return {"status": "paused"}


def handle_correct_resume(_params: dict) -> dict:
    CONTROL.request_resume()
    return {"status": "resumed"}


def handle_correct_cancel(_params: dict) -> dict:
    CONTROL.request_cancel()
    return {"status": "cancelling"}


def handle_correct_status(_params: dict) -> dict:
    return {
        "paused": CONTROL.is_paused(),
        "cancelled": CONTROL.is_cancelled(),
    }


# ---- diarization ----

def _repair_long_missing_speaker_cues(candidate: dict, audio: Path) -> dict:
    """Apply optional exact embeddings without relaxing transcript freezes."""
    from .diarizers.exact_embedding_fallback import repair_missing_evidence_cues

    source_segments = candidate.get("segments") or []
    repaired = repair_missing_evidence_cues(
        audio,
        candidate,
        on_progress=_make_progress("diarize"),
    )
    repaired_segments = repaired.get("segments") or []
    if _preserves_transcript_geometry(source_segments, repaired_segments):
        return repaired

    output = {
        **candidate,
        "segments": [dict(segment) for segment in source_segments],
        "stats": dict(candidate.get("stats") or {}),
    }
    output["stats"]["exact_embedding_fallback"] = {
        "available": False,
        "applied": False,
        "method": "campp_exact_missing_evidence_v1",
        "frozen_transcript_geometry": True,
        "reason": "transcript_geometry_guard_rejected_output",
    }
    return output

def handle_diarize(params: dict) -> dict:
    """params: { audio: str, segments: [{start,end,text}], n_speakers: int,
                 profiles: [{name, embedding}] }"""
    from .diarizers import diarize as _diarize
    audio = _existing_audio_path(params["audio"])
    frozen_segments = deepcopy(params.get("segments") or [])
    segments = deepcopy(frozen_segments)
    saved_diar = _load_saved_diarization_settings()
    saved_n_speakers = saved_diar.get("n_speakers", saved_diar.get("nSpeakers"))
    n_speakers_raw = params.get("n_speakers", params.get("nSpeakers"))
    # Tauri commands use camelCase at the JS boundary. Older packaged frontend sent
    # n_speakers, so Rust never received it and filled the default 2 before calling
    # Python. In the app, persisted settings are the source of truth.
    if n_speakers_raw is not None:
        pass
    elif saved_n_speakers is not None:
        n_speakers_raw = saved_n_speakers
    elif n_speakers_raw is None:
        n_speakers_raw = 0
    n_speakers = int(n_speakers_raw)
    engine = params.get("engine") or saved_diar.get("engine") or "auto"
    profiles = params.get("profiles") or []
    preserve_segmentation = bool(
        params.get("preserve_segmentation", params.get("preserveSegmentation", True))
    )

    if n_speakers <= 0:
        rec = _recommend_diarization_candidates(
            audio=audio,
            segments=segments,
            profiles=profiles,
            min_speakers=2,
            max_speakers=8,
            engine=engine,
            progress_method="diarize",
            preserve_segmentation=preserve_segmentation,
        )
        candidate_n = int(
            rec.get("recommended_candidate_n_speakers")
            or rec.get("recommended_n_speakers")
            or 0
        )
        selected = next(
            (
                c for c in rec.get("candidates", [])
                if int(c.get("n_speakers") or 0) == candidate_n
            ),
            (rec.get("candidates") or [None])[0],
        )
        if not selected:
            return {
                "segments": segments,
                "speakers": [],
                "matched_profiles": {},
                "stats": {
                    "embeddings": 0,
                    "duration_s": sum(
                        max(0.0, float(segment.get("end") or 0.0) - float(segment.get("start") or 0.0))
                        for segment in segments
                    ),
                    "clusters": 0,
                    "matched_profile_count": 0,
                    "segment_count": len(segments),
                    "status": "error",
                    "applied": False,
                    "auto": True,
                    "segmentation_preserved": preserve_segmentation,
                    "failure_reason": rec.get("reason") or "没有可应用的分人候选",
                    "errors": rec.get("errors") or [],
                },
            }
        selected_segments = selected.get("segments") or []
        if not _has_complete_speaker_assignment(segments, selected_segments):
            return {
                "segments": segments,
                "speakers": [],
                "matched_profiles": {},
                "stats": {
                    "embeddings": 0,
                    "duration_s": sum(
                        max(0.0, float(segment.get("end") or 0.0) - float(segment.get("start") or 0.0))
                        for segment in segments
                    ),
                    "clusters": 0,
                    "matched_profile_count": 0,
                    "segment_count": len(segments),
                    "status": "error",
                    "applied": False,
                    "auto": True,
                    "segmentation_preserved": preserve_segmentation,
                    "failure_reason": "分人候选没有为每个 transcript segment 提供 speaker",
                    "errors": rec.get("errors") or [],
                },
            }
        selected = _apply_historical_human_speaker_annotations(selected, audio)
        selected = _project_speaker_cues(selected)
        selected = _repair_long_missing_speaker_cues(selected, audio)
        finalized_segments = _finalize_speaker_metadata_only(
            frozen_segments, selected.get("segments") or []
        )
        if finalized_segments is None or not all(
            str(segment.get("speaker") or "").strip()
            for segment in finalized_segments
        ):
            return {
                "segments": deepcopy(frozen_segments),
                "speakers": [],
                "matched_profiles": {},
                "stats": {
                    "status": "error",
                    "applied": False,
                    "auto": True,
                    "segmentation_preserved": True,
                    "failure_reason": "最终分人结果试图改变冻结转录，已拒绝应用",
                    "errors": rec.get("errors") or [],
                },
            }
        selected["segments"] = finalized_segments

        summary_speakers = selected.get("summary", {}).get("speakers", [])
        speakers = [s["speaker"] for s in summary_speakers] or selected.get("speakers", [])
        stats = dict(selected.get("stats") or {})
        confidence = str(rec.get("confidence") or "")
        if confidence == "low":
            stats.update({
                "status": "partial" if rec.get("errors") else "ok",
                "applied": True,
                "errors": rec.get("errors") or [],
                "segmentation_preserved": bool(selected.get("segmentation_preserved", preserve_segmentation)),
                "auto": True,
                "auto_candidate_sweep": True,
                "clusters": len(speakers),
                "requested_n_speakers": 0,
                "recommended_n_speakers": int(rec.get("recommended_n_speakers") or len(speakers)),
                "recommended_run_n_speakers": candidate_n,
                "recommendation_confidence": rec.get("confidence"),
                "recommendation_confidence_reason": rec.get("confidence_reason"),
                "score_gap_to_next": rec.get("score_gap_to_next"),
                "risk_level": "high",
                "risk_reason": "自动分人置信度低，已保留说话人标签但需要抽听确认",
                "recommendation_reason": rec.get("reason") or selected.get("reason") or "",
                "voice_mix_summary": rec.get("voice_mix_summary") or selected.get("voice_mix_summary") or {},
                "mixed_voice_speakers": rec.get("mixed_voice_speakers") or selected.get("mixed_voice_speakers") or [],
                "severe_mixed_voice_speakers": rec.get("severe_mixed_voice_speakers") or selected.get("severe_mixed_voice_speakers") or [],
                "voice_mix_penalty": rec.get("voice_mix_penalty") or selected.get("voice_mix_penalty") or 0.0,
                "voice_line_groups": rec.get("voice_line_groups") or selected.get("voice_line_groups") or _speaker_voice_line_groups(selected.get("segments") or segments),
                "voice_line_refine_count": rec.get("voice_line_refine_count") or selected.get("voice_line_refine_count") or 0,
                "voice_line_review_count": rec.get("voice_line_review_count") or selected.get("voice_line_review_count") or 0,
                "voice_line_refine_reason": rec.get("voice_line_refine_reason") or selected.get("voice_line_refine_reason") or "",
                "review_segments": rec.get("review_segments") or selected.get("review_segments") or [],
            })
            return _simplify_diarization_response({
                "segments": selected.get("segments") or segments,
                "speakers": speakers,
                "matched_profiles": {},
                "stats": stats,
            })
        stats.update({
            "status": "partial" if rec.get("errors") else "ok",
            "applied": True,
            "errors": rec.get("errors") or [],
            "segmentation_preserved": bool(selected.get("segmentation_preserved", preserve_segmentation)),
            "auto": True,
            "auto_candidate_sweep": True,
            "clusters": len(speakers),
            "requested_n_speakers": 0,
            "recommended_n_speakers": int(rec.get("recommended_n_speakers") or len(speakers)),
            "recommended_run_n_speakers": candidate_n,
            "recommendation_confidence": rec.get("confidence"),
            "recommendation_confidence_reason": rec.get("confidence_reason"),
            "score_gap_to_next": rec.get("score_gap_to_next"),
            "risk_reason": (
                selected.get("resegmentation_reason")
                or selected.get("voice_line_refine_reason")
                or selected.get("local_leakage_reason")
                or selected.get("continuity_repair_reason")
                or selected.get("voice_band_repair_reason")
                or selected.get("smoothing_reason")
                or selected.get("short_sandwich_reason")
                or selected.get("reassignment_reason")
                or selected.get("merge_reason")
                or rec.get("reason")
                or selected.get("reason")
                or ""
            ),
            "recommendation_reason": rec.get("reason") or selected.get("reason") or "",
            "merge_map": rec.get("merge_map") or selected.get("merge_map") or {},
            "merge_distribution": rec.get("merge_distribution") or selected.get("merge_distribution") or {},
            "merge_reason": rec.get("merge_reason") or selected.get("merge_reason") or "",
            "reassignment_distribution": rec.get("reassignment_distribution") or selected.get("reassignment_distribution") or {},
            "reassignment_reason": rec.get("reassignment_reason") or selected.get("reassignment_reason") or "",
            "smoothing_distribution": rec.get("smoothing_distribution") or selected.get("smoothing_distribution") or {},
            "smoothing_reason": rec.get("smoothing_reason") or selected.get("smoothing_reason") or "",
            "short_sandwich_distribution": rec.get("short_sandwich_distribution") or selected.get("short_sandwich_distribution") or {},
            "short_sandwich_reason": rec.get("short_sandwich_reason") or selected.get("short_sandwich_reason") or "",
            "local_leakage_distribution": rec.get("local_leakage_distribution") or selected.get("local_leakage_distribution") or {},
            "local_leakage_reason": rec.get("local_leakage_reason") or selected.get("local_leakage_reason") or "",
            "continuity_repair_distribution": rec.get("continuity_repair_distribution") or selected.get("continuity_repair_distribution") or {},
            "continuity_repair_reason": rec.get("continuity_repair_reason") or selected.get("continuity_repair_reason") or "",
            "voice_band_repair_distribution": rec.get("voice_band_repair_distribution") or selected.get("voice_band_repair_distribution") or {},
            "voice_band_repair_reason": rec.get("voice_band_repair_reason") or selected.get("voice_band_repair_reason") or "",
            "voice_profiles": rec.get("voice_profiles") or selected.get("voice_profiles") or {},
            "voice_mix_summary": rec.get("voice_mix_summary") or selected.get("voice_mix_summary") or {},
            "mixed_voice_speakers": rec.get("mixed_voice_speakers") or selected.get("mixed_voice_speakers") or [],
            "severe_mixed_voice_speakers": rec.get("severe_mixed_voice_speakers") or selected.get("severe_mixed_voice_speakers") or [],
            "voice_mix_penalty": rec.get("voice_mix_penalty") or selected.get("voice_mix_penalty") or 0.0,
            "voice_line_groups": rec.get("voice_line_groups") or selected.get("voice_line_groups") or _speaker_voice_line_groups(selected.get("segments") or segments),
            "voice_line_refine_count": rec.get("voice_line_refine_count") or selected.get("voice_line_refine_count") or 0,
            "voice_line_review_count": rec.get("voice_line_review_count") or selected.get("voice_line_review_count") or 0,
            "voice_line_refine_reason": rec.get("voice_line_refine_reason") or selected.get("voice_line_refine_reason") or "",
            "resegmentation_count": rec.get("resegmentation_count") or selected.get("resegmentation_count") or 0,
            "resegmentation_reason": rec.get("resegmentation_reason") or selected.get("resegmentation_reason") or "",
            "voice_guard_count": rec.get("voice_guard_count") or selected.get("voice_guard_count") or 0,
            "voice_guard_reason": rec.get("voice_guard_reason") or selected.get("voice_guard_reason") or "",
            "handoff_split_count": rec.get("handoff_split_count") or selected.get("handoff_split_count") or 0,
            "handoff_split_reason": rec.get("handoff_split_reason") or selected.get("handoff_split_reason") or "",
            "handoff_voice_guard_distribution": rec.get("handoff_voice_guard_distribution") or selected.get("handoff_voice_guard_distribution") or {},
            "handoff_voice_guard_reason": rec.get("handoff_voice_guard_reason") or selected.get("handoff_voice_guard_reason") or "",
            "review_segments": rec.get("review_segments") or selected.get("review_segments") or [],
            "historical_human_annotation_count": (selected.get("stats") or {}).get("historical_human_annotation_count", 0),
            "historical_human_annotation_changed": (selected.get("stats") or {}).get("historical_human_annotation_changed", 0),
            "historical_human_annotation_sources": (selected.get("stats") or {}).get("historical_human_annotation_sources", []),
        })
        return _simplify_diarization_response({
            "segments": selected.get("segments") or segments,
            "speakers": speakers,
            "matched_profiles": selected.get("matched_profiles") or {},
            "stats": stats,
        })

    result = _diarize(
        audio=audio,
        segments=segments,
        n_speakers=n_speakers,
        profiles=profiles,
        engine=engine,
        on_progress=_make_progress("diarize"),
    )
    out_segments = _segments_with_diarization_speakers(segments, result.segments)
    if not _has_complete_speaker_assignment(segments, out_segments):
        return {
            "segments": deepcopy(frozen_segments),
            "speakers": [],
            "matched_profiles": {},
            "stats": {
                "embeddings": int((result.stats or {}).get("embeddings") or 0),
                "duration_s": sum(
                    max(0.0, float(segment.get("end") or 0.0) - float(segment.get("start") or 0.0))
                    for segment in segments
                ),
                "clusters": 0,
                "matched_profile_count": 0,
                "segment_count": len(segments),
                "status": "error",
                "applied": False,
                "segmentation_preserved": preserve_segmentation,
                "failure_reason": "分人结果没有为每个 transcript segment 提供 speaker",
                "errors": [],
            },
        }
    result_stats = dict(result.stats or {})
    # Senko may conservatively override a requested count when the acoustic
    # eigengap evidence strongly rejects it. Downstream scoring and smoothing
    # must use the count that was actually clustered, otherwise a guarded
    # 3-speaker result can be treated as 4 speakers and fragmented again.
    effective_n_speakers = int(
        result_stats.get("model_selected_n_speakers")
        or getattr(result, "cluster_count", 0)
        or len(result.speakers)
        or n_speakers
        or 0
    )
    candidate = {
        "n_speakers": effective_n_speakers,
        "speakers": result.speakers,
        "segments": out_segments,
        "matched_profiles": result.matched_profiles,
        "stats": result_stats,
        "summary": _speaker_summary(out_segments),
    }
    candidate.update(_score_diarization_candidate(candidate, effective_n_speakers))
    candidate = _refine_conflicting_voice_bands_with_pyin(candidate, audio)

    anchors: list[dict] = []
    if candidate.get("mergeable_speakers") and effective_n_speakers > 2:
        try:
            anchor_result = _diarize(
                audio=audio,
                segments=segments,
                n_speakers=2,
                profiles=profiles,
                engine=engine,
                on_progress=_make_progress("diarize"),
            )
            anchor_segments = _segments_with_diarization_speakers(segments, anchor_result.segments)
            anchor = {
                "n_speakers": 2,
                "speakers": anchor_result.speakers,
                "segments": anchor_segments,
                "matched_profiles": anchor_result.matched_profiles,
                "stats": dict(anchor_result.stats or {}),
                "summary": _speaker_summary(anchor_segments),
            }
            anchor.update(_score_diarization_candidate(anchor, 2))
            anchor = _refine_conflicting_voice_bands_with_pyin(anchor, audio)
            anchors.append(anchor)
        except Exception as exc:
            candidate["stats"]["fallback_reason"] = f"错挂片段复核失败: {type(exc).__name__}: {exc}"

    candidate = _postprocess_fixed_count_candidate(
        candidate,
        [*anchors, candidate],
        requested_n=effective_n_speakers,
        preserve_segmentation=preserve_segmentation,
    )
    candidate = _apply_historical_human_speaker_annotations(candidate, audio)
    candidate["review_segments"] = _build_review_segments(candidate)
    candidate = _annotate_segments_with_speaker_reviews(candidate)
    candidate = _project_speaker_cues(candidate)
    candidate = _repair_long_missing_speaker_cues(candidate, audio)
    finalized_segments = _finalize_speaker_metadata_only(
        frozen_segments, candidate.get("segments") or []
    )
    out_segments = finalized_segments or []
    if (
        finalized_segments is None
        or not all(str(segment.get("speaker") or "").strip() for segment in out_segments)
    ):
        return {
            "segments": deepcopy(frozen_segments),
            "speakers": [],
            "matched_profiles": {},
            "stats": {
                "embeddings": int((result.stats or {}).get("embeddings") or 0),
                "duration_s": sum(
                    max(0.0, float(segment.get("end") or 0.0) - float(segment.get("start") or 0.0))
                    for segment in segments
                ),
                "clusters": 0,
                "matched_profile_count": 0,
                "segment_count": len(segments),
                "status": "error",
                "applied": False,
                "segmentation_preserved": preserve_segmentation,
                "failure_reason": "分人后处理改变了 transcript 几何或留下未标注 segment",
                "errors": [],
            },
        }
    candidate["segments"] = out_segments
    summary_speakers = candidate.get("summary", {}).get("speakers", [])
    speakers = [s["speaker"] for s in summary_speakers] or [
        s for s in result.speakers
        if any(seg.get("speaker") == s for seg in out_segments)
    ]
    stats = dict(candidate.get("stats") or {})
    stats.update({
        "status": "ok",
        "applied": True,
        "segmentation_preserved": bool(candidate.get("segmentation_preserved", preserve_segmentation)),
        "clusters": len(speakers),
        "requested_n_speakers": int(n_speakers),
        "selected_score": float(candidate.get("score", 0.0)),
        "risk_level": (
            "high" if candidate.get("severe_mixed_voice_speakers") or candidate.get("fragile_speakers")
            else "medium" if (
                candidate.get("mixed_voice_speakers")
                or candidate.get("merge_map")
                or candidate.get("handoff_split_count")
                or candidate.get("handoff_voice_guard_distribution")
                or candidate.get("local_leakage_distribution")
                or candidate.get("continuity_repair_distribution")
                or candidate.get("voice_band_repair_distribution")
                or candidate.get("voice_guard_count")
                or candidate.get("reassignment_distribution")
                or candidate.get("smoothing_distribution")
                or candidate.get("short_sandwich_distribution")
            )
            else "low"
        ),
        "risk_reason": (
            candidate.get("postprocess_skipped_reason")
            or candidate.get("handoff_voice_guard_reason")
            or candidate.get("handoff_split_reason")
            or candidate.get("voice_line_refine_reason")
            or candidate.get("local_leakage_reason")
            or candidate.get("continuity_repair_reason")
            or candidate.get("voice_band_repair_reason")
            or candidate.get("voice_guard_reason")
            or candidate.get("smoothing_reason")
            or candidate.get("short_sandwich_reason")
            or candidate.get("reassignment_reason")
            or candidate.get("merge_reason")
            or ("存在严重声线混标" if candidate.get("severe_mixed_voice_speakers") else "")
            or ("存在声线混合风险" if candidate.get("mixed_voice_speakers") else "")
            or candidate.get("reason")
            or ""
        ),
        "postprocess_skipped_reason": candidate.get("postprocess_skipped_reason") or "",
        "recommendation_reason": candidate.get("reason") or "",
        "merge_map": candidate.get("merge_map") or {},
        "merge_distribution": candidate.get("merge_distribution") or {},
        "merge_reason": candidate.get("merge_reason") or "",
        "reassignment_distribution": candidate.get("reassignment_distribution") or {},
        "reassignment_reason": candidate.get("reassignment_reason") or "",
        "smoothing_distribution": candidate.get("smoothing_distribution") or {},
        "smoothing_reason": candidate.get("smoothing_reason") or "",
        "short_sandwich_distribution": candidate.get("short_sandwich_distribution") or {},
        "short_sandwich_reason": candidate.get("short_sandwich_reason") or "",
        "local_leakage_distribution": candidate.get("local_leakage_distribution") or {},
        "local_leakage_reason": candidate.get("local_leakage_reason") or "",
        "continuity_repair_distribution": candidate.get("continuity_repair_distribution") or {},
        "continuity_repair_reason": candidate.get("continuity_repair_reason") or "",
        "voice_band_repair_distribution": candidate.get("voice_band_repair_distribution") or {},
        "voice_band_repair_reason": candidate.get("voice_band_repair_reason") or "",
        "voice_profiles": candidate.get("voice_profiles") or stats.get("speaker_voice_summary") or {},
        "voice_mix_summary": candidate.get("voice_mix_summary") or {},
        "mixed_voice_speakers": candidate.get("mixed_voice_speakers") or [],
        "severe_mixed_voice_speakers": candidate.get("severe_mixed_voice_speakers") or [],
        "voice_mix_penalty": candidate.get("voice_mix_penalty") or 0.0,
        "voice_line_groups": candidate.get("voice_line_groups") or _speaker_voice_line_groups(candidate.get("segments") or []),
        "voice_line_refine_count": candidate.get("voice_line_refine_count") or 0,
        "voice_line_review_count": candidate.get("voice_line_review_count") or 0,
        "voice_line_refine_reason": candidate.get("voice_line_refine_reason") or "",
        "resegmentation_count": candidate.get("resegmentation_count") or 0,
        "resegmentation_reason": candidate.get("resegmentation_reason") or "",
        "voice_guard_count": candidate.get("voice_guard_count") or 0,
        "voice_guard_reason": candidate.get("voice_guard_reason") or "",
        "handoff_split_count": candidate.get("handoff_split_count") or 0,
        "handoff_split_reason": candidate.get("handoff_split_reason") or "",
        "handoff_voice_guard_distribution": candidate.get("handoff_voice_guard_distribution") or {},
        "handoff_voice_guard_reason": candidate.get("handoff_voice_guard_reason") or "",
        "review_segments": candidate.get("review_segments") or [],
    })
    return _simplify_diarization_response({
        "segments": out_segments,
        "speakers": speakers,
        "matched_profiles": result.matched_profiles,
        "stats": stats,
    })


def _candidate_actual_count(candidate: dict) -> int:
    return len(candidate.get("summary", {}).get("speakers") or [])


def _postprocess_fixed_count_candidate(
    candidate: dict,
    anchors: list[dict] | None,
    requested_n: int,
    preserve_segmentation: bool = False,
) -> dict:
    """Post-process fixed-count diarization without silently reducing speakers."""
    original_count = _candidate_actual_count(candidate)
    if preserve_segmentation:
        corrected = {
            **candidate,
            "segments": [dict(seg) for seg in candidate.get("segments") or []],
            "stats": dict(candidate.get("stats") or {}),
            "resegmentation_count": 0,
            "resegmentation_reason": "",
            "handoff_split_count": 0,
            "handoff_split_reason": "",
            "segmentation_preserved": True,
        }
    else:
        corrected = _resegment_mixed_speaker_segments(candidate)
        corrected = _split_handoff_segments(corrected)
        corrected["segmentation_preserved"] = False
    corrected = _repair_handoff_voice_guard_assignments(corrected)
    corrected = _reassign_isolated_fragile_segments(corrected, anchors)
    corrected = _smooth_short_sandwiched_segments(corrected)
    corrected = _smooth_windowed_sandwiched_runs(corrected)
    corrected = _smooth_alternating_local_speaker_leakage(corrected)
    corrected = _repair_discourse_continuity_assignments(corrected)
    corrected = _repair_voice_band_assignments(corrected)
    if preserve_segmentation:
        corrected = _project_speaker_cues(corrected)
    corrected_count = _candidate_actual_count(corrected)
    requested_floor = min(max(1, int(requested_n)), original_count)

    if int(requested_n) > 0 and corrected_count < requested_floor:
        preserved = {
            **candidate,
            "segments": [dict(seg) for seg in candidate.get("segments") or []],
            "stats": dict(candidate.get("stats") or {}),
            "summary": candidate.get("summary") or {},
        }
        preserved["reassignment_distribution"] = {}
        preserved["reassignment_reason"] = ""
        preserved["smoothing_distribution"] = {}
        preserved["smoothing_reason"] = ""
        preserved["short_sandwich_distribution"] = {}
        preserved["short_sandwich_reason"] = ""
        preserved["local_leakage_distribution"] = {}
        preserved["local_leakage_reason"] = ""
        preserved["continuity_repair_distribution"] = {}
        preserved["continuity_repair_reason"] = ""
        preserved["voice_band_repair_distribution"] = {}
        preserved["voice_band_repair_reason"] = ""
        preserved["handoff_split_count"] = 0
        preserved["handoff_split_reason"] = ""
        preserved["handoff_voice_guard_distribution"] = {}
        preserved["handoff_voice_guard_reason"] = ""
        preserved["resegmentation_count"] = 0
        preserved["resegmentation_reason"] = ""
        preserved["postprocess_skipped_reason"] = (
            f"已按手动指定 {int(requested_n)} 人保留原始分人；"
            f"自动纠偏会把 {original_count} 人降为 {corrected_count} 人，"
            "请抽听疑点段后再决定是否合并。"
        )
        preserved["review_segments"] = _build_review_segments(preserved)
        return preserved

    corrected["postprocess_skipped_reason"] = ""
    return corrected


def _repair_handoff_voice_guard_assignments(candidate: dict) -> dict:
    """Re-check handoff splits against strong short-window speaker evidence."""
    segments = [dict(s) for s in candidate.get("segments") or []]
    if not segments:
        candidate["handoff_voice_guard_distribution"] = {}
        candidate["handoff_voice_guard_reason"] = ""
        return candidate

    voice_profiles = _speaker_voice_profiles(segments)
    from collections import Counter, defaultdict

    corrections: dict[int, str] = {}
    distribution: dict[str, Counter] = defaultdict(Counter)
    review_segments: list[dict] = []

    def best_vote_target(seg: dict, current: str) -> str:
        votes = seg.get("speaker_votes")
        if not isinstance(votes, dict) or not votes:
            return ""
        clean = []
        total = 0.0
        for speaker, value in votes.items():
            try:
                duration = max(0.0, float(value))
            except (TypeError, ValueError):
                continue
            if duration <= 0:
                continue
            clean.append((duration, str(speaker)))
            total += duration
        if not clean or total <= 0:
            return ""
        clean.sort(reverse=True)
        best_duration, best_speaker = clean[0]
        if best_speaker == current:
            return ""
        current_share = _speaker_vote_share(seg, current) or 0.0
        best_share = best_duration / total
        if best_share >= 0.68 and best_share >= current_share + 0.35:
            return best_speaker
        return ""

    for idx, seg in enumerate(segments):
        if not (seg.get("speaker_handoff_split") or seg.get("speaker_handoff_bridge")):
            continue
        if seg.get("speaker_handoff_text_review"):
            continue
        current = str(seg.get("speaker") or "")
        if not current:
            continue
        target = best_vote_target(seg, current)
        if not target:
            continue
        seg_pitch = _voice_pitch(seg)
        if seg_pitch is not None:
            current_distance = _speaker_profile_distance(voice_profiles, current, seg_pitch)
            target_distance = _speaker_profile_distance(voice_profiles, target, seg_pitch)
            if target_distance == float("inf"):
                continue
            if current_distance != float("inf") and target_distance + 18.0 >= current_distance:
                review_segments.append(_review_segment_payload(
                    idx,
                    seg,
                    current,
                    target,
                    "声纹护栏：handoff 结果与底层投票冲突，但声线距离不足以安全改派，建议抽听确认",
                ))
                continue
        corrections[idx] = target
        distribution[current][target] += 1
        review_segments.append(_review_segment_payload(
            idx,
            seg,
            current,
            target,
            "声纹护栏：handoff 结果与底层短窗投票/声线强冲突，已按声纹证据纠回",
        ))

    if not corrections:
        candidate["handoff_voice_guard_distribution"] = {}
        candidate["handoff_voice_guard_reason"] = ""
        if review_segments:
            candidate["review_segments"] = _merge_review_segments([
                *(candidate.get("review_segments") or []),
                *review_segments,
            ])
        return candidate

    for idx, target in corrections.items():
        segments[idx]["speaker"] = target
        segments[idx]["speaker_handoff_review"] = True
        segments[idx]["speaker_handoff_voice_guard_repaired"] = True

    corrected = {**candidate, "segments": segments}
    _rescore_candidate(corrected)
    corrected["actual_n_speakers"] = len(corrected["summary"]["speakers"])
    corrected["speakers"] = [s["speaker"] for s in corrected["summary"]["speakers"]]
    payload = {
        speaker: {target: int(count) for target, count in counts.most_common()}
        for speaker, counts in distribution.items()
    }
    corrected["handoff_voice_guard_distribution"] = payload
    corrected["handoff_voice_guard_reason"] = "已按声纹护栏纠正 handoff 后处理造成的强证据冲突段"
    corrected["review_segments"] = _merge_review_segments([
        *(candidate.get("review_segments") or []),
        *review_segments,
    ])
    return corrected


def _speaker_summary(segments: list[dict]) -> dict:
    from collections import Counter, defaultdict
    import re

    counts = Counter(str(s.get("speaker")) for s in segments if s.get("speaker"))
    durations = defaultdict(float)
    short_counts = Counter()
    filler_counts = Counter()
    sandwiched_counts = Counter()
    turns = []
    filler_re = re.compile(r"^[嗯啊呃哦对是好可以行的了哈\s,，。.!！?？]+$")
    for idx, seg in enumerate(segments):
        speaker = str(seg.get("speaker") or "")
        if not speaker:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        duration = max(0.0, end - start)
        durations[speaker] += max(0.0, end - start)
        text = str(seg.get("text") or "").strip()
        if duration < 1.5:
            short_counts[speaker] += 1
        if text and filler_re.match(text):
            filler_counts[speaker] += 1
        prev_speaker = str(segments[idx - 1].get("speaker") or "") if idx > 0 else ""
        next_speaker = str(segments[idx + 1].get("speaker") or "") if idx + 1 < len(segments) else ""
        if prev_speaker and prev_speaker == next_speaker and prev_speaker != speaker:
            sandwiched_counts[speaker] += 1
        if turns and turns[-1]["speaker"] == speaker and start - turns[-1]["end"] <= 1.2:
            turns[-1]["end"] = end
            turns[-1]["segments"] += 1
        else:
            turns.append({"speaker": speaker, "start": start, "end": end, "segments": 1})

    turn_counts = Counter(t["speaker"] for t in turns)
    stable_turn_counts = Counter(
        t["speaker"]
        for t in turns
        if (t["end"] - t["start"]) >= 4.0 or t["segments"] >= 3
    )
    total = max(1, len(segments))
    total_duration = sum(durations.values()) or 1.0
    speakers = []
    for speaker in sorted(counts):
        segment_count = int(counts[speaker])
        duration_s = float(durations[speaker])
        speakers.append({
            "speaker": speaker,
            "segments": segment_count,
            "segment_ratio": segment_count / total,
            "duration_s": duration_s,
            "duration_ratio": duration_s / total_duration,
            "turns": int(turn_counts[speaker]),
            "stable_turns": int(stable_turn_counts[speaker]),
            "short_segments": int(short_counts[speaker]),
            "short_ratio": int(short_counts[speaker]) / max(1, segment_count),
            "filler_segments": int(filler_counts[speaker]),
            "filler_ratio": int(filler_counts[speaker]) / max(1, segment_count),
            "sandwiched_segments": int(sandwiched_counts[speaker]),
            "sandwiched_ratio": int(sandwiched_counts[speaker]) / max(1, segment_count),
        })
    return {
        "speakers": speakers,
        "turns": len(turns),
        "stable_turns": sum(int(s["stable_turns"]) for s in speakers),
    }


def _speaker_name_to_label(name: str) -> str:
    suffix = name.replace("SPEAKER_", "")
    return suffix or name


def _score_diarization_candidate(candidate: dict, requested_n: int) -> dict:
    speakers = candidate["summary"]["speakers"]
    if not speakers:
        return {"score": -999.0, "issues": ["未识别到说话人"], "reason": "未识别到说话人"}

    voice_mix_severity = _candidate_voice_mix_severity(candidate)
    voice_mix_summary = voice_mix_severity["voice_mix_summary"]
    mixed_voice_speakers = voice_mix_severity["mixed_voice_speakers"]
    severe_mixed_voice_speakers = voice_mix_severity["severe_mixed_voice_speakers"]
    voice_mix_penalty = float(voice_mix_severity["voice_mix_penalty"])
    formed_n = len(speakers)
    tiny = [
        s for s in speakers
        if s["segments"] <= 2 or s["duration_s"] < 3.0 or (s["stable_turns"] == 0 and s["segments"] < 6)
    ]
    tiny_names = {s["speaker"] for s in tiny}
    weak = [
        s for s in speakers
        if s["speaker"] not in tiny_names
        and (
            s["segments"] < 8
            or s["duration_s"] < 10.0
            or s["stable_turns"] == 0
            or (requested_n >= 4 and s["duration_s"] < 18.0)
            or (
                s["duration_s"] < 18.0
                and s["stable_turns"] <= 2
            )
        )
    ]
    weak_names = {s["speaker"] for s in weak}
    fragmented = [
        s for s in speakers
        if s["speaker"] not in tiny_names
        and (
            (
                s["turns"] >= 5
                and (s["duration_s"] / max(1, s["turns"])) < 3.0
                and s["stable_turns"] <= 3
            )
            or (
                s["turns"] >= 5
                and s.get("short_ratio", 0.0) >= 0.45
                and s.get("filler_ratio", 0.0) >= 0.35
            )
            or (
                s["turns"] >= 5
                and s.get("sandwiched_ratio", 0.0) >= 0.18
                and s.get("filler_ratio", 0.0) >= 0.30
            )
            or (
                s["duration_s"] < 18.0
                and s["turns"] >= 5
                and s.get("short_ratio", 0.0) >= 0.35
            )
        )
    ]
    fragmented_names = {s["speaker"] for s in fragmented}
    stable = [
        s for s in speakers
        if s["speaker"] not in tiny_names
        and s["speaker"] not in weak_names
        and s["speaker"] not in fragmented_names
    ]
    # A marginal speaker is not pure noise, but is too small to justify raising
    # the global count unless the lower-count candidates are clearly worse.
    marginal = [
        s for s in speakers
        if s["speaker"] not in tiny_names
        and (s["segment_ratio"] < 0.015 or s["duration_ratio"] < 0.010)
    ]
    mergeable = [*tiny, *fragmented]
    mergeable_names = {s["speaker"] for s in mergeable}
    for s in marginal:
        if s["speaker"] in mergeable_names:
            continue
        if (
            s["duration_s"] < 18.0
            or s["stable_turns"] == 0
            or (s.get("short_ratio", 0.0) >= 0.45 and s.get("filler_ratio", 0.0) >= 0.30)
        ):
            mergeable.append(s)
            mergeable_names.add(s["speaker"])
    max_ratio = max(s["segment_ratio"] for s in speakers)

    score = 0.0
    score += len(stable) * 6.0
    score += len(weak) * 0.5
    score -= len(tiny) * 6.0
    score -= len(fragmented) * 4.0
    score -= len(marginal) * 2.5
    if requested_n > 2 and max_ratio > 0.82:
        score -= (max_ratio - 0.82) * 12.0
    if requested_n >= 6:
        score -= max(0, requested_n - len(stable)) * 1.4
    if requested_n >= 5:
        score -= (len(weak) + len(tiny)) * 1.5
    if requested_n > formed_n:
        score -= (requested_n - formed_n) * 2.0
    if mixed_voice_speakers:
        score -= voice_mix_penalty
    if requested_n != len(stable):
        score -= max(0, requested_n - len(stable)) * 0.8
    if len(stable) >= 2:
        score += 1.0
    if len(stable) == requested_n:
        score += 2.0
    if len(stable) == 0:
        score -= 8.0

    issues = []
    if tiny:
        issues.append(f"{len(tiny)} 个碎片说话人")
    if weak:
        issues.append(f"{len(weak)} 个弱说话人")
    if fragmented:
        issues.append(f"{len(fragmented)} 个碎片式说话人")
    if marginal:
        issues.append(f"{len(marginal)} 个低占比说话人")
    if requested_n > formed_n:
        issues.append(f"实际只形成 {formed_n} 人")
    if mixed_voice_speakers:
        issues.append(f"{len(mixed_voice_speakers)} 个说话人疑似混入高低声线")
    if severe_mixed_voice_speakers:
        issues.append(f"{len(severe_mixed_voice_speakers)} 个说话人存在严重声线混标")
    if max_ratio > 0.86 and requested_n > 2:
        issues.append("主说话人占比过高")
    reason = " / ".join(issues) if issues else "说话人分布较稳定"
    return {
        "score": round(score, 3),
        "actual_n_speakers": formed_n,
        "stable_speakers": len(stable),
        "weak_speakers": len(weak),
        "tiny_speakers": len(tiny),
        "fragmented_speakers": len(fragmented),
        "marginal_speakers": len(marginal),
        "fragile_speakers": [s["speaker"] for s in [*tiny, *weak, *fragmented, *marginal]],
        "mergeable_speakers": [s["speaker"] for s in mergeable],
        "voice_mix_summary": voice_mix_summary,
        "mixed_voice_speakers": mixed_voice_speakers,
        "severe_mixed_voice_speakers": severe_mixed_voice_speakers,
        "voice_mix_penalty": round(float(voice_mix_penalty), 3),
        "voice_line_groups": _speaker_voice_line_groups(candidate.get("segments") or []),
        "dominant_ratio": round(max_ratio, 3),
        "issues": issues,
        "reason": reason,
    }


def _rescore_candidate(candidate: dict) -> None:
    candidate["summary"] = _speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, int(candidate["n_speakers"])))


def _speaker_duration_lookup(candidate: dict) -> dict[str, float]:
    return {
        str(s["speaker"]): float(s.get("duration_s", 0.0))
        for s in candidate.get("summary", {}).get("speakers", [])
    }


def _segment_duration(seg: dict) -> float:
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    return max(0.0, end - start)


def _review_segment_payload(
    idx: int,
    seg: dict,
    from_speaker: str,
    to_speaker: str,
    reason: str,
) -> dict:
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    return {
        "index": int(idx),
        "start": start,
        "end": end,
        "duration_s": round(max(0.0, end - start), 3),
        "text": str(seg.get("text") or ""),
        "from_speaker": from_speaker,
        "to_speaker": to_speaker,
        "reason": reason,
    }


def _merge_review_segments(review_segments: list[dict], limit: int = 160) -> list[dict]:
    merged: list[dict] = []
    by_index: dict[int, dict] = {}
    for item in sorted(review_segments, key=lambda s: (float(s.get("start", 0.0)), int(s.get("index", 0)))):
        idx = int(item.get("index", -1))
        if idx < 0 or idx not in by_index:
            normalized = dict(item)
            merged.append(normalized)
            if idx >= 0:
                by_index[idx] = normalized
            continue

        existing = by_index[idx]
        reason = str(item.get("reason") or "")
        if reason and reason not in str(existing.get("reason") or ""):
            existing["reason"] = f"{existing.get('reason') or ''}；{reason}".strip("；")
        existing_from = str(existing.get("from_speaker") or "")
        existing_to = str(existing.get("to_speaker") or "")
        item_to = str(item.get("to_speaker") or "")
        if existing_from and existing_to == existing_from and item_to and item_to != existing_from:
            existing["to_speaker"] = item_to
    return merged[:limit]


def _append_review_reason(existing: str, reason: str) -> str:
    existing = str(existing or "").strip()
    reason = str(reason or "").strip()
    if not reason:
        return existing
    if not existing:
        return reason
    if reason in existing:
        return existing
    return f"{existing}；{reason}"


def _segment_level_review_reason(reason: str) -> str:
    """Keep only high-signal review reasons on transcript rows.

    Full review metadata stays in `diarization_stats.review_segments`; this
    filter prevents the UI from painting dozens of weak/noisy "待确认" chips.
    """
    parts = [part.strip() for part in str(reason or "").split("；") if part.strip()]
    high_signal: list[str] = []
    noisy_prefixes = (
        "推荐置信度不足",
        "段内短声纹窗存在换人切点",
        "段内短声纹窗显示多个说话人：当前单一说话人标签需抽听确认",
        "段内短声纹窗显示多个说话人，但不满足安全切分条件",
        "段内短声纹窗疑似换人，但证据重叠/接近",
        "段内短声纹窗疑似换人，但一侧时长过短",
    )
    strong_markers = (
        "局部快速轮换风险",
        "局部夹心跳变",
        "短插话/边界跳变",
        "声线复核",
        "声线混合风险",
        "段内短声纹窗多次换人",
        "长段内检测到强新发言",
        "强新发言开场",
    )
    for part in parts:
        if part.startswith(noisy_prefixes):
            continue
        if any(marker in part for marker in strong_markers):
            high_signal.append(part)
    return "；".join(high_signal)


def _annotate_segments_with_speaker_reviews(candidate: dict) -> dict:
    """Copy review-list risks back onto the segment rows shown in the UI."""
    segments = [dict(seg) for seg in candidate.get("segments") or []]
    review_segments = list(candidate.get("review_segments") or [])
    for seg in segments:
        existing_reason = str(seg.get("speaker_review_reason") or "")
        kept_reason = _segment_level_review_reason(existing_reason)
        skip_segment_level_review = (
            seg.get("speaker_calibrated")
            or seg.get("voice_band_repaired")
            or seg.get("continuity_repaired")
            or seg.get("speaker_handoff_voice_guard_repaired")
        )
        if skip_segment_level_review or not kept_reason:
            seg.pop("speaker_assignment_review", None)
            if existing_reason and existing_reason != "已按历史人工标注校准":
                seg.pop("speaker_review_reason", None)
            continue
        seg["speaker_assignment_review"] = True
        seg["speaker_review_reason"] = kept_reason
    if not segments or not review_segments:
        candidate["segments"] = segments
        return candidate

    def mark(seg_idx: int, reason: str) -> None:
        if not 0 <= seg_idx < len(segments):
            return
        if segments[seg_idx].get("speaker_calibrated"):
            return
        if (
            segments[seg_idx].get("voice_band_repaired")
            or segments[seg_idx].get("continuity_repaired")
            or segments[seg_idx].get("speaker_handoff_voice_guard_repaired")
        ):
            return
        reason = _segment_level_review_reason(reason)
        if not reason:
            return
        segments[seg_idx]["speaker_assignment_review"] = True
        segments[seg_idx]["speaker_review_reason"] = _append_review_reason(
            str(segments[seg_idx].get("speaker_review_reason") or ""),
            reason,
        )

    for item in review_segments:
        reason = str(item.get("reason") or "")
        try:
            idx = int(item.get("index", -1))
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < len(segments):
            mark(idx, reason)
            continue

        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        for seg_idx, seg in enumerate(segments):
            try:
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", seg_start))
            except (TypeError, ValueError):
                continue
            overlap = max(0.0, min(seg_end, end) - max(seg_start, start))
            if overlap >= 0.25 or (abs(seg_start - start) <= 0.35 and abs(seg_end - end) <= 0.35):
                mark(seg_idx, reason)
                break

    candidate["segments"] = segments
    return candidate


def _segment_gap(a: dict, b: dict) -> float:
    a_start = float(a.get("start", 0.0))
    a_end = float(a.get("end", a_start))
    b_start = float(b.get("start", 0.0))
    b_end = float(b.get("end", b_start))
    if min(a_end, b_end) >= max(a_start, b_start):
        return 0.0
    return min(abs(a_start - b_end), abs(a_end - b_start))


def _select_merge_anchor(candidate: dict, anchors: list[dict] | None) -> dict | None:
    if not anchors:
        return None
    n_speakers = int(candidate.get("n_speakers") or 0)
    lower = [
        a for a in anchors
        if a is not candidate
        and int(a.get("n_speakers") or 0) < n_speakers
        and len(a.get("segments") or []) == len(candidate.get("segments") or [])
    ]
    if not lower:
        return None

    clean = [
        a for a in lower
        if not a.get("fragile_speakers")
        and int(a.get("actual_n_speakers") or len(a.get("speakers") or [])) >= 2
    ]
    pool = clean or lower
    # Use the coarsest clean split as a macro-speaker anchor. It prevents a
    # tiny forced cluster from being merged into a neighboring speaker that
    # belongs to a different broad voice group.
    return min(
        pool,
        key=lambda a: (
            int(a.get("actual_n_speakers") or len(a.get("speakers") or [])),
            int(a.get("n_speakers") or 0),
            float(a.get("score") or 0.0) * -1,
        ),
    )


def _stable_anchor_map(candidate: dict, anchor: dict | None, stable_set: set[str]) -> dict[str, str]:
    if not anchor:
        return {}
    from collections import Counter, defaultdict

    votes: dict[str, Counter] = defaultdict(Counter)
    anchor_segments = anchor.get("segments") or []
    for idx, seg in enumerate(candidate.get("segments") or []):
        if idx >= len(anchor_segments):
            break
        speaker = str(seg.get("speaker") or "")
        if speaker not in stable_set:
            continue
        anchor_speaker = str(anchor_segments[idx].get("speaker") or "")
        if not anchor_speaker:
            continue
        votes[speaker][anchor_speaker] += _segment_duration(seg)
    return {
        speaker: counts.most_common(1)[0][0]
        for speaker, counts in votes.items()
        if counts
    }


def _best_merge_target(
    segments: list[dict],
    idx: int,
    targets: list[str],
    duration_by_speaker: dict[str, float],
) -> str:
    if not targets:
        return ""

    seg = segments[idx]
    best_target = ""
    best_score = float("-inf")
    prev_speaker = str(segments[idx - 1].get("speaker") or "") if idx > 0 else ""
    next_speaker = str(segments[idx + 1].get("speaker") or "") if idx + 1 < len(segments) else ""

    for target in targets:
        best_gap = float("inf")
        local_turn_segments = 0
        for j, other in enumerate(segments):
            if j == idx or str(other.get("speaker") or "") != target:
                continue
            gap = _segment_gap(seg, other)
            best_gap = min(best_gap, gap)
            if gap <= 12.0:
                local_turn_segments += 1

        if best_gap == float("inf"):
            continue
        # Local proximity carries the most weight, but a strong stable speaker
        # should win near-ties over a tiny adjacent artifact. This fixes cases
        # where a forced C cluster is surrounded by A text but belongs to the
        # same broad two-speaker anchor as B.
        support = duration_by_speaker.get(target, 0.0)
        score = 1.0 / (1.0 + best_gap)
        score += min(0.12, support / 2400.0)
        score += min(0.06, local_turn_segments * 0.015)
        if prev_speaker == target:
            score += 0.08
        if next_speaker == target:
            score += 0.08
        if prev_speaker == target and next_speaker == target:
            score += 0.12

        if score > best_score:
            best_score = score
            best_target = target

    return best_target


def _merge_fragile_speakers(candidate: dict, anchors: list[dict] | None = None) -> dict:
    """Return a copy with likely pseudo-speakers merged into stable speakers.

    Fragile speakers are reassigned per segment. When lower-count candidates are
    available, their coarse speaker labels are used as a macro-speaker anchor so
    a short forced cluster is not blindly merged into its immediate neighbor.
    """
    fragile = set(candidate.get("mergeable_speakers") or candidate.get("fragile_speakers") or [])
    if not fragile:
        candidate["merge_map"] = {}
        candidate["merge_distribution"] = {}
        candidate["merge_reason"] = ""
        return candidate

    stable_names = [
        s["speaker"]
        for s in candidate["summary"]["speakers"]
        if s["speaker"] not in fragile
    ]
    if len(stable_names) < 2:
        candidate["merge_map"] = {}
        candidate["merge_distribution"] = {}
        candidate["merge_reason"] = ""
        return candidate

    out_segments = [dict(s) for s in candidate["segments"]]
    duration_by_speaker = _speaker_duration_lookup(candidate)
    stable_set = set(stable_names)
    anchor = _select_merge_anchor(candidate, anchors)
    anchor_segments = anchor.get("segments") if anchor else None
    stable_to_anchor = _stable_anchor_map(candidate, anchor, stable_set)

    from collections import Counter, defaultdict

    merge_distribution: dict[str, Counter] = defaultdict(Counter)
    review_segments: list[dict] = []

    for idx, seg in enumerate(out_segments):
        speaker = str(seg.get("speaker") or "")
        if speaker not in fragile:
            continue

        targets = stable_names
        if anchor_segments and idx < len(anchor_segments):
            anchor_speaker = str(anchor_segments[idx].get("speaker") or "")
            same_macro_targets = [
                stable for stable in stable_names
                if stable_to_anchor.get(stable) == anchor_speaker
            ]
            if same_macro_targets:
                targets = same_macro_targets

        best_speaker = _best_merge_target(out_segments, idx, targets, duration_by_speaker)
        if best_speaker:
            review_segments.append(_review_segment_payload(
                idx,
                seg,
                speaker,
                best_speaker,
                "碎片说话人逐段合并",
            ))
            seg["speaker"] = best_speaker
            merge_distribution[speaker][best_speaker] += 1

    if not merge_distribution:
        candidate["merge_map"] = {}
        candidate["merge_distribution"] = {}
        candidate["merge_reason"] = ""
        return candidate

    merge_map: dict[str, str] = {}
    distribution_payload: dict[str, dict[str, int]] = {}
    for speaker, counts in merge_distribution.items():
        merge_map[speaker] = counts.most_common(1)[0][0]
        distribution_payload[speaker] = {
            target: int(count)
            for target, count in counts.most_common()
        }

    merged = {**candidate, "segments": out_segments}
    _rescore_candidate(merged)
    merged["actual_n_speakers"] = len(merged["summary"]["speakers"])
    merged["speakers"] = [s["speaker"] for s in merged["summary"]["speakers"]]
    merged["merge_map"] = merge_map
    merged["merge_distribution"] = distribution_payload
    merged["review_segments"] = _merge_review_segments(review_segments)
    merge_parts = []
    for speaker, counts in distribution_payload.items():
        target_desc = "/".join(
            f"{target.replace('SPEAKER_', '')}{count}"
            for target, count in counts.items()
        )
        merge_parts.append(f"{speaker.replace('SPEAKER_', '')}->{target_desc}")
    if anchor:
        merged["merge_reason"] = "已按低人数候选对齐，逐段合并碎片式/弱说话人"
    else:
        merged["merge_reason"] = "已逐段合并碎片式/弱说话人"
    if merge_parts:
        merged["merge_reason"] = f"{merged['merge_reason']}（{' / '.join(merge_parts)}）"
    merged["reason"] = (
        merged["merge_reason"]
        if not merged.get("issues")
        else f"{merged['merge_reason']}；{merged['reason']}"
    )
    return merged


def _nearest_same_speaker_gap(
    segments: list[dict],
    idx: int,
    speaker: str,
    window_s: float = 18.0,
) -> float:
    seg = segments[idx]
    best = float("inf")
    for j, other in enumerate(segments):
        if j == idx or str(other.get("speaker") or "") != speaker:
            continue
        gap = _segment_gap(seg, other)
        if gap <= window_s:
            best = min(best, gap)
    return best


def _same_speaker_run_length(segments: list[dict], idx: int, speaker: str, max_gap_s: float = 1.2) -> int:
    total = 1
    prev_end = float(segments[idx].get("start", 0.0))
    j = idx - 1
    while j >= 0:
        other = segments[j]
        if str(other.get("speaker") or "") != speaker:
            break
        other_end = float(other.get("end", other.get("start", 0.0)))
        if prev_end - other_end > max_gap_s:
            break
        total += 1
        prev_end = float(other.get("start", 0.0))
        j -= 1

    next_start = float(segments[idx].get("end", segments[idx].get("start", 0.0)))
    j = idx + 1
    while j < len(segments):
        other = segments[j]
        if str(other.get("speaker") or "") != speaker:
            break
        other_start = float(other.get("start", 0.0))
        if other_start - next_start > max_gap_s:
            break
        total += 1
        next_start = float(other.get("end", other_start))
        j += 1
    return total


def _fragile_reassignment_has_acoustic_support(
    seg: dict,
    from_speaker: str,
    to_speaker: str,
    context_hits: int,
) -> tuple[bool, str]:
    """Require segment-level acoustic evidence before overriding a speaker."""
    if seg.get("speaker_overlap_risk"):
        return False, "声纹证据门禁：片段存在重叠语音风险，已保留原说话人并标为待确认"

    text = str(seg.get("text") or "").strip()
    lexical_text = re.sub(r"[^\w]+", "", text, flags=re.UNICODE)
    if _segment_duration(seg) <= 0.45 and not lexical_text and context_hits == 2:
        return True, ""

    target_share = _speaker_vote_share(seg, to_speaker)
    source_share = _speaker_vote_share(seg, from_speaker)
    if target_share is None or source_share is None:
        return False, "声纹证据门禁：缺少短窗声纹投票，已保留原说话人并标为待确认"

    margin = target_share - source_share
    if target_share >= 0.58 and margin >= 0.20:
        return True, ""

    confidence = _speaker_confidence(seg)
    if source_share >= target_share and (source_share >= 0.45 or (confidence or 0.0) >= 0.65):
        return False, "声纹证据门禁：原说话人声纹投票占优，已阻止上下文自动覆盖"
    return False, "声纹证据门禁：目标说话人声纹票数或领先幅度不足，已保留原说话人并标为待确认"


def _reassign_isolated_fragile_segments(candidate: dict, anchors: list[dict] | None = None) -> dict:
    """Correct isolated wrong-speaker assignments without reducing speaker count.

    This is intentionally narrower than `_merge_fragile_speakers`: it keeps the
    split produced by the selected speaker count, and only moves short isolated
    fragments when a lower-count anchor and local context agree. Continuous
    low-volume participants remain visible as their own speaker.
    """
    fragile = set(candidate.get("mergeable_speakers") or candidate.get("fragile_speakers") or [])
    if not fragile:
        candidate["reassignment_distribution"] = {}
        candidate["reassignment_reason"] = ""
        return candidate

    stable_names = [
        s["speaker"]
        for s in candidate.get("summary", {}).get("speakers", [])
        if s["speaker"] not in fragile
    ]
    if len(stable_names) < 2:
        candidate["reassignment_distribution"] = {}
        candidate["reassignment_reason"] = ""
        return candidate

    anchor = _select_merge_anchor(candidate, anchors)
    anchor_segments = anchor.get("segments") if anchor else None
    stable_set = set(stable_names)
    stable_to_anchor = _stable_anchor_map(candidate, anchor, stable_set)
    duration_by_speaker = _speaker_duration_lookup(candidate)
    out_segments = [dict(s) for s in candidate.get("segments", [])]
    voice_profiles = _speaker_voice_profiles(out_segments)

    from collections import Counter, defaultdict

    distribution: dict[str, Counter] = defaultdict(Counter)
    review_segments: list[dict] = []

    for idx, seg in enumerate(out_segments):
        speaker = str(seg.get("speaker") or "")
        if speaker not in fragile:
            continue

        duration = _segment_duration(seg)
        run_length = _same_speaker_run_length(out_segments, idx, speaker)
        nearest_same_gap = _nearest_same_speaker_gap(out_segments, idx, speaker)
        prev_speaker = str(out_segments[idx - 1].get("speaker") or "") if idx > 0 else ""
        next_speaker = str(out_segments[idx + 1].get("speaker") or "") if idx + 1 < len(out_segments) else ""

        targets = list(stable_names)
        anchor_speaker = ""
        if anchor_segments and idx < len(anchor_segments):
            anchor_speaker = str(anchor_segments[idx].get("speaker") or "")
            same_macro_targets = [
                stable for stable in stable_names
                if stable_to_anchor.get(stable) == anchor_speaker
            ]
            if same_macro_targets:
                targets = same_macro_targets

        best_speaker = _best_merge_target(out_segments, idx, targets, duration_by_speaker)
        if not best_speaker or best_speaker == speaker:
            continue

        context_hits = int(prev_speaker == best_speaker) + int(next_speaker == best_speaker)
        isolated = run_length <= 2 and nearest_same_gap > 8.0
        short_or_boundary = duration <= 6.5 or context_hits > 0
        anchor_support = bool(anchor_speaker and stable_to_anchor.get(best_speaker) == anchor_speaker)

        if isolated and short_or_boundary and (anchor_support or context_hits > 0):
            supported, support_reason = _fragile_reassignment_has_acoustic_support(
                seg,
                speaker,
                best_speaker,
                context_hits,
            )
            if not supported:
                review_segments.append(_review_segment_payload(
                    idx,
                    seg,
                    speaker,
                    best_speaker,
                    support_reason,
                ))
                continue
            blocked, block_reason = _voice_guard_blocks_reassignment(
                seg,
                speaker,
                best_speaker,
                voice_profiles,
            )
            if blocked:
                review_segments.append(_review_segment_payload(
                    idx,
                    seg,
                    speaker,
                    best_speaker,
                    block_reason,
                ))
                continue
            review_segments.append(_review_segment_payload(
                idx,
                seg,
                speaker,
                best_speaker,
                "孤立错挂片段已按低人数锚点/上下文纠偏",
            ))
            seg["speaker"] = best_speaker
            distribution[speaker][best_speaker] += 1

    if not distribution:
        if review_segments:
            candidate["review_segments"] = _merge_review_segments([
                *(candidate.get("review_segments") or []),
                *review_segments,
            ])
            candidate["voice_guard_reason"] = "声纹/声线证据门禁已阻止无证据的自动纠偏"
            candidate["voice_guard_count"] = len(review_segments)
        candidate["reassignment_distribution"] = {}
        candidate["reassignment_reason"] = ""
        return candidate

    corrected = {**candidate, "segments": out_segments}
    _rescore_candidate(corrected)
    corrected["actual_n_speakers"] = len(corrected["summary"]["speakers"])
    corrected["speakers"] = [s["speaker"] for s in corrected["summary"]["speakers"]]
    payload: dict[str, dict[str, int]] = {
        speaker: {target: int(count) for target, count in counts.most_common()}
        for speaker, counts in distribution.items()
    }
    corrected["reassignment_distribution"] = payload
    corrected["voice_profiles"] = voice_profiles
    if any("门禁" in str(item.get("reason") or "") for item in review_segments):
        corrected["voice_guard_reason"] = "声纹/声线证据门禁已阻止无证据的自动纠偏"
        corrected["voice_guard_count"] = sum(
            1 for item in review_segments
            if "门禁" in str(item.get("reason") or "")
        )
    corrected["review_segments"] = _merge_review_segments([
        *(candidate.get("review_segments") or []),
        *review_segments,
    ])
    parts = []
    for speaker, counts in payload.items():
        target_desc = "/".join(
            f"{target.replace('SPEAKER_', '')}{count}"
            for target, count in counts.items()
        )
        parts.append(f"{speaker.replace('SPEAKER_', '')}->{target_desc}")
    corrected["reassignment_reason"] = f"已保留人数，仅纠偏孤立错挂片段（{' / '.join(parts)}）"
    return corrected


def _speaker_runs(segments: list[dict], max_gap_s: float = 1.5) -> list[dict]:
    runs: list[dict] = []
    for idx, seg in enumerate(segments):
        speaker = str(seg.get("speaker") or "")
        if not speaker:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if runs and runs[-1]["speaker"] == speaker and start - float(runs[-1]["end"]) <= max_gap_s:
            runs[-1]["end"] = end
            runs[-1]["indices"].append(idx)
        else:
            runs.append({"speaker": speaker, "start": start, "end": end, "indices": [idx]})
    return runs


def _speaker_confidence(seg: dict) -> float | None:
    value = seg.get("speaker_confidence")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _speaker_vote_share(seg: dict, speaker: str) -> float | None:
    votes = seg.get("speaker_votes")
    if not isinstance(votes, dict) or not votes:
        return None
    total = 0.0
    target = 0.0
    for key, value in votes.items():
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        total += max(0.0, duration)
        if str(key) == speaker:
            target += max(0.0, duration)
    if total <= 0:
        return None
    return target / total


def _project_speaker_cues(candidate: dict) -> dict:
    """Project short-window speaker evidence onto immutable ASR sync cues.

    A single ASR segment can contain a real speaker handoff.  The transcript
    geometry stays frozen, while this additional diarization timeline lets the
    UI and exports represent the handoff at existing sync-cue boundaries.
    """
    segments = [dict(seg) for seg in candidate.get("segments") or []]
    projected_segments = 0
    projected_cues = 0
    review_cues = 0

    def union_duration(intervals: list[tuple[float, float]]) -> float:
        covered = 0.0
        current_start: float | None = None
        current_end: float | None = None
        for start, end in sorted(intervals):
            if current_start is None or current_end is None:
                current_start, current_end = start, end
            elif start <= current_end:
                current_end = max(current_end, end)
            else:
                covered += current_end - current_start
                current_start, current_end = start, end
        if current_start is not None and current_end is not None:
            covered += current_end - current_start
        return covered

    def split_text_for_handoff(text: str, wall_ratio: float, voice_ratio: float) -> tuple[str, str]:
        """Split display text without changing the immutable ASR cue."""
        if not text:
            return "", ""
        content_positions = [
            index
            for index, char in enumerate(text)
            if not char.isspace() and char not in "，。！？；：、,.!?;:"
        ]
        if len(content_positions) < 2:
            return "", ""
        blended_ratio = max(0.10, min(0.90, (wall_ratio + voice_ratio) / 2.0))
        left_content = max(
            1,
            min(len(content_positions) - 1, int(len(content_positions) * blended_ratio + 0.5)),
        )
        split_index = content_positions[left_content - 1] + 1
        if (
            0 < split_index < len(text)
            and text[split_index - 1].isascii()
            and text[split_index].isascii()
            and text[split_index - 1].isalnum()
            and text[split_index].isalnum()
        ):
            nearby = [
                index
                for index in range(max(1, split_index - 4), min(len(text), split_index + 5))
                if text[index - 1].isspace() or text[index].isspace()
            ]
            if nearby:
                split_index = min(nearby, key=lambda value: abs(value - split_index))
        return text[:split_index], text[split_index:]

    def cue_text(raw_cues: list, index: int) -> str:
        if index < 0 or index >= len(raw_cues) or not isinstance(raw_cues[index], dict):
            return ""
        return str(raw_cues[index].get("text") or "").strip()

    def has_closed_sentence(text: str) -> bool:
        return bool(re.search(r"[。！？!?；;：:]\s*$", str(text or "").strip()))

    def collapse_overlap_alternation_to_handoff(
        rows: list[dict],
        windows: list[tuple[float, float, str]],
        current: str,
        change_points: list[float],
        segment_confidence: float | None,
        overlap_risk: bool,
    ) -> tuple[list[dict], bool]:
        """Collapse overlapping-window A/B/A/B jitter to one supported handoff.

        Senko's 1.5-second windows overlap. Around a real turn boundary that can
        create several apparent label changes even when the evidence has a clear
        earlier/later ordering. Search only immutable sync-cue boundaries and
        fail closed when an exact cue strongly assigns against the proposed
        ordering.
        """
        if (
            not overlap_risk
            or segment_confidence is None
            or segment_confidence > 0.70
            or len(rows) < 3
            or len(change_points) < 3
            or len(windows) < 4
        ):
            return rows, False

        ordered_windows = sorted(windows, key=lambda item: (item[0], item[1], item[2]))
        window_labels = [speaker for _, _, speaker in ordered_windows]
        unique_speakers = set(window_labels)
        raw_transitions = sum(
            left != right
            for left, right in zip(window_labels, window_labels[1:])
        )
        if current not in unique_speakers or len(unique_speakers) != 2 or raw_transitions < 3:
            return rows, False

        alternate = next(speaker for speaker in unique_speakers if speaker != current)
        segment_start = min(float(row.get("start") or 0.0) for row in rows)
        segment_end = max(float(row.get("end") or segment_start) for row in rows)

        def covered(speaker: str, start: float, end: float) -> float:
            intervals = [
                (max(start, win_start), min(end, win_end))
                for win_start, win_end, win_speaker in ordered_windows
                if win_speaker == speaker
                and min(end, win_end) > max(start, win_start)
            ]
            return union_duration(intervals)

        proposals: list[tuple[float, float, float, int, str, str]] = []
        for boundary_index in range(1, len(rows)):
            boundary = float(rows[boundary_index].get("start") or segment_start)
            if not (segment_start + 0.55 <= boundary <= segment_end - 0.55):
                continue
            if min((abs(boundary - point) for point in change_points), default=999.0) > 1.5:
                continue
            for first_speaker, second_speaker in ((current, alternate), (alternate, current)):
                left_support = covered(first_speaker, segment_start, boundary)
                right_support = covered(second_speaker, boundary, segment_end)
                left_conflict = covered(second_speaker, segment_start, boundary)
                right_conflict = covered(first_speaker, boundary, segment_end)
                support = left_support + right_support
                conflict = left_conflict + right_conflict
                purity = support / max(0.001, support + conflict)
                margin = support - conflict
                if (
                    left_support < 1.5
                    or right_support < 0.55
                    or purity < 0.64
                    or margin < 1.0
                ):
                    continue

                exact_conflict = False
                for row_index, row in enumerate(rows):
                    intended = first_speaker if row_index < boundary_index else second_speaker
                    direct_speaker = str(row.get("_direct_speaker") or "")
                    if (
                        row.get("_direct_decision") == "assign"
                        and row.get("_direct_scope") == "exact_sync_cue"
                        and direct_speaker
                        and direct_speaker != intended
                    ):
                        exact_conflict = True
                        break
                if exact_conflict:
                    continue
                proposals.append((margin, support, purity, boundary_index, first_speaker, second_speaker))

        if not proposals:
            return rows, False
        margin, _support, purity, boundary_index, first_speaker, second_speaker = max(proposals)
        collapsed: list[dict] = []
        for row_index, source_row in enumerate(rows):
            row = dict(source_row)
            intended = first_speaker if row_index < boundary_index else second_speaker
            if str(row.get("speaker") or "") != intended:
                row["speaker"] = intended
                row["confidence"] = round(min(0.78, max(0.55, purity)), 3)
                row["source"] = "campp_overlap_dominant_handoff"
                row["review"] = True
                row["_boundary_inherited"] = True
                row["_has_evidence"] = True
            collapsed.append(row)
        return collapsed, True

    trusted_whole_segment_flags = (
        "speaker_calibrated",
        "speaker_voiceprint_reidentified",
        "voice_band_repaired",
        "continuity_repaired",
        "speaker_handoff_voice_guard_repaired",
    )

    speaker_durations: dict[str, float] = {}
    speaker_segment_counts: dict[str, int] = {}
    global_labels: list[str] = []
    timeline_start: float | None = None
    timeline_end: float | None = None
    for segment in segments:
        speaker = str(segment.get("speaker") or "")
        if not speaker:
            continue
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + end - start
        speaker_segment_counts[speaker] = speaker_segment_counts.get(speaker, 0) + 1
        global_labels.append(speaker)
        timeline_start = start if timeline_start is None else min(timeline_start, start)
        timeline_end = end if timeline_end is None else max(timeline_end, end)

    total_speaker_duration = sum(speaker_durations.values())
    global_transitions = sum(
        left != right
        for left, right in zip(global_labels, global_labels[1:])
    )
    timeline_duration = max(0.0, float(timeline_end or 0.0) - float(timeline_start or 0.0))
    transitions_per_minute = global_transitions / max(timeline_duration / 60.0, 1.0 / 60.0)
    balanced_two_speaker_dialogue = (
        len(speaker_durations) == 2
        and total_speaker_duration >= 20.0
        and len(global_labels) >= 8
        and min(speaker_durations.values()) / total_speaker_duration >= 0.30
        and min(speaker_segment_counts.values()) >= 3
        and global_transitions >= 4
        and transitions_per_minute >= 2.0
        and not candidate.get("mergeable_speakers")
    )

    for seg_index, seg in enumerate(segments):
        seg.pop("speaker_cues", None)
        seg.pop("speaker_cue_review", None)
        seg.pop("speaker_cue_mode", None)
        current = str(seg.get("speaker") or "")
        raw_cues = seg.get("sync_cues")
        raw_windows = seg.get("speaker_subsegments")
        raw_embedding_rows = seg.get("speaker_cue_embeddings")
        if (
            not current
            or not isinstance(raw_cues, list)
            or not raw_cues
            or (
                (not isinstance(raw_windows, list) or not raw_windows)
                and (not isinstance(raw_embedding_rows, list) or not raw_embedding_rows)
            )
            or any(seg.get(flag) for flag in trusted_whole_segment_flags)
        ):
            continue

        windows: list[tuple[float, float, str]] = []
        for item in raw_windows or []:
            if not isinstance(item, dict):
                continue
            speaker = str(item.get("speaker") or "")
            try:
                start = float(item.get("start"))
                end = float(item.get("end"))
            except (TypeError, ValueError):
                continue
            if speaker and end > start:
                windows.append((start, end, speaker))
        embedding_rows: dict[int, dict] = {}
        for item in raw_embedding_rows or []:
            if not isinstance(item, dict):
                continue
            try:
                cue_index = int(item.get("cue_index"))
            except (TypeError, ValueError):
                continue
            embedding_rows[cue_index] = dict(item)
        if not windows and not embedding_rows:
            continue

        cue_rows: list[dict] = []
        has_alternate = False
        segment_review_cues = 0
        intra_cue_context_handoff = False
        overlap_dominant_handoff = False
        runner_up_return_handoff = False
        dominant_interior_override = False
        sustained_review_handoff = False
        for cue_index, cue in enumerate(raw_cues):
            if not isinstance(cue, dict):
                continue
            try:
                cue_start = max(float(seg.get("start", 0.0)), float(cue.get("start")))
                cue_end = min(float(seg.get("end", cue_start)), float(cue.get("end")))
            except (TypeError, ValueError):
                continue
            if cue_end <= cue_start:
                continue

            coverage: dict[str, list[tuple[float, float]]] = {}
            for start, end, speaker in windows:
                overlap_start = max(cue_start, start)
                overlap_end = min(cue_end, end)
                if overlap_end > overlap_start:
                    coverage.setdefault(speaker, []).append((overlap_start, overlap_end))
            votes = {
                speaker: union_duration(intervals)
                for speaker, intervals in coverage.items()
            }

            total = sum(votes.values())
            cue_duration = cue_end - cue_start
            assigned = current
            confidence = float(_speaker_confidence(seg) or 0.5)
            margin = 1.0
            evidence_coverage = 0.0
            review = False
            source = "short_window_projection"
            direct_assign = False
            direct_decision = ""
            direct_scope = ""
            direct_speaker = ""
            direct_score = 0.0
            direct_overlap = 0.0
            direct_second_speaker = ""
            direct_second_score = 0.0
            window_best_speaker = ""
            window_confidence = 0.0
            window_margin = 0.0
            window_best_duration = 0.0
            window_coverage = 0.0
            if total > 0:
                ordered = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
                best_speaker, best_duration = ordered[0]
                second_duration = ordered[1][1] if len(ordered) > 1 else 0.0
                confidence = best_duration / total
                window_best_speaker = best_speaker
                window_confidence = confidence
                margin = (best_duration - second_duration) / total
                evidence_coverage = best_duration / cue_duration
                window_margin = margin
                window_best_duration = best_duration
                window_coverage = evidence_coverage
                if best_speaker == current:
                    assigned = current
                elif confidence >= 0.58 and margin >= 0.20:
                    assigned = best_speaker
                    has_alternate = True
                else:
                    review = True

                if seg.get("speaker_overlap_risk") and (confidence < 0.72 or margin < 0.30):
                    review = True

            direct = embedding_rows.get(cue_index)
            if direct:
                direct_decision = str(direct.get("decision") or "")
                direct_speaker = str(direct.get("speaker") or "")
                direct_scope = str(direct.get("embedding_scope") or "")
                try:
                    direct_score = float(direct.get("score") or 0.0)
                    direct_margin = float(direct.get("margin") or 0.0)
                    direct_coverage = float(direct.get("voice_coverage_ratio") or 0.0)
                    direct_overlap = float(direct.get("overlap_ratio") or 0.0)
                    direct_second_score = float(direct.get("second_score") or 0.0)
                except (TypeError, ValueError):
                    direct_score = 0.0
                    direct_margin = 0.0
                    direct_coverage = 0.0
                    direct_overlap = 0.0
                    direct_second_score = 0.0
                direct_second_speaker = str(direct.get("second_speaker") or "")
                if direct_decision == "assign" and direct_speaker:
                    assigned = direct_speaker
                    confidence = direct_score
                    margin = direct_margin
                    evidence_coverage = direct_coverage
                    review = False
                    source = "campp_sync_cue_embedding"
                    direct_assign = True
                elif direct_decision == "review":
                    assigned = current
                    confidence = direct_score
                    margin = direct_margin
                    evidence_coverage = direct_coverage
                    review = direct_speaker != current
                    source = "campp_sync_cue_embedding_review"
                elif direct_decision == "insufficient" and direct_scope == "exact_sync_cue":
                    # Exact-cue evidence is more local than the 1.5-second
                    # sliding windows. Keep an alternate only when both sources
                    # independently agree; otherwise fail closed to the segment.
                    cross_source_assign = (
                        direct_speaker
                        and direct_speaker != current
                        and direct_score >= 0.62
                        and direct_margin >= 0.10
                        and direct_coverage >= 0.50
                        and window_best_speaker == direct_speaker
                        and window_confidence >= 0.70
                        and direct_overlap < 0.08
                    )
                    weak_exact_sliding_assign = (
                        direct_speaker
                        and direct_speaker != current
                        and direct_speaker == window_best_speaker
                        and direct_score < 0.55
                        and direct_margin < 0.05
                        and direct_coverage < 0.15
                        and window_confidence >= 0.90
                        and window_margin >= 0.80
                        and window_best_duration >= 0.90
                        and window_coverage >= 0.35
                        and direct_overlap < 0.10
                    )
                    trusted_alternate = bool(cross_source_assign or weak_exact_sliding_assign)
                    assigned = direct_speaker if trusted_alternate else current
                    confidence = window_confidence if weak_exact_sliding_assign else direct_score
                    margin = window_margin if weak_exact_sliding_assign else direct_margin
                    evidence_coverage = window_coverage if weak_exact_sliding_assign else direct_coverage
                    review = False
                    source = (
                        "campp_exact_sliding_agreement"
                        if cross_source_assign
                        else (
                            "campp_sliding_rescue_weak_exact"
                            if weak_exact_sliding_assign
                            else "campp_exact_cue_insufficient"
                        )
                    )
                    direct_assign = trusted_alternate

            row = {
                "cue_index": int(cue_index),
                "start": round(cue_start, 3),
                "end": round(cue_end, 3),
                "text": str(cue.get("text") or ""),
                "speaker": assigned,
                "confidence": round(max(0.0, min(1.0, confidence)), 3),
                "source": source,
                "_margin": margin,
                "_evidence_coverage": evidence_coverage,
                "_has_evidence": bool(votes) or direct_assign,
                "_direct_assign": direct_assign,
                "_direct_decision": direct_decision,
                "_direct_scope": direct_scope,
                "_direct_speaker": direct_speaker,
                "_direct_score": direct_score,
                "_direct_margin": direct_margin if direct else 0.0,
                "_direct_overlap": direct_overlap,
                "_direct_second_speaker": direct_second_speaker,
                "_direct_second_score": direct_second_score,
                "_window_best_speaker": window_best_speaker,
                "_window_confidence": window_confidence,
                "_window_margin": window_margin,
                "_window_best_duration": window_best_duration,
                "_boundary_inherited": False,
            }
            if votes:
                row["votes"] = {
                    speaker: round(duration, 3)
                    for speaker, duration in sorted(votes.items())
                }
            if review:
                row["review"] = True
                segment_review_cues += 1
            cue_rows.append(row)

        change_points = [
            float(value)
            for value in seg.get("speaker_change_points") or []
            if isinstance(value, (int, float))
        ]
        previous_segment = segments[seg_index - 1] if seg_index > 0 else None
        next_segment = segments[seg_index + 1] if seg_index + 1 < len(segments) else None
        previous_speaker = str((previous_segment or {}).get("speaker") or "")
        next_speaker = str((next_segment or {}).get("speaker") or "")
        try:
            seg_start = float(seg.get("start") or 0.0)
            seg_end = float(seg.get("end") or seg_start)
            previous_gap = seg_start - float((previous_segment or {}).get("end") or seg_start)
            next_gap = float((next_segment or {}).get("start") or seg_end) - seg_end
        except (TypeError, ValueError):
            seg_start = 0.0
            seg_end = 0.0
            previous_gap = 999.0
            next_gap = 999.0

        # A complete short question can occupy the first sync cue while the
        # answer owns the rest of the ASR segment. Promote a review-band first
        # cue only when exact CAM++ evidence, independent sliding windows, the
        # following current-speaker cue, and the acoustic change point all
        # agree. This changes only the logical speaker timeline.
        if (
            len(cue_rows) >= 2
            and 1 <= len(change_points) <= 2
            and not bool(seg.get("speaker_overlap_risk"))
            and float(seg.get("overlap_ratio") or 0.0) < 0.05
            and (_speaker_confidence(seg) or 0.0) >= 0.75
        ):
            first_row = cue_rows[0]
            later_rows = cue_rows[1:]
            alternate = str(first_row.get("_direct_speaker") or "")
            first_end = float(first_row.get("end") or 0.0)
            first_duration = max(
                0.0,
                first_end - float(first_row.get("start") or first_end),
            )
            later_duration = max(0.0, seg_end - first_end)
            later_exact_current = [
                row
                for row in later_rows
                if row.get("_direct_scope") == "exact_sync_cue"
                and row.get("_direct_decision") == "assign"
                and str(row.get("_direct_speaker") or "") == current
                and float(row.get("_direct_score") or 0.0) >= 0.75
                and float(row.get("_direct_margin") or 0.0) >= 0.12
            ]
            if (
                alternate
                and alternate != current
                and first_row.get("_direct_scope") == "exact_sync_cue"
                and first_row.get("_direct_decision") == "review"
                and str(first_row.get("speaker") or "") == current
                and float(first_row.get("_direct_score") or 0.0) >= 0.68
                and float(first_row.get("_direct_margin") or 0.0) >= 0.04
                and float(first_row.get("_evidence_coverage") or 0.0) >= 0.65
                and float(first_row.get("_direct_overlap") or 0.0) < 0.05
                and str(first_row.get("_window_best_speaker") or "") == alternate
                and float(first_row.get("_window_confidence") or 0.0) >= 0.85
                and float(first_row.get("_window_margin") or 0.0) >= 0.60
                and first_duration >= 0.8
                and later_duration >= 0.8
                and all(str(row.get("speaker") or "") == current for row in later_rows)
                and bool(later_exact_current)
                and min(abs(point - first_end) for point in change_points) <= 0.75
            ):
                first_row["speaker"] = alternate
                first_row["confidence"] = round(
                    min(0.70, float(first_row.get("_direct_score") or 0.5)),
                    3,
                )
                first_row["source"] = "campp_review_cue_context_handoff"
                first_row["review"] = True
                first_row["_boundary_inherited"] = True

        # A short sliding-window assignment at the start of an ASR segment can
        # still contain most of the preceding turn. When the whole segment and
        # the following segment agree, and a later cue has direct evidence for
        # that speaker, treat the leading alternate as boundary bleed. Exact
        # cue assignments are intentionally excluded because they can preserve
        # a real short reply at the boundary.
        if len(cue_rows) >= 2 and change_points:
            first_row = cue_rows[0]
            first_duration = max(
                0.0,
                float(first_row.get("end") or 0.0) - float(first_row.get("start") or 0.0),
            )
            later_rows = cue_rows[1:]
            first_change = min(change_points)
            if (
                previous_speaker
                and previous_speaker != current
                and next_speaker == current
                and previous_gap <= 0.25
                and next_gap <= 0.35
                and (_speaker_confidence(seg) or 1.0) <= 0.70
                and bool(seg.get("speaker_overlap_risk"))
                and float(seg.get("overlap_ratio") or 0.0) >= 0.20
                and (_speaker_confidence(next_segment or {}) or 0.0) >= 0.60
                and str(first_row.get("speaker") or "") == previous_speaker
                and first_row.get("_direct_scope") == "sliding_window_weighted"
                and bool(first_row.get("_direct_assign"))
                and float(first_row.get("_direct_score") or 0.0) <= 0.80
                and first_duration <= 1.25
                and seg_start <= first_change <= seg_start + 1.50
                and len(later_rows) >= 2
                and all(str(row.get("speaker") or "") == current for row in later_rows)
                and any(bool(row.get("_direct_assign")) for row in later_rows)
            ):
                first_row["speaker"] = current
                first_row["confidence"] = round(float(_speaker_confidence(seg) or 0.5), 3)
                first_row["source"] = "campp_leading_sliding_bleed_guard"
                first_row["review"] = True
                first_row["_direct_assign"] = False
                first_row["_boundary_inherited"] = True

        # A strong whole-segment assignment can outvote a weak sub-second
        # sliding window that still contains the preceding speaker. Unlike the
        # lower-confidence guard above, this does not depend on the next ASR
        # segment because the conversation may legitimately switch again there.
        # Exact-cue assignments remain untouched so real short replies survive.
        if len(cue_rows) >= 3 and change_points:
            first_row = cue_rows[0]
            later_rows = cue_rows[1:]
            first_duration = max(
                0.0,
                float(first_row.get("end") or 0.0) - float(first_row.get("start") or 0.0),
            )
            first_change = min(change_points)
            current_vote_share = _speaker_vote_share(seg, current)
            later_direct_current = sum(
                bool(row.get("_direct_assign"))
                and str(row.get("_direct_speaker") or "") == current
                for row in later_rows
            )
            if (
                previous_speaker
                and previous_speaker != current
                and previous_gap <= 0.25
                and (_speaker_confidence(seg) or 0.0) >= 0.90
                and current_vote_share is not None
                and current_vote_share >= 0.85
                and str(first_row.get("speaker") or "") == previous_speaker
                and first_row.get("_direct_scope") == "sliding_window_weighted"
                and bool(first_row.get("_direct_assign"))
                and first_duration < 1.0
                and float(first_row.get("_direct_score") or 0.0) <= 0.72
                and float(first_row.get("_direct_margin") or 0.0) <= 0.20
                and seg_start <= first_change <= seg_start + 1.30
                and len(later_rows) >= 2
                and all(str(row.get("speaker") or "") == current for row in later_rows)
                and later_direct_current >= 2
            ):
                first_row["speaker"] = current
                first_row["confidence"] = round(float(_speaker_confidence(seg) or 0.5), 3)
                first_row["source"] = "campp_leading_weak_sliding_bleed_guard"
                first_row["review"] = True
                first_row["_direct_assign"] = False
                first_row["_boundary_inherited"] = True

        # If an ambiguous sub-second leading cue independently has the previous
        # speaker as both its second-nearest exact centroid and its strongest
        # sliding-window label, preserve that short reply. This is deliberately
        # limited to a near tie followed by a direct current-speaker cue.
        if len(cue_rows) >= 2:
            first_row = cue_rows[0]
            second_row = cue_rows[1]
            first_duration = max(
                0.0,
                float(first_row.get("end") or 0.0) - float(first_row.get("start") or 0.0),
            )
            if (
                previous_speaker
                and previous_speaker != current
                and previous_gap <= 0.75
                and 0.30 <= first_duration <= 1.0
                and str(first_row.get("speaker") or "") == current
                and first_row.get("_direct_decision") in {"review", "insufficient"}
                and float(first_row.get("_direct_margin") or 0.0) <= 0.03
                and str(first_row.get("_direct_second_speaker") or "") == previous_speaker
                and float(first_row.get("_direct_second_score") or 0.0) >= 0.68
                and str(first_row.get("_window_best_speaker") or "") == previous_speaker
                and float(first_row.get("_window_best_duration") or 0.0) >= 0.60
                and str(second_row.get("speaker") or "") == current
                and bool(second_row.get("_direct_assign"))
            ):
                first_row["speaker"] = previous_speaker
                first_row["confidence"] = round(
                    min(0.70, float(first_row.get("_direct_second_score") or 0.5)),
                    3,
                )
                first_row["source"] = "campp_ambiguous_boundary_inherit_previous"
                first_row["review"] = True
                first_row["_boundary_inherited"] = True

        # A sub-second window from the previous turn can bleed across an ASR
        # boundary. Do not expose it as a hard handoff when the remainder of a
        # high-confidence segment and the following segment agree on the current
        # speaker. Sustained alternate runs are intentionally left untouched.
        if len(cue_rows) >= 2 and change_points:
            first_row = cue_rows[0]
            first_change = min(change_points)
            later_rows = cue_rows[1:]
            if (
                previous_speaker
                and previous_speaker != current
                and next_speaker == current
                and previous_gap <= 0.25
                and next_gap <= 0.35
                and (_speaker_confidence(seg) or 0.0) >= 0.90
                and str(first_row.get("speaker") or "") == previous_speaker
                and bool(first_row.get("_direct_assign"))
                and float(first_row.get("_direct_score") or 0.0) < 0.78
                and float(first_row.get("_evidence_coverage") or 0.0) < 0.60
                and float(first_row.get("_window_best_duration") or 0.0) < 1.0
                and seg_start <= first_change <= seg_start + 1.5
                and all(str(row.get("speaker") or "") == current for row in later_rows)
                and any(bool(row.get("_direct_assign")) for row in later_rows)
            ):
                first_row["speaker"] = current
                first_row["confidence"] = round(float(_speaker_confidence(seg) or 0.5), 3)
                first_row["source"] = "campp_boundary_bleed_guard"
                first_row["review"] = True
                first_row["_direct_assign"] = False
                first_row["_boundary_inherited"] = True

        # The first cue after an ASR boundary may be overlap-heavy while both
        # its local sliding evidence and the next exact cue agree on the same
        # alternate speaker. Keep the handoff instead of failing the first cue
        # back to the whole-segment label.
        if len(cue_rows) >= 2:
            first_row = cue_rows[0]
            next_row = cue_rows[1]
            first_duration = max(
                0.0,
                float(first_row.get("end") or 0.0) - float(first_row.get("start") or 0.0),
            )
            alternate = str(next_row.get("speaker") or "")
            adjacent_exact_agreement = (
                float(first_row.get("_direct_score") or 0.0) >= 0.72
                and float(first_row.get("_direct_margin") or 0.0) >= 0.15
                and str(first_row.get("_window_best_speaker") or "") == alternate
                and float(first_row.get("_window_confidence") or 0.0) >= 0.72
                and float(first_row.get("_window_margin") or 0.0) >= 0.40
                and float(first_row.get("_direct_overlap") or 0.0) <= 0.25
            )
            if (
                first_row.get("_direct_scope") == "exact_sync_cue"
                and first_row.get("_direct_decision") in {"review", "insufficient"}
                and 0.30 <= first_duration <= 1.50
                and float(first_row.get("_direct_score") or 0.0) >= 0.64
                and alternate
                and alternate != current
                and bool(next_row.get("_direct_assign"))
                and str(first_row.get("_direct_speaker") or "") == alternate
                and str(first_row.get("_window_best_speaker") or "") == alternate
                and (
                    float(first_row.get("_window_confidence") or 0.0) >= 0.90
                    or adjacent_exact_agreement
                )
            ):
                first_row["speaker"] = alternate
                first_row["confidence"] = round(
                    min(0.70, float(next_row.get("confidence") or 0.5)),
                    3,
                )
                first_row["source"] = "campp_boundary_inherit_next_exact"
                first_row["review"] = True
                first_row["_boundary_inherited"] = True

        # A long ASR cue can contain a real turn boundary. Express that boundary
        # in the diarization-only timeline when both neighboring stable speakers
        # and the short-window evidence agree. The original segment and sync cue
        # remain byte-for-byte unchanged.
        if (
            len(cue_rows) == 1
            and len(raw_cues) == 1
            and len(change_points) == 1
            and previous_speaker == current
            and next_speaker
            and next_speaker != current
            and previous_gap <= 0.35
            and next_gap <= 0.35
            and (_speaker_confidence(seg) or 1.0) <= 0.65
        ):
            point = change_points[0]
            row = cue_rows[0]
            cue_start = float(row.get("start") or 0.0)
            cue_end = float(row.get("end") or cue_start)

            def side_evidence(start: float, end: float) -> tuple[str, float, float]:
                side_coverage: dict[str, list[tuple[float, float]]] = {}
                for win_start, win_end, speaker in windows:
                    overlap_start = max(start, win_start)
                    overlap_end = min(end, win_end)
                    if overlap_end > overlap_start:
                        side_coverage.setdefault(speaker, []).append((overlap_start, overlap_end))
                durations = {
                    speaker: union_duration(intervals)
                    for speaker, intervals in side_coverage.items()
                }
                ordered = sorted(durations.items(), key=lambda item: (-item[1], item[0]))
                if not ordered:
                    return "", 0.0, 0.0
                total = sum(durations.values())
                best_speaker, best_duration = ordered[0]
                confidence = best_duration / total if total > 0 else 0.0
                return best_speaker, best_duration, confidence

            left_speaker, left_voice, left_confidence = side_evidence(cue_start, point)
            right_speaker, right_voice, right_confidence = side_evidence(point, cue_end)
            direct_is_mixed = (
                row.get("_direct_scope") == "exact_sync_cue"
                and (
                    (
                        row.get("_direct_decision") == "insufficient"
                        and float(row.get("_direct_score") or 0.0) <= 0.60
                        and float(row.get("_direct_margin") or 0.0) <= 0.08
                    )
                    or (
                        row.get("_direct_decision") == "review"
                        and float(row.get("_direct_score") or 0.0) <= 0.68
                        and float(row.get("_direct_margin") or 0.0) <= 0.08
                        and float(row.get("_direct_overlap") or 0.0) <= 0.10
                    )
                )
            )
            raw_text = str(raw_cues[0].get("text") or "") if isinstance(raw_cues[0], dict) else ""
            wall_ratio = (point - cue_start) / max(cue_end - cue_start, 0.001)
            voice_ratio = left_voice / max(left_voice + right_voice, 0.001)
            left_text, right_text = split_text_for_handoff(raw_text, wall_ratio, voice_ratio)
            if (
                cue_start + 0.8 <= point <= cue_end - 0.8
                and direct_is_mixed
                and left_speaker == current
                and right_speaker == next_speaker
                and left_voice >= 0.75
                and right_voice >= 0.75
                and left_confidence >= 0.75
                and right_confidence >= 0.75
                and left_text
                and right_text
            ):
                cue_rows = [
                    {
                        "cue_index": 0,
                        "start": round(cue_start, 3),
                        "end": round(point, 3),
                        "text": left_text,
                        "speaker": current,
                        "confidence": round(min(0.85, left_confidence), 3),
                        "source": "campp_intracue_context_handoff",
                        "review": True,
                        "_has_evidence": True,
                        "_direct_assign": False,
                        "_boundary_inherited": True,
                    },
                    {
                        "cue_index": 0,
                        "start": round(point, 3),
                        "end": round(cue_end, 3),
                        "text": right_text,
                        "speaker": next_speaker,
                        "confidence": round(min(0.85, right_confidence), 3),
                        "source": "campp_intracue_context_handoff",
                        "review": True,
                        "_has_evidence": True,
                        "_direct_assign": False,
                        "_boundary_inherited": True,
                    },
                ]
                intra_cue_context_handoff = True

        # If the acoustic change point falls inside a short review-band cue,
        # the 1.5-second window often moves the boundary too early. Preserve
        # the preceding strong speaker for that indivisible cue and switch on
        # the next cue, without changing ASR text or timestamps.
        for cue_position in range(1, max(1, len(cue_rows) - 1)):
            if cue_position + 1 >= len(cue_rows):
                break
            previous_row = cue_rows[cue_position - 1]
            row = cue_rows[cue_position]
            next_row = cue_rows[cue_position + 1]
            row_start = float(row.get("start") or 0.0)
            row_end = float(row.get("end") or row_start)
            inside_change = any(
                row_start + 0.15 <= point <= row_end
                for point in change_points
            )
            previous_row_speaker = str(previous_row.get("speaker") or "")
            if (
                row.get("_direct_scope") == "exact_sync_cue"
                and row.get("_direct_decision") == "review"
                and 0.30 <= row_end - row_start <= 1.50
                and inside_change
                and float(row.get("_direct_score") or 0.0) < 0.67
                and bool(previous_row.get("_direct_assign"))
                and previous_row_speaker
                and previous_row_speaker != current
                and str(next_row.get("speaker") or "") == current
            ):
                row["speaker"] = previous_row_speaker
                row["confidence"] = round(
                    min(0.70, float(previous_row.get("confidence") or 0.5)),
                    3,
                )
                row["source"] = "campp_change_point_inherit_previous"
                row["review"] = True
                row["_boundary_inherited"] = True

        # ASR can produce a short cue over a VAD gap exactly at a speaker
        # boundary. Attach that gap to the following strong speaker instead of
        # defaulting it to the whole segment's label.
        for cue_position in range(1, max(1, len(cue_rows) - 1)):
            if cue_position + 1 >= len(cue_rows):
                break
            previous_row = cue_rows[cue_position - 1]
            row = cue_rows[cue_position]
            next_row = cue_rows[cue_position + 1]
            duration = max(0.0, float(row.get("end") or 0.0) - float(row.get("start") or 0.0))
            if (
                row.get("_direct_scope") == "exact_sync_cue"
                and row.get("_direct_decision") == "insufficient"
                and duration <= 0.75
                and float(row.get("_evidence_coverage") or 0.0) < 0.15
                and bool(next_row.get("_direct_assign"))
                and str(next_row.get("speaker") or "")
                and str(previous_row.get("speaker") or "") != str(next_row.get("speaker") or "")
            ):
                row["speaker"] = str(next_row.get("speaker"))
                row["confidence"] = round(min(0.7, float(next_row.get("confidence") or 0.5)), 3)
                row["source"] = "campp_boundary_inherit_next"
                row["review"] = True
                row["_boundary_inherited"] = True

        # A low-confidence segment can contain the returning speaker followed by
        # the current speaker. Stable centroids sometimes keep the first cues on
        # the current cluster by only a few cosine points. Require the same
        # runner-up on every leading cue, a recent clean turn from that speaker,
        # and a strong current-speaker terminal cue before exposing the handoff.
        if (
            len(cue_rows) >= 3
            and (_speaker_confidence(seg) or 1.0) <= 0.50
            and bool(seg.get("speaker_overlap_risk"))
        ):
            leading_rows = cue_rows[:-1]
            terminal_row = cue_rows[-1]
            runner_ups = {
                str(row.get("_direct_second_speaker") or "")
                for row in leading_rows
                if str(row.get("_direct_second_speaker") or "")
            }
            alternate = next(iter(runner_ups), "") if len(runner_ups) == 1 else ""
            current_vote_share = _speaker_vote_share(seg, current)
            alternate_vote_share = _speaker_vote_share(seg, alternate) if alternate else None
            prior_alternate_index = next(
                (
                    index
                    for index in range(seg_index - 1, max(-1, seg_index - 7), -1)
                    if str(segments[index].get("speaker") or "") == alternate
                    and (_speaker_confidence(segments[index]) or 0.0) >= 0.75
                ),
                None,
            )
            prior_alternate_recent = False
            if prior_alternate_index is not None:
                try:
                    prior_alternate_recent = (
                        seg_start - float(segments[prior_alternate_index].get("end") or seg_start) <= 60.0
                        and all(
                            str(row.get("speaker") or "") != current
                            for row in segments[prior_alternate_index + 1:seg_index]
                        )
                    )
                except (TypeError, ValueError):
                    prior_alternate_recent = False
            terminal_start = float(terminal_row.get("start") or 0.0)
            if (
                alternate
                and alternate != current
                and len(leading_rows) >= 2
                and all(str(row.get("speaker") or "") == current for row in leading_rows)
                and all(row.get("_direct_scope") == "exact_sync_cue" for row in leading_rows)
                and all(row.get("_direct_decision") in {"review", "insufficient"} for row in leading_rows)
                and all(str(row.get("_direct_speaker") or "") == current for row in leading_rows)
                and all(float(row.get("_direct_margin") or 0.0) <= 0.05 for row in leading_rows)
                and all(float(row.get("_direct_second_score") or 0.0) >= 0.55 for row in leading_rows)
                and str(terminal_row.get("speaker") or "") == current
                and bool(terminal_row.get("_direct_assign"))
                and str(terminal_row.get("_direct_speaker") or "") == current
                and float(terminal_row.get("_direct_score") or 0.0) >= 0.72
                and float(terminal_row.get("_direct_margin") or 0.0) >= 0.15
                and any(abs(point - terminal_start) <= 1.0 for point in change_points)
                and current_vote_share is not None
                and current_vote_share <= 0.50
                and alternate_vote_share is not None
                and alternate_vote_share >= 0.30
                and alternate_vote_share >= current_vote_share * 0.75
                and prior_alternate_recent
            ):
                for row in leading_rows:
                    row["speaker"] = alternate
                    row["confidence"] = round(
                        min(0.70, float(row.get("_direct_second_score") or 0.5)),
                        3,
                    )
                    row["source"] = "campp_consistent_runner_up_return"
                    row["review"] = True
                    row["_direct_assign"] = False
                    row["_boundary_inherited"] = True
                runner_up_return_handoff = True

        # A long, strong interior cue can reveal that short boundary windows
        # leaked from the following turn. Collapse current/alternate/current to
        # the interior speaker only when the leading cue is explicitly
        # insufficient, the terminal cue is sub-second sliding evidence, and
        # the preceding stable segment already belongs to the interior speaker.
        if len(cue_rows) == 3:
            first_row, middle_row, last_row = cue_rows
            middle_speaker = str(middle_row.get("speaker") or "")
            middle_duration = max(
                0.0,
                float(middle_row.get("end") or 0.0) - float(middle_row.get("start") or 0.0),
            )
            total_cue_duration = sum(
                max(0.0, float(row.get("end") or 0.0) - float(row.get("start") or 0.0))
                for row in cue_rows
            )
            last_duration = max(
                0.0,
                float(last_row.get("end") or 0.0) - float(last_row.get("start") or 0.0),
            )
            if (
                middle_speaker
                and middle_speaker != current
                and str(first_row.get("speaker") or "") == current
                and str(last_row.get("speaker") or "") == current
                and previous_speaker == middle_speaker
                and previous_gap <= 1.0
                and next_speaker == current
                and next_gap <= 0.35
                and (_speaker_confidence(seg) or 1.0) <= 0.65
                and bool(seg.get("speaker_overlap_risk"))
                and middle_duration >= 3.0
                and middle_duration >= total_cue_duration * 0.55
                and middle_row.get("_direct_scope") == "exact_sync_cue"
                and middle_row.get("_direct_decision") == "assign"
                and float(middle_row.get("_direct_score") or 0.0) >= 0.69
                and float(middle_row.get("_direct_margin") or 0.0) >= 0.18
                and first_row.get("_direct_scope") == "exact_sync_cue"
                and first_row.get("_direct_decision") in {"review", "insufficient"}
                and not bool(first_row.get("_direct_assign"))
                and float(first_row.get("_direct_score") or 0.0) <= 0.64
                and last_row.get("_direct_scope") == "sliding_window_weighted"
                and bool(last_row.get("_direct_assign"))
                and last_duration <= 1.0
                and float(last_row.get("_direct_score") or 0.0) <= 0.74
                and float(last_row.get("_direct_margin") or 0.0) <= 0.16
            ):
                for row in (first_row, last_row):
                    row["speaker"] = middle_speaker
                    row["confidence"] = round(
                        min(0.70, float(middle_row.get("confidence") or 0.5)),
                        3,
                    )
                    row["source"] = "campp_dominant_interior_boundary_guard"
                    row["review"] = True
                    row["_direct_assign"] = False
                    row["_boundary_inherited"] = True
                dominant_interior_override = True

        # A very short terminal fragment often belongs to the strong utterance
        # immediately before it. Longer exact-cue failures remain on the whole
        # segment speaker, which also permits a conservative return after a
        # short interjection.
        if len(cue_rows) >= 2:
            previous_row = cue_rows[-2]
            last_row = cue_rows[-1]
            last_duration = max(
                0.0,
                float(last_row.get("end") or 0.0) - float(last_row.get("start") or 0.0),
            )
            if (
                last_row.get("_direct_scope") == "exact_sync_cue"
                and last_row.get("_direct_decision") == "review"
                and last_duration <= 0.65
                and str(previous_row.get("speaker") or "") != current
            ):
                last_row["speaker"] = str(previous_row.get("speaker"))
                last_row["confidence"] = round(
                    min(0.7, float(previous_row.get("confidence") or 0.5)),
                    3,
                )
                last_row["source"] = "campp_boundary_inherit_previous"
                last_row["review"] = True
                last_row["_boundary_inherited"] = True
            elif (
                last_row.get("_direct_scope") == "exact_sync_cue"
                and last_row.get("_direct_decision") == "insufficient"
                and last_duration <= 0.40
                and bool(previous_row.get("_direct_assign"))
                and str(previous_row.get("speaker") or "") != current
            ):
                last_row["speaker"] = str(previous_row.get("speaker"))
                last_row["confidence"] = round(
                    min(0.65, float(previous_row.get("confidence") or 0.5)),
                    3,
                )
                last_row["source"] = "campp_boundary_inherit_previous"
                last_row["review"] = True
                last_row["_boundary_inherited"] = True

        # Sliding windows straddle a real boundary and can pull the next
        # speaker into the final sub-second cue too early. When the following
        # ASR segment confirms that speaker but the preceding sustained cue has
        # strong exact evidence, keep the final cue with the preceding turn and
        # move the visible handoff to the immutable segment boundary.
        if len(cue_rows) >= 2:
            previous_row = cue_rows[-2]
            last_row = cue_rows[-1]
            previous_row_speaker = str(previous_row.get("speaker") or "")
            last_row_speaker = str(last_row.get("speaker") or "")
            last_start = float(last_row.get("start") or 0.0)
            last_end = float(last_row.get("end") or last_start)
            last_duration = max(0.0, last_end - last_start)
            preceding_run_duration = 0.0
            for row in reversed(cue_rows[:-1]):
                if str(row.get("speaker") or "") != previous_row_speaker:
                    break
                preceding_run_duration += max(
                    0.0,
                    float(row.get("end") or 0.0) - float(row.get("start") or 0.0),
                )
            if (
                previous_row_speaker
                and last_row_speaker == next_speaker
                and previous_row_speaker != last_row_speaker
                and next_gap <= 0.35
                and last_duration <= 1.05
                and last_row.get("_direct_scope") == "sliding_window_weighted"
                and last_row.get("_direct_decision") == "assign"
                and bool(last_row.get("_direct_assign"))
                and float(last_row.get("_direct_score") or 0.0) <= 0.80
                and float(last_row.get("_direct_margin") or 0.0) <= 0.23
                and preceding_run_duration >= 1.50
                and previous_row.get("_direct_scope") == "exact_sync_cue"
                and previous_row.get("_direct_decision") == "assign"
                and float(previous_row.get("_direct_score") or 0.0) >= 0.75
                and float(previous_row.get("_direct_margin") or 0.0) >= 0.15
                and any(last_start - 1.50 <= point <= last_end for point in change_points)
            ):
                last_row["speaker"] = previous_row_speaker
                last_row["confidence"] = round(
                    min(0.70, float(previous_row.get("confidence") or 0.5)),
                    3,
                )
                last_row["source"] = "campp_terminal_sliding_bleed_guard"
                last_row["review"] = True
                last_row["_direct_assign"] = False
                last_row["_boundary_inherited"] = True

        # A sub-second alternate island surrounded by the whole-segment
        # speaker is usually sliding-window jitter. Preserve real short replies
        # when an exact cue embedding supports them; otherwise inherit the
        # stable outer turn.
        if len(cue_rows) >= 3 and (_speaker_confidence(seg) or 1.0) <= 0.65:
            cue_runs: list[dict] = []
            for row_index, row in enumerate(cue_rows):
                speaker = str(row.get("speaker") or "")
                duration = max(
                    0.0,
                    float(row.get("end") or 0.0) - float(row.get("start") or 0.0),
                )
                if cue_runs and cue_runs[-1]["speaker"] == speaker:
                    cue_runs[-1]["duration"] += duration
                    cue_runs[-1]["indices"].append(row_index)
                else:
                    cue_runs.append({
                        "speaker": speaker,
                        "duration": duration,
                        "indices": [row_index],
                    })
            for run_index in range(1, len(cue_runs) - 1):
                previous_run = cue_runs[run_index - 1]
                island_run = cue_runs[run_index]
                following_run = cue_runs[run_index + 1]
                if (
                    previous_run["speaker"] == current
                    and following_run["speaker"] == current
                    and island_run["speaker"] != current
                    and float(previous_run["duration"]) >= 1.50
                    and float(island_run["duration"]) <= 0.60
                    and float(following_run["duration"]) >= 0.40
                    and all(
                        cue_rows[index].get("_direct_scope") == "sliding_window_weighted"
                        and cue_rows[index].get("_direct_decision") == "assign"
                        and bool(cue_rows[index].get("_direct_assign"))
                        for index in island_run["indices"]
                    )
                    and any(
                        cue_rows[index].get("_direct_scope") == "exact_sync_cue"
                        and cue_rows[index].get("_direct_decision") == "assign"
                        and str(cue_rows[index].get("speaker") or "") == current
                        and float(cue_rows[index].get("_direct_score") or 0.0) >= 0.72
                        and float(cue_rows[index].get("_direct_margin") or 0.0) >= 0.15
                        for index in previous_run["indices"]
                    )
                ):
                    for index in island_run["indices"]:
                        row = cue_rows[index]
                        row["speaker"] = current
                        row["confidence"] = round(
                            min(0.70, float(_speaker_confidence(seg) or 0.5)),
                            3,
                        )
                        row["source"] = "campp_subsecond_sliding_island_guard"
                        row["review"] = True
                        row["_direct_assign"] = False
                        row["_boundary_inherited"] = True

        # Sync cues are alignment chunks, not guaranteed speaker turns. A
        # moderate-confidence B/A/B island inside one unfinished sentence is
        # usually centroid jitter rather than a real two-second interjection.
        # Keep strong or punctuated interjections, which are the cases where a
        # true return handoff is acoustically and linguistically supported.
        if len(cue_rows) == 3:
            first_row, middle_row, last_row = cue_rows
            middle_speaker = str(middle_row.get("speaker") or "")
            current_vote_share = _speaker_vote_share(seg, current)
            middle_vote_share = _speaker_vote_share(seg, middle_speaker)
            middle_duration = max(
                0.0,
                float(middle_row.get("end") or 0.0) - float(middle_row.get("start") or 0.0),
            )
            middle_window_votes = middle_row.get("votes") or {}
            try:
                current_window_vote = float(middle_window_votes.get(current) or 0.0)
                alternate_window_vote = float(middle_window_votes.get(middle_speaker) or 0.0)
            except (TypeError, ValueError):
                current_window_vote = 0.0
                alternate_window_vote = 0.0
            if (
                str(first_row.get("speaker") or "") == current
                and middle_speaker
                and middle_speaker != current
                and str(last_row.get("speaker") or "") == current
                and current_vote_share is not None
                and current_vote_share >= 0.78
                and middle_vote_share is not None
                and middle_vote_share <= 0.25
                and middle_duration <= 2.50
                and (_speaker_confidence(seg) or 0.0) >= 0.75
                and middle_row.get("_direct_scope") == "exact_sync_cue"
                and middle_row.get("_direct_decision") == "assign"
                and float(middle_row.get("_direct_score") or 0.0) <= 0.76
                and float(middle_row.get("_direct_margin") or 0.0) <= 0.18
                and float(middle_row.get("_direct_overlap") or 0.0) <= 0.05
                and current_window_vote >= 1.25 * max(alternate_window_vote, 0.001)
                and not has_closed_sentence(cue_text(raw_cues, 0))
                and not has_closed_sentence(cue_text(raw_cues, 1))
            ):
                middle_row["speaker"] = current
                middle_row["confidence"] = round(float(_speaker_confidence(seg) or 0.5), 3)
                middle_row["source"] = "campp_intrasentence_return_guard"
                middle_row["review"] = True
                middle_row["_direct_assign"] = False
                middle_row["_boundary_inherited"] = True

        # Two or more adjacent cue embeddings can agree on a real embedded
        # turn while each individual cue remains in the conservative review
        # band. Promote only a sustained run that is independently supported
        # by sliding windows, bounded by the whole-segment speaker on both
        # sides, and bracketed by acoustic change points. This affects only
        # speaker_cues; canonical ASR text, segment geometry, and sync cues stay
        # immutable.
        if len(cue_rows) >= 4 and 1 <= len(change_points) <= 3:
            run_start = 0
            while run_start < len(cue_rows):
                first = cue_rows[run_start]
                alternate = str(first.get("_direct_speaker") or "")
                eligible = (
                    alternate
                    and alternate != current
                    and first.get("_direct_decision") == "review"
                    and float(first.get("_direct_score") or 0.0) >= 0.65
                    and float(first.get("_direct_margin") or 0.0) >= 0.10
                    and float(first.get("_evidence_coverage") or 0.0) >= 0.30
                    and float(first.get("_direct_overlap") or 0.0) < 0.50
                    and first.get("_direct_scope") == "exact_sync_cue"
                    and str(first.get("_window_best_speaker") or "") == alternate
                )
                if not eligible:
                    run_start += 1
                    continue

                run_end = run_start + 1
                while run_end < len(cue_rows):
                    row = cue_rows[run_end]
                    if not (
                        str(row.get("_direct_speaker") or "") == alternate
                        and row.get("_direct_decision") == "review"
                        and float(row.get("_direct_score") or 0.0) >= 0.65
                        and float(row.get("_direct_margin") or 0.0) >= 0.10
                        and float(row.get("_evidence_coverage") or 0.0) >= 0.30
                        and float(row.get("_direct_overlap") or 0.0) < 0.50
                        and (
                            row.get("_direct_scope") == "sliding_window_weighted"
                            or str(row.get("_window_best_speaker") or "") == alternate
                        )
                    ):
                        break
                    run_end += 1

                run_rows = cue_rows[run_start:run_end]
                run_duration = sum(
                    max(0.0, float(row.get("end") or 0.0) - float(row.get("start") or 0.0))
                    for row in run_rows
                )
                before_rows = cue_rows[:run_start]
                after_rows = cue_rows[run_end:]

                def strong_current_boundary(rows: list[dict]) -> bool:
                    return any(
                        str(row.get("speaker") or "") == current
                        and str(row.get("_direct_speaker") or "") == current
                        and float(row.get("_direct_score") or 0.0) >= 0.68
                        for row in rows
                    )

                run_wall_start = float(run_rows[0].get("start") or 0.0)
                run_wall_end = float(run_rows[-1].get("end") or run_wall_start)
                boundaries_supported = (
                    any(abs(point - run_wall_start) <= 1.25 for point in change_points)
                    and any(abs(point - run_wall_end) <= 1.25 for point in change_points)
                )
                if (
                    len(run_rows) >= 2
                    and run_duration >= 1.50
                    and before_rows
                    and after_rows
                    and strong_current_boundary(before_rows)
                    and strong_current_boundary(after_rows)
                    and boundaries_supported
                    and (_speaker_confidence(seg) or 0.0) >= 0.70
                ):
                    for row in run_rows:
                        row["speaker"] = alternate
                        row["confidence"] = round(
                            min(0.72, float(row.get("_direct_score") or 0.5)),
                            3,
                        )
                        row["source"] = "campp_sustained_review_handoff"
                        row["review"] = True
                        row["_boundary_inherited"] = True
                    sustained_review_handoff = True
                    break
                run_start = max(run_end, run_start + 1)

        cue_rows, overlap_dominant_handoff = collapse_overlap_alternation_to_handoff(
            cue_rows,
            windows,
            current,
            change_points,
            _speaker_confidence(seg),
            bool(seg.get("speaker_overlap_risk")),
        )
        segment_review_cues = sum(bool(row.get("review")) for row in cue_rows)

        labels = [str(row.get("speaker") or "") for row in cue_rows]
        has_alternate = any(label and label != current for label in labels)
        transitions = sum(
            left != right
            for left, right in zip(labels, labels[1:])
        )
        unique_labels = {label for label in labels if label}
        change_point_count = len(seg.get("speaker_change_points") or [])
        segment_confidence = _speaker_confidence(seg)
        runs: list[dict] = []
        for row in cue_rows:
            label = str(row.get("speaker") or "")
            duration = max(0.0, float(row.get("end") or 0.0) - float(row.get("start") or 0.0))
            if runs and runs[-1]["speaker"] == label:
                runs[-1]["duration"] += duration
            else:
                runs.append({"speaker": label, "duration": duration})
        stable_single_handoff = (
            has_alternate
            and len(unique_labels) == 2
            and transitions == 1
            and 1 <= change_point_count <= 2
            and len(runs) == 2
            and all(float(run["duration"]) >= 0.8 for run in runs)
            and segment_confidence is not None
            and segment_confidence <= 0.65
        )
        stable_return_handoff = (
            has_alternate
            and len(unique_labels) == 2
            and transitions == 2
            and change_point_count == 2
            and len(runs) == 3
            and runs[0]["speaker"] == runs[2]["speaker"]
            and runs[0]["speaker"] != runs[1]["speaker"]
            and runs[1]["speaker"] == current
            and float(runs[0]["duration"]) >= 0.8
            and float(runs[1]["duration"]) >= 1.6
            and float(runs[2]["duration"]) >= 0.8
            and segment_confidence is not None
            and segment_confidence >= 0.75
        )
        balanced_dialogue_handoff = (
            balanced_two_speaker_dialogue
            and has_alternate
            and len(unique_labels) == 2
            and transitions >= 1
            and change_point_count >= transitions
            and change_point_count <= transitions + 2
            and len(runs) >= 2
            and all(float(run["duration"]) >= 0.45 for run in runs)
            and all(bool(row.get("_has_evidence")) for row in cue_rows)
            and all(float(row.get("_evidence_coverage") or 0.0) >= 0.15 for row in cue_rows)
            and segment_confidence is not None
            and segment_confidence <= 0.90
        )
        alternate_rows = [
            row for row in cue_rows
            if str(row.get("speaker") or "") != current
        ]
        direct_embedding_handoff = (
            has_alternate
            and len(unique_labels) == 2
            and transitions in {1, 2}
            and len(runs) in {2, 3}
            and all(float(run["duration"]) >= 0.45 for run in runs)
            and bool(alternate_rows)
            and all(
                bool(row.get("_direct_assign")) or bool(row.get("_boundary_inherited"))
                for row in alternate_rows
            )
            and all(
                not bool(row.get("review")) or bool(row.get("_boundary_inherited"))
                for row in alternate_rows
            )
            and (
                transitions == 1
                or (
                    len(runs) == 3
                    and runs[0]["speaker"] == runs[2]["speaker"]
                    and runs[0]["speaker"] != runs[1]["speaker"]
                )
            )
        )
        context_anchored_multi_handoff = (
            has_alternate
            and len(unique_labels) == 3
            and transitions == 2
            and len(runs) == 3
            and runs[1]["speaker"] == current
            and runs[0]["speaker"] == previous_speaker
            and runs[2]["speaker"] == next_speaker
            and previous_speaker
            and next_speaker
            and previous_speaker != current
            and next_speaker != current
            and previous_speaker != next_speaker
            and previous_gap <= 0.75
            and next_gap <= 0.10
            and float(runs[0]["duration"]) >= 0.30
            and float(runs[1]["duration"]) >= 2.0
            and float(runs[2]["duration"]) >= 0.80
            and bool(cue_rows[0].get("_boundary_inherited"))
            and bool(cue_rows[-1].get("_direct_assign"))
            and cue_rows[-1].get("_direct_decision") == "assign"
            and float(cue_rows[-1].get("_direct_score") or 0.0) >= 0.70
            and float(cue_rows[-1].get("_direct_margin") or 0.0) >= 0.15
        )
        if (
            stable_single_handoff
            or stable_return_handoff
            or balanced_dialogue_handoff
            or direct_embedding_handoff
            or context_anchored_multi_handoff
            or overlap_dominant_handoff
            or runner_up_return_handoff
            or dominant_interior_override
            or sustained_review_handoff
        ) and cue_rows:
            seg["speaker_cues"] = [
                {
                    key: value
                    for key, value in row.items()
                    if not key.startswith("_")
                }
                for row in cue_rows
            ]
            if sustained_review_handoff:
                seg["speaker_cue_mode"] = "campp_sustained_review_handoff"
            elif dominant_interior_override:
                seg["speaker_cue_mode"] = "campp_dominant_interior_override"
            elif runner_up_return_handoff:
                seg["speaker_cue_mode"] = "campp_consistent_runner_up_return"
            elif overlap_dominant_handoff:
                seg["speaker_cue_mode"] = "campp_overlap_dominant_handoff"
            elif context_anchored_multi_handoff:
                seg["speaker_cue_mode"] = "campp_context_anchored_multi_handoff"
            elif intra_cue_context_handoff:
                seg["speaker_cue_mode"] = "campp_intracue_context_handoff"
            elif direct_embedding_handoff:
                seg["speaker_cue_mode"] = "campp_sync_cue_embedding"
            elif balanced_dialogue_handoff and not (stable_single_handoff or stable_return_handoff):
                seg["speaker_cue_mode"] = "balanced_two_speaker_dialogue"
            if any(row.get("review") for row in cue_rows):
                seg["speaker_cue_review"] = True
            projected_segments += 1
            projected_cues += len(cue_rows)
            review_cues += segment_review_cues

    projected = {**candidate, "segments": segments}
    projected["speaker_cue_segment_count"] = projected_segments
    projected["speaker_cue_count"] = projected_cues
    projected["speaker_cue_review_count"] = review_cues
    return projected


def _materialize_projected_speaker_handoffs(candidate: dict) -> dict:
    """Turn one trusted cue-level A->B handoff into two transcript segments.

    The ASR text and sync cues are immutable atoms. A split is allowed only
    when direct cue embeddings, the short-window change timeline, and an
    existing sync-cue boundary agree. Ambiguous/overlapping evidence remains a
    review item instead of changing transcript geometry.
    """
    segments = [dict(seg) for seg in candidate.get("segments") or []]
    out: list[dict] = []
    review_segments: list[dict] = []
    split_count = 0
    tolerance = 0.02
    trusted_sources = {
        "campp_sync_cue_embedding",
        "campp_exact_sliding_agreement",
        "campp_sliding_rescue_weak_exact",
    }

    def reject(index: int, seg: dict, target: str, reason: str) -> None:
        held = dict(seg)
        held["speaker_resegmentation_review"] = True
        out.append(held)
        review_segments.append(_review_segment_payload(
            index,
            held,
            str(seg.get("speaker") or ""),
            target,
            reason,
        ))

    for index, seg in enumerate(segments):
        raw_cues = seg.get("sync_cues")
        speaker_cues = seg.get("speaker_cues")
        if not isinstance(speaker_cues, list) or len(speaker_cues) < 2:
            out.append(seg)
            continue

        target_speakers = [
            str(item.get("speaker") or "")
            for item in speaker_cues
            if isinstance(item, dict)
        ]
        target = next(
            (speaker for speaker in target_speakers if speaker and speaker != str(seg.get("speaker") or "")),
            str(seg.get("speaker") or ""),
        )
        if (
            not isinstance(raw_cues, list)
            or len(raw_cues) != len(speaker_cues)
            or len(raw_cues) < 2
        ):
            reject(index, seg, target, "段内换人证据存在，但同步 cue 不完整，已保留原段并标为待确认")
            continue

        try:
            segment_start = float(seg.get("start"))
            segment_end = float(seg.get("end"))
        except (TypeError, ValueError):
            reject(index, seg, target, "段内换人证据存在，但原段时间轴无效，已保留原段并标为待确认")
            continue

        valid_cues = True
        previous_end: float | None = None
        for cue, projected in zip(raw_cues, speaker_cues):
            if not isinstance(cue, dict) or not isinstance(projected, dict):
                valid_cues = False
                break
            try:
                cue_start = float(cue.get("start"))
                cue_end = float(cue.get("end"))
                projected_start = float(projected.get("start"))
                projected_end = float(projected.get("end"))
            except (TypeError, ValueError):
                valid_cues = False
                break
            if (
                cue_end <= cue_start
                or abs(cue_start - projected_start) > tolerance
                or abs(cue_end - projected_end) > tolerance
                or str(cue.get("text") or "") != str(projected.get("text") or "")
                or (previous_end is not None and abs(cue_start - previous_end) > tolerance)
            ):
                valid_cues = False
                break
            previous_end = cue_end
        cue_text = "".join(str(cue.get("text") or "") for cue in raw_cues if isinstance(cue, dict))
        if (
            not valid_cues
            or cue_text != str(seg.get("text") or "")
            or abs(float(raw_cues[0].get("start")) - segment_start) > tolerance
            or abs(float(raw_cues[-1].get("end")) - segment_end) > tolerance
        ):
            reject(index, seg, target, "段内换人证据存在，但同步 cue 未完整覆盖原文/时间轴，已保留原段并标为待确认")
            continue

        labels = [str(item.get("speaker") or "") for item in speaker_cues]
        transition_positions = [
            position
            for position in range(1, len(labels))
            if labels[position] != labels[position - 1]
        ]
        if (
            len(set(labels)) != 2
            or len(transition_positions) != 1
            or not all(labels)
            or seg.get("speaker_cue_mode") != "campp_sync_cue_embedding"
            or seg.get("speaker_cue_review")
            or seg.get("speaker_overlap_risk")
            or float(seg.get("overlap_ratio") or 0.0) >= 0.08
        ):
            reject(index, seg, target, "cue 级声纹仍存在多次跳变/重叠或待确认风险，已保留原段")
            continue

        boundary_index = transition_positions[0]
        left_boundary = speaker_cues[boundary_index - 1]
        right_boundary = speaker_cues[boundary_index]
        try:
            split_time = float(raw_cues[boundary_index - 1].get("end"))
            right_start = float(raw_cues[boundary_index].get("start"))
            left_confidence = float(left_boundary.get("confidence") or 0.0)
            right_confidence = float(right_boundary.get("confidence") or 0.0)
        except (TypeError, ValueError):
            reject(index, seg, target, "cue 级声纹边界缺少有效时间或置信度，已保留原段")
            continue
        left_duration = split_time - segment_start
        right_duration = segment_end - split_time
        if (
            abs(split_time - right_start) > tolerance
            or left_duration < 0.8
            or right_duration < 0.8
            or min(left_confidence, right_confidence) < 0.70
            or str(left_boundary.get("source") or "") not in trusted_sources
            or str(right_boundary.get("source") or "") not in trusted_sources
            or left_boundary.get("review")
            or right_boundary.get("review")
        ):
            reject(index, seg, target, "cue 边界两侧的直接声纹证据不足，已保留原段并标为待确认")
            continue

        change_points = []
        for value in seg.get("speaker_change_points") or []:
            try:
                point = float(value)
            except (TypeError, ValueError):
                continue
            if segment_start + 0.10 < point < segment_end - 0.10:
                change_points.append(point)
        cue_boundaries = [float(cue.get("end")) for cue in raw_cues[:-1]]
        if not change_points or not cue_boundaries:
            reject(index, seg, target, "缺少短窗声纹交接时间，已保留原段并标为待确认")
            continue
        acoustic_handoff = min(change_points, key=lambda point: abs(point - split_time))
        closest_boundary = min(cue_boundaries, key=lambda boundary: abs(boundary - acoustic_handoff))
        if abs(split_time - acoustic_handoff) > 0.75 or abs(split_time - closest_boundary) > tolerance:
            reject(index, seg, target, "短窗声纹交接点与最近同步 cue 边界不一致，已保留原段并标为待确认")
            continue

        left = dict(seg)
        right = dict(seg)
        for piece in (left, right):
            for key in (
                "speaker_cues",
                "speaker_cue_embeddings",
                "speaker_cue_review",
                "speaker_cue_mode",
                "speaker_subsegments",
                "speaker_change_points",
                "speaker_overlap_candidates",
                "speaker_resegmentation_review",
            ):
                piece.pop(key, None)
            # original_text predates timestamp alignment and cannot be safely
            # partitioned. The immutable, user-visible ASR text is kept in full.
            piece.pop("original_text", None)
            piece["speaker_resegmented"] = True
            piece["speaker_cue_split"] = True
            piece["speaker_split_from_index"] = index

        left["end"] = split_time
        left["text"] = "".join(str(cue.get("text") or "") for cue in raw_cues[:boundary_index])
        left["sync_cues"] = list(raw_cues[:boundary_index])
        left["speaker"] = labels[boundary_index - 1]
        left["speaker_confidence"] = round(left_confidence, 3)

        right["start"] = split_time
        right["text"] = "".join(str(cue.get("text") or "") for cue in raw_cues[boundary_index:])
        right["sync_cues"] = list(raw_cues[boundary_index:])
        right["speaker"] = labels[boundary_index]
        right["speaker_confidence"] = round(right_confidence, 3)

        out.extend([left, right])
        split_count += 1

    corrected = {**candidate, "segments": out}
    if split_count:
        _rescore_candidate(corrected)
        corrected["actual_n_speakers"] = len(corrected["summary"]["speakers"])
        corrected["speakers"] = [s["speaker"] for s in corrected["summary"]["speakers"]]
    corrected["cue_handoff_split_count"] = split_count
    corrected["cue_handoff_split_reason"] = (
        f"已在现有同步 cue 边界安全拆分 {split_count} 个段内换人片段"
        if split_count else ""
    )
    corrected["resegmentation_count"] = int(candidate.get("resegmentation_count") or 0) + split_count
    corrected["resegmentation_reason"] = (
        corrected["cue_handoff_split_reason"]
        or str(candidate.get("resegmentation_reason") or "")
    )
    if review_segments:
        corrected["review_segments"] = _merge_review_segments([
            *(candidate.get("review_segments") or []),
            *review_segments,
        ])
    corrected["segmentation_preserved"] = split_count == 0
    return corrected


def _voice_pitch(seg: dict) -> float | None:
    try:
        pitch = float(seg.get("voice_pitch_hz"))
        confidence = float(seg.get("voice_pitch_confidence") or 0.0)
    except (TypeError, ValueError):
        return None
    if pitch <= 0 or confidence < 0.25:
        return None
    return pitch


def _voice_confidence(seg: dict) -> float:
    try:
        return max(0.0, min(1.0, float(seg.get("voice_pitch_confidence") or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _voice_band_from_pitch_value(pitch: float | None, confidence: float = 1.0) -> str:
    if pitch is None or confidence < 0.25:
        return "unknown"
    if pitch < 155.0:
        return "low"
    if pitch > 185.0:
        return "high"
    return "mid"


def _speaker_voice_profiles(segments: list[dict]) -> dict[str, dict]:
    from collections import defaultdict

    values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for seg in segments:
        speaker = str(seg.get("speaker") or "")
        pitch = _voice_pitch(seg)
        if not speaker or pitch is None:
            continue
        weight = max(0.05, _voice_confidence(seg)) * max(0.1, _segment_duration(seg))
        values[speaker].append((pitch, weight))

    profiles: dict[str, dict] = {}
    for speaker, pairs in values.items():
        ordered = sorted(pairs, key=lambda item: item[0])
        total = sum(weight for _, weight in ordered)
        if total <= 0:
            continue
        cursor = 0.0
        median = ordered[-1][0]
        for pitch, weight in ordered:
            cursor += weight
            if cursor >= total / 2.0:
                median = pitch
                break
        confidence = min(1.0, total / 18.0)
        profiles[speaker] = {
            "pitch_hz": round(float(median), 1),
            "pitch_confidence": round(float(confidence), 3),
            "voice_band": _voice_band_from_pitch_value(median, confidence),
            "samples": len(pairs),
        }
    return profiles


def _speaker_voice_band_mix_summary(segments: list[dict]) -> dict[str, dict]:
    """Detect speakers whose assigned segments contain clearly mixed voice bands.

    This is a coarse gender/voice-line guardrail, not a public gender label.
    A speaker is marked mixed only when both low and high pitch bands have
    enough confident duration. Mid/unknown pitch is ignored so short pitch
    estimator noise does not split normal speech.
    """
    from collections import defaultdict

    stats: dict[str, dict] = defaultdict(lambda: {
        "duration_by_band": {"low": 0.0, "high": 0.0},
        "segments_by_band": {"low": 0, "high": 0},
        "weighted_pitch_by_band": {"low": [], "high": []},
    })
    for seg in segments:
        speaker = str(seg.get("speaker") or "")
        if not speaker:
            continue
        pitch = _voice_pitch(seg)
        confidence = _voice_confidence(seg)
        band = _voice_band_from_pitch_value(pitch, confidence)
        if pitch is None or confidence < 0.45 or band not in {"low", "high"}:
            continue
        duration = max(0.1, _segment_duration(seg))
        weight = duration * max(0.05, confidence)
        item = stats[speaker]
        item["duration_by_band"][band] += duration
        item["segments_by_band"][band] += 1
        item["weighted_pitch_by_band"][band].append((float(pitch), weight))

    summary: dict[str, dict] = {}
    for speaker, item in stats.items():
        low_duration = float(item["duration_by_band"]["low"])
        high_duration = float(item["duration_by_band"]["high"])
        total_duration = low_duration + high_duration
        low_segments = int(item["segments_by_band"]["low"])
        high_segments = int(item["segments_by_band"]["high"])
        total_segments = low_segments + high_segments
        if total_duration <= 0 or total_segments <= 0:
            continue

        dominant_band = "high" if high_duration >= low_duration else "low"
        minority_band = "low" if dominant_band == "high" else "high"
        minority_duration = min(low_duration, high_duration)
        minority_segments = low_segments if minority_band == "low" else high_segments
        minority_ratio = minority_duration / max(0.001, total_duration)
        mixed = (
            total_duration >= 18.0
            and total_segments >= 6
            and minority_duration >= 6.0
            and minority_segments >= 2
            and minority_ratio >= 0.16
        )
        severe_mixed = (
            mixed
            and minority_duration >= 10.0
            and min(low_duration, high_duration) >= 10.0
            and minority_ratio >= 0.22
        )

        def band_pitch(band: str) -> float | None:
            pairs = item["weighted_pitch_by_band"][band]
            if not pairs:
                return None
            ordered = sorted(pairs, key=lambda pair: pair[0])
            total = sum(weight for _, weight in ordered)
            cursor = 0.0
            for value, weight in ordered:
                cursor += weight
                if cursor >= total / 2.0:
                    return float(value)
            return float(ordered[-1][0])

        summary[speaker] = {
            "dominant_band": dominant_band,
            "minority_band": minority_band,
            "duration_by_band": {
                "low": round(low_duration, 3),
                "high": round(high_duration, 3),
            },
            "segments_by_band": {
                "low": low_segments,
                "high": high_segments,
            },
            "minority_ratio": round(float(minority_ratio), 3),
            "low_pitch_hz": round(band_pitch("low"), 1) if band_pitch("low") is not None else None,
            "high_pitch_hz": round(band_pitch("high"), 1) if band_pitch("high") is not None else None,
            "mixed": mixed,
            "severe_mixed": severe_mixed,
        }
    return summary


def _candidate_voice_mix_severity(candidate: dict) -> dict:
    # Always recompute from the current segments. Candidate dictionaries are
    # rescored after post-processing, and a cached empty summary from an earlier
    # pass would otherwise hide real high/low voice contamination.
    voice_mix = _speaker_voice_band_mix_summary(candidate.get("segments") or [])

    mixed = []
    severe = []
    penalty = 0.0
    for speaker, item in voice_mix.items():
        if not isinstance(item, dict) or not item.get("mixed"):
            continue
        name = str(speaker)
        mixed.append(name)
        minority_ratio = float(item.get("minority_ratio") or 0.0)
        duration_by_band = item.get("duration_by_band") or {}
        minority_duration = min(
            float(duration_by_band.get("low") or 0.0),
            float(duration_by_band.get("high") or 0.0),
        )
        item_penalty = 4.0 + min(8.0, minority_duration / 2.0) + min(4.0, minority_ratio * 12.0)
        if item.get("severe_mixed"):
            severe.append(name)
            item_penalty += 7.0
        penalty += item_penalty

    return {
        "voice_mix_summary": voice_mix,
        "mixed_voice_speakers": mixed,
        "severe_mixed_voice_speakers": severe,
        "voice_mix_penalty": round(float(penalty), 3),
    }


def _speaker_voice_line_groups(segments: list[dict]) -> dict:
    profiles = _speaker_voice_profiles(segments)
    first_seen: dict[str, float] = {}
    for seg in segments:
        speaker = str(seg.get("speaker") or "")
        if not speaker or speaker in first_seen:
            continue
        try:
            first_seen[speaker] = float(seg.get("start", 0.0))
        except (TypeError, ValueError):
            first_seen[speaker] = 0.0

    groups = {"low": [], "high": [], "mid": [], "unknown": []}
    for speaker in sorted(profiles, key=lambda name: first_seen.get(name, float("inf"))):
        profile = profiles.get(speaker) or {}
        band = str(profile.get("voice_band") or "unknown")
        if band not in groups:
            band = "unknown"
        groups[band].append(speaker)

    line_labels: dict[str, str] = {}
    label_prefix = {"low": "L", "high": "H", "mid": "M", "unknown": "U"}
    for band, speakers in groups.items():
        for idx, speaker in enumerate(speakers, start=1):
            line_labels[speaker] = f"{label_prefix[band]}{idx}"

    return {
        "profiles": profiles,
        "groups": groups,
        "line_labels": line_labels,
    }


def _voice_guard_blocks_reassignment(
    seg: dict,
    from_speaker: str,
    to_speaker: str,
    profiles: dict[str, dict],
) -> tuple[bool, str]:
    """Use coarse pitch as a guardrail against obvious cross-voice rewrites."""
    from_profile = profiles.get(from_speaker) or {}
    to_profile = profiles.get(to_speaker) or {}
    try:
        from_pitch = float(from_profile.get("pitch_hz"))
        to_pitch = float(to_profile.get("pitch_hz"))
        from_conf = float(from_profile.get("pitch_confidence") or 0.0)
        to_conf = float(to_profile.get("pitch_confidence") or 0.0)
    except (TypeError, ValueError):
        return False, ""
    if from_conf < 0.35 or to_conf < 0.35:
        return False, ""

    from_band = _voice_band_from_pitch_value(from_pitch, from_conf)
    to_band = _voice_band_from_pitch_value(to_pitch, to_conf)
    clearly_different_profiles = (
        {from_band, to_band} == {"low", "high"}
        and abs(from_pitch - to_pitch) >= 45.0
    )
    if not clearly_different_profiles:
        return False, ""

    seg_pitch = _voice_pitch(seg)
    if seg_pitch is None:
        return False, ""
    seg_conf = _voice_confidence(seg)
    from_distance = abs(seg_pitch - from_pitch)
    to_distance = abs(seg_pitch - to_pitch)
    if from_distance + 18.0 < to_distance and seg_conf >= 0.30:
        reason = (
            "声线护栏：片段音高更接近原说话人，且原/目标说话人声线差异明显，"
            "已阻止自动改派"
        )
        return True, reason
    return False, ""


def _speaker_profile_pitch(profiles: dict[str, dict], speaker: str) -> tuple[float | None, float, str]:
    profile = profiles.get(speaker) or {}
    try:
        pitch = float(profile.get("pitch_hz"))
        confidence = float(profile.get("pitch_confidence") or 0.0)
    except (TypeError, ValueError):
        return None, 0.0, "unknown"
    return pitch, confidence, _voice_band_from_pitch_value(pitch, confidence)


def _speaker_profile_distance(profiles: dict[str, dict], speaker: str, pitch: float) -> float:
    profile_pitch, profile_conf, _ = _speaker_profile_pitch(profiles, speaker)
    if profile_pitch is None or profile_conf < 0.35:
        return float("inf")
    return abs(float(pitch) - float(profile_pitch))


def _voice_band_conflicts_with_profile(seg: dict, speaker: str, profiles: dict[str, dict]) -> bool:
    seg_pitch = _voice_pitch(seg)
    if seg_pitch is None or _voice_confidence(seg) < 0.45:
        return False
    profile_pitch, profile_conf, profile_band = _speaker_profile_pitch(profiles, speaker)
    if profile_pitch is None or profile_conf < 0.45:
        return False
    seg_band = _voice_band_from_pitch_value(seg_pitch, _voice_confidence(seg))
    return {seg_band, profile_band} == {"low", "high"} and abs(seg_pitch - profile_pitch) >= 38.0


def _estimate_pyin_pitch_for_segment(audio: Path, seg: dict) -> tuple[float | None, float, str]:
    """Use non-JIT YIN as a conservative check for high/low voice-line conflicts."""
    try:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
    except (TypeError, ValueError):
        return None, 0.0, "unknown"
    duration = max(0.0, end - start)
    if duration < 0.35:
        return None, 0.0, "unknown"

    audio = audio.expanduser()
    cache_key = (
        str(audio),
        int(round(start * 1000)),
        int(round(end * 1000)),
        65.0,
        350.0,
    )
    if cache_key in _PYIN_PITCH_CACHE:
        return _PYIN_PITCH_CACHE[cache_key]

    try:
        import librosa
        import numpy as np
    except Exception:
        result = (None, 0.0, "unknown")
        _PYIN_PITCH_CACHE[cache_key] = result
        return result

    pad = 0.18
    load_start = max(0.0, start - pad)
    load_duration = duration + (start - load_start) + pad
    analysis_audio = audio
    try:
        from .diarizers.senko_diarizer import cached_analysis_wav

        analysis_audio = cached_analysis_wav(audio) or audio
    except Exception:
        pass
    try:
        samples, sr = librosa.load(
            str(analysis_audio),
            sr=16000,
            mono=True,
            offset=load_start,
            duration=min(load_duration, 18.0),
        )
    except Exception:
        result = (None, 0.0, "unknown")
        _PYIN_PITCH_CACHE[cache_key] = result
        return result
    if len(samples) < int(16000 * 0.30):
        result = (None, 0.0, "unknown")
        _PYIN_PITCH_CACHE[cache_key] = result
        return result

    try:
        f0 = librosa.yin(
            samples.astype("float32"),
            fmin=65.0,
            fmax=350.0,
            sr=int(sr),
            frame_length=2048,
            hop_length=320,
            center=False,
        )
        rms = librosa.feature.rms(
            y=samples.astype("float32"),
            frame_length=2048,
            hop_length=320,
            center=False,
        )[0]
    except Exception:
        result = (None, 0.0, "unknown")
        _PYIN_PITCH_CACHE[cache_key] = result
        return result
    if f0 is None or len(f0) == 0 or rms is None or len(rms) == 0:
        result = (None, 0.0, "unknown")
        _PYIN_PITCH_CACHE[cache_key] = result
        return result

    frame_count = min(len(f0), len(rms))
    f0 = np.asarray(f0[:frame_count], dtype=np.float32)
    rms = np.asarray(rms[:frame_count], dtype=np.float32)
    frame_times = np.arange(frame_count, dtype=np.float32) * (320.0 / float(sr)) + load_start
    in_segment = (frame_times >= start) & (frame_times <= end)
    local_energy = rms[in_segment]
    if local_energy.size == 0:
        result = (None, 0.0, "unknown")
        _PYIN_PITCH_CACHE[cache_key] = result
        return result
    energy_floor = max(1e-4, float(np.percentile(local_energy, 20)) * 0.75)
    valid = np.isfinite(f0) & (f0 >= 65.0) & (f0 <= 350.0) & (rms >= energy_floor) & in_segment
    if int(valid.sum()) < 2:
        result = (None, 0.0, "unknown")
        _PYIN_PITCH_CACHE[cache_key] = result
        return result

    pitches = np.asarray(f0[valid], dtype=np.float32)
    weights = np.maximum(np.asarray(rms[valid], dtype=np.float32), 1e-6)
    order = np.argsort(pitches)
    pitches = pitches[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    if cumulative.size == 0 or float(cumulative[-1]) <= 0:
        result = (None, 0.0, "unknown")
        _PYIN_PITCH_CACHE[cache_key] = result
        return result
    median = float(pitches[int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])
    coverage = float(valid.sum()) / max(1.0, float(in_segment.sum()))
    median_deviation = float(np.median(np.abs(pitches - median))) / max(1.0, median)
    stability = max(0.0, 1.0 - median_deviation / 0.20)
    confidence = min(1.0, coverage * (0.5 + 0.5 * stability))
    result = (median, round(confidence, 3), _voice_band_from_pitch_value(median, confidence))
    _PYIN_PITCH_CACHE[cache_key] = result
    return result


def _refine_conflicting_voice_bands_with_pyin(candidate: dict, audio: Path | None) -> dict:
    """Re-check only high/low voice-line conflicts with YIN before post-processing."""
    if audio is None:
        candidate["voice_line_refine_count"] = 0
        candidate["voice_line_refine_reason"] = ""
        return candidate
    segments = [dict(seg) for seg in candidate.get("segments") or []]
    if not segments:
        candidate["voice_line_refine_count"] = 0
        candidate["voice_line_refine_reason"] = ""
        return candidate

    profiles = _speaker_voice_profiles(segments)
    changes = 0
    reviewed = 0
    for seg in segments:
        speaker = str(seg.get("speaker") or "")
        if not speaker or not _voice_band_conflicts_with_profile(seg, speaker, profiles):
            continue
        old_pitch = _voice_pitch(seg)
        old_band = _voice_band_from_pitch_value(old_pitch, _voice_confidence(seg))
        pitch, confidence, band = _estimate_pyin_pitch_for_segment(audio, seg)
        if pitch is None or confidence < 0.04 or band == "unknown":
            reviewed += 1
            seg["voice_line_review"] = True
            continue
        if band != old_band or abs(float(pitch) - float(old_pitch or pitch)) >= 18.0:
            seg["voice_pitch_hz"] = round(float(pitch), 1)
            seg["voice_pitch_confidence"] = round(float(confidence), 3)
            seg["voice_band"] = band
            seg["voice_line_refined"] = True
            changes += 1
        if band == "mid":
            seg["voice_line_review"] = True
            reviewed += 1

    if not changes and not reviewed:
        candidate["voice_line_refine_count"] = 0
        candidate["voice_line_refine_reason"] = ""
        return candidate

    refined = {**candidate, "segments": segments}
    _rescore_candidate(refined)
    refined["actual_n_speakers"] = len(refined["summary"]["speakers"])
    refined["speakers"] = [s["speaker"] for s in refined["summary"]["speakers"]]
    refined["voice_line_refine_count"] = changes
    refined["voice_line_review_count"] = reviewed
    refined["voice_line_refine_reason"] = (
        f"已用 YIN 复核 {changes + reviewed} 个高低声线冲突段"
        + (f"，更新 {changes} 段声线" if changes else "")
        + (f"，{reviewed} 段标为声线疑点" if reviewed else "")
    )
    return refined


def _split_text_by_ratio(text: str, ratio: float, *, require_boundary: bool = False) -> tuple[str, str] | None:
    text = str(text or "")
    if not text:
        return None
    ratio = max(0.15, min(0.85, ratio))
    rough = int(round(len(text) * ratio))
    if rough <= 0 or rough >= len(text):
        return None
    break_chars = "。！？!?；;，,"
    candidates = [
        idx + 1
        for idx, ch in enumerate(text)
        if ch in break_chars
        and max(4, rough - 12) <= idx + 1 <= min(len(text) - 4, rough + 12)
    ]
    if require_boundary and not candidates:
        return None
    split_at = min(candidates, key=lambda idx: abs(idx - rough)) if candidates else rough
    left = text[:split_at].strip()
    right = text[split_at:].strip()
    if len(left) < 4 or len(right) < 4:
        return None
    return left, right


def _split_text_at_offset(text: str, offset: int) -> tuple[str, str] | None:
    text = str(text or "")
    if not text:
        return None
    offset = max(1, min(len(text) - 1, int(offset)))
    left = text[:offset].strip(" ，,。")
    right = text[offset:].strip()
    if len(left) < 4 or len(right) < 4:
        return None
    return left, right


def _handoff_phrase_offset(text: str) -> int | None:
    text = str(text or "")
    patterns = [
        r"(我还想(?:有几句话想说|了解一下|问一下|说一下))",
        r"(我想(?:了解一下|问一下|说一下|请问))",
        r"(请问)",
        r"(那我想)",
        r"(我有(?:个|一个)?(?:问题|建议|想法))",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and 6 <= match.start() <= max(6, len(text) - 6):
            return int(match.start())
    return None


def _strong_question_handoff_offset(text: str) -> int | None:
    text = str(text or "")
    patterns = [
        r"(我还想(?:有几句话想说|了解一下|问一下|说一下))",
        r"(我想(?:了解一下|问一下|说一下|请问))",
        r"(那我想)",
        r"(我有(?:个|一个)?(?:问题|建议|想法))",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and 6 <= match.start() <= max(6, len(text) - 6):
            return int(match.start())
    return None


def _speakers_have_clear_voice_contrast(
    from_speaker: str,
    to_speaker: str,
    profiles: dict[str, dict],
) -> bool:
    from_profile = profiles.get(from_speaker) or {}
    to_profile = profiles.get(to_speaker) or {}
    try:
        from_pitch = float(from_profile.get("pitch_hz"))
        to_pitch = float(to_profile.get("pitch_hz"))
        from_conf = float(from_profile.get("pitch_confidence") or 0.0)
        to_conf = float(to_profile.get("pitch_confidence") or 0.0)
    except (TypeError, ValueError):
        return False
    if from_conf < 0.35 or to_conf < 0.35:
        return False
    from_band = _voice_band_from_pitch_value(from_pitch, from_conf)
    to_band = _voice_band_from_pitch_value(to_pitch, to_conf)
    return {from_band, to_band} == {"low", "high"} and abs(from_pitch - to_pitch) >= 45.0


def _split_handoff_segments(candidate: dict) -> dict:
    """Split long ASR segments that contain an obvious speaker handoff.

    ASR sometimes keeps a previous speaker's tail and a new speaker's opening
    sentence in one text segment.  A single speaker label cannot be correct in
    that case.  This conservative splitter only fires when the following
    diarized context already shows the new speaker continuing.
    """
    segments = [dict(s) for s in candidate.get("segments") or []]
    if len(segments) < 3:
        candidate["handoff_split_count"] = 0
        candidate["handoff_split_reason"] = ""
        return candidate

    out: list[dict] = []
    split_count = 0
    review_segments: list[dict] = []
    bridge_relabels: dict[int, str] = {}
    low_confidence_bridge_relabels: set[int] = set()
    voice_profiles = _speaker_voice_profiles(segments)

    def supports_target(seg: dict, target: str) -> bool:
        """Only let text handoff cues override when diarizer votes agree."""
        share = _speaker_vote_share(seg, target)
        confidence = _speaker_confidence(seg)
        if share is not None:
            return share >= 0.35
        if confidence is not None:
            return confidence < 0.80
        return False

    def strong_evidence_blocks_target(seg: dict, from_speaker: str, target: str) -> tuple[bool, str]:
        """Do not let text cues overrule strong acoustic evidence.

        Text patterns like "我想问一下" are useful handoff hints, but they are
        weaker than short-window speaker votes and high/low voice-line evidence.
        When those acoustic signals strongly support the current speaker, the
        segment is left unchanged and surfaced as a review item.
        """
        target_share = _speaker_vote_share(seg, target)
        from_share = _speaker_vote_share(seg, from_speaker)
        if from_share is not None and from_share >= 0.68 and (target_share is None or target_share <= 0.20):
            return True, "声纹护栏：底层短窗投票强烈支持原说话人，已阻止文本线索自动改派"
        confidence = _speaker_confidence(seg)
        if confidence is not None and confidence >= 0.92 and (target_share is None or target_share <= 0.20):
            return True, "声纹护栏：当前说话人置信度高且目标说话人缺少投票支撑，已阻止自动改派"
        blocked, reason = _voice_guard_blocks_reassignment(seg, from_speaker, target, voice_profiles)
        if blocked:
            return True, reason
        return False, ""

    def can_text_override_to(idx: int, from_speaker: str, target: str) -> bool:
        target_share = _speaker_vote_share(segments[idx], target)
        current_share = _speaker_vote_share(segments[idx], from_speaker) or 0.0
        if target_share is None or target_share < max(0.45, current_share + 0.12):
            return False
        if not (
            _speakers_have_clear_voice_contrast(from_speaker, target, voice_profiles)
            and target_continuation_duration(idx, target) >= 8.0
        ):
            return False
        blocked, _reason = strong_evidence_blocks_target(segments[idx], from_speaker, target)
        return not blocked

    def can_review_text_handoff_to(idx: int, from_speaker: str, target: str) -> bool:
        """Allow strong turn-opening text to split a smeared ASR segment.

        A long ASR segment can contain the previous speaker's tail plus a new
        participant's first sentence.  The segment-level speaker vote often
        points to the previous speaker because the acoustic window spans both
        sides.  When the next turn is a stable different speaker, we split but
        mark the new side for review instead of presenting it as certain.
        """
        if not _speakers_have_clear_voice_contrast(from_speaker, target, voice_profiles):
            return False
        if target_continuation_duration(idx + 2, target) < 12.0 and target_continuation_duration(idx + 1, target) < 12.0:
            return False
        return True

    def blocked_text_override_target(idx: int, speaker: str) -> tuple[str, str]:
        if idx + 2 >= len(segments):
            return "", ""
        next_speaker = str(segments[idx + 1].get("speaker") or "")
        next2_speaker = str(segments[idx + 2].get("speaker") or "")
        next3_speaker = str(segments[idx + 3].get("speaker") or "") if idx + 3 < len(segments) else ""
        targets: list[tuple[int, str, int]] = []
        if next_speaker and next_speaker != speaker and (not next2_speaker or next2_speaker == next_speaker):
            targets.append((idx + 1, next_speaker, 0))
        if (
            next_speaker == speaker
            and next2_speaker
            and next2_speaker != speaker
            and (not next3_speaker or next3_speaker == next2_speaker)
            and _segment_duration(segments[idx + 1]) <= 5.5
        ):
            targets.append((idx + 2, next2_speaker, 1))
        for start_idx, target, _bridge_count in targets:
            if target_continuation_duration(start_idx, target) < 8.0:
                continue
            blocked, reason = strong_evidence_blocks_target(segments[idx], speaker, target)
            if blocked:
                return target, reason
        return "", ""

    def target_continuation_duration(start_idx: int, target: str, limit: int = 4) -> float:
        duration = 0.0
        for scan_idx in range(start_idx, min(len(segments), start_idx + limit)):
            if str(segments[scan_idx].get("speaker") or "") != target:
                continue
            duration += _segment_duration(segments[scan_idx])
        return duration

    def future_handoff_target(idx: int, speaker: str, allow_text_override: bool = False) -> tuple[str, list[int]]:
        if idx + 2 >= len(segments):
            return "", []
        next_speaker = str(segments[idx + 1].get("speaker") or "")
        next2_speaker = str(segments[idx + 2].get("speaker") or "")
        next3_speaker = str(segments[idx + 3].get("speaker") or "") if idx + 3 < len(segments) else ""
        text_override_target = (
            allow_text_override
            and next2_speaker
            and next2_speaker != speaker
            and (not next3_speaker or next3_speaker == next2_speaker)
            and (
                can_text_override_to(idx, speaker, next2_speaker)
                or can_review_text_handoff_to(idx, speaker, next2_speaker)
            )
        )
        if (
            next_speaker
            and next_speaker != speaker
            and (not next2_speaker or next2_speaker == next_speaker)
            and (
                supports_target(segments[idx], next_speaker)
                or (
                    allow_text_override
                    and (
                        can_text_override_to(idx, speaker, next_speaker)
                        or can_review_text_handoff_to(idx, speaker, next_speaker)
                    )
                )
            )
        ):
            return next_speaker, []
        if (
            next_speaker == speaker
            and next2_speaker
            and next2_speaker != speaker
            and (not next3_speaker or next3_speaker == next2_speaker)
            and _segment_duration(segments[idx + 1]) <= 5.5
            and (
                (
                    supports_target(segments[idx], next2_speaker)
                    and supports_target(segments[idx + 1], next2_speaker)
                )
                or text_override_target
            )
        ):
            return next2_speaker, [idx + 1]
        return "", []

    for idx, seg in enumerate(segments):
        if idx in bridge_relabels:
            target = bridge_relabels[idx]
            bridged = dict(seg)
            old = str(bridged.get("speaker") or "")
            if idx in low_confidence_bridge_relabels:
                bridged["speaker_handoff_bridge"] = True
                bridged["speaker_handoff_review"] = True
                bridged["speaker_handoff_text_review"] = True
                bridged["speaker_confidence"] = min(float(_speaker_confidence(bridged) or 0.55), 0.55)
                out.append(bridged)
                review_segments.append(_review_segment_payload(
                    idx,
                    bridged,
                    old,
                    target,
                    "强新发言开场后的桥接短段疑似换人，但缺少短窗声纹强证据，已保留原分人并标为待确认",
                ))
                continue
            blocked, reason = strong_evidence_blocks_target(bridged, old, target)
            if blocked:
                bridged["speaker_handoff_review"] = True
                out.append(bridged)
                review_segments.append(_review_segment_payload(
                    idx,
                    bridged,
                    old,
                    target,
                    reason,
                ))
                continue
            bridged["speaker"] = target
            bridged["speaker_handoff_bridge"] = True
            out.append(bridged)
            review_segments.append(_review_segment_payload(
                idx,
                bridged,
                old,
                target,
                "长段内检测到新发言开场句，后一短段已按同一新说话人衔接",
            ))
            continue

        speaker = str(seg.get("speaker") or "")
        text = str(seg.get("text") or "")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        duration = max(0.0, end - start)
        if idx == 0 or idx + 2 >= len(segments) or duration < 7.0 or len(text) < 28:
            out.append(seg)
            continue

        prev_speaker = str(segments[idx - 1].get("speaker") or "")
        strong_offset = _strong_question_handoff_offset(text)
        next_speaker, bridge_indices = future_handoff_target(
            idx,
            speaker,
            allow_text_override=strong_offset is not None,
        )
        if not speaker or speaker != prev_speaker or not next_speaker or next_speaker == speaker:
            if speaker and speaker == prev_speaker and strong_offset is not None:
                blocked_target, block_reason = blocked_text_override_target(idx, speaker)
                if blocked_target and block_reason:
                    held = dict(seg)
                    held["speaker_handoff_review"] = True
                    out.append(held)
                    review_segments.append(_review_segment_payload(
                        idx,
                        held,
                        speaker,
                        blocked_target,
                        block_reason,
                    ))
                    continue
            out.append(seg)
            continue

        offset = strong_offset if strong_offset is not None else _handoff_phrase_offset(text)
        if offset is None:
            out.append(seg)
            continue

        if isinstance(seg.get("sync_cues"), list) and seg.get("sync_cues"):
            held = dict(seg)
            held["speaker_handoff_review"] = True
            out.append(held)
            review_segments.append(_review_segment_payload(
                idx,
                held,
                speaker,
                next_speaker,
                "长段内疑似换人，但该段带精确同步时间戳；已保留原段并建议抽听确认",
            ))
            continue

        blocked, block_reason = strong_evidence_blocks_target(seg, speaker, next_speaker)
        low_confidence_text_handoff = (
            strong_offset is not None
            and can_review_text_handoff_to(idx, speaker, next_speaker)
            and not can_text_override_to(idx, speaker, next_speaker)
        )
        if blocked and not low_confidence_text_handoff:
            held = dict(seg)
            held["speaker_handoff_review"] = True
            out.append(held)
            review_segments.append(_review_segment_payload(
                idx,
                held,
                speaker,
                next_speaker,
                block_reason,
            ))
            continue

        ratio = offset / max(1, len(text))
        split_time = start + duration * ratio
        if split_time - start < 2.0 or end - split_time < 2.0:
            out.append(seg)
            continue
        split_text = (
            _split_text_at_offset(text, offset)
            if strong_offset is not None
            else _split_text_by_ratio(text, ratio)
        )
        if split_text is None:
            out.append(seg)
            continue
        left_text, right_text = split_text

        left = dict(seg)
        right = dict(seg)
        left["end"] = split_time
        left["text"] = left_text
        right["start"] = split_time
        right["text"] = right_text
        right["speaker"] = speaker if low_confidence_text_handoff else next_speaker
        right["speaker_confidence"] = min(
            0.55 if low_confidence_text_handoff else 0.8,
            max(
                float(_speaker_confidence(seg) or 0.5),
                float(_speaker_confidence(segments[idx + 1]) or 0.5),
            ),
        )
        right["speaker_handoff_split"] = True
        if low_confidence_text_handoff:
            right["speaker_handoff_review"] = True
            right["speaker_handoff_text_review"] = True
        right["speaker_split_from_index"] = idx
        out.extend([left, right])
        for bridge_idx in bridge_indices:
            bridge_relabels[bridge_idx] = next_speaker
            if low_confidence_text_handoff:
                low_confidence_bridge_relabels.add(bridge_idx)
        split_count += 1
        review_segments.append(_review_segment_payload(
            idx,
            right,
            speaker,
            next_speaker,
            (
                "长段内检测到强新发言开场句，但缺少短窗声纹强证据；已切分文本、保留原分人并标为待确认"
                if low_confidence_text_handoff
                else "长段内检测到新发言开场句，已按后续连续说话人切分"
            ),
        ))

    if split_count <= 0:
        candidate["handoff_split_count"] = 0
        candidate["handoff_split_reason"] = ""
        if review_segments:
            candidate["review_segments"] = _merge_review_segments([
                *(candidate.get("review_segments") or []),
                *review_segments,
            ])
        return candidate

    corrected = {**candidate, "segments": out}
    _rescore_candidate(corrected)
    corrected["actual_n_speakers"] = len(corrected["summary"]["speakers"])
    corrected["speakers"] = [s["speaker"] for s in corrected["summary"]["speakers"]]
    corrected["handoff_split_count"] = split_count
    corrected["handoff_split_reason"] = f"已切分 {split_count} 个疑似跨说话人的长 ASR 段"
    corrected["review_segments"] = _merge_review_segments([
        *(candidate.get("review_segments") or []),
        *review_segments,
    ])
    return corrected


def _resegment_mixed_speaker_segments(candidate: dict) -> dict:
    """Split ASR segments when short-window speaker evidence shows a handoff.

    This mirrors the common diarization pipeline shape: run speaker assignment
    on short acoustic windows first, then project that timeline back onto ASR
    text.  We only split when the segment contains exactly one clear handoff
    with enough duration on both sides; otherwise we keep the text intact and
    surface a review item.
    """
    segments = [dict(s) for s in candidate.get("segments") or []]
    if not segments:
        candidate["resegmentation_count"] = 0
        candidate["resegmentation_reason"] = ""
        return candidate

    out: list[dict] = []
    review_segments: list[dict] = []
    split_count = 0

    def timeline_runs(seg: dict) -> list[dict]:
        raw = seg.get("speaker_subsegments")
        if not isinstance(raw, list) or not raw:
            return []
        runs: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            speaker = str(item.get("speaker") or "")
            if not speaker:
                continue
            try:
                start = float(item.get("start"))
                end = float(item.get("end"))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            if runs and runs[-1]["speaker"] == speaker and abs(float(runs[-1]["end"]) - start) <= 0.08:
                runs[-1]["end"] = end
                runs[-1]["duration"] += end - start
            else:
                runs.append({
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                })
        return runs

    def compact_runs(runs: list[dict], min_piece_s: float = 0.55) -> list[dict]:
        if not runs:
            return []
        compact: list[dict] = []
        for run in runs:
            if run["duration"] < min_piece_s:
                if compact:
                    compact[-1]["end"] = run["end"]
                    compact[-1]["duration"] += run["duration"]
                continue
            compact.append(dict(run))
        return compact

    def has_overlapping_short_windows(runs: list[dict]) -> bool:
        ordered = sorted(runs, key=lambda item: (float(item["start"]), float(item["end"])))
        for prev, cur in zip(ordered, ordered[1:]):
            if float(cur["start"]) < float(prev["end"]) - 0.08:
                return True
        return False

    def dominant_handoff(runs: list[dict]) -> tuple[dict, dict] | None:
        """Find one stable speaker handoff from overlapping short-window votes.

        Senko windows overlap, so a real A→B handoff often appears as multiple
        overlapping A and B windows instead of two clean adjacent runs.  We use
        the weighted midpoint of each speaker's covered time and require two
        dominant speakers with enough duration on both sides.
        """
        by_speaker: dict[str, dict] = {}
        for run in runs:
            speaker = str(run.get("speaker") or "")
            if not speaker:
                continue
            start = float(run.get("start", 0.0))
            end = float(run.get("end", start))
            duration = max(0.0, float(run.get("duration") or (end - start)))
            if duration <= 0:
                continue
            item = by_speaker.setdefault(
                speaker,
                {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "duration": 0.0,
                    "weighted_mid": 0.0,
                },
            )
            item["start"] = min(float(item["start"]), start)
            item["end"] = max(float(item["end"]), end)
            item["duration"] = float(item["duration"]) + duration
            item["weighted_mid"] = float(item["weighted_mid"]) + ((start + end) / 2.0) * duration
        dominant = sorted(
            (item for item in by_speaker.values() if float(item["duration"]) >= 1.5),
            key=lambda item: float(item["duration"]),
            reverse=True,
        )
        if len(dominant) < 2:
            return None
        first_two = dominant[:2]
        if len(dominant) > 2 and float(dominant[2]["duration"]) >= 1.5:
            return None
        total = sum(float(item["duration"]) for item in by_speaker.values())
        if total <= 0:
            return None
        if sum(float(item["duration"]) for item in first_two) / total < 0.82:
            return None
        ordered = sorted(
            first_two,
            key=lambda item: float(item["weighted_mid"]) / max(0.001, float(item["duration"])),
        )
        first, second = ordered
        if str(first["speaker"]) == str(second["speaker"]):
            return None
        if float(first["duration"]) < 1.8 or float(second["duration"]) < 1.8:
            return None
        split_time = (
            max(float(first["start"]), float(second["start"]))
            + min(float(first["end"]), float(second["end"]))
        ) / 2.0
        if not (float(first["start"]) < split_time < float(second["end"])):
            split_time = (float(first["end"]) + float(second["start"])) / 2.0
        first = dict(first)
        second = dict(second)
        first["end"] = min(float(first["end"]), split_time)
        second["start"] = max(float(second["start"]), split_time)
        first["duration"] = max(0.0, split_time - float(first["start"]))
        second["duration"] = max(0.0, float(second["end"]) - split_time)
        if float(first["duration"]) < 1.2 or float(second["duration"]) < 1.2:
            return None
        return first, second

    for idx, seg in enumerate(segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        duration = max(0.0, end - start)
        text = str(seg.get("text") or "")
        runs = compact_runs(timeline_runs(seg))
        speakers = [run["speaker"] for run in runs]
        unique_speakers = [speaker for pos, speaker in enumerate(speakers) if speaker not in speakers[:pos]]
        handoff = dominant_handoff(runs)
        if (
            duration < 6.0
            or len(text) < 24
            or handoff is None
            or len(unique_speakers) != 2
        ):
            out.append(seg)
            if seg.get("speaker_overlap_risk") or len(unique_speakers) >= 2:
                seg["speaker_resegmentation_review"] = True
                review_segments.append(_review_segment_payload(
                    idx,
                    seg,
                    str(seg.get("speaker") or ""),
                    str(seg.get("speaker") or ""),
                    "段内短声纹窗显示多个说话人，但不满足安全切分条件，建议抽听确认",
            ))
            continue

        if isinstance(seg.get("sync_cues"), list) and seg.get("sync_cues"):
            held = dict(seg)
            held["speaker_resegmentation_review"] = True
            out.append(held)
            review_segments.append(_review_segment_payload(
                idx,
                held,
                str(seg.get("speaker") or ""),
                str(handoff[1]["speaker"]),
                "段内短声纹窗疑似换人，但该段带精确同步时间戳；已保留原段并建议抽听确认",
            ))
            continue

        first, second = handoff
        first_duration = float(first["duration"])
        second_duration = float(second["duration"])
        if first_duration < 1.5 or second_duration < 1.5:
            out.append(seg)
            review_segments.append(_review_segment_payload(
                idx,
                seg,
                str(seg.get("speaker") or ""),
                str(seg.get("speaker") or ""),
                "段内短声纹窗疑似换人，但一侧时长过短，建议抽听确认",
            ))
            continue

        split_time = max(start + 1.0, min(end - 1.0, float(second["start"])))
        if split_time <= start or split_time >= end:
            out.append(seg)
            continue
        split_text = _split_text_by_ratio(
            text,
            (split_time - start) / max(0.001, duration),
            require_boundary=True,
        )
        if split_text is None:
            out.append(seg)
            review_segments.append(_review_segment_payload(
                idx,
                seg,
                str(seg.get("speaker") or ""),
                str(second["speaker"]),
                "段内短声纹窗检测到换人，但文字无法安全切分，建议抽听确认",
            ))
            continue
        left_text, right_text = split_text
        left = dict(seg)
        right = dict(seg)
        left["end"] = split_time
        left["text"] = left_text
        left["speaker"] = str(first["speaker"])
        left["speaker_confidence"] = max(0.55, min(0.9, first_duration / max(0.001, duration)))
        left["speaker_resegmented"] = True
        left["speaker_subsegments"] = [first]
        left.pop("speaker_change_points", None)
        right["start"] = split_time
        right["text"] = right_text
        right["speaker"] = str(second["speaker"])
        right["speaker_confidence"] = max(0.55, min(0.9, second_duration / max(0.001, duration)))
        right["speaker_resegmented"] = True
        right["speaker_split_from_index"] = idx
        right["speaker_subsegments"] = [second]
        right.pop("speaker_change_points", None)
        ambiguous_split = bool(seg.get("speaker_overlap_risk"))
        overlapping_windows = has_overlapping_short_windows(runs)
        if overlapping_windows:
            ambiguous_split = True
        votes = seg.get("speaker_votes")
        if isinstance(votes, dict) and len(votes) >= 2:
            ordered_votes = sorted((float(v) for v in votes.values() if v is not None), reverse=True)
            total_votes = sum(max(0.0, v) for v in ordered_votes)
            if (
                total_votes > 0
                and len(ordered_votes) >= 2
                and (ambiguous_split or overlapping_windows)
                and (ordered_votes[0] - ordered_votes[1]) / total_votes < 0.18
            ):
                ambiguous_split = True
        if str(seg.get("voice_band") or "") == "mid":
            ambiguous_split = True
        if ambiguous_split:
            held = dict(seg)
            held["speaker_resegmentation_review"] = True
            out.append(held)
            review_segments.append(_review_segment_payload(
                idx,
                held,
                str(seg.get("speaker") or ""),
                str(second["speaker"]),
                "段内短声纹窗疑似换人，但证据重叠/接近，已保留原分人并标为待确认",
            ))
            continue
        out.extend([left, right])
        split_count += 1
        review_segments.append(_review_segment_payload(
            idx,
            right,
            str(seg.get("speaker") or ""),
            str(second["speaker"]),
            (
                "段内短声纹窗检测到明确换人，已按声纹时间线切分"
            ),
        ))

    if split_count <= 0:
        if review_segments:
            candidate = {**candidate, "segments": out}
            candidate["review_segments"] = _merge_review_segments([
                *(candidate.get("review_segments") or []),
                *review_segments,
            ])
        candidate["resegmentation_count"] = 0
        candidate["resegmentation_reason"] = ""
        return candidate

    corrected = {**candidate, "segments": out}
    _rescore_candidate(corrected)
    corrected["actual_n_speakers"] = len(corrected["summary"]["speakers"])
    corrected["speakers"] = [s["speaker"] for s in corrected["summary"]["speakers"]]
    corrected["resegmentation_count"] = split_count
    corrected["resegmentation_reason"] = f"已按短声纹时间线切分 {split_count} 个段内换人 ASR 段"
    corrected["review_segments"] = _merge_review_segments([
        *(candidate.get("review_segments") or []),
        *review_segments,
    ])
    return corrected


def _smooth_windowed_sandwiched_runs(candidate: dict) -> dict:
    """Fix local label drift introduced by long-audio windowed fallback clustering.

    This is deliberately scoped to the resemblyzer windowed fallback. It does
    not run for the normal short-audio path or for senko, and it only changes
    speaker labels when one run is sandwiched by the same speaker on both sides.
    """
    stats = candidate.get("stats") or {}
    if stats.get("assignment_mode") != "windowed_kmeans":
        candidate["smoothing_distribution"] = {}
        candidate["smoothing_reason"] = ""
        return candidate

    actual_n = int(candidate.get("actual_n_speakers") or len(candidate.get("summary", {}).get("speakers") or []))
    if actual_n != 2:
        candidate["smoothing_distribution"] = {}
        candidate["smoothing_reason"] = ""
        return candidate

    segments = [dict(s) for s in candidate.get("segments") or []]
    runs = _speaker_runs(segments)
    if len(runs) < 3:
        candidate["smoothing_distribution"] = {}
        candidate["smoothing_reason"] = ""
        return candidate

    from collections import Counter, defaultdict

    distribution: dict[str, Counter] = defaultdict(Counter)
    review_segments: list[dict] = []

    for i in range(1, len(runs) - 1):
        prev_run = runs[i - 1]
        run = runs[i]
        next_run = runs[i + 1]
        target = str(prev_run["speaker"])
        speaker = str(run["speaker"])
        if not target or target != str(next_run["speaker"]) or speaker == target:
            continue

        run_duration = float(run["end"]) - float(run["start"])
        prev_duration = float(prev_run["end"]) - float(prev_run["start"])
        next_duration = float(next_run["end"]) - float(next_run["start"])
        run_segments = len(run["indices"])
        medium_drift = (
            2 <= run_segments <= 3
            and 8.0 <= run_duration <= 25.0
            and run_duration <= 25.0
            and prev_duration >= 25.0
            and next_duration >= 20.0
        )
        if not medium_drift:
            continue

        for idx in run["indices"]:
            old = str(segments[idx].get("speaker") or "")
            if old == target:
                continue
            target_share = _speaker_vote_share(segments[idx], target)
            old_share = _speaker_vote_share(segments[idx], old) or 0.0
            if target_share is None or target_share < max(0.58, old_share + 0.22):
                review_segments.append(_review_segment_payload(
                    idx,
                    segments[idx],
                    old,
                    target,
                    "长音频窗口分人夹心疑点：底层声纹未明确支持改派，已保留原分人并建议抽听确认",
                ))
                continue
            review_segments.append(_review_segment_payload(
                idx,
                segments[idx],
                old,
                target,
                "长音频窗口分人夹心错挂纠偏",
            ))
            segments[idx]["speaker"] = target
            distribution[old][target] += 1

    if not distribution:
        candidate["smoothing_distribution"] = {}
        candidate["smoothing_reason"] = ""
        return candidate

    corrected = {**candidate, "segments": segments}
    _rescore_candidate(corrected)
    corrected["actual_n_speakers"] = len(corrected["summary"]["speakers"])
    corrected["speakers"] = [s["speaker"] for s in corrected["summary"]["speakers"]]
    payload: dict[str, dict[str, int]] = {
        speaker: {target: int(count) for target, count in counts.most_common()}
        for speaker, counts in distribution.items()
    }
    corrected["smoothing_distribution"] = payload
    corrected["review_segments"] = _merge_review_segments([
        *(candidate.get("review_segments") or []),
        *review_segments,
    ])
    parts = []
    for speaker, counts in payload.items():
        target_desc = "/".join(
            f"{target.replace('SPEAKER_', '')}{count}"
            for target, count in counts.items()
        )
        parts.append(f"{speaker.replace('SPEAKER_', '')}->{target_desc}")
    corrected["smoothing_reason"] = f"已纠偏长音频窗口分人的夹心错挂段（{' / '.join(parts)}）"
    return corrected


def _smooth_short_sandwiched_segments(candidate: dict) -> dict:
    """Smooth a low-confidence short reply surrounded by one stable speaker."""
    segments = [dict(segment) for segment in candidate.get("segments") or []]
    runs = _speaker_runs(segments)
    if len(runs) < 3:
        candidate["short_sandwich_distribution"] = {}
        candidate["short_sandwich_reason"] = ""
        return candidate

    from collections import Counter, defaultdict

    distribution: dict[str, Counter] = defaultdict(Counter)
    for pos in range(1, len(runs) - 1):
        previous = runs[pos - 1]
        current = runs[pos]
        following = runs[pos + 1]
        target = str(previous.get("speaker") or "")
        source = str(current.get("speaker") or "")
        if not target or target != str(following.get("speaker") or "") or source == target:
            continue
        indices = list(current.get("indices") or [])
        if len(indices) != 1:
            continue
        index = int(indices[0])
        segment = segments[index]
        duration = max(0.0, float(segment.get("end") or 0.0) - float(segment.get("start") or 0.0))
        previous_duration = float(previous.get("end") or 0.0) - float(previous.get("start") or 0.0)
        following_duration = float(following.get("end") or 0.0) - float(following.get("start") or 0.0)
        previous_gap = max(0.0, float(segment.get("start") or 0.0) - float(previous.get("end") or 0.0))
        following_gap = max(0.0, float(following.get("start") or 0.0) - float(segment.get("end") or 0.0))
        try:
            confidence = float(segment.get("speaker_confidence"))
        except (TypeError, ValueError):
            confidence = 1.0
        normalized_text = _normalized_annotation_text(segment.get("text"))
        filler_like = bool(re.match(r"^[嗯啊呃哦对是好可以行的了哈]+$", normalized_text))
        target_share = _speaker_vote_share(segment, target)
        source_share = _speaker_vote_share(segment, source)
        overlap_ratio = float(segment.get("overlap_ratio") or 0.0)
        previous_boundary = segments[int(previous["indices"][-1])]
        following_boundary = segments[int(following["indices"][0])]
        following_target_share = _speaker_vote_share(following_boundary, target)
        following_source_share = _speaker_vote_share(following_boundary, source) or 0.0
        boundary_confidences = (
            _speaker_confidence(previous_boundary),
            _speaker_confidence(following_boundary),
        )
        segment_pitch = _voice_pitch(segment)
        boundary_pitches = (
            _voice_pitch(previous_boundary),
            _voice_pitch(following_boundary),
        )
        pitch_continuity = (
            segment_pitch is not None
            and all(pitch is not None for pitch in boundary_pitches)
            and all(
                abs(segment_pitch - float(pitch)) / max(segment_pitch, float(pitch), 1.0) <= 0.15
                for pitch in boundary_pitches
            )
        )
        previous_pitch = boundary_pitches[0]
        exact_cue_scores = [
            float(row.get("score") or 0.0)
            for row in segment.get("speaker_cue_embeddings") or []
            if isinstance(row, dict) and row.get("embedding_scope") == "exact_sync_cue"
        ]
        previous_boundary_duration = max(
            0.0,
            float(previous_boundary.get("end") or 0.0)
            - float(previous_boundary.get("start") or 0.0),
        )
        pitch_confirmed_cluster_drift = (
            duration <= 1.60
            and confidence < 0.92
            and previous_gap <= 0.10
            and following_gap <= 0.10
            and previous_boundary_duration <= 1.20
            and segment_pitch is not None
            and previous_pitch is not None
            and 0.30 <= _voice_confidence(segment) <= 0.55
            and _voice_confidence(previous_boundary) >= 0.30
            and abs(segment_pitch - previous_pitch) / max(segment_pitch, previous_pitch, 1.0) <= 0.05
            and _voice_band_from_pitch_value(segment_pitch, _voice_confidence(segment))
            == _voice_band_from_pitch_value(previous_pitch, _voice_confidence(previous_boundary))
            != "unknown"
            and len(exact_cue_scores) == 1
            and exact_cue_scores[0] < 0.70
            and len(normalized_text) <= 12
        )
        positive_vote_speakers: set[str] = set()
        for speaker, value in (segment.get("speaker_votes") or {}).items():
            try:
                vote_duration = float(value)
            except (TypeError, ValueError):
                continue
            if vote_duration > 0.0:
                positive_vote_speakers.add(str(speaker))
        balanced_contextual_drift = (
            2.2 < duration <= 4.0
            and confidence <= 0.55
            and bool(segment.get("speaker_overlap_risk"))
            and overlap_ratio <= 0.08
            and positive_vote_speakers == {source, target}
            and target_share is not None
            and source_share is not None
            and 0.45 <= target_share <= source_share
            and source_share - target_share <= 0.08
            and all(value is not None and value >= 0.85 for value in boundary_confidences)
            and len(segment.get("speaker_change_points") or []) == 2
            and pitch_continuity
        )
        cue_embedding_rows = [
            row
            for row in segment.get("speaker_cue_embeddings") or []
            if isinstance(row, dict)
        ]
        embedding_ambiguous_contextual_drift = (
            2.2 < duration <= 4.5
            and confidence <= 0.55
            and bool(segment.get("speaker_overlap_risk"))
            and overlap_ratio <= 0.08
            and target_share is not None
            and target_share >= 0.20
            and source_share is not None
            and boundary_confidences[0] is not None
            and float(boundary_confidences[0]) >= 0.85
            and boundary_confidences[1] is not None
            and float(boundary_confidences[1]) >= 0.45
            and following_target_share is not None
            and following_target_share >= following_source_share
            and pitch_continuity
            and len(segment.get("speaker_change_points") or []) >= 3
            and bool(cue_embedding_rows)
            and all(
                str(row.get("decision") or "") == "insufficient"
                and float(row.get("score") or 0.0) <= 0.60
                and float(row.get("margin") or 0.0) <= 0.04
                and {
                    str(row.get("speaker") or ""),
                    str(row.get("second_speaker") or ""),
                } == {source, target}
                for row in cue_embedding_rows
            )
        )
        contextual_drift = balanced_contextual_drift or embedding_ambiguous_contextual_drift
        overlap_supported_micro_fragment = (
            duration <= 1.6
            and bool(segment.get("speaker_overlap_risk"))
            and overlap_ratio < 0.35
            and target_share is not None
            and source_share is not None
            and target_share >= 0.30
            and source_share - target_share <= 0.25
            and isinstance(segment.get("speaker_change_points"), list)
            and bool(segment.get("speaker_change_points"))
        )
        if (
            (duration > 2.2 and not contextual_drift)
            or previous_duration < 4.0
            or following_duration < 4.0
            or previous_gap > 0.75
            or following_gap > 0.75
            or (confidence >= 0.80 and not pitch_confirmed_cluster_drift)
            or (target_share is None and not pitch_confirmed_cluster_drift)
            or (target_share is not None and target_share < 0.20 and not pitch_confirmed_cluster_drift)
            or not (filler_like or len(normalized_text) <= 9 or contextual_drift)
            or (
                overlap_ratio >= 0.08
                and not overlap_supported_micro_fragment
                and not contextual_drift
            )
        ):
            continue
        segment["original_speaker"] = source
        segment["speaker"] = target
        segment["continuity_repaired"] = True
        segment["speaker_assignment_review"] = False
        segment["speaker_review_reason"] = (
            (
                "夹心漂移平滑：精确声纹近乎平票且前后同一声线"
                if embedding_ambiguous_contextual_drift
                else "夹心漂移平滑：前后高置信同一声线且当前短窗近乎平票"
            )
            if contextual_drift
            else (
                "同音色短句夹心平滑：相邻音高一致且精确声纹未达到强确认阈值"
                if pitch_confirmed_cluster_drift
                else "短句夹心平滑：前后稳定同一声线且短窗声纹支持"
            )
        )
        distribution[source][target] += 1

    if not distribution:
        candidate["short_sandwich_distribution"] = {}
        candidate["short_sandwich_reason"] = ""
        return candidate
    corrected = {**candidate, "segments": segments}
    _rescore_candidate(corrected)
    corrected["actual_n_speakers"] = len(corrected["summary"]["speakers"])
    corrected["speakers"] = [item["speaker"] for item in corrected["summary"]["speakers"]]
    corrected["short_sandwich_distribution"] = {
        source: {target: int(count) for target, count in counts.items()}
        for source, counts in distribution.items()
    }
    corrected["short_sandwich_reason"] = "已平滑低置信短句夹心跳变"
    return corrected


def _smooth_alternating_local_speaker_leakage(candidate: dict) -> dict:
    """Correct repeated local A-B-A-B speaker leakage without changing ASR text.

    Forced 4+ speaker clustering can split one speaker into a small local
    alternate label after a new participant appears.  The signature is not a
    globally weak speaker; it is a burst of short runs repeatedly sandwiched by
    the same dominant local speaker.  Keep real continuous runs visible, and
    only move the short alternating leakage runs.
    """
    actual_n = int(candidate.get("actual_n_speakers") or len(candidate.get("summary", {}).get("speakers") or []))
    if actual_n < 3:
        candidate["local_leakage_distribution"] = {}
        candidate["local_leakage_reason"] = ""
        return candidate

    segments = [dict(s) for s in candidate.get("segments") or []]
    voice_profiles = _speaker_voice_profiles(segments)
    runs = _speaker_runs(segments)
    if len(runs) < 5:
        candidate["local_leakage_distribution"] = {}
        candidate["local_leakage_reason"] = ""
        return candidate

    speaker_stats = {
        str(s.get("speaker") or ""): s
        for s in candidate.get("summary", {}).get("speakers", []) or []
    }
    corrections: dict[int, str] = {}
    review_segments: list[dict] = []

    from collections import Counter, defaultdict

    distribution: dict[str, Counter] = defaultdict(Counter)

    def run_duration(run: dict) -> float:
        return max(0.0, float(run.get("end", 0.0)) - float(run.get("start", 0.0)))

    def run_text(run: dict) -> str:
        return "".join(
            str(segments[idx].get("text") or "").strip()
            for idx in (run.get("indices") or [])
            if idx < len(segments)
        )

    for i in range(1, len(runs) - 1):
        run = runs[i]
        speaker = str(run.get("speaker") or "")
        if not speaker:
            continue

        duration = run_duration(run)
        run_segments = len(run.get("indices") or [])
        stats = speaker_stats.get(speaker, {})
        local_prev = runs[max(0, i - 8):i]
        local_next = runs[i + 1:min(len(runs), i + 9)]
        local_context = [*local_prev, *local_next]
        if len(local_context) < 3:
            continue

        neighbor_counts = Counter(str(r.get("speaker") or "") for r in local_context if str(r.get("speaker") or "") != speaker)
        if not neighbor_counts:
            continue
        target, target_hits = neighbor_counts.most_common(1)[0]
        if target_hits < 3:
            continue

        before_hits = sum(1 for r in local_prev if str(r.get("speaker") or "") == target)
        after_hits = sum(1 for r in local_next if str(r.get("speaker") or "") == target)
        if before_hits < 1 or after_hits < 1:
            continue

        nearest_same_gap = _nearest_same_run_gap(runs, i, speaker)
        surrounding_target_duration = sum(
            run_duration(r)
            for r in local_context
            if str(r.get("speaker") or "") == target
        )
        surrounding_speaker_duration = sum(
            run_duration(r)
            for r in local_context
            if str(r.get("speaker") or "") == speaker
        )
        local_speaker_hits = sum(1 for r in local_context if str(r.get("speaker") or "") == speaker)
        target_stats = speaker_stats.get(target, {})
        globally_small = (
            float(stats.get("duration_ratio") or 0.0) <= 0.07
            or float(stats.get("segment_ratio") or 0.0) <= 0.07
            or int(stats.get("turns") or 0) >= 7
        )
        leakage_prone = (
            float(stats.get("sandwiched_ratio") or 0.0) >= 0.06
            or (
                int(stats.get("turns") or 0) >= 6
                and (
                    float(stats.get("duration_ratio") or 0.0) <= 0.07
                    or float(stats.get("segment_ratio") or 0.0) <= 0.07
                )
            )
        )
        target_dominates = (
            float(target_stats.get("duration_ratio") or 0.0)
            >= max(0.08, float(stats.get("duration_ratio") or 0.0) * 1.25)
            or float(target_stats.get("segment_ratio") or 0.0)
            >= max(0.08, float(stats.get("segment_ratio") or 0.0) * 1.25)
            or surrounding_target_duration >= max(30.0, surrounding_speaker_duration * 1.5)
        )
        short_leak = (
            run_segments <= 3
            and duration <= 18.0
            and surrounding_target_duration >= 35.0
            and nearest_same_gap <= 28.0
            and leakage_prone
            and target_dominates
        )
        very_short_sandwich = (
            run_segments <= 2
            and duration <= 8.0
            and target_hits >= 2
            and surrounding_target_duration >= 20.0
            and leakage_prone
            and target_dominates
        )
        repeated_alternation = (
            local_speaker_hits >= 2
            and target_hits >= 4
            and duration <= 18.0
            and surrounding_target_duration >= 45.0
            and leakage_prone
            and (globally_small or target_dominates)
        )
        # Keep a real late entrant: one coherent block should not be rewritten
        # just because it is a minority speaker.
        coherent_real_run = (
            duration >= 28.0
            and run_segments >= 4
            and (nearest_same_gap > 45.0 or local_speaker_hits <= 1)
        )
        if coherent_real_run or not (short_leak or very_short_sandwich or repeated_alternation):
            continue

        text = run_text(run)
        # Avoid moving a long, content-heavy paragraph unless it is part of a
        # repeated local alternation pattern. This keeps real participant turns
        # visible while still fixing the B-D-B-D leakage shape.
        content_heavy = len(text) >= 80 and duration >= 14.0
        if content_heavy and not repeated_alternation:
            continue

        for idx in run.get("indices") or []:
            if idx >= len(segments):
                continue
            confidence = _speaker_confidence(segments[idx])
            if confidence is not None and confidence >= 0.72:
                continue
            target_share = _speaker_vote_share(segments[idx], target)
            current_share = _speaker_vote_share(segments[idx], speaker) or 0.0
            if target_share is None or target_share < max(0.58, current_share + 0.22):
                review_segments.append(_review_segment_payload(
                    idx,
                    segments[idx],
                    speaker,
                    target,
                    "局部交替串线疑点：底层声纹未明确支持改派，已保留原分人并建议抽听确认",
                ))
                continue
            blocked, block_reason = _voice_guard_blocks_reassignment(
                segments[idx],
                speaker,
                target,
                voice_profiles,
            )
            if blocked:
                review_segments.append(_review_segment_payload(
                    idx,
                    segments[idx],
                    speaker,
                    target,
                    block_reason,
                ))
                continue
            corrections[idx] = target
            old = str(segments[idx].get("speaker") or "")
            review_segments.append(_review_segment_payload(
                idx,
                segments[idx],
                old,
                target,
                "局部交替串线：该说话人短段反复夹在同一说话人上下文中，已按上下文纠偏",
            ))
            distribution[old][target] += 1

    if not corrections:
        if review_segments:
            candidate["review_segments"] = _merge_review_segments([
                *(candidate.get("review_segments") or []),
                *review_segments,
            ])
            candidate["voice_guard_reason"] = "声线护栏已阻止疑似跨声线自动纠偏"
            candidate["voice_guard_count"] = len(review_segments)
        candidate["local_leakage_distribution"] = {}
        candidate["local_leakage_reason"] = ""
        return candidate

    for idx, target in corrections.items():
        segments[idx]["speaker"] = target

    corrected = {**candidate, "segments": segments}
    _rescore_candidate(corrected)
    corrected["actual_n_speakers"] = len(corrected["summary"]["speakers"])
    corrected["speakers"] = [s["speaker"] for s in corrected["summary"]["speakers"]]
    payload: dict[str, dict[str, int]] = {
        speaker: {target: int(count) for target, count in counts.most_common()}
        for speaker, counts in distribution.items()
    }
    corrected["local_leakage_distribution"] = payload
    corrected["voice_profiles"] = voice_profiles
    if any("声线护栏" in str(item.get("reason") or "") for item in review_segments):
        corrected["voice_guard_reason"] = "声线护栏已阻止疑似跨声线自动纠偏"
        corrected["voice_guard_count"] = sum(
            1 for item in review_segments
            if "声线护栏" in str(item.get("reason") or "")
        )
    corrected["review_segments"] = _merge_review_segments([
        *(candidate.get("review_segments") or []),
        *review_segments,
    ])
    parts = []
    for speaker, counts in payload.items():
        target_desc = "/".join(
            f"{target.replace('SPEAKER_', '')}{count}"
            for target, count in counts.items()
        )
        parts.append(f"{speaker.replace('SPEAKER_', '')}->{target_desc}")
    corrected["local_leakage_reason"] = f"已纠偏局部交替串线段（{' / '.join(parts)}）"
    return corrected


def _repair_voice_band_assignments(candidate: dict) -> dict:
    """Repair obvious high/low voice-band speaker contamination.

    This pass is intentionally evidence-gated. It only rewrites a segment when
    the assigned speaker's global voice profile clearly conflicts with the
    segment pitch and diarizer votes also clearly support a different broad
    voice group. Voice pitch is a guardrail and review signal; it is not allowed
    to override the short-window speaker timeline by itself.
    """
    actual_n = int(candidate.get("actual_n_speakers") or len(candidate.get("summary", {}).get("speakers") or []))
    if actual_n < 2:
        candidate["voice_band_repair_distribution"] = {}
        candidate["voice_band_repair_reason"] = ""
        return candidate

    segments = [dict(s) for s in candidate.get("segments") or []]
    if len(segments) < 3:
        candidate["voice_band_repair_distribution"] = {}
        candidate["voice_band_repair_reason"] = ""
        return candidate

    voice_profiles = _speaker_voice_profiles(segments)
    reliable_speakers = {
        speaker
        for speaker, profile in voice_profiles.items()
        if float(profile.get("pitch_confidence") or 0.0) >= 0.45
        and str(profile.get("voice_band") or "") in {"low", "high"}
    }
    if len(reliable_speakers) < 2:
        candidate["voice_band_repair_distribution"] = {}
        candidate["voice_band_repair_reason"] = ""
        return candidate

    from collections import Counter, defaultdict

    distribution: dict[str, Counter] = defaultdict(Counter)
    corrections: dict[int, str] = {}
    review_segments: list[dict] = []

    runs = _speaker_runs(segments)
    run_by_index: dict[int, int] = {}
    for run_idx, run in enumerate(runs):
        for seg_idx in run.get("indices") or []:
            run_by_index[int(seg_idx)] = run_idx

    def run_duration(run: dict) -> float:
        return max(0.0, float(run.get("end", 0.0)) - float(run.get("start", 0.0)))

    def adjacent_context(idx: int, target: str) -> tuple[float, int, bool]:
        run_idx = run_by_index.get(idx)
        if run_idx is None:
            return 0.0, 0, False
        duration = 0.0
        hits = 0
        before = False
        after = False
        for scan in range(max(0, run_idx - 3), min(len(runs), run_idx + 4)):
            if scan == run_idx:
                continue
            run = runs[scan]
            if str(run.get("speaker") or "") != target:
                continue
            gap = _segment_gap(
                {"start": runs[run_idx]["start"], "end": runs[run_idx]["end"]},
                {"start": run["start"], "end": run["end"]},
            )
            if gap > 18.0:
                continue
            duration += run_duration(run)
            hits += 1
            if scan < run_idx:
                before = True
            if scan > run_idx:
                after = True
        return duration, hits, before and after

    def nearby_target_voice_support(idx: int, target: str, seg_band: str, window: int = 5) -> tuple[float, int]:
        duration = 0.0
        hits = 0
        for scan_idx in range(max(0, idx - window), min(len(segments), idx + window + 1)):
            if scan_idx == idx:
                continue
            other = segments[scan_idx]
            if str(other.get("speaker") or "") != target:
                continue
            other_pitch = _voice_pitch(other)
            if other_pitch is None:
                continue
            other_band = _voice_band_from_pitch_value(other_pitch, _voice_confidence(other))
            if other_band != seg_band:
                continue
            duration += _segment_duration(other)
            hits += 1
        return duration, hits

    def best_voice_target(idx: int, speaker: str, seg_pitch: float) -> tuple[str, str]:
        current_distance = _speaker_profile_distance(voice_profiles, speaker, seg_pitch)
        candidates: list[tuple[float, str, str]] = []
        current_share = _speaker_vote_share(segments[idx], speaker) or 0.0

        for target in sorted(reliable_speakers):
            if target == speaker:
                continue
            if not _speakers_have_clear_voice_contrast(speaker, target, voice_profiles):
                continue
            distance = _speaker_profile_distance(voice_profiles, target, seg_pitch)

            target_share = _speaker_vote_share(segments[idx], target)
            local_duration, local_hits, sandwiched = adjacent_context(idx, target)
            target_profile_pitch, _, target_band = _speaker_profile_pitch(voice_profiles, target)
            if target_profile_pitch is None:
                continue
            seg_band = _voice_band_from_pitch_value(seg_pitch, _voice_confidence(segments[idx]))
            nearby_voice_duration, nearby_voice_hits = nearby_target_voice_support(idx, target, seg_band)
            strong_vote_support = (
                target_share is not None
                and target_share >= max(0.58, current_share + 0.22)
            )
            strong_local_voice_sandwich = (
                sandwiched
                and local_hits >= 2
                and local_duration >= 18.0
                and target_band == seg_band
                and abs(float(seg_pitch) - float(target_profile_pitch)) <= 70.0
                and current_share < 0.88
            )
            strong_cross_voice_sandwich = (
                sandwiched
                and local_hits >= 2
                and local_duration >= 18.0
                and (strong_vote_support or strong_local_voice_sandwich)
                and (
                    (
                        target_band == seg_band
                        and abs(float(seg_pitch) - float(target_profile_pitch)) <= 70.0
                    )
                    or (
                        nearby_voice_hits >= 3
                        and nearby_voice_duration >= 16.0
                    )
                )
            )
            if distance == float("inf"):
                continue
            if not strong_cross_voice_sandwich and distance + 18.0 >= current_distance:
                continue
            target_context = strong_vote_support or strong_local_voice_sandwich
            if not target_context:
                continue

            reason = "声线复核：片段音高与当前说话人画像冲突，且目标说话人有局部/投票支撑"
            if strong_cross_voice_sandwich:
                reason = "声线复核：强高低声线夹心错挂，且短窗声纹投票支持目标说话人，已纠偏"
            elif sandwiched:
                reason = "声线复核：跨高低声线夹心错挂，且短窗声纹投票支持目标说话人，已纠偏"
            elif target_share is not None:
                reason = "声线复核：片段音高和底层声纹投票更接近目标说话人"

            score = 0.0
            score += max(0.0, current_distance - distance) / 100.0
            score += min(0.45, local_duration / 60.0)
            score += min(0.25, local_hits * 0.08)
            if sandwiched:
                score += 0.35
            if strong_cross_voice_sandwich:
                score += 0.55
            if target_share is not None:
                score += min(0.45, target_share)
            if target_band == seg_band:
                score += 0.15
            candidates.append((score, target, reason))

        if not candidates:
            return "", ""
        candidates.sort(reverse=True)
        return candidates[0][1], candidates[0][2]

    for idx, seg in enumerate(segments):
        speaker = str(seg.get("speaker") or "")
        if speaker not in reliable_speakers:
            continue
        if (
            seg.get("speaker_handoff_split")
            or seg.get("speaker_handoff_bridge")
            or seg.get("continuity_repaired")
        ):
            continue
        seg_pitch = _voice_pitch(seg)
        if seg_pitch is None or not _voice_band_conflicts_with_profile(seg, speaker, voice_profiles):
            continue

        target, reason = best_voice_target(idx, speaker, seg_pitch)
        if not target:
            review_segments.append(_review_segment_payload(
                idx,
                seg,
                speaker,
                speaker,
                "声线复核：片段音高与当前说话人画像冲突，但缺少安全改派目标，建议抽听确认",
            ))
            continue

        confidence = _speaker_confidence(seg)
        target_share = _speaker_vote_share(seg, target)
        speaker_share = _speaker_vote_share(seg, speaker)
        _, _, sandwiched = adjacent_context(idx, target)
        if target_share is None or target_share < max(0.58, (speaker_share or 0.0) + 0.22):
            review_segments.append(_review_segment_payload(
                idx,
                seg,
                speaker,
                target,
                "声线复核：音高/男女声线更像目标说话人，但短窗声纹投票不足，已保留原分人并建议抽听确认",
            ))
            continue
        if (
            confidence is not None
            and confidence >= 0.93
            and (target_share is None or target_share < 0.45)
            and not sandwiched
        ):
            review_segments.append(_review_segment_payload(
                idx,
                seg,
                speaker,
                target,
                "声线复核：音高更像目标说话人，但底层分人置信度高，建议抽听确认",
            ))
            continue

        corrections[idx] = target
        distribution[speaker][target] += 1
        review_segments.append(_review_segment_payload(idx, seg, speaker, target, reason))

    if not corrections:
        if review_segments:
            candidate["review_segments"] = _merge_review_segments([
                *(candidate.get("review_segments") or []),
                *review_segments,
            ])
            candidate["voice_band_repair_reason"] = "声线复核发现高低声线冲突段，已加入疑点抽听"
            candidate["voice_band_repair_distribution"] = {}
        else:
            candidate["voice_band_repair_distribution"] = {}
            candidate["voice_band_repair_reason"] = ""
        candidate["voice_profiles"] = voice_profiles
        return candidate

    for idx, target in corrections.items():
        segments[idx]["speaker"] = target
        segments[idx]["voice_band_repaired"] = True

    corrected = {**candidate, "segments": segments}
    _rescore_candidate(corrected)
    corrected["actual_n_speakers"] = len(corrected["summary"]["speakers"])
    corrected["speakers"] = [s["speaker"] for s in corrected["summary"]["speakers"]]
    payload: dict[str, dict[str, int]] = {
        speaker: {target: int(count) for target, count in counts.most_common()}
        for speaker, counts in distribution.items()
    }
    corrected["voice_band_repair_distribution"] = payload
    corrected["voice_profiles"] = voice_profiles
    corrected["review_segments"] = _merge_review_segments([
        *(candidate.get("review_segments") or []),
        *review_segments,
    ])
    parts = []
    for speaker, counts in payload.items():
        target_desc = "/".join(
            f"{target.replace('SPEAKER_', '')}{count}"
            for target, count in counts.items()
        )
        parts.append(f"{speaker.replace('SPEAKER_', '')}->{target_desc}")
    corrected["voice_band_repair_reason"] = f"已按高低声线一致性纠偏疑似男女/声线错挂段（{' / '.join(parts)}）"
    return corrected


def _repair_discourse_continuity_assignments(candidate: dict) -> dict:
    """Repair short speaker jumps that split one continuous sentence or argument.

    Diarization is acoustic-first, while ASR segments often cut a single
    sentence into several chunks. Discourse continuity is useful for surfacing
    suspicious cuts, but it is not a reliable speaker identity signal. This pass
    therefore only rewrites when short-window votes clearly support the target;
    otherwise it keeps the acoustic label and marks the segment for review.
    """
    actual_n = int(candidate.get("actual_n_speakers") or len(candidate.get("summary", {}).get("speakers") or []))
    if actual_n < 3:
        candidate["continuity_repair_distribution"] = {}
        candidate["continuity_repair_reason"] = ""
        return candidate

    segments = [dict(s) for s in candidate.get("segments") or []]
    runs = _speaker_runs(segments)
    if len(runs) < 3:
        candidate["continuity_repair_distribution"] = {}
        candidate["continuity_repair_reason"] = ""
        return candidate

    speaker_stats = {
        str(s.get("speaker") or ""): s
        for s in candidate.get("summary", {}).get("speakers", []) or []
    }
    voice_profiles = _speaker_voice_profiles(segments)

    from collections import Counter, defaultdict

    distribution: dict[str, Counter] = defaultdict(Counter)
    corrections: dict[int, str] = {}
    review_segments: list[dict] = []

    def run_duration(run: dict) -> float:
        return max(0.0, float(run.get("end", 0.0)) - float(run.get("start", 0.0)))

    def run_text(run: dict) -> str:
        return "".join(
            str(segments[idx].get("text") or "").strip()
            for idx in (run.get("indices") or [])
            if idx < len(segments)
        )

    def starts_like_continuation(text: str) -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        return bool(re.match(
            r"^(但是|但|而且|而|因为|所以|那么|如果|或者|只是|就是|这就是|接下来|然后|对[，,是吧]*|"
            r"不是代表|否认|承认|认同|接受|说明|表示|没有|我们|他|她|他们|它|这个|那个|其实)",
            text,
        ))

    def standalone_question_or_new_turn(text: str) -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        if _strong_question_handoff_offset(text) == 0:
            return True
        return bool(re.match(r"^(请问|我想问|我有个问题|那我想|我还想|您刚才|你刚才)", text))

    def ends_like_open_clause(text: str) -> bool:
        text = re.sub(r"[\s,，。.!！?？；;]+$", "", str(text or "").strip())
        if not text:
            return False
        return bool(re.search(
            r"(我们不|我们没有|不能|不是|但是|如果|因为|所以|或者|就是|接下来|这会|这就是|"
            r"代表|可以|需要|会|不|没|没有)$",
            text,
        ))

    def starts_like_open_clause_completion(text: str) -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        return bool(re.match(
            r"^(否认|承认|认同|接受|代表|说明|表示|意味着|等于|是说|看到|觉得|认为|"
            r"希望|需要|应该|可以|会|把|去|做)",
            text,
        ))

    for run_idx in range(1, len(runs) - 1):
        run = runs[run_idx]
        speaker = str(run.get("speaker") or "")
        prev_run = runs[run_idx - 1]
        next_run = runs[run_idx + 1]
        target = str(prev_run.get("speaker") or "")
        if not speaker or not target or speaker == target or target != str(next_run.get("speaker") or ""):
            continue

        duration = run_duration(run)
        run_segments = len(run.get("indices") or [])
        prev_duration = run_duration(prev_run)
        next_duration = run_duration(next_run)
        text = run_text(run)
        if standalone_question_or_new_turn(text):
            continue
        if run_segments > 4 or duration > 16.0:
            continue
        if prev_duration < 4.0 or next_duration < 4.0:
            continue

        stats = speaker_stats.get(speaker, {})
        target_stats = speaker_stats.get(target, {})
        local_speaker_gap = _nearest_same_run_gap(runs, run_idx, speaker)
        weak_or_local = (
            float(stats.get("duration_ratio") or 0.0) <= 0.08
            or int(stats.get("turns") or 0) >= 6
            or local_speaker_gap > 18.0
        )
        target_has_weight = (
            float(target_stats.get("duration_s") or 0.0) >= 45.0
            or prev_duration + next_duration >= 18.0
        )
        continuation = starts_like_continuation(text)
        previous_open_clause = ends_like_open_clause(run_text(prev_run))
        open_clause_bridge = previous_open_clause and starts_like_open_clause_completion(text)
        short_clause = duration <= 7.0 and len(text) <= 45
        medium_continuation_block = (
            run_segments <= 4
            and duration <= 16.0
            and continuation
            and local_speaker_gap <= 45.0
            and prev_duration + next_duration >= 24.0
        )
        strong_sentence_bridge = (
            previous_open_clause
            and (continuation or short_clause)
            and prev_duration >= 4.0
            and next_duration >= 4.0
        )
        strong_open_clause_bridge = (
            open_clause_bridge
            and run_segments <= 4
            and duration <= 16.0
            and prev_duration >= 4.0
            and next_duration >= 4.0
            and prev_duration + next_duration >= 18.0
        )
        split_sentence_shape = (
            prev_duration + next_duration >= 18.0
            and (continuation or short_clause or previous_open_clause)
        )
        if not (
            target_has_weight
            and (split_sentence_shape or medium_continuation_block)
            and (weak_or_local or strong_sentence_bridge or strong_open_clause_bridge or medium_continuation_block)
        ):
            continue

        for idx in run.get("indices") or []:
            if idx >= len(segments):
                continue
            seg = segments[idx]
            confidence = _speaker_confidence(seg)
            speaker_share = _speaker_vote_share(seg, speaker)
            target_share = _speaker_vote_share(seg, target)
            vote_supports_target = (
                target_share is not None
                and target_share >= max(0.58, (speaker_share or 0.0) + 0.22)
            )
            if not vote_supports_target and not strong_open_clause_bridge:
                review_segments.append(_review_segment_payload(
                    idx,
                    seg,
                    speaker,
                    target,
                    (
                        "语义连续性复核：疑似夹心断句，但底层声纹未明确支持改派，"
                        "已保留原分人并建议抽听确认"
                    ),
                ))
                continue

            same_voice_line = True
            seg_pitch = _voice_pitch(seg)
            if seg_pitch is not None:
                target_distance = _speaker_profile_distance(voice_profiles, target, seg_pitch)
                speaker_distance = _speaker_profile_distance(voice_profiles, speaker, seg_pitch)
                if target_distance == float("inf"):
                    same_voice_line = False
                if _speakers_have_clear_voice_contrast(speaker, target, voice_profiles):
                    same_voice_line = False
            if not same_voice_line:
                review_segments.append(_review_segment_payload(
                    idx,
                    seg,
                    speaker,
                    target,
                    "语义连续性复核：文本像同一论点，但声线/男女特征不支持安全合并，已保留原分人并建议抽听确认",
                ))
                continue

            blocked, block_reason = _voice_guard_blocks_reassignment(
                seg,
                speaker,
                target,
                voice_profiles,
            )
            # Speaker diarization is acoustic-first.  Discourse continuity can
            # fix same-voice sentence cuts, but it must not collapse clear
            # high/low voice-line turns into one speaker just because the text
            # reads as one argument.
            if blocked:
                review_segments.append(_review_segment_payload(
                    idx,
                    seg,
                    speaker,
                    target,
                    block_reason,
                ))
                continue

            corrections[idx] = target
            distribution[speaker][target] += 1
            review_segments.append(_review_segment_payload(
                idx,
                seg,
                speaker,
                target,
                (
                    "语义连续性复核：上一段开放句法未完成，当前段承接成同一句，已按上下文纠偏"
                    if strong_open_clause_bridge
                    else "语义连续性复核：短段夹在同一说话人上下文中，且文本承接同一论点，已纠偏"
                ),
            ))

    if not corrections:
        if review_segments:
            candidate["review_segments"] = _merge_review_segments([
                *(candidate.get("review_segments") or []),
                *review_segments,
            ])
            candidate["continuity_repair_distribution"] = {}
            candidate["continuity_repair_reason"] = "语义连续性复核发现疑似夹心断句，已加入疑点抽听"
        else:
            candidate["continuity_repair_distribution"] = {}
            candidate["continuity_repair_reason"] = ""
        return candidate

    for idx, target in corrections.items():
        segments[idx]["speaker"] = target
        segments[idx]["continuity_repaired"] = True

    corrected = {**candidate, "segments": segments}
    _rescore_candidate(corrected)
    corrected["actual_n_speakers"] = len(corrected["summary"]["speakers"])
    corrected["speakers"] = [s["speaker"] for s in corrected["summary"]["speakers"]]
    payload: dict[str, dict[str, int]] = {
        speaker: {target: int(count) for target, count in counts.most_common()}
        for speaker, counts in distribution.items()
    }
    corrected["continuity_repair_distribution"] = payload
    corrected["review_segments"] = _merge_review_segments([
        *(candidate.get("review_segments") or []),
        *review_segments,
    ])
    parts = []
    for speaker, counts in payload.items():
        target_desc = "/".join(
            f"{target.replace('SPEAKER_', '')}{count}"
            for target, count in counts.items()
        )
        parts.append(f"{speaker.replace('SPEAKER_', '')}->{target_desc}")
    corrected["continuity_repair_reason"] = f"已按语义连续性纠偏疑似夹心断句错挂段（{' / '.join(parts)}）"
    return corrected


def _nearest_same_run_gap(runs: list[dict], idx: int, speaker: str) -> float:
    run = runs[idx]
    best = float("inf")
    current = {"start": run["start"], "end": run["end"]}
    for j, other in enumerate(runs):
        if j == idx or str(other.get("speaker") or "") != speaker:
            continue
        gap = _segment_gap(current, {"start": other["start"], "end": other["end"]})
        best = min(best, gap)
    return best


def _build_local_assignment_review_segments(candidate: dict, limit: int = 10) -> list[dict]:
    """Flag local speaker-label jumps for human review without changing labels."""
    segments = candidate.get("segments") or []
    runs = _speaker_runs(segments)
    if len(runs) < 3:
        return []

    speaker_stats = {
        str(s.get("speaker") or ""): s
        for s in candidate.get("summary", {}).get("speakers", []) or []
    }
    fragile = set(candidate.get("fragile_speakers") or [])
    extras: list[dict] = []

    for i in range(1, len(runs) - 1):
        prev_run = runs[i - 1]
        run = runs[i]
        next_run = runs[i + 1]
        speaker = str(run.get("speaker") or "")
        prev_speaker = str(prev_run.get("speaker") or "")
        next_speaker = str(next_run.get("speaker") or "")
        if not speaker or speaker == prev_speaker or speaker == next_speaker:
            continue

        run_duration = float(run["end"]) - float(run["start"])
        prev_duration = float(prev_run["end"]) - float(prev_run["start"])
        next_duration = float(next_run["end"]) - float(next_run["start"])
        run_segments = len(run.get("indices") or [])
        nearest_same_gap = _nearest_same_run_gap(runs, i, speaker)
        stats = speaker_stats.get(speaker, {})
        globally_low_support = (
            speaker in fragile
            or int(stats.get("stable_turns") or 0) <= 2
            or float(stats.get("duration_ratio") or 0.0) <= 0.06
            or float(stats.get("segment_ratio") or 0.0) <= 0.06
        )
        run_text = "".join(
            str(segments[idx].get("text") or "").strip()
            for idx in (run.get("indices") or [])
            if idx < len(segments)
        )
        filler_like = bool(re.match(r"^[嗯啊呃哦对是好可以行的了哈\s,，。.!！?？]+$", run_text))
        enough_context = prev_duration >= 8.0 and next_duration >= 8.0
        short_run = run_segments <= 2 or run_duration <= 8.0
        medium_short_run = run_segments <= 4 and run_duration <= 18.0
        extreme_sandwich = (
            run_segments == 1
            and run_duration <= 3.5
            and (filler_like or len(run_text) <= 6)
        )
        isolated_low_support = globally_low_support and (
            short_run or (medium_short_run and nearest_same_gap > 18.0)
        )

        target = speaker
        reason = ""
        if prev_speaker == next_speaker and enough_context and (
            isolated_low_support
            or extreme_sandwich
            or (
                nearest_same_gap > 45.0
                and run_segments <= 2
                and run_duration <= 8.0
                and (prev_duration + next_duration) >= 35.0
            )
        ):
            target = prev_speaker
            reason = (
                f"局部夹心跳变：前后均为 {_speaker_name_to_label(prev_speaker)}，"
                f"当前 {_speaker_name_to_label(speaker)}，建议抽听确认"
            )
        elif enough_context and globally_low_support and short_run:
            reason = "短插话/边界跳变：该说话人局部支撑不足，建议抽听确认"

        if not reason:
            continue

        for seg_idx in (run.get("indices") or [])[:4]:
            if seg_idx >= len(segments):
                continue
            extras.append(_review_segment_payload(
                seg_idx,
                segments[seg_idx],
                speaker,
                target,
                reason,
            ))
            if len(extras) >= limit:
                return _merge_review_segments(extras, limit)

    return _merge_review_segments(extras, limit)


def _build_rapid_alternation_review_segments(candidate: dict, limit: int = 10) -> list[dict]:
    """Flag short windows where two speakers rapidly alternate with weak evidence.

    This is the common failure mode in heated meetings: the diarizer forms two
    nearby clusters, then swaps identity across short turns.  We do not rewrite
    labels here; we surface the whole pocket as a high-priority review region.
    """
    segments = candidate.get("segments") or []
    if len(segments) < 4:
        return []

    voice_mix = candidate.get("voice_mix_summary")
    if not isinstance(voice_mix, dict):
        voice_mix = _speaker_voice_band_mix_summary(segments)
    fragile = set(candidate.get("fragile_speakers") or [])

    extras: list[dict] = []
    window_s = 52.0
    min_switches = 4
    min_segments = 6
    i = 0
    while i < len(segments):
        try:
            start = float(segments[i].get("start", 0.0))
        except (TypeError, ValueError):
            i += 1
            continue
        j = i
        speakers: list[str] = []
        while j < len(segments):
            try:
                if float(segments[j].get("end", 0.0)) - start > window_s:
                    break
            except (TypeError, ValueError):
                break
            speaker = str(segments[j].get("speaker") or "")
            if speaker:
                speakers.append(speaker)
            j += 1

        unique = sorted(set(speakers))
        if len(speakers) >= min_segments and len(unique) == 2:
            switches = sum(
                1 for a, b in zip(speakers, speakers[1:])
                if a and b and a != b
            )
            weak_or_mixed = any(
                speaker in fragile or bool((voice_mix.get(speaker) or {}).get("mixed"))
                for speaker in unique
            )
            if switches >= min_switches and weak_or_mixed:
                for seg_idx in range(i, j):
                    speaker = str(segments[seg_idx].get("speaker") or "")
                    if not speaker:
                        continue
                    other = next((name for name in unique if name != speaker), speaker)
                    extras.append(_review_segment_payload(
                        seg_idx,
                        segments[seg_idx],
                        speaker,
                        other,
                        (
                            "局部快速轮换风险：短时间内两个说话人多次交替，且存在弱簇/声线混合，"
                            "容易出现 B/D 身份错位，建议连续抽听这一段"
                        ),
                    ))
                    if len(extras) >= limit:
                        return _merge_review_segments(extras, limit)
                i = max(i + 1, j - 1)
                continue
        i += 1

    return _merge_review_segments(extras, limit)


def _build_subsegment_change_review_segments(candidate: dict, limit: int = 10) -> list[dict]:
    """Flag segment-internal short-window speaker changes.

    The diarizer can expose short-window change points even when the final ASR
    segment keeps a single speaker label.  These are review signals first; they
    should be visible in the transcript instead of being hidden in metadata.
    """
    segments = candidate.get("segments") or []
    extras: list[dict] = []
    for idx, seg in enumerate(segments):
        try:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))
        except (TypeError, ValueError):
            continue
        seg_duration = max(0.0, seg_end - seg_start)
        text = str(seg.get("text") or "")
        if seg_duration < 1.2 and len(text) < 6:
            continue

        raw_points = seg.get("speaker_change_points")
        change_points: list[float] = []
        if isinstance(raw_points, list):
            for point in raw_points:
                try:
                    value = float(point)
                except (TypeError, ValueError):
                    continue
                if seg_start + 0.20 <= value <= seg_end - 0.20:
                    change_points.append(value)

        raw_subsegments = seg.get("speaker_subsegments")
        events: list[dict] = []
        if isinstance(raw_subsegments, list):
            for item in raw_subsegments:
                if not isinstance(item, dict):
                    continue
                speaker = str(item.get("speaker") or "")
                if not speaker:
                    continue
                try:
                    start = float(item.get("start"))
                    end = float(item.get("end"))
                except (TypeError, ValueError):
                    continue
                duration = max(0.0, min(end, seg_end) - max(start, seg_start))
                if duration <= 0:
                    continue
                events.append({
                    "speaker": speaker,
                    "start": max(start, seg_start),
                    "end": min(end, seg_end),
                    "duration": duration,
                })

        from collections import defaultdict

        duration_by_speaker: dict[str, float] = defaultdict(float)
        for event in events:
            duration_by_speaker[str(event["speaker"])] += float(event["duration"])
        total_duration = sum(duration_by_speaker.values())
        meaningful_speakers = [
            speaker
            for speaker, duration in duration_by_speaker.items()
            if duration >= 0.65 and (total_duration <= 0 or duration / total_duration >= 0.12)
        ]

        transitions = 0
        last_speaker = ""
        for event in sorted(events, key=lambda item: (float(item["start"]), float(item["end"]))):
            if float(event["duration"]) < 0.25:
                continue
            speaker = str(event["speaker"])
            if last_speaker and speaker != last_speaker:
                transitions += 1
            last_speaker = speaker

        if not change_points and len(meaningful_speakers) < 2 and transitions < 1:
            continue

        current = str(seg.get("speaker") or "")
        other = next((speaker for speaker in meaningful_speakers if speaker != current), current)
        if transitions >= 2:
            reason = "段内短声纹窗多次换人：一句里可能混入多位说话人，建议连续抽听确认"
        elif change_points:
            reason = "段内短声纹窗存在换人切点：当前单一说话人标签需抽听确认"
        else:
            reason = "段内短声纹窗显示多个说话人：当前单一说话人标签需抽听确认"
        extras.append(_review_segment_payload(idx, seg, current, other, reason))
        if len(extras) >= limit:
            break

    return _merge_review_segments(extras, limit)


def _build_review_segments(candidate: dict, limit: int = 64) -> list[dict]:
    existing = list(candidate.get("review_segments") or [])
    seen = {int(s.get("index", -1)) for s in existing}
    suspicious = set(candidate.get("mergeable_speakers") or []) | set(candidate.get("fragile_speakers") or [])
    local_review = _build_local_assignment_review_segments(candidate)
    voice_mix_review = _build_voice_mix_review_segments(candidate)
    rapid_review = _build_rapid_alternation_review_segments(candidate)
    subsegment_review = _build_subsegment_change_review_segments(candidate)
    if not suspicious:
        return _merge_review_segments([*rapid_review, *subsegment_review, *existing, *voice_mix_review, *local_review], limit)

    extra: list[dict] = []
    for idx, seg in enumerate(candidate.get("segments") or []):
        if idx in seen:
            continue
        speaker = str(seg.get("speaker") or "")
        if speaker not in suspicious:
            continue
        extra.append(_review_segment_payload(
            idx,
            seg,
            speaker,
            speaker,
            "弱/碎片说话人，建议抽听确认",
        ))
    return _merge_review_segments([*rapid_review, *subsegment_review, *existing, *voice_mix_review, *extra, *local_review], limit)


def _build_voice_mix_review_segments(candidate: dict, limit: int = 8) -> list[dict]:
    segments = candidate.get("segments") or []
    voice_mix = candidate.get("voice_mix_summary")
    if not isinstance(voice_mix, dict):
        voice_mix = _speaker_voice_band_mix_summary(segments)
    mixed_speakers = [
        str(speaker)
        for speaker, item in voice_mix.items()
        if bool((item or {}).get("mixed"))
    ]
    if not segments or not mixed_speakers:
        return []

    extras: list[dict] = []
    for speaker in mixed_speakers:
        item = voice_mix.get(speaker) or {}
        minority_band = str(item.get("minority_band") or "")
        dominant_band = str(item.get("dominant_band") or "")
        picked_minority = 0
        picked_dominant = 0
        for idx, seg in enumerate(segments):
            if str(seg.get("speaker") or "") != speaker:
                continue
            pitch = _voice_pitch(seg)
            band = _voice_band_from_pitch_value(pitch, _voice_confidence(seg))
            if band == minority_band and picked_minority < 4:
                extras.append(_review_segment_payload(
                    idx,
                    seg,
                    speaker,
                    speaker,
                    "声线混合风险：同一说话人内存在明显高低声线，建议抽听确认是否男女/声线混标",
                ))
                picked_minority += 1
            elif band == dominant_band and picked_dominant < 2:
                extras.append(_review_segment_payload(
                    idx,
                    seg,
                    speaker,
                    speaker,
                    "声线混合风险：该说话人同时包含另一类高低声线片段，建议与相邻疑点对照",
                ))
                picked_dominant += 1
            if picked_minority >= 4 and picked_dominant >= 2:
                break
        if len(extras) >= limit:
            return _merge_review_segments(extras, limit)
    return _merge_review_segments(extras, limit)


def _build_count_ambiguity_review_segments(best: dict, candidates: list[dict], limit: int = 8) -> list[dict]:
    """Return concrete snippets when adjacent speaker-count candidates are close.

    A low-confidence recommendation is not actionable unless the user knows
    what to listen to. When a higher-count candidate is close to the selected
    candidate, sample the small extra clusters from that higher-count candidate
    and show how they map back to the selected result.
    """
    best_segments = best.get("segments") or []
    best_actual = _candidate_actual_count(best)
    best_score = float(best.get("score") or 0.0)
    extras: list[dict] = []

    higher_close = sorted(
        [
            c for c in candidates
            if c is not best
            and _candidate_actual_count(c) > best_actual
            and (best_score - float(c.get("score") or 0.0)) <= 2.5
            and len(c.get("segments") or []) == len(best_segments)
        ],
        key=lambda c: (
            best_score - float(c.get("score") or 0.0),
            _candidate_actual_count(c),
        ),
    )
    if not higher_close:
        return []

    for candidate in higher_close[:2]:
        candidate_actual = _candidate_actual_count(candidate)
        speaker_stats = candidate.get("summary", {}).get("speakers", []) or []
        if not speaker_stats:
            continue
        small_speakers = [
            s for s in speaker_stats
            if float(s.get("segment_ratio") or 0.0) <= 0.08
            or float(s.get("duration_ratio") or 0.0) <= 0.06
            or int(s.get("segments") or 0) <= 12
        ]
        if not small_speakers:
            small_speakers = sorted(
                speaker_stats,
                key=lambda s: (float(s.get("duration_s") or 0.0), int(s.get("segments") or 0)),
            )[:1]

        candidate_segments = candidate.get("segments") or []
        sampled_speakers = sorted(
            small_speakers,
            key=lambda s: (float(s.get("duration_s") or 0.0), int(s.get("segments") or 0)),
        )[:3]
        per_speaker_limit = max(1, min(3, limit // max(1, len(sampled_speakers))))
        for speaker_info in sampled_speakers:
            speaker = str(speaker_info.get("speaker") or "")
            if not speaker:
                continue
            picked = 0
            for idx, seg in enumerate(candidate_segments):
                if str(seg.get("speaker") or "") != speaker:
                    continue
                target = ""
                if idx < len(best_segments):
                    target = str(best_segments[idx].get("speaker") or "")
                extras.append(_review_segment_payload(
                    idx,
                    seg,
                    speaker,
                    target or speaker,
                    f"候选人数接近：推荐 {best_actual} 人，{candidate_actual} 人候选中该簇需抽听确认",
                ))
                picked += 1
                if picked >= per_speaker_limit:
                    break
            if len(extras) >= limit:
                return _merge_review_segments(extras, limit)

    return _merge_review_segments(extras, limit)


def _build_low_confidence_speaker_review_segments(candidate: dict, limit: int = 8) -> list[dict]:
    """Sample concrete snippets from the least-supported speakers.

    Use this when the recommendation says "please listen" but no specific
    count-boundary cluster was found.  The smallest visible speakers are the
    fastest way for a user to confirm whether the count is real.
    """
    segments = candidate.get("segments") or []
    speaker_stats = candidate.get("summary", {}).get("speakers", []) or []
    if not segments or not speaker_stats:
        return []

    targets = sorted(
        speaker_stats,
        key=lambda s: (
            float(s.get("duration_ratio") or 0.0),
            float(s.get("segment_ratio") or 0.0),
            int(s.get("stable_turns") or 0),
        ),
    )[:3]
    extras: list[dict] = []
    for speaker_info in targets:
        speaker = str(speaker_info.get("speaker") or "")
        if not speaker:
            continue
        picked = 0
        for idx, seg in enumerate(segments):
            if str(seg.get("speaker") or "") != speaker:
                continue
            extras.append(_review_segment_payload(
                idx,
                seg,
                speaker,
                speaker,
                "推荐置信度不足：低占比说话人需抽听确认",
            ))
            picked += 1
            if picked >= 4:
                break
        if len(extras) >= limit:
            break
    return _merge_review_segments(extras, limit)


def _should_auto_merge_in_diarize(candidate: dict) -> bool:
    """Guard the one-click diarize path against swallowing real quiet speakers."""
    mergeable = set(candidate.get("mergeable_speakers") or [])
    if not mergeable:
        return False
    actual_n = int(candidate.get("actual_n_speakers") or len(candidate.get("summary", {}).get("speakers") or []))
    if actual_n >= 4:
        return True
    if actual_n < 3:
        return False

    by_name = {
        str(s["speaker"]): s
        for s in candidate.get("summary", {}).get("speakers", [])
    }
    for speaker in mergeable:
        s = by_name.get(speaker)
        if not s:
            continue
        if (
            s.get("duration_s", 0.0) < 18.0
            or s.get("segment_ratio", 1.0) < 0.04
            or (
                s.get("turns", 0) >= 5
                and s.get("short_ratio", 0.0) >= 0.55
                and s.get("filler_ratio", 0.0) >= 0.45
            )
        ):
            return True
    return False


def _candidate_has_meaningful_refinement(lower: dict, higher: dict) -> bool:
    if _candidate_resolves_voice_mix(lower, higher):
        higher["refinement_reason"] = (
            "保留更细人数：低人数候选中存在高低声线混标，"
            "高人数候选拆出了更一致的声线"
        )
        return True

    lower_actual = int(lower.get("actual_n_speakers") or len(lower.get("summary", {}).get("speakers") or []))
    higher_actual = int(higher.get("actual_n_speakers") or len(higher.get("summary", {}).get("speakers") or []))
    lower_severe_mix = set(lower.get("severe_mixed_voice_speakers") or [])
    higher_severe_mix = set(higher.get("severe_mixed_voice_speakers") or [])
    if (
        higher_severe_mix - lower_severe_mix
        or len(higher_severe_mix) > len(lower_severe_mix)
        or (higher_severe_mix and lower_actual < 4)
    ):
        return False

    if lower_actual < 3 or higher_actual != lower_actual + 1:
        return False
    if int(higher.get("tiny_speakers", 0)) > int(lower.get("tiny_speakers", 0)):
        return False
    added_weak_speakers = int(higher.get("weak_speakers", 0)) - int(lower.get("weak_speakers", 0))
    if added_weak_speakers > 0 and (lower_actual < 4 or added_weak_speakers > 1):
        return False
    if int(higher.get("fragmented_speakers", 0)) > 1:
        return False
    if int(higher.get("stable_speakers", 0)) < int(lower.get("stable_speakers", 0)):
        return False

    higher_segments = higher.get("segments") or []
    lower_segments = lower.get("segments") or []
    if len(higher_segments) != len(lower_segments):
        return False

    fragile_higher = set(higher.get("fragile_speakers") or [])
    higher_summary = {
        str(s["speaker"]): s
        for s in higher.get("summary", {}).get("speakers", [])
    }
    lower_summary = {
        str(s["speaker"]): s
        for s in lower.get("summary", {}).get("speakers", [])
    }

    from collections import Counter, defaultdict

    votes: dict[str, Counter] = defaultdict(Counter)
    durations: dict[str, Counter] = defaultdict(Counter)
    for lower_seg, higher_seg in zip(lower_segments, higher_segments):
        lower_speaker = str(lower_seg.get("speaker") or "")
        higher_speaker = str(higher_seg.get("speaker") or "")
        if not lower_speaker or not higher_speaker:
            continue
        votes[lower_speaker][higher_speaker] += 1
        durations[lower_speaker][higher_speaker] += _segment_duration(higher_seg)

    for lower_speaker, child_counts in votes.items():
        lower_stats = lower_summary.get(lower_speaker)
        if not lower_stats or lower_stats.get("duration_s", 0.0) < 45.0:
            continue
        children = [
            speaker
            for speaker, count in child_counts.items()
            if count >= 2 and durations[lower_speaker][speaker] >= 12.0
        ]
        stable_children = [
            speaker for speaker in children
            if speaker not in fragile_higher
            and higher_summary.get(speaker, {}).get("duration_s", 0.0) >= 45.0
            and (
                higher_summary.get(speaker, {}).get("stable_turns", 0) >= 4
                or (
                    higher_summary.get(speaker, {}).get("stable_turns", 0) >= 1
                    and higher_summary.get(speaker, {}).get("segments", 0) >= 12
                    and higher_summary.get(speaker, {}).get("duration_s", 0.0) >= 60.0
                )
            )
        ]
        review_children = [
            speaker for speaker in children
            if speaker in fragile_higher
            and durations[lower_speaker][speaker] >= 12.0
        ]
        if lower_actual >= 4:
            # Beyond four speakers, a tiny "review" child is more often an
            # over-split than a reliable new participant. Keep it visible in
            # review snippets, but do not let it raise the automatic count
            # unless it has enough duration and turn evidence.
            review_children = [
                speaker for speaker in review_children
                if (
                    float(higher_summary.get(speaker, {}).get("duration_s", 0.0)) >= 45.0
                    and int(higher_summary.get(speaker, {}).get("segments", 0)) >= 4
                    and int(higher_summary.get(speaker, {}).get("stable_turns", 0)) >= 2
                    and float(higher_summary.get(speaker, {}).get("duration_ratio", 0.0)) >= 0.015
                )
            ]
        if stable_children and review_children:
            higher["refinement_reason"] = (
                "保留更细人数：低人数候选中存在混合说话人，"
                "高人数候选拆出了稳定说话人和待抽听小簇"
            )
            return True
    return False


def _candidate_resolves_voice_mix(lower: dict, higher: dict) -> bool:
    lower_actual = int(lower.get("actual_n_speakers") or len(lower.get("summary", {}).get("speakers") or []))
    higher_actual = int(higher.get("actual_n_speakers") or len(higher.get("summary", {}).get("speakers") or []))
    if lower_actual < 2 or higher_actual <= lower_actual or higher_actual > lower_actual + 2:
        return False

    lower_segments = lower.get("segments") or []
    higher_segments = higher.get("segments") or []
    if not lower_segments or len(lower_segments) != len(higher_segments):
        return False

    mixed = lower.get("mixed_voice_speakers")
    if not mixed:
        voice_mix = lower.get("voice_mix_summary")
        if not isinstance(voice_mix, dict):
            voice_mix = _speaker_voice_band_mix_summary(lower_segments)
        mixed = [
            speaker
            for speaker, item in voice_mix.items()
            if bool((item or {}).get("mixed"))
        ]
    mixed_set = {str(speaker) for speaker in (mixed or [])}
    if not mixed_set:
        return False

    from collections import defaultdict

    by_parent_child: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {
            "duration": 0.0,
            "segments": 0,
            "low_duration": 0.0,
            "high_duration": 0.0,
        })
    )
    for lower_seg, higher_seg in zip(lower_segments, higher_segments):
        parent = str(lower_seg.get("speaker") or "")
        child = str(higher_seg.get("speaker") or "")
        if parent not in mixed_set or not child:
            continue
        pitch = _voice_pitch(higher_seg)
        band = _voice_band_from_pitch_value(pitch, _voice_confidence(higher_seg))
        if band not in {"low", "high"}:
            continue
        duration = max(0.1, _segment_duration(higher_seg))
        item = by_parent_child[parent][child]
        item["duration"] += duration
        item["segments"] += 1
        item[f"{band}_duration"] += duration

    for parent, child_stats in by_parent_child.items():
        reliable_children: list[tuple[str, str, float]] = []
        for child, stats in child_stats.items():
            duration = float(stats["duration"])
            segments = int(stats["segments"])
            if duration < 6.0 or segments < 2:
                continue
            low_duration = float(stats["low_duration"])
            high_duration = float(stats["high_duration"])
            dominant_band = "high" if high_duration >= low_duration else "low"
            dominant_duration = max(low_duration, high_duration)
            purity = dominant_duration / max(0.001, duration)
            if purity < 0.72:
                continue
            reliable_children.append((child, dominant_band, duration))

        has_low = any(band == "low" and duration >= 6.0 for _, band, duration in reliable_children)
        has_high = any(band == "high" and duration >= 6.0 for _, band, duration in reliable_children)
        if has_low and has_high:
            higher["refinement_reason"] = (
                "保留更细人数：低人数候选中存在高低声线混标，"
                "高人数候选拆出了更一致的声线"
            )
            return True
    return False


def _candidate_has_structural_refinement(lower: dict, higher: dict) -> bool:
    """Detect when a higher-count candidate splits a coarse mixed bucket.

    This is used only as a counterweight to the embedding model's lower-count
    bias. It requires stable, non-fragile child speakers and a clear parent
    mapping, so a lone noisy extra cluster cannot promote the speaker count.
    """
    lower_actual = int(lower.get("actual_n_speakers") or len(lower.get("summary", {}).get("speakers") or []))
    higher_actual = int(higher.get("actual_n_speakers") or len(higher.get("summary", {}).get("speakers") or []))
    if lower_actual < 2 or higher_actual <= lower_actual or higher_actual > lower_actual + 2:
        return False
    if _candidate_resolves_voice_mix(lower, higher):
        return True
    if (
        int(higher.get("tiny_speakers", 0)) > 0
        or int(higher.get("weak_speakers", 0)) > 0
        or int(higher.get("fragmented_speakers", 0)) > 0
        or int(higher.get("marginal_speakers", 0)) > 0
    ):
        return False
    if int(higher.get("stable_speakers", 0)) < higher_actual:
        return False

    higher_segments = higher.get("segments") or []
    lower_segments = lower.get("segments") or []
    if len(higher_segments) != len(lower_segments) or not higher_segments:
        return False

    lower_dominant = float(lower.get("dominant_ratio", 0.0) or 0.0)
    if lower_actual <= 2 and lower_dominant < 0.62:
        return False

    higher_summary = {
        str(s["speaker"]): s
        for s in higher.get("summary", {}).get("speakers", [])
    }
    lower_summary = {
        str(s["speaker"]): s
        for s in lower.get("summary", {}).get("speakers", [])
    }

    from collections import Counter, defaultdict

    durations_by_child: dict[str, Counter] = defaultdict(Counter)
    segments_by_child: dict[str, Counter] = defaultdict(Counter)
    switches_by_parent: Counter = Counter()
    previous_child_by_parent: dict[str, str] = {}
    for lower_seg, higher_seg in zip(lower_segments, higher_segments):
        lower_speaker = str(lower_seg.get("speaker") or "")
        higher_speaker = str(higher_seg.get("speaker") or "")
        if not lower_speaker or not higher_speaker:
            continue
        duration = _segment_duration(higher_seg)
        durations_by_child[higher_speaker][lower_speaker] += duration
        segments_by_child[higher_speaker][lower_speaker] += 1
        previous_child = previous_child_by_parent.get(lower_speaker)
        if previous_child and previous_child != higher_speaker:
            switches_by_parent[lower_speaker] += 1
        previous_child_by_parent[lower_speaker] = higher_speaker

    parent_children: dict[str, list[str]] = defaultdict(list)
    parent_child_durations: dict[str, list[float]] = defaultdict(list)
    for child, durations in durations_by_child.items():
        if not durations:
            continue
        child_stats = higher_summary.get(child)
        if not child_stats:
            continue
        total_duration = sum(float(v) for v in durations.values())
        parent, parent_duration = durations.most_common(1)[0]
        parent_stats = lower_summary.get(parent)
        if not parent_stats:
            continue
        purity = float(parent_duration) / max(1e-6, float(total_duration))
        parent_segment_count = int(segments_by_child[child][parent])
        child_duration = float(child_stats.get("duration_s", 0.0))
        child_turns = int(child_stats.get("stable_turns", 0))
        min_child_duration = 45.0 if lower_actual <= 2 else 25.0
        min_child_turns = 4 if lower_actual <= 2 else 1
        min_parent_segments = 12 if lower_actual <= 2 else 6
        if (
            purity >= 0.72
            and parent_segment_count >= min_parent_segments
            and child_duration >= min_child_duration
            and child_turns >= min_child_turns
            and float(parent_stats.get("duration_s", 0.0)) >= 45.0
        ):
            parent_children[parent].append(child)
            parent_child_durations[parent].append(child_duration)

    split_parents = [
        parent for parent, children in parent_children.items()
        if len(children) >= 2
        and (
            lower_actual > 2
            or switches_by_parent[parent] >= 6
            or min(parent_child_durations[parent] or [0.0]) >= 90.0
        )
    ]
    if split_parents:
        higher["refinement_reason"] = (
            "保留更细人数：低人数候选中存在混合说话人，"
            "高人数候选拆出了稳定子说话人"
        )
        return True
    return False


def _higher_count_has_weak_tail_over_split(lower: dict, higher: dict) -> bool:
    """Detect a common K-scan failure: one extra count rides on weak tail clusters.

    Forced clustering can make K=5 look better than K=4 by moving a small
    single-turn cluster to a new label and splitting one medium local block out
    of an existing speaker.  That shape is useful for review, but it should not
    automatically raise the recommended count unless the split proves a clear
    structural refinement.
    """
    lower_actual = int(lower.get("actual_n_speakers") or len(lower.get("summary", {}).get("speakers") or []))
    higher_actual = int(higher.get("actual_n_speakers") or len(higher.get("summary", {}).get("speakers") or []))
    if lower_actual < 3 or higher_actual < 5 or higher_actual != lower_actual + 1:
        return False

    fragile_count = (
        int(higher.get("tiny_speakers", 0))
        + int(higher.get("weak_speakers", 0))
        + int(higher.get("fragmented_speakers", 0))
        + int(higher.get("marginal_speakers", 0))
    )
    if fragile_count <= 0:
        return False
    if int(lower.get("stable_speakers", 0)) < int(higher.get("stable_speakers", 0)) - 1:
        return False
    if _candidate_has_meaningful_refinement(lower, higher):
        return False

    fragile_names = set(higher.get("fragile_speakers") or [])
    weak_tail = False
    borderline_stable = False
    for speaker in higher.get("summary", {}).get("speakers", []):
        name = str(speaker.get("speaker") or "")
        duration = float(speaker.get("duration_s", 0.0) or 0.0)
        segment_ratio = float(speaker.get("segment_ratio", 0.0) or 0.0)
        turns = int(speaker.get("turns", 0) or 0)
        stable_turns = int(speaker.get("stable_turns", 0) or 0)
        if name in fragile_names:
            if turns <= 1 or stable_turns <= 1 or segment_ratio < 0.035 or duration < 45.0:
                weak_tail = True
        elif turns <= 3 and stable_turns <= 3 and duration < 90.0:
            borderline_stable = True

    return weak_tail and borderline_stable


def _higher_count_adds_minor_split(lower: dict, higher: dict) -> bool:
    """Reject neat-looking extra speakers when the lower count is already clean."""
    lower_actual = int(lower.get("actual_n_speakers") or len(lower.get("summary", {}).get("speakers") or []))
    higher_actual = int(higher.get("actual_n_speakers") or len(higher.get("summary", {}).get("speakers") or []))
    if lower_actual < 2 or higher_actual != lower_actual + 1:
        return False
    if int(lower.get("stable_speakers", 0)) < lower_actual:
        return False
    if (
        int(lower.get("tiny_speakers", 0))
        + int(lower.get("weak_speakers", 0))
        + int(lower.get("fragmented_speakers", 0))
        + int(lower.get("marginal_speakers", 0))
    ) > 0:
        return False
    if _candidate_has_meaningful_refinement(lower, higher):
        return False
    if _candidate_has_structural_refinement(lower, higher):
        return False

    higher_speakers = list(higher.get("summary", {}).get("speakers") or [])
    if not higher_speakers:
        return False
    minor = min(
        higher_speakers,
        key=lambda s: (
            float(s.get("duration_s", 0.0) or 0.0),
            int(s.get("segments", 0) or 0),
        ),
    )
    duration = float(minor.get("duration_s", 0.0) or 0.0)
    duration_ratio = float(minor.get("duration_ratio", 0.0) or 0.0)
    segment_ratio = float(minor.get("segment_ratio", 0.0) or 0.0)
    stable_turns = int(minor.get("stable_turns", 0) or 0)
    turns = int(minor.get("turns", 0) or 0)
    return (
        duration < 35.0
        or duration_ratio < 0.12
        or segment_ratio < 0.12
        or stable_turns <= 2
        or turns <= 6
    )


def _model_recommended_n(candidate: dict) -> int:
    stats = candidate.get("stats") or {}
    try:
        return int(stats.get("model_recommended_n_speakers") or stats.get("recommended_n_speakers") or 0)
    except Exception:
        return 0


def _model_selected_score(candidate: dict) -> float | None:
    stats = candidate.get("stats") or {}
    value = stats.get("model_selected_score", stats.get("selected_score"))
    try:
        return float(value)
    except Exception:
        return None


def _model_recommended_score(candidate: dict) -> float | None:
    stats = candidate.get("stats") or {}
    value = stats.get("model_recommended_score", stats.get("recommended_score"))
    try:
        return float(value)
    except Exception:
        return None


def _model_recommended_confidence(candidate: dict) -> float:
    stats = candidate.get("stats") or {}
    try:
        return float(stats.get("model_recommended_confidence") or 0.0)
    except Exception:
        return 0.0


def _has_consistent_model_recommendation(candidate: dict, candidates: list[dict]) -> bool:
    model_n = _model_recommended_n(candidate)
    if model_n <= 0:
        return False
    compatible = [
        c for c in candidates
        if int(c.get("n_speakers") or 0) >= max(2, model_n)
    ]
    if not compatible:
        return False
    votes = sum(1 for c in compatible if _model_recommended_n(c) == model_n)
    return votes >= max(2, int(len(compatible) * 0.6))


def _model_under_count_refinement(
    current: dict,
    candidates: list[dict],
    model_n: int,
) -> dict | None:
    """Move up one count when a trusted acoustic prior exposes under-splitting."""
    current_actual = int(
        current.get("actual_n_speakers")
        or len(current.get("summary", {}).get("speakers") or [])
    )
    if (
        model_n <= current_actual
        or _model_recommended_confidence(current) < 0.40
        or not current.get("severe_mixed_voice_speakers")
    ):
        return None

    target_actual = current_actual + 1
    refinements = []
    for candidate in candidates:
        actual = int(
            candidate.get("actual_n_speakers")
            or len(candidate.get("summary", {}).get("speakers") or [])
        )
        if actual != target_actual or actual > model_n:
            continue
        fragile = int(candidate.get("weak_speakers", 0)) + int(candidate.get("tiny_speakers", 0))
        if fragile > int(current.get("weak_speakers", 0)) + int(current.get("tiny_speakers", 0)):
            continue
        if int(candidate.get("stable_speakers", 0)) < actual:
            continue
        refinements.append(candidate)
    if not refinements:
        return None
    return max(
        refinements,
        key=lambda candidate: (
            float(candidate.get("score") or 0.0),
            -int(candidate.get("n_speakers") or 0),
        ),
    )


def _model_penalizes_higher_count(anchor: dict, higher: dict, *, margin: float = 0.03) -> bool:
    anchor_score = _model_selected_score(anchor)
    higher_score = _model_selected_score(higher)
    if anchor_score is None or higher_score is None:
        recommended_score = _model_recommended_score(higher)
        if recommended_score is None:
            return False
        higher_score = higher_score if higher_score is not None else recommended_score
        anchor_score = anchor_score if anchor_score is not None else recommended_score
    return (anchor_score - higher_score) >= margin


def _candidate_embedding_count(candidate: dict) -> int:
    stats = candidate.get("stats") or {}
    for key in ("embeddings", "senko_subsegments"):
        try:
            value = int(stats.get(key) or 0)
        except Exception:
            value = 0
        if value > 0:
            return value
    return 0


def _model_conflict_refinement(anchor: dict, candidates: list[dict], model_n: int) -> dict | None:
    """Return a limited higher-count refinement when K=auto is too coarse."""
    anchor_actual = int(anchor.get("actual_n_speakers") or len(anchor.get("summary", {}).get("speakers") or []))
    upper_actual = min(8, anchor_actual + 2)
    if model_n <= 2:
        # When the embedding silhouette strongly prefers two people, jumping
        # straight to 5-8 is usually over-splitting. Keep those as visible
        # candidates, but make the automatic choice stop at a reviewable 3/4.
        upper_actual = min(upper_actual, 4)

    refinements = []
    for c in candidates:
        actual = int(c.get("actual_n_speakers") or len(c.get("summary", {}).get("speakers") or []))
        if actual <= anchor_actual or actual > upper_actual:
            continue
        if _candidate_has_structural_refinement(anchor, c):
            refinements.append(c)

    if not refinements:
        return None
    if model_n <= 2:
        return max(
            refinements,
            key=lambda c: (
                int(c.get("actual_n_speakers") or len(c.get("summary", {}).get("speakers") or [])),
                float(c.get("score") or 0.0),
            ),
        )
    return min(
        refinements,
        key=lambda c: (
            int(c.get("actual_n_speakers") or len(c.get("summary", {}).get("speakers") or [])),
            -float(c.get("score") or 0.0),
        ),
    )


def _trusted_high_count_model_anchor(
    anchor: dict,
    candidates: list[dict],
) -> bool:
    """Accept a consistent 5-8 person acoustic anchor with durable support.

    High-count meetings naturally contain low-share participants, so the
    ordinary weak-speaker score can undercount them. The model anchor may
    override that score only when every proposed speaker has independent,
    non-fragmentary evidence across the recording.
    """
    model_n = _model_recommended_n(anchor)
    actual_n = int(
        anchor.get("actual_n_speakers")
        or len(anchor.get("summary", {}).get("speakers") or [])
    )
    if (
        model_n < 5
        or actual_n != model_n
        or _model_recommended_confidence(anchor) < 0.40
        or not _has_consistent_model_recommendation(anchor, candidates)
        or int(anchor.get("tiny_speakers", 0)) > 0
        or int(anchor.get("fragmented_speakers", 0)) > 0
        or int(anchor.get("marginal_speakers", 0)) > 0
    ):
        return False
    speakers = [
        row
        for row in anchor.get("summary", {}).get("speakers") or []
        if isinstance(row, dict)
    ]
    if len(speakers) != model_n:
        return False
    return all(
        int(row.get("segments", 0) or 0) >= 5
        and float(row.get("duration_s", 0.0) or 0.0) >= 20.0
        and int(row.get("stable_turns", 0) or 0) >= 1
        for row in speakers
    )


def _medium_confidence_unresolved_count_fallback(
    current: dict,
    candidates: list[dict],
) -> dict | None:
    """Prefer one fewer speaker when a medium-confidence split stays mixed.

    Candidate fragmentation metrics are derived from ASR segment boundaries,
    so a harmless transcription re-segmentation can make the same acoustic
    cluster appear stable or fragmented.  Do not let that presentation detail
    turn a still-mixed, medium-confidence 3/4-person split into a hard count.
    The lower candidate must be acoustically plausible and contain no weak or
    fragmented speakers; high-confidence speaker-count evidence is untouched.
    """
    model_n = _model_recommended_n(current)
    actual_n = int(
        current.get("actual_n_speakers")
        or len(current.get("summary", {}).get("speakers") or [])
    )
    confidence = _model_recommended_confidence(current)
    severe_mix = set(current.get("severe_mixed_voice_speakers") or [])
    if (
        model_n not in {3, 4}
        or actual_n != model_n
        or not 0.40 <= confidence < 0.70
        or len(severe_mix) < max(1, (actual_n + 1) // 2)
        or not _has_consistent_model_recommendation(current, candidates)
    ):
        return None

    lower_actual = actual_n - 1
    lower_candidates = []
    for candidate in candidates:
        formed = int(
            candidate.get("actual_n_speakers")
            or len(candidate.get("summary", {}).get("speakers") or [])
        )
        if formed != lower_actual:
            continue
        fragile = (
            int(candidate.get("tiny_speakers", 0))
            + int(candidate.get("weak_speakers", 0))
            + int(candidate.get("fragmented_speakers", 0))
            + int(candidate.get("marginal_speakers", 0))
        )
        if fragile > 0 or int(candidate.get("stable_speakers", 0)) < lower_actual:
            continue
        recommended_score = _model_recommended_score(candidate)
        selected_score = _model_selected_score(candidate)
        if (
            recommended_score is None
            or selected_score is None
            or recommended_score <= 0.0
            or selected_score / recommended_score < 0.45
        ):
            continue
        lower_candidates.append(candidate)

    if not lower_candidates:
        return None
    fallback = max(
        lower_candidates,
        key=lambda candidate: (
            int(candidate.get("n_speakers") or 0) == lower_actual,
            float(candidate.get("score") or 0.0),
            -int(candidate.get("n_speakers") or 0),
        ),
    )
    current["over_split_guard_reason"] = (
        "底层人数证据只有中等置信，且高人数候选仍有严重声线混标"
    )
    fallback["model_guard_reason"] = (
        f"底层 {model_n} 人证据只有中等置信，且该候选仍有严重声线混标，"
        f"已采用声学仍可接受的 {lower_actual} 人结果"
    )
    return fallback


def _choose_diarization_candidate(candidates: list[dict]) -> dict:
    """Pick the practical speaker count, biasing against over-splitting.

    The highest raw score wins, but near-ties prefer the smaller count when the
    larger candidate only adds weak/tiny speakers. This keeps real low-volume
    participants (like 10 segments / 28s) while rejecting mostly-noise splits.
    """
    best = max(
        candidates,
        key=lambda c: (
            c["score"],
            c["stable_speakers"],
            -c["weak_speakers"],
            -c["tiny_speakers"],
            -c["n_speakers"],
        ),
    )
    by_actual: dict[int, dict] = {}
    for candidate in candidates:
        actual = int(
            candidate.get("actual_n_speakers")
            or len(candidate.get("summary", {}).get("speakers") or [])
        )
        current = by_actual.get(actual)
        if current is None:
            by_actual[actual] = candidate
            continue

        requested = int(candidate.get("n_speakers") or 0)
        current_requested = int(current.get("n_speakers") or 0)
        # A forced K run may collapse to fewer real clusters. Keep the run that
        # actually requested this count as the acoustic-model anchor; otherwise
        # a later K=5/6/7 run can overwrite the genuine K=4 candidate.
        candidate_rank = (
            requested == actual,
            float(candidate.get("score") or 0.0),
            -abs(requested - actual),
            -requested,
        )
        current_rank = (
            current_requested == actual,
            float(current.get("score") or 0.0),
            -abs(current_requested - actual),
            -current_requested,
        )
        if candidate_rank > current_rank:
            by_actual[actual] = candidate
    model_candidates = [
        c for c in candidates
        if _model_recommended_n(c) > 0
        and _model_recommended_n(c) in by_actual
        and _has_consistent_model_recommendation(c, candidates)
    ]
    trusted_high_count_anchor = None
    if model_candidates:
        # If the embedding model consistently says "K speakers" across forced
        # runs, use that as the baseline. Higher-K candidates must prove they are
        # a meaningful refinement; otherwise they are usually over-splitting one
        # speaker into several neat-looking buckets.
        model_n = min(_model_recommended_n(c) for c in model_candidates)
        anchor = by_actual.get(model_n)
        if anchor is not None:
            if _trusted_high_count_model_anchor(anchor, candidates):
                trusted_high_count_anchor = anchor
            if best is not anchor and _model_penalizes_higher_count(anchor, best):
                refinement = _model_conflict_refinement(anchor, candidates, model_n)
                if refinement is not None:
                    refinement["model_guard_reason"] = (
                        f"底层聚类更支持 {model_n} 人，但候选结果显示低人数中存在混合说话人"
                    )
                    best = refinement
                else:
                    guard_reason = f"底层聚类更支持 {model_n} 人，高人数候选疑似过度拆分"
                    best["over_split_guard_reason"] = guard_reason
                    anchor["model_guard_reason"] = guard_reason
                    best = anchor
            under_count_refinement = _model_under_count_refinement(best, candidates, model_n)
            if under_count_refinement is not None:
                previous_actual = int(
                    best.get("actual_n_speakers")
                    or len(best.get("summary", {}).get("speakers") or [])
                )
                refined_actual = int(
                    under_count_refinement.get("actual_n_speakers")
                    or len(under_count_refinement.get("summary", {}).get("speakers") or [])
                )
                under_count_refinement["model_guard_reason"] = (
                    f"底层声纹支持 {model_n} 人，当前 {previous_actual} 人仍有严重声线混标，"
                    f"已保守上调到 {refined_actual} 人"
                )
                best = under_count_refinement
    for c in candidates:
        if c is best or c["n_speakers"] >= best["n_speakers"]:
            continue
        cleaner = (
            c["tiny_speakers"] <= best["tiny_speakers"]
            and c["weak_speakers"] <= best["weak_speakers"]
        )
        same_support = c["stable_speakers"] >= best["stable_speakers"]
        if cleaner and same_support and c["score"] >= best["score"] - 2.0:
            best = c
            continue
        if (
            c["stable_speakers"] == best["stable_speakers"]
            and (c["weak_speakers"] + c["tiny_speakers"])
            < (best["weak_speakers"] + best["tiny_speakers"])
            and c["score"] >= best["score"] - 4.0
        ):
            best = c
    # A meaningful low-frequency participant may justify one higher count, but
    # the decision must be anchored to the result selected above. Mutating
    # ``best`` while scanning candidates used to permit 4 -> 5 -> 6 -> 7
    # cascades, where each individually plausible split amplified the previous
    # one. One recommendation pass may now promote at most one actual speaker.
    refinement_anchor = best
    anchor_actual = int(
        refinement_anchor.get("actual_n_speakers")
        or len(refinement_anchor.get("summary", {}).get("speakers") or [])
    )
    one_step_refinements = []
    for c in candidates:
        if c is refinement_anchor:
            continue
        candidate_actual = int(
            c.get("actual_n_speakers")
            or len(c.get("summary", {}).get("speakers") or [])
        )
        if (
            candidate_actual == anchor_actual + 1
            and c["score"] >= refinement_anchor["score"] - 8.0
            and _candidate_has_meaningful_refinement(refinement_anchor, c)
        ):
            one_step_refinements.append(c)
    if one_step_refinements:
        best = max(
            one_step_refinements,
            key=lambda c: (
                c["score"],
                c["stable_speakers"],
                -c["weak_speakers"],
                -c["tiny_speakers"],
                -c["n_speakers"],
            ),
        )

    for c in candidates:
        if c is best:
            continue
        same_actual = int(c.get("actual_n_speakers") or 0) == int(best.get("actual_n_speakers") or 0)
        if (
            same_actual
            and c.get("merge_map")
            and not best.get("merge_map")
            and c["score"] >= best["score"] - 5.0
        ):
            best = c
    for c in candidates:
        if c is best or c["n_speakers"] >= best["n_speakers"]:
            continue
        score_gap = float(best.get("score") or 0.0) - float(c.get("score") or 0.0)
        if score_gap > 3.0:
            continue
        if _higher_count_adds_minor_split(c, best):
            best["over_split_guard_reason"] = "高人数候选只增加低占比小簇，疑似把同一说话人过度拆分"
            c["model_guard_reason"] = "高人数候选只增加低占比小簇，已优先采用更稳的人数"
            best = c
    for c in candidates:
        if c is best or c["n_speakers"] >= best["n_speakers"]:
            continue
        score_gap = float(best.get("score") or 0.0) - float(c.get("score") or 0.0)
        if score_gap > 6.0:
            continue
        if _higher_count_has_weak_tail_over_split(c, best):
            best["over_split_guard_reason"] = "高人数候选包含低支持弱簇，疑似把局部片段过度拆成人"
            c["model_guard_reason"] = "高人数候选包含低支持弱簇，已优先采用更稳的人数"
            best = c
    unresolved_fallback = _medium_confidence_unresolved_count_fallback(best, candidates)
    if unresolved_fallback is not None:
        best = unresolved_fallback
    if trusted_high_count_anchor is not None:
        trusted_high_count_anchor["model_guard_reason"] = (
            f"底层声纹一致支持 {_model_recommended_n(trusted_high_count_anchor)} 人，"
            "且每位说话人均有独立持续声纹证据"
        )
        best = trusted_high_count_anchor
    return best


def _recommendation_confidence(candidates: list[dict], best: dict) -> tuple[str, str, float]:
    others = [c for c in candidates if c is not best]
    runner_up = max((c["score"] for c in others), default=-999.0)
    gap = round(float(best["score"]) - float(runner_up), 3)
    if best.get("severe_mixed_voice_speakers"):
        return "low", "推荐结果仍存在严重声线混标，建议提高/确认人数并抽听关键片段", gap
    if best.get("mixed_voice_speakers"):
        return "medium", "推荐结果存在声线混合风险，建议抽听关键片段", gap
    if best.get("refinement_reason"):
        return "low", f"{best['refinement_reason']}，建议抽听关键片段", gap
    if best.get("model_guard_reason"):
        return "medium", f"{best['model_guard_reason']}，建议抽听关键片段", gap
    if gap < 0:
        return "low", f"已优先采用可解释合并候选；最高分候选领先 {abs(gap):.1f}，建议抽听关键片段", gap
    fragile = (
        int(best.get("tiny_speakers", 0))
        + int(best.get("weak_speakers", 0))
        + int(best.get("fragmented_speakers", 0))
        + int(best.get("marginal_speakers", 0))
    )
    if gap >= 5.0 and fragile == 0:
        return "high", f"推荐分领先 {gap:.1f}，且无弱/碎片说话人", gap
    if gap >= 2.0 and fragile <= 1:
        return "medium", f"推荐分领先 {gap:.1f}，建议抽听关键片段", gap
    return "low", f"候选分差 {gap:.1f}，建议人工确认人数", gap


def _annotate_recommendation_stats(candidates: list[dict], best: dict) -> None:
    best_actual_n = int(best.get("actual_n_speakers") or len(best.get("speakers") or []))
    confidence, confidence_reason, score_gap_to_next = _recommendation_confidence(candidates, best)
    for c in candidates:
        score_gap = round(float(best["score"]) - float(c["score"]), 3)
        model_over_split = False
        model_n = _model_recommended_n(c)
        model_recommended_score = _model_recommended_score(c)
        model_selected_score = _model_selected_score(c)
        if model_n and int(c.get("actual_n_speakers") or len(c.get("speakers") or [])) > model_n:
            if (
                model_recommended_score is not None
                and model_selected_score is not None
                and (model_recommended_score - model_selected_score) >= 0.03
            ):
                model_over_split = True
        over_split = (
            int(c.get("actual_n_speakers") or len(c.get("speakers") or [])) > best_actual_n
            and (
                c["tiny_speakers"] > 0
                or c["weak_speakers"] > best["weak_speakers"]
                or c.get("fragmented_speakers", 0) > best.get("fragmented_speakers", 0)
                or score_gap >= 2.0
                or model_over_split
            )
        )
        severe_voice_mix = bool(c.get("severe_mixed_voice_speakers"))
        voice_mix = bool(c.get("mixed_voice_speakers"))
        if c is best:
            if severe_voice_mix:
                risk_level = "high"
                risk_reason = "推荐结果仍存在严重声线混标"
            elif voice_mix:
                risk_level = "medium"
                risk_reason = "推荐结果存在声线混合风险"
            elif confidence == "high":
                risk_level = "low"
                risk_reason = "推荐人数"
            else:
                risk_level = "medium"
                risk_reason = confidence_reason or "推荐结果置信度不足，建议人工确认"
        elif over_split and (
            c["tiny_speakers"] > 0
            or c.get("fragmented_speakers", 0) > 0
            or score_gap >= 4.0
            or model_over_split
        ):
            risk_level = "high"
            if model_over_split and model_n:
                risk_reason = f"底层聚类更支持 {model_n} 人，该候选疑似过度拆分"
            else:
                risk_reason = "人数偏多，容易把短噪声或碎片拆成新人"
        elif over_split or c["score"] < best["score"] - 2.0:
            risk_level = "medium"
            risk_reason = "候选结果不如推荐人数稳定"
        else:
            risk_level = "low"
            risk_reason = c["reason"]

        stats = dict(c.get("stats") or {})
        original_model_recommended_n = stats.get("model_recommended_n_speakers", stats.get("recommended_n_speakers"))
        original_model_recommended_score = stats.get("model_recommended_score", stats.get("recommended_score"))
        original_model_selected_score = stats.get("model_selected_score", stats.get("selected_score"))
        actual_n = int(c.get("actual_n_speakers") or len(c.get("summary", {}).get("speakers") or []))
        if c is best:
            display_risk_reason = (
                c.get("model_guard_reason")
                or c.get("resegmentation_reason")
                or c.get("voice_line_refine_reason")
                or c.get("handoff_voice_guard_reason")
                or c.get("handoff_split_reason")
                or c.get("local_leakage_reason")
                or c.get("continuity_repair_reason")
                or c.get("voice_band_repair_reason")
                or c.get("voice_guard_reason")
                or c.get("smoothing_reason")
                or c.get("short_sandwich_reason")
                or c.get("reassignment_reason")
                or c.get("refinement_reason")
                or risk_reason
            )
        elif model_over_split and model_n:
            display_risk_reason = risk_reason
        else:
            display_risk_reason = (
                c.get("handoff_split_reason")
                or c.get("handoff_voice_guard_reason")
                or c.get("voice_line_refine_reason")
                or c.get("local_leakage_reason")
                or c.get("continuity_repair_reason")
                or c.get("voice_band_repair_reason")
                or c.get("voice_guard_reason")
                or c.get("smoothing_reason")
                or c.get("short_sandwich_reason")
                or c.get("reassignment_reason")
                or c.get("refinement_reason")
                or risk_reason
            )

        stats.update({
            "clusters": actual_n,
            "model_recommended_n_speakers": original_model_recommended_n,
            "model_recommended_score": original_model_recommended_score,
            "model_selected_score": original_model_selected_score,
            "requested_n_speakers": int(c["n_speakers"]),
            "recommended_n_speakers": best_actual_n,
            "recommended_run_n_speakers": int(best["n_speakers"]),
            "recommended_score": float(best["score"]),
            "selected_score": float(c["score"]),
            "over_split_risk": bool(over_split),
            "over_split_score_gap": (
                score_gap
                if int(c.get("actual_n_speakers") or len(c.get("speakers") or [])) > best_actual_n
                else 0.0
            ),
            "risk_level": risk_level,
            "risk_reason": display_risk_reason,
            "recommendation_reason": c["reason"],
            "recommendation_confidence": confidence if c is best else None,
            "recommendation_confidence_reason": confidence_reason if c is best else None,
            "score_gap_to_next": score_gap_to_next if c is best else None,
            "merge_map": c.get("merge_map") or {},
            "merge_distribution": c.get("merge_distribution") or {},
            "merge_reason": c.get("merge_reason") or "",
            "reassignment_distribution": c.get("reassignment_distribution") or {},
            "reassignment_reason": c.get("reassignment_reason") or "",
            "smoothing_distribution": c.get("smoothing_distribution") or {},
            "smoothing_reason": c.get("smoothing_reason") or "",
            "short_sandwich_distribution": c.get("short_sandwich_distribution") or {},
            "short_sandwich_reason": c.get("short_sandwich_reason") or "",
            "local_leakage_distribution": c.get("local_leakage_distribution") or {},
            "local_leakage_reason": c.get("local_leakage_reason") or "",
            "continuity_repair_distribution": c.get("continuity_repair_distribution") or {},
            "continuity_repair_reason": c.get("continuity_repair_reason") or "",
            "voice_band_repair_distribution": c.get("voice_band_repair_distribution") or {},
            "voice_band_repair_reason": c.get("voice_band_repair_reason") or "",
            "voice_profiles": c.get("voice_profiles") or stats.get("speaker_voice_summary") or {},
            "voice_mix_summary": c.get("voice_mix_summary") or {},
            "mixed_voice_speakers": c.get("mixed_voice_speakers") or [],
            "severe_mixed_voice_speakers": c.get("severe_mixed_voice_speakers") or [],
            "voice_mix_penalty": c.get("voice_mix_penalty") or 0.0,
            "voice_line_groups": c.get("voice_line_groups") or _speaker_voice_line_groups(c.get("segments") or []),
            "voice_line_refine_count": c.get("voice_line_refine_count") or 0,
            "voice_line_review_count": c.get("voice_line_review_count") or 0,
            "voice_line_refine_reason": c.get("voice_line_refine_reason") or "",
            "resegmentation_count": c.get("resegmentation_count") or 0,
            "resegmentation_reason": c.get("resegmentation_reason") or "",
            "voice_guard_count": c.get("voice_guard_count") or 0,
            "voice_guard_reason": c.get("voice_guard_reason") or "",
            "handoff_split_count": c.get("handoff_split_count") or 0,
            "handoff_split_reason": c.get("handoff_split_reason") or "",
            "handoff_voice_guard_distribution": c.get("handoff_voice_guard_distribution") or {},
            "handoff_voice_guard_reason": c.get("handoff_voice_guard_reason") or "",
            "review_segments": _build_review_segments(c),
        })
        c["stats"] = stats


def _recommend_diarization_candidates(
    *,
    audio: Path,
    segments: list[dict],
    profiles: list[dict],
    min_speakers: int,
    max_speakers: int,
    engine: str | None = None,
    progress_method: str = "recommend_diarization",
    preserve_segmentation: bool = True,
) -> dict:
    """Run fixed speaker-count candidates and recommend a practical count."""
    from .diarizers import diarize as _diarize

    frozen_segments = deepcopy(segments)
    segments = deepcopy(frozen_segments)

    candidates = []
    errors = []
    for n_speakers in range(min_speakers, max_speakers + 1):
        if len(segments) == 0:
            break
        try:
            result = _diarize(
                audio=audio,
                segments=segments,
                n_speakers=n_speakers,
                profiles=profiles,
                engine=engine,
                on_progress=_make_progress(progress_method),
            )
        except Exception as exc:
            errors.append({
                "n_speakers": n_speakers,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        emb_count = int((result.stats or {}).get("embeddings") or (result.stats or {}).get("senko_subsegments") or 0)
        if emb_count > 0 and n_speakers >= emb_count:
            errors.append({
                "n_speakers": n_speakers,
                "error": f"有效声纹片段只有 {emb_count} 个，跳过 {n_speakers} 人候选",
            })
            break
        out_segments = _segments_with_diarization_speakers(segments, result.segments)
        if not _has_complete_speaker_assignment(segments, out_segments):
            errors.append({
                "n_speakers": n_speakers,
                "error": "分人结果没有为每个 transcript segment 提供 speaker",
            })
            continue
        candidate = {
            "n_speakers": n_speakers,
            "speakers": result.speakers,
            "segments": out_segments,
            "matched_profiles": result.matched_profiles,
            "stats": result.stats,
            "summary": _speaker_summary(out_segments),
        }
        candidate.update(_score_diarization_candidate(candidate, n_speakers))
        candidates.append(candidate)

    processed_candidates = []
    for candidate in candidates:
        if preserve_segmentation:
            processed = {
                **candidate,
                "segments": [dict(seg) for seg in candidate.get("segments") or []],
                "stats": dict(candidate.get("stats") or {}),
                "resegmentation_count": 0,
                "resegmentation_reason": "",
                "handoff_split_count": 0,
                "handoff_split_reason": "",
                "segmentation_preserved": True,
            }
        else:
            processed = _resegment_mixed_speaker_segments(candidate)
            processed = _split_handoff_segments(processed)
            processed["segmentation_preserved"] = False
        processed = _repair_handoff_voice_guard_assignments(processed)
        processed = _reassign_isolated_fragile_segments(processed, candidates)
        processed = _smooth_short_sandwiched_segments(processed)
        processed = _smooth_windowed_sandwiched_runs(processed)
        processed = _smooth_alternating_local_speaker_leakage(processed)
        processed = _repair_discourse_continuity_assignments(processed)
        processed = _repair_voice_band_assignments(processed)
        processed = _project_speaker_cues(processed)
        processed_candidates.append(processed)
    candidates = processed_candidates
    validated_candidates = []
    for candidate in candidates:
        candidate_segments = _finalize_speaker_metadata_only(
            frozen_segments, candidate.get("segments") or []
        )
        if candidate_segments is None or not all(
            str(segment.get("speaker") or "").strip()
            for segment in candidate_segments
        ):
            errors.append({
                "n_speakers": int(candidate.get("n_speakers") or 0),
                "error": "分人后处理改变冻结转录或留下未标注 transcript segment",
            })
            continue
        candidate["segments"] = candidate_segments
        validated_candidates.append(candidate)
    candidates = validated_candidates

    if not candidates:
        return {
            "recommended_n_speakers": 0,
            "candidates": [],
            "reason": "没有可分析的分人结果",
            "errors": errors,
        }

    best = _choose_diarization_candidate(candidates)
    best_index = candidates.index(best)
    refined_best = _refine_conflicting_voice_bands_with_pyin(best, audio)
    refined_best = _repair_voice_band_assignments(refined_best)
    refined_best = _project_speaker_cues(refined_best)
    refined_best = _repair_long_missing_speaker_cues(refined_best, audio)
    finalized_refined_segments = _finalize_speaker_metadata_only(
        frozen_segments, refined_best.get("segments") or []
    )
    if finalized_refined_segments is not None:
        refined_best["segments"] = finalized_refined_segments
    if (
        finalized_refined_segments is not None
        and all(
            str(segment.get("speaker") or "").strip()
            for segment in finalized_refined_segments
        )
    ):
        candidates[best_index] = refined_best
        best = refined_best
    else:
        errors.append({
            "n_speakers": int(best.get("n_speakers") or 0),
            "error": "最终声线复核改变了 transcript 几何，已保留复核前结果",
        })
    for c in candidates:
        c["review_segments"] = _build_review_segments(c)
    _annotate_recommendation_stats(candidates, best)
    best_actual_n = int(best.get("actual_n_speakers") or len(best.get("speakers") or []))
    confidence, confidence_reason, score_gap_to_next = _recommendation_confidence(candidates, best)
    best_review_segments = list(best.get("review_segments") or [])
    if confidence in {"low", "medium"}:
        extra_review_segments = _build_count_ambiguity_review_segments(best, candidates)
        if not extra_review_segments:
            extra_review_segments = _build_low_confidence_speaker_review_segments(best)
        best_review_segments = _merge_review_segments([*best_review_segments, *extra_review_segments], limit=64)
        best["review_segments"] = best_review_segments
        best_stats = dict(best.get("stats") or {})
        best_stats["review_segments"] = best_review_segments
        best["stats"] = best_stats

    for c in candidates:
        stats_review = (c.get("stats") or {}).get("review_segments")
        if stats_review:
            c["review_segments"] = stats_review
        safe_segments = deepcopy(c.get("segments") or [])
        _annotate_segments_with_speaker_reviews(c)
        finalized_segments = _finalize_speaker_metadata_only(
            frozen_segments, c.get("segments") or []
        )
        if finalized_segments is None:
            c["segments"] = safe_segments
            errors.append({
                "n_speakers": int(c.get("n_speakers") or 0),
                "error": "待确认标记试图改变冻结转录，已保留标记前结果",
            })
        else:
            c["segments"] = finalized_segments

    return _simplify_diarization_response({
        "recommended_n_speakers": best_actual_n,
        "recommended_candidate_n_speakers": best["n_speakers"],
        "reason": best.get("model_guard_reason") or best.get("refinement_reason") or best["reason"],
        "confidence": confidence,
        "confidence_reason": confidence_reason,
        "score_gap_to_next": score_gap_to_next,
        "merge_map": best.get("merge_map") or {},
        "merge_distribution": best.get("merge_distribution") or {},
        "merge_reason": best.get("merge_reason") or "",
        "reassignment_distribution": best.get("reassignment_distribution") or {},
        "reassignment_reason": best.get("reassignment_reason") or "",
        "smoothing_distribution": best.get("smoothing_distribution") or {},
        "smoothing_reason": best.get("smoothing_reason") or "",
        "short_sandwich_distribution": best.get("short_sandwich_distribution") or {},
        "short_sandwich_reason": best.get("short_sandwich_reason") or "",
        "local_leakage_distribution": best.get("local_leakage_distribution") or {},
        "local_leakage_reason": best.get("local_leakage_reason") or "",
        "continuity_repair_distribution": best.get("continuity_repair_distribution") or {},
        "continuity_repair_reason": best.get("continuity_repair_reason") or "",
        "voice_band_repair_distribution": best.get("voice_band_repair_distribution") or {},
        "voice_band_repair_reason": best.get("voice_band_repair_reason") or "",
        "voice_profiles": best.get("voice_profiles") or (best.get("stats") or {}).get("speaker_voice_summary") or {},
        "voice_mix_summary": best.get("voice_mix_summary") or {},
        "mixed_voice_speakers": best.get("mixed_voice_speakers") or [],
        "severe_mixed_voice_speakers": best.get("severe_mixed_voice_speakers") or [],
        "voice_mix_penalty": best.get("voice_mix_penalty") or 0.0,
        "voice_line_groups": best.get("voice_line_groups") or _speaker_voice_line_groups(best.get("segments") or []),
        "voice_line_refine_count": best.get("voice_line_refine_count") or 0,
        "voice_line_review_count": best.get("voice_line_review_count") or 0,
        "voice_line_refine_reason": best.get("voice_line_refine_reason") or "",
        "resegmentation_count": best.get("resegmentation_count") or 0,
        "resegmentation_reason": best.get("resegmentation_reason") or "",
        "voice_guard_count": best.get("voice_guard_count") or 0,
        "voice_guard_reason": best.get("voice_guard_reason") or "",
        "handoff_split_count": best.get("handoff_split_count") or 0,
        "handoff_split_reason": best.get("handoff_split_reason") or "",
        "handoff_voice_guard_distribution": best.get("handoff_voice_guard_distribution") or {},
        "handoff_voice_guard_reason": best.get("handoff_voice_guard_reason") or "",
        "review_segments": best_review_segments,
        "errors": errors,
        "candidates": candidates,
    })


def handle_recommend_diarization(params: dict) -> dict:
    """Run fixed speaker-count candidates and recommend a practical count."""
    audio = _existing_audio_path(params["audio"])
    segments = params.get("segments") or []
    profiles = params.get("profiles") or []
    min_speakers = max(2, int(params.get("min_speakers", params.get("minSpeakers", 2))))
    max_speakers = min(8, max(min_speakers, int(params.get("max_speakers", params.get("maxSpeakers", 8)))))
    saved_diar = _load_saved_diarization_settings()
    engine = params.get("engine") or saved_diar.get("engine") or "auto"
    preserve_segmentation = bool(
        params.get("preserve_segmentation", params.get("preserveSegmentation", True))
    )

    return _recommend_diarization_candidates(
        audio=audio,
        segments=segments,
        profiles=profiles,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        engine=engine,
        preserve_segmentation=preserve_segmentation,
    )


def handle_extract_voice_embedding(params: dict) -> dict:
    """params: { audio: str } → { embedding: list[float] (256), name?: str }"""
    from .diarizers import extract_voice_embedding
    audio = _existing_audio_path(params["audio"])
    emb = extract_voice_embedding(audio)
    return {
        "embedding": emb,
        "dims": len(emb),
        "audio": str(audio),
    }


def handle_preflight_voiceprint_anchors(params: dict) -> dict:
    """Read-only CAM++ quality preflight for the voiceprint anchor picker."""
    from .diarizers import preflight_voiceprint_anchor_candidates

    audio = _existing_audio_path(params["audio"])
    segments = params.get("segments") or []
    saved_diar = _load_saved_diarization_settings()
    engine = params.get("engine") or saved_diar.get("engine") or "auto"
    return preflight_voiceprint_anchor_candidates(
        audio=audio,
        segments=segments,
        engine=engine,
        on_progress=_make_progress("preflight_voiceprint_anchors"),
    )


def handle_reidentify_speakers(params: dict) -> dict:
    """Re-label current transcript from user-confirmed voice anchors.

    params: {
      audio: str,
      segments: [{start,end,text,speaker?}],
      anchors: [{start,end,speaker}],
      threshold?: float,
      review_threshold?: float,
      margin?: float,
      engine?: "auto"|"senko",
    }
    """
    from .diarizers import reidentify_with_voice_anchors

    audio = _existing_audio_path(params["audio"])
    segments = params.get("segments") or []
    anchors = params.get("anchors") or []
    saved_diar = _load_saved_diarization_settings()
    engine = params.get("engine") or saved_diar.get("engine") or "auto"
    threshold = float(params.get("threshold") or 0.78)
    review_threshold = float(params.get("review_threshold") or params.get("reviewThreshold") or 0.70)
    margin = float(params.get("margin") or 0.05)
    require_enrollment_quality = bool(
        params.get("require_enrollment_quality")
        if "require_enrollment_quality" in params
        else params.get("requireEnrollmentQuality", True)
    )
    result = reidentify_with_voice_anchors(
        audio=audio,
        segments=segments,
        anchors=anchors,
        threshold=threshold,
        review_threshold=review_threshold,
        margin=margin,
        require_enrollment_quality=require_enrollment_quality,
        engine=engine,
        on_progress=_make_progress("reidentify_speakers"),
    )
    result_segments = result.get("segments") or []
    if not _has_complete_speaker_assignment(segments, result_segments):
        raise RuntimeError("声纹回扫没有为每个 transcript segment 提供 speaker")
    if not _preserves_transcript_geometry(segments, result_segments):
        raise RuntimeError("声纹回扫改变了 transcript segment 数量、文字或时间轴")
    return _simplify_diarization_response(result)


def _run_asr_handler(handler: Any, params: dict) -> Any:
    with _asr_job_lock:
        return handler(params)


def _run_correction_handler(handler: Any, params: dict) -> Any:
    with _correction_job_lock:
        return handler(params)


HANDLERS: dict[str, Any] = {
    "check_model": handle_check_model,
    "probe_audio": handle_probe_audio,
    "asr_preflight_select": lambda params: _run_asr_handler(handle_asr_preflight_select, params),
    "environment": handle_environment,
    "transcribe": lambda params: _run_asr_handler(handle_transcribe, params),
    "correct": lambda params: _run_correction_handler(handle_correct, params),
    "polish": handle_polish,
    "translate_article": handle_translate_article,
    "correct_pause": handle_correct_pause,
    "correct_resume": handle_correct_resume,
    "correct_cancel": handle_correct_cancel,
    "correct_status": handle_correct_status,
    "diarize": handle_diarize,
    "recommend_diarization": handle_recommend_diarization,
    "extract_voice_embedding": handle_extract_voice_embedding,
    "preflight_voiceprint_anchors": handle_preflight_voiceprint_anchors,
    "reidentify_speakers": handle_reidentify_speakers,
}

# Methods that must run on the main thread (so they can interrupt long-running ops).
CONTROL_METHODS = {
    "correct_pause",
    "correct_resume",
    "correct_cancel",
    "correct_status",
}

ASR_METHODS = {"asr_preflight_select", "preflight_voiceprint_anchors", "transcribe"}
CORRECTION_METHODS = {"correct"}


# ---- main loop ----

def _dispatch(rid: Any, method: str, params: dict) -> None:
    """Execute a handler and emit response. Catches all exceptions."""
    if method not in HANDLERS:
        _emit({"id": rid, "error": {"code": -32601, "message": f"Method not found: {method}"}})
        return
    try:
        result = HANDLERS[method](params)
        _emit({"id": rid, "result": result})
    except Exception as e:  # noqa: BLE001
        _emit({
            "id": rid,
            "error": {
                "code": -32000,
                "message": str(e),
                "data": {"traceback": traceback.format_exc()},
            },
        })


def run() -> None:
    """Reader thread + worker pool. See module docstring."""
    request_q: queue.Queue = queue.Queue()
    stop_flag = threading.Event()

    def reader() -> None:
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except json.JSONDecodeError as e:
                    _emit({"error": {"code": -32700, "message": f"Parse error: {e}"}})
                    continue
                request_q.put(req)
        finally:
            stop_flag.set()
            # Wake up the main loop with a sentinel so it can shut down.
            request_q.put(None)

    threading.Thread(target=reader, name="ipc-reader", daemon=True).start()

    general_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ipc-worker")
    asr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ipc-asr")
    correction_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ipc-correction")

    try:
        while not stop_flag.is_set() or not request_q.empty():
            try:
                req = request_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if req is None:  # sentinel from reader on stdin EOF
                break

            rid = req.get("id")
            method = req.get("method")
            params = req.get("params", {}) or {}

            if method in CONTROL_METHODS:
                # Run synchronously on the main thread for instant response.
                _dispatch(rid, method, params)
            elif method in ASR_METHODS:
                asr_executor.submit(_dispatch, rid, method, params)
            elif method in CORRECTION_METHODS:
                correction_executor.submit(_dispatch, rid, method, params)
            else:
                # Dispatch to a worker so we keep reading control commands.
                general_executor.submit(_dispatch, rid, method, params)
    finally:
        asr_executor.shutdown(wait=False, cancel_futures=True)
        correction_executor.shutdown(wait=False, cancel_futures=True)
        general_executor.shutdown(wait=False, cancel_futures=True)
