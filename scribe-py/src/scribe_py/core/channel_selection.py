"""Read-only stereo channel quality evaluation.

This module is intentionally isolated from the active ASR and diarization
pipelines.  It reads an audio stream, evaluates whether one stereo channel can
replace the union without losing speech, and returns a JSON-serializable report.
It never creates a converted audio file or changes source timestamps.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class ChannelDecision(str, Enum):
    MIX = "mix"
    LEFT = "left"
    RIGHT = "right"


_SCHEMA_VERSION = 1
_METHOD = "stereo_energy_v1"
_ANALYSIS_SAMPLE_RATE = 8000
_FRAME_MS = 100
_MIN_UNION_RECALL = 0.98
_MAX_OTHER_ONLY_RATIO = 0.02
_MIN_QUALITY_MARGIN_DB = 4.0
_ANTI_PHASE_CORRELATION = -0.75
_MIN_ANTI_PHASE_OVERLAP = 0.80
_MAX_DURATION_RELATIVE_DELTA = 0.002
_MAX_DURATION_DELTA_S = 5.0


def _empty_channel_metrics() -> dict[str, Any]:
    return {
        "rms_dbfs": None,
        "noise_floor_dbfs": None,
        "speech_level_dbfs": None,
        "estimated_snr_db": None,
        "speech_threshold_dbfs": None,
        "speech_coverage": 0.0,
        "clipping_ratio": 0.0,
        "quality_score_db": None,
    }


def _base_report(source: Path) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "method": _METHOD,
        "status": "fallback",
        "decision": ChannelDecision.MIX.value,
        "reason": "not_evaluated",
        "error": "",
        "source": str(source),
        "read_only": True,
        "preserves_timing": True,
        "channels": 0,
        "channel_layout": "",
        "sample_rate": 0,
        "duration_s": 0.0,
        "decoded_duration_s": 0.0,
        "duration_delta_s": 0.0,
        "duration_unchanged": True,
        "analysis_sample_rate": _ANALYSIS_SAMPLE_RATE,
        "frame_ms": _FRAME_MS,
        "thresholds": {
            "min_union_recall": _MIN_UNION_RECALL,
            "max_other_only_ratio": _MAX_OTHER_ONLY_RATIO,
            "min_quality_margin_db": _MIN_QUALITY_MARGIN_DB,
            "anti_phase_correlation": _ANTI_PHASE_CORRELATION,
            "min_anti_phase_overlap": _MIN_ANTI_PHASE_OVERLAP,
            "max_duration_relative_delta": _MAX_DURATION_RELATIVE_DELTA,
            "max_duration_delta_s": _MAX_DURATION_DELTA_S,
        },
        "left": _empty_channel_metrics(),
        "right": _empty_channel_metrics(),
        "union_speech_coverage": 0.0,
        "left_union_recall": 0.0,
        "right_union_recall": 0.0,
        "left_only_speech_ratio": 0.0,
        "right_only_speech_ratio": 0.0,
        "complementary_speech_ratio": 0.0,
        "speech_disagreement_ratio": 0.0,
        "overlap_speech_ratio": 0.0,
        "channel_correlation": None,
        "anti_phase_risk": False,
        "quality_margin_db": 0.0,
    }


def _fallback(report: dict[str, Any], reason: str, error: str = "") -> dict[str, Any]:
    report["status"] = "fallback"
    report["decision"] = ChannelDecision.MIX.value
    report["reason"] = reason
    report["error"] = error
    return report


def _probe_audio(path: Path, ffprobe: str, timeout: float) -> dict[str, Any]:
    process = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,channel_layout,duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if process.returncode != 0:
        raise RuntimeError((process.stderr or f"ffprobe exited {process.returncode}").strip())
    payload = json.loads(process.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe found no audio stream")
    stream = streams[0]
    duration = stream.get("duration") or (payload.get("format") or {}).get("duration") or 0.0
    return {
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "channel_layout": str(stream.get("channel_layout") or ""),
        "duration_s": float(duration or 0.0),
    }


class _StereoAccumulator:
    def __init__(self, sample_rate: int, frame_ms: int) -> None:
        self.sample_rate = sample_rate
        self.frame_size = max(1, int(round(sample_rate * frame_ms / 1000.0)))
        self.pending = np.empty((0, 2), dtype=np.float32)
        self.frame_rms: list[list[float]] = [[], []]
        self.total_frames = 0
        self.clipped = np.zeros(2, dtype=np.int64)
        self.sample_sum = np.zeros(2, dtype=np.float64)
        self.sample_square_sum = np.zeros(2, dtype=np.float64)
        self.cross_sum = 0.0

    def add_pcm16(self, raw: bytes) -> None:
        usable = len(raw) - (len(raw) % 4)
        if usable <= 0:
            return
        values = np.frombuffer(raw[:usable], dtype="<i2").reshape(-1, 2)
        normalized = values.astype(np.float32) / 32768.0
        self.total_frames += int(normalized.shape[0])
        self.clipped += np.sum(np.abs(values.astype(np.int32)) >= 32760, axis=0)
        samples64 = normalized.astype(np.float64)
        self.sample_sum += np.sum(samples64, axis=0)
        self.sample_square_sum += np.sum(samples64 * samples64, axis=0)
        self.cross_sum += float(np.sum(samples64[:, 0] * samples64[:, 1]))

        combined = np.concatenate((self.pending, normalized), axis=0)
        complete = combined.shape[0] // self.frame_size
        if complete:
            framed = combined[: complete * self.frame_size].reshape(complete, self.frame_size, 2)
            rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
            self.frame_rms[0].extend(rms[:, 0].astype(float).tolist())
            self.frame_rms[1].extend(rms[:, 1].astype(float).tolist())
        self.pending = combined[complete * self.frame_size :].copy()

    def finish(self) -> None:
        if self.pending.shape[0] >= max(1, self.frame_size // 2):
            rms = np.sqrt(np.mean(self.pending * self.pending, axis=0) + 1e-12)
            self.frame_rms[0].append(float(rms[0]))
            self.frame_rms[1].append(float(rms[1]))
        self.pending = np.empty((0, 2), dtype=np.float32)

    def correlation(self) -> float | None:
        count = self.total_frames
        if count <= 0:
            return None
        covariance = self.cross_sum - (self.sample_sum[0] * self.sample_sum[1] / count)
        variance_left = self.sample_square_sum[0] - (self.sample_sum[0] ** 2 / count)
        variance_right = self.sample_square_sum[1] - (self.sample_sum[1] ** 2 / count)
        denominator = math.sqrt(max(variance_left, 0.0) * max(variance_right, 0.0))
        if denominator <= 1e-12:
            return None
        return max(-1.0, min(1.0, covariance / denominator))


def _decode_stereo(path: Path, ffmpeg: str, timeout: float) -> _StereoAccumulator:
    process = subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(_ANALYSIS_SAMPLE_RATE),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    timed_out = threading.Event()

    def terminate_on_timeout() -> None:
        timed_out.set()
        process.kill()

    timer = threading.Timer(timeout, terminate_on_timeout)
    timer.daemon = True
    timer.start()
    accumulator = _StereoAccumulator(_ANALYSIS_SAMPLE_RATE, _FRAME_MS)
    remainder = b""
    try:
        if process.stdout is None:
            raise RuntimeError("ffmpeg stdout pipe is unavailable")
        while True:
            chunk = process.stdout.read(64 * 1024)
            if not chunk:
                break
            chunk = remainder + chunk
            usable = len(chunk) - (len(chunk) % 4)
            accumulator.add_pcm16(chunk[:usable])
            remainder = chunk[usable:]
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        returncode = process.wait()
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
            process.wait()
    if timed_out.is_set():
        raise TimeoutError(f"ffmpeg channel analysis timed out after {timeout:.1f}s")
    if returncode != 0:
        raise RuntimeError(stderr.strip() or f"ffmpeg exited {returncode}")
    accumulator.finish()
    if accumulator.total_frames <= 0 or not accumulator.frame_rms[0]:
        raise RuntimeError("ffmpeg decoded no stereo samples")
    return accumulator


def _round_number(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _decoded_duration_matches(source_duration: float, decoded_duration: float) -> bool:
    if source_duration <= 0:
        return True
    # MP3/VBR container duration is commonly an estimate. Accept bounded drift,
    # but keep an absolute cap so a genuinely truncated decode cannot pass.
    tolerance = min(
        _MAX_DURATION_DELTA_S,
        max(0.1, source_duration * _MAX_DURATION_RELATIVE_DELTA),
    )
    return abs(decoded_duration - source_duration) <= tolerance


def _channel_metrics(
    frame_rms: list[float],
    *,
    clipped_samples: int,
    total_samples: int,
) -> tuple[dict[str, Any], np.ndarray]:
    rms = np.asarray(frame_rms, dtype=np.float64)
    db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    overall_rms = float(np.sqrt(np.mean(rms * rms)))
    noise_floor = float(np.percentile(db, 15))
    speech_level = float(np.percentile(db, 85))
    snr = max(0.0, min(60.0, speech_level - noise_floor))

    # A flat signal has no evidence that energy changes represent speech.
    if snr < 3.0:
        speech_threshold = speech_level + 3.0
        speech_mask = np.zeros(db.shape, dtype=bool)
    else:
        speech_threshold = noise_floor + min(8.0, max(4.0, snr * 0.5))
        speech_mask = db >= speech_threshold
    clipping_ratio = clipped_samples / max(total_samples, 1)
    quality_score = snr - min(20.0, clipping_ratio * 1000.0)
    metrics = {
        "rms_dbfs": _round_number(20.0 * math.log10(max(overall_rms, 1e-6)), 3),
        "noise_floor_dbfs": _round_number(noise_floor, 3),
        "speech_level_dbfs": _round_number(speech_level, 3),
        "estimated_snr_db": _round_number(snr, 3),
        "speech_threshold_dbfs": _round_number(speech_threshold, 3),
        "speech_coverage": _round_number(float(np.mean(speech_mask)), 6),
        "clipping_ratio": _round_number(clipping_ratio, 8),
        "quality_score_db": _round_number(quality_score, 3),
    }
    return metrics, speech_mask


def _apply_decision(report: dict[str, Any], accumulator: _StereoAccumulator) -> dict[str, Any]:
    left, left_speech = _channel_metrics(
        accumulator.frame_rms[0],
        clipped_samples=int(accumulator.clipped[0]),
        total_samples=accumulator.total_frames,
    )
    right, right_speech = _channel_metrics(
        accumulator.frame_rms[1],
        clipped_samples=int(accumulator.clipped[1]),
        total_samples=accumulator.total_frames,
    )
    frame_count = min(left_speech.size, right_speech.size)
    if frame_count <= 0:
        return _fallback(report, "insufficient_audio", "no complete analysis frames")
    left_speech = left_speech[:frame_count]
    right_speech = right_speech[:frame_count]
    union = left_speech | right_speech
    union_count = int(np.sum(union))
    report["left"] = left
    report["right"] = right
    if union_count <= 0:
        report["status"] = "ok"
        report["reason"] = "no_speech_evidence"
        return report

    left_only = int(np.sum(left_speech & ~right_speech))
    right_only = int(np.sum(right_speech & ~left_speech))
    overlap = int(np.sum(left_speech & right_speech))
    left_recall = float(np.sum(left_speech & union)) / union_count
    right_recall = float(np.sum(right_speech & union)) / union_count
    left_only_ratio = left_only / union_count
    right_only_ratio = right_only / union_count
    overlap_ratio = overlap / union_count
    correlation = accumulator.correlation()
    anti_phase = bool(
        correlation is not None
        and correlation <= _ANTI_PHASE_CORRELATION
        and overlap_ratio >= _MIN_ANTI_PHASE_OVERLAP
    )
    left_score = float(left["quality_score_db"] or 0.0)
    right_score = float(right["quality_score_db"] or 0.0)
    quality_margin = left_score - right_score

    report.update({
        "status": "ok",
        "union_speech_coverage": _round_number(union_count / frame_count, 6),
        "left_union_recall": _round_number(left_recall, 6),
        "right_union_recall": _round_number(right_recall, 6),
        "left_only_speech_ratio": _round_number(left_only_ratio, 6),
        "right_only_speech_ratio": _round_number(right_only_ratio, 6),
        "complementary_speech_ratio": _round_number(min(left_only_ratio, right_only_ratio), 6),
        "speech_disagreement_ratio": _round_number(left_only_ratio + right_only_ratio, 6),
        "overlap_speech_ratio": _round_number(overlap_ratio, 6),
        "channel_correlation": _round_number(correlation, 6),
        "anti_phase_risk": anti_phase,
        "quality_margin_db": _round_number(quality_margin, 3),
    })

    left_safe = left_recall >= _MIN_UNION_RECALL and right_only_ratio <= _MAX_OTHER_ONLY_RATIO
    right_safe = right_recall >= _MIN_UNION_RECALL and left_only_ratio <= _MAX_OTHER_ONLY_RATIO
    if left_safe and quality_margin >= _MIN_QUALITY_MARGIN_DB:
        report["decision"] = ChannelDecision.LEFT.value
        report["reason"] = "left_clearly_better"
    elif right_safe and quality_margin <= -_MIN_QUALITY_MARGIN_DB:
        report["decision"] = ChannelDecision.RIGHT.value
        report["reason"] = "right_clearly_better"
    elif not left_safe and not right_safe:
        report["reason"] = "complementary_speech"
    elif anti_phase:
        report["reason"] = "anti_phase_risk_quality_margin_insufficient"
    else:
        report["reason"] = "quality_margin_insufficient"
    return report


def evaluate_stereo_channel_selection(
    audio: Path | str,
    *,
    timeout: float = 60.0,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
) -> dict[str, Any]:
    """Return a conservative, JSON-stable channel decision without writing audio.

    A side is selected only when it retains at least 98% of union speech, the
    opposite side contributes no material unique speech, and the selected side's
    quality score is at least 4 dB better.  Every failure falls back to ``mix``.
    """
    source = Path(audio).expanduser()
    report = _base_report(source)
    if not source.is_file():
        return _fallback(report, "source_missing", f"audio file does not exist: {source}")

    ffprobe_bin = ffprobe or shutil.which("ffprobe")
    if not ffprobe_bin:
        return _fallback(report, "ffprobe_unavailable", "ffprobe not found")
    ffmpeg_bin = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return _fallback(report, "ffmpeg_unavailable", "ffmpeg not found")

    try:
        probe = _probe_audio(source, ffprobe_bin, timeout)
    except Exception as exc:  # noqa: BLE001
        return _fallback(report, "ffprobe_failed", f"{type(exc).__name__}: {exc}")
    report.update(probe)
    if int(probe.get("channels") or 0) != 2:
        report["status"] = "ok"
        report["reason"] = "not_stereo"
        return report

    try:
        accumulator = _decode_stereo(source, ffmpeg_bin, timeout)
    except Exception as exc:  # noqa: BLE001
        return _fallback(report, "ffmpeg_failed", f"{type(exc).__name__}: {exc}")

    decoded_duration = accumulator.total_frames / _ANALYSIS_SAMPLE_RATE
    source_duration = float(report.get("duration_s") or 0.0)
    duration_delta = decoded_duration - source_duration if source_duration > 0 else 0.0
    report["decoded_duration_s"] = round(decoded_duration, 6)
    report["duration_delta_s"] = round(duration_delta, 6)
    report["duration_unchanged"] = _decoded_duration_matches(
        source_duration,
        decoded_duration,
    )
    return _apply_decision(report, accumulator)


__all__ = ["ChannelDecision", "evaluate_stereo_channel_selection"]
