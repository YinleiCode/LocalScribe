from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from scribe_py.core import channel_selection
from scribe_py.core.channel_selection import ChannelDecision, evaluate_stereo_channel_selection


SR = 16000
DURATION = 4.0
EXPECTED_REPORT_KEYS = {
    "schema_version",
    "method",
    "status",
    "decision",
    "reason",
    "error",
    "source",
    "read_only",
    "preserves_timing",
    "channels",
    "channel_layout",
    "sample_rate",
    "duration_s",
    "decoded_duration_s",
    "duration_delta_s",
    "duration_unchanged",
    "analysis_sample_rate",
    "frame_ms",
    "thresholds",
    "left",
    "right",
    "union_speech_coverage",
    "left_union_recall",
    "right_union_recall",
    "left_only_speech_ratio",
    "right_only_speech_ratio",
    "complementary_speech_ratio",
    "speech_disagreement_ratio",
    "overlap_speech_ratio",
    "channel_correlation",
    "anti_phase_risk",
    "quality_margin_db",
}


def _tone(mask: np.ndarray, frequency: float = 220.0) -> np.ndarray:
    t = np.arange(mask.size, dtype=np.float64) / SR
    voice = 0.30 * np.sin(2.0 * math.pi * frequency * t)
    voice += 0.10 * np.sin(2.0 * math.pi * frequency * 2.0 * t)
    return voice * mask


def _speech_mask(*intervals: tuple[float, float]) -> np.ndarray:
    t = np.arange(int(SR * DURATION), dtype=np.float64) / SR
    mask = np.zeros(t.shape, dtype=np.float64)
    for start, end in intervals:
        mask[(t >= start) & (t < end)] = 1.0
    return mask


def _write_stereo(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    stereo = np.column_stack((left, right))
    pcm = np.clip(stereo, -0.999, 0.999)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def _noise(seed: int, amplitude: float, size: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, amplitude, size=size)


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required for channel integration tests")


def _assert_stable_json_schema(report: dict) -> None:
    assert set(report) == EXPECTED_REPORT_KEYS
    assert set(report["left"]) == set(report["right"])
    json.dumps(report, sort_keys=True)


def test_left_clean_right_noisy_selects_left(tmp_path: Path):
    _require_ffmpeg()
    mask = _speech_mask((0.5, 1.5), (2.0, 3.5))
    voice = _tone(mask)
    left = voice + _noise(1, 0.0008, voice.size)
    right = voice + _noise(2, 0.075, voice.size)
    audio = tmp_path / "left-clean.wav"
    _write_stereo(audio, left, right)

    report = evaluate_stereo_channel_selection(audio)

    assert report["decision"] == ChannelDecision.LEFT.value
    assert report["left_union_recall"] >= 0.98
    assert report["right_only_speech_ratio"] <= 0.02
    assert report["quality_margin_db"] >= 4.0


def test_right_clean_left_noisy_selects_right(tmp_path: Path):
    _require_ffmpeg()
    mask = _speech_mask((0.5, 1.5), (2.0, 3.5))
    voice = _tone(mask)
    left = voice + _noise(3, 0.075, voice.size)
    right = voice + _noise(4, 0.0008, voice.size)
    audio = tmp_path / "right-clean.wav"
    _write_stereo(audio, left, right)

    report = evaluate_stereo_channel_selection(audio)

    assert report["decision"] == ChannelDecision.RIGHT.value
    assert report["right_union_recall"] >= 0.98
    assert report["left_only_speech_ratio"] <= 0.02
    assert report["quality_margin_db"] <= -4.0


def test_dual_mono_keeps_mix(tmp_path: Path):
    _require_ffmpeg()
    mask = _speech_mask((0.5, 1.5), (2.0, 3.5))
    voice = _tone(mask) + _noise(5, 0.0008, mask.size)
    audio = tmp_path / "dual-mono.wav"
    _write_stereo(audio, voice, voice)

    report = evaluate_stereo_channel_selection(audio)

    assert report["decision"] == ChannelDecision.MIX.value
    assert report["reason"] == "quality_margin_insufficient"
    assert report["channel_correlation"] == pytest.approx(1.0, abs=1e-5)


def test_independent_left_and_right_speech_keeps_mix(tmp_path: Path):
    _require_ffmpeg()
    left_mask = _speech_mask((0.2, 1.8))
    right_mask = _speech_mask((2.2, 3.8))
    left = _tone(left_mask, 210.0) + _noise(6, 0.0008, left_mask.size)
    right = _tone(right_mask, 330.0) + _noise(7, 0.0008, right_mask.size)
    audio = tmp_path / "independent.wav"
    _write_stereo(audio, left, right)

    report = evaluate_stereo_channel_selection(audio)

    assert report["decision"] == ChannelDecision.MIX.value
    assert report["reason"] == "complementary_speech"
    assert report["left_union_recall"] < 0.98
    assert report["right_union_recall"] < 0.98
    assert report["complementary_speech_ratio"] > 0.20


def test_inverted_channels_report_anti_phase_without_forcing_side(tmp_path: Path):
    _require_ffmpeg()
    mask = _speech_mask((0.5, 1.5), (2.0, 3.5))
    voice = _tone(mask)
    audio = tmp_path / "anti-phase.wav"
    _write_stereo(audio, voice, -voice)

    report = evaluate_stereo_channel_selection(audio)

    assert report["decision"] == ChannelDecision.MIX.value
    assert report["anti_phase_risk"] is True
    assert report["channel_correlation"] <= -0.99
    assert report["reason"] == "anti_phase_risk_quality_margin_insufficient"


@pytest.mark.parametrize(
    ("missing_tool", "expected_reason"),
    [("ffprobe", "ffprobe_unavailable"), ("ffmpeg", "ffmpeg_unavailable")],
)
def test_ffmpeg_or_ffprobe_failure_falls_back_to_mix(
    monkeypatch,
    tmp_path: Path,
    missing_tool: str,
    expected_reason: str,
):
    audio = tmp_path / "exists.wav"
    audio.write_bytes(b"not-a-real-wave")
    monkeypatch.setattr(
        channel_selection.shutil,
        "which",
        lambda name: None if name == missing_tool else f"/unused/{name}",
    )

    report = evaluate_stereo_channel_selection(audio)

    assert report["decision"] == ChannelDecision.MIX.value
    assert report["status"] == "fallback"
    assert report["reason"] == expected_reason
    assert report["preserves_timing"] is True
    _assert_stable_json_schema(report)


def test_evaluation_is_read_only_and_duration_is_unchanged(tmp_path: Path):
    _require_ffmpeg()
    mask = _speech_mask((0.5, 3.5))
    voice = _tone(mask)
    audio = tmp_path / "duration.wav"
    _write_stereo(audio, voice, voice)
    before = hashlib.sha256(audio.read_bytes()).hexdigest()
    before_stat = audio.stat()

    report = evaluate_stereo_channel_selection(audio)

    after = hashlib.sha256(audio.read_bytes()).hexdigest()
    after_stat = audio.stat()
    assert before == after
    assert before_stat.st_size == after_stat.st_size
    assert before_stat.st_mtime_ns == after_stat.st_mtime_ns
    assert report["read_only"] is True
    assert report["preserves_timing"] is True
    assert report["duration_unchanged"] is True
    assert report["duration_s"] == pytest.approx(DURATION, abs=0.01)
    assert report["decoded_duration_s"] == pytest.approx(DURATION, abs=0.01)
    _assert_stable_json_schema(report)


@pytest.mark.parametrize(
    ("source_duration", "decoded_duration"),
    [
        (360.0123, 359.376),
        (2598.902375, 2594.328),
    ],
)
def test_vbr_container_duration_drift_is_tolerated(
    source_duration: float,
    decoded_duration: float,
):
    assert channel_selection._decoded_duration_matches(source_duration, decoded_duration) is True


@pytest.mark.parametrize(
    ("source_duration", "decoded_duration"),
    [
        (360.0, 358.5),
        (4000.0, 3994.8),
    ],
)
def test_truncated_decode_duration_is_rejected(
    source_duration: float,
    decoded_duration: float,
):
    assert channel_selection._decoded_duration_matches(source_duration, decoded_duration) is False
