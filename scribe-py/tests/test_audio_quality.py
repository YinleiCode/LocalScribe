from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from scribe_py.core import audio


def test_analyze_audio_quality_scores_low_loudness_and_silence(monkeypatch, tmp_path: Path):
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"fake")

    monkeypatch.setattr(audio, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        audio,
        "probe_audio",
        lambda _: {
            "duration": 10.0,
            "sample_rate": 8000,
            "channels": 2,
        },
    )

    stderr = """
    [Parsed_ebur128_0 @ 0x1] Summary:
      I:         -35.2 LUFS
      Peak:       -0.1 dBFS
    [silencedetect @ 0x2] silence_duration: 5.2
    """

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr=stderr)

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    report = audio.analyze_audio_quality_for_asr(sample)

    assert report["risk_level"] == "high"
    assert report["sample_rate"] == 8000
    assert report["channels"] == 2
    assert report["integrated_lufs"] == -35.2
    assert report["true_peak_dbfs"] == -0.1
    assert report["silence_ratio"] == 0.52
    assert "整体音量过低" in report["risk_reasons"]
    assert "静音占比过高" in report["risk_reasons"]


def test_adaptive_standardization_applies_loudnorm_for_quiet_audio(monkeypatch, tmp_path: Path):
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"fake")
    calls = []

    monkeypatch.setattr(audio, "find_ffmpeg", lambda: "ffmpeg")

    def fake_run(args, **kwargs):
        calls.append(args)
        out = Path(args[-1])
        out.write_bytes(b"RIFF" + b"0" * 128)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    out, stats = audio.standardize_audio_for_asr(
        sample,
        tmp_path,
        audio_quality={"integrated_lufs": -30.2, "true_peak_dbfs": -8.0, "silence_ratio": 0.05},
        mode="adaptive",
    )

    assert out.name == "asr_input_16k_mono.wav"
    assert stats["applied"] is True
    assert stats["mode"] == "adaptive"
    assert "loudness_normalization" in stats["applied_filters"]
    assert "loudnorm=I=-23:TP=-2:LRA=11" in stats["audio_filter"]
    assert any("-af" in call and "loudnorm=I=-23:TP=-2:LRA=11" in call for call in calls)


def test_analyze_audio_quality_estimates_noise_floor_and_snr(monkeypatch, tmp_path: Path):
    sample = tmp_path / "noisy.wav"
    sample.write_bytes(b"fake")

    monkeypatch.setattr(audio, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        audio,
        "probe_audio",
        lambda _: {
            "duration": 20.0,
            "sample_rate": 16000,
            "channels": 1,
        },
    )

    stderr = """
    [Parsed_ebur128_0 @ 0x1] Summary:
      I:         -22.0 LUFS
      Peak:       -4.0 dBFS
    lavfi.astats.Overall.RMS_level=-34.0
    lavfi.astats.Overall.RMS_level=-33.5
    lavfi.astats.Overall.RMS_level=-32.0
    lavfi.astats.Overall.RMS_level=-29.0
    lavfi.astats.Overall.RMS_level=-24.0
    lavfi.astats.Overall.RMS_level=-21.0
    """

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr=stderr)

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    report = audio.analyze_audio_quality_for_asr(sample)

    assert report["risk_level"] == "high"
    assert report["noise_floor_dbfs"] is not None
    assert report["speech_level_dbfs"] is not None
    assert report["estimated_snr_db"] < 16
    assert "信噪比偏低" in report["risk_reasons"] or "背景噪声明显" in report["risk_reasons"]


def test_adaptive_standardization_keeps_fidelity_for_noisy_audio(monkeypatch, tmp_path: Path):
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"fake")
    calls = []

    monkeypatch.setattr(audio, "find_ffmpeg", lambda: "ffmpeg")

    def fake_run(args, **kwargs):
        calls.append(args)
        out = Path(args[-1])
        out.write_bytes(b"RIFF" + b"0" * 128)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    _, stats = audio.standardize_audio_for_asr(
        sample,
        tmp_path,
        audio_quality={
            "integrated_lufs": -21.0,
            "true_peak_dbfs": -6.0,
            "noise_floor_dbfs": -34.0,
            "estimated_snr_db": 12.0,
            "risk_reasons": ["信噪比偏低", "背景噪声偏高"],
        },
        mode="adaptive",
    )

    assert stats["applied"] is True
    assert "adaptive_noise_reduction" not in stats["applied_filters"]
    assert "speech_bandpass" not in stats["applied_filters"]
    assert "afftdn" not in stats["audio_filter"]
    assert "loudness_normalization" not in stats["applied_filters"]
    assert not any("-af" in call for call in calls)
    assert any("adaptive 保持保真不自动降噪" in action for action in stats["skipped_actions"])


def test_standardization_selects_only_a_prevalidated_better_stereo_channel(
    monkeypatch, tmp_path: Path
):
    sample = tmp_path / "stereo.wav"
    sample.write_bytes(b"fake")
    calls = []
    monkeypatch.setattr(audio, "find_ffmpeg", lambda: "ffmpeg")

    def fake_run(args, **kwargs):
        calls.append(args)
        Path(args[-1]).write_bytes(b"RIFF" + b"0" * 128)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    report = {
        "status": "ok",
        "decision": "right",
        "channels": 2,
        "preserves_timing": True,
        "duration_unchanged": True,
    }

    _, stats = audio.standardize_audio_for_asr(
        sample,
        tmp_path,
        audio_quality={"integrated_lufs": -20.0},
        channel_selection=report,
        mode="adaptive",
    )

    assert stats["channel_decision"] == "right"
    assert "select_right_channel" in stats["applied_filters"]
    assert "downmix_mono" not in stats["applied_filters"]
    assert stats["audio_filter"] == "pan=mono|c0=c1"
    assert any("pan=mono|c0=c1" in call for call in calls)


def test_standardization_keeps_mix_when_channel_report_is_not_safe(
    monkeypatch, tmp_path: Path
):
    sample = tmp_path / "stereo.wav"
    sample.write_bytes(b"fake")
    calls = []
    monkeypatch.setattr(audio, "find_ffmpeg", lambda: "ffmpeg")

    def fake_run(args, **kwargs):
        calls.append(args)
        Path(args[-1]).write_bytes(b"RIFF" + b"0" * 128)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    report = {
        "status": "ok",
        "decision": "left",
        "channels": 2,
        "preserves_timing": True,
        "duration_unchanged": False,
    }

    _, stats = audio.standardize_audio_for_asr(
        sample,
        tmp_path,
        audio_quality={"integrated_lufs": -20.0},
        channel_selection=report,
        mode="adaptive",
    )

    assert stats["channel_decision"] == "mix"
    assert "downmix_mono" in stats["applied_filters"]
    assert stats["audio_filter"] == ""
    assert not any("-af" in call for call in calls)


def test_adaptive_does_not_amplify_quiet_noisy_audio(monkeypatch, tmp_path: Path):
    sample = tmp_path / "quiet-noisy.wav"
    sample.write_bytes(b"fake")
    calls = []

    monkeypatch.setattr(audio, "find_ffmpeg", lambda: "ffmpeg")

    def fake_run(args, **kwargs):
        calls.append(args)
        out = Path(args[-1])
        out.write_bytes(b"RIFF" + b"0" * 128)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    _, stats = audio.standardize_audio_for_asr(
        sample,
        tmp_path,
        audio_quality={
            "integrated_lufs": -32.4,
            "true_peak_dbfs": -0.3,
            "noise_floor_dbfs": -32.61,
            "estimated_snr_db": 2.08,
            "risk_reasons": ["整体音量过低", "信噪比过低", "背景噪声明显"],
        },
        mode="adaptive",
    )

    assert "loudness_normalization" not in stats["applied_filters"]
    assert "loudnorm" not in stats["audio_filter"]
    assert not any("-af" in call for call in calls)
    assert any("不做响度均衡以避免放大噪声" in action for action in stats["skipped_actions"])


def test_ai_denoise_standardization_uses_deepfilter_then_16k(monkeypatch, tmp_path: Path):
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"fake")
    calls = []

    monkeypatch.setattr(audio, "find_ffmpeg", lambda: "ffmpeg")

    def fake_run(args, **kwargs):
        calls.append(args)
        out = Path(args[-1])
        out.write_bytes(b"RIFF" + b"0" * 128)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    def fake_deepfilter(src: Path, out: Path, **kwargs):
        assert src.name == "ai_denoise_input_48k_mono.wav"
        out.write_bytes(b"RIFF" + b"1" * 128)
        return {"engine": "deepfilternet", "available": True, "applied": True, "input": str(src), "output": str(out), "method": "test"}

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    monkeypatch.setattr(audio, "_run_deepfilter_enhance", fake_deepfilter)

    out, stats = audio.standardize_audio_for_asr(
        sample,
        tmp_path,
        audio_quality={"integrated_lufs": -20.0, "true_peak_dbfs": -6.0, "silence_ratio": 0.05},
        mode="ai_denoise",
    )

    assert out.name == "asr_input_16k_mono.wav"
    assert stats["mode"] == "ai_denoise"
    assert stats["ai_denoise"]["applied"] is True
    assert "deepfilternet_ai_denoise" in stats["applied_filters"]
    assert any("-ar" in call and "48000" in call for call in calls)
    assert any("-ar" in call and "16000" in call for call in calls)


def test_ai_denoise_falls_back_to_adaptive_when_deepfilter_missing(monkeypatch, tmp_path: Path):
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"fake")

    monkeypatch.setattr(audio, "find_ffmpeg", lambda: "ffmpeg")

    def fake_run(args, **kwargs):
        out = Path(args[-1])
        out.write_bytes(b"RIFF" + b"0" * 128)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    def fake_deepfilter(src: Path, out: Path, **kwargs):
        return {"engine": "deepfilternet", "available": False, "applied": False, "input": str(src), "output": str(out), "error": "missing"}

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    monkeypatch.setattr(audio, "_run_deepfilter_enhance", fake_deepfilter)

    _, stats = audio.standardize_audio_for_asr(
        sample,
        tmp_path,
        audio_quality={
            "integrated_lufs": -21.0,
            "noise_floor_dbfs": -34.0,
            "estimated_snr_db": 12.0,
            "risk_reasons": ["背景噪声偏高"],
        },
        mode="ai_denoise",
    )

    assert stats["mode"] == "adaptive"
    assert stats["fallback_applied"] is True
    assert stats["fallback_mode"] == "adaptive"
    assert "adaptive_noise_reduction" not in stats["applied_filters"]
    assert "AI 降噪不可用或失败, 已回退为 adaptive 安全基线" in stats["skipped_actions"]


def test_ai_denoise_fallback_keeps_prevalidated_channel_selection(
    monkeypatch, tmp_path: Path
):
    sample = tmp_path / "stereo.wav"
    sample.write_bytes(b"fake")
    calls = []
    monkeypatch.setattr(audio, "find_ffmpeg", lambda: "ffmpeg")

    def fake_run(args, **kwargs):
        calls.append(args)
        out = Path(args[-1])
        out.write_bytes(b"RIFF" + b"0" * 128)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    monkeypatch.setattr(
        audio,
        "_run_deepfilter_enhance",
        lambda *_args, **_kwargs: {
            "engine": "deepfilternet",
            "available": False,
            "applied": False,
            "error": "missing",
        },
    )

    _, stats = audio.standardize_audio_for_asr(
        sample,
        tmp_path,
        audio_quality={"integrated_lufs": -20.0},
        channel_selection={
            "status": "ok",
            "decision": "right",
            "channels": 2,
            "preserves_timing": True,
            "duration_unchanged": True,
        },
        mode="ai_denoise",
    )

    assert stats["mode"] == "adaptive"
    assert stats["channel_decision"] == "right"
    assert "select_right_channel" in stats["applied_filters"]
    assert "downmix_mono" not in stats["applied_filters"]
    assert stats["audio_filter"] == "pan=mono|c0=c1"
    assert sum("pan=mono|c0=c1" in call for call in calls) == 2


def test_deepfilter_model_resolves_from_packaged_resources(monkeypatch, tmp_path: Path):
    packaged = tmp_path / "Resources" / "deepfilternet" / "DeepFilterNet3"
    (packaged / "checkpoints").mkdir(parents=True)
    (packaged / "config.ini").write_text("[train]\n", encoding="utf-8")
    (packaged / "checkpoints" / "model_120.ckpt.best").write_bytes(b"checkpoint")

    monkeypatch.setenv("LOCALSCRIBE_RESOURCES", str(tmp_path / "Resources"))
    monkeypatch.delenv("LOCALSCRIBE_DEEPFILTER_MODEL_DIR", raising=False)

    assert audio.resolve_deepfilter_model_dir() == packaged


def test_deepfilter_missing_model_returns_unavailable(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(audio, "_deepfilter_model_candidates", lambda model_name="DeepFilterNet3": [tmp_path / "missing"])

    stats = audio._run_deepfilter_enhance(tmp_path / "in.wav", tmp_path / "out.wav")

    assert stats["available"] is False
    assert stats["applied"] is False
    assert "model not found" in stats["error"]


def test_standardization_preserves_timing_and_does_not_remove_silence(monkeypatch, tmp_path: Path):
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"fake")

    monkeypatch.setattr(audio, "find_ffmpeg", lambda: "ffmpeg")

    def fake_run(args, **kwargs):
        out = Path(args[-1])
        out.write_bytes(b"RIFF" + b"0" * 128)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    _, stats = audio.standardize_audio_for_asr(
        sample,
        tmp_path,
        audio_quality={"integrated_lufs": -20.0, "true_peak_dbfs": -3.0, "silence_ratio": 0.36},
        mode="adaptive",
    )

    assert stats["preserves_timing"] is True
    assert "不删除静音, 避免破坏字幕/分人时间轴" in stats["skipped_actions"]
    assert "loudness_normalization" not in stats["applied_filters"]
