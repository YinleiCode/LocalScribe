"""ffmpeg/ffprobe helpers."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from collections import namedtuple


_PREPROCESS_MODES = {"off", "standard", "adaptive", "ai_denoise", "enhance"}
_DEEPFILTER_MODEL_NAME = "DeepFilterNet3"


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def find_ffprobe() -> str | None:
    return shutil.which("ffprobe")


def _normalize_preprocess_mode(mode: str | None) -> str:
    raw = (mode or os.environ.get("LOCALSCRIBE_AUDIO_PREPROCESS") or "adaptive").strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "none": "off",
        "disable": "off",
        "disabled": "off",
        "1": "adaptive",
        "true": "adaptive",
        "basic": "standard",
        "safe": "adaptive",
        "ai": "ai_denoise",
        "dnn": "ai_denoise",
        "deepfilter": "ai_denoise",
        "deepfilternet": "ai_denoise",
        "strong": "enhance",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in _PREPROCESS_MODES else "adaptive"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _audio_noise_risk(audio_quality: dict[str, Any] | None) -> bool:
    quality = audio_quality or {}
    reasons = " ".join(str(x) for x in (quality.get("risk_reasons") or []))
    if any(token in reasons for token in ("背景噪声", "信噪比", "噪声")):
        return True
    snr = quality.get("estimated_snr_db")
    noise_floor = quality.get("noise_floor_dbfs")
    try:
        if snr is not None and float(snr) < 16:
            return True
        if noise_floor is not None and snr is not None and float(noise_floor) > -40 and float(snr) < 20:
            return True
    except (TypeError, ValueError):
        return False
    return False


def _patch_torchaudio_for_deepfilter() -> None:
    """Provide the legacy torchaudio API DeepFilterNet 0.5 expects.

    New torchaudio builds removed `torchaudio.backend.common.AudioMetaData` and
    sometimes `torchaudio.info`.  DeepFilterNet only needs the metadata object
    and `info()`; use soundfile to supply them without downgrading torchaudio,
    which would be riskier for the ASR stack.
    """
    try:
        import types
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio as ta
    except Exception:
        return

    AudioMetaData = namedtuple(
        "AudioMetaData",
        ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
    )
    if not hasattr(ta, "info"):
        def _info(file: str, **_kwargs):
            info = sf.info(file)
            subtype = str(getattr(info, "subtype", "") or "")
            bits = 16 if "16" in subtype else 24 if "24" in subtype else 32 if "32" in subtype else 0
            return AudioMetaData(
                sample_rate=int(info.samplerate),
                num_frames=int(info.frames),
                num_channels=int(info.channels),
                bits_per_sample=bits,
                encoding=subtype or str(getattr(info, "format", "") or "UNKNOWN"),
            )
        ta.info = _info  # type: ignore[attr-defined]

    def _sf_load(file: str, **kwargs):
        frame_offset = int(kwargs.get("frame_offset") or 0)
        num_frames = int(kwargs.get("num_frames") or -1)
        stop = None if num_frames < 0 else frame_offset + num_frames
        data, sr = sf.read(file, start=frame_offset, stop=stop, dtype="float32", always_2d=True)
        tensor = torch.from_numpy(np.asarray(data).T.copy())
        return tensor, int(sr)

    def _sf_save(file: str, audio, sample_rate: int, **_kwargs):
        tensor = torch.as_tensor(audio).detach().cpu()
        if tensor.ndim == 1:
            array = tensor.numpy()
        else:
            array = tensor.numpy().T
        sf.write(file, array, int(sample_rate))

    ta.load = _sf_load  # type: ignore[assignment]
    ta.save = _sf_save  # type: ignore[assignment]

    backend_mod = sys.modules.get("torchaudio.backend")
    if backend_mod is None:
        backend_mod = types.ModuleType("torchaudio.backend")
        sys.modules["torchaudio.backend"] = backend_mod
    common_mod = sys.modules.get("torchaudio.backend.common")
    if common_mod is None:
        common_mod = types.ModuleType("torchaudio.backend.common")
        sys.modules["torchaudio.backend.common"] = common_mod
    common_mod.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]


def _valid_deepfilter_model_dir(path: Path) -> bool:
    checkpoints = path / "checkpoints"
    return (
        path.is_dir()
        and (path / "config.ini").is_file()
        and checkpoints.is_dir()
        and any(checkpoints.glob("model*.ckpt*"))
    )


def _deepfilter_model_candidates(model_name: str = _DEEPFILTER_MODEL_NAME) -> list[Path]:
    candidates: list[Path] = []

    explicit = os.environ.get("LOCALSCRIBE_DEEPFILTER_MODEL_DIR")
    if explicit:
        candidates.append(Path(explicit).expanduser())

    resources = os.environ.get("LOCALSCRIBE_RESOURCES")
    if resources:
        candidates.append(Path(resources).expanduser() / "deepfilternet" / model_name)

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "deepfilternet" / model_name)
        candidates.append(parent / "src-tauri" / "bundle-staging" / "deepfilternet" / model_name)

    cache_root = os.environ.get("LOCALSCRIBE_DEEPFILTER_CACHE")
    if cache_root:
        candidates.append(Path(cache_root).expanduser() / model_name)
    candidates.extend([
        Path.home() / "Library" / "Caches" / "DeepFilterNet" / model_name,
        Path.home() / ".cache" / "DeepFilterNet" / model_name,
    ])

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_deepfilter_model_dir(model_name: str = _DEEPFILTER_MODEL_NAME) -> Path | None:
    """Return a local DeepFilterNet model directory without triggering download."""
    for candidate in _deepfilter_model_candidates(model_name):
        if _valid_deepfilter_model_dir(candidate):
            return candidate
    return None


def ensure_deepfilter_model_dir(*, download: bool = False, model_name: str = _DEEPFILTER_MODEL_NAME) -> Path | None:
    """Find or optionally download the DeepFilterNet model for packaging.

    Runtime calls keep `download=False` so customer machines do not unexpectedly
    hit the network. Build scripts may pass `download=True` and then copy the
    returned cache directory into the app bundle.
    """
    existing = resolve_deepfilter_model_dir(model_name)
    if existing is not None:
        return existing
    if not download:
        return None
    try:
        _patch_torchaudio_for_deepfilter()
        from df.enhance import maybe_download_model

        downloaded = Path(maybe_download_model(model_name)).expanduser()
        return downloaded if _valid_deepfilter_model_dir(downloaded) else None
    except Exception:
        return None


def deepfilter_available() -> bool:
    """Return whether DeepFilterNet can be imported in the current runtime."""
    try:
        _patch_torchaudio_for_deepfilter()
        from df.enhance import enhance as _enhance  # noqa: F401
        from df.enhance import init_df as _init_df  # noqa: F401
        from df.enhance import load_audio as _load_audio  # noqa: F401
        from df.enhance import save_audio as _save_audio  # noqa: F401
    except Exception:
        return False
    return resolve_deepfilter_model_dir() is not None


def _deepfilter_missing_stats(src: Path, out: Path, reason: str) -> dict[str, Any]:
    return {
        "engine": "deepfilternet",
        "available": False,
        "applied": False,
        "input": str(src),
        "output": str(out),
        "method": "",
        "model_dir": "",
        "error": reason,
    }



def _run_deepfilter_enhance(src: Path, out: Path, *, timeout: float = 600.0) -> dict[str, Any]:
    """Enhance a mono wav with DeepFilterNet.

    Prefer the Python API because it lets us choose an exact output path.  Keep a
    CLI fallback for minor package API differences.  This is intentionally
    optional: if the dependency is missing, callers should fall back to the
    deterministic ffmpeg path.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "engine": "deepfilternet",
        "available": False,
        "applied": False,
        "input": str(src),
        "output": str(out),
        "method": "",
        "model_dir": "",
        "error": "",
    }
    model_dir = resolve_deepfilter_model_dir()
    if model_dir is None:
        return _deepfilter_missing_stats(src, out, "DeepFilterNet model not found locally")
    stats["model_dir"] = str(model_dir)
    try:
        _patch_torchaudio_for_deepfilter()
        from df.enhance import enhance, init_df, load_audio, save_audio

        model, df_state, _ = init_df(str(model_dir), log_file=None)
        sr = int(df_state.sr())
        audio, _meta = load_audio(str(src), sr=sr)
        enhanced = enhance(model, df_state, audio)
        save_audio(str(out), enhanced, sr)
        stats.update({"available": True, "applied": out.exists() and out.stat().st_size > 44, "method": "python_api"})
        if not stats["applied"]:
            stats["error"] = "DeepFilterNet produced no output"
        return stats
    except Exception as exc:  # noqa: BLE001
        stats["error"] = f"{type(exc).__name__}: {exc}"

    try:
        cli = shutil.which("deepFilter")
        cmd = [cli or sys.executable, "-m", "df.enhance"]
        if cli:
            cmd = [cli]
        tmp_out_dir = out.parent / "deepfilter_cli"
        tmp_out_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["-m", str(model_dir), "-o", str(tmp_out_dir), str(src)])
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        if proc.returncode != 0:
            stats["error"] = (proc.stderr or proc.stdout or f"DeepFilterNet exited {proc.returncode}").strip()
            return stats
        candidates = sorted(tmp_out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            stats["error"] = "DeepFilterNet CLI produced no wav output"
            return stats
        shutil.copyfile(candidates[0], out)
        stats.update({"available": True, "applied": out.exists() and out.stat().st_size > 44, "method": "cli"})
        return stats
    except Exception as exc:  # noqa: BLE001
        if not stats["error"]:
            stats["error"] = f"{type(exc).__name__}: {exc}"
        return stats


def probe_audio(audio: Path | str) -> dict:
    """Return {duration, size, format_name, has_audio_stream}. Raises if ffprobe missing."""
    ffprobe = find_ffprobe()
    if not ffprobe:
        raise RuntimeError("ffprobe not found in PATH. Install ffmpeg first.")
    proc = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration,size,format_name",
            "-show_streams",
            "-of", "json",
            str(audio),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    fmt = data.get("format", {})
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    first_audio = audio_streams[0] if audio_streams else {}
    return {
        "duration": float(fmt.get("duration", 0)),
        "size": int(fmt.get("size", 0)),
        "format_name": fmt.get("format_name"),
        "has_audio_stream": has_audio,
        "sample_rate": int(first_audio.get("sample_rate") or 0),
        "channels": int(first_audio.get("channels") or 0),
        "codec_name": first_audio.get("codec_name") or "",
    }


def build_asr_preprocess_plan(audio_quality: dict[str, Any] | None = None, *, mode: str | None = None) -> dict:
    """Return the local ASR preprocessing plan inspired by production ASR stacks.

    The default is deliberately conservative.  It always preserves timestamps:
    no silence deletion, no speed changes, no segmentation edits.  Model-based
    denoise is available via `ai_denoise`; `enhance` remains a deterministic
    ffmpeg-only fallback.
    """
    normalized_mode = _normalize_preprocess_mode(mode)
    quality = audio_quality or {}
    plan: dict[str, Any] = {
        "mode": normalized_mode,
        "enabled": normalized_mode != "off",
        "preserves_timing": True,
        "target_sample_rate": 16000,
        "target_channels": 1,
        "applied_filters": [],
        "ffmpeg_audio_filter": "",
        "ai_denoise_engine": "",
        "rationale": [],
        "skipped_actions": [],
        "industry_practices": [
            "音频质量门禁",
            "16kHz 单声道 PCM 标准化",
            "默认保真优先, 降噪必须通过预检或显式选择",
            "按需响度均衡",
            "不删除静音以保护时间轴",
        ],
    }
    if normalized_mode == "off":
        plan["rationale"].append("已关闭音频预处理")
        return plan

    plan["applied_filters"].extend(["downmix_mono", "resample_16k", "pcm_s16le"])
    plan["rationale"].append("统一为 ASR 友好的 16kHz 单声道 PCM")

    loudness = quality.get("integrated_lufs")
    peak = quality.get("true_peak_dbfs")
    silence_ratio = float(quality.get("silence_ratio") or 0.0)
    noise_risk = _audio_noise_risk(quality)
    filters: list[str] = []

    if normalized_mode == "enhance":
        filters.extend(["highpass=f=80", "lowpass=f=7600", "afftdn=nf=-25", "loudnorm=I=-23:TP=-2:LRA=11"])
        plan["applied_filters"].extend(["speech_bandpass", "stationary_noise_reduction", "loudness_normalization"])
        plan["rationale"].append("强增强模式: 轻量降噪 + 语音频段保留 + 响度均衡")
    elif normalized_mode == "ai_denoise":
        plan["ai_denoise_engine"] = "deepfilternet"
        plan["applied_filters"].append("deepfilternet_ai_denoise")
        plan["rationale"].append("AI 降噪模式: DeepFilterNet 人声增强,随后标准化为 ASR 输入")
        if loudness is None:
            plan["skipped_actions"].append("未取得 LUFS, AI 降噪后仅做格式标准化")
        elif float(loudness) < -26:
            filters.append("loudnorm=I=-23:TP=-2:LRA=11")
            plan["applied_filters"].append("loudness_normalization")
            plan["rationale"].append("检测到整体音量偏低, AI 降噪后做响度均衡")
    elif normalized_mode == "adaptive":
        if noise_risk:
            snr = quality.get("estimated_snr_db")
            floor = quality.get("noise_floor_dbfs")
            if snr is not None or floor is not None:
                detail = (
                    f"{f'估算 SNR {float(snr):.1f}dB' if snr is not None else ''}"
                    f"{'；' if snr is not None and floor is not None else ''}"
                    f"{f'噪声底 {float(floor):.1f}dBFS' if floor is not None else ''}"
                )
                plan["skipped_actions"].append(f"检测到背景噪声风险({detail}), adaptive 保持保真不自动降噪")
            else:
                plan["skipped_actions"].append("检测到背景噪声风险, adaptive 保持保真不自动降噪")
        if loudness is None:
            plan["skipped_actions"].append("未取得 LUFS, 仅做格式标准化")
        elif float(loudness) < -26 and noise_risk:
            plan["skipped_actions"].append(
                "整体音量偏低但同时存在背景噪声风险, adaptive 不做响度均衡以避免放大噪声"
            )
        elif float(loudness) < -26:
            filters.append("loudnorm=I=-23:TP=-2:LRA=11")
            plan["applied_filters"].append("loudness_normalization")
            plan["rationale"].append("检测到整体音量偏低, 自动做响度均衡")
        elif not noise_risk:
            plan["skipped_actions"].append("响度正常, 不额外改变音频")

    if peak is not None and float(peak) >= -0.2:
        plan["skipped_actions"].append("疑似削波/爆音无法靠本地滤镜可靠恢复")
    if silence_ratio >= 0.25:
        plan["skipped_actions"].append("不删除静音, 避免破坏字幕/分人时间轴")

    plan["ffmpeg_audio_filter"] = ",".join(filters)
    return plan


def analyze_audio_quality_for_asr(audio: Path | str, *, timeout: float = 60.0) -> dict:
    """Measure cheap, local audio quality signals before ASR.

    This intentionally avoids model-based enhancement.  It gives the product a
    repeatable quality gate: low loudness, clipping and too much silence are all
    practical predictors of poor ASR on real customer recordings.
    """
    path = Path(audio)
    stats = {
        "enabled": True,
        "source": str(path),
        "duration_s": 0.0,
        "sample_rate": 0,
        "channels": 0,
        "integrated_lufs": None,
        "true_peak_dbfs": None,
        "noise_floor_dbfs": None,
        "speech_level_dbfs": None,
        "estimated_snr_db": None,
        "rms_sample_count": 0,
        "silence_duration_s": 0.0,
        "silence_ratio": 0.0,
        "risk_level": "unknown",
        "risk_reasons": [],
        "error": "",
    }
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        stats["enabled"] = False
        stats["error"] = "ffmpeg not found"
        return stats
    try:
        info = probe_audio(path)
        stats.update({
            "duration_s": float(info.get("duration") or 0.0),
            "sample_rate": int(info.get("sample_rate") or 0),
            "channels": int(info.get("channels") or 0),
        })
    except Exception as exc:  # noqa: BLE001
        stats["error"] = str(exc)

    filters = (
        "ebur128=peak=true:framelog=verbose,"
        "silencedetect=noise=-45dB:d=0.5,"
        "astats=metadata=1:reset=0.5,"
        "ametadata=print:key=lavfi.astats.Overall.RMS_level"
    )
    try:
        proc = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-i",
                str(path),
                "-vn",
                "-af",
                filters,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        stats["error"] = f"audio quality analysis timed out after {timeout:.1f}s"
        return _score_audio_quality(stats)

    stderr = proc.stderr or ""
    if proc.returncode != 0:
        stats["error"] = stderr.strip() or f"ffmpeg exited {proc.returncode}"
        return _score_audio_quality(stats)

    integrated_matches = re.findall(r"(?:^|\s)I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", stderr, flags=re.MULTILINE)
    peak_matches = re.findall(r"(?:^|\s)Peak:\s*(-?\d+(?:\.\d+)?)\s*dBFS", stderr, flags=re.MULTILINE)
    silence_durations = [float(x) for x in re.findall(r"silence_duration:\s*(\d+(?:\.\d+)?)", stderr)]
    rms_levels = [
        float(x)
        for x in re.findall(r"lavfi\.astats\.Overall\.RMS_level=(-?\d+(?:\.\d+)?)", stderr)
        if x.lower() not in {"-inf", "inf"}
    ]
    if integrated_matches:
        stats["integrated_lufs"] = float(integrated_matches[-1])
    if peak_matches:
        stats["true_peak_dbfs"] = float(peak_matches[-1])
    if rms_levels:
        noise_floor = _percentile(rms_levels, 15)
        speech_level = _percentile(rms_levels, 85)
        stats["rms_sample_count"] = len(rms_levels)
        if noise_floor is not None:
            stats["noise_floor_dbfs"] = round(noise_floor, 2)
        if speech_level is not None:
            stats["speech_level_dbfs"] = round(speech_level, 2)
        if noise_floor is not None and speech_level is not None:
            stats["estimated_snr_db"] = round(max(0.0, speech_level - noise_floor), 2)
    if silence_durations:
        stats["silence_duration_s"] = round(sum(silence_durations), 3)
    duration = float(stats.get("duration_s") or 0.0)
    if duration > 0:
        stats["silence_ratio"] = round(min(float(stats["silence_duration_s"]) / duration, 1.0), 4)
    return _score_audio_quality(stats)


def _score_audio_quality(stats: dict) -> dict:
    reasons: list[str] = []
    duration = float(stats.get("duration_s") or 0.0)
    loudness = stats.get("integrated_lufs")
    peak = stats.get("true_peak_dbfs")
    noise_floor = stats.get("noise_floor_dbfs")
    estimated_snr = stats.get("estimated_snr_db")
    silence_ratio = float(stats.get("silence_ratio") or 0.0)
    sample_rate = int(stats.get("sample_rate") or 0)
    channels = int(stats.get("channels") or 0)

    if duration <= 0:
        reasons.append("无法读取有效音频时长")
    if sample_rate and sample_rate < 16000:
        reasons.append("采样率低于 16kHz")
    if channels > 1:
        reasons.append("多声道会被合并为单声道")
    if loudness is not None:
        if loudness < -32:
            reasons.append("整体音量过低")
        elif loudness < -26:
            reasons.append("整体音量偏低")
    if peak is not None and peak >= -0.2:
        reasons.append("疑似峰值削波/爆音")
    if estimated_snr is not None:
        if estimated_snr < 10:
            reasons.append("信噪比过低")
        elif estimated_snr < 16:
            reasons.append("信噪比偏低")
    if noise_floor is not None:
        if noise_floor > -35 and (estimated_snr is None or estimated_snr < 18):
            reasons.append("背景噪声明显")
        elif noise_floor > -40 and (estimated_snr is None or estimated_snr < 20):
            reasons.append("背景噪声偏高")
    if silence_ratio >= 0.45:
        reasons.append("静音占比过高")
    elif silence_ratio >= 0.25:
        reasons.append("静音占比较高")

    high_tokens = {"无法读取有效音频时长", "整体音量过低", "疑似峰值削波/爆音", "静音占比过高", "信噪比过低", "背景噪声明显"}
    if any(reason in high_tokens for reason in reasons):
        risk_level = "high"
    elif reasons:
        risk_level = "medium"
    else:
        risk_level = "low"
    stats["risk_reasons"] = reasons
    stats["risk_level"] = risk_level
    return stats


def _run_standardize_ffmpeg(
    *,
    ffmpeg: str,
    src: Path,
    out: Path,
    audio_filter: str,
    sample_rate: int = 16000,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(src),
        "-vn",
    ]
    if audio_filter:
        cmd.extend(["-af", audio_filter])
    cmd.extend([
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(out),
    ])
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _safe_channel_filter(channel_selection: dict | None) -> tuple[str, str]:
    """Return a prevalidated channel filter without changing timestamps."""
    report = channel_selection if isinstance(channel_selection, dict) else {}
    if (
        report.get("status") != "ok"
        or report.get("decision") not in {"left", "right"}
        or int(report.get("channels") or 0) != 2
        or report.get("preserves_timing") is not True
        or report.get("duration_unchanged") is not True
    ):
        return "", ""
    if report["decision"] == "left":
        return "pan=mono|c0=c0", "select_left_channel"
    return "pan=mono|c0=c1", "select_right_channel"


def _join_audio_filters(*filters: str) -> str:
    return ",".join(value for value in filters if value)


def _record_channel_filter(
    applied_filters: list[str],
    channel_filter_label: str,
) -> list[str]:
    if not channel_filter_label:
        return list(applied_filters)
    return [
        channel_filter_label if item == "downmix_mono" else item
        for item in applied_filters
    ]


def standardize_audio_for_asr(
    audio: Path | str,
    work_dir: Path | str,
    *,
    audio_quality: dict[str, Any] | None = None,
    channel_selection: dict[str, Any] | None = None,
    mode: str | None = None,
) -> tuple[Path, dict]:
    """Convert arbitrary input audio to 16 kHz mono WAV for ASR stability.

    The output always preserves timing.  `adaptive`/`enhance` use deterministic
    ffmpeg filters; `ai_denoise` optionally runs DeepFilterNet first and then
    returns to 16 kHz mono PCM for ASR.
    """
    src = Path(audio)
    stats = {
        "enabled": True,
        "applied": False,
        "source": str(src),
        "path": str(src),
        "format": None,
        "sample_rate": 16000,
        "channels": 1,
        "mode": _normalize_preprocess_mode(mode),
        "preserves_timing": True,
        "applied_filters": [],
        "audio_filter": "",
        "ai_denoise": {},
        "rationale": [],
        "skipped_actions": [],
        "fallback_applied": False,
        "channel_selection": dict(channel_selection or {}),
        "channel_decision": "mix",
        "error": "",
    }

    if os.environ.get("LOCALSCRIBE_STANDARDIZE_AUDIO", "1") == "0":
        stats["enabled"] = False
        stats["mode"] = "off"
        stats["skipped_actions"] = ["LOCALSCRIBE_STANDARDIZE_AUDIO=0"]
        return src, stats
    plan = build_asr_preprocess_plan(audio_quality, mode=stats["mode"])
    stats.update({
        "enabled": bool(plan.get("enabled")),
        "mode": plan.get("mode"),
        "preserves_timing": plan.get("preserves_timing", True),
        "applied_filters": list(plan.get("applied_filters") or []),
        "audio_filter": str(plan.get("ffmpeg_audio_filter") or ""),
        "ai_denoise_engine": str(plan.get("ai_denoise_engine") or ""),
        "rationale": list(plan.get("rationale") or []),
        "skipped_actions": list(plan.get("skipped_actions") or []),
        "industry_practices": list(plan.get("industry_practices") or []),
    })
    channel_filter, channel_filter_label = _safe_channel_filter(channel_selection)
    if channel_filter:
        stats["channel_decision"] = str(channel_selection["decision"])
        stats["applied_filters"] = _record_channel_filter(
            stats["applied_filters"],
            channel_filter_label,
        )
        stats["rationale"].append(
            "仅在单声道覆盖至少 98% 语音且质量领先至少 4dB 时选择更优声道"
        )
    if not plan.get("enabled"):
        return src, stats

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        stats["error"] = "ffmpeg not found"
        return src, stats

    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    out = work_path / "asr_input_16k_mono.wav"
    input_for_final = src
    quality_filter = stats["audio_filter"]
    if stats["mode"] == "ai_denoise":
        ai_input = work_path / "ai_denoise_input_48k_mono.wav"
        ai_output = work_path / "ai_denoise_deepfilter.wav"
        prep = _run_standardize_ffmpeg(
            ffmpeg=ffmpeg,
            src=src,
            out=ai_input,
            audio_filter=channel_filter,
            sample_rate=48000,
        )
        if prep.returncode == 0 and ai_input.exists() and ai_input.stat().st_size > 44:
            ai_stats = _run_deepfilter_enhance(ai_input, ai_output)
            stats["ai_denoise"] = ai_stats
            if ai_stats.get("applied") and ai_output.exists() and ai_output.stat().st_size > 44:
                input_for_final = ai_output
            else:
                fallback_plan = build_asr_preprocess_plan(audio_quality, mode="adaptive")
                stats["fallback_applied"] = True
                stats["fallback_mode"] = "adaptive"
                stats["skipped_actions"].append("AI 降噪不可用或失败, 已回退为 adaptive 安全基线")
                stats["error"] = str(ai_stats.get("error") or "DeepFilterNet unavailable")
                stats["mode"] = "adaptive"
                stats["applied_filters"] = _record_channel_filter(
                    list(fallback_plan.get("applied_filters") or []),
                    channel_filter_label,
                )
                stats["audio_filter"] = str(fallback_plan.get("ffmpeg_audio_filter") or "")
                quality_filter = stats["audio_filter"]
                stats["rationale"] = list(fallback_plan.get("rationale") or [])
        else:
            fallback_plan = build_asr_preprocess_plan(audio_quality, mode="adaptive")
            stats["fallback_applied"] = True
            stats["fallback_mode"] = "adaptive"
            stats["skipped_actions"].append("AI 降噪输入准备失败, 已回退为 adaptive 安全基线")
            stats["error"] = (prep.stderr or f"ffmpeg exited {prep.returncode}").strip()
            stats["mode"] = "adaptive"
            stats["applied_filters"] = _record_channel_filter(
                list(fallback_plan.get("applied_filters") or []),
                channel_filter_label,
            )
            stats["audio_filter"] = str(fallback_plan.get("ffmpeg_audio_filter") or "")
            quality_filter = stats["audio_filter"]
            stats["rationale"] = list(fallback_plan.get("rationale") or [])

    final_channel_filter = channel_filter if input_for_final == src else ""
    final_audio_filter = _join_audio_filters(final_channel_filter, quality_filter)
    stats["audio_filter"] = final_audio_filter
    proc = _run_standardize_ffmpeg(
        ffmpeg=ffmpeg,
        src=input_for_final,
        out=out,
        audio_filter=final_audio_filter,
    )
    if proc.returncode != 0 and final_audio_filter:
        primary_error = (proc.stderr or f"ffmpeg exited {proc.returncode}").strip()
        out.unlink(missing_ok=True)
        proc = _run_standardize_ffmpeg(ffmpeg=ffmpeg, src=src, out=out, audio_filter="")
        stats["fallback_applied"] = True
        stats["skipped_actions"].append("增强滤镜失败, 已回退为仅格式标准化")
        stats["error"] = primary_error
        stats["audio_filter"] = ""
        stats["channel_decision"] = "mix"
        stats["applied_filters"] = [x for x in stats["applied_filters"] if x in {"downmix_mono", "resample_16k", "pcm_s16le"}]
        if "downmix_mono" not in stats["applied_filters"]:
            stats["applied_filters"].insert(0, "downmix_mono")
    if proc.returncode != 0 or not out.exists() or out.stat().st_size <= 44:
        stats["error"] = (proc.stderr or stats["error"] or "ffmpeg produced no output").strip()
        return src, stats

    stats.update({"applied": True, "path": str(out), "format": "wav"})
    return out, stats
