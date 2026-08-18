"""Diarization — speaker labeling for transcribed segments.

**Active engine: Senko (CAM++ + CoreML), with Resemblyzer fallback**
  - 中文 DER ~13%(AISHELL-4 基准),专门为中文优化
  - macOS CoreML 加速,M 芯片上 96 分钟音频 ~47 秒
  - 输出 192 维 L2 归一化声纹中心
  - 长音频强制走 spectral 聚类(避免 macOS libomp + hdbscan 死锁)

旧 resemblyzer + KMeans 实现保留在 `resemblyzer_diarizer.py` 作为 fallback。

⚠️ 升级注意:senko 输出 192 维 vs 旧的 256 维,**用户已有的声纹样本需要重新上传**。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

from .resemblyzer_diarizer import (
    DiarizationResult as ResemblyzerDiarizationResult,
)

def _append_bundled_site_packages() -> None:
    bundled = os.environ.get("LOCALSCRIBE_BUNDLED_SITE_PACKAGES")
    if bundled and os.path.isdir(bundled) and bundled not in sys.path:
        sys.path.append(bundled)


try:
    import senko as _senko  # noqa: F401
except ModuleNotFoundError:
    _append_bundled_site_packages()
    try:
        import senko as _senko  # noqa: F401
    except ModuleNotFoundError:
        _SENKO_AVAILABLE = False
    else:
        _SENKO_AVAILABLE = True
else:
    _SENKO_AVAILABLE = True

try:
    from pyannote.audio import Pipeline as _pyannote_pipeline  # noqa: F401
except ModuleNotFoundError:
    _PYANNOTE_AVAILABLE = False
else:
    _PYANNOTE_AVAILABLE = True


def _senko_module():
    from . import senko_diarizer

    return senko_diarizer


def _resemblyzer_module():
    from . import resemblyzer_diarizer

    return resemblyzer_diarizer


def _pyannote_module():
    from . import pyannote_diarizer

    return pyannote_diarizer


def available_engines() -> list[str]:
    engines = ["senko" if _SENKO_AVAILABLE else "resemblyzer"]
    engines.append("resemblyzer")
    if _PYANNOTE_AVAILABLE:
        engines.append("pyannote")
    return list(dict.fromkeys(engines))


def diarize(
    audio: Path,
    segments: Sequence[dict],
    n_speakers: int = 0,
    profiles: Iterable[dict] | None = None,
    on_progress=None,
    engine: str | None = None,
) -> ResemblyzerDiarizationResult:
    selected_engine = str(engine or os.environ.get("LOCALSCRIBE_DIARIZATION_ENGINE") or "auto").strip().lower()
    if selected_engine in {"", "default"}:
        selected_engine = "auto"

    if selected_engine == "pyannote":
        return _pyannote_module().diarize(
            audio=audio,
            segments=segments,
            n_speakers=n_speakers,
            profiles=profiles,
            on_progress=on_progress,
        )

    if selected_engine in {"senko", "campp", "cam++"}:
        if not _SENKO_AVAILABLE:
            raise RuntimeError("senko module is not installed; cannot use diarization engine 'senko'")
        return _senko_module().diarize(
            audio=audio,
            segments=segments,
            n_speakers=n_speakers,
            profiles=profiles,
            on_progress=on_progress,
        )

    if selected_engine == "resemblyzer":
        result = _resemblyzer_module().diarize(
            audio=audio,
            segments=segments,
            n_speakers=n_speakers,
            profiles=profiles,
            on_progress=on_progress,
        )
        result.stats["engine"] = "resemblyzer"
        return result

    if selected_engine not in {"auto"}:
        raise ValueError(
            f"unknown diarization engine: {selected_engine}; "
            f"available: {', '.join(available_engines())}"
        )

    if _SENKO_AVAILABLE:
        return _senko_module().diarize(
            audio=audio,
            segments=segments,
            n_speakers=n_speakers,
            profiles=profiles,
            on_progress=on_progress,
        )

    if on_progress:
        on_progress({
            "stage": "diarize_fallback",
            "engine": "resemblyzer",
            "reason": "senko module is not installed",
        })
    result = _resemblyzer_module().diarize(
        audio=audio,
        segments=segments,
        n_speakers=n_speakers,
        profiles=profiles,
        on_progress=on_progress,
    )
    result.stats["engine"] = "resemblyzer"
    result.stats["fallback_reason"] = "senko module is not installed"
    return result


def extract_voice_embedding(audio: Path) -> list[float]:
    if _SENKO_AVAILABLE:
        return _senko_module().extract_voice_embedding(audio)
    return _resemblyzer_module().extract_voice_embedding(audio)


def reidentify_with_voice_anchors(
    audio: Path,
    segments: Sequence[dict],
    anchors: Sequence[dict],
    *,
    threshold: float = 0.78,
    review_threshold: float = 0.70,
    margin: float = 0.05,
    require_enrollment_quality: bool = True,
    on_progress=None,
    engine: str | None = None,
) -> dict:
    selected_engine = str(engine or os.environ.get("LOCALSCRIBE_DIARIZATION_ENGINE") or "auto").strip().lower()
    if selected_engine in {"", "default"}:
        selected_engine = "auto"
    if selected_engine not in {"auto", "senko", "campp", "cam++"}:
        raise RuntimeError("声纹锚点重识别目前仅支持本地 CAM++/Senko 引擎")
    if not _SENKO_AVAILABLE:
        raise RuntimeError("senko module is not installed; cannot use voiceprint re-identification")
    return _senko_module().reidentify_with_voice_anchors(
        audio=audio,
        segments=segments,
        anchors=anchors,
        threshold=threshold,
        review_threshold=review_threshold,
        margin=margin,
        require_enrollment_quality=require_enrollment_quality,
        on_progress=on_progress,
    )


def preflight_voiceprint_anchor_candidates(
    audio: Path,
    segments: Sequence[dict],
    *,
    on_progress=None,
    engine: str | None = None,
) -> dict:
    """Return only anchor segments that pass the CAM++ quality gates."""
    selected_engine = str(engine or os.environ.get("LOCALSCRIBE_DIARIZATION_ENGINE") or "auto").strip().lower()
    if selected_engine in {"", "default"}:
        selected_engine = "auto"
    if selected_engine not in {"auto", "senko", "campp", "cam++"}:
        raise RuntimeError("声纹锚点预检目前仅支持本地 CAM++/Senko 引擎")
    if not _SENKO_AVAILABLE:
        raise RuntimeError("senko module is not installed; cannot preflight voiceprint anchors")
    return _senko_module().preflight_voiceprint_anchor_candidates(
        audio=audio,
        segments=segments,
        on_progress=on_progress,
    )


DiarizationResult = ResemblyzerDiarizationResult

__all__ = [
    "available_engines",
    "diarize",
    "extract_voice_embedding",
    "preflight_voiceprint_anchor_candidates",
    "reidentify_with_voice_anchors",
    "DiarizationResult",
]
