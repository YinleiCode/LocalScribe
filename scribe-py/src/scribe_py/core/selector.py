"""Auto-select transcriber backend based on platform."""
from __future__ import annotations

import platform

from .transcriber_base import Transcriber


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _funasr_available() -> bool:
    try:
        import funasr  # noqa: F401
    except Exception:
        return False
    return True


def _native_backend() -> str:
    return "mlx" if is_apple_silicon() else "ct2"


def resolve_backend(backend: str = "auto") -> str:
    """Resolve optional ASR backends to a backend usable in this environment."""
    if backend == "auto":
        return default_backend()
    if backend in {"funasr", "sensevoice"} and not _funasr_available():
        return _native_backend()
    return backend


def make_transcriber(backend: str = "auto") -> Transcriber:
    """Return a Transcriber instance.

    backend:
      - "auto"  → SenseVoice if cached, otherwise MLX on Apple Silicon / faster-whisper
      - "mlx"   → 强制 MLX (Apple Silicon only)
      - "ct2"   → 强制 faster-whisper
      - "funasr" → FunASR Paraformer 中文 ASR(可选依赖)
      - "sensevoice" → FunASR SenseVoiceSmall(可选依赖)
      - "qwen3" → Qwen3-ASR MLX(显式实验后端,不会被 auto 选中)
    """
    backend = resolve_backend(backend)

    if backend == "mlx":
        from .transcriber_mlx import MLXTranscriber

        return MLXTranscriber()
    if backend == "ct2":
        from .transcriber_ct2 import CT2Transcriber

        return CT2Transcriber()
    if backend == "funasr":
        from .transcriber_funasr import FunASRTranscriber

        return FunASRTranscriber(backend_name="funasr")
    if backend == "sensevoice":
        from .transcriber_funasr import FunASRTranscriber

        return FunASRTranscriber(backend_name="sensevoice")
    if backend == "qwen3":
        from .transcriber_qwen3 import Qwen3ASRTranscriber

        return Qwen3ASRTranscriber()
    raise ValueError(f"Unknown backend: {backend!r}")


def default_model_id(backend: str = "auto") -> str:
    backend = resolve_backend(backend)
    if backend == "mlx":
        from .transcriber_mlx import DEFAULT_MODEL

        return DEFAULT_MODEL
    if backend == "ct2":
        from .transcriber_ct2 import DEFAULT_MODEL

        return DEFAULT_MODEL
    if backend == "funasr":
        from .transcriber_funasr import DEFAULT_MODEL

        return DEFAULT_MODEL
    if backend == "sensevoice":
        from .transcriber_funasr import SENSEVOICE_MODEL

        return SENSEVOICE_MODEL
    if backend == "qwen3":
        from .transcriber_qwen3 import DEFAULT_MODEL

        return DEFAULT_MODEL
    raise ValueError(f"Unknown backend: {backend!r}")


def default_backend() -> str:
    """Prefer the local Chinese-first ASR model when it is available."""
    if _funasr_available():
        try:
            from .transcriber_funasr import SENSEVOICE_MODEL, model_cached

            if model_cached(SENSEVOICE_MODEL):
                return "sensevoice"
        except Exception:
            pass
    return _native_backend()


def backend_unavailable_reason(backend: str = "auto") -> str:
    if backend in {"funasr", "sensevoice"} and not _funasr_available():
        return f"{backend} backend is not installed in this runtime; falling back to {resolve_backend(backend)}"
    if backend == "auto":
        try:
            from .transcriber_funasr import SENSEVOICE_MODEL, model_cached

            if model_cached(SENSEVOICE_MODEL) and not _funasr_available():
                return f"SenseVoice model cache exists, but FunASR is not installed; falling back to {resolve_backend(backend)}"
        except Exception:
            pass
    return ""
