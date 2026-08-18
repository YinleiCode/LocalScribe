"""Optional Qwen3-ASR backend for Apple Silicon quality evaluation.

This backend is deliberately opt-in. It lets the benchmark harness compare a
newer local ASR model against the frozen SenseVoice path without changing the
customer default or the playback/diarization pipeline.
"""
from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Any

from .transcriber_base import ProgressCallback, Transcriber
from .types import Segment, TranscribeOptions

DEFAULT_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"

_SENTENCE_RE = re.compile(r".*?[。！？!?；;](?:[\"'”’」』】])?|.+$", re.S)


def _language_name(language: str | None) -> str:
    value = (language or "zh").strip().lower()
    return {
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "yue": "Cantonese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
    }.get(value, language or "Chinese")


def _split_chunk_text(text: str, start: float, end: float) -> list[Segment]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    parts = [match.group(0).strip() for match in _SENTENCE_RE.finditer(cleaned)]
    parts = [part for part in parts if part]
    if not parts:
        parts = [cleaned]

    weights = [max(len(re.sub(r"\s+", "", part)), 1) for part in parts]
    total_weight = sum(weights)
    duration = max(end - start, 0.001)
    cursor = start
    segments: list[Segment] = []
    for index, (part, weight) in enumerate(zip(parts, weights)):
        part_end = end if index + 1 == len(parts) else cursor + duration * weight / total_weight
        segments.append(Segment(start=cursor, end=max(part_end, cursor), text=part))
        cursor = part_end
    return segments


class Qwen3ASRTranscriber(Transcriber):
    backend = "qwen3"

    def __init__(self):
        self._model: Any | None = None
        self._loaded_model_id: str | None = None

    def _load(self, model_id: str) -> Any:
        if self._model is not None and self._loaded_model_id == model_id:
            return self._model
        try:
            from mlx_audio.stt.utils import load_model
        except Exception as exc:
            raise RuntimeError(
                "Qwen3-ASR evaluation requires mlx-audio on Apple Silicon"
            ) from exc
        self._model = load_model(model_id)
        self._loaded_model_id = model_id
        return self._model

    def _run(
        self,
        audio: Path,
        options: TranscribeOptions,
        on_progress: ProgressCallback | None,
    ) -> tuple[list[Segment], str | None]:
        model_id = options.model_id or DEFAULT_MODEL
        if on_progress:
            on_progress({"stage": "loading_model", "backend": self.backend, "model": model_id})
        model = self._load(model_id)

        if on_progress:
            on_progress({"stage": "transcribing", "backend": self.backend, "model": model_id})
        try:
            from .audio import probe_audio

            audio_duration = float(probe_audio(audio).get("duration") or 0.0)
        except Exception:
            audio_duration = 0.0
        # Autoregressive ASR can loop on noisy audio. Bound output by audio
        # duration instead of allowing an unconditional 8k-token generation.
        max_tokens = min(8192, max(256, int(math.ceil(audio_duration * 12.0)) + 128))
        result = model.generate(
            str(audio),
            language=_language_name(options.language),
            max_tokens=max_tokens,
            temperature=0.0,
            repetition_penalty=1.1,
            repetition_context_size=256,
            chunk_duration=1200.0,
            verbose=False,
        )

        segments: list[Segment] = []
        raw_segments = list(getattr(result, "segments", None) or [])
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            segments.extend(
                _split_chunk_text(
                    str(item.get("text") or ""),
                    float(item.get("start") or 0.0),
                    float(item.get("end") or 0.0),
                )
            )
        if not segments:
            text = str(getattr(result, "text", "") or "").strip()
            segments = _split_chunk_text(text, 0.0, 0.001)

        output_chars = sum(len(segment.text or "") for segment in segments)
        chars_per_second = output_chars / audio_duration if audio_duration > 0 else 0.0
        generation_tokens = int(getattr(result, "generation_tokens", 0) or 0)
        hallucination_risk = bool(
            (audio_duration > 0 and chars_per_second > 12.0)
            or generation_tokens >= int(max_tokens * 0.98)
        )
        self.last_filter_stats = {
            "mode": "qwen3_asr",
            "model_family": "qwen3_asr",
            "raw_chunks": len(raw_segments),
            "prompt_tokens": int(getattr(result, "prompt_tokens", 0) or 0),
            "generation_tokens": generation_tokens,
            "max_tokens": max_tokens,
            "output_chars": output_chars,
            "chars_per_second": round(chars_per_second, 3),
            "has_hallucination_risk": hallucination_risk,
            "model_seconds": float(getattr(result, "total_time", 0.0) or 0.0),
            "timing_mode": "coarse_model_chunks",
            "timing_reliable": False,
            "timing_reason": "Qwen3-ASR experiment currently evaluates text quality only",
            "hotwords_supported": False,
        }
        return segments, options.language or "zh"
