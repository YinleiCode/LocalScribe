"""Cross-platform CTranslate2 (faster-whisper) transcriber."""
from __future__ import annotations

from pathlib import Path

from .hotwords import build_initial_prompt
from .transcriber_base import ProgressCallback, Transcriber
from .types import Segment, TranscribeOptions

DEFAULT_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"


class CT2Transcriber(Transcriber):
    backend = "faster-whisper"

    def __init__(self, device: str = "auto", compute_type: str = "auto"):
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._loaded_model_id: str | None = None

    def _load(self, model_id: str):
        from faster_whisper import WhisperModel

        if self._model is not None and self._loaded_model_id == model_id:
            return self._model
        self._model = WhisperModel(model_id, device=self.device, compute_type=self.compute_type)
        self._loaded_model_id = model_id
        return self._model

    def _run(
        self,
        audio: Path,
        options: TranscribeOptions,
        on_progress: ProgressCallback | None,
    ) -> tuple[list[Segment], str | None]:
        model_id = options.model_id or DEFAULT_MODEL
        model = self._load(model_id)

        kwargs = {
            "language": options.language,
            "beam_size": 5,
            "vad_filter": True,
            "word_timestamps": options.word_timestamps,
        }
        initial_prompt = build_initial_prompt(options.initial_prompt, options.hotwords)
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        seg_iter, info = model.transcribe(str(audio), **kwargs)

        segments: list[Segment] = []
        total_duration = info.duration if info else 0.0
        for s in seg_iter:
            text = s.text.strip()
            if not text:
                continue
            segments.append(Segment(start=float(s.start), end=float(s.end), text=text))
            if on_progress:
                on_progress({
                    "current": s.end,
                    "total": total_duration,
                    "preview": text,
                    "stage": "transcribing",
                })

        if on_progress:
            on_progress({"current": total_duration, "total": total_duration, "stage": "done"})
        return segments, info.language if info else None
