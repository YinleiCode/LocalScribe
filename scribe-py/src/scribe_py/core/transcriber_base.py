"""Transcriber abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import hashlib
import shutil
import tempfile
from typing import Callable

from .types import Segment, TranscribeOptions, TranscribeResult

ProgressCallback = Callable[[dict], None]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Transcriber(ABC):
    """所有转录后端的统一接口。子类实现 `_run`,基类负责计时和打包结果。"""

    backend: str = "base"

    @abstractmethod
    def _run(
        self,
        audio: Path,
        options: TranscribeOptions,
        on_progress: ProgressCallback | None,
    ) -> tuple[list[Segment], str | None]:
        """子类实现:返回 (segments, detected_language)。"""

    def transcribe(
        self,
        audio: Path | str,
        options: TranscribeOptions | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> TranscribeResult:
        import time

        audio_path = Path(audio)
        opts = options or TranscribeOptions()
        t0 = time.time()
        asr_audio = audio_path
        audio_stats: dict = {"enabled": False, "applied": False, "source": str(audio_path), "path": str(audio_path)}
        audio_quality_stats: dict = {"enabled": False}
        channel_selection_stats: dict = {
            "status": "fallback",
            "decision": "mix",
            "reason": "not_evaluated",
        }
        tmp_dir = tempfile.mkdtemp(prefix="localscribe-asr-")
        try:
            from .audio import analyze_audio_quality_for_asr, standardize_audio_for_asr

            audio_quality_stats = analyze_audio_quality_for_asr(audio_path)
            try:
                from .channel_selection import evaluate_stereo_channel_selection

                channel_selection_stats = evaluate_stereo_channel_selection(audio_path)
            except Exception as exc:  # Channel analysis must never disable ASR standardization.
                channel_selection_stats = {
                    "status": "fallback",
                    "decision": "mix",
                    "reason": "analysis_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "preserves_timing": True,
                    "duration_unchanged": False,
                }
            audio_quality_stats["channel_selection"] = channel_selection_stats
            if on_progress:
                on_progress({"stage": "audio_quality_analyzed", **audio_quality_stats})
                on_progress({"stage": "audio_channel_analyzed", **channel_selection_stats})
            asr_audio, audio_stats = standardize_audio_for_asr(
                audio_path,
                tmp_dir,
                audio_quality=audio_quality_stats,
                channel_selection=channel_selection_stats,
                mode=opts.audio_preprocess,
            )
            audio_stats["work_dir"] = tmp_dir
            audio_stats["original_audio"] = str(audio_path)
            audio_stats["standardized_sha256"] = _file_sha256(asr_audio)
            audio_stats["standardized_size_bytes"] = asr_audio.stat().st_size
            if on_progress:
                on_progress({"stage": "audio_standardized", **audio_stats})
        except Exception as exc:
            if audio_quality_stats == {"enabled": False}:
                audio_quality_stats = {"enabled": True, "error": str(exc)}
            audio_stats = {
                "enabled": True,
                "applied": False,
                "source": str(audio_path),
                "path": str(audio_path),
                "work_dir": tmp_dir,
                "original_audio": str(audio_path),
                "error": str(exc),
            }
            asr_audio = audio_path
            if on_progress:
                on_progress({"stage": "audio_standardize_failed", **audio_stats})

        try:
            segments, detected = self._run(asr_audio, opts, on_progress)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        try:
            from .text_normalizer import normalize_segments

            # Recording-specific profiles must be explicit. Environment
            # fallback made customer output depend on hidden process state.
            normalizer_profile = opts.normalizer_profile or None
            normalization_language = detected or opts.language
            self._text_normalization_language = normalization_language
            self._text_normalization_profile = normalizer_profile
            self._text_normalization_context_segments = [
                Segment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    original_text=segment.original_text,
                    speaker=segment.speaker,
                    sync_cues=segment.sync_cues,
                )
                for segment in segments
            ]
            normalized, text_stats = normalize_segments(
                segments,
                language=normalization_language,
                profile=normalizer_profile,
            )
            segments = normalized
            if on_progress:
                on_progress({"stage": "text_normalized", "stats": text_stats})
        except Exception as exc:
            text_stats = {"mode": "local_text_normalizer", "error": str(exc)}
            if on_progress:
                on_progress({"stage": "text_normalize_failed", "error": str(exc)})
        self._text_normalization_error = str(text_stats.get("error") or "")
        try:
            post_normalize = getattr(self, "_post_normalize_transcription", None)
            if callable(post_normalize):
                segments = post_normalize(segments, asr_audio, opts, on_progress)
            finalize = getattr(self, "_finalize_transcription_segments", None)
            if callable(finalize):
                finalize(segments)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            audio_stats["work_dir_cleaned"] = True
            self._text_normalization_context_segments = []
        elapsed = time.time() - t0
        last_segment_end = segments[-1].end if segments else 0.0
        duration = last_segment_end
        try:
            from .audio import probe_audio
            probed_duration = float(probe_audio(audio_path).get("duration") or 0.0)
            if probed_duration > 0:
                duration = probed_duration
        except Exception:
            duration = last_segment_end
        rtf = elapsed / duration if duration else 0.0
        filter_stats = getattr(self, "last_filter_stats", {}) or {}
        filter_stats = {**filter_stats, "audio_standardization": audio_stats, "audio_quality": audio_quality_stats}
        if "text_normalization" in filter_stats and isinstance(filter_stats["text_normalization"], dict):
            filter_stats["text_normalization"] = {**filter_stats["text_normalization"], **text_stats}
        else:
            filter_stats["text_normalization"] = text_stats
        return TranscribeResult(
            audio=str(audio_path),
            language=detected or opts.language,
            duration=duration,
            transcribe_seconds=elapsed,
            rtf=rtf,
            backend=self.backend,
            model_id=opts.model_id,
            segments=segments,
            filter_stats=filter_stats,
        )
