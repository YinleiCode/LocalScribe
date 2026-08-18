"""Pass-through corrector — used when LLM correction is disabled."""
from __future__ import annotations

from ..core.types import Segment
from .base import Corrector, ProgressCallback


class NoOpCorrector(Corrector):
    name = "noop"

    def correct(
        self,
        segments: list[Segment],
        context_hint: str = "",
        on_progress: ProgressCallback | None = None,
    ) -> list[Segment]:
        return [Segment(start=s.start, end=s.end, text=s.text, original_text=s.text) for s in segments]
