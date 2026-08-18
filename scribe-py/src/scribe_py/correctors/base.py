"""Corrector abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from ..core.types import Segment

ProgressCallback = Callable[[dict], None]


class Corrector(ABC):
    name: str = "base"

    @abstractmethod
    def correct(
        self,
        segments: list[Segment],
        context_hint: str = "",
        on_progress: ProgressCallback | None = None,
    ) -> list[Segment]:
        """Return corrected segments. Each Segment.original_text holds the source text."""
