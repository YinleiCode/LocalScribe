"""Shared pause/cancel state for in-flight LLM operations.

The IPC reader runs in its own thread so it can deliver `correct_pause` /
`correct_cancel` notifications while a long-running `correct` request is
processing batches. The corrector checks these flags between (and during)
batches and suspends or aborts gracefully.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class CorrectionControl:
    pause_event: threading.Event
    cancel_event: threading.Event

    @classmethod
    def new(cls) -> "CorrectionControl":
        return cls(pause_event=threading.Event(), cancel_event=threading.Event())

    def reset(self) -> None:
        self.pause_event.clear()
        self.cancel_event.clear()

    def request_pause(self) -> None:
        self.pause_event.set()

    def request_resume(self) -> None:
        self.pause_event.clear()

    def request_cancel(self) -> None:
        self.cancel_event.set()
        # Don't get stuck if user cancels while paused.
        self.pause_event.clear()

    def is_paused(self) -> bool:
        return self.pause_event.is_set()

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def wait_if_paused(self, poll_seconds: float = 0.3) -> bool:
        """Block while paused. Returns True if we were paused at least once."""
        if not self.pause_event.is_set():
            return False
        while self.pause_event.is_set() and not self.cancel_event.is_set():
            time.sleep(poll_seconds)
        return True


# Global singleton — only one correction runs at a time in the sidecar.
CONTROL: CorrectionControl = CorrectionControl.new()
