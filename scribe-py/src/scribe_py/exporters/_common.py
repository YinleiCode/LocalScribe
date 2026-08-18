"""Shared helpers for exporters."""
from __future__ import annotations


def fmt_ts(seconds: float, comma: bool = False) -> str:
    """Format seconds as HH:MM:SS.mmm (or HH:MM:SS,mmm for SRT)."""
    millis = int(round(seconds * 1000))
    h, rem = divmod(millis, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    sep = "," if comma else "."
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"
