"""Plain text export: one line per segment with `[HH:MM:SS.mmm - HH:MM:SS.mmm] text`."""
from __future__ import annotations

from pathlib import Path

from ..core.types import Segment
from ._common import fmt_ts


def render(segments: list[Segment], header: str | None = None) -> str:
    lines: list[str] = []
    if header:
        for h in header.splitlines():
            lines.append(f"# {h}")
        lines.append("")
    for s in segments:
        text = s.text.strip()
        if not text:
            continue
        lines.append(f"[{fmt_ts(s.start)} - {fmt_ts(s.end)}] {text}")
    return "\n".join(lines) + "\n"


def write(path: Path | str, segments: list[Segment], header: str | None = None) -> Path:
    p = Path(path)
    p.write_text(render(segments, header), encoding="utf-8")
    return p
