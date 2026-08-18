"""SRT subtitle export."""
from __future__ import annotations

from pathlib import Path

from ..core.types import Segment
from ._common import fmt_ts


def render(segments: list[Segment]) -> str:
    out: list[str] = []
    idx = 1
    for s in segments:
        text = s.text.strip()
        if not text:
            continue
        out.append(str(idx))
        out.append(f"{fmt_ts(s.start, comma=True)} --> {fmt_ts(s.end, comma=True)}")
        out.append(text)
        out.append("")
        idx += 1
    return "\n".join(out)


def write(path: Path | str, segments: list[Segment]) -> Path:
    p = Path(path)
    p.write_text(render(segments), encoding="utf-8")
    return p
