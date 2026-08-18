"""Markdown export with timestamp anchors (brief 第 130 行 v2 增强)."""
from __future__ import annotations

from pathlib import Path

from ..core.types import Segment
from ._common import fmt_ts


def render(segments: list[Segment], title: str | None = None) -> str:
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    for s in segments:
        text = s.text.strip()
        if not text:
            continue
        anchor_id = f"t-{int(s.start * 1000)}"
        lines.append(f'<a id="{anchor_id}"></a>')
        lines.append(f"**[{fmt_ts(s.start)}]** {text}")
        lines.append("")
    return "\n".join(lines)


def write(path: Path | str, segments: list[Segment], title: str | None = None) -> Path:
    p = Path(path)
    p.write_text(render(segments, title), encoding="utf-8")
    return p
