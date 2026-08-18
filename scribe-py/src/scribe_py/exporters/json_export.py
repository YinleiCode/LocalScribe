"""JSON export — full structured data with metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.types import Segment, TranscribeResult


def render(result: TranscribeResult, extra: dict[str, Any] | None = None) -> str:
    d = result.to_dict()
    if extra:
        d.update(extra)
    return json.dumps(d, ensure_ascii=False, indent=2)


def render_segments(segments: list[Segment], meta: dict[str, Any] | None = None) -> str:
    d: dict[str, Any] = {"segments": [s.to_dict() for s in segments]}
    if meta:
        d.update(meta)
    return json.dumps(d, ensure_ascii=False, indent=2)


def write(path: Path | str, result: TranscribeResult, extra: dict[str, Any] | None = None) -> Path:
    p = Path(path)
    p.write_text(render(result, extra), encoding="utf-8")
    return p
