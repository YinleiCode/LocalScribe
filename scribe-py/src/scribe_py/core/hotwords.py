"""Project glossary / ASR hotword helpers.

The recognizer cannot infer customer-specific names or terms from silence.  This
module gives every backend a common, auditable way to receive those words before
ASR runs, instead of relying on recording-specific post fixes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


_INLINE_SPLIT_RE = re.compile(r"[,，;；、|\t]+")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)、])\s*")
_MAX_TERM_LEN = 48


def _clean_term(value: Any) -> str:
    term = str(value or "").strip()
    term = _BULLET_RE.sub("", term).strip()
    term = term.strip("`\"'“”‘’[]()（）")
    return term


def _terms_from_json(value: Any) -> list[str]:
    terms: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                terms.append(item)
            elif isinstance(item, dict):
                terms.append(item.get("term") or item.get("name") or item.get("word") or "")
    elif isinstance(value, dict):
        raw = value.get("terms") or value.get("hotwords") or value.get("glossary")
        if raw is not None:
            terms.extend(_terms_from_json(raw))
        else:
            terms.extend(str(key) for key in value.keys())
    return terms


def parse_hotword_terms(text: str) -> list[str]:
    """Parse a user/customer term list while preserving order and de-duping.

    Supported formats:
    - plain lines: one term per line
    - comma/Chinese-comma separated terms
    - JSON list of strings or objects with `term` / `name` / `word`
    - JSON object with `terms`, `hotwords`, or `glossary`
    """
    raw = (text or "").strip()
    if not raw:
        return []

    candidates: list[str] = []
    if raw[0] in "[{":
        try:
            candidates.extend(_terms_from_json(json.loads(raw)))
        except Exception:
            candidates = []

    if not candidates:
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            candidates.extend(part for part in _INLINE_SPLIT_RE.split(line) if part.strip())

    seen: set[str] = set()
    terms: list[str] = []
    for candidate in candidates:
        term = _clean_term(candidate)
        if not term or len(term) > _MAX_TERM_LEN or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def load_hotword_terms(*, inline: str = "", file_path: str | Path | None = None) -> list[str]:
    parts: list[str] = []
    if inline:
        parts.append(inline)
    if file_path:
        path = Path(file_path).expanduser().resolve()
        parts.append(path.read_text(encoding="utf-8"))
    return parse_hotword_terms("\n".join(parts))


def build_hotword_string(terms: Iterable[str]) -> str:
    """Compact form for ASR engines that expose a hotword parameter."""
    return " ".join(term.strip() for term in terms if term and term.strip())


def build_initial_prompt(base_prompt: str = "", terms: Iterable[str] = ()) -> str:
    """Prompt form for Whisper-like backends that only support initial_prompt."""
    base = (base_prompt or "").strip()
    hotwords = [term.strip() for term in terms if term and term.strip()]
    if not hotwords:
        return base
    hint = "以下词语可能出现在录音中，请优先按这些写法转写：" + "、".join(hotwords)
    return "\n".join(part for part in [base, hint] if part)
