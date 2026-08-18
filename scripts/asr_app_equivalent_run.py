#!/usr/bin/env python3
"""Run the exact App ASR handler and atomically persist its response.

This is an evaluation harness. It calls ``handle_transcribe`` directly so
automatic high-noise review and all App request defaults are exercised without
copying that policy into another CLI implementation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PY_SRC = ROOT / "scribe-py" / "src"
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

from scribe_py.ipc import handle_transcribe  # noqa: E402


def build_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "audio": str(args.audio.expanduser().resolve()),
        "backend": args.backend,
        "language": args.language,
        "normalizer_profile": args.normalizer_profile or None,
        "audio_preprocess": args.audio_preprocess,
        "asr_quality_mode": args.asr_quality_mode,
        "word_timestamps": bool(args.word_timestamps),
        "timing_align": not bool(args.fast_timing),
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def summary(result: dict[str, Any], *, wall_seconds: float, output: Path) -> dict[str, Any]:
    segments = [row for row in result.get("segments") or [] if isinstance(row, dict)]
    strong = (result.get("filter_stats") or {}).get("strong_asr") or {}
    return {
        "ok": True,
        "output": str(output),
        "audio": result.get("audio"),
        "duration_s": result.get("duration"),
        "segments": len(segments),
        "characters": sum(len(str(row.get("text") or "")) for row in segments),
        "transcribe_seconds": result.get("transcribe_seconds"),
        "wall_seconds": round(wall_seconds, 3),
        "risk_level": (result.get("asr_quality") or {}).get("risk_level"),
        "strong_asr": {
            "enabled": strong.get("enabled"),
            "applied": strong.get("applied"),
            "trigger": strong.get("trigger"),
            "reason": strong.get("reason"),
            "cost_seconds": strong.get("cost_seconds"),
            "replacement_count": strong.get("replacement_count"),
            "candidate_count": strong.get("candidate_count"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行与桌面 App 完全相同的 ASR handler")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", default="sensevoice")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--normalizer-profile", default="")
    parser.add_argument("--audio-preprocess", default="adaptive")
    parser.add_argument("--asr-quality-mode", default="standard", choices=("standard", "strong"))
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--fast-timing", action="store_true")
    args = parser.parse_args(argv)

    audio = args.audio.expanduser().resolve()
    if not audio.is_file():
        raise SystemExit(f"audio not found: {audio}")
    output = args.out.expanduser().resolve()
    started = time.perf_counter()
    result = handle_transcribe(build_params(args))
    elapsed = time.perf_counter() - started
    write_json_atomic(output, result)
    print(json.dumps(summary(result, wall_seconds=elapsed, output=output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
