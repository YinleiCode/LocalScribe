#!/usr/bin/env python3
"""Evaluate the optional MLX Fun-ASR Nano port against a fixed manifest.

Run this script with the isolated ``mlx-audio-plus`` environment. The model is
loaded once and never becomes an App backend merely by passing this probe.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "mlx-community/Fun-ASR-Nano-2512-4bit"


def generation_token_limit(duration: float) -> int:
    return min(768, max(64, int(math.ceil(max(duration, 0.0) * 12.0)) + 32))


def has_repetition_risk(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if re.search(r"(.)\1{15,}", compact):
        return True
    if len(compact) < 40:
        return False
    grams = [compact[index : index + 4] for index in range(len(compact) - 3)]
    return bool(grams) and (1.0 - len(set(grams)) / len(grams)) > 0.75


def audio_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate) if info.samplerate else 0.0


def evaluate(manifest_path: Path, out_dir: Path, model_id: str) -> Path:
    from mlx_audio.stt.models.funasr import Model

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    items = [item for item in (manifest.get("items") or []) if isinstance(item, dict)]
    model = Model.from_pretrained(model_id)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        case_id = str(item.get("id") or f"case_{index:04d}")
        audio = Path(str(item.get("audio") or "")).expanduser().resolve()
        case_dir = out_dir / case_id / "funasr_nano"
        case_dir.mkdir(parents=True, exist_ok=True)
        duration = audio_duration(audio)
        max_tokens = generation_token_limit(duration)
        started = time.monotonic()
        row: dict[str, Any] = {
            "id": case_id,
            "audio": str(audio),
            "status": "error",
            "duration": duration,
            "max_tokens": max_tokens,
            "error": "",
        }
        try:
            result = model.generate(
                audio,
                max_tokens=max_tokens,
                temperature=0.0,
                language="zh",
            )
            text = str(result.text or "").strip()
            tokens = len(result.tokens or [])
            chars_per_second = len(text) / duration if duration > 0 else 0.0
            risk = bool(
                tokens >= int(max_tokens * 0.98)
                or chars_per_second > 12.0
                or has_repetition_risk(text)
            )
            payload = {
                "audio": str(audio),
                "backend": "funasr_nano",
                "model_id": model_id,
                "duration": duration,
                "language": result.language or "zh",
                "segments": [{"start": 0.0, "end": duration, "text": text}],
                "filter_stats": {
                    "mode": "funasr_nano_public_eval",
                    "generation_tokens": tokens,
                    "max_tokens": max_tokens,
                    "chars_per_second": round(chars_per_second, 3),
                    "has_hallucination_risk": risk,
                    "timing_reliable": False,
                },
            }
            (case_dir / "result.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (case_dir / "transcript.txt").write_text(text + "\n", encoding="utf-8")
            row.update(
                {
                    "status": "risk" if risk else "ok",
                    "text": text,
                    "tokens": tokens,
                    "chars_per_second": round(chars_per_second, 3),
                    "has_hallucination_risk": risk,
                }
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["cost_seconds"] = round(time.monotonic() - started, 3)
        rows.append(row)
        print(f"[{index}/{len(items)}] {case_id}: {row['status']}", flush=True)

    summary_path = out_dir / "funasr_nano_summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MLX Fun-ASR Nano on a fixed manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    summary = evaluate(
        Path(args.manifest).expanduser().resolve(),
        Path(args.out).expanduser().resolve(),
        args.model,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
