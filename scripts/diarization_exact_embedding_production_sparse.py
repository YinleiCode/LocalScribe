#!/usr/bin/env python3
"""Replay the production exact-embedding repair on sparse human annotations."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scribe-py/src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from diarization_gold_regression import collect_sources, evaluate, load_packs  # noqa: E402
from diarization_selective_fallback_sparse import _compare_reports  # noqa: E402
from scribe_py.diarizers.exact_embedding_fallback import (  # noqa: E402
    repair_missing_evidence_cues,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _annotation_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"\((\d+)\)", path.name)
    return (int(match.group(1)) if match else 0, path.name)


def _geometry(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start": segment.get("start"),
            "end": segment.get("end"),
            "text": segment.get("text"),
            "sync_cues": segment.get("sync_cues"),
        }
        for segment in segments
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    annotation_paths = sorted(
        args.annotations_dir.expanduser().resolve().glob(args.annotations_pattern),
        key=_annotation_sort_key,
    )
    if not annotation_paths:
        raise FileNotFoundError("no sparse annotation exports found")
    packs = load_packs(annotation_paths, args.manifest_root.expanduser().resolve())
    sources = collect_sources(packs, "senko")
    current_cache = _read_json(args.current_cache)
    baseline_report = evaluate(packs, sources, current_cache, "senko")
    output_cache = copy.deepcopy(current_cache)
    source_diagnostics: dict[str, Any] = {}
    changed_sources = 0

    for source_key, source in sorted((current_cache.get("sources") or {}).items()):
        if source.get("status") != "ok":
            continue
        audio = Path(str(source.get("audio") or "")).expanduser().resolve()
        if not audio.is_file():
            source_diagnostics[source_key] = {
                "available": False,
                "reason": "audio_missing",
                "audio": str(audio),
            }
            continue
        original_segments = [copy.deepcopy(row) for row in source.get("segments") or []]
        candidate = {
            "segments": original_segments,
            "stats": {"engine": "senko"},
        }
        repaired = repair_missing_evidence_cues(audio, candidate)
        repaired_segments = repaired.get("segments") or []
        if _geometry(repaired_segments) != _geometry(original_segments):
            raise RuntimeError(f"production repair changed transcript geometry: {source_key}")
        stats = dict((repaired.get("stats") or {}).get("exact_embedding_fallback") or {})
        source_diagnostics[source_key] = stats
        if stats.get("applied"):
            changed_sources += 1
        output_cache["sources"][source_key]["segments"] = repaired_segments

    candidate_report = evaluate(packs, sources, output_cache, "senko")
    comparison = _compare_reports(baseline_report, candidate_report)
    report = {
        "schema_version": 1,
        "production_module": True,
        "frozen_geometry": True,
        "comparison": comparison,
        "changed_sources": changed_sources,
        "source_diagnostics": source_diagnostics,
    }
    _write_json(args.out / "production_sparse_prediction_cache.json", output_cache)
    _write_json(args.out / "production_sparse_candidate_report.json", candidate_report)
    _write_json(args.out / "production_sparse_verify_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--annotations-pattern", default="通用分人人工标注*.json")
    parser.add_argument("--manifest-root", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--current-cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    args.out = args.out.expanduser().resolve()
    report = run(args)
    print(json.dumps({
        "ok": bool(report["comparison"]["passed"]),
        **report["comparison"],
        "changed_sources": report["changed_sources"],
        "report": str(args.out / "production_sparse_verify_report.json"),
    }, ensure_ascii=False))
    return 0 if report["comparison"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
