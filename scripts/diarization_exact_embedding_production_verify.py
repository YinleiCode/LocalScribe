#!/usr/bin/env python3
"""Score the production missing-evidence repair against frozen diarization truth."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scribe-py/src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from diarization_continuous_score import annotation_coverage, crop_recordings  # noqa: E402
from diarization_metrics import Segment, evaluate, load_annotations  # noqa: E402
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


def _key(uri: str, start: Any, end: Any) -> tuple[str, float, float]:
    return str(uri), round(float(start), 3), round(float(end), 3)


def _parse_sources(values: list[str]) -> dict[str, tuple[Path, str | None]]:
    sources: dict[str, tuple[Path, str | None]] = {}
    for value in values:
        uri, separator, raw_path = value.partition("=")
        if not separator or not uri.strip() or not raw_path.strip():
            raise ValueError(f"invalid --source: {value!r}; expected URI=PATH")
        path_value, fragment, source_key = raw_path.rpartition("#")
        path = Path(path_value if fragment else raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[uri.strip()] = (path, source_key.strip() if fragment else None)
    return sources


def _read_source_candidate(spec: tuple[Path, str | None]) -> dict[str, Any]:
    path, source_key = spec
    value = _read_json(path)
    if source_key is None:
        return value
    source = (value.get("sources") or {}).get(source_key)
    if not isinstance(source, dict):
        raise KeyError(f"source cache key not found: {source_key} in {path}")
    return dict(source)


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


def _exact_changes(uri: str, candidate: dict[str, Any]) -> dict[tuple[str, float, float], str]:
    changes: dict[tuple[str, float, float], str] = {}
    for segment in candidate.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        for cue in segment.get("speaker_cues") or []:
            if (
                isinstance(cue, dict)
                and cue.get("source") == "campp_exact_missing_evidence"
                and cue.get("speaker")
            ):
                changes[_key(uri, cue.get("start"), cue.get("end"))] = str(cue["speaker"])
    return changes


def _recordings(rows: list[dict[str, Any]]) -> dict[str, list[Segment]]:
    output: dict[str, list[Segment]] = defaultdict(list)
    for row in rows:
        uri = str(row.get("uri") or "")
        speaker = str(row.get("speaker") or "")
        start = float(row.get("start") or 0.0)
        end = float(row.get("end") or start)
        if uri and speaker and end > start:
            output[uri].append(Segment(uri, start, end, speaker))
    return dict(output)


def _snapshot(result) -> dict[str, Any]:
    aggregate = result.aggregate()
    return {
        "der": aggregate["der"],
        "jer": aggregate["jer"],
        "speaker_count_absolute_error": aggregate["speaker_count_absolute_error"],
        "recordings": {
            item.uri: {
                "der": item.der,
                "jer": item.jer,
                "confusion_s": item.confusion_s,
                "speaker_count": item.prediction_speaker_count,
            }
            for item in result.recordings
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _read_json(args.manifest)
    gold_json = _read_json(args.gold)
    current_json = _read_json(args.current)
    sources = _parse_sources(args.source)
    manifest_uris = {
        str(item.get("uri") or "")
        for item in manifest.get("items") or []
        if isinstance(item, dict)
    }
    if set(sources) != manifest_uris:
        raise ValueError("source URIs do not match manifest")

    baseline_rows = [dict(row) for row in current_json.get("segments") or []]
    output_rows = [dict(row) for row in baseline_rows]
    output_index = {
        _key(str(row.get("uri") or ""), row.get("start"), row.get("end")): row
        for row in output_rows
    }
    diagnostics: dict[str, Any] = {}
    all_changes: list[dict[str, Any]] = []
    for uri, source_spec in sorted(sources.items()):
        candidate = _read_source_candidate(source_spec)
        candidate["stats"] = dict(candidate.get("stats") or {})
        candidate["stats"].setdefault("engine", "senko")
        original_geometry = _geometry(candidate.get("segments") or [])
        audio = Path(str(candidate.get("audio") or "")).expanduser().resolve()
        repaired = repair_missing_evidence_cues(audio, candidate)
        if _geometry(repaired.get("segments") or []) != original_geometry:
            raise RuntimeError(f"production repair changed frozen transcript geometry: {uri}")
        stats = dict((repaired.get("stats") or {}).get("exact_embedding_fallback") or {})
        diagnostics[uri] = stats
        for cue_key, speaker in _exact_changes(uri, repaired).items():
            row = output_index.get(cue_key)
            if row is None:
                raise KeyError(f"repaired cue missing from frozen prediction: {cue_key}")
            previous = str(row.get("speaker") or "")
            row["speaker"] = speaker
            all_changes.append({
                "uri": uri,
                "start": cue_key[1],
                "end": cue_key[2],
                "from": previous,
                "to": speaker,
            })

    gold = load_annotations(args.gold)
    coverage_windows, _ = annotation_coverage(gold_json)
    gold = crop_recordings(gold, coverage_windows)
    baseline = _snapshot(
        evaluate(gold, crop_recordings(_recordings(baseline_rows), coverage_windows))
    )
    production = _snapshot(
        evaluate(gold, crop_recordings(_recordings(output_rows), coverage_windows))
    )
    prediction_path = args.out / "production_exact_embedding_prediction.json"
    _write_json(
        prediction_path,
        {
            "schema_version": 1,
            "kind": "continuous_diarization_prediction",
            "pack_id": manifest.get("pack_id"),
            "candidate_projection": "production_campp_exact_missing_evidence_v1",
            "segments": output_rows,
        },
    )
    report = {
        "schema_version": 1,
        "production_module": True,
        "frozen_geometry": True,
        "baseline": baseline,
        "production": production,
        "changes": all_changes,
        "source_diagnostics": diagnostics,
        "prediction": str(prediction_path),
    }
    _write_json(args.out / "production_verify_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--source", action="append", required=True, default=[])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    args.out = args.out.expanduser().resolve()
    report = run(args)
    print(json.dumps({
        "ok": True,
        "baseline_der": report["baseline"]["der"],
        "production_der": report["production"]["der"],
        "baseline_jer": report["baseline"]["jer"],
        "production_jer": report["production"]["jer"],
        "changed_cues": len(report["changes"]),
        "report": str(args.out / "production_verify_report.json"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
