#!/usr/bin/env python3
"""Replay the selective CAM++ VAD-gap fallback on sparse human annotations."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from diarization_gold_regression import (  # noqa: E402
    collect_sources,
    evaluate as evaluate_sparse,
    load_packs,
)
from diarization_selective_fallback_arena import (  # noqa: E402
    _candidate_cues,
    _interval_key,
    _local_model_paths,
    _make_modelscope_pipeline,
    _mapping_from_anchors,
    _model_votes,
    _read_json,
    _run_modelscope,
    _source_fingerprint,
    _write_json,
)


def _annotation_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"\((\d+)\)", path.name)
    return (int(match.group(1)) if match else 0, path.name)


def _current_cue_index(source_key: str, segments: list[dict[str, Any]]) -> dict:
    index: dict[tuple[str, float, float], str] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        projected = {
            _interval_key(
                source_key,
                row.get("start"),
                row.get("end"),
            ): str(row.get("speaker") or "")
            for row in segment.get("speaker_cues") or []
            if isinstance(row, dict) and row.get("speaker")
        }
        sync_cues = segment.get("sync_cues")
        if not isinstance(sync_cues, list) or not sync_cues:
            sync_cues = [segment]
        for cue in sync_cues:
            if not isinstance(cue, dict):
                continue
            key = _interval_key(source_key, cue.get("start"), cue.get("end"))
            index[key] = projected.get(key) or str(segment.get("speaker") or "")
    return index


def _window_overlap(cue: dict[str, Any], windows: list[tuple[float, float]]) -> bool:
    return any(
        min(float(cue["end"]), end) - max(float(cue["start"]), start) > 0.0
        for start, end in windows
    )


def _find_existing_cache(
    cache_dirs: list[Path], audio: Path, oracle_num: int
) -> Path | None:
    expected = _source_fingerprint(audio, oracle_num)
    for directory in cache_dirs:
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
            try:
                value = _read_json(path)
            except Exception:
                continue
            if value.get("fingerprint") == expected and isinstance(value.get("segments"), list):
                return path
    return None


def _proposal(
    cue: dict[str, Any],
    model_rows: list[dict[str, Any]],
    mapping: dict[str, str],
    diagnostics: dict[str, dict[str, float]],
) -> dict[str, Any] | None:
    votes, coverage, covered_seconds = _model_votes(
        model_rows, float(cue["start"]), float(cue["end"])
    )
    if not votes:
        return None
    ordered = sorted(votes.items(), key=lambda pair: (-pair[1], pair[0]))
    model_speaker, best_seconds = ordered[0]
    target = mapping.get(model_speaker, "")
    if not target:
        return None
    total = sum(votes.values())
    stats = diagnostics.get(model_speaker) or {}
    return {
        "key": list(cue["key"]),
        "uri": cue["uri"],
        "start": cue["start"],
        "end": cue["end"],
        "duration": cue["duration"],
        "text": cue["text"],
        "current": cue["current"],
        "target": target,
        "model_speaker": model_speaker,
        "coverage": coverage,
        "covered_seconds": covered_seconds,
        "purity": best_seconds / total if total > 0 else 0.0,
        "mapping_confidence": float(stats.get("confidence") or 0.0),
        "mapping_margin": float(stats.get("margin") or 0.0),
    }


def _accepted(proposal: dict[str, Any], gates: dict[str, float]) -> bool:
    return bool(
        proposal["target"] != proposal["current"]
        and proposal["duration"] >= gates["min_duration"]
        and proposal["coverage"] >= gates["min_coverage"]
        and proposal["purity"] >= gates["min_purity"]
        and proposal["mapping_confidence"] >= gates["min_mapping_confidence"]
        and proposal["mapping_margin"] >= gates["min_mapping_margin"]
    )


def _materialize_cue_changes(
    source_key: str,
    segments: list[dict[str, Any]],
    changes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accepted = {tuple(row["key"]): row for row in changes}
    output: list[dict[str, Any]] = []
    for segment in segments:
        copied = copy.deepcopy(segment)
        sync_cues = copied.get("sync_cues")
        if not isinstance(sync_cues, list) or not sync_cues:
            output.append(copied)
            continue
        existing = {
            _interval_key(source_key, row.get("start"), row.get("end")): dict(row)
            for row in copied.get("speaker_cues") or []
            if isinstance(row, dict)
        }
        cue_rows: list[dict[str, Any]] = []
        segment_changed = False
        for cue_index, cue in enumerate(sync_cues):
            if not isinstance(cue, dict):
                continue
            key = _interval_key(source_key, cue.get("start"), cue.get("end"))
            row = existing.get(key) or {
                "cue_index": cue_index,
                "start": float(cue.get("start") or 0.0),
                "end": float(cue.get("end") or 0.0),
                "text": str(cue.get("text") or ""),
                "speaker": str(copied.get("speaker") or ""),
                "confidence": float(copied.get("speaker_confidence") or 0.5),
                "source": "whole_segment_inheritance",
                "review": False,
            }
            proposal = accepted.get(key)
            if proposal is not None:
                row["speaker"] = proposal["target"]
                row["confidence"] = round(float(proposal["mapping_confidence"]), 3)
                row["source"] = "modelscope_missing_vad_selective_fallback"
                row["review"] = False
                segment_changed = True
            cue_rows.append(row)
        if segment_changed:
            copied["speaker_cues"] = cue_rows
            copied["speaker_cue_mode"] = "modelscope_missing_vad_selective_fallback"
        output.append(copied)
    return output


def _row_identity(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("audio_sha256") or row.get("source_key") or ""),
        int(round(float(row.get("review_start") or 0.0) * 1000)),
        int(round(float(row.get("review_end") or 0.0) * 1000)),
    )


def _compare_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_rows = {
        _row_identity(row): row
        for row in baseline.get("deduplicated_rows") or []
        if row.get("gold_turn_sequence")
    }
    candidate_rows = {
        _row_identity(row): row
        for row in candidate.get("deduplicated_rows") or []
        if row.get("gold_turn_sequence")
    }
    repaired: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []
    for key, before in baseline_rows.items():
        after = candidate_rows.get(key)
        if after is None:
            continue
        payload = {
            "recording": before.get("recording"),
            "start": before.get("review_start"),
            "end": before.get("review_end"),
            "before": before.get("current_prediction"),
            "after": after.get("current_prediction"),
            "gold": before.get("gold_turn_sequence"),
        }
        if not before.get("current_turn_correct") and after.get("current_turn_correct"):
            repaired.append(payload)
        elif before.get("current_turn_correct") and not after.get("current_turn_correct"):
            regressed.append(payload)
    return {
        "baseline_correct": baseline["summary"]["current_turn_correct"],
        "candidate_correct": candidate["summary"]["current_turn_correct"],
        "turn_scorable": candidate["summary"]["turn_scorable"],
        "repaired": repaired,
        "regressed": regressed,
        "passed": not regressed
        and candidate["summary"]["current_turn_correct"]
        >= baseline["summary"]["current_turn_correct"],
    }


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
    baseline_report = evaluate_sparse(packs, sources, current_cache, "senko")

    windows_by_source: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in baseline_report.get("deduplicated_rows") or []:
        if row.get("gold_turn_sequence"):
            windows_by_source[str(row.get("source_key") or "")].append(
                (float(row.get("review_start") or 0.0), float(row.get("review_end") or 0.0))
            )

    gates = {
        "min_duration": args.min_duration,
        "min_coverage": args.min_coverage,
        "min_purity": args.min_purity,
        "min_mapping_confidence": args.min_mapping_confidence,
        "min_mapping_margin": args.min_mapping_margin,
    }
    output_cache = copy.deepcopy(current_cache)
    model_paths = _local_model_paths(args.model_root.expanduser().resolve())
    pipeline = None
    all_proposals: list[dict[str, Any]] = []
    all_changes: list[dict[str, Any]] = []
    evaluated_sources: list[dict[str, Any]] = []
    cache_dirs = [path.expanduser().resolve() for path in args.existing_model_cache]

    for source_key, source in sorted((current_cache.get("sources") or {}).items()):
        windows = windows_by_source.get(source_key) or []
        if not windows or source.get("status") != "ok":
            continue
        segments = [dict(row) for row in source.get("segments") or []]
        current_index = _current_cue_index(source_key, segments)
        duration = max((float(row.get("end") or 0.0) for row in segments), default=0.0)
        cues = _candidate_cues(
            source_key,
            {"segments": segments},
            current_index,
            0.0,
            duration,
        )
        relevant = [
            cue
            for cue in cues
            if cue["missing"]
            and cue["duration"] >= gates["min_duration"]
            and _window_overlap(cue, windows)
        ]
        if not relevant:
            continue

        audio = Path(str(source.get("audio") or "")).expanduser().resolve()
        oracle_num = int(source.get("recommended_candidate_n_speakers") or 0)
        if not audio.is_file() or oracle_num < 1:
            continue
        existing = _find_existing_cache(cache_dirs, audio, oracle_num)
        cache_path = existing or (args.out / "modelscope_cache" / f"{source_key}.json")
        if pipeline is None and not cache_path.is_file():
            pipeline = _make_modelscope_pipeline(model_paths)
        if pipeline is None:
            # A cached result does not use the pipeline object.
            class _UnusedPipeline:
                def __call__(self, *_args, **_kwargs):
                    raise RuntimeError("unexpected cache miss")

            active_pipeline = _UnusedPipeline()
        else:
            active_pipeline = pipeline
        model_rows = _run_modelscope(
            active_pipeline,
            audio=audio,
            oracle_num=oracle_num,
            cache_path=cache_path,
        )
        mapping, diagnostics, _ = _mapping_from_anchors(cues, model_rows)
        proposals = [
            proposal
            for cue in relevant
            if (proposal := _proposal(cue, model_rows, mapping, diagnostics)) is not None
        ]
        changes = [proposal for proposal in proposals if _accepted(proposal, gates)]
        all_proposals.extend(proposals)
        all_changes.extend(changes)
        evaluated_sources.append(
            {
                "source_key": source_key,
                "audio": str(audio),
                "relevant_missing_cues": len(relevant),
                "proposals": len(proposals),
                "accepted": len(changes),
                "mapping": mapping,
                "mapping_diagnostics": diagnostics,
                "model_cache": str(cache_path),
            }
        )
        if changes:
            output_cache["sources"][source_key]["segments"] = _materialize_cue_changes(
                source_key, segments, changes
            )

    candidate_report = evaluate_sparse(packs, sources, output_cache, "senko")
    comparison = _compare_reports(baseline_report, candidate_report)
    args.out.mkdir(parents=True, exist_ok=True)
    _write_json(args.out / "hybrid_prediction_cache.json", output_cache)
    _write_json(args.out / "sparse_candidate_report.json", candidate_report)
    report = {
        "schema_version": 1,
        "evaluation_only": True,
        "gates": gates,
        "comparison": comparison,
        "evaluated_sources": evaluated_sources,
        "proposals": all_proposals,
        "accepted_changes": all_changes,
        "candidate_report": str(args.out / "sparse_candidate_report.json"),
        "prediction_cache": str(args.out / "hybrid_prediction_cache.json"),
    }
    _write_json(args.out / "selective_fallback_sparse_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--annotations-pattern", default="通用分人人工标注*.json")
    parser.add_argument("--manifest-root", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--current-cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path.home() / ".cache/modelscope/hub/models",
    )
    parser.add_argument("--existing-model-cache", action="append", type=Path, default=[])
    parser.add_argument("--min-duration", type=float, default=4.0)
    parser.add_argument("--min-coverage", type=float, default=0.25)
    parser.add_argument("--min-purity", type=float, default=0.65)
    parser.add_argument("--min-mapping-confidence", type=float, default=0.85)
    parser.add_argument("--min-mapping-margin", type=float, default=0.40)
    args = parser.parse_args(argv)
    args.out = args.out.expanduser().resolve()
    report = run(args)
    print(
        json.dumps(
            {
                "ok": True,
                **report["comparison"],
                "evaluated_sources": len(report["evaluated_sources"]),
                "accepted_changes": len(report["accepted_changes"]),
                "report": str(args.out / "selective_fallback_sparse_report.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
