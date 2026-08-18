#!/usr/bin/env python3
"""Evaluate a narrowly gated local CAM++ fallback for diarization VAD gaps.

This script is intentionally evaluation-only. It reads frozen transcript
geometry, never runs ASR, and writes a separate prediction/report under the
requested output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from diarization_continuous_score import annotation_coverage, crop_recordings  # noqa: E402
from diarization_metrics import (  # noqa: E402
    Segment,
    _maximum_weight_assignment,
    evaluate,
    load_annotations,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _round_time(value: Any) -> float:
    return round(float(value), 3)


def _interval_key(uri: str, start: Any, end: Any) -> tuple[str, float, float]:
    return str(uri), _round_time(start), _round_time(end)


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def _source_fingerprint(audio: Path, oracle_num: int) -> str:
    stat = audio.stat()
    payload = {
        "audio": str(audio.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "oracle_num": int(oracle_num),
        "model": "modelscope_campp_speaker_diarization_common",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _local_model_paths(model_root: Path) -> dict[str, Path]:
    candidates = {
        "diarization": [
            model_root / "iic/speech_campplus_speaker-diarization_common",
            model_root / "damo/speech_campplus_speaker-diarization_common",
        ],
        "speaker": [
            model_root / "damo/speech_campplus_sv_zh-cn_16k-common",
            model_root / "iic/speech_campplus_sv_zh-cn_16k-common",
        ],
        "change": [
            model_root / "damo/speech_campplus-transformer_scl_zh-cn_16k-common",
            model_root / "iic/speech_campplus-transformer_scl_zh-cn_16k-common",
        ],
        "vad": [
            model_root / "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            model_root / "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        ],
    }
    resolved: dict[str, Path] = {}
    for name, paths in candidates.items():
        path = next((item for item in paths if item.is_dir()), None)
        if path is None:
            raise FileNotFoundError(f"missing local ModelScope {name} model under {model_root}")
        resolved[name] = path.resolve()
    return resolved


def _make_modelscope_pipeline(paths: dict[str, Path]):
    os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from modelscope.models import Model
    from modelscope.pipelines.audio.segmentation_clustering_pipeline import (
        SegmentationClusteringPipeline,
    )

    model = Model.from_pretrained(str(paths["diarization"]), task="speaker-diarization")
    model.other_config["speaker_model"] = str(paths["speaker"])
    model.other_config["change_locator"] = str(paths["change"])
    model.other_config["vad_model"] = str(paths["vad"])
    return SegmentationClusteringPipeline(model=model)


def _decode_audio_16k(audio: Path) -> np.ndarray:
    """Decode without trimming so transcript timestamps remain unchanged."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required to decode source audio safely")
    decoded = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    samples = np.frombuffer(decoded.stdout, dtype="<f4").copy()
    if samples.size < 16000:
        raise RuntimeError(f"decoded audio is too short: {audio}")
    return samples


def _run_modelscope(
    pipeline,
    *,
    audio: Path,
    oracle_num: int,
    cache_path: Path,
) -> list[dict[str, Any]]:
    fingerprint = _source_fingerprint(audio, oracle_num)
    if cache_path.is_file():
        cached = _read_json(cache_path)
        if cached.get("fingerprint") == fingerprint and isinstance(cached.get("segments"), list):
            return cached["segments"]

    samples = _decode_audio_16k(audio)
    started = time.perf_counter()
    raw = pipeline(samples, oracle_num=int(oracle_num))
    rows: list[dict[str, Any]] = []
    for item in raw.get("text") or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        start, end, label = float(item[0]), float(item[1]), str(item[2])
        if end > start:
            rows.append({"start": start, "end": end, "speaker": f"MS_{label}"})
    if not rows:
        raise RuntimeError(f"ModelScope returned no diarization rows for {audio}")
    _write_json(
        cache_path,
        {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "audio": str(audio),
            "oracle_num": int(oracle_num),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "segments": rows,
        },
    )
    return rows


def _model_votes(
    rows: list[dict[str, Any]], start: float, end: float
) -> tuple[dict[str, float], float, float]:
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    all_intervals: list[tuple[float, float]] = []
    for row in rows:
        row_start = float(row["start"])
        row_end = float(row["end"])
        overlap_start = max(start, row_start)
        overlap_end = min(end, row_end)
        if overlap_end <= overlap_start:
            continue
        speaker = str(row["speaker"])
        intervals[speaker].append((overlap_start, overlap_end))
        all_intervals.append((overlap_start, overlap_end))
    votes = {speaker: _union_duration(values) for speaker, values in intervals.items()}
    covered = _union_duration(all_intervals)
    duration = max(1e-9, end - start)
    return votes, covered / duration, covered


def _cue_window_votes(
    segment: dict[str, Any], cue_index: int, start: float, end: float
) -> tuple[dict[str, float], float]:
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in segment.get("speaker_subsegments") or []:
        if not isinstance(row, dict):
            continue
        speaker = str(row.get("speaker") or "")
        if not speaker:
            continue
        overlap_start = max(start, float(row.get("start") or 0.0))
        overlap_end = min(end, float(row.get("end") or 0.0))
        if overlap_end > overlap_start:
            intervals[speaker].append((overlap_start, overlap_end))
    votes = {speaker: _union_duration(values) for speaker, values in intervals.items()}
    evidence = next(
        (
            dict(row)
            for row in segment.get("speaker_cue_embeddings") or []
            if isinstance(row, dict) and int(row.get("cue_index", -1)) == cue_index
        ),
        None,
    )
    return votes, float((evidence or {}).get("score") or 0.0)


def _cue_evidence(
    segment: dict[str, Any], cue_index: int, start: float, end: float, current: str
) -> dict[str, Any]:
    windows, _ = _cue_window_votes(segment, cue_index, start, end)
    direct = next(
        (
            dict(row)
            for row in segment.get("speaker_cue_embeddings") or []
            if isinstance(row, dict) and int(row.get("cue_index", -1)) == cue_index
        ),
        None,
    )
    total = sum(windows.values())
    ordered = sorted(windows.items(), key=lambda item: (-item[1], item[0]))
    window_speaker = ordered[0][0] if ordered else ""
    window_purity = ordered[0][1] / total if total > 0 else 0.0
    window_coverage = total / max(1e-9, end - start)
    direct_assign = bool(
        direct
        and direct.get("decision") == "assign"
        and str(direct.get("speaker") or "") == current
        and float(direct.get("score") or 0.0) >= 0.70
        and float(direct.get("margin") or 0.0) >= 0.06
    )
    window_assign = bool(
        window_speaker == current
        and window_purity >= 0.85
        and window_coverage >= 0.50
    )
    return {
        "missing": not windows and direct is None,
        "trusted": direct_assign or window_assign,
        "direct": direct,
        "window_speaker": window_speaker,
        "window_purity": window_purity,
        "window_coverage": window_coverage,
    }


def _candidate_cues(
    uri: str,
    candidate: dict[str, Any],
    current_index: dict[tuple[str, float, float], str],
    window_start: float,
    window_end: float,
) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(candidate.get("segments") or []):
        if not isinstance(segment, dict):
            continue
        raw_cues = segment.get("sync_cues")
        if not isinstance(raw_cues, list) or not raw_cues:
            raw_cues = [
                {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": segment.get("text"),
                }
            ]
        for cue_index, cue in enumerate(raw_cues):
            if not isinstance(cue, dict):
                continue
            start = float(cue.get("start") or 0.0)
            end = float(cue.get("end") or start)
            if end <= start or end <= window_start or start >= window_end:
                continue
            key = _interval_key(uri, start, end)
            current = current_index.get(key)
            if not current:
                continue
            evidence = _cue_evidence(segment, cue_index, start, end, current)
            confidence = segment.get("speaker_confidence")
            cues.append(
                {
                    "key": key,
                    "uri": uri,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "text": str(cue.get("text") or ""),
                    "current": current,
                    "segment_index": segment_index,
                    "cue_index": cue_index,
                    "segment_confidence": float(confidence) if confidence is not None else None,
                    "overlap_ratio": float(segment.get("overlap_ratio") or 0.0),
                    "overlap_risk": bool(segment.get("speaker_overlap_risk")),
                    **evidence,
                }
            )
    return cues


def _mapping_from_anchors(
    cues: list[dict[str, Any]], model_rows: list[dict[str, Any]]
) -> tuple[dict[str, str], dict[str, dict[str, float]], dict[tuple[str, str], float]]:
    weights: dict[tuple[str, str], float] = defaultdict(float)
    for cue in cues:
        if (
            not cue["trusted"]
            or cue["overlap_risk"]
            or cue["overlap_ratio"] >= 0.08
            or cue["duration"] < 0.8
        ):
            continue
        votes, coverage, _ = _model_votes(model_rows, cue["start"], cue["end"])
        if coverage < 0.50:
            continue
        for model_speaker, duration in votes.items():
            weights[(model_speaker, cue["current"])] += duration

    model_speakers = sorted({pair[0] for pair in weights})
    current_speakers = sorted({cue["current"] for cue in cues})
    mapping = _maximum_weight_assignment(model_speakers, current_speakers, weights)
    diagnostics: dict[str, dict[str, float]] = {}
    for model_speaker in model_speakers:
        ordered = sorted(
            (
                (current, weights.get((model_speaker, current), 0.0))
                for current in current_speakers
            ),
            key=lambda item: (-item[1], item[0]),
        )
        total = sum(value for _, value in ordered)
        mapped = mapping.get(model_speaker, "")
        mapped_value = weights.get((model_speaker, mapped), 0.0)
        second = max(
            (value for current, value in ordered if current != mapped),
            default=0.0,
        )
        diagnostics[model_speaker] = {
            "anchor_seconds": round(total, 3),
            "mapped_seconds": round(mapped_value, 3),
            "confidence": mapped_value / total if total > 0 else 0.0,
            "margin": (mapped_value - second) / total if total > 0 else 0.0,
        }
    return mapping, diagnostics, dict(weights)


def _prediction_recordings(rows: list[dict[str, Any]]) -> dict[str, list[Segment]]:
    output: dict[str, list[Segment]] = defaultdict(list)
    for row in rows:
        uri = str(row.get("uri") or "")
        speaker = str(row.get("speaker") or "")
        if not uri or not speaker:
            continue
        start = float(row.get("start") or 0.0)
        end = float(row.get("end") or start)
        if end > start:
            output[uri].append(Segment(uri, start, end, speaker))
    return dict(output)


def _metric_snapshot(result) -> dict[str, Any]:
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


def _apply_gate(
    prediction_rows: list[dict[str, Any]],
    proposals: list[dict[str, Any]],
    params: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: dict[tuple[str, float, float], dict[str, Any]] = {}
    for proposal in proposals:
        if (
            proposal["duration"] >= params["min_duration"]
            and proposal["coverage"] >= params["min_coverage"]
            and proposal["purity"] >= params["min_purity"]
            and proposal["mapping_confidence"] >= params["min_mapping_confidence"]
            and proposal["mapping_margin"] >= params["min_mapping_margin"]
            and proposal["target"] != proposal["current"]
        ):
            accepted[tuple(proposal["key"])] = proposal
    output: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for row in prediction_rows:
        copied = dict(row)
        key = _interval_key(str(row.get("uri") or ""), row.get("start"), row.get("end"))
        proposal = accepted.get(key)
        if proposal is not None:
            copied["speaker"] = proposal["target"]
            changes.append(proposal)
        output.append(copied)
    return output, changes


def _parameter_grid() -> Iterable[dict[str, float]]:
    for values in itertools.product(
        (1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
        (0.15, 0.25, 0.35, 0.50, 0.65),
        (0.65, 0.75, 0.85, 0.95),
        (0.55, 0.70, 0.85),
        (0.05, 0.20, 0.40),
    ):
        yield dict(
            zip(
                (
                    "min_duration",
                    "min_coverage",
                    "min_purity",
                    "min_mapping_confidence",
                    "min_mapping_margin",
                ),
                values,
            )
        )


def _parse_sources(values: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        uri, separator, raw_path = value.partition("=")
        if not separator or not uri.strip() or not raw_path.strip():
            raise ValueError(f"invalid --source value: {value!r}; expected URI=PATH")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[uri.strip()] = path
    return sources


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _read_json(args.manifest)
    gold_json = _read_json(args.gold)
    current_json = _read_json(args.current)
    sources = _parse_sources(args.source)
    items = {
        str(item.get("uri") or ""): item
        for item in manifest.get("items") or []
        if isinstance(item, dict)
    }
    if set(items) != set(sources):
        raise ValueError(f"source URIs {sorted(sources)} do not match manifest {sorted(items)}")

    prediction_rows = [dict(row) for row in current_json.get("segments") or []]
    current_index = {
        _interval_key(str(row.get("uri") or ""), row.get("start"), row.get("end")): str(
            row.get("speaker") or ""
        )
        for row in prediction_rows
    }
    candidate_data = {uri: _read_json(path) for uri, path in sources.items()}

    model_root = args.model_root.expanduser().resolve()
    paths = _local_model_paths(model_root)
    pipeline = _make_modelscope_pipeline(paths)
    model_rows_by_uri: dict[str, list[dict[str, Any]]] = {}
    cues_by_uri: dict[str, list[dict[str, Any]]] = {}
    mappings: dict[str, dict[str, str]] = {}
    mapping_diagnostics: dict[str, dict[str, dict[str, float]]] = {}
    proposals: list[dict[str, Any]] = []

    for uri in sorted(sources):
        item = items[uri]
        candidate = candidate_data[uri]
        audio = Path(str(candidate.get("audio") or "")).expanduser().resolve()
        if not audio.is_file():
            raise FileNotFoundError(f"candidate audio missing: {audio}")
        oracle_num = len(
            {
                str(row.get("speaker") or "")
                for row in prediction_rows
                if str(row.get("uri") or "") == uri and row.get("speaker")
            }
        )
        cache_path = args.out / "modelscope_cache" / f"{uri}.json"
        model_rows = _run_modelscope(
            pipeline,
            audio=audio,
            oracle_num=oracle_num,
            cache_path=cache_path,
        )
        cues = _candidate_cues(
            uri,
            candidate,
            current_index,
            float(item.get("window_start") or 0.0),
            float(item.get("window_end") or 0.0),
        )
        mapping, diagnostics, _ = _mapping_from_anchors(cues, model_rows)
        model_rows_by_uri[uri] = model_rows
        cues_by_uri[uri] = cues
        mappings[uri] = mapping
        mapping_diagnostics[uri] = diagnostics

        for cue in cues:
            if not cue["missing"]:
                continue
            votes, coverage, covered_seconds = _model_votes(
                model_rows, cue["start"], cue["end"]
            )
            if not votes:
                continue
            ordered = sorted(votes.items(), key=lambda pair: (-pair[1], pair[0]))
            model_speaker, best_seconds = ordered[0]
            total = sum(votes.values())
            target = mapping.get(model_speaker, "")
            mapping_stats = diagnostics.get(model_speaker) or {}
            if not target:
                continue
            proposals.append(
                {
                    "key": list(cue["key"]),
                    "uri": uri,
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
                    "mapping_confidence": float(mapping_stats.get("confidence") or 0.0),
                    "mapping_margin": float(mapping_stats.get("margin") or 0.0),
                }
            )

    gold = load_annotations(args.gold)
    coverage_windows, _ = annotation_coverage(gold_json)
    gold = crop_recordings(gold, coverage_windows)
    baseline_prediction = crop_recordings(
        _prediction_recordings(prediction_rows), coverage_windows
    )
    baseline_result = evaluate(gold, baseline_prediction)
    baseline = _metric_snapshot(baseline_result)

    arena: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_rows: list[dict[str, Any]] = prediction_rows
    best_changes: list[dict[str, Any]] = []
    for params in _parameter_grid():
        rows, changes = _apply_gate(prediction_rows, proposals, params)
        result = evaluate(
            gold,
            crop_recordings(_prediction_recordings(rows), coverage_windows),
        )
        metrics = _metric_snapshot(result)
        no_recording_regression = all(
            float(metrics["recordings"][uri]["der"] or 0.0)
            <= float(baseline["recordings"][uri]["der"] or 0.0) + 1e-9
            for uri in baseline["recordings"]
        )
        row = {
            "params": params,
            "changed_cues": len(changes),
            "changed_seconds": round(sum(item["duration"] for item in changes), 3),
            "no_recording_der_regression": no_recording_regression,
            "metrics": metrics,
        }
        arena.append(row)
        if not no_recording_regression or not changes:
            continue
        if float(metrics["der"] or 1.0) >= float(baseline["der"] or 0.0) - 1e-9:
            continue
        rank = (
            float(metrics["der"] or 1.0),
            float(metrics["jer"] or 1.0),
            len(changes),
        )
        if best is None or rank < best["rank"]:
            best = {"rank": rank, **row}
            best_rows = rows
            best_changes = changes

    arena.sort(
        key=lambda row: (
            0 if row["no_recording_der_regression"] else 1,
            float(row["metrics"]["der"] or 1.0),
            row["changed_cues"],
        )
    )
    prediction_path = args.out / "best_prediction.json"
    _write_json(
        prediction_path,
        {
            "schema_version": 1,
            "kind": "continuous_diarization_prediction",
            "pack_id": manifest.get("pack_id"),
            "candidate_projection": "modelscope_missing_vad_selective_fallback_eval",
            "segments": best_rows,
        },
    )
    report = {
        "schema_version": 1,
        "evaluation_only": True,
        "frozen_geometry": True,
        "baseline": baseline,
        "best": best,
        "best_changes": best_changes,
        "proposal_count": len(proposals),
        "proposals": proposals,
        "missing_evidence_cues": {
            uri: sum(bool(cue["missing"]) for cue in cues)
            for uri, cues in cues_by_uri.items()
        },
        "mappings": mappings,
        "mapping_diagnostics": mapping_diagnostics,
        "top_candidates": arena[:20],
        "prediction": str(prediction_path),
        "models": {name: str(path) for name, path in paths.items()},
    }
    _write_json(args.out / "selective_fallback_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate ModelScope CAM++ only on cues missing primary VAD evidence"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--source", action="append", default=[], required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path.home() / ".cache/modelscope/hub/models",
    )
    args = parser.parse_args(argv)
    args.out = args.out.expanduser().resolve()
    report = run(args)
    print(
        json.dumps(
            {
                "ok": True,
                "report": str(args.out / "selective_fallback_report.json"),
                "baseline_der": report["baseline"]["der"],
                "best_der": (report.get("best") or {}).get("metrics", {}).get("der"),
                "changed_cues": (report.get("best") or {}).get("changed_cues", 0),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
