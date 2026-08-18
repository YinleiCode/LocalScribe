#!/usr/bin/env python3
"""Evaluate exact CAM++ embeddings for transcript cues missed by primary VAD."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from diarization_continuous_score import annotation_coverage, crop_recordings  # noqa: E402
from diarization_metrics import Segment, evaluate, load_annotations  # noqa: E402
from diarization_selective_fallback_arena import (  # noqa: E402
    _candidate_cues,
    _decode_audio_16k,
    _interval_key,
    _local_model_paths,
    _read_json,
    _write_json,
)


SAMPLE_RATE = 16_000


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


def _make_sv_pipeline(model_path: Path):
    os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from modelscope.pipelines import pipeline

    return pipeline(task="speaker-verification", model=str(model_path))


def _candidate_current_index(uri: str, candidate: dict[str, Any]) -> dict:
    index: dict[tuple[str, float, float], str] = {}
    for segment in candidate.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        projected = {
            _interval_key(uri, row.get("start"), row.get("end")): str(
                row.get("speaker") or ""
            )
            for row in segment.get("speaker_cues") or []
            if isinstance(row, dict) and row.get("speaker")
        }
        raw_cues = segment.get("sync_cues")
        if not isinstance(raw_cues, list) or not raw_cues:
            raw_cues = [segment]
        for cue in raw_cues:
            if not isinstance(cue, dict):
                continue
            key = _interval_key(uri, cue.get("start"), cue.get("end"))
            index[key] = projected.get(key) or str(segment.get("speaker") or "")
    return index


def _window_samples(samples: np.ndarray, start: float, end: float) -> np.ndarray | None:
    left = max(0, int(round(start * SAMPLE_RATE)))
    right = min(len(samples), int(round(end * SAMPLE_RATE)))
    if right - left < int(1.45 * SAMPLE_RATE):
        return None
    return np.asarray(samples[left:right], dtype=np.float32)


def _rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    centered = values - float(np.mean(values))
    return float(np.sqrt(np.mean(np.square(centered), dtype=np.float64)))


def _energy_windows(
    samples: np.ndarray,
    start: float,
    end: float,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    duration = end - start
    if duration < 1.5:
        return []
    window_duration = min(3.0, duration)
    if duration <= window_duration + 1e-6:
        starts = [start]
    else:
        step = 0.5
        count = max(1, int(math.floor((duration - window_duration) / step)) + 1)
        starts = [start + index * step for index in range(count)]
        final_start = end - window_duration
        if not starts or abs(starts[-1] - final_start) > 0.05:
            starts.append(final_start)
    candidates: list[dict[str, Any]] = []
    for window_start in starts:
        window_end = min(end, window_start + window_duration)
        values = _window_samples(samples, window_start, window_end)
        if values is None:
            continue
        candidates.append(
            {
                "start": window_start,
                "end": window_end,
                "samples": values,
                "rms": _rms(values),
            }
        )
    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: -float(row["rms"])):
        midpoint = (float(candidate["start"]) + float(candidate["end"])) / 2.0
        if any(
            abs(
                midpoint
                - (float(existing["start"]) + float(existing["end"])) / 2.0
            )
            < 1.0
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _anchor_rank(cue: dict[str, Any]) -> tuple[float, ...]:
    direct = cue.get("direct") or {}
    direct_assign = float(
        direct.get("decision") == "assign"
        and str(direct.get("speaker") or "") == cue.get("current")
    )
    return (
        direct_assign,
        float(direct.get("score") or 0.0),
        float(direct.get("margin") or 0.0),
        float(cue.get("window_purity") or 0.0),
        float(cue.get("window_coverage") or 0.0),
        float(cue.get("segment_confidence") or 0.0),
        min(float(cue.get("duration") or 0.0), 8.0),
    )


def _select_anchors(cues: list[dict[str, Any]], limit_per_speaker: int = 24) -> list[dict]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cue in cues:
        if (
            cue.get("trusted")
            and not cue.get("overlap_risk")
            and float(cue.get("overlap_ratio") or 0.0) < 0.08
            and float(cue.get("duration") or 0.0) >= 1.5
            and cue.get("current")
        ):
            grouped[str(cue["current"])].append(cue)
    selected: list[dict[str, Any]] = []
    for speaker, rows in sorted(grouped.items()):
        speaker_rows: list[dict[str, Any]] = []
        for cue in sorted(rows, key=_anchor_rank, reverse=True):
            midpoint = (float(cue["start"]) + float(cue["end"])) / 2.0
            if any(
                abs(midpoint - (float(existing["start"]) + float(existing["end"])) / 2.0)
                < 3.0
                for existing in speaker_rows
            ):
                continue
            speaker_rows.append(cue)
            if len(speaker_rows) >= limit_per_speaker:
                break
        for cue in speaker_rows:
            selected.append({**cue, "anchor_speaker": speaker})
    return selected


def _robust_prototype(values: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        return None, {"available": False, "reason": "fewer_than_3_anchors"}
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    similarities = matrix @ matrix.T
    centrality = np.median(similarities, axis=1)
    medoid_index = int(np.argmax(centrality))
    medoid_scores = similarities[medoid_index]
    keep_count = max(3, int(math.ceil(matrix.shape[0] * 0.60)))
    keep_indices = np.argsort(medoid_scores)[::-1][:keep_count]
    retained = matrix[keep_indices]
    prototype = np.mean(retained, axis=0)
    prototype = prototype / (np.linalg.norm(prototype) + 1e-9)
    retained_scores = retained @ prototype
    return prototype.astype(np.float32), {
        "available": True,
        "input_anchors": int(matrix.shape[0]),
        "retained_anchors": int(retained.shape[0]),
        "medoid_similarity": round(float(centrality[medoid_index]), 4),
        "retained_median_similarity": round(float(np.median(retained_scores)), 4),
        "retained_min_similarity": round(float(np.min(retained_scores)), 4),
    }


def _embedding_cache_fingerprint(audio: Path, anchors: list[dict], targets: list[dict]) -> str:
    stat = audio.stat()
    payload = {
        "audio": str(audio.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "anchors": [list(row["key"]) for row in anchors],
        "targets": [list(row["key"]) for row in targets],
        "method": "modelscope_campp_sv_energy_windows_v1",
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _extract_source_embeddings(
    pipeline,
    *,
    audio: Path,
    anchors: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    cache_path: Path,
) -> dict[str, Any]:
    fingerprint = _embedding_cache_fingerprint(audio, anchors, targets)
    if cache_path.is_file():
        cached = _read_json(cache_path)
        if cached.get("fingerprint") == fingerprint:
            return cached

    samples = _decode_audio_16k(audio)
    descriptors: list[dict[str, Any]] = []
    waveforms: list[np.ndarray] = []
    for anchor in anchors:
        windows = _energy_windows(
            samples, float(anchor["start"]), float(anchor["end"]), limit=1
        )
        for window in windows:
            descriptors.append(
                {
                    "kind": "anchor",
                    "key": list(anchor["key"]),
                    "speaker": anchor["anchor_speaker"],
                    "start": window["start"],
                    "end": window["end"],
                    "rms": window["rms"],
                }
            )
            waveforms.append(window["samples"])
    for target in targets:
        windows = _energy_windows(
            samples, float(target["start"]), float(target["end"]), limit=3
        )
        for window in windows:
            descriptors.append(
                {
                    "kind": "target",
                    "key": list(target["key"]),
                    "current": target["current"],
                    "start": window["start"],
                    "end": window["end"],
                    "rms": window["rms"],
                }
            )
            waveforms.append(window["samples"])
    if not waveforms:
        raise RuntimeError(f"no valid exact embedding windows for {audio}")
    result = pipeline(waveforms, output_emb=True)
    embeddings = np.asarray(result["embs"], dtype=np.float32)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    if embeddings.shape[0] != len(descriptors):
        raise RuntimeError("speaker embedding count mismatch")
    for descriptor, embedding in zip(descriptors, embeddings):
        descriptor["embedding"] = embedding.astype(float).tolist()
    value = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "audio": str(audio),
        "rows": descriptors,
    }
    _write_json(cache_path, value)
    return value


def _proposals_from_embeddings(
    uri: str,
    targets: list[dict[str, Any]],
    cached: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anchor_values: dict[str, list[np.ndarray]] = defaultdict(list)
    target_values: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in cached.get("rows") or []:
        embedding = np.asarray(row.get("embedding") or [], dtype=np.float32)
        if embedding.ndim != 1 or embedding.size == 0:
            continue
        key = tuple(row.get("key") or [])
        if row.get("kind") == "anchor":
            anchor_values[str(row.get("speaker") or "")].append(embedding)
        elif row.get("kind") == "target" and len(key) == 3:
            target_values[key].append({**row, "embedding_value": embedding})

    speaker_ids: list[str] = []
    prototypes: list[np.ndarray] = []
    prototype_stats: dict[str, Any] = {}
    for speaker, values in sorted(anchor_values.items()):
        prototype, stats = _robust_prototype(np.stack(values))
        prototype_stats[speaker] = stats
        if prototype is not None:
            speaker_ids.append(speaker)
            prototypes.append(prototype)
    if len(prototypes) < 2:
        return [], {"available": False, "speakers": prototype_stats}
    prototype_matrix = np.stack(prototypes)
    proposals: list[dict[str, Any]] = []
    target_index = {tuple(row["key"]): row for row in targets}
    for key, windows in target_values.items():
        target = target_index.get(key)
        if target is None:
            continue
        votes: dict[str, int] = defaultdict(int)
        scored: list[dict[str, Any]] = []
        for window in windows:
            embedding = window["embedding_value"]
            scores = embedding @ prototype_matrix.T
            order = np.argsort(scores)[::-1]
            best_index = int(order[0])
            second_index = int(order[1])
            speaker = speaker_ids[best_index]
            votes[speaker] += 1
            scored.append(
                {
                    "speaker": speaker,
                    "score": float(scores[best_index]),
                    "margin": float(scores[best_index] - scores[second_index]),
                    "start": float(window["start"]),
                    "end": float(window["end"]),
                    "rms": float(window["rms"]),
                }
            )
        selected_speaker = sorted(
            votes,
            key=lambda speaker: (
                -votes[speaker],
                -max(row["score"] for row in scored if row["speaker"] == speaker),
                speaker,
            ),
        )[0]
        agreeing = [row for row in scored if row["speaker"] == selected_speaker]
        best = max(agreeing, key=lambda row: (row["score"] + row["margin"], row["rms"]))
        proposals.append(
            {
                "key": list(key),
                "uri": uri,
                "start": target["start"],
                "end": target["end"],
                "duration": target["duration"],
                "text": target["text"],
                "current": target["current"],
                "target": selected_speaker,
                "score": best["score"],
                "margin": best["margin"],
                "window_agreement": len(agreeing) / len(scored),
                "window_count": len(scored),
                "best_window": best,
            }
        )
    return proposals, {"available": True, "speakers": prototype_stats}


def _prediction_recordings(rows: list[dict[str, Any]]) -> dict[str, list[Segment]]:
    output: dict[str, list[Segment]] = defaultdict(list)
    for row in rows:
        uri = str(row.get("uri") or "")
        speaker = str(row.get("speaker") or "")
        start = float(row.get("start") or 0.0)
        end = float(row.get("end") or start)
        if uri and speaker and end > start:
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


def _parameter_grid() -> Iterable[dict[str, float]]:
    for values in itertools.product(
        (2.0, 4.0, 6.0),
        (0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
        (0.02, 0.05, 0.10, 0.15, 0.20),
        (0.50, 0.67, 1.0),
    ):
        yield dict(
            zip(
                ("min_duration", "min_score", "min_margin", "min_window_agreement"),
                values,
            )
        )


def _apply(
    rows: list[dict[str, Any]], proposals: list[dict[str, Any]], params: dict[str, float]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted = {
        tuple(proposal["key"]): proposal
        for proposal in proposals
        if proposal["target"] != proposal["current"]
        and proposal["duration"] >= params["min_duration"]
        and proposal["score"] >= params["min_score"]
        and proposal["margin"] >= params["min_margin"]
        and proposal["window_agreement"] >= params["min_window_agreement"]
    }
    output: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        key = _interval_key(str(row.get("uri") or ""), row.get("start"), row.get("end"))
        proposal = accepted.get(key)
        if proposal is not None:
            copied["speaker"] = proposal["target"]
            changes.append(proposal)
        output.append(copied)
    return output, changes


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
        raise ValueError("source URIs do not match continuous manifest")
    prediction_rows = [dict(row) for row in current_json.get("segments") or []]
    prediction_index = {
        _interval_key(str(row.get("uri") or ""), row.get("start"), row.get("end")): str(
            row.get("speaker") or ""
        )
        for row in prediction_rows
    }

    model_paths = _local_model_paths(args.model_root.expanduser().resolve())
    pipeline = _make_sv_pipeline(model_paths["speaker"])
    all_proposals: list[dict[str, Any]] = []
    source_diagnostics: dict[str, Any] = {}
    for uri, source_path in sorted(sources.items()):
        candidate = _read_json(source_path)
        current_index = _candidate_current_index(uri, candidate)
        current_index.update(
            {key: speaker for key, speaker in prediction_index.items() if key[0] == uri}
        )
        duration = max(
            (float(row.get("end") or 0.0) for row in candidate.get("segments") or []),
            default=0.0,
        )
        all_cues = _candidate_cues(uri, candidate, current_index, 0.0, duration)
        item = items[uri]
        targets = [
            cue
            for cue in all_cues
            if cue["missing"]
            and cue["duration"] >= 1.5
            and cue["end"] > float(item.get("window_start") or 0.0)
            and cue["start"] < float(item.get("window_end") or 0.0)
        ]
        anchors = _select_anchors(all_cues)
        audio = Path(str(candidate.get("audio") or "")).expanduser().resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        cached = _extract_source_embeddings(
            pipeline,
            audio=audio,
            anchors=anchors,
            targets=targets,
            cache_path=args.out / "embedding_cache" / f"{uri}.json",
        )
        proposals, diagnostics = _proposals_from_embeddings(uri, targets, cached)
        all_proposals.extend(proposals)
        source_diagnostics[uri] = {
            "anchors": len(anchors),
            "targets": len(targets),
            "proposals": len(proposals),
            "prototype_diagnostics": diagnostics,
        }

    gold = load_annotations(args.gold)
    coverage_windows, _ = annotation_coverage(gold_json)
    gold = crop_recordings(gold, coverage_windows)
    baseline_result = evaluate(
        gold, crop_recordings(_prediction_recordings(prediction_rows), coverage_windows)
    )
    baseline = _metric_snapshot(baseline_result)
    arena: list[dict[str, Any]] = []
    best = None
    best_rows = prediction_rows
    best_changes: list[dict[str, Any]] = []
    for params in _parameter_grid():
        rows, changes = _apply(prediction_rows, all_proposals, params)
        result = evaluate(
            gold, crop_recordings(_prediction_recordings(rows), coverage_windows)
        )
        metrics = _metric_snapshot(result)
        no_regression = all(
            float(metrics["recordings"][uri]["der"] or 0.0)
            <= float(baseline["recordings"][uri]["der"] or 0.0) + 1e-9
            for uri in baseline["recordings"]
        )
        row = {
            "params": params,
            "changed_cues": len(changes),
            "changed_seconds": round(sum(item["duration"] for item in changes), 3),
            "no_recording_der_regression": no_regression,
            "metrics": metrics,
        }
        arena.append(row)
        if (
            not no_regression
            or not changes
            or float(metrics["der"] or 1.0) >= float(baseline["der"] or 0.0) - 1e-9
        ):
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
    prediction_path = args.out / "best_exact_embedding_prediction.json"
    _write_json(
        prediction_path,
        {
            "schema_version": 1,
            "kind": "continuous_diarization_prediction",
            "pack_id": manifest.get("pack_id"),
            "candidate_projection": "campp_exact_embedding_missing_vad_eval",
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
        "proposals": all_proposals,
        "source_diagnostics": source_diagnostics,
        "top_candidates": arena[:20],
        "prediction": str(prediction_path),
        "speaker_model": str(model_paths["speaker"]),
    }
    _write_json(args.out / "exact_embedding_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
                "baseline_der": report["baseline"]["der"],
                "best_der": (report.get("best") or {}).get("metrics", {}).get("der"),
                "changed_cues": (report.get("best") or {}).get("changed_cues", 0),
                "report": str(args.out / "exact_embedding_report.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
