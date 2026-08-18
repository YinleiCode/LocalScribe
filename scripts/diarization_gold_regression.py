#!/usr/bin/env python3
"""Replay exported speaker annotations against the current diarization pipeline.

The script is evaluation-only. It never writes transcript JSON and never runs
ASR. Current predictions are cached after every source recording so a long run
can resume safely.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import itertools
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scribe-py" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from diarization_review_pack import _timeline_rows  # noqa: E402
from diarization_review_score import build_rows, normalize_sequence  # noqa: E402
from scribe_py.ipc import handle_recommend_diarization  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in (
        "scribe-py/src/scribe_py/ipc.py",
        "scribe-py/src/scribe_py/diarizers/senko_diarizer.py",
        "scribe-py/src/scribe_py/diarizers/__init__.py",
    ):
        path = PROJECT_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _geometry_projection(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start": row.get("start"),
            "end": row.get("end"),
            "text": row.get("text"),
            "sync_cues": row.get("sync_cues"),
        }
        for row in segments
    ]


def _geometry_hash(segments: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        _geometry_projection(segments),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_key(audio: Path, segments: list[dict[str, Any]], engine: str) -> str:
    stat = audio.stat()
    payload = {
        "audio": str(audio.resolve()),
        "audio_size": stat.st_size,
        "audio_mtime_ns": stat.st_mtime_ns,
        "geometry": _geometry_hash(segments),
        "engine": engine,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def index_manifests(root: Path) -> dict[str, Path]:
    manifests: dict[str, Path] = {}
    for path in sorted(root.rglob("通用分人验收清单.json")):
        value = _read_json(path)
        pack_id = str(value.get("pack_id") or "")
        if not pack_id:
            continue
        previous = manifests.get(pack_id)
        if previous is not None and previous.resolve() != path.resolve():
            raise ValueError(f"duplicate manifest pack_id {pack_id}: {previous}, {path}")
        manifests[pack_id] = path
    return manifests


def load_packs(annotation_paths: list[Path], manifest_root: Path) -> list[dict[str, Any]]:
    manifest_index = index_manifests(manifest_root)
    packs: list[dict[str, Any]] = []
    seen_pack_ids: set[str] = set()
    for annotation_path in annotation_paths:
        annotations = _read_json(annotation_path)
        pack_id = str(annotations.get("pack_id") or "")
        if not pack_id:
            raise ValueError(f"annotation has no pack_id: {annotation_path}")
        if pack_id in seen_pack_ids:
            raise ValueError(f"duplicate annotation export for pack_id {pack_id}")
        manifest_path = manifest_index.get(pack_id)
        if manifest_path is None:
            raise FileNotFoundError(f"manifest not found for pack_id {pack_id}")
        manifest = _read_json(manifest_path)
        rows = build_rows(manifest, annotations)
        packs.append(
            {
                "pack_id": pack_id,
                "annotation_path": annotation_path,
                "manifest_path": manifest_path,
                "annotations": annotations,
                "manifest": manifest,
                "rows": rows,
            }
        )
        seen_pack_ids.add(pack_id)
    return packs


def _pick_recommended_candidate(result: dict[str, Any]) -> dict[str, Any]:
    candidates = [row for row in result.get("candidates") or [] if isinstance(row, dict)]
    requested = int(result.get("recommended_candidate_n_speakers") or 0)
    candidate = next(
        (row for row in candidates if int(row.get("n_speakers") or 0) == requested),
        None,
    )
    if candidate is None:
        raise RuntimeError(
            f"recommended candidate {requested} missing from {len(candidates)} candidates"
        )
    return candidate


def _source_for_item(item: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    transcript = Path(str(item.get("transcript") or "")).expanduser().resolve()
    audio = Path(str(item.get("audio") or "")).expanduser().resolve()
    if not transcript.is_file():
        raise FileNotFoundError(f"transcript not found: {transcript}")
    if not audio.is_file():
        raise FileNotFoundError(f"audio not found: {audio}")
    data = _read_json(transcript)
    segments = data.get("segments") or []
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"transcript has no segments: {transcript}")
    return transcript, audio, data


def collect_sources(packs: list[dict[str, Any]], engine: str) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for pack in packs:
        for item in pack["manifest"].get("items") or []:
            if not isinstance(item, dict):
                continue
            transcript, audio, data = _source_for_item(item)
            segments = data.get("segments") or []
            key = _source_key(audio, segments, engine)
            source = sources.setdefault(
                key,
                {
                    "source_key": key,
                    "audio": audio,
                    "transcript": transcript,
                    "segments": segments,
                    "geometry_sha256": _geometry_hash(segments),
                    "references": [],
                },
            )
            source["references"].append(
                {
                    "pack_id": pack["pack_id"],
                    "recording": str(item.get("recording") or ""),
                    "item_id": str(item.get("id") or ""),
                }
            )
    return sources


def run_current_sources(
    sources: dict[str, dict[str, Any]],
    *,
    engine: str,
    cache_path: Path,
    runtime_fingerprint: str,
) -> dict[str, Any]:
    cache: dict[str, Any] = {
        "schema_version": 1,
        "runtime_fingerprint": runtime_fingerprint,
        "engine": engine,
        "sources": {},
    }
    if cache_path.is_file():
        previous = _read_json(cache_path)
        if (
            previous.get("runtime_fingerprint") == runtime_fingerprint
            and previous.get("engine") == engine
            and isinstance(previous.get("sources"), dict)
        ):
            cache = previous

    total = len(sources)
    for number, (key, source) in enumerate(sorted(sources.items()), start=1):
        existing = cache["sources"].get(key)
        if isinstance(existing, dict) and existing.get("status") == "ok":
            print(f"[{number}/{total}] cache {source['audio'].name}", flush=True)
            continue
        print(f"[{number}/{total}] run {source['audio'].name}", flush=True)
        started = time.perf_counter()
        row: dict[str, Any] = {
            "status": "error",
            "audio": str(source["audio"]),
            "transcript": str(source["transcript"]),
            "geometry_sha256_before": source["geometry_sha256"],
            "references": source["references"],
        }
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = handle_recommend_diarization(
                    {
                        "audio": str(source["audio"]),
                        "segments": source["segments"],
                        "min_speakers": 2,
                        "max_speakers": 8,
                        "profiles": [],
                        "engine": engine,
                        "preserve_segmentation": True,
                    }
                )
            candidate = _pick_recommended_candidate(result)
            current_segments = candidate.get("segments") or []
            after_hash = _geometry_hash(current_segments)
            if after_hash != source["geometry_sha256"]:
                raise RuntimeError("current diarization changed transcript geometry")
            row.update(
                {
                    "status": "ok",
                    "recommended_n_speakers": result.get("recommended_n_speakers"),
                    "recommended_candidate_n_speakers": result.get(
                        "recommended_candidate_n_speakers"
                    ),
                    "confidence": result.get("confidence"),
                    "confidence_reason": result.get("confidence_reason"),
                    "geometry_sha256_after": after_hash,
                    "segments": current_segments,
                    "candidate_stats": candidate.get("stats") or {},
                    "errors": result.get("errors") or [],
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        cache["sources"][key] = row
        _write_json_atomic(cache_path, cache)
    return cache


def _sequence_for_window(segments: list[dict[str, Any]], start: Any, end: Any) -> str:
    rows = _timeline_rows(
        {"segments": segments},
        float(start or 0.0),
        float(end or 0.0),
    )
    tokens: list[str] = []
    for row in rows:
        token = normalize_sequence(row.get("speaker"))
        for part in token.split("->") if token else []:
            if not tokens or tokens[-1] != part:
                tokens.append(part)
    return "->".join(tokens)


def _collapse_sequence(sequence: Any) -> str:
    output: list[str] = []
    for token in normalize_sequence(sequence).split("->"):
        if token and (not output or output[-1] != token):
            output.append(token)
    return "->".join(output)


def _best_label_mapping(pairs: list[tuple[str, str]]) -> dict[str, str]:
    current_labels = sorted(
        {
            token
            for current, _gold in pairs
            for token in normalize_sequence(current).split("->")
            if token
        }
    )
    gold_labels = sorted(
        {
            token
            for _current, gold in pairs
            for token in normalize_sequence(gold).split("->")
            if token
        }
    )
    if not current_labels or not gold_labels:
        return {label: label for label in current_labels}

    confusion: Counter[tuple[str, str]] = Counter()
    for current, gold in pairs:
        current_tokens = normalize_sequence(current).split("->")
        gold_tokens = normalize_sequence(gold).split("->")
        if len(current_tokens) != len(gold_tokens):
            continue
        confusion.update(zip(current_tokens, gold_tokens))

    targets = list(gold_labels)
    while len(targets) < len(current_labels):
        targets.append(f"__UNMAPPED_{len(targets) + 1}")
    best_score = -1
    best_mapping: dict[str, str] = {}
    for selected in itertools.permutations(targets, len(current_labels)):
        mapping = dict(zip(current_labels, selected))
        score = sum(confusion[(source, target)] for source, target in mapping.items())
        if score > best_score:
            best_score = score
            best_mapping = mapping
    return best_mapping


def _apply_mapping(sequence: str, mapping: dict[str, str]) -> str:
    output: list[str] = []
    for token in normalize_sequence(sequence).split("->"):
        if not token:
            continue
        mapped = mapping.get(token, f"?{token}")
        if mapped.startswith("__UNMAPPED_"):
            mapped = f"?{token}"
        if not output or output[-1] != mapped:
            output.append(mapped)
    return "->".join(output)


def _deduplicate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    selected: dict[tuple[str, int, int], dict[str, Any]] = {}
    duplicate_count = 0
    for row in rows:
        key = (
            str(row.get("audio_sha256") or row.get("source_key") or ""),
            int(round(float(row.get("review_start") or 0.0) * 1000)),
            int(round(float(row.get("review_end") or 0.0) * 1000)),
        )
        previous = selected.get(key)
        if previous is None:
            selected[key] = row
            continue
        duplicate_count += 1

        def rank(value: dict[str, Any]) -> tuple[int, int, str, str]:
            return (
                int(bool(value.get("verdict"))),
                int(bool(value.get("gold_turn_sequence"))),
                str(value.get("annotation_exported_at") or ""),
                str(value.get("pack_id") or ""),
            )

        if rank(row) > rank(previous):
            selected[key] = row
    return sorted(
        selected.values(),
        key=lambda row: (
            str(row.get("recording") or ""),
            float(row.get("review_start") or 0.0),
            str(row.get("id") or ""),
        ),
    ), duplicate_count


def _sequence_size(sequence: Any) -> int:
    normalized = _collapse_sequence(sequence)
    return len(normalized.split("->")) if normalized else 0


def _current_error_type(row: dict[str, Any]) -> str:
    if row.get("current_turn_correct"):
        return ""
    current_size = _sequence_size(row.get("current_prediction"))
    gold_size = _sequence_size(row.get("gold_turn_sequence"))
    if current_size < gold_size:
        return "漏掉换人"
    if current_size > gold_size:
        return "额外误切"
    return "人员或顺序错误"


def _turn_metrics_by(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "")].append(row)
    output: list[dict[str, Any]] = []
    for label, group in sorted(groups.items()):
        scorable = [row for row in group if row.get("gold_turn_sequence")]
        baseline_correct = sum(bool(row.get("baseline_turn_correct")) for row in scorable)
        current_correct = sum(bool(row.get("current_turn_correct")) for row in scorable)
        output.append(
            {
                key: label,
                "items": len(group),
                "turn_scorable": len(scorable),
                "baseline_correct": baseline_correct,
                "baseline_accuracy": round(baseline_correct / len(scorable), 4)
                if scorable
                else None,
                "current_correct": current_correct,
                "current_accuracy": round(current_correct / len(scorable), 4)
                if scorable
                else None,
                "repaired": sum(row.get("comparison") == "repaired" for row in group),
                "regressions": sum(row.get("comparison") == "regression" for row in group),
            }
        )
    return output


def evaluate(packs: list[dict[str, Any]], sources: dict[str, dict[str, Any]], cache: dict[str, Any], engine: str) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    source_cache = cache.get("sources") or {}
    audio_hashes: dict[Path, str] = {}
    for pack in packs:
        manifest_items = {
            str(item.get("id") or ""): item
            for item in pack["manifest"].get("items") or []
            if isinstance(item, dict)
        }
        for baseline in pack["rows"]:
            item = manifest_items[baseline["id"]]
            _transcript, audio, data = _source_for_item(item)
            if audio not in audio_hashes:
                audio_hashes[audio] = _sha256_file(audio)
            key = _source_key(audio, data.get("segments") or [], engine)
            current_source = source_cache.get(key) or {}
            current_raw = ""
            if current_source.get("status") == "ok":
                current_raw = _sequence_for_window(
                    current_source.get("segments") or [],
                    baseline.get("review_start"),
                    baseline.get("review_end"),
                )
            evaluated.append(
                {
                    **baseline,
                    "pack_id": pack["pack_id"],
                    "annotation_file": str(pack["annotation_path"]),
                    "manifest_file": str(pack["manifest_path"]),
                    "source_key": key,
                    "audio_sha256": audio_hashes[audio],
                    "annotation_exported_at": str(
                        pack["annotations"].get("exported_at") or ""
                    ),
                    "current_prediction_raw": current_raw,
                    "current_source_status": current_source.get("status", "missing"),
                    "current_source_error": current_source.get("error", ""),
                    "baseline_turn_prediction": _collapse_sequence(
                        baseline.get("baseline_detailed_prediction")
                    ),
                    "gold_turn_sequence": _collapse_sequence(baseline.get("gold_sequence")),
                }
            )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        groups[(row["pack_id"], row["recording"])].append(row)
    mappings: dict[str, dict[str, str]] = {}
    for (pack_id, recording), rows in groups.items():
        pairs = [
            (row["current_prediction_raw"], row["gold_turn_sequence"])
            for row in rows
            if row.get("current_prediction_raw") and row.get("gold_turn_sequence")
        ]
        mapping = _best_label_mapping(pairs)
        mappings[f"{pack_id}:{recording}"] = mapping
        for row in rows:
            row["current_prediction"] = _apply_mapping(
                row.get("current_prediction_raw") or "", mapping
            )
            scorable = bool(row.get("gold_turn_sequence"))
            row["baseline_turn_correct"] = bool(
                scorable
                and row.get("baseline_turn_prediction") == row.get("gold_turn_sequence")
            )
            row["current_turn_correct"] = bool(
                scorable and row.get("current_prediction") == row.get("gold_turn_sequence")
            )
            if not scorable:
                row["comparison"] = "not_exactly_scorable"
            elif row["baseline_turn_correct"] and row["current_turn_correct"]:
                row["comparison"] = "kept_correct"
            elif row["baseline_turn_correct"] and not row["current_turn_correct"]:
                row["comparison"] = "regression"
            elif not row["baseline_turn_correct"] and row["current_turn_correct"]:
                row["comparison"] = "repaired"
            else:
                row["comparison"] = "still_wrong"
            row["current_error_type"] = _current_error_type(row) if scorable else ""

    deduplicated, duplicate_count = _deduplicate_rows(evaluated)
    completed = [row for row in deduplicated if row["verdict"]]
    strict = [row for row in deduplicated if row["verdict"] not in {"", "uncertain"}]
    turn_scorable = [row for row in deduplicated if row.get("gold_turn_sequence")]
    baseline_turn_correct = sum(row["baseline_turn_correct"] for row in turn_scorable)
    current_turn_correct = sum(row["current_turn_correct"] for row in turn_scorable)
    comparisons = Counter(row["comparison"] for row in deduplicated)
    source_rows = list(source_cache.values())
    source_ok = [row for row in source_rows if row.get("status") == "ok"]

    by_recording = []
    recording_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in deduplicated:
        recording_groups[row["recording"]].append(row)
    for recording, rows in sorted(recording_groups.items()):
        recording_turns = [row for row in rows if row.get("gold_turn_sequence")]
        current_correct = sum(row.get("current_turn_correct") for row in recording_turns)
        baseline_correct = sum(row.get("baseline_turn_correct") for row in recording_turns)
        by_recording.append(
            {
                "recording": recording,
                "items": len(rows),
                "completed": sum(bool(row.get("verdict")) for row in rows),
                "turn_scorable": len(recording_turns),
                "baseline_turn_correct": baseline_correct,
                "baseline_turn_accuracy": round(baseline_correct / len(recording_turns), 4)
                if recording_turns
                else None,
                "current_turn_correct": current_correct,
                "current_turn_accuracy": round(current_correct / len(recording_turns), 4)
                if recording_turns
                else None,
                "repaired": sum(row.get("comparison") == "repaired" for row in rows),
                "regressions": sum(row.get("comparison") == "regression" for row in rows),
            }
        )
    by_category = _turn_metrics_by(deduplicated, "category")
    current_error_types = Counter(
        row.get("current_error_type")
        for row in turn_scorable
        if row.get("current_error_type")
    )

    return {
        "schema_version": 1,
        "limitations": [
            "这是稀疏人工抽检的片段级完整序列准确率，不是全录音 DER/JER。",
            "当前标签已在每个标注包/录音内做最优一一置换，A/B/C 字母顺序本身不计错。",
            "人员错、漏拆或切点错但未填写完整正确序列的项目，只计入历史严格结果，不能判断当前版本是否修复。",
        ],
        "summary": {
            "annotation_files": len(packs),
            "exported_items": len(evaluated),
            "deduplicated_items": len(deduplicated),
            "duplicate_items_removed": duplicate_count,
            "completed": len(completed),
            "pending": len(deduplicated) - len(completed),
            "historical_strict_scorable": len(strict),
            "historical_strict_correct": sum(row["verdict"] == "correct" for row in strict),
            "historical_strict_errors": sum(row["verdict"] != "correct" for row in strict),
            "historical_strict_accuracy": round(
                sum(row["verdict"] == "correct" for row in strict) / len(strict), 4
            )
            if strict
            else None,
            "turn_scorable": len(turn_scorable),
            "baseline_turn_correct": baseline_turn_correct,
            "baseline_turn_accuracy": round(baseline_turn_correct / len(turn_scorable), 4)
            if turn_scorable
            else None,
            "current_turn_correct": current_turn_correct,
            "current_turn_accuracy": round(current_turn_correct / len(turn_scorable), 4)
            if turn_scorable
            else None,
            "kept_correct": comparisons["kept_correct"],
            "repaired": comparisons["repaired"],
            "regressions": comparisons["regression"],
            "still_wrong": comparisons["still_wrong"],
            "not_exactly_scorable": comparisons["not_exactly_scorable"],
            "current_sources": len(source_rows),
            "current_sources_ok": len(source_ok),
            "current_sources_failed": len(source_rows) - len(source_ok),
            "geometry_preserved_sources": sum(
                row.get("geometry_sha256_before") == row.get("geometry_sha256_after")
                for row in source_ok
            ),
            "current_elapsed_seconds": round(
                sum(float(row.get("elapsed_seconds") or 0.0) for row in source_rows), 3
                ),
            "current_error_types": dict(current_error_types),
        },
        "verdict_counts": dict(Counter(row["verdict"] or "pending" for row in deduplicated)),
        "by_recording": by_recording,
        "by_category": by_category,
        "label_mappings": mappings,
        "current_source_summary": [
            {
                key: row.get(key)
                for key in (
                    "status",
                    "audio",
                    "transcript",
                    "recommended_n_speakers",
                    "recommended_candidate_n_speakers",
                    "confidence",
                    "elapsed_seconds",
                    "geometry_sha256_before",
                    "geometry_sha256_after",
                    "error",
                )
                if row.get(key) is not None
            }
            for row in source_rows
        ],
        "rows": evaluated,
        "deduplicated_rows": deduplicated,
    }


def _percent(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value) * 100:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# 中文真实录音人声分离回归验收\n\n",
        "## 结论摘要\n\n",
        f"- 人工标注文件：{summary['annotation_files']} 份，共导出 {summary['exported_items']} 个片段。\n",
        f"- 按音频内容与时间窗去重后：{summary['deduplicated_items']} 个；移除重复 {summary['duplicate_items_removed']} 个。\n",
        f"- 已判断：{summary['completed']}；待标注：{summary['pending']}。\n",
        f"- 历史严格结果：{summary['historical_strict_correct']}/{summary['historical_strict_scorable']}，通过率 {_percent(summary['historical_strict_accuracy'])}。\n",
        f"- 可做人员轮次序列同窗比较：{summary['turn_scorable']} 个。\n",
        f"- 历史版本人员轮次：{summary['baseline_turn_correct']}/{summary['turn_scorable']}，准确率 {_percent(summary['baseline_turn_accuracy'])}。\n",
        f"- 当前版本人员轮次：{summary['current_turn_correct']}/{summary['turn_scorable']}，准确率 {_percent(summary['current_turn_accuracy'])}。\n",
        f"- 当前相对变化：保持正确 {summary['kept_correct']}，修复 {summary['repaired']}，回归 {summary['regressions']}，仍错误 {summary['still_wrong']}。\n",
        f"- 当前错误结构：{ '，'.join(f'{key} {value}' for key, value in summary['current_error_types'].items()) or '无' }。\n",
        f"- 当前分人源：{summary['current_sources_ok']}/{summary['current_sources']} 成功；转录几何保持 {summary['geometry_preserved_sources']}/{summary['current_sources_ok']}。\n\n",
        "## 口径限制\n\n",
    ]
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}\n")
    lines.extend(
        [
            "\n## 按录音\n\n",
            "| 录音 | 片段 | 已判断 | 完整序列 | 历史准确率 | 当前准确率 | 修复 | 回归 |\n",
            "|---|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in report["by_recording"]:
        lines.append(
            f"| {row['recording']} | {row['items']} | {row['completed']} | {row['turn_scorable']} | "
            f"{_percent(row['baseline_turn_accuracy'])} | {_percent(row['current_turn_accuracy'])} | "
            f"{row['repaired']} | {row['regressions']} |\n"
        )
    lines.extend(
        [
            "\n## 按场景\n\n",
            "| 场景 | 片段 | 可比较 | 历史准确率 | 当前准确率 | 修复 | 回归 |\n",
            "|---|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in report["by_category"]:
        lines.append(
            f"| {row['category']} | {row['items']} | {row['turn_scorable']} | "
            f"{_percent(row['baseline_accuracy'])} | {_percent(row['current_accuracy'])} | "
            f"{row['repaired']} | {row['regressions']} |\n"
        )
    lines.extend(
        [
            "\n## 当前错误与回归\n\n",
            "| ID | 录音 | 场景 | 历史 | 当前原标签 | 当前对齐后 | 人工真值 | 错误类型 | 结果 |\n",
            "|---|---|---|---|---|---|---|---|---|\n",
        ]
    )
    for row in report["deduplicated_rows"]:
        if row.get("comparison") not in {"regression", "still_wrong"}:
            continue
        lines.append(
            f"| {row['id']} | {row['recording']} | {row['category']} | "
            f"{row['baseline_turn_prediction']} | {row['current_prediction_raw']} | "
            f"{row['current_prediction']} | {row['gold_turn_sequence']} | "
            f"{row['current_error_type']} | {row['comparison']} |\n"
        )
    lines.extend(
        [
            "\n## 下一道验收\n\n",
            "当前报告用于稀疏片段回归。要计算 DER/JER，还需在录音 3、录音 10 和一份未参与开发的真实多人录音上，各标注一段连续 10-15 分钟说话人时间真值。\n",
        ]
    )
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="汇总历史人工标注并回归当前分人算法")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path.home() / "Downloads",
    )
    parser.add_argument(
        "--annotations-pattern",
        default="通用分人人工标注*.json",
    )
    parser.add_argument("--manifest-root", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", choices=["senko", "auto"], default="senko")
    parser.add_argument("--audit-only", action="store_true", help="只审计标注，不运行当前分人")
    args = parser.parse_args(argv)

    annotation_paths = sorted(
        args.annotations_dir.expanduser().resolve().glob(args.annotations_pattern),
        key=lambda path: (
            int(match.group(1)) if (match := re.search(r"\((\d+)\)", path.name)) else 0,
            path.name,
        ),
    )
    if not annotation_paths:
        raise SystemExit("没有找到人工标注导出 JSON")
    packs = load_packs(annotation_paths, args.manifest_root.expanduser().resolve())
    sources = collect_sources(packs, args.engine)
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _runtime_fingerprint()
    cache_path = out_dir / "current_prediction_cache.json"
    if args.audit_only:
        cache = {
            "schema_version": 1,
            "runtime_fingerprint": fingerprint,
            "engine": args.engine,
            "sources": {},
        }
    else:
        cache = run_current_sources(
            sources,
            engine=args.engine,
            cache_path=cache_path,
            runtime_fingerprint=fingerprint,
        )
    report = evaluate(packs, sources, cache, args.engine)
    report.update(
        {
            "runtime_fingerprint": fingerprint,
            "engine": args.engine,
            "annotation_files": [str(path) for path in annotation_paths],
        }
    )
    json_path = out_dir / "中文真实录音分人回归验收.json"
    markdown_path = out_dir / "中文真实录音分人回归验收.md"
    _write_json_atomic(json_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "json": str(json_path),
                "markdown": str(markdown_path),
                "cache": str(cache_path),
                "summary": report["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
