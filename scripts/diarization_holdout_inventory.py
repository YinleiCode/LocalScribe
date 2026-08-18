#!/usr/bin/env python3
"""Build a content-hash-isolated audio holdout inventory.

Selection never inspects transcript text, recording names, or diarization
predictions. Any candidate whose bytes appeared in historical artifacts is
excluded before deterministic sampling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


AUDIO_SUFFIXES = {
    ".aac",
    ".amr",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}


def _resolved(path: Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


def _audio_files(roots: Iterable[Path], *, min_bytes: int) -> list[Path]:
    found: set[Path] = set()
    for raw_root in roots:
        root = _resolved(raw_root)
        paths = [root] if root.is_file() else root.rglob("*") if root.is_dir() else []
        for path in paths:
            try:
                if (
                    path.is_file()
                    and path.suffix.lower() in AUDIO_SUFFIXES
                    and path.stat().st_size >= min_bytes
                ):
                    found.add(path.resolve())
            except OSError:
                continue
    return sorted(found)


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _json_audio_references(roots: Iterable[Path]) -> tuple[set[Path], int, int]:
    references: set[Path] = set()
    json_files = 0
    unreadable = 0
    for raw_root in roots:
        root = _resolved(raw_root)
        paths = [root] if root.is_file() else root.rglob("*.json") if root.is_dir() else []
        for json_path in paths:
            if json_path.suffix.lower() != ".json":
                continue
            json_files += 1
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                unreadable += 1
                continue
            for value in _walk_strings(payload):
                raw = os.path.expandvars(value).strip()
                if not raw or Path(raw).suffix.lower() not in AUDIO_SUFFIXES:
                    continue
                candidate = Path(raw).expanduser()
                if candidate.is_absolute():
                    references.add(candidate.resolve())
                    continue
                references.add((json_path.parent / candidate).resolve())
                references.add((Path.cwd() / candidate).resolve())
    return references, json_files, unreadable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _duration_seconds(path: Path, ffprobe: str | None) -> float | None:
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return round(float(result.stdout.strip()), 3)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _selection_rank(seed: str, digest: str) -> str:
    return hashlib.sha256(f"{seed}:{digest}".encode("utf-8")).hexdigest()


def build_inventory(
    *,
    candidate_roots: list[Path],
    history_roots: list[Path],
    min_bytes: int = 500_000,
    min_duration_seconds: float = 120.0,
    max_duration_seconds: float = 3_600.0,
    select_count: int = 5,
    seed: str = "localscribe-holdout-v1",
    ffprobe: str | None = None,
) -> dict[str, Any]:
    candidates = _audio_files(candidate_roots, min_bytes=min_bytes)
    history_references, json_files, unreadable_json = _json_audio_references(history_roots)
    history_direct = _audio_files(history_roots, min_bytes=1)
    candidate_sizes = {path.stat().st_size for path in candidates}

    history_paths = history_references | set(history_direct)
    history_hashes: set[str] = set()
    hashed_history_files = 0
    for path in sorted(history_paths):
        try:
            if path.is_file() and path.stat().st_size in candidate_sizes:
                history_hashes.add(_sha256(path))
                hashed_history_files += 1
        except OSError:
            continue

    by_digest: dict[str, list[Path]] = {}
    for path in candidates:
        try:
            by_digest.setdefault(_sha256(path), []).append(path)
        except OSError:
            continue

    probe = ffprobe or shutil.which("ffprobe")
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for digest, duplicate_paths in sorted(by_digest.items()):
        canonical = max(duplicate_paths, key=lambda path: path.stat().st_mtime_ns)
        stat = canonical.stat()
        row = {
            "path": str(canonical),
            "sha256": digest,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
            "duplicate_paths": [str(path) for path in sorted(duplicate_paths)],
        }
        if any(path in history_paths for path in duplicate_paths):
            row["exclusion_reason"] = "historical_path_reference"
            excluded.append(row)
            continue
        if digest in history_hashes:
            row["exclusion_reason"] = "historical_content_hash"
            excluded.append(row)
            continue
        row["duration_seconds"] = _duration_seconds(canonical, probe)
        eligible.append(row)

    selection_pool = [
        row
        for row in eligible
        if row["duration_seconds"] is not None
        and min_duration_seconds <= float(row["duration_seconds"]) <= max_duration_seconds
    ]
    selection_pool.sort(key=lambda row: _selection_rank(seed, row["sha256"]))
    selected = selection_pool[: max(0, select_count)]
    selected_hashes = {row["sha256"] for row in selected}
    for row in eligible:
        row["selected"] = row["sha256"] in selected_hashes

    return {
        "schema_version": 1,
        "mode": "content_hash_isolated_diarization_holdout",
        "selection_policy": {
            "uses_transcript_text": False,
            "uses_recording_name_semantics": False,
            "uses_diarization_prediction": False,
            "seed": seed,
            "min_bytes": min_bytes,
            "min_duration_seconds": min_duration_seconds,
            "max_duration_seconds": max_duration_seconds,
            "select_count": select_count,
        },
        "candidate_roots": [str(_resolved(path)) for path in candidate_roots],
        "history_roots": [str(_resolved(path)) for path in history_roots],
        "summary": {
            "candidate_files": len(candidates),
            "candidate_unique_hashes": len(by_digest),
            "history_json_files": json_files,
            "history_json_unreadable": unreadable_json,
            "history_paths": len(history_paths),
            "history_files_hashed": hashed_history_files,
            "history_unique_hashes": len(history_hashes),
            "excluded_unique_audio": len(excluded),
            "eligible_unique_audio": len(eligible),
            "selection_pool": len(selection_pool),
            "selected": len(selected),
        },
        "selected": selected,
        "eligible": eligible,
        "excluded": excluded,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# LocalScribe 人声分离盲测清单\n\n",
        "- 选择过程不读取转录文字、文件名语义或当前分人结果。\n",
        "- 历史路径或音频内容 SHA-256 命中即排除。\n",
        f"- 候选文件：{summary['candidate_files']}；唯一音频：{summary['candidate_unique_hashes']}。\n",
        f"- 排除：{summary['excluded_unique_audio']}；可用：{summary['eligible_unique_audio']}；入选：{summary['selected']}。\n\n",
        "| 编号 | 时长 | 大小 | SHA-256 | 路径 |\n",
        "|---:|---:|---:|---|---|\n",
    ]
    for index, row in enumerate(report["selected"], start=1):
        duration = row.get("duration_seconds")
        duration_text = f"{float(duration) / 60:.1f} 分钟" if duration is not None else "未知"
        size_text = f"{int(row['size_bytes']) / 1024 / 1024:.1f} MB"
        lines.append(
            f"| {index} | {duration_text} | {size_text} | {row['sha256'][:12]} | {row['path']} |\n"
        )
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按内容哈希建立未参与开发的人声分离盲测集")
    parser.add_argument("--candidate-root", action="append", required=True, type=Path)
    parser.add_argument("--history-root", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-bytes", type=int, default=500_000)
    parser.add_argument("--min-duration-seconds", type=float, default=120.0)
    parser.add_argument("--max-duration-seconds", type=float, default=3_600.0)
    parser.add_argument("--select-count", type=int, default=5)
    parser.add_argument("--seed", default="localscribe-holdout-v1")
    args = parser.parse_args(argv)

    report = build_inventory(
        candidate_roots=args.candidate_root,
        history_roots=args.history_root,
        min_bytes=args.min_bytes,
        min_duration_seconds=args.min_duration_seconds,
        max_duration_seconds=args.max_duration_seconds,
        select_count=args.select_count,
        seed=args.seed,
    )
    out = _resolved(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out), "summary": report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
