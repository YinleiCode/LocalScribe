#!/usr/bin/env python3
"""Replay the atomic ASR consensus rule against saved model evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIBE_SRC = ROOT / "scribe-py" / "src"
if str(SCRIBE_SRC) not in sys.path:
    sys.path.insert(0, str(SCRIBE_SRC))

from scribe_py.core.strong_asr import (  # noqa: E402
    aligned_independent_consensus_rewrite,
    atomic_aligned_independent_consensus_rewrite,
    consensus_rewrite,
    independent_consensus_rewrite,
    phonetic_near_independent_consensus_rewrite,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _candidate_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        Path(str(item.get("audio") or "")).name,
        round(float(item.get("start") or 0.0), 3),
        str(item.get("from") or ""),
        str(item.get("to") or ""),
    )


def _replay_atomic_stage(
    diagnostic: dict[str, Any],
    evidence: dict[str, Any],
    *,
    global_context: str,
) -> list[dict[str, Any]]:
    if (
        diagnostic.get("audit_only")
        or diagnostic.get("technical_review_only")
        or not diagnostic.get("qwen_review_eligible")
        or diagnostic.get("qwen_skipped_reason")
    ):
        return []
    primary = str(evidence.get("primary") or "")
    paraformer = str(evidence.get("paraformer") or "")
    primary_redecode = str(evidence.get("primary_redecode") or "")
    qwen = str(evidence.get("qwen") or "")
    hallucination_risk = bool(diagnostic.get("qwen_hallucination_risk"))
    corrected, changes = consensus_rewrite(
        primary,
        primary_redecode,
        qwen,
        detector_text=paraformer,
        qwen_hallucination_risk=hallucination_risk,
        candidate_qwen_similarity=diagnostic.get("candidate_qwen_similarity"),
        primary_qwen_similarity=diagnostic.get("primary_qwen_similarity"),
        detector_qwen_similarity=diagnostic.get("detector_qwen_similarity"),
        global_context_text=global_context,
    )
    if not changes:
        corrected, changes = independent_consensus_rewrite(
            primary,
            paraformer,
            qwen,
            qwen_hallucination_risk=hallucination_risk,
            global_context_text=global_context,
        )
    if not changes:
        corrected, changes = phonetic_near_independent_consensus_rewrite(
            primary,
            paraformer,
            qwen,
            qwen_hallucination_risk=hallucination_risk,
            global_context_text=global_context,
        )
    length_preserving = all(
        len(str(item.get("normalized_from") or ""))
        == len(str(item.get("normalized_to") or ""))
        for item in changes
    )
    if changes and not length_preserving:
        return []
    corrected, _aligned_changes = aligned_independent_consensus_rewrite(
        corrected,
        paraformer,
        qwen,
        qwen_hallucination_risk=hallucination_risk,
        global_context_text=global_context,
    )
    _corrected, atomic_changes = atomic_aligned_independent_consensus_rewrite(
        corrected,
        paraformer,
        qwen,
        qwen_hallucination_risk=hallucination_risk,
        global_context_text=global_context,
    )
    return atomic_changes


def replay(paths: list[Path]) -> dict[str, Any]:
    files_scanned = 0
    files_with_evidence = 0
    windows_scanned = 0
    candidates: list[dict[str, Any]] = []

    for path in sorted(set(item.resolve() for item in paths)):
        files_scanned += 1
        data = _read_json(path)
        if data is None:
            continue
        strong = ((data.get("filter_stats") or {}).get("strong_asr") or {})
        diagnostics = strong.get("window_diagnostics")
        if not isinstance(diagnostics, list):
            continue
        files_with_evidence += 1
        global_context = "\n".join(
            str(segment.get("text") or "")
            for segment in data.get("segments") or []
            if isinstance(segment, dict)
        )
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue
            evidence = diagnostic.get("candidates")
            if not isinstance(evidence, dict):
                continue
            primary = str(evidence.get("primary") or "")
            paraformer = str(evidence.get("paraformer") or "")
            qwen = str(evidence.get("qwen") or "")
            if not primary or not paraformer or not qwen:
                continue
            windows_scanned += 1
            changes = _replay_atomic_stage(
                diagnostic,
                evidence,
                global_context=global_context,
            )
            for change in changes:
                candidates.append({
                    "file": str(path),
                    "audio": str(data.get("audio") or ""),
                    "window": diagnostic.get("window"),
                    "start": diagnostic.get("start"),
                    "end": diagnostic.get("end"),
                    "from": str(change.get("from") or ""),
                    "to": str(change.get("to") or ""),
                    "left_context": str(change.get("left_context") or ""),
                    "right_context": str(change.get("right_context") or ""),
                    "stored_consensus_path": str(
                        diagnostic.get("consensus_path") or ""
                    ),
                    "already_applied": "atomic_aligned" in str(
                        diagnostic.get("consensus_path") or ""
                    ),
                    "primary": primary,
                    "paraformer": paraformer,
                    "qwen": qwen,
                })

    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in candidates:
        key = _candidate_key(item)
        previous = unique.get(key)
        if previous is None or (
            not previous.get("already_applied") and item.get("already_applied")
        ):
            unique[key] = item
    unique_candidates = list(unique.values())
    unique_candidates.sort(
        key=lambda item: (
            Path(str(item.get("audio") or "")).name,
            float(item.get("start") or 0.0),
            str(item.get("from") or ""),
        )
    )
    return {
        "schema_version": 1,
        "mode": "atomic_aligned_independent_consensus_replay",
        "files_scanned": files_scanned,
        "files_with_saved_model_evidence": files_with_evidence,
        "windows_scanned": windows_scanned,
        "candidate_occurrences": len(candidates),
        "unique_candidate_count": len(unique_candidates),
        "new_unique_candidate_count": sum(
            not item["already_applied"] for item in unique_candidates
        ),
        "candidates": unique_candidates,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ASR 原子共识历史证据重放\n\n",
        f"- 扫描 JSON: {report['files_scanned']}\n",
        f"- 含多模型证据: {report['files_with_saved_model_evidence']}\n",
        f"- 重放窗口: {report['windows_scanned']}\n",
        f"- 唯一候选: {report['unique_candidate_count']}\n",
        f"- 历史结果中尚未应用: {report['new_unique_candidate_count']}\n\n",
        "| 音频 | 时间 | 替换 | 已应用 | 左右上下文 |\n",
        "|---|---:|---|---|---|\n",
    ]
    for item in report["candidates"]:
        audio = Path(str(item.get("audio") or "")).name.replace("|", "/")
        start = float(item.get("start") or 0.0)
        source = str(item.get("from") or "").replace("|", "/")
        target = str(item.get("to") or "").replace("|", "/")
        context = (
            str(item.get("left_context") or "")
            + "["
            + source
            + "→"
            + target
            + "]"
            + str(item.get("right_context") or "")
        ).replace("|", "/")
        lines.append(
            f"| {audio} | {start:.3f}s | {source} → {target} | "
            f"{'是' if item.get('already_applied') else '否'} | {context} |\n"
        )
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="对保存的 Paraformer/Qwen 证据只读重放原子共识规则"
    )
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        import jieba  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "jieba is required for the production token gate; run this script "
            "with LocalScribe's .venv or bundled Python"
        ) from exc

    paths: list[Path] = []
    for root in args.root:
        resolved = root.expanduser().resolve()
        if resolved.is_file():
            paths.append(resolved)
        elif resolved.is_dir():
            paths.extend(resolved.rglob("*.json"))
    report = replay(paths)
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "files_scanned": report["files_scanned"],
        "files_with_saved_model_evidence": report[
            "files_with_saved_model_evidence"
        ],
        "windows_scanned": report["windows_scanned"],
        "unique_candidate_count": report["unique_candidate_count"],
        "new_unique_candidate_count": report["new_unique_candidate_count"],
        "out": str(out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
