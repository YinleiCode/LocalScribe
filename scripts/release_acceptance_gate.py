#!/usr/bin/env python3
"""Run fail-closed structural release checks for known and unseen recordings."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from asr_cursor_freeze_gate import evaluate_gate  # noqa: E402
from transcript_sync_validate import validate_transcript  # noqa: E402


def _path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser().resolve()


def _labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=/path/to/transcript.json")
    return label.strip(), _path(raw_path.strip())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"file not found: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def evaluate_case(
    label: str,
    candidate_path: Path,
    *,
    baseline_path: Path | None = None,
    require_baseline: bool = False,
) -> dict[str, Any]:
    data = _read_json(candidate_path)
    sync, sync_errors = validate_transcript(candidate_path)
    failures = [
        {"code": "cursor_sync_invalid", "message": message}
        for message in sync_errors
    ]
    warnings: list[dict[str, str]] = []

    if require_baseline and baseline_path is None:
        failures.append({
            "code": "required_baseline_missing",
            "message": "已知验收录音没有提供冻结基线",
        })

    quality = data.get("asr_quality") or {}
    traditional_hits = list(quality.get("traditional_char_hits") or [])
    if traditional_hits:
        failures.append({
            "code": "traditional_chinese_present",
            "message": "仍存在繁体字: " + "、".join(str(item) for item in traditional_hits),
        })

    filter_stats = data.get("filter_stats") or {}
    audio_quality = quality.get("audio_quality") or filter_stats.get("audio_quality") or {}
    audio_risk = str(audio_quality.get("risk_level") or "unknown").lower()
    strong_review = filter_stats.get("strong_asr") or {}
    review_visible = bool(
        strong_review.get("review_recommended")
        or strong_review.get("enabled")
        or strong_review.get("applied")
    )
    if audio_risk == "high" and not review_visible:
        failures.append({
            "code": "high_risk_audio_not_surfaced",
            "message": "音频质量为 high，但没有触发本地复核或明确复核提示",
        })

    segments = list(data.get("segments") or [])
    speaker_count = len({str(row.get("speaker")) for row in segments if row.get("speaker")})
    diarization = data.get("diarization_stats") or {}
    if speaker_count:
        runtime_backend = str(diarization.get("runtime_backend") or "")
        if not runtime_backend:
            failures.append({
                "code": "diarization_backend_missing",
                "message": "存在说话人标签，但没有记录实际分人后端",
            })
        if diarization.get("applied") is False or str(diarization.get("status") or "ok") != "ok":
            failures.append({
                "code": "diarization_not_applied",
                "message": str(diarization.get("failure_reason") or "分人结果未正常应用"),
            })
        if diarization.get("segmentation_preserved") is False:
            failures.append({
                "code": "asr_geometry_not_preserved",
                "message": "分人结果改变了冻结 ASR 段落几何",
            })
        if diarization.get("fallback_reason") or diarization.get("vad_fallback_reason"):
            warnings.append({
                "code": "diarization_fallback_used",
                "message": str(
                    diarization.get("fallback_reason")
                    or diarization.get("vad_fallback_reason")
                ),
            })
        risk_level = str(diarization.get("risk_level") or "unknown")
        if risk_level in {"high", "medium"}:
            warnings.append({
                "code": "diarization_review_required",
                "message": str(diarization.get("risk_reason") or f"分人风险为 {risk_level}"),
            })

    frozen_gate = None
    if baseline_path is not None:
        if baseline_path.resolve() == candidate_path.resolve():
            failures.append({
                "code": "self_baseline_rejected",
                "message": "候选文件不能把自身作为冻结基线",
            })
        else:
            frozen_gate = evaluate_gate(
                baseline_path,
                candidate_path,
                text_mode="exact",
                require_segment_geometry=True,
            )
            if not frozen_gate.get("ok"):
                failures.append({
                    "code": "frozen_asr_or_cursor_regression",
                    "message": "候选结果与冻结 ASR/光标基线不一致",
                })

    return {
        "label": label,
        "candidate": str(candidate_path),
        "baseline": str(baseline_path) if baseline_path else None,
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "summary": {
            **sync,
            "backend": data.get("backend") or "",
            "model_id": data.get("model_id") or "",
            "audio_risk": audio_risk,
            "strong_review_visible": review_visible,
            "traditional_char_hits": traditional_hits,
            "speaker_count": speaker_count,
            "diarization_backend": diarization.get("runtime_backend") or "",
            "diarization_risk": diarization.get("risk_level") or "",
            "speaker_cue_segments": sum(bool(row.get("speaker_cues")) for row in segments),
        },
        "frozen_gate": frozen_gate,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# LocalScribe 发布验收门禁\n\n",
        f"- 总状态: {'通过' if report['ok'] else '失败'}\n",
        f"- 录音数: {len(report['cases'])}\n",
        f"- 失败录音: {report['failed_cases']}\n\n",
        "| 录音 | 状态 | 光标 | 音频风险 | 高噪声提示 | 分人数 | 分人后端 | 分人风险 | 失败原因 |\n",
        "|---|---|---|---|---|---:|---|---|---|\n",
    ]
    for case in report["cases"]:
        summary = case["summary"]
        failures = "；".join(item["message"] for item in case["failures"])
        lines.append(
            f"| {case['label']} | {'通过' if case['ok'] else '失败'} | {summary['status']} | "
            f"{summary['audio_risk']} | {'是' if summary['strong_review_visible'] else '否'} | "
            f"{summary['speaker_count']} | {summary['diarization_backend']} | "
            f"{summary['diarization_risk']} | {failures} |\n"
        )
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查未知录音结构质量和已知录音冻结基线")
    parser.add_argument("--case", action="append", required=True, type=_labeled_path)
    parser.add_argument("--baseline", action="append", default=[], type=_labeled_path)
    parser.add_argument(
        "--require-baseline",
        action="store_true",
        help="要求每个 --case 都提供独立冻结基线",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    baselines = dict(args.baseline)
    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    for label, candidate in args.case:
        try:
            cases.append(evaluate_case(
                label,
                candidate,
                baseline_path=baselines.get(label),
                require_baseline=args.require_baseline,
            ))
        except ValueError as exc:
            errors.append(str(exc))
            cases.append({
                "label": label,
                "candidate": str(candidate),
                "baseline": str(baselines[label]) if label in baselines else None,
                "ok": False,
                "failures": [{"code": "case_unreadable", "message": str(exc)}],
                "warnings": [],
                "summary": {
                    "status": "FAIL",
                    "audio_risk": "unknown",
                    "strong_review_visible": False,
                    "speaker_count": 0,
                    "diarization_backend": "",
                    "diarization_risk": "",
                },
                "frozen_gate": None,
            })

    report = {
        "schema_version": 1,
        "mode": "release_acceptance_gate",
        "ok": bool(cases) and all(case["ok"] for case in cases) and not errors,
        "failed_cases": sum(not case["ok"] for case in cases),
        "errors": errors,
        "cases": cases,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        out = args.out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
        out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(rendered)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
