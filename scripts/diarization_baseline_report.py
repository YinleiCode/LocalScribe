#!/usr/bin/env python3
"""Compare LocalScribe diarization against optional industry baselines.

This script is intentionally CLI-only and does not modify transcript files.  It
can run the current LocalScribe engine, optionally run pyannote when installed,
or read externally generated baseline JSON.  Without human gold labels it
reports consistency and risk metrics, not DER.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scribe-py" / "src"))

from scribe_py.diarizers import diarize as run_diarizer  # noqa: E402
from scribe_py.ipc import (  # noqa: E402
    _segments_with_diarization_speakers,
    _speaker_summary,
    handle_diarize,
)


@dataclass
class Case:
    label: str
    transcript: Path
    data: dict[str, Any]
    segments: list[dict[str, Any]]
    audio: Path | None


@dataclass
class EngineResult:
    case: str
    engine: str
    status: str
    segments: list[dict[str, Any]]
    stats: dict[str, Any]
    elapsed_s: float = 0.0
    error: str = ""
    audio: str = ""
    transcript: str = ""


@dataclass
class AgreementResult:
    score: float | None
    mapping: dict[str, str]


def _parse_label_path(value: str, flag: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"{flag} must be LABEL=PATH, got: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise SystemExit(f"{flag} label is empty: {value}")
    if not path.exists():
        raise SystemExit(f"{flag} file not found: {path}")
    return label, path


def _load_case(value: str) -> Case:
    label, path = _parse_label_path(value, "--case")
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = list(data.get("segments") or [])
    raw_audio = data.get("audio") or data.get("source_audio") or data.get("audio_path")
    audio = Path(raw_audio).expanduser() if raw_audio else None
    return Case(label=label, transcript=path, data=data, segments=segments, audio=audio)


def _load_baseline(value: str) -> EngineResult:
    label, path = _parse_label_path(value, "--baseline-json")
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = list(data.get("segments") or [])
    stats = dict(data.get("diarization_stats") or data.get("stats") or {})
    return EngineResult(
        case=str(data.get("case") or path.stem),
        engine=label,
        status="ok",
        segments=segments,
        stats=stats,
        audio=str(data.get("audio") or data.get("source_audio") or data.get("audio_path") or ""),
        transcript=str(data.get("transcript") or ""),
    )


def _run_engine(case: Case, engine: str, n_speakers: int) -> EngineResult:
    if not case.audio or not case.audio.exists():
        return EngineResult(
            case=case.label,
            engine=engine,
            status="error",
            segments=[],
            stats={},
            error=f"源音频不存在: {case.audio}",
            audio=str(case.audio or ""),
            transcript=str(case.transcript),
        )
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_diarizer(
                audio=case.audio,
                segments=case.segments,
                n_speakers=n_speakers,
                profiles=[],
                engine=engine,
            )
        out_segments = _segments_with_diarization_speakers(case.segments, result.segments)
        stats = dict(result.stats or {})
        stats["clusters"] = len(result.speakers)
        stats["requested_n_speakers"] = int(n_speakers)
        return EngineResult(
            case=case.label,
            engine=str(stats.get("engine") or engine),
            status="ok",
            segments=out_segments,
            stats=stats,
            elapsed_s=time.perf_counter() - started,
            audio=str(case.audio),
            transcript=str(case.transcript),
        )
    except Exception as exc:
        return EngineResult(
            case=case.label,
            engine=engine,
            status="error",
            segments=[],
            stats={},
            elapsed_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
            audio=str(case.audio or ""),
            transcript=str(case.transcript),
        )


def _run_app_pipeline(case: Case, engine: str, n_speakers: int) -> EngineResult:
    """Run the same diarization post-processing path used by the App, without ASR."""
    if not case.audio or not case.audio.exists():
        return EngineResult(
            case.label,
            f"app-{engine}",
            "error",
            [],
            {},
            error=f"源音频不存在: {case.audio}",
            audio=str(case.audio or ""),
            transcript=str(case.transcript),
        )
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            payload = handle_diarize({
                "audio": str(case.audio),
                "segments": case.segments,
                "n_speakers": int(n_speakers),
                "profiles": [],
                "engine": engine,
            })
        return EngineResult(
            case=case.label,
            engine=f"app-{engine}",
            status="ok",
            segments=list(payload.get("segments") or []),
            stats=dict(payload.get("stats") or {}),
            elapsed_s=time.perf_counter() - started,
            audio=str(case.audio),
            transcript=str(case.transcript),
        )
    except Exception as exc:
        return EngineResult(
            case=case.label,
            engine=f"app-{engine}",
            status="error",
            segments=[],
            stats={},
            elapsed_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
            audio=str(case.audio or ""),
            transcript=str(case.transcript),
        )


def _duration(segments: list[dict[str, Any]]) -> float:
    if not segments:
        return 0.0
    starts = [float(s.get("start") or 0.0) for s in segments]
    ends = [float(s.get("end") or s.get("start") or 0.0) for s in segments]
    return max(0.0, max(ends) - min(starts))


def _segment_duration(segment: dict[str, Any]) -> float:
    try:
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, end - start)


def _speaker_count(segments: list[dict[str, Any]]) -> int:
    return len({str(s.get("speaker")) for s in segments if s.get("speaker")})


def _distribution(segments: list[dict[str, Any]]) -> str:
    summary = _speaker_summary(segments).get("speakers") or []
    return " / ".join(
        f"{str(s.get('speaker')).replace('SPEAKER_', '')}:{s.get('segments')}段/{float(s.get('duration_s') or 0):.1f}s"
        for s in summary
    )


def _risk_counts(segments: list[dict[str, Any]]) -> tuple[int, int]:
    overlap = sum(1 for s in segments if s.get("speaker_overlap_risk"))
    resegmented = sum(1 for s in segments if s.get("speaker_resegmented"))
    return overlap, resegmented


def _agreement(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> AgreementResult:
    if not a or not b:
        return AgreementResult(score=None, mapping={})
    if len(a) != len(b):
        return AgreementResult(score=None, mapping={})
    total = 0
    same = 0
    mapping: dict[str, str] = {}
    used_targets: set[str] = set()
    confusion: dict[str, Counter] = defaultdict(Counter)
    for left, right in zip(a, b):
        left_speaker = str(left.get("speaker") or "")
        right_speaker = str(right.get("speaker") or "")
        if not left_speaker or not right_speaker:
            continue
        confusion[left_speaker][right_speaker] += 1
    for left_speaker, counts in confusion.items():
        for right_speaker, _count in counts.most_common():
            if right_speaker not in used_targets:
                mapping[left_speaker] = right_speaker
                used_targets.add(right_speaker)
                break
    for left, right in zip(a, b):
        left_speaker = str(left.get("speaker") or "")
        right_speaker = str(right.get("speaker") or "")
        if not left_speaker or not right_speaker:
            continue
        total += 1
        if mapping.get(left_speaker) == right_speaker:
            same += 1
    return AgreementResult(score=same / total if total else None, mapping=mapping)


def _segment_agreement(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> float | None:
    return _agreement(a, b).score


def _proxy_metrics(
    segments: list[dict[str, Any]],
    *,
    short_threshold_s: float = 1.0,
    fragment_threshold_s: float = 0.5,
) -> dict[str, Any]:
    labeled = [s for s in segments if s.get("speaker")]
    durations = [_segment_duration(s) for s in labeled]
    positive_durations = [d for d in durations if d > 0]
    span_s = _duration(segments)
    speaker_durations: Counter[str] = Counter()
    for seg, dur in zip(labeled, durations):
        speaker_durations[str(seg.get("speaker"))] += dur

    switches = 0
    previous = ""
    turns = 0
    for seg in labeled:
        speaker = str(seg.get("speaker") or "")
        if not speaker:
            continue
        if speaker != previous:
            turns += 1
            if previous:
                switches += 1
        previous = speaker

    short_segments = sum(1 for d in positive_durations if d < short_threshold_s)
    fragment_segments = sum(1 for d in positive_durations if d < fragment_threshold_s)
    denominator = len(positive_durations) or 1
    short_ratio = short_segments / denominator
    fragment_ratio = fragment_segments / denominator
    avg_segment_s = sum(positive_durations) / denominator if positive_durations else 0.0
    switch_rate_per_min = switches / (span_s / 60.0) if span_s > 0 else 0.0
    speaker_duration_total = sum(speaker_durations.values())
    dominant_ratio = (
        max(speaker_durations.values()) / speaker_duration_total
        if speaker_duration_total > 0 and speaker_durations
        else 0.0
    )

    risk_notes: list[str] = []
    if fragment_ratio >= 0.25:
        risk_notes.append("碎片段过多")
    elif fragment_ratio >= 0.12:
        risk_notes.append("碎片段偏多")
    if short_ratio >= 0.55:
        risk_notes.append("短句切分过密")
    elif short_ratio >= 0.35:
        risk_notes.append("短句偏多")
    if switch_rate_per_min >= 20:
        risk_notes.append("说话人跳变过频")
    elif switch_rate_per_min >= 12:
        risk_notes.append("切换频率偏高")
    if _speaker_count(segments) >= 2 and dominant_ratio >= 0.92:
        risk_notes.append("主讲占比过高")

    if any(note in risk_notes for note in ["碎片段过多", "短句切分过密", "说话人跳变过频"]):
        proxy_risk = "high"
    elif risk_notes:
        proxy_risk = "medium"
    else:
        proxy_risk = "low"

    return {
        "turns": turns,
        "speaker_switches": switches,
        "avg_segment_s": avg_segment_s,
        "short_segments": short_segments,
        "short_ratio": short_ratio,
        "fragment_segments": fragment_segments,
        "fragment_ratio": fragment_ratio,
        "switch_rate_per_min": switch_rate_per_min,
        "dominant_ratio": dominant_ratio,
        "missing_speaker_segments": len(segments) - len(labeled),
        "proxy_risk": proxy_risk,
        "risk_notes": risk_notes,
    }


def _fmt_ratio(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.1%}"


def _rows(results: list[EngineResult], reference_by_case: dict[str, EngineResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        segments = result.segments
        duration = _duration(segments)
        overlap, resegmented = _risk_counts(segments)
        metrics = _proxy_metrics(segments) if result.status == "ok" else {}
        ref = reference_by_case.get(result.case)
        agreement = _segment_agreement(segments, ref.segments) if ref and ref is not result else None
        rows.append({
            "录音": result.case,
            "引擎": result.engine,
            "状态": result.status,
            "人数": _speaker_count(segments) if result.status == "ok" else "",
            "段数": len(segments) if result.status == "ok" else "",
            "时长秒": f"{duration:.1f}" if duration else "",
            "耗时秒": f"{result.elapsed_s:.1f}" if result.elapsed_s else "",
            "RTF": f"{result.elapsed_s / duration:.3f}" if duration and result.elapsed_s else "",
            "平均段长秒": f"{float(metrics.get('avg_segment_s') or 0):.2f}" if metrics else "",
            "说话轮次": metrics.get("turns", "") if metrics else "",
            "切换次数": metrics.get("speaker_switches", "") if metrics else "",
            "每分钟切换": f"{float(metrics.get('switch_rate_per_min') or 0):.1f}" if metrics else "",
            "短段<1s": metrics.get("short_segments", "") if metrics else "",
            "短段占比": _fmt_ratio(float(metrics.get("short_ratio") or 0)) if metrics else "",
            "碎片<0.5s": metrics.get("fragment_segments", "") if metrics else "",
            "碎片占比": _fmt_ratio(float(metrics.get("fragment_ratio") or 0)) if metrics else "",
            "主讲占比": _fmt_ratio(float(metrics.get("dominant_ratio") or 0)) if metrics else "",
            "代理风险": metrics.get("proxy_risk", "") if metrics else "",
            "风险说明": "、".join(metrics.get("risk_notes") or []) if metrics else "",
            "重叠风险段": overlap if result.status == "ok" else "",
            "已切分段": resegmented if result.status == "ok" else "",
            "与基线一致率": f"{agreement:.1%}" if agreement is not None else "",
            "分布": _distribution(segments) if result.status == "ok" else "",
            "错误": result.error,
        })
    return rows


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = ["\t".join(headers)]
    for row in rows:
        lines.append("\t".join(str(row.get(h, "")) for h in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _disagreement_rows(
    results: list[EngineResult],
    reference_by_case: dict[str, EngineResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        ref = reference_by_case.get(result.case)
        if not ref or ref is result or result.status != "ok" or ref.status != "ok":
            continue
        agreement = _agreement(result.segments, ref.segments)
        if agreement.score is None:
            rows.append({
                "录音": result.case,
                "引擎": result.engine,
                "序号": "",
                "开始": "",
                "结束": "",
                "基线说话人": "",
                "当前说话人": "",
                "当前映射后": "",
                "文本": "",
                "原因": f"段数不同，无法逐段对齐: {len(result.segments)} vs {len(ref.segments)}",
            })
            continue
        for idx, (seg, ref_seg) in enumerate(zip(result.segments, ref.segments)):
            speaker = str(seg.get("speaker") or "")
            ref_speaker = str(ref_seg.get("speaker") or "")
            if not speaker or not ref_speaker:
                continue
            mapped_speaker = agreement.mapping.get(speaker, "")
            if mapped_speaker == ref_speaker:
                continue
            rows.append({
                "录音": result.case,
                "引擎": result.engine,
                "序号": idx,
                "开始": f"{float(seg.get('start') or 0):.2f}",
                "结束": f"{float(seg.get('end') or seg.get('start') or 0):.2f}",
                "基线说话人": ref_speaker,
                "当前说话人": speaker,
                "当前映射后": mapped_speaker,
                "文本": str(seg.get("text") or "")[:120].replace("\n", " "),
                "原因": "说话人不一致",
            })
    return rows


def _write_md(path: Path, rows: list[dict[str, Any]], has_gold: bool) -> None:
    lines = [
        "# 说话人分离基线对比报告",
        "",
        "> 本报告用于比较 LocalScribe 当前分人引擎与行业基线/外部基线。"
        + ("已提供 gold，可扩展 DER 指标。" if has_gold else "未提供人工 gold，因此一致率不是 DER，只用于发现差异和风险。"),
        "",
    ]
    if rows:
        headers = [
            "录音", "引擎", "状态", "人数", "段数", "耗时秒", "RTF",
            "平均段长秒", "说话轮次", "每分钟切换", "短段占比", "碎片占比",
            "主讲占比", "代理风险", "风险说明", "重叠风险段", "已切分段",
            "与基线一致率", "分布", "错误",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(h, "")).replace("|", "/") for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", str(value)).strip("._")
    return cleaned or "result"


def _write_predictions(out_dir: Path, results: list[EngineResult]) -> list[str]:
    prediction_dir = out_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for result in results:
        if result.status != "ok" or not result.segments:
            continue
        path = prediction_dir / f"{_safe_filename(result.case)}__{_safe_filename(result.engine)}.json"
        path.write_text(json.dumps({
            "case": result.case,
            "engine": result.engine,
            "audio": result.audio,
            "transcript": result.transcript,
            "segments": result.segments,
            "diarization_stats": result.stats,
            "elapsed_s": result.elapsed_s,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(str(path))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", default=[], help="LABEL=transcript.json，可重复")
    parser.add_argument("--out-dir", type=Path, default=Path("diarization_baseline_report"))
    parser.add_argument("--n-speakers", type=int, default=0, help="固定人数；0=引擎自动")
    parser.add_argument("--run-current", action="store_true", default=True)
    parser.add_argument("--current-engine", default="auto", choices=["auto", "senko", "resemblyzer"], help="当前引擎选择，默认 auto")
    parser.add_argument("--engine", action="append", choices=["auto", "senko", "resemblyzer", "pyannote"], help="指定要运行的分人引擎，可重复；设置后覆盖 --current-engine/--run-pyannote")
    parser.add_argument("--run-pyannote", action="store_true", help="显式启用可选 pyannote 基线")
    parser.add_argument("--app-pipeline", action="store_true", help="运行与 App 相同的分人后处理链，不重跑 ASR")
    parser.add_argument("--baseline-json", action="append", default=[], help="ENGINE=baseline_result.json，可重复")
    parser.add_argument("--gold", default="", help="预留 gold RTTM/JSON 路径；当前仅在报告中声明")
    args = parser.parse_args()

    if not args.case and not args.baseline_json:
        raise SystemExit("请至少提供 --case LABEL=transcript.json 或 --baseline-json ENGINE=PATH")

    cases = [_load_case(value) for value in args.case]
    results: list[EngineResult] = []
    engines = list(dict.fromkeys(args.engine or []))
    if not engines:
        if args.run_current:
            engines.append(args.current_engine)
        if args.run_pyannote:
            engines.append("pyannote")
    for case in cases:
        for engine in engines:
            results.append(_run_engine(case, engine, args.n_speakers))
        if args.app_pipeline:
            results.append(_run_app_pipeline(case, args.current_engine, args.n_speakers))

    baseline_results = [_load_baseline(value) for value in args.baseline_json]
    results.extend(baseline_results)

    reference_by_case: dict[str, EngineResult] = {}
    for result in results:
        if result.status == "ok" and result.case not in reference_by_case:
            reference_by_case[result.case] = result

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(results, reference_by_case)
    disagreement_rows = _disagreement_rows(results, reference_by_case)
    md_path = out_dir / "diarization_baseline_report.md"
    tsv_path = out_dir / "diarization_baseline_report.tsv"
    json_path = out_dir / "diarization_baseline_report.json"
    disagreement_path = out_dir / "diarization_disagreements.tsv"
    _write_md(md_path, rows, bool(args.gold))
    _write_tsv(tsv_path, rows)
    _write_tsv(disagreement_path, disagreement_rows)
    prediction_paths = _write_predictions(out_dir, results)
    json_path.write_text(json.dumps({
        "ok": True,
        "has_gold": bool(args.gold),
        "cases": [case.label for case in cases],
        "rows": rows,
        "disagreements": disagreement_rows,
        "predictions": prediction_paths,
        "results": [
            {
                "case": result.case,
                "engine": result.engine,
                "status": result.status,
                "stats": result.stats,
                "elapsed_s": result.elapsed_s,
                "error": result.error,
                "audio": result.audio,
                "transcript": result.transcript,
            }
            for result in results
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "markdown": str(md_path),
        "tsv": str(tsv_path),
        "json": str(json_path),
        "disagreements": str(disagreement_path),
        "cases": len(cases),
        "engines": sorted({r.engine for r in results}),
        "predictions": prediction_paths,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
