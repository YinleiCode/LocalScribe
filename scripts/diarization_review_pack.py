#!/usr/bin/env python3
"""Build a small, recording-agnostic speaker-diarization review pack.

The pack is evaluation-only: it reads transcript JSON and source audio, picks
diverse acoustic situations, extracts short WAV clips, and renders a local
Chinese review page. It never edits transcripts or applies human labels.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scribe-py" / "src"))


CATEGORY_ORDER = (
    "段内换人",
    "疑似错挂",
    "说话人切换",
    "重叠或短句",
    "稳定对照",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _fmt_time(value: float) -> str:
    total_ms = int(round(max(0.0, value) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def _short_speaker(value: Any) -> str:
    return str(value or "").replace("SPEAKER_", "")


def _safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return clean or "recording"


def _unreliable_timing_reason(data: dict[str, Any]) -> str:
    """Return an actionable reason only when timing is explicitly unreliable."""
    stats = data.get("filter_stats") or {}
    if not isinstance(stats, dict) or stats.get("timing_reliable") is not False:
        return ""
    reason = str(
        stats.get("timing_alignment_reason")
        or stats.get("timing_reason")
        or "transcript timing is explicitly unreliable"
    ).strip()
    return reason


def partition_review_cases(
    cases: list[tuple[str, Path, dict[str, Any]]],
) -> tuple[list[tuple[str, Path, dict[str, Any]]], list[dict[str, str]]]:
    eligible: list[tuple[str, Path, dict[str, Any]]] = []
    excluded: list[dict[str, str]] = []
    for label, transcript_path, data in cases:
        reason = _unreliable_timing_reason(data)
        if reason:
            excluded.append({
                "recording": label,
                "transcript": str(transcript_path.resolve()),
                "reason": reason,
            })
        else:
            eligible.append((label, transcript_path, data))
    return eligible, excluded


def _segment_duration(segment: dict[str, Any]) -> float:
    return max(0.0, _as_float(segment.get("end")) - _as_float(segment.get("start")))


def _spoken_text_units(value: Any) -> int:
    """Approximate spoken units without penalizing embedded Latin terms."""
    text = str(value or "")
    cjk_units = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))
    latin_words = len(re.findall(r"[A-Za-z0-9]+(?:[._+-][A-Za-z0-9]+)*", text))
    return cjk_units + latin_words


def _has_plausible_review_timing(segment: dict[str, Any]) -> bool:
    """Reject timestamp-compressed text while retaining genuine short replies."""
    duration = _segment_duration(segment)
    units = _spoken_text_units(segment.get("text"))
    if units <= 2 or duration >= 0.65:
        return True
    # Allow brief speech up to a deliberately generous 12 units/second. This
    # removes impossible tail compression without filtering fast interjections.
    return units <= max(2, math.ceil(duration * 12.0))


def _segment_confidence(segment: dict[str, Any]) -> float:
    return max(0.0, min(1.0, _as_float(segment.get("speaker_confidence"), 0.5)))


def _subsegment_speakers(segment: dict[str, Any]) -> set[str]:
    return {
        str(row.get("speaker") or "")
        for row in segment.get("speaker_subsegments") or []
        if isinstance(row, dict) and row.get("speaker")
    }


def _cue_speakers(segment: dict[str, Any]) -> list[str]:
    labels = [
        str(row.get("speaker") or "")
        for row in segment.get("speaker_cues") or []
        if isinstance(row, dict) and row.get("speaker")
    ]
    return _dedupe(labels)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and (not out or out[-1] != value):
            out.append(value)
    return out


def _bounded_window(
    start: float,
    end: float,
    *,
    duration: float,
    focus: float | None = None,
    max_seconds: float = 12.0,
) -> tuple[float, float]:
    start = max(0.0, start)
    end = min(duration, max(start + 0.05, end)) if duration > 0 else max(start + 0.05, end)
    if end - start <= max_seconds:
        return start, end
    midpoint = focus if focus is not None else (start + end) / 2
    bounded_start = max(start, midpoint - max_seconds / 2)
    bounded_end = min(end, bounded_start + max_seconds)
    bounded_start = max(start, bounded_end - max_seconds)
    return bounded_start, bounded_end


def _candidate(
    *,
    category: str,
    score: float,
    start: float,
    end: float,
    segment_index: int,
    reason: str,
    duration: float,
    focus: float | None = None,
    speakers: list[str] | None = None,
) -> dict[str, Any]:
    bounded_start, bounded_end = _bounded_window(
        start,
        end,
        duration=duration,
        focus=focus,
    )
    return {
        "category": category,
        "score": round(score, 3),
        "review_start": round(bounded_start, 3),
        "review_end": round(bounded_end, 3),
        "segment_index": int(segment_index),
        "reason": reason,
        "speakers": _dedupe([str(value or "") for value in speakers or []]),
    }


def build_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    segments = [row for row in data.get("segments") or [] if isinstance(row, dict)]
    duration = _as_float(data.get("duration"))
    if duration <= 0 and segments:
        duration = max(_as_float(row.get("end")) for row in segments)
    candidates: list[dict[str, Any]] = []
    reviewable = [_has_plausible_review_timing(segment) for segment in segments]

    for index, segment in enumerate(segments):
        start = _as_float(segment.get("start"))
        end = _as_float(segment.get("end"))
        if end <= start or not reviewable[index]:
            continue
        confidence = _segment_confidence(segment)
        change_points = [
            _as_float(value, -1.0)
            for value in segment.get("speaker_change_points") or []
            if _as_float(value, -1.0) > start and _as_float(value, -1.0) < end
        ]
        cue_labels = _cue_speakers(segment)
        subsegment_labels = _subsegment_speakers(segment)
        candidate_speakers = cue_labels or [str(segment.get("speaker") or "")]
        internal_handoff = len(cue_labels) > 1
        suspected_handoff = len(subsegment_labels) > 1 and bool(change_points)
        if internal_handoff or suspected_handoff:
            focus = change_points[0] if change_points else (start + end) / 2
            candidates.append(_candidate(
                category="段内换人",
                score=110.0 if internal_handoff else 92.0 - confidence * 8,
                start=start,
                end=end,
                segment_index=index,
                reason=(
                    "当前通用投影检测到段内说话人切换"
                    if internal_handoff
                    else "短窗声纹显示段内可能换人"
                ),
                duration=duration,
                focus=focus,
                speakers=candidate_speakers,
            ))

        if segment.get("speaker_assignment_review"):
            candidates.append(_candidate(
                category="疑似错挂",
                score=84.0 + (1.0 - confidence) * 12,
                start=start,
                end=end,
                segment_index=index,
                reason=str(segment.get("speaker_review_reason") or "当前说话人需要人工确认"),
                duration=duration,
                focus=change_points[0] if change_points else None,
                speakers=candidate_speakers,
            ))

        short = _segment_duration(segment) <= 1.4
        overlap = bool(segment.get("speaker_overlap_risk"))
        low_confidence = confidence < 0.65
        if short or overlap or low_confidence:
            reason_parts = []
            if overlap:
                reason_parts.append("可能重叠/插话")
            if short:
                reason_parts.append("短句声纹不足")
            if low_confidence:
                reason_parts.append("说话人置信度低")
            candidates.append(_candidate(
                category="重叠或短句",
                score=64.0 + (8 if overlap else 0) + (6 if low_confidence else 0),
                start=start,
                end=end,
                segment_index=index,
                reason="；".join(reason_parts),
                duration=duration,
                speakers=candidate_speakers,
            ))

        previous = segments[index - 1] if index > 0 else None
        following = segments[index + 1] if index + 1 < len(segments) else None
        stable = (
            _segment_duration(segment) >= 4.0
            and confidence >= 0.85
            and not segment.get("speaker_assignment_review")
            and not segment.get("speaker_overlap_risk")
            and len(subsegment_labels) <= 1
            and previous is not None
            and following is not None
            and previous.get("speaker") == segment.get("speaker") == following.get("speaker")
        )
        if stable:
            candidates.append(_candidate(
                category="稳定对照",
                score=35.0 + min(_segment_duration(segment), 12.0) / 12.0,
                start=start,
                end=end,
                segment_index=index,
                reason="高置信连续同人片段，用于检查是否出现误拆",
                duration=duration,
                speakers=candidate_speakers,
            ))

    for index in range(1, len(segments)):
        previous = segments[index - 1]
        current = segments[index]
        if not reviewable[index - 1] or not reviewable[index]:
            continue
        previous_speaker = str(previous.get("speaker") or "")
        current_speaker = str(current.get("speaker") or "")
        if not previous_speaker or not current_speaker or previous_speaker == current_speaker:
            continue
        boundary = max(_as_float(previous.get("start")), _as_float(current.get("start")))
        start = max(_as_float(previous.get("start")), boundary - 5.0)
        end = min(_as_float(current.get("end")), boundary + 5.0)
        confidence = min(_segment_confidence(previous), _segment_confidence(current))
        candidates.append(_candidate(
            category="说话人切换",
            score=76.0 + (1.0 - confidence) * 8,
            start=start,
            end=end,
            segment_index=index,
            reason=f"当前标签在 {_short_speaker(previous_speaker)}->{_short_speaker(current_speaker)} 之间切换",
            duration=duration,
            focus=boundary,
            speakers=[previous_speaker, current_speaker],
        ))

    return candidates


def _overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    overlap = max(
        0.0,
        min(_as_float(left.get("review_end")), _as_float(right.get("review_end")))
        - max(_as_float(left.get("review_start")), _as_float(right.get("review_start"))),
    )
    shorter = min(
        _as_float(left.get("review_end")) - _as_float(left.get("review_start")),
        _as_float(right.get("review_end")) - _as_float(right.get("review_start")),
    )
    return overlap / shorter if shorter > 0 else 0.0


def select_candidates(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    def add_best(rows: list[dict[str, Any]]) -> bool:
        for row in sorted(rows, key=lambda item: (-_as_float(item.get("score")), _as_float(item.get("review_start")))):
            if all(_overlap_ratio(row, existing) < 0.45 for existing in selected):
                selected.append(row)
                return True
        return False

    for category in CATEGORY_ORDER:
        if len(selected) >= limit:
            break
        add_best([row for row in candidates if row.get("category") == category])

    all_speakers = sorted({
        str(speaker)
        for row in candidates
        for speaker in row.get("speakers") or []
        if speaker
    })
    for speaker in all_speakers:
        if len(selected) >= limit:
            break
        if any(speaker in (row.get("speakers") or []) for row in selected):
            continue
        add_best([row for row in candidates if speaker in (row.get("speakers") or [])])

    if len(selected) < limit:
        addable = sorted(candidates, key=lambda item: (-_as_float(item.get("score")), _as_float(item.get("review_start"))))
        for row in addable:
            if len(selected) >= limit:
                break
            if row in selected:
                continue
            if all(_overlap_ratio(row, existing) < 0.45 for existing in selected):
                selected.append(row)

    return sorted(selected, key=lambda item: _as_float(item.get("review_start")))


def _cue_text(segment: dict[str, Any], cue: dict[str, Any]) -> str:
    projected_text = str(cue.get("text") or "").strip()
    if projected_text:
        return projected_text
    cue_index = int(_as_float(cue.get("cue_index"), -1))
    sync_cues = segment.get("sync_cues") or []
    if 0 <= cue_index < len(sync_cues) and isinstance(sync_cues[cue_index], dict):
        return str(sync_cues[cue_index].get("text") or "").strip()
    return str(segment.get("text") or "").strip()


def _timeline_rows(data: dict[str, Any], start: float, end: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(data.get("segments") or []):
        if not isinstance(segment, dict):
            continue
        seg_start = _as_float(segment.get("start"))
        seg_end = _as_float(segment.get("end"))
        if min(seg_end, end) - max(seg_start, start) <= 0:
            continue
        cues = [row for row in segment.get("speaker_cues") or [] if isinstance(row, dict)]
        if cues:
            for cue in cues:
                cue_start = _as_float(cue.get("start"))
                cue_end = _as_float(cue.get("end"))
                if min(cue_end, end) - max(cue_start, start) <= 0:
                    continue
                rows.append({
                    "segment_index": index,
                    "start": round(cue_start, 3),
                    "end": round(cue_end, 3),
                    "speaker": _short_speaker(cue.get("speaker")),
                    "text": _cue_text(segment, cue),
                    "source": "speaker_cue",
                })
        else:
            rows.append({
                "segment_index": index,
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "speaker": _short_speaker(segment.get("speaker")),
                "text": str(segment.get("text") or "").strip(),
                "source": "segment",
            })
    return rows


def build_review_items(
    cases: list[tuple[str, Path, dict[str, Any]]],
    *,
    per_case: int,
    padding_seconds: float,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for label, transcript_path, data in cases:
        timing_reason = _unreliable_timing_reason(data)
        if timing_reason:
            raise ValueError(
                f"cannot build time-based review clips for {label}: {timing_reason}"
            )
        audio_value = data.get("audio") or data.get("source_audio") or data.get("audio_path")
        audio = Path(str(audio_value or "")).expanduser()
        if not audio.is_file():
            raise FileNotFoundError(f"source audio does not exist for {label}: {audio}")
        duration = _as_float(data.get("duration"))
        selected = select_candidates(build_candidates(data), limit=per_case)
        for candidate in selected:
            review_start = _as_float(candidate.get("review_start"))
            review_end = _as_float(candidate.get("review_end"))
            clip_start = max(0.0, review_start - max(0.0, padding_seconds))
            clip_end = review_end + max(0.0, padding_seconds)
            if duration > 0:
                clip_end = min(duration, clip_end)
            focus_timeline = _timeline_rows(data, review_start, review_end)
            timeline = _timeline_rows(data, clip_start, clip_end)
            for row in timeline:
                row["context"] = (
                    min(_as_float(row.get("end")), review_end)
                    - max(_as_float(row.get("start")), review_start)
                    <= 0
                )
            prediction = "->".join(
                _dedupe([str(row.get("speaker") or "") for row in focus_timeline])
            )
            items.append({
                "recording": label,
                "transcript": str(transcript_path.resolve()),
                "audio": str(audio.resolve()),
                **candidate,
                "clip_start": round(clip_start, 3),
                "clip_end": round(max(clip_start + 0.05, clip_end), 3),
                "current_prediction": prediction,
                "timeline": timeline,
                "verdict": "",
                "correct_speaker_sequence": "",
                "notes": "",
            })

    for number, item in enumerate(items, start=1):
        item["id"] = f"DIA-{number:03d}"
    return items


def _extract_clip(audio: Path, output: Path, start: float, end: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{max(0.05, end - start):.3f}",
            "-i",
            str(audio),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0 or not output.is_file() or output.stat().st_size <= 44:
        output.unlink(missing_ok=True)
        raise RuntimeError((process.stderr or "ffmpeg produced no audio clip").strip())


def _manifest_hash(items: list[dict[str, Any]]) -> str:
    stable = [
        {
            "id": item.get("id"),
            "recording": item.get("recording"),
            "start": item.get("review_start"),
            "end": item.get("review_end"),
            "prediction": item.get("current_prediction"),
            "timeline": [
                {
                    "start": row.get("start"),
                    "end": row.get("end"),
                    "speaker": row.get("speaker"),
                    "context": bool(row.get("context")),
                }
                for row in item.get("timeline") or []
                if isinstance(row, dict)
            ],
        }
        for item in items
    ]
    return hashlib.sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def render_html(manifest: dict[str, Any]) -> str:
    manifest_json = json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c")
    title = html.escape(str(manifest.get("title") or "通用分人验收"))
    excluded = manifest.get("excluded_recordings") or []
    excluded_html = ""
    if excluded:
        rows = "".join(
            f"<li><strong>{html.escape(str(row.get('recording') or ''))}</strong>: "
            f"{html.escape(str(row.get('reason') or '时间轴不可靠'))}</li>"
            for row in excluded
            if isinstance(row, dict)
        )
        excluded_html = (
            '<aside class="warning"><strong>已排除时间轴不可靠的录音</strong>'
            f"<ul>{rows}</ul></aside>"
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f4f6f8; color: #182027; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-width: 320px; }}
    header {{ position: sticky; top: 0; z-index: 4; border-bottom: 1px solid #d9dee3; background: rgba(255,255,255,.96); }}
    .bar {{ max-width: 1180px; margin: 0 auto; padding: 14px 20px; display: flex; align-items: center; gap: 18px; }}
    h1 {{ margin: 0; font-size: 18px; line-height: 1.3; letter-spacing: 0; }}
    .progress {{ color: #53616d; font-size: 13px; white-space: nowrap; }}
    .spacer {{ flex: 1; }}
    button {{ min-height: 36px; border: 1px solid #b9c2ca; border-radius: 6px; background: #fff; color: #182027; padding: 0 12px; font-weight: 600; cursor: pointer; }}
    button:hover {{ border-color: #2368a2; color: #174f7c; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 18px 20px 48px; }}
    .warning {{ margin-bottom: 14px; border: 1px solid #c78232; border-radius: 6px; background: #fff8e8; color: #5f431d; padding: 12px 14px; font-size: 13px; line-height: 1.5; }}
    .warning ul {{ margin: 6px 0 0; padding-left: 20px; }}
    .filters {{ display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }}
    .filters button[aria-pressed="true"] {{ background: #1f6f54; border-color: #1f6f54; color: white; }}
    .list {{ display: grid; gap: 10px; }}
    article {{ border: 1px solid #d9dee3; border-radius: 8px; background: #fff; overflow: hidden; }}
    .head {{ display: grid; grid-template-columns: 90px minmax(120px, .8fr) minmax(120px, .7fr) minmax(180px, 1fr); gap: 12px; align-items: center; padding: 12px 14px; border-bottom: 1px solid #e7eaed; }}
    .id {{ font: 700 13px ui-monospace, SFMono-Regular, Menlo, monospace; color: #174f7c; }}
    .recording {{ font-weight: 650; overflow-wrap: anywhere; }}
    .time {{ font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; color: #53616d; }}
    .category {{ justify-self: start; border-radius: 4px; background: #edf2f5; color: #344651; padding: 4px 7px; font-size: 12px; }}
    .body {{ display: grid; grid-template-columns: minmax(280px, 1.35fr) minmax(260px, 1fr); gap: 18px; padding: 14px; }}
    audio {{ width: 100%; height: 36px; margin-bottom: 10px; }}
    .prediction {{ display: flex; align-items: center; gap: 8px; margin: 0 0 10px; font-size: 13px; }}
    .sequence {{ font: 700 13px ui-monospace, SFMono-Regular, Menlo, monospace; color: #8c3b16; }}
    .timeline {{ display: grid; gap: 7px; }}
    .line {{ display: grid; grid-template-columns: 28px 94px 1fr; gap: 8px; align-items: start; font-size: 13px; line-height: 1.5; }}
    .line.context {{ opacity: .52; }}
    .speaker {{ font-weight: 750; color: #1f6f54; }}
    .line-time {{ color: #68757f; font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; padding-top: 2px; }}
    .reason {{ color: #5f6c75; font-size: 12px; margin-top: 10px; }}
    .form {{ display: grid; gap: 10px; align-content: start; }}
    label {{ display: grid; gap: 5px; color: #53616d; font-size: 12px; }}
    select, input {{ width: 100%; min-height: 38px; border: 1px solid #b9c2ca; border-radius: 6px; background: #fff; color: #182027; padding: 7px 9px; font: inherit; }}
    select:focus, input:focus {{ outline: 2px solid #80b7e1; outline-offset: 1px; border-color: #2368a2; }}
    article.done {{ border-left: 4px solid #1f6f54; }}
    article.error {{ border-left: 4px solid #bd4b35; }}
    .empty {{ padding: 36px; text-align: center; color: #68757f; }}
    @media (max-width: 760px) {{
      .bar {{ padding: 12px; flex-wrap: wrap; gap: 8px; }}
      main {{ padding: 12px 10px 36px; }}
      .head {{ grid-template-columns: 72px 1fr; }}
      .body {{ grid-template-columns: 1fr; }}
      .line {{ grid-template-columns: 24px 82px 1fr; }}
      .spacer {{ display: none; }}
    }}
  </style>
</head>
<body>
  <header><div class="bar">
    <h1>{title}</h1>
    <div class="progress" id="progress">0 / 0</div>
    <div class="spacer"></div>
    <button id="export" type="button">导出标注</button>
  </div></header>
  <main>
    {excluded_html}
    <div class="filters" role="group" aria-label="筛选">
      <button type="button" data-filter="all" aria-pressed="true">全部</button>
      <button type="button" data-filter="pending" aria-pressed="false">待标注</button>
      <button type="button" data-filter="done" aria-pressed="false">已标注</button>
    </div>
    <div class="list" id="list"></div>
  </main>
  <script id="manifest" type="application/json">{manifest_json}</script>
  <script>
    const manifest = JSON.parse(document.getElementById('manifest').textContent);
    const storageKey = `localscribe-diarization-review:${{manifest.pack_id}}`;
    const saved = JSON.parse(localStorage.getItem(storageKey) || '{{}}');
    let filter = 'all';
    const verdicts = [
      ['', '待标注'], ['correct', '正确'], ['wrong_speaker', '人员错'],
      ['wrong_boundary', '切点错'], ['missed_split', '漏拆（实际多人）'],
      ['false_split', '误拆（实际同一人）'], ['uncertain', '不确定']
    ];
    const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    const itemState = (item) => ({{ verdict: '', correct_speaker_sequence: '', notes: '', ...(saved[item.id] || {{}}) }});
    function persist(id, patch) {{
      saved[id] = {{...itemState({{id}}), ...patch}};
      localStorage.setItem(storageKey, JSON.stringify(saved));
      render();
    }}
    function render() {{
      const list = document.getElementById('list');
      const visible = manifest.items.filter((item) => {{
        const done = Boolean(itemState(item).verdict);
        return filter === 'all' || (filter === 'done' ? done : !done);
      }});
      list.innerHTML = visible.length ? visible.map((item) => {{
        const state = itemState(item);
        const doneClass = state.verdict ? (state.verdict === 'correct' ? 'done' : 'error') : '';
        const timeline = item.timeline.map((row) => `<div class="line ${{row.context ? 'context' : ''}}"><span class="speaker">${{escapeHtml(row.speaker || '?')}}</span><span class="line-time">${{escapeHtml(row.start_label)}}-${{escapeHtml(row.end_label)}}</span><span>${{escapeHtml(row.text)}}</span></div>`).join('');
        const options = verdicts.map(([value, label]) => `<option value="${{value}}" ${{state.verdict === value ? 'selected' : ''}}>${{label}}</option>`).join('');
        return `<article class="${{doneClass}}" data-id="${{item.id}}">
          <div class="head"><div class="id">${{item.id}}</div><div class="recording">${{escapeHtml(item.recording)}}</div><div class="time">${{escapeHtml(item.review_time)}}</div><div class="category">${{escapeHtml(item.category)}}</div></div>
          <div class="body"><section><audio controls preload="metadata" src="${{escapeHtml(item.clip_path)}}"></audio><p class="prediction"><span>当前</span><span class="sequence">${{escapeHtml(item.current_prediction || '未标注')}}</span></p><div class="timeline">${{timeline}}</div><div class="reason">${{escapeHtml(item.reason)}}</div></section>
          <section class="form"><label>判断<select data-field="verdict">${{options}}</select></label><label>正确说话顺序<input data-field="correct_speaker_sequence" value="${{escapeHtml(state.correct_speaker_sequence)}}" placeholder="例如 B 或 A->D"></label><label>备注<input data-field="notes" value="${{escapeHtml(state.notes)}}"></label></section></div>
        </article>`;
      }}).join('') : '<div class="empty">没有符合当前筛选的片段</div>';
      const completed = manifest.items.filter((item) => itemState(item).verdict).length;
      document.getElementById('progress').textContent = `${{completed}} / ${{manifest.items.length}}`;
      list.querySelectorAll('select,input').forEach((control) => control.addEventListener('change', (event) => {{
        const article = event.target.closest('article');
        persist(article.dataset.id, {{[event.target.dataset.field]: event.target.value}});
      }}));
    }}
    document.querySelectorAll('[data-filter]').forEach((button) => button.addEventListener('click', () => {{
      filter = button.dataset.filter;
      document.querySelectorAll('[data-filter]').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
      render();
    }}));
    document.getElementById('export').addEventListener('click', () => {{
      const payload = {{
        schema_version: 1,
        pack_id: manifest.pack_id,
        source_manifest: manifest.manifest_filename,
        exported_at: new Date().toISOString(),
        items: manifest.items.map((item) => ({{
          id: item.id, recording: item.recording, category: item.category,
          review_start: item.review_start, review_end: item.review_end,
          current_prediction: item.current_prediction, ...itemState(item)
        }}))
      }};
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: 'application/json'}});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = '通用分人人工标注.json';
      link.click();
      URL.revokeObjectURL(link.href);
    }});
    render();
  </script>
</body>
</html>
"""


def _parse_case(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--case must be LABEL=/path/to/transcript.json")
    return label.strip(), Path(raw_path).expanduser().resolve()


def _project_cues(data: dict[str, Any]) -> dict[str, Any]:
    from scribe_py.ipc import _project_speaker_cues

    return _project_speaker_cues(data)


def write_pack(
    items: list[dict[str, Any]],
    *,
    out_dir: Path,
    dry_run: bool,
    excluded_cases: list[dict[str, str]] | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    for item in items:
        clip_name = f"{item['id']}_{_safe_name(str(item['recording']))}_{int(_as_float(item['review_start']) * 1000):010d}.wav"
        clip_path = clips_dir / clip_name
        item["clip_path"] = str(clip_path.relative_to(out_dir))
        item["review_time"] = f"{_fmt_time(_as_float(item['review_start']))}-{_fmt_time(_as_float(item['review_end']))}"
        for row in item.get("timeline") or []:
            row["start_label"] = _fmt_time(_as_float(row.get("start")))[3:8]
            row["end_label"] = _fmt_time(_as_float(row.get("end")))[3:8]
        if not dry_run:
            _extract_clip(
                Path(str(item["audio"])),
                clip_path,
                _as_float(item["clip_start"]),
                _as_float(item["clip_end"]),
            )

    pack_id = _manifest_hash(items)[:16]
    manifest = {
        "schema_version": 1,
        "pack_id": pack_id,
        "title": "通用分人验收",
        "manifest_filename": "通用分人验收清单.json",
        "item_count": len(items),
        "recordings": sorted({str(item.get("recording") or "") for item in items}),
        "excluded_recordings": list(excluded_cases or []),
        "category_counts": {
            category: sum(item.get("category") == category for item in items)
            for category in CATEGORY_ORDER
        },
        "items": items,
    }
    manifest_path = out_dir / manifest["manifest_filename"]
    html_path = out_dir / "开始标注.html"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(manifest), encoding="utf-8")
    return manifest_path, html_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成通用说话人分离人工验收包")
    parser.add_argument("--case", action="append", required=True, type=_parse_case, help="LABEL=/path/to/transcript.json，可重复")
    parser.add_argument("--out", required=True, type=Path, help="输出目录")
    parser.add_argument("--per-case", type=int, default=5, help="每份录音抽取片段数")
    parser.add_argument("--pad", type=float, default=1.5, help="片段前后附加音频秒数")
    parser.add_argument("--project-cues", action="store_true", help="先应用当前通用段内分人投影，再生成评测包")
    parser.add_argument("--dry-run", action="store_true", help="生成页面和清单但不切音频")
    args = parser.parse_args(argv)

    cases: list[tuple[str, Path, dict[str, Any]]] = []
    for label, path in args.case:
        data = _read_json(path)
        if args.project_cues:
            data = _project_cues(data)
        cases.append((label, path, data))

    eligible_cases, excluded_cases = partition_review_cases(cases)
    if not eligible_cases:
        reasons = "; ".join(
            f"{row['recording']}: {row['reason']}" for row in excluded_cases
        )
        raise SystemExit(f"没有可生成验收切片的可靠时间轴录音: {reasons}")

    items = build_review_items(
        eligible_cases,
        per_case=max(1, int(args.per_case)),
        padding_seconds=max(0.0, float(args.pad)),
    )
    manifest_path, html_path = write_pack(
        items,
        out_dir=args.out.expanduser().resolve(),
        dry_run=args.dry_run,
        excluded_cases=excluded_cases,
    )
    summary = {
        "ok": True,
        "items": len(items),
        "recordings": len(eligible_cases),
        "excluded_recordings": excluded_cases,
        "manifest": str(manifest_path),
        "html": str(html_path),
        "clips_created": 0 if args.dry_run else len(items),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
