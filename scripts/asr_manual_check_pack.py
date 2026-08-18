#!/usr/bin/env python3
"""Create a small, human-listenable verification pack for ASR changes.

The input is the JSON emitted by ``scripts/asr_compare_jsons.py`` for the
standard and strong ASR modes.  Every accepted automatic edit is grouped by
its review window, cut from the original audio, and rendered as a Chinese
side-by-side review sheet.  This script is deliberately read-only with
respect to transcripts: it never edits the standard or strong JSON files.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_time(value: float) -> str:
    total_ms = int(round(max(value, 0.0) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def _safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return clean or "review"


def _overlap_text(data: dict[str, Any], start: float, end: float) -> str:
    parts: list[str] = []
    for segment in data.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        seg_start = _as_float(segment.get("start"))
        seg_end = _as_float(segment.get("end"))
        if min(seg_end, end) - max(seg_start, start) <= 0:
            continue
        text = str(segment.get("text") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _window_key(item: dict[str, Any]) -> tuple[str, float, float]:
    return (
        str(item.get("case") or ""),
        round(_as_float(item.get("start")), 3),
        round(_as_float(item.get("end")), 3),
    )


def build_items(
    comparison: dict[str, Any],
    *,
    standard_dir: Path,
    strong_dir: Path,
    padding_seconds: float,
) -> list[dict[str, Any]]:
    """Group automatic substitutions into review windows without cutting audio."""
    grouped: OrderedDict[tuple[str, float, float], list[dict[str, Any]]] = OrderedDict()
    categories = {
        str(row.get("case") or ""): str(row.get("category") or "")
        for row in comparison.get("rows") or []
        if isinstance(row, dict)
    }
    for raw in comparison.get("manual_checks") or []:
        if not isinstance(raw, dict):
            continue
        key = _window_key(raw)
        if not key[0] or key[2] <= key[1]:
            continue
        grouped.setdefault(key, []).append(raw)

    items: list[dict[str, Any]] = []
    for number, ((case, start, end), changes) in enumerate(grouped.items(), start=1):
        standard_path = standard_dir / f"{case}.json"
        strong_path = strong_dir / f"{case}.json"
        if not standard_path.exists() or not strong_path.exists():
            missing = standard_path if not standard_path.exists() else strong_path
            raise FileNotFoundError(f"missing transcript for {case}: {missing}")
        standard = _read_json(standard_path)
        strong = _read_json(strong_path)
        audio = Path(str(strong.get("audio") or standard.get("audio") or "")).expanduser()
        if not audio.exists():
            raise FileNotFoundError(f"source audio for {case} does not exist: {audio}")
        duration = _as_float(strong.get("duration"), _as_float(standard.get("duration")))
        clip_start = max(0.0, start - max(padding_seconds, 0.0))
        clip_end = end + max(padding_seconds, 0.0)
        if duration > 0:
            clip_end = min(clip_end, duration)
        items.append(
            {
                "id": f"ASR-{number:03d}",
                "case": case,
                "category": categories.get(case, ""),
                "review_start": start,
                "review_end": end,
                "clip_start": clip_start,
                "clip_end": max(clip_end, clip_start + 0.05),
                "audio": str(audio.resolve()),
                "standard_text": _overlap_text(standard, start, end),
                "strong_text": _overlap_text(strong, start, end),
                "changes": [
                    {
                        "from": str(change.get("from") or ""),
                        "to": str(change.get("to") or ""),
                        "evidence": str(change.get("evidence") or ""),
                        "decision": "",
                        "correct_to": "",
                        "notes": "",
                    }
                    for change in changes
                ],
                "decision": "",
                "final_text": "",
                "notes": "",
            }
        )
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
            f"{max(end - start, 0.05):.3f}",
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
    if process.returncode != 0 or not output.exists() or output.stat().st_size <= 44:
        output.unlink(missing_ok=True)
        raise RuntimeError((process.stderr or "ffmpeg produced no audio clip").strip())


def render_markdown(items: list[dict[str, Any]]) -> str:
    lines = [
        "# ASR 高质量模式人工抽听\n\n",
        "本包只覆盖自动接受的改字。请以音频为准，不需要检查整段录音。\n\n",
        "操作：逐个播放 `clips/` 下同名音频。若高质量文本正确，回复 `ASR-001 确认`；若不正确，回复 `ASR-001：正确完整文字`。\n\n",
        "| ID | 样本 | 原始时间 | 自动改字 | 标准模式 | 高质量模式 |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for item in items:
        changes = "；".join(f"{change['from']} -> {change['to']}" for change in item["changes"])
        standard = str(item["standard_text"]).replace("\n", "<br>").replace("|", "\\|")
        strong = str(item["strong_text"]).replace("\n", "<br>").replace("|", "\\|")
        original_time = f"{_fmt_time(float(item['review_start']))}-{_fmt_time(float(item['review_end']))}"
        lines.append(
            f"| {item['id']} | {item['case']} | {original_time} | {changes} | {standard} | {strong} |\n"
        )
    lines.extend(
        [
            "\n## 音频文件\n\n",
        ]
    )
    for item in items:
        lines.append(f"- `{item['clip_path']}`\n")
    return "".join(lines)


def write_pack(
    items: list[dict[str, Any]],
    *,
    out_dir: Path,
    dry_run: bool,
) -> tuple[Path, Path]:
    clips_dir = out_dir / "clips"
    for item in items:
        clip_name = (
            f"{item['id']}_{_safe_name(str(item['case']))}_"
            f"{int(round(float(item['review_start']) * 1000)):010d}_"
            f"{int(round(float(item['review_end']) * 1000)):010d}.wav"
        )
        clip_path = clips_dir / clip_name
        item["clip_path"] = str(clip_path.relative_to(out_dir))
        if not dry_run:
            _extract_clip(
                Path(str(item["audio"])),
                clip_path,
                float(item["clip_start"]),
                float(item["clip_end"]),
            )
    manifest_path = out_dir / "人工确认模板.json"
    markdown_path = out_dir / "人工抽听说明.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "template": "ASR高质量模式人工抽听确认",
                "instruction": "以音频为准。decision 填确认或驳回；驳回时填写 final_text。",
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(items), encoding="utf-8")
    return manifest_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 ASR 自动改字整理为可抽听的音频对照包")
    parser.add_argument("--comparison", type=Path, required=True, help="asr_compare_jsons.py 输出的 comparison.json")
    parser.add_argument("--standard-dir", type=Path, required=True, help="标准模式 JSON 目录")
    parser.add_argument("--strong-dir", type=Path, required=True, help="高质量模式 JSON 目录")
    parser.add_argument("--out", type=Path, required=True, help="输出目录")
    parser.add_argument("--pad", type=float, default=2.0, help="每段前后保留的上下文秒数")
    parser.add_argument("--dry-run", action="store_true", help="仅生成说明和确认模板，不切音频")
    args = parser.parse_args(argv)

    comparison_path = args.comparison.expanduser().resolve()
    standard_dir = args.standard_dir.expanduser().resolve()
    strong_dir = args.strong_dir.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    items = build_items(
        _read_json(comparison_path),
        standard_dir=standard_dir,
        strong_dir=strong_dir,
        padding_seconds=args.pad,
    )
    manifest_path, markdown_path = write_pack(items, out_dir=out_dir, dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "ok": True,
                "items": len(items),
                "manifest": str(manifest_path),
                "markdown": str(markdown_path),
                "clips_created": 0 if args.dry_run else len(items),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
