#!/usr/bin/env python3
"""Turn an ASR gold sample template into a balanced listening pack."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


_TEXT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]")
_TYPE_PRIORITY = {"strong": 0, "weak": 1, "normal": 2}


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _text_chars(value: Any) -> int:
    return len(_TEXT_RE.findall(str(value or "")))


def _fmt_time(seconds: float) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def _safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return clean or "sample"


def select_rows(
    template: dict[str, Any],
    *,
    min_duration: float = 1.5,
    min_text_chars: int = 4,
    per_case: int = 3,
) -> list[dict[str, Any]]:
    """Select balanced rows, preferring strong/weak samples in each case."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    case_order: list[str] = []
    for raw in template.get("items") or []:
        if not isinstance(raw, dict):
            continue
        case = str(raw.get("case") or "").strip()
        start = float(raw.get("start") or 0.0)
        end = float(raw.get("end") or start)
        if not case or end - start < min_duration:
            continue
        if _text_chars(raw.get("current_text")) < min_text_chars:
            continue
        if case not in grouped:
            case_order.append(case)
        grouped[case].append(dict(raw))

    selected: list[dict[str, Any]] = []
    for case in case_order:
        rows = sorted(
            grouped[case],
            key=lambda row: (
                _TYPE_PRIORITY.get(str(row.get("sample_type") or "normal"), 99),
                float(row.get("start") or 0.0),
            ),
        )
        selected.extend(rows[: max(per_case, 0)])
    return selected


def build_pack_items(
    template: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    padding_seconds: float,
) -> list[dict[str, Any]]:
    transcripts = {
        str(case.get("case") or ""): Path(str(case.get("transcript") or "")).expanduser()
        for case in template.get("cases") or []
        if isinstance(case, dict)
    }
    transcript_cache: dict[Path, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    for number, row in enumerate(rows, start=1):
        case = str(row.get("case") or "")
        transcript_path = transcripts.get(case)
        if transcript_path is None or not transcript_path.exists():
            raise FileNotFoundError(f"transcript missing for case {case}: {transcript_path}")
        data = transcript_cache.setdefault(transcript_path, _read_json(transcript_path))
        audio = Path(str(data.get("audio") or "")).expanduser()
        if not audio.exists():
            raise FileNotFoundError(f"audio missing for case {case}: {audio}")
        start = float(row.get("start") or 0.0)
        end = float(row.get("end") or start)
        duration = float(data.get("duration") or data.get("duration_s") or 0.0)
        clip_start = max(0.0, start - max(padding_seconds, 0.0))
        clip_end = end + max(padding_seconds, 0.0)
        if duration > 0:
            clip_end = min(clip_end, duration)
        items.append(
            {
                "id": f"GOLD-{number:03d}",
                "case": case,
                "index": int(row.get("index") or 0),
                "sample_type": str(row.get("sample_type") or "normal"),
                "start": start,
                "end": end,
                "clip_start": clip_start,
                "clip_end": max(clip_end, clip_start + 0.05),
                "audio": str(audio.resolve()),
                "current_text": str(row.get("current_text") or ""),
                "correct_text": "",
                "decision": "",
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
        raise RuntimeError((process.stderr or "ffmpeg produced no clip").strip())


def _render_markdown(items: list[dict[str, Any]]) -> str:
    total_seconds = sum(float(item["clip_end"]) - float(item["clip_start"]) for item in items)
    lines = [
        "# ASR 通用人工标准集\n\n",
        f"- 录音数：{len({str(item['case']) for item in items})}\n",
        f"- 抽样数：{len(items)}\n",
        f"- 抽听总时长：约 {total_seconds / 60:.1f} 分钟\n\n",
        "逐个播放 `clips/` 中的音频。当前文字完全正确时回复 `GOLD-001 确认`；有错误时回复 `GOLD-001：正确完整文字`。\n\n",
        "| ID | 录音 | 类型 | 原录音时间 | 当前文字 |\n",
        "|---|---|---|---|---|\n",
    ]
    type_names = {"strong": "强疑点", "weak": "普通疑点", "normal": "正常抽查"}
    for item in items:
        text = str(item["current_text"]).replace("\n", "<br>").replace("|", "\\|")
        when = f"{_fmt_time(float(item['start']))}-{_fmt_time(float(item['end']))}"
        lines.append(
            f"| {item['id']} | {item['case']} | "
            f"{type_names.get(str(item['sample_type']), str(item['sample_type']))} | {when} | {text} |\n"
        )
    return "".join(lines)


def write_pack(items: list[dict[str, Any]], *, out_dir: Path, dry_run: bool = False) -> tuple[Path, Path]:
    clips_dir = out_dir / "clips"
    eval_clips_dir = out_dir / "eval_clips"
    for item in items:
        clip_name = (
            f"{item['id']}_{_safe_name(str(item['case']))}_"
            f"{int(round(float(item['start']) * 1000)):010d}_"
            f"{int(round(float(item['end']) * 1000)):010d}.wav"
        )
        clip_path = clips_dir / clip_name
        eval_clip_path = eval_clips_dir / clip_name
        item["clip_path"] = str(clip_path.relative_to(out_dir))
        item["eval_clip_path"] = str(eval_clip_path.relative_to(out_dir))
        if not dry_run:
            _extract_clip(
                Path(str(item["audio"])),
                clip_path,
                float(item["clip_start"]),
                float(item["clip_end"]),
            )
            _extract_clip(
                Path(str(item["audio"])),
                eval_clip_path,
                float(item["start"]),
                float(item["end"]),
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "人工标准答案模板.json"
    markdown_path = out_dir / "人工核对说明.md"
    json_path.write_text(
        json.dumps(
            {
                "template": "ASR通用人工标准集",
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(items), encoding="utf-8")
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="将 ASR gold 抽样模板打包成均衡人工抽听集")
    parser.add_argument("template", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-case", type=int, default=3)
    parser.add_argument("--min-duration", type=float, default=1.5)
    parser.add_argument("--min-text-chars", type=int, default=4)
    parser.add_argument("--pad", type=float, default=1.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    template_path = args.template.expanduser().resolve()
    template = _read_json(template_path)
    rows = select_rows(
        template,
        min_duration=max(args.min_duration, 0.0),
        min_text_chars=max(args.min_text_chars, 0),
        per_case=max(args.per_case, 0),
    )
    items = build_pack_items(template, rows, padding_seconds=max(args.pad, 0.0))
    json_path, markdown_path = write_pack(
        items,
        out_dir=args.out.expanduser().resolve(),
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "cases": len({item["case"] for item in items}),
                "items": len(items),
                "json": str(json_path),
                "markdown": str(markdown_path),
                "listening_clips_created": 0 if args.dry_run else len(items),
                "eval_clips_created": 0 if args.dry_run else len(items),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
