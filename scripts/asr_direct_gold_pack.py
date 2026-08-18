#!/usr/bin/env python3
"""Build a gold listening pack directly from source audio windows.

Unlike transcript-driven sampling, this workflow never trusts historical ASR
timestamps. It cuts a source-audio window first and transcribes that exact WAV
to guarantee that the review text and playback file refer to the same audio.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PY_SRC = ROOT / "scribe-py" / "src"
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

from scribe_py.core.audio import probe_audio  # noqa: E402
from scribe_py.core.selector import default_model_id, make_transcriber  # noqa: E402
from scribe_py.core.types import TranscribeOptions  # noqa: E402


_TEXT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]")
_ANCHORS = (0.10, 0.22, 0.35, 0.50, 0.64, 0.78, 0.90)
_APP_NORMALIZER_PROFILE = "legacy_general"
_APP_AUDIO_PREPROCESS = "adaptive"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def candidate_windows(duration: float, clip_seconds: float) -> list[tuple[float, float]]:
    """Return deterministic, widely distributed source-audio windows."""
    if duration <= 0:
        return [(0.0, max(clip_seconds, 0.1))]
    length = min(max(clip_seconds, 0.1), duration)
    if duration <= length:
        return [(0.0, duration)]
    windows: list[tuple[float, float]] = []
    seen: set[int] = set()
    for anchor in _ANCHORS:
        start = min(max((duration * anchor) - (length / 2), 0.0), duration - length)
        key = int(round(start * 1000))
        if key in seen:
            continue
        seen.add(key)
        windows.append((start, length))
    return windows


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")
    return ffmpeg


def extract_clip(audio: Path, output: Path, start: float, duration: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            _ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            f"{max(start, 0.0):.3f}",
            "-t",
            f"{max(duration, 0.1):.3f}",
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


def _fmt_time(seconds: float) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def _result_text(result: Any) -> str:
    return "\n".join(
        str(segment.text or "").strip()
        for segment in result.segments
        if str(segment.text or "").strip()
    )


def _text_chars(text: str) -> int:
    return len(_TEXT_RE.findall(text))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pack_id(items: list[dict[str, Any]]) -> str:
    identity = [
        {
            "case_id": item["case_id"],
            "audio_sha256": item["audio_sha256"],
            "start": item["start"],
            "duration": item["duration"],
            "clip_sha256": item["clip_sha256"],
        }
        for item in items
    ]
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"asr-direct-gold-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def render_markdown(items: list[dict[str, Any]]) -> str:
    lines = [
        "# ASR 通用人工标准集（原始音频直接抽样）\n\n",
        f"- 录音数：{len({item['case_id'] for item in items})}\n",
        f"- 抽样数：{len(items)}\n",
        f"- 音频总时长：约 {sum(float(item['duration']) for item in items) / 60:.1f} 分钟\n",
        "- 时间轴来源：原始音频绝对时间；未使用历史转录时间戳\n\n",
        "逐个播放 `clips/` 中的音频。当前文字完全正确时回复 `GOLD-001 确认`；有错误时回复 `GOLD-001：正确完整文字`。\n\n",
        "| ID | 录音 | 原录音时间 | 当前文字 |\n",
        "|---|---|---|---|\n",
    ]
    for item in items:
        text = str(item["current_text"]).replace("\n", "<br>").replace("|", "\\|")
        start = float(item["start"])
        end = start + float(item["duration"])
        lines.append(
            f"| {item['id']} | {item['case']} | {_fmt_time(start)}-{_fmt_time(end)} | {text} |\n"
        )
    return "".join(lines)


def render_html(payload: dict[str, Any]) -> str:
    """Render a self-contained review page that exports the gold JSON."""
    payload_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    title = html.escape(str(payload.get("title") or "ASR 通用盲测标注"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#172026; background:#f3f5f6; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; }} button,textarea {{ font:inherit; }}
    header {{ position:sticky; top:0; z-index:10; border-bottom:1px solid #d6dcdf; background:rgba(255,255,255,.97); }}
    .top {{ max-width:960px; margin:auto; padding:12px 18px; display:flex; align-items:center; gap:12px; }}
    h1 {{ margin:0; font-size:18px; letter-spacing:0; }} .progress {{ color:#5d6a72; }} .spacer {{ flex:1; }}
    .export {{ min-height:38px; border:0; border-radius:6px; padding:0 14px; color:white; background:#176447; font-weight:700; cursor:pointer; }}
    main {{ max-width:960px; margin:auto; padding:16px 18px 48px; display:grid; gap:12px; }}
    article {{ border:1px solid #d5dbdf; border-radius:8px; background:white; overflow:hidden; }}
    article.done {{ border-color:#589176; }} article.unusable {{ border-color:#bd694d; }}
    .meta {{ display:flex; gap:10px; align-items:baseline; padding:12px 14px; border-bottom:1px solid #e6eaec; }}
    .id {{ font-weight:800; color:#174f7c; }} .recording {{ font-weight:700; }} .time {{ margin-left:auto; color:#65727a; font:12px ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .body {{ padding:14px; display:grid; gap:10px; }} audio {{ width:100%; }}
    .label {{ color:#65727a; font-size:12px; }} .current {{ line-height:1.65; padding:10px 12px; background:#f5f7f8; border-left:3px solid #7593a6; }}
    textarea {{ width:100%; min-height:82px; resize:vertical; border:1px solid #b8c1c6; border-radius:6px; padding:10px 11px; line-height:1.55; }}
    textarea:focus {{ outline:2px solid #8ab6cf; border-color:#174f7c; }}
    .actions {{ display:flex; gap:8px; flex-wrap:wrap; }} .actions button {{ min-height:38px; border:1px solid #aeb8be; border-radius:6px; background:white; padding:0 13px; cursor:pointer; }}
    .actions .confirm {{ border-color:#176447; color:white; background:#176447; font-weight:700; }}
    .actions .correct {{ border-color:#174f7c; color:white; background:#174f7c; font-weight:700; }}
    .actions .unclear {{ color:#8a3d28; }} .status {{ margin-left:auto; align-self:center; color:#176447; font-weight:700; }}
    @media(max-width:640px) {{ .top,main {{ padding-left:10px; padding-right:10px; }} .meta {{ align-items:flex-start; flex-wrap:wrap; }} .time {{ width:100%; margin-left:0; }} .status {{ width:100%; margin-left:0; }} }}
  </style>
</head>
<body>
  <header><div class="top"><h1>{title}</h1><span class="progress" id="progress"></span><div class="spacer"></div><button class="export" id="export" type="button">导出标注</button></div></header>
  <main id="items"></main>
  <script id="manifest" type="application/json">{payload_json}</script>
  <script>
    const manifest = JSON.parse(document.getElementById('manifest').textContent);
    const storageKey = `localscribe-asr-gold:${{manifest.pack_id}}`;
    const saved = JSON.parse(localStorage.getItem(storageKey) || '{{}}');
    const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    const state = (item) => saved[item.id] || (saved[item.id] = {{decision:'', correct_text:'', notes:''}});
    const persist = () => {{ localStorage.setItem(storageKey, JSON.stringify(saved)); updateProgress(); }};
    const fmt = (s) => {{ const ms=Math.round(Math.max(0,s)*1000),h=Math.floor(ms/3600000),m=Math.floor(ms%3600000/60000),x=Math.floor(ms%60000/1000),z=ms%1000; return `${{String(h).padStart(2,'0')}}:${{String(m).padStart(2,'0')}}:${{String(x).padStart(2,'0')}}.${{String(z).padStart(3,'0')}}`; }};
    function updateProgress() {{
      const done = manifest.items.filter(item => state(item).decision).length;
      document.getElementById('progress').textContent = `${{done}} / ${{manifest.items.length}}`;
    }}
    function render() {{
      document.getElementById('items').innerHTML = manifest.items.map(item => {{
        const s=state(item), start=Number(item.start), end=start+Number(item.duration);
        const status=s.decision==='confirmed'?'正确':s.decision==='corrected'?'已修改':s.decision==='unusable'?'听不清':'';
        return `<article data-id="${{esc(item.id)}}" class="${{s.decision?'done ':''}}${{s.decision==='unusable'?'unusable':''}}"><div class="meta"><span class="id">${{esc(item.id)}}</span><span class="recording">${{esc(item.case)}}</span><span class="time">${{fmt(start)}} - ${{fmt(end)}}</span></div><div class="body"><audio controls preload="metadata" src="${{esc(item.clip_path)}}"></audio><div><div class="label">当前转文字</div><div class="current">${{esc(item.current_text).replace(/\\n/g,'<br>')}}</div></div><div><div class="label">正确完整文字</div><textarea aria-label="正确完整文字" placeholder="有错误时在这里修改">${{esc(s.correct_text || item.current_text)}}</textarea></div><div class="actions"><button type="button" class="confirm">文字正确</button><button type="button" class="correct">保存修改</button><button type="button" class="unclear">听不清/不计分</button><span class="status">${{status}}</span></div></div></article>`;
      }}).join('');
      bind(); updateProgress();
    }}
    function bind() {{
      document.querySelectorAll('article').forEach(root => {{
        const item=manifest.items.find(row=>row.id===root.dataset.id), textarea=root.querySelector('textarea');
        root.querySelector('.confirm').onclick=()=>{{ const s=state(item); s.decision='confirmed'; s.correct_text=item.current_text; persist(); render(); }};
        root.querySelector('.correct').onclick=()=>{{ const value=textarea.value.trim(); if(!value){{textarea.focus();return;}} const s=state(item); s.decision=value===item.current_text.trim()?'confirmed':'corrected'; s.correct_text=value; persist(); render(); }};
        root.querySelector('.unclear').onclick=()=>{{ const s=state(item); s.decision='unusable'; s.correct_text=''; persist(); render(); }};
        textarea.oninput=()=>{{ const s=state(item); s.correct_text=textarea.value; if(s.decision)s.decision=''; persist(); }};
      }});
    }}
    document.getElementById('export').onclick=()=>{{
      const items=manifest.items.map(item=>({{...item,...state(item)}}));
      const payload={{...manifest, exported_at:new Date().toISOString(), items}};
      const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}), a=document.createElement('a');
      a.href=URL.createObjectURL(blob); a.download='ASR通用人工标准答案.json'; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),1000);
    }};
    render();
  </script>
</body>
</html>"""


def build_pack(
    suite: dict[str, Any],
    *,
    out_dir: Path,
    clip_seconds: float,
    min_text_chars: int,
    normalizer_profile: str = _APP_NORMALIZER_PROFILE,
    audio_preprocess: str = _APP_AUDIO_PREPROCESS,
) -> list[dict[str, Any]]:
    clips_dir = out_dir / "clips"
    baseline_dir = out_dir / "baseline_results"
    transcriber = make_transcriber("sensevoice")
    model_id = default_model_id("sensevoice")
    items: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []

    for case in suite.get("cases") or []:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id") or "").strip()
        label = str(case.get("label") or case_id).strip()
        audio = Path(str(case.get("audio") or "")).expanduser().resolve()
        wanted = max(int(case.get("samples") or 0), 0)
        if not case_id or not audio.exists() or wanted <= 0:
            raise ValueError(f"invalid direct-gold case: {case}")
        duration = float(probe_audio(audio).get("duration") or 0.0)
        audio_sha256 = _sha256(audio)
        accepted = 0
        for candidate_index, (start, length) in enumerate(
            candidate_windows(duration, clip_seconds), start=1
        ):
            candidate = out_dir / "candidates" / f"{case_id}_{candidate_index:02d}.wav"
            extract_clip(audio, candidate, start, length)
            result = transcriber.transcribe(
                candidate,
                TranscribeOptions(
                    language="zh",
                    model_id=model_id,
                    normalizer_profile=normalizer_profile,
                    audio_preprocess=audio_preprocess,
                ),
            )
            text = _result_text(result)
            attempt = {
                "case_id": case_id,
                "candidate": candidate_index,
                "start": start,
                "duration": length,
                "text_chars": _text_chars(text),
                "accepted": False,
            }
            if _text_chars(text) < min_text_chars:
                attempts.append(attempt)
                continue

            accepted += 1
            item_id = f"GOLD-{len(items) + 1:03d}"
            clip_path = clips_dir / f"{item_id}_{case_id}_{int(round(start * 1000)):010d}.wav"
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate, clip_path)
            result_path = baseline_dir / f"{item_id}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            item = {
                "id": item_id,
                "case_id": case_id,
                "case": label,
                "audio": str(audio),
                "audio_sha256": audio_sha256,
                "start": round(start, 3),
                "duration": round(length, 3),
                "clip_path": str(clip_path.relative_to(out_dir)),
                "clip_sha256": _sha256(clip_path),
                "current_text": text,
                "correct_text": "",
                "decision": "",
                "notes": "",
                "baseline_result": str(result_path.relative_to(out_dir)),
            }
            items.append(item)
            attempt["accepted"] = True
            attempt["item_id"] = item_id
            attempts.append(attempt)
            if accepted >= wanted:
                break
        if accepted < wanted:
            raise RuntimeError(f"{label} only produced {accepted}/{wanted} speech windows")

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "template": "ASR原始音频直接抽样人工标准集",
        "title": "ASR 通用盲测标注",
        "pack_id": _pack_id(items),
        "timing_source": "source_audio_absolute",
        "asr_config": {
            "backend": "sensevoice",
            "model_id": model_id,
            "language": "zh",
            "normalizer_profile": normalizer_profile,
            "audio_preprocess": audio_preprocess,
        },
        "items": items,
    }
    (out_dir / "人工标准答案模板.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "人工核对说明.md").write_text(render_markdown(items), encoding="utf-8")
    (out_dir / "开始标注.html").write_text(render_html(payload), encoding="utf-8")
    (out_dir / "抽样诊断.json").write_text(
        json.dumps(attempts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从原始音频直接切片并生成 ASR 人工标准集")
    parser.add_argument("suite", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--min-text-chars", type=int, default=8)
    parser.add_argument("--normalizer-profile", default=_APP_NORMALIZER_PROFILE)
    parser.add_argument("--audio-preprocess", default=_APP_AUDIO_PREPROCESS)
    args = parser.parse_args(argv)

    out_dir = args.out.expanduser().resolve()
    items = build_pack(
        _read_json(args.suite.expanduser().resolve()),
        out_dir=out_dir,
        clip_seconds=max(args.clip_seconds, 0.1),
        min_text_chars=max(args.min_text_chars, 1),
        normalizer_profile=str(args.normalizer_profile or _APP_NORMALIZER_PROFILE),
        audio_preprocess=str(args.audio_preprocess or _APP_AUDIO_PREPROCESS),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "items": len(items),
                "out": str(out_dir),
                "markdown": str(out_dir / "人工核对说明.md"),
                "html": str(out_dir / "开始标注.html"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
