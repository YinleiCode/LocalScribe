#!/usr/bin/env python3
"""Build a blind continuous diarization annotation pack.

The pack intentionally excludes the system speaker labels from the manifest
and HTML. Predictions are written to a separate scoring directory. ASR text,
segment geometry, sync cues, and source audio are read-only inputs.
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
from collections import Counter
from pathlib import Path
from typing import Any


WINDOW_SECONDS = 600.0
WINDOW_STEP_SECONDS = 30.0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return clean or "recording"


def _audio_path(data: dict[str, Any], label: str) -> Path:
    raw = data.get("audio") or data.get("source_audio") or data.get("audio_path")
    path = Path(str(raw or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source audio does not exist for {label}: {path}")
    return path


def _timing_is_reliable(data: dict[str, Any]) -> bool:
    stats = data.get("filter_stats")
    return not isinstance(stats, dict) or stats.get("timing_reliable") is not False


def _cue_speaker_map(segment: dict[str, Any]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for cue in segment.get("speaker_cues") or []:
        if not isinstance(cue, dict):
            continue
        cue_index = int(_as_float(cue.get("cue_index"), -1))
        speaker = str(cue.get("speaker") or "").strip()
        if cue_index >= 0 and speaker:
            mapping[cue_index] = speaker
    return mapping


def prediction_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Project system labels onto immutable ASR cue geometry."""
    rows: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(data.get("segments") or []):
        if not isinstance(segment, dict):
            continue
        default_speaker = str(segment.get("speaker") or "").strip()
        cue_speakers = _cue_speaker_map(segment)
        sync_cues = [row for row in segment.get("sync_cues") or [] if isinstance(row, dict)]
        if sync_cues:
            for cue_index, cue in enumerate(sync_cues):
                start = _as_float(cue.get("start"), -1.0)
                end = _as_float(cue.get("end"), -1.0)
                speaker = cue_speakers.get(cue_index, default_speaker)
                if start < 0 or end <= start or not speaker:
                    continue
                rows.append({
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "speaker": speaker,
                    "text": str(cue.get("text") or "").strip(),
                    "segment_index": segment_index,
                    "cue_index": cue_index,
                })
            continue
        start = _as_float(segment.get("start"), -1.0)
        end = _as_float(segment.get("end"), -1.0)
        if start >= 0 and end > start and default_speaker:
            rows.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": default_speaker,
                "text": str(segment.get("text") or "").strip(),
                "segment_index": segment_index,
                "cue_index": None,
            })
    return sorted(rows, key=lambda row: (row["start"], row["end"]))


def _recording_end(data: dict[str, Any], rows: list[dict[str, Any]]) -> float:
    duration = _as_float(data.get("duration"))
    return max(duration, max((_as_float(row.get("end")) for row in rows), default=0.0))


def _clip_rows(rows: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    for row in rows:
        left = max(start, _as_float(row.get("start")))
        right = min(end, _as_float(row.get("end")))
        if right - left <= 0.001:
            continue
        clipped.append({**row, "start": round(left, 3), "end": round(right, 3)})
    return clipped


def _window_score(rows: list[dict[str, Any]], start: float, end: float) -> tuple[Any, ...]:
    clipped = _clip_rows(rows, start, end)
    durations: Counter[str] = Counter()
    sequence: list[str] = []
    speech_duration = 0.0
    for row in clipped:
        duration = _as_float(row["end"]) - _as_float(row["start"])
        speaker = str(row.get("speaker") or "")
        durations[speaker] += duration
        speech_duration += duration
        if not sequence or sequence[-1] != speaker:
            sequence.append(speaker)
    represented = [value for value in durations.values() if value >= 5.0]
    # Prefer windows that contain every sustained voice, then balanced turns,
    # then broad speech coverage. The same rule is applied to every recording.
    return (
        len(represented),
        round(min(represented), 3) if represented else 0.0,
        min(len(sequence) - 1, 40),
        round(speech_duration, 3),
        -round(start, 3),
    )


def choose_window(
    data: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    window_seconds: float = WINDOW_SECONDS,
    step_seconds: float = WINDOW_STEP_SECONDS,
) -> tuple[float, float, dict[str, Any]]:
    end = _recording_end(data, rows)
    if end <= 0:
        raise ValueError("recording has no valid timed speech")
    length = min(max(1.0, window_seconds), end)
    last_start = max(0.0, end - length)
    starts = [0.0]
    cursor = max(step_seconds, 1.0)
    while cursor < last_start:
        starts.append(cursor)
        cursor += max(step_seconds, 1.0)
    if last_start not in starts:
        starts.append(last_start)
    ranked = [(_window_score(rows, start, start + length), start) for start in starts]
    score, selected_start = max(ranked, key=lambda item: item[0])
    selected_end = min(end, selected_start + length)
    clipped = _clip_rows(rows, selected_start, selected_end)
    speakers = sorted({str(row.get("speaker") or "") for row in clipped})
    return round(selected_start, 3), round(selected_end, 3), {
        "selection_rule": "speaker_diversity_balance_turns_speech_coverage",
        "predicted_speaker_count_used_only_for_selection": len(speakers),
        "speech_cue_count": len(clipped),
        "score": list(score),
    }


def _extract_clip(audio: Path, output: Path, start: float, end: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", f"{start:.3f}", "-t", f"{max(0.05, end - start):.3f}",
            "-i", str(audio), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "aac", "-b:a", "64k", str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0 or not output.is_file() or output.stat().st_size <= 128:
        output.unlink(missing_ok=True)
        raise RuntimeError((process.stderr or "ffmpeg produced no audio clip").strip())


def _render_html(manifest: dict[str, Any]) -> str:
    manifest_json = json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c")
    title = html.escape(str(manifest.get("title") or "连续分人盲标"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#172026; background:#f3f5f6; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; }} button {{ font:inherit; }}
    header {{ position:sticky; top:0; z-index:10; border-bottom:1px solid #d5dade; background:rgba(255,255,255,.97); }}
    .top {{ max-width:1180px; margin:auto; padding:12px 18px; display:flex; align-items:center; gap:12px; }}
    h1 {{ margin:0; font-size:18px; letter-spacing:0; }} .spacer {{ flex:1; }}
    .export {{ min-height:38px; border:0; border-radius:6px; padding:0 14px; color:white; background:#176447; font-weight:700; cursor:pointer; }}
    main {{ max-width:1180px; margin:auto; padding:16px 18px 48px; }}
    nav {{ display:flex; gap:7px; margin-bottom:12px; flex-wrap:wrap; }}
    nav button {{ border:1px solid #b9c1c6; border-radius:5px; background:white; min-height:34px; padding:0 11px; cursor:pointer; }}
    nav button.active {{ border-color:#176447; color:white; background:#176447; }}
    article {{ display:none; }} article.active {{ display:block; }}
    .workspace {{ display:grid; grid-template-columns:minmax(310px,.8fr) minmax(420px,1.2fr); gap:14px; align-items:start; }}
    .panel {{ border:1px solid #d6dce0; border-radius:8px; background:white; overflow:hidden; }}
    .controls {{ position:sticky; top:64px; }} .section {{ padding:14px; border-bottom:1px solid #e5e9eb; }}
    .case-title {{ margin:0 0 4px; font-size:16px; }} .time-range {{ color:#65727a; font:12px ui-monospace,SFMono-Regular,Menlo,monospace; }}
    audio {{ width:100%; margin-top:12px; }} .clock {{ margin-top:8px; font:700 18px ui-monospace,SFMono-Regular,Menlo,monospace; color:#174f7c; }}
    .speaker-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:7px; }}
    .speaker-grid button {{ min-height:42px; border:1px solid #abb6bd; border-radius:6px; background:white; font-weight:800; cursor:pointer; }}
    .speaker-grid button.selected {{ border-color:#bd542e; color:white; background:#bd542e; }}
    .actions {{ display:grid; grid-template-columns:1fr auto auto; gap:7px; margin-top:9px; }}
    .actions button {{ min-height:40px; border:1px solid #aeb8be; border-radius:6px; background:#fff; padding:0 11px; cursor:pointer; }}
    .actions .add {{ border-color:#174f7c; color:white; background:#174f7c; font-weight:700; }}
    .actions .overlap.active {{ border-color:#bd542e; color:white; background:#bd542e; }}
    .markers {{ display:grid; gap:6px; max-height:280px; overflow:auto; }}
    .marker {{ display:grid; grid-template-columns:88px 1fr 32px; gap:7px; align-items:center; border-bottom:1px solid #edf0f2; padding:6px 0; }}
    .marker-time {{ border:0; padding:0; background:none; color:#174f7c; font:12px ui-monospace,SFMono-Regular,Menlo,monospace; text-align:left; cursor:pointer; }}
    .marker-speakers {{ font-weight:750; }} .delete {{ width:30px; height:30px; border:0; background:none; font-size:20px; color:#9d3b2f; cursor:pointer; }}
    .complete {{ width:100%; min-height:40px; border:1px solid #176447; border-radius:6px; color:#176447; background:white; font-weight:700; cursor:pointer; }}
    .complete.done {{ color:white; background:#176447; }}
    .transcript {{ max-height:calc(100vh - 108px); overflow:auto; }}
    .line {{ display:grid; grid-template-columns:88px 1fr; gap:10px; padding:9px 12px; border-bottom:1px solid #edf0f2; line-height:1.55; cursor:pointer; }}
    .line:hover {{ background:#f4f8fa; }} .line.current {{ background:#fff0b8; }}
    .line-time {{ color:#66747d; font:11px ui-monospace,SFMono-Regular,Menlo,monospace; padding-top:3px; }}
    .empty {{ color:#6b777f; font-size:13px; }}
    @media(max-width:820px) {{ .workspace {{ grid-template-columns:1fr; }} .controls {{ position:static; }} .transcript {{ max-height:none; }} .top {{ padding:10px; }} main {{ padding:10px; }} }}
  </style>
</head>
<body>
  <header><div class="top"><h1>{title}</h1><span id="progress"></span><div class="spacer"></div><button class="export" id="export" type="button">导出连续真值</button></div></header>
  <main><nav id="tabs"></nav><div id="cases"></div></main>
  <script id="manifest" type="application/json">{manifest_json}</script>
  <script>
    const manifest = JSON.parse(document.getElementById('manifest').textContent);
    const storageKey = `localscribe-continuous-review:${{manifest.pack_id}}`;
    const saved = JSON.parse(localStorage.getItem(storageKey) || '{{}}');
    const letters = ['A','B','C','D','E','F','G','H'];
    let activeCase = 0;
    const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    const fmt = (s) => {{ const ms=Math.round(Math.max(0,s)*1000), h=Math.floor(ms/3600000), m=Math.floor(ms%3600000/60000), x=Math.floor(ms%60000/1000), z=ms%1000; return `${{String(h).padStart(2,'0')}}:${{String(m).padStart(2,'0')}}:${{String(x).padStart(2,'0')}}.${{String(z).padStart(3,'0')}}`; }};
    const state = (item) => saved[item.id] || (saved[item.id] = {{markers:[], selected:['A'], complete:false}});
    const persist = () => {{ localStorage.setItem(storageKey, JSON.stringify(saved)); updateProgress(); }};
    function updateProgress() {{ const done=manifest.items.filter(i=>state(i).complete).length; document.getElementById('progress').textContent=`${{done}} / ${{manifest.items.length}}`; }}
    function markAt(item,audio,speakers) {{
      const s=state(item), time=Math.min(item.window_end,Math.max(item.window_start,item.window_start+audio.currentTime));
      const marker={{time:Math.round(time*1000)/1000,speakers:[...speakers]}};
      const old=s.markers.findIndex(x=>Math.abs(x.time-marker.time)<0.15);
      if(old>=0)s.markers[old]=marker; else s.markers.push(marker);
      s.markers.sort((a,b)=>a.time-b.time); s.complete=false; persist(); renderCase();
    }}
    function undoLast(item) {{ const s=state(item); s.markers.sort((a,b)=>a.time-b.time).pop(); s.complete=false; persist(); renderCase(); }}
    function renderShell() {{
      document.getElementById('tabs').innerHTML = manifest.items.map((item,i)=>`<button type="button" data-tab="${{i}}" class="${{i===activeCase?'active':''}}">${{esc(item.recording)}}</button>`).join('');
      document.getElementById('cases').innerHTML = manifest.items.map((item,i)=>`<article data-case="${{i}}" class="${{i===activeCase?'active':''}}"><div class="workspace"><section class="panel controls"><div class="section"><h2 class="case-title">${{esc(item.recording)}}</h2><div class="time-range">${{esc(item.window_label)}}</div><audio controls preload="metadata" src="${{esc(item.clip_path)}}"></audio><div class="clock">${{fmt(item.window_start)}}</div></div><div class="section"><div class="speaker-grid">${{letters.map(x=>`<button type="button" data-speaker="${{x}}">${{x}}</button>`).join('')}}</div><div class="actions"><button type="button" class="add">在当前时间开始</button><button type="button" class="overlap">重叠</button><button type="button" class="undo">撤销</button></div></div><div class="section"><div class="markers"></div></div><div class="section"><button type="button" class="complete">完成本段</button></div></section><section class="panel transcript">${{item.transcript.map((row,n)=>`<div class="line" data-row="${{n}}"><span class="line-time">${{esc(row.time_label)}}</span><span>${{esc(row.text)}}</span></div>`).join('')}}</section></div></article>`).join('');
      bind(); renderCase(); updateProgress();
    }}
    function renderCase() {{
      const item=manifest.items[activeCase], root=document.querySelector(`[data-case="${{activeCase}}"]`), s=state(item);
      root.querySelectorAll('[data-speaker]').forEach(b=>b.classList.toggle('selected',s.selected.includes(b.dataset.speaker)));
      root.querySelector('.overlap').classList.toggle('active',Boolean(s.overlap));
      root.querySelector('.markers').innerHTML=s.markers.length?s.markers.slice().sort((a,b)=>a.time-b.time).map((m,i)=>`<div class="marker"><button type="button" class="marker-time" data-seek="${{m.time}}">${{fmt(m.time)}}</button><span class="marker-speakers">${{esc(m.speakers.join('+'))}}</span><button type="button" class="delete" data-delete="${{i}}" title="删除">×</button></div>`).join(''):'<div class="empty">尚未标记</div>';
      const complete=root.querySelector('.complete'); complete.classList.toggle('done',s.complete); complete.textContent=s.complete?'已完成':'完成本段';
      root.querySelectorAll('[data-seek]').forEach(b=>b.onclick=()=>{{root.querySelector('audio').currentTime=Math.max(0,Number(b.dataset.seek)-item.window_start);}});
      root.querySelectorAll('[data-delete]').forEach(b=>b.onclick=()=>{{const ordered=s.markers.slice().sort((a,b)=>a.time-b.time); ordered.splice(Number(b.dataset.delete),1); s.markers=ordered; s.complete=false; persist(); renderCase();}});
    }}
    function bind() {{
      document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{{activeCase=Number(b.dataset.tab); document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x===b)); document.querySelectorAll('[data-case]').forEach(x=>x.classList.toggle('active',Number(x.dataset.case)===activeCase)); renderCase();}});
      manifest.items.forEach((item,i)=>{{
        const root=document.querySelector(`[data-case="${{i}}"]`), audio=root.querySelector('audio'), clock=root.querySelector('.clock'), lines=[...root.querySelectorAll('.line')];
        root.querySelectorAll('[data-speaker]').forEach(b=>b.onclick=()=>{{const s=state(item), x=b.dataset.speaker; if(s.overlap){{s.selected=s.selected.includes(x)?s.selected.filter(v=>v!==x):[...s.selected,x].sort(); if(!s.selected.length)s.selected=[x]; persist(); renderCase();}}else{{s.selected=[x]; markAt(item,audio,[x]);}}}});
        root.querySelector('.overlap').onclick=()=>{{const s=state(item); s.overlap=!s.overlap; if(!s.overlap&&s.selected.length>1)s.selected=[s.selected[0]]; persist(); renderCase();}};
        root.querySelector('.add').onclick=()=>markAt(item,audio,state(item).selected);
        root.querySelector('.undo').onclick=()=>undoLast(item);
        root.querySelector('.complete').onclick=()=>{{const s=state(item); if(!s.markers.length)return; s.complete=!s.complete; persist(); renderCase();}};
        lines.forEach((line,n)=>line.onclick=()=>{{audio.currentTime=Math.max(0,item.transcript[n].start-item.window_start); audio.play();}});
        const sync=()=>{{const now=item.window_start+audio.currentTime; clock.textContent=fmt(now); let active=-1; item.transcript.forEach((row,n)=>{{if(now>=row.start&&now<row.end)active=n;}}); lines.forEach((line,n)=>line.classList.toggle('current',n===active)); if(active>=0&&audio.paused===false)lines[active].scrollIntoView({{block:'nearest'}});}};
        audio.addEventListener('timeupdate',sync); audio.addEventListener('seeked',sync);
      }});
      document.addEventListener('keydown',(event)=>{{
        if(event.metaKey||event.ctrlKey||event.altKey||['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName))return;
        const item=manifest.items[activeCase],root=document.querySelector(`[data-case="${{activeCase}}"]`),audio=root.querySelector('audio');
        const letter=event.key.toUpperCase();
        if(letters.includes(letter)){{event.preventDefault();const s=state(item);s.overlap=false;s.selected=[letter];markAt(item,audio,[letter]);return;}}
        if(event.code==='Space'){{event.preventDefault();if(audio.paused)audio.play();else audio.pause();return;}}
        if(event.key==='Backspace'){{event.preventDefault();undoLast(item);return;}}
        if(event.key==='ArrowLeft'||event.key==='ArrowRight'){{event.preventDefault();audio.currentTime=Math.max(0,Math.min(audio.duration||item.window_end-item.window_start,audio.currentTime+(event.key==='ArrowLeft'?-2:2)));}}
      }});
    }}
    function intersectMarkers(item,s) {{
      const markers=s.markers.slice().sort((a,b)=>a.time-b.time), result=[];
      markers.forEach((marker,index)=>{{const regionEnd=index+1<markers.length?markers[index+1].time:item.window_end; item.speech_ranges.forEach(range=>{{const start=Math.max(marker.time,range.start),end=Math.min(regionEnd,range.end); if(end-start>0.001)marker.speakers.forEach(speaker=>result.push({{uri:item.uri,start:Math.round(start*1000)/1000,end:Math.round(end*1000)/1000,speaker}}));}});}});
      return result;
    }}
    document.getElementById('export').onclick=()=>{{
      const incomplete=manifest.items.filter(item=>!state(item).complete||!state(item).markers.length); if(incomplete.length){{alert(`还有 ${{incomplete.length}} 段未完成`);return;}}
      const segments=manifest.items.flatMap(item=>intersectMarkers(item,state(item)));
      const payload={{schema_version:1,kind:'continuous_diarization_gold',pack_id:manifest.pack_id,source_manifest:manifest.manifest_filename,exported_at:new Date().toISOString(),items:manifest.items.map(item=>({{id:item.id,recording:item.recording,uri:item.uri,window_start:item.window_start,window_end:item.window_end,markers:state(item).markers,complete:state(item).complete}})),segments}};
      const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}),link=document.createElement('a'); link.href=URL.createObjectURL(blob); link.download='连续分人人工真值.json'; link.click(); URL.revokeObjectURL(link.href);
    }};
    renderShell();
  </script>
</body>
</html>"""


def build_pack(
    cases: list[tuple[str, Path]],
    out_dir: Path,
    *,
    window_seconds: float = WINDOW_SECONDS,
    dry_run: bool = False,
) -> tuple[Path, Path, Path]:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    scoring_dir = out_dir / "scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}

    for number, (label, transcript_path) in enumerate(cases, start=1):
        transcript_path = transcript_path.expanduser().resolve()
        before = transcript_path.read_bytes()
        data = _read_json(transcript_path)
        if not _timing_is_reliable(data):
            raise ValueError(f"timing is explicitly unreliable for {label}")
        audio = _audio_path(data, label)
        rows = prediction_rows(data)
        window_start, window_end, diagnostics = choose_window(
            data, rows, window_seconds=window_seconds
        )
        clipped = _clip_rows(rows, window_start, window_end)
        uri = f"continuous-{number:02d}"
        clip_name = f"{number:02d}_{_safe_name(label)}_{int(window_start * 1000):010d}.m4a"
        clip_path = clips_dir / clip_name
        if not dry_run:
            _extract_clip(audio, clip_path, window_start, window_end)
        transcript_rows = [
            {
                "start": row["start"],
                "end": row["end"],
                "text": row["text"],
                "time_label": f"{_fmt_time(row['start'])[3:8]}-{_fmt_time(row['end'])[3:8]}",
            }
            for row in clipped
        ]
        speech_ranges = [{"start": row["start"], "end": row["end"]} for row in clipped]
        items.append({
            "id": f"CONT-{number:02d}",
            "recording": label,
            "uri": uri,
            "window_start": window_start,
            "window_end": window_end,
            "window_label": f"{_fmt_time(window_start)} - {_fmt_time(window_end)}",
            "clip_path": str(clip_path.relative_to(out_dir)),
            "transcript": transcript_rows,
            "speech_ranges": speech_ranges,
        })
        predictions.extend({
            "uri": uri,
            "start": row["start"],
            "end": row["end"],
            "speaker": row["speaker"],
        } for row in clipped)
        source_hashes[label] = _sha256(transcript_path)
        if transcript_path.read_bytes() != before:
            raise RuntimeError(f"source transcript changed while building pack: {transcript_path}")
        diagnostics_path = scoring_dir / f"{number:02d}_{_safe_name(label)}_window.json"
        diagnostics_path.write_text(json.dumps({
            "recording": label,
            "transcript": str(transcript_path),
            "audio": str(audio),
            "window_start": window_start,
            "window_end": window_end,
            **diagnostics,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    stable = [{key: item[key] for key in ("id", "recording", "uri", "window_start", "window_end")} for item in items]
    pack_id = hashlib.sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    manifest = {
        "schema_version": 1,
        "kind": "continuous_diarization_blind_review",
        "pack_id": pack_id,
        "title": "连续分人盲标",
        "manifest_filename": "连续分人盲标清单.json",
        "source_transcript_sha256": source_hashes,
        "system_speaker_labels_exposed": False,
        "items": items,
    }
    manifest_path = out_dir / manifest["manifest_filename"]
    html_path = out_dir / "开始连续盲标.html"
    prediction_path = scoring_dir / "当前通用分人预测.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    prediction_path.write_text(json.dumps({
        "schema_version": 1,
        "kind": "continuous_diarization_prediction",
        "pack_id": pack_id,
        "segments": predictions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(_render_html(manifest), encoding="utf-8")
    return manifest_path, html_path, prediction_path


def _parse_case(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--case must be LABEL=/path/to/transcript.json")
    return label.strip(), Path(raw_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成连续十分钟说话人盲标验收包")
    parser.add_argument("--case", action="append", required=True, type=_parse_case)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest, page, prediction = build_pack(
        args.case,
        args.out,
        window_seconds=max(1.0, args.window_seconds),
        dry_run=args.dry_run,
    )
    print(json.dumps({
        "ok": True,
        "manifest": str(manifest),
        "html": str(page),
        "prediction": str(prediction),
        "recordings": len(args.case),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
