"""转录脚本:用 mlx-whisper 在 Apple GPU 上转一个音频文件,输出 txt/srt/json 三件套。"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx_whisper

REPO = "mlx-community/whisper-large-v3-turbo"


def fmt_ts(seconds: float, comma: bool = False) -> str:
    millis = int(round(seconds * 1000))
    h, rem = divmod(millis, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    sep = "," if comma else "."
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def transcribe(audio: Path, out_dir: Path, language: str = "zh") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio.stem
    print(f"[transcribe] {audio.name} → {out_dir}/")

    t0 = time.time()
    result = mlx_whisper.transcribe(
        str(audio),
        path_or_hf_repo=REPO,
        language=language,
        word_timestamps=False,
    )
    elapsed = time.time() - t0

    segments = result["segments"]
    duration = segments[-1]["end"] if segments else 0
    rtf = elapsed / duration if duration else 0
    print(f"[done] {len(segments)} segments  duration={duration:.1f}s  cost={elapsed:.1f}s  rtf={rtf:.3f}x")

    txt_path = out_dir / f"{stem}.txt"
    srt_path = out_dir / f"{stem}.srt"
    json_path = out_dir / f"{stem}.json"

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"# {audio.name}\n")
        f.write(f"# language={result.get('language')}  duration={duration:.1f}s  segments={len(segments)}\n\n")
        for seg in segments:
            text = seg["text"].strip()
            if text:
                f.write(f"[{fmt_ts(seg['start'])} - {fmt_ts(seg['end'])}] {text}\n")

    with srt_path.open("w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            f.write(f"{idx}\n")
            f.write(f"{fmt_ts(seg['start'], comma=True)} --> {fmt_ts(seg['end'], comma=True)}\n")
            f.write(f"{text}\n\n")
            idx += 1

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "audio": audio.name,
                "language": result.get("language"),
                "duration": duration,
                "transcribe_seconds": elapsed,
                "rtf": rtf,
                "model": REPO,
                "segments": [
                    {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                    for s in segments
                    if s["text"].strip()
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[output]\n  - {txt_path}\n  - {srt_path}\n  - {json_path}")


if __name__ == "__main__":
    audio = Path(sys.argv[1] if len(sys.argv) > 1 else "雅各书一章.m4a")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "transcripts")
    if not audio.is_absolute():
        audio = Path(__file__).resolve().parent.parent / audio
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent.parent / out_dir
    transcribe(audio, out_dir)
