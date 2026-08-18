"""把 _corrected.json 整理成无时间戳的连续散文版本。"""
import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def to_prose(segments: list[dict], sentences_per_paragraph: int = 4) -> str:
    """拼接所有片段 → 按句号断句 → 每 N 句合并为一段。"""
    full = "".join(s["text"] for s in segments)
    full = re.sub(r"\s+", "", full)
    parts = re.split(r"(?<=[。！？])", full)
    parts = [p for p in parts if p.strip()]

    paragraphs = []
    buf = []
    for p in parts:
        buf.append(p)
        if len(buf) >= sentences_per_paragraph:
            paragraphs.append("".join(buf))
            buf = []
    if buf:
        paragraphs.append("".join(buf))
    return "\n\n".join(paragraphs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", nargs="?", default="transcripts/雅各书一章_corrected.json")
    ap.add_argument("--per-paragraph", type=int, default=4, help="每段几句话")
    args = ap.parse_args()

    src = Path(args.json_path)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    data = json.loads(src.read_text(encoding="utf-8"))
    segments = data["segments"]

    prose = to_prose(segments, args.per_paragraph)

    stem = src.stem.replace("_corrected", "")
    out = src.parent / f"{stem}_完整版.txt"
    out.write_text(
        f"# {data.get('audio')} — 完整文字稿\n"
        f"# 时长 {data['duration']:.1f}s · {len(segments)} 段 · 校对 {data.get('corrected_by', 'n/a')}\n\n"
        f"{prose}\n",
        encoding="utf-8",
    )

    sentence_count = len(re.findall(r"[。！？]", prose))
    char_count = len(re.sub(r"\s+", "", prose))
    print(f"[output] {out}")
    print(f"  段落={prose.count(chr(10)+chr(10)) + 1}  句子={sentence_count}  字数={char_count}")


if __name__ == "__main__":
    main()
