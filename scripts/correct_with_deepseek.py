"""校对脚本:把 transcripts/<name>.json 送到 DeepSeek 修同音字 / 标点 / 专有名词。

读取顺序:
  1) 同目录 .env 的 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL
  2) swarmpath-platform-next/packages/knowledge-api/.env
  3) 进程环境变量
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWARMPATH_ENV = Path("/Users/apple/gitCommit/SwarmPathAI/swarmpath-platform-next/packages/knowledge-api/.env")

SYSTEM_PROMPT_DEFAULT = """你是中文语音转写文本的校对助手。输入是 Whisper 模型转录的中文片段,可能含同音字错误、专有名词识别错误、标点缺失等。

任务:
1. 修正显然的同音字 / 错别字(尤其是宗教、人名、地名等专有名词)
2. 补充自然的标点符号
3. 不要改写语义,不要重新组织句子,不要合并/拆分片段
4. 不要添加原文没有的内容,不要省略原文有的内容
5. 保留每个片段对应的 idx,逐条返回

输出格式:严格的 JSON,字段:`segments: [{"idx": int, "text": "校对后文本"}]`,顺序与输入一致。"""


def load_env() -> dict:
    cfg = {}
    if SWARMPATH_ENV.exists():
        cfg.update(dotenv_values(SWARMPATH_ENV))
    local_env = PROJECT_ROOT / ".env"
    if local_env.exists():
        cfg.update(dotenv_values(local_env))
    cfg.update({k: v for k, v in os.environ.items() if k.startswith("DEEPSEEK_")})
    return cfg


def correct_batch(client: OpenAI, model: str, batch: list[dict], context_hint: str) -> list[dict]:
    user_payload = {
        "context_hint": context_hint,
        "segments": [{"idx": i, "text": s["text"]} for i, s in enumerate(batch)],
    }
    rsp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_DEFAULT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    payload = json.loads(rsp.choices[0].message.content)
    by_idx = {s["idx"]: s["text"] for s in payload.get("segments", [])}
    out = []
    for i, seg in enumerate(batch):
        corrected_text = by_idx.get(i, seg["text"]).strip()
        out.append({**seg, "text": corrected_text, "original_text": seg["text"]})
    return out


def fmt_ts(seconds: float, comma: bool = False) -> str:
    millis = int(round(seconds * 1000))
    h, rem = divmod(millis, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    sep = "," if comma else "."
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript_json", nargs="?", default="transcripts/雅各书一章.json")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--context", default="圣经新约雅各书第一章的中文朗读。常见专有名词:雅各、耶稣基督、圣经、福音。")
    args = ap.parse_args()

    json_path = Path(args.transcript_json)
    if not json_path.is_absolute():
        json_path = PROJECT_ROOT / json_path
    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = data["segments"]
    print(f"[input] {json_path.name}  {len(segments)} segments  duration={data['duration']:.1f}s")

    cfg = load_env()
    api_key = cfg.get("DEEPSEEK_API_KEY")
    base_url = cfg.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        sys.exit("ERROR: DEEPSEEK_API_KEY not found in env or .env files")
    print(f"[provider] {base_url}  model={args.model}  key={api_key[:6]}***")

    client = OpenAI(api_key=api_key, base_url=base_url)

    t0 = time.time()
    corrected = []
    for i in range(0, len(segments), args.batch_size):
        batch = segments[i : i + args.batch_size]
        try:
            out = correct_batch(client, args.model, batch, args.context)
        except Exception as e:
            print(f"[batch {i}-{i+len(batch)}] ERROR: {e}  → keep original")
            out = [{**s, "original_text": s["text"]} for s in batch]
        corrected.extend(out)
        print(f"[batch {i}-{i+len(batch)}] done")

    elapsed = time.time() - t0
    changed = sum(1 for s in corrected if s["text"] != s["original_text"])
    print(f"[done] cost={elapsed:.1f}s  changed={changed}/{len(corrected)} segments")

    stem = json_path.stem
    out_dir = json_path.parent
    txt_path = out_dir / f"{stem}_corrected.txt"
    srt_path = out_dir / f"{stem}_corrected.srt"
    diff_path = out_dir / f"{stem}_diff.txt"
    json_out = out_dir / f"{stem}_corrected.json"

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"# {data.get('audio')}  (corrected by {args.model})\n\n")
        for s in corrected:
            f.write(f"[{fmt_ts(s['start'])} - {fmt_ts(s['end'])}] {s['text']}\n")

    with srt_path.open("w", encoding="utf-8") as f:
        for idx, s in enumerate(corrected, start=1):
            text = s["text"].strip()
            if not text:
                continue
            f.write(f"{idx}\n{fmt_ts(s['start'], comma=True)} --> {fmt_ts(s['end'], comma=True)}\n{text}\n\n")

    with diff_path.open("w", encoding="utf-8") as f:
        f.write(f"# diff: {changed} changes / {len(corrected)} segments\n\n")
        for s in corrected:
            if s["text"] != s["original_text"]:
                f.write(f"[{fmt_ts(s['start'])}]\n  - {s['original_text']}\n  + {s['text']}\n\n")

    with json_out.open("w", encoding="utf-8") as f:
        json.dump({**data, "corrected_by": args.model, "segments": corrected}, f, ensure_ascii=False, indent=2)

    print(f"[output]\n  - {txt_path}\n  - {srt_path}\n  - {diff_path}\n  - {json_out}")


if __name__ == "__main__":
    main()
