"""把转录稿整理成正式文章版本(单次 LLM 调用)。"""
import argparse
import json
import re
import time
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWARMPATH_ENV = Path("/Users/apple/gitCommit/SwarmPathAI/swarmpath-platform-next/packages/knowledge-api/.env")

SYSTEM_PROMPT = """你是中文文章排版编辑。输入是一段语音转写后的文本(可能缺标点、断句不规整、有少量错字)。

任务:
1. 补全所有缺失的标点符号(逗号、句号、问号、感叹号、引号等)
2. 按语义和节奏分段(每段 3-6 句较合理)
3. 修正剩余的明显错别字 / 同音字
4. 不要增删原文意思,不要改写句子结构
5. 不要加入解释、注释、小标题
6. 直接输出排版后的纯文本,不要包裹任何说明

输出:整理后的纯文本文章,段与段之间用一个空行分隔。"""


def load_api_key():
    cfg = {}
    if SWARMPATH_ENV.exists():
        cfg.update(dotenv_values(SWARMPATH_ENV))
    return cfg.get("DEEPSEEK_API_KEY"), cfg.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", nargs="?", default="transcripts/雅各书一章_corrected.json")
    ap.add_argument("--model", default="deepseek-chat")
    args = ap.parse_args()

    src = Path(args.json_path)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    data = json.loads(src.read_text(encoding="utf-8"))
    raw = "".join(s["text"] for s in data["segments"])
    raw = re.sub(r"\s+", "", raw)

    api_key, base_url = load_api_key()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not found")

    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"[input] {len(raw)} chars  → {args.model}")

    t0 = time.time()
    rsp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw},
        ],
        temperature=0.1,
    )
    polished = rsp.choices[0].message.content.strip()
    print(f"[done] cost={time.time()-t0:.1f}s  output={len(polished)} chars")

    stem = src.stem.replace("_corrected", "")
    out = src.parent / f"{stem}_完整版.txt"
    out.write_text(
        f"# {data.get('audio')} — 完整文字稿\n"
        f"# 时长 {data['duration']:.1f}s · 校对+排版 {args.model}\n\n"
        f"{polished}\n",
        encoding="utf-8",
    )
    print(f"[output] {out}")


if __name__ == "__main__":
    main()
