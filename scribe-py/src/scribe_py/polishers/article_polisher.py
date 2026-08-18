"""LLM 整篇排版:把转录段落拼接 → 加标点 → 分段 → 输出连续散文。

DeepSeek 等 LLM 服务端有"单次输出 token 数"硬上限(通常 8K-32K),客户端传
`max_tokens=384000` 只是上限提示,服务端会按自己能力截断。所以长稿(>5000 字)
**必须分块** —— 否则会出现"原 25000 字仅生成 10000 字 (41%)"这种截断问题。

分块策略:
- 按 segments 边界(自然停顿)聚合,避免拆断句
- 每块 ≤ CHUNK_INPUT_CHARS(默认 4000 字)
- 中文 1 字 ≈ 0.5-0.8 token,4000 字输入 → 约 5K tokens 输出,远在 DeepSeek 8K 上限内
"""
from __future__ import annotations

import re

from openai import OpenAI

from ..core.types import Segment

SYSTEM_PROMPT = """你是中文文章排版编辑。输入是一段语音转写后的文本(可能缺标点、断句不规整、有少量错字)。

任务:
1. 补全所有缺失的标点符号(逗号、句号、问号、感叹号、引号等)
2. 按语义和节奏分段(每段 3-6 句较合理)
3. 修正剩余的明显错别字 / 同音字
4. 不要增删原文意思,不要改写句子结构
5. 不要加入解释、注释、小标题
6. 直接输出排版后的纯文本,不要包裹任何说明
7. **必须输出简体中文**(GB18030 字符集),严禁出现繁体字。若输入混入繁体字,统一转为简体

输出:整理后的纯文本文章,段与段之间用一个空行分隔。"""


SYSTEM_PROMPT_DIALOGUE = """你是中文对话整理编辑。输入是 N 个说话人的语音转写,**每行开头标有说话人名字**(格式:`【NAME】内容`)。

任务:
1. **严格保留每个回合的说话人归属** —— 不要把 A 的话改给 B,也不要凭空生成新的 speaker
2. **同一人连续多个回合,合并为一个段落**(去掉中间的【NAME】标记)
3. 给每个回合补全所有缺失的标点(逗号、句号、问号、感叹号、引号等)
4. 修正明显的错别字 / 同音字
5. 不增删原文意思,不改写语义
6. 不要加解释、注释、小标题
7. **必须输出简体中文**(GB18030 字符集),严禁繁体字。输入若混入繁体,统一转简体

输出格式(严格遵守):
```
**NAME1:** NAME1 的整段发言,合并相邻回合,补好标点。

**NAME2:** NAME2 的回应内容...

**NAME1:** NAME1 又说话了...
```

要点:
- 每个回合前用 `**NAME:**`(双星号包裹,冒号后一个空格)
- 回合之间空一行
- NAME 直接用输入里的【NAME】里那个名字,**原封不动复制**(可能是 SPEAKER_A 也可能是真实姓名如"陈总""客户")"""


class ArticlePolisher:
    name = "article_polisher"

    # 单次请求输入字数上限(中文字符);超过则分块。
    # 4000 字对应 ~5K tokens 输出,留 60% 余量给 DeepSeek 8K 输出上限,稳。
    CHUNK_INPUT_CHARS = 4000

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        temperature: float = 0.3,
        max_tokens: int = 384000,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def polish(self, segments: list[Segment], on_progress=None) -> dict:
        """Returns dict: {text, finish_reason, truncated, input_chars, chunks, mode}.

        on_progress(dict) 用于汇报分块进度(可选)。

        模式自动选择:
          - segments 里 unique speaker ≤ 1(或无 speaker)→ 流文章模式(原有逻辑)
          - ≥ 2 → 对话模式,输出 `**NAME:**` 对话格式,保留说话人归属
        """
        if not segments:
            return {"text": "", "finish_reason": "stop", "truncated": False,
                    "input_chars": 0, "chunks": 0, "mode": "monologue"}

        # 分人检测:用 segment.speaker(可能已被用户改名为 "陈总" 等)
        speakers_in_segs = {s.speaker for s in segments if getattr(s, "speaker", None)}
        is_dialogue = len(speakers_in_segs) >= 2

        if is_dialogue:
            return self._polish_dialogue(segments, on_progress)
        return self._polish_monologue(segments, on_progress)

    # ------- 单人模式(原有路径)-------
    def _polish_monologue(self, segments, on_progress=None) -> dict:
        seg_texts = []
        for s in segments:
            t = re.sub(r"\s+", "", s.text or "")
            if t:
                seg_texts.append(t)

        raw = "".join(seg_texts)
        total_chars = len(raw)
        if not raw:
            return {"text": "", "finish_reason": "stop", "truncated": False,
                    "input_chars": 0, "chunks": 0, "mode": "monologue"}

        if total_chars <= self.CHUNK_INPUT_CHARS:
            r = self._polish_text(raw, SYSTEM_PROMPT)
            r["input_chars"] = total_chars
            r["chunks"] = 1
            r["mode"] = "monologue"
            return r

        chunks = self._chunk_segment_texts(seg_texts, self.CHUNK_INPUT_CHARS)
        polished_parts: list[str] = []
        any_truncated = False
        for i, chunk_text in enumerate(chunks):
            if on_progress:
                on_progress({"stage": "polish_chunk", "current": i, "total": len(chunks)})
            r = self._polish_text(chunk_text, SYSTEM_PROMPT)
            polished_parts.append(r["text"])
            if r["truncated"]:
                any_truncated = True
        if on_progress:
            on_progress({"stage": "polish_chunk", "current": len(chunks), "total": len(chunks)})

        merged = "\n\n".join(t for t in polished_parts if t)
        return {
            "text": merged,
            "finish_reason": "length" if any_truncated else "stop",
            "truncated": any_truncated,
            "input_chars": total_chars,
            "chunks": len(chunks),
            "mode": "monologue",
        }

    # ------- 对话模式(多人)-------
    def _polish_dialogue(self, segments, on_progress=None) -> dict:
        """把多人 segments 合并为对话体。

        1. 先按 speaker 把相邻同人的 segments 合成一个 turn(去掉同人内部边界)
        2. 序列化成 `【NAME】内容` 一行一行,作为 LLM 输入
        3. 长稿分块时**优先在 speaker 切换处切**,避免把一个人发言切碎
        4. LLM 输出 `**NAME:**` 格式的对话
        """
        # Step 1: 合并相邻同人段落 → turns
        turns: list[tuple[str, str]] = []  # (speaker, text)
        for s in segments:
            who = getattr(s, "speaker", None) or "SPEAKER_?"
            txt = re.sub(r"\s+", "", s.text or "")
            if not txt:
                continue
            if turns and turns[-1][0] == who:
                turns[-1] = (who, turns[-1][1] + txt)
            else:
                turns.append((who, txt))

        if not turns:
            return {"text": "", "finish_reason": "stop", "truncated": False,
                    "input_chars": 0, "chunks": 0, "mode": "dialogue"}

        # 序列化每个 turn:`【NAME】内容\n`
        def serialize(ts: list[tuple[str, str]]) -> str:
            return "\n".join(f"【{w}】{t}" for w, t in ts)

        total_chars = sum(len(t) for _, t in turns)

        # Step 2: 决定单次 vs 分块
        if total_chars <= self.CHUNK_INPUT_CHARS:
            r = self._polish_text(serialize(turns), SYSTEM_PROMPT_DIALOGUE)
            r["input_chars"] = total_chars
            r["chunks"] = 1
            r["mode"] = "dialogue"
            return r

        # Step 3: 分块,在 speaker 边界(turn 之间)切
        chunks = self._chunk_turns(turns, self.CHUNK_INPUT_CHARS)
        polished_parts: list[str] = []
        any_truncated = False
        for i, chunk_turns in enumerate(chunks):
            if on_progress:
                on_progress({"stage": "polish_chunk", "current": i, "total": len(chunks)})
            r = self._polish_text(serialize(chunk_turns), SYSTEM_PROMPT_DIALOGUE)
            polished_parts.append(r["text"])
            if r["truncated"]:
                any_truncated = True
        if on_progress:
            on_progress({"stage": "polish_chunk", "current": len(chunks), "total": len(chunks)})

        merged = "\n\n".join(t for t in polished_parts if t)
        return {
            "text": merged,
            "finish_reason": "length" if any_truncated else "stop",
            "truncated": any_truncated,
            "input_chars": total_chars,
            "chunks": len(chunks),
            "mode": "dialogue",
        }

    @staticmethod
    def _chunk_turns(turns: list[tuple[str, str]], target_chars: int) -> list[list[tuple[str, str]]]:
        """按 turn 边界贪心打包到 target_chars 大小的块。永远不切断单个 turn。"""
        chunks: list[list[tuple[str, str]]] = []
        buf: list[tuple[str, str]] = []
        buf_len = 0
        for t in turns:
            tlen = len(t[1])
            if buf and buf_len + tlen > target_chars:
                chunks.append(buf)
                buf = [t]
                buf_len = tlen
            else:
                buf.append(t)
                buf_len += tlen
        if buf:
            chunks.append(buf)
        return chunks

    @staticmethod
    def _chunk_segment_texts(seg_texts: list[str], target_chars: int) -> list[str]:
        """贪心:在 segment 边界处把文本聚合到约 target_chars 长的块。

        永远不在 segment 内部切。如果单个 segment 已经超过 target_chars,它独占
        一块(此时该块输出可能仍被截断,但这是极端情况)。
        """
        chunks: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for t in seg_texts:
            # 加入后会超过 target、且 buf 非空 → 先把 buf 落盘
            if buf and buf_len + len(t) > target_chars:
                chunks.append("".join(buf))
                buf = [t]
                buf_len = len(t)
            else:
                buf.append(t)
                buf_len += len(t)
        if buf:
            chunks.append("".join(buf))
        return chunks

    def _polish_text(self, raw: str, system_prompt: str = SYSTEM_PROMPT) -> dict:
        """单次 LLM 排版调用。返回 {text, finish_reason, truncated}。"""
        rsp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        choice = rsp.choices[0]
        finish_reason = choice.finish_reason or "stop"
        text = (choice.message.content or "").strip()
        # 防御性:不管 prompt 怎么写,强制把繁体转简体(zhconv 纯字典,纯 Python)
        try:
            from zhconv import convert
            text = convert(text, "zh-hans")
        except Exception:
            pass
        return {
            "text": text,
            "finish_reason": finish_reason,
            "truncated": finish_reason == "length",
        }
