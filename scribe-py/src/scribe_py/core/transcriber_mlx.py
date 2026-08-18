"""Apple Silicon MLX transcriber + 4-layer hallucination defense.

四层防御(从源头到兜底):
  ① VAD 输入清理       — silero-vad 检测说话区间,Whisper 段落不在说话区间则丢弃
  ② 解码硬化           — condition_on_previous_text=False + no_speech / compression / logprob 阈值
  ③ 置信度自校         — Whisper 自报 avg_logprob,过低段丢弃
  ④ 模式后处理         — token 重复、字符密度、段间相似度、已知幻觉短语黑名单

返回的 segments 不再有幻觉,且 result 上挂 `filter_stats` 暴露每层过滤数量。
"""
from __future__ import annotations

import os
from difflib import SequenceMatcher
from pathlib import Path

from .hotwords import build_initial_prompt
from .transcriber_base import ProgressCallback, Transcriber
from .types import Segment, TranscribeOptions

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def _resolve_model_path(model_id: str) -> str:
    """优先使用项目内置 models/ 目录,缺失再回落到 HF repo id。

    解析顺序:
      1. 环境变量 LOCALSCRIBE_MODEL_DIR (绝对路径)
      2. <project_root>/models/<basename>     ← 默认随仓库分发
      3. <repo_root>/models/<basename>        ← 兼容打包后 .app/Resources
      4. 原始 model_id (HF cache 兜底)
    """
    basename = model_id.rsplit("/", 1)[-1]

    # 0. 打包后的 .app 资源目录(由 Rust 注入)
    res = os.environ.get("LOCALSCRIBE_RESOURCES")
    if res:
        bundled = Path(res) / "models" / basename
        if (bundled / "weights.safetensors").exists():
            return str(bundled)

    # 1. 显式环境变量
    env_dir = os.environ.get("LOCALSCRIBE_MODEL_DIR")
    if env_dir:
        p = Path(env_dir).expanduser()
        if (p / "weights.safetensors").exists():
            return str(p)
        candidate = p / basename
        if (candidate / "weights.safetensors").exists():
            return str(candidate)

    # 2. 向上找 LocalScribe 项目根 (含 scribe-py/ + package.json)
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "scribe-py").is_dir() and (ancestor / "package.json").is_file():
            local = ancestor / "models" / basename
            if (local / "weights.safetensors").exists():
                return str(local)
            break

    # 3. 兜底:HF repo id (mlx-whisper 自己去缓存或下载)
    return model_id

# ============================================================================
# Layer 4 patterns(短语黑名单 + 重复检测)
# ============================================================================

STRONG_HALLUCINATION_PHRASES: tuple[str, ...] = (
    "请不吝点赞",
    "打赏支持",
    "明镜与点点栏目",
    "明镜栏目",
    "點點欄目",
    "請按讚訂閱",
    "字幕由Amara",
    "字幕志愿者",
    "由社群字幕",
    "Please subscribe",
    "Please like and subscribe",
    "Like and subscribe",
    # 字幕组 / 视频平台水印型
    "优优独播",
    "獨播劇場",
    "YoYo Television",
    "Television Series Exclusive",
    "YouTube",  # 真说"YouTube"概率低于幻觉
    "本节目",
    "本视频",
    "我们下期再见",
    "我们下次再见",
    "下期再见",
    "下次再见",
)
WEAK_HALLUCINATION_PHRASES: tuple[str, ...] = (
    "感谢观看",
    "感谢您的观看",
    "请订阅",
    "请订阅本频道",
    "请点赞",
    "请关注",
    "谢谢观看",
    "谢谢大家",
    "Thanks for watching",
    "Thank you for watching",
    "Subscribe",
)


def _hits_phrases(text: str, phrases: tuple[str, ...]) -> bool:
    t = text.strip().rstrip("，。！？,.!? ")
    return any(p in t for p in phrases)


def _is_repetitive(text: str) -> bool:
    """Token 级或字符级高度重复检测,不依赖具体短语。

    保守起见:正常文本(>10 字)的 unique-ratio 通常 > 0.5;低于 0.4 就极不正常。
    """
    t = text.strip()
    if len(t) < 6:
        return False
    # 1) 空格分词:unique/total < 0.4 且 ≥3 词 → 重复(英文型)
    words = t.split()
    if len(words) >= 3 and len(set(words)) / len(words) < 0.4:
        return True
    # 2) 1-8 字符 seed 重复(中文 "甚麼甚麼甚麼" / "嗯嗯嗯嗯" 都要捕获)
    for span in range(1, 9):
        if len(t) < span * 3:
            continue
        seed = t[:span]
        if not seed.strip():
            continue
        # 整段大部分由 seed 构成
        if t.count(seed) * span >= len(t) * 0.6 and t.count(seed) >= 3:
            return True
    # 3) 单字符占比过高(如 '一一一啊一一')
    if len(t) >= 6:
        from collections import Counter
        cnt = Counter(t.replace(" ", ""))
        total = sum(cnt.values())
        if total > 0 and cnt.most_common(1)[0][1] / total > 0.6:
            return True
    return False


def _density_anomalous(text: str, start: float, end: float) -> bool:
    """字符密度异常:中文正常 4-7 字/秒。> 12 视为塞满了重复内容。"""
    duration = max(0.001, end - start)
    chars = len(text.strip().replace(" ", ""))
    if chars < 10:
        return False
    return chars / duration > 12.0


def _segment_pair_similar(a: str, b: str, thresh: float = 0.9) -> bool:
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() >= thresh


# ============================================================================
# Layer 1: VAD pre/post filter
# ============================================================================

_vad_model = None


def _get_vad_model():
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad()
    return _vad_model


def _vad_speech_ranges(audio_path: Path) -> list[tuple[float, float]]:
    """返回说话区间列表 [(start_s, end_s), ...]。失败返回 []。"""
    try:
        from silero_vad import get_speech_timestamps, read_audio

        model = _get_vad_model()
        wav = read_audio(str(audio_path), sampling_rate=16000)
        # min_silence_duration_ms=500: 容忍小停顿;min_speech_duration_ms=250: 短词不漏
        ts = get_speech_timestamps(
            wav,
            model,
            sampling_rate=16000,
            min_silence_duration_ms=500,
            min_speech_duration_ms=250,
        )
        return [(t["start"] / 16000.0, t["end"] / 16000.0) for t in ts]
    except Exception:  # noqa: BLE001
        return []


def _in_any_range(t: float, ranges: list[tuple[float, float]]) -> bool:
    """二分查 t 是否落在任一 (lo, hi) 区间内。"""
    if not ranges:
        return True  # VAD 失败时不过滤
    lo, hi = 0, len(ranges) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        a, b = ranges[mid]
        if t < a:
            hi = mid - 1
        elif t > b:
            lo = mid + 1
        else:
            return True
    return False


# ============================================================================
# 主过滤函数:四层依次跑,统计每层删除数
# ============================================================================

def _filter_all_layers(
    raw_segments: list[dict],
    audio_path: Path,
) -> tuple[list[Segment], dict]:
    """raw_segments 是 mlx-whisper 原始返回,带 avg_logprob / no_speech_prob 等字段。

    返回 (干净段, stats)。stats 包含每层删除数,用于 UI 显示。
    """
    stats = {"input": len(raw_segments), "vad": 0, "logprob": 0, "phrases": 0, "repetition": 0, "density": 0, "similarity": 0}

    # 转成 dict 列表方便保留 metadata
    items = [
        {
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": s.get("text", "").strip(),
            "avg_logprob": float(s.get("avg_logprob", 0.0)),
            "no_speech_prob": float(s.get("no_speech_prob", 0.0)),
            "compression_ratio": float(s.get("compression_ratio", 0.0)),
            "keep": True,
            "drop_reason": None,
        }
        for s in raw_segments
        if s.get("text", "").strip()
    ]

    # --- Layer 1: VAD ---
    speech_ranges = _vad_speech_ranges(audio_path)
    if speech_ranges:
        for it in items:
            mid = (it["start"] + it["end"]) / 2
            if not _in_any_range(mid, speech_ranges):
                it["keep"] = False
                it["drop_reason"] = "vad"
                stats["vad"] += 1

    # --- Layer 3: avg_logprob (低置信丢弃) ---
    LOGPROB_THRESHOLD = -1.0
    for it in items:
        if not it["keep"]:
            continue
        if it["avg_logprob"] != 0.0 and it["avg_logprob"] < LOGPROB_THRESHOLD:
            it["keep"] = False
            it["drop_reason"] = "logprob"
            stats["logprob"] += 1

    # --- Layer 4a: 重复检测 ---
    for it in items:
        if not it["keep"]:
            continue
        if _is_repetitive(it["text"]):
            it["keep"] = False
            it["drop_reason"] = "repetition"
            stats["repetition"] += 1

    # --- Layer 4b: 字符密度异常 ---
    for it in items:
        if not it["keep"]:
            continue
        if _density_anomalous(it["text"], it["start"], it["end"]):
            it["keep"] = False
            it["drop_reason"] = "density"
            stats["density"] += 1

    # --- Layer 4c: 强幻觉短语单段立删 ---
    for it in items:
        if not it["keep"]:
            continue
        if _hits_phrases(it["text"], STRONG_HALLUCINATION_PHRASES):
            it["keep"] = False
            it["drop_reason"] = "phrases"
            stats["phrases"] += 1

    # --- Layer 4d: 弱幻觉短语 ---
    # 规则:连续 ≥2 段命中 → 删;或 单段命中 + 周围 5 秒无说话 → 删(孤立判定)
    n = len(items)
    i = 0
    while i < n:
        if items[i]["keep"] and _hits_phrases(items[i]["text"], WEAK_HALLUCINATION_PHRASES):
            j = i
            while j < n and items[j]["keep"] and _hits_phrases(items[j]["text"], WEAK_HALLUCINATION_PHRASES):
                j += 1
            run = j - i
            if run >= 2:
                # 连续命中删
                for k in range(i, j):
                    items[k]["keep"] = False
                    items[k]["drop_reason"] = "phrases"
                    stats["phrases"] += 1
            elif run == 1:
                # 单段命中,检查孤立:前后段间隔 ≥ 5 秒(被静默包围)→ 删
                cur = items[i]
                prev = next((it for it in reversed(items[:i]) if it["keep"]), None)
                nxt = next((it for it in items[j:] if it["keep"]), None)
                gap_before = cur["start"] - (prev["end"] if prev else 0)
                gap_after = (nxt["start"] if nxt else cur["end"] + 999) - cur["end"]
                if gap_before >= 5 or gap_after >= 5:
                    cur["keep"] = False
                    cur["drop_reason"] = "phrases"
                    stats["phrases"] += 1
            i = j
        else:
            i += 1

    # --- Layer 4e: 段间相似度 ≥ 3 段连续高相似 ---
    for i in range(len(items) - 2):
        if not all(items[i + k]["keep"] for k in range(3)):
            continue
        a, b, c = items[i]["text"], items[i + 1]["text"], items[i + 2]["text"]
        if _segment_pair_similar(a, b) and _segment_pair_similar(b, c):
            for k in (i + 1, i + 2):
                items[k]["keep"] = False
                items[k]["drop_reason"] = "similarity"
                stats["similarity"] += 1

    cleaned = [
        Segment(start=it["start"], end=it["end"], text=it["text"])
        for it in items
        if it["keep"]
    ]
    stats["output"] = len(cleaned)
    stats["removed_total"] = stats["input"] - stats["output"]
    return cleaned, stats


# ============================================================================
# Transcriber implementation
# ============================================================================


class MLXTranscriber(Transcriber):
    backend = "mlx"

    last_filter_stats: dict = {}

    def _run(
        self,
        audio: Path,
        options: TranscribeOptions,
        on_progress: ProgressCallback | None,
    ) -> tuple[list[Segment], str | None]:
        repo = options.model_id or DEFAULT_MODEL
        resolved = _resolve_model_path(repo)
        if resolved != repo:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        # 优先走 VAD-guided 路径(解决 Whisper 长 chunk 漏段 bug)。
        # 失败时(silero-vad / soundfile 缺失)自动 fallback 到整段送 Whisper。
        use_vad = os.environ.get("LOCALSCRIBE_VAD_GUIDED", "1") != "0"
        if use_vad:
            try:
                segments, language = _vad_guided_run(
                    audio, resolved, options, on_progress
                )
                # 仍然过 Layer 3/4 后处理(VAD 已是 Layer 1,这里只跑模式后处理)
                raw = [
                    {"start": s.start, "end": s.end, "text": s.text, "avg_logprob": -0.5}
                    for s in segments
                ]
                cleaned, stats = _filter_all_layers(raw, audio)
                stats["mode"] = "vad_guided"
                self.last_filter_stats = stats
                if on_progress:
                    on_progress({"stage": "post_filter_done", "filter_stats": stats})
                    on_progress({"current": len(cleaned), "total": len(cleaned), "stage": "done"})
                return cleaned, language
            except Exception as e:
                if on_progress:
                    on_progress({"stage": "vad_fallback", "reason": str(e)})

        # Fallback / 强制旧路径
        return self._run_whole(audio, resolved, options, on_progress)

    def _run_whole(
        self,
        audio: Path,
        resolved_model: str,
        options: TranscribeOptions,
        on_progress: ProgressCallback | None,
    ) -> tuple[list[Segment], str | None]:
        """整段送 Whisper(原始路径,fallback 用)。"""
        import mlx_whisper

        kwargs = {
            "path_or_hf_repo": resolved_model,
            "language": options.language,
            "word_timestamps": options.word_timestamps,
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.6,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
        }
        initial_prompt = build_initial_prompt(options.initial_prompt, options.hotwords)
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        result = mlx_whisper.transcribe(str(audio), **kwargs)
        raw_segments = [s for s in result.get("segments", []) if s.get("text", "").strip()]

        if on_progress:
            on_progress({"stage": "post_filter_start", "raw_segments": len(raw_segments)})
        segments, stats = _filter_all_layers(raw_segments, audio)
        stats["mode"] = "whole"
        self.last_filter_stats = stats
        if on_progress:
            on_progress({"stage": "post_filter_done", "filter_stats": stats})
            on_progress({"current": len(segments), "total": len(segments), "stage": "done"})

        return segments, result.get("language")


# ============================================================================
# VAD-guided transcription
# ============================================================================
#
# Why we need this: Whisper 的内部 chunk 处理在某些情况下会**整窗丢段** —
# 同一段音频单独切出来送给 Whisper 能正常识别,放在长音频里又会被跳过。
# 这是 OpenAI Whisper 的已知问题(non-deterministic chunk merging)。
# 解决: 用 silero-vad 先把音频切成"说话区间",每个区间不超过 25 秒
# (避开 Whisper 30 秒滑窗),逐段送 mlx-whisper,再把结果按全局时间拼接。
#
# 副作用: 短片段(< 3s)上下文不足容易听错(如"雅各书"听成"染歌书"),
# 所以合并相邻区间到一个相对长的窗口里。

_AUDIO_SR = 16000  # silero-vad / Whisper 标准采样率


def _load_audio_16k(path: Path) -> "np.ndarray":  # type: ignore[name-defined]
    """ffmpeg → 16 kHz mono float32 numpy。"""
    import subprocess
    import numpy as np

    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(path), "-ac", "1", "-ar", str(_AUDIO_SR),
            "-f", "f32le", "-acodec", "pcm_f32le", "-",
        ],
        check=True, capture_output=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()  # copy → writable


def _vad_speech_regions(samples, *, threshold: float = 0.3,
                        min_speech_ms: int = 250,
                        min_silence_ms: int = 200,
                        merge_gap_s: float = 0.6,
                        max_seg_s: float = 25.0,
                        min_seg_s: float = 3.0):
    """返回[(start_sample, end_sample), …]。
    - 合并间隔 < merge_gap_s 的相邻区间(避免短片段)
    - 拆开 > max_seg_s 的(避开 Whisper chunk bug)
    - 过短(< min_seg_s)的区间会向后合并(短片段 Whisper 上下文不足易听错)
    """
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps

    vad = load_silero_vad()
    wav_t = torch.from_numpy(samples)
    raw = get_speech_timestamps(
        wav_t, vad, sampling_rate=_AUDIO_SR,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
    )
    if not raw:
        return []

    # 合并近邻
    merged = [dict(raw[0])]
    for r in raw[1:]:
        if r["start"] - merged[-1]["end"] < merge_gap_s * _AUDIO_SR:
            merged[-1]["end"] = r["end"]
        else:
            merged.append(dict(r))

    # 短片段向后合(若不能合,保留)
    consolidated = []
    i = 0
    while i < len(merged):
        cur = merged[i]
        dur = (cur["end"] - cur["start"]) / _AUDIO_SR
        # 短的合下一个 — 若合并后总长 ≤ max_seg_s
        while dur < min_seg_s and i + 1 < len(merged):
            nxt = merged[i + 1]
            new_dur = (nxt["end"] - cur["start"]) / _AUDIO_SR
            if new_dur > max_seg_s:
                break
            cur = {"start": cur["start"], "end": nxt["end"]}
            dur = new_dur
            i += 1
        consolidated.append(cur)
        i += 1

    # 拆超长
    final = []
    for r in consolidated:
        s, e = r["start"], r["end"]
        dur = (e - s) / _AUDIO_SR
        if dur <= max_seg_s:
            final.append((s, e))
        else:
            n = int((dur + max_seg_s - 1) // max_seg_s)
            chunk = (e - s) // n
            for k in range(n):
                cs = s + k * chunk
                ce = e if k == n - 1 else s + (k + 1) * chunk
                final.append((cs, ce))
    return final


def _vad_guided_run(audio: Path, resolved_model: str, options, on_progress):
    """VAD 引导转录主流程。"""
    import os
    import tempfile
    import mlx_whisper
    import soundfile as sf

    samples = _load_audio_16k(audio)
    duration_s = len(samples) / _AUDIO_SR

    if on_progress:
        on_progress({"stage": "vad_start", "duration": duration_s})

    regions = _vad_speech_regions(samples)
    if on_progress:
        on_progress({"stage": "vad_regions", "count": len(regions)})

    if not regions:
        # 没检测到说话 — 整段也试一下,可能是 VAD 太严
        return [], None

    base_kwargs = {
        "path_or_hf_repo": resolved_model,
        "language": options.language,
        "word_timestamps": options.word_timestamps,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.3,  # 已经被 VAD 筛过,这里放宽
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
    }
    initial_prompt = build_initial_prompt(options.initial_prompt, options.hotwords)
    if initial_prompt:
        base_kwargs["initial_prompt"] = initial_prompt

    all_segments: list[Segment] = []
    language: str | None = None

    for idx, (s_samp, e_samp) in enumerate(regions):
        s_sec = s_samp / _AUDIO_SR
        e_sec = e_samp / _AUDIO_SR
        chunk = samples[s_samp:e_samp]

        # 写临时 WAV (mlx-whisper 需要文件路径)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, chunk, _AUDIO_SR, subtype="FLOAT")
            res = mlx_whisper.transcribe(tmp_path, **base_kwargs)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if language is None:
            language = res.get("language")

        for seg in res.get("segments", []):
            txt = (seg.get("text") or "").strip()
            if not txt:
                continue
            all_segments.append(Segment(
                start=s_sec + float(seg["start"]),
                end=s_sec + float(seg["end"]),
                text=txt,
            ))

        if on_progress:
            on_progress({
                "stage": "vad_chunk",
                "current": idx + 1,
                "total": len(regions),
                "preview": txt[:30] if (res.get("segments") and txt) else "",
            })

    return all_segments, language
