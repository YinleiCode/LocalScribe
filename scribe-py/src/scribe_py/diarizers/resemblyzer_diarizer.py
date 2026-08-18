"""Resemblyzer + KMeans 说话人分离。

接受已转录的 segments (含时间戳),返回每段的 speaker 标签。可选地,提供"声纹库"
(name → 256d embedding)以把 SPEAKER_A/B 这种泛标签改成真实人名。

核心流程(同 /tmp/diarize.py 验证版):
  1. preprocess_wav 把音频转 16k mono float32
  2. VoiceEncoder.embed_utterance(return_partials=True) — 每段子嵌入 + 时间轴
  3. KMeans(n=n_speakers) 聚类
  4. 把每个 transcribe segment 映射到时间轴,多数投票决定 speaker
  5. (可选)每个 KMeans 中心 vs 声纹库,cosine 相似度最高且 ≥ 0.65 则用该人名

使用:
    from pathlib import Path
    from scribe_py.diarizers import diarize, extract_voice_embedding

    # 1) 用户上传声纹样本时,先离线提取
    emb = extract_voice_embedding(Path("三修.m4a"))   # → list[float], 256
    # 存进 settings.json: {"name": "三修", "embedding": emb}

    # 2) 转录后跑 diarization
    result = diarize(
        audio=Path("会议.m4a"),
        segments=transcribe_segments,
        n_speakers=2,
        profiles=[{"name": "三修", "embedding": emb_a},
                  {"name": "位总", "embedding": emb_b}],
    )
    # result.segments 每段加 .speaker 字段
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# Resemblyzer 默认采样率
SR = 16_000

# 匹配阈值:cosine 相似度 ≥ 此值 → 用真实姓名;否则保留 SPEAKER_X
MATCH_THRESHOLD = 0.65

# 自动 K 检测的搜索范围 + 单人判定阈值
AUTO_K_MIN = 2
AUTO_K_MAX = 8
SINGLE_SPEAKER_SILHOUETTE = 0.10  # 所有 K 的 silhouette 都低于此 → 判为单人
OVER_SPLIT_MARGIN = 0.03  # 强制 K 分数比推荐 K 低超过此值 → 明显过度拆分风险

# 聚类样本上限 — 超过则下采样跑聚类,再把全量按最近中心分配
# 长音频(96 min ≈ 16k 嵌入)直接跑层次聚类的 16k×16k 距离矩阵 = 1 GB,太重
# 下采样到 4k:质量近似不变,速度+内存大幅改善
CLUSTER_SAMPLE_LIMIT = 4000
WINDOWED_CLUSTER_MIN_AUDIO_S = 20 * 60
WINDOWED_CLUSTER_WINDOW_S = 10 * 60


@dataclass
class DiarizedSegment:
    start: float
    end: float
    text: str
    speaker: str  # "三修" / "位总" / "SPEAKER_A" 等


@dataclass
class DiarizationResult:
    segments: list[DiarizedSegment]
    speakers: list[str]            # 出现的所有说话人(去重)
    cluster_count: int             # KMeans 聚出的簇数
    matched_profiles: dict[str, str]  # cluster_id → matched profile name (空 = 未匹配)
    stats: dict


@dataclass
class _EmbeddingContext:
    duration_s: float
    speech_total_s: float
    vad_segments: int
    embeds: np.ndarray
    times: np.ndarray


_EMBEDDING_CACHE: dict[tuple[str, int, int], _EmbeddingContext] = {}


# ============================================================================
# Audio loading
# ============================================================================


def _load_audio_16k_mono(path: Path) -> np.ndarray:
    """ffmpeg → 16 kHz mono float32 [-1, 1]。"""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(path),
            "-ac", "1", "-ar", str(SR),
            "-f", "f32le", "-acodec", "pcm_f32le", "-",
        ],
        check=True, capture_output=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _embedding_cache_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))


def _get_embedding_context(audio: Path, on_progress=None) -> _EmbeddingContext:
    """Load audio, run VAD, and extract speaker embeddings once per source file."""
    import torch
    from resemblyzer import VoiceEncoder
    from silero_vad import load_silero_vad, get_speech_timestamps

    key = _embedding_cache_key(audio)
    cached = _EMBEDDING_CACHE.get(key)
    if cached is not None:
        if on_progress:
            on_progress({
                "stage": "diarize_cached_embeddings",
                "embeddings": int(len(cached.embeds)),
                "vad_segments": int(cached.vad_segments),
            })
        return cached

    if on_progress:
        on_progress({"stage": "diarize_load_audio"})

    raw = _load_audio_16k_mono(audio)
    duration_s = len(raw) / SR

    if on_progress:
        on_progress({"stage": "diarize_vad", "duration": duration_s})

    vad_model = load_silero_vad()
    ts_list = get_speech_timestamps(
        torch.from_numpy(raw), vad_model, sampling_rate=SR,
        threshold=0.3, min_speech_duration_ms=300, min_silence_duration_ms=200,
    )
    if not ts_list:
        ctx = _EmbeddingContext(
            duration_s=duration_s,
            speech_total_s=0.0,
            vad_segments=0,
            embeds=np.zeros((0, 256), dtype=np.float32),
            times=np.zeros((0,), dtype=np.float32),
        )
        _EMBEDDING_CACHE.clear()
        _EMBEDDING_CACHE[key] = ctx
        return ctx

    speech_total_s = sum(t["end"] - t["start"] for t in ts_list) / SR

    chunks = []
    chunk_map: list[tuple[int, int, int]] = []
    cur_off = 0
    for t in ts_list:
        chunk = raw[t["start"]:t["end"]]
        chunks.append(chunk)
        chunk_map.append((cur_off, t["start"], len(chunk)))
        cur_off += len(chunk)
    trimmed = np.concatenate(chunks).astype(np.float32)

    if on_progress:
        on_progress({"stage": "diarize_extract_embeddings",
                     "speech_seconds": speech_total_s, "vad_segments": len(ts_list)})

    encoder = VoiceEncoder(verbose=False)
    _full_emb, partial_embeds, wav_splits = encoder.embed_utterance(
        trimmed, return_partials=True, rate=2
    )
    embeds = np.asarray(partial_embeds, dtype=np.float32)
    norms = np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9
    embeds = embeds / norms

    times = np.zeros(len(wav_splits), dtype=np.float32)
    for i, sp in enumerate(wav_splits):
        center = (sp.start + sp.stop) // 2
        for trim_off, orig_off, length in chunk_map:
            if trim_off <= center < trim_off + length:
                times[i] = (orig_off + (center - trim_off)) / SR
                break
        else:
            times[i] = center / SR

    ctx = _EmbeddingContext(
        duration_s=duration_s,
        speech_total_s=speech_total_s,
        vad_segments=len(ts_list),
        embeds=embeds,
        times=times,
    )
    _EMBEDDING_CACHE.clear()
    _EMBEDDING_CACHE[key] = ctx
    return ctx


# ============================================================================
# Public API
# ============================================================================


def extract_voice_embedding(audio: Path) -> list[float]:
    """从一段音频提取 256 维声纹向量。建议样本 ≥ 5 秒纯人声。

    返回 Python list(json 可序列化),长度恒为 256。
    """
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder(verbose=False)
    samples = _load_audio_16k_mono(audio)
    # preprocess_wav 期待 ndarray 或 path,这里直接给 ndarray(并指定 source_sr)
    wav = preprocess_wav(samples, source_sr=SR)
    emb = encoder.embed_utterance(wav)
    # L2 归一化,后续 cosine 直接点积
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb.astype(float).tolist()


def diarize(
    audio: Path,
    segments: Sequence[dict],
    n_speakers: int = 2,
    profiles: Iterable[dict] | None = None,
    on_progress=None,
) -> DiarizationResult:
    """对 segments 打 speaker 标签。

    Args:
        audio: 音频文件路径
        segments: list of dict 含 "start"/"end"/"text" (秒,字符串)
        n_speakers: KMeans 簇数(1-8)。1 = 全归一人(跳过聚类)
        profiles: optional [{"name": str, "embedding": list[float]}]; 簇中心 vs 库匹配
        on_progress: callable(dict) — emit {"stage": ..., ...}

    Returns:
        DiarizationResult.segments: 每段加了 speaker 字段
    """
    from sklearn.cluster import KMeans

    profiles = list(profiles or [])
    requested_n = int(n_speakers)
    auto = requested_n <= 0
    n_speakers = 0 if auto else max(1, min(8, requested_n))

    ctx = _get_embedding_context(audio, on_progress)
    duration_s = ctx.duration_s
    embeds = ctx.embeds
    times = ctx.times
    n_emb = len(embeds)

    if n_emb == 0:
        # 整段没说话 → 单一簇
        return DiarizationResult(
            segments=[
                DiarizedSegment(start=float(s["start"]), end=float(s["end"]),
                                text=str(s.get("text") or ""), speaker="SPEAKER_A")
                for s in segments
            ],
            speakers=["SPEAKER_A"],
            cluster_count=1,
            matched_profiles={},
            stats={"embeddings": 0, "duration_s": duration_s, "clusters": 1,
                   "matched_profile_count": 0, "segment_count": len(segments),
                   "auto": auto, "silhouette_sweep": {}, "vad_segments": 0},
        )

    if on_progress:
        on_progress({
            "stage": "diarize_cluster",
            "embeddings": n_emb,
            "n_speakers": "auto" if auto else n_speakers,
        })

    auto_silhouette: dict[int, float] = {}
    recommended_n_speakers: int | None = None
    recommended_score: float | None = None
    selected_score: float | None = None
    assignment_mode = "global_kmeans"

    # ---- 长音频优化:下采样跑聚类,再全量按最近中心分配 ----
    if n_emb > CLUSTER_SAMPLE_LIMIT:
        # 等距下采样保证覆盖整段时间(防止只采到前半段)
        sample_idx = np.linspace(0, n_emb - 1, CLUSTER_SAMPLE_LIMIT).astype(int)
        embeds_fit = embeds[sample_idx]
    else:
        embeds_fit = embeds

    def _fit_kmeans(emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """KMeans + 单位归一中心。实测在 Resemblyzer 嵌入上比 Agglomerative 稳定 —
        Agglom average 倾向于把所有点合并成一类(chaining effect)。"""
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        lbl = km.fit_predict(emb)
        cents = km.cluster_centers_
        cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
        return lbl, cents

    def _assign_all(emb: np.ndarray, cents: np.ndarray) -> np.ndarray:
        """把全量嵌入按 cosine 距离最近的中心分配。"""
        # cents 已 L2 归一,emb 已 L2 归一 → 余弦相似度 = 点积
        sims = emb @ cents.T  # (N, K)
        return np.argmax(sims, axis=1)

    def _centroids_from_labels(emb: np.ndarray, lbl: np.ndarray, k: int) -> np.ndarray:
        cents = []
        for cluster_id in range(k):
            part = emb[lbl == cluster_id]
            if len(part) == 0:
                cents.append(np.zeros((emb.shape[1],), dtype=np.float32))
                continue
            cent = part.mean(axis=0)
            cent = cent / (np.linalg.norm(cent) + 1e-9)
            cents.append(cent)
        return np.asarray(cents, dtype=np.float32)

    def _earliest_cluster_mapping(local_times: np.ndarray, local_labels: np.ndarray, k: int) -> dict[int, int]:
        first_seen = []
        for cluster_id in range(k):
            idxs = np.where(local_labels == cluster_id)[0]
            first = float(local_times[idxs[0]]) if len(idxs) else float("inf")
            first_seen.append((first, cluster_id))
        return {
            cluster_id: canonical
            for canonical, (_first, cluster_id) in enumerate(sorted(first_seen))
        }

    def _align_centroids(prev: np.ndarray, local: np.ndarray) -> dict[int, int]:
        sims = local @ prev.T
        try:
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(-sims)
            return {int(row): int(col) for row, col in zip(rows, cols)}
        except Exception:
            pairs = [
                (float(sims[local_id, canonical_id]), local_id, canonical_id)
                for local_id in range(sims.shape[0])
                for canonical_id in range(sims.shape[1])
            ]
            used_local: set[int] = set()
            used_canonical: set[int] = set()
            mapping: dict[int, int] = {}
            for _score, local_id, canonical_id in sorted(pairs, reverse=True):
                if local_id in used_local or canonical_id in used_canonical:
                    continue
                mapping[local_id] = canonical_id
                used_local.add(local_id)
                used_canonical.add(canonical_id)
                if len(mapping) == sims.shape[0]:
                    break
            return mapping

    def _fit_windowed_kmeans(k: int, global_centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray, int] | None:
        if duration_s < WINDOWED_CLUSTER_MIN_AUDIO_S or k < 2 or len(embeds) < max(120, k * 20):
            return None

        out = np.full(len(embeds), -1, dtype=int)
        rolling_centroids: np.ndarray | None = None
        windows_used = 0
        starts = np.arange(0.0, duration_s + 0.001, WINDOWED_CLUSTER_WINDOW_S)
        for start in starts:
            end = min(duration_s + 0.001, start + WINDOWED_CLUSTER_WINDOW_S)
            mask = (times >= start) & (times < end)
            idxs = np.where(mask)[0]
            if len(idxs) < max(k * 8, 24):
                if rolling_centroids is not None and len(idxs):
                    out[idxs] = _assign_all(embeds[idxs], rolling_centroids)
                continue

            local_labels, local_centroids = _fit_kmeans(embeds[idxs], k)
            if rolling_centroids is None:
                mapping = _earliest_cluster_mapping(times[idxs], local_labels, k)
                aligned_centroids = np.zeros_like(local_centroids)
                for local_id, canonical_id in mapping.items():
                    aligned_centroids[canonical_id] = local_centroids[local_id]
                rolling_centroids = aligned_centroids
            else:
                mapping = _align_centroids(rolling_centroids, local_centroids)
                updated = rolling_centroids.copy()
                for local_id, canonical_id in mapping.items():
                    cent = 0.70 * rolling_centroids[canonical_id] + 0.30 * local_centroids[local_id]
                    updated[canonical_id] = cent / (np.linalg.norm(cent) + 1e-9)
                rolling_centroids = updated

            for local_id, canonical_id in mapping.items():
                out[idxs[local_labels == local_id]] = canonical_id
            windows_used += 1

        missing = np.where(out < 0)[0]
        if len(missing):
            fill_centroids = rolling_centroids if rolling_centroids is not None else global_centroids
            out[missing] = _assign_all(embeds[missing], fill_centroids)
        if windows_used == 0:
            return None
        return out, _centroids_from_labels(embeds, out, k), windows_used

    if (not auto and n_speakers == 1) or n_emb <= AUTO_K_MIN:
        # 显式单人 / 样本太少 → 一个簇
        labels = np.zeros(n_emb, dtype=int)
        centroids = embeds.mean(axis=0, keepdims=True)
        centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
        n_speakers_eff = 1
    elif auto:
        # ---- 自动 K 检测:silhouette 扫描 ----
        from sklearn.metrics import silhouette_score
        k_max = min(AUTO_K_MAX, max(AUTO_K_MIN, len(embeds_fit) // 4))
        best = None  # (k, score, fit_labels, centroids)
        for k in range(AUTO_K_MIN, k_max + 1):
            lbl, cents = _fit_kmeans(embeds_fit, k)
            try:
                score = float(silhouette_score(embeds_fit, lbl, metric="cosine"))
            except Exception:
                score = -1.0
            auto_silhouette[k] = score
            if best is None or score > best[1]:
                best = (k, score, lbl, cents)
        assert best is not None
        best_k, best_score, _, centroids = best
        recommended_n_speakers = best_k
        recommended_score = best_score
        if best_score < SINGLE_SPEAKER_SILHOUETTE:
            labels = np.zeros(n_emb, dtype=int)
            centroids = embeds.mean(axis=0, keepdims=True)
            centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
            n_speakers_eff = 1
            selected_score = best_score
        else:
            # 全量嵌入按 cosine 最近中心分配 — 比直接用聚类输出更稳(避免下采样的标签错位)
            labels = _assign_all(embeds, centroids)
            n_speakers_eff = best_k
            selected_score = best_score
    else:
        from sklearn.metrics import silhouette_score
        k_max = min(AUTO_K_MAX, max(AUTO_K_MIN, len(embeds_fit) // 4))
        best_k = None
        best_score = -1.0
        for k in range(AUTO_K_MIN, k_max + 1):
            lbl, _ = _fit_kmeans(embeds_fit, k)
            try:
                score = float(silhouette_score(embeds_fit, lbl, metric="cosine"))
            except Exception:
                score = -1.0
            auto_silhouette[k] = score
            if score > best_score:
                best_k = k
                best_score = score
        recommended_n_speakers = int(best_k or n_speakers)
        recommended_score = best_score
        selected_score = auto_silhouette.get(n_speakers)
        _, centroids = _fit_kmeans(embeds_fit, n_speakers)
        windowed = _fit_windowed_kmeans(n_speakers, centroids)
        if windowed is not None:
            labels, centroids, windows_used = windowed
            assignment_mode = "windowed_kmeans"
        else:
            labels = _assign_all(embeds, centroids)
            windows_used = 0
        n_speakers_eff = n_speakers

    if on_progress and auto_silhouette:
        on_progress({"stage": "diarize_auto_k", "silhouette": auto_silhouette,
                     "selected": n_speakers_eff})

    # ---- 匹配声纹库 ----
    matched: dict[int, str] = {}
    if profiles:
        prof_embs = []
        prof_names = []
        for p in profiles:
            emb = np.asarray(p.get("embedding", []), dtype=np.float32)
            if emb.shape != (256,):
                continue
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            prof_embs.append(emb)
            prof_names.append(p.get("name") or "SPEAKER")
        if prof_embs:
            prof_mat = np.stack(prof_embs)  # (P, 256)
            # 簇中心 vs 库:每个簇取 cosine 最高的 profile,要求 ≥ 阈值
            sims = centroids @ prof_mat.T  # (n_clusters, P)
            for cluster_id in range(centroids.shape[0]):
                best_p = int(np.argmax(sims[cluster_id]))
                best_sim = float(sims[cluster_id, best_p])
                if best_sim >= MATCH_THRESHOLD:
                    matched[cluster_id] = prof_names[best_p]

    if on_progress:
        on_progress({"stage": "diarize_assign", "matched": matched})

    # ---- 给每个 segment 投票 ----
    out_segments: list[DiarizedSegment] = []
    speakers_seen: list[str] = []

    def speaker_label(cluster_id: int) -> str:
        if cluster_id in matched:
            return matched[cluster_id]
        # 默认 SPEAKER_A / SPEAKER_B / ...
        return f"SPEAKER_{chr(ord('A') + cluster_id)}"

    for seg in segments:
        s, e = float(seg["start"]), float(seg["end"])
        # 找时间轴上落在 [s, e] 内的所有嵌入
        mask = (times >= s) & (times <= e)
        if mask.sum() == 0:
            # 段太短 → 取最近的一个
            nearest = int(np.argmin(np.abs(times - (s + e) / 2)))
            cluster_id = int(labels[nearest])
        else:
            # 多数投票
            votes = labels[mask]
            counts = np.bincount(votes, minlength=n_speakers_eff)
            cluster_id = int(np.argmax(counts))
        spk = speaker_label(cluster_id)
        if spk not in speakers_seen:
            speakers_seen.append(spk)
        out_segments.append(DiarizedSegment(
            start=s, end=e, text=str(seg.get("text") or ""),
            speaker=spk,
        ))

    stats = {
        "embeddings": int(n_emb),
        "duration_s": float(duration_s),
        "clusters": int(n_speakers_eff),
        "matched_profile_count": len(matched),
        "segment_count": len(out_segments),
        "auto": auto,
        "silhouette_sweep": auto_silhouette,
        "requested_n_speakers": int(requested_n),
        "recommended_n_speakers": int(recommended_n_speakers or n_speakers_eff),
        "recommended_score": recommended_score,
        "selected_score": selected_score,
        "assignment_mode": assignment_mode,
        "windowed_cluster_windows": int(locals().get("windows_used", 0)),
    }
    if (not auto
        and recommended_n_speakers is not None
        and recommended_score is not None
        and selected_score is not None
        and n_speakers_eff > recommended_n_speakers
    ):
        score_gap = recommended_score - selected_score
        stats["over_split_risk"] = score_gap >= OVER_SPLIT_MARGIN
        stats["over_split_score_gap"] = float(score_gap)
        stats["risk_level"] = "high" if score_gap >= OVER_SPLIT_MARGIN else "medium"
        stats["risk_reason"] = (
            f"forced {n_speakers_eff} speakers, but auto recommends "
            f"{recommended_n_speakers} speakers with a better clustering score"
        )
    else:
        stats["over_split_risk"] = False
        stats["over_split_score_gap"] = 0.0
        stats["risk_level"] = "low"

    return DiarizationResult(
        segments=out_segments,
        speakers=speakers_seen,
        cluster_count=n_speakers_eff,
        matched_profiles={f"SPEAKER_{chr(ord('A')+k)}": v for k, v in matched.items()},
        stats=stats,
    )
