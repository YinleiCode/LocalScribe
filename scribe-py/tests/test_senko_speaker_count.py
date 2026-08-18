from __future__ import annotations

import contextlib
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from scribe_py.diarizers import senko_diarizer as mod


def test_bundled_coreml_cache_is_redirected_outside_app(monkeypatch, tmp_path):
    cache_root = tmp_path / "Library/Caches/LocalScribe/senko/coreml"

    class _ModelPaths:
        cache_base_dir = property(lambda _self: Path("inside-app"))

    fake_config = types.SimpleNamespace(ModelPaths=_ModelPaths)
    monkeypatch.setenv("LOCALSCRIBE_SENKO_CACHE_DIR", str(cache_root))

    configured = mod._configure_senko_runtime_cache(fake_config)

    assert configured == cache_root
    assert _ModelPaths().cache_base_dir == cache_root


def _install_fake_senko(monkeypatch, diarizer_factory, *, darwin: bool = True):
    senko_module = types.ModuleType("senko")
    senko_config = types.ModuleType("senko.config")
    senko_config.DARWIN = darwin
    senko_config.ModelPaths = type("ModelPaths", (), {})
    senko_diarizer = types.ModuleType("senko.diarizer")
    senko_module.Diarizer = diarizer_factory
    senko_module.config = senko_config
    senko_module.diarizer = senko_diarizer
    monkeypatch.setitem(sys.modules, "senko", senko_module)
    monkeypatch.setitem(sys.modules, "senko.config", senko_config)
    monkeypatch.setitem(sys.modules, "senko.diarizer", senko_diarizer)
    monkeypatch.delenv("LOCALSCRIBE_SENKO_CACHE_DIR", raising=False)
    monkeypatch.setattr(mod, "_SENKO_COREML_FALLBACK_REASON", None)
    return senko_config, senko_diarizer


def test_make_diarizer_keeps_coreml_when_initialization_succeeds(monkeypatch):
    calls = []

    class _Diarizer:
        spectral_cluster = object()

        def __init__(self, **kwargs):
            calls.append(kwargs)

    senko_config, _ = _install_fake_senko(monkeypatch, _Diarizer)

    diarizer = mod._make_diarizer()

    assert calls == [{"device": "auto", "warmup": False, "quiet": True}]
    assert senko_config.DARWIN is True
    assert diarizer._localscribe_runtime_backend == mod.SENKO_COREML_RUNTIME
    assert diarizer._localscribe_fallback_reason is None


def test_make_diarizer_falls_back_to_silero_cpu_after_coreml_failure(monkeypatch):
    calls = []

    class _Diarizer:
        spectral_cluster = object()

        def __init__(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("device") == "auto":
                raise RuntimeError("Failed to build the model execution plan")

    senko_config, senko_diarizer_module = _install_fake_senko(monkeypatch, _Diarizer)

    first = mod._make_diarizer()
    second = mod._make_diarizer()

    cpu_args = {
        "device": "cpu",
        "vad": "silero",
        "clustering": "cpu",
        "warmup": False,
        "quiet": True,
    }
    assert calls == [
        {"device": "auto", "warmup": False, "quiet": True},
        cpu_args,
        cpu_args,
    ]
    assert senko_config.DARWIN is False
    assert senko_diarizer_module.torch is sys.modules["torch"]
    assert first._localscribe_runtime_backend == mod.SENKO_CPU_RUNTIME
    assert second._localscribe_runtime_backend == mod.SENKO_CPU_RUNTIME
    assert "Failed to build the model execution plan" in first._localscribe_fallback_reason


def test_silero_fallback_reads_bundled_pcm_without_torchaudio(tmp_path):
    wav_path = tmp_path / "speech.wav"
    samples = np.asarray((-32768, -8192, 0, 8192, 32767), dtype="<i2")
    with wave.open(str(wav_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(mod.SR)
        writer.writeframes(samples.tobytes())

    loaded = mod._read_silero_pcm_wav(wav_path)

    assert loaded.shape == (5,)
    assert loaded.dtype == sys.modules["torch"].float32
    assert np.allclose(loaded.numpy(), samples.astype(np.float32) / 32768.0)


def _cluster_embeddings(count: int, *, per_cluster: int = 24, dims: int = 12, noise: float = 0.015) -> np.ndarray:
    rng = np.random.default_rng(20260724 + count)
    rows = []
    for cluster in range(count):
        center = np.zeros(dims, dtype=np.float64)
        center[cluster] = 1.0
        for _ in range(per_cluster):
            vector = center + rng.normal(0.0, noise, size=dims)
            vector /= np.linalg.norm(vector)
            rows.append(vector)
    return np.asarray(rows, dtype=np.float32)


def _context(embeddings: np.ndarray) -> mod._SenkoEmbeddingContext:
    subsegments = [(index * 0.6, index * 0.6 + 1.5) for index in range(len(embeddings))]
    return mod._SenkoEmbeddingContext(
        vad_segments=[(0.0, subsegments[-1][1])],
        subsegments=subsegments,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        timing_stats={},
        subsegment_pitch_hz=np.zeros(len(embeddings), dtype=np.float32),
        subsegment_pitch_confidence=np.zeros(len(embeddings), dtype=np.float32),
    )


def test_cached_analysis_wav_is_reused_and_removed_with_embedding_cache(tmp_path):
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"source")
    wav = tmp_path / "meeting-16k.wav"
    wav.write_bytes(b"pcm")
    ctx = _context(_cluster_embeddings(2, per_cluster=4))
    ctx.analysis_wav = wav

    mod._clear_senko_embedding_cache()
    try:
        mod._SENKO_EMBEDDING_CACHE[mod._embedding_cache_key(source)] = ctx

        assert mod.cached_analysis_wav(source) == wav
        assert wav.is_file()
    finally:
        mod._clear_senko_embedding_cache()

    assert not wav.exists()


def test_spectral_workspace_and_same_count_result_are_reused_across_candidates():
    ctx = _context(_cluster_embeddings(3, per_cluster=4, dims=4))
    calls = {"similarity": 0, "cluster_counts": [], "perform": 0}

    class _InnerSpectral:
        k = 2

        def get_sim_mat(self, values):
            calls["similarity"] += 1
            return values @ values.T

        def p_pruning(self, values, _pval):
            return values

        def get_laplacian(self, values):
            values = values.copy()
            np.fill_diagonal(values, 0.0)
            return np.diag(np.sum(np.abs(values), axis=1)) - values

        def cluster_embs(self, _basis, count):
            calls["cluster_counts"].append(count)
            return np.arange(len(ctx.embeddings), dtype=int) % count

    class _CommonSpectral:
        cluster = _InnerSpectral()
        cluster_line = 1
        min_cluster_size = 1
        mer_cos = None

        @staticmethod
        def filter_minor_cluster(labels, _values, _minimum):
            return labels

    class _Diarizer:
        spectral_cluster = _CommonSpectral()
        umap_hdbscan_cluster = spectral_cluster
        _timing_stats = {}

        def _perform_clustering(self, values, subsegments):
            calls["perform"] += 1
            labels = np.asarray(self.spectral_cluster(values), dtype=int)
            centroids = {}
            raw = []
            for position, label in enumerate(labels):
                speaker = f"SPEAKER_{int(label) + 1:02d}"
                raw.append({
                    "speaker": speaker,
                    "start": subsegments[position][0],
                    "end": subsegments[position][1],
                })
            for label in np.unique(labels):
                centroids[f"SPEAKER_{int(label) + 1:02d}"] = np.mean(
                    values[labels == label],
                    axis=0,
                )
            return raw, list(raw), centroids

    diarizer = _Diarizer()
    first, _, _, first_labels = mod._cluster_senko_embeddings(diarizer, ctx)
    diarizer.spectral_cluster.cluster.k = 3
    second, _, _, _ = mod._cluster_senko_embeddings(diarizer, ctx)
    diarizer.spectral_cluster.cluster.k = 2
    third, _, _, third_labels = mod._cluster_senko_embeddings(diarizer, ctx)

    assert calls == {
        "similarity": 1,
        "cluster_counts": [2, 3],
        "perform": 2,
    }
    assert first["cluster_result_cache_hit"] is False
    assert first["spectral_workspace_cache_hit"] is False
    assert second["cluster_result_cache_hit"] is False
    assert second["spectral_workspace_cache_hit"] is True
    assert third["cluster_result_cache_hit"] is True
    assert np.array_equal(first_labels, third_labels)


def test_cached_spectral_labels_match_senko_native_partition():
    cluster_cpu = pytest.importorskip("senko.cluster.cluster_cpu")
    values = _cluster_embeddings(4, per_cluster=12, dims=8)
    ctx = _context(values)
    common = cluster_cpu.CommonClustering(
        cluster_type="spectral",
        cluster_line=10,
        min_cluster_size=1,
        min_num_spks=1,
        max_num_spks=8,
        pval=0.02,
        min_pnum=6,
        oracle_num=2,
    )

    class _Diarizer:
        spectral_cluster = common

    for count in (2, 3, 4):
        common.cluster.k = count
        native = np.asarray(common(values.copy()), dtype=int)
        cached, _ = mod._cached_spectral_labels(
            _Diarizer(),
            ctx,
            values,
            oracle_count=count,
            workspace_key=(False, 0, len(values)),
        )

        assert cached is not None
        assert np.array_equal(
            native[:, None] == native[None, :],
            cached[:, None] == cached[None, :],
        )


def test_embedding_cache_cleanup_also_releases_exact_cue_embeddings():
    mod._SENKO_EXACT_CUE_EMBEDDING_CACHE[("audio",)] = np.ones((2, 3), dtype=np.float32)

    mod._clear_senko_embedding_cache()

    assert mod._SENKO_EXACT_CUE_EMBEDDING_CACHE == {}


def test_stale_diarization_wav_cleanup_is_scoped_to_dead_localscribe_processes(
    monkeypatch,
    tmp_path,
):
    stale = tmp_path / "localscribe-diarization-111-stale.wav"
    active = tmp_path / "localscribe-diarization-222-active.wav"
    unrelated = tmp_path / "tmp-other-program.wav"
    malformed = tmp_path / "localscribe-diarization-not-a-pid.wav"
    stale.write_bytes(b"stale-pcm")
    active.write_bytes(b"active-pcm")
    unrelated.write_bytes(b"other-pcm")
    malformed.write_bytes(b"malformed-pcm")
    monkeypatch.setattr(mod, "_process_is_running", lambda pid: pid == 222)

    result = mod._cleanup_stale_diarization_wavs(tmp_path)

    assert result == {"removed_files": 1, "removed_bytes": len(b"stale-pcm")}
    assert not stale.exists()
    assert active.read_bytes() == b"active-pcm"
    assert unrelated.read_bytes() == b"other-pcm"
    assert malformed.read_bytes() == b"malformed-pcm"


def test_ffmpeg_analysis_wav_uses_localscribe_process_prefix(monkeypatch, tmp_path):
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"audio")
    monkeypatch.setattr(mod, "_STALE_DIARIZATION_WAVS_CLEANED", True)

    def fake_run(cmd, **_kwargs):
        Path(cmd[-1]).write_bytes(b"pcm")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    wav = mod._ffmpeg_to_16k_wav(source)
    try:
        assert wav.name.startswith(
            f"{mod._DIARIZATION_TEMP_PREFIX}{mod.os.getpid()}-"
        )
        assert wav.suffix == ".wav"
        assert wav.read_bytes() == b"pcm"
    finally:
        wav.unlink(missing_ok=True)


def test_acoustic_count_estimates_clear_two_clusters():
    result = mod._estimate_acoustic_speaker_count(_cluster_embeddings(2))

    assert result["available"] is True
    assert result["recommended_n_speakers"] == 2
    assert result["confidence_level"] in {"medium", "high"}
    assert result["eigengap_score"] > 0


def test_acoustic_count_estimates_clear_three_clusters():
    result = mod._estimate_acoustic_speaker_count(_cluster_embeddings(3))

    assert result["available"] is True
    assert result["recommended_n_speakers"] == 3
    assert result["confidence_level"] in {"medium", "high"}
    assert set(result["eigengaps"]) >= {"2", "3"}
    assert set(result["relative_eigengaps"]) >= {"2", "3"}


def test_acoustic_count_marks_ambiguous_embeddings_low_confidence():
    embeddings = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (8, 1))

    result = mod._estimate_acoustic_speaker_count(embeddings)

    assert result["available"] is True
    assert 2 <= result["recommended_n_speakers"] <= 7
    assert result["confidence_level"] == "low"
    assert result["reason"] == "ambiguous_eigengap"


def test_acoustic_count_clamps_bounds_to_two_through_eight():
    result = mod._estimate_acoustic_speaker_count(
        _cluster_embeddings(2),
        min_speakers=1,
        max_speakers=99,
    )
    insufficient = mod._estimate_acoustic_speaker_count(np.ones((3, 4), dtype=np.float32))

    assert result["min_speakers"] == 2
    assert result["max_speakers"] == 8
    assert 2 <= result["recommended_n_speakers"] <= 8
    assert insufficient["available"] is False
    assert insufficient["recommended_n_speakers"] == 2


def test_exact_cue_window_rejects_native_fbank_crash_lengths():
    assert mod._is_safe_exact_cue_window(1.0, 1.08) is False
    assert mod._is_safe_exact_cue_window(1.0, 1.15) is False
    assert mod._is_safe_exact_cue_window(1.0, 1.288) is False
    assert mod._is_safe_exact_cue_window(1.0, 1.56) is False
    assert mod._is_safe_exact_cue_window(1.0, 2.49) is False
    assert mod._is_safe_exact_cue_window(1.0, 2.50) is True
    assert mod._is_safe_exact_cue_window(1.0, 3.00) is True


def test_exact_cue_window_is_normalized_to_stable_fixed_length():
    assert mod._normalized_exact_cue_window(1.0, 2.49) is None
    assert mod._normalized_exact_cue_window(1.0, 2.50) == (1.0, 2.5)
    assert mod._normalized_exact_cue_window(10.0, 14.0) == (11.25, 12.75)


def test_exact_cue_embeddings_extract_mixed_lengths_one_window_at_a_time(monkeypatch, tmp_path):
    wav = tmp_path / "converted.wav"
    wav.write_bytes(b"wav")
    monkeypatch.setattr(mod, "_ffmpeg_to_16k_wav", lambda _audio: wav)

    senko_module = types.ModuleType("senko")
    senko_utils = types.ModuleType("senko.utils")

    @contextlib.contextmanager
    def _quiet():
        yield

    senko_utils.suppress_stdout_stderr = _quiet
    senko_module.utils = senko_utils
    monkeypatch.setitem(sys.modules, "senko", senko_module)
    monkeypatch.setitem(sys.modules, "senko.utils", senko_utils)

    class _Diarizer:
        def __init__(self):
            self.calls = []

        def _extract_fbank_features(self, _audio, windows):
            self.calls.append(list(windows))
            return np.ones((80,), dtype=np.float32), [10], [0], 80

        def _generate_embeddings(self, _features, _frames, _offsets, _feature_dim):
            index = len(self.calls)
            return np.asarray([[float(index), 1.0, 0.0]], dtype=np.float32)

    diarizer = _Diarizer()
    windows = [(1.0, 2.50), (3.0, 4.75), (5.0, 7.25)]
    source = tmp_path / "source.mp3"
    source.write_bytes(b"audio")

    embeddings = mod._extract_exact_cue_embeddings(
        source,
        diarizer,
        windows,
    )

    assert diarizer.calls == [[windows[0]], [windows[1]], [windows[2]]]
    assert embeddings.shape == (3, 3)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_acoustic_count_excludes_overlap_contaminated_embeddings():
    clean = _cluster_embeddings(2, per_cluster=12)
    contaminated = _cluster_embeddings(3, per_cluster=8)[-8:]
    embeddings = np.concatenate([clean, contaminated], axis=0)
    ctx = _context(embeddings)
    ctx.overlap_available = True
    ctx.subsegment_overlap_ratios = np.asarray(
        [0.0] * len(clean) + [0.5] * len(contaminated),
        dtype=np.float32,
    )

    class _Cluster:
        k = 2

    class _Spectral:
        cluster = _Cluster()

    class _Diarizer:
        spectral_cluster = _Spectral()
        _timing_stats = {}

        def _perform_clustering(self, values, subsegments):
            split = len(values) // 2
            return (
                [
                    {"speaker": "SPEAKER_01", "start": 0.0, "end": subsegments[split - 1][1]},
                    {"speaker": "SPEAKER_02", "start": subsegments[split][0], "end": subsegments[-1][1]},
                ],
                [],
                {
                    "SPEAKER_01": np.mean(values[:split], axis=0),
                    "SPEAKER_02": np.mean(values[split:], axis=0),
                },
            )

    result, _, _, _ = mod._cluster_senko_embeddings(_Diarizer(), ctx)
    estimate = result["acoustic_speaker_count"]

    assert estimate["total_embeddings"] == 32
    assert estimate["clean_embeddings"] == 24
    assert estimate["overlap_filtered_embeddings"] == 8
    assert estimate["overlap_filter_applied"] is True
    assert estimate["recommended_n_speakers"] == 2


def test_fixed_count_is_guarded_when_acoustic_evidence_strongly_rejects_it(monkeypatch):
    embeddings = _cluster_embeddings(2, per_cluster=12)
    ctx = _context(embeddings)

    class _Cluster:
        k = None

    class _Spectral:
        cluster = _Cluster()

    class _Diarizer:
        spectral_cluster = _Spectral()
        umap_hdbscan_cluster = _Spectral()
        _timing_stats = {}
        seen_fixed_count = None

        def _perform_clustering(self, values, _subsegments):
            self.seen_fixed_count = self.spectral_cluster.cluster.k
            split = len(values) // 2
            return (
                [
                    {"speaker": "SPEAKER_01", "start": 0.0, "end": 10.0},
                    {"speaker": "SPEAKER_02", "start": 10.0, "end": 20.0},
                ],
                [],
                {
                    "SPEAKER_01": np.mean(values[:split], axis=0),
                    "SPEAKER_02": np.mean(values[split:], axis=0),
                },
            )

    diarizer = _Diarizer()
    monkeypatch.setattr(mod, "_extract_senko_embeddings", lambda *_args, **_kwargs: ctx)
    monkeypatch.setattr(mod, "_make_diarizer", lambda: diarizer)

    senko_module = types.ModuleType("senko")
    senko_utils = types.ModuleType("senko.utils")

    @contextlib.contextmanager
    def _quiet():
        yield

    senko_utils.suppress_stdout_stderr = _quiet
    senko_module.utils = senko_utils
    monkeypatch.setitem(sys.modules, "senko", senko_module)
    monkeypatch.setitem(sys.modules, "senko.utils", senko_utils)

    result = mod.diarize(
        Path("unused.wav"),
        [
            {"start": 0.0, "end": 10.0, "text": "first"},
            {"start": 10.0, "end": 20.0, "text": "second"},
        ],
        n_speakers=4,
    )

    assert diarizer.seen_fixed_count == 2
    assert [segment.speaker for segment in result.segments] == ["SPEAKER_A", "SPEAKER_B"]
    assert result.stats["model_recommended_n_speakers"] == 2
    assert result.stats["model_recommended_score"] > 0
    assert result.stats["model_selected_n_speakers"] == 2
    assert result.stats["model_selected_score"] == result.stats["model_recommended_score"]
    assert result.stats["requested_n_speakers"] == 4
    assert result.stats["speaker_count_guard"]["guard_applied"] is True
    assert result.stats["speaker_count_guard"]["decision"] == "model_override"
    assert result.stats["speaker_count_guard"]["reason"] == (
        "model_evidence_strongly_rejects_requested_count"
    )
    assert result.stats["model_recommended_confidence_level"] in {"medium", "high"}
    assert result.stats["model_recommended_diagnostics"]["method"] == "campp_cosine_spectral_eigengap"


def test_requested_four_is_guarded_for_5707_like_three_speaker_evidence():
    decision = mod._resolve_requested_speaker_count(
        4,
        {
            "available": True,
            "recommended_n_speakers": 3,
            "confidence": 0.5022,
            "eigengap_score": 0.754,
            "eigengaps": {"2": 0.110, "3": 0.754, "4": 0.303},
        },
    )

    assert decision["requested_n_speakers"] == 4
    assert decision["selected_n_speakers"] == 3
    assert decision["oracle_n_speakers"] == 3
    assert decision["guard_applied"] is True
    assert decision["decision"] == "model_override"
    assert decision["recommended_eigengap"] == 0.754
    assert decision["requested_eigengap"] == 0.303
    assert decision["requested_gap_ratio"] == 0.401857


def test_model_cannot_increase_an_explicit_speaker_count():
    decision = mod._resolve_requested_speaker_count(
        3,
        {
            "available": True,
            "recommended_n_speakers": 4,
            "confidence": 0.90,
            "eigengap_score": 0.90,
            "eigengaps": {"3": 0.10, "4": 0.90},
        },
    )

    assert decision["selected_n_speakers"] == 3
    assert decision["oracle_n_speakers"] == 3
    assert decision["guard_applied"] is False
    assert decision["decision"] == "requested_preserved"
    assert decision["reason"] == (
        "higher_model_count_does_not_override_explicit_request"
    )


def test_requested_count_is_preserved_when_model_evidence_is_close():
    decision = mod._resolve_requested_speaker_count(
        4,
        {
            "available": True,
            "recommended_n_speakers": 3,
            "confidence": 0.72,
            "eigengap_score": 0.754,
            "eigengaps": {"3": 0.754, "4": 0.620},
        },
    )

    assert decision["selected_n_speakers"] == 4
    assert decision["oracle_n_speakers"] == 4
    assert decision["guard_applied"] is False
    assert decision["decision"] == "requested_preserved"
    assert decision["reason"] == "candidate_evidence_close"


def test_requested_count_is_preserved_when_model_confidence_is_low():
    decision = mod._resolve_requested_speaker_count(
        4,
        {
            "available": True,
            "recommended_n_speakers": 3,
            "confidence": 0.39,
            "eigengap_score": 0.90,
            "eigengaps": {"3": 0.90, "4": 0.10},
        },
    )

    assert decision["selected_n_speakers"] == 4
    assert decision["oracle_n_speakers"] == 4
    assert decision["guard_applied"] is False
    assert decision["reason"] == "model_confidence_too_low"


def test_stable_centroids_exclude_overlap_and_cluster_ambiguous_windows():
    a = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    ambiguous = np.asarray([0.71, 0.70, 0.0], dtype=np.float32)
    embeddings = np.stack([a, a, a, ambiguous, b, b, b, b])
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    overlap_ratios = np.asarray([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    speaker_ids, stable, stats = mod._build_stable_speaker_centroids(
        embeddings,
        labels,
        ["SPEAKER_01", "SPEAKER_02"],
        {"SPEAKER_01": a, "SPEAKER_02": b},
        overlap_ratios,
    )

    assert speaker_ids == ["SPEAKER_01", "SPEAKER_02"]
    assert np.allclose(stable[0], a)
    assert np.allclose(stable[1], b)
    assert stats["speakers"]["SPEAKER_01"]["clean_windows"] == 3
    assert stats["speakers"]["SPEAKER_01"]["trusted_windows"] == 3


def test_sync_cue_embedding_uses_double_threshold_and_overlap_guard():
    root_two = np.sqrt(2.0)
    embeddings = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0 / root_two, 1.0 / root_two],
        [0.0, 1.0],
    ], dtype=np.float32)
    segment = {
        "start": 0.0,
        "end": 4.0,
        "sync_cues": [
            {"start": 0.0, "end": 1.0, "text": "甲"},
            {"start": 1.0, "end": 2.0, "text": "乙"},
            {"start": 2.0, "end": 3.0, "text": "边界"},
            {"start": 3.0, "end": 4.0, "text": "抢话"},
        ],
    }

    rows = mod._speaker_cue_embedding_evidence(
        segment,
        embeddings,
        np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        ["SPEAKER_01", "SPEAKER_02"],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        {"SPEAKER_01": "SPEAKER_A", "SPEAKER_02": "SPEAKER_B"},
        overlap_ratios=np.zeros((4,), dtype=np.float32),
        overlap_intervals=[{"start": 3.0, "end": 4.0}],
        overlap_available=True,
    )

    assert [row["speaker"] for row in rows] == [
        "SPEAKER_A",
        "SPEAKER_B",
        "SPEAKER_B",
        "SPEAKER_B",
    ]
    assert [row["decision"] for row in rows] == ["assign", "assign", "review", "review"]
    assert rows[2]["margin"] == 0.0
    assert rows[3]["overlap_ratio"] == 1.0


def test_overlap_secondary_candidate_combines_window_and_nearby_turn_evidence():
    primary_timeline = [
        {"start": 0.0, "end": 6.0, "speaker": "raw_a"},
        {"start": 8.0, "end": 12.0, "speaker": "raw_b"},
    ]
    rows = mod._speaker_overlap_candidates_for_window(
        0.0,
        12.0,
        primary_timeline=primary_timeline,
        subsegments=[(4.5, 6.0), (8.0, 9.5)],
        sub_labels=np.asarray([0, 1]),
        speaker_ids=["raw_a", "raw_b"],
        overlap_intervals=[{
            "start": 5.0,
            "end": 5.5,
            "confidence": 0.8,
            "max_confidence": 0.95,
        }],
        label_map={"raw_a": "SPEAKER_A", "raw_b": "SPEAKER_B"},
    )

    assert rows == [{
        "start": 5.0,
        "end": 5.5,
        "primary_speaker": "SPEAKER_A",
        "secondary_speaker": "SPEAKER_B",
        "confidence": 0.8,
        "window_ratio": 0.0,
        "context_score": 0.4346,
        "candidate_score": 0.1521,
        "source": "osd_campp_context_v1",
    }]


def test_overlap_secondary_candidate_keeps_primary_label_and_uses_local_window_support():
    rows = mod._speaker_overlap_candidates_for_window(
        0.0,
        10.0,
        primary_timeline=[
            {"start": 0.0, "end": 5.0, "speaker": "raw_a"},
            {"start": 5.0, "end": 7.0, "speaker": "raw_b"},
            {"start": 7.0, "end": 10.0, "speaker": "raw_c"},
        ],
        subsegments=[(3.9, 4.7), (4.0, 4.8), (4.1, 4.9)],
        sub_labels=np.asarray([2, 2, 0]),
        speaker_ids=["raw_a", "raw_b", "raw_c"],
        overlap_intervals=[{"start": 4.2, "end": 4.8, "confidence": 0.9}],
        label_map={
            "raw_a": "SPEAKER_A",
            "raw_b": "SPEAKER_B",
            "raw_c": "SPEAKER_C",
        },
    )

    assert len(rows) == 1
    assert rows[0]["primary_speaker"] == "SPEAKER_A"
    assert rows[0]["secondary_speaker"] == "SPEAKER_C"
    assert rows[0]["window_ratio"] > 0.6


def test_overlap_secondary_candidate_rejects_weak_or_too_short_osd_intervals():
    common = {
        "primary_timeline": [
            {"start": 0.0, "end": 5.0, "speaker": "raw_a"},
            {"start": 20.0, "end": 25.0, "speaker": "raw_b"},
        ],
        "subsegments": [(0.0, 1.5)],
        "sub_labels": np.asarray([0]),
        "speaker_ids": ["raw_a", "raw_b"],
        "label_map": {"raw_a": "SPEAKER_A", "raw_b": "SPEAKER_B"},
    }

    weak = mod._speaker_overlap_candidates_for_window(
        0.0,
        5.0,
        overlap_intervals=[{"start": 1.0, "end": 1.5, "confidence": 0.54}],
        **common,
    )
    short = mod._speaker_overlap_candidates_for_window(
        0.0,
        5.0,
        overlap_intervals=[{"start": 1.0, "end": 1.1, "confidence": 0.9}],
        **common,
    )

    assert weak == []
    assert short == []


def test_sync_cue_embedding_accepts_clean_high_margin_short_voice():
    embedding = np.asarray([[0.70, 0.40, np.sqrt(0.35)]], dtype=np.float32)
    rows = mod._speaker_cue_embedding_evidence(
        {"start": 0.0, "end": 1.0, "sync_cues": [{"start": 0.0, "end": 1.0, "text": "短句"}]},
        embedding,
        np.asarray([0.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        ["SPEAKER_01", "SPEAKER_02"],
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        {"SPEAKER_01": "SPEAKER_A", "SPEAKER_02": "SPEAKER_B"},
    )

    assert rows[0]["score"] == 0.7
    assert rows[0]["margin"] == 0.3
    assert rows[0]["decision"] == "assign"


def test_exact_sync_cue_embedding_overrides_boundary_drag_from_sliding_window():
    exact = np.asarray([0.676, 0.465, np.sqrt(1.0 - 0.676**2 - 0.465**2)], dtype=np.float32)
    rows = mod._speaker_cue_embedding_evidence(
        {"start": 0.0, "end": 2.0, "sync_cues": [{"start": 0.0, "end": 2.0, "text": "边界句"}]},
        np.asarray([[0.45, 0.89, 0.05]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        ["SPEAKER_01", "SPEAKER_02"],
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        {"SPEAKER_01": "SPEAKER_A", "SPEAKER_02": "SPEAKER_B"},
        exact_embeddings={0: exact},
    )

    assert rows[0]["speaker"] == "SPEAKER_A"
    assert rows[0]["decision"] == "assign"
    assert rows[0]["embedding_scope"] == "exact_sync_cue"
    assert rows[0]["score"] == 0.676


def test_exact_sync_cue_high_margin_assigns_without_sliding_window_coverage():
    exact = np.asarray([0.80, 0.30, np.sqrt(1.0 - 0.80**2 - 0.30**2)], dtype=np.float32)
    rows = mod._speaker_cue_embedding_evidence(
        {"start": 0.0, "end": 1.0, "sync_cues": [{"start": 0.0, "end": 1.0, "text": "独立短句"}]},
        np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        np.asarray([10.0], dtype=np.float32),
        np.asarray([11.0], dtype=np.float32),
        ["SPEAKER_01", "SPEAKER_02"],
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        {"SPEAKER_01": "SPEAKER_A", "SPEAKER_02": "SPEAKER_B"},
        exact_embeddings={0: exact},
    )

    assert rows[0]["voice_coverage_ratio"] == 0.0
    assert rows[0]["speaker"] == "SPEAKER_A"
    assert rows[0]["decision"] == "assign"
    assert rows[0]["embedding_scope"] == "exact_sync_cue"


def test_exact_cue_selection_covers_sliding_window_drag_after_change():
    cues = [
        (0, 35.760, 35.934),
        (1, 35.934, 36.627),
        (2, 36.627, 39.169),
        (3, 39.169, 41.942),
        (4, 41.942, 43.214),
        (5, 43.214, 45.920),
        (6, 45.920, 46.853),
        (7, 46.853, 47.841),
    ]

    selected = mod._exact_cue_positions_near_changes(cues, [45.874])

    assert selected == {4, 5, 6, 7}
