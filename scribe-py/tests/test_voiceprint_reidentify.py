from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scribe_py.diarizers import senko_diarizer as mod


def _unit(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return arr / (np.linalg.norm(arr) + 1e-9)


def _fake_context(pattern: list[str], *, window: float = 2.0, step: float = 2.0) -> mod._SenkoEmbeddingContext:
    vectors = {
        "a": _unit([1.0, 0.0, 0.0, 0.0]),
        "a2": _unit([0.98, 0.04, 0.0, 0.0]),
        "b": _unit([0.0, 1.0, 0.0, 0.0]),
    }
    embeddings = np.stack([vectors[item] for item in pattern]).astype(np.float32)
    subsegments = [(idx * step, idx * step + window) for idx in range(len(pattern))]
    return mod._SenkoEmbeddingContext(
        vad_segments=[],
        subsegments=subsegments,
        embeddings=embeddings,
        timing_stats={},
        subsegment_pitch_hz=np.empty((len(pattern),), dtype=np.float32),
        subsegment_pitch_confidence=np.empty((len(pattern),), dtype=np.float32),
    )


def test_voiceprint_temporary_reidentify_allows_short_confirmed_anchor(monkeypatch):
    monkeypatch.setattr(mod, "_extract_senko_embeddings", lambda audio, on_progress=None: _fake_context(["a", "a2", "a", "b"]))

    result = mod.reidentify_with_voice_anchors(
        Path("dummy.wav"),
        segments=[
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A", "text": "hello"},
            {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_B", "text": "target"},
            {"start": 6.0, "end": 8.0, "speaker": "SPEAKER_C", "text": "other"},
        ],
        anchors=[{"start": 0.0, "end": 2.0, "speaker": "张三"}],
        threshold=0.78,
        review_threshold=0.70,
        margin=0.05,
        require_enrollment_quality=False,
    )

    assert result["segments"][1]["speaker"] == "张三"
    assert result["segments"][1]["speaker_voiceprint_reidentified"] is True
    assert result["profiles"][0]["enrollment_ready"] is False
    assert "too_short_enrollment" in result["profiles"][0]["enrollment_reasons"]
    assert result["stats"]["changed_segments"] == 2
    assert result["stats"]["require_enrollment_quality"] is False


def test_voiceprint_anchor_preflight_only_returns_consistent_windows(monkeypatch):
    monkeypatch.setattr(mod, "_extract_senko_embeddings", lambda audio, on_progress=None: _fake_context(["a", "b", "a", "a"]))

    result = mod.preflight_voiceprint_anchor_candidates(
        Path("dummy.wav"),
        segments=[
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_A", "text": "mixed"},
            {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_B", "text": "clean"},
        ],
    )

    assert result["stats"]["checked_segments"] == 2
    assert result["stats"]["eligible_segments"] == 1
    assert result["stats"]["rejected_segments"] == 1
    assert result["candidates"] == [{
        "index": 1,
        "speaker": "SPEAKER_B",
        "start": 4.0,
        "end": 8.0,
        "duration": 4.0,
        "text": "clean",
        "covered_seconds": 4.0,
        "quality": {
            "vector_count": 2,
            "pair_count": 1,
            "median_similarity": 1.0,
            "p10_similarity": 1.0,
            "min_similarity": 1.0,
            "stable": True,
        },
        "reason": "已通过 CAM++ 段内一致性和重叠语音预检",
    }]


def test_voiceprint_enrollment_rejects_less_than_ten_seconds(monkeypatch):
    monkeypatch.setattr(mod, "_extract_senko_embeddings", lambda audio, on_progress=None: _fake_context(["a", "a2", "a", "a2"]))

    with pytest.raises(ValueError, match="10 秒以上"):
        mod.reidentify_with_voice_anchors(
            Path("dummy.wav"),
            segments=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A"}],
            anchors=[{"start": 0.0, "end": 6.0, "speaker": "张三"}],
            require_enrollment_quality=True,
        )


def test_voiceprint_enrollment_rejects_mixed_anchor(monkeypatch):
    monkeypatch.setattr(mod, "_extract_senko_embeddings", lambda audio, on_progress=None: _fake_context(["a", "b", "a", "b", "a", "b"]))

    with pytest.raises(ValueError, match="无法通过声纹注册质量闸"):
        mod.reidentify_with_voice_anchors(
            Path("dummy.wav"),
            segments=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A"}],
            anchors=[{"start": 0.0, "end": 12.0, "speaker": "张三"}],
            require_enrollment_quality=True,
        )


def test_voiceprint_enrollment_accepts_clean_ten_second_profile(monkeypatch):
    monkeypatch.setattr(mod, "_extract_senko_embeddings", lambda audio, on_progress=None: _fake_context(["a", "a2", "a", "a2", "a", "b"]))

    result = mod.reidentify_with_voice_anchors(
        Path("dummy.wav"),
        segments=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A"}],
        anchors=[{"start": 0.0, "end": 10.0, "speaker": "张三"}],
        require_enrollment_quality=True,
    )

    profile = result["profiles"][0]
    assert profile["name"] == "张三"
    assert profile["enrollment_ready"] is True
    assert profile["enrollment_source"] == "user_confirmed_anchors"
    assert profile["sample_seconds"] == 10.0
    assert profile["sample_seconds_basis"] == "unique_clean_wallclock"
    assert profile["quality"]["vector_count"] == 5
    assert len(profile["embeddings"]) == 5


def test_voiceprint_enrollment_does_not_inflate_overlapping_windows(monkeypatch):
    context = _fake_context(["a", "a2", "a", "a2", "a", "a2", "a", "a2"], window=1.5, step=0.6)
    monkeypatch.setattr(mod, "_extract_senko_embeddings", lambda audio, on_progress=None: context)

    with pytest.raises(ValueError, match="10 秒以上"):
        mod.reidentify_with_voice_anchors(
            Path("dummy.wav"),
            segments=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A"}],
            anchors=[{"start": 0.0, "end": 5.7, "speaker": "张三"}],
            require_enrollment_quality=True,
        )


def test_voiceprint_enrollment_does_not_double_count_duplicate_anchor(monkeypatch):
    monkeypatch.setattr(mod, "_extract_senko_embeddings", lambda audio, on_progress=None: _fake_context(["a", "a2", "a", "a2"]))
    anchor = {"start": 0.0, "end": 6.0, "speaker": "张三"}

    with pytest.raises(ValueError, match="10 秒以上"):
        mod.reidentify_with_voice_anchors(
            Path("dummy.wav"),
            segments=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A"}],
            anchors=[anchor, dict(anchor)],
            require_enrollment_quality=True,
        )


@pytest.mark.parametrize("generic_name", ["SPEAKER_A", "SPEAKER_01", "speaker_1"])
def test_voiceprint_enrollment_rejects_generic_profile_name(monkeypatch, generic_name):
    monkeypatch.setattr(mod, "_extract_senko_embeddings", lambda audio, on_progress=None: _fake_context(["a", "a2", "a", "a2", "a", "a2"]))

    with pytest.raises(ValueError, match="generic_profile_name"):
        mod.reidentify_with_voice_anchors(
            Path("dummy.wav"),
            segments=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A"}],
            anchors=[{"start": 0.0, "end": 12.0, "speaker": generic_name}],
            require_enrollment_quality=True,
        )


def test_voiceprint_temporary_reidentify_allows_generic_profile_name(monkeypatch):
    monkeypatch.setattr(mod, "_extract_senko_embeddings", lambda audio, on_progress=None: _fake_context(["a", "a2", "b"]))

    result = mod.reidentify_with_voice_anchors(
        Path("dummy.wav"),
        segments=[{"start": 4.0, "end": 6.0, "speaker": "SPEAKER_B"}],
        anchors=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A"}],
        require_enrollment_quality=False,
    )

    assert result["profiles"][0]["name"] == "SPEAKER_A"
    assert result["profiles"][0]["enrollment_ready"] is False


def test_voiceprint_same_speaker_overlapping_anchors_use_unique_vectors(monkeypatch):
    monkeypatch.setattr(mod, "_extract_senko_embeddings", lambda audio, on_progress=None: _fake_context(["a", "a2", "a", "a2", "a", "a2"]))

    result = mod.reidentify_with_voice_anchors(
        Path("dummy.wav"),
        segments=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A"}],
        anchors=[
            {"start": 0.0, "end": 8.0, "speaker": "张三"},
            {"start": 4.0, "end": 12.0, "speaker": "张三"},
        ],
        require_enrollment_quality=True,
    )

    profile = result["profiles"][0]
    assert profile["sample_seconds"] == 12.0
    assert profile["anchor_count"] == 2
    assert profile["quality"]["vector_count"] == 6
    assert len(profile["embeddings"]) == 6


def test_voiceprint_rejects_overlapping_anchor_with_conflicting_label(monkeypatch):
    monkeypatch.setattr(mod, "_extract_senko_embeddings", lambda audio, on_progress=None: _fake_context(["a", "a2", "a", "a2", "a", "a2"]))

    result = mod.reidentify_with_voice_anchors(
        Path("dummy.wav"),
        segments=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A"}],
        anchors=[
            {"start": 0.0, "end": 12.0, "speaker": "张三"},
            {"start": 4.0, "end": 12.0, "speaker": "李四"},
        ],
        require_enrollment_quality=True,
    )

    assert [profile["name"] for profile in result["profiles"]] == ["张三"]
    assert result["stats"]["rejected_anchor_count"] == 1
    assert result["stats"]["rejected_anchors"][0]["reason"] == "conflicting_anchor_label"


def test_voiceprint_rejects_adjacent_labels_sharing_sliding_voice_window(monkeypatch):
    context = _fake_context(["a", "a", "a", "a", "a", "a"], window=1.5, step=0.6)
    monkeypatch.setattr(mod, "_extract_senko_embeddings", lambda audio, on_progress=None: context)

    result = mod.reidentify_with_voice_anchors(
        Path("dummy.wav"),
        segments=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A"}],
        anchors=[
            {"start": 0.0, "end": 2.0, "speaker": "张三"},
            {"start": 2.0, "end": 4.0, "speaker": "李四"},
        ],
        require_enrollment_quality=False,
    )

    assert [profile["name"] for profile in result["profiles"]] == ["张三"]
    assert result["stats"]["rejected_anchor_count"] == 1
    rejected = result["stats"]["rejected_anchors"][0]
    assert rejected["reason"] == "conflicting_anchor_voice_window"
    assert rejected["conflicts_with"] == ["张三"]
    assert rejected["subsegment_indices"]


def test_voiceprint_high_confidence_match_clears_stale_review_flags(monkeypatch):
    monkeypatch.setattr(mod, "_extract_senko_embeddings", lambda audio, on_progress=None: _fake_context(["a", "a2"]))

    result = mod.reidentify_with_voice_anchors(
        Path("dummy.wav"),
        segments=[{
            "start": 2.0,
            "end": 4.0,
            "speaker": "张三",
            "speaker_assignment_review": True,
            "speaker_voiceprint_review": True,
            "speaker_review_reason": "旧的待确认原因",
        }],
        anchors=[{"start": 0.0, "end": 2.0, "speaker": "张三"}],
        require_enrollment_quality=False,
    )

    segment = result["segments"][0]
    assert segment["speaker"] == "张三"
    assert segment["speaker_assignment_review"] is False
    assert segment["speaker_voiceprint_review"] is False
    assert segment["speaker_voiceprint_reidentified"] is True
    assert segment["speaker_review_reason"] == "已按声纹锚点回扫确认"
    assert result["stats"]["matched_segments"] == 1
    assert result["stats"]["changed_segments"] == 0


def test_voiceprint_rejects_overlap_contaminated_anchor(monkeypatch):
    ctx = _fake_context(["a", "a2", "a", "a2", "a", "a2"])
    ctx.overlap_available = True
    ctx.overlap_intervals = [{"start": 1.0, "end": 2.0, "confidence": 0.9}]
    ctx.subsegment_overlap_ratios = np.asarray([0.5, 0, 0, 0, 0, 0], dtype=np.float32)
    monkeypatch.setattr(mod, "_extract_senko_embeddings", lambda audio, on_progress=None: ctx)

    with pytest.raises(ValueError, match="overlapping_speech"):
        mod.reidentify_with_voice_anchors(
            Path("dummy.wav"),
            segments=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A"}],
            anchors=[{"start": 0.0, "end": 10.0, "speaker": "张三"}],
            require_enrollment_quality=False,
        )


def test_clustering_excludes_overlap_windows_but_assigns_all_embeddings():
    class Cluster:
        k = 2

    class Spectral:
        cluster = Cluster()

    class FakeDiarizer:
        spectral_cluster = Spectral()
        _timing_stats = {}

        def __init__(self):
            self.clustered_count = 0

        def _perform_clustering(self, embeddings, subsegments):
            self.clustered_count = len(embeddings)
            return (
                [
                    {"speaker": "SPEAKER_01", "start": 0.0, "end": 4.0},
                    {"speaker": "SPEAKER_02", "start": 6.0, "end": 20.0},
                ],
                [],
                {
                    "SPEAKER_01": _unit([1.0, 0.0, 0.0, 0.0]),
                    "SPEAKER_02": _unit([0.0, 1.0, 0.0, 0.0]),
                },
            )

    ctx = _fake_context(["a", "a2", "b", "b", "a", "b", "a", "b", "a", "b"])
    ctx.vad_segments = [(0.0, 20.0)]
    ctx.overlap_available = True
    ctx.subsegment_overlap_ratios = np.asarray([0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    diarizer = FakeDiarizer()

    result, subsegments, embeddings, labels = mod._cluster_senko_embeddings(diarizer, ctx)

    assert diarizer.clustered_count == 9
    assert result["overlap_filtered_subsegments"] == 1
    assert len(subsegments) == len(embeddings) == len(labels) == 10
    assert labels[2] == 1


def test_global_profile_assignment_prevents_duplicate_identity():
    assignment = mod._global_profile_assignment({
        "SPEAKER_01": {"张三": 0.92, "李四": 0.72},
        "SPEAKER_02": {"张三": 0.88, "李四": 0.81},
    }, min_score=0.70)

    assert assignment == {"SPEAKER_01": "张三", "SPEAKER_02": "李四"}
    assert len(set(assignment.values())) == len(assignment)
