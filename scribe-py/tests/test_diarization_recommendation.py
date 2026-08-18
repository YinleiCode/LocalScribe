from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from scribe_py.ipc import (
    _choose_diarization_candidate,
    _annotate_segments_with_speaker_reviews,
    _candidate_has_meaningful_refinement,
    _build_count_ambiguity_review_segments,
    _build_low_confidence_speaker_review_segments,
    _build_local_assignment_review_segments,
    _build_rapid_alternation_review_segments,
    _build_review_segments,
    _build_subsegment_change_review_segments,
    _merge_fragile_speakers,
    _higher_count_has_weak_tail_over_split,
    handle_diarize,
    handle_recommend_diarization,
    _apply_historical_human_speaker_annotations,
    _postprocess_fixed_count_candidate,
    _materialize_projected_speaker_handoffs,
    _project_speaker_cues,
    _preserves_transcript_geometry,
    _preserves_transcript_partition,
    _recommendation_confidence,
    _reassign_isolated_fragile_segments,
    _repair_handoff_voice_guard_assignments,
    _resegment_mixed_speaker_segments,
    _refine_conflicting_voice_bands_with_pyin,
    _repair_discourse_continuity_assignments,
    _repair_voice_band_assignments,
    _recommend_diarization_candidates,
    _score_diarization_candidate,
    _segments_with_diarization_speakers,
    _estimate_pyin_pitch_for_segment,
    _finalize_speaker_metadata_only,
    _speaker_voice_line_groups,
    _speaker_voice_band_mix_summary,
    _should_auto_merge_in_diarize,
    _split_handoff_segments,
    _smooth_alternating_local_speaker_leakage,
    _smooth_windowed_sandwiched_runs,
)
from scribe_py.diarizers.senko_diarizer import _ordered_senko_speakers


def _candidate(n_speakers: int, speakers: list[dict]) -> dict:
    total_segments = sum(s["segments"] for s in speakers) or 1
    total_duration = sum(s["duration_s"] for s in speakers) or 1.0
    normalized = []
    for idx, speaker in enumerate(speakers):
        normalized.append({
            "speaker": speaker.get("speaker", f"SPEAKER_{idx}"),
            "segments": speaker["segments"],
            "segment_ratio": speaker["segments"] / total_segments,
            "duration_s": speaker["duration_s"],
            "duration_ratio": speaker["duration_s"] / total_duration,
            "turns": speaker["turns"],
            "stable_turns": speaker["stable_turns"],
            "short_ratio": speaker.get("short_ratio", 0.0),
            "filler_ratio": speaker.get("filler_ratio", 0.0),
            "sandwiched_ratio": speaker.get("sandwiched_ratio", 0.0),
        })
    candidate = {
        "n_speakers": n_speakers,
        "speakers": [s["speaker"] for s in normalized],
        "segments": [
            {"start": float(i * 2), "end": float(i * 2 + 1), "text": "x", "speaker": speaker["speaker"]}
            for speaker in normalized
            for i in range(speaker["segments"])
        ],
        "summary": {"speakers": normalized},
    }
    candidate.update(_score_diarization_candidate(candidate, n_speakers))
    return candidate


def _candidate_from_labels(n_speakers: int, labels: list[str], durations: list[float] | None = None) -> dict:
    segments = []
    cursor = 0.0
    durations = durations or [1.0] * len(labels)
    for idx, (speaker, duration) in enumerate(zip(labels, durations)):
        segments.append({
            "start": cursor,
            "end": cursor + duration,
            "text": "对" if duration < 1.0 else f"seg {idx}",
            "speaker": speaker,
        })
        cursor += duration + 0.2
    candidate = {
        "n_speakers": n_speakers,
        "speakers": sorted(set(labels)),
        "segments": segments,
        "summary": __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(segments),
    }
    candidate.update(_score_diarization_candidate(candidate, n_speakers))
    return candidate


def _patch_diarization_postprocess_identity(monkeypatch) -> None:
    identity_names = [
        "_refine_conflicting_voice_bands_with_pyin",
        "_repair_handoff_voice_guard_assignments",
        "_split_handoff_segments",
        "_resegment_mixed_speaker_segments",
        "_reassign_isolated_fragile_segments",
        "_smooth_windowed_sandwiched_runs",
        "_smooth_alternating_local_speaker_leakage",
        "_repair_discourse_continuity_assignments",
        "_repair_voice_band_assignments",
        "_postprocess_fixed_count_candidate",
        "_apply_historical_human_speaker_annotations",
        "_annotate_segments_with_speaker_reviews",
    ]
    for name in identity_names:
        monkeypatch.setattr(f"scribe_py.ipc.{name}", lambda candidate, *args, **kwargs: candidate)
    monkeypatch.setattr("scribe_py.ipc._build_review_segments", lambda candidate: [])


def test_recommend_diarization_candidates_passes_selected_engine(monkeypatch, tmp_path):
    _patch_diarization_postprocess_identity(monkeypatch)
    seen: list[tuple[int, str | None]] = []
    voice_refine_counts: list[int] = []

    def fake_voice_refine(candidate, _audio):
        voice_refine_counts.append(int(candidate.get("n_speakers") or 0))
        return candidate

    monkeypatch.setattr("scribe_py.ipc._refine_conflicting_voice_bands_with_pyin", fake_voice_refine)

    def fake_diarize(*, audio, segments, n_speakers, profiles, engine=None, on_progress=None):
        seen.append((n_speakers, engine))
        labels = [f"SPEAKER_{chr(65 + i)}" for i in range(n_speakers)]
        return SimpleNamespace(
            segments=[
                {**seg, "speaker": labels[idx % len(labels)]}
                for idx, seg in enumerate(segments)
            ],
            speakers=labels,
            cluster_count=n_speakers,
            matched_profiles={},
            stats={"engine": engine or "auto", "embeddings": 99},
        )

    monkeypatch.setattr("scribe_py.diarizers.diarize", fake_diarize)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    segments = [
        {"start": 0.0, "end": 1.0, "text": "第一段"},
        {"start": 1.2, "end": 2.0, "text": "第二段"},
        {"start": 2.2, "end": 3.0, "text": "第三段"},
    ]

    out = _recommend_diarization_candidates(
        audio=audio,
        segments=segments,
        profiles=[],
        min_speakers=2,
        max_speakers=4,
        engine="resemblyzer",
    )

    assert seen == [(2, "resemblyzer"), (3, "resemblyzer"), (4, "resemblyzer")]
    assert len(voice_refine_counts) == 1
    assert out["candidates"]
    assert all(c["stats"]["engine"] == "resemblyzer" for c in out["candidates"])


def test_handle_diarize_passes_selected_engine_to_fixed_count(monkeypatch, tmp_path):
    _patch_diarization_postprocess_identity(monkeypatch)
    seen: list[str | None] = []

    def fake_diarize(*, audio, segments, n_speakers, profiles, engine=None, on_progress=None):
        seen.append(engine)
        return SimpleNamespace(
            segments=[
                {**seg, "speaker": "SPEAKER_A"}
                for seg in segments
            ],
            speakers=["SPEAKER_A"],
            cluster_count=1,
            matched_profiles={},
            stats={"engine": engine or "auto"},
        )

    monkeypatch.setattr("scribe_py.diarizers.diarize", fake_diarize)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")

    out = handle_diarize({
        "audio": str(audio),
        "segments": [{"start": 0.0, "end": 1.0, "text": "第一段"}],
        "n_speakers": 1,
        "engine": "pyannote",
        "profiles": [],
    })

    assert seen == ["pyannote"]
    assert out["stats"]["engine"] == "pyannote"
    assert out["segments"][0]["speaker"] == "SPEAKER_A"


def test_fixed_diarize_postprocess_uses_guarded_actual_speaker_count(monkeypatch, tmp_path):
    _patch_diarization_postprocess_identity(monkeypatch)
    seen_requested: list[int] = []

    def fake_diarize(*, segments, **_kwargs):
        return SimpleNamespace(
            segments=[
                {**segment, "speaker": f"SPEAKER_{chr(65 + index % 3)}"}
                for index, segment in enumerate(segments)
            ],
            speakers=["SPEAKER_A", "SPEAKER_B", "SPEAKER_C"],
            cluster_count=3,
            matched_profiles={},
            stats={
                "engine": "senko",
                "requested_n_speakers": 4,
                "model_selected_n_speakers": 3,
                "speaker_count_guard": {"overrode_requested": True},
            },
        )

    def capture_postprocess(candidate, _anchors, *, requested_n, preserve_segmentation):
        seen_requested.append(requested_n)
        candidate["actual_n_speakers"] = len(candidate["speakers"])
        candidate["segmentation_preserved"] = preserve_segmentation
        return candidate

    monkeypatch.setattr("scribe_py.diarizers.diarize", fake_diarize)
    monkeypatch.setattr(
        "scribe_py.ipc._postprocess_fixed_count_candidate",
        capture_postprocess,
    )
    monkeypatch.setattr("scribe_py.ipc._build_review_segments", lambda _candidate: [])
    monkeypatch.setattr(
        "scribe_py.ipc._annotate_segments_with_speaker_reviews",
        lambda candidate: candidate,
    )
    monkeypatch.setattr("scribe_py.ipc._project_speaker_cues", lambda candidate: candidate)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    segments = [
        {"start": float(index), "end": float(index + 1), "text": f"第{index + 1}段"}
        for index in range(3)
    ]

    out = handle_diarize({
        "audio": str(audio),
        "segments": segments,
        "n_speakers": 4,
        "engine": "senko",
        "profiles": [],
    })

    assert seen_requested == [3]
    assert out["stats"]["model_selected_n_speakers"] == 3
    assert out["stats"]["requested_n_speakers"] == 4
    assert out["speakers"] == ["SPEAKER_A", "SPEAKER_B", "SPEAKER_C"]


def test_senko_speaker_labels_follow_first_appearance_order():
    ordered = _ordered_senko_speakers(
        [
            {"speaker": "SPEAKER_03", "start": 12.0, "end": 20.0},
            {"speaker": "SPEAKER_01", "start": 0.0, "end": 8.0},
            {"speaker": "SPEAKER_02", "start": 8.0, "end": 12.0},
        ],
        {"SPEAKER_01": [], "SPEAKER_02": [], "SPEAKER_03": [], "SPEAKER_04": []},
    )

    assert ordered == ["SPEAKER_01", "SPEAKER_02", "SPEAKER_03", "SPEAKER_04"]


def test_diarization_preserves_asr_text_timestamps_and_metadata():
    source = [
        {
            "start": 1.25,
            "end": 3.5,
            "text": "已经校准过的简体正文。",
            "original_text": "原始识别正文",
            "asr_review": ["keep"],
        }
    ]
    diarized = [
        type("Diarized", (), {
            "start": 99.0,
            "end": 100.0,
            "text": "分人引擎不允许覆盖这段文字",
            "speaker": "SPEAKER_B",
        })()
    ]

    out = _segments_with_diarization_speakers(source, diarized)

    assert out == [{
        "start": 1.25,
        "end": 3.5,
        "text": "已经校准过的简体正文。",
        "original_text": "原始识别正文",
        "asr_review": ["keep"],
        "speaker": "SPEAKER_B",
    }]
    assert source[0].get("speaker") is None


def test_diarization_rerun_drops_stale_speaker_postprocess_metadata():
    source = [
        {
            "start": 1.25,
            "end": 3.5,
            "text": "重新跑分人时只保留转录文本。",
            "speaker": "SPEAKER_A",
            "speaker_confidence": 0.99,
            "speaker_votes": {"SPEAKER_A": 1.0},
            "speaker_overlap_candidates": [{
                "start": 1.5,
                "end": 2.0,
                "primary_speaker": "SPEAKER_A",
                "secondary_speaker": "SPEAKER_C",
            }],
            "voice_pitch_hz": 230.0,
            "voice_pitch_confidence": 0.9,
            "voice_band": "high",
            "speaker_handoff_split": True,
            "speaker_handoff_bridge": True,
            "voice_band_repaired": True,
            "asr_review": ["keep"],
        }
    ]
    diarized = [
        {
            "speaker": "SPEAKER_B",
            "speaker_confidence": 0.7,
            "speaker_votes": {"SPEAKER_B": 2.0},
            "voice_pitch_hz": 125.0,
            "voice_pitch_confidence": 0.8,
            "voice_band": "low",
        }
    ]

    out = _segments_with_diarization_speakers(source, diarized)

    assert out[0]["speaker"] == "SPEAKER_B"
    assert out[0]["speaker_votes"] == {"SPEAKER_B": 2.0}
    assert out[0]["voice_band"] == "low"
    assert out[0]["asr_review"] == ["keep"]
    assert "speaker_handoff_split" not in out[0]
    assert "speaker_handoff_bridge" not in out[0]
    assert "voice_band_repaired" not in out[0]
    assert "speaker_overlap_candidates" not in out[0]


def test_diarization_rerun_drops_all_identity_and_calibration_metadata():
    source = [{
        "start": 0.0,
        "end": 2.0,
        "text": "正文不能变化。",
        "speaker": "张三",
        "original_speaker": "SPEAKER_A",
        "speaker_voiceprint_reidentified": True,
        "speaker_voiceprint_review": True,
        "speaker_voiceprint_score": 0.91,
        "speaker_voiceprint_anchor": "张三",
        "speaker_calibrated": True,
        "speaker_calibration_source": "history.json",
        "voice_line_refined": True,
        "voice_line_review": True,
        "asr_review": ["keep"],
    }]

    out = _segments_with_diarization_speakers(source, [{"speaker": "SPEAKER_B"}])

    assert out == [{
        "start": 0.0,
        "end": 2.0,
        "text": "正文不能变化。",
        "speaker": "SPEAKER_B",
        "asr_review": ["keep"],
    }]


def test_diarization_projects_pyannote_overlap_ratio_to_canonical_field():
    out = _segments_with_diarization_speakers(
        [{"start": 0.0, "end": 1.0, "text": "重叠说话"}],
        [{
            "speaker": "SPEAKER_A",
            "speaker_overlap_risk": True,
            "speaker_overlap_ratio": 0.375,
        }],
    )

    assert out[0]["speaker_overlap_risk"] is True
    assert out[0]["overlap_ratio"] == 0.375
    assert "speaker_overlap_ratio" not in out[0]


def test_diarization_projects_overlap_second_speaker_metadata_without_touching_transcript():
    sync_cues = [{"start": 0.0, "end": 2.0, "text": "原始文字。"}]
    source = [{
        "start": 0.0,
        "end": 2.0,
        "text": "原始文字。",
        "sync_cues": sync_cues,
    }]
    out = _segments_with_diarization_speakers(source, [{
        "speaker": "SPEAKER_A",
        "speaker_overlap_candidates": [{
            "start": 0.5,
            "end": 1.25,
            "primary_speaker": "SPEAKER_A",
            "secondary_speaker": "SPEAKER_B",
            "confidence": 0.81234,
            "window_ratio": 0.45678,
            "context_score": 0.76543,
            "candidate_score": 0.72468,
            "source": "osd_campp_context_v1",
        }],
    }])

    assert out[0]["text"] == source[0]["text"]
    assert out[0]["start"] == source[0]["start"]
    assert out[0]["end"] == source[0]["end"]
    assert out[0]["sync_cues"] == sync_cues
    assert out[0]["speaker"] == "SPEAKER_A"
    assert out[0]["speaker_overlap_candidates"] == [{
        "start": 0.5,
        "end": 1.25,
        "primary_speaker": "SPEAKER_A",
        "secondary_speaker": "SPEAKER_B",
        "confidence": 0.8123,
        "window_ratio": 0.4568,
        "context_score": 0.7654,
        "candidate_score": 0.7247,
        "source": "osd_campp_context_v1",
    }]


def test_diarization_preserves_campp_cue_embedding_evidence_without_touching_asr_cues():
    sync_cues = [{"start": 0.0, "end": 1.0, "text": "原始文字。"}]
    source = [{"start": 0.0, "end": 1.0, "text": "原始文字。", "sync_cues": sync_cues}]
    diarized = [SimpleNamespace(
        speaker="SPEAKER_A",
        speaker_cue_embeddings=[{
            "cue_index": 0,
            "start": 0.0,
            "end": 1.0,
            "speaker": "SPEAKER_B",
            "score": 0.81234,
            "second_score": 0.70123,
            "second_speaker": "SPEAKER_A",
            "margin": 0.11111,
            "voice_coverage_seconds": 0.9,
            "voice_coverage_ratio": 0.9,
            "overlap_ratio": 0.0,
            "decision": "assign",
            "source": "campp_sync_cue_embedding",
        }],
    )]

    out = _segments_with_diarization_speakers(source, diarized)

    assert out[0]["text"] == source[0]["text"]
    assert out[0]["sync_cues"] == sync_cues
    assert out[0]["speaker_cue_embeddings"][0]["speaker"] == "SPEAKER_B"
    assert out[0]["speaker_cue_embeddings"][0]["score"] == 0.8123


def test_transcript_geometry_guard_rejects_sync_cue_changes():
    source = [{
        "start": 0.0,
        "end": 2.0,
        "text": "原始文字。",
        "sync_cues": [{"start": 0.0, "end": 2.0, "text": "原始文字。"}],
    }]
    changed = json.loads(json.dumps(source, ensure_ascii=False))
    changed[0]["sync_cues"][0]["end"] = 1.8

    assert _preserves_transcript_geometry(source, source)
    assert not _preserves_transcript_geometry(source, changed)


def test_fixed_count_postprocess_preserves_transcript_geometry_by_default(monkeypatch):
    candidate = _candidate_from_labels(2, ["SPEAKER_A", "SPEAKER_B"], [2.0, 2.0])
    original = [
        (segment["start"], segment["end"], segment["text"])
        for segment in candidate["segments"]
    ]

    def forbidden_split(_candidate):
        raise AssertionError("segmentation-changing postprocess must be disabled")

    monkeypatch.setattr("scribe_py.ipc._resegment_mixed_speaker_segments", forbidden_split)
    monkeypatch.setattr("scribe_py.ipc._split_handoff_segments", forbidden_split)
    for name in [
        "_repair_handoff_voice_guard_assignments",
        "_reassign_isolated_fragile_segments",
        "_smooth_short_sandwiched_segments",
        "_smooth_windowed_sandwiched_runs",
        "_smooth_alternating_local_speaker_leakage",
        "_repair_discourse_continuity_assignments",
        "_repair_voice_band_assignments",
    ]:
        monkeypatch.setattr(f"scribe_py.ipc.{name}", lambda value, *args: value)

    corrected = _postprocess_fixed_count_candidate(
        candidate,
        [candidate],
        requested_n=2,
        preserve_segmentation=True,
    )

    assert [
        (segment["start"], segment["end"], segment["text"])
        for segment in corrected["segments"]
    ] == original
    assert corrected["segmentation_preserved"] is True


def test_auto_diarize_reports_explicit_failure_when_all_candidates_fail(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    source = [{"start": 0.0, "end": 1.0, "text": "保留原文", "speaker": "SPEAKER_A"}]
    monkeypatch.setattr(
        "scribe_py.ipc._recommend_diarization_candidates",
        lambda **_kwargs: {
            "recommended_n_speakers": 0,
            "candidates": [],
            "reason": "全部候选失败",
            "errors": [{"n_speakers": 2, "error": "RuntimeError: failed"}],
        },
    )

    out = handle_diarize({
        "audio": str(audio),
        "segments": source,
        "n_speakers": 0,
        "profiles": [],
    })

    assert out["segments"] == source
    assert out["speakers"] == []
    assert out["stats"]["status"] == "error"
    assert out["stats"]["applied"] is False
    assert out["stats"]["errors"] == [{"n_speakers": 2, "error": "RuntimeError: failed"}]


def test_fixed_diarize_preserves_traditional_transcript_text(monkeypatch, tmp_path):
    _patch_diarization_postprocess_identity(monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    source = [{"start": 0.0, "end": 1.0, "text": "這個說話人不能改字。"}]
    monkeypatch.setattr(
        "scribe_py.diarizers.diarize",
        lambda **_kwargs: SimpleNamespace(
            segments=[{**source[0], "speaker": "SPEAKER_A"}],
            speakers=["SPEAKER_A"],
            cluster_count=1,
            matched_profiles={},
            stats={"engine": "senko"},
        ),
    )

    out = handle_diarize({
        "audio": str(audio),
        "segments": source,
        "n_speakers": 1,
        "profiles": [],
    })

    assert out["segments"][0]["text"] == "這個說話人不能改字。"


def test_recommend_diarization_preserves_traditional_transcript_text(monkeypatch, tmp_path):
    _patch_diarization_postprocess_identity(monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    source = [
        {"start": 0.0, "end": 1.0, "text": "這是第一段。"},
        {"start": 1.0, "end": 2.0, "text": "這是第二段。"},
    ]

    def fake_diarize(*, segments, n_speakers, **_kwargs):
        return SimpleNamespace(
            segments=[
                {**segment, "speaker": f"SPEAKER_{chr(65 + index % n_speakers)}"}
                for index, segment in enumerate(segments)
            ],
            speakers=[f"SPEAKER_{chr(65 + index)}" for index in range(n_speakers)],
            cluster_count=n_speakers,
            matched_profiles={},
            stats={"engine": "senko", "embeddings": 99},
        )

    monkeypatch.setattr("scribe_py.diarizers.diarize", fake_diarize)
    out = handle_recommend_diarization({
        "audio": str(audio),
        "segments": source,
        "min_speakers": 2,
        "max_speakers": 2,
        "profiles": [],
    })

    assert [segment["text"] for segment in out["candidates"][0]["segments"]] == [
        "這是第一段。",
        "這是第二段。",
    ]


def test_recommendation_applies_exact_repair_to_selected_candidate(monkeypatch, tmp_path):
    _patch_diarization_postprocess_identity(monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    source = [
        {"start": 0.0, "end": 1.0, "text": "第一段"},
        {"start": 1.0, "end": 2.0, "text": "第二段"},
    ]
    monkeypatch.setattr(
        "scribe_py.diarizers.diarize",
        lambda **_kwargs: SimpleNamespace(
            segments=[
                {**source[0], "speaker": "SPEAKER_A"},
                {**source[1], "speaker": "SPEAKER_B"},
            ],
            speakers=["SPEAKER_A", "SPEAKER_B"],
            cluster_count=2,
            matched_profiles={},
            stats={"engine": "senko", "embeddings": 99},
        ),
    )
    calls: list[int] = []

    def fake_exact_repair(candidate, repair_audio):
        assert repair_audio == audio.resolve()
        calls.append(int(candidate["n_speakers"]))
        output = json.loads(json.dumps(candidate))
        output.setdefault("stats", {})["exact_embedding_fallback"] = {
            "applied": True,
            "frozen_transcript_geometry": True,
        }
        return output

    monkeypatch.setattr("scribe_py.ipc._repair_long_missing_speaker_cues", fake_exact_repair)

    out = handle_recommend_diarization({
        "audio": str(audio),
        "segments": source,
        "min_speakers": 2,
        "max_speakers": 2,
        "profiles": [],
    })

    assert calls == [2]
    selected = out["candidates"][0]
    assert selected["stats"]["exact_embedding_fallback"]["applied"] is True
    assert _preserves_transcript_geometry(source, selected["segments"])


def test_recommendation_rejects_partial_speaker_assignment(monkeypatch, tmp_path):
    _patch_diarization_postprocess_identity(monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    source = [
        {"start": 0.0, "end": 1.0, "text": "第一段"},
        {"start": 1.0, "end": 2.0, "text": "第二段", "speaker": "OLD"},
    ]
    monkeypatch.setattr(
        "scribe_py.diarizers.diarize",
        lambda **_kwargs: SimpleNamespace(
            segments=[{"speaker": "SPEAKER_A"}],
            speakers=["SPEAKER_A"],
            cluster_count=1,
            matched_profiles={},
            stats={"engine": "senko", "embeddings": 99},
        ),
    )

    out = _recommend_diarization_candidates(
        audio=audio,
        segments=source,
        profiles=[],
        min_speakers=2,
        max_speakers=2,
    )

    assert out["candidates"] == []
    assert "speaker" in out["errors"][0]["error"]


def test_rapid_alternation_review_flags_weak_mixed_two_speaker_pocket():
    labels = [
        "SPEAKER_B",
        "SPEAKER_B",
        "SPEAKER_D",
        "SPEAKER_B",
        "SPEAKER_D",
        "SPEAKER_D",
        "SPEAKER_B",
        "SPEAKER_B",
        "SPEAKER_D",
        "SPEAKER_B",
    ]
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))
    for idx, seg in enumerate(candidate["segments"]):
        if seg["speaker"] == "SPEAKER_B":
            seg["voice_pitch_hz"] = 235.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "high"
        else:
            # Give the small cluster a mixed local voice line, which is exactly
            # the kind of evidence conflict that should be reviewed as a pocket
            # instead of silently trusting each short label.
            seg["voice_pitch_hz"] = 148.0 if idx in {2, 4} else 218.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low" if idx in {2, 4} else "high"
    candidate["fragile_speakers"] = ["SPEAKER_D"]
    candidate["voice_mix_summary"] = {
        "SPEAKER_D": {"mixed": True},
        "SPEAKER_B": {"mixed": False},
    }

    reviews = _build_rapid_alternation_review_segments(candidate)

    assert len(reviews) >= 6
    assert reviews[0]["index"] == 0
    assert any("局部快速轮换风险" in item["reason"] for item in reviews)


def test_subsegment_change_review_flags_hidden_internal_speaker_change():
    candidate = _candidate_from_labels(2, ["SPEAKER_B"], [4.0])
    candidate["segments"][0].update({
        "text": "有，我知道是谁我知道是谁，不是不是就是。",
        "speaker": "SPEAKER_D",
        "speaker_change_points": [1.6],
        "speaker_subsegments": [
            {"start": 0.0, "end": 1.4, "speaker": "SPEAKER_B", "duration": 1.4},
            {"start": 1.6, "end": 4.0, "speaker": "SPEAKER_D", "duration": 2.4},
        ],
    })

    reviews = _build_subsegment_change_review_segments(candidate)

    assert reviews
    assert reviews[0]["index"] == 0
    assert "段内短声纹窗" in reviews[0]["reason"]


def test_review_segments_are_copied_back_to_transcript_rows():
    candidate = _candidate_from_labels(2, ["SPEAKER_B", "SPEAKER_D"], [3.0, 3.0])
    candidate["review_segments"] = [{
        "index": 1,
        "start": candidate["segments"][1]["start"],
        "end": candidate["segments"][1]["end"],
        "duration_s": 3.0,
        "text": candidate["segments"][1]["text"],
        "from_speaker": "SPEAKER_D",
        "to_speaker": "SPEAKER_B",
        "reason": "局部夹心跳变：前后均为 B，当前 D，建议抽听确认",
    }]

    annotated = _annotate_segments_with_speaker_reviews(candidate)

    assert annotated["segments"][1]["speaker_assignment_review"] is True
    assert "局部夹心跳变" in annotated["segments"][1]["speaker_review_reason"]


def test_fragmented_extra_speakers_do_not_beat_clean_two_person_call():
    two = _candidate(2, [
        {"speaker": "SPEAKER_A", "segments": 89, "duration_s": 151.5, "turns": 24, "stable_turns": 15},
        {"speaker": "SPEAKER_B", "segments": 47, "duration_s": 78.5, "turns": 21, "stable_turns": 8},
    ])
    oversplit = _candidate(6, [
        {"speaker": "SPEAKER_A", "segments": 80, "duration_s": 138.6, "turns": 22, "stable_turns": 15},
        {"speaker": "SPEAKER_B", "segments": 33, "duration_s": 57.9, "turns": 13, "stable_turns": 7},
        {"speaker": "SPEAKER_C", "segments": 11, "duration_s": 11.3, "turns": 8, "stable_turns": 1},
        {"speaker": "SPEAKER_D", "segments": 12, "duration_s": 22.2, "turns": 8, "stable_turns": 1},
    ])

    best = _choose_diarization_candidate([two, oversplit])

    assert best is two
    assert oversplit["fragmented_speakers"] == 2


def test_fragmented_extra_speakers_merge_into_stable_neighbors():
    oversplit = _candidate(6, [
        {"speaker": "SPEAKER_A", "segments": 80, "duration_s": 138.6, "turns": 22, "stable_turns": 15},
        {"speaker": "SPEAKER_B", "segments": 33, "duration_s": 57.9, "turns": 13, "stable_turns": 7},
        {"speaker": "SPEAKER_C", "segments": 11, "duration_s": 11.3, "turns": 8, "stable_turns": 1},
        {"speaker": "SPEAKER_D", "segments": 12, "duration_s": 22.2, "turns": 8, "stable_turns": 1},
    ])

    merged = _merge_fragile_speakers(oversplit)

    assert merged["actual_n_speakers"] == 2
    assert set(merged["merge_map"]) == {"SPEAKER_C", "SPEAKER_D"}
    assert {s["speaker"] for s in merged["summary"]["speakers"]} == {"SPEAKER_A", "SPEAKER_B"}


def test_fragile_segments_use_lower_count_anchor_instead_of_context_only():
    anchor = _candidate_from_labels(
        2,
        ["SPEAKER_B", "SPEAKER_A", "SPEAKER_A", "SPEAKER_A", "SPEAKER_A"],
        [10.0, 8.0, 2.0, 8.0, 10.0],
    )
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_C", "SPEAKER_B", "SPEAKER_A"],
        [10.0, 8.0, 2.0, 8.0, 10.0],
    )
    candidate["fragile_speakers"] = ["SPEAKER_C"]
    candidate["mergeable_speakers"] = ["SPEAKER_C"]
    candidate["segments"][2]["speaker_votes"] = {"SPEAKER_C": 1.0, "SPEAKER_B": 7.0}

    merged = _merge_fragile_speakers(candidate, [anchor, candidate])

    assert merged["segments"][2]["speaker"] == "SPEAKER_B"
    assert merged["merge_distribution"] == {"SPEAKER_C": {"SPEAKER_B": 1}}


def test_default_postprocess_reassigns_only_isolated_fragments_without_merge_map():
    anchor = _candidate_from_labels(
        2,
        ["SPEAKER_B", "SPEAKER_A", "SPEAKER_A", "SPEAKER_A", "SPEAKER_A", "SPEAKER_B"],
        [10.0, 8.0, 4.0, 8.0, 10.0, 10.0],
    )
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_C", "SPEAKER_B", "SPEAKER_A", "SPEAKER_D"],
        [10.0, 8.0, 4.0, 8.0, 10.0, 10.0],
    )
    candidate["fragile_speakers"] = ["SPEAKER_C"]
    candidate["mergeable_speakers"] = ["SPEAKER_C"]
    candidate["segments"][2]["speaker_votes"] = {"SPEAKER_C": 1.0, "SPEAKER_B": 7.0}

    corrected = _reassign_isolated_fragile_segments(candidate, [anchor, candidate])

    assert corrected["actual_n_speakers"] == 3
    assert corrected["segments"][2]["speaker"] == "SPEAKER_B"
    assert corrected["segments"][5]["speaker"] == "SPEAKER_D"
    assert corrected["reassignment_distribution"] == {"SPEAKER_C": {"SPEAKER_B": 1}}
    assert corrected.get("merge_map") is None


def test_default_postprocess_keeps_isolated_speaker_when_original_votes_lead():
    anchor = _candidate_from_labels(
        2,
        ["SPEAKER_B", "SPEAKER_A", "SPEAKER_A", "SPEAKER_A", "SPEAKER_A", "SPEAKER_B"],
        [10.0, 8.0, 4.0, 8.0, 10.0, 10.0],
    )
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_C", "SPEAKER_B", "SPEAKER_A", "SPEAKER_D"],
        [10.0, 8.0, 4.0, 8.0, 10.0, 10.0],
    )
    candidate["fragile_speakers"] = ["SPEAKER_C"]
    candidate["mergeable_speakers"] = ["SPEAKER_C"]
    candidate["segments"][2]["speaker_votes"] = {"SPEAKER_C": 7.0, "SPEAKER_B": 1.0}
    candidate["segments"][2]["speaker_confidence"] = 0.875

    corrected = _reassign_isolated_fragile_segments(candidate, [anchor, candidate])

    assert corrected["segments"][2]["speaker"] == "SPEAKER_C"
    assert corrected["actual_n_speakers"] == 4
    assert corrected["reassignment_distribution"] == {}
    assert any("原说话人声纹投票占优" in item["reason"] for item in corrected["review_segments"])


def test_default_postprocess_does_not_reassign_overlap_risk():
    anchor = _candidate_from_labels(
        2,
        ["SPEAKER_B", "SPEAKER_A", "SPEAKER_A", "SPEAKER_A", "SPEAKER_A", "SPEAKER_B"],
        [10.0, 8.0, 4.0, 8.0, 10.0, 10.0],
    )
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_C", "SPEAKER_B", "SPEAKER_A", "SPEAKER_D"],
        [10.0, 8.0, 4.0, 8.0, 10.0, 10.0],
    )
    candidate["fragile_speakers"] = ["SPEAKER_C"]
    candidate["mergeable_speakers"] = ["SPEAKER_C"]
    candidate["segments"][2]["speaker_votes"] = {"SPEAKER_C": 1.0, "SPEAKER_B": 7.0}
    candidate["segments"][2]["speaker_overlap_risk"] = True

    corrected = _reassign_isolated_fragile_segments(candidate, [anchor, candidate])

    assert corrected["segments"][2]["speaker"] == "SPEAKER_C"
    assert corrected["reassignment_distribution"] == {}
    assert any("重叠语音风险" in item["reason"] for item in corrected["review_segments"])


def test_default_postprocess_does_not_reassign_text_without_acoustic_votes():
    anchor = _candidate_from_labels(
        2,
        ["SPEAKER_B", "SPEAKER_A", "SPEAKER_A", "SPEAKER_A", "SPEAKER_A", "SPEAKER_B"],
        [10.0, 8.0, 4.0, 8.0, 10.0, 10.0],
    )
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_C", "SPEAKER_B", "SPEAKER_A", "SPEAKER_D"],
        [10.0, 8.0, 4.0, 8.0, 10.0, 10.0],
    )
    candidate["fragile_speakers"] = ["SPEAKER_C"]
    candidate["mergeable_speakers"] = ["SPEAKER_C"]

    corrected = _reassign_isolated_fragile_segments(candidate, [anchor, candidate])

    assert corrected["segments"][2]["speaker"] == "SPEAKER_C"
    assert corrected["reassignment_distribution"] == {}
    assert any("缺少短窗声纹投票" in item["reason"] for item in corrected["review_segments"])


def test_default_postprocess_keeps_continuous_fragile_speaker_visible():
    anchor = _candidate_from_labels(
        2,
        ["SPEAKER_B", "SPEAKER_A", "SPEAKER_A", "SPEAKER_A", "SPEAKER_B"],
        [10.0, 8.0, 4.0, 4.0, 10.0],
    )
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_C", "SPEAKER_C", "SPEAKER_D"],
        [10.0, 8.0, 4.0, 4.0, 10.0],
    )
    candidate["fragile_speakers"] = ["SPEAKER_C"]
    candidate["mergeable_speakers"] = ["SPEAKER_C"]

    corrected = _reassign_isolated_fragile_segments(candidate, [anchor, candidate])

    assert corrected["segments"][2]["speaker"] == "SPEAKER_C"
    assert corrected["segments"][3]["speaker"] == "SPEAKER_C"
    assert corrected["actual_n_speakers"] == 4
    assert corrected["reassignment_distribution"] == {}


def test_fixed_count_postprocess_does_not_silently_reduce_requested_speakers():
    anchor = _candidate_from_labels(
        2,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_B", "SPEAKER_B"],
        [10.0, 8.0, 4.0, 10.0],
    )
    candidate = _candidate_from_labels(
        3,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_C", "SPEAKER_B"],
        [10.0, 8.0, 4.0, 10.0],
    )
    candidate["fragile_speakers"] = ["SPEAKER_C"]
    candidate["mergeable_speakers"] = ["SPEAKER_C"]

    corrected = _postprocess_fixed_count_candidate(candidate, [anchor, candidate], requested_n=3)

    assert corrected["actual_n_speakers"] == 3
    assert corrected["segments"][2]["speaker"] == "SPEAKER_C"
    assert corrected["reassignment_distribution"] == {}
    assert corrected["postprocess_skipped_reason"] == ""
    assert "声纹/声线证据门禁" in corrected["voice_guard_reason"]


def test_windowed_fallback_smooths_sandwiched_local_drift_only():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_A"] * 8 + ["SPEAKER_B"] * 3 + ["SPEAKER_A"] * 8 + ["SPEAKER_B"] * 8,
        [4.0] * 27,
    )
    candidate["stats"] = {"assignment_mode": "windowed_kmeans"}
    for idx in [8, 9, 10]:
        candidate["segments"][idx]["speaker_votes"] = {"SPEAKER_B": 1.0, "SPEAKER_A": 7.0}

    corrected = _smooth_windowed_sandwiched_runs(candidate)

    assert corrected["segments"][8]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][9]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][10]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][-1]["speaker"] == "SPEAKER_B"
    assert corrected["smoothing_distribution"] == {"SPEAKER_B": {"SPEAKER_A": 3}}


def test_windowed_smoothing_does_not_run_on_normal_global_candidate():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_A"] * 8 + ["SPEAKER_B"] * 3 + ["SPEAKER_A"] * 8,
        [4.0] * 19,
    )
    candidate["stats"] = {"assignment_mode": "global_kmeans"}

    corrected = _smooth_windowed_sandwiched_runs(candidate)

    assert corrected["segments"][8]["speaker"] == "SPEAKER_B"
    assert corrected["smoothing_distribution"] == {}


def test_alternating_local_speaker_leakage_is_smoothed_without_merging_real_block():
    labels = (
        ["SPEAKER_A"] * 8
        + ["SPEAKER_B"] * 5
        + ["SPEAKER_D"] * 2
        + ["SPEAKER_B"] * 6
        + ["SPEAKER_D"]
        + ["SPEAKER_B"] * 5
        + ["SPEAKER_D"] * 2
        + ["SPEAKER_B"] * 6
        + ["SPEAKER_C"] * 5
        + ["SPEAKER_D"] * 8
    )
    durations = [4.0] * len(labels)
    candidate = _candidate_from_labels(4, labels, durations)
    for idx in [13, 14, 21, 27, 28]:
        candidate["segments"][idx]["speaker_votes"] = {"SPEAKER_D": 1.0, "SPEAKER_B": 7.0}

    corrected = _smooth_alternating_local_speaker_leakage(candidate)

    # Short repeated D runs inside the B conversation are local leakage.
    for idx in [13, 14, 21, 27, 28]:
        assert corrected["segments"][idx]["speaker"] == "SPEAKER_B"
    # A later coherent D block is a real participant turn and must remain.
    assert all(seg["speaker"] == "SPEAKER_D" for seg in corrected["segments"][-8:])
    assert corrected["actual_n_speakers"] == 4
    assert corrected["local_leakage_distribution"] == {"SPEAKER_D": {"SPEAKER_B": 5}}


def test_alternating_smoothing_keeps_high_confidence_senko_segments():
    labels = (
        ["SPEAKER_A"] * 8
        + ["SPEAKER_B"] * 5
        + ["SPEAKER_D"] * 2
        + ["SPEAKER_B"] * 6
        + ["SPEAKER_D"]
        + ["SPEAKER_B"] * 5
        + ["SPEAKER_D"] * 2
        + ["SPEAKER_B"] * 6
    )
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))
    for seg in candidate["segments"]:
        if seg["speaker"] == "SPEAKER_D":
            seg["speaker_confidence"] = 0.9
            seg["speaker_votes"] = {"SPEAKER_D": 9.0, "SPEAKER_B": 1.0}

    corrected = _smooth_alternating_local_speaker_leakage(candidate)

    assert [seg["speaker"] for seg in corrected["segments"]] == labels
    assert corrected["local_leakage_distribution"] == {}


def test_alternating_smoothing_voice_guard_blocks_cross_voice_rewrite():
    labels = (
        ["SPEAKER_A"] * 8
        + ["SPEAKER_B"] * 5
        + ["SPEAKER_D"] * 2
        + ["SPEAKER_B"] * 6
        + ["SPEAKER_D"]
        + ["SPEAKER_B"] * 5
        + ["SPEAKER_D"] * 2
        + ["SPEAKER_B"] * 6
    )
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))
    for seg in candidate["segments"]:
        if seg["speaker"] == "SPEAKER_B":
            seg["voice_pitch_hz"] = 220.0
            seg["voice_pitch_confidence"] = 0.8
            seg["voice_band"] = "high"
        elif seg["speaker"] == "SPEAKER_D":
            seg["voice_pitch_hz"] = 115.0
            seg["voice_pitch_confidence"] = 0.8
            seg["voice_band"] = "low"

    corrected = _smooth_alternating_local_speaker_leakage(candidate)

    assert [seg["speaker"] for seg in corrected["segments"]] == labels
    assert corrected["local_leakage_distribution"] == {}
    assert corrected["voice_guard_count"] == 5
    assert "声线护栏" in corrected["voice_guard_reason"]


def test_alternating_smoothing_still_rewrites_same_voice_leakage():
    labels = (
        ["SPEAKER_A"] * 8
        + ["SPEAKER_B"] * 5
        + ["SPEAKER_D"] * 2
        + ["SPEAKER_B"] * 6
        + ["SPEAKER_D"]
        + ["SPEAKER_B"] * 5
        + ["SPEAKER_D"] * 2
        + ["SPEAKER_B"] * 6
    )
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))
    for seg in candidate["segments"]:
        if seg["speaker"] in {"SPEAKER_B", "SPEAKER_D"}:
            seg["voice_pitch_hz"] = 135.0 if seg["speaker"] == "SPEAKER_B" else 128.0
            seg["voice_pitch_confidence"] = 0.8
            seg["voice_band"] = "low"
    for idx in [13, 14, 21, 27, 28]:
        candidate["segments"][idx]["speaker_votes"] = {"SPEAKER_D": 1.0, "SPEAKER_B": 7.0}

    corrected = _smooth_alternating_local_speaker_leakage(candidate)

    for idx in [13, 14, 21, 27, 28]:
        assert corrected["segments"][idx]["speaker"] == "SPEAKER_B"
    assert corrected["local_leakage_distribution"] == {"SPEAKER_D": {"SPEAKER_B": 5}}


def test_discourse_continuity_repairs_short_sandwiched_clause():
    labels = (
        ["SPEAKER_A"] * 6
        + ["SPEAKER_B"]
        + ["SPEAKER_A"] * 6
        + ["SPEAKER_C"] * 4
    )
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))
    candidate["segments"][5]["text"] = "对，呃，是这样啊，那个营磊，我告诉你，这是两件事情，我们不。"
    candidate["segments"][6]["text"] = "否认所有的同工对团体的付出，那是肯定的。"
    candidate["segments"][7]["text"] = "但是不是代表，因为我们做的好是应该的。"
    for seg in candidate["segments"]:
        if seg["speaker"] == "SPEAKER_A":
            seg["voice_pitch_hz"] = 225.0
            seg["voice_pitch_confidence"] = 0.9
            seg["speaker_votes"] = {"SPEAKER_A": 8.0}
        elif seg["speaker"] == "SPEAKER_B":
            seg["voice_pitch_hz"] = 215.0
            seg["voice_pitch_confidence"] = 0.9
            seg["speaker_confidence"] = 0.6
            seg["speaker_votes"] = {"SPEAKER_B": 2.0, "SPEAKER_A": 7.0}
        else:
            seg["voice_pitch_hz"] = 130.0
            seg["voice_pitch_confidence"] = 0.9
            seg["speaker_votes"] = {"SPEAKER_C": 8.0}
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 4))

    corrected = _repair_discourse_continuity_assignments(candidate)

    assert corrected["segments"][6]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][6]["continuity_repaired"] is True
    assert corrected["continuity_repair_distribution"] == {"SPEAKER_B": {"SPEAKER_A": 1}}


def test_discourse_continuity_repairs_multi_segment_clause_after_open_negative():
    labels = (
        ["SPEAKER_A"] * 6
        + ["SPEAKER_B"] * 3
        + ["SPEAKER_A"] * 6
        + ["SPEAKER_C"] * 4
    )
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))
    candidate["segments"][5]["text"] = "这是两件事情，我们不。"
    candidate["segments"][6]["text"] = "否认所有人的付出，那是肯定的，也是感恩的。"
    candidate["segments"][7]["text"] = "但是不是代表，因为我们做得好是应该的。"
    candidate["segments"][8]["text"] = "但是如果说刚一转正。"
    candidate["segments"][9]["text"] = "接下来全都没有来服侍，这会造成新的问题。"
    for seg in candidate["segments"]:
        if seg["speaker"] == "SPEAKER_A":
            seg["voice_pitch_hz"] = 225.0
            seg["voice_pitch_confidence"] = 0.9
            seg["speaker_votes"] = {"SPEAKER_A": 8.0}
        elif seg["speaker"] == "SPEAKER_B":
            seg["voice_pitch_hz"] = 215.0
            seg["voice_pitch_confidence"] = 0.9
            seg["speaker_confidence"] = 0.6
            seg["speaker_votes"] = {"SPEAKER_B": 2.0, "SPEAKER_A": 7.0}
        else:
            seg["voice_pitch_hz"] = 130.0
            seg["voice_pitch_confidence"] = 0.9
            seg["speaker_votes"] = {"SPEAKER_C": 8.0}
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 4))

    corrected = _repair_discourse_continuity_assignments(candidate)

    for idx in [6, 7, 8]:
        assert corrected["segments"][idx]["speaker"] == "SPEAKER_A"
        assert corrected["segments"][idx]["continuity_repaired"] is True
    assert corrected["continuity_repair_distribution"] == {"SPEAKER_B": {"SPEAKER_A": 3}}


def test_discourse_continuity_keeps_standalone_question_turn():
    labels = ["SPEAKER_A"] * 6 + ["SPEAKER_B"] + ["SPEAKER_A"] * 6 + ["SPEAKER_C"] * 4
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))
    candidate["segments"][6]["text"] = "我想问一下，这个制度到底是怎么定的？"
    for seg in candidate["segments"]:
        if seg["speaker"] == "SPEAKER_A":
            seg["voice_pitch_hz"] = 225.0
            seg["voice_pitch_confidence"] = 0.9
        elif seg["speaker"] == "SPEAKER_B":
            seg["voice_pitch_hz"] = 135.0
            seg["voice_pitch_confidence"] = 0.9
        else:
            seg["voice_pitch_hz"] = 130.0
            seg["voice_pitch_confidence"] = 0.9
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 4))

    corrected = _repair_discourse_continuity_assignments(candidate)

    assert corrected["segments"][6]["speaker"] == "SPEAKER_B"
    assert corrected["continuity_repair_distribution"] == {}


def test_discourse_continuity_repairs_open_clause_even_when_votes_do_not_support_target():
    labels = (
        ["SPEAKER_A"] * 6
        + ["SPEAKER_B"]
        + ["SPEAKER_A"] * 6
        + ["SPEAKER_C"] * 4
    )
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))
    candidate["segments"][5]["text"] = "对，呃，是这样啊，那个营磊，我告诉你，这是两件事情，我们不。"
    candidate["segments"][6]["text"] = "否认所有的同工对团体的付出，那是肯定的。"
    candidate["segments"][7]["text"] = "但是不是代表，因为我们做的好是应该的。"
    for seg in candidate["segments"]:
        if seg["speaker"] == "SPEAKER_A":
            seg["voice_pitch_hz"] = 225.0
            seg["voice_pitch_confidence"] = 0.9
            seg["speaker_votes"] = {"SPEAKER_A": 8.0}
        elif seg["speaker"] == "SPEAKER_B":
            seg["voice_pitch_hz"] = 215.0
            seg["voice_pitch_confidence"] = 0.9
            seg["speaker_confidence"] = 0.6
            seg["speaker_votes"] = {"SPEAKER_B": 4.0, "SPEAKER_A": 3.0}
        else:
            seg["voice_pitch_hz"] = 130.0
            seg["voice_pitch_confidence"] = 0.9
            seg["speaker_votes"] = {"SPEAKER_C": 8.0}
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 4))

    corrected = _repair_discourse_continuity_assignments(candidate)

    assert corrected["segments"][6]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][6]["continuity_repaired"] is True
    assert corrected["continuity_repair_distribution"] == {"SPEAKER_B": {"SPEAKER_A": 1}}
    assert any("开放句法未完成" in item["reason"] for item in corrected["review_segments"])


def test_discourse_continuity_keeps_open_negative_clause_when_voice_line_disagrees():
    labels = (
        ["SPEAKER_A"] * 6
        + ["SPEAKER_B"] * 3
        + ["SPEAKER_A"] * 6
        + ["SPEAKER_C"] * 4
    )
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))
    candidate["segments"][5]["text"] = "对，呃，是这样啊，那个营磊，我告诉你，这是两件事情，我们不。"
    candidate["segments"][6]["text"] = "否认所有的同工对团体的付出，那是肯定的，也且是感恩的。"
    candidate["segments"][7]["text"] = "但是不是代表，因为我们做的好是应该的。"
    candidate["segments"][8]["text"] = "但是如果说刚一转正。"
    candidate["segments"][9]["text"] = "接下来全都没有来服侍，这会造成新同工。"
    for seg in candidate["segments"]:
        if seg["speaker"] == "SPEAKER_A":
            seg["voice_pitch_hz"] = 225.0
            seg["voice_pitch_confidence"] = 0.9
            seg["speaker_votes"] = {"SPEAKER_A": 8.0}
        elif seg["speaker"] == "SPEAKER_B":
            seg["voice_pitch_hz"] = 135.0
            seg["voice_pitch_confidence"] = 0.9
            seg["speaker_confidence"] = 1.0
            seg["speaker_votes"] = {"SPEAKER_B": 8.0}
        else:
            seg["voice_pitch_hz"] = 130.0
            seg["voice_pitch_confidence"] = 0.9
            seg["speaker_votes"] = {"SPEAKER_C": 8.0}
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 4))

    corrected = _repair_discourse_continuity_assignments(candidate)

    for idx in [6, 7, 8]:
        assert corrected["segments"][idx]["speaker"] == "SPEAKER_B"
        assert "continuity_repaired" not in corrected["segments"][idx]
    assert corrected["continuity_repair_distribution"] == {}
    assert any("声线/男女特征不支持安全合并" in item["reason"] for item in corrected["review_segments"])


def test_discourse_continuity_keeps_open_clause_across_voice_guard():
    labels = (
        ["SPEAKER_A"] * 6
        + ["SPEAKER_B"] * 3
        + ["SPEAKER_A"] * 6
        + ["SPEAKER_C"] * 4
    )
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))
    candidate["segments"][5]["text"] = "这是两件事情，我们不。"
    candidate["segments"][6]["text"] = "否认所有的同工对团体的付出，那是肯定的，也是感恩的。"
    candidate["segments"][7]["text"] = "但是不是代表，因为我们做的好是应该的。"
    candidate["segments"][8]["text"] = "但是如果说刚一转正。"
    candidate["segments"][9]["text"] = "接下来全都没有来服侍，这会造成新的问题。"
    for seg in candidate["segments"]:
        if seg["speaker"] == "SPEAKER_A":
            seg["voice_pitch_hz"] = 225.0
            seg["voice_pitch_confidence"] = 0.9
            seg["speaker_votes"] = {"SPEAKER_A": 8.0}
        elif seg["speaker"] == "SPEAKER_B":
            seg["voice_pitch_hz"] = 135.0
            seg["voice_pitch_confidence"] = 0.9
            seg["speaker_confidence"] = 0.6
            seg["speaker_votes"] = {"SPEAKER_B": 4.0, "SPEAKER_A": 3.0}
        else:
            seg["voice_pitch_hz"] = 130.0
            seg["voice_pitch_confidence"] = 0.9
            seg["speaker_votes"] = {"SPEAKER_C": 8.0}
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 4))

    corrected = _repair_discourse_continuity_assignments(candidate)

    for idx in [6, 7, 8]:
        assert corrected["segments"][idx]["speaker"] == "SPEAKER_B"
        assert "continuity_repaired" not in corrected["segments"][idx]
    assert corrected["continuity_repair_distribution"] == {}
    assert any("声线/男女特征不支持安全合并" in item["reason"] for item in corrected["review_segments"])


def test_voice_line_refine_is_noop_without_audio_path():
    candidate = _candidate_from_labels(2, ["SPEAKER_A", "SPEAKER_B"], [4.0, 4.0])

    refined = _refine_conflicting_voice_bands_with_pyin(candidate, None)

    assert refined["segments"] == candidate["segments"]
    assert refined["voice_line_refine_count"] == 0
    assert refined["voice_line_refine_reason"] == ""


def test_pyin_pitch_reuses_senko_analysis_wav(monkeypatch, tmp_path):
    from scribe_py import ipc as ipc_mod
    from scribe_py.diarizers import senko_diarizer

    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"source")
    cached_wav = tmp_path / "meeting-16k.wav"
    cached_wav.write_bytes(b"pcm")
    loaded = []

    def fake_load(path, **_kwargs):
        loaded.append(path)
        raise RuntimeError("stop after path selection")

    monkeypatch.setattr(senko_diarizer, "cached_analysis_wav", lambda _audio: cached_wav)
    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(load=fake_load))
    ipc_mod._PYIN_PITCH_CACHE.clear()

    result = _estimate_pyin_pitch_for_segment(
        source,
        {"start": 20.0, "end": 22.0},
    )

    assert result == (None, 0.0, "unknown")
    assert loaded == [str(cached_wav)]


def test_voice_line_refine_updates_conflicting_segment_when_pyin_disagrees(monkeypatch):
    labels = ["SPEAKER_A"] * 5 + ["SPEAKER_B"] + ["SPEAKER_A"] * 5 + ["SPEAKER_C"] * 5 + ["SPEAKER_B"] * 3
    candidate = _candidate_from_labels(3, labels, [4.0] * len(labels))
    for seg in candidate["segments"]:
        if seg["speaker"] == "SPEAKER_A":
            seg["voice_pitch_hz"] = 225.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "high"
        elif seg["speaker"] == "SPEAKER_B":
            seg["voice_pitch_hz"] = 120.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
        else:
            seg["voice_pitch_hz"] = 125.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
    candidate["segments"][5]["voice_pitch_hz"] = 240.0
    candidate["segments"][5]["voice_pitch_confidence"] = 0.9
    candidate["segments"][5]["voice_band"] = "high"
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 3))

    monkeypatch.setattr(
        "scribe_py.ipc._estimate_pyin_pitch_for_segment",
        lambda _audio, _seg: (150.0, 0.21, "low"),
    )

    refined = _refine_conflicting_voice_bands_with_pyin(candidate, Path("/tmp/fake.wav"))

    assert refined["segments"][5]["voice_pitch_hz"] == 150.0
    assert refined["segments"][5]["voice_band"] == "low"
    assert refined["segments"][5]["voice_line_refined"] is True
    assert refined["voice_line_refine_count"] == 1
    assert "YIN" in refined["voice_line_refine_reason"]


def test_voice_band_repair_reassigns_cross_gender_sandwich_when_context_agrees():
    labels = (
        ["SPEAKER_A"] * 5
        + ["SPEAKER_D"]
        + ["SPEAKER_A"] * 5
        + ["SPEAKER_C"] * 5
    )
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))
    for seg in candidate["segments"]:
        if seg["speaker"] == "SPEAKER_A":
            seg["voice_pitch_hz"] = 225.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "high"
            seg["speaker_votes"] = {"SPEAKER_A": 8.0}
        elif seg["speaker"] == "SPEAKER_C":
            seg["voice_pitch_hz"] = 135.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
            seg["speaker_votes"] = {"SPEAKER_C": 8.0}
    candidate["segments"][5]["voice_pitch_hz"] = 226.0
    candidate["segments"][5]["voice_pitch_confidence"] = 0.9
    candidate["segments"][5]["voice_band"] = "high"
    candidate["segments"][5]["speaker_votes"] = {"SPEAKER_A": 7.0, "SPEAKER_D": 1.0}
    candidate["segments"][5]["speaker_confidence"] = 0.6
    # Give D a reliable low profile elsewhere, so the sandwiched high segment
    # is a clear cross-voice assignment rather than an unknown profile.
    candidate["segments"].append({
        "start": 100.0,
        "end": 110.0,
        "text": "real low speaker",
        "speaker": "SPEAKER_D",
        "voice_pitch_hz": 125.0,
        "voice_pitch_confidence": 0.9,
        "voice_band": "low",
        "speaker_votes": {"SPEAKER_D": 9.0},
    })
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 4))

    corrected = _repair_voice_band_assignments(candidate)

    assert corrected["segments"][5]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][5]["voice_band_repaired"] is True
    assert corrected["voice_band_repair_distribution"] == {"SPEAKER_D": {"SPEAKER_A": 1}}
    assert "高低声线" in corrected["voice_band_repair_reason"]


def test_voice_band_repair_reviews_conflict_without_safe_target():
    labels = ["SPEAKER_A"] * 5 + ["SPEAKER_C"] * 2 + ["SPEAKER_B"] * 5 + ["SPEAKER_C"] * 5
    candidate = _candidate_from_labels(3, labels, [4.0] * len(labels))
    for seg in candidate["segments"]:
        if seg["speaker"] == "SPEAKER_A":
            seg["voice_pitch_hz"] = 225.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "high"
            seg["speaker_votes"] = {"SPEAKER_A": 8.0}
        elif seg["speaker"] == "SPEAKER_B":
            seg["voice_pitch_hz"] = 130.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
            seg["speaker_votes"] = {"SPEAKER_B": 8.0}
        elif seg["speaker"] == "SPEAKER_C":
            seg["voice_pitch_hz"] = 135.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
            seg["speaker_votes"] = {"SPEAKER_C": 8.0}
    candidate["segments"][6]["voice_pitch_hz"] = 224.0
    candidate["segments"][6]["voice_pitch_confidence"] = 0.9
    candidate["segments"][6]["voice_band"] = "high"
    candidate["segments"][6]["speaker_votes"] = {"SPEAKER_C": 8.0}
    candidate["segments"][6]["speaker_confidence"] = 0.98
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 3))

    corrected = _repair_voice_band_assignments(candidate)

    assert corrected["segments"][6]["speaker"] == "SPEAKER_C"
    assert corrected["voice_band_repair_distribution"] == {}
    assert any(item["index"] == 6 and "声线复核" in item["reason"] for item in corrected["review_segments"])


def test_voice_band_repair_reviews_strong_cross_voice_sandwich_when_target_votes_lag():
    labels = (
        ["SPEAKER_D"] * 4
        + ["SPEAKER_B"] * 5
        + ["SPEAKER_D"]
        + ["SPEAKER_B"] * 6
    )
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))
    for seg in candidate["segments"]:
        if seg["speaker"] == "SPEAKER_B":
            seg["voice_pitch_hz"] = 235.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "high"
            seg["speaker_votes"] = {"SPEAKER_B": 8.0}
        elif seg["speaker"] == "SPEAKER_D":
            seg["voice_pitch_hz"] = 130.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
            seg["speaker_votes"] = {"SPEAKER_D": 8.0}
    bad_idx = 9
    candidate["segments"][bad_idx]["voice_pitch_hz"] = 235.0
    candidate["segments"][bad_idx]["voice_pitch_confidence"] = 0.9
    candidate["segments"][bad_idx]["voice_band"] = "high"
    candidate["segments"][bad_idx]["speaker_votes"] = {"SPEAKER_B": 3.0, "SPEAKER_D": 13.0}
    candidate["segments"][bad_idx]["speaker_confidence"] = 0.82
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 4))

    corrected = _repair_voice_band_assignments(candidate)

    assert corrected["segments"][bad_idx]["speaker"] == "SPEAKER_D"
    assert "voice_band_repaired" not in corrected["segments"][bad_idx]
    assert corrected["voice_band_repair_distribution"] == {}
    assert any("短窗声纹投票不足" in item["reason"] for item in corrected["review_segments"])


def test_voice_band_repair_reassigns_when_pitch_and_votes_both_support_target():
    labels = (
        ["SPEAKER_D"] * 4
        + ["SPEAKER_B"] * 5
        + ["SPEAKER_D"]
        + ["SPEAKER_B"] * 6
    )
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))
    for seg in candidate["segments"]:
        if seg["speaker"] == "SPEAKER_B":
            seg["voice_pitch_hz"] = 235.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "high"
            seg["speaker_votes"] = {"SPEAKER_B": 8.0}
        elif seg["speaker"] == "SPEAKER_D":
            seg["voice_pitch_hz"] = 130.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
            seg["speaker_votes"] = {"SPEAKER_D": 8.0}
    bad_idx = 9
    candidate["segments"][bad_idx]["voice_pitch_hz"] = 235.0
    candidate["segments"][bad_idx]["voice_pitch_confidence"] = 0.9
    candidate["segments"][bad_idx]["voice_band"] = "high"
    candidate["segments"][bad_idx]["speaker_votes"] = {"SPEAKER_B": 10.0, "SPEAKER_D": 2.0}
    candidate["segments"][bad_idx]["speaker_confidence"] = 0.62
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 4))

    corrected = _repair_voice_band_assignments(candidate)

    assert corrected["segments"][bad_idx]["speaker"] == "SPEAKER_B"
    assert corrected["segments"][bad_idx]["voice_band_repaired"] is True
    assert corrected["voice_band_repair_distribution"] == {"SPEAKER_D": {"SPEAKER_B": 1}}


def test_voice_band_repair_reviews_local_voice_sandwich_when_votes_lag():
    labels = (
        ["SPEAKER_D"] * 4
        + ["SPEAKER_B"] * 5
        + ["SPEAKER_D"]
        + ["SPEAKER_B"] * 6
    )
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))
    for seg in candidate["segments"]:
        if seg["speaker"] == "SPEAKER_B":
            seg["voice_pitch_hz"] = 235.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "high"
            seg["speaker_votes"] = {"SPEAKER_B": 8.0}
        elif seg["speaker"] == "SPEAKER_D":
            seg["voice_pitch_hz"] = 130.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
            seg["speaker_votes"] = {"SPEAKER_D": 8.0}
    bad_idx = 9
    candidate["segments"][bad_idx]["voice_pitch_hz"] = 235.0
    candidate["segments"][bad_idx]["voice_pitch_confidence"] = 0.9
    candidate["segments"][bad_idx]["voice_band"] = "high"
    candidate["segments"][bad_idx]["speaker_votes"] = {"SPEAKER_B": 3.0, "SPEAKER_D": 10.0}
    candidate["segments"][bad_idx]["speaker_confidence"] = 0.76
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 4))

    corrected = _repair_voice_band_assignments(candidate)

    assert corrected["segments"][bad_idx]["speaker"] == "SPEAKER_D"
    assert "voice_band_repaired" not in corrected["segments"][bad_idx]
    assert corrected["voice_band_repair_distribution"] == {}
    assert any("短窗声纹投票不足" in item["reason"] for item in corrected["review_segments"])


def test_voice_band_repair_does_not_reverse_handoff_split_segments():
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A", "SPEAKER_A", "SPEAKER_D", "SPEAKER_D", "SPEAKER_A", "SPEAKER_C"],
        [6.0, 6.0, 4.0, 4.0, 6.0, 8.0],
    )
    for idx, seg in enumerate(candidate["segments"]):
        if seg["speaker"] == "SPEAKER_A":
            seg["voice_pitch_hz"] = 225.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "high"
        elif seg["speaker"] == "SPEAKER_D":
            seg["voice_pitch_hz"] = 125.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
        else:
            seg["voice_pitch_hz"] = 130.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
    candidate["segments"][2]["voice_pitch_hz"] = 245.0
    candidate["segments"][2]["voice_pitch_confidence"] = 0.9
    candidate["segments"][2]["voice_band"] = "high"
    candidate["segments"][2]["speaker_votes"] = {"SPEAKER_A": 9.0}
    candidate["segments"][2]["speaker_handoff_split"] = True
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 4))

    corrected = _repair_voice_band_assignments(candidate)

    assert corrected["segments"][2]["speaker"] == "SPEAKER_D"
    assert corrected["voice_band_repair_distribution"] == {}


def test_voice_band_repair_does_not_reverse_discourse_continuity_fix():
    labels = (
        ["SPEAKER_A"] * 6
        + ["SPEAKER_B"]
        + ["SPEAKER_A"] * 6
        + ["SPEAKER_C"] * 5
    )
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))
    candidate["segments"][5]["text"] = "这是两件事情，我们不。"
    candidate["segments"][6]["text"] = "否认所有的同工对团体的付出，那是肯定的。"
    candidate["segments"][7]["text"] = "但是不是代表，因为我们做的好是应该的。"
    for seg in candidate["segments"]:
        if seg["speaker"] == "SPEAKER_A":
            seg["voice_pitch_hz"] = 225.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "high"
            seg["speaker_votes"] = {"SPEAKER_A": 8.0}
        elif seg["speaker"] == "SPEAKER_B":
            seg["voice_pitch_hz"] = 215.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
            seg["speaker_confidence"] = 0.6
            seg["speaker_votes"] = {"SPEAKER_B": 2.0, "SPEAKER_A": 7.0}
        else:
            seg["voice_pitch_hz"] = 130.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
            seg["speaker_votes"] = {"SPEAKER_C": 8.0}
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 4))

    corrected = _repair_voice_band_assignments(_repair_discourse_continuity_assignments(candidate))

    assert corrected["segments"][6]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][6]["continuity_repaired"] is True


def test_voice_mix_summary_flags_speaker_with_real_high_low_contamination():
    labels = ["SPEAKER_A"] * 10 + ["SPEAKER_B"] * 12
    candidate = _candidate_from_labels(2, labels, [3.0] * len(labels))
    for idx, seg in enumerate(candidate["segments"]):
        if seg["speaker"] == "SPEAKER_A":
            seg["voice_pitch_hz"] = 225.0 if idx < 6 else 130.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "high" if idx < 6 else "low"
        else:
            seg["voice_pitch_hz"] = 128.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 2))

    mix = _speaker_voice_band_mix_summary(candidate["segments"])

    assert mix["SPEAKER_A"]["mixed"] is True
    assert candidate["mixed_voice_speakers"] == ["SPEAKER_A"]
    assert candidate["voice_mix_penalty"] > 0
    assert "高低声线" in candidate["reason"]


def test_severe_voice_mix_heavily_penalizes_same_speaker_high_low_bucket():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_A"] * 12 + ["SPEAKER_B"] * 8,
        [3.0] * 20,
    )
    for idx, seg in enumerate(candidate["segments"]):
        if seg["speaker"] == "SPEAKER_A":
            seg["voice_pitch_hz"] = 225.0 if idx < 6 else 125.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "high" if idx < 6 else "low"
        else:
            seg["voice_pitch_hz"] = 128.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 2))

    assert candidate["severe_mixed_voice_speakers"] == ["SPEAKER_A"]
    assert candidate["voice_mix_summary"]["SPEAKER_A"]["severe_mixed"] is True
    assert candidate["voice_mix_penalty"] >= 18.0
    assert "严重声线混标" in candidate["reason"]


def test_voice_line_groups_expose_broad_voice_bands_without_using_them_as_identity():
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_C", "SPEAKER_D"],
        [8.0, 8.0, 8.0, 8.0],
    )
    pitches = {
        "SPEAKER_A": 230.0,
        "SPEAKER_B": 130.0,
        "SPEAKER_C": 210.0,
        "SPEAKER_D": 125.0,
    }
    for seg in candidate["segments"]:
        seg["voice_pitch_hz"] = pitches[seg["speaker"]]
        seg["voice_pitch_confidence"] = 0.9
    groups = _speaker_voice_line_groups(candidate["segments"])

    assert groups["groups"]["high"] == ["SPEAKER_A", "SPEAKER_C"]
    assert groups["groups"]["low"] == ["SPEAKER_B", "SPEAKER_D"]
    assert groups["line_labels"]["SPEAKER_A"] == "H1"
    assert groups["line_labels"]["SPEAKER_D"] == "L2"


def test_higher_candidate_can_win_when_it_resolves_mixed_voice_bucket():
    lower = _candidate_from_labels(
        3,
        ["SPEAKER_A"] * 10 + ["SPEAKER_B"] * 12 + ["SPEAKER_C"] * 10,
        [3.0] * 32,
    )
    higher = _candidate_from_labels(
        4,
        ["SPEAKER_A"] * 6
        + ["SPEAKER_D"] * 4
        + ["SPEAKER_B"] * 12
        + ["SPEAKER_C"] * 10,
        [3.0] * 32,
    )
    for idx, seg in enumerate(lower["segments"]):
        if seg["speaker"] == "SPEAKER_A":
            seg["voice_pitch_hz"] = 225.0 if idx < 6 else 130.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "high" if idx < 6 else "low"
        elif seg["speaker"] == "SPEAKER_B":
            seg["voice_pitch_hz"] = 220.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "high"
        else:
            seg["voice_pitch_hz"] = 128.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
    for seg in higher["segments"]:
        if seg["speaker"] in {"SPEAKER_A", "SPEAKER_B"}:
            seg["voice_pitch_hz"] = 225.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "high"
        else:
            seg["voice_pitch_hz"] = 130.0
            seg["voice_pitch_confidence"] = 0.9
            seg["voice_band"] = "low"
    lower["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(lower["segments"])
    lower.update(_score_diarization_candidate(lower, 3))
    higher["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(higher["segments"])
    higher.update(_score_diarization_candidate(higher, 4))
    lower["score"] = 21.0
    higher["score"] = 18.0

    best = _choose_diarization_candidate([lower, higher])

    assert best is higher
    assert "高低声线混标" in higher["refinement_reason"]


def test_handoff_split_moves_new_question_to_following_speaker():
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A", "SPEAKER_A", "SPEAKER_D", "SPEAKER_D", "SPEAKER_B"],
        [8.0, 12.0, 4.0, 5.0, 5.0],
    )
    candidate["segments"][1]["text"] = (
        "所有的把那个爱换成你自己的名字读一遍，我告诉你什么都没总"
        "我还想有几句话想说，就是我还想了解一下，就是我们之前有的同工被。"
    )
    candidate["segments"][1]["speaker_votes"] = {"SPEAKER_A": 6.0, "SPEAKER_D": 4.0}
    candidate["segments"][1]["speaker_confidence"] = 0.6
    candidate["segments"][2]["speaker_votes"] = {"SPEAKER_D": 4.0}
    candidate["segments"][2]["speaker_confidence"] = 1.0

    corrected = _split_handoff_segments(candidate)

    assert corrected["handoff_split_count"] == 1
    assert len(corrected["segments"]) == len(candidate["segments"]) + 1
    assert corrected["segments"][1]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][2]["speaker"] == "SPEAKER_D"
    assert "我还想有几句话想说" in corrected["segments"][2]["text"]
    assert corrected["segments"][2]["speaker_handoff_split"] is True


def test_handoff_split_does_not_override_strong_diarizer_votes():
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A", "SPEAKER_A", "SPEAKER_D", "SPEAKER_D", "SPEAKER_B"],
        [8.0, 12.0, 4.0, 5.0, 5.0],
    )
    candidate["segments"][1]["text"] = (
        "所有的把那个爱换成你自己的名字读一遍，我告诉你什么都没总"
        "我还想有几句话想说，就是我还想了解一下，就是我们之前有的同工被。"
    )
    candidate["segments"][1]["speaker_votes"] = {"SPEAKER_A": 10.0}
    candidate["segments"][1]["speaker_confidence"] = 1.0

    corrected = _split_handoff_segments(candidate)

    assert corrected["handoff_split_count"] == 0
    assert [seg["speaker"] for seg in corrected["segments"]] == [
        "SPEAKER_A",
        "SPEAKER_A",
        "SPEAKER_D",
        "SPEAKER_D",
        "SPEAKER_B",
    ]
    assert any("声纹护栏" in item["reason"] for item in corrected.get("review_segments", []))


def test_handoff_split_marks_strong_question_handoff_for_review_even_with_strong_pre_handoff_votes():
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A", "SPEAKER_A", "SPEAKER_D", "SPEAKER_D", "SPEAKER_B"],
        [8.0, 12.0, 8.0, 7.0, 5.0],
    )
    candidate["segments"][1]["text"] = (
        "所有的把那个爱换成你自己的名字读一遍，我告诉你什么都没总"
        "我还想有几句话想说，就是我还想了解一下，就是我们之前有的同工被。"
    )
    candidate["segments"][1]["speaker_votes"] = {"SPEAKER_A": 10.0}
    candidate["segments"][1]["speaker_confidence"] = 1.0
    for idx in [0, 1]:
        candidate["segments"][idx]["voice_pitch_hz"] = 235.0
        candidate["segments"][idx]["voice_pitch_confidence"] = 0.9
    for idx in [2, 3]:
        candidate["segments"][idx]["voice_pitch_hz"] = 125.0
        candidate["segments"][idx]["voice_pitch_confidence"] = 0.9
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 4))

    corrected = _split_handoff_segments(candidate)

    assert corrected["segments"][1]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][2]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][2]["speaker_handoff_review"] is True
    assert corrected["handoff_split_count"] == 1
    assert any("待确认" in item["reason"] for item in corrected.get("review_segments", []))


def test_handoff_split_marks_bridge_for_review_when_voice_still_matches_current_speaker():
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A", "SPEAKER_A", "SPEAKER_A", "SPEAKER_D", "SPEAKER_D", "SPEAKER_B"],
        [8.0, 12.0, 3.0, 8.0, 7.0, 5.0],
    )
    candidate["segments"][1]["text"] = (
        "所有的把那个爱换成你自己的名字读一遍，我告诉你什么都没总"
        "我还想有几句话想说，就是我还想了解一下，就是我们之前有的同工被。"
    )
    candidate["segments"][1]["speaker_votes"] = {"SPEAKER_A": 6.0, "SPEAKER_D": 4.0}
    candidate["segments"][1]["speaker_confidence"] = 0.6
    candidate["segments"][2]["speaker_votes"] = {"SPEAKER_A": 2.0}
    candidate["segments"][2]["speaker_confidence"] = 1.0
    for idx in [0, 1, 2]:
        candidate["segments"][idx]["voice_pitch_hz"] = 225.0
        candidate["segments"][idx]["voice_pitch_confidence"] = 0.9
    for idx in [3, 4]:
        candidate["segments"][idx]["voice_pitch_hz"] = 125.0
        candidate["segments"][idx]["voice_pitch_confidence"] = 0.9

    corrected = _split_handoff_segments(candidate)

    assert corrected["segments"][1]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][2]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][2]["speaker_handoff_review"] is True
    assert corrected["segments"][3]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][3]["speaker_handoff_review"] is True
    assert corrected["handoff_split_count"] == 1


def test_handoff_split_uses_voice_contrast_when_segment_acoustics_support_target():
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A", "SPEAKER_A", "SPEAKER_A", "SPEAKER_D", "SPEAKER_D", "SPEAKER_B"],
        [8.0, 12.0, 3.0, 8.0, 7.0, 5.0],
    )
    candidate["segments"][1]["text"] = (
        "所有的把那个爱换成你自己的名字读一遍，我告诉你什么都没总"
        "我还想有几句话想说，就是我还想了解一下，就是我们之前有的同工被。"
    )
    candidate["segments"][1]["speaker_votes"] = {"SPEAKER_A": 3.0, "SPEAKER_D": 7.0}
    candidate["segments"][1]["speaker_confidence"] = 0.7
    candidate["segments"][2]["speaker_votes"] = {"SPEAKER_D": 2.0}
    candidate["segments"][2]["speaker_confidence"] = 0.7
    for idx in [0]:
        candidate["segments"][idx]["voice_pitch_hz"] = 225.0
        candidate["segments"][idx]["voice_pitch_confidence"] = 0.9
    for idx in [1, 2, 3, 4]:
        candidate["segments"][idx]["voice_pitch_hz"] = 125.0
        candidate["segments"][idx]["voice_pitch_confidence"] = 0.9

    corrected = _split_handoff_segments(candidate)

    assert corrected["handoff_split_count"] == 1
    assert corrected["segments"][1]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][2]["speaker"] == "SPEAKER_D"
    assert corrected["segments"][3]["speaker"] == "SPEAKER_D"
    assert corrected["segments"][3]["speaker_handoff_bridge"] is True


def test_handoff_split_marks_strong_question_handoff_for_review_when_segment_votes_are_smeared():
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A", "SPEAKER_A", "SPEAKER_A", "SPEAKER_D", "SPEAKER_D", "SPEAKER_B"],
        [8.0, 12.0, 3.0, 8.0, 7.0, 5.0],
    )
    candidate["segments"][1]["text"] = (
        "所有的把那个爱换成你自己的名字读一遍，我告诉你什么都没总"
        "我还想有几句话想说，就是我还想了解一下，就是我们之前有的同工被。"
    )
    candidate["segments"][1]["speaker_votes"] = {"SPEAKER_A": 10.0}
    candidate["segments"][1]["speaker_confidence"] = 1.0
    candidate["segments"][2]["speaker_votes"] = {"SPEAKER_A": 2.0}
    candidate["segments"][2]["speaker_confidence"] = 1.0
    for idx in [0, 1, 2]:
        candidate["segments"][idx]["voice_pitch_hz"] = 225.0
        candidate["segments"][idx]["voice_pitch_confidence"] = 0.9
    for idx in [3, 4]:
        candidate["segments"][idx]["voice_pitch_hz"] = 125.0
        candidate["segments"][idx]["voice_pitch_confidence"] = 0.9

    corrected = _split_handoff_segments(candidate)

    assert corrected["handoff_split_count"] == 1
    assert corrected["segments"][1]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][2]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][2]["speaker_handoff_review"] is True
    assert corrected["segments"][3]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][3]["speaker_handoff_review"] is True


def test_handoff_voice_guard_repairs_split_that_conflicts_with_votes_and_voice():
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A", "SPEAKER_D", "SPEAKER_D", "SPEAKER_D", "SPEAKER_B"],
        [8.0, 6.0, 4.0, 8.0, 5.0],
    )
    candidate["segments"][0]["voice_pitch_hz"] = 235.0
    candidate["segments"][0]["voice_pitch_confidence"] = 0.9
    candidate["segments"][1]["voice_pitch_hz"] = 240.0
    candidate["segments"][1]["voice_pitch_confidence"] = 0.9
    candidate["segments"][1]["speaker_votes"] = {"SPEAKER_A": 8.0}
    candidate["segments"][1]["speaker_handoff_split"] = True
    for idx in [2, 3]:
        candidate["segments"][idx]["voice_pitch_hz"] = 125.0
        candidate["segments"][idx]["voice_pitch_confidence"] = 0.9
        candidate["segments"][idx]["speaker_votes"] = {"SPEAKER_D": 8.0}
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 4))

    corrected = _repair_handoff_voice_guard_assignments(candidate)

    assert corrected["segments"][1]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][1]["speaker_handoff_voice_guard_repaired"] is True
    assert corrected["handoff_voice_guard_distribution"] == {"SPEAKER_D": {"SPEAKER_A": 1}}
    assert any("声纹护栏" in item["reason"] for item in corrected["review_segments"])


def test_resegment_mixed_speaker_segment_uses_short_window_timeline():
    candidate = _candidate_from_labels(3, ["SPEAKER_A", "SPEAKER_A", "SPEAKER_C"], [4.0, 10.0, 4.0])
    candidate["segments"][1]["text"] = "前半段是第一个人继续说明，后半段是第二个人开始回应这个问题。"
    candidate["segments"][1]["speaker"] = "SPEAKER_A"
    candidate["segments"][1]["speaker_subsegments"] = [
        {"start": 4.2, "end": 9.2, "speaker": "SPEAKER_A", "duration": 5.0},
        {"start": 9.2, "end": 14.2, "speaker": "SPEAKER_B", "duration": 5.0},
    ]
    candidate["segments"][1]["speaker_votes"] = {"SPEAKER_A": 5.0, "SPEAKER_B": 5.0}
    candidate["segments"][1]["speaker_change_points"] = [9.2]
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 3))

    corrected = _resegment_mixed_speaker_segments(candidate)

    assert corrected["resegmentation_count"] == 1
    assert len(corrected["segments"]) == 4
    assert corrected["segments"][1]["speaker"] == "SPEAKER_A"
    assert corrected["segments"][2]["speaker"] == "SPEAKER_B"
    assert corrected["segments"][1]["speaker_resegmented"] is True
    assert corrected["segments"][2]["speaker_resegmented"] is True
    assert "短声纹时间线" in corrected["resegmentation_reason"]


def test_resegment_mixed_speaker_segment_handles_overlapping_short_windows():
    candidate = _candidate_from_labels(3, ["SPEAKER_B"], [11.44])
    candidate["segments"][0]["start"] = 1811.091
    candidate["segments"][0]["end"] = 1822.531
    candidate["segments"][0]["text"] = (
        "对，是吧，如果是蓝英和金子，按照沟端的一个规定，"
        "不人性化处理啊，不人性化处理的话，是不是他俩早就应该被移出去了，但是没有。"
    )
    candidate["segments"][0]["speaker"] = "SPEAKER_D"
    candidate["segments"][0]["speaker_subsegments"] = [
        {"start": 1811.091, "end": 1811.346, "speaker": "SPEAKER_D", "duration": 0.254},
        {"start": 1811.091, "end": 1811.946, "speaker": "SPEAKER_D", "duration": 0.854},
        {"start": 1811.091, "end": 1812.546, "speaker": "SPEAKER_D", "duration": 1.454},
        {"start": 1811.646, "end": 1813.146, "speaker": "SPEAKER_D", "duration": 1.5},
        {"start": 1812.246, "end": 1813.746, "speaker": "SPEAKER_D", "duration": 1.5},
        {"start": 1812.846, "end": 1814.346, "speaker": "SPEAKER_D", "duration": 1.5},
        {"start": 1813.446, "end": 1814.946, "speaker": "SPEAKER_D", "duration": 1.5},
        {"start": 1814.046, "end": 1815.546, "speaker": "SPEAKER_D", "duration": 1.5},
        {"start": 1814.646, "end": 1816.146, "speaker": "SPEAKER_D", "duration": 1.5},
        {"start": 1815.246, "end": 1816.746, "speaker": "SPEAKER_D", "duration": 1.5},
        {"start": 1815.846, "end": 1817.346, "speaker": "SPEAKER_D", "duration": 1.5},
        {"start": 1816.446, "end": 1817.946, "speaker": "SPEAKER_B", "duration": 1.5},
        {"start": 1817.046, "end": 1818.546, "speaker": "SPEAKER_B", "duration": 1.5},
        {"start": 1817.646, "end": 1819.146, "speaker": "SPEAKER_B", "duration": 1.5},
        {"start": 1818.246, "end": 1819.746, "speaker": "SPEAKER_B", "duration": 1.5},
        {"start": 1818.846, "end": 1820.346, "speaker": "SPEAKER_B", "duration": 1.5},
        {"start": 1819.446, "end": 1820.946, "speaker": "SPEAKER_B", "duration": 1.5},
        {"start": 1820.046, "end": 1821.546, "speaker": "SPEAKER_B", "duration": 1.5},
        {"start": 1820.646, "end": 1822.146, "speaker": "SPEAKER_B", "duration": 1.5},
        {"start": 1821.246, "end": 1822.531, "speaker": "SPEAKER_B", "duration": 1.286},
    ]
    candidate["segments"][0]["speaker_votes"] = {"SPEAKER_D": 14.562, "SPEAKER_B": 14.057}
    candidate["segments"][0]["speaker_overlap_risk"] = True
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 3))

    corrected = _resegment_mixed_speaker_segments(candidate)

    assert corrected["resegmentation_count"] == 0
    assert len(corrected["segments"]) == 1
    assert corrected["segments"][0]["speaker"] == "SPEAKER_D"
    assert corrected["segments"][0]["speaker_resegmentation_review"] is True
    assert any("证据重叠/接近" in item["reason"] for item in corrected["review_segments"])


def test_resegment_mixed_speaker_segment_reviews_unsafe_short_mix():
    candidate = _candidate_from_labels(3, ["SPEAKER_A", "SPEAKER_A", "SPEAKER_C"], [4.0, 4.0, 4.0])
    candidate["segments"][1]["text"] = "短句里疑似有人插话。"
    candidate["segments"][1]["speaker_subsegments"] = [
        {"start": 4.2, "end": 5.2, "speaker": "SPEAKER_A", "duration": 1.0},
        {"start": 5.2, "end": 6.0, "speaker": "SPEAKER_B", "duration": 0.8},
        {"start": 6.0, "end": 8.2, "speaker": "SPEAKER_A", "duration": 2.2},
    ]
    candidate["segments"][1]["speaker_overlap_risk"] = True
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 3))

    corrected = _resegment_mixed_speaker_segments(candidate)

    assert corrected["resegmentation_count"] == 0
    assert len(corrected["segments"]) == 3
    assert any("短声纹窗" in item["reason"] for item in corrected["review_segments"])


def test_alternating_smoothing_keeps_single_coherent_minor_speaker_turn():
    labels = (
        ["SPEAKER_A"] * 8
        + ["SPEAKER_B"] * 8
        + ["SPEAKER_D"] * 8
        + ["SPEAKER_B"] * 8
        + ["SPEAKER_C"] * 6
    )
    candidate = _candidate_from_labels(4, labels, [4.0] * len(labels))

    corrected = _smooth_alternating_local_speaker_leakage(candidate)

    assert [seg["speaker"] for seg in corrected["segments"]] == labels
    assert corrected["local_leakage_distribution"] == {}


def test_short_filler_sandwiched_speaker_is_treated_as_fragmented():
    candidate = _candidate(4, [
        {"speaker": "SPEAKER_A", "segments": 127, "duration_s": 414.8, "turns": 38, "stable_turns": 26},
        {"speaker": "SPEAKER_B", "segments": 87, "duration_s": 259.3, "turns": 23, "stable_turns": 13},
        {
            "speaker": "SPEAKER_C",
            "segments": 22,
            "duration_s": 33.2,
            "turns": 11,
            "stable_turns": 6,
            "short_ratio": 0.55,
            "filler_ratio": 0.55,
            "sandwiched_ratio": 0.23,
        },
        {"speaker": "SPEAKER_D", "segments": 33, "duration_s": 88.3, "turns": 14, "stable_turns": 9},
    ])

    assert candidate["fragmented_speakers"] == 1
    assert candidate["fragile_speakers"] == ["SPEAKER_C"]
    assert candidate["mergeable_speakers"] == ["SPEAKER_C"]
    assert _should_auto_merge_in_diarize(candidate)


def test_weak_but_coherent_third_speaker_is_not_auto_mergeable():
    candidate = _candidate(3, [
        {"speaker": "SPEAKER_A", "segments": 80, "duration_s": 180.0, "turns": 20, "stable_turns": 14},
        {"speaker": "SPEAKER_B", "segments": 50, "duration_s": 120.0, "turns": 16, "stable_turns": 10},
        {
            "speaker": "SPEAKER_C",
            "segments": 7,
            "duration_s": 28.0,
            "turns": 2,
            "stable_turns": 1,
            "short_ratio": 0.0,
            "filler_ratio": 0.0,
            "sandwiched_ratio": 0.0,
        },
    ])

    assert candidate["weak_speakers"] == 1
    assert candidate["fragile_speakers"] == ["SPEAKER_C"]
    assert candidate["mergeable_speakers"] == []
    assert not _should_auto_merge_in_diarize(candidate)


def test_low_duration_fragment_should_not_make_five_person_candidate_win():
    four = _candidate(4, [
        {"speaker": "SPEAKER_A", "segments": 128, "duration_s": 415.1, "turns": 38, "stable_turns": 26},
        {"speaker": "SPEAKER_B", "segments": 102, "duration_s": 280.6, "turns": 24, "stable_turns": 15},
        {"speaker": "SPEAKER_D", "segments": 39, "duration_s": 99.9, "turns": 12, "stable_turns": 7},
    ])
    five = _candidate(5, [
        {"speaker": "SPEAKER_A", "segments": 127, "duration_s": 414.8, "turns": 38, "stable_turns": 26},
        {"speaker": "SPEAKER_B", "segments": 93, "duration_s": 271.0, "turns": 23, "stable_turns": 14},
        {"speaker": "SPEAKER_D", "segments": 37, "duration_s": 96.3, "turns": 12, "stable_turns": 7},
        {
            "speaker": "SPEAKER_E",
            "segments": 12,
            "duration_s": 13.5,
            "turns": 7,
            "stable_turns": 2,
            "short_ratio": 0.42,
            "filler_ratio": 0.17,
            "sandwiched_ratio": 0.17,
        },
    ])

    best = _choose_diarization_candidate([four, five])

    assert five["fragmented_speakers"] == 1
    assert "SPEAKER_E" in five["mergeable_speakers"]
    assert best is four


def test_minor_extra_split_does_not_beat_clean_two_person_candidate():
    two = _candidate(2, [
        {"speaker": "SPEAKER_A", "segments": 90, "duration_s": 151.5, "turns": 22, "stable_turns": 15},
        {"speaker": "SPEAKER_B", "segments": 46, "duration_s": 78.5, "turns": 19, "stable_turns": 9},
    ])
    three = _candidate(4, [
        {"speaker": "SPEAKER_A", "segments": 87, "duration_s": 145.8, "turns": 21, "stable_turns": 15},
        {"speaker": "SPEAKER_B", "segments": 36, "duration_s": 62.6, "turns": 15, "stable_turns": 8},
        {"speaker": "SPEAKER_C", "segments": 13, "duration_s": 21.5, "turns": 6, "stable_turns": 2},
    ])
    two["score"] = 15.0
    three["score"] = 16.2

    best = _choose_diarization_candidate([two, three])

    assert best is two
    assert "低占比小簇" in two["model_guard_reason"]


def test_review_only_micro_cluster_does_not_raise_stable_four_person_count():
    four = _candidate_from_labels(
        4,
        ["SPEAKER_A"] * 734
        + ["SPEAKER_B"] * 456
        + ["SPEAKER_D"] * 56
        + ["SPEAKER_E"] * 50,
        [1.8] * 734 + [2.0] * 456 + [1.9] * 56 + [1.9] * 50,
    )
    five = _candidate_from_labels(
        6,
        ["SPEAKER_A"] * 734
        + ["SPEAKER_B"] * 456
        + ["SPEAKER_C"] * 13
        + ["SPEAKER_D"] * 56
        + ["SPEAKER_E"] * 50,
        [1.8] * 734 + [2.0] * 456 + [1.95] * 13 + [1.9] * 56 + [1.9] * 50,
    )
    four["score"] = 27.0
    four["stable_speakers"] = 4
    four["weak_speakers"] = 0
    four["tiny_speakers"] = 0
    four["fragmented_speakers"] = 0
    four["marginal_speakers"] = 0
    five["score"] = 24.3
    five["stable_speakers"] = 5
    five["weak_speakers"] = 0
    five["tiny_speakers"] = 0
    five["fragmented_speakers"] = 0
    five["marginal_speakers"] = 1
    five["fragile_speakers"] = ["SPEAKER_C"]
    five["mergeable_speakers"] = ["SPEAKER_C"]

    best = _choose_diarization_candidate([four, five])

    assert best is four
    assert not _candidate_has_meaningful_refinement(four, five)


def test_five_person_weak_tail_does_not_beat_stable_four_person_shape():
    four = _candidate(4, [
        {"speaker": "SPEAKER_A", "segments": 126, "duration_s": 910.6, "turns": 8, "stable_turns": 7},
        {"speaker": "SPEAKER_B", "segments": 20, "duration_s": 153.6, "turns": 4, "stable_turns": 4},
        {"speaker": "SPEAKER_C", "segments": 26, "duration_s": 195.7, "turns": 7, "stable_turns": 7},
        {"speaker": "SPEAKER_D", "segments": 5, "duration_s": 32.8, "turns": 1, "stable_turns": 1},
    ])
    five = _candidate(5, [
        {"speaker": "SPEAKER_A", "segments": 117, "duration_s": 836.9, "turns": 9, "stable_turns": 8},
        {"speaker": "SPEAKER_B", "segments": 20, "duration_s": 153.6, "turns": 4, "stable_turns": 4},
        {"speaker": "SPEAKER_C", "segments": 25, "duration_s": 183.3, "turns": 6, "stable_turns": 6},
        {"speaker": "SPEAKER_D", "segments": 10, "duration_s": 86.1, "turns": 3, "stable_turns": 3},
        {"speaker": "SPEAKER_E", "segments": 5, "duration_s": 32.8, "turns": 1, "stable_turns": 1},
    ])

    best = _choose_diarization_candidate([four, five])

    assert five["score"] > four["score"]
    assert _higher_count_has_weak_tail_over_split(four, five)
    assert best is four


def test_meaningful_low_frequency_participant_promotes_only_one_count():
    four = _candidate(4, [
        {"speaker": "SPEAKER_A", "segments": 25, "duration_s": 250.0, "turns": 8, "stable_turns": 7},
        {"speaker": "SPEAKER_B", "segments": 20, "duration_s": 200.0, "turns": 7, "stable_turns": 6},
        {"speaker": "SPEAKER_C", "segments": 18, "duration_s": 180.0, "turns": 6, "stable_turns": 5},
        {"speaker": "SPEAKER_D", "segments": 5, "duration_s": 50.0, "turns": 1, "stable_turns": 1},
    ])
    five = _candidate(5, [
        {"speaker": "SPEAKER_A", "segments": 20, "duration_s": 200.0, "turns": 7, "stable_turns": 6},
        {"speaker": "SPEAKER_E", "segments": 5, "duration_s": 50.0, "turns": 2, "stable_turns": 2},
        {"speaker": "SPEAKER_B", "segments": 20, "duration_s": 200.0, "turns": 7, "stable_turns": 6},
        {"speaker": "SPEAKER_C", "segments": 18, "duration_s": 180.0, "turns": 6, "stable_turns": 5},
        {"speaker": "SPEAKER_D", "segments": 5, "duration_s": 50.0, "turns": 1, "stable_turns": 1},
    ])
    six = _candidate(6, [
        {"speaker": "SPEAKER_A", "segments": 15, "duration_s": 150.0, "turns": 6, "stable_turns": 5},
        {"speaker": "SPEAKER_F", "segments": 5, "duration_s": 50.0, "turns": 2, "stable_turns": 2},
        {"speaker": "SPEAKER_E", "segments": 5, "duration_s": 50.0, "turns": 2, "stable_turns": 2},
        {"speaker": "SPEAKER_B", "segments": 20, "duration_s": 200.0, "turns": 7, "stable_turns": 6},
        {"speaker": "SPEAKER_C", "segments": 18, "duration_s": 180.0, "turns": 6, "stable_turns": 5},
        {"speaker": "SPEAKER_D", "segments": 5, "duration_s": 50.0, "turns": 1, "stable_turns": 1},
    ])
    for candidate in (four, five, six):
        for segment in candidate["segments"]:
            segment["end"] = segment["start"] + 10.0
        candidate["severe_mixed_voice_speakers"] = ["SPEAKER_A"]
    four.update({
        "score": -3.36,
        "stable_speakers": 3,
        "weak_speakers": 1,
        "tiny_speakers": 0,
        "fragmented_speakers": 0,
        "marginal_speakers": 1,
        "fragile_speakers": ["SPEAKER_D"],
    })
    five.update({
        "score": -6.66,
        "stable_speakers": 3,
        "weak_speakers": 2,
        "tiny_speakers": 0,
        "fragmented_speakers": 0,
        "marginal_speakers": 2,
        "fragile_speakers": ["SPEAKER_D", "SPEAKER_E"],
    })
    six.update({
        "score": -7.0,
        "stable_speakers": 3,
        "weak_speakers": 3,
        "tiny_speakers": 0,
        "fragmented_speakers": 0,
        "marginal_speakers": 3,
        "fragile_speakers": ["SPEAKER_D", "SPEAKER_E", "SPEAKER_F"],
    })

    assert _candidate_has_meaningful_refinement(four, five)
    assert _candidate_has_meaningful_refinement(five, six)

    best = _choose_diarization_candidate([four, five, six])

    assert best is five


def test_meaningful_long_segment_participant_is_not_blocked_by_parent_segment_count():
    lower_labels = [
        "SPEAKER_A", "SPEAKER_A",
        "SPEAKER_A", "SPEAKER_A",
        "SPEAKER_B", "SPEAKER_B", "SPEAKER_B", "SPEAKER_B",
        "SPEAKER_B", "SPEAKER_B", "SPEAKER_B",
        "SPEAKER_B", "SPEAKER_B", "SPEAKER_B", "SPEAKER_B",
        "SPEAKER_B", "SPEAKER_B", "SPEAKER_B", "SPEAKER_B",
        "SPEAKER_C", "SPEAKER_C", "SPEAKER_C", "SPEAKER_C",
        "SPEAKER_D", "SPEAKER_D", "SPEAKER_D", "SPEAKER_D",
    ]
    higher_labels = [
        "SPEAKER_A", "SPEAKER_A",
        "SPEAKER_E", "SPEAKER_E",
        "SPEAKER_E", "SPEAKER_E", "SPEAKER_B", "SPEAKER_B",
        "SPEAKER_E", "SPEAKER_B", "SPEAKER_B",
        "SPEAKER_B", "SPEAKER_B", "SPEAKER_B", "SPEAKER_B",
        "SPEAKER_B", "SPEAKER_B", "SPEAKER_B", "SPEAKER_B",
        "SPEAKER_C", "SPEAKER_C", "SPEAKER_C", "SPEAKER_C",
        "SPEAKER_D", "SPEAKER_D", "SPEAKER_D", "SPEAKER_D",
    ]
    durations = [
        10.0, 10.0,
        0.1, 0.1,
        20.0, 20.0, 15.0, 15.0,
        30.0, 15.0, 15.0,
        5.0, 5.0, 5.0, 5.0,
        5.0, 5.0, 5.0, 5.0,
        20.0, 20.0, 20.0, 20.0,
        15.0, 15.0, 15.0, 15.0,
    ]
    four = _candidate_from_labels(4, lower_labels, durations)
    five = _candidate_from_labels(5, higher_labels, durations)
    for candidate in (four, five):
        candidate["severe_mixed_voice_speakers"] = ["SPEAKER_A"]
        candidate.update({
            "tiny_speakers": 0,
            "fragmented_speakers": 0,
            "marginal_speakers": 0,
            "stable_speakers": 3,
        })
    four.update({
        "score": -3.0,
        "weak_speakers": 1,
        "fragile_speakers": ["SPEAKER_D"],
    })
    five.update({
        "score": -6.5,
        "weak_speakers": 2,
        "fragile_speakers": ["SPEAKER_D", "SPEAKER_E"],
    })

    added = next(
        speaker
        for speaker in five["summary"]["speakers"]
        if speaker["speaker"] == "SPEAKER_E"
    )
    assert added["segments"] == 5
    assert added["duration_s"] >= 70.0
    assert _candidate_has_meaningful_refinement(four, five)
    assert _choose_diarization_candidate([four, five]) is five


def test_close_higher_count_candidate_produces_review_segments():
    four = _candidate_from_labels(
        4,
        ["SPEAKER_A"] * 20 + ["SPEAKER_B"] * 20 + ["SPEAKER_C"] * 6 + ["SPEAKER_D"] * 6,
        [3.0] * 52,
    )
    five = _candidate_from_labels(
        5,
        ["SPEAKER_A"] * 20 + ["SPEAKER_B"] * 20 + ["SPEAKER_C"] * 6 + ["SPEAKER_D"] * 4 + ["SPEAKER_E"] * 2,
        [3.0] * 52,
    )
    four["score"] = 27.0
    five["score"] = 26.2

    review = _build_count_ambiguity_review_segments(four, [four, five])

    assert review
    assert any("候选人数接近" in item["reason"] for item in review)
    assert any(item["from_speaker"] == "SPEAKER_E" for item in review)


def test_low_confidence_fallback_samples_low_share_speakers():
    candidate = _candidate_from_labels(
        4,
        ["SPEAKER_A"] * 20 + ["SPEAKER_B"] * 20 + ["SPEAKER_C"] * 4 + ["SPEAKER_D"] * 4,
        [3.0] * 48,
    )

    review = _build_low_confidence_speaker_review_segments(candidate)

    assert review
    assert any(item["from_speaker"] == "SPEAKER_C" for item in review)
    assert any("低占比说话人" in item["reason"] for item in review)


def test_auto_diarize_keeps_low_confidence_labels_for_review(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    source_segments = [
        {"start": 0.0, "end": 1.0, "text": "第一段"},
        {"start": 1.0, "end": 2.0, "text": "第二段"},
    ]
    labeled_segments = [
        {**source_segments[0], "speaker": "SPEAKER_A"},
        {**source_segments[1], "speaker": "SPEAKER_B"},
    ]

    def fake_recommend(**_kwargs):
        return {
            "recommended_n_speakers": 2,
            "recommended_candidate_n_speakers": 2,
            "confidence": "low",
            "confidence_reason": "测试低置信",
            "score_gap_to_next": 0.1,
            "reason": "测试推荐",
            "candidates": [{
                "n_speakers": 2,
                "speakers": ["SPEAKER_A", "SPEAKER_B"],
                "segments": labeled_segments,
                "matched_profiles": {},
                "stats": {},
                "summary": {
                    "speakers": [
                        {"speaker": "SPEAKER_A"},
                        {"speaker": "SPEAKER_B"},
                    ],
                },
            }],
        }

    monkeypatch.setattr("scribe_py.ipc._recommend_diarization_candidates", fake_recommend)
    out = handle_diarize({
        "audio": str(audio),
        "segments": source_segments,
        "n_speakers": 0,
        "profiles": [],
    })

    assert out["speakers"] == ["SPEAKER_A", "SPEAKER_B"]
    assert [seg.get("speaker") for seg in out["segments"]] == ["SPEAKER_A", "SPEAKER_B"]
    assert out["stats"]["recommendation_confidence"] == "low"
    assert out["stats"]["risk_level"] == "high"


def test_historical_human_annotations_are_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCALSCRIBE_REUSE_HUMAN_ANNOTATIONS", raising=False)
    monkeypatch.setattr(
        "scribe_py.ipc._find_human_speaker_annotation_files",
        lambda _audio: (_ for _ in ()).throw(AssertionError("default path must not scan historical labels")),
    )
    candidate = _candidate_from_labels(2, ["SPEAKER_A", "SPEAKER_B"])

    out = _apply_historical_human_speaker_annotations(candidate, tmp_path / "meeting.mp3")

    assert out is candidate
    assert [segment["speaker"] for segment in out["segments"]] == ["SPEAKER_A", "SPEAKER_B"]


def test_historical_human_annotations_are_reused_as_sparse_anchors(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALSCRIBE_REUSE_HUMAN_ANNOTATIONS", "1")
    transcript_root = tmp_path / "Library/Application Support/LocalScribe/transcripts"
    annotated_dir = transcript_root / "标准录音 3-previous"
    annotated_dir.mkdir(parents=True)
    annotation = annotated_dir / "标准录音 3_清洗后标注.json"
    annotation.write_text(json.dumps([
        {
            "序号": 1,
            "时间": "31:32.31 - 31:38.99",
            "你的标注": "D",
            "文本": "会跟你说了，这个这这个我觉得我没有较真。",
        }
    ], ensure_ascii=False), encoding="utf-8")

    audio_dir = tmp_path / "runtime" / "标准录音 3" / "audio"
    audio_dir.mkdir(parents=True)
    audio = audio_dir / "标准录音 3.mp3"
    audio.write_bytes(b"fake")
    candidate = _candidate_from_labels(2, ["SPEAKER_B", "SPEAKER_B"])
    candidate["segments"][1]["start"] = 1892.31
    candidate["segments"][1]["end"] = 1898.99
    candidate["segments"][1]["text"] = "会跟你说了，这个这这个我觉得我没有较真。"
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])

    monkeypatch.setattr("scribe_py.ipc.Path.home", lambda: tmp_path)
    out = _apply_historical_human_speaker_annotations(candidate, audio)

    assert out["segments"][1]["speaker"] == "SPEAKER_D"
    assert out["segments"][1]["speaker_calibrated"] is True
    assert out["stats"]["historical_human_annotation_changed"] == 1


def test_historical_annotations_prefer_time_and_support_review_table_header(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALSCRIBE_REUSE_HUMAN_ANNOTATIONS", "1")
    transcript_root = tmp_path / "Library/Application Support/LocalScribe/transcripts"
    annotated_dir = transcript_root / "测试录音-previous"
    annotated_dir.mkdir(parents=True)
    annotation = annotated_dir / "测试录音_分人问题标注表.json"
    annotation.write_text(json.dumps([
        {
            "序号": 0,
            "时间": "00:10.00 - 00:12.00",
            "正确speaker(你填A/B/C/D或同上一人)": "d",
            "文本": "这个旧序号不准，应该按时间锚到第二段。",
        },
        {
            "序号": 0,
            "时间": "00:12.00 - 00:14.00",
            "正确speaker(你填A/B/C/D或同上一人)": "同上一人",
            "文本": "这一段沿用上一位人工标注的人。",
        },
    ], ensure_ascii=False), encoding="utf-8")

    audio_dir = tmp_path / "runtime" / "测试录音" / "audio"
    audio_dir.mkdir(parents=True)
    audio = audio_dir / "测试录音.mp3"
    audio.write_bytes(b"fake")
    candidate = _candidate_from_labels(2, ["SPEAKER_B", "SPEAKER_B", "SPEAKER_B"])
    for seg, start, end in zip(candidate["segments"], [0.0, 10.0, 12.0], [2.0, 12.0, 14.0]):
        seg["start"] = start
        seg["end"] = end
    candidate["segments"][1]["text"] = "这个旧序号不准，应该按时间锚到第二段。"
    candidate["segments"][2]["text"] = "这一段沿用上一位人工标注的人。"
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])

    monkeypatch.setattr("scribe_py.ipc.Path.home", lambda: tmp_path)
    out = _apply_historical_human_speaker_annotations(candidate, audio)

    assert out["segments"][0]["speaker"] == "SPEAKER_B"
    assert out["segments"][1]["speaker"] == "SPEAKER_D"
    assert out["segments"][2]["speaker"] == "SPEAKER_D"
    assert out["stats"]["historical_human_annotation_changed"] == 2


def test_historical_annotations_do_not_apply_when_text_disagrees(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALSCRIBE_REUSE_HUMAN_ANNOTATIONS", "1")
    transcript_root = tmp_path / "Library/Application Support/LocalScribe/transcripts"
    annotated_dir = transcript_root / "测试录音-previous"
    annotated_dir.mkdir(parents=True)
    (annotated_dir / "测试录音_清洗后标注.json").write_text(json.dumps([{
        "序号": 0,
        "时间": "00:10.00 - 00:12.00",
        "你的标注": "D",
        "文本": "旧时间轴里的另一句话",
    }], ensure_ascii=False), encoding="utf-8")
    audio_dir = tmp_path / "runtime" / "测试录音" / "audio"
    audio_dir.mkdir(parents=True)
    audio = audio_dir / "测试录音.mp3"
    audio.write_bytes(b"fake")
    candidate = _candidate_from_labels(2, ["SPEAKER_B", "SPEAKER_B"])
    candidate["segments"][1].update({"start": 10.0, "end": 12.0, "text": "当前版本完全不同的内容"})
    monkeypatch.setattr("scribe_py.ipc.Path.home", lambda: tmp_path)

    out = _apply_historical_human_speaker_annotations(candidate, audio)

    assert out["segments"][1]["speaker"] == "SPEAKER_B"
    assert not out["segments"][1].get("speaker_calibrated")


def test_short_low_confidence_sandwich_inherits_stable_neighbor():
    candidate = _candidate_from_labels(3, ["SPEAKER_C", "SPEAKER_B", "SPEAKER_C"], [8.0, 1.8, 8.0])
    candidate["segments"][1].update({
        "text": "嗯，远程写数据库。",
        "speaker_confidence": 0.74,
        "speaker_votes": {"SPEAKER_B": 3.5, "SPEAKER_C": 1.3},
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_smooth_short_sandwiched_segments"])
    out = ipc_module._smooth_short_sandwiched_segments(candidate)

    assert out["segments"][1]["speaker"] == "SPEAKER_C"
    assert out["segments"][1]["continuity_repaired"] is True


def test_short_high_confidence_sandwich_is_not_rewritten():
    candidate = _candidate_from_labels(3, ["SPEAKER_C", "SPEAKER_B", "SPEAKER_C"], [8.0, 1.8, 8.0])
    candidate["segments"][1].update({
        "text": "我不同意。",
        "speaker_confidence": 0.95,
        "speaker_votes": {"SPEAKER_B": 4.5, "SPEAKER_C": 0.5},
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_smooth_short_sandwiched_segments"])
    out = ipc_module._smooth_short_sandwiched_segments(candidate)

    assert out["segments"][1]["speaker"] == "SPEAKER_B"


def test_same_pitch_short_sandwich_merges_cluster_drift():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_A"] * 4 + ["SPEAKER_B"] + ["SPEAKER_A"] * 4,
        [1.0] * 9,
    )
    cursor = 0.0
    for segment in candidate["segments"]:
        segment["start"] = cursor
        segment["end"] = cursor + 1.0
        cursor += 1.0
    candidate["segments"][3].update({
        "voice_pitch_hz": 210.0,
        "voice_pitch_confidence": 0.80,
    })
    candidate["segments"][4].update({
        "text": "这边继续。",
        "speaker_confidence": 0.84,
        "voice_pitch_hz": 211.0,
        "voice_pitch_confidence": 0.45,
        "speaker_cue_embeddings": [{
            "embedding_scope": "exact_sync_cue",
            "score": 0.66,
        }],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_smooth_short_sandwiched_segments"])
    out = ipc_module._smooth_short_sandwiched_segments(candidate)

    assert out["segments"][4]["speaker"] == "SPEAKER_A"
    assert out["segments"][4]["continuity_repaired"] is True
    assert "同音色短句夹心平滑" in out["segments"][4]["speaker_review_reason"]


def test_short_overlap_micro_fragment_inherits_stable_neighbor_when_votes_are_close():
    candidate = _candidate_from_labels(
        3,
        ["SPEAKER_B", "SPEAKER_C", "SPEAKER_B"],
        [8.0, 1.489, 8.0],
    )
    candidate["segments"][1].update({
        "text": "大家免得大家都。",
        "speaker_confidence": 0.618,
        "speaker_votes": {"SPEAKER_B": 0.454, "SPEAKER_C": 0.736},
        "speaker_overlap_risk": True,
        "overlap_ratio": 0.2861,
        "speaker_change_points": [8.95],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_smooth_short_sandwiched_segments"])
    out = ipc_module._smooth_short_sandwiched_segments(candidate)

    assert out["segments"][1]["speaker"] == "SPEAKER_B"
    assert out["segments"][1]["continuity_repaired"] is True


def test_short_overlap_real_interjection_is_not_swallowed_when_source_votes_dominate():
    candidate = _candidate_from_labels(
        3,
        ["SPEAKER_B", "SPEAKER_C", "SPEAKER_B"],
        [8.0, 1.489, 8.0],
    )
    candidate["segments"][1].update({
        "text": "我不同意。",
        "speaker_confidence": 0.72,
        "speaker_votes": {"SPEAKER_B": 0.2, "SPEAKER_C": 1.2},
        "speaker_overlap_risk": True,
        "overlap_ratio": 0.20,
        "speaker_change_points": [8.95],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_smooth_short_sandwiched_segments"])
    out = ipc_module._smooth_short_sandwiched_segments(candidate)

    assert out["segments"][1]["speaker"] == "SPEAKER_C"


def test_balanced_low_overlap_sandwich_inherits_matching_stable_voice():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_A"],
        [8.0, 3.6, 8.0],
    )
    candidate["segments"][0].update({
        "speaker_confidence": 0.98,
        "voice_pitch_hz": 241.0,
        "voice_pitch_confidence": 1.0,
    })
    candidate["segments"][1].update({
        "text": "这是一句长度超过短句门槛的连续表达。",
        "speaker_confidence": 0.505,
        "speaker_votes": {"SPEAKER_A": 1.47, "SPEAKER_B": 1.50},
        "speaker_overlap_risk": True,
        "overlap_ratio": 0.0,
        "speaker_change_points": [8.7, 11.4],
        "voice_pitch_hz": 229.0,
        "voice_pitch_confidence": 0.8,
    })
    candidate["segments"][2].update({
        "speaker_confidence": 0.99,
        "voice_pitch_hz": 253.0,
        "voice_pitch_confidence": 1.0,
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_smooth_short_sandwiched_segments"])
    out = ipc_module._smooth_short_sandwiched_segments(candidate)

    assert out["segments"][1]["speaker"] == "SPEAKER_A"
    assert out["segments"][1]["continuity_repaired"] is True
    assert "近乎平票" in out["segments"][1]["speaker_review_reason"]


def test_embedding_ambiguous_sandwich_inherits_matching_voice_context():
    candidate = _candidate_from_labels(
        3,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_A"],
        [12.27, 4.14, 6.05],
    )
    candidate["segments"][0].update({
        "speaker_confidence": 0.95,
        "voice_pitch_hz": 230.3,
        "voice_pitch_confidence": 1.0,
    })
    candidate["segments"][1].update({
        "text": "这是同一位说话人连续表达的完整一句。",
        "speaker_confidence": 0.468,
        "speaker_votes": {
            "SPEAKER_A": 2.566,
            "SPEAKER_B": 4.828,
            "SPEAKER_C": 2.926,
        },
        "speaker_overlap_risk": True,
        "overlap_ratio": 0.0,
        "speaker_change_points": [12.5, 13.1, 13.7, 14.3, 14.9, 15.5],
        "voice_pitch_hz": 219.7,
        "voice_pitch_confidence": 1.0,
        "speaker_cue_embeddings": [
            {"speaker": "SPEAKER_B", "second_speaker": "SPEAKER_A", "score": 0.5415, "margin": 0.0322, "decision": "insufficient", "embedding_scope": "exact_sync_cue"},
            {"speaker": "SPEAKER_B", "second_speaker": "SPEAKER_A", "score": 0.5670, "margin": 0.0108, "decision": "insufficient", "embedding_scope": "sliding_window_weighted"},
        ],
    })
    candidate["segments"][2].update({
        "speaker_confidence": 0.525,
        "speaker_votes": {"SPEAKER_A": 7.948, "SPEAKER_B": 7.128},
        "voice_pitch_hz": 253.0,
        "voice_pitch_confidence": 1.0,
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_smooth_short_sandwiched_segments"])
    out = ipc_module._smooth_short_sandwiched_segments(candidate)

    assert out["segments"][1]["speaker"] == "SPEAKER_A"
    assert out["segments"][1]["continuity_repaired"] is True
    assert "精确声纹近乎平票" in out["segments"][1]["speaker_review_reason"]


def test_balanced_sandwich_keeps_distinct_voice_interjection():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_A"],
        [8.0, 3.6, 8.0],
    )
    candidate["segments"][0].update({
        "speaker_confidence": 0.98,
        "voice_pitch_hz": 120.0,
        "voice_pitch_confidence": 1.0,
    })
    candidate["segments"][1].update({
        "text": "这是另一位说话人的完整插话。",
        "speaker_confidence": 0.505,
        "speaker_votes": {"SPEAKER_A": 1.47, "SPEAKER_B": 1.50},
        "speaker_overlap_risk": True,
        "overlap_ratio": 0.0,
        "speaker_change_points": [8.7, 11.4],
        "voice_pitch_hz": 230.0,
        "voice_pitch_confidence": 1.0,
    })
    candidate["segments"][2].update({
        "speaker_confidence": 0.99,
        "voice_pitch_hz": 125.0,
        "voice_pitch_confidence": 1.0,
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_smooth_short_sandwiched_segments"])
    out = ipc_module._smooth_short_sandwiched_segments(candidate)

    assert out["segments"][1]["speaker"] == "SPEAKER_B"


def test_local_assignment_review_flags_sandwiched_speaker_jump_without_rewriting():
    candidate = _candidate_from_labels(
        3,
        ["SPEAKER_A"] * 6
        + ["SPEAKER_C"]
        + ["SPEAKER_A"] * 6
        + ["SPEAKER_B"] * 8,
        [3.0] * 21,
    )
    before = [seg["speaker"] for seg in candidate["segments"]]

    review = _build_local_assignment_review_segments(candidate)

    assert [seg["speaker"] for seg in candidate["segments"]] == before
    assert review
    assert any(item["from_speaker"] == "SPEAKER_C" and item["to_speaker"] == "SPEAKER_A" for item in review)
    assert any("局部夹心跳变" in item["reason"] for item in review)


def test_acoustic_resegmentation_does_not_hard_split_without_text_boundary():
    candidate = _candidate_from_labels(2, ["SPEAKER_A"], [10.0])
    candidate["segments"][0].update({
        "text": "这句话本身是一个连续表达不应该因为声纹窗口抖动就硬拆成两个人",
        "speaker_subsegments": [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_A", "duration": 4.0},
            {"start": 4.0, "end": 10.0, "speaker": "SPEAKER_B", "duration": 6.0},
        ],
    })

    out = _resegment_mixed_speaker_segments(candidate)

    assert len(out["segments"]) == 1
    assert out["segments"][0]["text"] == "这句话本身是一个连续表达不应该因为声纹窗口抖动就硬拆成两个人"
    assert any("文字无法安全切分" in item["reason"] for item in out.get("review_segments", []))


def test_acoustic_resegmentation_can_split_at_natural_text_boundary():
    candidate = _candidate_from_labels(2, ["SPEAKER_A"], [10.0])
    candidate["segments"][0].update({
        "text": "前半句已经把当前观点说完了，后半句确实换了一个人继续补充说明。",
        "speaker_subsegments": [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_A", "duration": 4.0},
            {"start": 4.0, "end": 10.0, "speaker": "SPEAKER_B", "duration": 6.0},
        ],
    })

    out = _resegment_mixed_speaker_segments(candidate)

    assert len(out["segments"]) == 2
    assert out["segments"][0]["speaker"] == "SPEAKER_A"
    assert out["segments"][1]["speaker"] == "SPEAKER_B"
    assert out["segments"][0]["text"].endswith("，")


def test_acoustic_resegmentation_preserves_precise_sync_cues():
    candidate = _candidate_from_labels(2, ["SPEAKER_A"], [10.0])
    candidate["segments"][0].update({
        "text": "前半句已经把当前观点说完了，后半句确实换了一个人继续补充说明。",
        "sync_cues": [
            {"start": 0.0, "end": 4.0, "text": "前半句已经把当前观点说完了，"},
            {"start": 4.0, "end": 10.0, "text": "后半句确实换了一个人继续补充说明。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_A", "duration": 4.0},
            {"start": 4.0, "end": 10.0, "speaker": "SPEAKER_B", "duration": 6.0},
        ],
    })

    out = _resegment_mixed_speaker_segments(candidate)

    assert len(out["segments"]) == 1
    assert out["segments"][0]["sync_cues"] == candidate["segments"][0]["sync_cues"]
    assert any("精确同步时间戳" in item["reason"] for item in out.get("review_segments", []))


def _projected_cue_handoff_candidate(*, boundary_confidence: float = 0.82, change_point: float = 3.9) -> dict:
    left_text = "第一位说话人把问题完整说完，"
    right_text = "第二位说话人开始回答。"
    cues = [
        {"start": 0.0, "end": 4.0, "text": left_text},
        {"start": 4.0, "end": 8.0, "text": right_text},
    ]
    segment = {
        "start": 0.0,
        "end": 8.0,
        "text": left_text + right_text,
        "speaker": "SPEAKER_A",
        "speaker_confidence": 0.76,
        "sync_cues": cues,
        "speaker_cues": [
            {
                **cues[0],
                "speaker": "SPEAKER_A",
                "confidence": boundary_confidence,
                "source": "campp_sync_cue_embedding",
            },
            {
                **cues[1],
                "speaker": "SPEAKER_B",
                "confidence": boundary_confidence,
                "source": "campp_sync_cue_embedding",
            },
        ],
        "speaker_cue_mode": "campp_sync_cue_embedding",
        "speaker_change_points": [change_point],
        "overlap_ratio": 0.02,
    }
    candidate = {
        "n_speakers": 2,
        "speakers": ["SPEAKER_A", "SPEAKER_B"],
        "segments": [segment],
        "summary": __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary([segment]),
    }
    candidate.update(_score_diarization_candidate(candidate, 2))
    return candidate


def test_projected_cue_handoff_is_materialized_at_existing_cue_boundary():
    candidate = _projected_cue_handoff_candidate()
    original = json.loads(json.dumps(candidate["segments"]))

    out = _materialize_projected_speaker_handoffs(candidate)

    assert out["cue_handoff_split_count"] == 1
    assert out["resegmentation_count"] == 1
    assert out["segmentation_preserved"] is False
    assert len(out["segments"]) == 2
    assert [segment["speaker"] for segment in out["segments"]] == ["SPEAKER_A", "SPEAKER_B"]
    assert out["segments"][0]["end"] == out["segments"][1]["start"] == 4.0
    assert out["segments"][0]["sync_cues"] == original[0]["sync_cues"][:1]
    assert out["segments"][1]["sync_cues"] == original[0]["sync_cues"][1:]
    assert "".join(segment["text"] for segment in out["segments"]) == original[0]["text"]
    assert _preserves_transcript_partition(original, out["segments"])


def test_projected_cue_handoff_with_insufficient_boundary_evidence_stays_for_review():
    candidate = _projected_cue_handoff_candidate(boundary_confidence=0.69, change_point=2.0)
    original = json.loads(json.dumps(candidate["segments"]))

    out = _materialize_projected_speaker_handoffs(candidate)

    assert out["cue_handoff_split_count"] == 0
    assert out["segments"] == [{**original[0], "speaker_resegmentation_review": True}]
    assert any("证据不足" in item["reason"] for item in out.get("review_segments", []))
    assert _preserves_transcript_partition(original, out["segments"])


def test_transcript_partition_guard_rejects_text_cue_or_coverage_changes():
    candidate = _projected_cue_handoff_candidate()
    source = json.loads(json.dumps(candidate["segments"]))
    split = _materialize_projected_speaker_handoffs(candidate)["segments"]

    assert _preserves_transcript_partition(source, split)
    changed_text = json.loads(json.dumps(split))
    changed_text[1]["text"] += "错"
    assert not _preserves_transcript_partition(source, changed_text)
    changed_cue = json.loads(json.dumps(split))
    changed_cue[1]["sync_cues"][0]["start"] += 0.1
    assert not _preserves_transcript_partition(source, changed_cue)
    changed_coverage = json.loads(json.dumps(split))
    changed_coverage[1]["start"] += 0.1
    assert not _preserves_transcript_partition(source, changed_coverage)


def test_speaker_metadata_finalizer_rebuilds_from_frozen_asr_rows():
    source = [{
        "start": 0.0,
        "end": 2.0,
        "text": "冻结正文",
        "original_text": "冻结正文",
        "sync_cues": [{"start": 0.0, "end": 2.0, "text": "冻结正文"}],
        "speaker": "OLD",
        "speaker_cue_mode": "old_mode",
        "domain_field": {"keep": True},
    }]
    candidate = json.loads(json.dumps(source))
    candidate[0].update({
        "speaker": "SPEAKER_B",
        "speaker_confidence": 0.91,
        "speaker_cues": [{
            "cue_index": 0,
            "start": 0.0,
            "end": 2.0,
            "text": "冻结正文",
            "speaker": "SPEAKER_B",
            "confidence": 0.91,
            "source": "test",
        }],
        "speaker_cue_mode": "test_mode",
        "unexpected_diarization_field": "must-not-leak",
        "domain_field": {"keep": False},
    })

    finalized = _finalize_speaker_metadata_only(source, candidate)

    assert finalized is not None
    assert finalized[0]["speaker"] == "SPEAKER_B"
    assert finalized[0]["speaker_cue_mode"] == "test_mode"
    assert finalized[0]["domain_field"] == {"keep": True}
    assert "unexpected_diarization_field" not in finalized[0]
    assert finalized[0]["sync_cues"] == source[0]["sync_cues"]
    assert finalized[0]["sync_cues"] is not source[0]["sync_cues"]


def test_speaker_metadata_finalizer_rejects_frozen_geometry_changes():
    source = [{
        "start": 0.0,
        "end": 2.0,
        "text": "冻结正文",
        "sync_cues": [{"start": 0.0, "end": 2.0, "text": "冻结正文"}],
    }]
    for field, value in (
        ("start", 0.1),
        ("end", 2.1),
        ("text", "被修改"),
        ("sync_cues", [{"start": 0.0, "end": 1.9, "text": "冻结正文"}]),
    ):
        candidate = json.loads(json.dumps(source))
        candidate[0][field] = value
        candidate[0]["speaker"] = "SPEAKER_A"
        assert _finalize_speaker_metadata_only(source, candidate) is None


def test_handle_diarize_rejects_in_place_sync_cue_mutation(monkeypatch, tmp_path):
    _patch_diarization_postprocess_identity(monkeypatch)
    monkeypatch.setattr("scribe_py.ipc._project_speaker_cues", lambda candidate: candidate)
    monkeypatch.setattr(
        "scribe_py.ipc._repair_long_missing_speaker_cues",
        lambda candidate, _audio: candidate,
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    source = [{
        "start": 0.0,
        "end": 2.0,
        "text": "冻结正文",
        "sync_cues": [{"start": 0.0, "end": 2.0, "text": "冻结正文"}],
    }]
    original = json.loads(json.dumps(source))

    def mutating_diarize(*, segments, **_kwargs):
        segments[0]["sync_cues"][0]["end"] = 1.5
        return SimpleNamespace(
            segments=[{**segments[0], "speaker": "SPEAKER_A"}],
            speakers=["SPEAKER_A"],
            cluster_count=1,
            matched_profiles={},
            stats={"engine": "senko", "embeddings": 8},
        )

    monkeypatch.setattr("scribe_py.diarizers.diarize", mutating_diarize)

    out = handle_diarize({
        "audio": str(audio),
        "segments": source,
        "n_speakers": 1,
        "profiles": [],
    })

    assert source == original
    assert out["segments"] == original
    assert out["stats"]["status"] == "error"
    assert out["stats"]["applied"] is False
    assert "几何" in out["stats"]["failure_reason"]


def test_handle_diarize_incomplete_result_returns_frozen_input(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    source = [{
        "start": 0.0,
        "end": 2.0,
        "text": "冻结正文",
        "sync_cues": [{"start": 0.0, "end": 2.0, "text": "冻结正文"}],
    }]
    original = json.loads(json.dumps(source))

    def mutating_incomplete_diarize(*, segments, **_kwargs):
        segments[0]["text"] = "被污染"
        segments[0]["sync_cues"][0]["end"] = 1.5
        return SimpleNamespace(
            segments=[{**segments[0], "speaker": ""}],
            speakers=[],
            cluster_count=0,
            matched_profiles={},
            stats={"engine": "senko", "embeddings": 8},
        )

    monkeypatch.setattr("scribe_py.diarizers.diarize", mutating_incomplete_diarize)

    out = handle_diarize({
        "audio": str(audio),
        "segments": source,
        "n_speakers": 1,
        "profiles": [],
    })

    assert source == original
    assert out["segments"] == original
    assert out["stats"]["applied"] is False


def test_recommendation_review_annotation_cannot_change_frozen_geometry(monkeypatch, tmp_path):
    _patch_diarization_postprocess_identity(monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    source = [
        {"start": 0.0, "end": 1.0, "text": "第一段"},
        {"start": 1.0, "end": 2.0, "text": "第二段"},
    ]
    monkeypatch.setattr(
        "scribe_py.diarizers.diarize",
        lambda **_kwargs: SimpleNamespace(
            segments=[
                {**source[0], "speaker": "SPEAKER_A"},
                {**source[1], "speaker": "SPEAKER_B"},
            ],
            speakers=["SPEAKER_A", "SPEAKER_B"],
            cluster_count=2,
            matched_profiles={},
            stats={"engine": "senko", "embeddings": 99},
        ),
    )
    monkeypatch.setattr(
        "scribe_py.ipc._repair_long_missing_speaker_cues",
        lambda candidate, _audio: candidate,
    )

    def mutating_annotation(candidate):
        candidate["segments"][0]["text"] = "被污染"
        return candidate

    monkeypatch.setattr(
        "scribe_py.ipc._annotate_segments_with_speaker_reviews",
        mutating_annotation,
    )

    out = handle_recommend_diarization({
        "audio": str(audio),
        "segments": source,
        "min_speakers": 2,
        "max_speakers": 2,
        "profiles": [],
    })

    assert out["candidates"][0]["segments"] == [
        {**source[0], "speaker": "SPEAKER_A"},
        {**source[1], "speaker": "SPEAKER_B"},
    ]
    assert any("待确认标记" in item["error"] for item in out["errors"])


def test_transcript_partition_guard_rejects_legacy_or_cross_piece_cue_splits():
    legacy_source = [{"start": 0.0, "end": 4.0, "text": "历史原文"}]
    legacy_split = [
        {"start": 0.0, "end": 2.0, "text": "历史"},
        {"start": 2.0, "end": 4.0, "text": "原文"},
    ]
    assert not _preserves_transcript_partition(legacy_source, legacy_split)

    candidate = _projected_cue_handoff_candidate()
    source = json.loads(json.dumps(candidate["segments"]))
    split = _materialize_projected_speaker_handoffs(candidate)["segments"]
    split[0]["sync_cues"] = source[0]["sync_cues"]
    split[1]["sync_cues"] = []
    assert not _preserves_transcript_partition(source, split)


def test_low_risk_stable_segment_is_not_split_by_cue_materializer():
    candidate = _candidate_from_labels(2, ["SPEAKER_A"], [8.0])
    segment = candidate["segments"][0]
    segment.update({
        "text": "同一位说话人的稳定发言保持原样。",
        "sync_cues": [
            {"start": 0.0, "end": 4.0, "text": "同一位说话人的"},
            {"start": 4.0, "end": 8.0, "text": "稳定发言保持原样。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_A", "duration": 4.0},
            {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_A", "duration": 4.0},
        ],
    })
    original = json.loads(json.dumps(candidate["segments"]))

    out = _materialize_projected_speaker_handoffs(candidate)

    assert out["segments"] == original
    assert out["cue_handoff_split_count"] == 0
    assert out.get("review_segments") in (None, [])


def test_preserve_mode_keeps_safe_projected_cue_partition_logical(monkeypatch):
    candidate = _projected_cue_handoff_candidate()
    source = json.loads(json.dumps(candidate["segments"]))
    monkeypatch.setattr("scribe_py.ipc._project_speaker_cues", lambda value: value)
    for name in [
        "_repair_handoff_voice_guard_assignments",
        "_reassign_isolated_fragile_segments",
        "_smooth_short_sandwiched_segments",
        "_smooth_windowed_sandwiched_runs",
        "_smooth_alternating_local_speaker_leakage",
        "_repair_discourse_continuity_assignments",
        "_repair_voice_band_assignments",
    ]:
        monkeypatch.setattr(f"scribe_py.ipc.{name}", lambda value, *args: value)

    out = _postprocess_fixed_count_candidate(
        candidate,
        [candidate],
        requested_n=2,
        preserve_segmentation=True,
    )

    assert len(out["segments"]) == 1
    assert out["resegmentation_count"] == 0
    assert out["segmentation_preserved"] is True
    assert [cue["speaker"] for cue in out["segments"][0]["speaker_cues"]] == [
        "SPEAKER_A",
        "SPEAKER_B",
    ]
    assert _preserves_transcript_partition(source, out["segments"])


def test_auto_candidate_default_preserve_mode_keeps_safe_cue_handoff_logical(monkeypatch, tmp_path):
    template = _projected_cue_handoff_candidate()["segments"][0]
    source = [{
        key: value
        for key, value in template.items()
        if key in {"start", "end", "text", "sync_cues"}
    }]
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")

    monkeypatch.setattr(
        "scribe_py.diarizers.diarize",
        lambda **_kwargs: SimpleNamespace(
            segments=[{**source[0], "speaker": "SPEAKER_A", "speaker_confidence": 0.76}],
            speakers=["SPEAKER_A", "SPEAKER_B"],
            cluster_count=2,
            matched_profiles={},
            stats={"engine": "senko", "embeddings": 99},
        ),
    )

    def project(candidate):
        projected = {**candidate, "segments": [dict(segment) for segment in candidate["segments"]]}
        segment = projected["segments"][0]
        segment.update({
            "speaker_cues": json.loads(json.dumps(template["speaker_cues"])),
            "speaker_cue_mode": "campp_sync_cue_embedding",
            "speaker_change_points": [3.9],
            "overlap_ratio": 0.02,
        })
        return projected

    monkeypatch.setattr("scribe_py.ipc._project_speaker_cues", project)
    for name in [
        "_refine_conflicting_voice_bands_with_pyin",
        "_repair_handoff_voice_guard_assignments",
        "_reassign_isolated_fragile_segments",
        "_smooth_short_sandwiched_segments",
        "_smooth_windowed_sandwiched_runs",
        "_smooth_alternating_local_speaker_leakage",
        "_repair_discourse_continuity_assignments",
        "_repair_voice_band_assignments",
    ]:
        monkeypatch.setattr(f"scribe_py.ipc.{name}", lambda value, *args: value)

    out = _recommend_diarization_candidates(
        audio=audio,
        segments=source,
        profiles=[],
        min_speakers=2,
        max_speakers=2,
    )

    candidate = out["candidates"][0]
    assert len(candidate["segments"]) == 1
    assert candidate["resegmentation_count"] == 0
    assert candidate["segmentation_preserved"] is True
    assert [cue["speaker"] for cue in candidate["segments"][0]["speaker_cues"]] == [
        "SPEAKER_A",
        "SPEAKER_B",
    ]
    assert _preserves_transcript_partition(source, candidate["segments"])


def test_review_band_first_cue_projects_handoff_when_context_evidence_agrees():
    candidate = _candidate_from_labels(2, ["SPEAKER_D"], [16.444])
    segment = candidate["segments"][0]
    segment.update({
        "start": 189.586,
        "end": 206.03,
        "text": "双方从什么时候开始出，我2005年招这个孩子，女孩，我一直在基地。",
        "speaker": "SPEAKER_D",
        "speaker_confidence": 0.864,
        "overlap_ratio": 0.0,
        "speaker_change_points": [194.641],
        "sync_cues": [
            {"start": 189.586, "end": 194.231, "text": "双方从什么时候开始出，"},
            {"start": 194.231, "end": 197.908, "text": "我2005年招这个孩子，"},
            {"start": 197.908, "end": 198.592, "text": "女孩，"},
            {"start": 198.592, "end": 206.03, "text": "我一直在基地。"},
        ],
        "speaker_subsegments": [
            {"start": 189.586, "end": 193.358, "speaker": "SPEAKER_B"},
            {"start": 194.641, "end": 206.03, "speaker": "SPEAKER_D"},
        ],
        "speaker_cue_embeddings": [
            {
                "cue_index": 0,
                "decision": "review",
                "embedding_scope": "exact_sync_cue",
                "speaker": "SPEAKER_B",
                "score": 0.6933,
                "margin": 0.0496,
                "voice_coverage_ratio": 0.7249,
                "overlap_ratio": 0.0,
            },
            {
                "cue_index": 1,
                "decision": "assign",
                "embedding_scope": "exact_sync_cue",
                "speaker": "SPEAKER_D",
                "score": 0.7964,
                "margin": 0.1573,
                "voice_coverage_ratio": 0.8886,
                "overlap_ratio": 0.0,
            },
            *[
                {
                    "cue_index": index,
                    "decision": "assign",
                    "embedding_scope": "sliding_window_weighted",
                    "speaker": "SPEAKER_D",
                    "score": 0.86,
                    "margin": 0.15,
                    "voice_coverage_ratio": 1.0,
                    "overlap_ratio": 0.0,
                }
                for index in range(2, 4)
            ],
        ],
    })
    frozen = {
        key: json.loads(json.dumps(segment[key]))
        for key in ("start", "end", "text", "sync_cues")
    }

    out = _project_speaker_cues(candidate)
    projected = out["segments"][0]

    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_B",
        "SPEAKER_D",
        "SPEAKER_D",
        "SPEAKER_D",
    ]
    assert projected["speaker_cues"][0]["source"] == "campp_review_cue_context_handoff"
    assert {key: projected[key] for key in frozen} == frozen


def test_speaker_cues_project_internal_handoff_without_changing_transcript_geometry():
    candidate = _candidate_from_labels(2, ["SPEAKER_B"], [10.0])
    segment = candidate["segments"][0]
    segment.update({
        "text": "前半句由第一位说话人发言，后半句由第二位说话人回答。",
        "sync_cues": [
            {"start": 0.0, "end": 3.0, "text": "前半句由第一位说话人发言，"},
            {"start": 3.0, "end": 10.0, "text": "后半句由第二位说话人回答。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_D", "duration": 3.0},
            {"start": 3.0, "end": 10.0, "speaker": "SPEAKER_B", "duration": 7.0},
        ],
        "speaker_change_points": [3.0],
        "speaker_confidence": 0.60,
        "speaker_overlap_risk": True,
    })
    before = (segment["start"], segment["end"], segment["text"], list(segment["sync_cues"]))

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][0]
    assert (projected["start"], projected["end"], projected["text"], projected["sync_cues"]) == before
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == ["SPEAKER_D", "SPEAKER_B"]
    assert "".join(cue["text"] for cue in projected["speaker_cues"]) == projected["text"]
    assert out["speaker_cue_segment_count"] == 1


def test_speaker_cues_keep_segment_label_when_alternate_evidence_is_ambiguous():
    candidate = _candidate_from_labels(2, ["SPEAKER_B"], [6.0])
    candidate["segments"][0].update({
        "text": "这段短窗证据接近，不能强行制造段内换人。",
        "sync_cues": [
            {"start": 0.0, "end": 3.0, "text": "这段短窗证据接近，"},
            {"start": 3.0, "end": 6.0, "text": "不能强行制造段内换人。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 1.6, "speaker": "SPEAKER_D", "duration": 1.6},
            {"start": 0.0, "end": 1.4, "speaker": "SPEAKER_B", "duration": 1.4},
            {"start": 3.0, "end": 6.0, "speaker": "SPEAKER_B", "duration": 3.0},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert "speaker_cues" not in out["segments"][0]
    assert out["speaker_cue_segment_count"] == 0


def test_speaker_cues_do_not_override_stable_high_confidence_segment():
    candidate = _candidate_from_labels(2, ["SPEAKER_D"], [8.0])
    candidate["segments"][0].update({
        "text": "稳定整段结果不能被短窗边界拖尾改坏。",
        "speaker_confidence": 0.72,
        "sync_cues": [
            {"start": 0.0, "end": 3.0, "text": "稳定整段结果不能被"},
            {"start": 3.0, "end": 8.0, "text": "短窗边界拖尾改坏。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_A", "duration": 3.0},
            {"start": 3.0, "end": 8.0, "speaker": "SPEAKER_D", "duration": 5.0},
        ],
        "speaker_change_points": [3.0],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert out["segments"][0]["speaker"] == "SPEAKER_D"
    assert "speaker_cues" not in out["segments"][0]


def test_speaker_cues_project_stable_return_handoff_without_changing_geometry():
    candidate = _candidate_from_labels(2, ["SPEAKER_B"], [12.0])
    segment = candidate["segments"][0]
    segment.update({
        "text": "第一位先提问，第二位完整回答，第一位最后补充。",
        "speaker_confidence": 0.84,
        "sync_cues": [
            {"start": 0.0, "end": 2.5, "text": "第一位先提问，"},
            {"start": 2.5, "end": 10.0, "text": "第二位完整回答，"},
            {"start": 10.0, "end": 12.0, "text": "第一位最后补充。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_A", "duration": 2.5},
            {"start": 2.5, "end": 10.0, "speaker": "SPEAKER_B", "duration": 7.5},
            {"start": 10.0, "end": 12.0, "speaker": "SPEAKER_A", "duration": 2.0},
        ],
        "speaker_change_points": [2.5, 10.0],
        "speaker_overlap_risk": True,
    })
    before = (segment["start"], segment["end"], segment["text"], list(segment["sync_cues"]))

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][0]
    assert (projected["start"], projected["end"], projected["text"], projected["sync_cues"]) == before
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_A",
        "SPEAKER_B",
        "SPEAKER_A",
    ]
    assert out["speaker_cue_segment_count"] == 1


def test_speaker_cues_promote_sustained_review_run_without_changing_geometry():
    candidate = _candidate_from_labels(2, ["SPEAKER_A"], [10.0])
    segment = candidate["segments"][0]
    segment.update({
        "text": "第一位说明，第二位连续回答，第二位继续回答，第一位收尾。",
        "speaker_confidence": 0.80,
        "sync_cues": [
            {"start": 0.0, "end": 2.0, "text": "第一位说明，"},
            {"start": 2.0, "end": 4.0, "text": "第二位连续回答，"},
            {"start": 4.0, "end": 6.0, "text": "第二位继续回答，"},
            {"start": 6.0, "end": 10.0, "text": "第一位收尾。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A", "duration": 2.0},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_B", "duration": 2.0},
            {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_B", "duration": 2.0},
            {"start": 6.0, "end": 10.0, "speaker": "SPEAKER_A", "duration": 4.0},
        ],
        "speaker_change_points": [2.0, 6.0],
        "speaker_overlap_risk": True,
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_A", "score": 0.82, "margin": 0.20, "voice_coverage_ratio": 1.0, "overlap_ratio": 0.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 1, "speaker": "SPEAKER_B", "score": 0.69, "margin": 0.12, "voice_coverage_ratio": 1.0, "overlap_ratio": 0.0, "decision": "review", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 2, "speaker": "SPEAKER_B", "score": 0.71, "margin": 0.15, "voice_coverage_ratio": 0.8, "overlap_ratio": 0.2, "decision": "review", "embedding_scope": "sliding_window_weighted"},
            {"cue_index": 3, "speaker": "SPEAKER_A", "score": 0.84, "margin": 0.22, "voice_coverage_ratio": 1.0, "overlap_ratio": 0.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
        ],
    })
    frozen = (segment["start"], segment["end"], segment["text"], list(segment["sync_cues"]))

    out = _project_speaker_cues(candidate)

    projected = out["segments"][0]
    assert (projected["start"], projected["end"], projected["text"], projected["sync_cues"]) == frozen
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_A",
        "SPEAKER_B",
        "SPEAKER_B",
        "SPEAKER_A",
    ]
    assert projected["speaker_cue_mode"] == "campp_sustained_review_handoff"


def test_speaker_cues_do_not_promote_single_review_cue():
    candidate = _candidate_from_labels(2, ["SPEAKER_A"], [8.0])
    candidate["segments"][0].update({
        "speaker_confidence": 0.82,
        "sync_cues": [
            {"start": 0.0, "end": 2.0, "text": "第一位说明，"},
            {"start": 2.0, "end": 4.0, "text": "只有一段疑点，"},
            {"start": 4.0, "end": 6.0, "text": "第一位继续，"},
            {"start": 6.0, "end": 8.0, "text": "第一位结束。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_A", "duration": 2.0},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_B", "duration": 2.0},
            {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_A", "duration": 4.0},
        ],
        "speaker_change_points": [2.0, 4.0],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_A", "score": 0.82, "margin": 0.20, "voice_coverage_ratio": 1.0, "overlap_ratio": 0.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 1, "speaker": "SPEAKER_B", "score": 0.69, "margin": 0.12, "voice_coverage_ratio": 1.0, "overlap_ratio": 0.0, "decision": "review", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 2, "speaker": "SPEAKER_A", "score": 0.84, "margin": 0.22, "voice_coverage_ratio": 1.0, "overlap_ratio": 0.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 3, "speaker": "SPEAKER_A", "score": 0.83, "margin": 0.21, "voice_coverage_ratio": 1.0, "overlap_ratio": 0.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
        ],
    })

    out = _project_speaker_cues(candidate)

    assert "speaker_cues" not in out["segments"][0]


def test_speaker_cues_count_overlapping_windows_by_unique_time_coverage():
    candidate = _candidate_from_labels(2, ["SPEAKER_B"], [12.0])
    candidate["segments"][0].update({
        "text": "第一位先提问，第二位完整回答，第一位最后补充。",
        "speaker_confidence": 0.86,
        "sync_cues": [
            {"start": 0.0, "end": 2.8, "text": "第一位先提问，"},
            {"start": 2.8, "end": 10.0, "text": "第二位完整回答，"},
            {"start": 10.0, "end": 12.0, "text": "第一位最后补充。"},
        ],
        "speaker_subsegments": [
            {"start": 0.1, "end": 1.6, "speaker": "SPEAKER_A", "duration": 1.5},
            {"start": 1.8, "end": 3.3, "speaker": "SPEAKER_B", "duration": 1.5},
            {"start": 2.4, "end": 3.9, "speaker": "SPEAKER_B", "duration": 1.5},
            {"start": 3.0, "end": 10.0, "speaker": "SPEAKER_B", "duration": 7.0},
            {"start": 10.0, "end": 12.0, "speaker": "SPEAKER_A", "duration": 2.0},
        ],
        "speaker_change_points": [1.8, 10.0],
        "speaker_overlap_risk": True,
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][0]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_A",
        "SPEAKER_B",
        "SPEAKER_A",
    ]
    assert projected["speaker_cues"][0]["votes"] == {
        "SPEAKER_A": 1.5,
        "SPEAKER_B": 1.0,
    }


def test_speaker_cues_do_not_project_near_tied_return_pattern():
    candidate = _candidate_from_labels(2, ["SPEAKER_B"], [12.0])
    candidate["segments"][0].update({
        "text": "近似五五开的声纹序列不能直接拆成三段。",
        "speaker_confidence": 0.52,
        "sync_cues": [
            {"start": 0.0, "end": 3.0, "text": "第一段，"},
            {"start": 3.0, "end": 9.0, "text": "中间段，"},
            {"start": 9.0, "end": 12.0, "text": "最后段。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_A", "duration": 3.0},
            {"start": 3.0, "end": 9.0, "speaker": "SPEAKER_B", "duration": 6.0},
            {"start": 9.0, "end": 12.0, "speaker": "SPEAKER_A", "duration": 3.0},
        ],
        "speaker_change_points": [3.0, 9.0],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert "speaker_cues" not in out["segments"][0]
    assert out["speaker_cue_segment_count"] == 0


def test_speaker_cues_do_not_project_repeated_high_confidence_alternation():
    candidate = _candidate_from_labels(2, ["SPEAKER_B"], [12.0])
    candidate["segments"][0].update({
        "text": "多次快速交替不能直接写入正文说话人。",
        "speaker_confidence": 0.82,
        "sync_cues": [
            {"start": 0.0, "end": 3.0, "text": "第一段，"},
            {"start": 3.0, "end": 6.0, "text": "第二段，"},
            {"start": 6.0, "end": 9.0, "text": "第三段，"},
            {"start": 9.0, "end": 12.0, "text": "第四段。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_A", "duration": 3.0},
            {"start": 3.0, "end": 6.0, "speaker": "SPEAKER_B", "duration": 3.0},
            {"start": 6.0, "end": 9.0, "speaker": "SPEAKER_A", "duration": 3.0},
            {"start": 9.0, "end": 12.0, "speaker": "SPEAKER_B", "duration": 3.0},
        ],
        "speaker_change_points": [3.0, 6.0, 9.0],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert "speaker_cues" not in out["segments"][0]
    assert out["speaker_cue_segment_count"] == 0


def test_speaker_cues_collapse_overlap_window_jitter_to_one_dominant_handoff():
    candidate = _candidate_from_labels(2, ["SPEAKER_C"], [8.1])
    segment = candidate["segments"][0]
    segment.update({
        "text": "前面由第一位连续说明，最后由第二位接着补充。",
        "speaker_confidence": 0.61,
        "speaker_overlap_risk": True,
        "sync_cues": [
            {"start": 0.0, "end": 2.0, "text": "第一位说明，"},
            {"start": 2.0, "end": 4.0, "text": "中间有重叠，"},
            {"start": 4.0, "end": 6.0, "text": "第一位继续，"},
            {"start": 6.0, "end": 8.1, "text": "第二位补充。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_C", "duration": 1.5},
            {"start": 0.6, "end": 2.1, "speaker": "SPEAKER_C", "duration": 1.5},
            {"start": 1.8, "end": 3.3, "speaker": "SPEAKER_D", "duration": 1.5},
            {"start": 2.4, "end": 3.9, "speaker": "SPEAKER_D", "duration": 1.5},
            {"start": 4.0, "end": 5.5, "speaker": "SPEAKER_C", "duration": 1.5},
            {"start": 4.6, "end": 6.1, "speaker": "SPEAKER_C", "duration": 1.5},
            {"start": 6.0, "end": 7.5, "speaker": "SPEAKER_D", "duration": 1.5},
            {"start": 6.6, "end": 8.1, "speaker": "SPEAKER_D", "duration": 1.5},
        ],
        "speaker_change_points": [1.8, 4.0, 6.0],
    })
    frozen = (segment["start"], segment["end"], segment["text"], list(segment["sync_cues"]))

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][0]
    assert (projected["start"], projected["end"], projected["text"], projected["sync_cues"]) == frozen
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_C",
        "SPEAKER_C",
        "SPEAKER_C",
        "SPEAKER_D",
    ]
    assert projected["speaker_cue_mode"] == "campp_overlap_dominant_handoff"


def test_overlap_dominant_handoff_stays_review_only_on_exact_embedding_conflict():
    candidate = _candidate_from_labels(2, ["SPEAKER_C"], [8.1])
    candidate["segments"][0].update({
        "speaker_confidence": 0.61,
        "speaker_overlap_risk": True,
        "sync_cues": [
            {"start": 0.0, "end": 2.0, "text": "第一段，"},
            {"start": 2.0, "end": 4.0, "text": "第二段，"},
            {"start": 4.0, "end": 6.0, "text": "第三段，"},
            {"start": 6.0, "end": 8.1, "text": "第四段。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_C", "duration": 1.5},
            {"start": 0.6, "end": 2.1, "speaker": "SPEAKER_C", "duration": 1.5},
            {"start": 1.8, "end": 3.3, "speaker": "SPEAKER_D", "duration": 1.5},
            {"start": 2.4, "end": 3.9, "speaker": "SPEAKER_D", "duration": 1.5},
            {"start": 4.0, "end": 5.5, "speaker": "SPEAKER_C", "duration": 1.5},
            {"start": 4.6, "end": 6.1, "speaker": "SPEAKER_C", "duration": 1.5},
            {"start": 6.0, "end": 7.5, "speaker": "SPEAKER_D", "duration": 1.5},
            {"start": 6.6, "end": 8.1, "speaker": "SPEAKER_D", "duration": 1.5},
        ],
        "speaker_change_points": [1.8, 4.0, 6.0],
        "speaker_cue_embeddings": [
            {"cue_index": 3, "speaker": "SPEAKER_C", "score": 0.84, "margin": 0.20, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert "speaker_cues" not in out["segments"][0]
    assert out["speaker_cue_segment_count"] == 0


def test_overlap_dominant_handoff_ignores_sliding_window_conflict():
    candidate = _candidate_from_labels(2, ["SPEAKER_C"], [8.1])
    candidate["segments"][0].update({
        "speaker_confidence": 0.61,
        "speaker_overlap_risk": True,
        "sync_cues": [
            {"start": 0.0, "end": 2.0, "text": "第一段，"},
            {"start": 2.0, "end": 4.0, "text": "第二段，"},
            {"start": 4.0, "end": 6.0, "text": "第三段，"},
            {"start": 6.0, "end": 8.1, "text": "第四段。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_C", "duration": 1.5},
            {"start": 0.6, "end": 2.1, "speaker": "SPEAKER_C", "duration": 1.5},
            {"start": 1.8, "end": 3.3, "speaker": "SPEAKER_D", "duration": 1.5},
            {"start": 2.4, "end": 3.9, "speaker": "SPEAKER_D", "duration": 1.5},
            {"start": 4.0, "end": 5.5, "speaker": "SPEAKER_C", "duration": 1.5},
            {"start": 4.6, "end": 6.1, "speaker": "SPEAKER_C", "duration": 1.5},
            {"start": 6.0, "end": 7.5, "speaker": "SPEAKER_D", "duration": 1.5},
            {"start": 6.6, "end": 8.1, "speaker": "SPEAKER_D", "duration": 1.5},
        ],
        "speaker_change_points": [1.8, 4.0, 6.0],
        "speaker_cue_embeddings": [
            {
                "cue_index": 3,
                "speaker": "SPEAKER_C",
                "score": 0.78,
                "margin": 0.16,
                "voice_coverage_ratio": 1.0,
                "decision": "assign",
                "embedding_scope": "sliding_window_weighted",
            },
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert [cue["speaker"] for cue in out["segments"][0]["speaker_cues"]] == [
        "SPEAKER_C",
        "SPEAKER_C",
        "SPEAKER_C",
        "SPEAKER_D",
    ]


def test_speaker_cues_project_repeated_turns_for_balanced_two_speaker_dialogue():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_A", "SPEAKER_B"] * 4 + ["SPEAKER_A"],
        [4.0] * 9,
    )
    segment = candidate["segments"][1]
    segment.update({
        "speaker_confidence": 0.72,
        "sync_cues": [
            {"start": 4.2, "end": 5.2, "text": "第一位提问，"},
            {"start": 5.2, "end": 6.2, "text": "第二位回答，"},
            {"start": 6.2, "end": 7.2, "text": "第一位追问，"},
            {"start": 7.2, "end": 8.2, "text": "第二位补充。"},
        ],
        "speaker_subsegments": [
            {"start": 4.2, "end": 5.2, "speaker": "SPEAKER_A", "duration": 1.0},
            {"start": 5.2, "end": 6.2, "speaker": "SPEAKER_B", "duration": 1.0},
            {"start": 6.2, "end": 7.2, "speaker": "SPEAKER_A", "duration": 1.0},
            {"start": 7.2, "end": 8.2, "speaker": "SPEAKER_B", "duration": 1.0},
        ],
        "speaker_change_points": [5.2, 6.2, 7.2],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][1]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_A",
        "SPEAKER_B",
        "SPEAKER_A",
        "SPEAKER_B",
    ]
    assert projected["speaker_cue_mode"] == "balanced_two_speaker_dialogue"


def test_balanced_dialogue_projection_rejects_third_speaker_evidence():
    candidate = _candidate_from_labels(
        3,
        ["SPEAKER_A", "SPEAKER_B"] * 4 + ["SPEAKER_C"],
        [4.0] * 9,
    )
    segment = candidate["segments"][1]
    segment.update({
        "speaker_confidence": 0.72,
        "sync_cues": [
            {"start": 4.2, "end": 5.2, "text": "第一位提问，"},
            {"start": 5.2, "end": 6.2, "text": "第二位回答，"},
            {"start": 6.2, "end": 7.2, "text": "第一位追问，"},
            {"start": 7.2, "end": 8.2, "text": "第二位补充。"},
        ],
        "speaker_subsegments": [
            {"start": 4.2, "end": 5.2, "speaker": "SPEAKER_A", "duration": 1.0},
            {"start": 5.2, "end": 6.2, "speaker": "SPEAKER_B", "duration": 1.0},
            {"start": 6.2, "end": 7.2, "speaker": "SPEAKER_A", "duration": 1.0},
            {"start": 7.2, "end": 8.2, "speaker": "SPEAKER_B", "duration": 1.0},
        ],
        "speaker_change_points": [5.2, 6.2, 7.2],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert "speaker_cues" not in out["segments"][1]


def test_campp_cue_embedding_projects_local_handoff_in_unbalanced_dialogue():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_B"] * 9 + ["SPEAKER_A"],
        [4.0] * 10,
    )
    segment = candidate["segments"][0]
    segment.update({
        "speaker_confidence": 0.96,
        "sync_cues": [
            {"start": 0.0, "end": 1.0, "text": "第一位说。"},
            {"start": 1.0, "end": 2.0, "text": "第二位回答。"},
            {"start": 2.0, "end": 3.0, "text": "第一位继续。"},
        ],
        # Sliding windows drag the boundary and incorrectly keep the whole ASR
        # segment on B. Direct cue embeddings provide the local evidence.
        "speaker_subsegments": [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_B", "duration": 3.0},
        ],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_B", "score": 0.88, "margin": 0.18, "voice_coverage_ratio": 0.9, "decision": "assign"},
            {"cue_index": 1, "speaker": "SPEAKER_A", "score": 0.84, "margin": 0.12, "voice_coverage_ratio": 0.8, "decision": "assign"},
            {"cue_index": 2, "speaker": "SPEAKER_B", "score": 0.86, "margin": 0.14, "voice_coverage_ratio": 0.9, "decision": "assign"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][0]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_B",
        "SPEAKER_A",
        "SPEAKER_B",
    ]
    assert projected["speaker_cue_mode"] == "campp_sync_cue_embedding"
    assert projected["text"] == segment["text"]
    assert projected["sync_cues"] == segment["sync_cues"]


def test_subsecond_sliding_island_does_not_create_false_return_handoff():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_B", "SPEAKER_A", "SPEAKER_B"],
        [4.0, 5.61, 4.0],
    )
    segment = candidate["segments"][1]
    segment.update({
        "start": 4.0,
        "end": 9.61,
        "speaker_confidence": 0.542,
        "speaker_overlap_risk": True,
        "sync_cues": [
            {"start": 4.0, "end": 4.98, "text": "当前人开头，"},
            {"start": 4.98, "end": 8.65, "text": "当前人继续提问，"},
            {"start": 8.65, "end": 9.11, "text": "短暂滑窗，"},
            {"start": 9.11, "end": 9.61, "text": "当前人句尾。"},
        ],
        "speaker_subsegments": [
            {"start": 4.0, "end": 4.798, "speaker": "SPEAKER_A", "duration": 0.798},
            {"start": 5.102, "end": 6.602, "speaker": "SPEAKER_A", "duration": 1.5},
            {"start": 5.702, "end": 7.202, "speaker": "SPEAKER_A", "duration": 1.5},
            {"start": 6.302, "end": 7.802, "speaker": "SPEAKER_A", "duration": 1.5},
            {"start": 6.902, "end": 8.402, "speaker": "SPEAKER_B", "duration": 1.5},
            {"start": 7.502, "end": 9.002, "speaker": "SPEAKER_B", "duration": 1.5},
            {"start": 7.635, "end": 9.135, "speaker": "SPEAKER_B", "duration": 1.5},
            {"start": 9.587, "end": 9.61, "speaker": "SPEAKER_A", "duration": 0.023},
        ],
        "speaker_change_points": [6.902, 9.587],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_A", "score": 0.61, "margin": 0.06, "voice_coverage_ratio": 0.8, "decision": "insufficient", "embedding_scope": "sliding_window_weighted"},
            {"cue_index": 1, "speaker": "SPEAKER_A", "score": 0.74, "margin": 0.28, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 2, "speaker": "SPEAKER_B", "score": 0.72, "margin": 0.29, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "sliding_window_weighted"},
            {"cue_index": 3, "speaker": "SPEAKER_A", "score": 0.75, "margin": 0.01, "voice_coverage_ratio": 1.0, "decision": "insufficient", "embedding_scope": "sliding_window_weighted"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert "speaker_cues" not in out["segments"][1]


def test_terminal_sliding_bleed_moves_handoff_to_next_segment_boundary():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_B", "SPEAKER_A", "SPEAKER_A", "SPEAKER_A"],
        [4.0, 4.0, 4.43, 4.0],
    )
    segment = candidate["segments"][2]
    segment.update({
        "speaker_confidence": 0.558,
        "speaker_overlap_risk": True,
        "sync_cues": [
            {"start": 8.0, "end": 8.84, "text": "当前人回应，"},
            {"start": 8.84, "end": 9.28, "text": "短停顿，"},
            {"start": 9.28, "end": 11.93, "text": "另一人完整发言，"},
            {"start": 11.93, "end": 12.43, "text": "边界尾音。"},
        ],
        "speaker_subsegments": [
            {"start": 8.0, "end": 8.57, "speaker": "SPEAKER_A", "duration": 0.57},
            {"start": 9.51, "end": 11.61, "speaker": "SPEAKER_B", "duration": 2.1},
            {"start": 10.71, "end": 12.43, "speaker": "SPEAKER_A", "duration": 1.72},
        ],
        "speaker_change_points": [9.51, 10.71],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_A", "score": 0.74, "margin": 0.19, "voice_coverage_ratio": 0.68, "decision": "assign", "embedding_scope": "sliding_window_weighted"},
            {"cue_index": 2, "speaker": "SPEAKER_B", "score": 0.80, "margin": 0.27, "voice_coverage_ratio": 0.91, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 3, "speaker": "SPEAKER_A", "score": 0.79, "margin": 0.15, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "sliding_window_weighted"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert [cue["speaker"] for cue in out["segments"][2]["speaker_cues"]] == [
        "SPEAKER_A",
        "SPEAKER_A",
        "SPEAKER_B",
        "SPEAKER_B",
    ]


def test_moderate_intrasentence_return_jitter_does_not_create_handoff():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_B"] * 9 + ["SPEAKER_A"],
        [4.0] * 10,
    )
    segment = candidate["segments"][0]
    segment.update({
        "end": 5.9,
        "speaker_confidence": 0.83,
        "speaker_votes": {"SPEAKER_B": 8.3, "SPEAKER_A": 1.7},
        "sync_cues": [
            {"start": 0.0, "end": 2.8, "text": "这是同一句话的前半"},
            {"start": 2.8, "end": 5.2, "text": "这是同一句话的中间"},
            {"start": 5.2, "end": 5.9, "text": "部分。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 2.8, "speaker": "SPEAKER_B", "duration": 2.8},
            {"start": 2.8, "end": 5.2, "speaker": "SPEAKER_B", "duration": 2.4},
            {"start": 3.25, "end": 4.75, "speaker": "SPEAKER_A", "duration": 1.5},
            {"start": 5.2, "end": 5.9, "speaker": "SPEAKER_B", "duration": 0.7},
        ],
        "speaker_change_points": [2.8, 5.2],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_B", "score": 0.76, "margin": 0.17, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 1, "speaker": "SPEAKER_A", "score": 0.72, "margin": 0.15, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 2, "speaker": "SPEAKER_B", "score": 0.74, "margin": 0.12, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert "speaker_cues" not in out["segments"][0]


def test_consistent_runner_up_restores_returning_speaker_before_strong_current_onset():
    candidate = _candidate_from_labels(
        3,
        ["SPEAKER_B", "SPEAKER_D", "SPEAKER_A", "SPEAKER_A"],
        [4.0, 4.0, 15.0, 4.0],
    )
    candidate["segments"][0]["speaker_confidence"] = 0.98
    segment = candidate["segments"][2]
    segment.update({
        "speaker_confidence": 0.434,
        "speaker_overlap_risk": True,
        "speaker_votes": {
            "SPEAKER_A": 9.52,
            "SPEAKER_B": 7.888,
            "SPEAKER_C": 4.5,
        },
        "sync_cues": [
            {"start": 8.0, "end": 12.2, "text": "返回说话人先提问，"},
            {"start": 12.2, "end": 20.6, "text": "返回说话人继续，"},
            {"start": 20.6, "end": 23.0, "text": "当前人开始回答。"},
        ],
        "speaker_subsegments": [
            {"start": 8.0, "end": 12.2, "speaker": "SPEAKER_A", "duration": 4.2},
            {"start": 12.2, "end": 20.6, "speaker": "SPEAKER_A", "duration": 8.4},
            {"start": 20.6, "end": 23.0, "speaker": "SPEAKER_A", "duration": 2.4},
        ],
        "speaker_change_points": [9.0, 12.0, 19.8],
        "speaker_cue_embeddings": [
            {
                "cue_index": 0,
                "speaker": "SPEAKER_A",
                "score": 0.7395,
                "margin": 0.0194,
                "voice_coverage_ratio": 0.77,
                "decision": "review",
                "embedding_scope": "exact_sync_cue",
                "second_score": 0.7202,
                "second_speaker": "SPEAKER_B",
            },
            {
                "cue_index": 1,
                "speaker": "SPEAKER_A",
                "score": 0.601,
                "margin": 0.0375,
                "voice_coverage_ratio": 0.53,
                "decision": "insufficient",
                "embedding_scope": "exact_sync_cue",
                "second_score": 0.5634,
                "second_speaker": "SPEAKER_B",
            },
            {
                "cue_index": 2,
                "speaker": "SPEAKER_A",
                "score": 0.7453,
                "margin": 0.2353,
                "voice_coverage_ratio": 1.0,
                "decision": "assign",
                "embedding_scope": "exact_sync_cue",
                "second_score": 0.51,
                "second_speaker": "SPEAKER_B",
            },
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][2]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_B",
        "SPEAKER_B",
        "SPEAKER_A",
    ]
    assert projected["speaker_cue_mode"] == "campp_consistent_runner_up_return"
    assert projected["text"] == segment["text"]
    assert projected["sync_cues"] == segment["sync_cues"]


def test_dominant_interior_absorbs_weak_boundary_windows_before_next_turn():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_C", "SPEAKER_B", "SPEAKER_B"],
        [4.0, 9.342, 4.0],
    )
    candidate["segments"][0]["speaker_confidence"] = 0.83
    candidate["segments"][2]["speaker_confidence"] = 0.77
    segment = candidate["segments"][1]
    segment.update({
        "speaker_confidence": 0.607,
        "speaker_overlap_risk": True,
        "speaker_votes": {"SPEAKER_B": 6.938, "SPEAKER_C": 4.5},
        "sync_cues": [
            {"start": 4.0, "end": 6.349, "text": "同一人开头，"},
            {"start": 6.349, "end": 12.528, "text": "同一人继续说，"},
            {"start": 12.528, "end": 13.342, "text": "同一人结尾。"},
        ],
        "speaker_subsegments": [
            {"start": 4.0, "end": 6.349, "speaker": "SPEAKER_B", "duration": 2.349},
            {"start": 6.349, "end": 12.528, "speaker": "SPEAKER_C", "duration": 6.179},
            {"start": 12.528, "end": 13.342, "speaker": "SPEAKER_B", "duration": 0.814},
        ],
        "speaker_change_points": [8.335, 11.862],
        "speaker_cue_embeddings": [
            {
                "cue_index": 0,
                "speaker": "SPEAKER_B",
                "score": 0.6051,
                "margin": 0.0926,
                "voice_coverage_ratio": 0.59,
                "decision": "insufficient",
                "embedding_scope": "exact_sync_cue",
                "second_score": 0.5126,
                "second_speaker": "SPEAKER_A",
            },
            {
                "cue_index": 1,
                "speaker": "SPEAKER_C",
                "score": 0.7076,
                "margin": 0.2404,
                "voice_coverage_ratio": 0.66,
                "decision": "assign",
                "embedding_scope": "exact_sync_cue",
                "second_score": 0.4672,
                "second_speaker": "SPEAKER_A",
            },
            {
                "cue_index": 2,
                "speaker": "SPEAKER_B",
                "score": 0.7059,
                "margin": 0.1433,
                "voice_coverage_ratio": 1.0,
                "decision": "assign",
                "embedding_scope": "sliding_window_weighted",
                "second_score": 0.5627,
                "second_speaker": "SPEAKER_A",
            },
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][1]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_C",
        "SPEAKER_C",
        "SPEAKER_C",
    ]
    assert projected["speaker_cue_mode"] == "campp_dominant_interior_override"
    assert projected["text"] == segment["text"]
    assert projected["sync_cues"] == segment["sync_cues"]


def test_strong_terminal_exact_and_next_context_preserve_real_handoff():
    candidate = _candidate_from_labels(2, ["SPEAKER_A", "SPEAKER_D"], [6.0, 4.0])
    candidate["segments"][1]["start"] = 6.0
    candidate["segments"][1]["end"] = 10.0
    segment = candidate["segments"][0]
    segment.update({
        "speaker_confidence": 0.72,
        "sync_cues": [
            {"start": 0.0, "end": 2.0, "text": "当前人开始，"},
            {"start": 2.0, "end": 4.4, "text": "当前人接着说这"},
            {"start": 4.4, "end": 6.0, "text": "个结尾。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 4.4, "speaker": "SPEAKER_A", "duration": 4.4},
            {"start": 4.4, "end": 6.0, "speaker": "SPEAKER_D", "duration": 1.6},
        ],
        "speaker_change_points": [4.4],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_A", "score": 0.81, "margin": 0.20, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 1, "speaker": "SPEAKER_A", "score": 0.78, "margin": 0.18, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 2, "speaker": "SPEAKER_D", "score": 0.71, "margin": 0.22, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert [cue["speaker"] for cue in out["segments"][0]["speaker_cues"]] == [
        "SPEAKER_A",
        "SPEAKER_A",
        "SPEAKER_D",
    ]


def test_campp_review_band_keeps_current_speaker_inside_projected_handoff():
    candidate = _candidate_from_labels(2, ["SPEAKER_B"] * 8 + ["SPEAKER_A"], [4.0] * 9)
    segment = candidate["segments"][0]
    segment.update({
        "speaker_confidence": 0.94,
        "sync_cues": [
            {"start": 0.0, "end": 1.0, "text": "明确另一人。"},
            {"start": 1.0, "end": 2.0, "text": "边界不清。"},
            {"start": 2.0, "end": 3.0, "text": "当前人继续。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_B", "duration": 3.0},
        ],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_A", "score": 0.86, "margin": 0.15, "voice_coverage_ratio": 0.9, "decision": "assign"},
            {"cue_index": 1, "speaker": "SPEAKER_A", "score": 0.69, "margin": 0.02, "voice_coverage_ratio": 0.8, "decision": "review"},
            {"cue_index": 2, "speaker": "SPEAKER_B", "score": 0.88, "margin": 0.17, "voice_coverage_ratio": 0.9, "decision": "assign"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][0]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_A",
        "SPEAKER_B",
        "SPEAKER_B",
    ]
    assert projected["speaker_cues"][1]["review"] is True
    assert projected["speaker_cue_review"] is True


def test_exact_cue_boundary_smoothing_recovers_short_gap_and_terminal_fragment():
    candidate = _candidate_from_labels(2, ["SPEAKER_A"] * 8 + ["SPEAKER_B"], [4.0] * 9)
    segment = candidate["segments"][0]
    segment.update({
        "speaker_confidence": 0.91,
        "sync_cues": [
            {"start": 0.0, "end": 0.84, "text": "第一人结束，"},
            {"start": 0.84, "end": 1.28, "text": "短边界，"},
            {"start": 1.28, "end": 3.93, "text": "第二人完整发言"},
            {"start": 3.93, "end": 4.43, "text": "的句尾。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 0.84, "speaker": "SPEAKER_A", "duration": 0.84},
            {"start": 1.28, "end": 3.93, "speaker": "SPEAKER_B", "duration": 2.65},
            {"start": 3.93, "end": 4.43, "speaker": "SPEAKER_A", "duration": 0.5},
        ],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_A", "score": 0.78, "margin": 0.16, "voice_coverage_ratio": 0.8, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 1, "speaker": "SPEAKER_B", "score": 0.48, "margin": 0.02, "voice_coverage_ratio": 0.0, "decision": "insufficient", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 2, "speaker": "SPEAKER_B", "score": 0.82, "margin": 0.20, "voice_coverage_ratio": 0.9, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 3, "speaker": "SPEAKER_A", "score": 0.68, "margin": 0.10, "voice_coverage_ratio": 1.0, "decision": "review", "embedding_scope": "exact_sync_cue"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][0]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_A",
        "SPEAKER_B",
        "SPEAKER_B",
        "SPEAKER_B",
    ]
    assert projected["speaker_cues"][1]["source"] == "campp_boundary_inherit_next"
    assert projected["speaker_cues"][3]["source"] == "campp_boundary_inherit_previous"


def test_leading_overlap_cue_inherits_next_exact_speaker_on_cross_source_agreement():
    candidate = _candidate_from_labels(2, ["SPEAKER_A"] * 8 + ["SPEAKER_B"], [4.0] * 9)
    segment = candidate["segments"][0]
    segment.update({
        "end": 4.0,
        "speaker_confidence": 0.92,
        "speaker_overlap_risk": True,
        "sync_cues": [
            {"start": 0.0, "end": 0.8, "text": "第二人边界。"},
            {"start": 0.8, "end": 1.8, "text": "第二人继续。"},
            {"start": 1.8, "end": 2.9, "text": "第一人接话。"},
            {"start": 2.9, "end": 4.0, "text": "第一人继续。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 1.8, "speaker": "SPEAKER_B", "duration": 1.8},
            {"start": 1.8, "end": 4.0, "speaker": "SPEAKER_A", "duration": 2.2},
        ],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_B", "score": 0.65, "margin": 0.08, "voice_coverage_ratio": 0.8, "overlap_ratio": 0.20, "decision": "review", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 1, "speaker": "SPEAKER_B", "score": 0.82, "margin": 0.18, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 2, "speaker": "SPEAKER_A", "score": 0.85, "margin": 0.20, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 3, "speaker": "SPEAKER_A", "score": 0.84, "margin": 0.19, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][0]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_B",
        "SPEAKER_B",
        "SPEAKER_A",
        "SPEAKER_A",
    ]
    assert projected["speaker_cues"][0]["source"] == "campp_boundary_inherit_next_exact"


def test_adjacent_exact_cues_recover_overlap_heavy_leading_speaker():
    candidate = _candidate_from_labels(2, ["SPEAKER_A"] * 8 + ["SPEAKER_B"], [4.0] * 9)
    segment = candidate["segments"][0]
    segment.update({
        "end": 3.75,
        "speaker_confidence": 0.69,
        "speaker_overlap_risk": True,
        "sync_cues": [
            {"start": 0.0, "end": 0.75, "text": "第二人回应，"},
            {"start": 0.75, "end": 1.2, "text": "继续回应，"},
            {"start": 1.2, "end": 3.75, "text": "第一人继续。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 1.45, "speaker": "SPEAKER_B", "duration": 1.45},
            {"start": 0.0, "end": 0.2, "speaker": "SPEAKER_A", "duration": 0.2},
            {"start": 1.45, "end": 3.75, "speaker": "SPEAKER_A", "duration": 2.3},
        ],
        "speaker_change_points": [0.0, 1.45],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_B", "score": 0.75, "margin": 0.22, "voice_coverage_ratio": 0.0, "overlap_ratio": 0.18, "decision": "insufficient", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 1, "speaker": "SPEAKER_B", "score": 0.78, "margin": 0.20, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 2, "speaker": "SPEAKER_A", "score": 0.86, "margin": 0.28, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][0]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_B",
        "SPEAKER_B",
        "SPEAKER_A",
    ]
    assert projected["speaker_cues"][0]["source"] == "campp_boundary_inherit_next_exact"


def test_short_previous_turn_bleed_does_not_create_false_leading_handoff():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_B", "SPEAKER_A", "SPEAKER_A"],
        [4.0, 6.7, 5.0],
    )
    segment = candidate["segments"][1]
    segment.update({
        "speaker_confidence": 0.94,
        "sync_cues": [
            {"start": 4.0, "end": 6.0, "text": "当前人开头，"},
            {"start": 6.0, "end": 8.2, "text": "当前人继续，"},
            {"start": 8.2, "end": 10.2, "text": "当前人补充，"},
            {"start": 10.2, "end": 10.7, "text": "结束。"},
        ],
        "speaker_subsegments": [
            {"start": 4.0, "end": 4.895, "speaker": "SPEAKER_B", "duration": 0.895},
            {"start": 5.18, "end": 6.68, "speaker": "SPEAKER_A", "duration": 1.5},
            {"start": 6.0, "end": 10.7, "speaker": "SPEAKER_A", "duration": 4.7},
        ],
        "speaker_change_points": [5.18],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_B", "score": 0.74, "margin": 0.20, "voice_coverage_ratio": 0.44, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 1, "speaker": "SPEAKER_B", "score": 0.46, "margin": 0.03, "voice_coverage_ratio": 0.57, "decision": "insufficient", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 2, "speaker": "SPEAKER_A", "score": 0.85, "margin": 0.23, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 3, "speaker": "SPEAKER_A", "score": 0.88, "margin": 0.29, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert "speaker_cues" not in out["segments"][1]


def test_clean_short_previous_turn_is_preserved_as_real_handoff():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_B"],
        [2.0, 6.2, 4.0],
    )
    segment = candidate["segments"][1]
    segment.update({
        "speaker_confidence": 0.92,
        "sync_cues": [
            {"start": 2.0, "end": 2.94, "text": "前一人短回应，"},
            {"start": 2.94, "end": 6.3, "text": "当前人发言，"},
            {"start": 6.3, "end": 8.2, "text": "当前人继续。"},
        ],
        "speaker_subsegments": [
            {"start": 2.0, "end": 2.79, "speaker": "SPEAKER_A", "duration": 0.79},
            {"start": 2.13, "end": 2.79, "speaker": "SPEAKER_B", "duration": 0.66},
            {"start": 2.94, "end": 8.2, "speaker": "SPEAKER_B", "duration": 5.26},
        ],
        "speaker_change_points": [2.28],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_A", "score": 0.77, "margin": 0.11, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 1, "speaker": "SPEAKER_B", "score": 0.78, "margin": 0.14, "voice_coverage_ratio": 0.72, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 2, "speaker": "SPEAKER_B", "score": 0.77, "margin": 0.07, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "sliding_window_weighted"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert [cue["speaker"] for cue in out["segments"][1]["speaker_cues"]] == [
        "SPEAKER_A",
        "SPEAKER_B",
        "SPEAKER_B",
    ]


def test_sliding_window_leading_bleed_inherits_stable_current_context():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_B", "SPEAKER_D", "SPEAKER_D"],
        [4.0, 3.4, 5.0],
    )
    segment = candidate["segments"][1]
    segment.update({
        "speaker_confidence": 0.54,
        "speaker_overlap_risk": True,
        "overlap_ratio": 0.25,
        "sync_cues": [
            {"start": 4.2, "end": 5.3, "text": "当前人开头，"},
            {"start": 5.3, "end": 6.5, "text": "当前人继续，"},
            {"start": 6.5, "end": 7.6, "text": "当前人结束。"},
        ],
        "speaker_subsegments": [
            {"start": 4.2, "end": 5.3, "speaker": "SPEAKER_B", "duration": 1.1},
            {"start": 5.3, "end": 7.6, "speaker": "SPEAKER_D", "duration": 2.3},
        ],
        "speaker_change_points": [5.3],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_B", "score": 0.76, "margin": 0.18, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "sliding_window_weighted"},
            {"cue_index": 1, "speaker": "SPEAKER_D", "score": 0.78, "margin": 0.16, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 2, "speaker": "SPEAKER_D", "score": 0.80, "margin": 0.18, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
        ],
    })
    candidate["segments"][2]["speaker_confidence"] = 0.65

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert "speaker_cues" not in out["segments"][1]


def test_high_confidence_parent_removes_weak_leading_sliding_bleed_without_next_context():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_A"],
        [0.35, 4.75, 3.0],
    )
    segment = candidate["segments"][1]
    segment_start = float(segment["start"])
    frozen_cues = [
        {"start": segment_start, "end": segment_start + 0.86, "text": "短开头，"},
        {"start": segment_start + 0.86, "end": segment_start + 1.38, "text": "主说话人，"},
        {"start": segment_start + 1.38, "end": segment_start + 2.60, "text": "继续说明，"},
        {"start": segment_start + 2.60, "end": segment_start + 4.75, "text": "完成本轮发言。"},
    ]
    segment.update({
        "text": "".join(cue["text"] for cue in frozen_cues),
        "speaker_confidence": 0.912,
        "speaker_votes": {"SPEAKER_A": 0.58, "SPEAKER_B": 6.0},
        "sync_cues": frozen_cues,
        "speaker_subsegments": [
            {"start": segment_start, "end": segment_start + 0.53, "speaker": "SPEAKER_A", "duration": 0.53},
            {"start": segment_start + 1.20, "end": segment_start + 2.70, "speaker": "SPEAKER_B", "duration": 1.50},
            {"start": segment_start + 1.80, "end": segment_start + 3.30, "speaker": "SPEAKER_B", "duration": 1.50},
            {"start": segment_start + 2.40, "end": segment_start + 3.90, "speaker": "SPEAKER_B", "duration": 1.50},
            {"start": segment_start + 3.00, "end": segment_start + 4.50, "speaker": "SPEAKER_B", "duration": 1.50},
        ],
        "speaker_change_points": [segment_start + 1.20],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_A", "score": 0.693, "margin": 0.182, "voice_coverage_ratio": 0.61, "decision": "assign", "embedding_scope": "sliding_window_weighted"},
            {"cue_index": 1, "speaker": "SPEAKER_B", "score": 0.82, "margin": 0.25, "voice_coverage_ratio": 0.35, "decision": "insufficient", "embedding_scope": "sliding_window_weighted"},
            {"cue_index": 2, "speaker": "SPEAKER_B", "score": 0.85, "margin": 0.27, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "sliding_window_weighted"},
            {"cue_index": 3, "speaker": "SPEAKER_B", "score": 0.85, "margin": 0.28, "voice_coverage_ratio": 0.86, "decision": "assign", "embedding_scope": "exact_sync_cue"},
        ],
    })
    frozen_geometry = json.loads(json.dumps({
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"],
        "sync_cues": segment["sync_cues"],
    }, ensure_ascii=False))

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][1]
    assert projected["speaker"] == "SPEAKER_B"
    assert "speaker_cues" not in projected
    assert {
        "start": projected["start"],
        "end": projected["end"],
        "text": projected["text"],
        "sync_cues": projected["sync_cues"],
    } == frozen_geometry


def test_high_confidence_parent_does_not_override_exact_leading_reply():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_A"],
        [0.35, 4.0, 3.0],
    )
    segment = candidate["segments"][1]
    segment_start = float(segment["start"])
    segment.update({
        "speaker_confidence": 0.93,
        "speaker_votes": {"SPEAKER_A": 0.55, "SPEAKER_B": 6.2},
        "sync_cues": [
            {"start": segment_start, "end": segment_start + 0.80, "text": "真实短回应，"},
            {"start": segment_start + 0.80, "end": segment_start + 2.0, "text": "主说话人开始，"},
            {"start": segment_start + 2.0, "end": segment_start + 4.0, "text": "主说话人继续。"},
        ],
        "speaker_subsegments": [
            {"start": segment_start, "end": segment_start + 0.80, "speaker": "SPEAKER_A", "duration": 0.80},
            {"start": segment_start + 0.80, "end": segment_start + 4.0, "speaker": "SPEAKER_B", "duration": 3.20},
        ],
        "speaker_change_points": [segment_start + 0.80],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_A", "score": 0.70, "margin": 0.18, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 1, "speaker": "SPEAKER_B", "score": 0.84, "margin": 0.23, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 2, "speaker": "SPEAKER_B", "score": 0.86, "margin": 0.25, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert [cue["speaker"] for cue in out["segments"][1]["speaker_cues"]] == [
        "SPEAKER_A",
        "SPEAKER_B",
        "SPEAKER_B",
    ]
    assert out["segments"][1]["speaker_cues"][0]["source"] == "campp_sync_cue_embedding"


def test_ambiguous_leading_cue_uses_previous_speaker_on_cross_source_tie():
    candidate = _candidate_from_labels(
        3,
        ["SPEAKER_B", "SPEAKER_A", "SPEAKER_D"],
        [4.0, 4.0, 5.0],
    )
    segment = candidate["segments"][1]
    segment.update({
        "speaker_confidence": 0.72,
        "sync_cues": [
            {"start": 4.2, "end": 4.97, "text": "前一人短回应，"},
            {"start": 4.97, "end": 6.4, "text": "当前人开始，"},
            {"start": 6.4, "end": 8.2, "text": "当前人继续。"},
        ],
        "speaker_subsegments": [
            {"start": 4.2, "end": 4.97, "speaker": "SPEAKER_B", "duration": 0.77},
            {"start": 4.2, "end": 4.84, "speaker": "SPEAKER_D", "duration": 0.64},
            {"start": 4.54, "end": 5.0, "speaker": "SPEAKER_A", "duration": 0.46},
            {"start": 4.97, "end": 8.2, "speaker": "SPEAKER_A", "duration": 3.23},
        ],
        "speaker_change_points": [4.2, 4.54, 4.97],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_D", "second_speaker": "SPEAKER_B", "second_score": 0.721, "score": 0.728, "margin": 0.007, "voice_coverage_ratio": 1.0, "decision": "review", "embedding_scope": "sliding_window_weighted"},
            {"cue_index": 1, "speaker": "SPEAKER_A", "score": 0.82, "margin": 0.15, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "sliding_window_weighted"},
            {"cue_index": 2, "speaker": "SPEAKER_A", "score": 0.79, "margin": 0.13, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    assert [cue["speaker"] for cue in out["segments"][1]["speaker_cues"]] == [
        "SPEAKER_B",
        "SPEAKER_A",
        "SPEAKER_A",
    ]
    assert out["segments"][1]["speaker_cues"][0]["source"] == "campp_ambiguous_boundary_inherit_previous"


def test_context_anchored_three_speaker_handoff_is_projected_without_geometry_change():
    candidate = _candidate_from_labels(
        3,
        ["SPEAKER_B", "SPEAKER_A", "SPEAKER_D"],
        [4.0, 4.0, 5.0],
    )
    candidate["segments"][2]["start"] = 8.2
    candidate["segments"][2]["end"] = 13.2
    segment = candidate["segments"][1]
    frozen_cues = [
        {"start": 4.2, "end": 4.97, "text": "前一人短回应，"},
        {"start": 4.97, "end": 6.2, "text": "当前人开始，"},
        {"start": 6.2, "end": 7.0, "text": "当前人继续，"},
        {"start": 7.0, "end": 8.2, "text": "后一人接话。"},
    ]
    segment.update({
        "speaker_confidence": 0.72,
        "sync_cues": frozen_cues,
        "speaker_subsegments": [
            {"start": 4.2, "end": 4.97, "speaker": "SPEAKER_B", "duration": 0.77},
            {"start": 4.2, "end": 4.84, "speaker": "SPEAKER_D", "duration": 0.64},
            {"start": 4.54, "end": 7.0, "speaker": "SPEAKER_A", "duration": 2.46},
            {"start": 7.0, "end": 8.2, "speaker": "SPEAKER_D", "duration": 1.2},
        ],
        "speaker_change_points": [4.2, 4.54, 4.97, 7.0],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_D", "second_speaker": "SPEAKER_B", "second_score": 0.721, "score": 0.728, "margin": 0.007, "voice_coverage_ratio": 1.0, "decision": "review", "embedding_scope": "sliding_window_weighted"},
            {"cue_index": 1, "speaker": "SPEAKER_A", "score": 0.82, "margin": 0.15, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "sliding_window_weighted"},
            {"cue_index": 2, "speaker": "SPEAKER_A", "score": 0.79, "margin": 0.13, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 3, "speaker": "SPEAKER_D", "score": 0.76, "margin": 0.19, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][1]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_B",
        "SPEAKER_A",
        "SPEAKER_A",
        "SPEAKER_D",
    ]
    assert projected["speaker_cue_mode"] == "campp_context_anchored_multi_handoff"
    assert projected["sync_cues"] == frozen_cues


def test_single_long_cue_projects_context_anchored_internal_handoff():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_A", "SPEAKER_A", "SPEAKER_B"],
        [5.0, 5.7, 6.0],
    )
    segment = candidate["segments"][1]
    original_cues = [{"start": 5.0, "end": 10.7, "text": "前一位提问后一位回答。"}]
    segment.update({
        "speaker_confidence": 0.59,
        "speaker_overlap_risk": True,
        "sync_cues": original_cues,
        "speaker_subsegments": [
            {"start": 5.0, "end": 6.5, "speaker": "SPEAKER_A", "duration": 1.5},
            {"start": 6.0, "end": 7.5, "speaker": "SPEAKER_A", "duration": 1.5},
            {"start": 9.75, "end": 10.7, "speaker": "SPEAKER_B", "duration": 0.95},
        ],
        "speaker_change_points": [9.75],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_B", "score": 0.46, "margin": 0.03, "voice_coverage_ratio": 0.47, "decision": "insufficient", "embedding_scope": "exact_sync_cue"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][1]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_A",
        "SPEAKER_B",
    ]
    assert "".join(cue["text"] for cue in projected["speaker_cues"]) == original_cues[0]["text"]
    assert projected["sync_cues"] == original_cues
    assert projected["speaker_cue_mode"] == "campp_intracue_context_handoff"


def test_single_long_review_cue_projects_context_anchored_internal_handoff():
    candidate = _candidate_from_labels(
        2,
        ["SPEAKER_A", "SPEAKER_A", "SPEAKER_B"],
        [5.0, 5.7, 6.0],
    )
    segment = candidate["segments"][1]
    original_cues = [{"start": 5.0, "end": 10.7, "text": "前一位提问后一位回答。"}]
    segment.update({
        "speaker_confidence": 0.59,
        "speaker_overlap_risk": True,
        "sync_cues": original_cues,
        "speaker_subsegments": [
            {"start": 5.0, "end": 6.5, "speaker": "SPEAKER_A", "duration": 1.5},
            {"start": 6.0, "end": 7.5, "speaker": "SPEAKER_A", "duration": 1.5},
            {"start": 9.75, "end": 10.7, "speaker": "SPEAKER_B", "duration": 0.95},
        ],
        "speaker_change_points": [9.75],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_A", "score": 0.65, "margin": 0.04, "voice_coverage_ratio": 0.47, "overlap_ratio": 0.0, "decision": "review", "embedding_scope": "exact_sync_cue"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][1]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_A",
        "SPEAKER_B",
    ]
    assert projected["sync_cues"] == original_cues


def test_change_point_inside_review_cue_preserves_previous_strong_speaker():
    candidate = _candidate_from_labels(2, ["SPEAKER_B"] * 8 + ["SPEAKER_A"], [4.0] * 9)
    segment = candidate["segments"][0]
    cues = [
        {"start": 0.0, "end": 1.0, "text": "前一人。"},
        {"start": 1.0, "end": 2.0, "text": "前一人继续。"},
        {"start": 2.0, "end": 2.8, "text": "边界句。"},
        {"start": 2.8, "end": 3.8, "text": "后一人。"},
        {"start": 3.8, "end": 4.8, "text": "后一人继续。"},
        {"start": 4.8, "end": 5.8, "text": "后一人继续。"},
        {"start": 5.8, "end": 6.8, "text": "后一人结束。"},
    ]
    segment.update({
        "end": 6.8,
        "speaker_confidence": 0.91,
        "sync_cues": cues,
        "speaker_change_points": [2.55],
        "speaker_subsegments": [
            {"start": 0.0, "end": 2.4, "speaker": "SPEAKER_A", "duration": 2.4},
            {"start": 2.4, "end": 6.8, "speaker": "SPEAKER_B", "duration": 4.4},
        ],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_A", "score": 0.82, "margin": 0.18, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 1, "speaker": "SPEAKER_A", "score": 0.80, "margin": 0.16, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 2, "speaker": "SPEAKER_B", "score": 0.66, "margin": 0.04, "voice_coverage_ratio": 0.9, "decision": "review", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 3, "speaker": "SPEAKER_B", "score": 0.84, "margin": 0.19, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 4, "speaker": "SPEAKER_B", "score": 0.85, "margin": 0.20, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 5, "speaker": "SPEAKER_B", "score": 0.83, "margin": 0.18, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 6, "speaker": "SPEAKER_B", "score": 0.82, "margin": 0.17, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][0]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_A",
        "SPEAKER_A",
        "SPEAKER_A",
        "SPEAKER_B",
        "SPEAKER_B",
        "SPEAKER_B",
        "SPEAKER_B",
    ]
    assert projected["speaker_cues"][2]["source"] == "campp_change_point_inherit_previous"


def test_exact_cue_insufficient_terminal_returns_to_segment_speaker():
    candidate = _candidate_from_labels(2, ["SPEAKER_B"] * 8 + ["SPEAKER_A"], [4.0] * 9)
    segment = candidate["segments"][0]
    segment.update({
        "speaker_confidence": 0.92,
        "sync_cues": [
            {"start": 0.0, "end": 2.0, "text": "主说话人。"},
            {"start": 2.0, "end": 2.95, "text": "短插话。"},
            {"start": 2.95, "end": 3.95, "text": "主说话人返回。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_B", "duration": 2.0},
            {"start": 2.0, "end": 3.95, "speaker": "SPEAKER_A", "duration": 1.95},
        ],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_B", "score": 0.84, "margin": 0.18, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 1, "speaker": "SPEAKER_A", "score": 0.75, "margin": 0.11, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 2, "speaker": "SPEAKER_A", "score": 0.63, "margin": 0.06, "voice_coverage_ratio": 0.5, "decision": "insufficient", "embedding_scope": "exact_sync_cue"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][0]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_B",
        "SPEAKER_A",
        "SPEAKER_B",
    ]
    assert projected["speaker_cues"][2]["source"] == "campp_exact_cue_insufficient"


def test_exact_and_sliding_agreement_preserves_boundary_speaker_run():
    candidate = _candidate_from_labels(2, ["SPEAKER_A"] * 8 + ["SPEAKER_B"], [4.0] * 9)
    segment = candidate["segments"][0]
    segment.update({
        "end": 5.5,
        "speaker_confidence": 0.88,
        "speaker_overlap_risk": True,
        "sync_cues": [
            {"start": 0.0, "end": 1.2, "text": "第二人开头。"},
            {"start": 1.2, "end": 3.2, "text": "第一人发言。"},
            {"start": 3.2, "end": 5.2, "text": "第二人继续"},
            {"start": 5.2, "end": 5.5, "text": "句尾。"},
        ],
        "speaker_subsegments": [
            {"start": 0.0, "end": 1.2, "speaker": "SPEAKER_B", "duration": 1.2},
            {"start": 1.2, "end": 3.2, "speaker": "SPEAKER_A", "duration": 2.0},
            {"start": 3.2, "end": 5.5, "speaker": "SPEAKER_B", "duration": 2.3},
        ],
        "speaker_cue_embeddings": [
            {"cue_index": 0, "speaker": "SPEAKER_B", "score": 0.64, "margin": 0.11, "voice_coverage_ratio": 1.0, "decision": "insufficient", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 1, "speaker": "SPEAKER_A", "score": 0.82, "margin": 0.22, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 2, "speaker": "SPEAKER_B", "score": 0.84, "margin": 0.24, "voice_coverage_ratio": 1.0, "decision": "assign", "embedding_scope": "exact_sync_cue"},
            {"cue_index": 3, "speaker": "SPEAKER_B", "score": 0.50, "margin": 0.08, "voice_coverage_ratio": 1.0, "decision": "insufficient", "embedding_scope": "exact_sync_cue"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][0]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_B",
        "SPEAKER_A",
        "SPEAKER_B",
        "SPEAKER_B",
    ]
    assert projected["speaker_cues"][0]["source"] == "campp_exact_sliding_agreement"
    assert projected["speaker_cues"][3]["source"] == "campp_boundary_inherit_previous"


def test_sustained_sliding_run_survives_unusable_exact_boundary_embedding():
    candidate = _candidate_from_labels(2, ["SPEAKER_A"] * 8 + ["SPEAKER_B"], [4.0] * 9)
    segment = candidate["segments"][0]
    segment.update({
        "end": 7.8,
        "speaker_confidence": 0.64,
        "speaker_overlap_risk": True,
        "sync_cues": [
            {"start": 0.0, "end": 3.9, "text": "另一位说话人持续发言。"},
            {"start": 3.9, "end": 6.1, "text": "当前说话人接续。"},
            {"start": 6.1, "end": 7.8, "text": "当前说话人继续。"},
        ],
        "speaker_subsegments": [
            {"start": 2.0, "end": 3.5, "speaker": "SPEAKER_B", "duration": 1.5},
            {"start": 3.0, "end": 4.5, "speaker": "SPEAKER_B", "duration": 1.5},
            {"start": 4.6, "end": 6.1, "speaker": "SPEAKER_A", "duration": 1.5},
            {"start": 6.1, "end": 7.8, "speaker": "SPEAKER_A", "duration": 1.7},
        ],
        "speaker_change_points": [4.6],
        "speaker_cue_embeddings": [
            {
                "cue_index": 0,
                "speaker": "SPEAKER_B",
                "score": 0.47,
                "margin": 0.01,
                "voice_coverage_ratio": 0.0,
                "overlap_ratio": 0.07,
                "decision": "insufficient",
                "embedding_scope": "exact_sync_cue",
            },
            {"cue_index": 1, "speaker": "SPEAKER_A", "score": 0.81, "margin": 0.20, "voice_coverage_ratio": 1.0, "decision": "assign"},
            {"cue_index": 2, "speaker": "SPEAKER_A", "score": 0.84, "margin": 0.23, "voice_coverage_ratio": 1.0, "decision": "assign"},
        ],
    })

    ipc_module = __import__("scribe_py.ipc", fromlist=["_project_speaker_cues"])
    out = ipc_module._project_speaker_cues(candidate)

    projected = out["segments"][0]
    assert [cue["speaker"] for cue in projected["speaker_cues"]] == [
        "SPEAKER_B",
        "SPEAKER_A",
        "SPEAKER_A",
    ]
    assert projected["speaker_cues"][0]["source"] == "campp_sliding_rescue_weak_exact"


def test_discourse_continuity_does_not_override_acoustic_votes():
    candidate = _candidate_from_labels(
        3,
        ["SPEAKER_A", "SPEAKER_B", "SPEAKER_A", "SPEAKER_C"],
        [10.0, 8.0, 10.0, 12.0],
    )
    candidate["segments"][0]["text"] = "所以我们不"
    candidate["segments"][1].update({
        "text": "否认这件事本身很正常",
        "speaker_votes": {"SPEAKER_B": 3.8, "SPEAKER_A": 0.2},
        "speaker_confidence": 0.95,
        "voice_pitch_hz": 128.0,
        "voice_pitch_confidence": 0.9,
        "voice_band": "low",
    })
    candidate["segments"][2].update({
        "text": "这不很正常吗",
        "voice_pitch_hz": 220.0,
        "voice_pitch_confidence": 0.9,
        "voice_band": "high",
    })
    candidate["segments"][0].update({
        "voice_pitch_hz": 220.0,
        "voice_pitch_confidence": 0.9,
        "voice_band": "high",
    })
    candidate["summary"] = __import__("scribe_py.ipc", fromlist=["_speaker_summary"])._speaker_summary(candidate["segments"])
    candidate.update(_score_diarization_candidate(candidate, 3))

    out = _repair_discourse_continuity_assignments(candidate)

    assert out["segments"][1]["speaker"] == "SPEAKER_B"
    assert any("声线/男女特征不支持安全合并" in item["reason"] for item in out.get("review_segments", []))


def test_build_review_segments_includes_local_assignment_review_for_stable_candidates():
    candidate = _candidate_from_labels(
        3,
        ["SPEAKER_A"] * 6
        + ["SPEAKER_C"]
        + ["SPEAKER_A"] * 6
        + ["SPEAKER_B"] * 8,
        [3.0] * 21,
    )

    review = _build_review_segments(candidate)

    assert review
    assert any(item["from_speaker"] == "SPEAKER_C" for item in review)
    assert any("局部夹心跳变" in item["reason"] for item in review)


def test_near_tie_prefers_merged_candidate_with_same_actual_count():
    plain = _candidate(3, [
        {"speaker": "SPEAKER_A", "segments": 126, "duration_s": 412.8, "turns": 38, "stable_turns": 26},
        {"speaker": "SPEAKER_B", "segments": 86, "duration_s": 258.5, "turns": 22, "stable_turns": 13},
        {"speaker": "SPEAKER_C", "segments": 57, "duration_s": 124.3, "turns": 21, "stable_turns": 12},
    ])
    merged = _candidate(4, [
        {"speaker": "SPEAKER_A", "segments": 128, "duration_s": 415.1, "turns": 38, "stable_turns": 26},
        {"speaker": "SPEAKER_B", "segments": 102, "duration_s": 280.6, "turns": 24, "stable_turns": 15},
        {"speaker": "SPEAKER_D", "segments": 39, "duration_s": 99.9, "turns": 12, "stable_turns": 7},
    ])
    plain["score"] = 21.0
    merged["score"] = 16.2
    merged["actual_n_speakers"] = 3
    merged["merge_map"] = {"SPEAKER_C": "SPEAKER_B"}

    best = _choose_diarization_candidate([plain, merged])

    assert best is merged


def test_real_low_frequency_speaker_can_keep_four_person_recommendation():
    two = _candidate(2, [
        {"speaker": "SPEAKER_A", "segments": 413, "duration_s": 786.1, "turns": 86, "stable_turns": 55},
        {"speaker": "SPEAKER_B", "segments": 11, "duration_s": 29.6, "turns": 2, "stable_turns": 1},
    ])
    three = _candidate(3, [
        {"speaker": "SPEAKER_A", "segments": 330, "duration_s": 627.5, "turns": 76, "stable_turns": 43},
        {"speaker": "SPEAKER_B", "segments": 84, "duration_s": 160.6, "turns": 24, "stable_turns": 12},
        {"speaker": "SPEAKER_C", "segments": 10, "duration_s": 27.6, "turns": 1, "stable_turns": 1},
    ])
    four = _candidate(4, [
        {"speaker": "SPEAKER_A", "segments": 331, "duration_s": 628.6, "turns": 72, "stable_turns": 43},
        {"speaker": "SPEAKER_B", "segments": 45, "duration_s": 96.1, "turns": 12, "stable_turns": 8},
        {"speaker": "SPEAKER_C", "segments": 38, "duration_s": 63.3, "turns": 9, "stable_turns": 5},
        {"speaker": "SPEAKER_D", "segments": 10, "duration_s": 27.6, "turns": 1, "stable_turns": 1},
    ])

    best = _choose_diarization_candidate([two, three, four])

    assert best is four
    assert four["fragmented_speakers"] == 0


def test_model_silhouette_prevents_two_person_recording_from_being_recommended_as_eight():
    two = _candidate(2, [
        {"speaker": "SPEAKER_A", "segments": 32, "duration_s": 190.1, "turns": 18, "stable_turns": 12},
        {"speaker": "SPEAKER_B", "segments": 75, "duration_s": 409.9, "turns": 23, "stable_turns": 15},
    ])
    eight = _candidate(8, [
        {"speaker": "SPEAKER_A", "segments": 8, "duration_s": 46.1, "turns": 6, "stable_turns": 4},
        {"speaker": "SPEAKER_B", "segments": 22, "duration_s": 122.3, "turns": 9, "stable_turns": 7},
        {"speaker": "SPEAKER_C", "segments": 12, "duration_s": 68.0, "turns": 7, "stable_turns": 5},
        {"speaker": "SPEAKER_D", "segments": 17, "duration_s": 75.3, "turns": 8, "stable_turns": 6},
        {"speaker": "SPEAKER_E", "segments": 20, "duration_s": 101.0, "turns": 8, "stable_turns": 7},
        {"speaker": "SPEAKER_F", "segments": 11, "duration_s": 74.7, "turns": 6, "stable_turns": 5},
        {"speaker": "SPEAKER_G", "segments": 9, "duration_s": 69.5, "turns": 5, "stable_turns": 4},
        {"speaker": "SPEAKER_H", "segments": 8, "duration_s": 43.0, "turns": 5, "stable_turns": 4},
    ])
    for candidate, selected_score in [(two, 0.1666), (eight, 0.0636)]:
        candidate["stats"] = {
            "model_recommended_n_speakers": 2,
            "model_recommended_score": 0.1666,
            "model_selected_score": selected_score,
        }

    best = _choose_diarization_candidate([two, eight])

    assert best is two


def test_model_anchor_prefers_exact_requested_count_over_collapsed_higher_runs():
    two = _candidate(2, [
        {"speaker": "SPEAKER_A", "segments": 74, "duration_s": 635.0, "turns": 10, "stable_turns": 9},
        {"speaker": "SPEAKER_B", "segments": 25, "duration_s": 170.0, "turns": 8, "stable_turns": 7},
    ])
    four = _candidate(4, [
        {"speaker": "SPEAKER_A", "segments": 25, "duration_s": 264.0, "turns": 6, "stable_turns": 4},
        {"speaker": "SPEAKER_B", "segments": 35, "duration_s": 266.0, "turns": 15, "stable_turns": 11},
        {"speaker": "SPEAKER_C", "segments": 13, "duration_s": 91.0, "turns": 7, "stable_turns": 6},
        {"speaker": "SPEAKER_D", "segments": 26, "duration_s": 184.0, "turns": 9, "stable_turns": 8},
    ])
    collapsed_seven = _candidate(7, [
        {"speaker": "SPEAKER_A", "segments": 25, "duration_s": 264.0, "turns": 6, "stable_turns": 4},
        {"speaker": "SPEAKER_B", "segments": 35, "duration_s": 266.0, "turns": 15, "stable_turns": 11},
        {"speaker": "SPEAKER_C", "segments": 13, "duration_s": 91.0, "turns": 7, "stable_turns": 6},
        {"speaker": "SPEAKER_D", "segments": 26, "duration_s": 184.0, "turns": 9, "stable_turns": 8},
    ])
    collapsed_seven["actual_n_speakers"] = 4
    two["score"] = 15.0
    four["score"] = 12.6
    collapsed_seven["score"] = -2.0
    for candidate, selected_score in [
        (two, 0.35),
        (four, 0.75),
        (collapsed_seven, 0.06),
    ]:
        candidate["stats"] = {
            "model_recommended_n_speakers": 4,
            "model_recommended_score": 0.75,
            "model_selected_score": selected_score,
            "model_recommended_confidence": 0.77,
            "model_recommended_confidence_level": "high",
        }

    best = _choose_diarization_candidate([two, four, collapsed_seven])

    assert best is four
    assert "底层聚类更支持 4 人" in four["model_guard_reason"]


def test_model_two_person_bias_can_keep_structural_four_person_refinement():
    two = _candidate_from_labels(
        2,
        ["SPEAKER_A"] * 90 + ["SPEAKER_B"] * 240,
        [4.0] * 90 + [4.0] * 240,
    )
    four = _candidate_from_labels(
        4,
        ["SPEAKER_A"] * 90
        + ["SPEAKER_B", "SPEAKER_C"] * 80
        + ["SPEAKER_D"] * 80,
        [4.0] * 330,
    )
    for candidate, selected_score in [(two, 0.24), (four, 0.11)]:
        candidate["stats"] = {
            "model_recommended_n_speakers": 2,
            "model_recommended_score": 0.24,
            "model_selected_score": selected_score,
        }
    four["score"] = 27.0

    best = _choose_diarization_candidate([two, four])

    assert best is four
    assert "混合说话人" in four["refinement_reason"]
    assert "底层聚类更支持 2 人" in four["model_guard_reason"]


def test_model_three_person_anchor_prefers_minimal_structural_refinement():
    three = _candidate_from_labels(
        3,
        ["SPEAKER_A"] * 70 + ["SPEAKER_B"] * 50 + ["SPEAKER_C"] * 20,
        [4.0] * 140,
    )
    four = _candidate_from_labels(
        4,
        ["SPEAKER_A"] * 35
        + ["SPEAKER_D"] * 35
        + ["SPEAKER_B"] * 50
        + ["SPEAKER_C"] * 20,
        [4.0] * 140,
    )
    five = _candidate_from_labels(
        5,
        ["SPEAKER_A"] * 35
        + ["SPEAKER_D"] * 35
        + ["SPEAKER_B"] * 25
        + ["SPEAKER_E"] * 25
        + ["SPEAKER_C"] * 20,
        [4.0] * 140,
    )
    for candidate, selected_score in [(three, 0.21), (four, 0.16), (five, 0.16)]:
        candidate["stats"] = {
            "model_recommended_n_speakers": 3,
            "model_recommended_score": 0.21,
            "model_selected_score": selected_score,
        }
    three["score"] = 21.0
    four["score"] = 27.0
    five["score"] = 33.0

    best = _choose_diarization_candidate([three, four, five])

    assert best is four
    assert "混合说话人" in four["refinement_reason"]
    assert "底层聚类更支持 3 人" in four["model_guard_reason"]


def test_model_two_person_bias_rejects_time_drift_over_split():
    two = _candidate_from_labels(
        2,
        ["SPEAKER_A"] * 120 + ["SPEAKER_B"] * 120,
        [4.0] * 240,
    )
    four = _candidate_from_labels(
        4,
        ["SPEAKER_A"] * 60
        + ["SPEAKER_C"] * 60
        + ["SPEAKER_B"] * 60
        + ["SPEAKER_D"] * 60,
        [4.0] * 240,
    )
    for candidate, selected_score in [(two, 0.24), (four, 0.11)]:
        candidate["stats"] = {
            "model_recommended_n_speakers": 2,
            "model_recommended_score": 0.24,
            "model_selected_score": selected_score,
        }
    four["score"] = 27.0

    best = _choose_diarization_candidate([two, four])

    assert best is two
    assert "refinement_reason" not in four


def test_model_prior_conservatively_raises_severely_mixed_under_count_by_one():
    candidates = [
        _candidate(
            count,
            [
                {
                    "speaker": f"SPEAKER_{chr(65 + index)}",
                    "segments": 40,
                    "duration_s": 180.0,
                    "turns": 12,
                    "stable_turns": 10,
                }
                for index in range(count)
            ],
        )
        for count in range(2, 6)
    ]
    for candidate, selected_score in zip(candidates, [0.62, 0.27, 0.49, 0.05], strict=True):
        candidate["stats"] = {
            "model_recommended_n_speakers": 4,
            "model_recommended_score": 0.49,
            "model_selected_score": selected_score,
            "model_recommended_confidence": 0.51,
            "model_recommended_confidence_level": "medium",
        }
    two, three, four, five = candidates
    two["score"] = -31.0
    three["score"] = -48.0
    four["score"] = -47.0
    five["score"] = -60.0
    two["mixed_voice_speakers"] = ["SPEAKER_A", "SPEAKER_B"]
    two["severe_mixed_voice_speakers"] = ["SPEAKER_A", "SPEAKER_B"]

    best = _choose_diarization_candidate(candidates)

    assert best is three
    assert "保守上调到 3 人" in three["model_guard_reason"]


def test_medium_confidence_unresolved_four_person_split_falls_back_to_clean_three():
    candidates = [
        _candidate(
            count,
            [
                {
                    "speaker": f"SPEAKER_{chr(65 + index)}",
                    "segments": 20,
                    "duration_s": 90.0,
                    "turns": 8,
                    "stable_turns": 6,
                }
                for index in range(count)
            ],
        )
        for count in range(2, 6)
    ]
    for candidate, selected_score in zip(candidates, [0.62, 0.27, 0.49, 0.08], strict=True):
        candidate["stats"] = {
            "model_recommended_n_speakers": 4,
            "model_recommended_score": 0.49,
            "model_selected_score": selected_score,
            "model_recommended_confidence": 0.51,
            "model_recommended_confidence_level": "medium",
        }
    two, three, four, five = candidates
    two["score"] = -31.0
    three["score"] = -48.0
    four["score"] = -19.0
    five["score"] = -60.0
    four["severe_mixed_voice_speakers"] = ["SPEAKER_A", "SPEAKER_B"]
    four["mixed_voice_speakers"] = ["SPEAKER_A", "SPEAKER_B"]

    best = _choose_diarization_candidate(candidates)

    assert best is three
    assert "中等置信" in three["model_guard_reason"]
    assert "3 人结果" in three["model_guard_reason"]


def test_high_confidence_four_person_split_is_not_downgraded():
    three = _candidate(3, [
        {"speaker": "SPEAKER_A", "segments": 30, "duration_s": 150.0, "turns": 10, "stable_turns": 8},
        {"speaker": "SPEAKER_B", "segments": 30, "duration_s": 150.0, "turns": 10, "stable_turns": 8},
        {"speaker": "SPEAKER_C", "segments": 30, "duration_s": 150.0, "turns": 10, "stable_turns": 8},
    ])
    four = _candidate(4, [
        {"speaker": "SPEAKER_A", "segments": 24, "duration_s": 120.0, "turns": 8, "stable_turns": 6},
        {"speaker": "SPEAKER_B", "segments": 24, "duration_s": 120.0, "turns": 8, "stable_turns": 6},
        {"speaker": "SPEAKER_C", "segments": 24, "duration_s": 120.0, "turns": 8, "stable_turns": 6},
        {"speaker": "SPEAKER_D", "segments": 24, "duration_s": 120.0, "turns": 8, "stable_turns": 6},
    ])
    five = _candidate(5, [
        {"speaker": f"SPEAKER_{chr(65 + index)}", "segments": 18, "duration_s": 90.0, "turns": 6, "stable_turns": 5}
        for index in range(5)
    ])
    for candidate, selected_score in [(three, 0.20), (four, 0.91), (five, 0.04)]:
        candidate["stats"] = {
            "model_recommended_n_speakers": 4,
            "model_recommended_score": 0.91,
            "model_selected_score": selected_score,
            "model_recommended_confidence": 0.92,
            "model_recommended_confidence_level": "high",
        }
    three["score"] = 18.0
    four["score"] = 25.0
    five["score"] = 12.0
    four["severe_mixed_voice_speakers"] = ["SPEAKER_A", "SPEAKER_B"]
    four["mixed_voice_speakers"] = ["SPEAKER_A", "SPEAKER_B"]

    best = _choose_diarization_candidate([three, four, five])

    assert best is four


def test_consistent_high_count_model_anchor_keeps_durable_low_share_speakers():
    candidates = [
        _candidate(
            count,
            [
                {
                    "speaker": f"SPEAKER_{chr(65 + index)}",
                    "segments": 120 if index == 0 else 6,
                    "duration_s": 840.0 if index == 0 else 32.0 + index * 8.0,
                    "turns": 16 if index == 0 else 2,
                    "stable_turns": 15 if index == 0 else 1,
                }
                for index in range(count)
            ],
        )
        for count in range(2, 8)
    ]
    for candidate in candidates:
        candidate["stats"] = {
            "model_recommended_n_speakers": 6,
            "model_recommended_score": 0.38,
            "model_selected_score": 0.30,
            "model_recommended_confidence": 0.44,
        }
        candidate["score"] = -float(candidate["n_speakers"])
    six = candidates[4]
    six["stats"]["model_selected_score"] = 0.38
    six["weak_speakers"] = 3
    six["stable_speakers"] = 3
    six["tiny_speakers"] = 0
    six["fragmented_speakers"] = 0
    six["marginal_speakers"] = 0

    best = _choose_diarization_candidate(candidates)

    assert best is six
    assert "每位说话人均有独立持续声纹证据" in six["model_guard_reason"]


def test_high_count_model_anchor_rejects_fragmented_extra_speaker():
    candidates = [
        _candidate(
            count,
            [
                {
                    "speaker": f"SPEAKER_{chr(65 + index)}",
                    "segments": 30,
                    "duration_s": 120.0,
                    "turns": 8,
                    "stable_turns": 6,
                }
                for index in range(count)
            ],
        )
        for count in range(3, 7)
    ]
    for candidate in candidates:
        candidate["stats"] = {
            "model_recommended_n_speakers": 5,
            "model_recommended_score": 0.45,
            "model_selected_score": 0.30,
            "model_recommended_confidence": 0.50,
        }
        candidate["score"] = 20.0 - candidate["n_speakers"]
    three = candidates[0]
    five = candidates[2]
    five["fragmented_speakers"] = 1

    best = _choose_diarization_candidate(candidates)

    assert best is not five
    assert best is three


def test_unresolved_severe_voice_mix_does_not_promote_fragile_fourth_speaker():
    three = _candidate_from_labels(
        3,
        ["SPEAKER_A"] * 26 + ["SPEAKER_B"] * 46 + ["SPEAKER_D"] * 67,
        [5.0] * 139,
    )
    four = _candidate_from_labels(
        4,
        ["SPEAKER_A"] * 13
        + ["SPEAKER_C"] * 13
        + ["SPEAKER_B"] * 46
        + ["SPEAKER_D"] * 67,
        [5.0] * 139,
    )
    three["mixed_voice_speakers"] = ["SPEAKER_A", "SPEAKER_B", "SPEAKER_D"]
    three["severe_mixed_voice_speakers"] = ["SPEAKER_A", "SPEAKER_B", "SPEAKER_D"]
    four["mixed_voice_speakers"] = ["SPEAKER_A", "SPEAKER_B", "SPEAKER_D"]
    four["severe_mixed_voice_speakers"] = ["SPEAKER_A", "SPEAKER_B", "SPEAKER_D"]
    four["fragile_speakers"] = ["SPEAKER_C"]
    four["mergeable_speakers"] = ["SPEAKER_C"]
    four["tiny_speakers"] = 0
    four["weak_speakers"] = 0
    four["fragmented_speakers"] = 0
    four["stable_speakers"] = 3

    assert not _candidate_has_meaningful_refinement(three, four)


def test_low_confidence_model_prior_does_not_raise_under_count():
    candidates = [
        _candidate(
            count,
            [
                {
                    "speaker": f"SPEAKER_{chr(65 + index)}",
                    "segments": 40,
                    "duration_s": 180.0,
                    "turns": 12,
                    "stable_turns": 10,
                }
                for index in range(count)
            ],
        )
        for count in range(2, 6)
    ]
    for candidate in candidates:
        candidate["stats"] = {
            "model_recommended_n_speakers": 4,
            "model_recommended_score": 0.49,
            "model_selected_score": 0.30,
            "model_recommended_confidence": 0.20,
            "model_recommended_confidence_level": "low",
        }
        candidate["score"] = -60.0
    two = candidates[0]
    two["score"] = -31.0
    two["mixed_voice_speakers"] = ["SPEAKER_A", "SPEAKER_B"]
    two["severe_mixed_voice_speakers"] = ["SPEAKER_A", "SPEAKER_B"]

    best = _choose_diarization_candidate(candidates)

    assert best is two


def test_refined_four_person_candidate_can_beat_clean_three_person_mixed_bucket():
    three = _candidate_from_labels(
        3,
        ["SPEAKER_A"] * 80
        + ["SPEAKER_B"] * 60
        + ["SPEAKER_C"] * 34
        + ["SPEAKER_C"] * 12,
        [4.0] * 80 + [3.5] * 60 + [3.0] * 34 + [1.6] * 12,
    )
    four = _candidate_from_labels(
        4,
        ["SPEAKER_A"] * 80
        + ["SPEAKER_B"] * 60
        + ["SPEAKER_D"] * 34
        + ["SPEAKER_C"] * 12,
        [4.0] * 80 + [3.5] * 60 + [3.0] * 34 + [1.6] * 12,
    )
    # Mirror the real "标准录音 10" shape: three speakers looks clean, while
    # four speakers exposes one stable refined speaker plus one small review
    # cluster. Product behavior should preserve the refinement.
    three["score"] = 21.0
    four["score"] = 14.2
    four["stable_speakers"] = 3
    four["weak_speakers"] = 0
    four["tiny_speakers"] = 0
    four["fragmented_speakers"] = 1
    four["fragile_speakers"] = ["SPEAKER_C"]
    four["mergeable_speakers"] = ["SPEAKER_C"]

    best = _choose_diarization_candidate([three, four])

    assert best is four
    assert "混合说话人" in four["refinement_reason"]


def test_confidence_is_high_when_winner_has_clear_margin_and_no_fragile_speakers():
    two = _candidate(2, [
        {"speaker": "SPEAKER_A", "segments": 89, "duration_s": 151.5, "turns": 24, "stable_turns": 15},
        {"speaker": "SPEAKER_B", "segments": 47, "duration_s": 78.5, "turns": 21, "stable_turns": 8},
    ])
    bad = _candidate(5, [
        {"speaker": "SPEAKER_A", "segments": 89, "duration_s": 151.5, "turns": 24, "stable_turns": 15},
        {"speaker": "SPEAKER_B", "segments": 47, "duration_s": 78.5, "turns": 21, "stable_turns": 8},
        {"speaker": "SPEAKER_C", "segments": 1, "duration_s": 1.0, "turns": 1, "stable_turns": 0},
    ])

    confidence, _, _ = _recommendation_confidence([two, bad], two)

    assert confidence == "high"


def test_segment_review_annotation_filters_noisy_existing_flags():
    candidate = {
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "speaker": "SPEAKER_A",
                "text": "弱风险不应该刷满界面。",
                "speaker_assignment_review": True,
                "speaker_review_reason": "推荐置信度不足，建议抽听确认",
            },
            {
                "start": 3.0,
                "end": 6.0,
                "speaker": "SPEAKER_B",
                "text": "已经纠偏的段不要继续提示。",
                "speaker_assignment_review": True,
                "speaker_review_reason": "声线复核：强高低声线夹心错挂，且短窗声纹投票支持目标说话人，已纠偏",
                "continuity_repaired": True,
            },
            {
                "start": 6.0,
                "end": 9.0,
                "speaker": "SPEAKER_C",
                "text": "强风险仍然需要提示。",
                "speaker_assignment_review": True,
                "speaker_review_reason": "声线复核：片段音高与当前说话人画像冲突，但缺少安全改派目标，建议抽听确认",
            },
        ],
        "review_segments": [
            {
                "index": 0,
                "start": 0.0,
                "end": 3.0,
                "from_speaker": "SPEAKER_A",
                "to_speaker": "SPEAKER_A",
                "reason": "段内短声纹窗疑似换人，但证据重叠/接近，已保留原分人并标为待确认",
            },
            {
                "index": 2,
                "start": 6.0,
                "end": 9.0,
                "from_speaker": "SPEAKER_C",
                "to_speaker": "SPEAKER_D",
                "reason": "声线复核：片段音高与当前说话人画像冲突，但缺少安全改派目标，建议抽听确认",
            },
        ],
    }

    out = _annotate_segments_with_speaker_reviews(candidate)

    assert "speaker_assignment_review" not in out["segments"][0]
    assert "speaker_review_reason" not in out["segments"][0]
    assert "speaker_assignment_review" not in out["segments"][1]
    assert out["segments"][2]["speaker_assignment_review"] is True
    assert "声线复核" in out["segments"][2]["speaker_review_reason"]


if __name__ == "__main__":
    test_fragmented_extra_speakers_do_not_beat_clean_two_person_call()
    test_fragmented_extra_speakers_merge_into_stable_neighbors()
    test_fragile_segments_use_lower_count_anchor_instead_of_context_only()
    test_short_filler_sandwiched_speaker_is_treated_as_fragmented()
    test_weak_but_coherent_third_speaker_is_not_auto_mergeable()
    test_low_duration_fragment_should_not_make_five_person_candidate_win()
    test_near_tie_prefers_merged_candidate_with_same_actual_count()
    test_real_low_frequency_speaker_can_keep_four_person_recommendation()
    test_model_silhouette_prevents_two_person_recording_from_being_recommended_as_eight()
    test_confidence_is_high_when_winner_has_clear_margin_and_no_fragile_speakers()
    print("diarization recommendation tests passed")
