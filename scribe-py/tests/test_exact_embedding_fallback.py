from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scribe_py.diarizers import exact_embedding_fallback as mod


def _segment(start: float, speaker: str, *, missing: bool = False, duration: float = 2.0) -> dict:
    end = start + duration
    segment = {
        "start": start,
        "end": end,
        "text": f"{speaker}-{start}",
        "speaker": speaker,
        "speaker_confidence": 0.92,
        "sync_cues": [
            {
                "start": start,
                "end": end,
                "text": f"{speaker}-{start}",
                "words": [{"word": speaker, "start": start, "end": end}],
            }
        ],
    }
    if not missing:
        segment["speaker_subsegments"] = [
            {"start": start, "end": end, "speaker": speaker, "duration": duration}
        ]
    return segment


def _candidate(*, target_duration: float = 6.0, target_has_evidence: bool = False) -> dict:
    segments = [
        _segment(0.0, "SPEAKER_A"),
        _segment(4.0, "SPEAKER_A"),
        _segment(8.0, "SPEAKER_A"),
        _segment(12.0, "SPEAKER_B"),
        _segment(16.0, "SPEAKER_B"),
        _segment(20.0, "SPEAKER_B"),
        _segment(
            30.0,
            "SPEAKER_A",
            missing=not target_has_evidence,
            duration=target_duration,
        ),
    ]
    return {
        "segments": segments,
        "stats": {"engine": "senko"},
        "summary": {"speakers": ["SPEAKER_A", "SPEAKER_B"]},
    }


def _install_fake_runtime(monkeypatch, tmp_path: Path, *, target_vectors: list[np.ndarray]) -> Path:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio")
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(mod, "resolve_local_model_path", lambda: model)
    monkeypatch.setattr(
        mod,
        "_decode_audio_16k",
        lambda _audio: np.zeros(40 * mod.SAMPLE_RATE, dtype=np.float32),
    )
    monkeypatch.setattr(mod, "_get_pipeline", lambda _path: object())

    speaker_a = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    speaker_b = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    def fake_embeddings(_pipeline, waveforms):
        values = [speaker_a] * 3 + [speaker_b] * 3 + list(target_vectors)
        assert len(values) == len(waveforms)
        return np.stack(values)

    monkeypatch.setattr(mod, "_run_embeddings", fake_embeddings)
    return audio


def _frozen_geometry(candidate: dict) -> list[dict]:
    return [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "sync_cues": segment["sync_cues"],
        }
        for segment in candidate["segments"]
    ]


def test_high_confidence_missing_cue_changes_only_speaker_timeline(monkeypatch, tmp_path):
    speaker_b = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    candidate = _candidate()
    original = json.loads(json.dumps(candidate))
    frozen = _frozen_geometry(candidate)
    audio = _install_fake_runtime(
        monkeypatch,
        tmp_path,
        target_vectors=[speaker_b, speaker_b, speaker_b],
    )

    repaired = mod.repair_missing_evidence_cues(audio, candidate)

    target = repaired["segments"][-1]
    assert target["speaker"] == "SPEAKER_A"
    assert target["speaker_cues"][0]["speaker"] == "SPEAKER_B"
    assert target["speaker_cues"][0]["source"] == "campp_exact_missing_evidence"
    assert _frozen_geometry(repaired) == frozen
    assert candidate == original
    stats = repaired["stats"]["exact_embedding_fallback"]
    assert stats["applied"] is True
    assert stats["changed_cues"] == 1
    assert stats["changed_seconds"] == 6.0


def test_window_disagreement_fails_closed(monkeypatch, tmp_path):
    speaker_a = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    speaker_b = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    candidate = _candidate()
    audio = _install_fake_runtime(
        monkeypatch,
        tmp_path,
        target_vectors=[speaker_b, speaker_a, speaker_b],
    )

    repaired = mod.repair_missing_evidence_cues(audio, candidate)

    assert "speaker_cues" not in repaired["segments"][-1]
    stats = repaired["stats"]["exact_embedding_fallback"]
    assert stats["applied"] is False
    assert stats["reason"] == "no_proposal_passed_gates"


def test_cue_with_primary_evidence_never_runs_fallback(monkeypatch, tmp_path):
    candidate = _candidate(target_has_evidence=True)
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        mod,
        "resolve_local_model_path",
        lambda: (_ for _ in ()).throw(AssertionError("model must not be loaded")),
    )

    repaired = mod.repair_missing_evidence_cues(audio, candidate)

    assert repaired["segments"] == candidate["segments"]
    stats = repaired["stats"]["exact_embedding_fallback"]
    assert stats["available"] is True
    assert stats["reason"] == "no_long_missing_evidence_cues"


def test_short_missing_cue_never_runs_fallback(monkeypatch, tmp_path):
    candidate = _candidate(target_duration=3.99)
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        mod,
        "resolve_local_model_path",
        lambda: (_ for _ in ()).throw(AssertionError("model must not be loaded")),
    )

    repaired = mod.repair_missing_evidence_cues(audio, candidate)

    assert repaired["segments"] == candidate["segments"]
    assert repaired["stats"]["exact_embedding_fallback"]["candidate_cues"] == 0


def test_non_campp_primary_engine_never_runs_fallback(monkeypatch, tmp_path):
    candidate = _candidate()
    candidate["stats"]["engine"] = "pyannote"
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        mod,
        "resolve_local_model_path",
        lambda: (_ for _ in ()).throw(AssertionError("model must not be loaded")),
    )

    repaired = mod.repair_missing_evidence_cues(audio, candidate)

    assert repaired["segments"] == candidate["segments"]
    stats = repaired["stats"]["exact_embedding_fallback"]
    assert stats["reason"] == "primary_engine_not_campp"
    assert stats["primary_engine"] == "pyannote"


def test_human_confirmed_target_never_runs_fallback(monkeypatch, tmp_path):
    candidate = _candidate()
    target_index = len(candidate["segments"]) - 1
    candidate["human_annotation_reuse"] = {
        "rows": [{"index": target_index, "correct_speaker": "SPEAKER_A"}]
    }
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        mod,
        "resolve_local_model_path",
        lambda: (_ for _ in ()).throw(AssertionError("model must not be loaded")),
    )

    repaired = mod.repair_missing_evidence_cues(audio, candidate)

    assert repaired["segments"] == candidate["segments"]
    stats = repaired["stats"]["exact_embedding_fallback"]
    assert stats["candidate_cues"] == 0
    assert stats["reason"] == "no_long_missing_evidence_cues"


def test_missing_model_fails_closed_without_changing_segments(monkeypatch, tmp_path):
    candidate = _candidate()
    original_segments = json.loads(json.dumps(candidate["segments"]))
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        mod,
        "resolve_local_model_path",
        lambda: (_ for _ in ()).throw(FileNotFoundError("not bundled")),
    )

    repaired = mod.repair_missing_evidence_cues(audio, candidate)

    assert repaired["segments"] == original_segments
    stats = repaired["stats"]["exact_embedding_fallback"]
    assert stats["available"] is False
    assert stats["applied"] is False
    assert stats["reason"] == "fallback_unavailable"


def test_ipc_geometry_guard_rejects_bad_fallback(monkeypatch, tmp_path):
    from scribe_py import ipc

    candidate = _candidate()
    original_segments = json.loads(json.dumps(candidate["segments"]))
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio")

    def bad_fallback(_audio, value, *, on_progress=None):
        output = json.loads(json.dumps(value))
        output["segments"][0]["text"] = "mutated"
        output["segments"][0]["sync_cues"][0]["end"] += 1.0
        return output

    monkeypatch.setattr(mod, "repair_missing_evidence_cues", bad_fallback)

    repaired = ipc._repair_long_missing_speaker_cues(candidate, audio)

    assert repaired["segments"] == original_segments
    assert repaired["stats"]["exact_embedding_fallback"]["reason"] == (
        "transcript_geometry_guard_rejected_output"
    )


def test_local_model_resolver_prefers_bundled_cache(monkeypatch, tmp_path):
    bundled = tmp_path / "resources/modelscope/hub"
    model = bundled / "models/damo/speech_campplus_sv_zh-cn_16k-common"
    model.mkdir(parents=True)
    (model / "campplus_cn_common.bin").write_bytes(b"weights")
    (model / "config.yaml").write_text("model: campp\n", encoding="utf-8")
    monkeypatch.setenv("LOCALSCRIBE_MODELSCOPE_CACHE", str(bundled))
    monkeypatch.delenv("MODELSCOPE_CACHE", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "empty-home"))

    assert mod.resolve_local_model_path() == model.resolve()
