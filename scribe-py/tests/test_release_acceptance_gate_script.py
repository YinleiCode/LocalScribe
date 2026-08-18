from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "release_acceptance_gate.py"
SPEC = importlib.util.spec_from_file_location("release_acceptance_gate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_transcript(path: Path, *, risk: str = "low", review: bool = False) -> None:
    payload = {
        "backend": "sensevoice",
        "model_id": "iic/SenseVoiceSmall",
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "这是一段测试。",
                "sync_cues": [{"start": 0.0, "end": 2.0, "text": "这是一段测试。"}],
                "speaker": "SPEAKER_A",
            }
        ],
        "asr_quality": {
            "traditional_char_hits": [],
            "audio_quality": {"risk_level": risk},
        },
        "filter_stats": {
            "strong_asr": {"review_recommended": review, "enabled": False, "applied": False},
        },
        "diarization_stats": {
            "status": "ok",
            "applied": True,
            "runtime_backend": "coreml_pyannote_campp",
            "segmentation_preserved": True,
            "risk_level": "low",
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_unseen_low_risk_recording_passes_structural_gate(tmp_path: Path):
    transcript = tmp_path / "candidate.json"
    _write_transcript(transcript)

    result = MODULE.evaluate_case("未见录音", transcript)

    assert result["ok"] is True
    assert result["summary"]["status"] == "PASS"
    assert result["summary"]["speaker_count"] == 1


def test_high_risk_recording_must_surface_review(tmp_path: Path):
    transcript = tmp_path / "candidate.json"
    _write_transcript(transcript, risk="high", review=False)

    result = MODULE.evaluate_case("高噪声录音", transcript)

    assert result["ok"] is False
    assert {item["code"] for item in result["failures"]} == {"high_risk_audio_not_surfaced"}

    _write_transcript(transcript, risk="high", review=True)
    assert MODULE.evaluate_case("高噪声录音", transcript)["ok"] is True


def test_frozen_baseline_rejects_text_or_cursor_change(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_transcript(baseline)
    _write_transcript(candidate)
    assert MODULE.evaluate_case("冻结录音", candidate, baseline_path=baseline)["ok"] is True

    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["segments"][0]["text"] = "文字发生变化。"
    payload["segments"][0]["sync_cues"][0]["text"] = "文字发生变化。"
    candidate.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    result = MODULE.evaluate_case("冻结录音", candidate, baseline_path=baseline)
    assert result["ok"] is False
    assert "frozen_asr_or_cursor_regression" in {item["code"] for item in result["failures"]}


def test_release_gate_rejects_candidate_as_its_own_baseline(tmp_path: Path):
    candidate = tmp_path / "candidate.json"
    _write_transcript(candidate)

    result = MODULE.evaluate_case("错误自基线", candidate, baseline_path=candidate)

    assert result["ok"] is False
    assert "self_baseline_rejected" in {item["code"] for item in result["failures"]}


def test_release_gate_can_require_an_independent_baseline(tmp_path: Path):
    candidate = tmp_path / "candidate.json"
    _write_transcript(candidate)

    result = MODULE.evaluate_case("已知录音", candidate, require_baseline=True)

    assert result["ok"] is False
    assert "required_baseline_missing" in {item["code"] for item in result["failures"]}


def test_release_gate_rejects_zero_duration_cursor_cue(tmp_path: Path):
    transcript = tmp_path / "candidate.json"
    _write_transcript(transcript)
    payload = json.loads(transcript.read_text(encoding="utf-8"))
    payload["segments"][0]["sync_cues"][0]["end"] = 0.0
    transcript.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    result = MODULE.evaluate_case("零时长光标", transcript)

    assert result["ok"] is False
    assert result["summary"]["zero_duration_cues"] == 1
    assert "cursor_sync_invalid" in {item["code"] for item in result["failures"]}


def test_release_gate_rejects_overlapping_cursor_cues(tmp_path: Path):
    transcript = tmp_path / "candidate.json"
    _write_transcript(transcript)
    payload = json.loads(transcript.read_text(encoding="utf-8"))
    payload["segments"][0]["sync_cues"] = [
        {"start": 0.0, "end": 1.5, "text": "这是一段"},
        {"start": 1.0, "end": 2.0, "text": "测试。"},
    ]
    transcript.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    result = MODULE.evaluate_case("重叠光标", transcript)

    assert result["ok"] is False
    assert result["summary"]["overlapping_cues"] == 1
    assert "cursor_sync_invalid" in {item["code"] for item in result["failures"]}
