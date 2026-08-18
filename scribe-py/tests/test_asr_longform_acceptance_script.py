from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "asr_longform_acceptance.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("asr_longform_acceptance", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_valid_case(tmp_path: Path) -> Path:
    transcript = tmp_path / "demo.json"
    transcript.write_text(
        json.dumps(
            {
                "audio": str(tmp_path / "demo.wav"),
                "language": "zh",
                "duration": 4.0,
                "transcribe_seconds": 0.4,
                "rtf": 0.1,
                "backend": "sensevoice",
                "model_id": "iic/SenseVoiceSmall",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.5,
                        "text": "第一段。",
                        "speaker": "SPEAKER_A",
                        "sync_cues": [
                            {"start": 0.0, "end": 0.8, "text": "第一"},
                            {"start": 0.8, "end": 1.5, "text": "段。"},
                        ],
                    },
                    {"start": 2.0, "end": 3.8, "text": "第二段。", "speaker": "SPEAKER_B"},
                ],
                "filter_stats": {
                    "audio_standardization": {"mode": "adaptive"},
                    "settings": {"sensevoice_timing_align": True},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    transcript.with_suffix(".txt").write_text(
        "[00:00:00.000 - 00:00:01.500] [SPEAKER_A] 第一段。\n"
        "[00:00:02.000 - 00:00:03.800] [SPEAKER_B] 第二段。\n",
        encoding="utf-8",
    )
    transcript.with_suffix(".srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,500\n[SPEAKER_A] 第一段。\n\n"
        "2\n00:00:02,000 --> 00:00:03,800\n[SPEAKER_B] 第二段。\n",
        encoding="utf-8",
    )
    return transcript


def test_validate_case_accepts_valid_longform_exports(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)

    result = mod.validate_case("demo", transcript, require_exports=True)

    assert result.ok is True
    assert result.errors == []
    assert result.metrics["segments"] == 2
    assert result.metrics["chars"] == 8
    assert result.metrics["sync_cues"] == 2
    assert result.metrics["max_gap_s"] == 0.5
    assert result.exports["txt"].text_matches is True
    assert result.exports["srt"].text_matches is True


def test_validate_case_rejects_backwards_and_out_of_bounds_timestamps(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    data = json.loads(transcript.read_text(encoding="utf-8"))
    data["segments"][1]["start"] = -1
    data["segments"][1]["end"] = 6
    transcript.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    result = mod.validate_case("demo", transcript)

    assert result.ok is False
    assert any("negative timestamp" in error for error in result.errors)
    assert any("exceeds duration" in error for error in result.errors)


def test_validate_case_rejects_sync_cue_outside_segment(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    data = json.loads(transcript.read_text(encoding="utf-8"))
    data["segments"][0]["sync_cues"][1]["end"] = 2.0
    transcript.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    result = mod.validate_case("demo", transcript)

    assert result.ok is False
    assert any("cue 1: timestamp outside segment" in error for error in result.errors)


def test_validate_case_rejects_export_text_mismatch(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    transcript.with_suffix(".srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,500\n错误文字。\n\n"
        "2\n00:00:02,000 --> 00:00:03,800\n第二段。\n",
        encoding="utf-8",
    )

    result = mod.validate_case("demo", transcript, require_exports=True)

    assert result.ok is False
    assert result.exports["srt"].text_matches is False
    assert any("SRT text differs" in error for error in result.errors)


def test_cli_writes_reports_and_returns_failure_for_invalid_case(tmp_path: Path):
    transcript = _write_valid_case(tmp_path)
    data = json.loads(transcript.read_text(encoding="utf-8"))
    data["segments"] = []
    transcript.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    out_dir = tmp_path / "report"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--case",
            f"demo={transcript}",
            "--out-dir",
            str(out_dir),
            "--require-exports",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    report = json.loads((out_dir / "asr_longform_acceptance.json").read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert "transcript has no segments" in report["cases"][0]["errors"]
    assert (out_dir / "asr_longform_acceptance.md").is_file()


def test_validate_case_rejects_unreliable_requested_precise_timing(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    data = json.loads(transcript.read_text(encoding="utf-8"))
    data["filter_stats"].update({
        "timing_mode": "coarse_text_distribution",
        "timing_reliable": False,
        "timing_alignment_ok": False,
    })
    transcript.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    result = mod.validate_case("demo", transcript)

    assert result.ok is False
    assert "precise timing was requested but the result is marked unreliable" in result.errors
    assert "precise timing alignment failed" in result.errors


def _strict_schema2(
    attempted: list[list[float]],
    recognized: list[list[float]],
    failed: list[list[float]],
    *,
    pad_s: float = 0.5,
) -> dict:
    def digest(ranges: list[list[float]]) -> str:
        return hashlib.sha256(
            json.dumps(ranges, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()

    recognized_set = {tuple(item) for item in recognized}
    failed_set = {tuple(item) for item in failed}
    windows = []
    duration = max((end for _start, end in attempted), default=0.0)
    for start, end in attempted:
        core = (start, end)
        windows.append({
            "core_start": start,
            "core_end": end,
            "decode_start": max(0.0, start - pad_s),
            "decode_end": min(duration, end + pad_s),
            "status": "recognized" if core in recognized_set else "failed" if core in failed_set else "invalid",
            "reason": "test",
            "recognition_source": "strict_probe",
            "speech_duration_s": round(end - start, 3),
            "normalized_chars": 2,
            "chars_per_s": 2.0,
        })
    return {
        "coverage_schema_version": 2,
        "strict_core_max_chunk_s": 1.5,
        "strict_decode_context_pad_s": pad_s,
        "strict_probe_windows": windows,
        "strict_probe_windows_truncated": False,
        "strict_partition": {
            "attempted_ranges": attempted,
            "recognized_ranges": recognized,
            "failed_ranges": failed,
            "attempted_partition_sha256": digest(attempted),
            "recognized_partition_sha256": digest(recognized),
            "failed_partition_sha256": digest(failed),
            "covered_count": len(recognized),
            "failed_count": len(failed),
            "partition_valid": sorted(recognized + failed) == attempted,
        },
    }


def _set_speech_coverage(transcript: Path, coverage: dict) -> None:
    data = json.loads(transcript.read_text(encoding="utf-8"))
    data.setdefault("filter_stats", {})["speech_coverage"] = coverage
    transcript.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_speech_coverage_diagnostics_reports_complete_coverage():
    from scribe_py.core.transcriber_funasr import _speech_coverage_diagnostics
    from scribe_py.core.types import Segment

    result = _speech_coverage_diagnostics(
        [(0.0, 10.0)],
        [Segment(start=0.0, end=10.0, text="完整")],
        duration=10.0,
        collar_s=0.0,
    )

    assert result["status"] == "ok"
    assert result["speech_coverage_ratio"] == 1.0
    assert result["uncovered_speech_s"] == 0.0
    assert result["max_uncovered_speech_s"] == 0.0


def test_speech_coverage_diagnostics_detects_middle_gap():
    from scribe_py.core.transcriber_funasr import _speech_coverage_diagnostics
    from scribe_py.core.types import Segment

    result = _speech_coverage_diagnostics(
        [(0.0, 10.0)],
        [
            Segment(start=0.0, end=3.0, text="前段"),
            Segment(start=7.0, end=10.0, text="后段"),
        ],
        duration=10.0,
        collar_s=0.0,
    )

    assert result["speech_coverage_ratio"] == 0.6
    assert result["max_uncovered_speech_s"] == 4.0
    assert result["uncovered_speech_ranges"] == [{"start": 3.0, "end": 7.0, "duration": 4.0}]


def test_speech_coverage_ignores_trailing_silence_after_last_speech():
    from scribe_py.core.transcriber_funasr import _speech_coverage_diagnostics
    from scribe_py.core.types import Segment

    result = _speech_coverage_diagnostics(
        [(0.0, 20.0)],
        [Segment(start=0.0, end=20.0, text="结尾之后都是静音")],
        duration=60.0,
        collar_s=0.0,
    )

    assert result["speech_coverage_ratio"] == 1.0
    assert result["trailing_uncovered_speech_s"] == 0.0


def test_speech_coverage_detects_uncovered_trailing_speech():
    from scribe_py.core.transcriber_funasr import _speech_coverage_diagnostics
    from scribe_py.core.types import Segment

    result = _speech_coverage_diagnostics(
        [(0.0, 22.0)],
        [Segment(start=0.0, end=20.0, text="漏掉最后两秒")],
        duration=60.0,
        collar_s=0.0,
    )

    assert result["trailing_uncovered_speech_s"] == 2.0
    assert result["max_uncovered_speech_s"] == 2.0


def test_speech_coverage_collar_absorbs_small_boundary_difference():
    from scribe_py.core.transcriber_funasr import _speech_coverage_diagnostics
    from scribe_py.core.types import Segment

    result = _speech_coverage_diagnostics(
        [(0.0, 10.0)],
        [Segment(start=0.4, end=9.6, text="边界容差")],
        duration=10.0,
        collar_s=0.5,
    )

    assert result["speech_coverage_ratio"] == 1.0
    assert result["leading_uncovered_speech_s"] == 0.0
    assert result["trailing_uncovered_speech_s"] == 0.0


def test_validate_case_requires_available_speech_coverage(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)

    result = mod.validate_case("demo", transcript, require_speech_coverage=True)

    assert result.ok is False
    assert "speech coverage diagnostics are missing" in result.errors


def test_validate_case_fails_speech_coverage_thresholds(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    _set_speech_coverage(
        transcript,
        {
            "status": "ok",
            "reason": "test",
            "basis": "model_timestamps",
            "speech_duration_s": 10.0,
            "covered_speech_s": 6.0,
            "uncovered_speech_s": 4.0,
            "speech_coverage_ratio": 0.6,
            "max_uncovered_speech_s": 4.0,
            "leading_uncovered_speech_s": 0.0,
            "trailing_uncovered_speech_s": 0.0,
            "speech_intervals": [{"start": 0.0, "end": 10.0}],
            "covered_intervals": [{"start": 0.0, "end": 3.0}, {"start": 7.0, "end": 10.0}],
        },
    )

    result = mod.validate_case("demo", transcript, require_speech_coverage=True)

    assert result.ok is False
    assert any("speech coverage ratio" in error for error in result.errors)
    assert any("maximum uncovered speech" in error for error in result.errors)


def test_validate_case_rejects_no_speech_when_transcript_contains_text(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    _set_speech_coverage(
        transcript,
        {
            "status": "no_speech",
            "reason": "vad_detected_no_speech",
            "basis": "silero_vad_no_speech",
            "speech_duration_s": 0.0,
            "covered_speech_s": 0.0,
            "uncovered_speech_s": 0.0,
            "speech_coverage_ratio": None,
            "max_uncovered_speech_s": 0.0,
            "leading_uncovered_speech_s": 0.0,
            "trailing_uncovered_speech_s": 0.0,
        },
    )

    result = mod.validate_case("demo", transcript, require_speech_coverage=True)

    assert result.ok is False
    assert "speech coverage reports no_speech but transcript contains text" in result.errors


def test_cli_speech_coverage_thresholds_are_enforced(tmp_path: Path):
    transcript = _write_valid_case(tmp_path)
    _set_speech_coverage(
        transcript,
        {
            "status": "ok",
            "reason": "test",
            "basis": "model_timestamps",
            "speech_duration_s": 10.0,
            "covered_speech_s": 9.5,
            "uncovered_speech_s": 0.5,
            "speech_coverage_ratio": 0.95,
            "max_uncovered_speech_s": 0.5,
            "leading_uncovered_speech_s": 0.0,
            "trailing_uncovered_speech_s": 0.5,
            "speech_intervals": [{"start": 0.0, "end": 10.0}],
            "covered_intervals": [{"start": 0.0, "end": 9.5}],
        },
    )
    out_dir = tmp_path / "coverage-report"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--case",
            f"demo={transcript}",
            "--out-dir",
            str(out_dir),
            "--require-speech-coverage",
            "--min-speech-coverage-ratio",
            "0.99",
            "--max-uncovered-speech-seconds",
            "3",
            "--max-edge-uncovered-speech-seconds",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    report = json.loads((out_dir / "asr_longform_acceptance.json").read_text(encoding="utf-8"))
    assert report["cases"][0]["metrics"]["speech_coverage_min_ratio"] == 0.99
    assert any("speech coverage ratio" in error for error in report["cases"][0]["errors"])


def test_validate_case_rejects_time_span_only_coverage_evidence(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    _set_speech_coverage(
        transcript,
        {
            "status": "ok",
            "reason": "retroactive_segment_comparison",
            "basis": "final_segments",
            "speech_duration_s": 10.0,
            "covered_speech_s": 10.0,
            "uncovered_speech_s": 0.0,
            "speech_coverage_ratio": 1.0,
            "max_uncovered_speech_s": 0.0,
            "leading_uncovered_speech_s": 0.0,
            "trailing_uncovered_speech_s": 0.0,
            "speech_intervals": [{"start": 0.0, "end": 10.0}],
            "covered_intervals": [{"start": 0.0, "end": 10.0}],
        },
    )

    result = mod.validate_case("demo", transcript, require_speech_coverage=True)

    assert result.ok is False
    assert "speech coverage basis is not strict recognition evidence: final_segments" in result.errors


def test_wallclock_vad_records_recognized_and_failed_chunks(tmp_path: Path, monkeypatch):
    import wave

    from scribe_py.core.transcriber_funasr import FunASRTranscriber
    from scribe_py.core.types import TranscribeOptions

    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000 * 3)

    transcriber = FunASRTranscriber(backend_name="sensevoice")
    monkeypatch.setattr(transcriber, "_speech_ranges", lambda _audio: [(0.0, 1.0), (2.0, 3.0)])

    def fake_generate(_model, chunk_path, _options, *, sensevoice):
        assert sensevoice is True
        if Path(chunk_path).name == "chunk_0000.wav":
            return [{"text": "识别成功", "language": "zh"}]
        return [{"text": "", "language": "zh"}]

    monkeypatch.setattr(transcriber, "_generate", fake_generate)

    segments, language, stats = transcriber._run_sensevoice_wallclock_vad(
        object(),
        audio,
        TranscribeOptions(model_id="iic/SenseVoiceSmall", language="zh"),
        None,
    )

    assert language == "zh"
    assert segments
    assert stats["wallclock_vad_chunks"] == 2
    assert stats["wallclock_recognized_chunks"] == 1
    assert stats["wallclock_failed_chunks"] == 1
    assert transcriber._wallclock_recognized_ranges == [(0.0, 1.0)]
    assert transcriber._wallclock_failed_ranges == [(2.0, 3.0)]


def test_recognition_chunk_coverage_does_not_apply_boundary_collar():
    from scribe_py.core.transcriber_funasr import _speech_coverage_diagnostics
    from scribe_py.core.types import Segment

    result = _speech_coverage_diagnostics(
        [(0.0, 3.0)],
        [
            Segment(start=0.0, end=1.0, text="成功"),
            Segment(start=2.0, end=3.0, text="成功"),
        ],
        duration=3.0,
        collar_s=0.0,
    )

    assert result["uncovered_speech_s"] == 1.0
    assert result["max_uncovered_speech_s"] == 1.0
    assert result["speech_coverage_ratio"] == 0.666667


def test_wallclock_vad_rejects_low_text_density_chunk(tmp_path: Path, monkeypatch):
    import wave

    from scribe_py.core.transcriber_funasr import FunASRTranscriber
    from scribe_py.core.types import TranscribeOptions

    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000 * 10)

    transcriber = FunASRTranscriber(backend_name="sensevoice")
    monkeypatch.setattr(transcriber, "_speech_ranges", lambda _audio: [(0.0, 10.0)])
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, _audio, _options, *, sensevoice: [{"text": "啊", "language": "zh"}],
    )

    segments, _language, stats = transcriber._run_sensevoice_wallclock_vad(
        object(),
        audio,
        TranscribeOptions(model_id="iic/SenseVoiceSmall", language="zh"),
        None,
    )

    assert segments == []
    assert stats["wallclock_recognized_chunks"] == 0
    assert stats["wallclock_failed_chunks"] == 1
    assert stats["wallclock_low_density_chunks"] == 1
    assert transcriber._wallclock_failure_reasons[0]["reason"] == "low_text_density"


def test_validate_case_rejects_inconsistent_coverage_evidence(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    _set_speech_coverage(
        transcript,
        {
            "status": "ok",
            "reason": "test",
            "basis": "wallclock_strict_windows",
            "wallclock_max_chunk_s": 1.5,
            "speech_duration_s": 10.0,
            "covered_speech_s": 10.0,
            "uncovered_speech_s": 0.0,
            "speech_coverage_ratio": 1.2,
            "max_uncovered_speech_s": 0.0,
            "leading_uncovered_speech_s": 0.0,
            "trailing_uncovered_speech_s": 0.0,
            "wallclock_attempted_chunks": 2,
            "wallclock_recognized_chunks": 1,
            "wallclock_failed_chunks": 0,
            "speech_intervals": [{"start": 0.0, "end": 10.0}],
            "covered_intervals": [{"start": 0.0, "end": 10.0}],
        },
    )

    result = mod.validate_case("demo", transcript, require_speech_coverage=True)

    assert result.ok is False
    assert "speech coverage ratio must be between 0 and 1" in result.errors
    assert "wallclock recognized plus failed chunks does not equal attempted chunks" in result.errors


def test_funasr_production_path_preserves_no_speech_status(tmp_path: Path, monkeypatch):
    from scribe_py.core import audio as audio_module
    from scribe_py.core.transcriber_funasr import FunASRTranscriber, SENSEVOICE_MODEL
    from scribe_py.core.types import TranscribeOptions

    transcriber = FunASRTranscriber(backend_name="sensevoice")
    monkeypatch.setattr(transcriber, "_load", lambda _model_id: object())
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, _audio, _options, *, sensevoice: [{"text": "", "language": "zh"}],
    )
    monkeypatch.setattr(audio_module, "probe_audio", lambda _audio: {"duration": 4.0})

    def no_speech(_audio):
        transcriber._speech_ranges_status = "no_speech"
        transcriber._speech_ranges_reason = "vad_detected_no_speech"
        return []

    monkeypatch.setattr(transcriber, "_speech_ranges", no_speech)
    transcriber._run(
        tmp_path / "silent.wav",
        TranscribeOptions(model_id=SENSEVOICE_MODEL, timing_align=False),
        None,
    )

    coverage = transcriber.last_filter_stats["speech_coverage"]
    assert coverage["status"] == "no_speech"
    assert coverage["basis"] == "silero_vad_no_speech"


def test_merge_speech_ranges_splits_single_long_vad_range():
    from scribe_py.core.transcriber_funasr import _merge_speech_ranges

    chunks = _merge_speech_ranges([(2.0, 39.0)], max_chunk_s=15.0, max_gap_s=0.75)

    assert chunks == [(2.0, 17.0), (17.0, 32.0), (32.0, 39.0)]
    assert all(end - start <= 15.0 for start, end in chunks)


def test_validate_case_accepts_empty_transcript_with_verified_no_speech(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    data = json.loads(transcript.read_text(encoding="utf-8"))
    data["segments"] = []
    data.setdefault("filter_stats", {})["speech_coverage"] = {
        "status": "no_speech",
        "reason": "vad_detected_no_speech",
        "basis": "silero_vad_no_speech",
        "speech_duration_s": 0.0,
        "covered_speech_s": 0.0,
        "uncovered_speech_s": 0.0,
        "speech_coverage_ratio": None,
        "max_uncovered_speech_s": 0.0,
        "leading_uncovered_speech_s": 0.0,
        "trailing_uncovered_speech_s": 0.0,
    }
    silence_transcript = tmp_path / "silence.json"
    silence_transcript.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    result = mod.validate_case("silence", silence_transcript, require_speech_coverage=True)

    assert result.ok is True


def test_validate_case_recomputes_leading_uncovered_speech(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    _set_speech_coverage(
        transcript,
        {
            "status": "ok",
            "reason": "test",
            "basis": "model_timestamps",
            "speech_duration_s": 10.0,
            "covered_speech_s": 8.0,
            "uncovered_speech_s": 2.0,
            "speech_coverage_ratio": 0.8,
            "max_uncovered_speech_s": 2.0,
            "leading_uncovered_speech_s": 0.0,
            "trailing_uncovered_speech_s": 0.0,
            "speech_intervals": [{"start": 0.0, "end": 10.0}],
            "covered_intervals": [{"start": 2.0, "end": 10.0}],
        },
    )

    result = mod.validate_case("demo", transcript, require_speech_coverage=True)

    assert result.ok is False
    assert result.metrics["leading_uncovered_speech_s"] == 2.0
    assert any("reported leading uncovered speech differs" in error for error in result.errors)
    assert any("leading uncovered speech 2.000s exceeds" in error for error in result.errors)


def test_items_have_timing_rejects_zero_length_timestamps():
    from scribe_py.core.transcriber_funasr import _items_have_timing

    assert _items_have_timing([{"timestamp": [[0, 0], [100, 100]]}]) is False
    assert _items_have_timing([{"timestamp": [[0, 120]]}]) is True
    assert _items_have_timing([{"sentence_info": [{"start": 200, "end": 200}]}]) is False
    assert _items_have_timing([{"sentence_info": [{"start": 200, "end": 420}]}]) is True


def test_merge_speech_ranges_default_chunks_are_at_most_five_seconds():
    from scribe_py.core.transcriber_funasr import _merge_speech_ranges

    chunks = _merge_speech_ranges([(0.0, 12.0)])

    assert chunks == [(0.0, 5.0), (5.0, 10.0), (10.0, 12.0)]


def test_validate_case_accepts_strict_window_evidence(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    _set_speech_coverage(
        transcript,
        {
            "status": "ok",
            "reason": "test",
            "basis": "wallclock_strict_windows",
            "wallclock_max_chunk_s": 1.5,
            "speech_duration_s": 3.0,
            "covered_speech_s": 3.0,
            "uncovered_speech_s": 0.0,
            "speech_coverage_ratio": 1.0,
            "max_uncovered_speech_s": 0.0,
            "leading_uncovered_speech_s": 0.0,
            "trailing_uncovered_speech_s": 0.0,
            "wallclock_attempted_chunks": 2,
            "wallclock_recognized_chunks": 2,
            "wallclock_failed_chunks": 0,
            "speech_intervals": [{"start": 0.0, "end": 3.0}],
            "covered_intervals": [{"start": 0.0, "end": 3.0}],
            **_strict_schema2(
                [[0.0, 1.5], [1.5, 3.0]],
                [[0.0, 1.5], [1.5, 3.0]],
                [],
            ),
        },
    )

    result = mod.validate_case("demo", transcript, require_speech_coverage=True)

    assert result.ok is True


def test_validate_case_rejects_exactly_three_second_gap(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    _set_speech_coverage(
        transcript,
        {
            "status": "ok",
            "reason": "test",
            "basis": "wallclock_strict_windows",
            "wallclock_max_chunk_s": 1.5,
            "speech_duration_s": 10.0,
            "covered_speech_s": 7.0,
            "uncovered_speech_s": 3.0,
            "speech_coverage_ratio": 0.7,
            "max_uncovered_speech_s": 3.0,
            "leading_uncovered_speech_s": 0.0,
            "trailing_uncovered_speech_s": 0.0,
            "wallclock_attempted_chunks": 4,
            "wallclock_recognized_chunks": 3,
            "wallclock_failed_chunks": 1,
            "speech_intervals": [{"start": 0.0, "end": 10.0}],
            "covered_intervals": [{"start": 0.0, "end": 3.0}, {"start": 6.0, "end": 10.0}],
        },
    )

    result = mod.validate_case(
        "demo",
        transcript,
        require_speech_coverage=True,
        min_speech_coverage_ratio=0.0,
        max_uncovered_speech_seconds=3.0,
    )

    assert result.ok is False
    assert any("maximum uncovered speech 3.000s reaches 3.000s limit" in error for error in result.errors)


def test_strict_coverage_uses_one_and_half_second_windows(tmp_path: Path, monkeypatch):
    import wave

    from scribe_py.core.transcriber_funasr import FunASRTranscriber
    from scribe_py.core.types import TranscribeOptions

    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000 * 3)

    transcriber = FunASRTranscriber(backend_name="sensevoice")
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_STRICT_COVERAGE", "1")
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_MAX_CHUNK_S", "9")
    monkeypatch.setattr(transcriber, "_speech_ranges", lambda _audio: [(0.0, 3.0)])
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, _audio, _options, *, sensevoice: [{"text": "测试文字", "language": "zh"}],
    )

    options = TranscribeOptions(model_id="iic/SenseVoiceSmall", language="zh")
    _segments, _language, anchor_stats = transcriber._run_sensevoice_wallclock_vad(
        object(),
        audio,
        options,
        None,
    )
    probe_stats = transcriber._run_sensevoice_strict_coverage_probe(
        object(),
        audio,
        options,
        None,
        duration=3.0,
    )

    assert anchor_stats["wallclock_strict_coverage"] is False
    assert anchor_stats["wallclock_max_chunk_s"] == 9.0
    assert anchor_stats["wallclock_vad_chunks"] == 1
    assert probe_stats["strict_core_max_chunk_s"] == 1.5
    assert probe_stats["strict_decode_context_pad_s"] == 0.5
    assert transcriber._wallclock_attempted_ranges == [(0.0, 1.5), (1.5, 3.0)]
    assert transcriber._wallclock_recognized_ranges == [(0.0, 1.5), (1.5, 3.0)]
    assert transcriber._wallclock_failed_ranges == []
    assert [
        (item["core_start"], item["core_end"], item["decode_start"], item["decode_end"])
        for item in probe_stats["strict_probe_windows"]
    ] == [
        (0.0, 1.5, 0.0, 1.5),
        (1.5, 3.0, 1.5, 3.0),
    ]
    assert probe_stats["strict_partition"]["partition_valid"] is True


def test_wallclock_timing_anchor_defaults_to_twenty_five_second_chunks(
    tmp_path: Path,
    monkeypatch,
):
    import wave

    from scribe_py.core.transcriber_funasr import FunASRTranscriber
    from scribe_py.core.types import TranscribeOptions

    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000 * 30)

    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_MAX_CHUNK_S", raising=False)
    transcriber = FunASRTranscriber(backend_name="sensevoice")
    monkeypatch.setattr(transcriber, "_speech_ranges", lambda _audio: [(0.0, 30.0)])
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, _audio, _options, *, sensevoice: [{"text": "测试文字", "language": "zh"}],
    )

    _segments, _language, stats = transcriber._run_sensevoice_wallclock_vad(
        object(),
        audio,
        TranscribeOptions(model_id="iic/SenseVoiceSmall", language="zh"),
        None,
    )

    assert stats["wallclock_anchor_max_chunk_s"] == 25.0
    assert stats["wallclock_vad_chunks"] == 2
    assert transcriber._wallclock_attempted_ranges == [(0.0, 25.0), (25.0, 30.0)]


def _recovery_attempt(
    framing: str,
    residual: str,
    *,
    pad_s: float,
    provider_id: str = "sensevoice-primary",
    provider_kind: str = "primary_asr",
    model_id: str = "iic/SenseVoiceSmall",
    model_family: str = "sensevoice",
) -> dict:
    attempt = {
        "framing": framing,
        "pad_s": pad_s,
        "raw": residual,
        "normalized": residual,
        "residual": residual,
        "residual_text": residual,
        "status": "valid",
        "slice_start": 1.0,
        "slice_end": 2.0,
        "slice_sha256": hashlib.sha256(f"{provider_id}:{framing}".encode("utf-8")).hexdigest(),
        "provider_id": provider_id,
        "provider_kind": provider_kind,
        "model_id": model_id,
        "model_family": model_family,
        "model_revision": "test-revision" if provider_kind == "independent_asr" else None,
        "config_sha256": "test-config-sha256" if provider_kind == "independent_asr" else None,
        "weights_manifest_sha256": "test-weights-manifest-sha256" if provider_kind == "independent_asr" else None,
        "hallucination_risk": False,
    }
    attempt["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "provider_id": provider_id,
                "model_id": model_id,
                "model_revision": attempt["model_revision"],
                "config_sha256": attempt["config_sha256"],
                "weights_manifest_sha256": attempt["weights_manifest_sha256"],
                "slice_sha256": attempt["slice_sha256"],
                "raw": residual,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return attempt


def _qwen_recovery_attempt(framing: str, residual: str, *, pad_s: float) -> dict:
    return _recovery_attempt(
        framing,
        residual,
        pad_s=pad_s,
        provider_id="qwen3-independent",
        provider_kind="independent_asr",
        model_id="mlx-community/Qwen3-ASR-1.7B-8bit",
        model_family="qwen3_asr",
    )


def _recovery_snapshot(
    *,
    segment_count: int,
    text_sha256: str,
    recognized_ranges: list[list[float]],
    failed_ranges: list[list[float]],
) -> dict:
    import json as _json

    recognized_ranges = sorted(recognized_ranges)
    failed_ranges = sorted(failed_ranges)
    attempted_ranges = sorted(recognized_ranges + failed_ranges)

    def digest(ranges: list[list[float]]) -> str:
        return hashlib.sha256(
            _json.dumps(ranges, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    return {
        "segment_count": segment_count,
        "text_sha256": text_sha256,
        "covered_count": len(recognized_ranges),
        "failed_count": len(failed_ranges),
        "attempted_ranges": attempted_ranges,
        "recognized_ranges": recognized_ranges,
        "failed_ranges": failed_ranges,
        "attempted_partition_sha256": digest(attempted_ranges),
        "recognized_partition_sha256": digest(recognized_ranges),
        "failed_partition_sha256": digest(failed_ranges),
        "partition_valid": True,
    }


def _set_valid_recovery_coverage(transcript: Path, recovery: dict, *, recognized: int, failed: int) -> None:
    data = json.loads(transcript.read_text(encoding="utf-8"))
    data.setdefault("filter_stats", {})["speech_coverage"] = {
        "status": "ok",
        "reason": "test",
        "basis": "wallclock_strict_windows",
        "wallclock_max_chunk_s": 1.5,
        "speech_duration_s": 4.0,
        "covered_speech_s": 4.0,
        "uncovered_speech_s": 0.0,
        "speech_coverage_ratio": 1.0,
        "max_uncovered_speech_s": 0.0,
        "leading_uncovered_speech_s": 0.0,
        "trailing_uncovered_speech_s": 0.0,
        "wallclock_attempted_chunks": recognized + failed,
        "wallclock_recognized_chunks": recognized,
        "wallclock_failed_chunks": failed,
        "speech_intervals": [{"start": 0.0, "end": 4.0}],
        "covered_intervals": [{"start": 0.0, "end": 4.0}],
        "local_recovery": recovery,
    }
    transcript.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_validate_case_rejects_audit_recovery_that_claims_changes(tmp_path: Path):
    import hashlib
    import re

    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    data = json.loads(transcript.read_text(encoding="utf-8"))
    compact = re.sub(r"\s+", "", "".join(item["text"] for item in data["segments"]))
    text_hash = hashlib.sha256(compact.encode("utf-8")).hexdigest()
    recovery = {
        "mode": "audit",
        "requested_mode": "audit",
        "diagnostic": None,
        "pending_windows": 1,
        "pending_groups": 1,
        "attempts": 2,
        "matched_existing": 0,
        "inserted": 1,
        "rejected": 0,
        "error": 0,
        "before": {"segment_count": 2, "text_sha256": text_hash, "covered_count": 1, "failed_count": 1},
        "after": {"segment_count": 3, "text_sha256": text_hash, "covered_count": 1, "failed_count": 1},
        "details": [{
            "start": 1.5,
            "end": 2.0,
            "window_count": 1,
            "original_failures": [{"start": 1.5, "end": 2.0, "reason": "empty_transcript"}],
            "attempts": [
                _recovery_attempt("exact", "新增内容", pad_s=0.0),
                _recovery_attempt("pad0.5", "新增内容", pad_s=0.5),
            ],
            "decision": "insert_accepted",
            "consensus": "新增内容",
            "evidence_framings": ["exact", "pad0.5"],
            "inserted_text": "新增内容",
        }],
        "details_truncated": False,
    }
    _set_valid_recovery_coverage(transcript, recovery, recognized=1, failed=1)

    result = mod.validate_case("demo", transcript)

    assert result.ok is False
    assert "audit local recovery changed segment_count" in result.errors
    assert "audit local recovery segment count differs from transcript" in result.errors


def test_validate_case_rejects_forged_same_framing_recovery_evidence(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    data = json.loads(transcript.read_text(encoding="utf-8"))
    data["segments"].insert(1, {"start": 1.5, "end": 2.0, "text": "新增内容"})
    transcript.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    transcript.with_suffix(".txt").unlink()
    transcript.with_suffix(".srt").unlink()
    recovery = {
        "mode": "merge",
        "requested_mode": "merge",
        "diagnostic": None,
        "pending_windows": 1,
        "pending_groups": 1,
        "attempts": 2,
        "matched_existing": 0,
        "inserted": 1,
        "rejected": 0,
        "error": 0,
        "before": {"segment_count": 2, "text_sha256": "before", "covered_count": 0, "failed_count": 1},
        "after": {"segment_count": 3, "text_sha256": "after", "covered_count": 1, "failed_count": 0},
        "details": [{
            "start": 1.5,
            "end": 2.0,
            "window_count": 1,
            "original_failures": [{"start": 1.5, "end": 2.0, "reason": "empty_transcript"}],
            "attempts": [
                _recovery_attempt("exact", "新增内容", pad_s=0.0),
                _recovery_attempt("exact", "新增内容", pad_s=0.0),
            ],
            "decision": "insert_accepted",
            "consensus": "新增内容",
            "evidence_framings": ["exact", "exact"],
            "inserted_text": "新增内容",
        }],
        "details_truncated": False,
    }
    _set_valid_recovery_coverage(transcript, recovery, recognized=1, failed=0)

    result = mod.validate_case("demo", transcript)

    assert result.ok is False
    assert any("repeats provider framing sensevoice-primary:exact" in error for error in result.errors)
    assert any("decision is not supported by attempts" in error for error in result.errors)


def test_validate_case_rejects_conflicting_recovery_residuals(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    data = json.loads(transcript.read_text(encoding="utf-8"))
    data["segments"].insert(1, {"start": 1.5, "end": 2.0, "text": "新增内容"})
    transcript.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    transcript.with_suffix(".txt").unlink()
    transcript.with_suffix(".srt").unlink()
    recovery = {
        "mode": "merge",
        "requested_mode": "merge",
        "diagnostic": None,
        "pending_windows": 1,
        "pending_groups": 1,
        "attempts": 2,
        "matched_existing": 0,
        "inserted": 1,
        "rejected": 0,
        "error": 0,
        "before": {"segment_count": 2, "text_sha256": "before", "covered_count": 0, "failed_count": 1},
        "after": {"segment_count": 3, "text_sha256": "after", "covered_count": 1, "failed_count": 0},
        "details": [{
            "start": 1.5,
            "end": 2.0,
            "window_count": 1,
            "original_failures": [{"start": 1.5, "end": 2.0, "reason": "empty_transcript"}],
            "attempts": [
                _recovery_attempt("exact", "新增内容", pad_s=0.0),
                _recovery_attempt("pad0.5", "冲突内容", pad_s=0.5),
            ],
            "decision": "insert_accepted",
            "consensus": "新增内容",
            "evidence_framings": ["exact", "pad0.5"],
            "inserted_text": "新增内容",
        }],
        "details_truncated": False,
    }
    _set_valid_recovery_coverage(transcript, recovery, recognized=1, failed=0)

    result = mod.validate_case("demo", transcript)

    assert result.ok is False
    assert any("decision is not supported by attempts" in error for error in result.errors)


def test_validate_case_accepts_off_recovery_with_pending_windows(tmp_path: Path):
    import hashlib
    import re

    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    data = json.loads(transcript.read_text(encoding="utf-8"))
    compact = re.sub(r"\s+", "", "".join(item["text"] for item in data["segments"]))
    text_hash = hashlib.sha256(compact.encode("utf-8")).hexdigest()
    recovery = {
        "mode": "off",
        "requested_mode": "off",
        "diagnostic": None,
        "pending_windows": 1,
        "pending_groups": 1,
        "attempts": 0,
        "matched_existing": 0,
        "inserted": 0,
        "rejected": 0,
        "error": 0,
        "before": _recovery_snapshot(
            segment_count=2, text_sha256=text_hash,
            recognized_ranges=[[0.0, 1.0]], failed_ranges=[[1.0, 2.0]],
        ),
        "after": _recovery_snapshot(
            segment_count=2, text_sha256=text_hash,
            recognized_ranges=[[0.0, 1.0]], failed_ranges=[[1.0, 2.0]],
        ),
        "details": [],
        "details_truncated": False,
    }
    _set_valid_recovery_coverage(transcript, recovery, recognized=1, failed=1)

    result = mod.validate_case("demo", transcript)

    assert result.ok is True
    assert result.metrics["local_recovery_mode"] == "off"


def test_local_recovery_validator_rejects_merge_without_independent_evidence():
    import hashlib

    mod = _load_script()
    segments = [{"start": 0.0, "end": 1.0, "text": "新增内容"}]
    text_hash = hashlib.sha256("新增内容".encode("utf-8")).hexdigest()
    recovery = {
        "mode": "merge",
        "pending_windows": 1,
        "pending_groups": 1,
        "attempts": 2,
        "matched_existing": 0,
        "inserted": 1,
        "rejected": 0,
        "error": 0,
        "before": _recovery_snapshot(
            segment_count=0, text_sha256=hashlib.sha256(b"").hexdigest(),
            recognized_ranges=[], failed_ranges=[[0.0, 1.0]],
        ),
        "after": _recovery_snapshot(
            segment_count=1, text_sha256=text_hash,
            recognized_ranges=[[0.0, 1.0]], failed_ranges=[],
        ),
        "details": [{
            "start": 0.0,
            "end": 1.0,
            "window_count": 1,
            "original_failures": [{"start": 0.0, "end": 1.0, "reason": "empty_transcript"}],
            "left_context": "",
            "right_context": "",
            "local_reference": "",
            "attempts": [
                _recovery_attempt("exact", "新增内容", pad_s=0.0),
                _recovery_attempt("pad0.5", "新增内容", pad_s=0.5),
            ],
            "decision": "insert_accepted",
            "consensus": "新增内容",
            "evidence_framings": ["exact", "pad0.5"],
            "inserted_text": "新增内容",
        }],
        "details_truncated": False,
    }
    coverage = {"wallclock_recognized_chunks": 1, "wallclock_failed_chunks": 0}

    errors, _warnings, _metrics = mod._validate_local_recovery(
        recovery, segments=segments, raw_coverage=coverage
    )

    assert any("merge local recovery provider" in error for error in errors)
    assert any("decision is not supported by attempts" in error for error in errors)


def test_local_recovery_validator_accepts_qwen_independent_merge(monkeypatch):
    mod = _load_script()
    segments = [{"start": 0.0, "end": 1.0, "text": "新增内容", "original_text": "新增内容"}]
    empty_hash = hashlib.sha256(b"").hexdigest()
    text_hash = hashlib.sha256("新增内容".encode("utf-8")).hexdigest()
    attempts = [
        _recovery_attempt("exact", "新增内容", pad_s=0.0),
        _recovery_attempt("pad0.5", "新增内容", pad_s=0.5),
        _qwen_recovery_attempt("pad1.0", "新增内容", pad_s=1.0),
    ]
    evidence_ids = sorted(
        "|".join((item["provider_id"], item["framing"], item["slice_sha256"]))
        for item in attempts
    )
    recovery = {
        "mode": "merge",
        "requested_mode": "merge",
        "diagnostic": None,
        "provider": {
            "requested": "qwen3",
            "available": True,
            "error": None,
            "provider_id": "qwen3-independent",
            "provider_kind": "independent_asr",
            "model_id": "mlx-community/Qwen3-ASR-1.7B-8bit",
            "model_family": "qwen3_asr",
            "model_revision": "test-revision",
            "config_sha256": "test-config-sha256",
            "weights_manifest_sha256": "test-weights-manifest-sha256",
        },
        "text_normalization": {"language": "zh", "profile": None, "error": None},
        "pending_windows": 1,
        "pending_groups": 1,
        "attempts": 3,
        "matched_existing": 0,
        "inserted": 1,
        "rejected": 0,
        "error": 0,
        "before": _recovery_snapshot(
            segment_count=0,
            text_sha256=empty_hash,
            recognized_ranges=[],
            failed_ranges=[[0.0, 1.0]],
        ),
        "after": _recovery_snapshot(
            segment_count=1,
            text_sha256=text_hash,
            recognized_ranges=[[0.0, 1.0]],
            failed_ranges=[],
        ),
        "details": [{
            "start": 0.0,
            "end": 1.0,
            "window_count": 1,
            "original_failures": [{"start": 0.0, "end": 1.0, "reason": "empty_transcript"}],
            "left_context": "",
            "right_context": "",
            "overlapping_context": "",
            "local_reference": "",
            "min_required_chars": 2,
            "attempts": attempts,
            "decision": "insert_accepted",
            "consensus": "新增内容",
            "primary_status": "valid",
            "primary_consensus": "新增内容",
            "primary_evidence_framings": ["exact", "pad0.5"],
            "evidence_framings": ["exact", "pad0.5", "pad1.0"],
            "evidence_providers": ["qwen3-independent", "sensevoice-primary"],
            "evidence_models": ["iic/SenseVoiceSmall", "mlx-community/Qwen3-ASR-1.7B-8bit"],
            "evidence_ids": evidence_ids,
            "inserted_raw_text": "新增内容",
            "inserted_text": "新增内容",
        }],
        "details_truncated": False,
    }

    errors, _warnings, _metrics = mod._validate_local_recovery(
        recovery,
        segments=segments,
        raw_coverage={"wallclock_recognized_chunks": 1, "wallclock_failed_chunks": 0},
        normalization_language="zh",
        normalization_profile=None,
    )

    assert errors == []

    forged = json.loads(json.dumps(recovery))
    forged["details"][0]["inserted_raw_text"] = "新 增 内 容"
    forged_segments = [{"start": 0.0, "end": 1.0, "text": "新增内容", "original_text": "新 增 内 容"}]
    forged_errors, _warnings, _metrics = mod._validate_local_recovery(
        forged,
        segments=forged_segments,
        raw_coverage={"wallclock_recognized_chunks": 1, "wallclock_failed_chunks": 0},
        normalization_language="zh",
        normalization_profile=None,
    )
    assert any("inserted raw text differs from canonical evidence" in error for error in forged_errors)

    missing_original_errors, _warnings, _metrics = mod._validate_local_recovery(
        recovery,
        segments=[{"start": 0.0, "end": 1.0, "text": "新增内容"}],
        raw_coverage={"wallclock_recognized_chunks": 1, "wallclock_failed_chunks": 0},
        normalization_language="zh",
        normalization_profile=None,
    )
    assert any("inserted segment is missing or duplicated" in error for error in missing_original_errors)

    wrong_language = json.loads(json.dumps(recovery))
    wrong_language["text_normalization"]["language"] = "auto"
    language_errors, _warnings, _metrics = mod._validate_local_recovery(
        wrong_language,
        segments=segments,
        raw_coverage={"wallclock_recognized_chunks": 1, "wallclock_failed_chunks": 0},
        normalization_language="zh",
        normalization_profile=None,
    )
    assert any("normalization language differs from transcript" in error for error in language_errors)

    import scribe_py.core.text_normalizer as text_normalizer_module

    monkeypatch.setattr(text_normalizer_module, "normalize_segments", lambda *_args, **_kwargs: ([], {"mode": "test"}))
    for safe_mode in ("audit", "merge"):
        safe_rejection = json.loads(json.dumps(recovery))
        safe_rejection["mode"] = safe_mode
        safe_rejection["requested_mode"] = safe_mode
        safe_rejection["inserted"] = 0
        safe_rejection["rejected"] = 1
        safe_rejection["before"] = _recovery_snapshot(
            segment_count=0,
            text_sha256=empty_hash,
            recognized_ranges=[],
            failed_ranges=[[0.0, 1.0]],
        )
        safe_rejection["after"] = dict(safe_rejection["before"])
        safe_detail = safe_rejection["details"][0]
        safe_detail["evidence_decision"] = "insert_accepted"
        safe_detail["decision"] = "rejected"
        safe_detail["normalization_rejection_reason"] = "ValueError:test"
        safe_detail["inserted_text"] = ""
        safe_errors, _warnings, _metrics = mod._validate_local_recovery(
            safe_rejection,
            segments=[],
            raw_coverage={"wallclock_recognized_chunks": 0, "wallclock_failed_chunks": 1},
            normalization_language="zh",
            normalization_profile=None,
        )
        assert safe_errors == []


def test_local_recovery_validator_rejects_forged_raw_normalization():
    import hashlib

    mod = _load_script()
    segments = [{"start": 1.0, "end": 2.0, "text": "新增内容"}]
    text_hash = hashlib.sha256("新增内容".encode("utf-8")).hexdigest()
    forged_attempts = [
        {
            **_recovery_attempt("exact", "新增内容", pad_s=0.0),
            "raw": "冲突甲乙",
        },
        {
            **_recovery_attempt("pad0.5", "新增内容", pad_s=0.5),
            "raw": "冲突丙丁",
        },
    ]
    recovery = {
        "mode": "merge",
        "pending_windows": 1,
        "pending_groups": 1,
        "attempts": 2,
        "matched_existing": 0,
        "inserted": 1,
        "rejected": 0,
        "error": 0,
        "before": {"segment_count": 0, "text_sha256": hashlib.sha256(b"").hexdigest(), "covered_count": 0, "failed_count": 1},
        "after": {"segment_count": 1, "text_sha256": text_hash, "covered_count": 1, "failed_count": 0},
        "details": [{
            "start": 1.0,
            "end": 2.0,
            "window_count": 1,
            "original_failures": [{"start": 1.0, "end": 2.0, "reason": "empty_transcript"}],
            "left_context": "",
            "right_context": "",
            "local_reference": "",
            "attempts": forged_attempts,
            "decision": "insert_accepted",
            "consensus": "新增内容",
            "evidence_framings": ["exact", "pad0.5"],
            "inserted_text": "新增内容",
        }],
        "details_truncated": False,
    }

    errors, _warnings, _metrics = mod._validate_local_recovery(
        recovery,
        segments=segments,
        raw_coverage={"wallclock_recognized_chunks": 1, "wallclock_failed_chunks": 0},
    )

    assert any("normalized text is forged" in error for error in errors)
    assert any("attempt residual is forged" in error for error in errors)


def test_local_recovery_validator_binds_inserted_text_to_consensus():
    import hashlib

    mod = _load_script()
    segments = [{"start": 1.0, "end": 2.0, "text": "其他内容"}]
    text_hash = hashlib.sha256("其他内容".encode("utf-8")).hexdigest()
    recovery = {
        "mode": "merge",
        "pending_windows": 1,
        "pending_groups": 1,
        "attempts": 2,
        "matched_existing": 0,
        "inserted": 1,
        "rejected": 0,
        "error": 0,
        "before": {"segment_count": 0, "text_sha256": hashlib.sha256(b"").hexdigest(), "covered_count": 0, "failed_count": 1},
        "after": {"segment_count": 1, "text_sha256": text_hash, "covered_count": 1, "failed_count": 0},
        "details": [{
            "start": 1.0,
            "end": 2.0,
            "window_count": 1,
            "original_failures": [{"start": 1.0, "end": 2.0, "reason": "empty_transcript"}],
            "left_context": "",
            "right_context": "",
            "local_reference": "",
            "attempts": [
                _recovery_attempt("exact", "新增内容", pad_s=0.0),
                _recovery_attempt("pad0.5", "新增内容", pad_s=0.5),
            ],
            "decision": "insert_accepted",
            "consensus": "新增内容",
            "evidence_framings": ["exact", "pad0.5"],
            "inserted_raw_text": "其他内容",
            "inserted_text": "其他内容",
        }],
        "details_truncated": False,
    }

    errors, _warnings, _metrics = mod._validate_local_recovery(
        recovery,
        segments=segments,
        raw_coverage={"wallclock_recognized_chunks": 1, "wallclock_failed_chunks": 0},
    )

    assert any("evidence decision is not supported" in error for error in errors)
    assert any("final decision disagrees with normalization gate" in error for error in errors)


def test_local_recovery_validator_rejects_matched_existing_without_attempt_evidence():
    import hashlib

    mod = _load_script()
    segments = [{"start": 0.0, "end": 2.0, "text": "已有正文"}]
    text_hash = hashlib.sha256("已有正文".encode("utf-8")).hexdigest()
    recovery = {
        "mode": "merge",
        "pending_windows": 1,
        "pending_groups": 1,
        "attempts": 0,
        "matched_existing": 1,
        "inserted": 0,
        "rejected": 0,
        "error": 0,
        "before": {"segment_count": 1, "text_sha256": text_hash, "covered_count": 0, "failed_count": 1},
        "after": {"segment_count": 1, "text_sha256": text_hash, "covered_count": 1, "failed_count": 0},
        "details": [{
            "start": 1.0,
            "end": 2.0,
            "window_count": 1,
            "original_failures": [{"start": 1.0, "end": 2.0, "reason": "empty_transcript"}],
            "left_context": "",
            "right_context": "",
            "local_reference": "已有正文",
            "attempts": [],
            "decision": "matched_existing",
            "consensus": "已有正文",
            "evidence_framings": ["exact"],
            "inserted_text": "",
        }],
        "details_truncated": False,
    }

    errors, _warnings, _metrics = mod._validate_local_recovery(
        recovery,
        segments=segments,
        raw_coverage={"wallclock_recognized_chunks": 1, "wallclock_failed_chunks": 0},
    )

    assert any("decision is not supported by attempts" in error for error in errors)


def test_local_recovery_validator_rebuilds_context_from_transcript():
    mod = _load_script()
    segments = [{"start": 0.0, "end": 3.0, "text": "真实正文"}]
    text_hash = hashlib.sha256("真实正文".encode("utf-8")).hexdigest()
    snapshot = _recovery_snapshot(
        segment_count=1,
        text_sha256=text_hash,
        recognized_ranges=[],
        failed_ranges=[[1.0, 2.0]],
    )
    recovery = {
        "mode": "audit",
        "pending_windows": 1,
        "pending_groups": 1,
        "attempts": 2,
        "matched_existing": 1,
        "inserted": 0,
        "rejected": 0,
        "error": 0,
        "before": snapshot,
        "after": dict(snapshot),
        "details": [{
            "start": 1.0,
            "end": 2.0,
            "window_count": 1,
            "original_failures": [{"start": 1.0, "end": 2.0, "reason": "empty_transcript"}],
            "left_context": "",
            "right_context": "",
            "overlapping_context": "伪造正文",
            "local_reference": "伪造正文",
            "min_required_chars": 2,
            "attempts": [
                _recovery_attempt("exact", "伪造正文", pad_s=0.0),
                _recovery_attempt("pad0.5", "伪造正文", pad_s=0.5),
            ],
            "decision": "matched_existing",
            "consensus": "伪造正文",
            "evidence_framings": ["exact", "pad0.5"],
            "inserted_text": "",
        }],
        "details_truncated": False,
    }

    errors, _warnings, _metrics = mod._validate_local_recovery(
        recovery,
        segments=segments,
        raw_coverage={"wallclock_recognized_chunks": 0, "wallclock_failed_chunks": 1},
    )

    assert any("local_reference is not derived from transcript" in error for error in errors)
    assert any("decision is not supported by attempts" in error for error in errors)


def test_local_recovery_validator_rejects_truncated_machine_evidence():
    mod = _load_script()
    failed_ranges = [[float(index), float(index) + 0.5] for index in range(21)]
    empty_hash = hashlib.sha256(b"").hexdigest()
    snapshot = _recovery_snapshot(
        segment_count=0,
        text_sha256=empty_hash,
        recognized_ranges=[],
        failed_ranges=failed_ranges,
    )
    recovery = {
        "mode": "audit",
        "pending_windows": 21,
        "pending_groups": 21,
        "attempts": 0,
        "matched_existing": 0,
        "inserted": 0,
        "rejected": 21,
        "error": 0,
        "before": snapshot,
        "after": dict(snapshot),
        "details": [],
        "details_truncated": True,
    }

    errors, _warnings, _metrics = mod._validate_local_recovery(
        recovery,
        segments=[],
        raw_coverage={"wallclock_recognized_chunks": 0, "wallclock_failed_chunks": 21},
    )

    assert "local recovery machine-verifiable details must not be truncated" in errors


def test_local_recovery_validator_recomputes_partition_hashes():
    mod = _load_script()
    empty_hash = hashlib.sha256(b"").hexdigest()
    before = _recovery_snapshot(
        segment_count=0,
        text_sha256=empty_hash,
        recognized_ranges=[],
        failed_ranges=[[0.0, 1.0]],
    )
    before["attempted_partition_sha256"] = "forged"
    recovery = {
        "mode": "off",
        "pending_windows": 1,
        "pending_groups": 1,
        "attempts": 0,
        "matched_existing": 0,
        "inserted": 0,
        "rejected": 0,
        "error": 0,
        "before": before,
        "after": _recovery_snapshot(
            segment_count=0,
            text_sha256=empty_hash,
            recognized_ranges=[],
            failed_ranges=[[0.0, 1.0]],
        ),
        "details": [],
        "details_truncated": False,
    }

    errors, _warnings, _metrics = mod._validate_local_recovery(
        recovery,
        segments=[],
        raw_coverage={"wallclock_recognized_chunks": 0, "wallclock_failed_chunks": 1},
    )

    assert "local recovery before attempted_partition_sha256 is invalid" in errors


def test_local_recovery_validator_rejects_partially_overlapping_partitions():
    mod = _load_script()
    empty_hash = hashlib.sha256(b"").hexdigest()
    snapshot = _recovery_snapshot(
        segment_count=0,
        text_sha256=empty_hash,
        recognized_ranges=[[0.0, 2.0]],
        failed_ranges=[[1.0, 3.0]],
    )
    recovery = {
        "mode": "off",
        "pending_windows": 1,
        "pending_groups": 1,
        "attempts": 0,
        "matched_existing": 0,
        "inserted": 0,
        "rejected": 0,
        "error": 0,
        "before": snapshot,
        "after": dict(snapshot),
        "details": [],
        "details_truncated": False,
    }

    errors, _warnings, _metrics = mod._validate_local_recovery(
        recovery,
        segments=[],
        raw_coverage={"wallclock_recognized_chunks": 1, "wallclock_failed_chunks": 1},
    )

    assert any("recognized and failed ranges overlap" in error for error in errors)
    assert any("partition_valid is invalid" in error for error in errors)


def test_local_recovery_validator_binds_before_text_to_reconstructed_segments():
    mod = _load_script()
    segments = [{"start": 0.0, "end": 1.0, "text": "新增内容"}]
    after_hash = hashlib.sha256("新增内容".encode("utf-8")).hexdigest()
    before = _recovery_snapshot(
        segment_count=0,
        text_sha256="forged",
        recognized_ranges=[],
        failed_ranges=[[0.0, 1.0]],
    )
    after = _recovery_snapshot(
        segment_count=1,
        text_sha256=after_hash,
        recognized_ranges=[[0.0, 1.0]],
        failed_ranges=[],
    )
    recovery = {
        "mode": "merge",
        "pending_windows": 1,
        "pending_groups": 1,
        "attempts": 2,
        "matched_existing": 0,
        "inserted": 1,
        "rejected": 0,
        "error": 0,
        "before": before,
        "after": after,
        "details": [{
            "start": 0.0,
            "end": 1.0,
            "window_count": 1,
            "original_failures": [{"start": 0.0, "end": 1.0, "reason": "empty_transcript"}],
            "left_context": "",
            "right_context": "",
            "overlapping_context": "",
            "local_reference": "",
            "min_required_chars": 2,
            "attempts": [
                _recovery_attempt("exact", "新增内容", pad_s=0.0),
                _recovery_attempt("pad0.5", "新增内容", pad_s=0.5),
            ],
            "decision": "insert_accepted",
            "consensus": "新增内容",
            "evidence_framings": ["exact", "pad0.5"],
            "inserted_text": "新增内容",
        }],
        "details_truncated": False,
    }

    errors, _warnings, _metrics = mod._validate_local_recovery(
        recovery,
        segments=segments,
        raw_coverage={"wallclock_recognized_chunks": 1, "wallclock_failed_chunks": 0},
    )

    assert "merge local recovery before text hash differs from reconstructed transcript" in errors


def test_local_recovery_validator_uses_trusted_density_threshold():
    mod = _load_script()
    segments = [{"start": 10.0, "end": 11.0, "text": "后文"}]
    text_hash = hashlib.sha256("后文".encode("utf-8")).hexdigest()
    snapshot = _recovery_snapshot(
        segment_count=1,
        text_sha256=text_hash,
        recognized_ranges=[],
        failed_ranges=[[0.0, 10.0]],
    )
    recovery = {
        "mode": "audit",
        "pending_windows": 1,
        "pending_groups": 1,
        "attempts": 2,
        "matched_existing": 0,
        "inserted": 1,
        "rejected": 0,
        "error": 0,
        "before": snapshot,
        "after": dict(snapshot),
        "details": [{
            "start": 0.0,
            "end": 10.0,
            "window_count": 1,
            "original_failures": [{"start": 0.0, "end": 10.0, "reason": "empty_transcript"}],
            "left_context": "",
            "right_context": "后文",
            "overlapping_context": "",
            "local_reference": "",
            "min_required_chars": 2,
            "attempts": [
                _recovery_attempt("exact", "你好", pad_s=0.0),
                _recovery_attempt("pad0.5", "你好", pad_s=0.5),
            ],
            "decision": "insert_accepted",
            "consensus": "你好",
            "evidence_framings": ["exact", "pad0.5"],
            "inserted_text": "你好",
        }],
        "details_truncated": False,
    }

    errors, _warnings, _metrics = mod._validate_local_recovery(
        recovery,
        segments=segments,
        raw_coverage={"wallclock_recognized_chunks": 0, "wallclock_failed_chunks": 1},
        min_chars_per_s=0.75,
    )

    assert any("density threshold is not trusted" in error for error in errors)
    assert any("decision is not supported by attempts" in error for error in errors)


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    [
        (None, None),
        ("standardized_hash", "standardized audio hash is invalid"),
        ("slice_hash", "slice_sha256 is invalid"),
        ("slice_bounds", "slice_end is invalid"),
    ],
)
def test_local_recovery_validator_recomputes_all_audit_audio_evidence(
    tmp_path: Path,
    tamper: str | None,
    expected_error: str | None,
):
    import soundfile as sf

    from scribe_py.core.audio import standardize_audio_for_asr

    mod = _load_script()
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000 * 2)
    prepared, standardization = standardize_audio_for_asr(
        audio,
        tmp_path / "prepared",
        mode="off",
    )
    standardization["standardized_sha256"] = mod._file_sha256(prepared)
    slice_data, _rate = sf.read(
        str(prepared), dtype="float32", start=0, stop=16000, always_2d=True
    )
    attempt = _recovery_attempt("exact", "测试内容", pad_s=0.0)
    attempt.update({
        "slice_start": 0.0,
        "slice_end": 1.0,
        "slice_sha256": hashlib.sha256(slice_data[:, 0].tobytes()).hexdigest(),
    })
    if tamper == "standardized_hash":
        standardization["standardized_sha256"] = "forged"
    elif tamper == "slice_hash":
        attempt["slice_sha256"] = "forged"
    elif tamper == "slice_bounds":
        attempt["slice_end"] = 1.25
    attempt["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "provider_id": attempt["provider_id"],
                "model_id": attempt["model_id"],
                "model_revision": attempt.get("model_revision"),
                "config_sha256": attempt.get("config_sha256"),
                "weights_manifest_sha256": attempt.get("weights_manifest_sha256"),
                "slice_sha256": attempt["slice_sha256"],
                "raw": attempt["raw"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    empty_hash = hashlib.sha256(b"").hexdigest()
    snapshot = _recovery_snapshot(
        segment_count=0,
        text_sha256=empty_hash,
        recognized_ranges=[],
        failed_ranges=[[0.0, 1.0]],
    )
    recovery = {
        "mode": "audit",
        "requested_mode": "audit",
        "diagnostic": None,
        "text_normalization": {"language": "zh", "profile": None, "error": None},
        "pending_windows": 1,
        "pending_groups": 1,
        "attempts": 1,
        "matched_existing": 0,
        "inserted": 0,
        "rejected": 1,
        "error": 0,
        "before": snapshot,
        "after": dict(snapshot),
        "details": [{
            "start": 0.0,
            "end": 1.0,
            "window_count": 1,
            "original_failures": [{"start": 0.0, "end": 1.0, "reason": "empty_transcript"}],
            "left_context": "",
            "right_context": "",
            "overlapping_context": "",
            "local_reference": "",
            "min_required_chars": 2,
            "attempts": [attempt],
            "decision": "rejected",
            "consensus": "",
            "primary_status": "",
            "primary_consensus": "",
            "primary_evidence_framings": [],
            "evidence_framings": [],
            "evidence_providers": [],
            "evidence_models": [],
            "evidence_ids": [],
            "inserted_raw_text": "",
            "inserted_text": "",
        }],
        "details_truncated": False,
    }

    errors, _warnings, _metrics = mod._validate_local_recovery(
        recovery,
        segments=[],
        raw_coverage={"wallclock_recognized_chunks": 0, "wallclock_failed_chunks": 1},
        audio_path=audio,
        preprocess_mode="off",
        audio_standardization=standardization,
        audio_quality={},
        enforce_audio_evidence=True,
    )

    if expected_error is None:
        assert errors == []
    else:
        assert any(expected_error in error for error in errors)


def test_recovery_validator_matches_single_primary_existing_plus_qwen_policy():
    mod = _load_script()
    from scribe_py.core.sensevoice_recovery import analyze_recovery_candidate, decide_recovery_attempts

    primary = analyze_recovery_candidate(
        "目标正文",
        framing="exact",
        pad_s=0.0,
        left="",
        right="",
        local_reference="前文目标正文后文",
    )
    primary["slice_sha256"] = "primary-slice"
    qwen = analyze_recovery_candidate(
        "目标正文",
        framing="pad1.0",
        pad_s=1.0,
        left="",
        right="",
        local_reference="前文目标正文后文",
        provider_id="qwen3-independent",
        provider_kind="independent_asr",
        model_id="mlx-community/Qwen3-ASR-1.7B-8bit",
        model_family="qwen3_asr",
    )
    qwen.update({
        "slice_sha256": "qwen-slice",
        "model_revision": "test-revision",
        "config_sha256": "config-sha256",
        "weights_manifest_sha256": "weights-manifest-sha256",
    })
    production = decide_recovery_attempts([primary, qwen])

    verified = []
    for attempt in (primary, qwen):
        item = dict(attempt)
        item.update({
            "verified_provider_id": attempt["provider_id"],
            "verified_provider_kind": attempt["provider_kind"],
            "verified_model_id": attempt["model_id"],
            "verified_model_family": attempt["model_family"],
            "verified_status": attempt["status"],
            "verified_normalized": attempt["normalized"],
            "verified_residual": attempt["residual"],
            "verified_residual_text": attempt["residual_text"],
        })
        verified.append(item)
    replay = mod._recovery_decision_from_attempts(verified)

    assert production["decision"] == replay["decision"] == "matched_existing"
    assert production["consensus"] == replay["consensus"] == "目标正文"
    assert production["evidence_providers"] == replay["evidence_providers"]
    assert production["evidence_ids"] == replay["evidence_ids"]


def test_validate_case_rejects_schema2_oversized_strict_core(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    coverage = {
        "status": "ok",
        "reason": "test",
        "basis": "wallclock_strict_windows",
        "wallclock_max_chunk_s": 1.5,
        "speech_duration_s": 3.0,
        "covered_speech_s": 3.0,
        "uncovered_speech_s": 0.0,
        "speech_coverage_ratio": 1.0,
        "max_uncovered_speech_s": 0.0,
        "leading_uncovered_speech_s": 0.0,
        "trailing_uncovered_speech_s": 0.0,
        "wallclock_attempted_chunks": 1,
        "wallclock_recognized_chunks": 1,
        "wallclock_failed_chunks": 0,
        "speech_intervals": [{"start": 0.0, "end": 3.0}],
        "covered_intervals": [{"start": 0.0, "end": 3.0}],
        **_strict_schema2([[0.0, 3.0]], [[0.0, 3.0]], []),
    }
    _set_speech_coverage(transcript, coverage)

    result = mod.validate_case("demo", transcript, require_speech_coverage=True)

    assert result.ok is False
    assert any("exceeds 1.5s core limit" in error for error in result.errors)


def test_validate_case_rejects_decode_padding_as_coverage_credit(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    coverage = {
        "status": "ok",
        "reason": "test",
        "basis": "wallclock_strict_windows",
        "wallclock_max_chunk_s": 1.5,
        "speech_duration_s": 3.0,
        "covered_speech_s": 2.5,
        "uncovered_speech_s": 0.5,
        "speech_coverage_ratio": 2.5 / 3.0,
        "max_uncovered_speech_s": 0.5,
        "leading_uncovered_speech_s": 0.0,
        "trailing_uncovered_speech_s": 0.5,
        "wallclock_attempted_chunks": 1,
        "wallclock_recognized_chunks": 1,
        "wallclock_failed_chunks": 0,
        "speech_intervals": [{"start": 0.0, "end": 3.0}],
        "covered_intervals": [{"start": 0.0, "end": 2.5}],
        **_strict_schema2([[0.5, 2.0]], [[0.5, 2.0]], []),
    }
    _set_speech_coverage(transcript, coverage)

    result = mod.validate_case("demo", transcript, require_speech_coverage=True)

    assert result.ok is False
    assert "strict recognized core ranges differ from covered intervals" in result.errors


def test_strict_probe_padding_cannot_replace_empty_core_evidence(tmp_path: Path, monkeypatch):
    import wave

    from scribe_py.core.transcriber_funasr import FunASRTranscriber
    from scribe_py.core.types import TranscribeOptions

    audio = tmp_path / "echo.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000 * 3)
    transcriber = FunASRTranscriber(backend_name="sensevoice")
    monkeypatch.setattr(transcriber, "_speech_ranges", lambda _audio: [(1.0, 2.0)])
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, chunk_path, _options, *, sensevoice: [
            {"text": "邻句回声" if str(chunk_path).endswith("_context.wav") else "", "language": "zh"}
        ],
    )

    stats = transcriber._run_sensevoice_strict_coverage_probe(
        object(),
        audio,
        TranscribeOptions(model_id="iic/SenseVoiceSmall", language="zh"),
        None,
        duration=3.0,
    )

    assert transcriber._wallclock_recognized_ranges == []
    assert transcriber._wallclock_failed_ranges == [(1.0, 2.0)]
    assert stats["strict_probe_windows"][0]["reason"] == "empty_core_transcript"


def test_validate_case_rejects_all_failed_windows_with_forged_covered_intervals(tmp_path: Path):
    mod = _load_script()
    transcript = _write_valid_case(tmp_path)
    coverage = {
        "status": "ok",
        "reason": "test",
        "basis": "wallclock_strict_windows",
        "wallclock_max_chunk_s": 1.5,
        "speech_duration_s": 3.0,
        "covered_speech_s": 3.0,
        "uncovered_speech_s": 0.0,
        "speech_coverage_ratio": 1.0,
        "max_uncovered_speech_s": 0.0,
        "leading_uncovered_speech_s": 0.0,
        "trailing_uncovered_speech_s": 0.0,
        "wallclock_attempted_chunks": 2,
        "wallclock_recognized_chunks": 0,
        "wallclock_failed_chunks": 2,
        "speech_intervals": [{"start": 0.0, "end": 3.0}],
        "covered_intervals": [{"start": 0.0, "end": 3.0}],
        **_strict_schema2(
            [[0.0, 1.5], [1.5, 3.0]],
            [],
            [[0.0, 1.5], [1.5, 3.0]],
        ),
    }
    _set_speech_coverage(transcript, coverage)

    result = mod.validate_case("demo", transcript, require_speech_coverage=True)

    assert result.ok is False
    assert "strict recognized core ranges differ from covered intervals" in result.errors
