from __future__ import annotations

import sys
import wave
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from scribe_py.core.asr_quality import build_asr_quality_report
from scribe_py.core import transcriber_funasr as funasr_module
from scribe_py.core.text_normalizer import normalize_segments, normalize_transcript_text, simplify_chinese_value
from scribe_py.core.transcriber_base import Transcriber
from scribe_py.core.transcriber_funasr import (
    SENSEVOICE_MODEL,
    FunASRTranscriber,
    _align_segments_to_timing_anchor,
    _build_sync_cues_from_char_times,
    _guard_unreliable_sync_cues,
    _repair_nonpositive_sync_cues_preserving_segments,
    _paraformer_timing_preflight,
    _realign_sync_cues_preserving_segments,
    _segments_from_text_and_timestamps,
    model_cached,
    resolve_model_path,
)
from scribe_py.core.types import Segment, TranscribeOptions


def test_transcribe_runs_recovery_hook_after_normalization_and_cleans_audio(tmp_path: Path):
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)

    class RecordingTranscriber(Transcriber):
        backend = "recording"

        def __init__(self):
            self.post_text = ""
            self.post_audio_exists = False

        def _run(self, prepared_audio, _options, _on_progress):
            assert prepared_audio.is_file()
            return [Segment(0.0, 1.0, "當我們進入學校的時候")], "zh"

        def _post_normalize_transcription(self, segments, prepared_audio, _options, _on_progress):
            self.post_text = segments[0].text
            self.post_audio_exists = prepared_audio.is_file()
            return segments

    transcriber = RecordingTranscriber()
    result = transcriber.transcribe(audio, TranscribeOptions(language="auto", audio_preprocess="off"))

    assert transcriber.post_text == "当我们进入学校的时候。"
    assert transcriber._text_normalization_language == "zh"
    assert transcriber.post_audio_exists is True
    assert result.filter_stats["audio_standardization"]["work_dir_cleaned"] is True
    assert not Path(result.filter_stats["audio_standardization"]["work_dir"]).exists()
    assert result.filter_stats["audio_standardization"]["standardized_sha256"]


def test_channel_analysis_failure_does_not_disable_audio_standardization(
    monkeypatch, tmp_path: Path
):
    from scribe_py.core import channel_selection

    source = tmp_path / "audio.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)

    monkeypatch.setattr(
        channel_selection,
        "evaluate_stereo_channel_selection",
        lambda _audio: (_ for _ in ()).throw(RuntimeError("analysis unavailable")),
    )

    class RecordingTranscriber(Transcriber):
        backend = "recording"

        def __init__(self):
            self.prepared_audio = None

        def _run(self, prepared_audio, _options, _on_progress):
            self.prepared_audio = prepared_audio
            return [Segment(0.0, 1.0, "测试")], "zh"

    transcriber = RecordingTranscriber()
    result = transcriber.transcribe(
        source,
        TranscribeOptions(language="zh", audio_preprocess="adaptive"),
    )

    standardization = result.filter_stats["audio_standardization"]
    selection = result.filter_stats["audio_quality"]["channel_selection"]
    assert standardization["applied"] is True
    assert standardization["channel_decision"] == "mix"
    assert transcriber.prepared_audio != source
    assert selection["reason"] == "analysis_failed"
    assert "analysis unavailable" in selection["error"]


def test_funasr_generation_is_reproducible_and_restores_torch_rng(monkeypatch, tmp_path: Path):
    import random

    numpy = __import__("numpy")
    torch = __import__("torch")

    class RandomDitherModel:
        def generate(self, **_kwargs):
            return [{
                "text": (
                    f"{random.random():.12f}:"
                    f"{numpy.random.random():.12f}:"
                    f"{torch.rand(1).item():.12f}"
                )
            }]

    monkeypatch.setenv("LOCALSCRIBE_FUNASR_INFERENCE_SEED", "7")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"test")
    transcriber = FunASRTranscriber()
    options = TranscribeOptions(language="zh", model_id=SENSEVOICE_MODEL)

    torch.manual_seed(1234)
    random.seed(1234)
    numpy.random.seed(1234)
    caller_state = torch.random.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = numpy.random.get_state()
    first = transcriber._generate(RandomDitherModel(), audio, options, sensevoice=True)
    assert torch.equal(torch.random.get_rng_state(), caller_state)
    assert random.getstate() == python_state
    assert numpy.array_equal(numpy.random.get_state()[1], numpy_state[1])
    assert first and first[0]["text"]

    torch.manual_seed(9999)
    random.seed(9999)
    numpy.random.seed(9999)
    second_caller_state = torch.random.get_rng_state().clone()
    second_python_state = random.getstate()
    second_numpy_state = numpy.random.get_state()
    second = transcriber._generate(RandomDitherModel(), audio, options, sensevoice=True)
    assert torch.equal(torch.random.get_rng_state(), second_caller_state)
    assert random.getstate() == second_python_state
    assert numpy.array_equal(numpy.random.get_state()[1], second_numpy_state[1])
    assert second and second[0]["text"]
    assert first == second


def test_paraformer_timing_preflight_is_generic_and_density_bounded(monkeypatch):
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_PARAFORMER_PREFLIGHT", raising=False)
    low_density = _paraformer_timing_preflight(
        [Segment(start=0.0, end=0.0, text="这是一段较短的转录正文。")],
        600.0,
    )
    normal_density = _paraformer_timing_preflight(
        [Segment(start=0.0, end=0.0, text="正常语速内容" * 400)],
        600.0,
    )
    long_low_density = _paraformer_timing_preflight(
        [Segment(start=0.0, end=0.0, text="低密度内容")],
        1800.0,
    )
    too_long = _paraformer_timing_preflight(
        [Segment(start=0.0, end=0.0, text="低密度内容")],
        9000.0,
    )

    assert low_density["selected"] is True
    assert low_density["reason"] == "low_source_text_density"
    assert normal_density["selected"] is False
    assert normal_density["reason"] == "source_text_density_normal"
    assert long_low_density["selected"] is False
    assert long_low_density["reason"] == "recording_too_long"
    assert long_low_density["max_duration_s"] == 1200.0
    assert too_long["selected"] is False
    assert too_long["reason"] == "recording_too_long"


def test_funasr_timestamps_are_milliseconds_even_when_small():
    segments = _segments_from_text_and_timestamps(
        "大家一定要认清这个情况。",
        [[110, 270], [270, 350], [350, 450], [450, 690], [690, 890], [890, 1010], [1010, 1150], [1150, 1330], [1330, 1450], [1450, 1610], [1610, 1730]],
        sensevoice=False,
    )

    assert segments
    assert segments[0].start == 0.11
    assert segments[0].end < 2.0


def test_funasr_timestamps_emit_phrase_sync_cues():
    text = "有点搅扰，大家一定要认清这个情况，并不是说这里面说哎。"
    timestamps = [[idx * 200, idx * 200 + 160] for idx, _ch in enumerate(text)]
    segments = _segments_from_text_and_timestamps(text, timestamps, sensevoice=False)

    assert len(segments) == 1
    cues = segments[0].sync_cues
    assert cues
    assert [cue["text"] for cue in cues] == ["有点搅扰，", "大家一定要认清这个情况，", "并不是说这里面说哎。"]
    assert cues[0]["start"] == segments[0].start
    assert cues[-1]["end"] == segments[0].end
    assert all(cues[idx]["end"] <= cues[idx + 1]["start"] for idx in range(len(cues) - 1))


def test_funasr_sync_cues_survive_missing_punctuation_timestamps():
    text = "有点搅扰，大家一定要认清这个情况。"
    timed_chars = [ch for ch in text if ch not in "，。！？；：、,.!?;:"]
    timestamps = [[idx * 180, idx * 180 + 140] for idx, _ch in enumerate(timed_chars)]
    segments = _segments_from_text_and_timestamps(text, timestamps, sensevoice=False)

    assert len(segments) == 1
    assert segments[0].start == 0.0
    assert segments[0].sync_cues
    assert [cue["text"] for cue in segments[0].sync_cues] == ["有点搅扰，", "大家一定要认清这个情况。"]


def test_funasr_sync_cues_do_not_emit_punctuation_only_cues():
    text = "你看这不就是一个悖论了吗？"
    timestamps = [[idx * 180, idx * 180 + 140] for idx, _ch in enumerate(text)]
    segments = _segments_from_text_and_timestamps(text, timestamps, sensevoice=False)

    assert len(segments) == 1
    cues = segments[0].sync_cues
    assert cues
    assert cues[-1]["text"].endswith("？")
    assert all(cue["text"].strip("，。！？；：、,.!?;:") for cue in cues)


def test_sync_cue_duration_repair_merges_collapsed_cues_without_changing_segment():
    source = [
        Segment(
            start=10.0,
            end=14.0,
            text="前句，好吧，后句。",
            sync_cues=[
                {"start": 10.0, "end": 12.0, "text": "前句，"},
                {"start": 12.0, "end": 12.0, "text": "好吧，"},
                {"start": 12.0, "end": 14.0, "text": "后句。"},
            ],
        )
    ]

    repaired, stats = _repair_nonpositive_sync_cues_preserving_segments(source)

    assert [(row.start, row.end, row.text) for row in repaired] == [(10.0, 14.0, source[0].text)]
    assert "".join(str(cue["text"]) for cue in repaired[0].sync_cues or []) == source[0].text
    assert all(float(cue["end"]) > float(cue["start"]) for cue in repaired[0].sync_cues or [])
    assert stats["zero_duration_cues_before"] == 1
    assert stats["zero_duration_cues_after"] == 0
    assert stats["geometry_preserved"] is True


def test_sync_cue_duration_repair_uses_segment_bounds_when_every_cue_is_collapsed():
    source = [
        Segment(
            start=4.0,
            end=6.0,
            text="好吧，",
            sync_cues=[{"start": 5.0, "end": 5.0, "text": "好吧，"}],
        )
    ]

    repaired, stats = _repair_nonpositive_sync_cues_preserving_segments(source)

    assert repaired[0].sync_cues == [
        {"start": 4.0, "end": 6.0, "text": "好吧，", "reliable": False}
    ]
    assert stats["zero_duration_cues_after"] == 0


def test_sync_cue_duration_repair_merges_overlapping_cues_as_unreliable():
    source = [
        Segment(
            start=10.0,
            end=16.0,
            text="前半句，后半句。",
            sync_cues=[
                {"start": 10.0, "end": 13.0, "text": "前半句，"},
                {"start": 10.0, "end": 16.0, "text": "后半句。"},
            ],
        )
    ]

    repaired, stats = _repair_nonpositive_sync_cues_preserving_segments(source)

    assert repaired[0].sync_cues == [
        {
            "start": 10.0,
            "end": 16.0,
            "text": "前半句，后半句。",
            "reliable": False,
        }
    ]
    assert stats["overlapping_cues_before"] == 1
    assert stats["overlapping_cues_after"] == 0
    assert [(row.start, row.end, row.text) for row in repaired] == [(10.0, 16.0, source[0].text)]


def test_sync_cue_guard_disables_highlight_without_vad_speech_support():
    source = [
        Segment(
            start=10.0,
            end=13.0,
            text="这句话没有可靠时间。",
            sync_cues=[{"start": 10.0, "end": 13.0, "text": "这句话没有可靠时间。"}],
        )
    ]

    guarded, stats = _guard_unreliable_sync_cues(
        source,
        [(0.0, 5.0)],
        vad_status="ok",
    )

    assert guarded[0].sync_cues == [
        {
            "start": 10.0,
            "end": 13.0,
            "text": "这句话没有可靠时间。",
            "reliable": False,
        }
    ]
    assert stats["unreliable_cues"] == 1
    assert stats["geometry_preserved"] is True


def test_sensevoice_without_timestamps_uses_wallclock_vad_fallback(monkeypatch, tmp_path: Path):
    transcriber = FunASRTranscriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD", "1")
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN", "0")
    monkeypatch.setattr(transcriber, "_load", lambda model_id: object())
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda model, audio, options, *, sensevoice: [{"text": "整段没有时间戳。", "language": "zh"}],
    )
    monkeypatch.setattr(
        transcriber,
        "_run_sensevoice_wallclock_vad",
        lambda model, audio, options, on_progress: (
            [
                Segment(start=10.0, end=12.0, text="第一段真实时间。"),
                Segment(start=20.0, end=22.0, text="第二段真实时间。"),
            ],
            "zh",
            {
                "timing_mode": "wallclock_vad_chunks",
                "timing_reliable": True,
                "timing_reason": "test",
                "wallclock_vad_ranges": 2,
                "wallclock_vad_chunks": 2,
            },
        ),
    )

    segments, lang = transcriber._run(
        tmp_path / "audio.wav",
        TranscribeOptions(model_id=SENSEVOICE_MODEL),
        None,
    )

    assert lang == "zh"
    assert 9.5 <= segments[0].start <= 10.0
    assert 19.5 <= segments[1].start <= 20.0
    assert [seg.end for seg in segments] == [12.0, 22.0]
    assert transcriber.last_filter_stats["timing_mode"] == "wallclock_vad_chunks"
    assert transcriber.last_filter_stats["timing_reliable"] is True


def test_sensevoice_without_timestamps_aligns_full_text_to_wallclock_anchor(monkeypatch, tmp_path: Path):
    transcriber = FunASRTranscriber()
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD", raising=False)
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN", "1")
    monkeypatch.setattr(transcriber, "_load", lambda model_id: object())
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda model, audio, options, *, sensevoice: [{"text": "第一句完整文字。第二句完整文字。", "language": "zh"}],
    )
    monkeypatch.setattr(
        transcriber,
        "_run_sensevoice_wallclock_vad",
        lambda model, audio, options, on_progress: (
            [
                Segment(start=10.0, end=12.0, text="第一句完整文字。"),
                Segment(start=20.0, end=22.0, text="第二句完整文字。"),
            ],
            "zh",
            {
                "timing_mode": "wallclock_vad_chunks",
                "timing_reliable": True,
                "timing_reason": "test",
                "wallclock_vad_ranges": 2,
                "wallclock_vad_chunks": 2,
            },
        ),
    )

    segments, lang = transcriber._run(
        tmp_path / "audio.wav",
        TranscribeOptions(model_id=SENSEVOICE_MODEL),
        None,
    )

    assert lang == "zh"
    assert [seg.text for seg in segments] == ["第一句完整文字。", "第二句完整文字。"]
    assert segments[0].start >= 9.5
    assert segments[1].start >= 19.5
    assert transcriber.last_filter_stats["timing_mode"] == "aligned_to_wallclock_anchor"
    assert transcriber.last_filter_stats["timing_reliable"] is True
    assert transcriber.last_filter_stats["equal_char_ratio"] == 1.0


def test_sensevoice_paraformer_anchor_is_opt_in_by_default(monkeypatch, tmp_path: Path):
    transcriber = FunASRTranscriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN", "1")
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_PARAFORMER_ANCHOR", raising=False)
    monkeypatch.setattr(transcriber, "_load", lambda model_id: object())
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda model, audio, options, *, sensevoice: [{"text": "第一句完整文字。第二句完整文字。", "language": "zh"}],
    )
    monkeypatch.setattr(
        transcriber,
        "_run_sensevoice_wallclock_vad",
        lambda model, audio, options, on_progress: (
            [
                Segment(start=10.0, end=12.0, text="第一句完整文字。"),
                Segment(start=20.0, end=22.0, text="第二句完整文字。"),
            ],
            "zh",
            {
                "timing_mode": "wallclock_vad_chunks",
                "timing_reliable": True,
                "timing_reason": "test",
                "wallclock_vad_ranges": 2,
                "wallclock_vad_chunks": 2,
            },
        ),
    )

    segments, _ = transcriber._run(
        tmp_path / "audio.wav",
        TranscribeOptions(model_id=SENSEVOICE_MODEL),
        None,
    )

    assert 9.5 <= segments[0].start <= 10.0
    assert 19.5 <= segments[1].start <= 20.0
    assert transcriber.last_filter_stats["timing_mode"] == "aligned_to_wallclock_anchor"
    assert transcriber.last_filter_stats["settings"]["sensevoice_paraformer_anchor"] is False


def test_sensevoice_timing_alignment_is_opt_in_by_default(monkeypatch, tmp_path: Path):
    transcriber = FunASRTranscriber()
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD", raising=False)
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN", raising=False)
    monkeypatch.setattr(transcriber, "_load", lambda model_id: object())
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda model, audio, options, *, sensevoice: [{"text": "第一句。第二句。", "language": "zh"}],
    )
    monkeypatch.setattr(
        transcriber,
        "_run_sensevoice_wallclock_vad",
        lambda model, audio, options, on_progress: (_ for _ in ()).throw(
            AssertionError("wallclock timing alignment should be opt-in")
        ),
    )

    segments, _ = transcriber._run(
        tmp_path / "audio.wav",
        TranscribeOptions(model_id=SENSEVOICE_MODEL),
        None,
    )

    assert segments
    assert transcriber.last_filter_stats["timing_mode"] == "coarse_text_distribution"
    assert transcriber.last_filter_stats["settings"]["sensevoice_timing_align"] is False


def test_sensevoice_timing_alignment_can_be_enabled_per_request(monkeypatch, tmp_path: Path):
    transcriber = FunASRTranscriber()
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD", raising=False)
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN", raising=False)
    monkeypatch.setattr(transcriber, "_load", lambda model_id: object())
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda model, audio, options, *, sensevoice: [{"text": "第一句完整文字。第二句完整文字。", "language": "zh"}],
    )
    monkeypatch.setattr(
        transcriber,
        "_run_sensevoice_wallclock_vad",
        lambda model, audio, options, on_progress: (
            [
                Segment(start=10.0, end=12.0, text="第一句完整文字。"),
                Segment(start=20.0, end=22.0, text="第二句完整文字。"),
            ],
            "zh",
            {
                "timing_mode": "wallclock_vad_chunks",
                "timing_reliable": True,
                "timing_reason": "test",
                "wallclock_vad_ranges": 2,
                "wallclock_vad_chunks": 2,
            },
        ),
    )

    segments, _ = transcriber._run(
        tmp_path / "audio.wav",
        TranscribeOptions(model_id=SENSEVOICE_MODEL, timing_align=True),
        None,
    )

    assert [seg.text for seg in segments] == ["第一句完整文字。", "第二句完整文字。"]
    assert transcriber.last_filter_stats["timing_mode"] == "aligned_to_wallclock_anchor"
    assert transcriber.last_filter_stats["settings"]["sensevoice_timing_align"] is True


def test_sensevoice_uses_cached_paraformer_after_wallclock_alignment_fails(monkeypatch, tmp_path: Path):
    transcriber = FunASRTranscriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN", "1")
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_PARAFORMER_ANCHOR", raising=False)
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD", raising=False)
    monkeypatch.setattr(funasr_module, "model_cached", lambda model_id: True)
    monkeypatch.setattr(transcriber, "_load", lambda model_id: object())

    source_text = "第一句完整文字。第二句完整文字。"

    def generate(_model, _audio, _options, *, sensevoice):
        if sensevoice:
            return [{"text": source_text, "language": "zh"}]
        chars = [ch for ch in source_text if ch not in "。"]
        return [{
            "text": source_text,
            "language": "zh",
            "timestamp": [[index * 100, index * 100 + 80] for index, _ch in enumerate(chars)],
        }]

    monkeypatch.setattr(transcriber, "_generate", generate)
    monkeypatch.setattr(
        transcriber,
        "_run_sensevoice_wallclock_vad",
        lambda model, audio, options, on_progress: (
            [Segment(start=10.0, end=12.0, text="这段锚点与正文完全无关。")],
            "zh",
            {
                "timing_mode": "wallclock_vad_chunks",
                "timing_reliable": True,
                "wallclock_vad_chunks": 1,
            },
        ),
    )

    segments, _ = transcriber._run(
        tmp_path / "audio.wav",
        TranscribeOptions(model_id=SENSEVOICE_MODEL),
        None,
    )

    assert "".join(segment.text for segment in segments) == source_text
    assert transcriber.last_filter_stats["timing_mode"] == "aligned_to_paraformer_recovery_anchor"
    assert transcriber.last_filter_stats["timing_reliable"] is True
    assert transcriber.last_filter_stats["wallclock_alignment_ok"] is False
    assert transcriber.last_filter_stats["wallclock_alignment_reason"] == "source_anchor_text_too_different"
    assert transcriber.last_filter_stats["paraformer_anchor_mode"] == "recovery"
    assert transcriber.last_filter_stats["equal_char_ratio"] == 1.0
    assert transcriber.last_filter_stats["timing_alignment_reason"] is None
    assert transcriber.last_filter_stats["settings"]["sensevoice_paraformer_anchor"] is True
    detector_segments, detector_source, detector_is_paraformer = (
        transcriber.strong_asr_detector_snapshot()
    )
    assert detector_is_paraformer is True
    assert detector_source == "paraformer_recovery_timing_anchor"
    assert "".join(segment.text for segment in detector_segments) == source_text


def test_sensevoice_low_density_preflight_skips_expensive_wallclock_chunks(monkeypatch, tmp_path: Path):
    from scribe_py.core import audio as audio_module

    transcriber = FunASRTranscriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN", "1")
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_PARAFORMER_ANCHOR", raising=False)
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD", raising=False)
    monkeypatch.setattr(funasr_module, "model_cached", lambda model_id: True)
    monkeypatch.setattr(audio_module, "probe_audio", lambda _audio: {"duration": 600.0})
    monkeypatch.setattr(transcriber, "_speech_ranges", lambda _audio: [])
    monkeypatch.setattr(transcriber, "_load", lambda model_id: object())

    source_text = "庭审双方正在核实事实。"

    def generate(_model, _audio, _options, *, sensevoice):
        if sensevoice:
            return [{"text": source_text, "language": "zh"}]
        chars = [ch for ch in source_text if ch not in "。"]
        return [{
            "text": source_text,
            "language": "zh",
            "timestamp": [[index * 100, index * 100 + 80] for index, _ch in enumerate(chars)],
        }]

    monkeypatch.setattr(transcriber, "_generate", generate)
    monkeypatch.setattr(
        transcriber,
        "_run_sensevoice_wallclock_vad",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("low-density preflight should skip wallclock chunk ASR")
        ),
    )

    segments, _ = transcriber._run(
        tmp_path / "audio.wav",
        TranscribeOptions(model_id=SENSEVOICE_MODEL),
        None,
    )

    assert "".join(segment.text for segment in segments) == source_text
    assert transcriber.last_filter_stats["timing_mode"] == "aligned_to_paraformer_recovery_anchor"
    assert transcriber.last_filter_stats["timing_reliable"] is True
    assert transcriber.last_filter_stats["paraformer_anchor_mode"] == "recovery"
    assert transcriber.last_filter_stats["paraformer_preflight"]["selected"] is True


def test_sensevoice_rejects_single_item_coarse_paraformer_anchor(monkeypatch, tmp_path: Path):
    from scribe_py.core import audio as audio_module

    transcriber = FunASRTranscriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN", "1")
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_PARAFORMER_ANCHOR", raising=False)
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD", raising=False)
    monkeypatch.setattr(funasr_module, "model_cached", lambda _model_id: True)
    monkeypatch.setattr(audio_module, "probe_audio", lambda _audio: {"duration": 600.0})
    monkeypatch.setattr(transcriber, "_speech_ranges", lambda _audio: [])
    monkeypatch.setattr(transcriber, "_load", lambda _model_id: object())

    source_text = "第一句完整文字。第二句完整文字。"

    def generate(_model, _audio, _options, *, sensevoice):
        if sensevoice:
            return [{"text": source_text, "language": "zh"}]
        return [{
            "text": source_text,
            "language": "zh",
            "timestamp": [[0, 590000]],
        }]

    monkeypatch.setattr(transcriber, "_generate", generate)
    wallclock_calls = []

    def wallclock(*_args, **_kwargs):
        wallclock_calls.append(True)
        return (
            [
                Segment(start=10.0, end=12.0, text="第一句完整文字。"),
                Segment(start=20.0, end=22.0, text="第二句完整文字。"),
            ],
            "zh",
            {
                "timing_mode": "wallclock_vad_chunks",
                "timing_reliable": True,
                "wallclock_vad_chunks": 2,
            },
        )

    monkeypatch.setattr(transcriber, "_run_sensevoice_wallclock_vad", wallclock)

    segments, _ = transcriber._run(
        tmp_path / "audio.wav",
        TranscribeOptions(model_id=SENSEVOICE_MODEL),
        None,
    )

    assert wallclock_calls == [True]
    assert "".join(segment.text for segment in segments) == source_text
    assert transcriber.last_filter_stats["timing_mode"] == "aligned_to_wallclock_anchor"
    assert transcriber.last_filter_stats["paraformer_anchor_ok"] is False
    assert (
        transcriber.last_filter_stats["paraformer_anchor_reason"]
        == "single_item_coarse_timestamp_projection"
    )
    assert transcriber.last_filter_stats["paraformer_anchor_timing_precision"] == "coarse"


def test_long_low_density_recording_keeps_wallclock_timeline(monkeypatch, tmp_path: Path):
    from scribe_py.core import audio as audio_module

    transcriber = FunASRTranscriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN", "1")
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_PARAFORMER_ANCHOR", raising=False)
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_PARAFORMER_PREFLIGHT_MAX_S", raising=False)
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD", raising=False)
    monkeypatch.setattr(funasr_module, "model_cached", lambda _model_id: True)
    monkeypatch.setattr(audio_module, "probe_audio", lambda _audio: {"duration": 2398.443})
    monkeypatch.setattr(transcriber, "_speech_ranges", lambda _audio: [])
    monkeypatch.setattr(transcriber, "_load", lambda _model_id: object())

    source_text = "第一句完整文字。第二句完整文字。"
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, _audio, _options, *, sensevoice: (
            [{"text": source_text, "language": "zh"}]
            if sensevoice
            else (_ for _ in ()).throw(
                AssertionError("long recordings must not preflight with Paraformer")
            )
        ),
    )
    wallclock_calls = []

    def wallclock(*_args, **_kwargs):
        wallclock_calls.append(True)
        return (
            [
                Segment(start=1.0, end=4.0, text="第一句完整文字。"),
                Segment(start=8.0, end=11.0, text="第二句完整文字。"),
            ],
            "zh",
            {
                "timing_mode": "wallclock_vad_chunks",
                "timing_reliable": True,
                "wallclock_vad_chunks": 2,
            },
        )

    monkeypatch.setattr(transcriber, "_run_sensevoice_wallclock_vad", wallclock)

    segments, _ = transcriber._run(
        tmp_path / "audio.wav",
        TranscribeOptions(model_id=SENSEVOICE_MODEL),
        None,
    )

    assert wallclock_calls == [True]
    assert "".join(segment.text for segment in segments) == source_text
    assert transcriber.last_filter_stats["timing_mode"] == "aligned_to_wallclock_anchor"
    assert transcriber.last_filter_stats["timing_reliable"] is True
    assert transcriber.last_filter_stats["paraformer_preflight"]["selected"] is False
    assert transcriber.last_filter_stats["paraformer_preflight"]["reason"] == "recording_too_long"
    assert transcriber.last_filter_stats["settings"]["sensevoice_paraformer_anchor"] is False


def test_sensevoice_explicitly_disabled_paraformer_does_not_auto_recover(monkeypatch, tmp_path: Path):
    transcriber = FunASRTranscriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN", "1")
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_PARAFORMER_ANCHOR", "0")
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD", raising=False)
    monkeypatch.setattr(funasr_module, "model_cached", lambda model_id: True)
    monkeypatch.setattr(transcriber, "_load", lambda model_id: object())
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda model, audio, options, *, sensevoice: [{"text": "第一句完整文字。", "language": "zh"}],
    )
    monkeypatch.setattr(
        transcriber,
        "_run_sensevoice_wallclock_vad",
        lambda model, audio, options, on_progress: (
            [Segment(start=10.0, end=12.0, text="完全不同内容。")],
            "zh",
            {"timing_mode": "wallclock_vad_chunks", "timing_reliable": True},
        ),
    )

    segments, _ = transcriber._run(
        tmp_path / "audio.wav",
        TranscribeOptions(model_id=SENSEVOICE_MODEL),
        None,
    )

    assert segments
    assert transcriber.last_filter_stats["timing_mode"] == "coarse_text_distribution"
    assert transcriber.last_filter_stats["timing_reliable"] is False
    assert transcriber.last_filter_stats["settings"]["sensevoice_paraformer_anchor"] is False


def test_sensevoice_timing_alignment_request_false_overrides_env(monkeypatch, tmp_path: Path):
    transcriber = FunASRTranscriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN", "1")
    monkeypatch.delenv("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD", raising=False)
    monkeypatch.setattr(transcriber, "_load", lambda model_id: object())
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda model, audio, options, *, sensevoice: [{"text": "第一句。第二句。", "language": "zh"}],
    )
    monkeypatch.setattr(
        transcriber,
        "_run_sensevoice_wallclock_vad",
        lambda model, audio, options, on_progress: (_ for _ in ()).throw(
            AssertionError("request timing_align=False should skip wallclock alignment")
        ),
    )

    segments, _ = transcriber._run(
        tmp_path / "audio.wav",
        TranscribeOptions(model_id=SENSEVOICE_MODEL, timing_align=False),
        None,
    )

    assert segments
    assert transcriber.last_filter_stats["timing_mode"] == "coarse_text_distribution"
    assert transcriber.last_filter_stats["settings"]["sensevoice_timing_align"] is False


def test_sensevoice_coarse_timing_is_marked_unreliable_when_fallback_disabled(monkeypatch, tmp_path: Path):
    from scribe_py.core import audio as audio_module

    transcriber = FunASRTranscriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD", "0")
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN", "0")
    monkeypatch.setattr(audio_module, "probe_audio", lambda audio: {"duration": 100.0})
    monkeypatch.setattr(transcriber, "_load", lambda model_id: object())
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda model, audio, options, *, sensevoice: [{"text": "第一句。第二句。", "language": "zh"}],
    )

    segments, _ = transcriber._run(
        tmp_path / "audio.wav",
        TranscribeOptions(model_id=SENSEVOICE_MODEL),
        None,
    )

    assert segments
    assert segments[0].start == 0.0
    assert segments[-1].end == 100.0
    assert transcriber.last_filter_stats["timing_mode"] == "coarse_text_distribution"
    assert transcriber.last_filter_stats["timing_reliable"] is False


def test_timing_alignment_rejects_unrelated_anchor_text():
    aligned, stats = _align_segments_to_timing_anchor(
        [Segment(start=0.0, end=0.0, text="第一句完整文字。")],
        [Segment(start=10.0, end=12.0, text="完全不同内容。")],
        min_equal_ratio=0.8,
    )

    assert aligned == []
    assert stats["timing_alignment_ok"] is False
    assert stats["timing_alignment_reason"] == "source_anchor_text_too_different"


def test_timing_alignment_outputs_monotonic_segments():
    aligned, stats = _align_segments_to_timing_anchor(
        [
            Segment(start=0.0, end=0.0, text="第一句完整文字。"),
            Segment(start=0.0, end=0.0, text="第二句完整文字。"),
        ],
        [
            Segment(
                start=0.0,
                end=1.0,
                text="第一句完整文字。",
                sync_cues=[{"start": 0.0, "end": 1.0, "text": "第一句完整文字。"}],
            ),
            Segment(
                start=0.92,
                end=2.0,
                text="第二句完整文字。",
                sync_cues=[{"start": 0.92, "end": 2.0, "text": "第二句完整文字。"}],
            ),
        ],
        min_equal_ratio=0.8,
    )

    assert stats["timing_alignment_ok"] is True
    assert len(aligned) == 2
    assert aligned[1].start >= aligned[0].end
    assert aligned[0].sync_cues
    assert aligned[1].sync_cues


def test_timing_alignment_keeps_short_deleted_prefix_next_to_later_speech():
    aligned, stats = _align_segments_to_timing_anchor(
        [
            Segment(start=0.0, end=0.0, text="前文一致。"),
            Segment(start=0.0, end=0.0, text="哦，后文一致。"),
        ],
        [
            Segment(
                start=0.0,
                end=2.0,
                text="前文一致。",
                sync_cues=[{"start": 0.0, "end": 2.0, "text": "前文一致。"}],
            ),
            Segment(
                start=20.0,
                end=22.0,
                text="后文一致。",
                sync_cues=[{"start": 20.0, "end": 22.0, "text": "后文一致。"}],
            ),
        ],
        min_equal_ratio=0.8,
    )

    assert stats["timing_alignment_ok"] is True
    assert aligned[1].start >= 19.0
    assert aligned[1].sync_cues


def test_timing_alignment_does_not_spread_multi_segment_deletion_across_silence():
    aligned, stats = _align_segments_to_timing_anchor(
        [
            Segment(start=0.0, end=0.0, text="前文。"),
            Segment(start=0.0, end=0.0, text="噪声。"),
            Segment(start=0.0, end=0.0, text="哦，后文。"),
        ],
        [
            Segment(
                start=0.0,
                end=2.0,
                text="前文。",
                sync_cues=[{"start": 0.0, "end": 2.0, "text": "前文。"}],
            ),
            Segment(
                start=20.0,
                end=22.0,
                text="后文。",
                sync_cues=[{"start": 20.0, "end": 22.0, "text": "后文。"}],
            ),
        ],
        min_equal_ratio=0.5,
    )

    assert stats["timing_alignment_ok"] is True
    assert aligned[1].end <= 5.0
    assert aligned[2].start >= 19.0


def test_sync_cues_split_at_long_acoustic_gap():
    text = "等下真没有吧。"
    char_times = [
        (1.0, 1.2),
        (10.0, 10.2),
        (10.2, 10.4),
        (10.4, 10.6),
        (10.6, 10.8),
        (10.8, 11.0),
        (11.0, 11.0),
    ]

    cues = _build_sync_cues_from_char_times(
        text,
        char_times,
        segment_start=1.0,
        segment_end=11.0,
    )

    assert [cue["text"] for cue in cues] == ["等", "下真没有吧。"]
    assert cues[0]["end"] < cues[1]["start"]


def test_timing_alignment_distributes_replaced_leading_segment_across_anchor_interval():
    aligned, stats = _align_segments_to_timing_anchor(
        [
            Segment(start=0.0, end=0.0, text="甲乙丙丁戊己。"),
            Segment(start=0.0, end=0.0, text="后续内容完全一致。"),
        ],
        [
            Segment(
                start=0.0,
                end=4.0,
                text="天地玄黄宇宙。",
                sync_cues=[{"start": 0.0, "end": 4.0, "text": "天地玄黄宇宙。"}],
            ),
            Segment(
                start=4.0,
                end=8.0,
                text="后续内容完全一致。",
                sync_cues=[{"start": 4.0, "end": 8.0, "text": "后续内容完全一致。"}],
            ),
        ],
        min_equal_ratio=0.4,
    )

    assert stats["timing_alignment_ok"] is True
    assert stats["estimated_timing_chars"] >= 6
    assert aligned[0].start == 0.0
    assert aligned[0].end >= 3.0
    assert aligned[1].start >= aligned[0].end
    assert all(
        float(cue["end"]) > float(cue["start"])
        for segment in aligned
        for cue in segment.sync_cues or []
    )


def test_timing_alignment_distributes_replaced_trailing_segment_across_anchor_interval():
    aligned, stats = _align_segments_to_timing_anchor(
        [
            Segment(start=0.0, end=0.0, text="前面内容完全一致。"),
            Segment(start=0.0, end=0.0, text="甲乙丙丁戊己。"),
        ],
        [
            Segment(
                start=0.0,
                end=4.0,
                text="前面内容完全一致。",
                sync_cues=[{"start": 0.0, "end": 4.0, "text": "前面内容完全一致。"}],
            ),
            Segment(
                start=4.0,
                end=8.0,
                text="天地玄黄宇宙。",
                sync_cues=[{"start": 4.0, "end": 8.0, "text": "天地玄黄宇宙。"}],
            ),
        ],
        min_equal_ratio=0.4,
    )

    assert stats["timing_alignment_ok"] is True
    assert aligned[1].start >= 3.5
    assert aligned[1].end >= 7.5
    assert aligned[1].end > aligned[1].start


def test_timing_alignment_reflows_leading_segment_omitted_by_anchor():
    aligned, stats = _align_segments_to_timing_anchor(
        [
            Segment(start=0.0, end=0.0, text="甲乙丙丁戊己。"),
            Segment(start=0.0, end=0.0, text="后续内容完全一致。"),
        ],
        [
            Segment(
                start=0.0,
                end=8.0,
                text="后续内容完全一致。",
                sync_cues=[{"start": 0.0, "end": 8.0, "text": "后续内容完全一致。"}],
            ),
        ],
        min_equal_ratio=0.4,
    )

    assert stats["timing_alignment_ok"] is True
    assert stats["repaired_collapsed_edges"] == ["leading"]
    assert aligned[0].end >= 2.0
    assert aligned[1].start >= aligned[0].end


def test_timing_alignment_reflows_trailing_segment_omitted_by_anchor():
    aligned, stats = _align_segments_to_timing_anchor(
        [
            Segment(start=0.0, end=0.0, text="前面内容完全一致。"),
            Segment(start=0.0, end=0.0, text="甲乙丙丁戊己。"),
        ],
        [
            Segment(
                start=0.0,
                end=8.0,
                text="前面内容完全一致。",
                sync_cues=[{"start": 0.0, "end": 8.0, "text": "前面内容完全一致。"}],
            ),
        ],
        min_equal_ratio=0.4,
    )

    assert stats["timing_alignment_ok"] is True
    assert stats["repaired_collapsed_edges"] == ["trailing"]
    assert aligned[1].start >= 4.0
    assert aligned[1].end >= 7.5


def test_timing_alignment_keeps_short_trailing_acknowledgement_out_of_edge_reflow():
    aligned, stats = _align_segments_to_timing_anchor(
        [
            Segment(start=0.0, end=0.0, text="前面内容完全一致。"),
            Segment(start=0.0, end=0.0, text="好。"),
        ],
        [
            Segment(
                start=0.0,
                end=8.0,
                text="前面内容完全一致。",
                sync_cues=[{"start": 0.0, "end": 8.0, "text": "前面内容完全一致。"}],
            ),
        ],
        min_equal_ratio=0.4,
    )

    assert stats["timing_alignment_ok"] is True
    assert stats["repaired_collapsed_edges"] == []
    assert aligned[1].end - aligned[1].start >= 0.15


def test_sync_cue_realign_repairs_cross_segment_text_without_changing_geometry():
    anchor = [
        Segment(
            start=0.0,
            end=1.0,
            text="前面的工",
            sync_cues=[{"start": 0.0, "end": 1.0, "text": "前面的工"}],
        ),
        Segment(
            start=1.0,
            end=2.0,
            text="作，后文。",
            sync_cues=[{"start": 1.0, "end": 2.0, "text": "作，后文。"}],
        ),
    ]
    normalized = [
        Segment(
            start=0.0,
            end=1.05,
            text="前面的工作。",
            sync_cues=[{"start": 0.0, "end": 0.9, "text": "前面的工"}],
        ),
        Segment(
            start=1.05,
            end=2.0,
            text="后文。",
            sync_cues=[{"start": 1.0, "end": 2.0, "text": "作，后文。"}],
        ),
    ]
    geometry_before = [(segment.start, segment.end, segment.text) for segment in normalized]

    repaired, stats = _realign_sync_cues_preserving_segments(normalized, anchor)

    assert [(segment.start, segment.end, segment.text) for segment in repaired] == geometry_before
    assert stats["repaired_segments"] == 2
    assert stats["mismatched_segments_after"] == 0
    assert [
        _sync_text
        for segment in repaired
        for _sync_text in ["".join(str(cue["text"]) for cue in segment.sync_cues or [])]
    ] == ["前面的工作。", "后文。"]
    assert all(
        segment.start <= float(cue["start"]) <= float(cue["end"]) <= segment.end
        for segment in repaired
        for cue in segment.sync_cues or []
    )


def test_sync_cue_realign_does_not_collapse_replaced_edge_text():
    anchor = [
        Segment(
            start=0.0,
            end=4.0,
            text="天地玄黄宇宙，后续一致。",
            sync_cues=[{"start": 0.0, "end": 4.0, "text": "天地玄黄宇宙，后续一致。"}],
        )
    ]
    normalized = [
        Segment(
            start=0.0,
            end=4.0,
            text="甲乙丙丁戊己，后续一致。",
            sync_cues=[{"start": 0.0, "end": 4.0, "text": "天地玄黄宇宙，后续一致。"}],
        )
    ]

    repaired, stats = _realign_sync_cues_preserving_segments(
        normalized,
        anchor,
        min_equal_ratio=0.3,
    )

    assert stats["repaired_segments"] == 1
    assert repaired[0].start == 0.0
    assert repaired[0].end == 4.0
    assert "".join(str(cue["text"]) for cue in repaired[0].sync_cues or []) == normalized[0].text
    assert all(
        float(cue["end"]) > float(cue["start"])
        for cue in repaired[0].sync_cues or []
    )


def test_funasr_resolves_bundled_modelscope_cache(monkeypatch, tmp_path: Path):
    cache = tmp_path / "modelscope" / "hub"
    model = cache / "models" / "iic" / "SenseVoiceSmall"
    model.mkdir(parents=True)
    (model / "config.yaml").write_text("model: SenseVoiceSmall\n", encoding="utf-8")
    (model / "model.pt").write_bytes(b"fake")

    monkeypatch.setenv("LOCALSCRIBE_MODELSCOPE_CACHE", str(cache))

    assert model_cached("iic/SenseVoiceSmall") is True
    assert resolve_model_path("iic/SenseVoiceSmall") == str(model)


def test_normalizer_forces_simplified_chinese_in_text_and_original_text():
    segments, stats = normalize_segments([
        Segment(start=0.0, end=2.0, text="當我們進入學校的時候")
    ])

    assert segments[0].text == "当我们进入学校的时候。"
    assert segments[0].original_text == "当我们进入学校的时候"
    assert "當" not in segments[0].text
    assert "進" not in segments[0].text
    assert "學" not in segments[0].text
    assert "當" not in (segments[0].original_text or "")
    assert stats["simplified_segments"] == 1


def test_asr_quality_detects_less_common_traditional_characters():
    report = build_asr_quality_report([
        Segment(start=0.0, end=2.0, text="现在學校出现狀况。")
    ])

    assert "學" in report["traditional_char_hits"]
    assert "狀" in report["traditional_char_hits"]
    assert "仍存在繁体字" in report["risk_reasons"]


def test_normalizer_rewrites_traditional_aspect_particle_but_not_zhuzuo():
    text, _ = normalize_transcript_text("不是代表说他跟著团契没有关系，他正在写著作。")
    report = build_asr_quality_report([Segment(start=0.0, end=2.0, text=text)])

    assert "跟着团契" in text
    assert "著作" in text
    assert "著" not in report["traditional_char_hits"]
    assert "仍存在繁体字" not in report["risk_reasons"]


def test_recursive_simplifier_skips_paths_and_config_fields():
    payload = {
        "audio": "/tmp/錄音/當我們.m4a",
        "docx_path": "/tmp/會議紀要.docx",
        "model_id": "mlx-community/whisper-large-v3-turbo",
        "segments": [
            {
                "text": "當我們進入學校的時候，他跟著團契。",
                "original_text": "當我們進入學校的時候",
            }
        ],
        "diarization_stats": {
            "review_segments": [{"text": "好 謝謝 張目中。"}],
        },
        "filter_stats": {
            "speech_coverage": {
                "local_recovery": {
                    "details": [{
                        "inserted_raw_text": "當我們進入學校的時候",
                        "attempts": [{"raw": "當我們進入學校的時候", "residual_text": "當我們"}],
                    }]
                }
            }
        },
    }

    cleaned = simplify_chinese_value(payload)

    assert cleaned["audio"] == payload["audio"]
    assert cleaned["docx_path"] == payload["docx_path"]
    assert cleaned["model_id"] == payload["model_id"]
    assert cleaned["segments"][0]["text"] == "当我们进入学校的时候，他跟着团契。"
    assert cleaned["segments"][0]["original_text"] == "當我們進入學校的時候"
    recovery = cleaned["filter_stats"]["speech_coverage"]["local_recovery"]
    assert recovery["details"][0]["inserted_raw_text"] == "當我們進入學校的時候"
    assert recovery["details"][0]["attempts"][0]["raw"] == "當我們進入學校的時候"
    assert recovery["details"][0]["attempts"][0]["residual_text"] == "當我們"
    assert cleaned["diarization_stats"]["review_segments"][0]["text"] == "好 谢谢 张目中。"


def test_segment_serializes_original_text_even_when_unchanged():
    payload = Segment(0.0, 1.0, "新增内容", original_text="新增内容").to_dict()

    assert payload["original_text"] == "新增内容"


def test_normalizer_does_not_rewrite_uncertain_words_without_confirmed_context():
    text, stats = normalize_transcript_text(
        "有点矫正嗯，我们先看这个设备怎么调整",
        context="圣诞节 教会 赞美",
        language="zh",
    )

    assert "矫正嗯" in text
    assert "焦躁" not in text
    assert stats["terminal_punctuation_added"] is True


def test_default_normalizer_applies_only_unambiguous_character_cleanup():
    text, stats = normalize_transcript_text(
        "即江的安排里有活务，多多少美的毛盾，那很正。",
        context="普通会议安排",
        language="zh",
    )

    assert text == "即将要安排里有活动，多多少少的矛盾，那很正常。"
    # No named person, meeting-specific phrase, or domain vocabulary is enabled
    # by the customer App's default path.
    assert stats["profile"] is None
    assert stats["lexical_rewrites_enabled"] is False
    assert stats["safe_replacements"] >= 5


def test_normalizer_applies_family_legal_context_asr_cleanup():
    text, stats = normalize_transcript_text(
        "就骚谣我，咱那调查去，我绝对不承受，他真的不辱我，我觉他好极了。"
        "对吧回训慎重考虑一下，好吧啊，现在别拿出这个家长的气词。"
        "说的直接您好养，不是你们养过子女了。一个我刚才说了，跟司女也商量了。"
        "是重好虑规定，如果仍然坚持离婚。",
        context="子女 孩子 家长 离婚 婚姻 调查 侮辱",
        language="zh",
        profile="legacy_general",
    )

    assert "他造谣我" in text
    assert "咱们可以调查去" in text
    assert "侮辱我" in text
    assert "我对他好极了" in text
    assert "回去再慎重考虑一下" in text
    assert "家长的气势" in text
    assert "您得靠子女养活" in text
    assert "跟子女也商量商量" in text
    assert "慎重考虑婚姻问题" in text
    assert "骚谣" not in text
    assert "司女" not in text
    assert stats["safe_replacements"] >= 8
    assert "命中家庭/调解场景 ASR 混淆" in stats["asr_review_reasons"]


def test_normalizer_does_not_apply_family_legal_cleanup_without_context():
    text, stats = normalize_transcript_text(
        "这个字段叫司女，回训慎重是测试用例里的代号。",
        context="技术测试 数据字段",
        language="zh",
    )

    assert "司女" in text
    assert "回训慎重" in text
    assert stats["safe_replacements"] == 0


def test_normalizer_applies_confirmed_standard3_phrase_fix():
    text, _ = normalize_transcript_text(
        "有点矫正嗯，大家一定要认出这个情况，并不是说这里面说哎这个人平时脾挺好的，很自然就回不好了",
        context="圣诞节 教会 赞美 这个人平时脾挺好的",
        language="zh",
        profile="legacy_general",
    )

    assert "有点搅扰" in text
    assert "认清这个情况" in text
    assert "这个人平时脾气挺好的" in text
    assert "怎么突然间脾气不好了" in text
    assert "焦躁" not in text


def test_normalizer_marks_confirmed_fix_for_review():
    segments, stats = normalize_segments([
        Segment(
            start=22.747,
            end=32.347,
            text="有点矫正嗯，大家一定要认清这个情况，并不是说这里面说哎这个人平时脾挺好的，很自然就回不好了",
        )
    ], profile="legacy_general")

    assert segments[0].text == "有点搅扰，大家一定要认清这个情况，并不是说这里面说哎这个人平时脾气挺好的，怎么突然间脾气不好了。"
    assert stats["asr_review_segment_count"] == 1
    assert "已应用通用强上下文纠错" in stats["asr_review_segments"][0]["reasons"]


def test_normalizer_merges_youth_fellowship_boundary_error():
    segments, stats = normalize_segments([
        Segment(start=182.393, end=183.645, text="我相信在清。"),
        Segment(
            start=183.645,
            end=196.166,
            text="团气的每一位君子的每人没有人希望看到团气是就是该怎么讲，里面有很多很多的就是巴不得这个团气乱",
        ),
    ], profile="legacy_general")

    assert len(segments) == 1
    assert segments[0].start == 182.393
    assert segments[0].end == 196.166
    assert segments[0].text == (
        "我相信在青年团契的每一位姊妹，没有人希望看到团契是，就是该怎么讲，"
        "里面有很多很多的就是巴不得这个团契乱。"
    )
    assert stats["contextual_asr_merges"] == 1
    assert stats["asr_review_segment_count"] >= 1


def test_normalizer_merges_split_negative_good_in_church_context():
    segments, stats = normalize_segments([
        Segment(
            start=4.84,
            end=16.01,
            text="我相信在青年团气的每一位女子内，没有人希望看到团气是就是该怎么讲，里面有很多很多的就是巴不得这个团气乱，巴不得这个团气不",
        ),
        Segment(start=16.01, end=27.18, text="好，巴不得团气有好多好多的矛盾"),
    ], profile="legacy_general")

    assert len(segments) == 1
    assert "团契是，就是该怎么讲" in segments[0].text
    assert "巴不得这个团契不好，巴不得团契有好多好多的矛盾" in segments[0].text
    assert stats["contextual_asr_merges"] == 1


def test_normalizer_applies_confirmed_prayer_phrase_fix():
    text, stats = normalize_transcript_text(
        "当然文字也很也很温暖我呀，也很也很温暖我，但是最有质量的因为我们好好，那么我就说到这里吧",
        context="最好的表达方式也不是文字 当然文字也很温暖 我就说到这里",
        language="zh",
        profile="legacy_general",
    )

    assert "当然文字也很温暖，我呀也很温暖" in text
    assert "但是最有力量的是为我祷告" in text
    assert "最有质量" not in text
    assert "已应用通用强上下文纠错" in stats["asr_review_reasons"]


def test_normalizer_does_not_rewrite_quality_without_prayer_context():
    text, _ = normalize_transcript_text(
        "这个方案里面最有质量的是后面的交付流程",
        context="项目评审 交付流程",
        language="zh",
    )

    assert "最有质量" in text
    assert "为我祷告" not in text


def test_normalizer_unifies_confirmed_names_in_standard3_context():
    segments, stats = normalize_segments([
        Segment(start=340.0, end=346.0, text="然后因为今天李会没有来，本来我想说这里要因为李会是一个。"),
        Segment(start=359.0, end=365.0, text="是理慧，如果是按照我们的守则来说的话，你会去做这个动作。"),
        Segment(start=365.0, end=369.0, text="因为对方是男人和金子，所以他不好意思。"),
    ], profile="standard3")

    joined = "\n".join(s.text for s in segments)
    assert "李会没有来" in joined
    assert "李会去做这个动作" in joined
    assert "兰艺和金子" in joined
    assert "理慧" not in joined
    assert "男人和金子" not in joined
    assert stats["asr_review_segment_count"] >= 2


def test_default_profile_does_not_hardcode_recording_specific_names_in_strong_context():
    segments, stats = normalize_segments([
        Segment(start=340.0, end=346.0, text="然后因为今天李会没有来，本来我想说这里要因为李会是一个。"),
        Segment(start=359.0, end=365.0, text="是理慧，如果是按照我们的守则来说的话，你会去做这个动作。"),
        Segment(start=365.0, end=369.0, text="因为对方是男人和金子，所以他不好意思。"),
    ])

    joined = "\n".join(s.text for s in segments)
    assert "李会没有来" in joined
    assert "理慧" in joined
    assert "男人和金子" in joined
    assert "兰艺和金子" not in joined
    assert "已应用通用强上下文纠错" not in "\n".join(
        reason
        for item in stats["asr_review_segments"]
        for reason in item["reasons"]
    )


def test_normalizer_applies_rule_context_fixes_for_standard3():
    text, stats = normalize_transcript_text(
        "这就是为什么要把这个守的来做人性化的原因举例子，其实我的性免已经举完了。",
        context="守则 制度 金子 兰艺 服侍",
        language="zh",
        profile="legacy_general",
    )

    assert "这个守则拿来做人性化的原因" in text
    assert "我的例子就举完了" in text
    assert "守的来" not in text
    assert "性免" not in text
    assert "已应用通用强上下文纠错" in stats["asr_review_reasons"]


def test_normalizer_applies_gold_lanyi_segment_fixes_for_standard3():
    text, _ = normalize_transcript_text(
        "管有关个地，大家其他的从东其实也有私底下或者怎么样的来问我说凭什么他们俩那么长时间，他们俩没有来一年多没有没有不事，但还。",
        context="金子 兰艺 守则 他们俩没有来 一年多 服侍 群里",
        language="zh",
        profile="legacy_general",
    )

    assert "大家其他的从中其实也有私底下或怎样的人问我说，为什么他们俩那么长时间" in text
    assert "一年多没有服侍，但还" in text
    assert "管有关个地" not in text


def test_normalizer_handles_confirmed_lanyi_variant_without_llm():
    text, _ = normalize_transcript_text(
        "管有关地，大家其他的从东其实也有私底下或者怎么样的来问我说凭什么他们俩那么长时间，他们俩没有来一年多没有没有不事，但还在。",
        context="金子 兰艺 守则 他们俩没有来 一年多 服侍 群里",
        language="zh",
        profile="legacy_general",
    )

    assert text.startswith("大家其他的从中其实也有私底下或怎样的人问我说，为什么")
    assert "一年多没有服侍，但还在。" in text
    assert "管有关地" not in text
    assert "没有没有不事" not in text


def test_default_profile_does_not_apply_recording_specific_name_or_phrase_answers():
    text, stats = normalize_transcript_text(
        "李慧说这个方案里面最有质量的是后面的交付流程，蓝艺负责整理。",
        context="项目评审 交付流程",
        language="zh",
    )

    assert "李慧" in text
    assert "蓝艺" in text
    assert "最有质量" in text
    assert "李会" not in text
    assert "兰艺" not in text
    assert "为我祷告" not in text
    assert stats["profile"] is None


def test_default_profile_does_not_hardcode_technical_meeting_phrase_answers():
    text, stats = normalize_transcript_text(
        "也就是说，因为我们流写分离这块需要做一些改造。",
        context="技术会议 缓存 Redis 数据库 同步",
        language="zh",
    )

    assert "流写分离" in text
    assert "读写分离" not in text
    assert stats["profile"] is None
    assert stats["lexical_rewrites_enabled"] is False


def test_default_profile_preserves_strong_context_standard3_like_errors():
    text, stats = normalize_transcript_text(
        "再举一个就是也是跟金子，管有关个地，大家其他的从东其实也有私底下或者怎么样的来问我说凭什么他们俩那么长时间。",
        context="金子 守则 他们俩 群里",
        language="zh",
    )

    assert "管有关个地" in text
    assert "大家其他的从中" not in text
    assert "这个金子和兰艺" not in text
    assert "命中明显不通顺 ASR 片段" not in stats["asr_review_reasons"]
    assert "已应用通用强上下文纠错" not in stats["asr_review_reasons"]
    assert stats["lexical_rewrites_enabled"] is False


def test_default_profile_does_not_apply_strong_context_rules_without_domain_context():
    text, stats = normalize_transcript_text(
        "再举一个就是也是跟金子，管有关个地，大家其他的从东其实也有私底下或者怎么样的来问我说凭什么他们俩那么长时间。",
        context="项目复盘 用户访谈 采购流程",
        language="zh",
    )

    assert "管有关个地" in text
    assert "大家其他的从中" not in text
    assert "已应用通用强上下文纠错" not in stats["asr_review_reasons"]


def test_normalizer_applies_low_risk_asr_cleanup():
    text, _ = normalize_transcript_text(
        "，但是这里想知告诉大家，这是神的保手里面，这言是怎么说的吗？",
        context="教会 团契 祷告 服侍",
        language="zh",
        profile="legacy_general",
    )

    assert text.startswith("但是这里想告诉大家")
    assert "保守" in text
    assert "箴言是怎么说" in text
    assert not text.startswith("，")


def test_standard3_profile_fixes_confirmed_fellowship_variants():
    text, _ = normalize_transcript_text(
        "现在我们这青年团体都不叫团气了，就来团戏服侍了。",
        context="青年团契 服侍 教会",
        language="zh",
        profile="standard3",
    )

    assert "青年团契" in text
    assert "团契服侍" in text
    assert "团戏" not in text


def test_legacy_profile_fixes_confirmed_fellowship_variants_with_context():
    text, _ = normalize_transcript_text(
        "现在我们这青年团体都不叫团气了，就来团戏服侍了。",
        context="青年团契 服侍 教会",
        language="zh",
        profile="legacy_general",
    )

    assert "青年团契" in text
    assert "团契服侍" in text
    assert "团戏" not in text


def test_standard3_profile_merges_split_but_still_in_group():
    segments, stats = normalize_segments([
        Segment(start=526.0, end=527.0, text="他们俩没有来一年多没有没有不事，但还。"),
        Segment(start=527.1, end=527.6, text="在群里。"),
    ], profile="standard3")

    assert len(segments) == 1
    assert "一年多没有服侍，但还在群里。" in segments[0].text
    assert stats["contextual_asr_merges"] == 1


def test_legacy_profile_merges_split_but_still_in_group():
    segments, stats = normalize_segments([
        Segment(start=526.0, end=527.0, text="他们俩没有来一年多没有没有不事，但还。"),
        Segment(start=527.1, end=527.6, text="在群里。"),
    ], profile="legacy_general")

    assert len(segments) == 1
    assert "一年多没有服侍，但还在群里。" in segments[0].text
    assert stats["contextual_asr_merges"] == 1


def test_standard3_profile_merges_split_but_still_in_followed_by_group():
    segments, stats = normalize_segments([
        Segment(start=469.4, end=480.8, text="管有关地，大家其他的从东其实也有私底下或者怎么样的来问我说凭什么他们俩那么长时间，他们俩没有来一年多没有没有不事，但还在。"),
        Segment(start=480.8, end=481.4, text="群里。"),
    ], profile="standard3")

    assert len(segments) == 1
    assert segments[0].text.startswith("大家其他的从中其实也有私底下或怎样的人问我说，为什么")
    assert "一年多没有服侍，但还在群里。" in segments[0].text
    assert "管有关地" not in segments[0].text
    assert stats["contextual_asr_merges"] == 1


def test_technical_meeting_confusions_are_contextual_only():
    text, stats = normalize_transcript_text(
        "也就是说因为我们流写。",
        context="双活改造，读写分离，缓存，Redis，DNS，数据库切换。",
        profile="legacy_general",
    )

    assert text == "也就是说因为我们读写分离这块。"
    assert stats["safe_replacements"] >= 1
    assert "技术会议" in " ".join(stats["asr_review_reasons"])


def test_technical_cache_confusions_normalize_with_tech_context():
    text, stats = normalize_transcript_text(
        "分离这块也需要做一些工作的改造，要缓存，我们有，比如说我们管都管了，那ice的缓存数据，请他他就直接丢了，就我们怎么同入。",
        context="分离这块也需要做一些工作的改造，要缓存，我们有 redis，比如说缓存数据。",
        profile="legacy_general",
    )

    assert "有缓存，我们有 Redis" in text
    assert "我们缓存挂了" in text
    assert "Redis 的缓存数据" in text
    assert "相当于它就直接丢了" in text
    assert "那我们怎么同步" in text
    assert stats["safe_replacements"] >= 3


def test_technical_terms_do_not_rewrite_without_context():
    text, stats = normalize_transcript_text(
        "这个双模设计可以先不讨论。",
        context="今天主要聊会议安排。",
    )

    assert text == "这个双模设计可以先不讨论。"
    assert stats["safe_replacements"] == 0


def test_generic_normalizer_is_auditable_and_does_not_rewrite_known_answers():
    segments, stats = normalize_segments([
        Segment(
            start=0.0,
            end=5.0,
            text="當然文字也很溫暖但是最有質量的因為我們好好",
        )
    ])

    assert segments[0].text == "当然文字也很温暖但是最有质量的因为我们好好。"
    assert "为我祷告" not in segments[0].text
    assert stats["profile"] is None
    assert stats["lexical_rewrites_enabled"] is False
    assert stats["safe_replacements"] == 0
    assert stats["raw_text_sha256"] != stats["final_text_sha256"]
    assert stats["raw_chars"] > 0
    assert stats["final_chars"] > 0
