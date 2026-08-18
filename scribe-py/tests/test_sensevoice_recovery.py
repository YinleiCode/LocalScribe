from __future__ import annotations

import hashlib
import wave
from pathlib import Path

from scribe_py.core.sensevoice_recovery import (
    analyze_recovery_candidate,
    decide_recovery_attempts,
    deduplicate_candidate,
    group_failure_windows,
    is_repeated_hallucination,
    local_reference_from_segments,
    normalize_recovery_text,
)
from scribe_py.core.transcriber_funasr import (
    FunASRTranscriber,
    _cached_huggingface_snapshot,
    _recovery_snapshot,
    _suppress_vad_unsupported_segments,
)
from scribe_py.core.types import Segment, TranscribeOptions


def _write_silent_wav(path: Path, seconds: int = 5) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000 * seconds)


def test_vad_unsupported_text_guard_marks_hold_music_without_changing_timeline():
    segments = [
        Segment(start=0.0, end=6.0, text="这是正常讲话。"),
        Segment(
            start=10.0,
            end=30.0,
            text="走吧见。",
            sync_cues=[
                {"start": 10.0, "end": 20.0, "text": "走吧"},
                {"start": 20.0, "end": 30.0, "text": "见。"},
            ],
        ),
        Segment(start=30.0, end=30.5, text="。"),
        Segment(start=40.0, end=52.0, text="这段虽然没有 VAD 支持，但是文字密度正常。"),
    ]

    output, stats = _suppress_vad_unsupported_segments(
        segments,
        [(0.0, 6.0)],
        vad_status="ok",
    )

    assert [segment.text for segment in output] == [
        "这是正常讲话。",
        "（非语音）",
        "",
        "这段虽然没有 VAD 支持，但是文字密度正常。",
    ]
    assert [(segment.start, segment.end) for segment in output] == [
        (segment.start, segment.end) for segment in segments
    ]
    assert [
        (cue["start"], cue["end"])
        for cue in output[1].sync_cues or []
    ] == [(10.0, 20.0), (20.0, 30.0)]
    assert output[1].original_text == "走吧见。"
    assert stats["applied"] is True
    assert stats["suppressed_segments"] == 2
    assert stats["input_segments"] == stats["output_segments"] == 4
    assert stats["segment_geometry_preserved"] is True
    assert stats["sync_cue_boundaries_preserved"] is True
    assert stats["uses_recording_name"] is False
    assert stats["uses_fixed_transcript_phrases"] is False


def test_vad_unsupported_text_guard_fails_open_without_vad_evidence():
    segments = [Segment(start=10.0, end=30.0, text="很短。")]

    output, stats = _suppress_vad_unsupported_segments(
        segments,
        [],
        vad_status="unavailable",
    )

    assert output == segments
    assert stats["applied"] is False
    assert stats["reason"] == "vad_evidence_unavailable"


def test_vad_unsupported_text_guard_suppresses_punctuation_inside_speech():
    segments = [
        Segment(start=10.0, end=10.5, text="。", sync_cues=[
            {"start": 10.0, "end": 10.5, "text": "。"},
        ]),
    ]

    output, stats = _suppress_vad_unsupported_segments(
        segments,
        [(9.0, 12.0)],
        vad_status="ok",
    )

    assert output[0].text == ""
    assert output[0].original_text == "。"
    assert output[0].sync_cues == [{"start": 10.0, "end": 10.5, "text": ""}]
    assert (output[0].start, output[0].end) == (10.0, 10.5)
    assert stats["suppressed_segments"] == 1
    assert stats["items"][0]["reason"] == "standalone_punctuation"


def _attempt(
    raw: str,
    framing: str,
    *,
    pad_s: float = 0.0,
    left: str = "",
    right: str = "",
    reference: str = "",
    provider_id: str = "sensevoice-primary",
    provider_kind: str = "primary_asr",
    model_id: str = "iic/SenseVoiceSmall",
    model_family: str = "sensevoice",
    hallucination_risk: bool = False,
):
    attempt = analyze_recovery_candidate(
        raw,
        framing=framing,
        pad_s=pad_s,
        left=left,
        right=right,
        local_reference=reference,
        provider_id=provider_id,
        provider_kind=provider_kind,
        model_id=model_id,
        model_family=model_family,
        hallucination_risk=hallucination_risk,
    )
    attempt["slice_sha256"] = f"{provider_id}:{framing}"
    return attempt


def _qwen_attempt(raw: str, framing: str = "pad1.0", **kwargs):
    attempt = _attempt(
        raw,
        framing,
        pad_s=kwargs.pop("pad_s", 1.0),
        provider_id="qwen3-independent",
        provider_kind="independent_asr",
        model_id="mlx-community/Qwen3-ASR-1.7B-8bit",
        model_family="qwen3_asr",
        **kwargs,
    )
    attempt.update({
        "model_revision": "test-revision",
        "config_sha256": "config-sha256",
        "weights_manifest_sha256": "weights-manifest-sha256",
    })
    return attempt


def _enable_qwen_provider(transcriber: FunASRTranscriber, monkeypatch, text: str) -> None:
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_PROVIDER", "qwen3")
    metadata = {
        "provider_id": "qwen3-independent",
        "provider_kind": "independent_asr",
        "model_id": "mlx-community/Qwen3-ASR-1.7B-8bit",
        "model_family": "qwen3_asr",
        "model_revision": "test-revision",
        "config_sha256": "config-sha256",
        "weights_manifest_sha256": "weights-manifest-sha256",
    }
    monkeypatch.setattr(transcriber, "_load_local_recovery_provider", lambda _name: (object(), metadata, None))
    monkeypatch.setattr(transcriber, "_run_local_recovery_provider", lambda *_args: (text, {}))


def _pending_transcriber() -> FunASRTranscriber:
    transcriber = FunASRTranscriber(backend_name="sensevoice")
    transcriber._wallclock_attempted_ranges = [(1.0, 2.0)]
    transcriber._wallclock_recognized_ranges = []
    transcriber._wallclock_failed_ranges = [(1.0, 2.0)]
    transcriber._wallclock_failure_reasons = [
        {"start": 1.0, "end": 2.0, "reason": "low_text_density"}
    ]
    return transcriber


def test_cached_qwen_snapshot_requires_weights_and_builds_manifest(tmp_path: Path, monkeypatch):
    import huggingface_hub

    snapshot = tmp_path / "snapshots" / "revision-123"
    snapshot.mkdir(parents=True)
    config = snapshot / "config.json"
    config.write_text('{"model_type":"qwen3_asr"}', encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"model-weights")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *_args: str(config))

    resolved, metadata, error = _cached_huggingface_snapshot("test/model")

    assert resolved == snapshot
    assert error is None
    assert metadata["model_revision"] == "revision-123"
    assert metadata["config_sha256"]
    assert metadata["weights_manifest_sha256"]
    assert metadata["weight_files"] == 1


def test_group_failure_windows_merges_only_adjacent_fifty_ms():
    groups = group_failure_windows([
        {"start": 0.0, "end": 1.0, "reason": "a"},
        {"start": 1.05, "end": 2.0, "reason": "b"},
        {"start": 2.051, "end": 3.0, "reason": "c"},
    ])

    assert [(group["start"], group["end"]) for group in groups] == [(0.0, 2.0), (2.051, 3.0)]
    assert [item["reason"] for item in groups[0]["windows"]] == ["a", "b"]


def test_normalization_removes_event_labels_punctuation_and_nfkc_folds():
    assert normalize_recovery_text("<|zh|><|Speech|> ＡＢＣ， 你好！[music]") == "abc你好"


def test_deduplicate_candidate_uses_exact_left_and_right_overlaps():
    result = deduplicate_candidate("今天继续讨论明天", "我们今天", "明天见")

    assert result["left_overlap_chars"] == 2
    assert result["right_overlap_chars"] == 2
    assert result["residual_normalized"] == "继续讨论"
    assert result["residual_text"] == "继续讨论"


def test_local_reference_includes_overlapping_and_nearest_sides():
    context = local_reference_from_segments(
        [
            Segment(0.0, 1.0, "左侧"),
            Segment(1.0, 3.0, "覆盖正文"),
            Segment(3.0, 4.0, "右侧"),
        ],
        1.5,
        2.0,
    )

    assert context == {
        "left": "左侧",
        "right": "右侧",
        "overlapping": "覆盖正文",
        "reference": "覆盖正文",
    }


def test_candidate_in_local_reference_is_matched_existing():
    decision = decide_recovery_attempts([
        _attempt("目标正文", "exact", reference="前文目标正文后文"),
        _attempt("目标正文", "pad0.5", pad_s=0.5, reference="前文目标正文后文"),
        _qwen_attempt("目标正文", reference="前文目标正文后文"),
    ])

    assert decision["decision"] == "matched_existing"
    assert decision["inserted_text"] == ""


def test_primary_framings_without_independent_model_are_not_accepted():
    decision = decide_recovery_attempts([
        _attempt("新增内容", "exact"),
        _attempt("新增内容", "pad0.5", pad_s=0.5),
    ])

    assert decision["decision"] == "rejected"
    assert decision["primary_consensus"] == "新增内容"


def test_one_matching_framing_is_not_enough_to_repair_coverage():
    decision = decide_recovery_attempts([
        _attempt("目标正文", "exact", reference="前文目标正文后文"),
        _attempt("其他内容", "pad0.5", pad_s=0.5, reference="前文目标正文后文"),
    ])

    assert decision["decision"] == "rejected"


def test_one_primary_existing_match_plus_qwen_only_confirms_existing_text():
    decision = decide_recovery_attempts([
        _attempt("目标正文", "exact", reference="前文目标正文后文"),
        _attempt("其他内容", "pad0.5", pad_s=0.5, reference="前文目标正文后文"),
        _qwen_attempt("目标正文", reference="前文目标正文后文"),
    ])

    assert decision["decision"] == "matched_existing"
    assert decision["consensus"] == "目标正文"
    assert decision["inserted_text"] == ""
    assert decision["evidence_providers"] == ["qwen3-independent", "sensevoice-primary"]


def test_short_and_repeated_exact_existing_candidates_bypass_insertion_gates():
    short = analyze_recovery_candidate(
        "啊",
        framing="exact",
        pad_s=0.0,
        left="",
        right="",
        local_reference="前文啊后文",
        min_required_chars=4,
    )
    repeated = analyze_recovery_candidate(
        "谢谢谢谢谢谢",
        framing="pad0.5",
        pad_s=0.5,
        left="",
        right="",
        local_reference="他说谢谢谢谢谢谢然后结束",
        min_required_chars=8,
    )

    assert short["status"] == "matched_existing"
    assert short["residual"] == ""
    assert repeated["status"] == "matched_existing"
    assert repeated["residual"] == ""


def test_one_primary_valid_plus_qwen_cannot_authorize_insertion():
    decision = decide_recovery_attempts([
        _attempt("新增内容", "exact"),
        _qwen_attempt("新增内容"),
    ])

    assert decision["decision"] == "rejected"
    assert decision["inserted_text"] == ""


def test_unresolved_overlapping_reference_blocks_automatic_insertion():
    attempt = _attempt(
        "已有正文新增内容",
        "exact",
        reference="已有正文",
    )

    assert attempt["status"] == "rejected"
    assert attempt["rejection_reason"] == "overlapping_reference_unresolved"


def test_two_distinct_framings_with_identical_residual_are_inserted():
    attempts = [
        _attempt("前文新增内容后文", "exact", left="前文", right="后文"),
        _attempt("前文，新增内容，后文", "pad0.5", pad_s=0.5, left="前文", right="后文"),
        _qwen_attempt("前文新增内容后文", left="前文", right="后文"),
    ]

    decision = decide_recovery_attempts(attempts)

    assert decision["decision"] == "insert_accepted"
    assert decision["consensus"] == "新增内容"
    assert decision["evidence_framings"] == ["exact", "pad0.5", "pad1.0"]
    assert decision["evidence_providers"] == ["qwen3-independent", "sensevoice-primary"]
    assert normalize_recovery_text(decision["inserted_text"]) == "新增内容"


def test_qwen_conflict_and_hallucination_risk_are_rejected():
    primary = [
        _attempt("新增内容", "exact"),
        _attempt("新增内容", "pad0.5", pad_s=0.5),
    ]

    conflict = decide_recovery_attempts(primary + [_qwen_attempt("冲突内容")])
    hallucination = decide_recovery_attempts(
        primary + [_qwen_attempt("新增内容", hallucination_risk=True)]
    )

    assert conflict["decision"] == "rejected"
    assert hallucination["decision"] == "rejected"


def test_same_framing_does_not_count_as_two_evidence_sources():
    attempts = [
        _attempt("新增内容", "exact"),
        _attempt("新增内容", "exact"),
    ]

    assert decide_recovery_attempts(attempts)["decision"] == "rejected"


def test_conflicts_single_character_and_repeated_hallucination_are_rejected():
    conflict = decide_recovery_attempts([
        _attempt("甲乙", "exact"),
        _attempt("丙丁", "pad0.5", pad_s=0.5),
        _attempt("戊己", "pad1.0", pad_s=1.0),
    ])
    single = _attempt("啊", "exact")
    repeated = _attempt("谢谢谢谢谢谢", "exact")

    assert conflict["decision"] == "rejected"
    assert single["status"] == "rejected"
    assert single["rejection_reason"] == "single_character"
    assert repeated["status"] == "rejected"
    assert repeated["rejection_reason"] == "repeated_hallucination"
    assert is_repeated_hallucination("abcabcabc") is True


def test_audit_runs_recovery_but_changes_nothing(tmp_path: Path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_silent_wav(audio)
    transcriber = _pending_transcriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "audit")
    calls: list[str] = []

    def fake_generate(_model, chunk_path, _options, *, sensevoice):
        assert sensevoice is True
        calls.append(Path(chunk_path).stem)
        return [{"text": "新增内容", "language": "zh"}]

    monkeypatch.setattr(transcriber, "_generate", fake_generate)
    original = [Segment(0.0, 1.0, "前文"), Segment(2.0, 3.0, "后文")]

    segments, stats = transcriber._run_sensevoice_local_recovery(
        object(),
        audio,
        TranscribeOptions(model_id="iic/SenseVoiceSmall"),
        list(original),
        duration=5.0,
        on_progress=None,
    )

    assert segments == original
    assert transcriber._wallclock_recognized_ranges == []
    assert transcriber._wallclock_failed_ranges == [(1.0, 2.0)]
    assert stats["inserted"] == 0
    assert stats["rejected"] == 1
    assert stats["attempts"] == 2
    assert stats["before"] == stats["after"]
    assert len(calls) == 2


def test_audit_runs_qwen_even_when_primary_framings_conflict(tmp_path: Path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_silent_wav(audio)
    transcriber = _pending_transcriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "audit")
    _enable_qwen_provider(transcriber, monkeypatch, "独立结果")

    def fake_generate(_model, chunk_path, _options, *, sensevoice):
        stem = Path(chunk_path).stem
        if stem.endswith("exact"):
            text = "甲乙"
        elif stem.endswith("pad0_5"):
            text = "丙丁"
        else:
            text = "戊己"
        return [{"text": text, "language": "zh"}]

    monkeypatch.setattr(transcriber, "_generate", fake_generate)

    segments, stats = transcriber._run_sensevoice_local_recovery(
        object(),
        audio,
        TranscribeOptions(model_id="iic/SenseVoiceSmall"),
        [],
        duration=5.0,
        on_progress=None,
    )

    assert segments == []
    assert stats["mode"] == "audit"
    assert stats["attempts"] == 5
    assert stats["rejected"] == 1
    assert [item["provider_id"] for item in stats["details"][0]["attempts"]][-1] == "qwen3-independent"


def test_merge_runs_qwen_for_single_primary_existing_without_primary_consensus(tmp_path: Path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_silent_wav(audio)
    transcriber = _pending_transcriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "merge")
    _enable_qwen_provider(transcriber, monkeypatch, "目标正文")

    def fake_generate(_model, chunk_path, _options, *, sensevoice):
        assert sensevoice is True
        stem = Path(chunk_path).stem
        return [{"text": "目标正文" if stem.endswith("exact") else "冲突内容", "language": "zh"}]

    monkeypatch.setattr(transcriber, "_generate", fake_generate)
    original = [Segment(0.0, 3.0, "目标正文")]
    before_hash = hashlib.sha256("目标正文".encode()).hexdigest()

    segments, stats = transcriber._run_sensevoice_local_recovery(
        object(),
        audio,
        TranscribeOptions(model_id="iic/SenseVoiceSmall"),
        list(original),
        duration=5.0,
        on_progress=None,
    )

    assert segments == original
    assert stats["mode"] == "merge"
    assert stats["attempts"] == 5
    assert stats["matched_existing"] == 1
    assert stats["inserted"] == 0
    assert stats["details"][0]["decision"] == "matched_existing"
    assert stats["details"][0]["evidence_providers"] == ["qwen3-independent", "sensevoice-primary"]
    assert stats["before"]["text_sha256"] == before_hash
    assert stats["after"]["text_sha256"] == before_hash
    assert stats["before"]["segment_count"] == stats["after"]["segment_count"] == 1
    assert transcriber._wallclock_recognized_ranges == [(1.0, 2.0)]
    assert transcriber._wallclock_failed_ranges == []


def test_merge_request_without_provider_downgrades_to_audit(tmp_path: Path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_silent_wav(audio)
    transcriber = _pending_transcriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "merge")
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, _audio, _options, *, sensevoice: [{"text": "目标正文", "language": "zh"}],
    )
    original = [Segment(0.0, 3.0, "前文目标正文后文")]

    segments, stats = transcriber._run_sensevoice_local_recovery(
        object(), audio, TranscribeOptions(model_id="iic/SenseVoiceSmall"), list(original), duration=5.0, on_progress=None
    )

    assert segments == original
    assert stats["mode"] == "audit"
    assert stats["requested_mode"] == "merge"
    assert stats["diagnostic"] == "merge_requires_qwen3_independent_provider"
    assert stats["matched_existing"] == 0
    assert stats["rejected"] == 1
    assert stats["details"][0]["inserted_text"] == ""
    assert transcriber._wallclock_recognized_ranges == []
    assert transcriber._wallclock_failed_ranges == [(1.0, 2.0)]
    assert stats["before"] == stats["after"]


def test_merge_inserts_only_with_qwen_independent_consensus(tmp_path: Path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_silent_wav(audio)
    transcriber = _pending_transcriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "merge")
    _enable_qwen_provider(transcriber, monkeypatch, "前文新增内容后文")

    def fake_generate(_model, _audio, _options, *, sensevoice):
        assert sensevoice is True
        return [{"text": "前文新增内容后文", "language": "zh"}]

    monkeypatch.setattr(transcriber, "_generate", fake_generate)
    original = [Segment(0.0, 1.0, "前文"), Segment(2.0, 3.0, "后文")]

    segments, stats = transcriber._run_sensevoice_local_recovery(
        object(), audio, TranscribeOptions(model_id="iic/SenseVoiceSmall"), list(original), duration=5.0, on_progress=None
    )

    assert [segment.text for segment in segments] == ["前文", "新增内容", "后文"]
    assert stats["mode"] == "merge"
    assert stats["requested_mode"] == "merge"
    assert stats["inserted"] == 1
    assert stats["attempts"] == 3
    assert stats["before"] != stats["after"]
    assert transcriber._wallclock_recognized_ranges == [(1.0, 2.0)]
    assert transcriber._wallclock_failed_ranges == []


def test_merge_normalizes_inserted_candidate_with_transcript_normalizer(tmp_path: Path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_silent_wav(audio)
    transcriber = _pending_transcriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "merge")
    _enable_qwen_provider(transcriber, monkeypatch, "當我們進入學校的時候")
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, _audio, _options, *, sensevoice: [
            {"text": "當我們進入學校的時候", "language": "zh"}
        ],
    )

    segments, stats = transcriber._run_sensevoice_local_recovery(
        object(),
        audio,
        TranscribeOptions(model_id="iic/SenseVoiceSmall", language="auto"),
        [],
        duration=5.0,
        on_progress=None,
        normalization_language="zh",
    )

    assert [segment.text for segment in segments] == ["当我们进入学校的时候。"]
    assert segments[0].original_text == "當我們進入學校的時候"
    assert stats["details"][0]["inserted_raw_text"] == "當我們進入學校的時候"
    assert stats["details"][0]["inserted_text"] == "当我们进入学校的时候。"


def test_merge_records_safe_rejection_when_candidate_normalization_loses_identity(tmp_path: Path, monkeypatch):
    import scribe_py.core.transcriber_funasr as funasr_module

    audio = tmp_path / "audio.wav"
    _write_silent_wav(audio)
    transcriber = _pending_transcriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "merge")
    _enable_qwen_provider(transcriber, monkeypatch, "新增内容")
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, _audio, _options, *, sensevoice: [{"text": "新增内容", "language": "zh"}],
    )
    monkeypatch.setattr(funasr_module, "normalize_segments", lambda *_args, **_kwargs: ([], {"mode": "test"}))

    segments, stats = transcriber._run_sensevoice_local_recovery(
        object(),
        audio,
        TranscribeOptions(model_id="iic/SenseVoiceSmall", language="zh"),
        [],
        duration=5.0,
        on_progress=None,
        normalization_language="zh",
    )

    detail = stats["details"][0]
    assert segments == []
    assert stats["inserted"] == 0
    assert stats["rejected"] == 1
    assert detail["evidence_decision"] == "insert_accepted"
    assert detail["decision"] == "rejected"
    assert detail["normalization_rejection_reason"]
    assert detail["inserted_text"] == ""


def test_merge_downgrades_when_transcript_normalization_failed(tmp_path: Path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_silent_wav(audio)
    transcriber = _pending_transcriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "merge")
    _enable_qwen_provider(transcriber, monkeypatch, "新增内容")
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, _audio, _options, *, sensevoice: [{"text": "新增内容", "language": "zh"}],
    )

    segments, stats = transcriber._run_sensevoice_local_recovery(
        object(),
        audio,
        TranscribeOptions(model_id="iic/SenseVoiceSmall", language="zh"),
        [],
        duration=5.0,
        on_progress=None,
        normalization_error="normalizer_failed",
    )

    assert segments == []
    assert stats["mode"] == "audit"
    assert stats["diagnostic"] == "merge_disabled_text_normalization_failed"
    assert stats["before"] == stats["after"]


def test_merge_downgrades_when_qwen_provider_is_unavailable(tmp_path: Path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_silent_wav(audio)
    transcriber = _pending_transcriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "merge")
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_PROVIDER", "qwen3")
    metadata = {
        "provider_id": "qwen3-independent",
        "provider_kind": "independent_asr",
        "model_id": "mlx-community/Qwen3-ASR-1.7B-8bit",
        "model_family": "qwen3_asr",
    }
    monkeypatch.setattr(
        transcriber,
        "_load_local_recovery_provider",
        lambda _name: (None, metadata, "qwen3_model_not_cached"),
    )
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, _audio, _options, *, sensevoice: [{"text": "新增内容", "language": "zh"}],
    )

    segments, stats = transcriber._run_sensevoice_local_recovery(
        object(),
        audio,
        TranscribeOptions(model_id="iic/SenseVoiceSmall"),
        [],
        duration=5.0,
        on_progress=None,
    )

    assert segments == []
    assert stats["mode"] == "audit"
    assert stats["diagnostic"] == "merge_downgraded:qwen3_model_not_cached"
    assert stats["provider"]["available"] is False
    assert transcriber._wallclock_failed_ranges == [(1.0, 2.0)]


def test_finalizer_rebuilds_snapshots_after_text_normalization(tmp_path: Path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_silent_wav(audio)
    transcriber = _pending_transcriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "merge")
    _enable_qwen_provider(transcriber, monkeypatch, "前文新增内容后文")
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, _audio, _options, *, sensevoice: [{"text": "前文新增内容后文", "language": "zh"}],
    )
    original = [Segment(0.0, 1.0, "前文"), Segment(2.0, 3.0, "后文")]
    _segments, stats = transcriber._run_sensevoice_local_recovery(
        object(),
        audio,
        TranscribeOptions(model_id="iic/SenseVoiceSmall"),
        list(original),
        duration=5.0,
        on_progress=None,
    )
    normalized = [
        Segment(0.0, 1.0, "前文。"),
        Segment(1.0, 2.0, "新增内容", original_text="新增内容"),
        Segment(2.0, 3.0, "后文。"),
    ]
    transcriber.last_filter_stats = {"speech_coverage": {"local_recovery": stats}}

    transcriber._finalize_transcription_segments(normalized)

    recovery = transcriber.last_filter_stats["speech_coverage"]["local_recovery"]
    assert recovery["before"]["segment_count"] == 2
    assert recovery["after"]["segment_count"] == 3
    assert recovery["before"]["text_sha256"] == hashlib.sha256("前文。后文。".encode()).hexdigest()
    assert recovery["after"]["text_sha256"] == hashlib.sha256("前文。新增内容后文。".encode()).hexdigest()


def test_audit_finalizer_allows_normalizer_text_and_segment_changes():
    transcriber = _pending_transcriber()
    original = [Segment(0.0, 1.0, "第一句"), Segment(2.0, 3.0, "第二句")]
    snapshot = _recovery_snapshot(
        original,
        transcriber._wallclock_attempted_ranges,
        transcriber._wallclock_recognized_ranges,
        transcriber._wallclock_failed_ranges,
    )
    recovery = {"mode": "audit", "before": snapshot, "after": snapshot}
    transcriber.last_filter_stats = {"speech_coverage": {"local_recovery": recovery}}
    normalized = [Segment(0.0, 3.0, "第一句，第二句。")]

    transcriber._finalize_transcription_segments(normalized)

    assert recovery["normalization_changed_segments"] is True
    assert recovery["partition_preserved_after_normalization"] is True
    assert recovery.get("diagnostic") != "audit_local_recovery_partition_changed_after_snapshot"


def test_audit_finalizer_records_partition_change_without_losing_transcript():
    transcriber = _pending_transcriber()
    segments = [Segment(0.0, 1.0, "正文。")]
    snapshot = _recovery_snapshot(
        segments,
        transcriber._wallclock_attempted_ranges,
        transcriber._wallclock_recognized_ranges,
        transcriber._wallclock_failed_ranges,
    )
    recovery = {"mode": "audit", "before": snapshot, "after": snapshot}
    transcriber.last_filter_stats = {"speech_coverage": {"local_recovery": recovery}}
    transcriber._wallclock_failed_ranges = []

    transcriber._finalize_transcription_segments(segments)

    assert recovery["partition_preserved_after_normalization"] is False
    assert recovery["diagnostic"] == "audit_local_recovery_partition_changed_after_snapshot"


def test_merge_conflict_remains_failed_and_uses_pad_two(tmp_path: Path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_silent_wav(audio)
    transcriber = _pending_transcriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "merge")

    def fake_generate(_model, chunk_path, _options, *, sensevoice):
        stem = Path(chunk_path).stem
        if stem.endswith("exact"):
            text = "甲乙"
        elif stem.endswith("pad0_5"):
            text = "丙丁"
        else:
            text = "戊己"
        return [{"text": text, "language": "zh"}]

    monkeypatch.setattr(transcriber, "_generate", fake_generate)
    original = [Segment(0.0, 1.0, "前文"), Segment(2.0, 3.0, "后文")]

    segments, stats = transcriber._run_sensevoice_local_recovery(
        object(), audio, TranscribeOptions(model_id="iic/SenseVoiceSmall"), list(original), duration=5.0, on_progress=None
    )

    assert segments == original
    assert stats["rejected"] == 1
    assert stats["attempts"] == 4
    assert transcriber._wallclock_recognized_ranges == []
    assert transcriber._wallclock_failed_ranges == [(1.0, 2.0)]


def test_invalid_mode_falls_back_to_off_with_diagnostic(tmp_path: Path, monkeypatch):
    audio = tmp_path / "audio.wav"
    _write_silent_wav(audio)
    transcriber = _pending_transcriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "unsafe")
    monkeypatch.setattr(transcriber, "_generate", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()))
    original = [Segment(0.0, 1.0, "原文")]

    segments, stats = transcriber._run_sensevoice_local_recovery(
        object(), audio, TranscribeOptions(model_id="iic/SenseVoiceSmall"), list(original), duration=5.0, on_progress=None
    )

    assert segments == original
    assert stats["mode"] == "off"
    assert stats["requested_mode"] == "unsafe"
    assert stats["diagnostic"] == "invalid_mode:unsafe;fallback_off"
    assert stats["attempts"] == 0


def test_wallclock_short_punctuation_candidate_is_not_a_segment(tmp_path: Path, monkeypatch):
    audio = tmp_path / "short.wav"
    _write_silent_wav(audio, seconds=1)
    transcriber = FunASRTranscriber(backend_name="sensevoice")
    monkeypatch.setattr(transcriber, "_speech_ranges", lambda _audio: [(0.0, 0.5)])
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, _audio, _options, *, sensevoice: [{"text": "<|Speech|>!!!", "language": "zh"}],
    )

    segments, _language, stats = transcriber._run_sensevoice_wallclock_vad(
        object(), audio, TranscribeOptions(model_id="iic/SenseVoiceSmall"), None
    )

    assert segments == []
    assert stats["wallclock_recognized_chunks"] == 0
    assert stats["wallclock_failed_chunks"] == 1
    assert transcriber._wallclock_failure_reasons[0]["reason"] == "low_text_density"


def test_adjacent_failed_windows_are_recovered_independently(tmp_path: Path, monkeypatch):
    audio = tmp_path / "long.wav"
    _write_silent_wav(audio, seconds=10)
    transcriber = FunASRTranscriber(backend_name="sensevoice")
    windows = [(0.0, 1.5), (1.5, 3.0), (3.0, 4.5), (4.5, 6.0), (6.0, 7.5), (7.5, 9.0), (9.0, 10.0)]
    transcriber._wallclock_attempted_ranges = list(windows)
    transcriber._wallclock_recognized_ranges = []
    transcriber._wallclock_failed_ranges = list(windows)
    transcriber._wallclock_failure_reasons = [
        {"start": start, "end": end, "reason": "empty_transcript"} for start, end in windows
    ]
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "merge")
    _enable_qwen_provider(transcriber, monkeypatch, "unused")

    def recovery_text(chunk_path: Path) -> str:
        group_index = int(chunk_path.stem.split("_")[1])
        return f"内容{group_index}"

    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, chunk_path, _options, *, sensevoice: [
            {"text": recovery_text(Path(chunk_path)), "language": "zh"}
        ],
    )
    monkeypatch.setattr(
        transcriber,
        "_run_local_recovery_provider",
        lambda _provider, chunk_path, _options: (recovery_text(Path(chunk_path)), {}),
    )
    original = [Segment(10.0, 11.0, "后文")]

    segments, stats = transcriber._run_sensevoice_local_recovery(
        object(), audio, TranscribeOptions(model_id="iic/SenseVoiceSmall"), list(original), duration=10.0, on_progress=None
    )

    assert len(segments) == 8
    assert stats["mode"] == "merge"
    assert stats["pending_groups"] == 7
    assert stats["inserted"] == 7
    assert stats["rejected"] == 0
    assert transcriber._wallclock_recognized_ranges == windows
    assert transcriber._wallclock_failed_ranges == []
    assert stats["before"] != stats["after"]
    assert stats["after"]["partition_valid"] is True


def test_neighbor_echo_does_not_mark_failed_window_as_existing(tmp_path: Path, monkeypatch):
    audio = tmp_path / "echo.wav"
    _write_silent_wav(audio)
    transcriber = _pending_transcriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "merge")
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, _audio, _options, *, sensevoice: [{"text": "左侧回声", "language": "zh"}],
    )
    original = [Segment(0.0, 1.0, "左侧回声"), Segment(2.0, 3.0, "后文")]

    segments, stats = transcriber._run_sensevoice_local_recovery(
        object(), audio, TranscribeOptions(model_id="iic/SenseVoiceSmall"), list(original), duration=5.0, on_progress=None
    )

    assert segments == original
    assert stats["matched_existing"] == 0
    assert stats["rejected"] == 1
    assert transcriber._wallclock_failed_ranges == [(1.0, 2.0)]


def test_identical_clamped_audio_slices_do_not_form_two_framing_consensus(tmp_path: Path, monkeypatch):
    audio = tmp_path / "whole.wav"
    _write_silent_wav(audio, seconds=5)
    transcriber = FunASRTranscriber(backend_name="sensevoice")
    transcriber._wallclock_attempted_ranges = [(0.0, 5.0)]
    transcriber._wallclock_recognized_ranges = []
    transcriber._wallclock_failed_ranges = [(0.0, 5.0)]
    transcriber._wallclock_failure_reasons = [{"start": 0.0, "end": 5.0, "reason": "empty_transcript"}]
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "merge")
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, _audio, _options, *, sensevoice: [{"text": "完整内容测试", "language": "zh"}],
    )

    segments, stats = transcriber._run_sensevoice_local_recovery(
        object(), audio, TranscribeOptions(model_id="iic/SenseVoiceSmall"), [], duration=5.0, on_progress=None
    )

    assert segments == []
    assert stats["rejected"] == 1
    assert stats["attempts"] == 4
    slice_hashes = {attempt["slice_sha256"] for attempt in stats["details"][0]["attempts"]}
    assert len(slice_hashes) == 1
    assert transcriber._wallclock_failed_ranges == [(0.0, 5.0)]


def test_off_mode_never_marks_details_truncated(tmp_path: Path, monkeypatch):
    transcriber = FunASRTranscriber(backend_name="sensevoice")
    windows = [(float(index * 2), float(index * 2 + 1)) for index in range(21)]
    transcriber._wallclock_attempted_ranges = list(windows)
    transcriber._wallclock_recognized_ranges = []
    transcriber._wallclock_failed_ranges = list(windows)
    transcriber._wallclock_failure_reasons = [
        {"start": start, "end": end, "reason": "empty_transcript"} for start, end in windows
    ]
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "off")

    _segments, stats = transcriber._run_sensevoice_local_recovery(
        object(), tmp_path / "unused.wav", TranscribeOptions(model_id="iic/SenseVoiceSmall"), [], duration=50.0, on_progress=None
    )

    assert stats["pending_groups"] == 21
    assert stats["details"] == []
    assert stats["details_truncated"] is False


def test_local_recovery_reads_only_requested_audio_slices(tmp_path: Path, monkeypatch):
    import soundfile as sf

    audio = tmp_path / "audio.wav"
    _write_silent_wav(audio)
    transcriber = _pending_transcriber()
    monkeypatch.setenv("LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE", "audit")
    monkeypatch.setattr(
        transcriber,
        "_generate",
        lambda _model, _audio, _options, *, sensevoice: [{"text": "新增内容", "language": "zh"}],
    )
    real_read = sf.read
    calls: list[dict] = []

    def tracked_read(*args, **kwargs):
        calls.append(dict(kwargs))
        return real_read(*args, **kwargs)

    monkeypatch.setattr(sf, "read", tracked_read)
    transcriber._run_sensevoice_local_recovery(
        object(),
        audio,
        TranscribeOptions(model_id="iic/SenseVoiceSmall"),
        [Segment(0.0, 1.0, "前文"), Segment(2.0, 3.0, "后文")],
        duration=5.0,
        on_progress=None,
    )

    assert calls
    assert all(call.get("start") is not None and call.get("stop") is not None for call in calls)
    assert all(call.get("always_2d") is True for call in calls)


def test_speech_ranges_streams_audio_in_bounded_reads(tmp_path: Path, monkeypatch):
    import sys
    import types
    import soundfile as sf
    import scribe_py.core.transcriber_funasr as funasr_module

    audio = tmp_path / "vad.wav"
    _write_silent_wav(audio, seconds=3)
    real_read = sf.read
    calls: list[dict] = []

    def tracked_read(*args, **kwargs):
        calls.append(dict(kwargs))
        return real_read(*args, **kwargs)

    fake_vad = types.SimpleNamespace(
        load_silero_vad=lambda: object(),
        get_speech_timestamps=lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(sf, "read", tracked_read)
    monkeypatch.setitem(sys.modules, "silero_vad", fake_vad)
    monkeypatch.setattr(funasr_module, "_SILERO_VAD_MODEL", None)

    transcriber = FunASRTranscriber(backend_name="sensevoice")
    ranges = transcriber._speech_ranges(audio)

    assert ranges == []
    assert calls
    assert all(call.get("start") is not None and call.get("stop") is not None for call in calls)
    assert all(call.get("always_2d") is True for call in calls)


def test_asymmetric_context_accepts_only_matching_existing_core_evidence():
    from scribe_py.core.sensevoice_recovery import decide_asymmetric_context_evidence

    result = decide_asymmetric_context_evidence(
        core_start=1.0, core_end=2.0,
        left_start=0.5, left_end=2.0,
        right_start=1.0, right_end=2.5,
        left_text="目标正文", right_text="目标正文",
        left_slice_sha256="left", right_slice_sha256="right",
        local_reference="前文目标正文后文", speech_duration_s=1.0,
    )

    assert result["decision"] == "matched_existing"
    assert result["consensus"] == "目标正文"
    assert result["evidence_sha256"]


def test_asymmetric_context_rejects_different_neighbor_echoes():
    from scribe_py.core.sensevoice_recovery import decide_asymmetric_context_evidence

    result = decide_asymmetric_context_evidence(
        core_start=1.0, core_end=2.0,
        left_start=0.5, left_end=2.0,
        right_start=1.0, right_end=2.5,
        left_text="前一句", right_text="后一句",
        left_slice_sha256="left", right_slice_sha256="right",
        local_reference="前一句目标后一句", speech_duration_s=1.0,
    )

    assert result["decision"] == "rejected"
    assert result["rejection_reason"] == "asymmetric_text_disagreement"


def test_asymmetric_context_rejects_same_clamped_slice():
    from scribe_py.core.sensevoice_recovery import decide_asymmetric_context_evidence

    result = decide_asymmetric_context_evidence(
        core_start=0.0, core_end=1.0,
        left_start=0.0, left_end=1.0,
        right_start=0.0, right_end=1.0,
        left_text="目标", right_text="目标",
        left_slice_sha256="same", right_slice_sha256="same",
        local_reference="目标", speech_duration_s=1.0,
    )

    assert result["decision"] == "rejected"
    assert result["rejection_reason"] in {"framing_intersection_differs_from_core", "non_independent_slices"}


def test_anchor_character_ownership_accepts_unique_exact_run():
    from scribe_py.core.sensevoice_recovery import build_anchor_character_ownership
    result=build_anchor_character_ownership(final_text='甲乙丙丁戊己庚辛壬癸',anchor_chunks=[{'start':0.0,'end':2.0,'text':'甲乙丙丁戊己庚辛壬癸','status':'recognized'}],strict_windows=[{'core_start':0.2,'core_end':1.8,'speech_duration_s':1.6}],boundary_guard_s=0.0)
    assert result['equal_char_ratio']==1.0
    assert len(result['claims'])==1
    assert result['claims'][0]['owned_chars']>=2


def test_anchor_character_ownership_rejects_repeated_non_unique_phrase():
    from scribe_py.core.sensevoice_recovery import build_anchor_character_ownership
    text='我们就是这样我们就是这样'
    result=build_anchor_character_ownership(final_text=text,anchor_chunks=[{'start':0.0,'end':2.0,'text':text,'status':'recognized'}],strict_windows=[{'core_start':0.2,'core_end':1.8,'speech_duration_s':1.6}],boundary_guard_s=0.0,max_unique_context_chars=6)
    assert result['claims']==[]


def test_anchor_character_ownership_ignores_failed_anchor_chunk():
    from scribe_py.core.sensevoice_recovery import build_anchor_character_ownership
    result=build_anchor_character_ownership(final_text='甲乙丙丁戊己庚辛',anchor_chunks=[{'start':0.0,'end':2.0,'text':'甲乙丙丁戊己庚辛','status':'failed'}],strict_windows=[{'core_start':0.0,'core_end':2.0,'speech_duration_s':2.0}],boundary_guard_s=0.0)
    assert result['anchor_normalized_chars']==0
    assert result['claims']==[]


def test_parse_paraformer_native_timestamps_keeps_integer_ms_chars():
    from scribe_py.core.sensevoice_recovery import parse_paraformer_native_timestamps
    result=parse_paraformer_native_timestamps([{'text':'你 好','timestamp':[[110,270],[270,350]]}],duration_s=1.0)
    assert [(u['normalized'],u['start_ms'],u['end_ms']) for u in result['units']]==[('你',110,270),('好',270,350)]
    assert result['coarse_fallback_count']==0


def test_parse_paraformer_native_timestamps_rejects_bad_shapes():
    from scribe_py.core.sensevoice_recovery import parse_paraformer_native_timestamps
    result=parse_paraformer_native_timestamps([{'text':'你 好','timestamp':[[0,100]]},{'text':'你','timestamp':[[100,100]]}],duration_s=1.0)
    assert result['units']==[]
    assert [x['reason'] for x in result['rejected_items']]==['unit_timestamp_count_mismatch','invalid_or_non_monotonic_timestamp']


def test_paraformer_native_ownership_uses_only_contained_exact_chars():
    from scribe_py.core.sensevoice_recovery import build_paraformer_native_ownership
    chars='甲乙丙丁戊己庚辛'
    units=[{'native_id':f'u{i}','normalized':c,'native_character':True,'start_ms':i*200,'end_ms':i*200+150} for i,c in enumerate(chars)]
    result=build_paraformer_native_ownership(final_text=chars,native_units=units,strict_windows=[{'core_start':0.2,'core_end':1.4,'speech_duration_s':1.2}])
    assert len(result['claims'])==1
    assert result['claims'][0]['padding_credit_chars']==0
    assert all(i not in result['claims'][0]['native_ids'] for i in ['u0','u7'])


def test_parse_paraformer_native_timestamps_allows_small_interval_overlap_with_monotonic_starts():
    from scribe_py.core.sensevoice_recovery import parse_paraformer_native_timestamps
    result=parse_paraformer_native_timestamps([{'text':'你 好','timestamp':[[100,220],[200,300]]}],duration_s=1.0)
    assert len(result['units'])==2
    assert result['rejected_items']==[]
