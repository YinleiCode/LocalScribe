from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "diarization_continuous_review.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("diarization_continuous_review", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prediction_rows_uses_cue_labels_without_changing_text_geometry():
    mod = _load_script()
    data = {"segments": [{
        "start": 0,
        "end": 4,
        "speaker": "SPEAKER_A",
        "text": "甲乙",
        "sync_cues": [
            {"start": 0, "end": 2, "text": "甲"},
            {"start": 2, "end": 4, "text": "乙"},
        ],
        "speaker_cues": [
            {"cue_index": 0, "start": 0, "end": 2, "speaker": "SPEAKER_A"},
            {"cue_index": 1, "start": 2, "end": 4, "speaker": "SPEAKER_B"},
        ],
    }]}

    rows = mod.prediction_rows(data)

    assert [(row["start"], row["end"], row["speaker"], row["text"]) for row in rows] == [
        (0.0, 2.0, "SPEAKER_A", "甲"),
        (2.0, 4.0, "SPEAKER_B", "乙"),
    ]


def test_choose_window_prefers_sustained_speaker_diversity():
    mod = _load_script()
    rows = [
        {"start": 0, "end": 10, "speaker": "A", "text": "一"},
        {"start": 10, "end": 20, "speaker": "A", "text": "二"},
        {"start": 60, "end": 70, "speaker": "A", "text": "三"},
        {"start": 70, "end": 80, "speaker": "B", "text": "四"},
        {"start": 80, "end": 90, "speaker": "C", "text": "五"},
    ]

    start, end, diagnostics = mod.choose_window(
        {"duration": 100}, rows, window_seconds=40, step_seconds=20
    )

    assert (start, end) == (60.0, 100.0)
    assert diagnostics["predicted_speaker_count_used_only_for_selection"] == 3


def test_build_pack_is_blind_and_keeps_source_read_only(tmp_path: Path):
    mod = _load_script()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF" + b"0" * 100)
    transcript = tmp_path / "meeting.json"
    transcript.write_text(json.dumps({
        "audio": str(audio),
        "duration": 20,
        "segments": [
            {"start": 0, "end": 10, "speaker": "SPEAKER_A", "text": "第一段"},
            {"start": 10, "end": 20, "speaker": "SPEAKER_B", "text": "第二段"},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    before = transcript.read_bytes()

    manifest_path, html_path, prediction_path = mod.build_pack(
        [("盲测会议", transcript)], tmp_path / "pack", window_seconds=20, dry_run=True
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    page = html_path.read_text(encoding="utf-8")
    assert transcript.read_bytes() == before
    assert manifest["system_speaker_labels_exposed"] is False
    assert "speaker" not in json.dumps(manifest["items"], ensure_ascii=False).lower()
    assert "SPEAKER_A" not in page and "SPEAKER_B" not in page
    assert {row["speaker"] for row in prediction["segments"]} == {"SPEAKER_A", "SPEAKER_B"}
    assert "导出连续真值" in page
    assert 'class="overlap"' in page
    assert "document.addEventListener('keydown'" in page
    assert "markAt(item,audio,[letter])" in page
