from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "asr_full_pipeline_gold_score.py"
SPEC = importlib.util.spec_from_file_location("asr_full_pipeline_gold_score", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_best_local_alignment_ignores_neighboring_context():
    result = mod.best_local_alignment("婚姻问题慎重考虑了吗", "前文无关婚姻问题慎重考虑了吗后文无关")

    assert result["edit_distance"] == 0
    assert result["matched"] == "婚姻问题慎重考虑了吗"


def test_extract_timeline_text_prefers_sync_cues():
    result = {
        "segments": [
            {
                "start": 0.0,
                "end": 10.0,
                "text": "整段文字不应重复",
                "sync_cues": [
                    {"start": 0.0, "end": 4.0, "text": "前文"},
                    {"start": 4.0, "end": 8.0, "text": "目标文字"},
                    {"start": 8.0, "end": 10.0, "text": "后文"},
                ],
            }
        ]
    }

    text, stats = mod.extract_timeline_text(result, start=4.0, end=8.0, pad_seconds=0.0)

    assert text == "目标文字"
    assert stats["cue_parts"] == 1
    assert stats["segment_parts"] == 0


def test_score_full_results_matches_audio_and_reports_real_error(tmp_path: Path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio")
    result_path = tmp_path / "full.json"
    result = {
        "audio": str(audio),
        "segments": [
            {"start": 9.0, "end": 14.0, "text": "婚姻问题真么考虑了吗"},
        ],
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    gold = {
        "items": [
            {
                "id": "GOLD-001",
                "case_id": "court",
                "audio": str(audio),
                "start": 10.0,
                "duration": 3.0,
                "decision": "corrected",
                "correct_text": "婚姻问题慎重考虑了吗",
            }
        ]
    }

    report = mod.score_full_results(gold, mod.result_index([result_path]), pad_seconds=1.0)

    assert report["scored_rows"] == 1
    assert report["missing_full_results"] == 0
    assert report["total_edit_distance"] == 2
    assert report["rows"][0]["matched_text_normalized"] == "婚姻问题真么考虑了吗"


def test_missing_full_result_is_explicit():
    report = mod.score_full_results(
        {
            "items": [
                {
                    "id": "GOLD-001",
                    "audio": "/tmp/missing.wav",
                    "start": 0.0,
                    "duration": 1.0,
                    "decision": "confirmed",
                    "correct_text": "测试",
                }
            ]
        },
        {},
    )

    assert report["scored_rows"] == 0
    assert report["missing_full_results"] == 1
    assert report["rows"][0]["status"] == "missing_full_result"
