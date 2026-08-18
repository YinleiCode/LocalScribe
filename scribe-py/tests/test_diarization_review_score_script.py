from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "diarization_review_score.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("diarization_review_score", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _manifest():
    return {
        "pack_id": "pack",
        "items": [
            {"id": "DIA-001", "recording": "一", "category": "稳定对照", "current_prediction": "SPEAKER_A"},
            {"id": "DIA-002", "recording": "一", "category": "段内换人", "current_prediction": "A->B"},
            {"id": "DIA-003", "recording": "二", "category": "说话人切换", "current_prediction": "B-A"},
        ],
    }


def test_score_uses_current_prediction_as_gold_for_confirmed_rows():
    mod = _load_script()
    annotations = {
        "pack_id": "pack",
        "items": [
            {"id": "DIA-001", "verdict": "correct", "correct_speaker_sequence": "", "notes": ""},
            {"id": "DIA-002", "verdict": "wrong_speaker", "correct_speaker_sequence": "B_A", "notes": ""},
            {"id": "DIA-003", "verdict": "uncertain", "correct_speaker_sequence": "", "notes": ""},
        ],
    }
    rows = mod.build_rows(_manifest(), annotations)
    metrics = mod.score_rows(rows)
    assert rows[0]["gold_sequence"] == "A"
    assert rows[1]["gold_sequence"] == "B->A"
    assert metrics["scorable"] == 2
    assert metrics["correct"] == 1
    assert metrics["accuracy"] == 0.5


def test_override_is_explicit_and_preserves_original_note():
    mod = _load_script()
    annotations = {
        "pack_id": "pack",
        "items": [
            {"id": "DIA-001", "verdict": "", "correct_speaker_sequence": "", "notes": "实际是B"},
            {"id": "DIA-002", "verdict": "correct", "correct_speaker_sequence": "", "notes": ""},
            {"id": "DIA-003", "verdict": "correct", "correct_speaker_sequence": "", "notes": ""},
        ],
    }
    rows = mod.build_rows(
        _manifest(),
        annotations,
        overrides={"DIA-001": {"verdict": "false_split", "correct_speaker_sequence": "B"}},
    )
    assert rows[0]["override_applied"] is True
    assert rows[0]["notes"] == "实际是B"
    assert rows[0]["gold_sequence"] == "B"


def test_mismatched_pack_is_rejected():
    mod = _load_script()
    annotations = {"pack_id": "other", "items": []}
    try:
        mod.build_rows(_manifest(), annotations)
    except ValueError as exc:
        assert "pack_id" in str(exc)
    else:
        raise AssertionError("expected pack mismatch to fail")


def test_candidate_prediction_does_not_redefine_confirmed_gold():
    mod = _load_script()
    annotations = {
        "pack_id": "pack",
        "items": [
            {"id": "DIA-001", "verdict": "correct", "correct_speaker_sequence": "", "notes": ""},
            {"id": "DIA-002", "verdict": "correct", "correct_speaker_sequence": "", "notes": ""},
            {"id": "DIA-003", "verdict": "uncertain", "correct_speaker_sequence": "", "notes": ""},
        ],
    }

    rows = mod.build_rows(
        _manifest(),
        annotations,
        predictions={"DIA-001": "B", "DIA-002": "A->B"},
    )
    metrics = mod.score_rows(rows)

    assert rows[0]["baseline_prediction"] == "A"
    assert rows[0]["prediction"] == "B"
    assert rows[0]["gold_sequence"] == "A"
    assert rows[0]["prediction_correct"] is False
    assert rows[1]["prediction_correct"] is True
    assert metrics["correct"] == 1
    assert metrics["strict_scorable"] == 2
    assert metrics["strict_correct"] == 1
    assert metrics["strict_accuracy"] == 0.5


def test_strict_score_counts_errors_without_full_speaker_sequence():
    mod = _load_script()
    annotations = {
        "pack_id": "pack",
        "items": [
            {"id": "DIA-001", "verdict": "correct", "correct_speaker_sequence": "", "notes": ""},
            {"id": "DIA-002", "verdict": "missed_split", "correct_speaker_sequence": "", "notes": "实际两人"},
            {"id": "DIA-003", "verdict": "wrong_speaker", "correct_speaker_sequence": "", "notes": "人员错"},
        ],
    }

    rows = mod.build_rows(_manifest(), annotations)
    metrics = mod.score_rows(rows)

    assert metrics["strict_scorable"] == 3
    assert metrics["strict_correct"] == 1
    assert metrics["strict_errors"] == 2
    assert metrics["strict_accuracy"] == 0.3333
    assert metrics["scorable"] == 1
    assert {row["id"] for row in metrics["strict_error_items"]} == {"DIA-002", "DIA-003"}


def test_concatenated_labels_are_scored_against_per_cue_timeline():
    mod = _load_script()
    manifest = {
        "pack_id": "pack",
        "items": [{
            "id": "DIA-001",
            "recording": "一",
            "category": "段内换人",
            "current_prediction": "D->C",
            "timeline": [
                {"speaker": "B", "context": True},
                {"speaker": "D", "context": False},
                {"speaker": "C", "context": False},
                {"speaker": "C", "context": False},
                {"speaker": "C", "context": False},
            ],
        }],
    }
    annotations = {
        "pack_id": "pack",
        "items": [{
            "id": "DIA-001",
            "verdict": "wrong_speaker",
            "correct_speaker_sequence": "ddcc",
            "notes": "",
        }],
    }

    rows = mod.build_rows(manifest, annotations)

    assert mod.normalize_sequence("BAAA") == "B->A->A->A"
    assert rows[0]["baseline_prediction"] == "D->C"
    assert rows[0]["baseline_detailed_prediction"] == "D->C->C->C"
    assert rows[0]["prediction"] == "D->C->C->C"
    assert rows[0]["gold_sequence"] == "D->D->C->C"
    assert rows[0]["prediction_correct"] is False
