from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "diarization_sparse_score.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("diarization_sparse_score", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _score(tmp_path: Path, predicted_labels: list[str], expected_labels: list[str]) -> dict[str, Any]:
    prediction_path = tmp_path / "prediction.json"
    annotation_path = tmp_path / "annotations.json"
    segments = []
    annotations = []
    for index, (predicted, expected) in enumerate(zip(predicted_labels, expected_labels, strict=True)):
        text = f"第{index}条"
        segments.append({
            "start": float(index),
            "end": float(index + 1),
            "speaker": predicted,
            "text": text,
        })
        annotations.append({
            "序号": index,
            "时间": f"00:{index:02d}.00 - 00:{index + 1:02d}.00",
            "你的标注": expected,
            "文本": text,
        })
    prediction_path.write_text(json.dumps({"segments": segments}, ensure_ascii=False), encoding="utf-8")
    annotation_path.write_text(json.dumps(annotations, ensure_ascii=False), encoding="utf-8")
    return _load_script().score(prediction_path, annotation_path)


def test_global_label_permutation_scores_by_maximum_agreement(tmp_path: Path):
    result = _score(
        tmp_path,
        ["SPEAKER_C", "SPEAKER_A", "SPEAKER_B", "SPEAKER_C", "SPEAKER_A", "SPEAKER_B"],
        ["A", "B", "C", "A", "B", "C"],
    )

    assert result["mapping"] == {"A": "B", "B": "C", "C": "A"}
    assert result["raw_accuracy"] == 0.0
    assert result["mapped_accuracy"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["mapping_status"] == "determined"
    assert all(row["raw_predicted"] != row["mapped_predicted"] for row in result["rows"])
    assert all(row["mapped_correct"] for row in result["rows"])


def test_mixed_matching_and_permuted_labels_use_one_global_mapping(tmp_path: Path):
    result = _score(
        tmp_path,
        ["speaker_a", "SPEAKER_A", "SPEAKER_B", "SPEAKER_B", "SPEAKER_C"],
        ["A", "A", "C", "C", "B"],
    )

    assert result["mapping"] == {"A": "A", "B": "C", "C": "B"}
    assert result["raw_correct"] == 2
    assert result["mapped_correct"] == 5
    assert result["raw_accuracy"] == 0.4
    assert result["mapped_accuracy"] == 1.0
    assert [row["raw_predicted"] for row in result["rows"]] == ["A", "A", "B", "B", "C"]


def test_ambiguous_expected_sets_keep_mapping_one_to_one(tmp_path: Path):
    result = _score(
        tmp_path,
        ["SPEAKER_X", "SPEAKER_Y", "SPEAKER_X", "SPEAKER_Y"],
        ["B/D", "B/D", "B/D", "B/D"],
    )

    assert result["mapping"] == {"X": "B", "Y": "D"}
    assert len({label for label in result["mapping"].values() if label is not None}) == 2
    assert result["mapped_accuracy"] == 1.0
    assert result["mapping_status"] == "underdetermined_multiple_optima"
    assert result["mapping_is_determined"] is False
    assert result["mapping_optimal_solution_count"] == 2
    assert all(row["expected"] == ["B", "D"] for row in result["rows"])


def test_single_expected_label_reports_underdetermined_calibration(tmp_path: Path):
    result = _score(
        tmp_path,
        ["SPEAKER_A", "SPEAKER_A", "SPEAKER_B"],
        ["C", "C", "C"],
    )

    assert result["mapping"] == {"A": "C", "B": None}
    assert result["mapped_accuracy"] == 0.6667
    assert result["mapping_status"] == "underdetermined_single_expected_label"
    assert result["mapping_is_determined"] is False
    assert "multi-speaker separation" in result["mapping_note"]


def test_no_review_rows_returns_empty_mapping_and_null_accuracies(tmp_path: Path):
    result = _score(tmp_path, [], [])

    assert result["total"] == 0
    assert result["mapping"] == {}
    assert result["raw_accuracy"] is None
    assert result["mapped_accuracy"] is None
    assert result["mapping_method"] == "maximum_agreement_one_to_one"
    assert result["mapping_calibration_rows"] == 0
    assert result["mapping_optimal_solution_count"] == 0
    assert result["mapping_status"] == "no_reviewed_rows"
    assert result["mapping_is_determined"] is False
    assert result["rows"] == []


def test_sparse_score_uses_speaker_cue_at_reviewed_time(tmp_path: Path):
    prediction_path = tmp_path / "prediction.json"
    annotation_path = tmp_path / "annotations.json"
    prediction_path.write_text(json.dumps({"segments": [{
        "start": 0.0,
        "end": 6.0,
        "speaker": "SPEAKER_B",
        "text": "前半句提问，后半句回答。",
        "speaker_cues": [
            {"cue_index": 0, "start": 0.0, "end": 2.0, "speaker": "SPEAKER_D"},
            {"cue_index": 1, "start": 2.0, "end": 6.0, "speaker": "SPEAKER_B"},
        ],
    }]}, ensure_ascii=False), encoding="utf-8")
    annotation_path.write_text(json.dumps([{
        "序号": 1,
        "时间": "00:00.20 - 00:01.80",
        "你的标注": "D",
    }], ensure_ascii=False), encoding="utf-8")

    result = _load_script().score(prediction_path, annotation_path)

    assert result["raw_correct"] == 1
    assert result["rows"][0]["raw_predicted"] == "D"
    assert result["rows"][0]["prediction_resolution"] == "speaker_cue"
