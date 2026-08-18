from __future__ import annotations

import numpy as np

from scribe_py.diarizers import overlap_detector


def _frame_index(seconds: float) -> int:
    return int(round(seconds / overlap_detector.FRAME_STEP))


def test_powerset_overlap_probability_uses_two_speaker_classes():
    probabilities = np.asarray([
        [0.02, 0.10, 0.10, 0.08, 0.30, 0.25, 0.15],
        [0.05, 0.70, 0.10, 0.05, 0.04, 0.03, 0.03],
    ], dtype=np.float32)

    scores = overlap_detector._powerset_overlap_probabilities(probabilities)

    assert np.allclose(scores, [0.70, 0.10], atol=1e-6)


def test_powerset_overlap_probability_accepts_log_softmax_output():
    probabilities = np.asarray([[0.1, 0.2, 0.2, 0.1, 0.15, 0.15, 0.1]], dtype=np.float64)

    scores = overlap_detector._powerset_overlap_probabilities(np.log(probabilities))

    assert np.allclose(scores, [0.4], atol=1e-6)


def test_scores_to_intervals_applies_hysteresis_and_merges_short_gap():
    scores = np.zeros(_frame_index(2.0), dtype=np.float32)
    scores[_frame_index(0.20):_frame_index(0.55)] = 0.8
    scores[_frame_index(0.55):_frame_index(0.62)] = 0.2
    scores[_frame_index(0.62):_frame_index(1.00)] = 0.75

    intervals = overlap_detector.scores_to_overlap_intervals(
        scores,
        onset=0.55,
        offset=0.45,
        min_duration=0.1,
        min_gap=0.1,
        total_duration=2.0,
    )

    assert len(intervals) == 1
    assert 0.15 <= intervals[0]["start"] <= 0.25
    assert 0.95 <= intervals[0]["end"] <= 1.05
    assert intervals[0]["max_confidence"] == 0.8


def test_map_overlap_to_segments_uses_interval_union_and_preserves_input():
    source = [
        {"start": 0.0, "end": 2.0, "text": "one", "speaker": "SPEAKER_A"},
        {"start": 2.0, "end": 3.0, "text": "two", "speaker_overlap_risk": True},
    ]
    intervals = [
        {"start": 0.5, "end": 1.0},
        {"start": 0.8, "end": 1.3},
    ]

    mapped = overlap_detector.map_overlap_to_segments(source, intervals)

    assert "overlap_ratio" not in source[0]
    assert mapped[0]["overlap_ratio"] == 0.4
    assert mapped[0]["speaker_overlap_risk"] is True
    assert mapped[0]["text"] == "one"
    assert mapped[0]["start"] == 0.0
    assert mapped[0]["end"] == 2.0
    assert mapped[1]["overlap_ratio"] == 0.0
    assert mapped[1]["speaker_overlap_risk"] is True


def test_filter_contaminated_windows_supports_tuples_and_dicts():
    tuple_windows = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    intervals = [{"start": 1.4, "end": 1.7}]

    clean, rejected = overlap_detector.partition_contaminated_windows(
        tuple_windows,
        intervals,
        max_overlap_ratio=0.02,
        padding=0.0,
    )

    assert clean == [(0.0, 1.0), (2.0, 3.0)]
    assert rejected == [(1.0, 2.0)]
    dict_windows = [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}]
    assert overlap_detector.filter_contaminated_windows(
        dict_windows,
        intervals,
        max_overlap_ratio=0.4,
        padding=0.0,
    ) == dict_windows


def test_detect_overlaps_uses_injected_coreml_model():
    class FakeCoreMLModel:
        def predict(self, inputs):
            assert inputs["audio"].shape == (1, 1, 160000)
            probabilities = np.zeros((1, 589, 7), dtype=np.float32)
            probabilities[..., 1] = 1.0
            probabilities[:, 100:120, 1] = 0.1
            probabilities[:, 100:120, 4] = 0.9
            return {"segments": probabilities}

    result = overlap_detector.detect_overlaps(
        np.zeros(160000, dtype=np.float32),
        model_path=__file__,
        coreml_model=FakeCoreMLModel(),
        min_duration=0.1,
    )

    assert result["available"] is True
    assert result["backend"] == "senko_segmentation_coreml"
    assert len(result["overlap_intervals"]) == 1
    assert result["stats"]["frame_count"] > 500


def test_detect_overlaps_safely_degrades_when_model_is_missing(tmp_path):
    result = overlap_detector.detect_overlaps(
        np.zeros(1600, dtype=np.float32),
        model_path=tmp_path / "missing.mlmodelc",
    )

    assert result["available"] is False
    assert result["backend"] == "none"
    assert result["overlap_intervals"] == []
    assert result["frame_confidence"]["scores"] == []
    assert "FileNotFoundError" in result["error"]


def test_empty_audio_is_available_and_does_not_load_model():
    result = overlap_detector.detect_overlaps(np.empty((0,), dtype=np.float32))

    assert result["available"] is True
    assert result["backend"] == "empty_audio"
    assert result["overlap_intervals"] == []

