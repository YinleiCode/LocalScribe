import json
from pathlib import Path


FIXTURE = Path(__file__).resolve().parents[2] / "experiments" / "asr_recording10_human_truth.json"


def test_recording10_human_truth_fixture_is_complete_and_consistent():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = data["cases"]
    summary = data["summary"]

    assert data["version"] == 1
    assert len(data["audio_sha256"]) == 64
    assert len(data["baseline_text_sha256"]) == 64
    assert [case["id"] for case in cases] == list(range(1, 8))
    assert all(case["start"] < case["end"] for case in cases)
    assert all(case["human_text"] for case in cases)
    assert sum(case["verdict"] == "true_omission" for case in cases) == summary["true_omission_windows"]
    assert sum(case["verdict"].startswith("substitution") for case in cases) == summary["substitution_windows"]
    assert sum(case["verdict"].startswith("already_present") for case in cases) == summary["coverage_false_positive_windows"]


def test_recording10_corrections_are_one_way_and_bound_to_reviewed_cases():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    case_ids = {case["id"] for case in data["cases"]}
    corrections = data["corrections"]

    assert {item["case_id"] for item in corrections} == {1, 2, 6, 7}
    assert all(item["case_id"] in case_ids for item in corrections)
    assert all(item["old"] and item["new"] and item["old"] != item["new"] for item in corrections)
