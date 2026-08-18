from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "diarization_metrics.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("diarization_metrics", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _segment(mod, start: float, end: float, speaker: str, uri: str = "demo"):
    return mod.Segment(uri=uri, start=start, end=end, speaker=speaker)


def test_permuted_speaker_names_have_zero_der_and_jer():
    mod = _load_script()
    reference = [
        _segment(mod, 0.0, 5.0, "张三"),
        _segment(mod, 5.0, 10.0, "李四"),
    ]
    prediction = [
        _segment(mod, 0.0, 5.0, "SPEAKER_B"),
        _segment(mod, 5.0, 10.0, "SPEAKER_A"),
    ]

    result = mod.evaluate_recording("demo", reference, prediction)

    assert result.der == pytest.approx(0.0)
    assert result.jer == pytest.approx(0.0)
    assert result.speaker_mapping == {"SPEAKER_A": "李四", "SPEAKER_B": "张三"}
    assert result.speaker_count_error == 0


def test_assignment_is_global_optimum_instead_of_greedy():
    mod = _load_script()
    rows = ["X", "Y"]
    columns = ["A", "B"]
    weights = {
        ("X", "A"): 9.0,
        ("X", "B"): 8.0,
        ("Y", "A"): 7.0,
        ("Y", "B"): 0.0,
    }

    mapping = mod._maximum_weight_assignment(rows, columns, weights)

    assert mapping == {"X": "B", "Y": "A"}


def test_der_breakdown_reports_miss_false_alarm_and_confusion():
    mod = _load_script()
    reference = [_segment(mod, 0.0, 10.0, "A")]
    prediction = [
        _segment(mod, 0.0, 7.0, "X"),
        _segment(mod, 8.0, 10.0, "Y"),
        _segment(mod, 10.0, 12.0, "Y"),
    ]

    result = mod.evaluate_recording("demo", reference, prediction)

    assert result.miss_s == pytest.approx(1.0)
    assert result.false_alarm_s == pytest.approx(2.0)
    assert result.confusion_s == pytest.approx(2.0)
    assert result.der == pytest.approx(0.5)
    assert result.speaker_count_error == 1
    assert result.speaker_count_absolute_error == 1


def test_overlap_is_scored_as_reference_speaker_time():
    mod = _load_script()
    reference = [
        _segment(mod, 0.0, 10.0, "A"),
        _segment(mod, 5.0, 10.0, "B"),
    ]
    prediction = [_segment(mod, 0.0, 10.0, "X")]

    result = mod.evaluate_recording("demo", reference, prediction)

    assert result.reference_speaker_time_s == pytest.approx(15.0)
    assert result.miss_s == pytest.approx(5.0)
    assert result.false_alarm_s == pytest.approx(0.0)
    assert result.confusion_s == pytest.approx(0.0)
    assert result.der == pytest.approx(1.0 / 3.0)


def test_jer_uses_optimal_jaccard_mapping():
    mod = _load_script()
    reference = [_segment(mod, 0.0, 10.0, "A")]
    prediction = [_segment(mod, 0.0, 7.0, "X")]

    result = mod.evaluate_recording("demo", reference, prediction)

    assert result.jer == pytest.approx(0.3)
    assert result.per_reference_speaker_jer == {"A": pytest.approx(0.3)}
    assert result.jer_speaker_mapping == {"X": "A"}


def test_load_rttm_and_json_support_overlapping_speakers(tmp_path: Path):
    mod = _load_script()
    gold = tmp_path / "gold.rttm"
    gold.write_text(
        "SPEAKER meeting 1 0.000 5.000 <NA> <NA> alice <NA> <NA>\n"
        "SPEAKER meeting 1 3.000 2.000 <NA> <NA> bob <NA> <NA>\n",
        encoding="utf-8",
    )
    prediction = tmp_path / "prediction.json"
    prediction.write_text(json.dumps({
        "case": "meeting",
        "segments": [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_A"},
            {"start": 3.0, "duration": 2.0, "speakers": ["SPEAKER_B"]},
        ],
    }), encoding="utf-8")

    result = mod.evaluate_files(gold, prediction)

    assert result.aggregate()["der"] == pytest.approx(0.0)
    assert result.aggregate()["jer"] == pytest.approx(0.0)
    assert len(mod.load_rttm(gold)["meeting"]) == 2
    assert len(mod.load_json(prediction)["meeting"]) == 2


def test_single_recording_files_align_even_when_uri_names_differ(tmp_path: Path):
    mod = _load_script()
    gold = tmp_path / "human.json"
    prediction = tmp_path / "engine.json"
    gold.write_text(json.dumps([
        {"start": 0.0, "end": 2.0, "speaker": "A"},
    ]), encoding="utf-8")
    prediction.write_text(json.dumps([
        {"start": 0.0, "end": 2.0, "speaker": "X"},
    ]), encoding="utf-8")

    result = mod.evaluate_files(gold, prediction)

    assert len(result.recordings) == 1
    assert result.aggregate()["der"] == pytest.approx(0.0)


def test_cli_writes_chinese_json_and_markdown_reports(tmp_path: Path):
    gold = tmp_path / "gold.json"
    prediction = tmp_path / "prediction.json"
    gold.write_text(json.dumps({
        "uri": "demo",
        "segments": [
            {"start": 0.0, "end": 4.0, "speaker": "张三"},
            {"start": 4.0, "end": 8.0, "speaker": "李四"},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    prediction.write_text(json.dumps({
        "uri": "demo",
        "segments": [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_B"},
            {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_A"},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    out_dir = tmp_path / "report"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--gold",
            str(gold),
            "--prediction",
            str(prediction),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = json.loads(proc.stdout)
    report = json.loads(Path(stdout["json"]).read_text(encoding="utf-8"))
    markdown = Path(stdout["markdown"]).read_text(encoding="utf-8")
    assert stdout["ok"] is True
    assert stdout["DER"] == pytest.approx(0.0)
    assert report["状态"] == "成功"
    assert report["汇总"]["DER百分比"] == "0.00%"
    assert report["逐录音"][0]["DER最优说话人映射_预测到真值"] == {
        "SPEAKER_A": "李四",
        "SPEAKER_B": "张三",
    }
    assert "说话人分离真实评测报告" in markdown
    assert "DER 分解" in markdown
    assert "SPEAKER_A→李四" in markdown


def test_invalid_json_time_range_is_rejected(tmp_path: Path):
    mod = _load_script()
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({
        "segments": [{"start": 2.0, "end": 1.0, "speaker": "A"}],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="时间范围无效"):
        mod.load_json(path)
