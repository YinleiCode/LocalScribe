from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "diarization_continuous_score.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("diarization_continuous_score", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def test_score_pack_reports_overlap_error_and_count(tmp_path: Path):
    mod = _load_script()
    pack_id = "test-pack"
    pack_dir = tmp_path / "pack"
    (pack_dir / "scoring").mkdir(parents=True)
    _write(pack_dir / "连续分人盲标清单.json", {
        "pack_id": pack_id,
        "items": [{"id": "CONT-01", "window_start": 0, "window_end": 4}],
    })
    _write(pack_dir / "scoring" / "当前通用分人预测.json", {
        "pack_id": pack_id,
        "segments": [{"uri": "meeting", "start": 0, "end": 4, "speaker": "X"}],
    })
    gold = _write(tmp_path / "gold.json", {
        "kind": "continuous_diarization_gold",
        "pack_id": pack_id,
        "items": [{
            "id": "CONT-01",
            "uri": "meeting",
            "window_start": 0,
            "window_end": 4,
            "complete": True,
            "markers": [{"time": 0, "speakers": ["A"]}],
        }],
        "segments": [
            {"uri": "meeting", "start": 0, "end": 4, "speaker": "A"},
            {"uri": "meeting", "start": 2, "end": 4, "speaker": "B"},
        ],
    })

    result = mod.score_pack(gold, pack_dir, tmp_path / "score")
    report = json.loads(Path(result["json"]).read_text(encoding="utf-8"))

    assert abs(result["DER"] - 1 / 3) < 1e-6
    assert result["overlap_error_rate"] == 0.5
    assert result["speaker_count_absolute_error"] == 1
    assert report["连续验收时长分钟"] == 0.067
    assert report["有效人工覆盖分钟"] == 0.067


def test_annotation_coverage_excludes_unlabeled_leading_audio():
    mod = _load_script()
    coverage, omitted = mod.annotation_coverage({"items": [{
        "uri": "meeting",
        "window_start": 10,
        "window_end": 30,
        "markers": [{"time": 14, "speakers": ["A"]}],
    }]})
    recordings = {
        "meeting": [mod.Segment("meeting", 10, 20, "X")],
    }

    cropped = mod.crop_recordings(recordings, coverage)

    assert coverage == {"meeting": (14.0, 30.0)}
    assert omitted == 4.0
    assert cropped["meeting"] == [mod.Segment("meeting", 14.0, 20, "X")]


def test_validate_inputs_rejects_incomplete_review(tmp_path: Path):
    mod = _load_script()
    pack_id = "test-pack"
    manifest = _write(tmp_path / "manifest.json", {
        "pack_id": pack_id,
        "items": [{"id": "CONT-01"}, {"id": "CONT-02"}],
    })
    prediction = _write(tmp_path / "prediction.json", {
        "pack_id": pack_id,
        "segments": [{"start": 0, "end": 1, "speaker": "X"}],
    })
    gold = _write(tmp_path / "gold.json", {
        "kind": "continuous_diarization_gold",
        "pack_id": pack_id,
        "items": [{
            "id": "CONT-01",
            "complete": True,
            "markers": [{"time": 0, "speakers": ["A"]}],
        }],
        "segments": [{"start": 0, "end": 1, "speaker": "A"}],
    })

    try:
        mod.validate_inputs(gold, prediction, manifest)
    except ValueError as exc:
        assert "CONT-02" in str(exc)
    else:
        raise AssertionError("incomplete continuous review must not be scored")
