from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "asr_gold_sample.py"
EXPECTED_ITEM_FIELDS = ["case", "index", "start", "end", "current_text", "reasons", "sample_type", "correct_text"]


def _load_script_module():
    spec = importlib.util.spec_from_file_location("asr_gold_sample", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_transcript(path: Path, count: int = 6) -> None:
    payload = {
        "backend": "sensevoice",
        "segments": [
            {"start": float(i), "end": float(i + 1), "text": f"第{i}段正常文本。"}
            for i in range(count)
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_quality(path: Path) -> None:
    payload = {
        "mode": "local_asr_quality",
        "review": {
            "segment_count": 3,
            "strong_segment_count": 1,
            "segments": [
                {
                    "index": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "第0段正常文本。",
                    "reasons": ["疑似明显语义不顺"],
                },
                {
                    "index": 1,
                    "start": 1.0,
                    "end": 2.0,
                    "text": "第1段正常文本。",
                    "reasons": ["疑似重复词"],
                },
            ],
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_sample_case_includes_strong_weak_and_normal_rows(tmp_path: Path):
    mod = _load_script_module()
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    transcript = case_dir / "result.json"
    _write_transcript(transcript)
    _write_quality(case_dir / "ASR质量检查.json")

    payload = mod.build_template(
        [f"录音A={transcript}"],
        per_case_strong=1,
        per_case_weak=1,
        per_case_normal=2,
        seed=7,
    )

    types = [row["sample_type"] for row in payload["items"]]
    assert types.count("strong") == 1
    assert types.count("weak") == 1
    assert types.count("normal") == 2
    assert payload["cases"][0]["strong_candidate_count"] == 1
    assert payload["cases"][0]["weak_candidate_count"] == 1
    assert payload["cases"][0]["normal_candidate_count"] == 4
    assert payload["items"][0]["case"] == "录音A"
    assert payload["items"][0]["correct_text"] == ""
    assert payload["items"][0]["reasons"] == ["疑似明显语义不顺"]


def test_missing_quality_sidecar_samples_normal_segments(tmp_path: Path):
    mod = _load_script_module()
    transcript = tmp_path / "result.json"
    _write_transcript(transcript, count=4)

    payload = mod.build_template(
        [f"无质检={transcript}"],
        per_case_strong=2,
        per_case_weak=2,
        per_case_normal=3,
        seed=1,
    )

    assert [row["sample_type"] for row in payload["items"]] == ["normal", "normal", "normal"]
    assert all(row["reasons"] == [] for row in payload["items"])
    assert payload["cases"][0]["quality_sidecar"] == ""
    assert payload["cases"][0]["strong_candidate_count"] == 0
    assert payload["cases"][0]["weak_candidate_count"] == 0
    assert payload["cases"][0]["normal_candidate_count"] == 4


def test_output_item_fields_are_stable(tmp_path: Path):
    mod = _load_script_module()
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    transcript = case_dir / "result.json"
    _write_transcript(transcript)
    _write_quality(case_dir / "ASR质量检查.json")

    payload = mod.build_template([f"录音A={transcript}"], per_case_strong=1, per_case_weak=1, per_case_normal=1, seed=0)

    assert list(payload["items"][0].keys()) == EXPECTED_ITEM_FIELDS
    assert set(payload["items"][0]) == set(EXPECTED_ITEM_FIELDS)
    assert isinstance(payload["items"][0]["start"], float)
    assert isinstance(payload["items"][0]["end"], float)
    assert isinstance(payload["items"][0]["current_text"], str)


def test_cli_writes_template_json(tmp_path: Path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    transcript = case_dir / "result.json"
    _write_transcript(transcript)
    _write_quality(case_dir / "ASR质量检查.json")
    out = tmp_path / "gold_sample.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--case",
            f"录音A={transcript}",
            "--out",
            str(out),
            "--per-case-strong",
            "1",
            "--per-case-weak",
            "1",
            "--per-case-normal",
            "1",
            "--seed",
            "9",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    stdout = json.loads(proc.stdout)
    assert stdout["ok"] is True
    assert stdout["items"] == 3
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["template"] == "ASR人工校对gold抽样模板"
    assert {row["sample_type"] for row in payload["items"]} == {"strong", "weak", "normal"}
    assert list(payload["items"][0].keys()) == EXPECTED_ITEM_FIELDS
