from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "asr_benchmark.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("asr_benchmark", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recommendation_prefers_lower_asr_risk_before_speed():
    mod = _load_script_module()
    rows = [
        {
            "audio": "demo.mp3",
            "backend": "fast",
            "model": "fast-model",
            "status": "ok",
            "risk_level": "medium",
            "strong_review_count": 0,
            "review_count": 1,
            "punctuation_ratio": "1.0000",
            "traditional_count": 0,
            "rtf": "0.1",
            "cost_s": "1.0",
            "out_dir": "/tmp/fast",
        },
        {
            "audio": "demo.mp3",
            "backend": "stable",
            "model": "stable-model",
            "status": "ok",
            "risk_level": "low",
            "strong_review_count": 0,
            "review_count": 2,
            "punctuation_ratio": "1.0000",
            "traditional_count": 0,
            "rtf": "0.8",
            "cost_s": "8.0",
            "out_dir": "/tmp/stable",
        },
    ]

    recommendation = mod.build_recommendation(rows)

    assert recommendation["recommended_backend"] == "stable"
    assert recommendation["recommendations"][0]["ranked"][0]["backend"] == "stable"


def test_recommendation_groups_multiple_audios_and_records_failed_backends(tmp_path: Path):
    mod = _load_script_module()
    rows = [
        {
            "audio": "a.mp3",
            "backend": "sensevoice",
            "model": "iic/SenseVoiceSmall",
            "status": "ok",
            "risk_level": "low",
            "strong_review_count": 0,
            "review_count": 1,
            "punctuation_ratio": "1.0000",
            "traditional_count": 0,
            "rtf": "0.5",
            "cost_s": "5",
            "out_dir": "/tmp/a/sensevoice",
        },
        {
            "audio": "b.mp3",
            "backend": "mlx",
            "model": "mlx-model",
            "status": "error",
            "error": "missing dependency",
            "out_dir": "/tmp/b/mlx",
        },
        {
            "audio": "b.mp3",
            "backend": "ct2",
            "model": "ct2-model",
            "status": "ok",
            "risk_level": "medium",
            "strong_review_count": 1,
            "review_count": 4,
            "punctuation_ratio": "0.9000",
            "traditional_count": 0,
            "rtf": "1.2",
            "cost_s": "12",
            "out_dir": "/tmp/b/ct2",
        },
    ]

    out = tmp_path / "bench"
    out.mkdir()
    mod.write_recommendation(out, rows)

    payload = json.loads((out / "recommendation.json").read_text(encoding="utf-8"))
    assert len(payload["recommendations"]) == 2
    assert payload["recommendations"][0]["audio"] == "a.mp3"
    assert payload["recommendations"][1]["recommended_backend"] == "ct2"
    assert payload["recommendations"][1]["failed"][0]["backend"] == "mlx"
    markdown = (out / "recommendation.md").read_text(encoding="utf-8")
    assert "ASR 后端推荐" in markdown
    assert "missing dependency" in markdown
