from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "asr_app_equivalent_run.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("asr_app_equivalent_run", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_params_matches_app_defaults(tmp_path: Path):
    mod = _load_script()
    audio = tmp_path / "demo.wav"
    args = argparse.Namespace(
        audio=audio,
        backend="sensevoice",
        language="zh",
        normalizer_profile="",
        audio_preprocess="adaptive",
        asr_quality_mode="standard",
        word_timestamps=False,
        fast_timing=False,
    )

    assert mod.build_params(args) == {
        "audio": str(audio.resolve()),
        "backend": "sensevoice",
        "language": "zh",
        "normalizer_profile": None,
        "audio_preprocess": "adaptive",
        "asr_quality_mode": "standard",
        "word_timestamps": False,
        "timing_align": True,
    }


def test_atomic_json_and_summary(tmp_path: Path):
    mod = _load_script()
    output = tmp_path / "result.json"
    result = {
        "audio": "/audio.wav",
        "duration": 10.0,
        "segments": [{"text": "你好"}],
        "transcribe_seconds": 2.0,
        "asr_quality": {"risk_level": "high"},
        "filter_stats": {
            "strong_asr": {
                "enabled": True,
                "applied": True,
                "trigger": "auto_high_noise",
                "replacement_count": 1,
            }
        },
    }

    mod.write_json_atomic(output, result)
    row = mod.summary(result, wall_seconds=2.5, output=output)

    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert row["characters"] == 2
    assert row["strong_asr"]["trigger"] == "auto_high_noise"
