from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "asr_funasr_nano_eval.py"
SPEC = importlib.util.spec_from_file_location("asr_funasr_nano_eval", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_generation_token_limit_is_duration_bounded():
    assert MODULE.generation_token_limit(0) == 64
    assert MODULE.generation_token_limit(20) == 272
    assert MODULE.generation_token_limit(1000) == 768


def test_repetition_risk_detects_autoregressive_loop():
    assert MODULE.has_repetition_risk("我" * 20)
    assert not MODULE.has_repetition_risk("这是正常的中文会议转录内容。")
