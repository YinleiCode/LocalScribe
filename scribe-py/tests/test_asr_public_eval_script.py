from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "asr_public_eval.py"
SPEC = importlib.util.spec_from_file_location("asr_public_eval", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalize_cer_text_ignores_punctuation_and_unifies_script():
    assert MODULE.normalize_cer_text("讀寫 分離，Redis！") == "读写分离redis"


def test_edit_distance_counts_chinese_character_errors():
    assert MODULE.edit_distance("读写分离", "流写分离") == 1
    assert MODULE.edit_distance("缓存挂了", "缓存") == 2
