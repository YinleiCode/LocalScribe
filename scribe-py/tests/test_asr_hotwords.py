from __future__ import annotations

import sys
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from scribe_py.core.hotwords import build_initial_prompt, build_hotword_string, parse_hotword_terms


def test_parse_hotword_terms_from_plain_text_and_commas():
    terms = parse_hotword_terms("""
    # customer glossary
    李会，兰艺, 金子
    - 青年团契
    1. 为我祷告
    李会
    """)

    assert terms == ["李会", "兰艺", "金子", "青年团契", "为我祷告"]


def test_parse_hotword_terms_from_json_glossary():
    terms = parse_hotword_terms(
        '{"glossary":[{"term":"LocalScribe"},{"name":"DeepSeek"},{"word":"Hermes"}]}'
    )

    assert terms == ["LocalScribe", "DeepSeek", "Hermes"]


def test_build_prompts_from_hotwords():
    terms = ["李会", "兰艺"]

    assert build_hotword_string(terms) == "李会 兰艺"
    prompt = build_initial_prompt("请使用简体中文。", terms)
    assert "请使用简体中文。" in prompt
    assert "李会、兰艺" in prompt
