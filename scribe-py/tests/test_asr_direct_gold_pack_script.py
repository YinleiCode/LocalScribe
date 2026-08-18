from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "asr_direct_gold_pack.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("asr_direct_gold_pack", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_windows_are_bounded_and_distributed():
    mod = _load_script()

    windows = mod.candidate_windows(100.0, 10.0)

    assert len(windows) == 7
    assert all(0.0 <= start <= 90.0 and length == 10.0 for start, length in windows)
    assert windows[0][0] < windows[-1][0]


def test_candidate_windows_use_whole_short_audio():
    mod = _load_script()

    assert mod.candidate_windows(4.0, 10.0) == [(0.0, 4.0)]


def test_render_markdown_states_direct_audio_timing():
    mod = _load_script()
    markdown = mod.render_markdown(
        [
            {
                "id": "GOLD-001",
                "case_id": "demo",
                "case": "示例",
                "start": 5.0,
                "duration": 10.0,
                "current_text": "当前文字。",
            }
        ]
    )

    assert "未使用历史转录时间戳" in markdown
    assert "GOLD-001 确认" in markdown


def test_render_html_supports_local_review_and_json_export():
    mod = _load_script()
    page = mod.render_html(
        {
            "title": "ASR 通用盲测标注",
            "pack_id": "pack-demo",
            "items": [
                {
                    "id": "GOLD-001",
                    "case": "陌生录音",
                    "start": 5.0,
                    "duration": 10.0,
                    "clip_path": "clips/one.wav",
                    "current_text": "当前文字。",
                }
            ],
        }
    )

    assert "<audio" in page
    assert "文字正确" in page
    assert "保存修改" in page
    assert "听不清/不计分" in page
    assert "localStorage" in page
    assert "ASR通用人工标准答案.json" in page
    assert "当前文字。" in page
    assert "replace(/\\n/g" in page


def test_pack_id_changes_when_source_or_clip_changes():
    mod = _load_script()
    item = {
        "case_id": "demo",
        "audio_sha256": "a" * 64,
        "start": 1.0,
        "duration": 10.0,
        "clip_sha256": "b" * 64,
    }

    first = mod._pack_id([item])
    second = mod._pack_id([{**item, "clip_sha256": "c" * 64}])

    assert first.startswith("asr-direct-gold-")
    assert first != second
