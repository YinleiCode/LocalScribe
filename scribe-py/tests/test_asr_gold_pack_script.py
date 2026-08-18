from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "asr_gold_pack.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("asr_gold_pack", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_rows_filters_short_and_punctuation_only_rows_and_balances_cases():
    mod = _load_script()
    template = {
        "items": [
            {"case": "A", "start": 0, "end": 0.5, "current_text": "太短", "sample_type": "strong"},
            {"case": "A", "start": 1, "end": 3, "current_text": "。", "sample_type": "weak"},
            {"case": "A", "start": 3, "end": 6, "current_text": "强疑点文本", "sample_type": "strong"},
            {"case": "A", "start": 6, "end": 9, "current_text": "正常抽查文本", "sample_type": "normal"},
            {"case": "B", "start": 0, "end": 4, "current_text": "普通疑点文本", "sample_type": "weak"},
        ]
    }

    rows = mod.select_rows(template, min_duration=1.5, min_text_chars=4, per_case=2)

    assert [(row["case"], row["sample_type"]) for row in rows] == [
        ("A", "strong"),
        ("A", "normal"),
        ("B", "weak"),
    ]


def test_build_and_write_dry_pack_uses_transcript_audio(tmp_path: Path):
    mod = _load_script()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio-placeholder")
    transcript = tmp_path / "result.json"
    transcript.write_text(
        json.dumps({"audio": str(audio), "duration": 20.0, "segments": []}),
        encoding="utf-8",
    )
    template = {
        "cases": [{"case": "A", "transcript": str(transcript)}],
        "items": [],
    }
    rows = [
        {
            "case": "A",
            "index": 2,
            "start": 5.0,
            "end": 8.0,
            "sample_type": "weak",
            "current_text": "当前转写文本。",
        }
    ]

    items = mod.build_pack_items(template, rows, padding_seconds=1.5)
    json_path, markdown_path = mod.write_pack(items, out_dir=tmp_path / "pack", dry_run=True)

    assert items[0]["clip_start"] == 3.5
    assert items[0]["clip_end"] == 9.5
    assert json_path.exists()
    assert markdown_path.exists()
    assert items[0]["clip_path"].startswith("clips/")
    assert items[0]["eval_clip_path"].startswith("eval_clips/")
    assert "GOLD-001 确认" in markdown_path.read_text(encoding="utf-8")
