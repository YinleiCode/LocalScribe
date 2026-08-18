from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "asr_manual_check_pack.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("asr_manual_check_pack", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_transcript(path: Path, *, audio: Path, text: str) -> None:
    path.write_text(
        json.dumps(
            {
                "audio": str(audio),
                "duration": 30.0,
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "前文。"},
                    {"start": 10.0, "end": 20.0, "text": text},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_build_items_groups_multiple_changes_in_one_audio_window(tmp_path: Path):
    mod = _load_script()
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"not-cut-in-this-test")
    standard_dir = tmp_path / "standard"
    strong_dir = tmp_path / "strong"
    standard_dir.mkdir()
    strong_dir.mkdir()
    _write_transcript(standard_dir / "demo.json", audio=audio, text="我们流写分离和看出业务。")
    _write_transcript(strong_dir / "demo.json", audio=audio, text="我们读写分离和探测业务。")
    comparison = {
        "rows": [{"case": "demo", "category": "未见样本"}],
        "manual_checks": [
            {"case": "demo", "start": 10.0, "end": 20.0, "from": "流", "to": "读"},
            {"case": "demo", "start": 10.0, "end": 20.0, "from": "看出", "to": "探测"},
        ],
    }

    items = mod.build_items(
        comparison,
        standard_dir=standard_dir,
        strong_dir=strong_dir,
        padding_seconds=2.0,
    )

    assert len(items) == 1
    assert items[0]["id"] == "ASR-001"
    assert items[0]["clip_start"] == 8.0
    assert items[0]["clip_end"] == 22.0
    assert [change["to"] for change in items[0]["changes"]] == ["读", "探测"]
    assert items[0]["strong_text"] == "我们读写分离和探测业务。"


def test_write_pack_dry_run_writes_a_reusable_confirmation_template(tmp_path: Path):
    mod = _load_script()
    items = [
        {
            "id": "ASR-001",
            "case": "demo",
            "review_start": 1.0,
            "review_end": 2.0,
            "clip_start": 0.0,
            "clip_end": 4.0,
            "audio": str(tmp_path / "source.wav"),
            "standard_text": "标准文本。",
            "strong_text": "高质量文本。",
            "changes": [{"from": "标", "to": "高", "evidence": "consensus"}],
            "decision": "",
            "final_text": "",
            "notes": "",
        }
    ]

    manifest, markdown = mod.write_pack(items, out_dir=tmp_path / "pack", dry_run=True)

    assert manifest.exists()
    assert markdown.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["items"][0]["clip_path"].startswith("clips/ASR-001_demo_")
    assert "ASR-001 确认" in markdown.read_text(encoding="utf-8")
