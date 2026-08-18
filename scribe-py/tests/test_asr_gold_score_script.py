from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "asr_gold_score.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("asr_gold_score", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_punctuation_and_whitespace_are_ignored_by_default():
    mod = _load_script_module()

    assert mod.normalize_for_cer("你好， 世界。") == "你好世界"
    assert mod.edit_distance(
        mod.normalize_for_cer("你好，世界。"),
        mod.normalize_for_cer("你好世界"),
    ) == 0


def test_json_template_scores_filled_rows_and_skips_empty_rows(tmp_path: Path):
    mod = _load_script_module()
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "source_json": "demo.json",
                "items": [
                    {"id": 1, "time": "00:00:00.000 - 00:00:01.000", "current_text": "你好世界", "correct_text": "你好世间"},
                    {"id": 2, "time": "00:00:01.000 - 00:00:02.000", "current_text": "今天下雨", "correct_text": "今天下雨了"},
                    {"id": 3, "time": "00:00:02.000 - 00:00:03.000", "current_text": "空答案", "correct_text": ""},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows_raw, metadata = mod._load_gold_rows(gold)
    summary, rows = mod.score_gold_rows(rows_raw)

    assert metadata["source_format"] == "json"
    assert summary["total_rows"] == 3
    assert summary["filled_gold_rows"] == 2
    assert summary["skipped_empty_rows"] == 1
    assert summary["total_reference_chars"] == 9
    assert summary["total_edit_distance"] == 2
    assert summary["overall_cer"] == 2 / 9
    assert [row.edit_distance for row in rows] == [1, 1]


def test_csv_template_with_utf8_bom_is_supported(tmp_path: Path):
    mod = _load_script_module()
    gold = tmp_path / "gold.csv"
    with gold.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "current_text", "correct_text", "notes"])
        writer.writeheader()
        writer.writerow({"id": "1", "current_text": "慎重好虑", "correct_text": "慎重考虑", "notes": "人工确认"})

    rows_raw, metadata = mod._load_gold_rows(gold)
    summary, rows = mod.score_gold_rows(rows_raw)

    assert metadata["source_format"] == "csv"
    assert summary["filled_gold_rows"] == 1
    assert rows[0].id == "1"
    assert rows[0].notes == "人工确认"
    assert rows[0].edit_distance == 1
    assert rows[0].cer == 0.25


def test_cli_writes_chinese_markdown_and_json_report(tmp_path: Path):
    gold = tmp_path / "gold.json"
    out = tmp_path / "reports"
    gold.write_text(
        json.dumps(
            [
                {"id": "a", "time": "00:00:00.000 - 00:00:01.000", "current_text": "你好，世界。", "correct_text": "你好世界"},
                {"id": "b", "time": "00:00:01.000 - 00:00:02.000", "current_text": "没填", "correct_text": ""},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(gold), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    stdout = json.loads(proc.stdout)
    assert stdout["filled_gold_rows"] == 1
    assert stdout["skipped_empty_rows"] == 1
    assert stdout["overall_cer"] == 0
    report_json = out / "ASR回归评分报告.json"
    report_md = out / "ASR回归评分报告.md"
    assert report_json.exists()
    assert report_md.exists()
    assert "ASR 人工标准答案回归评分" in report_md.read_text(encoding="utf-8")
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["summary"]["overall_cer"] == 0


def test_empty_current_template_does_not_fail(tmp_path: Path):
    mod = _load_script_module()
    rows = [
        {"id": 1, "current_text": "当前文本", "correct_text": ""},
        {"id": 2, "current_text": "当前文本二", "correct_text": "   "},
    ]

    summary, scored = mod.score_gold_rows(rows)

    assert scored == []
    assert summary["filled_gold_rows"] == 0
    assert summary["skipped_empty_rows"] == 2
    assert summary["overall_cer"] is None
