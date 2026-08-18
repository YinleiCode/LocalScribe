from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "asr_benchmark_gold_score.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("asr_benchmark_gold_score", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_result(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"segments": [{"text": text}]}, ensure_ascii=False), encoding="utf-8")


def test_score_benchmark_excludes_partial_rows_and_ranks_lower_cer_first(tmp_path: Path):
    mod = _load_script()
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(
        json.dumps(
            {
                "items": [
                    {"id": "GOLD-001", "case": "A", "correct_text": "你好世界", "decision": "确认"},
                    {"id": "GOLD-002", "case": "A", "correct_text": "", "decision": "部分校正"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    benchmark = tmp_path / "benchmark"
    _write_result(benchmark / "GOLD-001_demo" / "sensevoice" / "result.json", "你好世界")
    _write_result(benchmark / "GOLD-001_demo" / "qwen3" / "result.json", "你好世间")

    report = mod.score_benchmark(gold_path, benchmark)

    assert report["filled_gold_items"] == 1
    assert report["excluded_partial_items"] == ["GOLD-002"]
    assert [row["backend"] for row in report["summaries"]] == ["sensevoice", "qwen3"]
    assert report["summaries"][0]["cer"] == 0.0
    assert report["summaries"][1]["edit_distance"] == 1
