from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "asr_regression_report.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("asr_regression_report", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_transcript(path: Path, *, profile: str = "") -> None:
    payload = {
        "backend": "sensevoice",
        "model_id": "demo",
        "duration": 3.0,
        "transcribe_seconds": 1.0,
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "你好世界。"},
            {"start": 1.0, "end": 2.0, "text": "今天下雨"},
        ],
        "filter_stats": {"text_normalization": {"profile": profile}},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_quality(path: Path) -> None:
    payload = {
        "mode": "local_asr_quality",
        "segments": 2,
        "chars": 8,
        "punctuation_ratio": 0.5,
        "traditional_char_hits": ["聽"],
        "hotwords": {
            "count": 2,
            "exact_hit_count": 1,
            "missing_terms": ["张三"],
        },
        "review": {
            "segment_count": 2,
            "strong_segment_count": 1,
            "segments": [{"index": 1, "reasons": ["疑似明显语义不顺"]}],
        },
        "term_consistency": {
            "candidate_count": 3,
            "candidates": [],
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_render_multi_case_report_with_gold_and_profile_warning(tmp_path: Path):
    mod = _load_script_module()
    case_a_dir = tmp_path / "case_a"
    case_a_dir.mkdir()
    transcript_a = case_a_dir / "result.json"
    _write_transcript(transcript_a)
    _write_quality(case_a_dir / "ASR质量检查.json")

    case_b_dir = tmp_path / "case_b"
    case_b_dir.mkdir()
    transcript_b = case_b_dir / "result.json"
    _write_transcript(transcript_b, profile="standard3")

    gold = tmp_path / "a_gold.json"
    gold.write_text(
        json.dumps(
            {
                "items": [
                    {"index": 0, "current_text": "你好世界。", "correct_text": "你好世间"},
                    {"index": 1, "current_text": "今天下雨", "correct_text": "今天下雨了"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = mod.load_cases(
        [f"录音A={transcript_a}", f"录音B={transcript_b}"],
        [f"录音A={gold}"],
        None,
    )
    report = mod.render_markdown(cases)

    assert "ASR 通用回归评测汇总" in report
    assert "| 总本地疑点数 | 2 |" in report
    assert "| 总强疑点数 | 1 |" in report
    assert "| 总同音/近音实体一致性候选数 | 3 |" in report
    assert "| 录音A | 2 | 8 | 2 | 1 | 3 | 1 | 50.0% | 张三 | 22.22% | 22.22% | 通用 | 可按通用能力观察 |" in report
    assert "录音B 使用了非空 profile" in report
    assert "不能直接视为纯通用 ASR 能力" in report
    assert "gold 评分口径" in report


def test_cli_writes_customer_facing_markdown(tmp_path: Path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    transcript = case_dir / "result.json"
    _write_transcript(transcript)
    out = tmp_path / "report.md"

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--case", f"真实录音1={transcript}", "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    stdout = json.loads(proc.stdout)
    assert stdout["ok"] is True
    assert stdout["cases"] == 1
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "逐条录音指标" in text
    assert "未提供 gold 标注" in text
