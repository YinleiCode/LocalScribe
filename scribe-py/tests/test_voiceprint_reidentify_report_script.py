from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "voiceprint_reidentify_report.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("voiceprint_reidentify_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_global_assignment_uses_best_unique_mapping():
    module = _load_script_module()
    scores = {
        "SPEAKER_A": {"张三": 0.90, "李四": 0.80},
        "SPEAKER_B": {"张三": 0.89, "李四": 0.10},
    }

    result = module.global_one_to_one_assignment(scores)

    assert result == {"SPEAKER_A": "李四", "SPEAKER_B": "张三"}
    assert len(set(result.values())) == len(result)


def test_global_assignment_leaves_low_confidence_speaker_unassigned():
    module = _load_script_module()
    scores = {
        "SPEAKER_A": {"张三": 0.91, "李四": 0.30},
        "SPEAKER_B": {"张三": 0.70, "李四": 0.49},
    }

    result = module.global_one_to_one_assignment(scores, min_score=0.75)

    assert result == {"SPEAKER_A": "张三"}


def test_report_cli_writes_chinese_md_tsv_and_json(tmp_path: Path):
    original = tmp_path / "before.json"
    reidentified = tmp_path / "after.json"
    original.write_text(json.dumps({
        "segments": [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_A", "text": "第一段"},
            {"start": 3.0, "end": 7.0, "speaker": "SPEAKER_B", "text": "第二段"},
            {"start": 7.0, "end": 9.0, "speaker": "SPEAKER_B", "text": "第三段"},
        ],
        "diarization_stats": {"risk_level": "high"},
    }, ensure_ascii=False), encoding="utf-8")
    reidentified.write_text(json.dumps({
        "segments": [
            {"start": 0.0, "end": 3.0, "speaker": "张三", "original_speaker": "SPEAKER_A", "text": "第一段", "speaker_voiceprint_reidentified": True, "speaker_voiceprint_score": 0.92, "speaker_voiceprint_anchor": "张三"},
            {"start": 3.0, "end": 7.0, "speaker": "李四", "original_speaker": "SPEAKER_B", "text": "第二段", "speaker_voiceprint_reidentified": True, "speaker_voiceprint_score": 0.88, "speaker_voiceprint_anchor": "李四"},
            {"start": 7.0, "end": 9.0, "speaker": "SPEAKER_B", "text": "第三段", "speaker_assignment_review": True, "speaker_voiceprint_review": True, "speaker_voiceprint_score": 0.73, "speaker_voiceprint_anchor": "李四", "speaker_review_reason": "分数位于边界带"},
        ],
        "profiles": [
            {"name": "张三", "anchor_count": 2, "sample_seconds": 12.5, "enrollment_ready": True, "enrollment_reasons": [], "quality": {"median_similarity": 0.91, "min_similarity": 0.84, "vector_count": 6}},
            {"name": "李四", "anchor_count": 2, "sample_seconds": 11.0, "enrollment_ready": True, "enrollment_reasons": [], "quality": {"median_similarity": 0.89, "min_similarity": 0.81, "vector_count": 5}},
        ],
        "stats": {"risk_level": "low"},
    }, ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--original", str(original), "--reidentified", str(reidentified), "--out-dir", str(tmp_path / "out"), "--prefix", "demo"],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["changed_segments"] == 2
    assert payload["review_segments"] == 1
    assert payload["risk_change"] == "降低"
    report = json.loads(Path(payload["json"]).read_text(encoding="utf-8"))
    assert report["distribution"]["before"][0]["speaker"] == "SPEAKER_A"
    assert report["change_pairs"] == [
        {"before_speaker": "SPEAKER_A", "after_speaker": "张三", "segments": 1},
        {"before_speaker": "SPEAKER_B", "after_speaker": "李四", "segments": 1},
    ]
    assert report["profile_quality"][0]["enrollment_ready"] is True
    markdown = Path(payload["markdown"]).read_text(encoding="utf-8")
    assert "声纹重识别前后对比报告" in markdown
    assert "分人风险: 高 -> 低（降低）" in markdown
    assert "分数位于边界带" in markdown
    tsv = Path(payload["tsv"]).read_text(encoding="utf-8-sig")
    assert "每人分布" in tsv
    assert "改派段" in tsv
    assert "锚点质量" in tsv


def test_build_report_detects_same_level_review_risk_change():
    module = _load_script_module()
    before = {
        "segments": [
            {"start": 0, "end": 1, "speaker": "A", "speaker_assignment_review": True},
            {"start": 1, "end": 2, "speaker": "A"},
        ],
        "diarization_stats": {"risk_level": "medium"},
    }
    after = {
        "segments": [
            {"start": 0, "end": 1, "speaker": "A"},
            {"start": 1, "end": 2, "speaker": "A"},
        ],
        "diarization_stats": {"risk_level": "medium"},
    }

    report = module.build_report(before, after)

    assert report["risk"]["change"] == "同级但待确认减少"
