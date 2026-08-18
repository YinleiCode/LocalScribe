from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "asr_disagreement_review_pack.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("asr_disagreement_review_pack", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_result(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"segments": [{"text": text}]}, ensure_ascii=False), encoding="utf-8")


def test_disagreement_pack_prioritizes_auxiliary_consensus_and_preserves_case_coverage(tmp_path: Path):
    mod = _load_script()
    template = tmp_path / "人工标准答案模板.json"
    template.write_text(
        json.dumps(
            {
                "items": [
                    {"id": "GOLD-001", "case_id": "a", "case": "A", "clip_path": "a.wav", "current_text": "今天讨论项目进度"},
                    {"id": "GOLD-002", "case_id": "a", "case": "A", "clip_path": "b.wav", "current_text": "完全错误的一长句话"},
                    {"id": "GOLD-003", "case_id": "b", "case": "B", "clip_path": "c.wav", "current_text": "客户确认部署方案"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    benchmark = tmp_path / "benchmark"
    values = {
        "GOLD-001_demo": ("今天讨论项目进度", "今天讨论项目进度"),
        "GOLD-002_demo": ("应该使用正确候选文本", "应该使用正确候选文本"),
        "GOLD-003_demo": ("客户确认部署方式", "客户确认部署方案"),
    }
    for case, (paraformer, qwen3) in values.items():
        _write_result(benchmark / case / "funasr" / "result.json", paraformer)
        _write_result(benchmark / case / "qwen3" / "result.json", qwen3)

    rows = mod.build_rows(template, benchmark)
    assert rows[0]["id"] == "GOLD-002"
    assert rows[0]["auxiliary_consensus_against_primary"] is True

    selected = mod.select_rows(rows, limit=2, per_case=1)
    assert {row["case_id"] for row in selected} == {"a", "b"}
    assert "不会自动替换" in mod.render_markdown(selected, total=len(rows))


def test_disagreement_metrics_are_low_for_identical_text():
    mod = _load_script()
    metrics = mod.disagreement_metrics("同一段文字", "同一段文字", "同一段文字")

    assert metrics["disagreement_score"] == 0.0
    assert metrics["auxiliary_consensus_against_primary"] is False


def test_materialize_review_clips_numbers_selected_audio(tmp_path: Path):
    mod = _load_script()
    source = tmp_path / "source.wav"
    source.write_bytes(b"RIFFtest")

    rows = mod.materialize_review_clips(
        [{"id": "GOLD-007", "clip_path": str(source)}],
        tmp_path / "review",
    )

    copied = Path(rows[0]["review_clip_path"])
    assert copied.name == "01_GOLD-007.wav"
    assert copied.read_bytes() == source.read_bytes()
