from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "asr_entity_review.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("asr_entity_review", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_case(tmp_path: Path) -> tuple[Path, Path]:
    transcript = tmp_path / "case.json"
    transcript.write_text(
        json.dumps(
            {
                "backend": "sensevoice",
                "model_id": "iic/SenseVoiceSmall",
                "segments": [
                    {"start": 0, "end": 5, "text": "兰艺和金子一起沟通。"},
                    {"start": 5, "end": 10, "text": "蓝衣同学后来跟金子确认。"},
                    {"start": 10, "end": 15, "text": "因为对方是男人和金子，所以不好意思。"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    quality = tmp_path / "ASR质量检查.json"
    quality.write_text(
        json.dumps(
            {
                "mode": "local_asr_quality",
                "term_consistency": {
                    "candidate_count": 2,
                    "candidates": [
                        {
                            "id": "term-consistency-1",
                            "kind": "phonetic_entity",
                            "action": "review",
                            "confidence": 0.94,
                            "phonetic_key": "lan-yi",
                            "terms": ["兰艺", "蓝衣"],
                            "suggested_canonical": None,
                            "total_count": 2,
                            "variants": [
                                {"text": "兰艺", "count": 1, "contexts": [{"index": 0, "start": 0, "end": 5, "text": "兰艺和金子一起沟通。"}]},
                                {"text": "蓝衣", "count": 1, "contexts": [{"index": 1, "start": 5, "end": 10, "text": "蓝衣同学后来跟金子确认。"}]},
                            ],
                            "contexts": [
                                {"index": 0, "start": 0, "end": 5, "text": "兰艺和金子一起沟通。"},
                                {"index": 1, "start": 5, "end": 10, "text": "蓝衣同学后来跟金子确认。"},
                            ],
                            "reason": "相同/近似读音的实体写法在上下文中反复出现，建议确认标准写法；系统不自动替换。",
                        },
                        {
                            "id": "term-consistency-2",
                            "kind": "entity_drift",
                            "action": "review",
                            "confidence": 0.72,
                            "phonetic_key": "lan-yi",
                            "terms": ["男人", "兰艺"],
                            "suggested_canonical": None,
                            "total_count": 2,
                            "variants": [
                                {"text": "男人", "count": 0, "contexts": []},
                                {"text": "兰艺", "count": 1, "contexts": [{"index": 0, "start": 0, "end": 5, "text": "兰艺和金子一起沟通。"}]},
                            ],
                            "contexts": [
                                {"index": 10, "start": 10, "end": 15, "text": "因为对方是男人和金子，所以不好意思。"}
                            ],
                            "reason": "“男人和金子”像把人名/实体识别成普通词；建议人工核对该实体，不自动替换。",
                        },
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return transcript, quality


def test_export_review_writes_chinese_checklist_and_decision_template(tmp_path: Path):
    mod = _load_script_module()
    transcript, quality = _write_case(tmp_path)
    out = tmp_path / "review"

    result = mod.export_review(transcript, quality_json=quality, out_dir=out)
    payload = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    markdown = Path(result["md"]).read_text(encoding="utf-8")

    assert payload["candidate_count"] == 2
    assert payload["decisions"][0]["action"] == "skip"
    assert payload["decisions"][0]["allow_global_unify"] is True
    assert payload["decisions"][1]["allow_global_unify"] is False
    assert payload["decisions"][1]["occurrence_replacements"][0]["index"] == 10
    assert "ASR 实体一致性核对清单" in markdown
    assert "同音实体" in markdown
    assert "实体漂移" in markdown
    assert "系统不自动替换" in markdown


def test_apply_review_skips_by_default_and_does_not_modify_text(tmp_path: Path):
    mod = _load_script_module()
    transcript, quality = _write_case(tmp_path)
    out = tmp_path / "review"
    review_json = Path(mod.export_review(transcript, quality_json=quality, out_dir=out)["json"])
    output = tmp_path / "applied.json"

    result = mod.apply_review(transcript, review_json, out_json=output, write_quality=False)
    data = json.loads(output.read_text(encoding="utf-8"))

    assert result["replacement_count"] == 0
    assert [seg["text"] for seg in data["segments"]] == [
        "兰艺和金子一起沟通。",
        "蓝衣同学后来跟金子确认。",
        "因为对方是男人和金子，所以不好意思。",
    ]


def test_apply_review_unifies_confirmed_phonetic_entity_only(tmp_path: Path):
    mod = _load_script_module()
    transcript, quality = _write_case(tmp_path)
    out = tmp_path / "review"
    review_json = Path(mod.export_review(transcript, quality_json=quality, out_dir=out)["json"])
    payload = json.loads(review_json.read_text(encoding="utf-8"))
    payload["decisions"][0]["action"] = "unify"
    payload["decisions"][0]["canonical_text"] = "兰艺"
    review_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output = tmp_path / "applied.json"

    result = mod.apply_review(transcript, review_json, out_json=output, write_quality=False)
    data = json.loads(output.read_text(encoding="utf-8"))

    assert result["replacement_count"] == 1
    assert data["segments"][0]["text"] == "兰艺和金子一起沟通。"
    assert data["segments"][1]["text"] == "兰艺同学后来跟金子确认。"
    assert data["segments"][2]["text"] == "因为对方是男人和金子，所以不好意思。"
    assert data["filter_stats"]["entity_consistency_review"]["replacement_count"] == 1


def test_apply_review_rejects_global_unify_for_entity_drift(tmp_path: Path):
    mod = _load_script_module()
    transcript, quality = _write_case(tmp_path)
    out = tmp_path / "review"
    review_json = Path(mod.export_review(transcript, quality_json=quality, out_dir=out)["json"])
    payload = json.loads(review_json.read_text(encoding="utf-8"))
    payload["decisions"][1]["action"] = "unify"
    payload["decisions"][1]["canonical_text"] = "兰艺"
    payload["decisions"][1]["replace_terms"] = ["男人"]
    review_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output = tmp_path / "applied.json"

    result = mod.apply_review(transcript, review_json, out_json=output, write_quality=False)
    data = json.loads(output.read_text(encoding="utf-8"))

    assert result["replacement_count"] == 0
    assert result["skipped"][0]["id"] == "term-consistency-2"
    assert "不允许全局统一" in result["skipped"][0]["reason"]
    assert data["segments"][2]["text"] == "因为对方是男人和金子，所以不好意思。"


def test_apply_review_can_replace_explicit_occurrence_for_entity_drift(tmp_path: Path):
    mod = _load_script_module()
    transcript, quality = _write_case(tmp_path)
    out = tmp_path / "review"
    review_json = Path(mod.export_review(transcript, quality_json=quality, out_dir=out)["json"])
    payload = json.loads(review_json.read_text(encoding="utf-8"))
    payload["decisions"][1]["action"] = "replace_occurrences"
    payload["decisions"][1]["occurrence_replacements"] = [{"index": 2, "from": "男人", "to": "兰艺"}]
    review_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output = tmp_path / "applied.json"

    result = mod.apply_review(transcript, review_json, out_json=output, write_quality=False)
    data = json.loads(output.read_text(encoding="utf-8"))

    assert result["replacement_count"] == 1
    assert data["segments"][2]["text"] == "因为对方是兰艺和金子，所以不好意思。"


def test_cli_export_and_apply(tmp_path: Path):
    transcript, quality = _write_case(tmp_path)
    review_dir = tmp_path / "review"

    export_proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "export",
            str(transcript),
            "--quality-json",
            str(quality),
            "--out",
            str(review_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert export_proc.returncode == 0
    export_result = json.loads(export_proc.stdout)
    review_json = Path(export_result["json"])
    payload = json.loads(review_json.read_text(encoding="utf-8"))
    payload["decisions"][0]["action"] = "unify"
    payload["decisions"][0]["canonical_text"] = "兰艺"
    review_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output = tmp_path / "applied.json"

    apply_proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "apply",
            str(transcript),
            "--review-json",
            str(review_json),
            "--out-json",
            str(output),
            "--no-quality",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert apply_proc.returncode == 0
    apply_result = json.loads(apply_proc.stdout)
    assert apply_result["replacement_count"] == 1
    assert output.exists()
