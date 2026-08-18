import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[2] / "scripts" / "diarization_gold_regression.py"
    spec = importlib.util.spec_from_file_location("diarization_gold_regression", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_best_label_mapping_ignores_arbitrary_speaker_letters():
    module = _load_module()

    mapping = module._best_label_mapping(
        [
            ("C", "A"),
            ("D", "B"),
            ("C->D", "A->B"),
        ]
    )

    assert mapping == {"C": "A", "D": "B"}
    assert module._apply_mapping("C->D->D", mapping) == "A->B"


def test_best_label_mapping_marks_extra_current_speaker_unmapped():
    module = _load_module()

    mapping = module._best_label_mapping([("A->B->C", "A->B->B")])
    mapped = module._apply_mapping("A->B->C", mapping)

    assert "?" in mapped


def test_collapse_sequence_keeps_turns_but_ignores_repeated_cues():
    module = _load_module()

    assert module._collapse_sequence("B->A->A->C->C") == "B->A->C"
    assert module._collapse_sequence("SPEAKER_B B A A") == "B->A"


def test_deduplicate_rows_prefers_latest_completed_annotation():
    module = _load_module()
    common = {
        "audio_sha256": "audio",
        "review_start": 1.0,
        "review_end": 2.0,
        "recording": "测试",
        "id": "DIA-001",
    }
    rows, duplicate_count = module._deduplicate_rows(
        [
            {
                **common,
                "pack_id": "old",
                "annotation_exported_at": "2026-07-01T00:00:00Z",
                "verdict": "correct",
                "gold_turn_sequence": "A",
            },
            {
                **common,
                "pack_id": "new",
                "annotation_exported_at": "2026-07-02T00:00:00Z",
                "verdict": "wrong_speaker",
                "gold_turn_sequence": "B",
            },
        ]
    )

    assert duplicate_count == 1
    assert len(rows) == 1
    assert rows[0]["pack_id"] == "new"


def test_current_error_type_separates_missed_and_extra_turns():
    module = _load_module()

    assert module._current_error_type(
        {"current_turn_correct": False, "current_prediction": "A", "gold_turn_sequence": "A->B"}
    ) == "漏掉换人"
    assert module._current_error_type(
        {"current_turn_correct": False, "current_prediction": "A->B->A", "gold_turn_sequence": "A"}
    ) == "额外误切"
    assert module._current_error_type(
        {"current_turn_correct": False, "current_prediction": "A->B", "gold_turn_sequence": "B->A"}
    ) == "人员或顺序错误"


def test_load_packs_matches_annotation_to_manifest_by_pack_id(tmp_path):
    module = _load_module()
    manifest_root = tmp_path / "output"
    manifest_dir = manifest_root / "pack"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "pack_id": "pack-1",
        "items": [
            {
                "id": "DIA-001",
                "recording": "测试",
                "category": "稳定对照",
                "review_start": 0,
                "review_end": 1,
                "current_prediction": "A",
                "timeline": [{"speaker": "A", "context": False}],
            }
        ],
    }
    (manifest_dir / "通用分人验收清单.json").write_text(
        __import__("json").dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    annotations = {
        "pack_id": "pack-1",
        "items": [
            {
                "id": "DIA-001",
                "verdict": "correct",
                "correct_speaker_sequence": "",
                "notes": "",
            }
        ],
    }
    annotation_path = tmp_path / "annotation.json"
    annotation_path.write_text(
        __import__("json").dumps(annotations, ensure_ascii=False), encoding="utf-8"
    )

    packs = module.load_packs([annotation_path], manifest_root)

    assert len(packs) == 1
    assert packs[0]["pack_id"] == "pack-1"
    assert packs[0]["rows"][0]["gold_sequence"] == "A"
