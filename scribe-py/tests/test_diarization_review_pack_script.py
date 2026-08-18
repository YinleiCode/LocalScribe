from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "diarization_review_pack.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("diarization_review_pack", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _segment(
    start: float,
    end: float,
    speaker: str,
    text: str,
    *,
    confidence: float = 1.0,
    review: bool = False,
    overlap: bool = False,
):
    return {
        "start": start,
        "end": end,
        "speaker": speaker,
        "text": text,
        "speaker_confidence": confidence,
        "speaker_assignment_review": review,
        "speaker_overlap_risk": overlap,
        "speaker_subsegments": [{"start": start, "end": end, "speaker": speaker}],
    }


def test_select_candidates_covers_diverse_categories():
    mod = _load_script()
    segments = [
        _segment(0, 5, "SPEAKER_A", "稳定一"),
        _segment(5, 10, "SPEAKER_A", "稳定二"),
        _segment(10, 15, "SPEAKER_A", "稳定三"),
        _segment(15, 21, "SPEAKER_B", "边界切换", confidence=0.8),
        _segment(21, 22, "SPEAKER_B", "短句", confidence=0.5, overlap=True),
        _segment(22, 28, "SPEAKER_B", "疑似错挂", confidence=0.6, review=True),
        _segment(28, 34, "SPEAKER_B", "第二个稳定说话人"),
        _segment(34, 40, "SPEAKER_A", "独立边界切换"),
    ]
    segments[3].update({
        "sync_cues": [
            {"start": 15, "end": 18, "text": "前半"},
            {"start": 18, "end": 21, "text": "后半"},
        ],
        "speaker_cues": [
            {"cue_index": 0, "start": 15, "end": 18, "speaker": "SPEAKER_A"},
            {"cue_index": 1, "start": 18, "end": 21, "speaker": "SPEAKER_B"},
        ],
        "speaker_change_points": [18],
    })
    data = {"duration": 42, "segments": segments}
    selected = mod.select_candidates(mod.build_candidates(data), limit=5)
    categories = {row["category"] for row in selected}
    assert "段内换人" in categories
    assert "疑似错挂" in categories
    assert "说话人切换" in categories
    assert "重叠或短句" in categories
    assert "稳定对照" in categories


def test_build_candidates_rejects_timestamp_compressed_long_text():
    mod = _load_script()
    segments = [
        _segment(0, 3, "SPEAKER_A", "前一段"),
        _segment(
            3,
            3.3,
            "SPEAKER_B",
            "但是交给我的必须是纸质的好吧",
            confidence=0.4,
            review=True,
        ),
        _segment(3.3, 6, "SPEAKER_A", "后一段"),
    ]

    candidates = mod.build_candidates({"duration": 6, "segments": segments})

    assert all(row["segment_index"] != 1 for row in candidates)
    assert all(row["segment_index"] != 2 for row in candidates)


def test_build_candidates_keeps_short_meaningful_interjection():
    mod = _load_script()
    segments = [
        _segment(0, 2, "SPEAKER_A", "前一段"),
        _segment(2, 2.15, "SPEAKER_B", "对", confidence=0.5),
        _segment(2.15, 4, "SPEAKER_A", "后一段"),
    ]

    candidates = mod.build_candidates({"duration": 4, "segments": segments})

    assert any(
        row["segment_index"] == 1 and row["category"] == "重叠或短句"
        for row in candidates
    )


def test_select_candidates_covers_each_predicted_speaker():
    mod = _load_script()
    segments = [
        _segment(0, 5, "SPEAKER_A", "甲发言", confidence=0.5, review=True),
        _segment(6, 11, "SPEAKER_B", "乙发言", confidence=0.5, review=True),
        _segment(12, 17, "SPEAKER_C", "丙发言", confidence=0.5, review=True),
        _segment(18, 23, "SPEAKER_D", "丁发言", confidence=0.5, review=True),
    ]

    selected = mod.select_candidates(
        mod.build_candidates({"duration": 24, "segments": segments}),
        limit=4,
    )

    assert {
        speaker
        for row in selected
        for speaker in row["speakers"]
    } == {"SPEAKER_A", "SPEAKER_B", "SPEAKER_C", "SPEAKER_D"}


def test_build_review_items_keeps_transcript_read_only(tmp_path: Path):
    mod = _load_script()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF" + b"0" * 100)
    transcript_path = tmp_path / "meeting.json"
    data = {
        "audio": str(audio),
        "duration": 20,
        "segments": [
            _segment(0, 6, "SPEAKER_A", "第一段"),
            _segment(6, 12, "SPEAKER_B", "第二段", confidence=0.6, review=True),
            _segment(12, 18, "SPEAKER_B", "第三段"),
        ],
    }
    transcript_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    before = transcript_path.read_bytes()
    items = mod.build_review_items(
        [("会议", transcript_path, data)],
        per_case=3,
        padding_seconds=1.5,
    )
    assert items
    assert transcript_path.read_bytes() == before
    assert all(item["recording"] == "会议" for item in items)
    assert all(item["current_prediction"] for item in items)


def test_build_review_items_shows_padded_context_without_changing_prediction(tmp_path: Path):
    mod = _load_script()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF" + b"0" * 100)
    transcript_path = tmp_path / "meeting.json"
    data = {
        "audio": str(audio),
        "duration": 12,
        "segments": [
            _segment(0, 4, "SPEAKER_A", "前文"),
            _segment(4, 8, "SPEAKER_B", "复核中心", confidence=0.5, review=True),
            _segment(8, 12, "SPEAKER_B", "后文"),
        ],
    }

    items = mod.build_review_items(
        [("会议", transcript_path, data)],
        per_case=1,
        padding_seconds=1.5,
    )

    assert items[0]["current_prediction"] == "B"
    assert [row["speaker"] for row in items[0]["timeline"]] == ["A", "B", "B"]
    assert [row["context"] for row in items[0]["timeline"]] == [True, False, True]


def test_unreliable_timing_is_excluded_and_cannot_build_clips(tmp_path: Path):
    mod = _load_script()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF" + b"0" * 100)
    transcript_path = tmp_path / "meeting.json"
    data = {
        "audio": str(audio),
        "duration": 12,
        "filter_stats": {
            "timing_reliable": False,
            "timing_alignment_reason": "source_anchor_text_too_different",
        },
        "segments": [
            _segment(0, 6, "SPEAKER_A", "第一段"),
            _segment(6, 12, "SPEAKER_B", "第二段", confidence=0.5, review=True),
        ],
    }

    eligible, excluded = mod.partition_review_cases([("会议", transcript_path, data)])

    assert eligible == []
    assert excluded == [{
        "recording": "会议",
        "transcript": str(transcript_path.resolve()),
        "reason": "source_anchor_text_too_different",
    }]
    try:
        mod.build_review_items(
            [("会议", transcript_path, data)],
            per_case=1,
            padding_seconds=1.5,
        )
    except ValueError as exc:
        assert "source_anchor_text_too_different" in str(exc)
    else:
        raise AssertionError("explicitly unreliable timing must not produce clips")


def test_write_dry_run_renders_local_chinese_review_page(tmp_path: Path):
    mod = _load_script()
    items = [{
        "id": "DIA-001",
        "recording": "测试会议",
        "transcript": "/tmp/test.json",
        "audio": "/tmp/test.wav",
        "category": "说话人切换",
        "score": 80,
        "review_start": 10,
        "review_end": 15,
        "segment_index": 2,
        "reason": "A->B",
        "clip_start": 8.5,
        "clip_end": 16.5,
        "current_prediction": "A->B",
        "timeline": [{"start": 10, "end": 15, "speaker": "B", "text": "你好", "source": "segment"}],
        "verdict": "",
        "correct_speaker_sequence": "",
        "notes": "",
    }]
    manifest_path, html_path = mod.write_pack(
        items,
        out_dir=tmp_path / "pack",
        dry_run=True,
        excluded_cases=[{"recording": "错位录音", "reason": "时间轴不可靠"}],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    page = html_path.read_text(encoding="utf-8")
    assert manifest["item_count"] == 1
    assert manifest["category_counts"]["说话人切换"] == 1
    assert "通用分人验收" in page
    assert "导出标注" in page
    assert "localStorage" in page
    assert "测试会议" in page
    assert manifest["excluded_recordings"] == [{"recording": "错位录音", "reason": "时间轴不可靠"}]
    assert "已排除时间轴不可靠的录音" in page
    assert "错位录音" in page
