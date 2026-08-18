from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "asr_local_recheck.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("asr_local_recheck", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recheck_reads_review_segments_from_filter_stats():
    mod = _load_script_module()
    data = {
        "filter_stats": {
            "text_normalization": {
                "asr_review_segments": [
                    {
                        "index": 1,
                        "start": 1.0,
                        "end": 2.0,
                        "text": "有点矫正嗯。",
                        "reasons": ["命中已知 ASR 易混淆词"],
                    }
                ]
            }
        }
    }

    review, selection = mod._review_segments(data)

    assert len(review) == 1
    assert review[0]["index"] == 1
    assert selection["strong_segment_count"] == 1


def test_recheck_defaults_to_strong_segments_and_skips_weak_spot_checks():
    mod = _load_script_module()
    data = {
        "filter_stats": {
            "text_normalization": {
                "asr_review_segments": [
                    {
                        "index": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "text": "这个这个可以。",
                        "reasons": ["疑似重复词"],
                    },
                    {
                        "index": 1,
                        "start": 1.0,
                        "end": 2.0,
                        "text": "管有关个地。",
                        "reasons": ["命中明显不通顺 ASR 片段"],
                    },
                ]
            }
        }
    }

    review, selection = mod._review_segments(data)

    assert [item["index"] for item in review] == [1]
    assert selection["total_segment_count"] == 2
    assert selection["skipped_weak_count"] == 1


def test_recheck_can_include_all_segments_when_requested():
    mod = _load_script_module()
    data = {
        "asr_quality": {
            "review": {
                "segments": [
                    {"index": 0, "start": 0.0, "end": 1.0, "text": "。", "reasons": ["只有标点/空白"]},
                    {"index": 1, "start": 1.0, "end": 2.0, "text": "不辱我。", "reasons": ["疑似明显语义不顺"]},
                ]
            }
        }
    }

    review, selection = mod._review_segments(data, scope="all")

    assert [item["index"] for item in review] == [0, 1]
    assert selection["skipped_weak_count"] == 0


def test_recheck_report_is_chinese_and_non_destructive(tmp_path: Path):
    mod = _load_script_module()
    item = mod.RecheckItem(
        index=1,
        start=1.0,
        end=2.0,
        clip_start=0.0,
        clip_end=3.0,
        original_text="有点矫正嗯。",
        current_text="有点矫正嗯。",
        reasons=["命中已知 ASR 易混淆词"],
        candidate_text="有点矫搅啊。",
        candidate_segments=1,
        backend="funasr",
        model="paraformer-zh",
        clip_path=tmp_path / "clip.wav",
    )

    report = mod._render_report(
        [item],
        tmp_path / "input.json",
        tmp_path / "audio.mp3",
        {
            "scope": "strong",
            "total_segment_count": 2,
            "strong_segment_count": 1,
            "weak_segment_count": 1,
            "skipped_weak_count": 1,
        },
    )

    assert "# ASR 本地疑点复核" in report
    assert "强疑点数: 1" in report
    assert "跳过弱抽查数: 1" in report
    assert "候选文本只供参考" in report
    assert "作为本地复听证据之一" in report
    assert "不会自动替换原转录" in report
    assert "复核范围" in report
    assert "有点矫搅啊" in report


def test_recheck_report_can_render_multiple_local_candidates(tmp_path: Path):
    mod = _load_script_module()
    item = mod.RecheckItem(
        index=1,
        start=1.0,
        end=2.0,
        clip_start=0.0,
        clip_end=3.0,
        original_text="有点矫正嗯。",
        current_text="有点矫正嗯。",
        reasons=["命中已知 ASR 易混淆词"],
        candidate_text="有点搅扰。",
        candidate_segments=1,
        backend="funasr",
        model="paraformer-zh",
        clip_path=tmp_path / "clip.wav",
        candidates=[
            mod.RecheckCandidate("funasr", "paraformer-zh", "有点搅扰。", 1),
            mod.RecheckCandidate("qwen3", "qwen3-asr", "有点搅扰。", 1),
            mod.RecheckCandidate("sensevoice", "iic/SenseVoiceSmall", "有点矫正。", 1),
        ],
    )

    report = mod._render_report(
        [item],
        tmp_path / "input.json",
        tmp_path / "audio.mp3",
        {"scope": "strong", "total_segment_count": 1, "strong_segment_count": 1, "weak_segment_count": 0, "skipped_weak_count": 0},
    )

    assert "funasr/paraformer-zh: 有点搅扰。" in report
    assert "qwen3/qwen3-asr: 有点搅扰。" in report
    assert "sensevoice/iic/SenseVoiceSmall: 有点矫正。" in report
    assert "Primary / Current" in report
    assert "current↔paraformer" in report
    assert "审计建议" in report


def test_recheck_parse_compare_backends_and_model_overrides():
    mod = _load_script_module()
    backends = mod._selected_backends("funasr", "funasr,qwen3,sensevoice,funasr")
    overrides = mod._model_overrides(
        "funasr=paraformer-zh,qwen3=mlx-community/Qwen3-ASR-1.7B-8bit,sensevoice=iic/SenseVoiceSmall",
        backends,
    )

    assert backends == ["funasr", "qwen3", "sensevoice"]
    assert overrides == {
        "funasr": "paraformer-zh",
        "qwen3": "mlx-community/Qwen3-ASR-1.7B-8bit",
        "sensevoice": "iic/SenseVoiceSmall",
    }


def test_severe_low_similarity_is_audit_only_and_never_allows_replacement():
    mod = _load_script_module()
    candidates = [
        mod.RecheckCandidate("funasr", "paraformer-zh", "今天需要检查数据库缓存。", 1),
        mod.RecheckCandidate("qwen3", "qwen3-asr", "明天下午讨论客户合同。", 1),
    ]

    audit = mod._build_audit("大家现在开始会议。", candidates)

    assert audit["status"] == "severe_low_similarity"
    assert audit["severe_low_similarity"] is True
    assert audit["auto_replace_allowed"] is False
    assert audit["severe_low_similarity_threshold"] == 0.45
    assert audit["primary_current"] == "大家现在开始会议。"
    assert audit["paraformer_candidate"] == "今天需要检查数据库缓存。"
    assert audit["qwen3_candidate"] == "明天下午讨论客户合同。"
    assert "禁止" in audit["recommendation"]


def test_three_way_audit_reports_all_pairwise_similarities():
    mod = _load_script_module()
    candidates = [
        mod.RecheckCandidate("funasr", "paraformer-zh", "我们需要读写分离。", 1),
        mod.RecheckCandidate("qwen3", "qwen3-asr", "我们需要做读写分离。", 1),
    ]

    audit = mod._build_audit("我们需要做读写分离。", candidates)

    assert set(audit["similarities"]) == {
        "primary_paraformer",
        "primary_qwen3",
        "paraformer_qwen3",
    }
    assert audit["similarities"]["primary_qwen3"] == 1.0
    assert audit["auto_replace_allowed"] is False


def test_dry_run_with_qwen3_writes_audit_without_changing_source(
    monkeypatch, tmp_path: Path
):
    mod = _load_script_module()
    transcript = tmp_path / "input.json"
    payload = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "管有关个地。"}],
        "filter_stats": {
            "text_normalization": {
                "asr_review_segments": [
                    {
                        "index": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "text": "管有关个地。",
                        "reasons": ["命中明显不通顺 ASR 片段"],
                    }
                ]
            }
        },
    }
    transcript.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    source_before = transcript.read_bytes()
    out_dir = tmp_path / "review"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            str(transcript),
            "--dry-run",
            "--compare-backends",
            "funasr,qwen3",
            "--out",
            str(out_dir),
        ],
    )

    assert mod.main() == 0

    assert transcript.read_bytes() == source_before
    report = json.loads((out_dir / "ASR本地疑点复核.json").read_text(encoding="utf-8"))
    assert report["mode"] == "audit_only"
    assert report["auto_replace_allowed"] is False
    assert [item["backend"] for item in report["items"][0]["candidates"]] == [
        "funasr",
        "qwen3",
    ]
    assert report["items"][0]["primary_current"] == "管有关个地。"
    assert report["items"][0]["paraformer_candidate"]["backend"] == "funasr"
    assert report["items"][0]["qwen3_candidate"]["backend"] == "qwen3"
    assert report["items"][0]["audit"]["auto_replace_allowed"] is False


def test_extract_clip_timeout_does_not_hang(monkeypatch, tmp_path: Path):
    mod = _load_script_module()

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args", "ffmpeg"), timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    try:
        mod._extract_clip(tmp_path / "audio.mp3", tmp_path / "clip.wav", 0, 1, timeout=0.01)
    except RuntimeError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("expected timeout RuntimeError")


def test_source_audio_resolves_relative_path_next_to_transcript(tmp_path: Path):
    mod = _load_script_module()
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"fake")
    transcript = tmp_path / "case.json"
    transcript.write_text("{}", encoding="utf-8")

    resolved = mod._source_audio("", {"audio": "audio.mp3"}, transcript)

    assert resolved == audio.resolve()


def test_source_audio_falls_back_to_same_name_next_to_transcript(tmp_path: Path):
    mod = _load_script_module()
    audio = tmp_path / "source.mp3"
    audio.write_bytes(b"fake")
    transcript = tmp_path / "case.json"
    transcript.write_text("{}", encoding="utf-8")

    resolved = mod._source_audio("", {"audio": "/missing/source.mp3"}, transcript)

    assert resolved == audio.resolve()


def test_source_audio_error_tells_user_to_pass_audio(tmp_path: Path):
    mod = _load_script_module()
    transcript = tmp_path / "case.json"
    transcript.write_text("{}", encoding="utf-8")

    try:
        mod._source_audio("", {"audio": "missing.mp3"}, transcript)
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("expected SystemExit")

    assert "audio not found from transcript JSON" in message
    assert "--audio" in message


def test_limit_zero_means_all_selected_items():
    mod = _load_script_module()
    review = [{"index": 1}, {"index": 2}]

    assert mod._selected_review_items(review, 0) == review
    assert mod._selected_review_items(review, 1) == [{"index": 1}]


def test_default_fixture_selects_22_strong_and_skips_17_weak():
    mod = _load_script_module()
    json_path = mod.DEFAULT_JSON
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))

    review, selection = mod._review_segments(data, transcript_json=json_path)

    assert len(review) == 22
    assert selection["strong_segment_count"] == 22
    assert selection["weak_segment_count"] == 17
    assert selection["skipped_weak_count"] == 17
