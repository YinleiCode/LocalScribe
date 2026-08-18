from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "asr_cursor_freeze_gate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("asr_cursor_freeze_gate", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_transcript(path: Path, segments: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"segments": segments}, ensure_ascii=False), encoding="utf-8")


def _segment(start: float, end: float, text: str, cues: list[tuple[float, float, str]]) -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "text": text,
        "sync_cues": [
            {"start": cue_start, "end": cue_end, "text": cue_text}
            for cue_start, cue_end, cue_text in cues
        ],
    }


def _failure_codes(result: dict[str, Any]) -> set[str]:
    return {failure["code"] for failure in result["failures"]}


def test_gate_passes_same_text_with_changed_valid_cursor_geometry(tmp_path: Path):
    mod = _load_script()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_transcript(baseline, [_segment(0.0, 2.0, "第一句。第二句。", [(0.0, 1.0, "第一句。"), (1.0, 2.0, "第二句。")])])
    _write_transcript(candidate, [_segment(0.1, 2.1, "第一句。第二句。", [(0.1, 1.2, "第一句。"), (1.2, 2.1, "第二句。")])])

    result = mod.evaluate_gate(baseline, candidate)

    assert result["ok"] is True
    assert result["baseline"]["text_sha256"] == result["candidate"]["text_sha256"]
    assert result["failures"] == []


def test_strict_segment_geometry_rejects_retiming(tmp_path: Path):
    mod = _load_script()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_transcript(baseline, [_segment(0.0, 2.0, "同一段文字。", [(0.0, 2.0, "同一段文字。")])])
    _write_transcript(candidate, [_segment(0.1, 2.1, "同一段文字。", [(0.1, 2.1, "同一段文字。")])])

    relaxed = mod.evaluate_gate(baseline, candidate)
    strict = mod.evaluate_gate(baseline, candidate, require_segment_geometry=True)

    assert relaxed["ok"] is True
    assert strict["ok"] is False
    assert "segment_geometry_changed" in _failure_codes(strict)


def test_strict_segment_geometry_rejects_sync_cue_retiming(tmp_path: Path):
    mod = _load_script()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_transcript(
        baseline,
        [_segment(0.0, 2.0, "第一句。第二句。", [(0.0, 1.0, "第一句。"), (1.0, 2.0, "第二句。")])],
    )
    _write_transcript(
        candidate,
        [_segment(0.0, 2.0, "第一句。第二句。", [(0.0, 1.2, "第一句。"), (1.2, 2.0, "第二句。")])],
    )

    relaxed = mod.evaluate_gate(baseline, candidate)
    strict = mod.evaluate_gate(baseline, candidate, require_segment_geometry=True)

    assert relaxed["ok"] is True
    assert strict["ok"] is False
    assert "segment_geometry_changed" in _failure_codes(strict)


def test_normalized_text_equality_must_be_explicitly_enabled(tmp_path: Path):
    mod = _load_script()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_transcript(baseline, [_segment(0.0, 1.0, "Ａ B", [(0.0, 1.0, "Ａ B")])])
    _write_transcript(candidate, [_segment(0.0, 1.0, "AB", [(0.0, 1.0, "AB")])])

    exact = mod.evaluate_gate(baseline, candidate, text_mode="exact")
    normalized = mod.evaluate_gate(baseline, candidate, text_mode="normalized")

    assert exact["ok"] is False
    assert "transcript_text_changed" in _failure_codes(exact)
    assert normalized["ok"] is True
    assert normalized["compared_hash"] == "normalized_text_sha256"


def test_invalid_baseline_cursor_is_diagnostic_and_does_not_block_repair(tmp_path: Path):
    mod = _load_script()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_transcript(baseline, [_segment(0.0, 1.0, "需要修复。", [(0.0, 0.0, "错误。")])])
    _write_transcript(candidate, [_segment(0.0, 1.0, "需要修复。", [(0.0, 1.0, "需要修复。")])])

    result = mod.evaluate_gate(baseline, candidate)

    assert result["ok"] is True
    assert {failure["code"] for failure in result["baseline_diagnostics"]} == {
        "cue_text_mismatch",
        "cue_zero_duration",
    }
    assert result["failures"] == []


def test_identical_baseline_diagnostics_are_inherited_not_regressions(tmp_path: Path):
    mod = _load_script()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    inherited_segment = _segment(0.0, 1.0, "完整文字。", [(0.0, 1.0, "旧错误。")])
    _write_transcript(baseline, [inherited_segment])
    _write_transcript(candidate, [inherited_segment])

    result = mod.evaluate_gate(baseline, candidate, require_segment_geometry=True)

    assert result["ok"] is True
    assert result["failures"] == []
    assert {item["code"] for item in result["candidate_diagnostics"]} == {"cue_text_mismatch"}
    assert result["inherited_diagnostics"] == result["candidate_diagnostics"]


def test_new_diagnostic_at_another_location_is_a_regression(tmp_path: Path):
    mod = _load_script()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    good = _segment(0.0, 1.0, "前句。", [(0.0, 1.0, "前句。")])
    inherited_bad = _segment(1.0, 2.0, "后句。", [(1.0, 2.0, "旧错误。")])
    newly_bad = _segment(0.0, 1.0, "前句。", [(0.0, 1.0, "新错误。")])
    _write_transcript(baseline, [good, inherited_bad])
    _write_transcript(candidate, [newly_bad, inherited_bad])

    result = mod.evaluate_gate(baseline, candidate)

    assert result["ok"] is False
    assert "candidate_cue_text_mismatch" in _failure_codes(result)
    assert result["failures"][0]["location"] == "segments[0]"
    assert result["inherited_diagnostics"][0]["location"] == "segments[1]"


def test_gate_rejects_cue_text_mismatch_and_zero_duration_cue(tmp_path: Path):
    mod = _load_script()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_transcript(baseline, [_segment(0.0, 1.0, "完整文字。", [(0.0, 1.0, "完整文字。")])])
    _write_transcript(candidate, [_segment(0.0, 1.0, "完整文字。", [(0.0, 0.0, "错误文字。")])])

    result = mod.evaluate_gate(baseline, candidate)
    codes = _failure_codes(result)

    assert result["ok"] is False
    assert "candidate_cue_text_mismatch" in codes
    assert "candidate_cue_zero_duration" in codes
    assert result["candidate"]["cue_text_mismatches"] == 1
    assert result["candidate"]["zero_duration_cues"] == 1


def test_gate_rejects_overlapping_cues(tmp_path: Path):
    mod = _load_script()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_transcript(
        baseline,
        [_segment(0.0, 2.0, "前句。后句。", [(0.0, 1.0, "前句。"), (1.0, 2.0, "后句。")] )],
    )
    _write_transcript(
        candidate,
        [_segment(0.0, 2.0, "前句。后句。", [(0.0, 1.5, "前句。"), (1.0, 2.0, "后句。")] )],
    )

    result = mod.evaluate_gate(baseline, candidate)

    assert result["ok"] is False
    assert "candidate_cue_overlap" in _failure_codes(result)
    assert result["candidate"]["overlapping_cues"] == 1


def test_gate_rejects_non_monotonic_segment_and_cue_times(tmp_path: Path):
    mod = _load_script()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    good_segments = [
        _segment(0.0, 1.0, "前句。", [(0.0, 1.0, "前句。")]),
        _segment(1.0, 2.0, "后句。", [(1.0, 2.0, "后句。")]),
    ]
    bad_segments = [
        _segment(1.0, 2.0, "前句。", [(1.0, 2.0, "前句。")]),
        _segment(0.5, 1.5, "后句。", [(0.5, 1.5, "后句。")]),
    ]
    _write_transcript(baseline, good_segments)
    _write_transcript(candidate, bad_segments)

    result = mod.evaluate_gate(baseline, candidate)
    codes = _failure_codes(result)

    assert "candidate_segment_start_non_monotonic" in codes
    assert "candidate_segment_end_non_monotonic" in codes
    assert "candidate_cue_start_non_monotonic" in codes
    assert "candidate_cue_end_non_monotonic" in codes


def test_short_cue_limits_default_to_baseline_and_allow_explicit_override(tmp_path: Path):
    mod = _load_script()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_transcript(
        baseline,
        [_segment(0.0, 1.0, "甲乙丙", [(0.0, 0.1, "甲"), (0.1, 1.0, "乙丙")])],
    )
    _write_transcript(
        candidate,
        [_segment(0.0, 1.0, "甲乙丙", [(0.0, 0.1, "甲"), (0.1, 0.2, "乙"), (0.2, 1.0, "丙")])],
    )

    default_result = mod.evaluate_gate(baseline, candidate, short_cue_threshold_ms=150.0)
    overridden = mod.evaluate_gate(
        baseline,
        candidate,
        short_cue_threshold_ms=150.0,
        max_short_cues=2,
        max_short_cue_ratio=0.67,
    )

    assert default_result["ok"] is False
    assert "short_cue_count_exceeded" in _failure_codes(default_result)
    assert "short_cue_ratio_exceeded" in _failure_codes(default_result)
    assert overridden["ok"] is True
    assert overridden["limits"]["short_cue_limits_source"] == {"count": "explicit", "ratio": "explicit"}


def test_cli_returns_nonzero_and_machine_readable_failure(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_transcript(baseline, [_segment(0.0, 1.0, "基线。", [(0.0, 1.0, "基线。")])])
    _write_transcript(candidate, [_segment(0.0, 1.0, "变化。", [(0.0, 1.0, "变化。")])])

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(baseline), str(candidate)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert "transcript_text_changed" in _failure_codes(payload)
