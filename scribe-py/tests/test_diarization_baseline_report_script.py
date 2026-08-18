from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "diarization_baseline_report.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("diarization_baseline_report", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_segment_agreement_aligns_labels_by_confusion():
    mod = _load_script()

    left = [
        {"speaker": "SPEAKER_A"},
        {"speaker": "SPEAKER_A"},
        {"speaker": "SPEAKER_B"},
        {"speaker": "SPEAKER_B"},
    ]
    right = [
        {"speaker": "spk_2"},
        {"speaker": "spk_2"},
        {"speaker": "spk_1"},
        {"speaker": "spk_1"},
    ]

    assert mod._segment_agreement(left, right) == 1.0


def test_proxy_metrics_flags_fragmented_speaker_jumps():
    mod = _load_script()

    segments = [
        {"start": 0.0, "end": 0.3, "speaker": "SPEAKER_A"},
        {"start": 0.3, "end": 0.6, "speaker": "SPEAKER_B"},
        {"start": 0.6, "end": 0.9, "speaker": "SPEAKER_A"},
        {"start": 0.9, "end": 1.2, "speaker": "SPEAKER_B"},
    ]

    metrics = mod._proxy_metrics(segments)
    assert metrics["proxy_risk"] == "high"
    assert metrics["fragment_segments"] == 4
    assert "碎片段过多" in metrics["risk_notes"]


def test_cli_writes_chinese_report_from_baseline_json(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "case": "demo",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "你好", "speaker": "SPEAKER_A"},
            {"start": 2.0, "end": 4.0, "text": "您好", "speaker": "SPEAKER_B"},
        ],
        "stats": {"engine": "pyannote"},
    }, ensure_ascii=False), encoding="utf-8")
    out_dir = tmp_path / "report"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline-json",
            f"pyannote={baseline}",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    md = Path(payload["markdown"]).read_text(encoding="utf-8")
    assert "说话人分离基线对比报告" in md
    assert "未提供人工 gold" in md
    assert "pyannote" in md
    assert "代理风险" in md
    assert (out_dir / "diarization_baseline_report.tsv").exists()
    assert (out_dir / "diarization_baseline_report.json").exists()
    assert (out_dir / "diarization_disagreements.tsv").exists()


def test_cli_disagreement_file_shows_mapped_speaker_label(tmp_path: Path):
    base = tmp_path / "base.json"
    alt = tmp_path / "alt.json"
    base.write_text(json.dumps({
        "case": "demo",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "你好", "speaker": "SPEAKER_A"},
            {"start": 2.0, "end": 4.0, "text": "您好", "speaker": "SPEAKER_B"},
        ],
        "stats": {"engine": "base"},
    }, ensure_ascii=False), encoding="utf-8")
    alt.write_text(json.dumps({
        "case": "demo",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "你好", "speaker": "spk_1"},
            {"start": 2.0, "end": 4.0, "text": "您好", "speaker": "spk_1"},
        ],
        "stats": {"engine": "alt"},
    }, ensure_ascii=False), encoding="utf-8")
    out_dir = tmp_path / "report"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline-json",
            f"base={base}",
            "--baseline-json",
            f"alt={alt}",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    disagreements = (out_dir / "diarization_disagreements.tsv").read_text(encoding="utf-8")
    assert "当前映射后" in disagreements
    assert "说话人不一致" in disagreements


def test_prediction_artifact_preserves_source_audio_and_transcript(tmp_path: Path):
    mod = _load_script()
    result = mod.EngineResult(
        case="独立盲测",
        engine="app-senko",
        status="ok",
        segments=[{"start": 0.0, "end": 1.0, "text": "你好", "speaker": "SPEAKER_A"}],
        stats={"engine": "senko"},
        audio="/tmp/audio.mp3",
        transcript="/tmp/transcript.json",
    )

    paths = mod._write_predictions(tmp_path, [result])
    payload = json.loads(Path(paths[0]).read_text(encoding="utf-8"))

    assert payload["audio"] == "/tmp/audio.mp3"
    assert payload["transcript"] == "/tmp/transcript.json"
