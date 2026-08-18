from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "diarization_apply_annotations.py"


def test_apply_annotations_script_writes_calibrated_copy(tmp_path: Path):
    transcript = tmp_path / "demo.json"
    transcript.write_text(json.dumps({
        "duration": 4.0,
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "你好", "speaker": "SPEAKER_A"},
            {"start": 2.0, "end": 4.0, "text": "您好", "speaker": "SPEAKER_B"},
        ],
        "diarization_stats": {"engine": "test"},
    }, ensure_ascii=False), encoding="utf-8")
    annotations = tmp_path / "ann.json"
    annotations.write_text(json.dumps([
        {"序号": 0, "你的标注": "A"},
        {"序号": 1, "你的标注": "A"},
    ], ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--transcript",
            str(transcript),
            "--annotations",
            str(annotations),
            "--out-dir",
            str(tmp_path / "out"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["annotation_count"] == 2
    assert payload["changed"] == 1
    assert payload["accuracy_before"] == 0.5
    calibrated = json.loads(Path(payload["json"]).read_text(encoding="utf-8"))
    assert calibrated["segments"][1]["speaker"] == "SPEAKER_A"
    assert calibrated["segments"][1]["original_speaker"] == "SPEAKER_B"
    assert calibrated["segments"][1]["speaker_calibrated"] is True
    report = Path(payload["report"]).read_text(encoding="utf-8")
    assert "说话人标注校准报告" in report
