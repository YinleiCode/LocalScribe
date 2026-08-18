from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "diarization_interval_projection.py"


def _module():
    spec = importlib.util.spec_from_file_location("diarization_interval_projection", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_project_speaker_uses_total_overlap_per_speaker() -> None:
    module = _module()
    intervals = [
        (0.0, 0.4, "A"),
        (0.4, 0.7, "B"),
        (0.7, 1.0, "B"),
    ]

    speaker, overlap, distance = module.project_speaker(0.0, 1.0, intervals)

    assert speaker == "B"
    assert abs(overlap - 0.6) < 1e-9
    assert distance == 0.0


def test_project_speaker_uses_nearest_interval_when_vad_has_gap() -> None:
    module = _module()
    intervals = [(0.0, 1.0, "A"), (3.0, 4.0, "B")]

    speaker, overlap, distance = module.project_speaker(1.1, 1.3, intervals)

    assert speaker == "A"
    assert overlap == 0.0
    assert abs(distance - 0.2) < 1e-9
