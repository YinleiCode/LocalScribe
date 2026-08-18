from __future__ import annotations

import importlib.util
import json
import shutil
import wave
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "diarization_holdout_inventory.py"
SPEC = importlib.util.spec_from_file_location("diarization_holdout_inventory", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_wav(path: Path, *, sample: int, seconds: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(int(sample).to_bytes(2, "little", signed=True) * 16_000 * seconds)


def test_inventory_excludes_renamed_historical_audio_by_content_hash(tmp_path: Path):
    candidates = tmp_path / "candidates"
    history = tmp_path / "history"
    used_source = history / "source.wav"
    used_copy = candidates / "renamed.wav"
    unseen = candidates / "unseen.wav"
    _write_wav(used_source, sample=10)
    used_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(used_source, used_copy)
    _write_wav(unseen, sample=20)
    (history / "manifest.json").write_text(
        json.dumps({"audio": str(used_source)}),
        encoding="utf-8",
    )

    report = MODULE.build_inventory(
        candidate_roots=[candidates],
        history_roots=[history],
        min_bytes=1,
        min_duration_seconds=0,
        max_duration_seconds=10,
        select_count=5,
        ffprobe=None,
    )

    assert report["summary"]["candidate_unique_hashes"] == 2
    assert report["summary"]["excluded_unique_audio"] == 1
    assert [Path(row["path"]).name for row in report["eligible"]] == ["unseen.wav"]
    assert report["excluded"][0]["exclusion_reason"] == "historical_content_hash"
    assert report["selection_policy"]["uses_transcript_text"] is False


def test_inventory_excludes_exact_path_referenced_by_history(tmp_path: Path):
    candidates = tmp_path / "candidates"
    history = tmp_path / "history"
    audio = candidates / "referenced.wav"
    _write_wav(audio, sample=30)
    history.mkdir()
    (history / "manifest.json").write_text(
        json.dumps({"nested": [{"audio": str(audio)}]}),
        encoding="utf-8",
    )

    report = MODULE.build_inventory(
        candidate_roots=[candidates],
        history_roots=[history],
        min_bytes=1,
        min_duration_seconds=0,
        max_duration_seconds=10,
        select_count=1,
        ffprobe=None,
    )

    assert report["summary"]["eligible_unique_audio"] == 0
    assert report["excluded"][0]["exclusion_reason"] == "historical_path_reference"
