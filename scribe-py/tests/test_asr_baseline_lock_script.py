from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "asr_baseline_lock.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("asr_baseline_lock", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_transcript(path: Path, *, text: str, md5: str = "placeholder") -> None:
    path.write_text(json.dumps({
        "backend": "sensevoice",
        "model_id": "iic/SenseVoiceSmall",
        "duration": 12.0,
        "transcribe_seconds": 1.2,
        "rtf": 0.1,
        "segments": [{"start": 0.0, "end": 12.0, "text": text}],
        "filter_stats": {
            "audio_standardization": {
                "mode": "adaptive",
                "applied_filters": ["downmix_mono", "resample_16k", "pcm_s16le"],
            },
            "settings": {"sensevoice_timing_align": True},
            "text_normalization": {
                "segments_changed": 0,
                "safe_replacements": 0,
                "first_mention_phonetic_consistency": {"replacement_count": 0},
            },
        },
    }, ensure_ascii=False), encoding="utf-8")


def test_baseline_lock_passes_matching_custom_baseline(tmp_path: Path):
    mod = _load_script()
    transcript = tmp_path / "demo.json"
    _write_transcript(transcript, text="当前转录稳定。")
    data = json.loads(transcript.read_text(encoding="utf-8"))
    compact = mod._compact_text(data["segments"])
    summary = mod._summarize(data, compact)
    mod.DEFAULT_BASELINES["demo"] = {
        "segments": summary["segments"],
        "chars": summary["chars"],
        "md5": summary["md5"],
        "sha256": summary["sha256"],
        "backend": "sensevoice",
        "model_id": "iic/SenseVoiceSmall",
        "preprocess_mode": "adaptive",
        "applied_filters": ["downmix_mono", "resample_16k", "pcm_s16le"],
        "timing_align": True,
    }

    case = mod._load_case(f"demo={transcript}", strict_filters=True)

    assert case.failures == []


def test_baseline_lock_reports_changed_standardized_audio_when_baseline_tracks_it(tmp_path: Path):
    mod = _load_script()
    transcript = tmp_path / "demo.json"
    _write_transcript(transcript, text="同一段转录。")
    data = json.loads(transcript.read_text(encoding="utf-8"))
    data["filter_stats"]["audio_standardization"]["standardized_sha256"] = "current-pcm"
    transcript.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    summary = mod._summarize(data, mod._compact_text(data["segments"]))
    mod.DEFAULT_BASELINES["pcm-demo"] = {
        "segments": summary["segments"],
        "chars": summary["chars"],
        "md5": summary["md5"],
        "sha256": summary["sha256"],
        "backend": "sensevoice",
        "model_id": "iic/SenseVoiceSmall",
        "preprocess_mode": "adaptive",
        "applied_filters": ["downmix_mono", "resample_16k", "pcm_s16le"],
        "timing_align": True,
        "standardized_audio_sha256": "baseline-pcm",
    }

    case = mod._load_case(f"pcm-demo={transcript}", strict_filters=True)

    assert any("standardized_audio_sha256" in failure for failure in case.failures)


def test_default_cases_fall_back_to_latest_versioned_history_directory(
    tmp_path: Path,
    monkeypatch,
):
    mod = _load_script()
    root = tmp_path / "Library" / "Application Support" / "LocalScribe" / "transcripts"
    versioned = root / "标准录音 10-20250709-0311" / "标准录音 10.json"
    versioned.parent.mkdir(parents=True)
    versioned.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mod.Path, "home", classmethod(lambda cls: tmp_path))

    values = mod._default_case_values()

    assert values[1] == f"标准录音 10={versioned}"


def test_cli_fails_when_text_hash_changes(tmp_path: Path):
    transcript = tmp_path / "demo.json"
    _write_transcript(transcript, text="这不是录音三的基线。")
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--case",
            f"标准录音 3={transcript}",
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    report = (out_dir / "asr_baseline_lock.md").read_text(encoding="utf-8")
    assert "FAIL" in report
    assert "md5" in report


def test_cli_accepts_versioned_external_baseline(tmp_path: Path):
    mod = _load_script()
    transcript = tmp_path / "demo.json"
    _write_transcript(transcript, text="当前通用模式已经稳定。")
    data = json.loads(transcript.read_text(encoding="utf-8"))
    summary = mod._summarize(data, mod._compact_text(data["segments"]))
    baseline_file = tmp_path / "baselines.json"
    baseline_file.write_text(
        json.dumps(
            {
                "version": "p0-test",
                "baselines": {
                    "demo": {
                        "aliases": ["demo-alias"],
                        "segments": summary["segments"],
                        "chars": summary["chars"],
                        "md5": summary["md5"],
                        "sha256": summary["sha256"],
                        "backend": summary["backend"],
                        "model_id": summary["model_id"],
                        "preprocess_mode": summary["preprocess_mode"],
                        "applied_filters": summary["applied_filters"],
                        "timing_align": summary["timing_align"],
                        "timing_mode": summary["timing_mode"],
                        "lexical_rewrites_enabled": summary["lexical_rewrites_enabled"],
                        "funasr_inference_seed": summary["funasr_inference_seed"],
                        "textnorm_safe_replacements": summary["textnorm_safe_replacements"],
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline-file",
            str(baseline_file),
            "--case",
            f"demo-alias={transcript}",
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert json.loads(proc.stdout)["ok"] is True
    report = json.loads((out_dir / "asr_baseline_lock.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert "demo" in report["baselines"]
