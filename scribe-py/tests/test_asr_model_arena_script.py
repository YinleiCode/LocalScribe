from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
import wave
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "asr_model_arena.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("asr_model_arena", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_wav(path: Path, duration: float = 1.0, sample_rate: int = 8000) -> None:
    frames = int(duration * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


def _case_result(
    mod,
    *,
    case_id: str = "case-1",
    text: str = "你好",
    duration: float = 2.0,
    inference: float = 1.0,
    load: float = 0.0,
    status: str = "ok",
    error: str = "",
    peak_rss: float | None = 128.0,
):
    return {
        "case_id": case_id,
        "clip_path": f"/{case_id}.wav",
        "status": status,
        "raw_text": text,
        "display_text": text,
        "normalized_text": mod.normalize_text(text),
        "segments": [{"start": 0.0, "end": duration, "text": text}],
        "timing_reliable": False,
        "alignment": dict(mod.ALIGNMENT_NOT_RUN),
        "load_seconds": load,
        "inference_seconds": inference,
        "audio_duration": duration,
        "peak_rss_mb": peak_rss,
        "accelerator_memory": {
            "provider": "mlx",
            "available": True,
            "peak_mb": 256.0,
            "raw_bytes": 256 * 1024 * 1024,
        },
        "error": error,
    }


def _round(
    mod,
    *,
    variant: str,
    repeat: int,
    cases: list[dict],
    status: str = "ok",
    load: float = 0.5,
    revision: str = "test-revision",
    load_includes_download: bool = False,
):
    return {
        "schema_version": mod.SCHEMA_VERSION,
        "kind": "arena_worker_round",
        "variant": variant,
        "repeat": repeat,
        "required": True,
        "adapter": "localscribe",
        "model_id": "fake-model",
        "revision": revision,
        "status": status,
        "load_seconds": load,
        "load_includes_download": load_includes_download,
        "started_at": "2026-07-16T00:00:00Z",
        "completed_at": "2026-07-16T00:00:01Z",
        "cases": cases,
        "error": "" if status == "ok" else "failed",
    }


def test_normalization_nfkc_opencc_lowercase_and_ignores_punctuation_space():
    mod = _load_script_module()

    class FakeOpenCC:
        def convert(self, value: str) -> str:
            return value.replace("臺", "台").replace("灣", "湾")

    assert mod.normalize_text(" ＡＢＣ， 臺灣！\n", converter=FakeOpenCC()) == "abc台湾"
    assert mod.normalize_text("Hello—WORLD...", converter=None) == "helloworld"


def test_truth_text_prefers_correct_text_over_current_text():
    mod = _load_script_module()

    assert mod._truth_text({"current_text": "当前文本", "correct_text": "人工正确文本"}) == "人工正确文本"


def test_truth_text_rejects_empty_correct_text_and_never_uses_current_text():
    mod = _load_script_module()

    with pytest.raises(ValueError, match="correct_text must be non-empty"):
        mod._truth_text({"correct_text": "  ", "gold_text": "不应回退"})
    with pytest.raises(ValueError, match="non-empty"):
        mod._truth_text({"current_text": "不能猜作真值"})


def test_truth_text_uses_only_nonempty_documented_fallbacks_in_order():
    mod = _load_script_module()

    text, source = mod._truth_text_and_source(
        {
            "gold_text": "",
            "reference_text": "人工参考",
            "text": "较低优先级",
            "transcript": "更低优先级",
            "current_text": "禁止使用",
        }
    )

    assert text == "人工参考"
    assert source == "reference_text"


def test_gold_eval_clip_path_wins_and_records_path_and_truth_sources(tmp_path: Path):
    mod = _load_script_module()
    regular = tmp_path / "regular.wav"
    evaluation = tmp_path / "evaluation.wav"
    _write_wav(regular)
    _write_wav(evaluation)
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "case-1",
                        "clip_path": regular.name,
                        "eval_clip_path": evaluation.name,
                        "gold_text": "真值",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    item = mod.load_gold_items(gold_path)[0]

    assert item["clip_path"] == str(evaluation.resolve())
    assert item["clip_path_input"] == evaluation.name
    assert item["clip_path_source"] == "eval_clip_path"
    assert item["gold_text_source"] == "gold_text"


def test_empty_eval_clip_path_falls_back_to_clip_path(tmp_path: Path):
    mod = _load_script_module()
    clip = tmp_path / "clip.wav"
    _write_wav(clip)
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(
        json.dumps(
            {"items": [{"clip_path": clip.name, "eval_clip_path": " ", "text": "真值"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    item = mod.load_gold_items(gold_path)[0]

    assert item["clip_path"] == str(clip.resolve())
    assert item["clip_path_source"] == "clip_path"


def test_levenshtein_backtrace_emits_deterministic_substitution_deletion_insertion():
    mod = _load_script_module()
    substitution = mod.levenshtein_alignment("abc", "adc")
    deletion = mod.levenshtein_alignment("abc", "ac")
    insertion = mod.levenshtein_alignment("ac", "abc")

    assert substitution["substitutions"] == 1
    assert substitution["deletions"] == 0
    assert substitution["insertions"] == 0
    assert [item["op"] for item in substitution["operations"]] == ["=", "S", "="]
    assert deletion["deletions"] == 1
    assert insertion["insertions"] == 1


def test_micro_cer_sums_s_d_i_before_dividing():
    mod = _load_script_module()
    alignments = [
        mod.levenshtein_alignment("abcd", "abxd"),
        mod.levenshtein_alignment("xy", "x"),
        mod.levenshtein_alignment("z", "zz"),
    ]

    metric = mod.micro_cer(alignments)

    assert metric == {
        "reference_chars": 7,
        "substitutions": 1,
        "deletions": 1,
        "insertions": 1,
        "errors": 3,
        "cer": pytest.approx(3 / 7),
    }


def test_aggregate_rtf_cold_rtf_and_repeat_stability_hashes():
    mod = _load_script_module()
    config = {
        "variants": [
            {
                "name": "required-model",
                "adapter": "localscribe",
                "backend": "qwen3",
                "model_id": "fake-model",
                "required": True,
            }
        ]
    }
    cases = [
        {
            "case_id": "case-1",
            "clip_path": "/case-1.wav",
            "gold_text": "你好",
            "gold_normalized": "你好",
        }
    ]
    rounds = [
        _round(
            mod,
            variant="required-model",
            repeat=1,
            load=2.0,
            cases=[_case_result(mod, text="你好。", duration=4.0, inference=1.0, load=2.0)],
        ),
        _round(
            mod,
            variant="required-model",
            repeat=2,
            load=2.0,
            cases=[_case_result(mod, text="你号", duration=4.0, inference=3.0)],
        ),
    ]

    results = mod.aggregate_results(config, cases, rounds, {"required-model"}, repeats=2)
    summary = results["variants"][0]

    assert summary["status"] == "ok"
    assert summary["rtf"] == pytest.approx(4.0 / 8.0)
    assert summary["cold_rtf"] == pytest.approx((4.0 + 4.0) / 8.0)
    assert summary["micro_cer"]["cer"] == pytest.approx(1 / 4)
    assert summary["dataset_raw_unique_output_count"] == 2
    assert summary["dataset_normalized_unique_output_count"] == 2
    assert summary["max_case_raw_unique_count"] == 2
    assert summary["max_case_normalized_unique_count"] == 2
    stability = summary["repeat_stability"][0]
    assert len(stability["raw_hashes"]) == 2
    assert len(stability["normalized_hashes"]) == 2
    assert stability["raw_unique_count"] == 2
    assert stability["normalized_unique_count"] == 2
    assert stability["max_self_cer"] == pytest.approx(0.5)


def test_optional_default_disabled_is_skipped_and_not_fatal():
    mod = _load_script_module()
    config = {
        "variants": [
            {
                "name": "required",
                "adapter": "localscribe",
                "model_id": "required-model",
                "required": True,
            },
            {
                "name": "optional",
                "adapter": "mlx_audio",
                "model_id": "optional-model",
                "required": False,
                "enabled_by_default": False,
            },
        ]
    }
    cases = [{"case_id": "case-1", "clip_path": "/x.wav", "gold_text": "好", "gold_normalized": "好"}]
    rounds = [_round(mod, variant="required", repeat=1, cases=[_case_result(mod, text="好")])]

    results = mod.aggregate_results(config, cases, rounds, {"required"}, repeats=1)

    assert results["status"] == "ok"
    assert results["variants"][1]["status"] == "skipped"
    assert results["policy"]["required_failures"] == []
    assert results["policy"]["optional_failures"] == []


def test_required_variant_failure_is_fatal():
    mod = _load_script_module()
    config = {
        "variants": [
            {
                "name": "required",
                "adapter": "localscribe",
                "model_id": "required-model",
                "required": True,
            }
        ]
    }
    cases = [{"case_id": "case-1", "clip_path": "/x.wav", "gold_text": "好", "gold_normalized": "好"}]
    failed_case = _case_result(mod, status="error", text="", error="boom")
    rounds = [_round(mod, variant="required", repeat=1, cases=[failed_case], status="error")]

    results = mod.aggregate_results(config, cases, rounds, {"required"}, repeats=1)

    assert results["status"] == "failed"
    assert results["variants"][0]["status"] == "error"
    assert results["policy"]["required_failures"] == ["required"]


def test_rss_units_match_macos_bytes_and_linux_kibibytes():
    mod = _load_script_module()
    assert mod.rss_to_mb(100 * 1024 * 1024, "darwin") == pytest.approx(100.0)
    assert mod.rss_to_mb(100 * 1024, "linux") == pytest.approx(100.0)


def test_report_and_tsv_include_required_arena_fields_and_alignment_policy():
    mod = _load_script_module()
    config = {
        "variants": [
            {
                "name": "model",
                "adapter": "localscribe",
                "model_id": "fake-model",
                "required": True,
            }
        ]
    }
    cases = [{"case_id": "case-1", "clip_path": "/x.wav", "gold_text": "你好", "gold_normalized": "你好"}]
    rounds = [_round(mod, variant="model", repeat=1, cases=[_case_result(mod, text="你好")])]
    results = mod.aggregate_results(config, cases, rounds, {"model"}, repeats=1)

    markdown = mod.results_markdown(results)
    tsv = mod.results_tsv(results)

    assert "Micro CER" in markdown
    assert "Cold RTF" in markdown
    assert "Max self CER" in markdown
    assert "Alignment: `not_run`" in markdown
    assert "required failures are fatal" in markdown
    assert "micro_cer" in tsv.splitlines()[0]
    case = results["rounds"][0]["cases"][0]
    assert case["alignment"] == {"status": "not_run", "method": None, "details": None}
    assert {
        "raw_text",
        "display_text",
        "segments",
        "timing_reliable",
        "alignment",
        "load_seconds",
        "inference_seconds",
        "audio_duration",
        "peak_rss_mb",
        "accelerator_memory",
    } <= set(case)


def test_parent_orchestration_uses_fake_worker_and_writes_atomic_outputs(tmp_path: Path, monkeypatch):
    mod = _load_script_module()
    clip_dir = tmp_path / "gold"
    clip_dir.mkdir()
    clip = clip_dir / "clip.wav"
    _write_wav(clip, duration=2.0)
    gold_path = clip_dir / "gold.json"
    gold_path.write_text(
        json.dumps({"items": [{"id": "case-1", "clip_path": "clip.wav", "text": "你好"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "asr-model-arena-config-v1",
                "defaults": {"repeats": 1, "timeout_seconds": 30},
                "variants": [
                    {
                        "name": "fake-required",
                        "adapter": "localscribe",
                        "backend": "qwen3",
                        "model_id": "fake-model",
                        "revision": "fake-required-revision",
                        "required": True,
                        "enabled_by_default": True,
                    },
                    {
                        "name": "fake-optional",
                        "adapter": "mlx_audio",
                        "model_id": "fake-optional",
                        "revision": "fake-optional-revision",
                        "required": False,
                        "enabled_by_default": False,
                        "load_model_kwargs": {"revision": "fake-optional-revision"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_worker(python_executable, request_path, output_path, timeout):
        request = json.loads(request_path.read_text(encoding="utf-8"))
        payload = _round(
            mod,
            variant=request["variant"]["name"],
            repeat=request["repeat"],
            load=0.25,
            revision=request["variant"]["revision"],
            cases=[
                {
                    **_case_result(mod, case_id=item["case_id"], text="你好。", duration=2.0, inference=0.5, load=0.25),
                    "clip_path": item["clip_path"],
                }
                for item in request["cases"]
            ],
        )
        payload["request_id"] = request["request_id"]
        mod.atomic_write_json(output_path, payload)
        return {
            "returncode": 0,
            "elapsed_seconds": 0.01,
            "stdout": "Fetching model shard",
            "stderr": "",
        }

    monkeypatch.setattr(mod, "run_worker_subprocess", fake_worker)
    out_dir = tmp_path / "out"

    exit_code, results = mod.run_arena(
        config_path=config_path,
        gold_path=gold_path,
        out_dir=out_dir,
    )

    assert exit_code == 0
    assert results["status"] == "ok"
    assert results["variants"][0]["status"] == "ok"
    assert results["variants"][0]["load_includes_download"] is True
    assert results["variants"][0]["cold_rtf"] is None
    assert results["variants"][1]["status"] == "skipped"
    for filename in ("arena_results.json", "arena_results.tsv", "arena_report.md", "run_manifest.json"):
        assert (out_dir / filename).is_file()
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["inputs"]["config"]["sha256"] == mod.sha256_file(config_path)
    assert manifest["inputs"]["gold"]["sha256"] == mod.sha256_file(gold_path)
    assert manifest["inputs"]["clips"][0]["sha256"] == mod.sha256_file(clip)
    assert manifest["inputs"]["clips"][0]["path_source"] == "clip_path"
    assert manifest["resolved_variants"][0]["revision"] == "fake-required-revision"
    assert manifest["resolved_variants"][0]["target_python"]["sys_prefix"] == sys.prefix
    report = (out_dir / "arena_report.md").read_text(encoding="utf-8")
    assert "fake-required" in report
    assert "fake-optional" in report
    assert "network transfer is not a valid local model load measurement" in report


def test_worker_output_validation_rejects_missing_alignment_schema():
    mod = _load_script_module()
    variant = {
        "name": "model",
        "adapter": "localscribe",
        "model_id": "fake-model",
        "revision": "test-revision",
        "required": True,
    }
    cases = [{"case_id": "case-1", "clip_path": "/x.wav"}]
    payload = _round(mod, variant="model", repeat=1, cases=[_case_result(mod)])
    payload["cases"][0]["clip_path"] = "/x.wav"
    payload["cases"][0].pop("alignment")

    problem = mod.validate_worker_output(payload, variant, 1, cases)

    assert problem is not None
    assert "alignment" in problem


def test_mlx_generate_api_rejects_silently_dropped_required_settings(tmp_path: Path):
    mod = _load_script_module()

    def narrow_generate(audio, language=None):
        return "should not run"

    with pytest.raises(RuntimeError, match="cannot honor required settings"):
        mod._call_with_supported_kwargs(
            narrow_generate,
            tmp_path / "clip.wav",
            {"language": "Chinese", "max_tokens": 256, "stream": False},
        )


def test_wrapped_missing_dependency_is_unavailable():
    mod = _load_script_module()
    try:
        try:
            raise ModuleNotFoundError("missing runtime")
        except ModuleNotFoundError as cause:
            raise RuntimeError("adapter unavailable") from cause
    except RuntimeError as exc:
        assert mod._classify_exception(exc) == "unavailable"


def test_resume_identity_changes_when_clip_hash_or_variant_changes():
    mod = _load_script_module()
    cases = [{"case_id": "case-1", "clip_path": "/clip.wav", "clip_sha256": "aaa"}]
    variant = {"name": "model", "adapter": "localscribe", "model_id": "one", "required": True}
    first = mod._worker_request(variant, 1, cases)
    changed_clip = mod._worker_request(variant, 1, [{**cases[0], "clip_sha256": "bbb"}])
    changed_model = mod._worker_request({**variant, "model_id": "two"}, 1, cases)

    assert first["request_id"] != changed_clip["request_id"]
    assert first["request_id"] != changed_model["request_id"]


def test_required_python_environment_variable_does_not_fall_back(monkeypatch):
    mod = _load_script_module()
    monkeypatch.delenv("ARENA_ISOLATED_PYTHON", raising=False)

    executable, error = mod._resolve_python(
        {
            "python_env_var": "ARENA_ISOLATED_PYTHON",
            "python_env_required": True,
        }
    )

    assert executable is None
    assert "ARENA_ISOLATED_PYTHON" in error


def test_resolve_python_preserves_configured_virtualenv_symlink(tmp_path: Path, monkeypatch):
    mod = _load_script_module()
    python_path = tmp_path / "venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.symlink_to(sys.executable)
    monkeypatch.setenv("ARENA_ISOLATED_PYTHON", str(python_path))

    executable, error = mod._resolve_python({"python_env_var": "ARENA_ISOLATED_PYTHON"})

    assert error is None
    assert executable == str(python_path.absolute())
    assert executable != str(python_path.resolve())


def test_default_config_pins_revisions_and_requires_all_isolated_environments():
    mod = _load_script_module()

    config = mod._load_config(ROOT / "experiments" / "asr_model_arena_phase1.json")
    variants = {item["name"]: item for item in config["variants"]}

    assert variants["sensevoice-prod"]["revision"] == "70514a3da51f1160f51d18449dab6128bbd4928b"
    assert variants["qwen3-mlx-1.7b-8bit-prod"]["revision"] == "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"
    assert variants["qwen3-official-0.6b"]["revision"] == "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
    assert variants["qwen3-official-1.7b"]["revision"] == "7278e1e70fe206f11671096ffdd38061171dd6e5"
    assert all(item["python_env_required"] is True for item in variants.values())
    for name in ("qwen3-official-0.6b", "qwen3-official-1.7b"):
        assert variants[name]["load_model_kwargs"]["revision"] == variants[name]["revision"]


def test_mlx_lock_matches_exact_isolated_environment_freeze():
    expected = {
        "annotated-doc": "0.0.4",
        "anyio": "4.14.2",
        "certifi": "2026.6.17",
        "cffi": "2.1.0",
        "click": "8.4.2",
        "filelock": "3.30.2",
        "fsspec": "2026.6.0",
        "h11": "0.16.0",
        "hf-xet": "1.5.2",
        "httpcore": "1.0.9",
        "httpx": "0.28.1",
        "huggingface-hub": "1.23.0",
        "idna": "3.18",
        "jinja2": "3.1.6",
        "markdown-it-py": "4.2.0",
        "markupsafe": "3.0.3",
        "mdurl": "0.1.2",
        "miniaudio": "1.71",
        "mlx": "0.32.0",
        "mlx-audio": "0.4.5",
        "mlx-lm": "0.31.3",
        "mlx-metal": "0.32.0",
        "numpy": "2.5.1",
        "packaging": "26.2",
        "protobuf": "7.35.1",
        "pycparser": "3.0",
        "pygments": "2.20.0",
        "pyyaml": "6.0.3",
        "regex": "2026.7.10",
        "rich": "15.0.0",
        "safetensors": "0.8.0",
        "scipy": "1.18.0",
        "sentencepiece": "0.2.2",
        "shellingham": "1.5.4",
        "sounddevice": "0.5.5",
        "tokenizers": "0.22.2",
        "tqdm": "4.68.4",
        "transformers": "5.12.1",
        "typer": "0.27.0",
        "typing-extensions": "4.16.0",
    }
    lock_path = ROOT / "experiments" / "asr_model_arena_mlx.lock"
    actual = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        package, version = line.split("==", 1)
        actual[package] = version

    assert actual == expected


def test_worker_subprocess_strips_python_environment_and_preserves_model_environment(
    tmp_path: Path, monkeypatch
):
    mod = _load_script_module()
    captured = {}
    monkeypatch.setenv("PYTHONPATH", "/polluting/path")
    monkeypatch.setenv("PYTHONHOME", "/polluting/home")
    monkeypatch.setenv("VIRTUAL_ENV", "/polluting/venv")
    monkeypatch.setenv("HF_HOME", "/models/hf")
    monkeypatch.setenv("MODELSCOPE_CACHE", "/models/modelscope")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="x" * 13000 + " Downloading model",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    process = mod.run_worker_subprocess(
        sys.executable, tmp_path / "request.json", tmp_path / "output.json", 30
    )

    env = captured["env"]
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert "VIRTUAL_ENV" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["HF_HOME"] == "/models/hf"
    assert env["MODELSCOPE_CACHE"] == "/models/modelscope"
    assert process["load_includes_download"] is True
    assert process["stdout"].endswith("...[truncated]")


def test_target_python_fingerprint_records_prefix_packages_and_inference_environment(monkeypatch):
    mod = _load_script_module()
    monkeypatch.setenv("HF_HOME", "/tmp/test-hf-home")
    monkeypatch.setenv("LOCALSCRIBE_FUNASR_INFERENCE_SEED", "1")
    variant = {"adapter": "mlx_audio", "expected_packages": {"mlx-audio": "0.4.5"}}

    fingerprint = mod._worker_fingerprint(variant, sys.executable, {"sha256": "lock-sha"})
    first_request = mod._worker_request(
        {"name": "model", "adapter": "mlx_audio", "model_id": "fake", "revision": "rev"},
        1,
        [{"case_id": "case-1", "clip_path": "/clip.wav", "clip_sha256": "sha"}],
        fingerprint,
    )
    monkeypatch.setenv("LOCALSCRIBE_FUNASR_INFERENCE_SEED", "2")
    changed_fingerprint = mod._worker_fingerprint(variant, sys.executable, {"sha256": "lock-sha"})
    changed_request = mod._worker_request(
        {"name": "model", "adapter": "mlx_audio", "model_id": "fake", "revision": "rev"},
        1,
        [{"case_id": "case-1", "clip_path": "/clip.wav", "clip_sha256": "sha"}],
        changed_fingerprint,
    )

    assert fingerprint["target_python_query_ok"] is True
    assert fingerprint["target_python"]["sys_prefix"] == sys.prefix
    assert "mlx-audio" in fingerprint["target_python"]["packages"]
    assert fingerprint["target_python"]["environment_variables"]["HF_HOME"] == "/tmp/test-hf-home"
    assert fingerprint["target_python"]["environment_variables"]["PYTHONNOUSERSITE"] == "1"
    assert first_request["request_id"] != changed_request["request_id"]


def test_failed_target_identity_query_disables_resume_even_for_matching_cached_request(
    tmp_path: Path, monkeypatch
):
    mod = _load_script_module()
    clip = tmp_path / "clip.wav"
    _write_wav(clip)
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(
        json.dumps({"items": [{"clip_path": clip.name, "correct_text": "真值"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "asr-model-arena-config-v1",
                "defaults": {"repeats": 1, "timeout_seconds": 30},
                "variants": [
                    {
                        "name": "model",
                        "adapter": "localscribe",
                        "backend": "qwen3",
                        "model_id": "fake-model",
                        "revision": "fixed-revision",
                        "required": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mod,
        "_worker_fingerprint",
        lambda *args, **kwargs: {
            "target_python": None,
            "target_python_query_ok": False,
            "target_python_query_error": "query failed",
        },
    )
    worker_calls = []

    def fake_worker(python_executable, request_path, output_path, timeout):
        worker_calls.append(request_path)
        request = json.loads(request_path.read_text(encoding="utf-8"))
        payload = _round(
            mod,
            variant="model",
            repeat=1,
            revision="fixed-revision",
            cases=[
                {
                    **_case_result(mod, text="真值"),
                    "clip_path": request["cases"][0]["clip_path"],
                }
            ],
        )
        payload["request_id"] = request["request_id"]
        mod.atomic_write_json(output_path, payload)
        return {"returncode": 0, "elapsed_seconds": 0.01, "stdout": "", "stderr": ""}

    monkeypatch.setattr(mod, "run_worker_subprocess", fake_worker)
    out_dir = tmp_path / "out"
    mod.run_arena(config_path=config_path, gold_path=gold_path, out_dir=out_dir)
    _, results = mod.run_arena(
        config_path=config_path,
        gold_path=gold_path,
        out_dir=out_dir,
        resume=True,
    )

    assert len(worker_calls) == 2
    assert results["rounds"][0].get("resumed") is not True
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["resolved_variants"][0]["resume_identity_query_ok"] is False
    assert manifest["resolved_variants"][0]["resume_identity_query_error"] == "query failed"


def test_parent_failure_round_uses_null_rss_and_validator_requires_rss_only_for_success():
    mod = _load_script_module()
    variant = {
        "name": "model",
        "adapter": "localscribe",
        "model_id": "fake-model",
        "revision": "test-revision",
        "required": True,
    }
    cases = [{"case_id": "case-1", "clip_path": "/missing.wav"}]
    failure = mod._failure_round(variant, 1, cases, "timeout", "timed out")

    assert failure["cases"][0]["peak_rss_mb"] is None
    assert mod.validate_worker_output(failure, variant, 1, cases) is None

    success = _round(mod, variant="model", repeat=1, cases=[_case_result(mod, peak_rss=None)])
    success["cases"][0]["clip_path"] = "/missing.wav"
    problem = mod.validate_worker_output(success, variant, 1, cases)
    assert problem is not None
    assert "peak_rss_mb" in problem


def test_download_detection_is_case_insensitive_and_invalidates_variant_cold_rtf():
    mod = _load_script_module()
    assert mod._process_output_includes_download({"stdout": "", "stderr": "Downloading weights"}) is True
    assert mod._process_output_includes_download({"stdout": "FETCHING config", "stderr": ""}) is True
    config = {
        "variants": [
            {
                "name": "model",
                "adapter": "localscribe",
                "model_id": "fake-model",
                "revision": "test-revision",
                "required": True,
            }
        ]
    }
    cases = [{"case_id": "case-1", "clip_path": "/x.wav", "gold_text": "好", "gold_normalized": "好"}]
    rounds = [
        _round(
            mod,
            variant="model",
            repeat=1,
            load=2.0,
            load_includes_download=True,
            cases=[_case_result(mod, text="好", duration=2.0, inference=1.0, load=2.0)],
        )
    ]

    results = mod.aggregate_results(config, cases, rounds, {"model"}, repeats=1)
    summary = results["variants"][0]

    assert summary["rtf"] == pytest.approx(0.5)
    assert summary["cold_rtf"] is None
    assert summary["load_includes_download"] is True
    assert "Cold RTF omitted" in summary["warnings"][0]
    assert results["rounds"][0]["cases"][0]["cold_rtf"] is None


def test_huggingface_refs_main_must_match_configured_revision(tmp_path: Path, monkeypatch):
    mod = _load_script_module()
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    refs_main = mod._huggingface_model_cache_path("org/model") / "refs" / "main"
    refs_main.parent.mkdir(parents=True)
    refs_main.write_text("expected-revision\n", encoding="utf-8")

    identity = mod._verify_huggingface_main_revision("org/model", "expected-revision")

    assert identity["refs_main_exists"] is True
    assert identity["refs_main_revision"] == "expected-revision"
    with pytest.raises(RuntimeError, match="revision mismatch"):
        mod._verify_huggingface_main_revision("org/model", "different-revision")


def test_sensevoice_identity_records_configured_revision_and_optional_model_hash(
    tmp_path: Path, monkeypatch
):
    mod = _load_script_module()
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "modelscope"))
    monkeypatch.setattr(mod, "_modelscope_model_candidates", lambda model_id: [])

    missing = mod._sensevoice_model_identity(object(), "iic/SenseVoiceSmall", "configured-revision")
    assert missing["configured_revision"] == "configured-revision"
    assert missing["model_pt_sha256"] is None

    model_dir = tmp_path / "actual-model"
    model_dir.mkdir()
    model_pt = model_dir / "model.pt"
    model_pt.write_bytes(b"small fake model")
    transcriber = types.SimpleNamespace(_model=types.SimpleNamespace(model_path=str(model_dir)))
    present = mod._sensevoice_model_identity(transcriber, "iic/SenseVoiceSmall", "configured-revision")

    assert present["model_pt_path"] == str(model_pt)
    assert present["model_pt_sha256"] == mod.sha256_file(model_pt)


def test_official_mlx_loader_forces_configured_revision(monkeypatch):
    mod = _load_script_module()
    captured = {}

    class FakeModel:
        def generate(self, audio, **kwargs):
            return ""

    def fake_load_model(model_id, **kwargs):
        captured["model_id"] = model_id
        captured["kwargs"] = kwargs
        return FakeModel()

    mlx_audio = types.ModuleType("mlx_audio")
    stt = types.ModuleType("mlx_audio.stt")
    utils = types.ModuleType("mlx_audio.stt.utils")
    utils.load_model = fake_load_model
    mlx_audio.stt = stt
    stt.utils = utils
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils)

    _infer, _load_seconds, identity = mod._load_mlx_audio_adapter(
        {
            "model_id": "Qwen/Qwen3-ASR-0.6B",
            "revision": "fixed-revision",
            "load_model_kwargs": {},
        },
        ROOT,
    )

    assert captured["model_id"] == "Qwen/Qwen3-ASR-0.6B"
    assert captured["kwargs"]["revision"] == "fixed-revision"
    assert identity["configured_revision"] == "fixed-revision"


def test_inference_identity_includes_strict_coverage_configuration():
    mod = _load_script_module()

    assert "LOCALSCRIBE_SENSEVOICE_STRICT_COVERAGE" in mod._INFERENCE_ENV_NAMES
    assert "LOCALSCRIBE_SENSEVOICE_STRICT_COVERAGE_MAX_CHUNK_S" in mod._INFERENCE_ENV_NAMES
    assert "LOCALSCRIBE_SENSEVOICE_STRICT_COVERAGE_CONTEXT_PAD_S" in mod._INFERENCE_ENV_NAMES
    assert "LOCALSCRIBE_SENSEVOICE_COVERAGE_MIN_CHARS_PER_S" in mod._INFERENCE_ENV_NAMES
