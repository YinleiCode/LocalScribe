#!/usr/bin/env python3
"""Phase-one ASR model arena with isolated, serial worker processes.

The parent process intentionally uses the Python standard library for orchestration.
Model runtimes are imported only by the ``worker`` subcommand.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unicodedata
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "asr_model_arena_phase1.json"
SCHEMA_VERSION = "asr-model-arena-v1"
VALID_STATUSES = {
    "ok",
    "skipped",
    "unavailable",
    "error",
    "timeout",
    "oom",
    "invalid_output",
}
FAILURE_PRIORITY = {
    "ok": 0,
    "skipped": 1,
    "unavailable": 2,
    "error": 3,
    "invalid_output": 4,
    "oom": 5,
    "timeout": 6,
}
ALIGNMENT_NOT_RUN = {"status": "not_run", "method": None, "details": None}
_OPENCC_UNSET = object()
_opencc_converter: Any = _OPENCC_UNSET
_MEASURE_CURRENT_RSS = object()
_DOWNLOAD_OUTPUT_RE = re.compile(r"\b(?:fetching|downloading)\b", re.IGNORECASE)
_WORKER_ENV_REMOVE = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
_INFERENCE_ENV_NAMES = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "MODELSCOPE_CACHE",
    "LOCALSCRIBE_MODELSCOPE_CACHE",
    "LOCALSCRIBE_ALLOW_MODEL_DOWNLOAD",
    "LOCALSCRIBE_FUNASR_BATCH_SIZE_S",
    "LOCALSCRIBE_FUNASR_DEVICE",
    "LOCALSCRIBE_FUNASR_INFERENCE_SEED",
    "LOCALSCRIBE_FUNASR_PUNC",
    "LOCALSCRIBE_FUNASR_VAD_MAX_MS",
    "LOCALSCRIBE_FUNASR_VERBOSE",
    "LOCALSCRIBE_SENSEVOICE_BATCH_SIZE_S",
    "LOCALSCRIBE_SENSEVOICE_MERGE_LENGTH_S",
    "LOCALSCRIBE_SENSEVOICE_MERGE_VAD",
    "LOCALSCRIBE_SENSEVOICE_SPEECH_COVERAGE",
    "LOCALSCRIBE_SENSEVOICE_STRICT_COVERAGE",
    "LOCALSCRIBE_SENSEVOICE_STRICT_COVERAGE_MAX_CHUNK_S",
    "LOCALSCRIBE_SENSEVOICE_STRICT_COVERAGE_CONTEXT_PAD_S",
    "LOCALSCRIBE_SENSEVOICE_COVERAGE_MIN_CHARS_PER_S",
    "LOCALSCRIBE_SENSEVOICE_MIN_SPEECH_COVERAGE_RATIO",
    "LOCALSCRIBE_SENSEVOICE_MAX_UNCOVERED_SPEECH_S",
    "LOCALSCRIBE_SENSEVOICE_MAX_EDGE_UNCOVERED_SPEECH_S",
    "LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_MODE",
    "LOCALSCRIBE_SENSEVOICE_LOCAL_RECOVERY_PROVIDER",
    "LOCALSCRIBE_SENSEVOICE_PARAFORMER_ANCHOR",
    "LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN",
    "LOCALSCRIBE_SENSEVOICE_TIMING_ALIGN_MIN_RATIO",
    "LOCALSCRIBE_SENSEVOICE_VAD_MAX_MS",
    "LOCALSCRIBE_SENSEVOICE_WALLCLOCK_MAX_CHUNK_S",
    "LOCALSCRIBE_SENSEVOICE_WALLCLOCK_MAX_GAP_S",
    "LOCALSCRIBE_SENSEVOICE_WALLCLOCK_MIN_SILENCE_MS",
    "LOCALSCRIBE_SENSEVOICE_WALLCLOCK_MIN_SPEECH_MS",
    "LOCALSCRIBE_SENSEVOICE_WALLCLOCK_PAD_MS",
    "LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD",
    "LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD_FIRST",
    "LOCALSCRIBE_SENSEVOICE_WALLCLOCK_VAD_THRESHOLD",
    "XDG_CACHE_HOME",
    "TOKENIZERS_PARALLELISM",
    "MLX_METAL_CACHE_DIR",
    "CUDA_VISIBLE_DEVICES",
)
_TARGET_PACKAGE_NAMES = (
    "scribe-py",
    "mlx-audio",
    "mlx",
    "mlx-lm",
    "transformers",
    "huggingface-hub",
    "funasr",
    "modelscope",
    "torch",
    "numpy",
    "safetensors",
    "tokenizers",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return cleaned or "variant"


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _truncate(value: Any, limit: int = 12000) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _get_opencc_converter() -> Any | None:
    global _opencc_converter
    if _opencc_converter is not _OPENCC_UNSET:
        return _opencc_converter
    try:
        from opencc import OpenCC  # type: ignore

        _opencc_converter = OpenCC("t2s")
    except Exception:
        _opencc_converter = None
    return _opencc_converter


def normalize_text(text: str, converter: Any = _OPENCC_UNSET) -> str:
    """Normalize for CER: NFKC, optional traditional-to-simplified, lowercase,
    then remove punctuation and whitespace/separator characters.
    """

    value = unicodedata.normalize("NFKC", str(text or ""))
    actual_converter = _get_opencc_converter() if converter is _OPENCC_UNSET else converter
    if actual_converter is not None:
        try:
            value = actual_converter.convert(value)
        except Exception:
            pass
    value = value.lower()
    return "".join(
        char
        for char in value
        if not char.isspace()
        and not unicodedata.category(char).startswith("P")
        and not unicodedata.category(char).startswith("Z")
    )


def levenshtein_alignment(reference: str, hypothesis: str) -> dict[str, Any]:
    """Return deterministic character edit counts and a backtrace.

    Ties prefer substitution, then deletion, then insertion. This makes the
    emitted S/D/I sequence stable across Python versions and runs.
    """

    ref = str(reference)
    hyp = str(hypothesis)
    rows = len(ref) + 1
    cols = len(hyp) + 1
    dp = [[0] * cols for _ in range(rows)]
    back: list[list[str | None]] = [[None] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, cols):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "="
                continue
            candidates = (
                (dp[i - 1][j - 1] + 1, 0, "S"),
                (dp[i - 1][j] + 1, 1, "D"),
                (dp[i][j - 1] + 1, 2, "I"),
            )
            cost, _rank, operation = min(candidates)
            dp[i][j] = cost
            back[i][j] = operation

    operations: list[dict[str, Any]] = []
    substitutions = deletions = insertions = 0
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        operation = back[i][j]
        if operation == "=":
            operations.append({"op": "=", "ref": ref[i - 1], "hyp": hyp[j - 1]})
            i -= 1
            j -= 1
        elif operation == "S":
            substitutions += 1
            operations.append({"op": "S", "ref": ref[i - 1], "hyp": hyp[j - 1]})
            i -= 1
            j -= 1
        elif operation == "D":
            deletions += 1
            operations.append({"op": "D", "ref": ref[i - 1], "hyp": ""})
            i -= 1
        elif operation == "I":
            insertions += 1
            operations.append({"op": "I", "ref": "", "hyp": hyp[j - 1]})
            j -= 1
        else:
            raise RuntimeError(f"invalid Levenshtein backtrace at ({i}, {j})")
    operations.reverse()
    errors = substitutions + deletions + insertions
    cer = errors / len(ref) if ref else (0.0 if not hyp else None)
    return {
        "reference_chars": len(ref),
        "hypothesis_chars": len(hyp),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "cer": cer,
        "operations": operations,
    }


def micro_cer(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    substitutions = deletions = insertions = reference_chars = 0
    for item in items:
        substitutions += int(item.get("substitutions") or 0)
        deletions += int(item.get("deletions") or 0)
        insertions += int(item.get("insertions") or 0)
        reference_chars += int(item.get("reference_chars") or 0)
    errors = substitutions + deletions + insertions
    return {
        "reference_chars": reference_chars,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "cer": errors / reference_chars if reference_chars else (0.0 if errors == 0 else None),
    }


def self_cer(left: str, right: str) -> float:
    alignment = levenshtein_alignment(left, right)
    denominator = max(len(left), len(right), 1)
    return float(alignment["errors"]) / denominator


def rss_to_mb(value: float, system: str | None = None) -> float:
    """Convert resource.ru_maxrss to MiB (bytes on macOS, KiB elsewhere)."""

    current = (system or sys.platform).lower()
    divisor = 1024.0 * 1024.0 if current.startswith("darwin") else 1024.0
    return float(value) / divisor


def peak_rss_mb() -> float:
    return round(rss_to_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), 3)


def _reset_accelerator_peak() -> None:
    try:
        import mlx.core as mx  # type: ignore

        reset = getattr(mx, "reset_peak_memory", None)
        if callable(reset):
            reset()
    except Exception:
        return


def accelerator_memory() -> dict[str, Any]:
    try:
        import mlx.core as mx  # type: ignore

        getter = getattr(mx, "get_peak_memory", None)
        if not callable(getter):
            return {"provider": "mlx", "available": False, "peak_mb": None, "raw_bytes": None}
        raw_bytes = int(getter() or 0)
        return {
            "provider": "mlx",
            "available": True,
            "peak_mb": round(raw_bytes / (1024.0 * 1024.0), 3),
            "raw_bytes": raw_bytes,
        }
    except Exception as exc:
        return {
            "provider": "mlx",
            "available": False,
            "peak_mb": None,
            "raw_bytes": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else 0.0
    except Exception:
        pass
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode == 0:
            return max(_finite_float(completed.stdout.strip()), 0.0)
    except Exception:
        pass
    return 0.0


def _nonempty_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _truth_text_and_source(item: dict[str, Any]) -> tuple[str, str]:
    if "correct_text" in item:
        correct = _nonempty_text(item.get("correct_text"))
        if correct is None:
            raise ValueError("gold item correct_text must be non-empty when present")
        return correct, "correct_text"
    for key in ("gold_text", "reference_text", "text", "transcript"):
        fallback = _nonempty_text(item.get(key))
        if fallback is not None:
            return fallback, key
    raise ValueError(
        "gold item is missing a non-empty correct_text/gold_text/reference_text/text/transcript"
    )


def _truth_text(item: dict[str, Any]) -> str:
    return _truth_text_and_source(item)[0]


def load_gold_items(gold_path: Path, case_limit: int | None = None) -> list[dict[str, Any]]:
    payload = read_json(gold_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("gold JSON must be an object containing an items array")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload["items"]):
        if case_limit is not None and len(items) >= case_limit:
            break
        if not isinstance(raw, dict):
            raise ValueError(f"gold items[{index}] must be an object")
        eval_clip_value = _nonempty_text(raw.get("eval_clip_path"))
        clip_value = eval_clip_value or _nonempty_text(raw.get("clip_path"))
        clip_source = "eval_clip_path" if eval_clip_value is not None else "clip_path"
        if clip_value is None:
            raise ValueError(f"gold items[{index}] is missing a non-empty eval_clip_path/clip_path")
        clip_path = Path(str(clip_value)).expanduser()
        if not clip_path.is_absolute():
            clip_path = (gold_path.parent / clip_path).resolve()
        else:
            clip_path = clip_path.resolve()
        if not clip_path.is_file():
            raise FileNotFoundError(f"gold clip not found: {clip_path}")
        case_id = str(raw.get("id") or raw.get("case_id") or f"case-{index + 1:04d}")
        if case_id in seen:
            raise ValueError(f"duplicate gold case id: {case_id}")
        seen.add(case_id)
        truth, truth_source = _truth_text_and_source(raw)
        items.append(
            {
                "case_id": case_id,
                "clip_path": str(clip_path),
                "clip_path_input": str(clip_value),
                "clip_path_source": clip_source,
                "clip_sha256": sha256_file(clip_path),
                "gold_text": truth,
                "gold_text_source": truth_source,
                "gold_normalized": normalize_text(truth),
                "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
            }
        )
    if not items:
        raise ValueError("gold JSON contains no cases after applying case-limit")
    return items


def _package_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        except Exception as exc:
            versions[name] = f"error:{type(exc).__name__}"
    return versions


def _git_info(root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if commit.returncode == 0:
            info["commit"] = commit.stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if branch.returncode == 0:
            info["branch"] = branch.stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _relevant_environment(config: dict[str, Any]) -> dict[str, Any]:
    env_names = {
        "LANG",
        "LC_ALL",
        *_INFERENCE_ENV_NAMES,
    }
    for variant in config.get("variants") or []:
        if isinstance(variant, dict) and variant.get("python_env_var"):
            env_names.add(str(variant["python_env_var"]))
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
        "environment_variables": {name: os.environ.get(name) for name in sorted(env_names)},
        "packages": _package_versions(
            ["opencc", "opencc-python-reimplemented", "mlx-audio", "mlx", "funasr", "modelscope"]
        ),
    }


def _worker_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in _WORKER_ENV_REMOVE:
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _huggingface_hub_cache(env: dict[str, str] | None = None) -> Path:
    values = os.environ if env is None else env
    direct = values.get("HF_HUB_CACHE") or values.get("HUGGINGFACE_HUB_CACHE")
    if direct:
        return Path(direct).expanduser()
    hf_home = values.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    xdg_cache = values.get("XDG_CACHE_HOME")
    cache_root = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return cache_root / "huggingface" / "hub"


def _huggingface_model_cache_path(model_id: str, env: dict[str, str] | None = None) -> Path:
    cache_name = "models--" + "--".join(part for part in model_id.split("/") if part)
    return _huggingface_hub_cache(env) / cache_name


def _verify_huggingface_main_revision(model_id: str, revision: str) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "provider": "huggingface",
        "configured_revision": revision,
        "refs_main_path": None,
        "refs_main_revision": None,
        "refs_main_exists": False,
    }
    expanded = Path(model_id).expanduser()
    if expanded.exists() or "/" not in model_id:
        identity["provider"] = "local"
        return identity
    refs_main = _huggingface_model_cache_path(model_id) / "refs" / "main"
    identity["refs_main_path"] = str(refs_main)
    if not refs_main.exists():
        return identity
    identity["refs_main_exists"] = True
    try:
        cached_revision = refs_main.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"cannot read HuggingFace refs/main for {model_id}: {exc}") from exc
    identity["refs_main_revision"] = cached_revision
    if cached_revision != revision:
        raise RuntimeError(
            f"HuggingFace refs/main revision mismatch for {model_id}: "
            f"configured {revision}, cached {cached_revision or '<empty>'}"
        )
    return identity


def _modelscope_model_candidates(model_id: str) -> list[Path]:
    explicit = Path(model_id).expanduser()
    candidates: list[Path] = [explicit] if explicit.exists() else []
    cache_values = [
        os.environ.get("LOCALSCRIBE_MODELSCOPE_CACHE"),
        os.environ.get("MODELSCOPE_CACHE"),
        str(Path.home() / ".cache" / "modelscope" / "hub"),
    ]
    parts = [part for part in model_id.split("/") if part]
    if len(parts) >= 2:
        for raw_root in cache_values:
            if not raw_root:
                continue
            root = Path(raw_root).expanduser()
            if root.name == "models":
                candidates.append(root.joinpath(*parts))
            else:
                candidates.append(root.joinpath("models", *parts))
                candidates.append(root.joinpath(*parts))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _sensevoice_model_identity(
    transcriber: Any, model_id: str, revision: str
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "provider": "modelscope",
        "configured_revision": revision,
        "resolved_model_path": None,
        "model_pt_path": None,
        "model_pt_sha256": None,
    }
    candidates: list[Path] = []
    errors: list[str] = []
    for value in (
        getattr(getattr(transcriber, "_model", None), "model_path", None),
        getattr(transcriber, "_resolved_model_id", None),
    ):
        if value:
            try:
                candidates.append(Path(str(value)).expanduser())
            except Exception as exc:
                errors.append(f"invalid resolved model path: {type(exc).__name__}: {exc}")
    candidates.extend(_modelscope_model_candidates(model_id))
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        model_path = candidate / "model.pt" if candidate.is_dir() else candidate
        if model_path.name != "model.pt" or not model_path.is_file():
            continue
        identity["resolved_model_path"] = str(model_path.parent)
        identity["model_pt_path"] = str(model_path)
        try:
            identity["model_pt_sha256"] = sha256_file(model_path)
        except OSError as exc:
            errors.append(f"cannot hash {model_path}: {type(exc).__name__}: {exc}")
        break
    if errors:
        identity["identity_errors"] = errors
    return identity


def _classify_exception(exc: BaseException) -> str:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    message = " | ".join(f"{type(item).__name__}: {item}" for item in chain).lower()
    if any(isinstance(item, (ModuleNotFoundError, ImportError)) for item in chain):
        return "unavailable"
    if "out of memory" in message or "memoryerror" in message or "metal command buffer" in message:
        return "oom"
    return "error"


def _round_status(statuses: Iterable[str]) -> str:
    values = [status if status in VALID_STATUSES else "invalid_output" for status in statuses]
    if not values:
        return "invalid_output"
    return max(values, key=lambda value: FAILURE_PRIORITY[value])


def _empty_case_result(
    case: dict[str, Any],
    status: str,
    error: str = "",
    *,
    collect_accelerator_memory: bool = True,
    peak_rss_value: Any = _MEASURE_CURRENT_RSS,
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "clip_path": case["clip_path"],
        "status": status,
        "raw_text": "",
        "display_text": "",
        "normalized_text": "",
        "segments": [],
        "timing_reliable": False,
        "alignment": dict(ALIGNMENT_NOT_RUN),
        "load_seconds": 0.0,
        "inference_seconds": 0.0,
        "audio_duration": audio_duration(Path(case["clip_path"])),
        "peak_rss_mb": peak_rss_mb() if peak_rss_value is _MEASURE_CURRENT_RSS else peak_rss_value,
        "accelerator_memory": (
            accelerator_memory()
            if collect_accelerator_memory
            else {"provider": None, "available": False, "peak_mb": None, "raw_bytes": None}
        ),
        "error": error,
    }


def _segment_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _standardize_segments(raw_segments: Any) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if raw_segments is None:
        return segments
    try:
        source = list(raw_segments)
    except TypeError:
        return segments
    for item in source:
        text = str(_segment_value(item, "text", "") or "")
        start = max(_finite_float(_segment_value(item, "start", 0.0)), 0.0)
        end = max(_finite_float(_segment_value(item, "end", start)), start)
        segment = {"start": start, "end": end, "text": text}
        original = _segment_value(item, "original_text", None)
        if original is not None and str(original) != text:
            segment["raw_text"] = str(original)
        segments.append(segment)
    return segments


def _join_segment_text(segments: Sequence[dict[str, Any]], raw: bool = False) -> str:
    key = "raw_text" if raw else "text"
    return "".join(str(item.get(key, item.get("text", "")) or "") for item in segments).strip()


def _load_localscribe_adapter(
    variant: dict[str, Any], repo_root: Path
) -> tuple[Callable[[Path, float], dict[str, Any]], float, dict[str, Any]]:
    py_src = repo_root / "scribe-py" / "src"
    if str(py_src) not in sys.path:
        sys.path.insert(0, str(py_src))

    backend = str(variant.get("backend") or "")
    model_id = str(variant.get("model_id") or "")
    if backend == "sensevoice":
        from scribe_py.core.transcriber_funasr import FunASRTranscriber  # type: ignore

        transcriber = FunASRTranscriber(backend_name="sensevoice")
    elif backend == "qwen3":
        from scribe_py.core.transcriber_qwen3 import Qwen3ASRTranscriber  # type: ignore

        transcriber = Qwen3ASRTranscriber()
    else:
        raise ValueError(f"localscribe adapter only supports exact backends sensevoice/qwen3, got {backend!r}")
    from scribe_py.core.types import TranscribeOptions  # type: ignore

    if not model_id:
        raise ValueError("localscribe adapter requires an explicit model_id")
    started = time.perf_counter()
    transcriber._load(model_id)  # exact backend/model; deliberately bypass selector fallback
    load_seconds = time.perf_counter() - started
    revision = str(variant.get("revision") or "")
    model_identity = (
        _sensevoice_model_identity(transcriber, model_id, revision)
        if backend == "sensevoice"
        else {"provider": "huggingface", "configured_revision": revision}
    )

    options_config = variant.get("options") if isinstance(variant.get("options"), dict) else {}

    def infer(path: Path, _duration: float) -> dict[str, Any]:
        options = TranscribeOptions(
            language=options_config.get("language", "zh"),
            model_id=model_id,
            initial_prompt=str(options_config.get("initial_prompt") or ""),
            hotwords=list(options_config.get("hotwords") or []),
            word_timestamps=bool(options_config.get("word_timestamps", False)),
            timing_align=options_config.get("timing_align"),
            normalizer_profile=options_config.get("normalizer_profile"),
            audio_preprocess=str(options_config.get("audio_preprocess") or "adaptive"),
        )
        result = transcriber.transcribe(path, options)
        if str(getattr(result, "backend", "")) != backend:
            raise RuntimeError(
                f"adapter backend mismatch: requested {backend!r}, result reported {getattr(result, 'backend', None)!r}"
            )
        if str(getattr(result, "model_id", "")) != model_id:
            raise RuntimeError(
                f"adapter model mismatch: requested {model_id!r}, result reported {getattr(result, 'model_id', None)!r}"
            )
        segments = _standardize_segments(getattr(result, "segments", []))
        raw_text = _join_segment_text(segments, raw=True)
        display_text = _join_segment_text(segments)
        stats = getattr(result, "filter_stats", {}) or {}
        return {
            "raw_text": raw_text,
            "display_text": display_text,
            "segments": segments,
            "timing_reliable": bool(stats.get("timing_reliable", False)),
            "backend_metadata": {
                "backend": getattr(result, "backend", backend),
                "model_id": getattr(result, "model_id", model_id),
                "language": getattr(result, "language", None),
                "filter_stats": stats,
            },
        }

    return infer, load_seconds, model_identity


def _call_with_supported_kwargs(function: Callable[..., Any], audio: Path, kwargs: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(str(audio), **kwargs)
    parameters = signature.parameters
    accepts_extra = any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values())
    missing = sorted(key for key in kwargs if key not in parameters and not accepts_extra)
    if missing:
        raise RuntimeError(
            "mlx_audio generate API cannot honor required settings: " + ", ".join(missing)
        )
    return function(str(audio), **kwargs)


def _extract_mlx_output(result: Any, duration: float) -> dict[str, Any]:
    if isinstance(result, str):
        text = result
        raw_segments: Any = []
    elif isinstance(result, dict):
        text = str(result.get("text") or result.get("transcript") or "")
        raw_segments = result.get("segments")
    else:
        text = str(getattr(result, "text", "") or "")
        raw_segments = getattr(result, "segments", None)
    segments = _standardize_segments(raw_segments)
    if not segments and text:
        segments = [{"start": 0.0, "end": max(duration, 0.0), "text": text}]
    display_text = _join_segment_text(segments) or text.strip()
    return {
        "raw_text": text.strip() or display_text,
        "display_text": display_text,
        "segments": segments,
        "timing_reliable": bool(
            (result.get("timing_reliable") if isinstance(result, dict) else getattr(result, "timing_reliable", False))
        ),
        "backend_metadata": {
            "language": result.get("language") if isinstance(result, dict) else getattr(result, "language", None),
            "prompt_tokens": result.get("prompt_tokens") if isinstance(result, dict) else getattr(result, "prompt_tokens", None),
            "generation_tokens": (
                result.get("generation_tokens") if isinstance(result, dict) else getattr(result, "generation_tokens", None)
            ),
            "model_seconds": result.get("total_time") if isinstance(result, dict) else getattr(result, "total_time", None),
        },
    }


def _load_mlx_audio_adapter(
    variant: dict[str, Any], _repo_root: Path
) -> tuple[Callable[[Path, float], dict[str, Any]], float, dict[str, Any]]:
    try:
        from mlx_audio.stt.utils import load_model  # type: ignore
    except Exception as exc:
        raise RuntimeError("mlx_audio adapter requires mlx-audio with STT support") from exc
    model_id = str(variant.get("model_id") or "")
    if not model_id:
        raise ValueError("mlx_audio adapter requires an explicit model_id")
    revision = str(variant.get("revision") or "")
    load_kwargs = dict(variant.get("load_model_kwargs")) if isinstance(variant.get("load_model_kwargs"), dict) else {}
    configured_load_revision = str(load_kwargs.get("revision") or "")
    if configured_load_revision and configured_load_revision != revision:
        raise RuntimeError(
            f"load_model revision {configured_load_revision} does not match configured revision {revision}"
        )
    load_kwargs["revision"] = revision
    started = time.perf_counter()
    model = load_model(model_id, **load_kwargs)
    load_seconds = time.perf_counter() - started
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise RuntimeError(
            f"loaded exact model {model_id!r}, but it does not expose a supported generate(audio, ...) API"
        )
    generation = variant.get("generation") if isinstance(variant.get("generation"), dict) else {}
    language = generation.get("language", "Chinese")

    def infer(path: Path, duration: float) -> dict[str, Any]:
        max_tokens = generation.get("max_tokens")
        if max_tokens is None:
            max_tokens = min(8192, max(256, int(math.ceil(duration * 12.0)) + 128))
        kwargs = {
            "language": language,
            "max_tokens": int(max_tokens),
            "temperature": float(generation.get("temperature", 0.0)),
            "repetition_penalty": float(generation.get("repetition_penalty", 1.1)),
            "repetition_context_size": int(generation.get("repetition_context_size", 256)),
            "chunk_duration": float(generation.get("chunk_duration", 1200.0)),
            "verbose": bool(generation.get("verbose", False)),
            "stream": False,
        }
        result = _call_with_supported_kwargs(generate, path, kwargs)
        if inspect.isgenerator(result):
            raise RuntimeError("mlx_audio generate returned a stream despite stream=False; refusing ambiguous output")
        return _extract_mlx_output(result, duration)

    return infer, load_seconds, {"provider": "huggingface", "configured_revision": revision}


def _load_worker_adapter(
    variant: dict[str, Any], repo_root: Path
) -> tuple[Callable[[Path, float], dict[str, Any]], float, dict[str, Any]]:
    adapter = str(variant.get("adapter") or "")
    if adapter == "localscribe":
        return _load_localscribe_adapter(variant, repo_root)
    if adapter == "mlx_audio":
        return _load_mlx_audio_adapter(variant, repo_root)
    raise ValueError(f"unsupported worker adapter: {adapter!r}")


def run_worker_request(request: dict[str, Any]) -> dict[str, Any]:
    started_at = utc_now()
    variant = request.get("variant") if isinstance(request.get("variant"), dict) else {}
    cases = request.get("cases") if isinstance(request.get("cases"), list) else []
    repeat = int(request.get("repeat") or 1)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "arena_worker_round",
        "variant": str(variant.get("name") or ""),
        "repeat": repeat,
        "required": bool(variant.get("required", True)),
        "adapter": variant.get("adapter"),
        "model_id": variant.get("model_id"),
        "revision": variant.get("revision"),
        "request_id": request.get("request_id"),
        "status": "error",
        "load_seconds": 0.0,
        "load_includes_download": False,
        "started_at": started_at,
        "completed_at": None,
        "cases": [],
        "error": "",
        "model_identity": {"configured_revision": variant.get("revision")},
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _package_versions(["scribe-py", "mlx-audio", "mlx", "funasr", "modelscope"]),
        },
    }
    repo_root = Path(str(request.get("repo_root") or ROOT)).resolve()
    _reset_accelerator_peak()
    try:
        expected_packages = variant.get("expected_packages")
        if isinstance(expected_packages, dict):
            for package_name, expected_version in expected_packages.items():
                try:
                    actual_version = importlib.metadata.version(str(package_name))
                except importlib.metadata.PackageNotFoundError as exc:
                    raise ModuleNotFoundError(
                        f"required package {package_name}=={expected_version} is not installed"
                    ) from exc
                if actual_version != str(expected_version):
                    raise RuntimeError(
                        f"required package version mismatch: {package_name}=={expected_version}, got {actual_version}"
                    )
        revision = str(variant.get("revision") or "")
        if variant.get("adapter") == "mlx_audio" or variant.get("backend") == "qwen3":
            result["model_identity"] = _verify_huggingface_main_revision(
                str(variant.get("model_id") or ""), revision
            )
        infer, load_seconds, loaded_identity = _load_worker_adapter(variant, repo_root)
        result["model_identity"].update(loaded_identity)
        result["load_seconds"] = round(load_seconds, 6)
    except BaseException as exc:
        if variant.get("backend") == "sensevoice":
            try:
                result["model_identity"].update(
                    _sensevoice_model_identity(
                        object(),
                        str(variant.get("model_id") or ""),
                        str(variant.get("revision") or ""),
                    )
                )
            except Exception as identity_exc:
                result["model_identity"]["identity_error"] = (
                    f"{type(identity_exc).__name__}: {identity_exc}"
                )
        status = _classify_exception(exc)
        message = f"{type(exc).__name__}: {exc}"
        result["status"] = status
        result["error"] = message
        result["traceback"] = traceback.format_exc()
        result["cases"] = [_empty_case_result(case, status, message) for case in cases]
        result["completed_at"] = utc_now()
        result["peak_rss_mb"] = peak_rss_mb()
        result["accelerator_memory"] = accelerator_memory()
        return result

    case_results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_result = _empty_case_result(case, "error")
        case_result["load_seconds"] = round(result["load_seconds"] if index == 0 else 0.0, 6)
        path = Path(str(case["clip_path"]))
        duration = case_result["audio_duration"]
        started = time.perf_counter()
        try:
            output = infer(path, duration)
            elapsed = time.perf_counter() - started
            case_result.update(
                {
                    "status": "ok",
                    "raw_text": str(output.get("raw_text") or ""),
                    "display_text": str(output.get("display_text") or ""),
                    "segments": output.get("segments") if isinstance(output.get("segments"), list) else [],
                    "timing_reliable": bool(output.get("timing_reliable", False)),
                    "alignment": dict(ALIGNMENT_NOT_RUN),
                    "inference_seconds": round(elapsed, 6),
                    "peak_rss_mb": peak_rss_mb(),
                    "accelerator_memory": accelerator_memory(),
                    "backend_metadata": output.get("backend_metadata") or {},
                    "error": "",
                }
            )
            case_result["normalized_text"] = normalize_text(case_result["display_text"])
        except BaseException as exc:
            elapsed = time.perf_counter() - started
            status = _classify_exception(exc)
            case_result.update(
                {
                    "status": status,
                    "inference_seconds": round(elapsed, 6),
                    "peak_rss_mb": peak_rss_mb(),
                    "accelerator_memory": accelerator_memory(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        case_results.append(case_result)

    result["cases"] = case_results
    result["status"] = _round_status(item["status"] for item in case_results)
    result["error"] = "; ".join(item["error"] for item in case_results if item.get("error"))
    result["completed_at"] = utc_now()
    result["peak_rss_mb"] = peak_rss_mb()
    result["accelerator_memory"] = accelerator_memory()
    return result


def worker_main(args: argparse.Namespace) -> int:
    request_path = Path(args.request).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    try:
        request = read_json(request_path)
        if not isinstance(request, dict):
            raise ValueError("worker request must be a JSON object")
        payload = run_worker_request(request)
    except BaseException as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "arena_worker_round",
            "variant": "",
            "repeat": 0,
            "required": True,
            "adapter": None,
            "model_id": None,
            "status": _classify_exception(exc),
            "load_seconds": 0.0,
            "started_at": utc_now(),
            "completed_at": utc_now(),
            "cases": [],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    atomic_write_json(output_path, payload)
    return 0


def _parse_variants(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    selected: set[str] = set()
    for value in values:
        selected.update(part.strip() for part in str(value).split(",") if part.strip())
    return selected


def _resolve_python(variant: dict[str, Any]) -> tuple[str | None, str | None]:
    env_name = str(variant.get("python_env_var") or "")
    configured = os.environ.get(env_name) if env_name else None
    if env_name and bool(variant.get("python_env_required", False)) and not configured:
        return None, f"required Python environment variable is unset: {env_name}"
    candidate = configured or variant.get("python") or sys.executable
    if not candidate:
        return None, f"no Python executable configured{f' via {env_name}' if env_name else ''}"
    expanded = os.path.expandvars(os.path.expanduser(str(candidate)))
    if os.path.sep in expanded:
        path = Path(expanded)
        if not path.is_file() or not os.access(path, os.X_OK):
            source = f"environment variable {env_name}" if configured else "configuration"
            return None, f"Python executable from {source} is unavailable: {expanded}"
        return os.path.abspath(path), None
    resolved = shutil.which(expanded)
    if not resolved:
        return None, f"Python executable is unavailable: {expanded}"
    return resolved, None


def _query_target_python_identity(python_executable: str) -> tuple[dict[str, Any] | None, str | None]:
    query_code = """
import importlib.metadata
import json
import os
import platform
import sys

packages = json.loads(sys.argv[1])
env_names = json.loads(sys.argv[2])
versions = {}
for name in packages:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
    except Exception as exc:
        versions[name] = f"error:{type(exc).__name__}"
print(json.dumps({
    "python": sys.version,
    "python_executable": sys.executable,
    "sys_prefix": sys.prefix,
    "sys_base_prefix": sys.base_prefix,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "packages": versions,
    "environment_variables": {name: os.environ.get(name) for name in env_names},
}, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [
                python_executable,
                "-c",
                query_code,
                json.dumps(list(_TARGET_PACKAGE_NAMES)),
                json.dumps(list(_INFERENCE_ENV_NAMES) + ["PYTHONNOUSERSITE"]),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=_worker_environment(),
        )
    except Exception as exc:
        return None, f"target Python identity query failed: {type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        detail = _truncate(completed.stderr or completed.stdout, 2000).strip()
        return None, f"target Python identity query exited {completed.returncode}: {detail}"
    try:
        payload = json.loads(completed.stdout)
    except Exception as exc:
        return None, f"target Python identity query returned invalid JSON: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict) or not _nonempty_text(payload.get("sys_prefix")):
        return None, "target Python identity query omitted sys_prefix"
    packages = payload.get("packages")
    environment = payload.get("environment_variables")
    if not isinstance(packages, dict) or not isinstance(environment, dict):
        return None, "target Python identity query omitted packages/environment_variables"
    return payload, None


def _worker_fingerprint(
    variant: dict[str, Any], python_executable: str | None, lock_input: dict[str, Any] | None
) -> dict[str, Any]:
    files = [Path(__file__).resolve()]
    if variant.get("adapter") == "localscribe":
        files.extend(
            [
                ROOT / "scribe-py" / "src" / "scribe_py" / "core" / "transcriber_base.py",
                ROOT / "scribe-py" / "src" / "scribe_py" / "core" / "types.py",
                ROOT / "scribe-py" / "src" / "scribe_py" / "core" / "audio.py",
                ROOT / "scribe-py" / "src" / "scribe_py" / "core" / "text_normalizer.py",
                ROOT / "scribe-py" / "src" / "scribe_py" / "core" / "sensevoice_recovery.py",
                ROOT / "scribe-py" / "src" / "scribe_py" / "core" / (
                    "transcriber_funasr.py" if variant.get("backend") == "sensevoice" else "transcriber_qwen3.py"
                ),
            ]
        )
    identities = []
    for path in files:
        if path.is_file():
            identities.append({"path": str(path), "sha256": sha256_file(path)})
    python_identity: dict[str, Any] = {"path": python_executable}
    if python_executable:
        try:
            stat = Path(python_executable).stat()
            python_identity.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        except OSError:
            pass
    target_identity: dict[str, Any] | None = None
    target_identity_error: str | None = "target Python executable is unavailable"
    if python_executable:
        target_identity, target_identity_error = _query_target_python_identity(python_executable)
    return {
        "python": python_identity,
        "target_python": target_identity,
        "target_python_query_ok": target_identity_error is None,
        "target_python_query_error": target_identity_error,
        "files": identities,
        "expected_packages": variant.get("expected_packages") or {},
        "lock_sha256": (lock_input or {}).get("sha256"),
    }


def _worker_request(
    variant: dict[str, Any],
    repeat: int,
    cases: list[dict[str, Any]],
    worker_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "arena_worker_request",
        "repo_root": str(ROOT),
        "variant": variant,
        "repeat": repeat,
        "worker_fingerprint": worker_fingerprint or {},
        "cases": [
            {
                "case_id": item["case_id"],
                "clip_path": item["clip_path"],
                "clip_sha256": item.get("clip_sha256"),
            }
            for item in cases
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["request_id"] = sha256_text(canonical)
    return payload


def _failure_round(
    variant: dict[str, Any],
    repeat: int,
    cases: list[dict[str, Any]],
    status: str,
    error: str,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "arena_worker_round",
        "variant": variant.get("name"),
        "repeat": repeat,
        "required": bool(variant.get("required", True)),
        "adapter": variant.get("adapter"),
        "model_id": variant.get("model_id"),
        "revision": variant.get("revision"),
        "request_id": request_id,
        "status": status,
        "load_seconds": 0.0,
        "load_includes_download": False,
        "started_at": utc_now(),
        "completed_at": utc_now(),
        "cases": [
            _empty_case_result(
                item,
                status,
                error,
                collect_accelerator_memory=False,
                peak_rss_value=None,
            )
            for item in cases
        ],
        "error": error,
        "model_identity": {"configured_revision": variant.get("revision")},
    }


def validate_worker_output(
    payload: Any,
    variant: dict[str, Any],
    repeat: int,
    cases: list[dict[str, Any]],
    request_id: str | None = None,
) -> str | None:
    if not isinstance(payload, dict):
        return "worker output is not a JSON object"
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != "arena_worker_round":
        return "worker output has an invalid schema_version/kind"
    if payload.get("variant") != variant.get("name"):
        return "worker output variant/repeat does not match the request"
    try:
        payload_repeat = int(payload.get("repeat"))
    except (TypeError, ValueError):
        return "worker output repeat is not an integer"
    if payload_repeat != repeat:
        return "worker output variant/repeat does not match the request"
    if payload.get("adapter") != variant.get("adapter") or payload.get("model_id") != variant.get("model_id"):
        return "worker output adapter/model_id does not match the request"
    if payload.get("revision") != variant.get("revision"):
        return "worker output configured revision does not match the request"
    if bool(payload.get("required")) != bool(variant.get("required", True)):
        return "worker output required policy does not match the request"
    if request_id is not None and payload.get("request_id") != request_id:
        return "worker output request_id does not match the current inputs/configuration"
    if payload.get("status") not in VALID_STATUSES:
        return f"worker output has invalid status: {payload.get('status')!r}"
    output_cases = payload.get("cases")
    if not isinstance(output_cases, list) or len(output_cases) != len(cases):
        return "worker output case count does not match the request"
    expected_ids = [item["case_id"] for item in cases]
    actual_ids = [item.get("case_id") if isinstance(item, dict) else None for item in output_cases]
    if actual_ids != expected_ids:
        return "worker output case IDs/order do not match the request"
    expected_paths = [str(item["clip_path"]) for item in cases]
    actual_paths = [str(item.get("clip_path")) if isinstance(item, dict) else None for item in output_cases]
    if actual_paths != expected_paths:
        return "worker output clip paths/order do not match the request"
    required_fields = {
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
        "status",
    }
    for item in output_cases:
        if not isinstance(item, dict):
            return "worker output contains a non-object case"
        missing = sorted(required_fields - set(item))
        if missing:
            return f"worker case {item.get('case_id')!r} is missing fields: {', '.join(missing)}"
        if item.get("status") not in VALID_STATUSES:
            return f"worker case {item.get('case_id')!r} has invalid status"
        alignment = item.get("alignment")
        if not isinstance(alignment, dict) or alignment.get("status") != "not_run":
            return f"worker case {item.get('case_id')!r} has invalid alignment schema"
        for field in ("load_seconds", "inference_seconds", "audio_duration"):
            value = item.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                return f"worker case {item.get('case_id')!r} has invalid {field}"
        peak_rss = item.get("peak_rss_mb")
        if item.get("status") == "ok" or peak_rss is not None:
            if (
                not isinstance(peak_rss, (int, float))
                or not math.isfinite(float(peak_rss))
                or float(peak_rss) < 0
            ):
                return f"worker case {item.get('case_id')!r} has invalid peak_rss_mb"
        if not isinstance(item.get("segments"), list) or not isinstance(item.get("accelerator_memory"), dict):
            return f"worker case {item.get('case_id')!r} has invalid segments/accelerator_memory"
    derived_status = _round_status(item["status"] for item in output_cases)
    if payload.get("status") != derived_status:
        return f"worker round status {payload.get('status')!r} disagrees with case status {derived_status!r}"
    return None


def run_worker_subprocess(
    python_executable: str,
    request_path: Path,
    output_path: Path,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [python_executable, str(Path(__file__).resolve()), "worker", "--request", str(request_path), "--output", str(output_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_worker_environment(),
    )
    return {
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "stdout": _truncate(completed.stdout),
        "stderr": _truncate(completed.stderr),
        "load_includes_download": _output_includes_download(completed.stdout, completed.stderr),
    }


def _output_includes_download(stdout: Any, stderr: Any) -> bool:
    output = f"{stdout or ''}\n{stderr or ''}"
    return bool(_DOWNLOAD_OUTPUT_RE.search(output))


def _process_output_includes_download(process: dict[str, Any]) -> bool:
    recorded = process.get("load_includes_download")
    if isinstance(recorded, bool):
        return recorded
    return _output_includes_download(process.get("stdout"), process.get("stderr"))


def _enrich_round(round_payload: dict[str, Any], gold_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    enriched = dict(round_payload)
    load_includes_download = bool(round_payload.get("load_includes_download", False))
    enriched_cases: list[dict[str, Any]] = []
    for raw_case in round_payload.get("cases") or []:
        case = dict(raw_case)
        gold = gold_by_id.get(str(case.get("case_id")))
        if gold:
            case["gold_text"] = gold["gold_text"]
            case["gold_normalized"] = gold["gold_normalized"]
        if case.get("status") == "ok" and gold:
            normalized = normalize_text(str(case.get("display_text") or ""))
            case["normalized_text"] = normalized
            case["raw_hash"] = sha256_text(str(case.get("raw_text") or ""))
            case["normalized_hash"] = sha256_text(normalized)
            case["cer"] = levenshtein_alignment(gold["gold_normalized"], normalized)
            duration = _finite_float(case.get("audio_duration"))
            inference = _finite_float(case.get("inference_seconds"))
            load = _finite_float(case.get("load_seconds"))
            case["rtf"] = inference / duration if duration > 0 else None
            case["cold_rtf"] = (
                (load + inference) / duration
                if duration > 0 and not load_includes_download
                else None
            )
        else:
            case.setdefault("normalized_text", "")
            case["raw_hash"] = None
            case["normalized_hash"] = None
            case["cer"] = None
            case["rtf"] = None
            case["cold_rtf"] = None
        enriched_cases.append(case)
    enriched["cases"] = enriched_cases
    return enriched


def _variant_summary(
    variant: dict[str, Any], rounds: list[dict[str, Any]], cases: list[dict[str, Any]], selected: bool
) -> dict[str, Any]:
    name = str(variant.get("name") or "")
    base = {
        "name": name,
        "required": bool(variant.get("required", True)),
        "selected": selected,
        "adapter": variant.get("adapter"),
        "backend": variant.get("backend"),
        "model_id": variant.get("model_id"),
        "revision": variant.get("revision"),
        "status": "skipped",
        "rounds": len(rounds),
        "cases_expected": len(cases) * len(rounds),
        "cases_ok": 0,
        "cases_failed": 0,
        "micro_cer": micro_cer([]),
        "rtf": None,
        "cold_rtf": None,
        "load_seconds": 0.0,
        "inference_seconds": 0.0,
        "audio_duration": 0.0,
        "peak_rss_mb": None,
        "accelerator_peak_mb": None,
        "dataset_raw_unique_output_count": 0,
        "dataset_normalized_unique_output_count": 0,
        "max_case_raw_unique_count": 0,
        "max_case_normalized_unique_count": 0,
        "max_self_cer": 0.0,
        "load_includes_download": False,
        "repeat_stability": [],
        "errors": [],
        "warnings": [],
    }
    if not selected:
        base["errors"] = ["variant not selected (optional/default-disabled or filtered by --variants)"]
        return base
    if not rounds:
        base["status"] = "invalid_output"
        base["errors"] = ["selected variant has no rounds"]
        return base

    all_cases = [item for round_payload in rounds for item in round_payload.get("cases") or []]
    base["status"] = _round_status(
        [round_payload.get("status", "invalid_output") for round_payload in rounds]
        + [item.get("status", "invalid_output") for item in all_cases]
    )
    ok_cases = [item for item in all_cases if item.get("status") == "ok"]
    base["cases_expected"] = len(cases) * len(rounds)
    base["cases_ok"] = len(ok_cases)
    base["cases_failed"] = len(all_cases) - len(ok_cases)
    if len(ok_cases) == base["cases_expected"] and all(round_payload.get("status") == "ok" for round_payload in rounds):
        base["status"] = "ok"
    base["micro_cer"] = micro_cer(item["cer"] for item in ok_cases if isinstance(item.get("cer"), dict))
    base["load_seconds"] = round(sum(_finite_float(item.get("load_seconds")) for item in rounds), 6)
    base["inference_seconds"] = round(sum(_finite_float(item.get("inference_seconds")) for item in ok_cases), 6)
    base["audio_duration"] = round(sum(_finite_float(item.get("audio_duration")) for item in ok_cases), 6)
    base["load_includes_download"] = any(
        bool(round_payload.get("load_includes_download", False)) for round_payload in rounds
    )
    if base["audio_duration"] > 0:
        base["rtf"] = base["inference_seconds"] / base["audio_duration"]
        if not base["load_includes_download"]:
            base["cold_rtf"] = (base["load_seconds"] + base["inference_seconds"]) / base["audio_duration"]
    rss_values = [_finite_float(item.get("peak_rss_mb"), -1.0) for item in all_cases]
    rss_values = [value for value in rss_values if value >= 0]
    base["peak_rss_mb"] = max(rss_values) if rss_values else None
    accel_values = [
        _finite_float((item.get("accelerator_memory") or {}).get("peak_mb"), -1.0)
        for item in all_cases
        if isinstance(item.get("accelerator_memory"), dict)
    ]
    accel_values = [value for value in accel_values if value >= 0]
    base["accelerator_peak_mb"] = max(accel_values) if accel_values else None
    raw_hashes = {str(item.get("raw_hash")) for item in ok_cases if item.get("raw_hash")}
    normalized_hashes = {str(item.get("normalized_hash")) for item in ok_cases if item.get("normalized_hash")}
    base["dataset_raw_unique_output_count"] = len(raw_hashes)
    base["dataset_normalized_unique_output_count"] = len(normalized_hashes)

    by_case: dict[str, list[dict[str, Any]]] = {}
    for item in ok_cases:
        by_case.setdefault(str(item.get("case_id")), []).append(item)
    stability: list[dict[str, Any]] = []
    for gold_case in cases:
        case_id = gold_case["case_id"]
        outputs = by_case.get(case_id, [])
        raw_values = [str(item.get("raw_text") or "") for item in outputs]
        normalized_values = [str(item.get("normalized_text") or "") for item in outputs]
        maximum = 0.0
        for left_index in range(len(normalized_values)):
            for right_index in range(left_index + 1, len(normalized_values)):
                maximum = max(maximum, self_cer(normalized_values[left_index], normalized_values[right_index]))
        stability.append(
            {
                "case_id": case_id,
                "repeat_count": len(outputs),
                "raw_hashes": [sha256_text(value) for value in raw_values],
                "normalized_hashes": [sha256_text(value) for value in normalized_values],
                "raw_unique_count": len(set(raw_values)),
                "normalized_unique_count": len(set(normalized_values)),
                "max_self_cer": maximum,
            }
        )
    base["repeat_stability"] = stability
    base["max_case_raw_unique_count"] = max((item["raw_unique_count"] for item in stability), default=0)
    base["max_case_normalized_unique_count"] = max(
        (item["normalized_unique_count"] for item in stability), default=0
    )
    base["max_self_cer"] = max((item["max_self_cer"] for item in stability), default=0.0)
    errors: list[str] = []
    for round_payload in rounds:
        if round_payload.get("error"):
            errors.append(f"repeat {round_payload.get('repeat')}: {round_payload['error']}")
        for item in round_payload.get("cases") or []:
            if item.get("status") != "ok" and item.get("error"):
                errors.append(f"repeat {round_payload.get('repeat')} case {item.get('case_id')}: {item['error']}")
    base["errors"] = list(dict.fromkeys(errors))
    if base["load_includes_download"]:
        base["warnings"] = [
            "Cold RTF omitted because at least one round logged model fetching/downloading; "
            "network transfer is not local model load time."
        ]
    return base


def aggregate_results(
    config: dict[str, Any],
    cases: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    selected_names: set[str],
    repeats: int,
) -> dict[str, Any]:
    gold_by_id = {item["case_id"]: item for item in cases}
    enriched_rounds = [_enrich_round(item, gold_by_id) for item in rounds]
    summaries: list[dict[str, Any]] = []
    for variant in config.get("variants") or []:
        name = str(variant.get("name") or "")
        variant_rounds = [item for item in enriched_rounds if item.get("variant") == name]
        summaries.append(_variant_summary(variant, variant_rounds, cases, name in selected_names))
    required_failures = [
        item["name"]
        for item in summaries
        if item["required"] and item["selected"] and item["status"] != "ok"
    ]
    optional_failures = [
        item["name"]
        for item in summaries
        if not item["required"] and item["selected"] and item["status"] != "ok"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "arena_results",
        "generated_at": utc_now(),
        "status": "failed" if required_failures else "ok",
        "repeats": repeats,
        "case_count": len(cases),
        "selected_variants": sorted(selected_names),
        "policy": {
            "required_failures_are_fatal": True,
            "optional_failures_are_fatal": False,
            "required_failures": required_failures,
            "optional_failures": optional_failures,
        },
        "normalization": {
            "unicode": "NFKC",
            "traditional_to_simplified": "opencc-t2s-if-available",
            "lowercase": True,
            "ignored": ["unicode punctuation", "whitespace", "unicode separators"],
            "opencc_available": _get_opencc_converter() is not None,
        },
        "cases": [
            {
                "case_id": item["case_id"],
                "clip_path": item["clip_path"],
                "clip_path_source": item.get("clip_path_source"),
                "gold_text": item["gold_text"],
                "gold_text_source": item.get("gold_text_source"),
                "gold_normalized": item["gold_normalized"],
            }
            for item in cases
        ],
        "variants": summaries,
        "rounds": enriched_rounds,
    }


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def results_tsv(results: dict[str, Any]) -> str:
    fields = [
        "variant",
        "required",
        "selected",
        "status",
        "model_id",
        "cases_ok",
        "cases_expected",
        "micro_cer",
        "substitutions",
        "deletions",
        "insertions",
        "rtf",
        "cold_rtf",
        "load_seconds",
        "inference_seconds",
        "peak_rss_mb",
        "accelerator_peak_mb",
        "dataset_raw_unique_output_count",
        "dataset_normalized_unique_output_count",
        "max_case_raw_unique_count",
        "max_case_normalized_unique_count",
        "max_self_cer",
        "load_includes_download",
        "warnings",
        "errors",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for item in results.get("variants") or []:
        cer = item.get("micro_cer") or {}
        writer.writerow(
            {
                "variant": item.get("name"),
                "required": str(bool(item.get("required"))).lower(),
                "selected": str(bool(item.get("selected"))).lower(),
                "status": item.get("status"),
                "model_id": item.get("model_id"),
                "cases_ok": item.get("cases_ok"),
                "cases_expected": item.get("cases_expected"),
                "micro_cer": _format_number(cer.get("cer")),
                "substitutions": cer.get("substitutions"),
                "deletions": cer.get("deletions"),
                "insertions": cer.get("insertions"),
                "rtf": _format_number(item.get("rtf")),
                "cold_rtf": _format_number(item.get("cold_rtf")),
                "load_seconds": _format_number(item.get("load_seconds"), 3),
                "inference_seconds": _format_number(item.get("inference_seconds"), 3),
                "peak_rss_mb": _format_number(item.get("peak_rss_mb"), 1),
                "accelerator_peak_mb": _format_number(item.get("accelerator_peak_mb"), 1),
                "dataset_raw_unique_output_count": item.get("dataset_raw_unique_output_count"),
                "dataset_normalized_unique_output_count": item.get("dataset_normalized_unique_output_count"),
                "max_case_raw_unique_count": item.get("max_case_raw_unique_count"),
                "max_case_normalized_unique_count": item.get("max_case_normalized_unique_count"),
                "max_self_cer": _format_number(item.get("max_self_cer")),
                "load_includes_download": str(bool(item.get("load_includes_download"))).lower(),
                "warnings": " | ".join(item.get("warnings") or []),
                "errors": " | ".join(item.get("errors") or []),
            }
        )
    return buffer.getvalue()


def results_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# ASR Model Arena — Phase 1\n\n",
        f"- Status: **{results.get('status')}**\n",
        f"- Cases: {results.get('case_count')}\n",
        f"- Repeats: {results.get('repeats')}\n",
        "- Alignment: `not_run` (phase-one text-quality arena)\n",
        "- Failure policy: required failures are fatal; optional failures are reported but non-fatal.\n\n",
        "## Results\n\n",
        "| Variant | Required | Status | Cases | Micro CER | S/D/I | RTF | Cold RTF | Peak RSS MB | Accelerator MB | Max case raw unique | Max case normalized unique | Max self CER | Download in load |\n",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for item in results.get("variants") or []:
        cer = item.get("micro_cer") or {}
        lines.append(
            "| {name} | {required} | {status} | {cases_ok}/{cases_expected} | {micro} | "
            "{s}/{d}/{i} | {rtf} | {cold} | {rss} | {accel} | {raw} | {norm} | {self_cer} | {download} |\n".format(
                name=item.get("name"),
                required="yes" if item.get("required") else "no",
                status=item.get("status"),
                cases_ok=item.get("cases_ok"),
                cases_expected=item.get("cases_expected"),
                micro=_format_number(cer.get("cer")),
                s=cer.get("substitutions", 0),
                d=cer.get("deletions", 0),
                i=cer.get("insertions", 0),
                rtf=_format_number(item.get("rtf")),
                cold=_format_number(item.get("cold_rtf")),
                rss=_format_number(item.get("peak_rss_mb"), 1),
                accel=_format_number(item.get("accelerator_peak_mb"), 1),
                raw=item.get("max_case_raw_unique_count"),
                norm=item.get("max_case_normalized_unique_count"),
                self_cer=_format_number(item.get("max_self_cer")),
                download="yes" if item.get("load_includes_download") else "no",
            )
        )
    download_warnings = [item for item in results.get("variants") or [] if item.get("load_includes_download")]
    lines.extend(["\n## Load timing warnings\n\n"])
    if not download_warnings:
        lines.append("None.\n")
    else:
        for item in download_warnings:
            lines.append(
                f"- **{item.get('name')}**: Cold RTF is omitted because at least one round logged "
                "Fetching/Downloading; network transfer is not a valid local model load measurement.\n"
            )
    lines.extend(["\n## Repeat stability\n\n"])
    for item in results.get("variants") or []:
        lines.append(f"### {item.get('name')}\n\n")
        if not item.get("repeat_stability"):
            lines.append(f"- Status: {item.get('status')}\n\n")
            continue
        lines.extend(
            [
                "| Case | Repeats | Raw unique count | Normalized unique count | Max self CER |\n",
                "|---|---:|---:|---:|---:|\n",
            ]
        )
        for stability in item["repeat_stability"]:
            lines.append(
                f"| {stability['case_id']} | {stability['repeat_count']} | {stability['raw_unique_count']} | "
                f"{stability['normalized_unique_count']} | {_format_number(stability['max_self_cer'])} |\n"
            )
        lines.append("\n")
    failures = [item for item in results.get("variants") or [] if item.get("errors")]
    lines.extend(["## Failures and skips\n\n"])
    if not failures:
        lines.append("None.\n")
    else:
        for item in failures:
            lines.append(f"### {item.get('name')} ({item.get('status')})\n\n")
            for error in item.get("errors") or []:
                lines.append(f"- {str(error).replace(chr(10), ' ')}\n")
            lines.append("\n")
    lines.extend(
        [
            "## Metric definitions\n\n",
            "- **Micro CER** = total `(S + D + I)` divided by total normalized reference characters.\n",
            "- **RTF** = total inference seconds divided by total audio duration.\n",
            "- **Cold RTF** = `(model load seconds + inference seconds)` divided by total audio duration; omitted if any round included fetching/downloading.\n",
            "- **Dataset unique output counts** count distinct outputs across all cases and repeats; they are not a repeat-stability metric.\n",
            "- **Max self CER** = maximum pairwise repeat edit distance divided by the longer normalized output length.\n",
        ]
    )
    return "".join(lines)


def _load_config(config_path: Path) -> dict[str, Any]:
    payload = read_json(config_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("variants"), list):
        raise ValueError("arena config must be a JSON object containing a variants array")
    if payload.get("schema_version") != "asr-model-arena-config-v1":
        raise ValueError("arena config schema_version must be asr-model-arena-config-v1")
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("arena config defaults must be an object")
    if "repeats" in defaults and (not isinstance(defaults["repeats"], int) or defaults["repeats"] < 1):
        raise ValueError("arena config defaults.repeats must be a positive integer")
    if "timeout_seconds" in defaults and (
        not isinstance(defaults["timeout_seconds"], (int, float)) or defaults["timeout_seconds"] <= 0
    ):
        raise ValueError("arena config defaults.timeout_seconds must be positive")
    names: set[str] = set()
    slugs: set[str] = set()
    for index, variant in enumerate(payload["variants"]):
        if not isinstance(variant, dict):
            raise ValueError(f"config variants[{index}] must be an object")
        name = str(variant.get("name") or "")
        if not name or name in names:
            raise ValueError(f"config variant name is missing or duplicated: {name!r}")
        names.add(name)
        slug = safe_slug(name)
        if slug in slugs:
            raise ValueError(f"config variant names collide after path sanitization: {name!r}")
        slugs.add(slug)
        adapter = variant.get("adapter")
        if adapter not in {"localscribe", "mlx_audio"}:
            raise ValueError(f"variant {name!r} has unsupported adapter {adapter!r}")
        if not isinstance(variant.get("model_id"), str) or not variant["model_id"].strip():
            raise ValueError(f"variant {name!r} requires a non-empty model_id")
        if not isinstance(variant.get("revision"), str) or not variant["revision"].strip():
            raise ValueError(f"variant {name!r} requires a non-empty revision")
        for boolean_field in ("required", "enabled_by_default", "python_env_required"):
            if boolean_field in variant and not isinstance(variant[boolean_field], bool):
                raise ValueError(f"variant {name!r} field {boolean_field} must be boolean")
        for object_field in ("options", "generation", "load_model_kwargs"):
            if object_field in variant and not isinstance(variant[object_field], dict):
                raise ValueError(f"variant {name!r} field {object_field} must be an object")
        expected_packages = variant.get("expected_packages", {})
        if not isinstance(expected_packages, dict) or any(
            not isinstance(package, str)
            or not package.strip()
            or not isinstance(version, str)
            or not version.strip()
            for package, version in expected_packages.items()
        ):
            raise ValueError(f"variant {name!r} expected_packages must map package names to exact versions")
        if adapter == "localscribe" and variant.get("backend") not in {"sensevoice", "qwen3"}:
            raise ValueError(f"variant {name!r} localscribe backend must be sensevoice or qwen3")
        if variant.get("python_env_required") and not variant.get("python_env_var"):
            raise ValueError(f"variant {name!r} requires python_env_var when python_env_required is true")
        if adapter == "mlx_audio":
            load_revision = str((variant.get("load_model_kwargs") or {}).get("revision") or "")
            if load_revision != variant["revision"]:
                raise ValueError(
                    f"variant {name!r} load_model_kwargs.revision must match revision"
                )
    return payload


def run_arena(
    *,
    config_path: Path,
    gold_path: Path,
    out_dir: Path,
    repeats: int | None = None,
    variants: Sequence[str] | None = None,
    case_limit: int | None = None,
    timeout: float | None = None,
    resume: bool = False,
) -> tuple[int, dict[str, Any]]:
    config_path = config_path.expanduser().resolve()
    gold_path = gold_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    config = _load_config(config_path)
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    repeat_count = int(repeats if repeats is not None else defaults.get("repeats", 1))
    worker_timeout = float(timeout if timeout is not None else defaults.get("timeout_seconds", 1800))
    if repeat_count < 1:
        raise ValueError("repeats must be at least 1")
    if worker_timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if case_limit is not None and case_limit < 1:
        raise ValueError("case-limit must be at least 1")
    cases = load_gold_items(gold_path, case_limit)

    requested = _parse_variants(variants)
    config_names = {str(item["name"]) for item in config["variants"]}
    if requested is not None:
        unknown = sorted(requested - config_names)
        if unknown:
            raise ValueError(f"unknown variants: {', '.join(unknown)}")
        selected_names = requested
    else:
        selected_names = {
            str(item["name"])
            for item in config["variants"]
            if bool(item.get("enabled_by_default", True))
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    requests_dir = out_dir / "requests"
    runs_dir = out_dir / "runs"
    requests_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "arena_run_manifest",
        "started_at": started_at,
        "completed_at": None,
        "status": "running",
        "arguments": {
            "config": str(config_path),
            "gold": str(gold_path),
            "out_dir": str(out_dir),
            "repeats": repeat_count,
            "variants": sorted(selected_names),
            "case_limit": case_limit,
            "timeout_seconds": worker_timeout,
            "resume": resume,
        },
        "inputs": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "gold": {"path": str(gold_path), "sha256": sha256_file(gold_path)},
            "lock": None,
            "clips": [
                {
                    "case_id": item["case_id"],
                    "path": item["clip_path"],
                    "path_source": item.get("clip_path_source"),
                    "sha256": item["clip_sha256"],
                    "gold_text_source": item.get("gold_text_source"),
                }
                for item in cases
            ],
        },
        "git": _git_info(ROOT),
        "environment": _relevant_environment(config),
        "resolved_variants": [],
    }
    atomic_write_json(out_dir / "run_manifest.json", manifest)

    try:
        lock_value = config.get("lock_file")
        if lock_value:
            lock_path = Path(str(lock_value)).expanduser()
            if not lock_path.is_absolute():
                lock_path = (config_path.parent / lock_path).resolve()
            if not lock_path.is_file():
                raise FileNotFoundError(f"configured lock file not found: {lock_path}")
            manifest["inputs"]["lock"] = {"path": str(lock_path), "sha256": sha256_file(lock_path)}
        rounds: list[dict[str, Any]] = []
        for variant in config["variants"]:
            name = str(variant["name"])
            python_executable, python_error = _resolve_python(variant)
            selected = name in selected_names
            resolved_variant = {
                "name": name,
                "selected": selected,
                "required": bool(variant.get("required", True)),
                "python_env_var": variant.get("python_env_var"),
                "python_env_required": bool(variant.get("python_env_required", False)),
                "python_executable": python_executable,
                "python_error": python_error,
                "model_id": variant.get("model_id"),
                "revision": variant.get("revision"),
                "target_python": None,
                "resume_identity_query_ok": None,
                "resume_identity_query_error": None,
            }
            manifest["resolved_variants"].append(resolved_variant)
            if not selected:
                continue
            variant_dir = runs_dir / safe_slug(name)
            variant_dir.mkdir(parents=True, exist_ok=True)
            fingerprint = _worker_fingerprint(
                variant, python_executable, manifest["inputs"].get("lock")
            )
            resolved_variant["target_python"] = fingerprint.get("target_python")
            resolved_variant["resume_identity_query_ok"] = fingerprint.get("target_python_query_ok")
            resolved_variant["resume_identity_query_error"] = fingerprint.get("target_python_query_error")
            resume_identity_safe = bool(fingerprint.get("target_python_query_ok"))
            for repeat in range(1, repeat_count + 1):
                output_path = variant_dir / f"repeat_{repeat:03d}.json"
                request_path = requests_dir / f"{safe_slug(name)}_repeat_{repeat:03d}.json"
                request_payload = _worker_request(variant, repeat, cases, fingerprint)
                atomic_write_json(request_path, request_payload)
                if resume and resume_identity_safe and output_path.is_file():
                    try:
                        cached = read_json(output_path)
                        problem = validate_worker_output(
                            cached, variant, repeat, cases, request_payload["request_id"]
                        )
                        if problem is None:
                            cached["resumed"] = True
                            rounds.append(cached)
                            continue
                    except Exception:
                        pass
                output_path.unlink(missing_ok=True)
                if python_error or not python_executable:
                    payload = _failure_round(
                        variant,
                        repeat,
                        cases,
                        "unavailable",
                        python_error or "Python unavailable",
                        request_id=request_payload["request_id"],
                    )
                    atomic_write_json(output_path, payload)
                    rounds.append(payload)
                    continue
                try:
                    process = run_worker_subprocess(python_executable, request_path, output_path, worker_timeout)
                except subprocess.TimeoutExpired as exc:
                    payload = _failure_round(
                        variant,
                        repeat,
                        cases,
                        "timeout",
                        f"worker exceeded timeout of {worker_timeout:g}s",
                        request_id=request_payload["request_id"],
                    )
                    payload["process"] = {
                        "returncode": None,
                        "elapsed_seconds": worker_timeout,
                        "stdout": _truncate(exc.stdout),
                        "stderr": _truncate(exc.stderr),
                        "load_includes_download": _output_includes_download(exc.stdout, exc.stderr),
                    }
                    payload["load_includes_download"] = _process_output_includes_download(payload["process"])
                    atomic_write_json(output_path, payload)
                    rounds.append(payload)
                    continue
                except Exception as exc:
                    status = "unavailable" if isinstance(exc, OSError) else "error"
                    payload = _failure_round(
                        variant,
                        repeat,
                        cases,
                        status,
                        f"worker launch failed: {type(exc).__name__}: {exc}",
                        request_id=request_payload["request_id"],
                    )
                    payload["process"] = {
                        "returncode": None,
                        "elapsed_seconds": 0.0,
                        "stdout": "",
                        "stderr": _truncate(str(exc)),
                    }
                    payload["load_includes_download"] = _process_output_includes_download(payload["process"])
                    atomic_write_json(output_path, payload)
                    rounds.append(payload)
                    continue
                try:
                    payload = read_json(output_path)
                except Exception as exc:
                    returncode = int(process.get("returncode") or 0)
                    status = "oom" if returncode in {-9, 137} else "invalid_output"
                    payload = _failure_round(
                        variant,
                        repeat,
                        cases,
                        status,
                        f"worker did not produce readable JSON: {type(exc).__name__}: {exc}",
                        request_id=request_payload["request_id"],
                    )
                problem = validate_worker_output(
                    payload, variant, repeat, cases, request_payload["request_id"]
                )
                returncode = int(process.get("returncode") or 0)
                if returncode != 0 and problem is None:
                    problem = f"worker exited with nonzero status {returncode}"
                if problem is not None:
                    status = "oom" if returncode in {-9, 137} else "invalid_output"
                    payload = _failure_round(
                        variant,
                        repeat,
                        cases,
                        status,
                        problem,
                        request_id=request_payload["request_id"],
                    )
                payload["process"] = process
                payload["load_includes_download"] = _process_output_includes_download(process)
                atomic_write_json(output_path, payload)
                rounds.append(payload)

        results = aggregate_results(config, cases, rounds, selected_names, repeat_count)
        results["config"] = {"path": str(config_path), "sha256": manifest["inputs"]["config"]["sha256"]}
        results["gold"] = {"path": str(gold_path), "sha256": manifest["inputs"]["gold"]["sha256"]}
        atomic_write_json(out_dir / "arena_results.json", results)
        atomic_write_text(out_dir / "arena_results.tsv", results_tsv(results))
        atomic_write_text(out_dir / "arena_report.md", results_markdown(results))
        manifest["completed_at"] = utc_now()
        manifest["status"] = results["status"]
        manifest["outputs"] = {
            "results_json": {
                "path": str(out_dir / "arena_results.json"),
                "sha256": sha256_file(out_dir / "arena_results.json"),
            },
            "results_tsv": {
                "path": str(out_dir / "arena_results.tsv"),
                "sha256": sha256_file(out_dir / "arena_results.tsv"),
            },
            "report_markdown": {
                "path": str(out_dir / "arena_report.md"),
                "sha256": sha256_file(out_dir / "arena_report.md"),
            },
        }
        atomic_write_json(out_dir / "run_manifest.json", manifest)
        return (0 if results["status"] == "ok" else 1), results
    except Exception as exc:
        manifest["completed_at"] = utc_now()
        manifest["status"] = "failed"
        manifest["error"] = _truncate(f"{type(exc).__name__}: {exc}")
        atomic_write_json(out_dir / "run_manifest.json", manifest)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated ASR model arena variants against human gold clips.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="orchestrate arena workers")
    run_parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="arena variant JSON")
    run_parser.add_argument("--gold", required=True, help="human gold JSON containing items")
    run_parser.add_argument("--out-dir", required=True, help="output directory")
    run_parser.add_argument("--repeats", type=int, default=None)
    run_parser.add_argument("--variants", nargs="+", default=None, help="variant names (space- or comma-separated)")
    run_parser.add_argument("--case-limit", type=int, default=None)
    run_parser.add_argument("--timeout", type=float, default=None, help="seconds per variant/repeat worker")
    run_parser.add_argument("--resume", action="store_true")
    worker_parser = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--request", required=True)
    worker_parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    actual = list(sys.argv[1:] if argv is None else argv)
    if not actual or actual[0] not in {"run", "worker", "-h", "--help"}:
        actual.insert(0, "run")
    parser = _build_parser()
    args = parser.parse_args(actual)
    if args.command == "worker":
        return worker_main(args)
    try:
        exit_code, results = run_arena(
            config_path=Path(args.config),
            gold_path=Path(args.gold),
            out_dir=Path(args.out_dir),
            repeats=args.repeats,
            variants=args.variants,
            case_limit=args.case_limit,
            timeout=args.timeout,
            resume=args.resume,
        )
    except Exception as exc:
        parser.error(f"{type(exc).__name__}: {exc}")
        return 2
    print(f"[asr-model-arena] status={results['status']} out={Path(args.out_dir).expanduser().resolve()}")
    for item in results.get("variants") or []:
        print(
            f"  {item['name']}: {item['status']} "
            f"cases={item['cases_ok']}/{item['cases_expected']} "
            f"micro_cer={_format_number((item.get('micro_cer') or {}).get('cer'))}"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
