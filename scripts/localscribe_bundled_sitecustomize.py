"""Offline-only Hugging Face setup installed as bundled Python's sitecustomize."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import MutableMapping


MODEL_ID = "mlx-community/Qwen3-ASR-1.7B-8bit"
MANIFEST_RELATIVE = Path("huggingface/offline-asr-manifest.json")


def configure_bundled_runtime_cache(
    environ: MutableMapping[str, str] | None = None,
    *,
    home: Path | None = None,
) -> bool:
    """Keep runtime-generated caches outside the signed App bundle."""
    env = os.environ if environ is None else environ
    if not env.get("LOCALSCRIBE_RESOURCES", "").strip():
        return False

    cache_root = Path(
        env.get("LOCALSCRIBE_CACHE_DIR", "").strip()
        or ((home or Path.home()) / "Library/Caches/LocalScribe")
    ).expanduser()
    env["LOCALSCRIBE_CACHE_DIR"] = str(cache_root)
    env["LOCALSCRIBE_SENKO_CACHE_DIR"] = str(cache_root / "senko/coreml")
    env["NUMBA_CACHE_DIR"] = str(cache_root / "numba")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if environ is None:
        sys.dont_write_bytecode = True
    return True


def _manifest_is_complete(resources: Path) -> tuple[bool, str]:
    manifest_path = resources / MANIFEST_RELATIVE
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "manifest_missing_or_invalid"
    if manifest.get("format_version") != 1 or manifest.get("model_id") != MODEL_ID:
        return False, "manifest_model_mismatch"
    entries = manifest.get("required_files")
    if not isinstance(entries, list) or not entries:
        return False, "manifest_files_missing"
    root = resources.resolve()
    for entry in entries:
        if not isinstance(entry, dict):
            return False, "manifest_file_invalid"
        try:
            path = (resources / str(entry.get("relative_path") or "")).resolve(strict=True)
            path.relative_to(root)
            expected_size = int(entry.get("size") or -1)
        except (OSError, ValueError, TypeError):
            return False, "model_file_missing"
        if not path.is_file() or path.stat().st_size != expected_size:
            return False, "model_file_size_mismatch"
    if importlib.util.find_spec("mlx_audio") is None:
        return False, "mlx_audio_missing"
    return True, "ready"


def configure_bundled_offline_asr(
    environ: MutableMapping[str, str] | None = None,
) -> tuple[bool, str]:
    env = os.environ if environ is None else environ
    raw_resources = env.get("LOCALSCRIBE_RESOURCES", "").strip()
    if not raw_resources:
        return False, "not_bundled"
    resources = Path(raw_resources).expanduser()
    ready, reason = _manifest_is_complete(resources)

    # A customer build must never fetch models at runtime. Invalid or damaged
    # bundles point at an empty cache so the existing review path safely skips.
    hf_home = resources / ("huggingface" if ready else "huggingface-disabled")
    hub = hf_home / "hub"
    env["HF_HOME"] = str(hf_home)
    env["HF_HUB_CACHE"] = str(hub)
    env["HUGGINGFACE_HUB_CACHE"] = str(hub)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["LOCALSCRIBE_OFFLINE_QWEN_AVAILABLE"] = "1" if ready else "0"
    env["LOCALSCRIBE_OFFLINE_QWEN_REASON"] = reason
    return ready, reason


configure_bundled_runtime_cache()
configure_bundled_offline_asr()
