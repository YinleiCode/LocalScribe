#!/usr/bin/env python3
"""Build-time integrity manifest and verifier for bundled offline Qwen3-ASR."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


MODEL_ID = "mlx-community/Qwen3-ASR-1.7B-8bit"
MANIFEST_RELATIVE = Path("huggingface/offline-asr-manifest.json")
DEFAULT_MINIMUM_MODEL_BYTES = 1_000_000_000
REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
    "model.safetensors",
    "model.safetensors.index.json",
)
_REVISION_RE = re.compile(r"^[0-9A-Za-z._-]+$")


class BundleValidationError(RuntimeError):
    pass


def cache_dir_name(model_id: str = MODEL_ID) -> str:
    return "models--" + model_id.replace("/", "--")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_revision(value: str) -> str:
    revision = value.strip()
    if not revision or not _REVISION_RE.fullmatch(revision):
        raise BundleValidationError(f"invalid Hugging Face revision: {revision!r}")
    return revision


def resolve_snapshot(hub_root: Path, model_id: str = MODEL_ID) -> tuple[Path, str]:
    cache_dir = hub_root / cache_dir_name(model_id)
    ref = cache_dir / "refs/main"
    if not ref.is_file():
        raise BundleValidationError(f"missing model ref: {ref}")
    revision = _safe_revision(ref.read_text(encoding="utf-8"))
    snapshot = cache_dir / "snapshots" / revision
    if not snapshot.is_dir():
        raise BundleValidationError(f"missing model snapshot: {snapshot}")
    return snapshot, revision


def _validate_snapshot_files(
    snapshot: Path,
    *,
    minimum_model_bytes: int = DEFAULT_MINIMUM_MODEL_BYTES,
) -> list[Path]:
    files: list[Path] = []
    for name in REQUIRED_SNAPSHOT_FILES:
        path = snapshot / name
        if not path.is_file():
            raise BundleValidationError(f"missing Qwen3-ASR file: {path}")
        try:
            path.resolve(strict=True)
        except OSError as exc:
            raise BundleValidationError(f"broken Qwen3-ASR file link: {path}: {exc}") from exc
        if path.stat().st_size <= 0:
            raise BundleValidationError(f"empty Qwen3-ASR file: {path}")
        files.append(path)

    model_path = snapshot / "model.safetensors"
    if model_path.stat().st_size < minimum_model_bytes:
        raise BundleValidationError(
            f"Qwen3-ASR weights are incomplete: {model_path} "
            f"({model_path.stat().st_size} < {minimum_model_bytes} bytes)"
        )

    try:
        index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"invalid Qwen3-ASR weight index: {exc}") from exc
    referenced = set((index.get("weight_map") or {}).values())
    if referenced != {"model.safetensors"}:
        raise BundleValidationError(f"unexpected Qwen3-ASR weight files: {sorted(referenced)}")
    return files


def write_manifest(
    resources: Path,
    *,
    model_id: str = MODEL_ID,
    minimum_model_bytes: int = DEFAULT_MINIMUM_MODEL_BYTES,
) -> dict[str, Any]:
    resources = resources.expanduser().resolve()
    hub_root = resources / "huggingface/hub"
    snapshot, revision = resolve_snapshot(hub_root, model_id)
    files = _validate_snapshot_files(snapshot, minimum_model_bytes=minimum_model_bytes)
    entries: list[dict[str, Any]] = []
    for path in files:
        entries.append(
            {
                "relative_path": path.relative_to(resources).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest: dict[str, Any] = {
        "format_version": 1,
        "feature": "offline_high_risk_asr_review",
        "model_id": model_id,
        "revision": revision,
        "cache_root_relative": "huggingface/hub",
        "model_cache_relative": f"huggingface/hub/{cache_dir_name(model_id)}",
        "snapshot_relative": snapshot.relative_to(resources).as_posix(),
        "required_files": entries,
        "model_bytes": (snapshot / "model.safetensors").stat().st_size,
        "total_required_bytes": sum(int(item["size"]) for item in entries),
    }
    manifest_path = resources / MANIFEST_RELATIVE
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _inside_resources(resources: Path, relative: str) -> Path:
    candidate = (resources / relative).resolve(strict=True)
    try:
        candidate.relative_to(resources.resolve())
    except ValueError as exc:
        raise BundleValidationError(f"manifest path escapes Resources: {relative}") from exc
    return candidate


def verify_manifest(
    resources: Path,
    *,
    check_hash: bool = False,
    minimum_model_bytes: int = DEFAULT_MINIMUM_MODEL_BYTES,
) -> dict[str, Any]:
    resources = resources.expanduser().resolve()
    manifest_path = resources / MANIFEST_RELATIVE
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"missing or invalid offline ASR manifest: {manifest_path}: {exc}") from exc

    if manifest.get("format_version") != 1:
        raise BundleValidationError("unsupported offline ASR manifest version")
    if manifest.get("model_id") != MODEL_ID:
        raise BundleValidationError(f"unexpected offline ASR model: {manifest.get('model_id')!r}")

    hub_root = _inside_resources(resources, str(manifest.get("cache_root_relative") or ""))
    snapshot, revision = resolve_snapshot(hub_root, MODEL_ID)
    if revision != manifest.get("revision"):
        raise BundleValidationError("offline ASR manifest revision does not match refs/main")
    expected_snapshot = _inside_resources(resources, str(manifest.get("snapshot_relative") or ""))
    if snapshot.resolve() != expected_snapshot:
        raise BundleValidationError("offline ASR snapshot path does not match manifest")

    snapshot_files = _validate_snapshot_files(snapshot, minimum_model_bytes=minimum_model_bytes)
    entries = manifest.get("required_files")
    if not isinstance(entries, list) or len(entries) != len(REQUIRED_SNAPSHOT_FILES):
        raise BundleValidationError("offline ASR manifest has an incomplete required_files list")
    manifest_names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise BundleValidationError("invalid offline ASR manifest file entry")
        relative = str(entry.get("relative_path") or "")
        path = _inside_resources(resources, relative)
        manifest_names.add(Path(relative).name)
        actual_size = path.stat().st_size
        if actual_size != int(entry.get("size") or -1):
            raise BundleValidationError(
                f"offline ASR file size mismatch: {relative}: "
                f"{actual_size} != {entry.get('size')}"
            )
        if check_hash:
            actual_hash = _sha256(path)
            if actual_hash != entry.get("sha256"):
                raise BundleValidationError(f"offline ASR SHA-256 mismatch: {relative}")
    if manifest_names != set(REQUIRED_SNAPSHOT_FILES):
        raise BundleValidationError("offline ASR manifest file names do not match the required set")
    if {path.name for path in snapshot_files} != manifest_names:
        raise BundleValidationError("offline ASR snapshot and manifest disagree")

    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "revision": revision,
        "model_bytes": (snapshot / "model.safetensors").stat().st_size,
        "total_required_bytes": sum(path.stat().st_size for path in snapshot_files),
        "hash_checked": check_hash,
        "resources": str(resources),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write-manifest", help="write a SHA-256 manifest for a staged model")
    write.add_argument("--resources", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="verify a staged or app Resources directory")
    verify.add_argument("--resources", type=Path, required=True)
    verify.add_argument("--check-hash", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "write-manifest":
            manifest = write_manifest(args.resources)
            result = {
                "status": "ok",
                "model_id": manifest["model_id"],
                "revision": manifest["revision"],
                "model_bytes": manifest["model_bytes"],
                "total_required_bytes": manifest["total_required_bytes"],
                "manifest": str(args.resources.expanduser().resolve() / MANIFEST_RELATIVE),
            }
        else:
            result = verify_manifest(args.resources, check_hash=bool(args.check_hash))
    except BundleValidationError as exc:
        print(f"offline ASR bundle invalid: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
