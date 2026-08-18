from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


_ASR_REVIEW_STATUSES = {
    "pending",
    "confirmed_present",
    "confirmed_missing",
    "substitution",
    "noise",
    "resolved",
}
_STEM_LOCKS: dict[str, threading.RLock] = {}
_STEM_LOCKS_GUARD = threading.Lock()
_LIBRARY_NAMESPACE_LOCK = threading.RLock()
_LIBRARY_META_REQUIRED_FIELDS = {
    "stem",
    "audio_filename",
    "audio_path",
    "duration",
    "segments",
    "backend",
    "model_id",
    "created_at",
    "updated_at",
    "has_corrected",
    "has_polished",
    "correction_model",
    "correction_changed",
    "correction_glossary",
    "polish_model",
    "polish_source",
}


@dataclass(frozen=True)
class FileCopy:
    source: Path


FilePayload = bytes | FileCopy


def _project_root() -> Path | None:
    env = os.environ.get("LOCALSCRIBE_DEV_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "package.json").exists() and (p / "scribe-py").is_dir():
            return p

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "package.json").exists() and (parent / "scribe-py").is_dir():
            return parent
    return None


def data_root() -> Path:
    bundled = os.environ.get("LOCALSCRIBE_RESOURCES")
    if bundled:
        root = Path.home() / "Library/Application Support/LocalScribe"
        root.mkdir(parents=True, exist_ok=True)
        return root
    return _project_root() or (Path.home() / "Library/Application Support/LocalScribe")


def library_root() -> Path:
    return data_root() / "transcripts"


def articles_root() -> Path:
    return data_root() / "articles"


def uploads_root() -> Path:
    root = data_root() / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def settings_path() -> Path:
    return data_root() / "settings.json"


def secrets_path() -> Path:
    return data_root() / "secrets.json"


def sanitize_stem(stem: str) -> str:
    cleaned = "".join("_" if ch in {"/", "\\"} or ord(ch) < 32 else ch for ch in stem.strip())
    cleaned = cleaned.strip(". ").strip()
    return cleaned or "meeting"


def now_ts() -> int:
    return int(time.time())


def now_iso8601() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_filename(title: str) -> str:
    trimmed = title.strip()
    if not trimmed:
        return "untitled"
    out = []
    for ch in trimmed:
        safe = ch.isalnum() or ch in {"-", "_", " "} or ord(ch) > 127
        out.append(ch if safe else "_")
    return " ".join("".join(out).split())[:80] or "untitled"


def unique_path(base: Path, name: str, suffix: str) -> Path:
    primary = base / f"{name}{suffix}"
    if not primary.exists():
        return primary
    for n in range(2, 1000):
        candidate = base / f"{name} ({n}){suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate a unique path for {primary}")


def read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(path, flags)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def fsync_directory(path: Path) -> None:
    _fsync_directory(path)


def _ensure_parent(path: Path) -> None:
    parent = path.parent
    if parent.is_dir():
        return
    missing: list[Path] = []
    current = parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    parent.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        if created.parent.is_dir():
            _fsync_directory(created.parent)


def _create_temp(path: Path, kind: str = "") -> tuple[int, Path]:
    _ensure_parent(path)
    infix = f".{kind}" if kind else ""
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}{infix}.",
        suffix=".tmp",
    )
    return fd, Path(temp_name)


def _write_payload(fd: int, payload: FilePayload) -> None:
    with os.fdopen(fd, "wb") as temp_file:
        if isinstance(payload, FileCopy):
            source = payload.source
            try:
                source_size = source.stat().st_size
            except OSError as exc:
                raise RuntimeError(f"cannot inspect source file: {source}") from exc
            if source_size <= 0:
                raise ValueError(f"source file is empty: {source}")
            copied = 0
            with source.open("rb") as source_file:
                while True:
                    chunk = source_file.read(1024 * 1024)
                    if not chunk:
                        break
                    temp_file.write(chunk)
                    copied += len(chunk)
            if copied != source_size:
                raise RuntimeError(
                    f"source file changed while copying: expected {source_size} bytes, copied {copied}"
                )
        else:
            temp_file.write(payload)
        temp_file.flush()
        os.fsync(temp_file.fileno())


def _stage_payload(path: Path, payload: FilePayload, kind: str = "") -> Path:
    fd, temp_path = _create_temp(path, kind)
    try:
        _write_payload(fd, payload)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _cleanup_temp(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def transactional_write_files(updates: Sequence[tuple[Path, FilePayload]]) -> None:
    """Atomically replace a related file set, restoring the old set on failure."""
    normalized = [(Path(path), payload) for path, payload in updates]
    if not normalized:
        return
    targets = [path for path, _ in normalized]
    if len(set(targets)) != len(targets):
        raise ValueError("transaction contains duplicate target paths")

    created_directories: set[Path] = set()
    for target in targets:
        current = target.parent
        while not current.exists():
            created_directories.add(current)
            current = current.parent
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    committed: list[Path] = []
    try:
        for target in targets:
            if target.exists():
                if not target.is_file():
                    raise IsADirectoryError(f"transaction target is not a file: {target}")
                backups[target] = _stage_payload(target, FileCopy(target), "backup")
            else:
                backups[target] = None
        for target, payload in normalized:
            staged[target] = _stage_payload(target, payload)

        for target in targets:
            temp_path = staged[target]
            os.replace(temp_path, target)
            staged[target] = None  # type: ignore[assignment]
            committed.append(target)
            _fsync_directory(target.parent)
    except Exception as error:
        for temp_path in staged.values():
            _cleanup_temp(temp_path)

        rollback_errors: list[str] = []
        failed_backups: set[Path] = set()
        for target in reversed(committed):
            backup = backups.get(target)
            try:
                if backup is None:
                    target.unlink(missing_ok=True)
                    _fsync_directory(target.parent)
                else:
                    os.replace(backup, target)
                    backups[target] = None
                    _fsync_directory(target.parent)
            except Exception as rollback_error:
                failed_backups.add(target)
                rollback_errors.append(f"restore {target}: {rollback_error}")

        for target, backup in backups.items():
            if target not in failed_backups:
                _cleanup_temp(backup)
        if rollback_errors:
            raise RuntimeError(
                f"file transaction failed: {error}; rollback failed: {'; '.join(rollback_errors)}"
            ) from error
        for directory in sorted(created_directories, key=lambda path: len(path.parts), reverse=True):
            try:
                directory.rmdir()
                _fsync_directory(directory.parent)
            except FileNotFoundError:
                pass
            except OSError:
                # A concurrent/external writer may have populated it; never remove their files.
                pass
        raise
    else:
        for backup in backups.values():
            _cleanup_temp(backup)


def atomic_write_bytes(path: Path, contents: bytes) -> None:
    transactional_write_files([(path, contents)])


def atomic_write_text(path: Path, contents: str) -> None:
    atomic_write_bytes(path, contents.encode("utf-8"))


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_bytes(path, json_bytes(data))


def write_json(path: Path, data: Any) -> None:
    atomic_write_json(path, data)


def stem_lock(stem: str) -> threading.RLock:
    normalized = unicodedata.normalize("NFC", sanitize_stem(stem)).casefold()
    with _STEM_LOCKS_GUARD:
        return _STEM_LOCKS.setdefault(normalized, threading.RLock())


def asr_review_lock(stem: str) -> threading.RLock:
    return stem_lock(stem)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def validate_library_meta(meta: Any, path: Path | None = None) -> None:
    label = f"metadata {path}" if path is not None else "library metadata"
    if not isinstance(meta, dict):
        raise ValueError(f"{label} must be a JSON object")
    missing = sorted(_LIBRARY_META_REQUIRED_FIELDS.difference(meta))
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")

    string_fields = ("stem", "audio_filename", "backend", "model_id")
    for field in string_fields:
        if not isinstance(meta.get(field), str):
            raise ValueError(f"{label}.{field} must be a string")
    if not meta["stem"].strip():
        raise ValueError(f"{label}.stem must not be empty")
    if "raw_filename" in meta:
        raw_filename = meta["raw_filename"]
        if (
            not isinstance(raw_filename, str)
            or Path(raw_filename).name != raw_filename
            or not raw_filename.endswith(".json")
        ):
            raise ValueError(f"{label}.raw_filename must be a plain JSON filename")

    if meta.get("audio_path") is not None and not isinstance(meta["audio_path"], str):
        raise ValueError(f"{label}.audio_path must be a string or null")
    if not _is_finite_number(meta.get("duration")) or meta["duration"] < 0:
        raise ValueError(f"{label}.duration must be a non-negative finite number")
    if type(meta.get("segments")) is not int or meta["segments"] < 0:
        raise ValueError(f"{label}.segments must be a non-negative integer")
    for field in ("created_at", "updated_at"):
        if type(meta.get(field)) is not int or meta[field] < 0:
            raise ValueError(f"{label}.{field} must be a non-negative integer")
    for field in ("has_corrected", "has_polished"):
        if type(meta.get(field)) is not bool:
            raise ValueError(f"{label}.{field} must be a boolean")
    for field in ("correction_model", "polish_model", "polish_source"):
        if meta.get(field) is not None and not isinstance(meta[field], str):
            raise ValueError(f"{label}.{field} must be a string or null")
    if meta.get("correction_changed") is not None and (
        type(meta["correction_changed"]) is not int or meta["correction_changed"] < 0
    ):
        raise ValueError(f"{label}.correction_changed must be a non-negative integer or null")
    if meta.get("correction_glossary") is not None and not isinstance(
        meta["correction_glossary"], list
    ):
        raise ValueError(f"{label}.correction_glossary must be an array or null")


def read_library_meta(path: Path, expected_stem: str | None = None) -> dict[str, Any]:
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except UnicodeDecodeError as exc:
        raise ValueError(f"parse metadata {path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"read metadata {path}: {exc}") from exc

    def reject_nonstandard_number(value: str) -> Any:
        raise ValueError(f"non-standard JSON number: {value}")

    try:
        meta = json.loads(contents, parse_constant=reject_nonstandard_number)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"parse metadata {path}: {exc}") from exc
    try:
        validate_library_meta(meta, path)
    except ValueError as exc:
        raise ValueError(f"validate metadata {path}: {exc}") from exc
    if expected_stem is not None and meta["stem"] != sanitize_stem(expected_stem):
        raise ValueError(
            f"validate metadata {path}: stem {meta['stem']!r} does not match directory "
            f"{sanitize_stem(expected_stem)!r}"
        )
    return meta


def write_library_meta(path: Path, meta: dict[str, Any]) -> None:
    validate_library_meta(meta, path)
    atomic_write_json(path, meta)


def default_library_meta(stem: str) -> dict[str, Any]:
    return {
        "stem": sanitize_stem(stem),
        "audio_filename": "",
        "audio_path": None,
        "duration": 0.0,
        "segments": 0,
        "backend": "",
        "model_id": "",
        "created_at": 0,
        "updated_at": 0,
        "has_corrected": False,
        "has_polished": False,
        "correction_model": None,
        "correction_changed": None,
        "correction_glossary": None,
        "polish_model": None,
        "polish_source": None,
    }


def validate_asr_review(review: Any) -> None:
    if not isinstance(review, dict):
        raise ValueError("ASR human review must be a JSON object")
    if type(review.get("schema_version")) is not int or review["schema_version"] != 1:
        raise ValueError("ASR human review schema_version must be 1")
    items = review.get("items")
    if not isinstance(items, list):
        raise ValueError("ASR human review items must be an array")

    ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"ASR human review items[{index}] must be a JSON object")

        item_id = item.get("id")
        if not isinstance(item_id, str):
            raise ValueError(f"ASR human review items[{index}].id must be a string")
        if not item_id.strip():
            raise ValueError(f"ASR human review items[{index}].id must not be empty")
        if item_id in ids:
            raise ValueError(f"ASR human review item id must be unique: {item_id}")
        ids.add(item_id)

        start = item.get("start")
        end = item.get("end")
        if not _is_finite_number(start):
            raise ValueError(f"ASR human review items[{index}].start must be a finite number")
        if not _is_finite_number(end):
            raise ValueError(f"ASR human review items[{index}].end must be a finite number")
        if start < 0 or start >= end:
            raise ValueError(f"ASR human review items[{index}] must satisfy 0 <= start < end")

        status = item.get("status")
        if not isinstance(status, str):
            raise ValueError(f"ASR human review items[{index}].status must be a string")
        if status not in _ASR_REVIEW_STATUSES:
            raise ValueError(f"ASR human review items[{index}].status is invalid: {status}")

        for field in ("heard_text", "note", "replacement_text"):
            if field in item and not isinstance(item[field], str):
                raise ValueError(
                    f"ASR human review items[{index}].{field} must be a string when present"
                )


def write_asr_review(path: Path, review: Any) -> None:
    validate_asr_review(review)
    atomic_write_json(path, review)


def read_asr_review(path: Path) -> dict[str, Any] | None:
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except UnicodeDecodeError as exc:
        raise ValueError(f"parse ASR human review sidecar {path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"read ASR human review sidecar {path}: {exc}") from exc

    def reject_nonstandard_number(value: str) -> Any:
        raise ValueError(f"non-standard JSON number: {value}")

    try:
        review = json.loads(contents, parse_constant=reject_nonstandard_number)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"parse ASR human review sidecar {path}: {exc}") from exc
    try:
        validate_asr_review(review)
    except ValueError as exc:
        raise ValueError(f"validate ASR human review sidecar {path}: {exc}") from exc
    return review


def source_audio_copy_update(
    stem: str, source_audio: str | None, audio_filename: str
) -> tuple[Path, FileCopy] | None:
    if not source_audio:
        return None
    src = Path(source_audio).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"source audio not found: {src}")
    try:
        if src.stat().st_size <= 0:
            raise ValueError(f"source audio is empty: {src}")
    except OSError as exc:
        raise RuntimeError(f"cannot inspect source audio: {src}") from exc
    ext = src.suffix or Path(audio_filename).suffix or ".audio"
    clean_stem = sanitize_stem(stem)
    target = library_root() / clean_stem / "audio" / f"{clean_stem}{ext}"
    try:
        if src.resolve() == target.resolve():
            return target, FileCopy(target)
    except OSError:
        pass
    return target, FileCopy(src)


def copy_source_audio(stem: str, source_audio: str | None, audio_filename: str) -> str | None:
    update = source_audio_copy_update(stem, source_audio, audio_filename)
    if update is None:
        return None
    target, payload = update
    try:
        if payload.source.resolve() == target.resolve() and target.is_file():
            return str(target)
    except OSError:
        pass
    try:
        transactional_write_files([(target, payload)])
    except Exception as exc:
        raise RuntimeError(f"copy source audio failed: {payload.source} -> {target}") from exc
    return str(target)


def stable_audio_path(task_dir: Path, preferred: str | None = None) -> str | None:
    audio_dir = task_dir / "audio"
    if not audio_dir.is_dir():
        return None
    if preferred:
        preferred_path = audio_dir / Path(preferred).name
        try:
            if preferred_path.is_file() and preferred_path.stat().st_size > 0:
                return str(preferred_path)
        except OSError:
            pass
    candidates: list[Path] = []
    for path in sorted(audio_dir.iterdir()):
        if path.name.startswith(".") or path.name.endswith(".tmp") or not path.is_file():
            continue
        try:
            if path.stat().st_size > 0:
                candidates.append(path)
        except OSError:
            continue
    if not candidates:
        return None
    if len(candidates) > 1:
        raise RuntimeError(
            f"stable audio is ambiguous for {task_dir.name}: {len(candidates)} candidates"
        )
    return str(candidates[0])


def archive_library_task(stem: str) -> str | None:
    stem = sanitize_stem(stem)
    with _LIBRARY_NAMESPACE_LOCK:
        with stem_lock(stem):
            root = library_root()
            src = root / stem
            if not src.exists():
                return None
            meta_path = src / "task.json"
            meta = read_library_meta(meta_path, expected_stem=stem)
            original_meta = meta_path.read_bytes()
            root.mkdir(parents=True, exist_ok=True)
            base_name = f"{stem}-{now_ts()}"
            destination_lock: threading.RLock | None = None
            dest: Path | None = None
            for suffix in range(1000):
                name = base_name if suffix == 0 else f"{base_name} ({suffix + 1})"
                candidate_lock = stem_lock(name)
                candidate_lock.acquire()
                candidate = root / name
                if not candidate.exists():
                    destination_lock = candidate_lock
                    dest = candidate
                    break
                candidate_lock.release()
            if dest is None or destination_lock is None:
                raise RuntimeError(f"could not allocate archive destination for {stem}")
            archived_meta_path = dest / "task.json"
            renamed = False
            try:
                src.rename(dest)
                renamed = True
                _fsync_directory(root)
                meta = dict(meta)
                meta["stem"] = dest.name
                meta["audio_path"] = stable_audio_path(dest, meta.get("audio_path"))
                write_library_meta(archived_meta_path, meta)
                return str(dest)
            except Exception as error:
                rollback_errors: list[str] = []
                if renamed:
                    try:
                        atomic_write_bytes(archived_meta_path, original_meta)
                    except Exception as rollback_error:
                        rollback_errors.append(f"restore metadata: {rollback_error}")
                    try:
                        dest.rename(src)
                        _fsync_directory(root)
                    except Exception as rollback_error:
                        rollback_errors.append(f"restore directory: {rollback_error}")
                    if rollback_errors:
                        raise RuntimeError(
                            f"archive migration failed: {error}; rollback failed: "
                            + "; ".join(rollback_errors)
                        ) from error
                raise
            finally:
                destination_lock.release()
