from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from . import ipc as sidecar_ipc
from .core.text_normalizer import simplify_chinese_value
from .web_runtime import (
    FileCopy,
    archive_library_task,
    articles_root,
    atomic_write_text,
    data_root,
    default_library_meta,
    fsync_directory,
    json_bytes,
    library_root,
    now_iso8601,
    now_ts,
    read_asr_review,
    read_json,
    read_library_meta,
    safe_filename,
    sanitize_stem,
    secrets_path,
    settings_path,
    source_audio_copy_update,
    stable_audio_path,
    stem_lock,
    transactional_write_files,
    unique_path,
    uploads_root,
    validate_asr_review,
    validate_library_meta,
    write_asr_review,
    write_json,
)

app = FastAPI(title="LocalScribe Web API", version="1.0.3")
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:11517", "http://127.0.0.1:11517"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MEDIA_EXTS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


DEFAULT_SETTINGS: dict[str, Any] = {
    "model_id": "mlx-community/whisper-large-v3-turbo",
    "backend": "auto",
    "language": "zh",
    "asr_hotwords": "",
    "asr_quality_mode": "standard",
    "audio_preprocess": "adaptive",
    "transcript_sync": "precise",
    "output_formats": ["txt", "srt", "json"],
    "output_dir": None,
    "correction": {
        "enabled": False,
        "auto_pipeline": False,
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "mode": "medium",
        "batch_size": 30,
        "context_hint": "",
        "use_glossary": True,
        "concurrency": 15,
        "advanced": {
            "temperature": 0.1,
            "max_tokens": 8192,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "polish": {
        "enabled": False,
        "model": "deepseek-v4-flash",
        "advanced": {
            "temperature": 0.3,
            "max_tokens": 384000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "translation": {
        "model": "deepseek-v4-flash",
        "advanced": {
            "temperature": 0.3,
            "max_tokens": 384000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "diarization": {"enabled": False, "engine": "auto", "n_speakers": 0, "speakers": []},
}


def _error(message: str, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail=message)


def _is_reserved_speaker_profile_name(value: Any) -> bool:
    return str(value or "").strip().upper().startswith("SPEAKER_")


def _validated_settings(settings: dict[str, Any], *, drop_reserved: bool = False) -> tuple[dict[str, Any], bool]:
    normalized = json.loads(json.dumps(settings))
    diarization = normalized.get("diarization")
    if not isinstance(diarization, dict):
        return normalized, False
    speakers = diarization.get("speakers")
    if not isinstance(speakers, list):
        return normalized, False
    reserved = [
        profile
        for profile in speakers
        if isinstance(profile, dict) and _is_reserved_speaker_profile_name(profile.get("name"))
    ]
    if reserved and not drop_reserved:
        names = ", ".join(str(profile.get("name") or "") for profile in reserved[:3])
        raise _error(f"SPEAKER_* is a per-recording placeholder and cannot be saved as a voice profile: {names}")
    if reserved:
        diarization["speakers"] = [profile for profile in speakers if profile not in reserved]
        return normalized, True
    return normalized, False


def _with_api_key(params: dict[str, Any]) -> dict[str, Any]:
    provider = params.get("provider") or "deepseek"
    key = read_json(secrets_path(), {}).get(provider)
    if not key:
        raise _error(f"No API key stored for provider {provider!r}", 401)
    out = dict(params)
    out["api_key"] = key
    return out


def _task_dir(stem: str) -> Path:
    return library_root() / sanitize_stem(stem)


def _meta_path(stem: str) -> Path:
    return _task_dir(stem) / "task.json"


def _load_meta(stem: str) -> dict[str, Any]:
    path = _meta_path(stem)
    try:
        return read_library_meta(path, expected_stem=stem)
    except FileNotFoundError as exc:
        raise ValueError(f"metadata missing: {path}") from exc


def _load_meta_for_raw(stem: str) -> dict[str, Any]:
    path = _meta_path(stem)
    if path.exists():
        return read_library_meta(path, expected_stem=stem)
    task_dir = _task_dir(stem)
    if task_dir.exists() and any(task_dir.iterdir()):
        raise ValueError(f"metadata missing: {path}")
    return default_library_meta(stem)


def _write_text(path: Path, text: str) -> None:
    atomic_write_text(path, simplify_chinese_value(text))


def _text_bytes(text: str) -> bytes:
    if not isinstance(text, str):
        raise ValueError("library text payload must be a string")
    return simplify_chinese_value(text).encode("utf-8")


def _strict_json_text(contents: str, label: str) -> Any:
    def reject_nonstandard_number(value: str) -> Any:
        raise ValueError(f"non-standard JSON number: {value}")

    try:
        return json.loads(contents, parse_constant=reject_nonstandard_number)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"parse {label}: {exc}") from exc


def _strict_json(path: Path, label: str) -> Any:
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except UnicodeDecodeError as exc:
        raise ValueError(f"parse {label} {path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"read {label} {path}: {exc}") from exc
    return _strict_json_text(contents, f"{label} {path}")


def _stable_audio_for_save(
    stem: str, meta: dict[str, Any], args: dict[str, Any]
) -> tuple[str | None, tuple[Path, FileCopy] | None]:
    current = meta.get("audio_path")
    existing = stable_audio_path(_task_dir(stem), current if isinstance(current, str) else None)
    if existing:
        return existing, None
    copy_source = args.get("source_audio")
    if not copy_source and isinstance(current, str):
        current_path = Path(current)
        try:
            if current_path.is_file() and current_path.stat().st_size > 0:
                copy_source = current
        except OSError:
            pass
    update = source_audio_copy_update(stem, copy_source, args["audio_filename"])
    if update is None:
        return None, None
    target, payload = update
    try:
        if payload.source.resolve() == target.resolve() and target.is_file():
            return str(target), None
    except OSError:
        pass
    return str(target), update


def _library_meta_from_raw(
    args: dict[str, Any], previous: dict[str, Any], audio_path: str | None
) -> dict[str, Any]:
    stem = sanitize_stem(args["stem"])
    result = args.get("result")
    if not isinstance(result, dict):
        raise ValueError("library raw result must be a JSON object")
    duration = result["duration"] if "duration" in result else 0
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration):
        raise ValueError("library raw result.duration must be a finite number")
    segments = result["segments"] if "segments" in result else []
    if not isinstance(segments, list):
        raise ValueError("library raw result.segments must be an array")
    backend = result["backend"] if "backend" in result else ""
    model_id = result["model_id"] if "model_id" in result else ""
    if not isinstance(backend, str) or not isinstance(model_id, str):
        raise ValueError("library raw result backend and model_id must be strings")
    audio_filename = args.get("audio_filename")
    if not isinstance(audio_filename, str):
        raise ValueError("library raw audio_filename must be a string")

    meta = dict(previous)
    now = now_ts()
    meta.update(
        {
            "stem": stem,
            "raw_filename": f"{stem}.json",
            "audio_filename": audio_filename,
            "audio_path": audio_path,
            "updated_at": now,
            "duration": float(duration),
            "segments": len(segments),
            "backend": backend,
            "model_id": model_id,
        }
    )
    if meta.get("created_at") == 0:
        meta["created_at"] = now
    validate_library_meta(meta)
    return meta


def _save_raw_and_corrected(args: dict[str, Any]) -> dict[str, Any]:
    raw = args["raw"]
    corrected = args.get("corrected")
    clear_corrected = bool(args.get("clear_corrected", False))
    stem = sanitize_stem(raw["stem"])
    raw_json_text = raw.get("json")
    if not isinstance(raw_json_text, str):
        raise ValueError("library raw JSON payload must be a string")
    raw_value = _strict_json_text(raw_json_text, "raw JSON payload")
    if not isinstance(raw_value, dict):
        raise ValueError("library raw JSON payload must be a JSON object")
    if not isinstance(raw.get("audio_filename"), str):
        raise ValueError("library raw audio_filename must be a string")

    if corrected is not None:
        if not isinstance(corrected, dict):
            raise ValueError("library corrected payload must be a JSON object")
        corrected_stem = sanitize_stem(corrected["stem"])
        if corrected_stem != stem:
            raise ValueError(
                f"raw/corrected stem mismatch: raw={stem!r}, corrected={corrected_stem!r}"
            )
        for key in ("txt", "srt", "json", "diff", "model"):
            if not isinstance(corrected.get(key), str):
                raise ValueError(f"library corrected {key} must be a string")
        for key in ("changed", "total"):
            value = corrected.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"library corrected {key} must be a non-negative integer")
        glossary = corrected.get("glossary")
        if glossary is not None and not isinstance(glossary, list):
            raise ValueError("library corrected glossary must be an array or null")

    with stem_lock(stem):
        target = _task_dir(stem)
        previous = _load_meta_for_raw(stem)
        audio_path, audio_update = _stable_audio_for_save(stem, previous, raw)
        meta = _library_meta_from_raw(raw, previous, audio_path)
        if corrected is not None:
            meta.update(
                {
                    "has_corrected": True,
                    "correction_model": corrected["model"],
                    "correction_changed": corrected["changed"],
                    "correction_glossary": corrected.get("glossary"),
                }
            )
        elif clear_corrected:
            meta.update(
                {
                    "has_corrected": False,
                    "correction_model": None,
                    "correction_changed": None,
                    "correction_glossary": None,
                }
            )
        validate_library_meta(meta)

        updates: list[tuple[Path, bytes | FileCopy]] = [
            (target / f"{stem}.txt", _text_bytes(raw["txt"])),
            (target / f"{stem}.srt", _text_bytes(raw["srt"])),
            # The raw JSON is an immutable caller artifact: never normalize or reserialize it.
            (target / f"{stem}.json", raw_json_text.encode("utf-8")),
        ]
        if corrected is not None:
            updates.extend(
                [
                    (target / f"{stem}_corrected.txt", _text_bytes(corrected["txt"])),
                    (target / f"{stem}_corrected.srt", _text_bytes(corrected["srt"])),
                    (target / f"{stem}_corrected.json", _text_bytes(corrected["json"])),
                    (target / f"{stem}_diff.txt", _text_bytes(corrected["diff"])),
                ]
            )
        if audio_update is not None:
            updates.append(audio_update)
        updates.append((target / "task.json", json_bytes(meta)))
        transactional_write_files(updates)
        return meta


def _parse_frontmatter(raw: str, path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "title": path.stem,
        "filename": path.name,
        "path": str(path),
        "source_audio": None,
        "source_stem": None,
        "duration_seconds": None,
        "char_count": len(raw),
        "model": None,
        "based_on": None,
        "tags": [],
        "note": None,
        "created_at": "",
        "modified_at": time_iso(path.stat().st_mtime if path.exists() else now_ts()),
    }
    if raw.startswith("---\n"):
        end = raw.find("\n---\n", 4)
        if end != -1:
            for line in raw[4:end].splitlines():
                key, sep, val = line.partition(":")
                if not sep:
                    continue
                val = val.strip().strip('"').strip("'")
                if key == "title":
                    meta["title"] = val
                elif key in {"source_audio", "source_stem", "model", "based_on", "note", "created_at"}:
                    meta[key] = val
                elif key == "duration_seconds":
                    meta[key] = float(val) if val else None
                elif key == "char_count":
                    meta[key] = int(val) if val.isdigit() else len(raw)
                elif key == "tags":
                    meta["tags"] = [x.strip().strip('"').strip("'") for x in val.strip("[]").split(",") if x.strip()]
    if not meta["created_at"]:
        meta["created_at"] = meta["modified_at"]
    return meta


def time_iso(ts: float) -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "data_root": str(data_root())}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    raw_name = Path(file.filename or "audio").name
    stem = sanitize_stem(Path(raw_name).stem)
    suffix = Path(raw_name).suffix or ".audio"
    target = unique_path(uploads_root(), stem, suffix)
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"path": str(target), "filename": raw_name, "size": target.stat().st_size}


@app.get("/api/media")
def media(path: str) -> FileResponse:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise _error(f"Media file not found: {p}", 404)
    if p.suffix.lower() not in _MEDIA_EXTS:
        raise _error("Unsupported media file type", 400)
    return FileResponse(p)


@app.post("/api/invoke/{method}")
async def invoke(method: str, request: Request) -> JSONResponse:
    params = await request.json()
    try:
        result = await run_in_threadpool(dispatch, method, params or {})
        return JSONResponse({"result": result})
    except HTTPException:
        raise
    except Exception as exc:
        raise _error(str(exc), 500) from exc


def dispatch(method: str, params: dict[str, Any]) -> Any:
    if method == "load_settings":
        stored = {**DEFAULT_SETTINGS, **read_json(settings_path(), {})}
        validated, changed = _validated_settings(stored, drop_reserved=True)
        if changed:
            write_json(settings_path(), validated)
        return validated
    if method == "save_settings":
        validated, _ = _validated_settings(params["settings"])
        write_json(settings_path(), validated)
        return None
    if method == "set_api_key":
        secrets = read_json(secrets_path(), {})
        secrets[params["provider"]] = params["api_key"]
        write_json(secrets_path(), secrets)
        return None
    if method == "has_api_key":
        return bool(read_json(secrets_path(), {}).get(params["provider"]))
    if method == "get_api_key":
        key = read_json(secrets_path(), {}).get(params["provider"])
        if not key:
            raise _error(f"No API key found for provider: {params['provider']}", 404)
        return key
    if method == "delete_api_key":
        secrets = read_json(secrets_path(), {})
        secrets.pop(params["provider"], None)
        write_json(secrets_path(), secrets)
        return None

    if method == "correct_segments":
        return sidecar_ipc.HANDLERS["correct"](_with_api_key(params))
    if method == "polish_article":
        return sidecar_ipc.handle_polish(_with_api_key(params))
    if method == "translate_article":
        return sidecar_ipc.handle_translate_article(_with_api_key(params))
    if method == "library_save_raw":
        return _save_raw_and_corrected({"raw": params["args"]})
    if method == "library_save_raw_and_corrected":
        return _save_raw_and_corrected(params["args"])
    if method == "library_save_asr_review":
        args = params["args"]
        stem = sanitize_stem(args["stem"])
        try:
            validate_asr_review(args["review"])
        except ValueError as exc:
            raise _error(str(exc)) from exc
        with stem_lock(stem):
            _load_meta(stem)
            write_asr_review(_task_dir(stem) / "asr_human_review.json", args["review"])
        return None
    if method == "library_save_corrected":
        args = params["args"]
        stem = sanitize_stem(args["stem"])
        with stem_lock(stem):
            target = _task_dir(stem)
            meta = dict(_load_meta(stem))
            meta.update(
                {
                    "stem": stem,
                    "has_corrected": True,
                    "correction_model": args["model"],
                    "correction_changed": args["changed"],
                    "correction_glossary": args.get("glossary"),
                    "updated_at": now_ts(),
                }
            )
            validate_library_meta(meta)
            transactional_write_files(
                [
                    (target / f"{stem}_corrected.txt", _text_bytes(args["txt"])),
                    (target / f"{stem}_corrected.srt", _text_bytes(args["srt"])),
                    (target / f"{stem}_corrected.json", _text_bytes(args["json"])),
                    (target / f"{stem}_diff.txt", _text_bytes(args["diff"])),
                    (target / "task.json", json_bytes(meta)),
                ]
            )
            return meta
    if method == "library_save_polished":
        args = params["args"]
        stem = sanitize_stem(args["stem"])
        with stem_lock(stem):
            body = f"# {stem} - 完整文字稿\n# 排版 {args['model']}\n\n{args['text']}\n"
            meta = dict(_load_meta(stem))
            meta.update(
                {
                    "stem": stem,
                    "has_polished": True,
                    "polish_model": args["model"],
                    "polish_source": args.get("source"),
                    "updated_at": now_ts(),
                }
            )
            validate_library_meta(meta)
            target = _task_dir(stem)
            transactional_write_files(
                [
                    (target / f"{stem}_完整版.txt", _text_bytes(body)),
                    (target / "task.json", json_bytes(meta)),
                ]
            )
            return meta
    if method == "library_list":
        root = library_root()
        if not root.exists():
            return []
        items: list[dict[str, Any]] = []
        for path in root.iterdir():
            if not path.is_dir():
                continue
            with stem_lock(path.name):
                try:
                    meta = read_library_meta(path / "task.json", expected_stem=path.name)
                    items.append(meta)
                except Exception as exc:
                    logger.warning("skipping corrupt library item %s: %s", path.name, exc)
        return sorted(items, key=lambda item: -item["updated_at"])
    if method == "library_load":
        stem = sanitize_stem(params["stem"])
        task_dir = _task_dir(stem)
        with stem_lock(stem):
            if not task_dir.is_dir():
                raise _error(f"library item not found: {stem}", 404)
            meta = _load_meta(stem)
            raw_filename = meta.get("raw_filename")
            raw_path = task_dir / (
                raw_filename if isinstance(raw_filename, str) else f"{stem}.json"
            )
            if not raw_path.exists():
                original_stem = sanitize_stem(Path(meta["audio_filename"]).stem)
                original_path = task_dir / f"{original_stem}.json"
                if original_path.exists():
                    raw_path = original_path
                else:
                    candidates = [
                        candidate
                        for candidate in sorted(task_dir.glob("*.json"))
                        if candidate.name not in {"task.json", "asr_human_review.json"}
                        and not candidate.name.endswith("_corrected.json")
                    ]
                    if len(candidates) == 1:
                        raw_path = candidates[0]
                    else:
                        matching_candidates = []
                        for candidate in candidates:
                            try:
                                candidate_json = _strict_json(candidate, "raw transcript")
                            except (ValueError, RuntimeError):
                                continue
                            if (
                                isinstance(candidate_json, dict)
                                and isinstance(candidate_json.get("segments"), list)
                                and isinstance(candidate_json.get("backend"), str)
                                and isinstance(candidate_json.get("model_id"), str)
                            ):
                                matching_candidates.append(candidate)
                        if len(matching_candidates) == 1:
                            raw_path = matching_candidates[0]
            if not raw_path.is_file():
                raise _error(f"library item not found: {stem}", 404)
            raw_json = _strict_json(raw_path, "raw transcript")
            if isinstance(raw_json, dict) and meta.get("audio_path"):
                raw_json["audio"] = meta["audio_path"]
            asr_human_review = read_asr_review(task_dir / "asr_human_review.json")
            corrected_json = None
            if meta["has_corrected"]:
                corrected_path = task_dir / f"{stem}_corrected.json"
                if not corrected_path.exists():
                    corrected_path = next(
                        iter(sorted(task_dir.glob("*_corrected.json"))), corrected_path
                    )
                if not corrected_path.is_file():
                    raise RuntimeError(f"committed corrected result missing: {stem}")
                corrected_json = _strict_json(corrected_path, "corrected transcript")
            polished_text = None
            if meta["has_polished"]:
                polished_path = task_dir / f"{stem}_完整版.txt"
                if not polished_path.exists():
                    polished_path = next(
                        iter(sorted(task_dir.glob("*_完整版.txt"))), polished_path
                    )
                if not polished_path.is_file():
                    raise RuntimeError(f"committed polished result missing: {stem}")
                lines = polished_path.read_text(encoding="utf-8").splitlines()
                polished_text = "\n".join([line for line in lines if not line.startswith("# ")]).strip()
            return {
                "meta": meta,
                "raw_json": raw_json,
                "asr_human_review": asr_human_review,
                "corrected_json": corrected_json,
                "polished_text": polished_text,
            }
    if method == "library_delete":
        stem = sanitize_stem(params["stem"])
        with stem_lock(stem):
            task_dir = _task_dir(stem)
            if task_dir.exists():
                shutil.rmtree(task_dir)
                if task_dir.parent.is_dir():
                    fsync_directory(task_dir.parent)
        return None
    if method == "library_archive":
        stem = sanitize_stem(params["stem"])
        return archive_library_task(stem)
    if method == "library_root_path":
        library_root().mkdir(parents=True, exist_ok=True)
        return str(library_root())

    if method == "article_save":
        args = params["args"]
        root = articles_root()
        root.mkdir(parents=True, exist_ok=True)
        filename = safe_filename(args["title"])
        target = root / f"{filename}.md" if args.get("overwrite") else unique_path(root, filename, ".md")
        created_at = now_iso8601()
        tags = args.get("tags") or []
        fm = [
            "---",
            f"title: {json.dumps(args['title'], ensure_ascii=False)}",
            f"char_count: {len(args['content'])}",
            f"created_at: {created_at}",
        ]
        for key in ["source_audio", "source_stem", "duration_seconds", "model", "based_on", "note"]:
            if args.get(key) is not None and args.get(key) != "":
                fm.append(f"{key}: {json.dumps(args[key], ensure_ascii=False)}")
        if tags:
            fm.append("tags: [" + ", ".join(json.dumps(t, ensure_ascii=False) for t in tags) + "]")
        body = "\n".join(fm) + "\n---\n\n" + f"# {args['title']}\n\n{args['content'].strip()}\n"
        _write_text(target, body)
        return _parse_frontmatter(body, target)
    if method == "article_list":
        root = articles_root()
        if not root.exists():
            return []
        items = [_parse_frontmatter(p.read_text(encoding="utf-8"), p) for p in root.glob("*.md")]
        return sorted(items, key=lambda x: x["modified_at"], reverse=True)
    if method == "article_delete":
        (articles_root() / params["filename"]).unlink(missing_ok=True)
        return None
    if method == "article_rename":
        src = articles_root() / params["oldFilename"]
        if not src.exists():
            raise _error("article not found", 404)
        target = unique_path(articles_root(), safe_filename(params["newTitle"]), ".md")
        raw = src.read_text(encoding="utf-8")
        lines = raw.splitlines()
        in_frontmatter = bool(lines and lines[0] == "---")
        frontmatter_closed = False
        next_lines: list[str] = []
        for line in lines:
            if in_frontmatter and line.startswith("title:"):
                next_lines.append(f"title: {json.dumps(params['newTitle'], ensure_ascii=False)}")
                continue
            if in_frontmatter and line == "---" and next_lines:
                frontmatter_closed = True
                in_frontmatter = False
                next_lines.append(line)
                continue
            if frontmatter_closed and line.startswith("# "):
                next_lines.append(f"# {params['newTitle']}")
                frontmatter_closed = False
                continue
            next_lines.append(line)
        raw = "\n".join(next_lines) + ("\n" if raw.endswith("\n") else "")
        src.rename(target)
        target.write_text(raw, encoding="utf-8")
        return _parse_frontmatter(raw, target)
    if method == "article_read":
        path = articles_root() / params["filename"]
        if not path.exists():
            raise _error("article not found", 404)
        return path.read_text(encoding="utf-8")
    if method == "articles_root_path":
        articles_root().mkdir(parents=True, exist_ok=True)
        return str(articles_root())

    if method == "open_url":
        _open_path(params["url"])
        return None
    if method == "reveal_models_dir":
        model_id = params.get("model_id") or "mlx-community/whisper-large-v3-turbo"
        path = (data_root() / "models" / model_id.rsplit("/", 1)[-1])
        path.mkdir(parents=True, exist_ok=True)
        _open_path(str(path))
        return str(path)

    mapped = {
        "environment": sidecar_ipc.handle_environment,
        "check_model": sidecar_ipc.handle_check_model,
        "check_model_cache": sidecar_ipc.handle_check_model,
        "probe_audio": sidecar_ipc.handle_probe_audio,
        "asr_preflight_select": sidecar_ipc.HANDLERS["asr_preflight_select"],
        "transcribe": sidecar_ipc.HANDLERS["transcribe"],
        "diarize": sidecar_ipc.handle_diarize,
        "recommend_diarization": sidecar_ipc.handle_recommend_diarization,
        "extract_voice_embedding": sidecar_ipc.handle_extract_voice_embedding,
        "preflight_voiceprint_anchors": sidecar_ipc.handle_preflight_voiceprint_anchors,
        "reidentify_speakers": sidecar_ipc.handle_reidentify_speakers,
        "correct_pause": sidecar_ipc.handle_correct_pause,
        "correct_resume": sidecar_ipc.handle_correct_resume,
        "correct_cancel": sidecar_ipc.handle_correct_cancel,
        "correct_status": sidecar_ipc.handle_correct_status,
    }
    if method in mapped:
        return mapped[method](params)
    raise _error(f"Method not found: {method}", 404)


def _open_path(path: str) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
    elif sys.platform.startswith("win"):
        subprocess.Popen(["cmd", "/C", "start", "", path])
    else:
        subprocess.Popen(["xdg-open", path])


def main() -> None:
    import uvicorn

    port = int(os.environ.get("LOCALSCRIBE_WEB_PORT", "8765"))
    uvicorn.run("scribe_py.web_server:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    main()
