from __future__ import annotations

import json
import stat
import threading
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from scribe_py import ipc
from scribe_py.ipc import (
    ASR_METHODS,
    CORRECTION_METHODS,
    _run_asr_handler,
    _run_correction_handler,
)


def test_asr_handler_lock_serializes_jobs():
    first_entered = threading.Event()
    second_entered = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    counter_lock = threading.Lock()
    active = 0
    max_active = 0

    def handler(params):
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        if params["job"] == 1:
            first_entered.set()
            assert release_first.wait(timeout=2)
        else:
            second_entered.set()
        with counter_lock:
            active -= 1
        return params["job"]

    first = threading.Thread(target=lambda: _run_asr_handler(handler, {"job": 1}))
    def run_second():
        second_started.set()
        _run_asr_handler(handler, {"job": 2})

    second = threading.Thread(target=run_second)
    first.start()
    assert first_entered.wait(timeout=1)
    second.start()
    assert second_started.wait(timeout=1)
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_entered.is_set()
    assert max_active == 1


def test_asr_lane_serializes_all_audio_model_work():
    assert ASR_METHODS == {
        "asr_preflight_select",
        "preflight_voiceprint_anchors",
        "transcribe",
    }
    assert CORRECTION_METHODS == {"correct"}


def test_web_dispatch_uses_shared_asr_handler_lock(monkeypatch):
    from scribe_py import web_server

    entered = threading.Event()
    release = threading.Event()
    second_finished = threading.Event()

    def fake_transcribe(params):
        if params["job"] == 1:
            entered.set()
            assert release.wait(timeout=2)
        return params["job"]

    monkeypatch.setattr(ipc, "handle_transcribe", fake_transcribe)
    first = threading.Thread(target=lambda: web_server.dispatch("transcribe", {"job": 1}))
    second = threading.Thread(
        target=lambda: (web_server.dispatch("transcribe", {"job": 2}), second_finished.set())
    )
    first.start()
    assert entered.wait(timeout=1)
    second.start()
    assert not second_finished.wait(timeout=0.05)
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_finished.is_set()


def test_web_dispatch_uses_shared_correction_handler_lock(monkeypatch):
    from scribe_py import web_server

    entered = threading.Event()
    release = threading.Event()
    second_finished = threading.Event()

    def fake_correct(params):
        if params["job"] == 1:
            entered.set()
            assert release.wait(timeout=2)
        return params["job"]

    monkeypatch.setattr(ipc, "handle_correct", fake_correct)
    monkeypatch.setattr(web_server, "_with_api_key", lambda params: params)
    first = threading.Thread(target=lambda: web_server.dispatch("correct_segments", {"job": 1}))
    second = threading.Thread(
        target=lambda: (web_server.dispatch("correct_segments", {"job": 2}), second_finished.set())
    )
    first.start()
    assert entered.wait(timeout=1)
    second.start()
    assert not second_finished.wait(timeout=0.05)
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_finished.is_set()


def test_web_dispatch_routes_voiceprint_reidentify(monkeypatch):
    from scribe_py import web_server

    expected = {"segments": [{"speaker": "SPEAKER_A"}]}
    monkeypatch.setattr(
        ipc,
        "handle_reidentify_speakers",
        lambda params: {**expected, "received": params},
    )

    params = {"audio": "meeting.wav", "segments": [], "anchors": []}
    assert web_server.dispatch("reidentify_speakers", params) == {
        **expected,
        "received": params,
    }


def test_web_dispatch_routes_voiceprint_anchor_preflight(monkeypatch):
    from scribe_py import web_server

    expected = {"candidates": []}
    monkeypatch.setattr(
        ipc,
        "handle_preflight_voiceprint_anchors",
        lambda params: {**expected, "received": params},
    )

    params = {"audio": "meeting.wav", "segments": []}
    assert web_server.dispatch("preflight_voiceprint_anchors", params) == {
        **expected,
        "received": params,
    }


def test_correction_handler_lock_serializes_jobs():
    entered = threading.Event()
    release = threading.Event()
    second_finished = threading.Event()

    def handler(params):
        if params["job"] == 1:
            entered.set()
            assert release.wait(timeout=2)
        return params["job"]

    first = threading.Thread(target=lambda: _run_correction_handler(handler, {"job": 1}))
    second = threading.Thread(
        target=lambda: (_run_correction_handler(handler, {"job": 2}), second_finished.set())
    )
    first.start()
    assert entered.wait(timeout=1)
    second.start()
    assert not second_finished.wait(timeout=0.05)
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_finished.is_set()


def test_web_audio_copy_failure_is_not_silent(tmp_path, monkeypatch):
    from scribe_py import web_runtime

    monkeypatch.setattr(web_runtime, "library_root", lambda: tmp_path / "library")
    with pytest.raises(FileNotFoundError):
        web_runtime.copy_source_audio("demo", str(tmp_path / "missing.wav"), "missing.wav")

    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    copied = web_runtime.copy_source_audio("demo", str(source), "source.wav")

    assert copied is not None
    assert (tmp_path / "library" / "demo" / "audio" / "demo.wav").read_bytes() == b"audio"


def _valid_library_meta(stem: str = "demo", audio_path: str | None = None):
    return {
        "stem": stem,
        "audio_filename": f"{stem}.wav",
        "audio_path": audio_path,
        "duration": 0.0,
        "segments": 0,
        "backend": "test",
        "model_id": "test-model",
        "created_at": 1,
        "updated_at": 1,
        "has_corrected": False,
        "has_polished": False,
        "correction_model": None,
        "correction_changed": None,
        "correction_glossary": None,
        "polish_model": None,
        "polish_source": None,
    }



def test_web_archive_keeps_history_loadable_and_preserves_raw_bytes(tmp_path, monkeypatch):
    from scribe_py import web_runtime, web_server

    root = tmp_path / "library"
    monkeypatch.setattr(web_runtime, "library_root", lambda: root)
    monkeypatch.setattr(web_server, "library_root", lambda: root)
    task = root / "demo"
    audio = task / "audio" / "demo.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"old-audio")
    raw_path = task / "demo.json"
    raw_path.write_bytes(
        b'{ "audio" : "old.wav", "segments": [], "backend": "test", "model_id": "model" }\n'
    )
    raw_before = raw_path.read_bytes()
    (task / "task.json").write_text(
        json.dumps(_valid_library_meta("demo", str(audio))),
        encoding="utf-8",
    )

    archived_path = web_server.dispatch("library_archive", {"stem": "demo"})

    assert archived_path is not None
    archived = Path(archived_path)
    archived_stem = archived.name
    loaded = web_server.dispatch("library_load", {"stem": archived_stem})
    assert loaded["meta"]["stem"] == archived_stem
    assert loaded["meta"]["audio_path"].startswith(str(archived))
    assert loaded["raw_json"]["audio"] == loaded["meta"]["audio_path"]
    assert Path(loaded["meta"]["audio_path"]).read_bytes() == b"old-audio"
    assert (archived / "demo.json").read_bytes() == raw_before
    assert not task.exists()


def test_web_archive_rolls_back_directory_when_parent_fsync_fails(tmp_path, monkeypatch):
    from scribe_py import web_runtime, web_server

    root = tmp_path / "library"
    monkeypatch.setattr(web_runtime, "library_root", lambda: root)
    monkeypatch.setattr(web_server, "library_root", lambda: root)
    raw_path = _write_web_library_task(root, "demo")
    raw_before = raw_path.read_bytes()
    meta_before = (root / "demo" / "task.json").read_bytes()
    original_fsync_directory = web_runtime._fsync_directory
    failed = False

    def fail_archive_root_once(path):
        nonlocal failed
        if Path(path) == root and not failed:
            failed = True
            raise OSError("injected archive fsync failure")
        return original_fsync_directory(path)

    monkeypatch.setattr(web_runtime, "_fsync_directory", fail_archive_root_once)

    with pytest.raises(OSError, match="injected archive fsync failure"):
        web_server.dispatch("library_archive", {"stem": "demo"})

    assert failed
    assert (root / "demo" / "demo.json").read_bytes() == raw_before
    assert (root / "demo" / "task.json").read_bytes() == meta_before
    assert [path.name for path in root.iterdir()] == ["demo"]


def test_web_archive_loads_raw_with_unrelated_audio_filename_and_custom_shape(
    tmp_path, monkeypatch
):
    root = tmp_path / "library"
    _, web_server = _patch_web_library_roots(monkeypatch, root)
    raw_json = '{ "custom" : 1, "繁體" : true }\n'
    args = _raw_save_args("edited-copy", raw_json)
    args["audio_filename"] = "recording.wav"
    web_server.dispatch("library_save_raw", {"args": args})
    (root / "edited-copy" / "other.json").write_text('{"report": true}', encoding="utf-8")

    archived_path = Path(web_server.dispatch("library_archive", {"stem": "edited-copy"}))
    loaded = web_server.dispatch("library_load", {"stem": archived_path.name})

    assert loaded["raw_json"] == {"custom": 1, "繁體": True}
    assert (archived_path / "edited-copy.json").read_bytes() == raw_json.encode("utf-8")


def _valid_asr_review():
    return {
        "schema_version": 1,
        "reviewer": "human",
        "extension": {"accepted": True},
        "items": [
            {
                "id": "segment-1",
                "start": 0,
                "end": 1.25,
                "status": "pending",
                "heard_text": "测试",
                "note": "listen again",
                "replacement_text": "",
                "item_extension": ["preserved"],
            }
        ],
    }


def _write_web_library_task(root: Path, stem: str = "demo") -> Path:
    task = root / stem
    task.mkdir(parents=True)
    raw_path = task / f"{stem}.json"
    raw_path.write_bytes(
        b'{\n  "segments": [],\n  "backend": "test",\n  "model_id": "test-model"\n}\n'
    )
    (task / "task.json").write_text(
        json.dumps(_valid_library_meta(stem)),
        encoding="utf-8",
    )
    return raw_path


def test_web_asr_review_save_load_round_trip_preserves_extensions_and_raw(tmp_path, monkeypatch):
    from scribe_py import web_server

    root = tmp_path / "transcripts"
    monkeypatch.setattr(web_server, "library_root", lambda: root)
    raw_path = _write_web_library_task(root)
    raw_before = raw_path.read_bytes()
    review = _valid_asr_review()

    with pytest.raises(ValueError, match="metadata missing"):
        web_server.dispatch(
            "library_save_asr_review",
            {"args": {"stem": "../demo", "review": review}},
        )
    sidecar = root / "_demo" / "asr_human_review.json"
    assert not sidecar.exists()
    assert raw_path.read_bytes() == raw_before

    loaded = web_server.dispatch("library_load", {"stem": "demo"})
    assert loaded["asr_human_review"] is None

    web_server.dispatch(
        "library_save_asr_review",
        {"args": {"stem": "demo", "review": review}},
    )
    loaded = web_server.dispatch("library_load", {"stem": "demo"})
    assert loaded["asr_human_review"] == review
    assert raw_path.read_bytes() == raw_before


def test_web_asr_review_second_save_atomically_replaces_entire_sidecar(tmp_path, monkeypatch):
    from scribe_py import web_runtime, web_server

    root = tmp_path / "transcripts"
    monkeypatch.setattr(web_server, "library_root", lambda: root)
    _write_web_library_task(root)
    old_review = _valid_asr_review()
    old_review["obsolete"] = "old trailing data that must disappear"
    web_server.dispatch(
        "library_save_asr_review",
        {"args": {"stem": "demo", "review": old_review}},
    )
    sidecar = root / "demo" / "asr_human_review.json"
    old_contents = sidecar.read_text(encoding="utf-8")
    replacement = {"schema_version": 1, "items": []}
    original_fsync = web_runtime.os.fsync
    original_replace = web_runtime.os.replace
    fsync_modes = []
    replacement_sources = []

    def track_fsync(fd):
        fsync_modes.append(web_runtime.os.fstat(fd).st_mode)
        return original_fsync(fd)

    def track_replace(src, dst):
        replacement_sources.append(Path(src))
        return original_replace(src, dst)

    monkeypatch.setattr(web_runtime.os, "fsync", track_fsync)
    monkeypatch.setattr(web_runtime.os, "replace", track_replace)

    web_server.dispatch(
        "library_save_asr_review",
        {"args": {"stem": "demo", "review": replacement}},
    )
    web_server.dispatch(
        "library_save_asr_review",
        {"args": {"stem": "demo", "review": replacement}},
    )

    assert sidecar.read_text(encoding="utf-8") == json.dumps(
        replacement, ensure_ascii=False, indent=2, allow_nan=False
    )
    assert len(sidecar.read_text(encoding="utf-8")) < len(old_contents)
    assert all(path.parent == sidecar.parent for path in replacement_sources)
    assert len(set(replacement_sources)) == 2
    assert any(stat.S_ISREG(mode) for mode in fsync_modes)
    assert any(stat.S_ISDIR(mode) for mode in fsync_modes)
    assert list(sidecar.parent.glob(f".{sidecar.name}.*.tmp")) == []

    monkeypatch.setattr(web_runtime.os, "fsync", original_fsync)

    def fail_replace(src, dst):
        assert Path(src).parent == sidecar.parent
        assert Path(dst) == sidecar
        raise OSError("injected replace failure")

    monkeypatch.setattr(web_runtime.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        web_server.dispatch(
            "library_save_asr_review",
            {"args": {"stem": "demo", "review": old_review}},
        )
    monkeypatch.setattr(web_runtime.os, "replace", original_replace)

    assert json.loads(sidecar.read_text(encoding="utf-8")) == replacement
    assert list(sidecar.parent.glob(f".{sidecar.name}.*.tmp")) == []


def test_fastapi_invoke_saves_and_loads_asr_review(tmp_path, monkeypatch):
    from scribe_py import web_server

    root = tmp_path / "transcripts"
    monkeypatch.setattr(web_server, "library_root", lambda: root)
    _write_web_library_task(root)
    review = _valid_asr_review()
    client = TestClient(web_server.app)

    saved = client.post(
        "/api/invoke/library_save_asr_review",
        json={"args": {"stem": "demo", "review": review}},
    )
    loaded = client.post("/api/invoke/library_load", json={"stem": "demo"})

    assert saved.status_code == 200
    assert saved.json() == {"result": None}
    assert loaded.status_code == 200
    assert loaded.json()["result"]["asr_human_review"] == review


@pytest.mark.parametrize(
    ("review", "message"),
    [
        (["not", "an", "object"], "must be a JSON object"),
        ({"schema_version": True, "items": []}, "schema_version must be 1"),
        ({"schema_version": 2, "items": []}, "schema_version must be 1"),
        ({"schema_version": 1, "items": {}}, "items must be an array"),
        (
            {"schema_version": 1, "items": [{"id": " ", "start": 0, "end": 1, "status": "pending"}]},
            "id must not be empty",
        ),
        (
            {
                "schema_version": 1,
                "items": [
                    {"id": "same", "start": 0, "end": 1, "status": "pending"},
                    {"id": "same", "start": 1, "end": 2, "status": "resolved"},
                ],
            },
            "id must be unique",
        ),
        (
            {"schema_version": 1, "items": [{"id": "range", "start": -1, "end": 1, "status": "pending"}]},
            "0 <= start < end",
        ),
        (
            {"schema_version": 1, "items": [{"id": "finite", "start": 0, "end": float("inf"), "status": "pending"}]},
            "end must be a finite number",
        ),
        (
            {
                "schema_version": 1,
                "items": [{"id": "finite", "start": float("nan"), "end": 1, "status": "pending"}],
            },
            "start must be a finite number",
        ),
        (
            {"schema_version": 1, "items": [{"id": "status", "start": 0, "end": 1, "status": "approved"}]},
            "status is invalid",
        ),
        (
            {
                "schema_version": 1,
                "items": [
                    {
                        "id": "text",
                        "start": 0,
                        "end": 1,
                        "status": "pending",
                        "replacement_text": None,
                    }
                ],
            },
            "replacement_text must be a string",
        ),
    ],
)
def test_web_asr_review_save_rejects_invalid_schema(tmp_path, monkeypatch, review, message):
    from scribe_py import web_server

    root = tmp_path / "transcripts"
    monkeypatch.setattr(web_server, "library_root", lambda: root)

    with pytest.raises(HTTPException) as caught:
        web_server.dispatch(
            "library_save_asr_review",
            {"args": {"stem": "demo", "review": review}},
        )

    assert caught.value.status_code == 400
    assert message in caught.value.detail
    assert not (root / "demo" / "asr_human_review.json").exists()


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("{not valid json", "parse ASR human review sidecar"),
        ('{"schema_version":1,"items":[],"extension":NaN}', "parse ASR human review sidecar"),
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "items": [{"id": "x", "start": 0, "end": 1, "status": "approved"}],
                }
            ),
            "validate ASR human review sidecar",
        ),
    ],
)
def test_web_library_load_reports_corrupt_or_invalid_asr_review(tmp_path, monkeypatch, contents, message):
    from scribe_py import web_server

    root = tmp_path / "transcripts"
    monkeypatch.setattr(web_server, "library_root", lambda: root)
    _write_web_library_task(root)
    sidecar = root / "demo" / "asr_human_review.json"
    sidecar.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        web_server.dispatch("library_load", {"stem": "demo"})

    assert message in str(caught.value)
    assert "asr_human_review.json" in str(caught.value)


def _patch_web_library_roots(monkeypatch, root: Path):
    from scribe_py import web_runtime, web_server

    monkeypatch.setattr(web_runtime, "library_root", lambda: root)
    monkeypatch.setattr(web_server, "library_root", lambda: root)
    return web_runtime, web_server


def _raw_save_args(stem: str, raw_json: str, source_audio: str | None = None):
    return {
        "stem": stem,
        "audio_filename": f"{stem}.wav",
        "source_audio": source_audio,
        "txt": "繁體逐字稿",
        "srt": "1\n00:00:00,000 --> 00:00:01,000\n繁體字幕\n",
        "json": raw_json,
        "result": {
            "duration": 1.25,
            "segments": [{"start": 0, "end": 1.25, "text": "繁體"}],
            "backend": "test",
            "model_id": "test-model",
        },
    }


def _corrected_save_args(stem: str):
    return {
        "stem": stem,
        "txt": "校对稿",
        "srt": "1\n00:00:00,000 --> 00:00:01,000\n校对稿\n",
        "json": '{"segments":[{"start":0,"end":1.25,"text":"校对稿"}]}',
        "diff": "- 原文\n+ 校对稿\n",
        "model": "test-corrector",
        "changed": 1,
        "total": 1,
        "glossary": [{"from": "原文", "to": "校对稿"}],
    }


def _snapshot_files(root: Path):
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_web_library_save_raw_and_corrected_supports_atomic_frontend_persistence(
    tmp_path, monkeypatch
):
    root = tmp_path / "transcripts"
    _, web_server = _patch_web_library_roots(monkeypatch, root)
    raw_json = '{"segments":[],"backend":"test","model_id":"test-model"}'

    saved = web_server.dispatch(
        "library_save_raw_and_corrected",
        {
            "args": {
                "raw": _raw_save_args("demo", raw_json),
                "corrected": _corrected_save_args("demo"),
            }
        },
    )

    assert saved["has_corrected"] is True
    assert saved["correction_model"] == "test-corrector"
    assert saved["correction_changed"] == 1
    assert (root / "demo" / "demo.json").read_text(encoding="utf-8") == raw_json
    assert (root / "demo" / "demo_corrected.txt").read_text(encoding="utf-8") == "校对稿"


def test_web_library_save_raw_and_corrected_accepts_raw_only(
    tmp_path, monkeypatch
):
    root = tmp_path / "transcripts"
    _, web_server = _patch_web_library_roots(monkeypatch, root)
    raw_json = '{"segments":[],"backend":"test","model_id":"test-model"}'

    saved = web_server.dispatch(
        "library_save_raw_and_corrected",
        {"args": {"raw": _raw_save_args("demo", raw_json)}},
    )

    assert saved["has_corrected"] is False
    assert (root / "demo" / "demo.json").read_text(encoding="utf-8") == raw_json


def test_web_library_save_raw_and_corrected_can_clear_corrected_metadata(
    tmp_path, monkeypatch
):
    root = tmp_path / "transcripts"
    _, web_server = _patch_web_library_roots(monkeypatch, root)
    raw_json = '{"segments":[],"backend":"test","model_id":"test-model"}'
    raw_args = _raw_save_args("demo", raw_json)

    web_server.dispatch(
        "library_save_raw_and_corrected",
        {
            "args": {
                "raw": raw_args,
                "corrected": _corrected_save_args("demo"),
            }
        },
    )
    saved = web_server.dispatch(
        "library_save_raw_and_corrected",
        {"args": {"raw": raw_args, "clear_corrected": True}},
    )

    assert saved["has_corrected"] is False
    assert saved["correction_model"] is None
    assert saved["correction_changed"] is None
    assert saved["correction_glossary"] is None
    assert (root / "demo" / "demo_corrected.json").is_file()


def test_web_raw_save_preserves_exact_utf8_bytes_and_overlays_audio_only_in_memory(
    tmp_path, monkeypatch
):
    root = tmp_path / "transcripts"
    _, web_server = _patch_web_library_roots(monkeypatch, root)
    source = tmp_path / "source.wav"
    source.write_bytes(b"stable-audio")
    raw_json = (
        '{  "audio" : "caller-original.wav",\n'
        ' "traditional" : "繁體臺灣", "escaped" : "\\u4e2d\\u6587",\n'
        ' "segments" : [], "backend" : "test", "model_id" : "test-model" }\n'
    )

    meta = web_server.dispatch(
        "library_save_raw",
        {"args": _raw_save_args("demo", raw_json, str(source))},
    )

    raw_path = root / "demo" / "demo.json"
    expected = raw_json.encode("utf-8")
    assert raw_path.read_bytes() == expected
    assert meta["audio_path"] == str(root / "demo" / "audio" / "demo.wav")
    loaded = web_server.dispatch("library_load", {"stem": "demo"})
    assert loaded["raw_json"]["audio"] == meta["audio_path"]
    assert raw_path.read_bytes() == expected


def test_web_corrupt_metadata_isolated_from_list_and_reported_by_load_and_saves(
    tmp_path, monkeypatch, caplog
):
    root = tmp_path / "transcripts"
    _, web_server = _patch_web_library_roots(monkeypatch, root)
    _write_web_library_task(root, "good")

    corrupt = root / "corrupt"
    corrupt.mkdir(parents=True)
    (corrupt / "task.json").write_text("{not valid json", encoding="utf-8")
    bad_type = root / "bad-type"
    bad_type.mkdir(parents=True)
    invalid_meta = _valid_library_meta("bad-type")
    invalid_meta["created_at"] = "yesterday"
    (bad_type / "task.json").write_text(json.dumps(invalid_meta), encoding="utf-8")
    missing_fields = root / "missing-fields"
    missing_fields.mkdir(parents=True)
    (missing_fields / "task.json").write_text(
        json.dumps({"stem": "missing-fields", "created_at": 1, "updated_at": 1}),
        encoding="utf-8",
    )
    wrong_stem = root / "wrong-stem"
    wrong_stem.mkdir(parents=True)
    (wrong_stem / "task.json").write_text(
        json.dumps(_valid_library_meta("someone-else")), encoding="utf-8"
    )

    caplog.set_level("WARNING", logger="scribe_py.web_server")
    listed = web_server.dispatch("library_list", {})

    assert [item["stem"] for item in listed] == ["good"]
    assert "skipping corrupt library item corrupt" in caplog.text
    assert "skipping corrupt library item bad-type" in caplog.text
    assert "skipping corrupt library item missing-fields" in caplog.text
    assert "skipping corrupt library item wrong-stem" in caplog.text
    for stem in ("corrupt", "bad-type", "missing-fields", "wrong-stem"):
        with pytest.raises(ValueError, match="metadata"):
            web_server.dispatch("library_load", {"stem": stem})

    before = _snapshot_files(corrupt)
    save_calls = [
        ("library_save_raw", {"args": _raw_save_args("corrupt", '{"segments":[]}')}),
        (
            "library_save_corrected",
            {
                "args": {
                    "stem": "corrupt",
                    "txt": "new",
                    "srt": "new",
                    "json": '{"segments":[]}',
                    "diff": "new",
                    "model": "corrector",
                    "changed": 1,
                    "glossary": None,
                }
            },
        ),
        (
            "library_save_polished",
            {"args": {"stem": "corrupt", "text": "new", "model": "polisher"}},
        ),
        (
            "library_save_asr_review",
            {"args": {"stem": "corrupt", "review": _valid_asr_review()}},
        ),
    ]
    for method, params in save_calls:
        with pytest.raises(ValueError, match="metadata"):
            web_server.dispatch(method, params)
        assert _snapshot_files(corrupt) == before


def test_web_multifile_save_failure_rolls_back_complete_old_file_set(tmp_path, monkeypatch):
    root = tmp_path / "transcripts"
    web_runtime, web_server = _patch_web_library_roots(monkeypatch, root)
    _write_web_library_task(root, "demo")
    first = {
        "stem": "demo",
        "txt": "old corrected",
        "srt": "old srt",
        "json": '{"segments":[{"text":"old"}]}',
        "diff": "old diff",
        "model": "old-model",
        "changed": 1,
        "glossary": None,
    }
    web_server.dispatch("library_save_corrected", {"args": first})
    task = root / "demo"
    before = _snapshot_files(task)
    original_replace = web_runtime.os.replace
    failed = False

    def fail_metadata_once(src, dst):
        nonlocal failed
        if Path(dst).name == "task.json" and not failed:
            failed = True
            raise OSError("injected metadata replace failure")
        return original_replace(src, dst)

    monkeypatch.setattr(web_runtime.os, "replace", fail_metadata_once)
    second = {
        **first,
        "txt": "new corrected",
        "srt": "new srt",
        "json": '{"segments":[{"text":"new"}]}',
        "diff": "new diff",
        "model": "new-model",
    }

    with pytest.raises(OSError, match="injected metadata replace failure"):
        web_server.dispatch("library_save_corrected", {"args": second})

    assert failed
    assert _snapshot_files(task) == before
    assert list(task.rglob("*.tmp")) == []


def test_web_failed_first_raw_save_removes_new_directories_and_allows_retry(tmp_path, monkeypatch):
    root = tmp_path / "transcripts"
    web_runtime, web_server = _patch_web_library_roots(monkeypatch, root)
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    raw_json = '{"segments":[],"backend":"test","model_id":"test-model"}\n'
    args = _raw_save_args("demo", raw_json, str(source))
    original_replace = web_runtime.os.replace
    failed = False

    def fail_second_artifact_once(src, dst):
        nonlocal failed
        if Path(dst).name == "demo.srt" and not failed:
            failed = True
            raise OSError("injected first-save failure")
        return original_replace(src, dst)

    monkeypatch.setattr(web_runtime.os, "replace", fail_second_artifact_once)
    with pytest.raises(OSError, match="injected first-save failure"):
        web_server.dispatch("library_save_raw", {"args": args})

    assert failed
    assert not (root / "demo").exists()
    monkeypatch.setattr(web_runtime.os, "replace", original_replace)

    saved = web_server.dispatch("library_save_raw", {"args": args})
    assert saved["stem"] == "demo"
    assert (root / "demo" / "demo.json").read_bytes() == raw_json.encode("utf-8")
    assert (root / "demo" / "audio" / "demo.wav").read_bytes() == b"audio"


def test_web_raw_save_migrates_external_audio_to_stable_library_copy(tmp_path, monkeypatch):
    root = tmp_path / "transcripts"
    _, web_server = _patch_web_library_roots(monkeypatch, root)
    task = root / "demo"
    task.mkdir(parents=True)
    external = tmp_path / "external.wav"
    external.write_bytes(b"external-audio")
    (task / "task.json").write_text(
        json.dumps(_valid_library_meta("demo", str(external))), encoding="utf-8"
    )
    (task / "demo.json").write_text(
        '{"segments":[],"backend":"test","model_id":"test-model"}', encoding="utf-8"
    )
    raw_json = '{"segments":[],"backend":"test","model_id":"test-model","version":2}'

    meta = web_server.dispatch(
        "library_save_raw", {"args": _raw_save_args("demo", raw_json)}
    )

    stable = root / "demo" / "audio" / "demo.wav"
    assert meta["audio_path"] == str(stable)
    assert stable.read_bytes() == b"external-audio"


def test_web_save_and_archive_are_serialized_by_the_same_stem_lock(tmp_path, monkeypatch):
    root = tmp_path / "transcripts"
    _, web_server = _patch_web_library_roots(monkeypatch, root)
    initial_raw = '{"audio":"old.wav","segments":[],"backend":"test","model_id":"test-model"}\n'
    web_server.dispatch(
        "library_save_raw", {"args": _raw_save_args("demo", initial_raw)}
    )

    entered_save = threading.Event()
    release_save = threading.Event()
    archive_started = threading.Event()
    archive_finished = threading.Event()
    original_transaction = web_server.transactional_write_files
    failures = []
    archived_paths = []

    def blocking_transaction(updates):
        entered_save.set()
        assert release_save.wait(timeout=2)
        return original_transaction(updates)

    monkeypatch.setattr(web_server, "transactional_write_files", blocking_transaction)
    next_raw = (
        '{ "audio" : "caller.wav", "segments" : [], '
        '"backend" : "test", "model_id" : "test-model", "version" : 2 }\n'
    )

    def save_task():
        try:
            web_server.dispatch(
                "library_save_raw", {"args": _raw_save_args("demo", next_raw)}
            )
        except Exception as exc:
            failures.append(exc)

    def archive_task():
        archive_started.set()
        try:
            archived_paths.append(web_server.dispatch("library_archive", {"stem": "demo"}))
        except Exception as exc:
            failures.append(exc)
        finally:
            archive_finished.set()

    save_thread = threading.Thread(target=save_task)
    archive_thread = threading.Thread(target=archive_task)
    save_thread.start()
    assert entered_save.wait(timeout=1)
    archive_thread.start()
    assert archive_started.wait(timeout=1)
    assert not archive_finished.wait(timeout=0.1)

    release_save.set()
    save_thread.join(timeout=3)
    archive_thread.join(timeout=3)

    assert not save_thread.is_alive()
    assert not archive_thread.is_alive()
    assert failures == []
    archived = Path(archived_paths[0])
    assert archived.is_dir()
    assert (archived / "demo.json").read_bytes() == next_raw.encode("utf-8")
    assert not (root / "demo").exists()


def test_app_transcribe_request_does_not_enable_recording_specific_normalizer_profile():
    source = (Path(__file__).parents[2] / "src" / "hooks" / "usePipeline.ts").read_text()

    assert 'normalizer_profile: "legacy_general"' not in source


@pytest.mark.parametrize("profile", ["standard3", "legacy_general", "STANDARD3", ""])
def test_app_sidecar_ignores_recording_specific_normalizer_profiles(profile: str):
    assert ipc._app_normalizer_profile({"normalizer_profile": profile}) is None
