#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf

from scribe_py.core.sensevoice_recovery import decide_asymmetric_context_evidence
from scribe_py.core.transcriber_funasr import FunASRTranscriber, _clean_text, _speech_coverage_diagnostics
from scribe_py.core.types import Segment, TranscribeOptions


def _decode(transcriber: FunASRTranscriber, model: Any, audio: Path, options: TranscribeOptions, start: float, end: float, path: Path) -> tuple[str, str]:
    info = sf.info(str(audio))
    data, sr = sf.read(
        str(audio), dtype="float32",
        start=max(0, int(round(start * info.samplerate))),
        stop=min(info.frames, int(round(end * info.samplerate))),
    )
    sf.write(str(path), data, sr)
    result = transcriber._generate(model, path, options, sensevoice=True)
    items = result if isinstance(result, list) else [result]
    text = " ".join(
        cleaned
        for item in items
        if isinstance(item, dict)
        for cleaned in [_clean_text(str(item.get("text") or item.get("raw_text") or ""), sensevoice=True)]
        if cleaned
    ).strip()
    return text, hashlib.sha256(path.read_bytes()).hexdigest()


def _local_reference(segments: list[dict[str, Any]], start: float, end: float) -> str:
    return "".join(
        str(item.get("text") or "")
        for item in segments
        if float(item.get("end") or 0.0) > start and float(item.get("start") or 0.0) < end
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay strict failed cores with asymmetric SenseVoice context")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pads", default="0.25,0.5,1.0")
    parser.add_argument("--model", default="iic/SenseVoiceSmall")
    args = parser.parse_args()

    artifact = json.loads(args.transcript.read_text())
    coverage = artifact["filter_stats"]["speech_coverage"]
    windows = list(coverage.get("strict_probe_windows") or [])
    failed = [item for item in windows if item.get("status") == "failed"]
    segments = list(artifact.get("segments") or [])
    duration = float(artifact.get("duration") or sf.info(str(args.audio)).duration)
    options = TranscribeOptions(model_id=args.model, language="zh", audio_preprocess="off")
    transcriber = FunASRTranscriber(backend_name="sensevoice")
    model = transcriber._load(args.model)
    pads = [float(item) for item in args.pads.split(",") if item.strip()]
    results: list[dict[str, Any]] = []

    baseline_recognized = [
        (float(item["core_start"]), float(item["core_end"]))
        for item in windows if item.get("status") == "recognized"
    ]
    speech_ranges = [
        (float(item["start"]), float(item["end"]))
        for item in coverage.get("speech_intervals") or []
    ]
    with tempfile.TemporaryDirectory(prefix="localscribe-asymmetric-replay-") as tmp:
        root = Path(tmp)
        for pad in pads:
            accepted: list[tuple[float, float]] = []
            details = []
            for index, window in enumerate(failed):
                core_start = float(window["core_start"])
                core_end = float(window["core_end"])
                left_start, left_end = max(0.0, core_start - pad), core_end
                right_start, right_end = core_start, min(duration, core_end + pad)
                try:
                    left_text, left_hash = _decode(transcriber, model, args.audio, options, left_start, left_end, root / f"p{pad}_{index}_left.wav")
                    right_text, right_hash = _decode(transcriber, model, args.audio, options, right_start, right_end, root / f"p{pad}_{index}_right.wav")
                    decision = decide_asymmetric_context_evidence(
                        core_start=core_start, core_end=core_end,
                        left_start=left_start, left_end=left_end,
                        right_start=right_start, right_end=right_end,
                        left_text=left_text, right_text=right_text,
                        left_slice_sha256=left_hash, right_slice_sha256=right_hash,
                        local_reference=_local_reference(segments, core_start, core_end),
                        speech_duration_s=float(window.get("speech_duration_s") or core_end - core_start),
                        min_chars_per_s=float(coverage.get("wallclock_min_chars_per_s") or 0.75),
                    )
                except Exception as exc:
                    decision = {"decision": "rejected", "rejection_reason": f"replay_failed:{type(exc).__name__}"}
                if decision.get("decision") == "matched_existing":
                    accepted.append((core_start, core_end))
                details.append({"core_start": core_start, "core_end": core_end, **decision})
            diagnostics = _speech_coverage_diagnostics(
                speech_ranges,
                [Segment(start=start, end=end, text="recognized") for start, end in baseline_recognized + accepted],
                duration=duration, collar_s=0.0, vad_status="ok", vad_reason="asymmetric_replay",
            )
            results.append({
                "pad_s": pad,
                "failed_windows": len(failed),
                "accepted_windows": len(accepted),
                "recovered_wallclock_s": round(sum(end - start for start, end in accepted), 3),
                "projected": diagnostics,
                "details": details,
            })

    output = {
        "audio": str(args.audio.resolve()),
        "transcript": str(args.transcript.resolve()),
        "frozen_text_sha256": coverage.get("local_recovery", {}).get("after", {}).get("text_sha256"),
        "baseline_failed_windows": len(failed),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({
        "out": str(args.out),
        "results": [
            {"pad_s": item["pad_s"], "accepted_windows": item["accepted_windows"], "projected_ratio": item["projected"].get("speech_coverage_ratio"), "projected_max_gap_s": item["projected"].get("max_uncovered_speech_s")}
            for item in results
        ],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
