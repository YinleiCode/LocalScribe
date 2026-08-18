#!/usr/bin/env python3
"""批量检查说话人分离推荐结果。

脚本不修改历史文件。它会对仍能找到源音频的转录跑 2-8 人候选，
输出中文 TSV 表格，方便直接贴到测试记录或客户说明里。
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scribe-py" / "src"))

from scribe_py.ipc import handle_recommend_diarization  # noqa: E402


SKIP_SUFFIXES = (
    "_report.json",
    "_corrected.json",
)

EXPECTED_SPEAKERS = {
    "冯文良": 2,
    "冯文梁": 2,
    "吴玉泉": 2,
    "标准录音 2": 4,
}


CONFIDENCE_LABEL = {
    "high": "高",
    "medium": "中",
    "low": "低",
}


def iter_transcripts(root: Path):
    for path in sorted(root.glob("*/*.json")):
        if path.name == "task.json" or path.name.endswith(SKIP_SUFFIXES):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        segments = data.get("segments") or []
        audio = data.get("audio") or data.get("source_audio") or data.get("audio_path")
        if isinstance(segments, list) and segments and audio:
            yield path, data, Path(audio)


def audio_key(audio: Path) -> tuple[str, int, int]:
    resolved = audio.expanduser().resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns)


def segment_shape_key(segments: list[dict]) -> str:
    """Fingerprint only ASR timing/text, not existing speaker labels."""
    digest = hashlib.sha1()
    for seg in segments:
        digest.update(f"{float(seg.get('start') or 0):.3f}|{float(seg.get('end') or 0):.3f}|".encode())
        digest.update(str(seg.get("text") or "").encode("utf-8", errors="ignore"))
        digest.update(b"\n")
    return digest.hexdigest()


def print_row(values: list[Any], sink) -> None:
    print("\t".join(str(v) for v in values), file=sink)


def expected_for_name(name: str) -> int | None:
    for marker, expected in EXPECTED_SPEAKERS.items():
        if marker in name:
            return expected
    return None


def fmt_bool(value: bool) -> str:
    return "是" if value else "否"


def speaker_label(name: str) -> str:
    return str(name).replace("SPEAKER_", "")


def fmt_review_segments(items: list[dict], limit: int = 5) -> str:
    parts = []
    for s in items[:limit]:
        start = float(s.get("start") or 0.0)
        minute = int(start // 60)
        second = int(start % 60)
        from_speaker = speaker_label(s.get("from_speaker", ""))
        to_speaker = speaker_label(s.get("to_speaker", ""))
        change = (
            f"{from_speaker}->{to_speaker}"
            if from_speaker and to_speaker and from_speaker != to_speaker
            else f"{from_speaker or to_speaker}待确认"
        )
        parts.append(
            f"{minute:02d}:{second:02d} {change}"
        )
    if len(items) > limit:
        parts.append(f"另{len(items) - limit}段")
    return " / ".join(parts)


def pick_candidate(rec: dict, candidate_n: int) -> dict | None:
    candidates = rec.get("candidates") or []
    if not candidates:
        return None
    return next(
        (c for c in candidates if int(c.get("n_speakers") or 0) == candidate_n),
        candidates[0],
    )


def fmt_distribution(candidate: dict | None) -> str:
    if not candidate:
        return ""
    speakers = (candidate.get("summary") or {}).get("speakers") or []
    return " / ".join(
        f"{speaker_label(s.get('speaker', ''))}{s.get('segments', '')}"
        for s in speakers
    )


def fmt_recommend_error(rec: dict) -> str:
    reason = str(rec.get("reason") or rec.get("error") or "").strip()
    errors = rec.get("errors") or []
    if isinstance(errors, list) and errors:
        error_text = " / ".join(str(e) for e in errors[:3])
        return f"{reason}; {error_text}" if reason else error_text
    return reason or "没有可用候选结果，通常是音频损坏或解码失败"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(Path.home() / "Library/Application Support/LocalScribe/transcripts"),
        help="LocalScribe transcript library root",
    )
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条有效转录; 0=不限")
    parser.add_argument("--out", type=Path, default=None, help="可选 TSV 输出路径")
    parser.add_argument("--include", action="append", default=[], help="只处理名称包含该文本的录音; 可重复传入")
    parser.add_argument("--max-duration", type=float, default=0.0, help="跳过超过该秒数的录音; 0=不限")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    out_handle = None
    sink = sys.stdout
    if args.out:
        out_path = args.out.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_handle = out_path.open("w", encoding="utf-8")
        sink = out_handle

    print_row([
        "录音名",
        "段数",
        "源音频存在",
        "当前人数",
        "推荐人数",
        "期望人数",
        "真值结果",
        "置信度",
        "候选人数",
        "分布",
        "合并",
        "抽听片段",
        "评审结论",
        "原因",
        "缓存",
    ], sink)

    cache: dict[tuple[tuple[str, int, int], str], dict] = {}
    processed = 0
    try:
        for path, data, audio in iter_transcripts(root):
            name = path.parent.name
            if args.include and not any(marker in name for marker in args.include):
                continue
            duration = float(data.get("duration") or 0.0)
            if args.max_duration and duration > args.max_duration:
                continue
            if args.limit and processed >= args.limit:
                break
            processed += 1
            segments = data.get("segments") or []
            current = len({s.get("speaker") for s in segments if s.get("speaker")})
            if not audio.exists():
                print_row([
                    path.parent.name,
                    len(segments),
                    "否",
                    current,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "需排查",
                    f"源音频不存在: {audio}",
                    "",
                ], sink)
                continue

            try:
                key = (audio_key(audio), segment_shape_key(segments))
                cache_hit = key in cache
                if cache_hit:
                    rec = cache[key]
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        rec = handle_recommend_diarization({
                            "audio": str(audio),
                            "segments": segments,
                            "min_speakers": args.min_speakers,
                            "max_speakers": args.max_speakers,
                            "profiles": [],
                        })
                    cache[key] = rec

                candidate_n = int(
                    rec.get("recommended_candidate_n_speakers")
                    or rec.get("recommended_n_speakers")
                    or 0
                )
                best = pick_candidate(rec, candidate_n)
                if best is None:
                    expected = expected_for_name(path.parent.name)
                    recommended = int(rec.get("recommended_n_speakers") or 0)
                    print_row([
                        path.parent.name,
                        len(segments),
                        "是",
                        current,
                        recommended or "",
                        expected if expected is not None else "",
                        "",
                        CONFIDENCE_LABEL.get(str(rec.get("confidence") or ""), str(rec.get("confidence") or "")),
                        candidate_n or "",
                        "",
                        "",
                        "",
                        "需排查",
                        fmt_recommend_error(rec),
                        "命中" if cache_hit else "新算",
                    ], sink)
                    continue
                distribution = fmt_distribution(best)
                merge_map = rec.get("merge_map") or {}
                merge = " / ".join(
                    f"{speaker_label(k)}->{speaker_label(v)}"
                    for k, v in merge_map.items()
                )
                review_segments = rec.get("review_segments") or best.get("review_segments") or []
                review = []
                recommended = int(rec.get("recommended_n_speakers") or 0)
                if current and current != recommended:
                    review.append(f"当前{current}!=推荐{recommended}")
                if (
                    best.get("weak_speakers", 0)
                    or best.get("tiny_speakers", 0)
                    or best.get("fragmented_speakers", 0)
                    or best.get("marginal_speakers", 0)
                ):
                    review.append("含弱/碎片说话人")
                expected = expected_for_name(path.parent.name)
                truth = ""
                if expected is not None:
                    truth = "通过" if recommended == expected else "未通过"
                    if truth == "未通过":
                        review.append(f"真值{expected}!=推荐{recommended}")
                if not review:
                    review.append("低风险")
                print_row([
                    path.parent.name,
                    len(segments),
                    "是",
                    current,
                    recommended,
                    expected if expected is not None else "",
                    truth,
                    CONFIDENCE_LABEL.get(str(rec.get("confidence") or ""), str(rec.get("confidence") or "")),
                    candidate_n,
                    distribution,
                    merge,
                    fmt_review_segments(review_segments),
                    "、".join(review),
                    str(rec.get("reason") or ""),
                    "命中" if cache_hit else "新算",
                ], sink)
            except Exception as exc:
                print_row([
                    path.parent.name,
                    len(segments),
                    "是",
                    current,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "需排查",
                    f"{type(exc).__name__}: {exc}",
                    "",
                ], sink)
    finally:
        if out_handle:
            out_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
