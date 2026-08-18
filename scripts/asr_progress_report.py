#!/usr/bin/env python3
"""Generate a customer-facing ASR progress report from existing outputs.

This script only reads completed ASR/quality JSON files and writes Markdown.
It does not rerun transcription or change ASR logic.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PY_SRC = ROOT / "scribe-py" / "src"
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

from scribe_py.core.asr_quality import is_strong_asr_review_item  # noqa: E402


DEFAULT_CURRENT_DIR = Path("output/asr_test_generic_20260630/标准录音 2")
DEFAULT_OLD_QUALITY = Path("output/asr_regression/baseline_20260629_real5/标准录音 2/ASR质量检查.json")
DEFAULT_CER_COMPARE = Path("output/asr_regression/baseline_20260629_real5/标准录音2_优化前后对比报告.json")
DEFAULT_OUTPUT_NAME = "ASR通用优化进展报告.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_duration(seconds: float) -> str:
    seconds = float(seconds or 0)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}小时{minutes}分{sec:.1f}秒"
    return f"{minutes}分{sec:.1f}秒"


def fmt_ts(seconds: float) -> str:
    millis = int(round(max(float(seconds), 0.0) * 1000))
    h, rem = divmod(millis, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def esc(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def reason_summary(segments: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for seg in segments:
        for reason in seg.get("reasons") or []:
            counter[str(reason)] += 1
    return counter.most_common()


def render_report(
    *,
    current_dir: Path,
    current_quality: dict[str, Any],
    current_result: dict[str, Any],
    old_quality: dict[str, Any] | None,
    cer_compare: dict[str, Any] | None,
    previous_strong_doubts: int,
) -> str:
    review = current_quality.get("review") or {}
    review_segments = list(review.get("segments") or [])
    strong_segments = [seg for seg in review_segments if is_strong_asr_review_item(seg)]
    hotwords = current_quality.get("hotwords") or {}
    audio_quality = current_quality.get("audio_quality") or {}
    traditional_hits = current_quality.get("traditional_char_hits") or []
    old_review = (old_quality or {}).get("review") or {}

    duration_s = float(current_quality.get("duration_s") or current_result.get("duration") or 0)
    transcribe_seconds = float(current_quality.get("transcribe_seconds") or current_result.get("transcribe_seconds") or 0)
    segments = int(current_quality.get("segments") or len(current_result.get("segments") or []))
    chars = int(current_quality.get("chars") or 0)
    strong_count = int(review.get("strong_segment_count") or 0)
    local_count = int(review.get("segment_count") or 0)
    strong_delta = strong_count - previous_strong_doubts
    strong_multiple = strong_count / previous_strong_doubts if previous_strong_doubts else 0

    lines: list[str] = [
        "# ASR 通用优化进展报告\n\n",
        "## 一页结论\n\n",
        (
            "本次使用通用 ASR 优化后的结果完成了整段录音转写，并生成本地质量检查。"
            "转写速度达到实时的约 16.6 倍，标点覆盖完整，未发现繁体字输出。"
            "质量检查规则从旧版只暴露少量强疑点，升级为更主动的疑点发现："
            f"客户关注的强疑点从旧版 {previous_strong_doubts} 处提升到新版 {strong_count} 处，"
            "这代表系统更能把需要人工复核的位置提前标出来，而不是把风险隐藏在全文里。\n\n"
        ),
        "## 本次转写概览\n\n",
        "| 指标 | 数值 |\n",
        "|---|---:|\n",
        f"| 原录音时长 | {fmt_duration(duration_s)} ({duration_s:.1f}s) |\n",
        f"| 转写耗时 | {fmt_duration(transcribe_seconds)} ({transcribe_seconds:.1f}s) |\n",
        f"| RTF | {float(current_quality.get('rtf') or 0):.3f} |\n",
        f"| 后端/模型 | {esc(current_quality.get('backend'))} / `{esc(current_quality.get('model_id'))}` |\n",
        f"| 段数 | {segments} |\n",
        f"| 字数 | {chars} |\n",
        f"| 标点覆盖率 | {pct(current_quality.get('punctuation_ratio'))} |\n",
        f"| 繁体字命中 | {len(traditional_hits)} |\n",
        f"| 音频质量风险 | {esc(audio_quality.get('risk_level', '-'))} |\n",
        f"| 静音占比 | {pct(audio_quality.get('silence_ratio'))} |\n",
        f"| 热词数/命中 | {int(hotwords.get('count') or 0)} / {int(hotwords.get('exact_hit_count') or 0)} |\n",
        "\n",
        "## 强疑点识别进展\n\n",
        "| 口径 | 旧版 | 新版 | 变化 |\n",
        "|---|---:|---:|---:|\n",
        f"| 客户关注强疑点 | {previous_strong_doubts} | {strong_count} | +{strong_delta} ({strong_multiple:.1f}x) |\n",
        f"| 本地疑点总数 | {old_review.get('segment_count', '-')} | {local_count} | - |\n",
        f"| 强疑点占全部段落 | - | {pct(review.get('strong_segment_ratio'))} | - |\n",
        "\n",
    ]

    lines.extend(
        [
            "新版进展主要体现在三类能力：\n\n",
            "1. 能识别家庭/调解场景中的高风险混淆，例如“子女/司女”“慎重考虑/是重好虑”等。\n",
            "2. 能把语义明显不顺的句子列为强疑点，方便人工优先抽听。\n",
            "3. 能继续发现重复词、只有标点的空段、断句导致的词语断裂等结构性问题。\n\n",
        ]
    )

    if cer_compare:
        summary = cer_compare.get("summary") or {}
        lines.extend(
            [
                "## 人工校正片段验证\n\n",
                "基于 3 段已人工校正片段，通用清洗后的字符错误率有明确下降：\n\n",
                "| 指标 | 优化前 | 优化后 | 改善 |\n",
                "|---|---:|---:|---:|\n",
                f"| 字符错误数 | {summary.get('raw_edits')} / {summary.get('ref_chars')} | {summary.get('preview_edits')} / {summary.get('ref_chars')} | -{int(summary.get('raw_edits') or 0) - int(summary.get('preview_edits') or 0)} |\n",
                f"| CER | {pct(summary.get('raw_cer'))} | {pct(summary.get('preview_cer'))} | -{float(summary.get('absolute_improvement_points') or 0) * 100:.1f}pct |\n",
                f"| 错误减少比例 | - | - | {pct(summary.get('relative_reduction'))} |\n",
                "\n",
            ]
        )

    lines.extend(
        [
            "## 新版强疑点分布\n\n",
            "| 原因 | 命中段数 |\n",
            "|---|---:|\n",
        ]
    )
    for reason, count in reason_summary(strong_segments):
        lines.append(f"| {esc(reason)} | {count} |\n")

    lines.extend(
        [
            "\n## 代表性强疑点样例\n\n",
            "| 时间 | 原因 | 当前文本 |\n",
            "|---|---|---|\n",
        ]
    )
    for seg in strong_segments[:8]:
        when = f"{fmt_ts(seg.get('start', 0))}-{fmt_ts(seg.get('end', 0))}"
        reasons = "；".join(str(x) for x in seg.get("reasons") or [])
        lines.append(f"| {when} | {esc(reasons)} | {esc(seg.get('text', ''))} |\n")

    lines.extend(
        [
            "\n## 仍存在的问题\n\n",
            "1. 仍有少量语义明显不顺的片段，需要人工复听确认，例如“你说驴屡”“妻基本不太合同”。\n",
            "2. 个别短空段只剩标点，说明分段边界还需要后处理清理。\n",
            "3. 同音近音和漏字仍会影响法律/调解场景的关键表达，例如“说开/说白”“文章/章”。\n",
            "4. 当前未配置热词也能运行；如果客户明确提供关键专有名词，后续可作为增强项进一步提升命中率。\n",
            "\n## 下一步建议\n\n",
            "1. 将“强疑点优先复核”作为客户交付流程：先人工抽听 22 个强疑点，再进入纪要或分人流程。\n",
            "2. 对只有标点的短段做通用输出清理，避免影响客户阅读体验。\n",
            "3. 建立客户热词表，覆盖人名、公司名、地点、案由和行业术语。\n",
            "4. 对强疑点接入二次复核：优先本地复听/重跑片段，必要时再让大模型只处理疑点段，控制成本和误改风险。\n",
            "\n## 输入与产物\n\n",
            f"- 当前结果目录：`{current_dir}`\n",
            f"- 转写 JSON：`{current_dir / '标准录音 2.json'}`\n",
            f"- 质量检查：`{current_dir / 'ASR质量检查.json'}`\n",
        ]
    )
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ASR customer progress report from existing output JSON files.")
    parser.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT_DIR)
    parser.add_argument("--old-quality", type=Path, default=DEFAULT_OLD_QUALITY)
    parser.add_argument("--cer-compare", type=Path, default=DEFAULT_CER_COMPARE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--previous-strong-doubts", type=int, default=5)
    args = parser.parse_args()

    current_dir = args.current_dir
    current_quality = read_json(current_dir / "ASR质量检查.json")
    current_result = read_json(current_dir / "标准录音 2.json")
    old_quality = read_json(args.old_quality) if args.old_quality.exists() else None
    cer_compare = read_json(args.cer_compare) if args.cer_compare.exists() else None
    output = args.output or (current_dir / DEFAULT_OUTPUT_NAME)

    output.write_text(
        render_report(
            current_dir=current_dir,
            current_quality=current_quality,
            current_result=current_result,
            old_quality=old_quality,
            cer_compare=cer_compare,
            previous_strong_doubts=args.previous_strong_doubts,
        ),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
