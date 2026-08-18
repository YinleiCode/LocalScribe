// Client-side formatters for display + export.

import type { Segment, TranscribeResult } from "./ipc";
import { segmentCues } from "./transcript-sync";

function speakerAwareSegments(segments: Segment[]): Segment[] {
  const out: Segment[] = [];
  for (const segment of segments) {
    const cues = segmentCues(segment);
    const hasProjectedHandoff = cues.some((cue) => cue.speaker && cue.speaker !== segment.speaker);
    if (!hasProjectedHandoff) {
      out.push(segment);
      continue;
    }
    for (const cue of cues) {
      const piece: Segment = {
        ...segment,
        start: cue.start,
        end: cue.end,
        text: cue.text,
        speaker: cue.speaker || segment.speaker,
        sync_cues: [{ start: cue.start, end: cue.end, text: cue.text }],
      };
      delete piece.speaker_cues;
      out.push(piece);
    }
  }
  return out;
}

export function fmtTs(seconds: number, comma = false): string {
  const ms = Math.round(seconds * 1000);
  const h = Math.floor(ms / 3_600_000).toString().padStart(2, "0");
  const m = Math.floor((ms % 3_600_000) / 60_000).toString().padStart(2, "0");
  const s = Math.floor((ms % 60_000) / 1000).toString().padStart(2, "0");
  const milli = (ms % 1000).toString().padStart(3, "0");
  return `${h}:${m}:${s}${comma ? "," : "."}${milli}`;
}

export function fmtDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}秒`;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}分${s}秒`;
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export function buildTxt(segments: Segment[], header?: string): string {
  const lines: string[] = [];
  if (header) {
    for (const h of header.split("\n")) lines.push(`# ${h}`);
    lines.push("");
  }
  for (const s of speakerAwareSegments(segments)) {
    const t = s.text.trim();
    if (!t) continue;
    const spk = s.speaker ? `[${s.speaker}] ` : "";
    lines.push(`[${fmtTs(s.start)} - ${fmtTs(s.end)}] ${spk}${t}`);
  }
  return lines.join("\n") + "\n";
}

export function buildSrt(segments: Segment[]): string {
  const out: string[] = [];
  let idx = 1;
  for (const s of speakerAwareSegments(segments)) {
    const t = s.text.trim();
    if (!t) continue;
    out.push(String(idx));
    out.push(`${fmtTs(s.start, true)} --> ${fmtTs(s.end, true)}`);
    // SRT 标准里没有正式的 speaker 字段,放在文本前用方括号是惯例
    out.push(s.speaker ? `[${s.speaker}] ${t}` : t);
    out.push("");
    idx += 1;
  }
  return out.join("\n");
}

export function buildJson(result: TranscribeResult, extra?: Record<string, unknown>): string {
  return JSON.stringify({ ...result, ...(extra ?? {}) }, null, 2);
}

export function buildMd(segments: Segment[], title?: string): string {
  const lines: string[] = [];
  if (title) {
    lines.push(`# ${title}`);
    lines.push("");
  }
  let lastSpeaker: string | undefined;
  for (const s of speakerAwareSegments(segments)) {
    const t = s.text.trim();
    if (!t) continue;
    // 说话人切换时插一个 H2 分隔,同人连续段不重复标
    if (s.speaker && s.speaker !== lastSpeaker) {
      lines.push("");
      lines.push(`## ${s.speaker}`);
      lines.push("");
      lastSpeaker = s.speaker;
    }
    lines.push(`<a id="t-${Math.floor(s.start * 1000)}"></a>`);
    lines.push(`**[${fmtTs(s.start)}]** ${t}`);
    lines.push("");
  }
  return lines.join("\n");
}
