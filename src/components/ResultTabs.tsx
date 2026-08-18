import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from "react";
import clsx from "clsx";

import { buildJson, buildMd, buildSrt, buildTxt, fmtDuration } from "../lib/format";
import {
  captureTaskCorrectedRevision,
  committedRawRevisionForStem,
  correctedArtifactFingerprint,
  enqueueStemWrite,
  persistCorrectedArtifact,
  persistCorrectedTransaction,
  type CorrectedArtifact,
  type CorrectedRevision,
  type CorrectedSavePayload,
} from "../lib/corrected-persistence";
import type {
  ASRHumanReview,
  ASRHumanReviewItemStatus,
  ASRHumanReviewStatus,
  ASRLocalRecoveryDecision,
  ASRStrongReviewStats,
  DiarizationCandidate,
  RecommendDiarizationResponse,
  Segment,
  SpeechCoverageReport,
  TermConsistencyCandidate,
  TranscribeResult,
  VoiceprintAnchor,
  VoiceprintAnchorPreflightCandidate,
  VoiceprintProfile,
  VoiceprintReidentifyResponse,
} from "../lib/ipc";
import { assertDiarizationPreservesTranscript, ipc, libraryStemFromFilename, libraryStemKey } from "../lib/ipc";
import { downloadTextFile, localMediaUrl, showMessage } from "../lib/runtime";
import { groupArticleSpeakerTurns } from "../lib/article-speaker-groups";
import { syncSpeakerMetadata } from "../lib/speaker-metadata";
import {
  activeCueAt,
  activeCueIndexAt,
  activeSegmentIndexAt,
  segmentCues,
  type TranscriptSyncCue,
} from "../lib/transcript-sync";
import { useSettings } from "../stores/settings-store";
import { type Task, useTasks } from "../stores/tasks-store";
import { Article, Check, Copy, Download, FileText, Hourglass, Lock, Mic, Pause, Pencil, Play, Refresh, Globe, Warning } from "./Icons";
import PinArticleDialog from "./PinArticleDialog";

type Tab = "raw" | "corrected" | "article" | "translated";
type ViewMode = "timeline" | "dialog";

type SpeakerChipSegmentFlags = Pick<
  Segment,
  | "speaker_handoff_review"
  | "speaker_handoff_text_review"
  | "speaker_resegmentation_review"
  | "speaker_assignment_review"
  | "speaker_review_reason"
  | "voice_band_repaired"
  | "continuity_repaired"
  | "speaker_handoff_voice_guard_repaired"
  | "speaker_overlap_risk"
  | "speaker_overlap_candidates"
  | "speaker_voiceprint_reidentified"
  | "speaker_voiceprint_review"
  | "speaker_voiceprint_score"
  | "speaker_voiceprint_anchor"
>;

const TAB_META: Record<Tab, { label: string; icon: React.ReactNode }> = {
  raw: { label: "原文", icon: <FileText size={13} /> },
  corrected: { label: "校对", icon: <Pencil size={13} /> },
  article: { label: "文章", icon: <Article size={13} /> },
  translated: { label: "译文", icon: <Globe size={13} /> },
};

// VSCode-friendly palette,8 路循环
const SPEAKER_PALETTE = [
  "text-sky-300 border-sky-300/40 bg-sky-500/10",
  "text-orange-300 border-orange-300/40 bg-orange-500/10",
  "text-emerald-300 border-emerald-300/40 bg-emerald-500/10",
  "text-pink-300 border-pink-300/40 bg-pink-500/10",
  "text-violet-300 border-violet-300/40 bg-violet-500/10",
  "text-yellow-300 border-yellow-300/40 bg-yellow-500/10",
  "text-cyan-300 border-cyan-300/40 bg-cyan-500/10",
  "text-rose-300 border-rose-300/40 bg-rose-500/10",
];

function collectSpeakers(segments: Segment[]): string[] {
  const out: string[] = [];
  for (const s of segments) {
    const overlapNames = (s.speaker_overlap_candidates ?? []).flatMap((candidate) => [
      candidate.primary_speaker,
      candidate.secondary_speaker,
    ]);
    const names = [s.speaker, ...(s.speaker_cues ?? []).map((cue) => cue.speaker), ...overlapNames];
    for (const name of names) {
      const speaker = normalizeSpeakerName(name || "");
      if (speaker && isValidSpeakerAnchorName(speaker) && !out.includes(speaker)) out.push(speaker);
    }
  }
  return out;
}

function normalizeSpeakerName(value: string): string {
  const raw = value.trim();
  if (!raw) return "";
  const match = raw.match(/^(?:SPEAKER_)?([A-Za-z])$/);
  if (match) return `SPEAKER_${match[1].toUpperCase()}`;
  const speakerMatch = raw.match(/^SPEAKER_([A-Za-z])$/i);
  if (speakerMatch) return `SPEAKER_${speakerMatch[1].toUpperCase()}`;
  return raw;
}

function isValidSpeakerAnchorName(value: string): boolean {
  const normalized = normalizeSpeakerName(value);
  if (!normalized) return false;
  if (/[\\/|,，、;；\s]/.test(normalized)) return false;
  return true;
}

function displaySpeakerName(value?: string): string | undefined {
  const normalized = normalizeSpeakerName(value || "");
  return normalized || value;
}

function speakerChipClass(speakers: string[], who: string): string {
  const idx = speakers.indexOf(who);
  return SPEAKER_PALETTE[idx >= 0 ? idx % SPEAKER_PALETTE.length : 0];
}

function shortSpeakerName(value: string): string {
  return normalizeSpeakerName(value).replace(/^SPEAKER_/, "");
}

function speakerOverlapSummary(segment?: SpeakerChipSegmentFlags): string {
  const pairs: string[] = [];
  for (const candidate of segment?.speaker_overlap_candidates ?? []) {
    const primary = shortSpeakerName(candidate.primary_speaker);
    const secondary = shortSpeakerName(candidate.secondary_speaker);
    if (!primary || !secondary) continue;
    const pair = `${primary}+${secondary}`;
    if (!pairs.includes(pair)) pairs.push(pair);
  }
  if (!pairs.length) return "";
  const visible = pairs.slice(0, 2).join(" / ");
  const remainder = pairs.length > 2 ? ` +${pairs.length - 2}` : "";
  return `重叠 ${visible}${remainder}，待确认`;
}

function segmentSpeakerReviewReason(segment?: SpeakerChipSegmentFlags): string {
  if (!segment) return "";
  const reasons: string[] = [];
  if (segment.speaker_handoff_review || segment.speaker_handoff_text_review) reasons.push("换人边界待确认");
  if (segment.speaker_resegmentation_review) reasons.push("段内短声纹待确认");
  if (segment.speaker_assignment_review) reasons.push(segment.speaker_review_reason || "分人待确认");
  if (segment.voice_band_repaired) reasons.push("已按局部声线纠偏");
  if (segment.continuity_repaired) reasons.push("已按语义连续性纠偏");
  if (segment.speaker_handoff_voice_guard_repaired) reasons.push("已按声纹护栏纠偏");
  if (segment.speaker_voiceprint_reidentified) reasons.push(`已按声纹锚点回扫${segment.speaker_voiceprint_score ? `(${segment.speaker_voiceprint_score.toFixed(2)})` : ""}`);
  if (segment.speaker_voiceprint_review) reasons.push(`声纹锚点接近 ${segment.speaker_voiceprint_anchor || "目标说话人"}，待确认`);
  const overlapSummary = speakerOverlapSummary(segment);
  if (overlapSummary) reasons.push(overlapSummary);
  else if (segment.speaker_overlap_risk) reasons.push("可能有重叠/插话");
  return reasons.join(" / ");
}

function isSpeakerReviewSegment(segment?: Segment): boolean {
  return Boolean(segmentSpeakerReviewReason(segment));
}

function isUnresolvedSpeakerReviewSegment(segment?: SpeakerChipSegmentFlags): boolean {
  if (!segment) return false;
  if (
    segment.speaker_handoff_review
    || segment.speaker_handoff_text_review
    || segment.speaker_resegmentation_review
  ) return true;
  if (
    segment.speaker_assignment_review
    || segment.speaker_voiceprint_review
    || segment.speaker_overlap_risk
    || Boolean(segment.speaker_overlap_candidates?.length)
  ) {
    return true;
  }
  if (
    segment.voice_band_repaired
    || segment.continuity_repaired
    || segment.speaker_handoff_voice_guard_repaired
    || segment.speaker_voiceprint_reidentified
  ) return false;
  return Boolean(segmentSpeakerReviewReason(segment));
}

function combinedSpeakerFlags(segments: Segment[]): SpeakerChipSegmentFlags | undefined {
  if (!segments.some(isSpeakerReviewSegment)) return undefined;
  return {
    speaker_handoff_review: segments.some((s) => s.speaker_handoff_review),
    speaker_handoff_text_review: segments.some((s) => s.speaker_handoff_text_review),
    speaker_resegmentation_review: segments.some((s) => s.speaker_resegmentation_review),
    speaker_assignment_review: segments.some((s) => s.speaker_assignment_review),
    speaker_review_reason: segments.map((s) => s.speaker_review_reason).find(Boolean),
    voice_band_repaired: segments.some((s) => s.voice_band_repaired),
    continuity_repaired: segments.some((s) => s.continuity_repaired),
    speaker_handoff_voice_guard_repaired: segments.some((s) => s.speaker_handoff_voice_guard_repaired),
    speaker_overlap_risk: segments.some((s) => s.speaker_overlap_risk),
    speaker_overlap_candidates: segments.flatMap((s) => s.speaker_overlap_candidates ?? []),
    speaker_voiceprint_reidentified: segments.some((s) => s.speaker_voiceprint_reidentified),
    speaker_voiceprint_review: segments.some((s) => s.speaker_voiceprint_review),
    speaker_voiceprint_score: segments.map((s) => s.speaker_voiceprint_score).find(Boolean),
    speaker_voiceprint_anchor: segments.map((s) => s.speaker_voiceprint_anchor).find(Boolean),
  };
}

function diarizationHint(result: NonNullable<Task["result"]>, speakerCount: number): {
  text: string;
  className: string;
  title?: string;
} | null {
  const stats = result.diarization_stats;
  if (!stats) return null;
  const recommended = stats.recommended_n_speakers;
  const selected = stats.clusters || speakerCount;
  const risky = stats.risk_level === "high" || stats.over_split_risk;
  const uncertain = risky || stats.recommendation_confidence !== "high";
  const failedAutoApply = stats.applied === false;
  const parts: string[] = [];
  if (failedAutoApply) {
    parts.push("分人失败，原文已保留");
  } else if (recommended) {
    parts.push(`${uncertain ? "疑似" : "推荐"} ${recommended} 人`);
  }
  if (selected) parts.push(`当前 ${selected} 人`);
  if (risky) {
    parts.push("需人工确认");
  } else if (stats.risk_level === "medium") {
    parts.push("可能过度拆分");
  }
  if (stats.engine) parts.push(`引擎 ${stats.engine}`);
  if (stats.runtime_backend) parts.push(`后端 ${stats.runtime_backend}`);
  if (stats.fallback_reason) parts.push("已降级");
  if (!parts.length) return null;
  const className = stats.risk_level === "high" || stats.over_split_risk
    ? "text-warn border-warn/35 bg-warn/10"
    : stats.risk_level === "medium" || failedAutoApply
      ? "text-yellow-300 border-yellow-300/35 bg-yellow-500/10"
      : "text-fg-mute border-border bg-bg-panel";
  const titleParts = [
    stats.postprocess_skipped_reason || null,
    stats.risk_reason || null,
    stats.recommendation_confidence_reason || null,
    stats.recommended_score != null ? `推荐分数 ${stats.recommended_score.toFixed(3)}` : null,
    stats.selected_score != null ? `当前分数 ${stats.selected_score.toFixed(3)}` : null,
    stats.over_split_score_gap != null ? `差值 ${stats.over_split_score_gap.toFixed(3)}` : null,
    stats.fallback_reason ? `fallback: ${stats.fallback_reason}` : null,
  ].filter(Boolean);
  return { text: parts.join(" · "), className, title: titleParts.join(" / ") || undefined };
}

function speakerShortName(name: string): string {
  return name.replace("SPEAKER_", "");
}

function reviewSegmentLabel(s: { start: number; from_speaker: string; to_speaker: string }): string {
  const from = speakerShortName(s.from_speaker);
  const to = speakerShortName(s.to_speaker);
  const change = from && to && from !== to ? `${from}->${to}` : `${from || to}待确认`;
  return `${formatTimeShort(s.start)} ${change}`;
}

function formatSpeakerDistribution(distribution: Record<string, Record<string, number>>): string {
  return Object.entries(distribution)
    .map(([from, targets]) => {
      const detail = Object.entries(targets)
        .map(([to, count]) => `${speakerShortName(to)}${count}`)
        .join("/");
      return `${speakerShortName(from)}->${detail}`;
    })
    .join(" / ");
}

function persistenceStem(task: Task): string {
  return task.libraryStem ?? libraryStemFromFilename(task.filename);
}

function authoritativeTaskForStem(stem: string, preferredTaskId?: string | null): Task | undefined {
  const state = useTasks.getState();
  const key = libraryStemKey(stem);
  const matches = (candidate: Task) => libraryStemKey(persistenceStem(candidate)) === key;
  return state.tasks.find((candidate) => candidate.id === preferredTaskId && matches(candidate))
    ?? state.tasks.find((candidate) => candidate.id === state.activeId && matches(candidate))
    ?? state.tasks.find(matches);
}

function isTaskCurrentTarget(taskId: string): boolean {
  const state = useTasks.getState();
  return state.activeId === taskId
    && state.tasks.some((task) => task.id === taskId && task.stage !== "cancelled");
}

function assertTaskCurrentTarget(taskId: string): void {
  if (!isTaskCurrentTarget(taskId)) {
    throw new Error("当前任务已切换，本次异步结果已丢弃");
  }
}

function stringFingerprint(prefix: string, canonical: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < canonical.length; index += 1) {
    hash ^= canonical.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return `${prefix}:${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function segmentRevisionFingerprint(segments: Segment[]): string {
  return stringFingerprint(`segments-v2:${segments.length}`, JSON.stringify(segments));
}

function resultRevisionFingerprint(result: TranscribeResult): string {
  return stringFingerprint("result-v1", JSON.stringify({
    segments: result.segments,
    diarization_stats: result.diarization_stats,
  }));
}

function currentTaskResultFingerprint(taskId: string): string | null {
  const task = useTasks.getState().tasks.find((candidate) => candidate.id === taskId);
  return task?.result ? resultRevisionFingerprint(task.result) : null;
}

function assertTaskResultRevision(taskId: string, expected: string): void {
  assertTaskCurrentTarget(taskId);
  if (currentTaskResultFingerprint(taskId) !== expected) {
    throw new Error("当前任务正文已变化，本次异步结果已过期，请重新运行");
  }
}

function buildRawSavePayload(task: Task, result: TranscribeResult) {
  const stem = persistenceStem(task);
  return {
    stem,
    audio_filename: task.filename,
    source_audio: result.audio || task.audio,
    txt: buildTxt(result.segments, `${task.filename}\nbackend=${result.backend} duration=${result.duration.toFixed(1)}s segments=${result.segments.length}`),
    srt: buildSrt(result.segments),
    json: buildJson(result),
    result,
  };
}

async function saveRawResult(task: Task, result: TranscribeResult) {
  const payload = buildRawSavePayload(task, result);
  await enqueueStemWrite(payload.stem, () => ipc.librarySaveRaw(payload));
}

function correctedArtifactWithSegments(
  base: CorrectedArtifact,
  segments: Segment[],
): CorrectedArtifact {
  const changed = segments.filter((segment) =>
    segment.original_text != null && segment.original_text !== segment.text,
  ).length;
  return {
    ...base,
    segments,
    changed,
    total: segments.length,
  };
}

function humanCorrectedArtifact(task: Task): CorrectedArtifact {
  if (task.corrected) return task.corrected;
  const segments = (task.result?.segments ?? []).map((segment) => ({
    ...segment,
    original_text: segment.original_text ?? segment.text,
  }));
  return {
    segments,
    changed: 0,
    total: segments.length,
    model: "human-review",
  };
}

function buildResultTabsCorrectedPayload(stem: string, artifact: CorrectedArtifact): CorrectedSavePayload {
  const segments = artifact.segments;
  const diffLines: string[] = [`# diff: ${artifact.changed} changes / ${artifact.total} segments`, ""];
  for (const segment of segments) {
    if (segment.original_text != null && segment.original_text !== segment.text) {
      diffLines.push(
        `[${formatTimeShort(segment.start)}]\n  - ${segment.original_text}\n  + ${segment.text}\n`,
      );
    }
  }
  return {
    stem,
    txt: buildTxt(segments),
    srt: buildSrt(segments),
    json: JSON.stringify({
      stem,
      corrected_by: artifact.model,
      changed: artifact.changed,
      total: artifact.total,
      glossary: artifact.glossary,
      segments,
    }, null, 2),
    diff: diffLines.join("\n"),
    model: artifact.model,
    changed: artifact.changed,
    total: artifact.total,
    glossary: artifact.glossary,
  };
}
function saveCorrectedSegments(
  task: Task,
  segments: Segment[],
  corrected: CorrectedArtifact | undefined = task.corrected,
  expectedRevision: CorrectedRevision = captureTaskCorrectedRevision(task),
): Promise<CorrectedArtifact | undefined> {
  if (!corrected) return Promise.resolve(undefined);
  const artifact = correctedArtifactWithSegments(corrected, segments);

  return persistCorrectedArtifact({
    expectedRevision,
    artifact,
    buildPayload: buildResultTabsCorrectedPayload,
    commitToStore: true,
    requireActive: true,
  });
}
function mergeVoiceprintProfiles(existing: VoiceprintProfile[], incoming: VoiceprintProfile[]): VoiceprintProfile[] {
  function normalizedVector(vector: number[] | undefined): number[] | null {
    if (!vector?.length) return null;
    const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    if (!Number.isFinite(norm) || norm <= 1e-9) return null;
    return vector.map((value) => value / norm);
  }

  function profileVectors(profile: VoiceprintProfile): number[][] {
    const rows: number[][] = [];
    for (const vector of profile.embeddings ?? []) {
      const normalized = normalizedVector(vector);
      if (normalized) rows.push(normalized);
    }
    if (rows.length === 0) {
      const normalized = normalizedVector(profile.embedding);
      if (normalized) rows.push(normalized);
    }
    return rows;
  }

  function deduplicateVectors(vectors: number[][]): number[][] {
    const unique: number[][] = [];
    for (const vector of vectors) {
      if (unique.some((candidate) =>
        candidate.length === vector.length
        && candidate.reduce((sum, value, index) => sum + value * vector[index], 0) >= 0.999999,
      )) continue;
      unique.push(vector);
    }
    return unique;
  }

  function mergeVectors(vectors: number[][]): number[] {
    if (!vectors.length) return [];
    const dims = vectors[0].length;
    const acc = Array.from({ length: dims }, () => 0);
    let count = 0;
    for (const vector of vectors) {
      if (vector.length !== dims) continue;
      count += 1;
      vector.forEach((value, idx) => { acc[idx] += value; });
    }
    if (count === 0) return [];
    return normalizedVector(acc.map((value) => value / count)) ?? [];
  }

  const byName = new Map<string, VoiceprintProfile>();
  for (const profile of existing) {
    if (profile.name) {
      const vectors = deduplicateVectors(profileVectors(profile));
      byName.set(profile.name, {
        ...profile,
        embeddings: vectors,
        embedding: mergeVectors(vectors),
      });
    }
  }
  for (const profile of incoming) {
    const name = (profile.name || "").trim();
    if (!name || /^SPEAKER_.+$/i.test(name)) continue;
    const prev = byName.get(name);
    const previousVectors = prev ? deduplicateVectors(profileVectors(prev)) : [];
    const incomingVectors = deduplicateVectors(profileVectors(profile));
    const vectors = deduplicateVectors([...previousVectors, ...incomingVectors]);
    const addedVectors = Math.max(0, vectors.length - previousVectors.length);
    const additionRatio = incomingVectors.length > 0
      ? Math.min(1, addedVectors / incomingVectors.length)
      : 0;
    const previousSeconds = prev?.sample_seconds ?? 0;
    const addedSeconds = (profile.sample_seconds ?? 0) * additionRatio;
    const previousAnchors = prev?.anchor_count ?? 0;
    const addedAnchors = addedVectors > 0
      ? Math.max(1, Math.round((profile.anchor_count ?? 0) * additionRatio))
      : 0;
    byName.set(name, {
      ...(prev ?? {}),
      ...profile,
      name,
      embedding: mergeVectors(vectors),
      embeddings: vectors,
      anchor_count: previousAnchors + addedAnchors,
      sample_seconds: previousSeconds + addedSeconds,
      enrollment_source: "user_confirmed_anchors",
      created_at: prev?.created_at || profile.created_at || new Date().toISOString(),
    });
  }
  return Array.from(byName.values());
}

function replaceTermsInText(text: string, terms: string[], canonical: string): { text: string; count: number } {
  let next = text;
  let count = 0;
  for (const term of terms) {
    const cleanTerm = term.trim();
    if (!cleanTerm || cleanTerm === canonical) continue;
    const hits = next.split(cleanTerm).length - 1;
    if (hits <= 0) continue;
    next = next.split(cleanTerm).join(canonical);
    count += hits;
  }
  return { text: next, count };
}

function updateSegmentsByText(
  segments: Segment[],
  updater: (text: string, index: number) => { text: string; count: number },
): { segments: Segment[]; replacementCount: number; touchedCount: number } {
  let replacementCount = 0;
  let touchedCount = 0;
  const next = segments.map((segment, index) => {
    const result = updater(segment.text, index);
    if (result.text === segment.text) return segment;
    replacementCount += result.count;
    touchedCount += 1;
    return {
      ...segment,
      original_text: segment.original_text ?? segment.text,
      text: result.text,
    };
  });
  return { segments: next, replacementCount, touchedCount };
}

function withoutTermConsistencyCandidates(
  result: TranscribeResult,
  candidateIds: string[],
): TranscribeResult {
  if (!result.asr_quality?.term_consistency) return result;
  const remove = new Set(candidateIds);
  const candidates = (result.asr_quality.term_consistency.candidates ?? [])
    .filter((candidate) => !remove.has(candidate.id));
  return {
    ...result,
    asr_quality: {
      ...result.asr_quality,
      term_consistency: {
        ...result.asr_quality.term_consistency,
        candidate_count: candidates.length,
        candidates,
      },
    },
  };
}

type SpeakerDisplaySegment = Segment & { sourceSegmentIndex: number };

function speakerDisplayCues(segment: Segment): ReturnType<typeof segmentCues> {
  const baseCues = segmentCues(segment);
  const projected = (segment.speaker_cues ?? []).filter((cue) => (
    typeof cue.text === "string"
    && cue.text.length > 0
    && Number.isFinite(cue.start)
    && Number.isFinite(cue.end)
    && cue.end > cue.start
  ));
  const projectedText = projected.map((cue) => cue.text).join("");
  const monotonic = projected.every((cue, index) => (
    index === 0 || cue.start >= projected[index - 1].end
  ));
  if (projected.length >= 2 && projectedText === segment.text && monotonic) {
    return projected.map((cue) => {
      const timingCue = baseCues.find((item) => (
        Math.min(item.end, cue.end) - Math.max(item.start, cue.start)
      ) > 0.001);
      return {
        text: cue.text || "",
        start: cue.start,
        end: cue.end,
        source: "sync" as const,
        reliable: timingCue?.reliable ?? true,
        speaker: cue.speaker,
        speakerConfidence: cue.confidence,
        speakerReview: cue.review,
      };
    });
  }
  return baseCues;
}

function speakerDisplaySegments(segments: Segment[]): SpeakerDisplaySegment[] {
  const display: SpeakerDisplaySegment[] = [];
  segments.forEach((segment, sourceSegmentIndex) => {
    const cues = speakerDisplayCues(segment);
    const hasProjectedHandoff = cues.some((cue) => (
      cue.speaker
      && normalizeSpeakerName(cue.speaker) !== normalizeSpeakerName(segment.speaker || "")
    ));
    if (!hasProjectedHandoff) {
      display.push({ ...segment, sourceSegmentIndex });
      return;
    }

    const originalCues = segment.original_text != null
      ? segmentCues({ ...segment, text: segment.original_text })
      : [];
    cues.forEach((cue, cueIndex) => {
      const piece: SpeakerDisplaySegment = {
        ...segment,
        start: cue.start,
        end: cue.end,
        text: cue.text,
        speaker: cue.speaker || segment.speaker,
        sync_cues: [{ start: cue.start, end: cue.end, text: cue.text }],
        sourceSegmentIndex,
      };
      if (segment.original_text != null) {
        piece.original_text = originalCues[cueIndex]?.text ?? cue.text;
      }
      delete piece.speaker_cues;
      display.push(piece);
    });
  });
  return display;
}

/** 把连续 ≤ 1.2s 间隔的同一 speaker 段合并成 turn(对话视图用) */
function groupBySpeakerTurns<T extends Segment>(segments: T[]): Array<{
  speaker?: string;
  start: number;
  end: number;
  segments: T[];
}> {
  const turns: Array<{ speaker?: string; start: number; end: number; segments: T[] }> = [];
  for (const s of segments) {
    const speaker = displaySpeakerName(s.speaker);
    const last = turns[turns.length - 1];
    const sameSpeaker = last && last.speaker === speaker;
    const closeInTime = last && s.start - last.end < 1.2;
    const reviewBoundary = isUnresolvedSpeakerReviewSegment(s) || Boolean(last?.segments.some(isUnresolvedSpeakerReviewSegment));
    const blockTooLong = last && (s.end - last.start) > 20.0;
    if (last && sameSpeaker && closeInTime && !reviewBoundary && !blockTooLong) {
      last.end = s.end;
      last.segments.push(s);
    } else {
      turns.push({ speaker, start: s.start, end: s.end, segments: [s] });
    }
  }
  return turns;
}

type TranscriptCue = TranscriptSyncCue;

type Props = {
  task: Task;
  onCorrect: () => Promise<void> | void;
  onPolish: () => Promise<void> | void;
  onPipelineFull: () => Promise<void> | void;
  onOpenSettings: () => void;
  onArticleSaved?: () => void;
};

export default function ResultTabs({ task, onCorrect, onPolish, onPipelineFull, onOpenSettings, onArticleSaved }: Props) {
  const [tab, setTab] = useState<Tab>("raw");
  const hasCorrected = !!task.corrected;
  const hasPolished = !!task.polished;
  const hasTranslated = !!task.translated;
  const setResult = useTasks((s) => s.setResult);
  const setPolished = useTasks((s) => s.setPolished);
  const setTranslatedTop = useTasks((s) => s.setTranslated);
  const setTaskError = useTasks((s) => s.setError);

  // ---- 自动同步:文章/译文 里的 speaker 名按"首次出现顺序"对齐到原文/校对里的名 ----
  // 用途:用户在文章生成 *之后* 才在原文里改名;旧文本没被替换,这里上线时补救。
  useEffect(() => {
    const result = task.result;
    if (!result) return;
    // 当前(可能已改名)的原文 speaker,按首次出现顺序
    const rawSpeakers: string[] = [];
    for (const s of result.segments) {
      const speaker = displaySpeakerName(s.speaker);
      if (speaker && isValidSpeakerAnchorName(speaker) && !rawSpeakers.includes(speaker)) rawSpeakers.push(speaker);
    }
    if (rawSpeakers.length < 2) return; // 单人或没分人,文章是流文章,无需同步

    // 从文本里抽出对话头(支持 `**X:**` / `**X：**` / `**X**:` 等)
    const parseHeaders = (txt: string): string[] => {
      const out: string[] = [];
      // 行首 `**name:**` 或 `**name：**` 或 `**name**:`
      const re = /^\*\*([^*\n]+?)(?:\s*[:：]\s*\*\*|\*\*\s*[:：])/gm;
      let m: RegExpExecArray | null;
      while ((m = re.exec(txt))) {
        const name = m[1].trim();
        if (name && !out.includes(name)) out.push(name);
      }
      return out;
    };

    const escapeRegex = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const applyMapping = (
      txt: string,
      mapping: Record<string, string>,
    ): string => {
      let out = txt;
      for (const [oldN, newN] of Object.entries(mapping)) {
        const O = escapeRegex(oldN);
        out = out.replace(new RegExp(`\\*\\*${O}\\s*[:\\uFF1A]\\s*\\*\\*`, "g"), `**${newN}:**`);
        out = out.replace(new RegExp(`^\\*\\*${O}\\*\\*\\s*[:\\uFF1A]`, "gm"), `**${newN}:**`);
      }
      return out;
    };

    // 同步文章
    if (task.polished) {
      const articleSpeakers = parseHeaders(task.polished.text);
      if (articleSpeakers.length >= 2) {
        const mapping: Record<string, string> = {};
        for (let i = 0; i < articleSpeakers.length && i < rawSpeakers.length; i++) {
          if (articleSpeakers[i] !== rawSpeakers[i]) {
            mapping[articleSpeakers[i]] = rawSpeakers[i];
          }
        }
        if (Object.keys(mapping).length) {
          const newText = applyMapping(task.polished.text, mapping);
          if (newText !== task.polished.text) {
            console.log("[auto-sync] article speakers updated:", mapping);
            setPolished(task.id, { ...task.polished, text: newText });
            const stem = persistenceStem(task);
            ipc.librarySavePolished({
              stem,
              text: newText,
              model: task.polished.model,
              source: task.polished.source,
            }).catch((error) => {
              setTaskError(task.id, `文章内容已更新，但保存失败: ${String(error)}`);
            });
          }
        }
      }
    }

    // 同步译文
    if (task.translated) {
      const tSpeakers = parseHeaders(task.translated.text);
      if (tSpeakers.length >= 2) {
        const mapping: Record<string, string> = {};
        for (let i = 0; i < tSpeakers.length && i < rawSpeakers.length; i++) {
          if (tSpeakers[i] !== rawSpeakers[i]) {
            mapping[tSpeakers[i]] = rawSpeakers[i];
          }
        }
        if (Object.keys(mapping).length) {
          const newText = applyMapping(task.translated.text, mapping);
          if (newText !== task.translated.text) {
            console.log("[auto-sync] translation speakers updated:", mapping);
            setTranslatedTop(task.id, { ...task.translated, text: newText });
          }
        }
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task.result, task.polished?.text, task.translated?.text]);

  // Auto-jump to the most informative tab when new data arrives.
  useEffect(() => {
    if (hasTranslated) setTab("translated");
    else if (hasPolished) setTab("article");
    else if (hasCorrected) setTab("corrected");
  }, [hasTranslated, hasPolished, hasCorrected]);

  const result = task.result;

  // 有 speaker → 默认对话视图;没有 → 时间戳列表
  const hasSpeakers = (result?.segments ?? []).some((s) => !!s.speaker);
  const [viewMode, setViewMode] = useState<ViewMode>("timeline");
  useEffect(() => {
    setViewMode(hasSpeakers ? "dialog" : "timeline");
  }, [hasSpeakers]);

  /** 全局重命名说话人 — 所有 raw/corrected segments 中 speaker===oldName 的都替换为 newName,
      并把改动写回 raw/corrected JSON 持久化。 */
  const renameSpeaker = async (oldName: string, newName: string) => {
    if (!result) return;
    const normalizedOldName = normalizeSpeakerName(oldName);
    const normalizedNewName = normalizeSpeakerName(newName);
    if (!normalizedOldName || !isValidSpeakerAnchorName(normalizedNewName)) return;
    const renameSeg = (s: Segment): Segment => {
      const speaker = normalizeSpeakerName(s.speaker || "") === normalizedOldName
        ? normalizedNewName
        : s.speaker;
      const speakerCues = s.speaker_cues?.map((cue) => (
        normalizeSpeakerName(cue.speaker) === normalizedOldName
          ? { ...cue, speaker: normalizedNewName }
          : cue
      ));
      const overlapCandidates = s.speaker_overlap_candidates?.map((candidate) => ({
        ...candidate,
        primary_speaker: normalizeSpeakerName(candidate.primary_speaker) === normalizedOldName
          ? normalizedNewName
          : candidate.primary_speaker,
        secondary_speaker: normalizeSpeakerName(candidate.secondary_speaker) === normalizedOldName
          ? normalizedNewName
          : candidate.secondary_speaker,
      }));
      return speaker === s.speaker
        && speakerCues === s.speaker_cues
        && overlapCandidates === s.speaker_overlap_candidates
        ? s
        : {
            ...s,
            speaker,
            speaker_cues: speakerCues,
            speaker_overlap_candidates: overlapCandidates,
          };
    };

    // 1. raw segments
    const newSegs = result.segments.map(renameSeg);
    const nextResult = { ...result, segments: newSegs };
    const correctedBefore = task.corrected;
    setResult(task.id, nextResult);
    const correctedRevision = captureTaskCorrectedRevision(task);
    const nextCorrected = correctedBefore
      ? correctedArtifactWithSegments(
          correctedBefore,
          correctedBefore.segments.map(renameSeg),
        )
      : undefined;

    // 2. 持久化到 raw JSON(transcripts/<stem>/<stem>.json)
    const stem = persistenceStem(task);
    try {
      await saveRawResult(task, nextResult);
    } catch (e) {
      setTaskError(task.id, `说话人名称已更新，但原始结果保存失败: ${String(e)}`);
    }

    // 3. 持久化到 corrected JSON(若有)
    if (nextCorrected) {
      try {
        await saveCorrectedSegments(
          task,
          nextCorrected.segments,
          nextCorrected,
          correctedRevision,
        );
      } catch (e) {
        setTaskError(task.id, `说话人名称已更新，但校对结果保存失败: ${String(e)}`);
      }
    }

    // 4. 同步到 polished 文章 / 译文(对话体里的对话头改名)
    //    兼容 LLM 输出的多种变体格式:
    //      `**陈总:**`(标准格式,中冒号)
    //      `**陈总:**`(英冒号)
    //      `**陈总：**`(中文全角冒号)
    //      `**陈总**:` / `**陈总**：`(冒号在 ** 外面)
    //    用严格的"行首 + ** 包裹 + 冒号" 模式,不会误改正文中的偶然重名
    const escapeRegex = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const O = escapeRegex(oldName);
    const renameInArticle = (txt: string): string => {
      let out = txt;
      // a) `**OLD:**` / `**OLD：**`(冒号在 ** 内)
      out = out.replace(new RegExp(`\\*\\*${O}\\s*[:\\uFF1A]\\s*\\*\\*`, "g"), `**${normalizedNewName}:**`);
      // b) `**OLD**:` / `**OLD**：`(冒号在 ** 外)— 只匹配行首,避免误伤正文
      out = out.replace(new RegExp(`^\\*\\*${O}\\*\\*\\s*[:\\uFF1A]`, "gm"), `**${normalizedNewName}:**`);
      return out;
    };

    if (task.polished) {
      const newText = renameInArticle(task.polished.text);
      if (newText !== task.polished.text) {
        setPolished(task.id, { ...task.polished, text: newText });
        try {
          await ipc.librarySavePolished({
            stem,
            text: newText,
            model: task.polished.model,
            source: task.polished.source,
          });
        } catch (e) {
          setTaskError(task.id, `说话人名称已更新，但文章保存失败: ${String(e)}`);
        }
      }
    }
    if (task.translated) {
      const newText = renameInArticle(task.translated.text);
      if (newText !== task.translated.text) {
        setTranslatedTop(task.id, { ...task.translated, text: newText });
      }
    }
  };

  const persistEntitySegments = async (
    nextResult: TranscribeResult,
    nextCorrectedSegments?: Segment[],
    previousCorrected: CorrectedRevision = captureTaskCorrectedRevision(task),
  ) => {
    const nextCorrected = previousCorrected.artifact && nextCorrectedSegments
      ? correctedArtifactWithSegments(previousCorrected.artifact, nextCorrectedSegments)
      : undefined;
    setResult(task.id, nextResult);
    const correctedRevisionAfterRaw = captureTaskCorrectedRevision(task);
    const failures: string[] = [];
    try {
      await saveRawResult(task, nextResult);
    } catch (e) {
      failures.push(`原始结果: ${String(e)}`);
    }
    if (nextCorrected) {
      try {
        await saveCorrectedSegments(
          task,
          nextCorrected.segments,
          nextCorrected,
          correctedRevisionAfterRaw,
        );
      } catch (e) {
        failures.push(`校对结果: ${String(e)}`);
      }
    }
    if (failures.length) {
      const message = `实体修改已保留，但保存失败: ${failures.join("；")}`;
      setTaskError(task.id, message);
      throw new Error(message);
    }
  };

  const applyEntityUnify = async (candidate: TermConsistencyCandidate, canonical: string) => {
    if (!result) return { replacementCount: 0, touchedCount: 0 };
    const cleanCanonical = canonical.trim();
    if (!cleanCanonical) throw new Error("请先填写标准写法");
    const terms = (candidate.terms ?? []).map((term) => term.trim()).filter(Boolean);
    const rawUpdate = updateSegmentsByText(result.segments, (text) =>
      replaceTermsInText(text, terms, cleanCanonical),
    );
    if (rawUpdate.replacementCount <= 0) return rawUpdate;
    const previousCorrected = captureTaskCorrectedRevision(task);
    const correctedUpdate = previousCorrected.artifact
      ? updateSegmentsByText(previousCorrected.artifact.segments, (text) => replaceTermsInText(text, terms, cleanCanonical))
      : undefined;
    await persistEntitySegments(
      withoutTermConsistencyCandidates({ ...result, segments: rawUpdate.segments }, [candidate.id]),
      correctedUpdate?.segments,
      previousCorrected,
    );
    return rawUpdate;
  };

  const applyEntityOccurrence = async (
    candidate: TermConsistencyCandidate,
    contextIndex: number,
    fromText: string,
    toText: string,
  ) => {
    if (!result) return { replacementCount: 0, touchedCount: 0 };
    const from = fromText.trim();
    const to = toText.trim();
    if (!from || !to) throw new Error("请填写要替换的原词和目标写法");
    const rawUpdate = updateSegmentsByText(result.segments, (text, index) => {
      if (index !== contextIndex) return { text, count: 0 };
      return replaceTermsInText(text, [from], to);
    });
    if (rawUpdate.replacementCount <= 0) return rawUpdate;
    const previousCorrected = captureTaskCorrectedRevision(task);
    const correctedUpdate = previousCorrected.artifact
      ? updateSegmentsByText(previousCorrected.artifact.segments, (text, index) => {
          if (index !== contextIndex) return { text, count: 0 };
          return replaceTermsInText(text, [from], to);
        })
      : undefined;
    await persistEntitySegments(
      withoutTermConsistencyCandidates({ ...result, segments: rawUpdate.segments }, [candidate.id]),
      correctedUpdate?.segments,
      previousCorrected,
    );
    return rawUpdate;
  };

  if (!result) {
    return <div className="text-sm text-text-mute">（转录尚未完成）</div>;
  }

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      {/* VSCode-style editor tab bar */}
      <header className="shrink-0 flex items-center justify-between bg-tabbar border-b border-border">
        <div className="flex items-center">
          {(["raw", "corrected", "article", "translated"] as Tab[]).map((k) => {
            const isActive = tab === k;
            const isReady =
              k === "raw" ||
              (k === "corrected" && hasCorrected) ||
              (k === "article" && hasPolished) ||
              (k === "translated" && hasTranslated);

            // 判断当前 tab 是否正在处理中
            const isProcessing =
              (k === "raw" && (task.stage === "transcribing" || task.stage === "diarizing")) ||
              (k === "corrected" && (task.stage === "correcting" || task.stage === "correcting_paused")) ||
              (k === "article" && task.stage === "polishing") ||
              (k === "translated" && task.stage === "translating");

            return (
              <button
                key={k}
                onClick={() => setTab(k)}
                className={clsx(
                  "btn-tab",
                  isActive && "btn-tab-active",
                  isActive && "bg-editor",
                  isProcessing && !isActive && "animate-pulse text-accent",
                )}
              >
                {TAB_META[k].icon}
                <span>{TAB_META[k].label}</span>
                {isReady && k !== "raw" && <Check size={10} className="text-ok" />}
                {isProcessing && <Hourglass size={10} className="text-accent animate-spin" />}
              </button>
            );
          })}
        </div>
        <div className="px-3 flex items-center gap-3 text-ui-sm text-fg-mute">
          {hasSpeakers && (
            <div className="inline-flex border border-border rounded-sm overflow-hidden">
              <button
                onClick={() => setViewMode("dialog")}
                className={clsx(
                  "px-2 py-0.5 text-xs",
                  viewMode === "dialog"
                    ? "bg-accent/20 text-accent"
                    : "text-fg-mute hover:text-fg",
                )}
                title="按说话人合并为对话气泡"
              >
                对话
              </button>
              <button
                onClick={() => setViewMode("timeline")}
                className={clsx(
                  "px-2 py-0.5 text-xs border-l border-border",
                  viewMode === "timeline"
                    ? "bg-accent/20 text-accent"
                    : "text-fg-mute hover:text-fg",
                )}
                title="逐段带时间戳"
              >
                时间戳
              </button>
            </div>
          )}
          {result.language && (
            <div className="inline-flex border border-border rounded-sm overflow-hidden">
              <span className="px-2 py-0.5 text-xs bg-accent/10 text-accent" title="识别语言">
                {result.language === "zh" && "中文"}
                {result.language === "en" && "English"}
                {result.language === "ja" && "日本語"}
                {result.language === "ko" && "한국어"}
                {!["zh", "en", "ja", "ko"].includes(result.language) && result.language.toUpperCase()}
              </span>
            </div>
          )}
          <span>
            {result.backend} · {fmtDuration(result.duration)} · {result.segments.length} 段
          </span>
        </div>
      </header>

      <div className="flex-1 min-h-0 overflow-auto px-6 py-4 bg-editor">
        {tab === "raw" && (
          <RawTabContent
            task={task}
            viewMode={viewMode}
            onRenameSpeaker={renameSpeaker}
            onApplyEntityUnify={applyEntityUnify}
            onApplyEntityOccurrence={applyEntityOccurrence}
          />
        )}
        {tab === "corrected" &&
          (hasCorrected ? (
            <CorrectedSegments
              segments={task.corrected!.segments}
              changed={task.corrected!.changed}
              total={task.corrected!.total}
              model={task.corrected!.model}
              viewMode={viewMode}
              onRenameSpeaker={renameSpeaker}
            />
          ) : task.stage === "cancelled" ? (
            <div className="py-12 px-4 text-center text-ui text-fg-mute">
              任务已取消。如需重新校对，请重新导入或恢复该任务。
            </div>
          ) : (
            <CorrectionCTA
              busy={task.stage === "correcting"}
              onCorrect={onCorrect}
              onPipelineFull={onPipelineFull}
              onOpenSettings={onOpenSettings}
            />
          ))}
        {tab === "article" &&
          (hasPolished ? (
            <ArticleView
              text={task.polished!.text}
              model={task.polished!.model}
              source={task.polished!.source}
              truncated={task.polished!.truncated}
              inputChars={task.polished!.input_chars}
              segments={task.polished!.source === "corrected" && task.corrected
                ? task.corrected.segments
                : task.result!.segments}
              audio={task.result!.audio || task.audio}
            />
          ) : (
            <PolishCTA
              busy={task.stage === "polishing"}
              hasCorrected={hasCorrected}
              onPolish={onPolish}
              onOpenSettings={onOpenSettings}
            />
          ))}
        {tab === "translated" &&
          (hasTranslated ? (
            <TranslatedView
              text={task.translated!.text}
              sourceLanguage={task.translated!.source_language}
              targetLanguage={task.translated!.target_language}
              model={task.translated!.model}
              truncated={task.translated!.truncated}
              inputChars={task.translated!.input_chars}
            />
          ) : (
            <div className="py-12 px-4 text-center flex flex-col items-center gap-3">
              <Globe size={28} className="text-fg-mute" />
              <div className="text-ui text-fg-dim max-w-md leading-relaxed">
                翻译功能可将文章翻译成其他语言。请先完成文章排版，然后点击底部的"翻译"按钮。
              </div>
            </div>
          ))}
      </div>

      <ExportBar
        task={task}
        tab={tab}
        onCorrect={onCorrect}
        onPolish={onPolish}
        onArticleSaved={onArticleSaved}
      />
    </div>
  );
}

// ============================================================================
// 内容渲染
// ============================================================================

function SpeakerChip({
  speakers,
  who,
  segment,
  onRename,
}: {
  speakers: string[];
  who?: string;
  segment?: SpeakerChipSegmentFlags;
  onRename?: (oldName: string, newName: string) => void;
}) {
  // window.prompt() 在 Tauri WKWebView 里被禁用 → 改用 inline 输入框
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(who ?? "");
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  if (!speakers.length) return null;
  const clickable = !!(who && onRename);
  const reviewReason = segmentSpeakerReviewReason(segment);
  const overlapSummary = speakerOverlapSummary(segment);
  const reviewed = Boolean(reviewReason);
  const fixed = Boolean(
    reviewed
    && !isUnresolvedSpeakerReviewSegment(segment)
    && (
      segment?.voice_band_repaired
      || segment?.continuity_repaired
      || segment?.speaker_handoff_voice_guard_repaired
      || segment?.speaker_voiceprint_reidentified
    ),
  );

  const commit = () => {
    const next = normalizeSpeakerName(draft);
    if (next && !isValidSpeakerAnchorName(next)) {
      setDraft(who ?? "");
      setEditing(false);
      return;
    }
    if (next && next !== who && onRename && who) {
      onRename(who, next);
    }
    setEditing(false);
  };
  const cancel = () => {
    setDraft(who ?? "");
    setEditing(false);
  };

  if (editing) {
    return (
      <input
        ref={inputRef}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit();
          else if (e.key === "Escape") cancel();
        }}
        onBlur={commit}
        onClick={(e) => e.stopPropagation()}
        className={clsx(
          "shrink-0 px-1.5 py-0.5 rounded-sm border text-xs font-medium whitespace-nowrap",
          "bg-bg outline-none",
          who ? speakerChipClass(speakers, who) : "text-fg-mute border-border",
          reviewed && "border-warn/70 bg-warn/10 text-warn",
        )}
        style={{ width: `${Math.max(14, draft.length + 2)}ch` }}
      />
    );
  }

  return (
    <span
      onClick={
        clickable
          ? (e) => {
              e.stopPropagation();
              setDraft(who ?? "");
              setEditing(true);
            }
          : undefined
      }
      title={[
        clickable ? "点击改名(全局生效:所有标着此说话人的段都会一起换)" : "",
        reviewReason,
      ].filter(Boolean).join("；") || undefined}
      className={clsx(
        "shrink-0 px-1.5 py-0.5 rounded-sm border text-xs font-medium whitespace-nowrap select-none inline-flex items-center gap-1",
        who ? speakerChipClass(speakers, who) : "text-fg-mute border-border",
        reviewed && "border-warn/70 bg-warn/10 text-warn",
        clickable && "cursor-pointer hover:brightness-125",
      )}
    >
      <span>{who ?? "?"}</span>
      {reviewed && (
        <span className="text-[10px] font-normal">
          {overlapSummary || (fixed ? "已纠偏" : "待确认")}
        </span>
      )}
    </span>
  );
}

type TaskBoundDiarizationRecommendation = {
  taskId: string;
  sourceFingerprint: string;
  response: RecommendDiarizationResponse;
};

type VoiceprintUndoSnapshot = {
  taskId: string;
  before: TranscribeResult;
  correctedBefore?: CorrectedArtifact;
  appliedFingerprint: string;
  correctedAppliedFingerprint?: string;
};

function RawTabContent({ task, viewMode, onRenameSpeaker, onApplyEntityUnify, onApplyEntityOccurrence }: {
  task: Task;
  viewMode: ViewMode;
  onRenameSpeaker?: (oldName: string, newName: string) => void;
  onApplyEntityUnify?: (candidate: TermConsistencyCandidate, canonical: string) => Promise<{ replacementCount: number; touchedCount: number }>;
  onApplyEntityOccurrence?: (
    candidate: TermConsistencyCandidate,
    contextIndex: number,
    fromText: string,
    toText: string,
  ) => Promise<{ replacementCount: number; touchedCount: number }>;
}) {
  const segments = task.result!.segments;
  const speakers = collectSpeakers(segments);
  const hint = diarizationHint(task.result!, speakers.length);
  const diarizationAnalyzed = Boolean(task.result!.diarization_stats);
  const settings = useSettings((s) => s.settings);
  const patchDiarization = useSettings((s) => s.patchDiarization);
  const setDiarizationArtifacts = useTasks((s) => s.setDiarizationArtifacts);

  const [busy, setBusy] = useState(false);
  const [recommending, setRecommending] = useState(false);
  const [applying, setApplying] = useState<number | null>(null);
  const [recommendation, setRecommendation] = useState<TaskBoundDiarizationRecommendation | null>(null);
  const [voiceprintSnapshot, setVoiceprintSnapshot] = useState<VoiceprintUndoSnapshot | null>(null);
  const [voiceprintComparison, setVoiceprintComparison] = useState<VoiceprintComparison | null>(null);
  const autoSpeakerCount = (settings.diarization?.n_speakers ?? 0) <= 0;
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const coveragePlaybackEndRef = useRef<number | null>(null);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [coveragePlaybackId, setCoveragePlaybackId] = useState<string | null>(null);
  const [audioPlaying, setAudioPlaying] = useState(false);
  const [autoFollowTranscript, setAutoFollowTranscript] = useState(true);
  const audioSource = task.result?.audio || task.audio;
  const coverageReport = task.result!.filter_stats?.speech_coverage;
  const strongAsrReview = task.result!.filter_stats?.strong_asr;
  const timingReliable = task.result!.filter_stats?.timing_reliable !== false;
  const timingFailureReason = task.result!.filter_stats?.timing_alignment_reason
    || task.result!.filter_stats?.timing_reason
    || "时间锚点不足";
  const transcriptSyncTime = playbackTime;
  const activeSegmentIndex = useMemo(
    () => (timingReliable ? activeSegmentIndexAt(segments, transcriptSyncTime) : null),
    [segments, timingReliable, transcriptSyncTime],
  );
  const activeSegment = activeSegmentIndex == null ? null : segments[activeSegmentIndex] ?? null;
  const activeCue = useMemo(
    () => activeCueAt(activeSegment, transcriptSyncTime),
    [activeSegment, transcriptSyncTime],
  );

  useEffect(() => {
    audioRef.current?.pause();
    coveragePlaybackEndRef.current = null;
    setCoveragePlaybackId(null);
    setAudioPlaying(false);
    setPlaybackTime(0);
    setRecommendation(null);
    setVoiceprintSnapshot(null);
    setVoiceprintComparison(null);
    setError(null);
  }, [task.id, audioSource]);

  function finishCoveragePlayback() {
    coveragePlaybackEndRef.current = null;
    setCoveragePlaybackId(null);
    setAudioPlaying(false);
  }

  function seekToSegment(segment: Segment, play = true) {
    const audio = audioRef.current;
    if (!audio) return;
    coveragePlaybackEndRef.current = null;
    setCoveragePlaybackId(null);
    setAudioPlaying(false);
    const targetTime = Math.max(0, segment.start + 0.02);
    audio.currentTime = targetTime;
    setPlaybackTime(targetTime);
    setAutoFollowTranscript(true);
    if (play) {
      audio.play().catch(() => {
        setAudioPlaying(false);
        /* 用户手势限制时只定位,不强制播放 */
      });
    }
  }

  function handlePlaybackTimeChange(time: number) {
    const stopAt = coveragePlaybackEndRef.current;
    if (stopAt != null && time >= stopAt - 0.02) {
      const audio = audioRef.current;
      coveragePlaybackEndRef.current = null;
      setCoveragePlaybackId(null);
      if (audio) {
        audio.pause();
        if (Math.abs(audio.currentTime - stopAt) > 0.03) audio.currentTime = stopAt;
      }
      setPlaybackTime(stopAt);
      setAudioPlaying(false);
      return;
    }
    setPlaybackTime(time);
  }

  async function playCoverageRange(id: string, start: number, end: number) {
    const audio = audioRef.current;
    if (!audio) throw new Error("音频播放器尚未准备好");
    const clipStart = Math.max(0, start - 2);
    const requestedEnd = Math.max(clipStart + 0.05, end + 2);
    const clipEnd = Number.isFinite(audio.duration) && audio.duration > 0
      ? Math.min(audio.duration, requestedEnd)
      : requestedEnd;
    coveragePlaybackEndRef.current = clipEnd;
    setCoveragePlaybackId(id);
    setAutoFollowTranscript(true);
    audio.currentTime = clipStart;
    setPlaybackTime(clipStart);
    try {
      await audio.play();
    } catch (e) {
      coveragePlaybackEndRef.current = null;
      setCoveragePlaybackId(null);
      throw e;
    }
  }

  async function restoreLatestCommittedArtifacts(
    stem: string,
    corrected?: CorrectedArtifact,
  ): Promise<void> {
    while (true) {
      const rawRevision = committedRawRevisionForStem(stem);
      const latestTask = authoritativeTaskForStem(rawRevision.stem, rawRevision.taskId);
      if (!latestTask?.result) {
        throw new Error("当前 stem 没有可恢复的 committed raw");
      }
      const effectiveCorrected = corrected ?? latestTask.corrected;
      await ipc.librarySaveRawAndCorrected({
        raw: buildRawSavePayload({ ...latestTask, libraryStem: rawRevision.stem }, latestTask.result),
        corrected: effectiveCorrected
          ? buildResultTabsCorrectedPayload(rawRevision.stem, effectiveCorrected)
          : undefined,
        clear_corrected: !effectiveCorrected,
      });
      const after = committedRawRevisionForStem(stem);
      if (
        after.version === rawRevision.version
        && after.taskId === rawRevision.taskId
        && after.stem === rawRevision.stem
        && after.fingerprint === rawRevision.fingerprint
      ) {
        return;
      }
    }
  }

  async function persistSpeakerArtifacts({
    baseTask,
    requestTaskId,
    sourceFingerprint,
    nextResult,
    correctedRevision,
    nextCorrected,
  }: {
    baseTask: Task;
    requestTaskId: string;
    sourceFingerprint: string;
    nextResult: TranscribeResult;
    correctedRevision: CorrectedRevision;
    nextCorrected?: CorrectedArtifact;
  }): Promise<void> {
    const resultIsCurrent = () =>
      isTaskCurrentTarget(requestTaskId)
      && currentTaskResultFingerprint(requestTaskId) === sourceFingerprint;
    assertTaskResultRevision(requestTaskId, sourceFingerprint);
    assertTaskCurrentTarget(requestTaskId);

    if (nextCorrected) {
      await persistCorrectedTransaction({
        expectedRevision: correctedRevision,
        artifact: nextCorrected,
        requireActive: true,
        isCurrent: resultIsCurrent,
        persist: () => ipc.librarySaveRawAndCorrected({
          raw: buildRawSavePayload(baseTask, nextResult),
          corrected: buildResultTabsCorrectedPayload(persistenceStem(baseTask), nextCorrected),
        }).then(() => undefined),
        restore: ({ stem, artifact: latestCorrected }) =>
          restoreLatestCommittedArtifacts(stem, latestCorrected),
        commit: (artifact) => {
          setDiarizationArtifacts(requestTaskId, nextResult, artifact);
        },
      });
      return;
    }

    const stem = persistenceStem(baseTask);
    committedRawRevisionForStem(stem);
    await enqueueStemWrite(stem, async () => {
      if (!resultIsCurrent()) {
        throw new Error("speaker 结果在排队期间已过期，未写入磁盘");
      }
      await ipc.librarySaveRawAndCorrected({
        raw: buildRawSavePayload(baseTask, nextResult),
      });
      if (!resultIsCurrent()) {
        await restoreLatestCommittedArtifacts(stem);
        throw new Error("speaker 保存期间正文已变化；已恢复最新磁盘版本，本次旧结果未应用");
      }
      setDiarizationArtifacts(requestTaskId, nextResult);
    });
  }

  async function applyDiarization(
    segmentsWithSpeakers: Segment[],
    stats: TranscribeResult["diarization_stats"],
    sourceFingerprint: string,
  ): Promise<string> {
    const requestTaskId = task.id;
    assertTaskResultRevision(requestTaskId, sourceFingerprint);
    if (stats?.applied === false) {
      throw new Error(stats.failure_reason || "说话人分离没有可应用结果");
    }
    if (!segmentsWithSpeakers.every((segment) => Boolean(segment.speaker))) {
      throw new Error("说话人分离没有返回有效 speaker 标签");
    }
    assertDiarizationPreservesTranscript(segments, segmentsWithSpeakers);

    const nextResult = {
      ...task.result!,
      segments: segmentsWithSpeakers,
      diarization_stats: stats,
    };
    const previousCorrected = captureTaskCorrectedRevision(task);
    const nextCorrected = previousCorrected.artifact
      ? correctedArtifactWithSegments(
          previousCorrected.artifact,
          syncSpeakerMetadata(
            segmentsWithSpeakers,
            previousCorrected.artifact.segments,
            { requireFullMatch: true },
          ),
        )
      : undefined;
    const appliedFingerprint = resultRevisionFingerprint(nextResult);
    try {
      await persistSpeakerArtifacts({
        baseTask: task,
        requestTaskId,
        sourceFingerprint,
        nextResult,
        correctedRevision: previousCorrected,
        nextCorrected,
      });
    } catch (error) {
      throw new Error(`分人结果保存失败，已保留原结果: ${String(error)}`);
    }
    return appliedFingerprint;
  }

  async function rerun() {
    const requestTaskId = task.id;
    const sourceFingerprint = resultRevisionFingerprint(task.result!);
    setError(null);
    setBusy(true);
    try {
      const audio = task.result?.audio || task.audio;
      const dr = await ipc.diarize({
        audio,
        segments,
        n_speakers: settings.diarization?.n_speakers ?? 0,
        engine: settings.diarization?.engine || "auto",
        profiles: settings.diarization?.speakers ?? [],
      });
      assertTaskResultRevision(requestTaskId, sourceFingerprint);
      await applyDiarization(dr.segments, dr.stats, sourceFingerprint);
    } catch (e) {
      if (isTaskCurrentTarget(requestTaskId)) setError(String(e));
    } finally {
      if (isTaskCurrentTarget(requestTaskId)) setBusy(false);
    }
  }

  async function recommend() {
    const requestTaskId = task.id;
    const sourceFingerprint = resultRevisionFingerprint(task.result!);
    setError(null);
    setRecommending(true);
    try {
      const audio = task.result?.audio || task.audio;
      const rec = await ipc.recommendDiarization({
        audio,
        segments,
        min_speakers: 2,
        max_speakers: 8,
        engine: settings.diarization?.engine || "auto",
        profiles: settings.diarization?.speakers ?? [],
      });
      assertTaskResultRevision(requestTaskId, sourceFingerprint);
      if (!rec.candidates.length) {
        const details = rec.errors?.map((item) => item.error).filter(Boolean).join("；");
        throw new Error(rec.reason || details || "没有可用的说话人候选");
      }
      setRecommendation({ taskId: requestTaskId, sourceFingerprint, response: rec });
    } catch (e) {
      if (isTaskCurrentTarget(requestTaskId)) setError(String(e));
    } finally {
      if (isTaskCurrentTarget(requestTaskId)) setRecommending(false);
    }
  }

  async function applyCandidate(candidate: DiarizationCandidate, persistCount = false) {
    const requestTaskId = task.id;
    const actualN = candidate.actual_n_speakers ?? candidate.speakers.length;
    const sourceFingerprint = recommendation?.taskId === requestTaskId
      ? recommendation.sourceFingerprint
      : "";
    setError(null);
    setApplying(candidate.n_speakers);
    try {
      if (!sourceFingerprint) throw new Error("推荐结果不属于当前任务，请重新分析");
      assertTaskResultRevision(requestTaskId, sourceFingerprint);
      const appliedFingerprint = await applyDiarization(
        candidate.segments,
        candidate.stats,
        sourceFingerprint,
      );
      if (persistCount) {
        assertTaskResultRevision(requestTaskId, appliedFingerprint);
        await patchDiarization({
          enabled: true,
          n_speakers: actualN,
        });
      }
    } catch (e) {
      if (isTaskCurrentTarget(requestTaskId)) setError(String(e));
    } finally {
      if (isTaskCurrentTarget(requestTaskId)) setApplying(null);
    }
  }

  async function applyVoiceprintReidentify(
    response: VoiceprintReidentifyResponse,
    saveProfiles: boolean,
    sourceFingerprint: string,
  ) {
    const requestTaskId = task.id;
    assertTaskResultRevision(requestTaskId, sourceFingerprint);
    if (!response.segments.every((segment) => Boolean(segment.speaker))) {
      throw new Error("声纹回扫没有为每个 transcript segment 提供 speaker");
    }
    assertDiarizationPreservesTranscript(segments, response.segments);
    const snapshot = JSON.parse(JSON.stringify(task.result!)) as TranscribeResult;
    const previousCorrected = captureTaskCorrectedRevision(task);
    const correctedBefore = previousCorrected.artifact
      ? JSON.parse(JSON.stringify(previousCorrected.artifact)) as CorrectedArtifact
      : undefined;
    const nextCorrected = previousCorrected.artifact
      ? correctedArtifactWithSegments(
          previousCorrected.artifact,
          syncSpeakerMetadata(
            response.segments,
            previousCorrected.artifact.segments,
            { requireFullMatch: true },
          ),
        )
      : undefined;
    const currentStats = task.result!.diarization_stats ?? {
      embeddings: 0,
      duration_s: task.result!.duration,
      clusters: speakers.length,
      matched_profile_count: 0,
      segment_count: response.segments.length,
    };
    const nextStats = {
      ...currentStats,
      voiceprint_reidentify: response.stats,
      risk_reason: response.stats.reason || currentStats.risk_reason,
    };
    const nextResult = { ...task.result!, segments: response.segments, diarization_stats: nextStats };
    const appliedFingerprint = resultRevisionFingerprint(nextResult);
    await persistSpeakerArtifacts({
      baseTask: task,
      requestTaskId,
      sourceFingerprint,
      nextResult,
      correctedRevision: previousCorrected,
      nextCorrected,
    });
    setVoiceprintSnapshot({
      taskId: requestTaskId,
      before: snapshot,
      correctedBefore,
      appliedFingerprint,
      correctedAppliedFingerprint: nextCorrected
        ? correctedArtifactFingerprint(nextCorrected)
        : undefined,
    });
    setVoiceprintComparison({
      changedSegments: response.stats.changed_segments,
      matchedSegments: response.stats.matched_segments,
      reviewSegments: response.stats.review_segments,
      profileCount: response.stats.profile_count,
      rejectedAnchors: response.stats.rejected_anchor_count,
      rejectedProfiles: response.stats.rejected_profile_count ?? 0,
      speakersBefore: collectSpeakers(snapshot.segments).length,
      speakersAfter: collectSpeakers(response.segments).length,
    });

    const readyProfiles = response.profiles.filter((profile) => profile.enrollment_ready !== false);
    if (saveProfiles && readyProfiles.length > 0) {
      assertTaskResultRevision(requestTaskId, appliedFingerprint);
      await patchDiarization((current) => ({
        speakers: mergeVoiceprintProfiles(current.speakers ?? [], readyProfiles),
      }));
    }
  }

  async function undoVoiceprintReidentify() {
    if (!voiceprintSnapshot) return;
    const requestTaskId = task.id;
    if (voiceprintSnapshot.taskId !== requestTaskId) {
      setError("撤销快照不属于当前任务，已拒绝应用");
      return;
    }
    try {
      assertTaskResultRevision(requestTaskId, voiceprintSnapshot.appliedFingerprint);
      const currentTask = useTasks.getState().tasks.find((candidate) => candidate.id === requestTaskId);
      if (!currentTask?.result) throw new Error("当前任务已不存在");
      const currentCorrected = captureTaskCorrectedRevision(currentTask);
      if (
        (voiceprintSnapshot.correctedAppliedFingerprint
          && currentCorrected.fingerprint !== voiceprintSnapshot.correctedAppliedFingerprint)
        || (!voiceprintSnapshot.correctedAppliedFingerprint && currentCorrected.artifact)
      ) {
        throw new Error("校对稿已在声纹回扫后发生变化，已拒绝覆盖撤销");
      }
      const restoredResult: TranscribeResult = {
        ...currentTask.result,
        segments: voiceprintSnapshot.before.segments,
        diarization_stats: voiceprintSnapshot.before.diarization_stats,
      };
      await persistSpeakerArtifacts({
        baseTask: currentTask,
        requestTaskId,
        sourceFingerprint: voiceprintSnapshot.appliedFingerprint,
        nextResult: restoredResult,
        correctedRevision: currentCorrected,
        nextCorrected: voiceprintSnapshot.correctedBefore,
      });
      if (!isTaskCurrentTarget(requestTaskId)) return;
      setVoiceprintSnapshot(null);
      setVoiceprintComparison(null);
    } catch (e) {
      if (isTaskCurrentTarget(requestTaskId)) setError(`撤销声纹重识别失败: ${String(e)}`);
      throw e;
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-border/60">
        <div className="min-w-0 flex-1 text-ui-sm text-fg-mute">
          {speakers.length > 0 ? (
            <div className="flex flex-wrap items-center gap-1.5">
              <span>识别到 {speakers.length} 位说话人</span>
              {speakers.map((speaker) => (
                <SpeakerChip
                  key={speaker}
                  speakers={speakers}
                  who={speaker}
                  onRename={onRenameSpeaker}
                />
              ))}
              <span className="text-fg-mute">点击标签改名</span>
            </div>
          ) : (
            !timingReliable
              ? "时间轴不可靠，已暂停说话人分离"
              : diarizationAnalyzed
              ? "已分析说话人，置信度不足，需点推荐人数后人工应用"
              : "未运行说话人分离 - 点右侧按钮跑一次"
          )}
        </div>
        {hint && (
          <div
            className={clsx("shrink-0 rounded-sm border px-1.5 py-0.5 text-xs", hint.className)}
            title={hint.title}
          >
            {hint.text}
          </div>
        )}
        <div className="flex items-center gap-2">
          <button
            onClick={recommend}
            disabled={!timingReliable || busy || recommending || applying !== null}
            className="btn-ghost flex items-center gap-1.5 text-ui-sm"
            title="试跑 2-8 人候选,推荐最稳的人数"
          >
            <Hourglass size={12} className={recommending ? "animate-spin" : ""} />
            {recommending ? "分析中..." : "推荐人数"}
          </button>
          <button
            onClick={rerun}
            disabled={!timingReliable || busy || recommending || applying !== null}
            className="btn-ghost flex items-center gap-1.5 text-ui-sm"
            title={autoSpeakerCount ? "不重新转录,试跑 2-8 人候选并应用推荐人数" : "不重新转录,按设置人数重新识别说话人"}
          >
            <Refresh size={12} className={busy ? "animate-spin" : ""} />
            {busy ? "分人中..." : autoSpeakerCount ? "自动检测人数" : speakers.length > 0 ? "重新跑分人" : "运行说话人分离"}
          </button>
        </div>
      </div>
      {recommendation && (
        <DiarizationRecommendation
          recommendation={recommendation.response}
          applying={applying}
          currentSpeakers={speakers.length}
          onApply={applyCandidate}
        />
      )}
      {error && (
        <div className="text-ui-sm text-err bg-err/10 border border-err/30 rounded-sm px-3 py-2">
          {error}
        </div>
      )}
      {!timingReliable && (
        <div className="text-ui-sm text-err bg-err/10 border border-err/30 rounded-sm px-3 py-2">
          音频与文字时间轴未通过校验：{timingFailureReason}。已关闭文字跟随和说话人分离，请重新转录。
        </div>
      )}
      <ASRStrongReviewStatus
        review={strongAsrReview}
        onPlayRange={(id, start, end) => {
          void playCoverageRange(id, start, end).catch((playError) => setError(String(playError)));
        }}
      />
      <TranscriptAudioSyncBar
        audio={audioSource}
        audioRef={audioRef}
        playbackTime={playbackTime}
        activeSegment={activeSegment}
        activeCue={activeCue}
        autoFollow={timingReliable && autoFollowTranscript}
        timingReliable={timingReliable}
        onTimeChange={handlePlaybackTimeChange}
        onPlayingChange={setAudioPlaying}
        onPlaybackEnded={finishCoveragePlayback}
        onToggleAutoFollow={() => {
          if (timingReliable) setAutoFollowTranscript((value) => !value);
        }}
      />
      {coverageReport && (
        <SpeechCoverageReviewPanel
          task={task}
          report={coverageReport}
          playingId={audioPlaying ? coveragePlaybackId : null}
          onPlayRange={playCoverageRange}
        />
      )}
      {timingReliable && (
        <VoiceprintReidentifyPanel
          key={task.id}
          task={task}
          segments={segments}
          activeSegment={activeSegment}
          activeSegmentIndex={activeSegmentIndex}
          speakers={speakers}
          engine={settings.diarization?.engine || "auto"}
          onApply={applyVoiceprintReidentify}
          comparison={voiceprintComparison}
          onUndo={voiceprintSnapshot ? undoVoiceprintReidentify : undefined}
        />
      )}
      <EntityConsistencyPanel
        result={task.result!}
        onApplyUnify={onApplyEntityUnify}
        onApplyOccurrence={onApplyEntityOccurrence}
      />
      <RawSegments
        segments={segments}
        viewMode={viewMode}
        activeSegmentIndex={activeSegmentIndex}
        playbackTime={transcriptSyncTime}
        autoFollow={autoFollowTranscript}
        onSeekSegment={timingReliable ? seekToSegment : undefined}
        onRenameSpeaker={onRenameSpeaker}
      />
    </div>
  );
}

function strongReviewReason(reason: string): string {
  if (reason === "qwen_model_not_cached") return "Qwen3 本地模型不完整，未执行复核";
  if (reason === "paraformer_model_not_cached") return "Paraformer 本地模型不完整，未执行复核";
  if (reason.startsWith("paraformer_failed:")) return "Paraformer 复核失败，已保留原转录";
  if (reason.startsWith("qwen_failed:")) return "Qwen3 复核失败，已保留原转录";
  if (reason.startsWith("strong_review_failed:")) return "本地高质量复核失败，已保留原转录";
  if (reason === "audit_only_candidates_recorded") return "存在严重分歧候选，已保留原转录并标记审计";
  if (reason === "no_confirmed_consensus_changes") return "本地复核完成，没有满足双重证据的改字";
  if (reason === "consensus_changes_applied") return "本地复核完成，已应用双重证据确认的局部改字";
  return reason || "本地复核状态未知";
}

function ASRStrongReviewStatus({
  review,
  onPlayRange,
}: {
  review?: ASRStrongReviewStats;
  onPlayRange: (id: string, start: number, end: number) => void;
}) {
  if (!review || (!review.enabled && !review.review_recommended)) return null;

  if (review.review_recommended && !review.enabled) {
    const snr = review.auto_review_decision?.estimated_snr_db;
    const reason = String(review.reason || "");
    const status = reason === "independent_reference_unavailable"
      ? "缺少独立复核证据，已保留原转录"
      : reason === "high_noise_independent_review_required"
        ? "极端噪声且缺少独立模型证据，已保留原转录并建议复核"
      : "高噪声录音的自动复核已关闭，当前保留原转录";
    return (
      <div className="flex flex-wrap items-center gap-2 rounded-sm border border-warn/35 bg-warn/10 px-3 py-2 text-ui-sm text-warn">
        <Warning size={13} />
        <span>{status}</span>
        {typeof snr === "number" && <span className="font-mono text-fg-mute">SNR {snr.toFixed(1)} dB</span>}
      </div>
    );
  }

  const reason = String(review.reason || "");
  const failed = reason.includes("not_cached") || reason.includes("failed:");
  const auditCount = Number(review.audit_only_window_count || 0);
  const replacementCount = Number(review.replacement_count || 0);
  const auditCandidates = (review.audit_only_candidates || []).slice(0, 5);
  const automaticStatus = reason === "no_confirmed_consensus_changes"
    ? "高噪声录音自动复核完成，没有满足双重证据的改字"
    : reason === "consensus_changes_applied"
      ? "高噪声录音自动复核完成，已应用双重证据确认的局部改字"
      : reason === "audit_only_candidates_recorded"
        ? "高噪声录音自动复核发现严重分歧，已保留原转录并标记审计"
        : strongReviewReason(reason);
  const statusText = review.trigger === "auto_high_noise" && !failed
    ? automaticStatus
    : strongReviewReason(reason);
  return (
    <section className="rounded-sm border border-border bg-bg-panel/50">
      <div
        className={clsx(
          "flex flex-wrap items-center gap-2 px-3 py-2 text-ui-sm",
          failed || auditCount > 0 ? "text-warn" : "text-ok",
        )}
      >
        {failed || auditCount > 0 ? <Warning size={13} /> : <Check size={13} />}
        <span>{statusText}</span>
        {replacementCount > 0 && <span>改字 {replacementCount} 处</span>}
        {auditCount > 0 && <span>待审计 {auditCount} 个窗口</span>}
        {typeof review.cost_seconds === "number" && (
          <span className="font-mono text-fg-mute">{review.cost_seconds.toFixed(1)}s</span>
        )}
      </div>
      {auditCandidates.length > 0 && (
        <details className="border-t border-border/60 px-3 py-2 text-ui-sm">
          <summary className="cursor-pointer text-fg-dim">查看三方候选</summary>
          <div className="mt-2 divide-y divide-border/50">
            {auditCandidates.map((candidate, index) => {
              const start = Number(candidate.start || 0);
              const end = Math.max(start, Number(candidate.end || start));
              return (
                <div key={`${start}-${end}-${index}`} className="grid gap-1 py-2 first:pt-0 last:pb-0">
                  <div className="flex items-center gap-2 text-fg-mute">
                    <button
                      type="button"
                      className="btn-ghost grid h-6 w-6 shrink-0 place-items-center p-0"
                      title="播放审计窗口"
                      aria-label="播放审计窗口"
                      onClick={() => onPlayRange(`strong-asr-${index}`, start, end)}
                    >
                      <Play size={11} />
                    </button>
                    <span className="font-mono">{formatTimeShort(start)}-{formatTimeShort(end)}</span>
                    {typeof candidate.primary_para_similarity === "number" && (
                      <span>主模型/Paraformer {candidate.primary_para_similarity.toFixed(3)}</span>
                    )}
                  </div>
                  <div><span className="text-fg-mute">当前：</span>{candidate.primary || "无"}</div>
                  <div><span className="text-fg-mute">Paraformer：</span>{candidate.paraformer || "无"}</div>
                  <div><span className="text-fg-mute">Qwen3：</span>{candidate.qwen || "无"}</div>
                </div>
              );
            })}
          </div>
        </details>
      )}
    </section>
  );
}

function TranscriptAudioSyncBar({
  audio,
  audioRef,
  playbackTime,
  activeSegment,
  activeCue,
  autoFollow,
  timingReliable,
  onTimeChange,
  onPlayingChange,
  onPlaybackEnded,
  onToggleAutoFollow,
}: {
  audio: string;
  audioRef: RefObject<HTMLAudioElement>;
  playbackTime: number;
  activeSegment: Segment | null;
  activeCue: TranscriptCue | null;
  autoFollow: boolean;
  timingReliable: boolean;
  onTimeChange: (time: number) => void;
  onPlayingChange: (playing: boolean) => void;
  onPlaybackEnded: () => void;
  onToggleAutoFollow: () => void;
}) {
  const [src, setSrc] = useState("");
  const [error, setError] = useState<string | null>(null);
  const onTimeChangeRef = useRef(onTimeChange);

  useEffect(() => {
    onTimeChangeRef.current = onTimeChange;
  }, [onTimeChange]);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    setSrc("");
    if (!audio) return;
    localMediaUrl(audio)
      .then((url) => {
        if (!cancelled) setSrc(url);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [audio]);

  useEffect(() => {
    const element = audioRef.current;
    if (!src || !element) return;

    let animationFrame: number | null = null;
    const stopFrameLoop = () => {
      if (animationFrame != null) cancelAnimationFrame(animationFrame);
      animationFrame = null;
    };
    const updateFromPlaybackClock = () => {
      onTimeChangeRef.current(element.currentTime);
      if (!element.paused && !element.ended) {
        animationFrame = requestAnimationFrame(updateFromPlaybackClock);
      } else {
        animationFrame = null;
      }
    };
    const startFrameLoop = () => {
      stopFrameLoop();
      animationFrame = requestAnimationFrame(updateFromPlaybackClock);
    };
    const stopAndSync = () => {
      stopFrameLoop();
      onTimeChangeRef.current(element.currentTime);
    };

    element.addEventListener("play", startFrameLoop);
    element.addEventListener("pause", stopAndSync);
    element.addEventListener("ended", stopAndSync);
    if (!element.paused && !element.ended) startFrameLoop();

    return () => {
      stopFrameLoop();
      element.removeEventListener("play", startFrameLoop);
      element.removeEventListener("pause", stopAndSync);
      element.removeEventListener("ended", stopAndSync);
    };
  }, [audioRef, src]);

  if (!audio) return null;

  return (
    <section className="rounded-sm border border-border bg-bg-panel/70 px-3 py-2">
      <div className="flex flex-wrap items-center gap-3">
        <div className="min-w-0 flex-1">
          {src ? (
            <audio
              ref={audioRef}
              src={src}
              controls
              preload="metadata"
              className="block h-8 w-full min-w-[260px]"
              onTimeUpdate={(event) => onTimeChange(event.currentTarget.currentTime)}
              onSeeked={(event) => onTimeChange(event.currentTarget.currentTime)}
              onLoadedMetadata={(event) => onTimeChange(event.currentTarget.currentTime)}
              onPlay={() => onPlayingChange(true)}
              onPause={() => onPlayingChange(false)}
              onEnded={() => {
                onPlayingChange(false);
                onPlaybackEnded();
              }}
              onError={() => {
                onPlaybackEnded();
                setError("音频无法在当前环境播放");
              }}
            />
          ) : (
            <div className="h-8 flex items-center text-ui-sm text-fg-mute">
              正在准备音频播放器...
            </div>
          )}
        </div>
        <button
          onClick={onToggleAutoFollow}
          disabled={!timingReliable}
          className={clsx(
            "shrink-0 rounded-sm border px-2 py-1 text-ui-sm",
            timingReliable && autoFollow
              ? "border-accent/35 bg-accent/10 text-accent"
              : "border-border bg-bg text-fg-mute hover:text-fg",
          )}
          title="播放时自动滚动并高亮当前转录段"
        >
          {!timingReliable ? "时间轴不可用" : autoFollow ? "文字跟随中" : "文字不跟随"}
        </button>
      </div>
      <div className="mt-2 flex flex-wrap items-start gap-2 text-ui-sm">
        <span className="shrink-0 font-mono text-accent">{formatTimeShort(playbackTime)}</span>
        <span
          className={clsx(
            "shrink-0 rounded-sm border px-1.5 py-0.5 text-[11px]",
            !timingReliable
              ? "border-err/35 bg-err/10 text-err"
              : activeCue?.source === "sync"
              ? "border-ok/35 bg-ok/10 text-ok"
              : "border-sky-300/30 bg-sky-500/10 text-sky-300",
          )}
          title={!timingReliable ? "时间轴校验未通过" : activeCue?.source === "sync" ? "使用模型返回的短语级时间轴" : "当前结果没有短语级时间轴，只跟随到当前转录段"}
        >
          {!timingReliable ? "不可同步" : activeCue?.source === "sync" ? "精确同步" : "段级同步"}
        </span>
        <span className="shrink-0 text-fg-mute">当前</span>
        <span className="min-w-0 flex-1 truncate text-fg-dim">
          {activeCue?.text || activeSegment?.text || "播放后会自动定位到对应文字"}
        </span>
      </div>
      {error && (
        <div className="mt-2 text-ui-sm text-err">
          {error}
        </div>
      )}
    </section>
  );
}

type CoverageReviewStatus = ASRHumanReviewItemStatus;

type JsonObject = Record<string, unknown>;

type CoverageReviewSourceItem = {
  id: string;
  start: number;
  end: number;
  reasons: JsonObject[];
  recoveryDetails: JsonObject[];
  sourcePresent: boolean;
};

type CoverageReviewItemModel = CoverageReviewSourceItem & {
  status: CoverageReviewStatus;
  heardText: string;
  replacementText: string;
  note: string;
};

type CoverageReviewDraft = Partial<Pick<CoverageReviewItemModel, "status" | "heardText" | "replacementText" | "note">>;

const COVERAGE_REVIEW_STATUS_META: Record<CoverageReviewStatus, { label: string; className: string }> = {
  pending: { label: "待复核", className: "border-warn/35 bg-warn/10 text-warn" },
  confirmed_present: { label: "确认已有", className: "border-ok/35 bg-ok/10 text-ok" },
  confirmed_missing: { label: "确认漏字", className: "border-err/35 bg-err/10 text-err" },
  substitution: { label: "错词替换", className: "border-orange-300/35 bg-orange-500/10 text-orange-300" },
  noise: { label: "噪声/非语音", className: "border-border bg-bg text-fg-mute" },
  resolved: { label: "已应用", className: "border-accent/35 bg-accent/10 text-accent" },
};

const USER_SELECTABLE_COVERAGE_STATUSES: CoverageReviewStatus[] = [
  "pending",
  "confirmed_present",
  "confirmed_missing",
  "substitution",
  "noise",
];

function asJsonObject(value: unknown): JsonObject | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as JsonObject;
}

function finiteNumber(value: unknown): number | null {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : value == null ? "" : String(value);
}

function normalizedCoverageRange(value: unknown): { id: string; start: number; end: number } | null {
  let start: number | null = null;
  let end: number | null = null;
  if (Array.isArray(value) && value.length >= 2) {
    start = finiteNumber(value[0]);
    end = finiteNumber(value[1]);
  } else {
    const record = asJsonObject(value);
    if (record) {
      start = finiteNumber(record.start ?? record.start_s);
      end = finiteNumber(record.end ?? record.end_s);
    }
  }
  if (start == null || end == null || end <= start) return null;
  const startMs = Math.max(0, Math.round(start * 1000));
  const endMs = Math.max(startMs + 1, Math.round(end * 1000));
  return {
    id: `coverage:${startMs}-${endMs}`,
    start: startMs / 1000,
    end: endMs / 1000,
  };
}

function pushUniqueRecord(target: JsonObject[], value: JsonObject) {
  const signature = JSON.stringify(value);
  if (!target.some((item) => JSON.stringify(item) === signature)) target.push(value);
}

function buildCoverageReviewSourceItems(report: SpeechCoverageReport): CoverageReviewSourceItem[] {
  const reportRecord = asJsonObject(report) ?? {};
  const recovery = asJsonObject(reportRecord.local_recovery);
  const recoveryBefore = asJsonObject(recovery?.before);
  const recoveryAfter = asJsonObject(recovery?.after);
  const byId = new Map<string, CoverageReviewSourceItem>();

  function ensure(value: unknown): CoverageReviewSourceItem | null {
    const range = normalizedCoverageRange(value);
    if (!range) return null;
    const existing = byId.get(range.id);
    if (existing) return existing;
    const item: CoverageReviewSourceItem = {
      ...range,
      reasons: [],
      recoveryDetails: [],
      sourcePresent: true,
    };
    byId.set(item.id, item);
    return item;
  }

  const rangeLists = [
    reportRecord.wallclock_failed_ranges,
    reportRecord.failed_ranges,
    reportRecord.uncovered_speech_ranges,
    recoveryBefore?.failed_ranges,
    recoveryAfter?.failed_ranges,
  ];
  for (const list of rangeLists) {
    if (!Array.isArray(list)) continue;
    for (const value of list) ensure(value);
  }

  const failureReasons = [
    ...(Array.isArray(reportRecord.wallclock_failure_reasons) ? reportRecord.wallclock_failure_reasons : []),
    ...(Array.isArray(reportRecord.failure_reasons) ? reportRecord.failure_reasons : []),
  ];
  for (const value of failureReasons) {
    const reason = asJsonObject(value);
    if (!reason) continue;
    const item = ensure(reason);
    if (item) pushUniqueRecord(item.reasons, reason);
  }

  const recoveryDetails = Array.isArray(recovery?.details) ? recovery.details : [];
  for (const value of recoveryDetails) {
    const detail = asJsonObject(value);
    if (!detail) continue;
    const originals = Array.isArray(detail.original_failures) ? detail.original_failures : [];
    if (originals.length > 0) {
      for (const originalValue of originals) {
        const original = asJsonObject(originalValue);
        const item = ensure(originalValue);
        if (!item) continue;
        if (original) pushUniqueRecord(item.reasons, original);
        pushUniqueRecord(item.recoveryDetails, detail);
      }
    } else {
      const item = ensure(detail);
      if (item) pushUniqueRecord(item.recoveryDetails, detail);
    }
  }

  return Array.from(byId.values()).sort((left, right) => left.start - right.start || left.end - right.end);
}

function isCoverageReviewStatus(value: unknown): value is CoverageReviewStatus {
  return value === "pending"
    || value === "confirmed_present"
    || value === "confirmed_missing"
    || value === "substitution"
    || value === "noise"
    || value === "resolved";
}

function parseCoverageIdRange(id: string): { id: string; start: number; end: number } | null {
  const match = id.match(/^coverage:(\d+)-(\d+)$/);
  if (!match) return null;
  return normalizedCoverageRange([Number(match[1]) / 1000, Number(match[2]) / 1000]);
}

function savedCoverageReviewItems(review: ASRHumanReview | undefined): CoverageReviewItemModel[] {
  const record = asJsonObject(review);
  const rawItems = Array.isArray(record?.items)
    ? record.items
    : Array.isArray(record?.entries)
      ? record.entries
      : [];
  const byId = new Map<string, CoverageReviewItemModel>();
  for (const value of rawItems) {
    const item = asJsonObject(value);
    if (!item) continue;
    const rawId = stringValue(item.id);
    const range = normalizedCoverageRange(item) ?? parseCoverageIdRange(rawId);
    if (!range) continue;
    const storedStatus = item.review_status ?? item.status;
    const legacyCorrectedText = stringValue(item.corrected_text);
    const explicitReplacementText = stringValue(item.replacement_text);
    const heardText = stringValue(
      item.heard_text
      ?? item.heardText
      ?? (explicitReplacementText ? "" : legacyCorrectedText),
    );
    const status: CoverageReviewStatus = isCoverageReviewStatus(storedStatus)
      ? storedStatus
      : storedStatus === "approved"
        ? heardText || explicitReplacementText ? "resolved" : "confirmed_present"
        : storedStatus === "needs_changes"
          ? heardText || explicitReplacementText ? "substitution" : "pending"
          : "pending";
    const replacementText = explicitReplacementText
      || (status === "substitution" ? legacyCorrectedText : "");
    byId.set(range.id, {
      ...range,
      status,
      heardText,
      replacementText,
      note: stringValue(item.note),
      reasons: [],
      recoveryDetails: [],
      sourcePresent: false,
    });
  }
  return Array.from(byId.values());
}

function savedCoverageReviewRecords(review: ASRHumanReview | undefined): Map<string, JsonObject> {
  const record = asJsonObject(review);
  const rawItems = Array.isArray(record?.items)
    ? record.items
    : Array.isArray(record?.entries)
      ? record.entries
      : [];
  const byId = new Map<string, JsonObject>();
  for (const value of rawItems) {
    const item = asJsonObject(value);
    if (!item) continue;
    const range = normalizedCoverageRange(item) ?? parseCoverageIdRange(stringValue(item.id));
    if (range) byId.set(range.id, item);
  }
  return byId;
}

function mergeCoverageReviewItems(
  sourceItems: CoverageReviewSourceItem[],
  savedItems: CoverageReviewItemModel[],
  resetSavedStatuses: boolean,
): CoverageReviewItemModel[] {
  const savedById = new Map(savedItems.map((item) => [item.id, item]));
  const merged = sourceItems.map((source) => {
    const saved = savedById.get(source.id);
    return {
      ...source,
      status: resetSavedStatuses ? "pending" as const : saved?.status ?? "pending",
      heardText: resetSavedStatuses ? "" : saved?.heardText ?? "",
      replacementText: resetSavedStatuses ? "" : saved?.replacementText ?? "",
      note: resetSavedStatuses ? "" : saved?.note ?? "",
    };
  });
  if (!resetSavedStatuses) {
    for (const saved of savedItems) {
      if (!sourceItems.some((source) => source.id === saved.id)) {
        merged.push(saved);
      }
    }
  }
  return merged.sort((left, right) => left.start - right.start || left.end - right.end);
}

type CoverageReviewFingerprint = {
  failedPartitionHash: string;
  recoveryAfterTextSha256: string;
  transcriptFingerprint: string;
  backend: string;
  modelId: string;
  segmentCount: number;
};

function coveragePartitionHash(report: SpeechCoverageReport, sourceItems: CoverageReviewSourceItem[]): string {
  const reportRecord = asJsonObject(report) ?? {};
  const recovery = asJsonObject(reportRecord.local_recovery);
  const before = asJsonObject(recovery?.before);
  const after = asJsonObject(recovery?.after);
  const canonical = [
    before?.failed_partition_sha256,
    reportRecord.failed_partition_sha256,
    reportRecord.coverage_partition_sha256,
    after?.failed_partition_sha256,
  ].map(stringValue).find(Boolean);
  return canonical || `coverage-ranges-v1:${sourceItems.map((item) => item.id).join("|")}`;
}

function currentCoverageReviewFingerprint(
  task: Task,
  report: SpeechCoverageReport,
  sourceItems: CoverageReviewSourceItem[],
): CoverageReviewFingerprint {
  const reportRecord = asJsonObject(report) ?? {};
  const recovery = asJsonObject(reportRecord.local_recovery);
  const after = asJsonObject(recovery?.after);
  return {
    failedPartitionHash: coveragePartitionHash(report, sourceItems),
    recoveryAfterTextSha256: stringValue(after?.text_sha256),
    transcriptFingerprint: segmentRevisionFingerprint(task.result?.segments ?? []),
    backend: task.result?.backend || "",
    modelId: task.result?.model_id || "",
    segmentCount: task.result?.segments.length || 0,
  };
}

function savedCoverageReviewFingerprint(review: ASRHumanReview | undefined): CoverageReviewFingerprint | null {
  const record = asJsonObject(review);
  if (!record) return null;
  const source = asJsonObject(record.source) ?? {};
  return {
    failedPartitionHash: [
      record.coverage_partition_sha256,
      record.source_failed_partition_sha256,
      record.failed_partition_sha256,
      record.partition_hash,
    ].map(stringValue).find(Boolean) || "",
    recoveryAfterTextSha256: stringValue(source.recovery_after_text_sha256 || source.text_sha256),
    transcriptFingerprint: stringValue(source.transcript_fingerprint),
    backend: stringValue(source.backend),
    modelId: stringValue(source.model_id),
    segmentCount: finiteNumber(source.segment_count) ?? -1,
  };
}

function coverageReviewStaleReasons(
  saved: CoverageReviewFingerprint | null,
  current: CoverageReviewFingerprint,
): string[] {
  if (!saved) return [];
  const reasons: string[] = [];
  if (saved.failedPartitionHash !== current.failedPartitionHash) reasons.push("failed partition hash");
  if (saved.recoveryAfterTextSha256 !== current.recoveryAfterTextSha256) reasons.push("recovery after text_sha256");
  if (saved.transcriptFingerprint !== current.transcriptFingerprint) reasons.push("current transcript fingerprint");
  if (saved.backend !== current.backend) reasons.push("backend");
  if (saved.modelId !== current.modelId) reasons.push("model");
  if (saved.segmentCount !== current.segmentCount) reasons.push("segment_count");
  return reasons;
}

function coveragePartitionInvalid(report: SpeechCoverageReport): boolean {
  const reportRecord = asJsonObject(report) ?? {};
  const recovery = asJsonObject(reportRecord.local_recovery);
  const before = asJsonObject(recovery?.before);
  const after = asJsonObject(recovery?.after);
  return before?.partition_valid === false || after?.partition_valid === false;
}

function coverageDetailsTruncated(report: SpeechCoverageReport): boolean {
  const reportRecord = asJsonObject(report) ?? {};
  const recovery = asJsonObject(reportRecord.local_recovery);
  return reportRecord.wallclock_failure_details_truncated === true || recovery?.details_truncated === true;
}

function overlapSeconds(segment: Segment, start: number, end: number): number {
  return Math.max(0, Math.min(segment.end, end) - Math.max(segment.start, start));
}

type CoverageReviewTarget =
  | { kind: "insert" }
  | { kind: "replace"; index: number; segment: Segment };

function coverageReviewTarget(
  segments: Segment[],
  item: Pick<CoverageReviewItemModel, "start" | "end">,
): CoverageReviewTarget {
  let bestIndex = -1;
  let bestOverlap = 0;
  segments.forEach((segment, index) => {
    const overlap = overlapSeconds(segment, item.start, item.end);
    if (overlap > bestOverlap) {
      bestOverlap = overlap;
      bestIndex = index;
    }
  });
  return bestIndex >= 0 && bestOverlap > 0
    ? { kind: "replace", index: bestIndex, segment: segments[bestIndex] }
    : { kind: "insert" };
}

function normalizedReviewCharCount(value: string): number {
  return Array.from(value.normalize("NFKC").replace(/[^\p{L}\p{N}]+/gu, "")).length;
}

function normalizedReviewComparable(value: string): string {
  return value.normalize("NFKC").replace(/\s+/g, " ").trim();
}

function applyCoverageReviewToSegments(
  segments: Segment[],
  item: CoverageReviewItemModel,
  heardText: string,
  replacementText: string,
  targetLabel: string,
): Segment[] {
  const target = coverageReviewTarget(segments, item);
  if (item.status === "confirmed_missing" && target.kind === "insert") {
    if (segments.some((segment) => overlapSeconds(segment, item.start, item.end) > 0)) {
      throw new Error(`${targetLabel}中该时间范围与现有段落重叠，已拒绝插入`);
    }
    const insertedText = heardText.trim();
    if (!insertedText) throw new Error("请先填写实际听到的漏字内容");
    const inserted: Segment = {
      start: item.start,
      end: item.end,
      text: insertedText,
      original_text: "",
    };
    return [...segments, inserted].sort((left, right) => left.start - right.start || left.end - right.end);
  }

  if (target.kind !== "replace") {
    throw new Error(`${targetLabel}中没有与 ${formatCoverageTime(item.start)}-${formatCoverageTime(item.end)} 重叠的段落`);
  }
  const replacement = replacementText.trim();
  if (!replacement) throw new Error("请填写纠正后的完整段落");
  if (normalizedReviewComparable(replacement) === normalizedReviewComparable(target.segment.text)) {
    throw new Error("纠正后的完整段落必须与当前目标段不同，不能在未修改正文时标记 resolved");
  }
  const originalChars = normalizedReviewCharCount(target.segment.text);
  const replacementChars = normalizedReviewCharCount(replacement);
  const minimumChars = Math.max(1, Math.ceil(originalChars * 0.5));
  if (replacementChars < minimumChars) {
    throw new Error(`纠正后的完整段落至少需 ${minimumChars} 个规范化字符（原段 ${originalChars} 个）`);
  }
  return segments.map((segment, index) => index === target.index
    ? {
        ...segment,
        original_text: segment.original_text ?? segment.text,
        text: replacement,
        sync_cues: undefined,
      }
    : segment);
}

function coverageReasonText(reason: JsonObject): string {
  const code = stringValue(reason.reason) || "unknown";
  const labels: Record<string, string> = {
    empty_transcript: "未识别出文字",
    low_text_density: "识别文字密度过低",
    chunk_inference_failed: "分块识别失败",
    unknown: "原因未知",
  };
  const details = [
    finiteNumber(reason.speech_duration_s) != null ? `语音 ${finiteNumber(reason.speech_duration_s)!.toFixed(2)}s` : "",
    finiteNumber(reason.normalized_chars) != null ? `字符 ${finiteNumber(reason.normalized_chars)}` : "",
    finiteNumber(reason.chars_per_s) != null ? `密度 ${finiteNumber(reason.chars_per_s)!.toFixed(2)}/s` : "",
  ].filter(Boolean);
  return `${labels[code] || code}${details.length ? `（${details.join(" · ")}）` : ""}`;
}

function recoveryDecisionLabel(detail: JsonObject): string {
  const decision = stringValue(detail.decision || detail.evidence_decision);
  const labels: Record<string, string> = {
    matched_existing: "局部复核认为正文已包含",
    insert_accepted: "局部复核建议补入",
    rejected: "局部复核证据不足",
    error: "局部复核失败",
  };
  const text = stringValue(detail.inserted_text || detail.consensus || detail.primary_consensus);
  return `${labels[decision] || decision || "局部复核详情"}${text ? `：“${text}”` : ""}`;
}

function recoveryAttemptText(value: unknown): string {
  const attempt = asJsonObject(value);
  if (!attempt) return "";
  const framing = stringValue(attempt.framing) || "尝试";
  const provider = stringValue(attempt.provider_id);
  const status = stringValue(attempt.status) || "unknown";
  const text = stringValue(attempt.residual_text || attempt.normalized || attempt.raw);
  const rejection = stringValue(attempt.rejection_reason || attempt.error);
  return `${provider ? `${provider}/` : ""}${framing}: ${status}${text ? `「${text}」` : ""}${rejection ? `（${rejection}）` : ""}`;
}

function formatCoverageTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const rest = (seconds % 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${rest}`;
}

function SpeechCoverageReviewPanel({
  task,
  report,
  playingId,
  onPlayRange,
}: {
  task: Task;
  report: SpeechCoverageReport;
  playingId: string | null;
  onPlayRange: (id: string, start: number, end: number) => Promise<void>;
}) {
  const setAsrHumanReview = useTasks((state) => state.setAsrHumanReview);
  const sourceItems = useMemo(() => buildCoverageReviewSourceItems(report), [report]);
  const savedItems = useMemo(() => savedCoverageReviewItems(task.asrHumanReview), [task.asrHumanReview]);
  const savedItemRecords = useMemo(() => savedCoverageReviewRecords(task.asrHumanReview), [task.asrHumanReview]);
  const currentFingerprint = useMemo(
    () => currentCoverageReviewFingerprint(task, report, sourceItems),
    [task.id, task.result, report, sourceItems],
  );
  const savedFingerprint = useMemo(
    () => savedCoverageReviewFingerprint(task.asrHumanReview),
    [task.asrHumanReview],
  );
  const staleReasons = useMemo(
    () => coverageReviewStaleReasons(savedFingerprint, currentFingerprint),
    [savedFingerprint, currentFingerprint],
  );
  const reviewIsStale = staleReasons.length > 0;
  const baseItems = useMemo(
    () => mergeCoverageReviewItems(sourceItems, savedItems, reviewIsStale),
    [sourceItems, savedItems, reviewIsStale],
  );
  const partitionInvalid = coveragePartitionInvalid(report);
  const detailsTruncated = coverageDetailsTruncated(report);
  const reportRecord = asJsonObject(report) ?? {};
  const coverageRatio = finiteNumber(reportRecord.speech_coverage_ratio);
  const reportStatus = stringValue(reportRecord.status);
  const reportReason = stringValue(reportRecord.reason);
  const basis = stringValue(reportRecord.basis);
  const resetKey = `${task.id}|${JSON.stringify(currentFingerprint)}|${sourceItems.map((item) => item.id).join(",")}`;

  const [drafts, setDrafts] = useState<Record<string, CoverageReviewDraft>>({});
  const draftsRef = useRef<Record<string, CoverageReviewDraft>>({});
  const itemRevisionsRef = useRef<Record<string, number>>({});
  const [savingReview, setSavingReview] = useState(false);
  const [applyingId, setApplyingId] = useState<string | null>(null);
  const [hotwordBusyId, setHotwordBusyId] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    draftsRef.current = {};
    itemRevisionsRef.current = {};
    setDrafts({});
    setMessage(null);
    setError(null);
  }, [resetKey]);

  const items = useMemo(() => baseItems.map((item) => ({
    ...item,
    ...(drafts[item.id] ?? {}),
  })), [baseItems, drafts]);
  const dirty = Object.keys(drafts).length > 0;

  function patchItem(id: string, patch: CoverageReviewDraft) {
    itemRevisionsRef.current[id] = (itemRevisionsRef.current[id] ?? 0) + 1;
    const next = {
      ...draftsRef.current,
      [id]: { ...(draftsRef.current[id] ?? {}), ...patch },
    };
    draftsRef.current = next;
    setDrafts(next);
  }

  function clearItemDraft(id: string) {
    const next = { ...draftsRef.current };
    delete next[id];
    draftsRef.current = next;
    setDrafts(next);
  }

  function currentItem(id: string): CoverageReviewItemModel | null {
    const base = baseItems.find((item) => item.id === id);
    return base ? { ...base, ...(draftsRef.current[id] ?? {}) } : null;
  }

  function editableSnapshot(item: CoverageReviewItemModel): string {
    return JSON.stringify({
      status: item.status,
      heardText: item.heardText,
      replacementText: item.replacementText,
      note: item.note,
    });
  }

  function itemSnapshotStillCurrent(id: string, revision: number, snapshot: string): boolean {
    const latest = currentItem(id);
    return (itemRevisionsRef.current[id] ?? 0) === revision
      && latest != null
      && editableSnapshot(latest) === snapshot;
  }

  function buildReviewPayload(reviewItems: CoverageReviewItemModel[], updatedAt: string): ASRHumanReview {
    const previous = asJsonObject(task.asrHumanReview) ?? {};
    const previousTopLevel: JsonObject = { ...previous };
    for (const key of ["schema_version", "status", "items", "entries", "source", "updated_at"]) {
      delete previousTopLevel[key];
    }
    const previousSource = asJsonObject(previous.source) ?? {};
    const recovery = asJsonObject(reportRecord.local_recovery);
    const localRecoveryMode = ["off", "audit", "merge"].includes(stringValue(recovery?.mode))
      ? stringValue(recovery?.mode) as "off" | "audit" | "merge"
      : undefined;
    const topStatus: ASRHumanReviewStatus = reviewItems.some(
      (item) => item.status === "confirmed_missing" || item.status === "substitution",
    )
      ? "needs_changes"
      : reviewItems.some((item) => item.status === "pending")
        ? "pending"
        : "approved";

    return {
      ...previousTopLevel,
      schema_version: 1,
      status: topStatus,
      items: reviewItems.map((item) => {
        const previousItem = reviewIsStale ? {} : savedItemRecords.get(item.id) ?? {};
        const rawDecision = item.recoveryDetails
          .map((detail) => stringValue(detail.decision || detail.evidence_decision))
          .find(Boolean);
        const sourceDecision = (["matched_existing", "insert_accepted", "rejected", "error"] as string[])
          .includes(rawDecision || "")
          ? rawDecision as ASRLocalRecoveryDecision
          : undefined;
        const sourceEvidenceIds = Array.from(new Set(item.recoveryDetails.flatMap((detail) =>
          Array.isArray(detail.evidence_ids)
            ? detail.evidence_ids.filter((value): value is string => typeof value === "string")
            : [],
        )));
        return {
          ...previousItem,
          id: item.id,
          start: item.start,
          end: item.end,
          status: item.status,
          review_status: item.status,
          source_decision: sourceDecision || previousItem.source_decision as ASRLocalRecoveryDecision | undefined,
          source_evidence_ids: sourceEvidenceIds.length
            ? sourceEvidenceIds
            : previousItem.source_evidence_ids as readonly string[] | undefined,
          corrected_text: item.replacementText || item.heardText || undefined,
          heard_text: item.heardText || undefined,
          replacement_text: item.replacementText || undefined,
          note: item.note || undefined,
          reviewed_at: updatedAt,
        };
      }),
      source: {
        ...previousSource,
        backend: currentFingerprint.backend,
        model_id: currentFingerprint.modelId,
        segment_count: currentFingerprint.segmentCount,
        text_sha256: currentFingerprint.recoveryAfterTextSha256 || undefined,
        recovery_after_text_sha256: currentFingerprint.recoveryAfterTextSha256 || undefined,
        transcript_fingerprint: currentFingerprint.transcriptFingerprint,
        local_recovery_mode: localRecoveryMode,
      },
      coverage_partition_sha256: currentFingerprint.failedPartitionHash,
      reviewer: stringValue(previous.reviewer) || undefined,
      created_at: stringValue(previous.created_at) || updatedAt,
      updated_at: updatedAt,
    };
  }

  async function saveReview() {
    const requestTaskId = task.id;
    const itemSnapshot = items.map((item) => ({ ...item }));
    const revisionSnapshot = Object.fromEntries(
      itemSnapshot.map((item) => [item.id, itemRevisionsRef.current[item.id] ?? 0]),
    );
    setSavingReview(true);
    setError(null);
    setMessage(null);
    try {
      assertTaskCurrentTarget(requestTaskId);
      const unchanged = itemSnapshot.every((item) =>
        itemSnapshotStillCurrent(item.id, revisionSnapshot[item.id] ?? 0, editableSnapshot(item)),
      );
      if (!unchanged) throw new Error("复核内容在保存前已变化，请重试");
      const review = buildReviewPayload(itemSnapshot, new Date().toISOString());
      assertTaskCurrentTarget(requestTaskId);
      await ipc.librarySaveAsrReview({
        stem: persistenceStem(task),
        review,
      });
      if (!isTaskCurrentTarget(requestTaskId)) return;
      const stillUnchanged = itemSnapshot.every((item) =>
        itemSnapshotStillCurrent(item.id, revisionSnapshot[item.id] ?? 0, editableSnapshot(item)),
      );
      if (!stillUnchanged) {
        setError("复核已落盘，但编辑内容在完成前发生变化；当前修改仍保持未保存状态。");
        return;
      }
      setAsrHumanReview(requestTaskId, review);
      draftsRef.current = {};
      itemRevisionsRef.current = {};
      setDrafts({});
      setMessage("复核记录已保存；canonical raw transcript/evidence 未被修改。");
    } catch (e) {
      if (isTaskCurrentTarget(requestTaskId)) {
        setError(`复核保存失败：${String(e)}。当前编辑保持 dirty，未修改 resolved 状态。`);
      }
    } finally {
      if (isTaskCurrentTarget(requestTaskId)) setSavingReview(false);
    }
  }

  async function applyToTranscript(item: CoverageReviewItemModel) {
    if (!task.result) return;
    if (item.status !== "confirmed_missing" && item.status !== "substitution") {
      setError("只有“确认漏字”或“错词替换”可以应用到人工校对稿。");
      return;
    }
    const requestTaskId = task.id;
    const revision = itemRevisionsRef.current[item.id] ?? 0;
    const snapshot = editableSnapshot(item);
    const itemSnapshot = items.map((current) => ({ ...current }));
    const storeTask = useTasks.getState().tasks.find((candidate) => candidate.id === requestTaskId);
    if (!storeTask?.result) {
      setError("当前任务正文已不存在，无法应用人工复核。");
      return;
    }
    const expectedCorrected = captureTaskCorrectedRevision(storeTask);
    const correctedBase = humanCorrectedArtifact(storeTask);
    const target = coverageReviewTarget(correctedBase.segments, item);
    if (item.status === "substitution" && target.kind !== "replace") {
      setError("错词替换必须命中一个现有校对段落；当前范围无重叠，已拒绝应用。");
      return;
    }
    setApplyingId(item.id);
    setError(null);
    setMessage(null);
    let correctedSaved = false;
    try {
      assertTaskCurrentTarget(requestTaskId);
      const nextSegments = applyCoverageReviewToSegments(
        correctedBase.segments,
        item,
        item.heardText,
        item.replacementText,
        "校对稿",
      );
      const nextCorrected = correctedArtifactWithSegments(correctedBase, nextSegments);
      assertTaskCurrentTarget(requestTaskId);
      await saveCorrectedSegments(
        storeTask,
        nextSegments,
        nextCorrected,
        expectedCorrected,
      );
      correctedSaved = true;
      if (!isTaskCurrentTarget(requestTaskId)) return;
      if (!itemSnapshotStillCurrent(item.id, revision, snapshot)) {
        setError("人工校对稿已保存，但本条复核内容在完成前发生变化；未标记 resolved，复核保持 dirty。");
        return;
      }

      const resolvedItems = itemSnapshot.map((current) => current.id === item.id
        ? { ...current, status: "resolved" as const }
        : current);
      const review = buildReviewPayload(resolvedItems, new Date().toISOString());
      assertTaskCurrentTarget(requestTaskId);
      if (!itemSnapshotStillCurrent(item.id, revision, snapshot)) {
        throw new Error("复核内容已变化，未保存 resolved 状态");
      }
      await ipc.librarySaveAsrReview({
        stem: persistenceStem(task),
        review,
      });
      if (!isTaskCurrentTarget(requestTaskId)) return;
      if (!itemSnapshotStillCurrent(item.id, revision, snapshot)) {
        setError("人工校对稿与复核记录已落盘，但本条内容在完成前变化；界面保持未 resolved，请重新确认。");
        return;
      }
      setAsrHumanReview(requestTaskId, review);
      clearItemDraft(item.id);
      setMessage(expectedCorrected.artifact
        ? "已更新并保存人工校对稿；复核记录已标记为“已应用”。"
        : "已从 canonical raw 克隆并创建 human-review 校对稿；复核记录已标记为“已应用”。");
    } catch (e) {
      if (isTaskCurrentTarget(requestTaskId)) {
        setError(correctedSaved
          ? `人工校对稿已保存，但复核记录保存失败：${String(e)}。本条保持 dirty/未 resolved。`
          : `人工校对稿保存失败：${String(e)}。本条保持 dirty，复核未标记 resolved。`);
      }
    } finally {
      if (isTaskCurrentTarget(requestTaskId)) setApplyingId(null);
    }
  }

  async function addHotword(item: CoverageReviewItemModel) {
    const requestTaskId = task.id;
    const hotword = item.heardText.trim();
    if (!hotword) {
      setError("请先填写要加入热词的文字。");
      return;
    }
    setHotwordBusyId(item.id);
    setError(null);
    setMessage(null);
    try {
      const result = await useSettings.getState().appendAsrHotword(
        hotword,
        () => assertTaskCurrentTarget(requestTaskId),
      );
      if (!isTaskCurrentTarget(requestTaskId)) return;
      if (!result.added) {
        setMessage(`热词“${hotword}”已存在，未重复添加。`);
        return;
      }
      setMessage(`已加入热词：“${hotword}”。`);
    } catch (e) {
      if (isTaskCurrentTarget(requestTaskId)) setError(`加入热词失败：${String(e)}`);
    } finally {
      if (isTaskCurrentTarget(requestTaskId)) setHotwordBusyId(null);
    }
  }

  const correctedPreview = humanCorrectedArtifact(task);

  return (
    <section className="rounded-sm border border-warn/35 bg-warn/5">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-warn/20 px-3 py-2">
        <div className="min-w-0 text-ui-sm">
          <div className="flex flex-wrap items-center gap-2 text-fg">
            <Warning size={13} className={items.length ? "text-warn" : "text-ok"} />
            <span>覆盖疑点复核</span>
            <span className={items.length ? "text-warn" : "text-ok"}>
              {items.length ? `${items.length} 处` : "未发现疑点"}
            </span>
            {dirty && <span className="text-warn">有未保存修改</span>}
            {coverageRatio != null && (
              <span className="text-fg-mute">覆盖率 {(coverageRatio * 100).toFixed(1)}%</span>
            )}
            {basis && <span className="font-mono text-xs text-fg-mute">{basis}</span>}
          </div>
          <div className="mt-0.5 text-fg-mute">
            播放会包含疑点前后各 2 秒；机器 raw transcript/evidence 只读，人工应用只创建或更新 corrected。
          </div>
        </div>
        <button
          className="btn-primary h-7 px-2 text-ui-sm"
          disabled={savingReview || applyingId !== null}
          onClick={() => void saveReview()}
        >
          {savingReview ? "保存中..." : "保存复核"}
        </button>
      </div>

      {reviewIsStale && (
        <div className="mx-3 mt-2 rounded-sm border border-warn/40 bg-warn/10 px-2 py-1.5 text-ui-sm text-warn">
          已保存复核已过期（{staleReasons.join("、")}）。仅保留当前 source IDs，全部重置为 pending；旧 saved-only 项不会写回当前复核。
        </div>
      )}
      {partitionInvalid && (
        <div className="mx-3 mt-2 rounded-sm border border-err/35 bg-err/10 px-2 py-1.5 text-ui-sm text-err">
          机器覆盖分区校验失败，以下疑点只能作为人工线索，不能视为完整证据。
        </div>
      )}
      {detailsTruncated && (
        <div className="mx-3 mt-2 rounded-sm border border-warn/35 bg-warn/10 px-2 py-1.5 text-ui-sm text-warn">
          机器失败原因或局部恢复详情已截断，疑点列表可能不完整。
        </div>
      )}
      {reportStatus && reportStatus !== "ok" && (
        <div className="mx-3 mt-2 text-ui-sm text-fg-mute">
          覆盖检查状态：{reportStatus}{reportReason ? `（${reportReason}）` : ""}
        </div>
      )}
      {(message || error) && (
        <div className={clsx(
          "mx-3 mt-2 rounded-sm border px-2 py-1.5 text-ui-sm",
          error ? "border-err/35 bg-err/10 text-err" : "border-ok/35 bg-ok/10 text-ok",
        )}>
          {error || message}
        </div>
      )}

      {items.length === 0 ? (
        <div className="px-3 py-3 text-ui-sm text-fg-mute">
          当前 speech coverage 报告没有 failed range、failure reason 或 local recovery detail。
        </div>
      ) : (
        <div className="divide-y divide-warn/15">
          {items.map((item) => {
            const statusMeta = COVERAGE_REVIEW_STATUS_META[item.status];
            const target = coverageReviewTarget(correctedPreview.segments, item);
            const canApply = item.status === "confirmed_missing" || item.status === "substitution";
            const requiresReplacement = canApply && target.kind === "replace";
            const substitutionWithoutTarget = item.status === "substitution" && target.kind === "insert";
            const applicationTextReady = requiresReplacement
              ? Boolean(item.replacementText.trim())
              : Boolean(item.heardText.trim());
            const itemBusy = savingReview || applyingId === item.id || hotwordBusyId === item.id;
            return (
              <div key={item.id} className="space-y-2 px-3 py-3 text-ui-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    className={clsx(
                      "btn-ghost h-7 px-2 text-ui-sm",
                      playingId === item.id && "border-accent/40 bg-accent/10 text-accent",
                    )}
                    disabled={itemBusy}
                    onClick={() => {
                      setError(null);
                      void onPlayRange(item.id, item.start, item.end).catch((e) => {
                        setError(`播放失败：${String(e)}`);
                      });
                    }}
                  >
                    {playingId === item.id ? "播放中..." : "播放前后 2 秒"}
                  </button>
                  <span className="font-mono text-fg">
                    {formatCoverageTime(item.start)} - {formatCoverageTime(item.end)}
                  </span>
                  <span className="font-mono text-xs text-fg-mute">{item.id}</span>
                  {!item.sourcePresent && (
                    <span className="rounded-sm border border-warn/35 bg-warn/10 px-1.5 py-0.5 text-xs text-warn">
                      旧复核项
                    </span>
                  )}
                  <select
                    className={clsx(
                      "ml-auto h-7 rounded-sm border px-2 text-ui-sm outline-none",
                      statusMeta.className,
                    )}
                    value={item.status}
                    disabled={itemBusy}
                    onChange={(event) => {
                      const status = event.target.value as CoverageReviewStatus;
                      const patch: CoverageReviewDraft = { status };
                      if (
                        (status === "substitution" || status === "confirmed_missing")
                        && target.kind === "replace"
                        && !item.replacementText.trim()
                      ) {
                        patch.replacementText = target.segment.text;
                      }
                      patchItem(item.id, patch);
                    }}
                  >
                    {item.status === "resolved" && (
                      <option value="resolved">{COVERAGE_REVIEW_STATUS_META.resolved.label}</option>
                    )}
                    {USER_SELECTABLE_COVERAGE_STATUSES.map((status) => (
                      <option key={status} value={status}>{COVERAGE_REVIEW_STATUS_META[status].label}</option>
                    ))}
                  </select>
                </div>

                {item.reasons.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 text-xs text-fg-mute">
                    {item.reasons.map((reason, index) => (
                      <span key={`${item.id}-reason-${index}`} className="rounded-sm border border-warn/25 bg-bg/60 px-1.5 py-0.5">
                        {coverageReasonText(reason)}
                      </span>
                    ))}
                  </div>
                )}

                {item.recoveryDetails.map((detail, index) => {
                  const attempts = Array.isArray(detail.attempts)
                    ? detail.attempts.map(recoveryAttemptText).filter(Boolean)
                    : [];
                  const localReference = stringValue(detail.local_reference);
                  return (
                    <details key={`${item.id}-recovery-${index}`} className="rounded-sm border border-border/60 bg-bg/50 px-2 py-1.5">
                      <summary className="cursor-pointer text-fg-dim">
                        {recoveryDecisionLabel(detail)}
                      </summary>
                      <div className="mt-1 space-y-1 text-xs text-fg-mute">
                        {localReference && <div>邻近正文：{localReference}</div>}
                        {attempts.map((attempt, attemptIndex) => (
                          <div key={`${item.id}-attempt-${attemptIndex}`}>{attempt}</div>
                        ))}
                      </div>
                    </details>
                  );
                })}

                <div className="grid gap-2 md:grid-cols-2">
                  <label className="space-y-1">
                    <span className="text-xs text-fg-mute">实际听到（无重叠漏字插入 / 加入热词）</span>
                    <textarea
                      className="input min-h-[56px] w-full resize-y py-1.5"
                      value={item.heardText}
                      disabled={itemBusy}
                      onChange={(event) => patchItem(item.id, { heardText: event.target.value })}
                      placeholder="填写实际听到的文字"
                    />
                  </label>
                  <label className="space-y-1">
                    <span className="text-xs text-fg-mute">备注</span>
                    <textarea
                      className="input min-h-[56px] w-full resize-y py-1.5"
                      value={item.note}
                      disabled={itemBusy}
                      onChange={(event) => patchItem(item.id, { note: event.target.value })}
                      placeholder="可记录判断依据、说话人或环境噪声"
                    />
                  </label>
                </div>

                {requiresReplacement && target.kind === "replace" && (
                  <label className="block space-y-1">
                    <span className="text-xs text-fg-mute">
                      纠正后的完整段落（当前目标：{target.segment.text}）
                    </span>
                    <textarea
                      className="input min-h-[72px] w-full resize-y py-1.5"
                      value={item.replacementText}
                      disabled={itemBusy}
                      onChange={(event) => patchItem(item.id, { replacementText: event.target.value })}
                      placeholder="必须填写完整段落，规范化字符数不得低于原段的 50%"
                    />
                  </label>
                )}
                {substitutionWithoutTarget && (
                  <div className="text-xs text-err">当前范围未与任何 corrected 段落重叠，错词替换不可应用。</div>
                )}

                <div className="flex flex-wrap items-center gap-2">
                  <button
                    className="btn-primary h-7 px-2 text-ui-sm"
                    disabled={
                      !canApply
                      || applyingId !== null
                      || savingReview
                      || hotwordBusyId !== null
                      || !applicationTextReady
                      || substitutionWithoutTarget
                    }
                    onClick={() => void applyToTranscript(item)}
                    title={canApply
                      ? requiresReplacement
                        ? "只替换 corrected 中重叠最大的目标段，raw 保持只读"
                        : "仅在 corrected 中无任何重叠时插入人工 segment，raw 保持只读"
                      : "先把状态改为确认漏字或错词替换"}
                  >
                    {applyingId === item.id ? "应用中..." : "应用到人工校对稿"}
                  </button>
                  <button
                    className="btn-ghost h-7 px-2 text-ui-sm"
                    disabled={itemBusy || applyingId !== null || hotwordBusyId !== null || !item.heardText.trim()}
                    onClick={() => void addHotword(item)}
                  >
                    {hotwordBusyId === item.id ? "添加中..." : "加入热词"}
                  </button>
                  <span className={clsx("rounded-sm border px-1.5 py-0.5 text-xs", statusMeta.className)}>
                    {statusMeta.label}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

type LocalVoiceprintAnchor = VoiceprintAnchor & { id: string };
type VoiceprintComparison = {
  changedSegments: number;
  matchedSegments: number;
  reviewSegments: number;
  profileCount: number;
  rejectedAnchors: number;
  rejectedProfiles: number;
  speakersBefore: number;
  speakersAfter: number;
};
type VoiceprintAnchorPreflight = {
  taskId: string;
  sourceFingerprint: string;
  candidates: VoiceprintAnchorPreflightCandidate[];
  reason: string;
};

function voiceprintAnchorSeconds(
  anchors: Array<Pick<VoiceprintAnchor, "speaker" | "start" | "end">>,
  speaker: string,
): number {
  const intervals = anchors
    .filter((anchor) => normalizeSpeakerName(anchor.speaker) === speaker && anchor.end > anchor.start)
    .map((anchor) => [anchor.start, anchor.end] as const)
    .sort((left, right) => left[0] - right[0] || left[1] - right[1]);
  let seconds = 0;
  let currentStart: number | null = null;
  let currentEnd = 0;
  for (const [start, end] of intervals) {
    if (currentStart == null) {
      currentStart = start;
      currentEnd = end;
    } else if (start <= currentEnd) {
      currentEnd = Math.max(currentEnd, end);
    } else {
      seconds += currentEnd - currentStart;
      currentStart = start;
      currentEnd = end;
    }
  }
  if (currentStart != null) seconds += currentEnd - currentStart;
  return seconds;
}

function recommendedVoiceprintAnchors<T extends VoiceprintAnchor & { index: number; duration: number }>(
  candidates: T[],
  existing: LocalVoiceprintAnchor[],
  speaker: string,
): T[] {
  const existingKeys = new Set(
    existing.map((anchor) => `${normalizeSpeakerName(anchor.speaker)}:${anchor.index ?? ""}:${anchor.start.toFixed(2)}`),
  );
  const picked: T[] = [];
  let seconds = voiceprintAnchorSeconds(existing, speaker);
  for (const candidate of candidates) {
    const key = `${candidate.speaker}:${candidate.index}:${candidate.start.toFixed(2)}`;
    if (existingKeys.has(key)) continue;
    picked.push(candidate);
    seconds = voiceprintAnchorSeconds([...existing, ...picked], speaker);
    if (seconds >= 10 || picked.length >= 5) break;
  }
  return picked;
}

function voiceprintReidentifyErrorMessage(error: unknown): string {
  const raw = String(error).replace(/^Error:\s*/i, "");
  if (raw.includes("mixed_voice_anchor") || raw.includes("mixed_voice_profile")) {
    return "所选片段的段内声纹不一致，可能包含两个人，请试听后换一个更纯净的片段";
  }
  if (raw.includes("overlapping_speech")) {
    return "所选片段检测到重叠说话，请换一个没有抢话或插话的片段";
  }
  if (raw.includes("too_little_speech") || raw.includes("too_short_enrollment")) {
    return "所选片段的有效单人语音不足，请增加同一人的清晰片段";
  }
  return raw;
}

function VoiceprintReidentifyPanel({
  task,
  segments,
  activeSegment,
  activeSegmentIndex,
  speakers,
  engine,
  onApply,
  comparison,
  onUndo,
}: {
  task: Task;
  segments: Segment[];
  activeSegment: Segment | null;
  activeSegmentIndex: number | null;
  speakers: string[];
  engine: string;
  onApply: (
    response: VoiceprintReidentifyResponse,
    saveProfiles: boolean,
    sourceFingerprint: string,
  ) => Promise<void>;
  comparison: VoiceprintComparison | null;
  onUndo?: () => Promise<void>;
}) {
  const [anchors, setAnchors] = useState<LocalVoiceprintAnchor[]>([]);
  const [preflight, setPreflight] = useState<VoiceprintAnchorPreflight | null>(null);
  const [draftSpeaker, setDraftSpeaker] = useState("");
  const [saveProfiles, setSaveProfiles] = useState(false);
  const [busy, setBusy] = useState(false);
  const [preflighting, setPreflighting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const speakersKey = speakers.join("|");
  const segmentsFingerprint = useMemo(() => segmentRevisionFingerprint(segments), [segments]);
  const speakerChoices = useMemo(() => {
    const choices: string[] = [];
    for (const speaker of speakers) {
      const normalized = normalizeSpeakerName(speaker);
      if (normalized && isValidSpeakerAnchorName(normalized) && !choices.includes(normalized)) choices.push(normalized);
    }
    return choices;
  }, [speakersKey]);
  const preflightCandidatesBySpeaker = useMemo(() => {
    const groups = new Map<string, VoiceprintAnchorPreflightCandidate[]>();
    if (!preflight) return groups;
    for (const candidate of preflight.candidates) {
      const speaker = normalizeSpeakerName(candidate.speaker);
      if (!speaker) continue;
      const group = groups.get(speaker) || [];
      group.push(candidate);
      groups.set(speaker, group);
    }
    return groups;
  }, [preflight]);

  useEffect(() => {
    const next = normalizeSpeakerName(activeSegment?.speaker || speakerChoices[0] || "");
    if (next) setDraftSpeaker(next);
  }, [activeSegmentIndex, activeSegment?.speaker, speakerChoices]);

  useEffect(() => {
    setPreflight(null);
    setMessage(null);
    setError(null);
  }, [task.id, segmentsFingerprint]);

  if (!segments.length || !speakers.length) return null;

  const hasActive = activeSegment && activeSegmentIndex != null;
  const cleanSpeaker = normalizeSpeakerName(draftSpeaker);
  const genericSpeaker = /^SPEAKER_.+$/i.test(cleanSpeaker);
  const selectedAnchorSeconds = voiceprintAnchorSeconds(anchors, cleanSpeaker);
  const preflightFresh = preflight?.taskId === task.id
    && preflight.sourceFingerprint === resultRevisionFingerprint(task.result!);
  const selectedCandidates = preflightFresh && cleanSpeaker
    ? (preflightCandidatesBySpeaker.get(cleanSpeaker) || [])
    : [];
  const recommendedCandidates = cleanSpeaker && preflightFresh
    ? recommendedVoiceprintAnchors(selectedCandidates, anchors, cleanSpeaker)
    : [];

  async function preflightAnchors() {
    const requestTaskId = task.id;
    const sourceFingerprint = resultRevisionFingerprint(task.result!);
    setPreflighting(true);
    setError(null);
    setMessage(null);
    try {
      const response = await ipc.preflightVoiceprintAnchors({
        audio: task.result?.audio || task.audio,
        segments,
        engine,
      });
      assertTaskResultRevision(requestTaskId, sourceFingerprint);
      if (!isTaskCurrentTarget(requestTaskId)) return;
      setPreflight({
        taskId: requestTaskId,
        sourceFingerprint,
        candidates: response.candidates,
        reason: response.stats.reason,
      });
      setMessage(response.stats.reason);
    } catch (e) {
      if (isTaskCurrentTarget(requestTaskId)) setError(voiceprintReidentifyErrorMessage(e));
    } finally {
      if (isTaskCurrentTarget(requestTaskId)) setPreflighting(false);
    }
  }

  function addActiveAnchor() {
    if (!activeSegment || activeSegmentIndex == null) return;
    const speaker = cleanSpeaker || normalizeSpeakerName(activeSegment.speaker || "");
    if (!isValidSpeakerAnchorName(speaker)) {
      setError("每个声纹锚点只能对应一个说话人，请填 B/C/D 或 SPEAKER_B/C/D，不要填 b/c");
      return;
    }
    if (!speaker) {
      setError("请先填写这个锚点对应的说话人");
      return;
    }
    if (!preflightFresh) {
      setError("请先点击“分析可用锚点”，系统会用 CAM++ 预检当前段是否可用于声纹回扫");
      return;
    }
    const activeCandidate = preflight?.candidates.find((candidate) => candidate.index === activeSegmentIndex);
    if (!activeCandidate) {
      setError("当前段未通过 CAM++ 声纹质量门禁，请选择分析结果中的可用片段");
      return;
    }
    const conflictingAnchor = anchors.find((anchor) =>
      normalizeSpeakerName(anchor.speaker) !== speaker
      && Math.min(anchor.end, activeSegment.end) - Math.max(anchor.start, activeSegment.start) > 0.05,
    );
    if (conflictingAnchor) {
      setError(`当前片段已分配给 ${normalizeSpeakerName(conflictingAnchor.speaker)}，不能同时注册给 ${speaker}`);
      return;
    }
    const exists = anchors.some((anchor) =>
      anchor.index === activeSegmentIndex
      && Math.abs(anchor.start - activeSegment.start) < 0.05,
    );
    if (exists) {
      setError("当前片段已经加入声纹锚点");
      return;
    }
    setError(null);
    setMessage(null);
    setAnchors((prev) => [
      ...prev,
      {
        id: `${Date.now()}-${activeSegmentIndex}`,
        index: activeSegmentIndex,
        speaker,
        start: activeSegment.start,
        end: activeSegment.end,
        text: activeSegment.text,
      },
    ]);
  }

  function addCandidateAnchor(candidate: VoiceprintAnchorPreflightCandidate) {
    const conflictingAnchor = anchors.find((anchor) =>
      normalizeSpeakerName(anchor.speaker) !== candidate.speaker
      && Math.min(anchor.end, candidate.end) - Math.max(anchor.start, candidate.start) > 0.05,
    );
    if (conflictingAnchor) {
      setError(`推荐片段已分配给 ${normalizeSpeakerName(conflictingAnchor.speaker)}，不能同时注册给 ${candidate.speaker}`);
      return;
    }
    const exists = anchors.some((anchor) =>
      anchor.index === candidate.index
      && Math.abs(anchor.start - candidate.start) < 0.05,
    );
    if (exists) {
      setError("该推荐片段已经加入声纹锚点");
      return;
    }
    setError(null);
    setMessage(null);
    setAnchors((prev) => [
      ...prev,
      {
        id: `rec-${Date.now()}-${candidate.index}`,
        index: candidate.index,
        speaker: candidate.speaker,
        start: candidate.start,
        end: candidate.end,
        text: candidate.text,
      },
    ]);
  }

  async function runReidentify(anchorsForRun: LocalVoiceprintAnchor[]) {
    const requestTaskId = task.id;
    const sourceFingerprint = resultRevisionFingerprint(task.result!);
    const normalizedAnchors = anchorsForRun
      .map((anchor) => ({ ...anchor, speaker: normalizeSpeakerName(anchor.speaker) }))
      .filter((anchor) => isValidSpeakerAnchorName(anchor.speaker));
    if (!normalizedAnchors.length) {
      setError("至少先加入 1 个确认无误的说话人片段");
      return;
    }
    if (normalizedAnchors.length !== anchors.length) {
      setError("锚点里有 b/c 这类无效标签，请清空后按 B、C、D 分别加入");
      return;
    }
    if (saveProfiles) {
      const genericLabels = Array.from(new Set(
        normalizedAnchors
          .map((anchor) => anchor.speaker)
          .filter((speaker) => /^SPEAKER_.+$/i.test(speaker)),
      ));
      if (genericLabels.length > 0) {
        setError(`保存到长期声纹库前，请先把 ${genericLabels.join("、")} 改成真实姓名`);
        return;
      }
    }
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const response = await ipc.reidentifySpeakers({
        audio: task.result?.audio || task.audio,
        segments,
        anchors: normalizedAnchors,
        engine,
        threshold: 0.78,
        review_threshold: 0.70,
        margin: 0.05,
        require_enrollment_quality: saveProfiles,
      });
      assertTaskResultRevision(requestTaskId, sourceFingerprint);
      await onApply(response, saveProfiles, sourceFingerprint);
      if (!isTaskCurrentTarget(requestTaskId)) return;
      setMessage(response.stats.reason || `已按声纹锚点回扫，改派 ${response.stats.changed_segments} 段`);
    } catch (e) {
      if (isTaskCurrentTarget(requestTaskId)) setError(voiceprintReidentifyErrorMessage(e));
    } finally {
      if (isTaskCurrentTarget(requestTaskId)) setBusy(false);
    }
  }

  async function reidentifyWithRecommendedAnchors() {
    if (!isValidSpeakerAnchorName(cleanSpeaker)) {
      setError("请先选择一个有效说话人");
      return;
    }
    if (!preflightFresh) {
      setError("请先点击“分析可用锚点”");
      return;
    }
    if (!recommendedCandidates.length) {
      setError("当前说话人没有通过 CAM++ 质量门禁的候选，请选择其他说话人或重新分析");
      return;
    }

    const existingKeys = new Set(
      anchors.map((anchor) => `${normalizeSpeakerName(anchor.speaker)}:${anchor.index ?? ""}:${anchor.start.toFixed(2)}`),
    );
    const nextAnchors = [
      ...anchors,
      ...recommendedCandidates
        .filter((candidate) => !existingKeys.has(`${cleanSpeaker}:${candidate.index}:${candidate.start.toFixed(2)}`))
        .map((candidate) => ({
          id: `rec-${Date.now()}-${candidate.index}`,
          index: candidate.index,
          speaker: cleanSpeaker,
          start: candidate.start,
          end: candidate.end,
          text: candidate.text,
        })),
    ];
    setAnchors(nextAnchors);
    await runReidentify(nextAnchors);
  }

  async function undoReidentify() {
    if (!onUndo) return;
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      await onUndo();
      setMessage("已撤销本次声纹回扫");
    } catch (e) {
      setError(voiceprintReidentifyErrorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="rounded-sm border border-border bg-bg-panel/70">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/60 px-3 py-2">
        <div className="min-w-0 text-ui-sm">
          <div className="flex items-center gap-2 text-fg">
            <Mic size={13} className="text-accent" />
            <span>声纹重识别</span>
            <span className="text-fg-mute">{anchors.length} 个锚点</span>
          </div>
          <div className="mt-0.5 text-fg-mute">
            确认干净片段后回扫全文；累计 10 秒以上才可入库，只改高置信匹配，接近但不确定的标待确认。
          </div>
        </div>
        <button
          onClick={() => void runReidentify(anchors)}
          disabled={busy || anchors.length === 0}
          className="btn-primary flex items-center gap-1.5 text-ui-sm"
          title="不重新转录，只用已选声纹锚点重新识别当前全文说话人"
        >
          <Refresh size={12} className={busy ? "animate-spin" : ""} />
          {busy ? "回扫中..." : "重新识别全文"}
        </button>
      </div>
      <div className="space-y-2 px-3 py-2">
        {comparison && (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-border/60 pb-2 text-ui-sm">
            <span className="text-fg">
              改派 {comparison.changedSegments} 段
            </span>
            <span className="text-fg-dim">
              高置信命中 {comparison.matchedSegments} 段
            </span>
            <span className={comparison.reviewSegments > 0 ? "text-warn" : "text-fg-dim"}>
              待确认 {comparison.reviewSegments} 段
            </span>
            <span className="text-fg-mute">
              说话人 {comparison.speakersBefore} → {comparison.speakersAfter} · 声纹 {comparison.profileCount} 人
            </span>
            {(comparison.rejectedAnchors > 0 || comparison.rejectedProfiles > 0) && (
              <span className="text-warn">
                质量闸拒绝 {comparison.rejectedAnchors + comparison.rejectedProfiles} 项
              </span>
            )}
            {onUndo && (
              <button
                className="btn-ghost ml-auto flex h-7 items-center gap-1 px-2 text-ui-sm"
                disabled={busy}
                onClick={() => void undoReidentify()}
                title="恢复声纹回扫前的说话人标签"
              >
                <Refresh size={12} />
                撤销本次
              </button>
            )}
          </div>
        )}
        <div className="flex flex-wrap items-center gap-2 text-ui-sm">
          <span className="text-fg-mute">当前段</span>
          <span className="font-mono text-fg-mute">
            {hasActive ? formatTimeShort(activeSegment!.start) : "--:--"}
          </span>
          {hasActive && (
            <span className="max-w-[42rem] truncate text-fg-dim">
              {activeSegment!.text}
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <select
            className="h-7 rounded-sm border border-border bg-bg px-2 text-ui-sm text-fg"
            value={draftSpeaker}
            onChange={(event) => setDraftSpeaker(event.target.value)}
          >
            {draftSpeaker && isValidSpeakerAnchorName(draftSpeaker) && !speakerChoices.includes(normalizeSpeakerName(draftSpeaker)) && (
              <option value={draftSpeaker}>{normalizeSpeakerName(draftSpeaker)}</option>
            )}
            {speakerChoices.map((speaker) => (
              <option key={speaker} value={speaker}>{speaker}</option>
            ))}
          </select>
          <input
            className="input h-7 w-36"
            value={draftSpeaker}
            onChange={(event) => setDraftSpeaker(event.target.value)}
            placeholder="说话人/姓名"
          />
          <button
            className="btn-ghost h-7 px-2 text-ui-sm"
            disabled={!hasActive || !preflightFresh}
            onClick={addActiveAnchor}
            title="仅允许加入已通过 CAM++ 声纹质量预检的当前段"
          >
            加入当前段
          </button>
          <button
            className="btn-ghost flex h-7 items-center gap-1 px-2 text-ui-sm"
            disabled={preflighting || busy}
            onClick={() => void preflightAnchors()}
            title="只分析当前录音的声纹锚点质量，不会改动文字、时间轴、光标或说话人标签"
          >
            <Mic size={12} className={preflighting ? "animate-pulse" : ""} />
            {preflighting ? "分析锚点中..." : "分析可用锚点"}
          </button>
          <button
            className="btn-primary h-7 px-2 text-ui-sm"
            disabled={!recommendedCandidates.length || busy || !preflightFresh}
            onClick={() => void reidentifyWithRecommendedAnchors()}
            title="自动加入当前说话人的合格推荐锚点，并立即安全回扫全文"
          >
            用推荐锚点回扫
          </button>
          <label className="inline-flex items-center gap-1.5 text-ui-sm text-fg-mute">
            <input
              type="checkbox"
              checked={saveProfiles}
              onChange={(event) => setSaveProfiles(event.target.checked)}
            />
            保存人工锚点到声纹库
          </label>
          {genericSpeaker && saveProfiles && (
            <span className="text-xs text-err">
              {cleanSpeaker} 是系统临时标签，不能保存到长期声纹库
            </span>
          )}
          {cleanSpeaker && (
            <span className={clsx("text-xs", selectedAnchorSeconds >= 10 ? "text-ok" : "text-fg-mute")}>
              {cleanSpeaker} 已选 {selectedAnchorSeconds.toFixed(1)}s / 10s
            </span>
          )}
        </div>
        {preflightFresh && selectedCandidates.length > 0 && (
          <div className="space-y-1 rounded-sm border border-border/60 bg-bg/60 px-2 py-1.5">
            <div className="flex flex-wrap items-center gap-2 text-xs text-fg-mute">
              <span>已通过声纹质量门禁</span>
              <span>{cleanSpeaker}</span>
              <span>{selectedCandidates.length} 个候选</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {selectedCandidates.slice(0, 10).map((candidate) => (
                <button
                  key={`${candidate.speaker}-${candidate.index}-${candidate.start}`}
                  className="rounded-sm border border-border bg-bg px-1.5 py-0.5 text-xs text-fg-mute hover:border-accent/40 hover:text-accent"
                  onClick={() => addCandidateAnchor(candidate)}
                  title={`${candidate.reason}\n段内相似度 ${candidate.quality.median_similarity?.toFixed(2) ?? "单窗"}`}
                >
                  {formatTimeShort(candidate.start)} · {candidate.duration.toFixed(1)}s
                </button>
              ))}
            </div>
          </div>
        )}
        {anchors.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {anchors.map((anchor) => (
              <button
                key={anchor.id}
                className="rounded-sm border border-accent/30 bg-accent/10 px-1.5 py-0.5 text-xs text-accent hover:bg-accent/20"
                onClick={() => setAnchors((prev) => prev.filter((item) => item.id !== anchor.id))}
                title="点击移除这个锚点"
              >
                {normalizeSpeakerName(anchor.speaker)} · {formatTimeShort(anchor.start)}
              </button>
            ))}
            <button
              className="rounded-sm border border-border bg-bg px-1.5 py-0.5 text-xs text-fg-mute hover:text-fg"
              onClick={() => setAnchors([])}
            >
              清空
            </button>
          </div>
        )}
        {(message || error) && (
          <div className={clsx(
            "rounded-sm border px-2 py-1 text-ui-sm",
            error ? "border-err/30 bg-err/10 text-err" : "border-ok/30 bg-ok/10 text-ok",
          )}>
            {error || message}
          </div>
        )}
      </div>
    </section>
  );
}

function DiarizationRecommendation({
  recommendation,
  applying,
  currentSpeakers,
  onApply,
}: {
  recommendation: RecommendDiarizationResponse;
  applying: number | null;
  currentSpeakers: number;
  onApply: (candidate: DiarizationCandidate, persistCount?: boolean) => Promise<void> | void;
}) {
  const recommended = recommendation.recommended_n_speakers;
  const recommendedCandidateN = recommendation.recommended_candidate_n_speakers ?? recommended;
  const sorted = [...recommendation.candidates].sort((a, b) => a.n_speakers - b.n_speakers);
  const recommendedCandidate = sorted.find((c) => c.n_speakers === recommendedCandidateN);
  const confidence = recommendation.confidence ?? "medium";
  const canPersistRecommended = confidence !== "low";
  const confidenceClass = confidence === "high"
    ? "border-ok/35 bg-ok/10 text-ok"
    : confidence === "low"
      ? "border-warn/35 bg-warn/10 text-warn"
      : "border-yellow-300/35 bg-yellow-500/10 text-yellow-300";
  const confidenceLabel = confidence === "high" ? "高置信" : confidence === "low" ? "低置信" : "中置信";
  if (!sorted.length) return null;

  return (
    <div className="rounded-sm border border-border bg-bg-panel/70">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/60 px-3 py-2">
        <div className="min-w-0 text-ui-sm text-fg">
          <div className="flex flex-wrap items-center gap-2">
            <span>
              推荐 <span className="text-accent font-medium">{recommended || "?"}</span> 人
              {recommendedCandidateN !== recommended && (
                <span className="ml-1 text-fg-mute">
                  （由 {recommendedCandidateN} 人候选生成）
                </span>
              )}
            </span>
            <span
              className={clsx("rounded-sm border px-1.5 py-0.5 text-xs", confidenceClass)}
              title={recommendation.confidence_reason}
            >
              {confidenceLabel}
            </span>
          </div>
          <div className="mt-1 text-fg-mute">
            {recommendation.reassignment_reason || recommendation.merge_reason || recommendation.reason}
            {recommendation.confidence_reason && (
              <span className="ml-2">{recommendation.confidence_reason}</span>
            )}
            <span className="ml-2">候选按 2-8 人评估，可手动应用任一结果。</span>
          </div>
          {recommendation.reassignment_distribution && Object.keys(recommendation.reassignment_distribution).length > 0 && (
            <div className="mt-1 text-xs text-fg-mute">
              纠偏: {formatSpeakerDistribution(recommendation.reassignment_distribution)}
            </div>
          )}
          {recommendation.merge_map && Object.keys(recommendation.merge_map).length > 0 && (
            <div className="mt-1 text-xs text-fg-mute">
              合并: {Object.entries(recommendation.merge_map)
                .map(([from, to]) => `${from.replace("SPEAKER_", "")}->${to.replace("SPEAKER_", "")}`)
                .join(" / ")}
            </div>
          )}
          {recommendation.review_segments && recommendation.review_segments.length > 0 && (
            <div className="mt-1 text-xs text-fg-mute">
              抽听: {recommendation.review_segments.slice(0, 5)
                .map(reviewSegmentLabel)
                .join(" / ")}
            </div>
          )}
        </div>
        {recommendedCandidate && (
          <button
            onClick={() => onApply(recommendedCandidate, canPersistRecommended)}
            disabled={applying !== null}
            className="btn-primary text-ui-sm"
            title={canPersistRecommended ? "应用推荐结果，并把实际推荐人数保存为后续默认分人数量" : "低置信推荐只应用本次结果，不保存为默认人数"}
          >
            {applying === recommendedCandidate.n_speakers
              ? "应用中…"
              : canPersistRecommended
                ? "应用推荐并设为默认"
                : "应用推荐"}
          </button>
        )}
      </div>
      <div className="divide-y divide-border/50">
        {sorted.map((candidate) => {
          const actualN = candidate.actual_n_speakers ?? candidate.speakers.length;
          const isRecommended = candidate.n_speakers === recommendedCandidateN;
          const isCurrent = actualN === currentSpeakers;
          const distribution = candidate.summary.speakers
            .map((s) => `${s.speaker.replace("SPEAKER_", "")} ${s.segments}`)
            .join(" / ");
          const mergeTitle = candidate.merge_map && Object.keys(candidate.merge_map).length
            ? `merge=${JSON.stringify(candidate.merge_map)}`
            : "";
          const reassignmentTitle = candidate.reassignment_distribution && Object.keys(candidate.reassignment_distribution).length
            ? `reassign=${JSON.stringify(candidate.reassignment_distribution)}`
            : "";
          const postprocessTitle = candidate.postprocess_skipped_reason || "";
          const reviewTitle = candidate.review_segments?.length
            ? `review=${candidate.review_segments
              .slice(0, 8)
              .map(reviewSegmentLabel)
              .join("; ")}`
            : "";
          return (
            <div
              key={candidate.n_speakers}
              className={clsx(
                "flex flex-wrap items-center gap-2 px-3 py-2 text-ui-sm",
                isRecommended && "bg-accent/10",
              )}
            >
              <div className="w-16 shrink-0 text-fg">
                {actualN} 人
                {actualN !== candidate.n_speakers && (
                  <span className="ml-1 text-fg-mute">({candidate.n_speakers})</span>
                )}
              </div>
              <div className="min-w-[180px] flex-1 text-fg-mute">
                {distribution}
              </div>
              {isCurrent && (
                <div className="shrink-0 rounded-sm border border-ok/35 bg-ok/10 px-1.5 py-0.5 text-xs text-ok">
                  当前
                </div>
              )}
              <div
                className={clsx(
                  "shrink-0 rounded-sm border px-1.5 py-0.5 text-xs",
                  candidate.tiny_speakers > 0
                    ? "border-warn/35 bg-warn/10 text-warn"
                    : candidate.weak_speakers > 0
                      || (candidate.fragmented_speakers ?? 0) > 0
                      || (candidate.marginal_speakers ?? 0) > 0
                      ? "border-yellow-300/35 bg-yellow-500/10 text-yellow-300"
                    : "border-border bg-bg text-fg-mute",
                )}
                title={`score=${candidate.score}; stable=${candidate.stable_speakers}; weak=${candidate.weak_speakers}; tiny=${candidate.tiny_speakers}; fragmented=${candidate.fragmented_speakers ?? 0}; marginal=${candidate.marginal_speakers ?? 0}; ${postprocessTitle}; ${reassignmentTitle}; ${mergeTitle}; ${reviewTitle}`}
              >
                {candidate.reason}
              </div>
              <button
                onClick={() => onApply(candidate, false)}
                disabled={applying !== null}
                className={clsx("btn-ghost text-ui-sm", isRecommended && "text-accent")}
                title={`应用 ${actualN} 人分离结果`}
              >
                {applying === candidate.n_speakers ? "应用中…" : isRecommended ? "应用推荐" : "应用"}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function entityKindLabel(kind?: string): string {
  if (kind === "phonetic_entity") return "同音实体";
  if (kind === "entity_drift") return "实体漂移";
  if (kind === "orthographic_term") return "字形相近";
  return kind || "候选";
}

function canGlobalUnify(candidate: TermConsistencyCandidate): boolean {
  return candidate.kind === "phonetic_entity" || candidate.kind === "orthographic_term" || !candidate.kind;
}

function EntityConsistencyPanel({
  result,
  onApplyUnify,
  onApplyOccurrence,
}: {
  result: TranscribeResult;
  onApplyUnify?: (candidate: TermConsistencyCandidate, canonical: string) => Promise<{ replacementCount: number; touchedCount: number }>;
  onApplyOccurrence?: (
    candidate: TermConsistencyCandidate,
    contextIndex: number,
    fromText: string,
    toText: string,
  ) => Promise<{ replacementCount: number; touchedCount: number }>;
}) {
  const candidates = result.asr_quality?.term_consistency?.candidates ?? [];
  const visible = candidates.filter((candidate) => (candidate.terms ?? []).length >= 2);
  const [expanded, setExpanded] = useState(false);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [busyId, setBusyId] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!visible.length) return null;

  async function handleUnify(candidate: TermConsistencyCandidate) {
    if (!onApplyUnify) return;
    const canonical = (drafts[candidate.id] || candidate.suggested_canonical || candidate.terms?.[0] || "").trim();
    setBusyId(candidate.id);
    setError(null);
    setMessage(null);
    try {
      const result = await onApplyUnify(candidate, canonical);
      if (result.replacementCount <= 0) {
        setMessage("没有找到需要替换的文本。");
      } else {
        setMessage(`已统一 ${result.replacementCount} 处，影响 ${result.touchedCount} 段。`);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setBusyId(null);
    }
  }

  async function handleOccurrence(candidate: TermConsistencyCandidate, contextIndex: number, fromText: string) {
    if (!onApplyOccurrence) return;
    const toText = (drafts[candidate.id] || "").trim();
    if (!toText) {
      setError("请先填写目标写法");
      return;
    }
    setBusyId(`${candidate.id}:${contextIndex}`);
    setError(null);
    setMessage(null);
    try {
      const result = await onApplyOccurrence(candidate, contextIndex, fromText, toText);
      if (result.replacementCount <= 0) {
        setMessage("该段没有找到要替换的词。");
      } else {
        setMessage(`已替换 ${result.replacementCount} 处，影响 ${result.touchedCount} 段。`);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <section className="rounded-sm border border-border bg-bg-panel/70">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/60 px-3 py-2">
        <div className="min-w-0 text-ui-sm">
          <div className="flex items-center gap-2 text-fg">
            <Warning size={13} className="text-warn" />
            <span>实体核对</span>
            <span className="text-fg-mute">{visible.length} 组候选</span>
          </div>
          <div className="mt-0.5 text-fg-mute">
            确认后才会统一文本；实体漂移只按单段替换，不做全局替换。
          </div>
        </div>
        <button
          onClick={() => setExpanded((value) => !value)}
          className="btn-ghost h-6 px-2 text-ui-sm"
        >
          {expanded ? "收起" : "查看并处理"}
        </button>
      </div>
      {(message || error) && (
        <div className={clsx(
          "mx-3 mt-2 rounded-sm border px-2 py-1 text-ui-sm",
          error ? "border-err/30 bg-err/10 text-err" : "border-ok/30 bg-ok/10 text-ok",
        )}>
          {error || message}
        </div>
      )}
      {expanded && (
        <div className="divide-y divide-border/50">
          {visible.map((candidate) => {
            const globalUnify = canGlobalUnify(candidate);
            const canonical = drafts[candidate.id] ?? candidate.suggested_canonical ?? candidate.terms?.[0] ?? "";
            const terms = (candidate.terms ?? []).join("、");
            const contexts = (candidate.contexts ?? []).slice(0, 4);
            return (
              <div key={candidate.id} className="px-3 py-2 text-ui-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <span className={clsx(
                    "rounded-sm border px-1.5 py-0.5 text-xs",
                    candidate.kind === "entity_drift"
                      ? "border-warn/35 bg-warn/10 text-warn"
                      : "border-accent/30 bg-accent/10 text-accent",
                  )}>
                    {entityKindLabel(candidate.kind)}
                  </span>
                  <span className="font-mono text-fg-mute">{candidate.phonetic_key || "-"}</span>
                  <span className="min-w-0 flex-1 text-fg break-words">{terms}</span>
                  <span className="text-fg-mute">{candidate.total_count} 次</span>
                </div>
                <div className="mt-1 text-fg-mute">{candidate.reason}</div>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <input
                    className="input h-6 w-36"
                    value={canonical}
                    onChange={(e) => setDrafts((prev) => ({ ...prev, [candidate.id]: e.target.value }))}
                    placeholder={globalUnify ? "标准写法" : "目标写法"}
                  />
                  {globalUnify ? (
                    <button
                      className="btn-primary h-6 px-2 text-ui-sm"
                      disabled={busyId !== null || !onApplyUnify}
                      onClick={() => handleUnify(candidate)}
                      title="只替换本候选组中的写法"
                    >
                      {busyId === candidate.id ? "应用中..." : "应用本组"}
                    </button>
                  ) : (
                    <span className="text-xs text-fg-mute">选择下面具体段落应用</span>
                  )}
                </div>
                {!!contexts.length && (
                  <div className="mt-2 space-y-1">
                    {contexts.map((context, idx) => {
                      const termInContext = (candidate.terms ?? []).find((term) => context.text.includes(term)) ?? "";
                      return (
                        <div key={`${context.index}-${idx}`} className="flex items-start gap-2 rounded-sm bg-bg/50 px-2 py-1">
                          <span className="w-12 shrink-0 text-right font-mono text-xs text-fg-mute">
                            {formatTimeShort(context.start)}
                          </span>
                          <span className="min-w-0 flex-1 break-words text-fg-mute">
                            {context.text}
                          </span>
                          {!globalUnify && (
                            <button
                              className="btn-ghost h-6 px-2 text-ui-sm"
                              disabled={busyId !== null || !onApplyOccurrence || !termInContext}
                              onClick={() => handleOccurrence(candidate, context.index, termInContext)}
                              title={termInContext ? `仅替换第 ${context.index} 段里的“${termInContext}”` : "本段没有可替换候选词"}
                            >
                              {busyId === `${candidate.id}:${context.index}` ? "应用中..." : "应用此段"}
                            </button>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

function RawSegments({ segments, viewMode, activeSegmentIndex, playbackTime, autoFollow, onSeekSegment, onRenameSpeaker }: {
  segments: Segment[];
  viewMode: ViewMode;
  activeSegmentIndex?: number | null;
  playbackTime?: number;
  autoFollow?: boolean;
  onSeekSegment?: (segment: Segment) => void;
  onRenameSpeaker?: (oldName: string, newName: string) => void;
}) {
  const speakers = collectSpeakers(segments);
  const hasSpeakers = speakers.length > 0;
  const itemRefs = useRef<Record<number, HTMLLIElement | null>>({});
  const dialogSegments = useMemo(
    () => (viewMode === "dialog" && hasSpeakers ? speakerDisplaySegments(segments) : []),
    [hasSpeakers, segments, viewMode],
  );
  const turns = useMemo(
    () => (viewMode === "dialog" && hasSpeakers ? groupBySpeakerTurns(dialogSegments) : []),
    [dialogSegments, hasSpeakers, viewMode],
  );
  const activeDisplaySegment = activeSegmentIndex == null
    ? null
    : dialogSegments.find((segment) => (
      segment.sourceSegmentIndex === activeSegmentIndex
      && (
        playbackTime == null
        || (playbackTime >= segment.start - 0.03 && playbackTime <= segment.end + 0.18)
      )
    )) ?? dialogSegments.find((segment) => segment.sourceSegmentIndex === activeSegmentIndex) ?? null;
  const activeTurnIndex = activeDisplaySegment == null
    ? null
    : turns.findIndex((turn) => turn.segments.includes(activeDisplaySegment));

  useEffect(() => {
    if (!autoFollow) return;
    const targetIndex = viewMode === "dialog" && hasSpeakers ? activeTurnIndex : activeSegmentIndex;
    if (targetIndex == null || targetIndex < 0) return;
    itemRefs.current[targetIndex]?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [activeSegmentIndex, activeTurnIndex, autoFollow, hasSpeakers, viewMode]);

  const renderCueText = (segment: Segment, segmentIndex: number) => {
    const cues = segmentCues(segment);
    const isSegmentActive = segmentIndex === activeSegmentIndex && playbackTime != null;
    const hasSyncCues = cues.some((cue) => cue.source === "sync");
    const activeCueIndex = isSegmentActive ? activeCueIndexAt(segment, playbackTime ?? 0) : null;
    return cues.map((cue, cueIndex) => {
      const active = activeCueIndex === cueIndex;
      return (
        <button
          key={`${segmentIndex}-${cueIndex}-${cue.start}`}
          type="button"
          data-cue-index={cueIndex}
          data-cue-source={cue.source}
          data-cue-reliable={cue.reliable ? "true" : "false"}
          data-cue-start={cue.start}
          data-cue-end={cue.end}
          onClick={(event) => {
            event.stopPropagation();
            onSeekSegment?.({ ...segment, start: cue.start, end: cue.end, text: cue.text });
          }}
          className={clsx(
            "transcript-cue-button inline rounded-[4px] px-1 py-0.5 -mx-0.5 text-left align-baseline transition-colors duration-100",
            onSeekSegment && "cursor-pointer hover:bg-hover/60",
            isSegmentActive && !active && "text-fg",
            active && cue.source === "sync" && "bg-sky-400/42 text-white shadow-[inset_0_0_0_1px_rgba(125,211,252,0.72)]",
            active && cue.source === "segment" && !hasSyncCues && "text-white",
          )}
          title={`${formatTimeShort(cue.start)} - ${formatTimeShort(cue.end)}`}
        >
          {cue.text}
        </button>
      );
    });
  };

  if (viewMode === "dialog" && hasSpeakers) {
    return (
      <ul className="space-y-3 text-ui leading-relaxed">
        {turns.map((t, i) => {
          return (
          <li
            key={i}
            ref={(node) => { itemRefs.current[i] = node; }}
            onClick={() => onSeekSegment?.(t.segments[0])}
            className={clsx(
              "flex gap-3 min-w-0 group rounded-sm px-2 py-1.5 transition-colors",
              onSeekSegment && "cursor-pointer hover:bg-hover/70",
              i === activeTurnIndex && "shadow-[inset_2px_0_0_0_rgba(56,189,248,0.9)]",
            )}
          >
            <div className="flex flex-col items-start gap-1 shrink-0">
              <SpeakerChip
                speakers={speakers}
                who={t.speaker}
                segment={combinedSpeakerFlags(t.segments)}
                onRename={onRenameSpeaker}
              />
              <span className="text-ui-sm text-fg-mute font-mono">
                {formatTimeShort(t.start)}
              </span>
            </div>
            <div className="flex-1 min-w-0 break-words text-fg">
              {t.segments.map((s) => (
                <span key={`${s.start}-${s.end}`}>
                  {renderCueText(s, s.sourceSegmentIndex)}
                </span>
              ))}
            </div>
          </li>
          );
        })}
      </ul>
    );
  }

  // 时间戳模式
  return (
    <ul className="space-y-1 font-mono text-ui leading-relaxed">
      {segments.map((s, i) => {
        return (
        <li
          key={i}
          ref={(node) => { itemRefs.current[i] = node; }}
          onClick={() => onSeekSegment?.(s)}
          className={clsx(
            "flex gap-3 min-w-0 group items-start rounded-sm px-2 py-1 transition-colors",
            onSeekSegment && "cursor-pointer hover:bg-hover/70",
            i === activeSegmentIndex && "shadow-[inset_2px_0_0_0_rgba(56,189,248,0.9)]",
          )}
        >
          <span className="text-ui-sm text-fg-mute pt-0.5 whitespace-nowrap shrink-0 select-none w-12 text-right">
            {formatTimeShort(s.start)}
          </span>
          {hasSpeakers && <SpeakerChip speakers={speakers} who={displaySpeakerName(s.speaker)} segment={s} onRename={onRenameSpeaker} />}
          <span className="flex-1 min-w-0 break-words text-fg">
            {renderCueText(s, i)}
          </span>
        </li>
        );
      })}
    </ul>
  );
}

function CorrectedSegments({
  segments,
  changed,
  total,
  model,
  viewMode,
  onRenameSpeaker,
}: {
  segments: Segment[];
  changed: number;
  total: number;
  model: string;
  viewMode: ViewMode;
  onRenameSpeaker?: (oldName: string, newName: string) => void;
}) {
  const speakers = collectSpeakers(segments);
  const hasSpeakers = speakers.length > 0;

  const header = (
    <div className="text-ui-sm text-fg-mute pb-2 border-b border-border/60">
      模型 <span className="text-fg-dim font-mono">{model}</span> · 改动 {changed}/{total} 段
      <span className="ml-3 text-ok">绿色</span> 为校对后,
      <span className="text-err line-through ml-1">删除线</span> 为原文
    </div>
  );

  if (viewMode === "dialog" && hasSpeakers) {
    const turns = groupBySpeakerTurns(speakerDisplaySegments(segments));
    return (
      <div className="space-y-3">
        {header}
        <ul className="space-y-3 text-ui leading-relaxed">
          {turns.map((t, i) => {
            const anyChanged = t.segments.some(
              (s) => s.original_text && s.original_text !== s.text,
            );
            return (
              <li key={i} className="flex gap-3 min-w-0">
                <div className="flex flex-col items-start gap-1 shrink-0">
                  <SpeakerChip
                    speakers={speakers}
                    who={t.speaker}
                    segment={combinedSpeakerFlags(t.segments)}
                    onRename={onRenameSpeaker}
                  />
                  <span className="text-ui-sm text-fg-mute font-mono">
                    {formatTimeShort(t.start)}
                  </span>
                </div>
                <div className="flex-1 min-w-0 break-words">
                  <div className={anyChanged ? "text-ok" : "text-fg"}>
                    {t.segments.map((s) => s.text).join("")}
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {header}
      <ul className="space-y-1 font-mono text-ui leading-relaxed">
        {segments.map((s, i) => {
          const changedSeg = s.original_text && s.original_text !== s.text;
          return (
            <li key={i} className="flex gap-3 min-w-0 items-start">
              <span className="text-ui-sm text-fg-mute pt-0.5 whitespace-nowrap shrink-0 select-none w-12 text-right">
                {formatTimeShort(s.start)}
              </span>
              {hasSpeakers && <SpeakerChip speakers={speakers} who={displaySpeakerName(s.speaker)} segment={s} onRename={onRenameSpeaker} />}
              <div className="flex-1 min-w-0 break-words">
                {changedSeg && (
                  <div className="text-ui-sm text-err/80 line-through">
                    {s.original_text}
                  </div>
                )}
                <div className={changedSeg ? "text-ok" : "text-fg"}>{s.text}</div>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function ArticleView({
  text,
  model,
  source,
  truncated,
  inputChars,
  segments,
  audio,
}: {
  text: string;
  model: string;
  source?: "corrected" | "raw";
  truncated?: boolean;
  inputChars?: number;
  segments: Segment[];
  audio: string;
}) {
  const groups = useMemo(() => groupArticleSpeakerTurns(segments), [segments]);
  const [view, setView] = useState<"speakers" | "article">("speakers");
  const isCorrected = source === "corrected";
  // Heuristic fallback:LLM 没传 finish_reason 时,根据输入/输出比判断
  const ratio = inputChars && inputChars > 0 ? text.length / inputChars : null;
  const looksTruncated = truncated || (ratio !== null && ratio < 0.7);
  const completenessPct = ratio ? Math.round(ratio * 100) : null;
  return (
    <div className="space-y-4 max-w-3xl mx-auto">
      {/* 顶部:状态徽章 + 元数据 */}
      <div className="flex flex-wrap items-center gap-2 text-ui-sm text-fg-mute pb-2 border-b border-border/60">
        {/* 完整性徽章(主要) */}
        {looksTruncated ? (
          <span className="px-2 py-0.5 rounded-sm border text-ui-sm bg-warn/15 text-warn border-warn/40 font-medium">
            ⚠ 内容不完整(可能被 max_tokens 截断)
          </span>
        ) : (
          <span className="px-2 py-0.5 rounded-sm border text-ui-sm bg-ok/15 text-ok border-ok/40 font-medium">
            ✓ 完整生成
          </span>
        )}
        {/* 来源徽章 */}
        <span
          className={clsx(
            "px-2 py-0.5 rounded-sm border text-ui-sm",
            isCorrected
              ? "bg-ok/10 text-ok border-ok/30"
              : "bg-warn/10 text-warn border-warn/30",
          )}
        >
          {isCorrected ? "基于已校对稿" : "基于原始转录"}
        </span>
        <span>模型 <span className="text-fg-dim font-mono">{model}</span></span>
        <span>·</span>
        <span>{text.length} 字</span>
        {inputChars && (
          <>
            <span>·</span>
            <span>原 {inputChars} 字 ({completenessPct}%)</span>
          </>
        )}
      </div>

      <div className="flex items-center border border-border rounded-sm overflow-hidden w-fit">
        <button
          onClick={() => setView("speakers")}
          className={clsx(
            "px-2.5 py-1 text-ui-sm",
            view === "speakers" ? "bg-accent/20 text-accent" : "text-fg-mute hover:text-fg",
          )}
          title="按时间顺序查看连续发言，并播放对应音频"
        >
          按人发言
        </button>
        <button
          onClick={() => setView("article")}
          className={clsx(
            "px-2.5 py-1 text-ui-sm border-l border-border",
            view === "article" ? "bg-accent/20 text-accent" : "text-fg-mute hover:text-fg",
          )}
          title="查看 AI 整理后的文章"
        >
          整理文章
        </button>
      </div>

      {/* 详细告警条(仅截断时显示,提供补救建议) */}
      {looksTruncated && (
        <div className="bg-warn/5 border border-warn/30 rounded-sm p-3 text-ui-sm text-fg leading-relaxed">
          <div className="font-medium text-warn mb-1">检测到生成内容不完整</div>
          <div className="text-fg-dim">
            原始 {inputChars ?? "?"} 字仅生成了 {text.length} 字
            {completenessPct ? ` (${completenessPct}%)` : ""}。
            可能原因:LLM 输出 token 数被限制截断。
          </div>
          <div className="text-fg-dim mt-1">
            建议:打开 设置 → 校对 → 高级参数 → <span className="font-mono text-fg">排版 · 最大输出</span>,
            提高 <span className="font-mono text-fg">max_tokens</span>(已最大 384000),然后重新点 "整理为文章"。
          </div>
        </div>
      )}

      {view === "speakers" ? (
        <ArticleSpeakerGroups groups={groups} audio={audio} />
      ) : (
        <DialogueOrArticleContent text={text} />
      )}
    </div>
  );
}

function ArticleSpeakerGroups({
  groups,
  audio,
}: {
  groups: ReturnType<typeof groupArticleSpeakerTurns>;
  audio: string;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const stopAtRef = useRef<number | null>(null);
  const [src, setSrc] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const updatePlaybackTimeRef = useRef<(time: number) => void>(() => {});
  const speakers = useMemo(
    () => Array.from(new Set(groups.map((group) => group.speaker).filter((speaker): speaker is string => Boolean(speaker)))),
    [groups],
  );
  const activeGroupIndex = useMemo(() => groups.findIndex((group) => (
    playbackTime >= group.start - 0.03 && playbackTime <= group.end + 0.12
  )), [groups, playbackTime]);

  useEffect(() => {
    let cancelled = false;
    setSrc("");
    setError(null);
    setPlaybackTime(0);
    setPlaying(false);
    stopAtRef.current = null;
    audioRef.current?.pause();
    if (!audio) return;
    localMediaUrl(audio)
      .then((url) => {
        if (!cancelled) setSrc(url);
      })
      .catch(() => {
        if (!cancelled) setError("原始音频无法在当前环境播放");
      });
    return () => {
      cancelled = true;
    };
  }, [audio]);

  const updatePlaybackTime = useCallback((time: number) => {
    const stopAt = stopAtRef.current;
    if (stopAt != null && time >= stopAt - 0.02) {
      const player = audioRef.current;
      stopAtRef.current = null;
      if (player) {
        player.pause();
        if (Math.abs(player.currentTime - stopAt) > 0.03) player.currentTime = stopAt;
      }
      setPlaybackTime(stopAt);
      setPlaying(false);
      return;
    }
    setPlaybackTime(time);
  }, []);

  useEffect(() => {
    updatePlaybackTimeRef.current = updatePlaybackTime;
  }, [updatePlaybackTime]);

  useEffect(() => {
    const player = audioRef.current;
    if (!src || !player) return;
    let animationFrame: number | null = null;
    const stopFrameLoop = () => {
      if (animationFrame != null) cancelAnimationFrame(animationFrame);
      animationFrame = null;
    };
    const updateFromPlaybackClock = () => {
      updatePlaybackTimeRef.current(player.currentTime);
      if (!player.paused && !player.ended) {
        animationFrame = requestAnimationFrame(updateFromPlaybackClock);
      } else {
        animationFrame = null;
      }
    };
    const startFrameLoop = () => {
      stopFrameLoop();
      animationFrame = requestAnimationFrame(updateFromPlaybackClock);
    };
    player.addEventListener("play", startFrameLoop);
    player.addEventListener("pause", stopFrameLoop);
    player.addEventListener("ended", stopFrameLoop);
    if (!player.paused && !player.ended) startFrameLoop();
    return () => {
      stopFrameLoop();
      player.removeEventListener("play", startFrameLoop);
      player.removeEventListener("pause", stopFrameLoop);
      player.removeEventListener("ended", stopFrameLoop);
    };
  }, [src]);

  const playGroup = async (index: number) => {
    const player = audioRef.current;
    const group = groups[index];
    if (!player || !group) return;
    if (index === activeGroupIndex && !player.paused) {
      player.pause();
      setPlaying(false);
      return;
    }
    const start = Math.max(0, group.start);
    const requestedEnd = Math.max(start + 0.05, group.end);
    const end = Number.isFinite(player.duration) && player.duration > 0
      ? Math.min(player.duration, requestedEnd)
      : requestedEnd;
    stopAtRef.current = end;
    player.currentTime = start;
    setPlaybackTime(start);
    try {
      await player.play();
    } catch {
      stopAtRef.current = null;
      setError("音频播放失败，请点击播放器重试");
    }
  };

  if (!groups.length) {
    return <div className="py-8 text-center text-ui text-fg-mute">没有可按人归类的转录内容。</div>;
  }

  return (
    <section className="space-y-3">
      <div className="rounded-sm border border-border bg-bg-panel/70 px-3 py-2">
        {src ? (
          <audio
            ref={audioRef}
            src={src}
            controls
            preload="metadata"
            className="block h-8 w-full"
            onTimeUpdate={(event) => updatePlaybackTime(event.currentTarget.currentTime)}
            onSeeked={(event) => {
              stopAtRef.current = null;
              updatePlaybackTime(event.currentTarget.currentTime);
            }}
            onLoadedMetadata={(event) => updatePlaybackTime(event.currentTarget.currentTime)}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => {
              stopAtRef.current = null;
              setPlaying(false);
            }}
            onError={() => setError("原始音频无法在当前环境播放")}
          />
        ) : (
          <div className="h-8 flex items-center text-ui-sm text-fg-mute">
            {audio ? "正在准备原始音频..." : "此任务没有可播放的原始音频"}
          </div>
        )}
        <div className="mt-2 flex items-center gap-2 text-ui-sm text-fg-mute">
          <span className="font-mono text-accent">{formatTimeShort(playbackTime)}</span>
          <span>{activeGroupIndex >= 0 ? "当前语句" : "未播放"}</span>
        </div>
        {error && <div className="mt-2 text-ui-sm text-err">{error}</div>}
      </div>

      <ol className="space-y-2">
        {groups.map((group, index) => {
          const active = activeGroupIndex === index;
          const canPlay = Boolean(src);
          const isPlayingGroup = active && playing;
          return (
            <li
              key={`${group.start}-${group.end}-${index}`}
              className={clsx(
                "grid grid-cols-[auto_auto_minmax(0,1fr)] items-start gap-3 border-l-2 px-3 py-2 transition-colors",
                active ? "border-accent" : "border-border bg-transparent hover:bg-bg-panel/60",
              )}
            >
              <button
                onClick={() => void playGroup(index)}
                disabled={!canPlay}
                className={clsx(
                  "mt-0.5 grid h-7 w-7 place-items-center rounded-sm border",
                  canPlay
                    ? "border-border text-fg-mute hover:border-accent/50 hover:text-accent"
                    : "border-border/50 text-fg-mute/50 cursor-not-allowed",
                )}
                title={isPlayingGroup ? "暂停当前发言" : "播放这一组发言"}
                aria-label={isPlayingGroup ? "暂停当前发言" : "播放这一组发言"}
              >
                {isPlayingGroup ? <Pause size={14} /> : <Play size={14} />}
              </button>
              <div className="min-w-[74px] pt-0.5">
                {group.speaker ? (
                  <SpeakerChip speakers={speakers} who={group.speaker} />
                ) : (
                  <span className="inline-flex border border-border rounded-sm px-1.5 py-0.5 text-ui-sm text-fg-mute">
                    未分人
                  </span>
                )}
                <div className="mt-1 whitespace-nowrap font-mono text-ui-sm text-fg-mute">
                  {formatTimeShort(group.start)} - {formatTimeShort(group.end)}
                </div>
              </div>
              <ArticleSpeakerGroupText cues={group.cues} playbackTime={active ? playbackTime : null} />
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function ArticleSpeakerGroupText({
  cues,
  playbackTime,
}: {
  cues: ReturnType<typeof groupArticleSpeakerTurns>[number]["cues"];
  playbackTime: number | null;
}) {
  const activeCueIndex = useMemo(() => {
    if (playbackTime == null) return -1;
    for (let index = 0; index < cues.length; index += 1) {
      const cue = cues[index];
      const nextStart = cues[index + 1]?.start;
      const end = Math.min(cue.end + 0.12, nextStart ?? Number.POSITIVE_INFINITY);
      if (playbackTime >= cue.start - 0.03 && playbackTime <= end) return index;
    }
    return -1;
  }, [cues, playbackTime]);

  return (
    <p className="min-w-0 whitespace-pre-wrap break-words text-ui leading-relaxed text-fg">
      {cues.map((cue, index) => (
        <span
          key={`${cue.start}-${cue.end}-${index}`}
          className={clsx(
            "rounded-[2px] transition-colors",
            activeCueIndex === index && "bg-accent/25 text-fg",
          )}
        >
          {cue.text}
        </span>
      ))}
    </p>
  );
}

// ============================================================================
// 对话体内容渲染:自动识别 **NAME:** 头,渲染成 SpeakerChip + 内容
// 输入若不含对话头,降级为纯文章渲染(whitespace-pre-wrap)。
// ============================================================================

function DialogueOrArticleContent({ text }: { text: string }) {
  // 解析:逐行扫,**NAME:** 标记每个回合起点
  // 兼容三种格式:
  //   **陈总:** 内容
  //   **陈总:**\n内容(标题独占一行,内容下一行)
  //   普通段落(无标记)
  const turns: { speaker: string; content: string }[] = [];
  const headerRe = /^\*\*([^*\n]+?):\*\*\s*/;
  const lines = text.split(/\r?\n/);
  let current: { speaker: string; content: string } | null = null;
  for (const line of lines) {
    const m = line.match(headerRe);
    if (m) {
      if (current) turns.push(current);
      current = { speaker: m[1].trim(), content: line.slice(m[0].length).trim() };
    } else if (current) {
      // 续行附加进当前回合
      const trimmed = line.trim();
      if (current.content && trimmed) current.content += "\n" + trimmed;
      else if (trimmed) current.content = trimmed;
      else if (current.content) current.content += "\n";  // 保留空行作段间隔
    }
  }
  if (current) turns.push(current);

  // 没找到任何对话头 → 降级
  if (turns.length === 0) {
    return <article className="text-fg leading-loose whitespace-pre-wrap text-ui-lg">{text}</article>;
  }

  const speakers = Array.from(new Set(turns.map((t) => t.speaker)));
  return (
    <article className="space-y-4 text-fg leading-loose text-ui-lg">
      {turns.map((t, i) => (
        <div key={i} className="flex gap-2 items-start">
          <div className="pt-0.5">
            <SpeakerChip speakers={speakers} who={t.speaker} />
          </div>
          <div className="flex-1 whitespace-pre-wrap">{t.content}</div>
        </div>
      ))}
    </article>
  );
}

// ============================================================================
// 译文视图
// ============================================================================

function TranslatedView({
  text,
  sourceLanguage,
  targetLanguage,
  model,
  truncated,
  inputChars,
}: {
  text: string;
  sourceLanguage: string | null;
  targetLanguage: string;
  model: string;
  truncated?: boolean;
  inputChars?: number;
}) {
  const looksTruncated = truncated || (inputChars && text.length < inputChars * 0.8);
  const completenessPct = inputChars ? Math.round((text.length / inputChars) * 100) : null;

  const langNames: Record<string, string> = {
    zh: "中文",
    en: "English",
    ja: "日本語",
    ko: "한국어",
  };

  return (
    <div className="space-y-3">
      {/* 元信息条 */}
      <div className="flex items-center gap-3 text-ui-sm text-fg-mute">
        {!looksTruncated && (
          <span className="px-2 py-0.5 rounded-sm border border-ok/30 bg-ok/10 text-ok text-ui-sm">
            ✓ 翻译完成
          </span>
        )}
        {/* 语言方向 */}
        <span className="px-2 py-0.5 rounded-sm border border-accent/30 bg-accent/10 text-accent">
          {sourceLanguage ? langNames[sourceLanguage] || sourceLanguage.toUpperCase() : "?"} → {langNames[targetLanguage] || targetLanguage.toUpperCase()}
        </span>
        <span>模型 <span className="text-fg-dim font-mono">{model}</span></span>
        <span>·</span>
        <span>{text.length} 字</span>
        {inputChars && (
          <>
            <span>·</span>
            <span>原 {inputChars} 字 ({completenessPct}%)</span>
          </>
        )}
      </div>

      {/* 截断警告 */}
      {looksTruncated && (
        <div className="bg-warn/5 border border-warn/30 rounded-sm p-3 text-ui-sm text-fg leading-relaxed">
          <div className="font-medium text-warn mb-1">检测到翻译内容不完整</div>
          <div className="text-fg-dim">
            原文 {inputChars ?? "?"} 字仅翻译了 {text.length} 字
            {completenessPct ? ` (${completenessPct}%)` : ""}。
            可能原因:LLM 输出 token 数被限制截断。
          </div>
          <div className="text-fg-dim mt-1">
            建议:打开 设置 → 校对 → 高级参数 → <span className="font-mono text-fg">排版 · 最大输出</span>,
            提高 <span className="font-mono text-fg">max_tokens</span>,然后重新翻译。
          </div>
        </div>
      )}

      <DialogueOrArticleContent text={text} />
    </div>
  );
}

// ============================================================================
// CTA 占位(没数据时显示触发按钮 / 提示)
// ============================================================================

function CorrectionCTA({
  busy,
  onCorrect,
  onPipelineFull,
  onOpenSettings,
}: {
  busy: boolean;
  onCorrect: () => Promise<void> | void;
  onPipelineFull: () => Promise<void> | void;
  onOpenSettings: () => void;
}) {
  const enabled = useSettings((s) => s.settings.correction.enabled);
  const hasApiKey = useSettings((s) => s.hasApiKey);
  const provider = useSettings((s) => s.settings.correction.provider);
  const model = useSettings((s) => s.settings.correction.model);

  const [error, setError] = useState<string | null>(null);

  async function trigger() {
    setError(null);
    try {
      await onCorrect();
    } catch (e) {
      setError(String(e));
    }
  }

  if (busy) {
    return (
      <div className="text-ui text-warn py-16 text-center flex flex-col items-center gap-2">
        <Hourglass size={28} />
        <div>正在校对(LLM 调用中)</div>
      </div>
    );
  }

  if (!enabled || !hasApiKey) {
    return (
      <div className="py-12 px-4 text-center flex flex-col items-center gap-3">
        <Pencil size={28} className="text-fg-mute" />
        <div className="text-ui text-fg-dim max-w-md leading-relaxed">
          {!enabled
            ? "未启用 LLM 校对。在设置中开启后可对此转录做字级校对。"
            : `LLM 校对已启用,但 ${provider} 的 API Key 尚未配置。`}
        </div>
        <button onClick={onOpenSettings} className="btn mt-1">前往设置</button>
      </div>
    );
  }

  return (
    <div className="py-12 px-4 text-center flex flex-col items-center gap-3">
      <Pencil size={28} className="text-accent" />
      <div className="text-ui text-fg-dim max-w-md leading-relaxed">
        将转录段落送至 <span className="font-mono text-fg">{model}</span> 做字级校对(修同音字 / 错别字 / ASR 冗余)。
        <br />每段保留原文用于对比。
      </div>
      {error && <div className="text-ui-sm text-err">{error}</div>}
      <div className="flex justify-center gap-2 mt-1">
        <button onClick={trigger} className="btn">开始校对</button>
        <button
          onClick={async () => {
            setError(null);
            try {
              await onPipelineFull();
            } catch (e) {
              setError(String(e));
            }
          }}
          className="btn"
          title="校对完成后自动接力做整篇排版"
        >
          校对 + 排版(一键)
        </button>
        <button onClick={onOpenSettings} className="btn-ghost">设置</button>
      </div>
    </div>
  );
}

function PolishCTA({
  busy,
  hasCorrected,
  onPolish,
  onOpenSettings,
}: {
  busy: boolean;
  hasCorrected: boolean;
  onPolish: () => Promise<void> | void;
  onOpenSettings: () => void;
}) {
  const enabled = useSettings((s) => s.settings.correction.enabled);
  const hasApiKey = useSettings((s) => s.hasApiKey);
  const model = useSettings((s) => s.settings.polish.model);

  const [error, setError] = useState<string | null>(null);

  async function trigger() {
    setError(null);
    try {
      await onPolish();
    } catch (e) {
      setError(String(e));
    }
  }

  if (busy) {
    return (
      <div className="text-ui text-ok py-16 text-center flex flex-col items-center gap-2">
        <Hourglass size={28} />
        <div>正在生成文章</div>
      </div>
    );
  }

  if (!enabled || !hasApiKey) {
    return (
      <div className="py-12 px-4 text-center flex flex-col items-center gap-3">
        <Article size={28} className="text-fg-mute" />
        <div className="text-ui text-fg-dim max-w-md leading-relaxed">
          整篇排版需要 LLM。请先在设置中启用并配置 API Key。
        </div>
        <button onClick={onOpenSettings} className="btn mt-1">前往设置</button>
      </div>
    );
  }

  return (
    <div className="py-12 px-4 text-center flex flex-col items-center gap-3">
      <Article size={28} className="text-accent" />
      <div
        className={clsx(
          "px-2.5 py-1 rounded-sm border text-ui-sm",
          hasCorrected
            ? "bg-ok/10 text-ok border-ok/30"
            : "bg-warn/10 text-warn border-warn/30",
        )}
      >
        {hasCorrected ? "将基于已校对稿生成" : "将基于原始转录生成(未校对)"}
      </div>
      <div className="text-ui text-fg-dim max-w-md leading-relaxed">
        拼成连续散文,自动加标点和分段,输出完整文字稿。模型 <span className="font-mono text-fg">{model}</span>
        {!hasCorrected && (
          <div className="mt-2 text-ui-sm text-fg-mute">
            建议先在「校对」标签运行一次校对,可显著提升排版质量。
          </div>
        )}
      </div>
      {error && <div className="text-ui-sm text-err">{error}</div>}
      <div className="flex justify-center gap-2 mt-1">
        <button onClick={trigger} className="btn">整理为文章</button>
        <button onClick={onOpenSettings} className="btn-ghost">设置</button>
      </div>
    </div>
  );
}

// ============================================================================
// 导出栏
// ============================================================================

function ExportBar({
  task,
  tab,
  onCorrect,
  onPolish,
  onArticleSaved,
}: {
  task: Task;
  tab: Tab;
  onCorrect: () => Promise<void> | void;
  onPolish: () => Promise<void> | void;
  onArticleSaved?: () => void;
}) {
  const isCorrecting = task.stage === "correcting" || task.stage === "correcting_paused";
  const isPolishing = task.stage === "polishing";
  const isTranslating = task.stage === "translating";
  const [pinOpen, setPinOpen] = useState(false);
  const [showTranslateMenu, setShowTranslateMenu] = useState(false);
  const settings = useSettings((s) => s.settings);
  const hasApiKey = useSettings((s) => s.hasApiKey);
  const setTranslated = useTasks((s) => s.setTranslated);
  const setStage = useTasks((s) => s.setStage);
  const setError = useTasks((s) => s.setError);

  // Close translate menu when clicking outside
  useEffect(() => {
    if (!showTranslateMenu) return;
    const handleClick = () => setShowTranslateMenu(false);
    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, [showTranslateMenu]);

  async function copy(text: string) {
    await navigator.clipboard.writeText(text);
  }
  async function download(name: string, text: string) {
    await downloadTextFile(name, text);
  }

  async function translateTo(targetLang: string) {
    console.log("[translateTo] called with targetLang:", targetLang);
    console.log("[translateTo] task.polished:", task.polished);
    console.log("[translateTo] task.stage:", task.stage);

    if (!task.polished) {
      console.warn("[translateTo] No polished text, returning");
      alert("请先完成文章排版后再翻译");
      return;
    }
    setShowTranslateMenu(false);

    try {
      console.log("[translateTo] Starting translation...");
      setStage(task.id, "translating");

      const result = await ipc.translateArticle({
        text: task.polished.text,
        source_language: task.result?.language || settings.language || undefined,
        target_language: targetLang,
        glossary: task.corrected?.glossary,
        provider: settings.correction.provider,
        base_url: settings.correction.base_url,
        model: settings.translation.model,
        temperature: settings.translation.advanced.temperature,
        max_tokens: settings.translation.advanced.max_tokens,
        top_p: settings.translation.advanced.top_p,
        frequency_penalty: settings.translation.advanced.frequency_penalty,
        presence_penalty: settings.translation.advanced.presence_penalty,
      });

      setTranslated(task.id, {
        text: result.text,
        source_language: result.source_language,
        target_language: result.target_language,
        model: result.model,
        truncated: result.truncated,
        finish_reason: result.finish_reason,
        input_chars: result.input_chars,
      });
    } catch (e) {
      const errorMsg = `翻译失败: ${String(e)}`;
      console.error("[translateTo] Error:", e);
      console.error("[translateTo] Error stack:", e instanceof Error ? e.stack : "no stack");

      showMessage(errorMsg, "翻译错误", "error");

      setError(task.id, errorMsg);
      setStage(task.id, "polished");
    }
  }

  const stem = persistenceStem(task);
  const result = task.result!;
  const segments =
    tab === "corrected" && task.corrected ? task.corrected.segments : result.segments;

  // For the article tab when polished is missing, no export buttons make sense.
  if (tab === "article" && !task.polished) {
    return null;
  }
  // For corrected tab when no correction yet, hide too.
  if (tab === "corrected" && !task.corrected) {
    return null;
  }
  // For translated tab when no translation yet, hide too.
  if (tab === "translated" && !task.translated) {
    return null;
  }

  return (
    <div className="shrink-0 flex items-center gap-1 px-3 h-9 border-t border-border/60 bg-tabbar/60">
      {tab === "translated" && task.translated ? (
        <>
          <ActionBtn icon={<Copy size={12} />} label="复制译文" onClick={() => copy(task.translated!.text)} />
          <ActionBtn icon={<Download size={12} />} label=".txt" onClick={() => download(`${stem}_译文_${task.translated!.target_language}.txt`, task.translated!.text)} />
          <ActionBtn
            icon={<Download size={12} />}
            label=".md"
            onClick={() => {
              const langNames: Record<string, string> = { zh: "中文", en: "English", ja: "日本語", ko: "한국어" };
              const srcLang = task.translated!.source_language ? langNames[task.translated!.source_language] || task.translated!.source_language : "?";
              const tgtLang = langNames[task.translated!.target_language] || task.translated!.target_language;
              const md = `# ${stem} (译文)\n\n> _${srcLang} → ${tgtLang} · ${task.translated!.model} · ${task.translated!.text.length} 字_\n\n${task.translated!.text}\n`;
              download(`${stem}_译文_${task.translated!.target_language}.md`, md);
            }}
          />
        </>
      ) : tab === "article" && task.polished ? (
        <>
          <ActionBtn icon={<Copy size={12} />} label="复制全文" onClick={() => copy(task.polished!.text)} />
          <ActionBtn icon={<Download size={12} />} label=".txt" onClick={() => download(`${stem}_完整版.txt`, task.polished!.text)} />
          <ActionBtn
            icon={<Download size={12} />}
            label=".md"
            onClick={() => {
              const meta = task.polished!.source === "corrected" ? "校对+排版" : "原文+排版";
              const md = `# ${stem}\n\n> _${meta} · ${task.polished!.model} · ${task.polished!.text.length} 字_\n\n${task.polished!.text}\n`;
              download(`${stem}_完整版.md`, md);
            }}
          />
          <span className="mx-1 h-4 w-px bg-border" />
          <ActionBtn
            icon={<Lock size={12} />}
            label="保存到文章库"
            onClick={() => setPinOpen(true)}
            title="保存为带语义化文件名的 markdown,AI agent 可通过 articles/ 读取"
          />
          <ActionBtn
            icon={<Refresh size={12} />}
            label={isPolishing ? "排版中..." : "重新生成"}
            onClick={() => onPolish()}
            disabled={isPolishing}
            title="重新跑一遍排版(基于当前校对稿/原文)"
          />
          <span className="mx-1 h-4 w-px bg-border" />
          {/* 翻译按钮 */}
          <div className="relative">
            <ActionBtn
              icon={<Globe size={12} />}
              label={isTranslating ? "翻译中..." : "翻译"}
              onClick={(e) => {
                e.stopPropagation();
                setShowTranslateMenu(!showTranslateMenu);
              }}
              disabled={isTranslating || !hasApiKey}
              title={hasApiKey ? "翻译文章到其他语言" : "需要配置 API Key"}
            />
            {showTranslateMenu && (
              <div className="absolute bottom-full left-0 mb-1 bg-sidebar border border-border rounded-sm shadow-lg py-1 min-w-[120px] z-50">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    translateTo("zh");
                  }}
                  className="w-full px-3 py-1.5 text-left text-sm hover:bg-accent/10 hover:text-accent"
                >
                  中文
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    translateTo("en");
                  }}
                  className="w-full px-3 py-1.5 text-left text-sm hover:bg-accent/10 hover:text-accent"
                >
                  English
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    translateTo("ja");
                  }}
                  className="w-full px-3 py-1.5 text-left text-sm hover:bg-accent/10 hover:text-accent"
                >
                  日本語
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    translateTo("ko");
                  }}
                  className="w-full px-3 py-1.5 text-left text-sm hover:bg-accent/10 hover:text-accent"
                >
                  한국어
                </button>
              </div>
            )}
          </div>
        </>
      ) : (
        <>
          <ActionBtn icon={<Copy size={12} />} label="复制(带时间戳)" onClick={() => copy(buildTxt(segments))} />
          <ActionBtn icon={<Copy size={12} />} label="复制纯文本" onClick={() => copy(segments.map((s) => s.text).join(""))} />
          <span className="mx-1 h-4 w-px bg-border" />
          <ActionBtn icon={<Download size={12} />} label=".txt" onClick={() => download(`${stem}${tab === "corrected" ? "_corrected" : ""}.txt`, buildTxt(segments, `${stem} (${tab})`))} />
          <ActionBtn icon={<Download size={12} />} label=".srt" onClick={() => download(`${stem}${tab === "corrected" ? "_corrected" : ""}.srt`, buildSrt(segments))} />
          <ActionBtn icon={<Download size={12} />} label=".md" onClick={() => download(`${stem}.md`, buildMd(segments, stem))} />
          <ActionBtn icon={<Download size={12} />} label=".json" onClick={() => download(`${stem}.json`, buildJson(result))} />
          {tab === "corrected" && task.corrected && task.stage !== "cancelled" && (
            <>
              <span className="mx-1 h-4 w-px bg-border" />
              <ActionBtn
                icon={<Refresh size={12} />}
                label={isCorrecting ? "校对中..." : "重新校对"}
                onClick={() => onCorrect()}
                disabled={isCorrecting}
                title="重新跑一遍校对(覆盖当前结果)"
              />
            </>
          )}
        </>
      )}

      {pinOpen && task.polished && (
        <PinArticleDialog
          defaultTitle={stem}
          content={task.polished.text}
          source_audio={task.audio}
          source_stem={persistenceStem(task)}
          duration_seconds={task.result?.duration}
          model={task.polished.model}
          based_on={task.polished.source}
          onClose={(saved) => {
            setPinOpen(false);
            if (saved) onArticleSaved?.();
          }}
        />
      )}
    </div>
  );
}

function ActionBtn({
  icon,
  label,
  onClick,
  disabled,
  title,
}: {
  icon: React.ReactNode;
  label: string;
  onClick: (e: React.MouseEvent) => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className="btn-ghost h-6 px-2 text-ui-sm"
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

function formatTimeShort(seconds: number): string {
  const m = Math.floor(seconds / 60).toString().padStart(2, "0");
  const s = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}
