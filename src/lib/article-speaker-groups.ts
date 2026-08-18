import type { Segment } from "./ipc.ts";
import { segmentCues, type TranscriptSyncCue } from "./transcript-sync.ts";

export type ArticleSpeakerGroup = {
  /** Undefined means diarization was not available. It is never replaced with a fabricated A/B label. */
  speaker?: string;
  start: number;
  end: number;
  text: string;
  cues: ArticleSpeakerCue[];
  segments: Segment[];
};

export type ArticleSpeakerCue = {
  start: number;
  end: number;
  text: string;
};

type SpeakerCue = TranscriptSyncCue & { speaker?: string };
type PunctuationCue = ArticleSpeakerCue & { speaker?: string };

const PHRASE_END_RE = /[，。！？；：,.!?;:]$/;

function normalizedText(value: string): string {
  return (value || "").replace(/\s+/g, "");
}

function normalizedSpeaker(value?: string): string | undefined {
  const raw = value?.trim();
  if (!raw) return undefined;
  const match = raw.match(/^(?:SPEAKER_)?([A-Za-z])$/i);
  return match ? `SPEAKER_${match[1].toUpperCase()}` : raw;
}

function projectedSpeakerCues(segment: Segment): SpeakerCue[] | null {
  const cues = (segment.speaker_cues ?? [])
    .filter((cue) => (
      typeof cue.text === "string"
      && cue.text.trim().length > 0
      && Number.isFinite(cue.start)
      && Number.isFinite(cue.end)
      && cue.end > cue.start
    ))
    .sort((left, right) => left.start - right.start);

  if (!cues.length) return null;
  const monotonic = cues.every((cue, index) => index === 0 || cue.start >= cues[index - 1].end);
  const cueText = cues.map((cue) => cue.text || "").join("");
  if (!monotonic || normalizedText(cueText) !== normalizedText(segment.text)) return null;

  return cues.map((cue) => ({
    text: cue.text || "",
    start: Math.max(segment.start, cue.start),
    end: Math.min(segment.end, cue.end),
    source: "sync" as const,
    reliable: true,
    speaker: normalizedSpeaker(cue.speaker),
  })).filter((cue) => cue.end > cue.start);
}

function displayCues(segment: Segment): SpeakerCue[] {
  const projected = projectedSpeakerCues(segment);
  if (projected?.length) return projected;
  const speaker = normalizedSpeaker(segment.speaker);
  return segmentCues(segment).map((cue) => ({
    ...cue,
    speaker: normalizedSpeaker(cue.speaker) ?? speaker,
  }));
}

function splitCueAtPunctuation(cue: SpeakerCue): PunctuationCue[] {
  const parts = cue.text.match(/[^，。！？；：,.!?;:]+[，。！？；：,.!?;:]*/g) ?? [cue.text];
  const cleanParts = parts.map((part) => part.trim()).filter(Boolean);
  if (cleanParts.length <= 1) {
    return [{ start: cue.start, end: cue.end, text: cue.text.trim(), speaker: cue.speaker }];
  }

  const weights = cleanParts.map((part) => Math.max(1, Array.from(part.replace(/\s+/g, "")).length));
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  const duration = Math.max(0.05, cue.end - cue.start);
  let elapsedWeight = 0;
  return cleanParts.map((part, index) => {
    const start = cue.start + duration * (elapsedWeight / totalWeight);
    elapsedWeight += weights[index];
    const end = index === cleanParts.length - 1
      ? cue.end
      : cue.start + duration * (elapsedWeight / totalWeight);
    return { start, end: Math.max(start + 0.01, end), text: part, speaker: cue.speaker };
  });
}

function punctuationPhrases(cues: SpeakerCue[]): PunctuationCue[] {
  const phrases: PunctuationCue[] = [];
  let pending: PunctuationCue | null = null;

  for (const cue of cues.flatMap(splitCueAtPunctuation)) {
    if (pending && pending.speaker !== cue.speaker) {
      phrases.push(pending);
      pending = null;
    }
    if (!pending) {
      pending = { ...cue };
    } else {
      pending.end = cue.end;
      pending.text += cue.text;
    }
    if (PHRASE_END_RE.test(cue.text)) {
      phrases.push(pending);
      pending = null;
    }
  }
  if (pending) phrases.push(pending);
  return phrases;
}

/**
 * Produces chronological speech turns for the article view. Consecutive turns are
 * merged only when their resolved speaker is identical; A -> B -> A stays three turns.
 */
export function groupArticleSpeakerTurns(segments: Segment[]): ArticleSpeakerGroup[] {
  const ordered = segments
    .map((segment, index) => ({ segment, index }))
    .sort((left, right) => left.segment.start - right.segment.start || left.index - right.index);
  const groups: ArticleSpeakerGroup[] = [];

  for (const { segment } of ordered) {
    for (const cue of punctuationPhrases(displayCues(segment))) {
      const text = cue.text.trim();
      if (!text) continue;
      const speaker = normalizedSpeaker(cue.speaker) ?? normalizedSpeaker(segment.speaker);
      const previous = groups[groups.length - 1];
      if (previous && previous.speaker === speaker) {
        previous.end = Math.max(previous.end, cue.end);
        previous.text += text;
        previous.cues.push({ start: cue.start, end: cue.end, text });
        if (!previous.segments.includes(segment)) previous.segments.push(segment);
      } else {
        groups.push({
          speaker,
          start: cue.start,
          end: cue.end,
          text,
          cues: [{ start: cue.start, end: cue.end, text }],
          segments: [segment],
        });
      }
    }
  }
  return groups;
}
