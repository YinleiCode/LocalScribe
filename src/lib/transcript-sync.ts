export type TranscriptSyncCue = {
  text: string;
  start: number;
  end: number;
  source: "sync" | "segment";
  reliable: boolean;
  speaker?: string;
  speakerConfidence?: number;
  speakerReview?: boolean;
};

export type TranscriptSyncSegment = {
  text: string;
  start: number;
  end: number;
  sync_cues?: Array<{ text: string; start: number; end: number; reliable?: boolean }>;
  speaker_cues?: Array<{
    cue_index: number;
    start: number;
    end: number;
    text?: string;
    speaker: string;
    confidence?: number;
    review?: boolean;
  }>;
};

const ACTIVE_START_EPSILON_S = 0.03;
const ACTIVE_END_HOLD_S = 0.18;
const BOUNDARY_CHARS = new Set([
  "，", "。", "！", "？", "；", "：", "、", ",", ".", "!", "?", ";", ":",
]);

function hasDisplayText(segment?: TranscriptSyncSegment): boolean {
  return Boolean((segment?.text || "").replace(/[，。！？；：、,.!?;:\s]/g, "").trim());
}

export function activeSegmentIndexAt(
  segments: TranscriptSyncSegment[],
  time: number,
): number | null {
  if (!segments.length || !Number.isFinite(time)) return null;

  let lo = 0;
  let hi = segments.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (segments[mid].start <= time + ACTIVE_START_EPSILON_S) lo = mid + 1;
    else hi = mid;
  }

  for (let index = Math.min(lo - 1, segments.length - 1); index >= 0; index -= 1) {
    const segment = segments[index];
    if (time > segment.end + ACTIVE_END_HOLD_S) break;
    if (
      hasDisplayText(segment)
      && time >= segment.start - ACTIVE_START_EPSILON_S
      && time <= segment.end + ACTIVE_END_HOLD_S
    ) {
      return index;
    }
  }
  return null;
}

function normalizedCueText(text: string): string {
  return (text || "").replace(/\s+/g, "");
}

function cueTimingWeight(text: string): number {
  const chars = Math.max(1, Array.from((text || "").replace(/\s+/g, "")).length);
  const softPauses = text.match(/[，、,]/g)?.length ?? 0;
  const hardPauses = text.match(/[。！？；：.!?;:]/g)?.length ?? 0;
  return chars + softPauses * 1.2 + hardPauses * 1.8;
}

function chooseDisplayTextSplit(chars: string[], rough: number, min: number, max: number): number {
  const safeMin = Math.max(1, min);
  const safeMax = Math.min(chars.length - 1, max);
  if (safeMin >= safeMax) return Math.max(1, Math.min(chars.length - 1, rough));

  let best = -1;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = safeMin; index <= safeMax; index += 1) {
    if (!BOUNDARY_CHARS.has(chars[index - 1])) continue;
    const distance = Math.abs(index - rough);
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  }
  return best > 0 ? best : Math.max(safeMin, Math.min(safeMax, rough));
}

function splitDisplayTextForCues(text: string, cueTexts: string[]): string[] {
  const displayText = (text || "").trim();
  if (!displayText) return [];
  if (cueTexts.length <= 1) return [displayText];

  const chars = Array.from(displayText);
  const weights = cueTexts.map((cue) => Math.max(1, cueTimingWeight(cue)));
  const totalWeight = weights.reduce((sum, value) => sum + value, 0) || cueTexts.length;
  const pieces: string[] = [];
  let consumedWeight = 0;
  let cursor = 0;

  for (let index = 0; index < cueTexts.length - 1; index += 1) {
    consumedWeight += weights[index];
    const remaining = cueTexts.length - index - 1;
    const rough = Math.round(chars.length * (consumedWeight / totalWeight));
    const splitAt = chooseDisplayTextSplit(chars, rough, cursor + 1, chars.length - remaining);
    pieces.push(chars.slice(cursor, splitAt).join(""));
    cursor = splitAt;
  }
  pieces.push(chars.slice(cursor).join(""));
  return pieces;
}

export function segmentCues(segment: TranscriptSyncSegment): TranscriptSyncCue[] {
  const raw = Array.isArray(segment.sync_cues) ? segment.sync_cues : [];
  const cues = raw
    .map((cue) => ({
      text: String(cue.text || "").trim(),
      start: Math.max(segment.start, Number(cue.start)),
      end: Math.min(segment.end, Number(cue.end)),
      source: "sync" as const,
      reliable: cue.reliable !== false,
    }))
    .filter((cue) => cue.text && Number.isFinite(cue.start) && Number.isFinite(cue.end))
    .map((cue) => ({ ...cue, end: Math.min(segment.end, Math.max(cue.end, cue.start + 0.05)) }))
    .filter((cue) => cue.end > cue.start)
    .sort((left, right) => left.start - right.start);

  if (!cues.length) {
    return [{
      text: segment.text,
      start: segment.start,
      end: segment.end,
      source: "segment",
      reliable: true,
    }];
  }

  cues[0].start = segment.start;
  cues[cues.length - 1].end = segment.end;
  for (let index = 1; index < cues.length; index += 1) {
    if (cues[index].start < cues[index - 1].end) {
      const midpoint = (cues[index].start + cues[index - 1].end) / 2;
      cues[index - 1].end = midpoint;
      cues[index].start = midpoint;
    }
  }

  const cueTextMatchesSegment = normalizedCueText(cues.map((cue) => cue.text).join(""))
    === normalizedCueText(segment.text);
  const displayPieces = cueTextMatchesSegment
    ? cues.map((cue) => cue.text)
    : splitDisplayTextForCues(segment.text, cues.map((cue) => cue.text));

  const speakerCues = Array.isArray(segment.speaker_cues) ? segment.speaker_cues : [];
  return cues.map((cue, index) => {
    const projected = speakerCues.find((item) => item.cue_index === index)
      ?? speakerCues.find((item) => (
        Math.min(cue.end, item.end) - Math.max(cue.start, item.start)
      ) >= Math.min(0.25, Math.max(0.05, (cue.end - cue.start) * 0.5)));
    return {
      ...cue,
      text: displayPieces[index] || "",
      speaker: projected?.speaker,
      speakerConfidence: projected?.confidence,
      speakerReview: projected?.review,
    };
  })
    .filter((cue) => cue.text);
}

export function activeCueIndexAt(
  segment: TranscriptSyncSegment | null,
  time: number,
): number | null {
  if (!segment || !Number.isFinite(time)) return null;
  const cues = segmentCues(segment);
  for (let index = 0; index < cues.length; index += 1) {
    const cue = cues[index];
    if (!cue.reliable) continue;
    const nextStart = index < cues.length - 1 ? cues[index + 1].start : null;
    const holdEnd = nextStart == null
      ? Math.min(segment.end + ACTIVE_END_HOLD_S, cue.end + ACTIVE_END_HOLD_S)
      : Math.min(cue.end + ACTIVE_END_HOLD_S, nextStart);
    if (time >= cue.start - ACTIVE_START_EPSILON_S && time <= holdEnd) return index;
  }
  return null;
}

export function activeCueAt(
  segment: TranscriptSyncSegment | null,
  time: number,
): TranscriptSyncCue | null {
  const index = activeCueIndexAt(segment, time);
  if (segment == null || index == null) return null;
  return segmentCues(segment)[index] ?? null;
}
