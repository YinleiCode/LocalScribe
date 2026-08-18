import type { Segment } from "./ipc";

const RANGE_TOLERANCE_S = 0.001;

const SPEAKER_METADATA_KEYS = [
  "speaker",
  "speaker_confidence",
  "speaker_votes",
  "speaker_subsegments",
  "speaker_change_points",
  "speaker_cues",
  "speaker_cue_embeddings",
  "speaker_cue_review",
  "speaker_cue_mode",
  "speaker_cue_split",
  "speaker_overlap_risk",
  "overlap_ratio",
  "speaker_overlap_confidence",
  "speaker_overlap_candidates",
  "speaker_overlap_ratio",
  "speaker_resegmented",
  "speaker_resegmentation_review",
  "speaker_handoff_review",
  "speaker_assignment_review",
  "speaker_review_reason",
  "speaker_handoff_voice_guard_repaired",
  "voice_pitch_hz",
  "voice_pitch_confidence",
  "voice_band",
  "speaker_handoff_split",
  "speaker_handoff_bridge",
  "speaker_handoff_relabel",
  "speaker_handoff_text_review",
  "speaker_split_from_index",
  "voice_band_repaired",
  "continuity_repaired",
  "original_speaker",
  "speaker_voiceprint_reidentified",
  "speaker_voiceprint_review",
  "speaker_voiceprint_score",
  "speaker_voiceprint_anchor",
  "speaker_calibrated",
  "speaker_calibration_source",
  "voice_line_refined",
  "voice_line_review",
] as const satisfies readonly (keyof Segment)[];

type SyncSpeakerMetadataOptions = {
  requireFullMatch?: boolean;
};

function sameTime(left: number, right: number): boolean {
  return Number.isFinite(left)
    && Number.isFinite(right)
    && Math.abs(left - right) <= RANGE_TOLERANCE_S;
}

function copySpeakerMetadata(source: Segment, target: Segment): Segment {
  const next = { ...target } as Segment;
  const mutable = next as unknown as Record<string, unknown>;
  const sourceRecord = source as unknown as Record<string, unknown>;
  for (const key of SPEAKER_METADATA_KEYS) {
    const value = sourceRecord[key];
    if (value === undefined) delete mutable[key];
    else mutable[key] = value;
  }
  return next;
}

function immutableCuePartition(
  pieces: Segment[],
  target: Segment,
): NonNullable<Segment["speaker_cues"]> | null {
  const targetCues = target.sync_cues;
  if (!targetCues?.length || pieces.length < 2) return null;
  if (!sameTime(targetCues[0].start, target.start)) return null;
  if (!sameTime(targetCues[targetCues.length - 1].end, target.end)) return null;

  for (let index = 0; index < targetCues.length; index += 1) {
    const cue = targetCues[index];
    if (!Number.isFinite(cue.start) || !Number.isFinite(cue.end) || cue.end <= cue.start) return null;
    if (index > 0 && !sameTime(cue.start, targetCues[index - 1].end)) return null;
  }

  const sourceCues = pieces.flatMap((piece) => piece.sync_cues ?? []);
  if (sourceCues.length !== targetCues.length) return null;
  for (let index = 0; index < targetCues.length; index += 1) {
    const sourceCue = sourceCues[index];
    const targetCue = targetCues[index];
    if (
      !sameTime(sourceCue.start, targetCue.start)
      || !sameTime(sourceCue.end, targetCue.end)
      || sourceCue.text !== targetCue.text
    ) return null;
  }

  const projected: NonNullable<Segment["speaker_cues"]> = [];
  let pieceIndex = 0;
  for (let cueIndex = 0; cueIndex < targetCues.length; cueIndex += 1) {
    const cue = targetCues[cueIndex];
    while (
      pieceIndex < pieces.length - 1
      && cue.start >= pieces[pieceIndex].end - RANGE_TOLERANCE_S
    ) pieceIndex += 1;
    const piece = pieces[pieceIndex];
    if (
      !piece.speaker
      || cue.start < piece.start - RANGE_TOLERANCE_S
      || cue.end > piece.end + RANGE_TOLERANCE_S
    ) return null;
    projected.push({
      cue_index: cueIndex,
      start: cue.start,
      end: cue.end,
      speaker: piece.speaker,
      confidence: Number.isFinite(piece.speaker_confidence)
        ? Number(piece.speaker_confidence)
        : 0,
      source: "cue_partition_projection",
    });
  }

  const cueBoundaries = new Set(targetCues.slice(0, -1).map((cue) => cue.end));
  if (!pieces.slice(0, -1).every((piece) => (
    Array.from(cueBoundaries).some((boundary) => sameTime(piece.end, boundary))
  ))) return null;
  return projected;
}

function partitionForTarget(
  source: Segment[],
  sourceIndex: number,
  target: Segment,
): { pieces: Segment[]; nextSourceIndex: number } | null {
  const first = source[sourceIndex];
  if (!first || !sameTime(first.start, target.start)) return null;
  const pieces: Segment[] = [];
  let expectedStart = target.start;
  let index = sourceIndex;
  while (index < source.length) {
    const piece = source[index];
    if (
      !sameTime(piece.start, expectedStart)
      || piece.end > target.end + RANGE_TOLERANCE_S
      || piece.end <= piece.start
    ) return null;
    pieces.push(piece);
    expectedStart = piece.end;
    index += 1;
    if (sameTime(expectedStart, target.end)) {
      return { pieces, nextSourceIndex: index };
    }
  }
  return null;
}

function unmatchedSegment(target: Segment, requireFullMatch: boolean): Segment {
  if (requireFullMatch) {
    throw new Error("校对稿时间轴与原文不一致，已拒绝重新分人以避免 speaker 信息错位");
  }
  return target;
}

export function syncSpeakerMetadata(
  source: Segment[],
  target: Segment[],
  options: SyncSpeakerMetadataOptions = {},
): Segment[] {
  const requireFullMatch = options.requireFullMatch === true;
  let sourceIndex = 0;
  const synchronized = target.map((targetSegment) => {
    const partition = partitionForTarget(source, sourceIndex, targetSegment);
    if (!partition) return unmatchedSegment(targetSegment, requireFullMatch);
    sourceIndex = partition.nextSourceIndex;

    if (partition.pieces.length === 1) {
      return copySpeakerMetadata(partition.pieces[0], targetSegment);
    }

    const speakerCues = immutableCuePartition(partition.pieces, targetSegment);
    if (!speakerCues) return unmatchedSegment(targetSegment, requireFullMatch);
    return {
      ...copySpeakerMetadata(partition.pieces[0], targetSegment),
      speaker_cues: speakerCues,
      speaker_cue_review: false,
    };
  });

  if (requireFullMatch && sourceIndex !== source.length) {
    throw new Error("分人结果包含无法对应校对稿的额外片段，已拒绝应用");
  }
  return synchronized;
}
