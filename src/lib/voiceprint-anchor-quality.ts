import type { Segment } from "./ipc";

export const MAX_VOICEPRINT_ANCHOR_OVERLAP_RATIO = 0.02;

export function hasVoiceprintAnchorOverlapRisk(segment: Segment): boolean {
  const overlapRatio = Number(segment.overlap_ratio);
  return Boolean(
    segment.speaker_overlap_risk
    || segment.speaker_overlap_candidates?.length
    || (Number.isFinite(overlapRatio) && overlapRatio > MAX_VOICEPRINT_ANCHOR_OVERLAP_RATIO)
  );
}
