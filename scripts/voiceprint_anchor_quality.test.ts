import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_VOICEPRINT_ANCHOR_OVERLAP_RATIO,
  hasVoiceprintAnchorOverlapRisk,
} from "../src/lib/voiceprint-anchor-quality.ts";

test("rejects an anchor when OSD overlap exceeds the backend quality gate", () => {
  assert.equal(hasVoiceprintAnchorOverlapRisk({
    start: 13.699,
    end: 26.246,
    text: "清晰长段",
    overlap_ratio: 0.0277,
  }), true);
});

test("uses the same strict overlap threshold as the backend", () => {
  assert.equal(hasVoiceprintAnchorOverlapRisk({
    start: 0,
    end: 10,
    text: "边界样本",
    overlap_ratio: MAX_VOICEPRINT_ANCHOR_OVERLAP_RATIO,
  }), false);
});

test("rejects explicit overlap risk and second-speaker candidates", () => {
  assert.equal(hasVoiceprintAnchorOverlapRisk({
    start: 0,
    end: 10,
    text: "重叠风险",
    speaker_overlap_risk: true,
  }), true);
  assert.equal(hasVoiceprintAnchorOverlapRisk({
    start: 0,
    end: 10,
    text: "第二说话人候选",
    speaker_overlap_candidates: [{
      start: 1,
      end: 2,
      primary_speaker: "SPEAKER_A",
      secondary_speaker: "SPEAKER_B",
      confidence: 0.8,
      window_ratio: 0.2,
      context_score: 0.7,
      candidate_score: 0.75,
      source: "osd_campp_context_v1",
    }],
  }), true);
});

test("keeps a clean segment eligible", () => {
  assert.equal(hasVoiceprintAnchorOverlapRisk({
    start: 0,
    end: 10,
    text: "无重叠片段",
    overlap_ratio: 0,
  }), false);
});
