import assert from "node:assert/strict";
import test from "node:test";

import {
  activeCueIndexAt,
  activeSegmentIndexAt,
  segmentCues,
  type TranscriptSyncSegment,
} from "../src/lib/transcript-sync.ts";

test("does not keep the previous segment highlighted through a long silence", () => {
  const segments: TranscriptSyncSegment[] = [
    { start: 0, end: 1, text: "第一句。" },
    { start: 4, end: 5, text: "第二句。" },
  ];
  assert.equal(activeSegmentIndexAt(segments, 1.1), 0);
  assert.equal(activeSegmentIndexAt(segments, 2), null);
  assert.equal(activeSegmentIndexAt(segments, 3.8), null);
  assert.equal(activeSegmentIndexAt(segments, 4), 1);
});

test("does not keep the last cue active after its bounded hold", () => {
  const segment: TranscriptSyncSegment = {
    start: 0,
    end: 2,
    text: "第一句，第二句。",
    sync_cues: [
      { start: 0, end: 1, text: "第一句，" },
      { start: 1, end: 2, text: "第二句。" },
    ],
  };
  assert.equal(activeCueIndexAt(segment, 1.5), 1);
  assert.equal(activeCueIndexAt(segment, 2.1), 1);
  assert.equal(activeCueIndexAt(segment, 2.3), null);
});

test("does not bridge a long acoustic gap between cues in one segment", () => {
  const segment: TranscriptSyncSegment = {
    start: 175.035,
    end: 192.026,
    text: "等下真没有吧。",
    sync_cues: [
      { start: 175.035, end: 175.155, text: "等" },
      { start: 188.562, end: 192.026, text: "下真没有吧。" },
    ],
  };

  assert.equal(activeCueIndexAt(segment, 175.1), 0);
  assert.equal(activeCueIndexAt(segment, 180), null);
  assert.equal(activeCueIndexAt(segment, 188.7), 1);
});

test("renders unreliable timing text without automatically highlighting it", () => {
  const segment: TranscriptSyncSegment = {
    start: 10,
    end: 16,
    text: "时间依据不足。",
    sync_cues: [
      { start: 10, end: 16, text: "时间依据不足。", reliable: false },
    ],
  };

  assert.equal(segmentCues(segment).map((cue) => cue.text).join(""), segment.text);
  assert.equal(activeCueIndexAt(segment, 12), null);
});

test("uses exact cue text when it already matches the displayed segment", () => {
  const cues = segmentCues({
    start: 10,
    end: 12,
    text: "第一句，第二句。",
    sync_cues: [
      { start: 10, end: 11, text: "第一句，" },
      { start: 11, end: 12, text: "第二句。" },
    ],
  });
  assert.deepEqual(cues.map((cue) => cue.text), ["第一句，", "第二句。"]);
});

test("falls back to complete display text for legacy mismatched cue metadata", () => {
  const segment: TranscriptSyncSegment = {
    start: 10,
    end: 12,
    text: "当前完整正文。",
    sync_cues: [
      { start: 10, end: 11, text: "旧文字，" },
      { start: 11, end: 12, text: "跨段内容。" },
    ],
  };
  assert.equal(segmentCues(segment).map((cue) => cue.text).join(""), segment.text);
});

test("projects speaker metadata onto existing sync cues without changing cue geometry", () => {
  const segment: TranscriptSyncSegment = {
    start: 10,
    end: 14,
    text: "第一位发言，第二位回答。",
    sync_cues: [
      { start: 10, end: 12, text: "第一位发言，" },
      { start: 12, end: 14, text: "第二位回答。" },
    ],
    speaker_cues: [
      { cue_index: 0, start: 10, end: 12, speaker: "SPEAKER_D", confidence: 0.91 },
      { cue_index: 1, start: 12, end: 14, speaker: "SPEAKER_B", confidence: 0.88 },
    ],
  };

  const cues = segmentCues(segment);

  assert.deepEqual(cues.map((cue) => cue.speaker), ["SPEAKER_D", "SPEAKER_B"]);
  assert.deepEqual(cues.map((cue) => [cue.start, cue.end]), [[10, 12], [12, 14]]);
  assert.equal(cues.map((cue) => cue.text).join(""), segment.text);
});
