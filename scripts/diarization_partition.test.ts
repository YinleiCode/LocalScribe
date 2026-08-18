import assert from "node:assert/strict";
import test from "node:test";

import {
  assertDiarizationPreservesTranscript,
  type DiarizationTranscriptSegment as Segment,
} from "../src/lib/diarization-partition.ts";

const source: Segment[] = [{
  start: 0,
  end: 8,
  text: "第一位发言，第二位回答。",
  sync_cues: [
    { start: 0, end: 4, text: "第一位发言，" },
    { start: 4, end: 8, text: "第二位回答。" },
  ],
}];

function safePartition(): Segment[] {
  return [
    {
      start: 0,
      end: 4,
      text: "第一位发言，",
      speaker: "SPEAKER_A",
      sync_cues: [{ start: 0, end: 4, text: "第一位发言，" }],
    },
    {
      start: 4,
      end: 8,
      text: "第二位回答。",
      speaker: "SPEAKER_B",
      sync_cues: [{ start: 4, end: 8, text: "第二位回答。" }],
    },
  ];
}

test("accepts a diarization-only split at immutable cue boundaries", () => {
  assert.doesNotThrow(() => assertDiarizationPreservesTranscript(source, safePartition()));
});

test("rejects text, cue, or time coverage changes", () => {
  const textChanged = structuredClone(safePartition());
  textChanged[1].text += "错";
  assert.throws(() => assertDiarizationPreservesTranscript(source, textChanged), /改变了.*文字/);

  const cueChanged = structuredClone(safePartition());
  cueChanged[1].sync_cues![0].start += 0.1;
  assert.throws(
    () => assertDiarizationPreservesTranscript(source, cueChanged),
    /同步 cue|cue 原子边界/,
  );

  const coverageChanged = structuredClone(safePartition());
  coverageChanged[1].start += 0.1;
  assert.throws(() => assertDiarizationPreservesTranscript(source, coverageChanged), /时间覆盖/);
});

test("rejects splitting a legacy segment without sync cues", () => {
  const legacySource: Segment[] = [{ start: 0, end: 2, text: "完整原文" }];
  const unsafe: Segment[] = [
    { start: 0, end: 1, text: "完整", speaker: "SPEAKER_A" },
    { start: 1, end: 2, text: "原文", speaker: "SPEAKER_B" },
  ];
  assert.throws(
    () => assertDiarizationPreservesTranscript(legacySource, unsafe),
    /没有精确 cue/,
  );
});

test("rejects assigning a cue across a split-piece boundary", () => {
  const unsafe = safePartition();
  unsafe[0].sync_cues = structuredClone(source[0].sync_cues);
  unsafe[1].sync_cues = [];
  assert.throws(
    () => assertDiarizationPreservesTranscript(source, unsafe),
    /cue 原子边界/,
  );
});
