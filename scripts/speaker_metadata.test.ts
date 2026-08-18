import assert from "node:assert/strict";
import test from "node:test";

import type { Segment } from "../src/lib/ipc.ts";
import { syncSpeakerMetadata } from "../src/lib/speaker-metadata.ts";

const cues = [
  { start: 0, end: 2, text: "原始第一句。" },
  { start: 2, end: 4, text: "原始第二句。" },
];

function splitSource(): Segment[] {
  return [
    {
      start: 0,
      end: 2,
      text: cues[0].text,
      sync_cues: [structuredClone(cues[0])],
      speaker: "SPEAKER_A",
      speaker_confidence: 0.91,
      speaker_resegmented: true,
    },
    {
      start: 2,
      end: 4,
      text: cues[1].text,
      sync_cues: [structuredClone(cues[1])],
      speaker: "SPEAKER_B",
      speaker_confidence: 0.88,
      speaker_resegmented: true,
    },
  ];
}

function correctedTarget(): Segment {
  return {
    start: 0,
    end: 4,
    text: "人工校对后的第一句。人工校对后的第二句。",
    original_text: cues.map((cue) => cue.text).join(""),
    sync_cues: structuredClone(cues),
    speaker: "OLD_SPEAKER",
  };
}

test("projects a safe source split onto an unchanged corrected parent", () => {
  const target = correctedTarget();
  const before = structuredClone(target);
  const output = syncSpeakerMetadata(splitSource(), [target], { requireFullMatch: true });

  assert.equal(output.length, 1);
  assert.equal(output[0].text, before.text);
  assert.equal(output[0].original_text, before.original_text);
  assert.deepEqual(output[0].sync_cues, before.sync_cues);
  assert.deepEqual(output[0].speaker_cues?.map((cue) => cue.speaker), ["SPEAKER_A", "SPEAKER_B"]);
  assert.deepEqual(output[0].speaker_cues?.map((cue) => [cue.start, cue.end]), [[0, 2], [2, 4]]);
});

test("keeps exact-range metadata synchronization compatible", () => {
  const source: Segment[] = [{
    start: 0,
    end: 4,
    text: "原文",
    speaker: "SPEAKER_C",
    speaker_confidence: 0.95,
  }];
  const target: Segment[] = [{ start: 0, end: 4, text: "校对文字" }];
  const output = syncSpeakerMetadata(source, target, { requireFullMatch: true });
  assert.equal(output[0].text, "校对文字");
  assert.equal(output[0].speaker, "SPEAKER_C");
  assert.equal(output[0].speaker_confidence, 0.95);
});

test("rejects split projection without complete immutable cues", () => {
  const withoutCues = correctedTarget();
  delete withoutCues.sync_cues;
  assert.throws(
    () => syncSpeakerMetadata(splitSource(), [withoutCues], { requireFullMatch: true }),
    /时间轴与原文不一致/,
  );

  const changedCue = correctedTarget();
  changedCue.sync_cues![1].text = "被改动的 cue";
  assert.throws(
    () => syncSpeakerMetadata(splitSource(), [changedCue], { requireFullMatch: true }),
    /时间轴与原文不一致/,
  );
});

test("rejects source split points that are not target cue boundaries", () => {
  const source = splitSource();
  source[0].end = 1.5;
  source[0].sync_cues![0].end = 1.5;
  source[1].start = 1.5;
  source[1].sync_cues![0].start = 1.5;
  assert.throws(
    () => syncSpeakerMetadata(source, [correctedTarget()], { requireFullMatch: true }),
    /时间轴与原文不一致/,
  );
});
