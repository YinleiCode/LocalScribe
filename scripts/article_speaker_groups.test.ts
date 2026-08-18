import assert from "node:assert/strict";
import test from "node:test";

import type { Segment } from "../src/lib/ipc.ts";
import { groupArticleSpeakerTurns } from "../src/lib/article-speaker-groups.ts";

test("merges consecutive turns from the same speaker", () => {
  const segments: Segment[] = [
    { start: 0, end: 2, text: "第一句。", speaker: "SPEAKER_A" },
    { start: 2.2, end: 4, text: "第二句。", speaker: "A" },
    { start: 4, end: 5, text: "我来回答。", speaker: "SPEAKER_B" },
  ];

  const groups = groupArticleSpeakerTurns(segments);

  assert.deepEqual(groups.map((group) => [group.speaker, group.start, group.end, group.text]), [
    ["SPEAKER_A", 0, 4, "第一句。第二句。"],
    ["SPEAKER_B", 4, 5, "我来回答。"],
  ]);
  assert.deepEqual(groups[0].cues, [
    { start: 0, end: 2, text: "第一句。" },
    { start: 2.2, end: 4, text: "第二句。" },
  ]);
});

test("does not merge a speaker across another speaker's turn", () => {
  const segments: Segment[] = [
    { start: 0, end: 1, text: "A一。", speaker: "SPEAKER_A" },
    { start: 1, end: 2, text: "B一。", speaker: "SPEAKER_B" },
    { start: 2, end: 3, text: "A二。", speaker: "SPEAKER_A" },
  ];

  assert.deepEqual(groupArticleSpeakerTurns(segments).map((group) => group.speaker), [
    "SPEAKER_A",
    "SPEAKER_B",
    "SPEAKER_A",
  ]);
});

test("splits a source segment when its exact speaker cues switch people", () => {
  const segments: Segment[] = [{
    start: 10,
    end: 14,
    text: "第一位发言，第二位回答。",
    speaker: "SPEAKER_A",
    speaker_cues: [
      { cue_index: 0, start: 10, end: 12, text: "第一位发言，", speaker: "SPEAKER_A", confidence: 0.9, source: "test" },
      { cue_index: 1, start: 12, end: 14, text: "第二位回答。", speaker: "SPEAKER_B", confidence: 0.9, source: "test" },
    ],
  }];

  assert.deepEqual(groupArticleSpeakerTurns(segments).map((group) => [group.speaker, group.start, group.end, group.text]), [
    ["SPEAKER_A", 10, 12, "第一位发言，"],
    ["SPEAKER_B", 12, 14, "第二位回答。"],
  ]);
});

test("keeps unlabeled speech unlabeled instead of inventing a speaker", () => {
  const groups = groupArticleSpeakerTurns([
    { start: 0, end: 1, text: "未分人的内容。" },
  ]);

  assert.equal(groups.length, 1);
  assert.equal(groups[0].speaker, undefined);
});

test("splits article highlighting phrases at commas and sentence endings", () => {
  const groups = groupArticleSpeakerTurns([{
    start: 0,
    end: 4,
    text: "第一句，第二句。",
    speaker: "SPEAKER_A",
  }]);

  assert.deepEqual(groups[0].cues.map((cue) => cue.text), ["第一句，", "第二句。"]);
  assert.equal(groups[0].cues[0].start, 0);
  assert.equal(groups[0].cues[1].end, 4);
  assert.ok(groups[0].cues[0].end <= groups[0].cues[1].start);
});
