export type DiarizationTranscriptSegment = {
  start: number;
  end: number;
  text: string;
  sync_cues?: Array<{ start: number; end: number; text: string }>;
};

export function assertDiarizationPreservesTranscript(
  source: DiarizationTranscriptSegment[],
  output: DiarizationTranscriptSegment[],
): void {
  const tolerance = 1e-6;
  let outputIndex = 0;
  for (let sourceIndex = 0; sourceIndex < source.length; sourceIndex += 1) {
    const before = source[sourceIndex];
    const pieces: DiarizationTranscriptSegment[] = [];
    let expectedStart = before.start;
    while (outputIndex < output.length) {
      const piece = output[outputIndex];
      if (
        piece.end <= piece.start
        || Math.abs(piece.start - expectedStart) > tolerance
        || piece.end > before.end + tolerance
      ) {
        throw new Error(`分人结果改变了第 ${sourceIndex + 1} 段的时间覆盖，已拒绝应用`);
      }
      pieces.push(piece);
      outputIndex += 1;
      expectedStart = piece.end;
      if (Math.abs(piece.end - before.end) <= tolerance) break;
    }
    if (pieces.length === 0 || Math.abs(expectedStart - before.end) > tolerance) {
      throw new Error(`分人结果没有完整覆盖第 ${sourceIndex + 1} 段，已拒绝应用`);
    }
    if (pieces.map((piece) => piece.text).join("") !== before.text) {
      throw new Error(`分人结果改变了第 ${sourceIndex + 1} 段文字，已拒绝应用`);
    }
    if (pieces.length > 1 && !before.sync_cues?.length) {
      throw new Error(`分人结果试图拆分没有精确 cue 的第 ${sourceIndex + 1} 段，已拒绝应用`);
    }
    if (pieces.length > 1 && pieces.some((piece) => {
      const cues = piece.sync_cues ?? [];
      return cues.length === 0
        || Math.abs(cues[0].start - piece.start) > tolerance
        || Math.abs(cues[cues.length - 1].end - piece.end) > tolerance;
    })) {
      throw new Error(`分人结果没有按第 ${sourceIndex + 1} 段的 cue 原子边界拆分，已拒绝应用`);
    }
    const outputCues = pieces.flatMap((piece) => piece.sync_cues ?? []);
    if (JSON.stringify(outputCues) !== JSON.stringify(before.sync_cues ?? [])) {
      throw new Error(`分人结果改变了第 ${sourceIndex + 1} 段同步 cue，已拒绝应用`);
    }
  }
  if (outputIndex !== output.length) {
    throw new Error(`分人结果包含 ${output.length - outputIndex} 个多余片段，已拒绝应用`);
  }
}
