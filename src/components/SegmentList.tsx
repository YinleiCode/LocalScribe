import { fmtTs } from "../lib/format";
import type { Segment } from "../lib/ipc";

type Props = {
  segments: Segment[];
  showDiff?: boolean;
  emptyHint?: string;
};

// VSCode-friendly palette for up to 8 speakers (循环复用,够多人会议)
const SPEAKER_COLORS = [
  "text-sky-300 border-sky-300/40 bg-sky-500/10",
  "text-orange-300 border-orange-300/40 bg-orange-500/10",
  "text-emerald-300 border-emerald-300/40 bg-emerald-500/10",
  "text-pink-300 border-pink-300/40 bg-pink-500/10",
  "text-violet-300 border-violet-300/40 bg-violet-500/10",
  "text-yellow-300 border-yellow-300/40 bg-yellow-500/10",
  "text-cyan-300 border-cyan-300/40 bg-cyan-500/10",
  "text-rose-300 border-rose-300/40 bg-rose-500/10",
];

function speakerClass(speakers: string[], who: string): string {
  const idx = speakers.indexOf(who);
  return SPEAKER_COLORS[idx >= 0 ? idx % SPEAKER_COLORS.length : 0];
}

function speakerReviewReason(s: Segment): string {
  const reasons: string[] = [];
  if (s.speaker_handoff_review || s.speaker_handoff_text_review) reasons.push("换人边界待确认");
  if (s.speaker_resegmentation_review) reasons.push("段内短声纹待确认");
  if (s.speaker_assignment_review) reasons.push(s.speaker_review_reason || "分人待确认");
  if (s.voice_band_repaired) reasons.push("已按局部声线纠偏");
  if (s.continuity_repaired) reasons.push("已按语义连续性纠偏");
  if (s.speaker_handoff_voice_guard_repaired) reasons.push("已按声纹护栏纠偏");
  if (s.speaker_overlap_risk) reasons.push("可能有重叠/插话");
  return reasons.join(" / ");
}

export default function SegmentList({ segments, showDiff = false, emptyHint }: Props) {
  if (!segments.length) {
    return (
      <div className="text-sm text-text-mute py-12 text-center">
        {emptyHint ?? "(暂无内容)"}
      </div>
    );
  }
  // 收集所有说话人,按出现顺序固定颜色
  const speakers: string[] = [];
  for (const s of segments) {
    if (s.speaker && !speakers.includes(s.speaker)) speakers.push(s.speaker);
  }
  const showSpeakers = speakers.length > 0;

  return (
    <ul className="space-y-1.5 font-mono text-sm leading-relaxed">
      {segments.map((s, i) => {
        const changed = showDiff && s.original_text && s.original_text !== s.text;
        const reviewReason = speakerReviewReason(s);
        return (
          <li key={i} className="flex gap-3">
            <span className="text-xs text-text-mute pt-0.5 whitespace-nowrap shrink-0">
              {fmtTs(s.start)}
            </span>
            {showSpeakers && (
              <span
                className={
                  "shrink-0 px-1.5 py-0.5 rounded-sm border text-xs font-medium whitespace-nowrap inline-flex items-center gap-1 " +
                  (reviewReason
                    ? "text-yellow-300 border-yellow-300/60 bg-yellow-500/10"
                    : s.speaker ? speakerClass(speakers, s.speaker) : "text-text-mute border-bg-border")
                }
                title={reviewReason || "说话人"}
              >
                <span>{s.speaker ?? "?"}</span>
                {reviewReason && <span className="text-[10px] font-normal">待确认</span>}
              </span>
            )}
            <div className="flex-1 min-w-0">
              {changed && (
                <div className="text-xs text-red-400/70 line-through truncate">
                  {s.original_text}
                </div>
              )}
              <div className={changed ? "text-emerald-300" : "text-text"}>
                {s.text}
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
