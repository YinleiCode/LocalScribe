import clsx from "clsx";
import { useTasks, type Task, type TaskStage } from "../stores/tasks-store";
import { fmtDuration } from "../lib/format";
import { cancelCorrection, cancelTask, pauseCorrection, resumeCorrection, retryTaskPersistence } from "../hooks/usePipeline";
import {
  Check,
  Close,
  FileText,
  Hourglass,
  Pause,
  Play,
  Refresh,
  Trash,
  Warning,
} from "./Icons";

const STAGE_LABEL: Record<TaskStage, string> = {
  queued: "等待",
  transcribing: "转录",
  diarizing: "分人",
  saving: "保存",
  transcribed: "转录完成",
  correcting: "校对",
  correcting_paused: "暂停",
  corrected: "校对完成",
  polishing: "排版",
  polished: "完成",
  translating: "翻译",
  translated: "翻译完成",
  error: "失败",
  cancelled: "取消",
};

const STAGE_COLOR: Record<TaskStage, string> = {
  queued: "text-fg-mute",
  transcribing: "text-accent",
  diarizing: "text-accent",
  saving: "text-accent",
  transcribed: "text-fg-dim",
  correcting: "text-warn",
  correcting_paused: "text-warn/70",
  corrected: "text-fg-dim",
  polishing: "text-ok",
  polished: "text-ok",
  translating: "text-violet-400",
  translated: "text-fg-dim",
  error: "text-err",
  cancelled: "text-fg-mute",
};

function StageIcon({ stage, className }: { stage: TaskStage; className?: string }) {
  if (stage === "transcribing" || stage === "diarizing" || stage === "saving" || stage === "correcting" || stage === "polishing" || stage === "translating")
    return <Hourglass size={11} className={className} />;
  if (stage === "transcribed" || stage === "corrected" || stage === "polished" || stage === "translated")
    return <Check size={11} className={className} />;
  if (stage === "correcting_paused") return <Pause size={11} className={className} />;
  if (stage === "error") return <Warning size={11} className={className} />;
  return <span className={`inline-block w-2 h-2 rounded-full bg-current ${className ?? ""}`} />;
}

function progressPct(t: Task): number {
  if (!t.progress.total) return 0;
  return Math.min(100, Math.round((t.progress.current / t.progress.total) * 100));
}

const RUNNING_STAGES: TaskStage[] = [
  "transcribing",
  "diarizing",
  "saving",
  "correcting",
  "polishing",
  "translating",
];
function isRunning(t: Task): boolean {
  return RUNNING_STAGES.includes(t.stage);
}

/** 排序优先级:跑动中 > 等待 > 已完成 > 失败/取消 */
function stagePriority(s: TaskStage): number {
  if (RUNNING_STAGES.includes(s)) return 0;
  if (s === "queued") return 1;
  if (s === "correcting_paused") return 1;
  if (s === "polished" || s === "corrected" || s === "transcribed" || s === "translated") return 2;
  return 3; // error / cancelled
}

export default function TaskQueue() {
  const tasks = useTasks((s) => s.tasks);
  const activeId = useTasks((s) => s.activeId);
  const setActive = useTasks((s) => s.setActive);
  const remove = useTasks((s) => s.remove);
  const clearAll = useTasks((s) => s.clearAll);

  // 队列只显示"在跑 / 等待 / 失败"的任务。已完成(transcribed/corrected/polished/translated)
  // 自动从队列消失 — 它们已经在"历史库"里了,避免双重显示混淆
  const FINISHED: TaskStage[] = ["polished", "corrected", "transcribed", "translated"];
  const visible = tasks.filter((t) => !FINISHED.includes(t.stage));

  if (visible.length === 0) {
    return (
      <div className="px-3 py-2 text-ui-sm text-fg-mute">
        {tasks.length > 0
          ? "全部完成 — 在下方「历史库」查看"
          : "还没有任务。从上方拖入或选择文件。"}
      </div>
    );
  }

  // 跑动中 → 队列顶部;同优先级里新的在前
  const sorted = [...visible].sort((a, b) => {
    const pa = stagePriority(a.stage);
    const pb = stagePriority(b.stage);
    if (pa !== pb) return pa - pb;
    return (b.createdAt ?? 0) - (a.createdAt ?? 0);
  });

  return (
    <div className="text-ui">
      {sorted.map((t) => {
        const pct = progressPct(t);
        const isActive = activeId === t.id;
        const running = isRunning(t);
        const showProgress = running;
        return (
          <div
            key={t.id}
            onClick={() => setActive(t.id)}
            className={clsx(
              "list-item flex-col items-stretch py-1.5 relative",
              isActive && "list-item-active",
              running && !isActive && "bg-accent/10",
              running && "shadow-[inset_3px_0_0_0_theme(colors.accent.DEFAULT)]",
            )}
          >
            {running && (
              <span
                className="absolute top-1.5 right-2 inline-block w-1.5 h-1.5 rounded-full bg-accent animate-pulse"
                title="正在处理"
              />
            )}
            <div className="flex items-center gap-2 min-w-0">
              <FileText size={12} className="text-fg-mute shrink-0" />
              <span className="flex-1 min-w-0 truncate font-mono text-ui">{t.filename}</span>
              <span className={clsx("flex items-center gap-1 text-ui-sm shrink-0", STAGE_COLOR[t.stage])}>
                <StageIcon stage={t.stage} />
                {STAGE_LABEL[t.stage]}
                {showProgress && ` ${pct}%`}
              </span>
            </div>

            {showProgress && (
              <div className="mt-1 h-[3px] rounded bg-border/60 overflow-hidden">
                <div className="h-full bg-accent transition-all" style={{ width: `${pct}%` }} />
              </div>
            )}

            {t.progress.preview && (t.stage === "transcribing" || t.stage === "saving") && (
              <div className="mt-1 text-ui-sm text-fg-mute truncate">{t.progress.preview}</div>
            )}

            {t.result && (
              <>
                <div className="mt-0.5 text-ui-sm text-fg-mute">
                  {fmtDuration(t.result.duration)} · {t.result.segments.length} 段 · RTF {t.result.rtf.toFixed(3)}
                </div>
                {t.result.filter_stats && (t.result.filter_stats.removed_total ?? 0) > 0 && (
                  <div className="mt-0.5 text-ui-sm text-warn/80" title={JSON.stringify(t.result.filter_stats, null, 2)}>
                    已过滤幻觉 {t.result.filter_stats.removed_total} 段
                    {t.result.filter_stats.vad ? ` · VAD ${t.result.filter_stats.vad}` : ""}
                    {t.result.filter_stats.logprob ? ` · 低置信 ${t.result.filter_stats.logprob}` : ""}
                    {t.result.filter_stats.repetition ? ` · 重复 ${t.result.filter_stats.repetition}` : ""}
                    {t.result.filter_stats.phrases ? ` · 黑词 ${t.result.filter_stats.phrases}` : ""}
                    {t.result.filter_stats.density ? ` · 密度异常 ${t.result.filter_stats.density}` : ""}
                    {t.result.filter_stats.similarity ? ` · 相似 ${t.result.filter_stats.similarity}` : ""}
                  </div>
                )}
              </>
            )}

            {t.error && (
              <div className="mt-1 text-ui-sm text-err truncate" title={t.error}>{t.error}</div>
            )}

            <div className="mt-1 flex items-center gap-0.5">
              {t.stage === "correcting" && (
                <>
                  <ToolbarBtn icon={<Pause size={11} />} label="暂停" onClick={() => pauseCorrection(t.id)} />
                  <ToolbarBtn icon={<Close size={11} />} label="取消" onClick={() => cancelCorrection(t.id)} className="hover:text-err" />
                </>
              )}
              {t.stage === "correcting_paused" && (
                <>
                  <ToolbarBtn icon={<Play size={11} />} label="继续" onClick={() => resumeCorrection(t.id)} className="text-ok hover:text-ok" />
                  <ToolbarBtn icon={<Close size={11} />} label="取消" onClick={() => cancelCorrection(t.id)} className="hover:text-err" />
                </>
              )}
              {(t.stage === "transcribing" || t.stage === "diarizing") && (
                <ToolbarBtn
                  icon={<Close size={11} />}
                  label="取消"
                  onClick={() => cancelTask(t.id)}
                  className="hover:text-err"
                />
              )}
              {t.stage === "error" && t.result && (
                <ToolbarBtn
                  icon={<Refresh size={11} />}
                  label="重试保存"
                  onClick={() => void retryTaskPersistence(t.id)}
                  className="text-accent hover:text-accent"
                />
              )}
              {t.stage === "queued" && (
                <ToolbarBtn
                  icon={<Close size={11} />}
                  label="取消"
                  onClick={() => cancelTask(t.id)}
                  className="hover:text-err"
                />
              )}
              <span className="flex-1" />
              <ToolbarBtn
                icon={<Trash size={11} />}
                label="删除"
                onClick={() => remove(t.id)}
                className="hover:text-err"
              />
            </div>
          </div>
        );
      })}

      {tasks.length > 1 && (
        <div className="px-3 py-1.5 border-t border-border/60">
          <button onClick={clearAll} className="btn-ghost h-6 text-ui-sm">
            <Trash size={11} /> 清空队列
          </button>
        </div>
      )}
    </div>
  );
}

function ToolbarBtn({
  icon,
  label,
  onClick,
  className,
}: {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  className?: string;
}) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className={clsx("flex items-center gap-1 px-1.5 h-5 text-ui-sm text-fg-mute hover:text-fg rounded-sm hover:bg-hover", className)}
      title={label}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}
