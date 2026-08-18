import { FileText, Refresh, Warning } from "./Icons";
import type { LibraryMeta } from "../lib/ipc";
import { fmtDuration } from "../lib/format";

export type DuplicateChoice = "load" | "redo" | "cancel";

type Props = {
  filename: string;
  existing: LibraryMeta;
  onChoose: (choice: DuplicateChoice) => void;
};

export default function DuplicateFileDialog({ filename, existing, onChoose }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-sidebar border border-border rounded-sm shadow-2xl max-w-lg w-full m-6">
        <header className="flex items-center gap-2 px-4 py-3 border-b border-border">
          <Warning size={16} className="text-warn" />
          <h2 className="text-ui-lg font-medium text-fg">同名文件已存在</h2>
        </header>

        <div className="px-4 py-4 space-y-3 text-ui">
          <div className="text-fg-dim leading-relaxed">
            历史库里已有同名转录:<span className="font-mono text-fg ml-1">{filename}</span>
          </div>

          <div className="bg-editor/60 border border-border rounded-sm p-3 text-ui-sm space-y-1">
            <div className="flex items-center gap-2">
              <FileText size={12} className="text-fg-mute" />
              <span className="font-mono text-fg-dim">{existing.stem}</span>
            </div>
            <div className="text-fg-mute pl-5 space-y-0.5">
              <div>
                {fmtDuration(existing.duration)} · {existing.segments} 段 · {existing.backend}
              </div>
              <div>
                创建于 {new Date(existing.created_at * 1000).toLocaleString()}
                {existing.has_corrected && " · 已校对"}
                {existing.has_polished && " · 已排版"}
              </div>
            </div>
          </div>

          <div className="text-ui-sm text-fg-mute leading-relaxed">
            选择如何处理这次上传:
          </div>
        </div>

        <div className="flex gap-2 px-4 py-3 border-t border-border bg-editor/40">
          <button
            onClick={() => onChoose("cancel")}
            className="btn-ghost"
          >
            取消
          </button>
          <span className="flex-1" />
          <button
            onClick={() => onChoose("load")}
            className="btn-ghost"
            title="不重新转录,直接打开已有的结果"
          >
            <FileText size={12} />
            <span>载入已有</span>
          </button>
          <button
            onClick={() => onChoose("redo")}
            className="btn"
            title="把旧的转录归档为带时间戳的目录,然后重新跑一次"
          >
            <Refresh size={12} />
            <span>归档旧的 + 重新转录</span>
          </button>
        </div>
      </div>
    </div>
  );
}
