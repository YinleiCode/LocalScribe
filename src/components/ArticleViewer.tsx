import { useEffect, useState } from "react";

import { ipc, type ArticleMeta } from "../lib/ipc";
import { fmtDuration } from "../lib/format";
import { downloadTextFile, openExternalTarget } from "../lib/runtime";
import { Article, Close, Copy, Download, FolderOpen, Pencil } from "./Icons";

type Props = {
  meta: ArticleMeta;
  onClose: () => void;
  onChanged: () => void;
};

export default function ArticleViewer({ meta, onClose, onChanged }: Props) {
  const [content, setContent] = useState<string>("");
  const [body, setBody] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);
  const [newTitle, setNewTitle] = useState(meta.title);

  useEffect(() => {
    (async () => {
      try {
        const raw = await ipc.articleRead(meta.filename);
        setContent(raw);
        // Strip frontmatter for preview
        if (raw.startsWith("---\n")) {
          const end = raw.indexOf("\n---\n", 4);
          if (end !== -1) {
            setBody(raw.slice(end + 5).replace(/^# .+\n+/, "").trim());
            return;
          }
        }
        setBody(raw);
      } catch (e) {
        setError(String(e));
      }
    })();
    setNewTitle(meta.title);
  }, [meta.filename]);

  async function copy() {
    await navigator.clipboard.writeText(body);
  }

  async function exportMd() {
    await downloadTextFile(meta.filename, content);
  }

  async function showInFinder() {
    try {
      await openExternalTarget(meta.path);
    } catch {
      /* noop */
    }
  }

  async function applyRename() {
    if (!newTitle.trim() || newTitle.trim() === meta.title) {
      setRenaming(false);
      return;
    }
    try {
      await ipc.articleRename(meta.filename, newTitle.trim());
      setRenaming(false);
      onChanged();
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      {/* Header */}
      <header className="shrink-0 flex items-center justify-between bg-tabbar border-b border-border px-3 h-9">
        <div className="flex items-center gap-2 min-w-0">
          <Article size={14} className="text-accent shrink-0" />
          {renaming ? (
            <input
              autoFocus
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              onBlur={applyRename}
              onKeyDown={(e) => {
                if (e.key === "Enter") applyRename();
                if (e.key === "Escape") {
                  setRenaming(false);
                  setNewTitle(meta.title);
                }
              }}
              className="input h-6 text-ui flex-1 min-w-0"
            />
          ) : (
            <span
              onDoubleClick={() => setRenaming(true)}
              className="text-ui font-medium text-fg truncate cursor-text"
              title="双击重命名"
            >
              {meta.title}
            </span>
          )}
        </div>
        <button onClick={onClose} className="btn-ghost h-6 px-2" title="关闭">
          <Close size={11} />
          <span>关闭</span>
        </button>
      </header>

      {/* Metadata */}
      <div className="shrink-0 px-6 pt-4 pb-2 text-ui-sm text-fg-mute flex flex-wrap items-center gap-3 border-b border-border/60">
        {meta.source_audio && (
          <span title={meta.source_audio}>来源 <span className="font-mono text-fg-dim">{meta.source_stem ?? "audio"}</span></span>
        )}
        {meta.duration_seconds && <span>· {fmtDuration(meta.duration_seconds)}</span>}
        <span>· {meta.char_count} 字</span>
        {meta.model && <span>· 模型 <span className="font-mono text-fg-dim">{meta.model}</span></span>}
        {meta.based_on && (
          <span
            className={
              meta.based_on === "corrected"
                ? "px-1.5 py-0 rounded-sm border bg-ok/10 text-ok border-ok/30"
                : "px-1.5 py-0 rounded-sm border bg-warn/10 text-warn border-warn/30"
            }
          >
            {meta.based_on === "corrected" ? "校对版" : "原文版"}
          </span>
        )}
        <span className="ml-auto text-fg-mute">{meta.created_at}</span>
      </div>

      {/* Tags + note */}
      {(meta.tags.length > 0 || meta.note) && (
        <div className="shrink-0 px-6 py-2 text-ui-sm flex flex-wrap items-center gap-2 border-b border-border/60">
          {meta.tags.map((t) => (
            <span key={t} className="px-1.5 py-0 rounded-sm bg-accent/20 text-accent text-[11px]">
              {t}
            </span>
          ))}
          {meta.note && <span className="text-fg-mute italic">— {meta.note}</span>}
        </div>
      )}

      {/* Body */}
      <div className="flex-1 min-h-0 overflow-auto px-6 py-6 bg-editor">
        {error ? (
          <div className="text-err text-ui">{error}</div>
        ) : (
          <article className="text-fg leading-loose whitespace-pre-wrap text-ui-lg max-w-3xl mx-auto">
            {body || "(loading...)"}
          </article>
        )}
      </div>

      {/* Action bar */}
      <div className="shrink-0 flex items-center gap-1 px-3 h-9 border-t border-border/60 bg-tabbar/60">
        <button onClick={copy} className="btn-ghost h-6 px-2 text-ui-sm">
          <Copy size={12} />
          <span>复制全文</span>
        </button>
        <button onClick={exportMd} className="btn-ghost h-6 px-2 text-ui-sm">
          <Download size={12} />
          <span>导出 .md</span>
        </button>
        <span className="mx-1 h-4 w-px bg-border" />
        <button onClick={() => setRenaming(true)} className="btn-ghost h-6 px-2 text-ui-sm">
          <Pencil size={12} />
          <span>重命名</span>
        </button>
        <button onClick={showInFinder} className="btn-ghost h-6 px-2 text-ui-sm" title={meta.path}>
          <FolderOpen size={12} />
          <span>打开 .md</span>
        </button>
      </div>
    </div>
  );
}
