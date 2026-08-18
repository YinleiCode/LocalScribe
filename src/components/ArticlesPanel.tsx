import { useEffect, useState } from "react";
import clsx from "clsx";

import { ipc, type ArticleMeta } from "../lib/ipc";
import { fmtDuration } from "../lib/format";
import { openExternalTarget } from "../lib/runtime";
import { Article, Close, FolderOpen, Refresh } from "./Icons";

type Props = {
  activeFilename: string | null;
  onSelect: (article: ArticleMeta) => void;
  refreshKey?: number;
};

export default function ArticlesPanel({ activeFilename, onSelect, refreshKey }: Props) {
  const [items, setItems] = useState<ArticleMeta[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      setItems(await ipc.articleList());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, [refreshKey]);

  async function openFolder() {
    try {
      const path = await ipc.articlesRootPath();
      await openExternalTarget(path);
    } catch (e) {
      setError(String(e));
    }
  }

  async function deleteArticle(m: ArticleMeta) {
    if (!confirm(`删除文章库文章「${m.title}」?(磁盘文件也会被删除)`)) return;
    try {
      await ipc.articleDelete(m.filename);
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  if (items.length === 0 && !loading && !error) {
    return (
      <div className="px-3 py-2 text-ui-sm text-fg-mute">
        文章库为空。在文章 tab 里点 <span className="text-fg-dim">保存到文章库</span> 保存。
      </div>
    );
  }

  return (
    <div>
      <div className="px-3 py-1 flex items-center justify-between text-ui-sm text-fg-mute">
        <button onClick={openFolder} className="hover:text-fg flex items-center gap-1">
          <FolderOpen size={11} />
          <span>打开 articles/</span>
        </button>
        <button
          onClick={refresh}
          disabled={loading}
          className="hover:text-fg flex items-center gap-1"
        >
          <Refresh size={11} />
          <span>{loading ? "刷新中" : "刷新"}</span>
        </button>
      </div>

      {error && <div className="px-3 py-1 text-ui-sm text-err truncate">{error}</div>}

      <div>
        {items.map((m) => (
          <div
            key={m.filename}
            onClick={() => onSelect(m)}
            className={clsx(
              "list-item flex-col items-stretch py-1.5 group",
              activeFilename === m.filename && "list-item-active",
            )}
          >
            <div className="flex items-center gap-2 min-w-0">
              <Article size={12} className="text-accent shrink-0" />
              <span className="flex-1 min-w-0 truncate text-ui">{m.title}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteArticle(m);
                }}
                className="opacity-0 group-hover:opacity-100 hover:text-err transition-opacity text-fg-mute"
                title="删除"
              >
                <Close size={11} />
              </button>
            </div>
            <div className="text-ui-sm text-fg-mute mt-0.5">
              {m.char_count} 字
              {m.duration_seconds ? ` · ${fmtDuration(m.duration_seconds)}` : ""}
              {m.based_on === "corrected" ? " · 校对版" : m.based_on === "raw" ? " · 原文版" : ""}
            </div>
            {m.tags.length > 0 && (
              <div className="text-ui-sm mt-0.5 flex flex-wrap gap-1">
                {m.tags.map((t) => (
                  <span
                    key={t}
                    className="px-1 py-0 rounded-sm bg-accent/20 text-accent text-[10px]"
                  >
                    {t}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
