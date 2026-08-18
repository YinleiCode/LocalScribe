import { useEffect, useState } from "react";
import clsx from "clsx";

import { ipc, type LibraryMeta } from "../lib/ipc";
import { fmtDuration } from "../lib/format";
import { openExternalTarget } from "../lib/runtime";
import { useTasks } from "../stores/tasks-store";
import { Article, Close, FileText, FolderOpen, Pencil, Refresh } from "./Icons";

export default function LibraryPanel() {
  const [items, setItems] = useState<LibraryMeta[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const tasks = useTasks((s) => s.tasks);
  const add = useTasks((s) => s.add);
  const setActive = useTasks((s) => s.setActive);
  const setStage = useTasks((s) => s.setStage);
  const setResult = useTasks((s) => s.setResult);
  const setAsrHumanReview = useTasks((s) => s.setAsrHumanReview);
  const setCorrected = useTasks((s) => s.setCorrected);
  const setPolished = useTasks((s) => s.setPolished);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const list = await ipc.libraryList();
      setItems(list);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, [tasks.length]);

  async function openLibraryFolder() {
    try {
      const path = await ipc.libraryRootPath();
      await openExternalTarget(path);
    } catch (e) {
      setError(String(e));
    }
  }

  async function loadItem(stem: string) {
    try {
      const loaded = await ipc.libraryLoad(stem);
      const audio = loaded.meta.audio_path || loaded.raw_json.audio || stem;
      const rawJson = { ...loaded.raw_json, audio };
      const id = add(audio, loaded.meta.stem);
      setStage(id, "transcribed");
      setResult(id, rawJson);
      if (loaded.asr_human_review) {
        setAsrHumanReview(id, loaded.asr_human_review);
      }
      if (loaded.corrected_json) {
        setCorrected(id, {
          segments: loaded.corrected_json.segments,
          changed: loaded.corrected_json.changed ?? 0,
          total: loaded.corrected_json.total ?? loaded.corrected_json.segments.length,
          model: loaded.corrected_json.corrected_by ?? loaded.meta.correction_model ?? "?",
          glossary: loaded.corrected_json.glossary ?? loaded.meta.correction_glossary ?? undefined,
        });
      }
      if (loaded.polished_text) {
        const src = loaded.meta.polish_source as "corrected" | "raw" | null;
        setPolished(id, {
          text: loaded.polished_text,
          model: loaded.meta.polish_model ?? "?",
          source: src ?? (loaded.meta.has_corrected ? "corrected" : "raw"),
        });
      }
      setActive(id);
    } catch (e) {
      setError(String(e));
    }
  }

  async function deleteItem(stem: string) {
    if (!confirm(`从历史库删除「${stem}」?(磁盘文件也会被删除)`)) return;
    try {
      await ipc.libraryDelete(stem);
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  if (items.length === 0 && !loading && !error) {
    return (
      <div className="px-3 py-2 text-ui-sm text-fg-mute">
        历史库为空。完成转录后会自动保存到此处。
      </div>
    );
  }

  return (
    <div>
      <div className="px-3 py-1 flex items-center justify-between text-ui-sm text-fg-mute">
        <button onClick={openLibraryFolder} className="hover:text-fg flex items-center gap-1">
          <FolderOpen size={11} />
          <span>打开 transcripts/</span>
        </button>
        <button onClick={refresh} disabled={loading} className="hover:text-fg flex items-center gap-1">
          <Refresh size={11} />
          <span>{loading ? "刷新中" : "刷新"}</span>
        </button>
      </div>

      {error && (
        <div className="px-3 py-1 text-ui-sm text-err truncate">{error}</div>
      )}

      <div>
        {items.map((m) => (
          <div
            key={m.stem}
            onClick={() => loadItem(m.stem)}
            className={clsx(
              "list-item flex-col items-stretch py-1.5 group",
            )}
          >
            <div className="flex items-center gap-2 min-w-0">
              <FileText size={12} className="text-fg-mute shrink-0" />
              <span className="flex-1 min-w-0 truncate font-mono text-ui">
                {m.audio_filename || m.stem}
              </span>
              <span className="flex items-center gap-1 shrink-0 text-fg-mute">
                {m.has_corrected && <Pencil size={11} className="text-warn" />}
                {m.has_polished && <Article size={11} className="text-ok" />}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteItem(m.stem);
                  }}
                  className="opacity-0 group-hover:opacity-100 hover:text-err transition-opacity"
                  title="删除"
                >
                  <Close size={11} />
                </button>
              </span>
            </div>
            <div className="text-ui-sm text-fg-mute mt-0.5">
              {fmtDuration(m.duration)} · {m.segments} 段 · {m.backend}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
