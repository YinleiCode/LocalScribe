import { useEffect, useState } from "react";
import { ipc, type ArticleMeta } from "../lib/ipc";
import { Article, Check } from "./Icons";

type Props = {
  defaultTitle: string;
  content: string;
  source_audio?: string;
  source_stem?: string;
  duration_seconds?: number;
  model?: string;
  based_on?: "corrected" | "raw";
  onClose: (saved?: ArticleMeta) => void;
};

export default function PinArticleDialog({
  defaultTitle,
  content,
  source_audio,
  source_stem,
  duration_seconds,
  model,
  based_on,
  onClose,
}: Props) {
  const [title, setTitle] = useState(defaultTitle);
  const [tags, setTags] = useState("");
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<ArticleMeta | null>(null);

  useEffect(() => {
    setTitle(defaultTitle);
  }, [defaultTitle]);

  async function save() {
    setError(null);
    setSaving(true);
    try {
      const meta = await ipc.articleSave({
        title: title.trim() || defaultTitle,
        content,
        source_audio,
        source_stem,
        duration_seconds,
        model,
        based_on,
        tags: tags
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
        note: note.trim() || undefined,
      });
      setSaved(meta);
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-sidebar border border-border rounded-sm shadow-2xl max-w-lg w-full m-6">
        <header className="flex items-center gap-2 px-4 py-3 border-b border-border">
          <Article size={16} className="text-accent" />
          <h2 className="text-ui-lg font-medium text-fg">保存为文章库条目</h2>
        </header>

        {!saved ? (
          <>
            <div className="px-4 py-4 space-y-3 text-ui">
              <div>
                <label className="block text-ui-sm text-fg-dim mb-1">标题</label>
                <input
                  className="input w-full"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="给这篇文章一个语义化的标题(AI agent 通过它读取)"
                />
              </div>

              <div>
                <label className="block text-ui-sm text-fg-dim mb-1">
                  标签<span className="text-fg-mute"> · 逗号分隔(可选)</span>
                </label>
                <input
                  className="input w-full"
                  value={tags}
                  onChange={(e) => setTags(e.target.value)}
                  placeholder="例:meeting, ai-business, 2026Q2"
                />
              </div>

              <div>
                <label className="block text-ui-sm text-fg-dim mb-1">
                  备注<span className="text-fg-mute">(可选,会写入 frontmatter)</span>
                </label>
                <textarea
                  className="textarea w-full"
                  rows={2}
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  placeholder="这次讨论的核心是..."
                />
              </div>

              <div className="text-ui-sm text-fg-mute leading-relaxed bg-editor/40 border border-border rounded-sm p-2">
                文章会以 markdown + YAML frontmatter 写入 <span className="font-mono text-fg-dim">articles/{title || defaultTitle}.md</span>。
                AI agent 可通过 glob <span className="font-mono text-fg-dim">articles/*.md</span> 读取整个文章库。
              </div>

              {error && <div className="text-ui-sm text-err">{error}</div>}
            </div>

            <div className="flex gap-2 px-4 py-3 border-t border-border bg-editor/40">
              <button onClick={() => onClose()} className="btn-ghost">
                取消
              </button>
              <span className="flex-1" />
              <button onClick={save} disabled={saving || !title.trim()} className="btn">
                {saving ? "保存中..." : "保存到文章库"}
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="px-4 py-6 space-y-3 text-ui flex flex-col items-center text-center">
              <Check size={32} className="text-ok" />
              <div className="text-ui-lg text-fg">已保存到文章库</div>
              <div className="text-ui-sm text-fg-dim font-mono break-all">
                {saved.path}
              </div>
              <div className="text-ui-sm text-fg-mute">
                AI agent 可读取:<span className="font-mono">{saved.filename}</span>
              </div>
            </div>
            <div className="flex gap-2 px-4 py-3 border-t border-border bg-editor/40">
              <span className="flex-1" />
              <button onClick={() => onClose(saved)} className="btn">
                完成
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
