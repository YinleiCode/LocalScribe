import { useState } from "react";

import { ipc, type ModelStatus } from "../lib/ipc";
import { Check, FolderOpen, Refresh, Warning } from "./Icons";

type Props = {
  status: ModelStatus;
  onRecheck: () => Promise<void> | void;
};

const MODEL_SIZE = "约 1.5 GB";
const HF_URL = "https://huggingface.co/mlx-community/whisper-large-v3-turbo/tree/main";
const HF_MIRROR_URL = "https://hf-mirror.com/mlx-community/whisper-large-v3-turbo/tree/main";

export default function ModelMissingScreen({ status, onRecheck }: Props) {
  const [busy, setBusy] = useState<"none" | "open-folder" | "recheck">("none");
  const [error, setError] = useState<string | null>(null);

  const expected = status.expected_local_path || "(未知,请先启动一次主程序)";

  async function openFolder() {
    setBusy("open-folder");
    setError(null);
    try {
      await ipc.revealModelsDir(status.model_id);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy("none");
    }
  }

  async function openLink(url: string) {
    setError(null);
    try {
      await ipc.openUrl(url);
    } catch (e) {
      setError(String(e));
    }
  }

  async function recheck() {
    setBusy("recheck");
    setError(null);
    try {
      await onRecheck();
    } finally {
      setBusy("none");
    }
  }

  return (
    <div className="flex-1 overflow-y-auto bg-editor">
      <div className="max-w-2xl mx-auto px-6 py-10 space-y-6">
        <header className="flex items-start gap-3">
          <Warning size={20} className="text-warn shrink-0 mt-0.5" />
          <div className="min-w-0">
            <h1 className="text-ui-xl font-medium text-fg">未检测到 Whisper 模型</h1>
            <p className="text-ui-sm text-fg-mute mt-1">
              LocalScribe 需要 <span className="font-mono">{status.model_id}</span> 模型才能转录。
              首次使用请按下面三步把权重文件放到指定位置。
            </p>
          </div>
        </header>

        <section className="bg-sidebar border border-border rounded-sm">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2">
            <span className="inline-flex items-center justify-center w-5 h-5 text-ui-sm rounded-sm bg-accent text-white">1</span>
            <h2 className="text-ui font-medium text-fg">下载权重文件</h2>
            <span className="text-ui-sm text-fg-mute ml-auto">{MODEL_SIZE}</span>
          </div>
          <div className="px-4 py-3 space-y-2 text-ui-sm text-fg-dim">
            <p>从下面任一来源下载,需要的两个文件:</p>
            <ul className="list-disc pl-5 space-y-0.5 font-mono text-fg">
              <li>weights.safetensors (~1.5 GB)</li>
              <li>config.json (&lt; 1 KB)</li>
            </ul>
            <div className="flex gap-2 pt-2">
              <button onClick={() => openLink(HF_URL)} className="btn-ghost">
                HuggingFace (海外)
              </button>
              <button onClick={() => openLink(HF_MIRROR_URL)} className="btn-ghost">
                hf-mirror.com (国内)
              </button>
            </div>
          </div>
        </section>

        <section className="bg-sidebar border border-border rounded-sm">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2">
            <span className="inline-flex items-center justify-center w-5 h-5 text-ui-sm rounded-sm bg-accent text-white">2</span>
            <h2 className="text-ui font-medium text-fg">放到指定目录</h2>
          </div>
          <div className="px-4 py-3 space-y-2 text-ui-sm text-fg-dim">
            <p>把下载的两个文件放到下面这个文件夹(不存在会自动创建):</p>
            <div className="font-mono text-ui-sm bg-editor border border-border rounded-sm px-3 py-2 break-all text-fg select-all">
              {expected}
            </div>
            <button
              onClick={openFolder}
              disabled={busy === "open-folder"}
              className="btn-ghost flex items-center gap-1.5"
            >
              <FolderOpen size={14} />
              在 Finder 中打开此目录
            </button>
          </div>
        </section>

        <section className="bg-sidebar border border-border rounded-sm">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2">
            <span className="inline-flex items-center justify-center w-5 h-5 text-ui-sm rounded-sm bg-accent text-white">3</span>
            <h2 className="text-ui font-medium text-fg">回到这里点击重新检测</h2>
          </div>
          <div className="px-4 py-3 space-y-2 text-ui-sm text-fg-dim">
            <p>放好文件后点这个按钮 — 检测到即可开始使用。</p>
            <button
              onClick={recheck}
              disabled={busy === "recheck"}
              className="btn flex items-center gap-1.5"
            >
              {busy === "recheck" ? (
                <>
                  <Refresh size={14} className="animate-spin" />
                  检测中…
                </>
              ) : (
                <>
                  <Check size={14} />
                  我已放好,重新检测
                </>
              )}
            </button>
          </div>
        </section>

        {error && (
          <div className="text-ui-sm text-err bg-err/10 border border-err/30 rounded-sm px-3 py-2">
            {error}
          </div>
        )}

        <footer className="text-ui-sm text-fg-mute pt-2 border-t border-border/50 leading-relaxed">
          <p>
            进阶: 也可设置环境变量 <span className="font-mono text-fg">LOCALSCRIBE_MODEL_DIR</span> 指向其他位置(例如外置硬盘或 NAS),
            或者直接重跑 <span className="font-mono text-fg">./install.sh</span> 让脚本自动下载到 <span className="font-mono text-fg">models/</span>。
          </p>
        </footer>
      </div>
    </div>
  );
}
