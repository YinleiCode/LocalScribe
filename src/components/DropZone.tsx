import { useEffect, useState } from "react";
import clsx from "clsx";

import { FileAdd } from "./Icons";
import { isTauriRuntime, listenEvent, pickAudioPaths, uploadBrowserFile } from "../lib/runtime";

const ACCEPTED_EXTS = ["m4a", "mp3", "wav", "ogg", "flac", "aac", "opus", "mp4", "mov", "mkv", "webm"];

type Props = {
  onPick: (paths: string[]) => void;
  disabled?: boolean;
};

export default function DropZone({ onPick, disabled }: Props) {
  const [hovering, setHovering] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let active = true;
    (async () => {
      if (!isTauriRuntime()) return;
      const u = await listenEvent<{ paths: string[]; type: string }>("tauri://drag-drop", (payload) => {
        if (disabled) return;
        const paths = (payload?.paths ?? []).filter((p) => {
          const ext = p.split(".").pop()?.toLowerCase() ?? "";
          return ACCEPTED_EXTS.includes(ext);
        });
        if (paths.length) onPick(paths);
        setHovering(false);
      });
      const enter = await listenEvent("tauri://drag-enter", () => setHovering(true));
      const leave = await listenEvent("tauri://drag-leave", () => setHovering(false));
      if (!active) {
        u(); enter(); leave();
      } else {
        unlisten = () => { u(); enter(); leave(); };
      }
    })();
    return () => {
      active = false;
      unlisten?.();
    };
  }, [onPick, disabled]);

  async function pickFile() {
    if (disabled) return;
    const paths = await pickAudioPaths(ACCEPTED_EXTS);
    onPick(paths);
  }

  async function uploadFiles(files: FileList | File[]) {
    if (disabled || uploading) return;
    setUploading(true);
    setError(null);
    try {
      const accepted = Array.from(files).filter((file) => {
        const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
        return ACCEPTED_EXTS.includes(ext);
      });
      const uploaded = [];
      for (const file of accepted) {
        uploaded.push(await uploadBrowserFile(file));
      }
      if (uploaded.length) onPick(uploaded.map((f) => f.path));
    } catch (e) {
      setError(String(e));
    } finally {
      setUploading(false);
      setHovering(false);
    }
  }

  return (
    <button
      onClick={isTauriRuntime() ? pickFile : undefined}
      disabled={disabled || uploading}
      onDragOver={(e) => {
        if (isTauriRuntime()) return;
        e.preventDefault();
        setHovering(true);
      }}
      onDragLeave={() => !isTauriRuntime() && setHovering(false)}
      onDrop={(e) => {
        if (isTauriRuntime()) return;
        e.preventDefault();
        uploadFiles(e.dataTransfer.files);
      }}
      className={clsx(
        "relative",
        "w-full py-4 px-3 rounded-sm border border-dashed transition-colors",
        "flex flex-col items-center gap-1.5 text-ui-sm",
        hovering
          ? "border-accent bg-accent/10 text-accent"
          : "border-border bg-transparent hover:border-accent/60 hover:bg-hover text-fg-dim",
        (disabled || uploading) && "opacity-50 cursor-not-allowed",
      )}
    >
      {!isTauriRuntime() && (
        <input
          type="file"
          multiple
          accept={ACCEPTED_EXTS.map((ext) => `.${ext}`).join(",")}
          disabled={disabled || uploading}
          onChange={(e) => {
            if (e.currentTarget.files) uploadFiles(e.currentTarget.files);
            e.currentTarget.value = "";
          }}
          className="absolute inset-0 opacity-0 cursor-pointer disabled:cursor-not-allowed"
          aria-label="选择音视频文件"
        />
      )}
      <FileAdd size={20} className={hovering ? "text-accent" : "text-fg-mute"} />
      <span className="text-fg">{uploading ? "导入中" : hovering ? "松开以导入" : "拖入文件或点击选择"}</span>
      <span className="text-ui-sm text-fg-mute">
        {error || (uploading ? "大文件会先复制到本地" : "m4a / mp3 / wav / mp4 ...")}
      </span>
    </button>
  );
}
