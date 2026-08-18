export type UnlistenFn = () => void;

export const WEB_API_BASE =
  ((import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env?.VITE_LOCALSCRIBE_API)
  || "http://127.0.0.1:8765";

export function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export async function invokeCommand<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
  if (isTauriRuntime()) {
    const { invoke } = await import("@tauri-apps/api/core");
    return invoke<T>(method, params);
  }

  const res = await fetch(`${WEB_API_BASE}/api/invoke/${method}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || data.error || `HTTP ${res.status}`);
  }
  return data.result as T;
}

export async function listenEvent<T>(
  event: string,
  handler: (payload: T) => void,
): Promise<UnlistenFn> {
  if (!isTauriRuntime()) {
    void event;
    void handler;
    return () => {};
  }
  const { listen } = await import("@tauri-apps/api/event");
  return listen<T>(event, (e) => handler(e.payload));
}

export async function pickAudioPaths(extensions: string[]): Promise<string[]> {
  if (!isTauriRuntime()) {
    return [];
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  const selected = await open({
    multiple: true,
    filters: [{ name: "Audio / Video", extensions }],
  });
  if (!selected) return [];
  return Array.isArray(selected) ? selected : [selected];
}

export async function uploadBrowserFile(file: File): Promise<{ path: string; filename: string; size: number }> {
  const body = new FormData();
  body.append("file", file, file.name);
  const res = await fetch(`${WEB_API_BASE}/api/upload`, { method: "POST", body });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Upload failed: HTTP ${res.status}`);
  }
  return data;
}

export async function downloadTextFile(filename: string, text: string): Promise<void> {
  if (isTauriRuntime()) {
    const [{ save, message }, { writeTextFile }, { downloadDir, join }] = await Promise.all([
      import("@tauri-apps/plugin-dialog"),
      import("@tauri-apps/plugin-fs"),
      import("@tauri-apps/api/path"),
    ]);
    let defaultPath = filename;
    try {
      defaultPath = await join(await downloadDir(), filename);
    } catch {
      // keep filename fallback
    }
    const ext = filename.match(/\.([a-zA-Z0-9]+)$/)?.[1]?.toLowerCase() || "txt";
    const filterName: Record<string, string> = {
      txt: "Text",
      md: "Markdown",
      srt: "SubRip Subtitle",
      json: "JSON",
    };
    const path = await save({
      defaultPath,
      filters: [{ name: filterName[ext] || ext.toUpperCase(), extensions: [ext] }],
    });
    if (!path) return;
    await writeTextFile(path, text);
    await message(`已保存到:\n${path}`, { title: "下载完成", kind: "info" }).catch(() => {});
    return;
  }

  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function showMessage(text: string, title = "LocalScribe", kind: "info" | "error" = "info"): Promise<void> {
  if (isTauriRuntime()) {
    const { message } = await import("@tauri-apps/plugin-dialog");
    await message(text, { title, kind });
    return;
  }
  window.alert(`${title}\n\n${text}`);
}

export async function openExternalTarget(target: string): Promise<void> {
  if (isTauriRuntime()) {
    const { open } = await import("@tauri-apps/plugin-shell");
    await open(target);
    return;
  }
  if (/^https?:\/\//.test(target)) {
    window.open(target, "_blank", "noopener,noreferrer");
    return;
  }
  await invokeCommand("open_url", { url: target });
}

export async function localMediaUrl(path: string): Promise<string> {
  const target = path.trim();
  if (!target) return "";
  if (/^(https?:|blob:|data:|asset:|file:)/i.test(target)) return target;
  if (isTauriRuntime()) {
    const { convertFileSrc } = await import("@tauri-apps/api/core");
    const prepared = await invokeCommand<{ path: string; optimized: boolean }>(
      "prepare_media_playback",
      { audio: target },
    );
    return convertFileSrc(prepared.path);
  }
  return `${WEB_API_BASE}/api/media?path=${encodeURIComponent(target)}`;
}
