import { create } from "zustand";
import type { ASRHumanReview, GlossaryEntry, Segment, TranscribeResult } from "../lib/ipc";

export type TaskStage =
  | "queued"
  | "transcribing"
  | "diarizing"
  | "saving"
  | "transcribed"
  | "correcting"
  | "correcting_paused"
  | "corrected"
  | "polishing"
  | "polished"
  | "translating"
  | "translated"
  | "error"
  | "cancelled";

export type Task = {
  id: string;
  audio: string;
  filename: string;
  libraryStem?: string;
  stage: TaskStage;
  progress: { current: number; total: number; preview?: string };
  error?: string;
  result?: TranscribeResult;
  asrHumanReview?: ASRHumanReview;
  corrected?: { segments: Segment[]; changed: number; total: number; model: string; glossary?: GlossaryEntry[] };
  polished?: {
    text: string;
    model: string;
    source: "corrected" | "raw";
    truncated?: boolean;
    finish_reason?: string;
    input_chars?: number;
  };
  translated?: {
    text: string;
    source_language: string | null;
    target_language: string;
    model: string;
    truncated?: boolean;
    finish_reason?: string;
    input_chars?: number;
  };
  createdAt: number;
};

type TasksStore = {
  tasks: Task[];
  activeId: string | null;
  add: (audio: string, libraryStem?: string) => string;
  setStage: (id: string, stage: TaskStage) => void;
  setProgress: (id: string, progress: Task["progress"]) => void;
  setResult: (id: string, result: TranscribeResult) => void;
  setDiarizationArtifacts: (
    id: string,
    result: TranscribeResult,
    corrected?: Task["corrected"],
  ) => void;
  setAsrHumanReview: (id: string, review: Task["asrHumanReview"]) => void;
  setAudio: (id: string, audio: string, filename?: string) => void;
  setLibraryStem: (id: string, stem: string) => void;
  setCorrected: (id: string, corrected: Task["corrected"]) => void;
  setPolished: (id: string, polished: Task["polished"]) => void;
  setTranslated: (id: string, translated: Task["translated"]) => void;
  setError: (id: string, error: string) => void;
  clearError: (id: string) => void;
  setCancelled: (id: string, reason?: string) => void;
  setActive: (id: string | null) => void;
  remove: (id: string) => void;
  clearAll: () => void;
};

function basename(path: string): string {
  return path.split(/[\\/]/).pop() || path;
}

export const useTasks = create<TasksStore>((set) => ({
  tasks: [],
  activeId: null,

  add: (audio, libraryStem) => {
    const id = crypto.randomUUID();
    const task: Task = {
      id,
      audio,
      filename: basename(audio),
      libraryStem,
      stage: "queued",
      progress: { current: 0, total: 0 },
      createdAt: Date.now(),
    };
    // 新任务放最前面 + 自动选中(用户期望:拖入即看到正在处理的项)
    set((s) => ({ tasks: [task, ...s.tasks], activeId: id }));
    return id;
  },

  setStage: (id, stage) =>
    set((s) => ({
      tasks: s.tasks.map((t) =>
        t.id === id && (t.stage !== "cancelled" || stage === "cancelled") ? { ...t, stage } : t,
      ),
    })),

  setProgress: (id, progress) =>
    set((s) => ({
      tasks: s.tasks.map((t) => (t.id === id && t.stage !== "cancelled" ? { ...t, progress } : t)),
    })),

  setResult: (id, result) =>
    set((s) => ({
      tasks: s.tasks.map((t) =>
        t.id === id && t.stage !== "cancelled" ? { ...t, result, stage: "transcribed" } : t,
      ),
    })),

  setDiarizationArtifacts: (id, result, corrected) =>
    set((s) => ({
      tasks: s.tasks.map((t) =>
        t.id === id && t.stage !== "cancelled"
          ? { ...t, result, ...(corrected ? { corrected } : {}) }
          : t,
      ),
    })),

  setAsrHumanReview: (id, asrHumanReview) =>
    set((s) => ({
      tasks: s.tasks.map((t) =>
        t.id === id && t.stage !== "cancelled" ? { ...t, asrHumanReview } : t,
      ),
    })),

  setAudio: (id, audio, filename) =>
    set((s) => ({
      tasks: s.tasks.map((t) =>
        t.id === id && t.stage !== "cancelled"
          ? { ...t, audio, filename: filename ?? t.filename }
          : t,
      ),
    })),

  setLibraryStem: (id, libraryStem) =>
    set((s) => ({
      tasks: s.tasks.map((t) =>
        t.id === id && t.stage !== "cancelled" ? { ...t, libraryStem } : t,
      ),
    })),

  setCorrected: (id, corrected) =>
    set((s) => ({
      tasks: s.tasks.map((t) =>
        t.id === id && t.stage !== "cancelled" ? { ...t, corrected, stage: "corrected" } : t,
      ),
    })),

  setPolished: (id, polished) =>
    set((s) => ({
      tasks: s.tasks.map((t) =>
        t.id === id && t.stage !== "cancelled" ? { ...t, polished, stage: "polished" } : t,
      ),
    })),

  setTranslated: (id, translated) =>
    set((s) => ({
      tasks: s.tasks.map((t) =>
        t.id === id && t.stage !== "cancelled" ? { ...t, translated, stage: "translated" } : t,
      ),
    })),

  setError: (id, error) =>
    set((s) => ({
      tasks: s.tasks.map((t) =>
        t.id === id && t.stage !== "cancelled" ? { ...t, error, stage: "error" } : t,
      ),
    })),

  clearError: (id) =>
    set((s) => ({
      tasks: s.tasks.map((t) => (t.id === id ? { ...t, error: undefined } : t)),
    })),

  setCancelled: (id, reason = "用户取消") =>
    set((s) => ({
      tasks: s.tasks.map((t) =>
        t.id === id ? { ...t, stage: "cancelled", error: reason } : t,
      ),
    })),

  setActive: (id) => set({ activeId: id }),

  remove: (id) =>
    set((s) => {
      const tasks = s.tasks.filter((t) => t.id !== id);
      const activeId = s.activeId === id ? tasks[0]?.id ?? null : s.activeId;
      return { tasks, activeId };
    }),

  clearAll: () => set({ tasks: [], activeId: null }),
}));
