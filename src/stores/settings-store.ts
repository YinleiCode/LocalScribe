import { create } from "zustand";
import { ipc, type AppSettings, type CorrectionMode } from "../lib/ipc";

type ValueOrUpdater<T> = T | ((current: T) => T);
type PartialOrUpdater<T> = Partial<T> | ((current: T) => Partial<T>);
type JsonObject = Record<string, unknown>;

export type AppendAsrHotwordsResult = {
  added: string[];
  skipped: string[];
};

type SettingsMutation<T> = {
  next: AppSettings;
  result: T;
};

// One module-lifetime queue for every settings.json mutation. Mutation builders
// run only after prior writes settle, so they always derive from the latest
// committed Zustand state and reach the backend in invocation order.
let settingsWriteTail: Promise<void> = Promise.resolve();
let settingsCommitRevision = 0;

const DEFAULT_SETTINGS: AppSettings = {
  model_id: "iic/SenseVoiceSmall",
  backend: "auto",
  language: "zh",
  asr_hotwords: "",
  asr_quality_mode: "standard",
  audio_preprocess: "adaptive",
  transcript_sync: "precise",
  output_formats: ["txt", "srt", "json"],
  output_dir: null,
  correction: {
    enabled: false,
    auto_pipeline: false,
    provider: "deepseek",
    base_url: "https://api.deepseek.com",
    model: "deepseek-v4-flash",
    mode: "medium",
    batch_size: 30,
    context_hint: "",
    use_glossary: true,
    concurrency: 15,
    advanced: {
      temperature: 0.1,
      max_tokens: 8192,
      top_p: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.0,
    },
  },
  diarization: {
    enabled: false,
    engine: "auto",
    n_speakers: 0, // 0 = 自动检测
    speakers: [],
  },
  polish: {
    enabled: false,
    model: "deepseek-v4-flash",
    advanced: {
      temperature: 0.3,
      max_tokens: 384000,
      top_p: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.0,
    },
  },
  translation: {
    model: "deepseek-v4-flash",
    advanced: {
      temperature: 0.3,
      max_tokens: 384000,
      top_p: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.0,
    },
  },
};

function normalizeSettings(settings: AppSettings): AppSettings {
  if (
    (settings.backend === "auto" || settings.backend === "sensevoice")
    && settings.model_id.startsWith("mlx-community/")
  ) {
    return { ...settings, model_id: DEFAULT_SETTINGS.model_id };
  }
  return settings;
}

function withDefaults(settings: AppSettings): AppSettings {
  return normalizeSettings({
    ...DEFAULT_SETTINGS,
    ...settings,
    correction: { ...DEFAULT_SETTINGS.correction, ...(settings.correction ?? {}) },
    polish: { ...DEFAULT_SETTINGS.polish, ...(settings.polish ?? {}) },
    translation: { ...DEFAULT_SETTINGS.translation, ...(settings.translation ?? {}) },
    diarization: { ...DEFAULT_SETTINGS.diarization, ...(settings.diarization ?? {}) },
  });
}

function resolveValue<T>(current: T, value: ValueOrUpdater<T>): T {
  return typeof value === "function"
    ? (value as (current: T) => T)(current)
    : value;
}

function resolvePatch<T>(current: T, patch: PartialOrUpdater<T>): Partial<T> {
  return typeof patch === "function"
    ? (patch as (current: T) => Partial<T>)(current)
    : patch;
}

function enqueueSettingsWrite<T>(
  readCurrent: () => AppSettings,
  commit: (next: AppSettings) => void,
  build: (current: AppSettings) => SettingsMutation<T>,
): Promise<T> {
  const run = settingsWriteTail.catch(() => undefined).then(async () => {
    const current = readCurrent();
    const mutation = build(current);
    const next = normalizeSettings(mutation.next);
    if (next !== current) {
      await ipc.saveSettings(next);
      settingsCommitRevision += 1;
      commit(next);
    }
    return mutation.result;
  });
  settingsWriteTail = run.then(() => undefined, () => undefined);
  return run;
}

function asJsonObject(value: unknown): JsonObject | null {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function normalizeHotwordKey(value: string): string {
  return value.replace(/\s+/g, " ").trim().toLocaleLowerCase();
}

type HotwordObjectEntry = { term: string; key: "term" | "name" | "word" };

function parseHotwordObjectEntry(value: unknown): HotwordObjectEntry | null {
  const record = asJsonObject(value);
  if (!record) return null;
  const candidates = (["term", "name", "word"] as const)
    .map((key) => ({ key, term: typeof record[key] === "string" ? record[key].trim() : "" }))
    .filter((entry) => entry.term);
  if (!candidates.length) return null;
  const distinct = new Set(candidates.map((entry) => normalizeHotwordKey(entry.term)));
  if (distinct.size > 1) {
    throw new Error("热词对象同时包含不同的 term/name/word，无法无损判断应使用哪一个");
  }
  return candidates[0];
}

function parseHotwordArray(value: unknown[]): {
  terms: string[];
  append: (term: string) => unknown[];
} {
  const terms: string[] = [];
  let objectKey: "term" | "name" | "word" | null = null;
  for (const entry of value) {
    if (typeof entry === "string") {
      const term = entry.trim();
      if (!term) throw new Error("热词 JSON 数组包含空字符串，无法无损理解");
      terms.push(term);
      continue;
    }
    const parsed = parseHotwordObjectEntry(entry);
    if (!parsed) {
      throw new Error("热词 JSON 数组仅支持字符串或含 term/name/word 的对象");
    }
    objectKey ??= parsed.key;
    terms.push(parsed.term);
  }
  return {
    terms,
    append: (term) => [...value, objectKey ? { [objectKey]: term } : term],
  };
}

export function appendHotwordSetting(current: string, hotword: string): { value: string; added: boolean } {
  const clean = hotword.replace(/\s+/g, " ").trim();
  if (!clean) return { value: current, added: false };
  const cleanKey = normalizeHotwordKey(clean);
  const trimmed = current.trim();
  if (trimmed) {
    let parsed: unknown;
    let parsedJson = false;
    try {
      parsed = JSON.parse(trimmed) as unknown;
      parsedJson = true;
    } catch {
      if (/^[\[{]/.test(trimmed)) {
        throw new Error("当前热词看起来是 JSON，但格式无效；为避免损坏配置，已拒绝文本拼接");
      }
    }
    if (parsedJson) {
      if (Array.isArray(parsed)) {
        const array = parseHotwordArray(parsed);
        if (array.terms.some((term) => normalizeHotwordKey(term) === cleanKey)) {
          return { value: current, added: false };
        }
        return { value: JSON.stringify(array.append(clean), null, 2), added: true };
      }
      const record = asJsonObject(parsed);
      if (!record) {
        throw new Error("热词 JSON 必须是数组，或包含 hotwords/glossary/terms 数组的对象");
      }
      const fields = (["terms", "hotwords", "glossary"] as const)
        .filter((field) => Object.prototype.hasOwnProperty.call(record, field));
      if (!fields.length) {
        throw new Error("热词 JSON 对象缺少 hotwords、glossary 或 terms 数组，已拒绝修改");
      }
      const parsedFields = fields.map((field) => {
        if (!Array.isArray(record[field])) {
          throw new Error(`热词 JSON 字段 ${field} 必须是数组`);
        }
        return { field, array: parseHotwordArray(record[field] as unknown[]) };
      });
      const selected = parsedFields.find((entry) => entry.array.terms.length > 0) ?? parsedFields[0];
      const allTerms = parsedFields.flatMap((entry) => entry.array.terms);
      if (allTerms.some((term) => normalizeHotwordKey(term) === cleanKey)) {
        return { value: current, added: false };
      }
      return {
        value: JSON.stringify({
          ...record,
          [selected.field]: selected.array.append(clean),
        }, null, 2),
        added: true,
      };
    }
  }
  const existing = current
    .split(/[\n,，;；、|\t]+/)
    .map(normalizeHotwordKey)
    .filter(Boolean);
  if (existing.includes(cleanKey)) return { value: current, added: false };
  return {
    value: current.trimEnd() ? `${current.trimEnd()}\n${clean}` : clean,
    added: true,
  };
}

type SettingsStore = {
  settings: AppSettings;
  loaded: boolean;
  hasApiKey: boolean;
  loadFromBackend: () => Promise<void>;
  save: (next: AppSettings) => Promise<void>;
  patch: <K extends keyof AppSettings>(key: K, value: ValueOrUpdater<AppSettings[K]>) => Promise<void>;
  patchCorrection: (patch: PartialOrUpdater<AppSettings["correction"]>) => Promise<void>;
  patchPolish: (patch: PartialOrUpdater<AppSettings["polish"]>) => Promise<void>;
  patchTranslation: (patch: PartialOrUpdater<AppSettings["translation"]>) => Promise<void>;
  patchDiarization: (patch: PartialOrUpdater<AppSettings["diarization"]>) => Promise<void>;
  appendAsrHotword: (hotword: string, beforeWrite?: () => void) => Promise<{ added: boolean }>;
  appendAsrHotwords: (hotwords: string[], beforeWrite?: () => void) => Promise<AppendAsrHotwordsResult>;
  setApiKey: (provider: string, key: string) => Promise<void>;
  refreshHasApiKey: () => Promise<void>;
};

export const useSettings = create<SettingsStore>((set, get) => ({
  settings: DEFAULT_SETTINGS,
  loaded: false,
  hasApiKey: false,

  loadFromBackend: async () => {
    const loadRevision = settingsCommitRevision;
    try {
      const s = await ipc.loadSettings();
      if (settingsCommitRevision === loadRevision) {
        set({ settings: withDefaults(s), loaded: true });
      } else {
        set({ loaded: true });
      }
      await get().refreshHasApiKey();
    } catch (e) {
      console.warn("loadSettings failed, using defaults", e);
      set({ loaded: true });
      await get().refreshHasApiKey();
    }
  },

  save: (next) => enqueueSettingsWrite(
    () => get().settings,
    (committed) => set({ settings: committed }),
    () => ({ next, result: undefined }),
  ),

  patch: (key, value) => enqueueSettingsWrite(
    () => get().settings,
    (committed) => set({ settings: committed }),
    (current) => {
      const resolved = resolveValue(current[key], value);
      let next = { ...current, [key]: resolved };
      if (key === "backend" && (resolved === "auto" || resolved === "sensevoice")) {
        next = { ...next, model_id: DEFAULT_SETTINGS.model_id };
      }
      return { next, result: undefined };
    },
  ),

  patchCorrection: (patch) => enqueueSettingsWrite(
    () => get().settings,
    (committed) => set({ settings: committed }),
    (current) => ({
      next: {
        ...current,
        correction: { ...current.correction, ...resolvePatch(current.correction, patch) },
      },
      result: undefined,
    }),
  ),

  patchPolish: (patch) => enqueueSettingsWrite(
    () => get().settings,
    (committed) => set({ settings: committed }),
    (current) => ({
      next: {
        ...current,
        polish: { ...current.polish, ...resolvePatch(current.polish, patch) },
      },
      result: undefined,
    }),
  ),

  patchTranslation: (patch) => enqueueSettingsWrite(
    () => get().settings,
    (committed) => set({ settings: committed }),
    (current) => ({
      next: {
        ...current,
        translation: { ...current.translation, ...resolvePatch(current.translation, patch) },
      },
      result: undefined,
    }),
  ),

  patchDiarization: (patch) => enqueueSettingsWrite(
    () => get().settings,
    (committed) => set({ settings: committed }),
    (current) => {
      const diarization = current.diarization ?? DEFAULT_SETTINGS.diarization;
      return {
        next: {
          ...current,
          diarization: { ...diarization, ...resolvePatch(diarization, patch) },
        },
        result: undefined,
      };
    },
  ),

  appendAsrHotword: async (hotword, beforeWrite) => {
    const result = await get().appendAsrHotwords([hotword], beforeWrite);
    return { added: result.added.length > 0 };
  },

  appendAsrHotwords: (hotwords, beforeWrite) => enqueueSettingsWrite(
    () => get().settings,
    (committed) => set({ settings: committed }),
    (current) => {
      beforeWrite?.();
      let value = current.asr_hotwords || "";
      const added: string[] = [];
      const skipped: string[] = [];
      for (const hotword of hotwords) {
        const clean = hotword.replace(/\s+/g, " ").trim();
        if (!clean) {
          skipped.push(hotword);
          continue;
        }
        const appended = appendHotwordSetting(value, clean);
        value = appended.value;
        (appended.added ? added : skipped).push(clean);
      }
      return {
        next: added.length > 0 ? { ...current, asr_hotwords: value } : current,
        result: { added, skipped },
      };
    },
  ),

  setApiKey: async (provider, key) => {
    await ipc.setApiKey(provider, key);
    set({ hasApiKey: true });
  },

  refreshHasApiKey: async () => {
    const provider = get().settings.correction.provider;
    try {
      const has = await ipc.hasApiKey(provider);
      set({ hasApiKey: has });
    } catch {
      set({ hasApiKey: false });
    }
  },
}));

export const CORRECTION_MODES: { value: CorrectionMode; label: string; hint: string }[] = [
  { value: "light", label: "轻", hint: "只修明显错别字 / 同音字" },
  { value: "medium", label: "中", hint: "错字 + 专名 + 删冗余字(推荐)" },
  { value: "heavy", label: "重", hint: "上述 + 口头禅 / 重复词清理" },
];
