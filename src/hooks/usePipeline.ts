// 流水线 hook:
// - 自动:只跑转录(快,无网络依赖)
// - 手动:校对 / 排版 通过暴露的函数按需触发(在 ResultTabs 按钮点击时调用)

import { useCallback, useEffect, useReducer, useRef } from "react";
import {
  captureTaskCorrectedRevision,
  committedRawRevisionForStem,
  correctedArtifactFingerprint,
  enqueueStemWrite,
  persistCorrectedArtifact,
  type CorrectedArtifact,
  type CorrectedSavePayload,
} from "../lib/corrected-persistence";
import { assertDiarizationPreservesTranscript, ipc, libraryStemFromFilename, libraryStemKey, onProgress } from "../lib/ipc";
import { isTauriRuntime } from "../lib/runtime";
import type { ASRPreprocessMode, ASRPreprocessSetting, TranscribeResult } from "../lib/ipc";
import { buildJson, buildSrt, buildTxt, fmtTs } from "../lib/format";
import { useSettings } from "../stores/settings-store";
import { useTasks } from "../stores/tasks-store";

// 模块级流水线状态 —— hook 和外部的 cancelTask() 共享同一份。
const pendingCommittedResults = new Map<string, TranscribeResult>();

const pipelineState = {
  /** 当前拥有转写流水线执行权的 task。只有 owner 的 finally 可以释放。 */
  activeTaskId: null as string | null,
  /** 已被用户取消的 task id 集合 —— 流水线 async 块每 await 完检查,命中就丢结果 */
  cancelledIds: new Set<string>(),
  cancelPromises: new Map<string, Promise<void>>(),
  diarizationCancelledIds: new Set<string>(),
};

const correctionState = {
  activeTaskId: null as string | null,
  cancelledIds: new Set<string>(),
};

const ASR_PREPROCESS_MODES = new Set<ASRPreprocessMode>(["off", "standard", "adaptive", "ai_denoise", "enhance"]);
const SENSEVOICE_MODEL_ID = "iic/SenseVoiceSmall";

class DiarizationPersistenceError extends Error {}

function concretePreprocessMode(value: string | undefined): ASRPreprocessMode | null {
  return ASR_PREPROCESS_MODES.has(value as ASRPreprocessMode) ? (value as ASRPreprocessMode) : null;
}

function effectiveAsrModelId(backend: string | undefined, modelId: string | undefined): string | undefined {
  const normalizedBackend = backend || "auto";
  if ((normalizedBackend === "auto" || normalizedBackend === "sensevoice") && modelId?.startsWith("mlx-community/")) {
    return SENSEVOICE_MODEL_ID;
  }
  return modelId;
}

function resultRevisionFingerprint(result: TranscribeResult): string {
  const canonical = JSON.stringify({
    segments: result.segments,
    diarization_stats: result.diarization_stats,
  });
  let hash = 0x811c9dc5;
  for (let index = 0; index < canonical.length; index += 1) {
    hash ^= canonical.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return `pipeline-result-v1:${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function authoritativePipelineTaskForStem(stem: string, preferredTaskId?: string) {
  const state = useTasks.getState();
  const key = libraryStemKey(stem);
  const matches = (task: { libraryStem?: string; filename: string }) =>
    libraryStemKey(task.libraryStem ?? libraryStemFromFilename(task.filename)) === key;
  return state.tasks.find((task) => task.id === preferredTaskId && matches(task))
    ?? state.tasks.find((task) => task.id === state.activeId && matches(task))
    ?? state.tasks.find(matches);
}

function selectedPipelineAudio(result: { filter_stats?: Record<string, unknown> }, fallbackAudio: string): string {
  const stats = result.filter_stats?.audio_standardization as Record<string, unknown> | undefined;
  const path = typeof stats?.path === "string" ? stats.path : "";
  const applied = Boolean(stats?.applied);
  const cleaned = Boolean(stats?.work_dir_cleaned);
  return applied && path && !cleaned ? path : fallbackAudio;
}


function buildPipelineCorrectedPayload(stem: string, artifact: CorrectedArtifact): CorrectedSavePayload {
  const diffLines: string[] = [`# diff: ${artifact.changed} changes / ${artifact.total} segments`, ""];
  for (const segment of artifact.segments) {
    if (segment.original_text && segment.original_text !== segment.text) {
      diffLines.push(`[${fmtTs(segment.start)}]\n  - ${segment.original_text}\n  + ${segment.text}\n`);
    }
  }
  return {
    stem,
    txt: buildTxt(artifact.segments, `${stem} (corrected by ${artifact.model})`),
    srt: buildSrt(artifact.segments),
    json: JSON.stringify({
      stem,
      corrected_by: artifact.model,
      changed: artifact.changed,
      total: artifact.total,
      glossary: artifact.glossary,
      segments: artifact.segments,
    }, null, 2),
    diff: diffLines.join("\n"),
    model: artifact.model,
    changed: artifact.changed,
    total: artifact.total,
    glossary: artifact.glossary,
  };
}

export function usePipeline() {
  const tasks = useTasks((s) => s.tasks);
  const setStage = useTasks((s) => s.setStage);
  const setProgress = useTasks((s) => s.setProgress);
  const setResult = useTasks((s) => s.setResult);
  const setAudio = useTasks((s) => s.setAudio);
  const setLibraryStem = useTasks((s) => s.setLibraryStem);
  const setPolished = useTasks((s) => s.setPolished);
  const setError = useTasks((s) => s.setError);
  const setCancelled = useTasks((s) => s.setCancelled);

  const settings = useSettings((s) => s.settings);
  const settingsLoaded = useSettings((s) => s.loaded);

  const transcribingIdRef = useRef<string | null>(null);
  const correctingIdRef = useRef<string | null>(null);
  const [schedulerTick, wakeScheduler] = useReducer((value: number) => value + 1, 0);
  const [correctionSchedulerTick, wakeCorrectionScheduler] = useReducer((value: number) => value + 1, 0);

  // Forward sidecar progress events to whichever task is currently running.
  // 关键:统一单位到 0-100 百分比。伪进度(下方 useEffect)也是 0-100,这样两者
  // 不会因为 total 字段含义切换(分块数 vs 100)而导致 UI 反复跳。
  // 同时不允许进度倒退(防止 fake 估算偏大后被真值暴跌覆盖)。
  useEffect(() => {
    let unsubT: (() => void) | undefined;
    let unsubC: (() => void) | undefined;
    let unsubP: (() => void) | undefined;
    onProgress("transcribe", (data) => {
      const id = transcribingIdRef.current;
      if (!id || pipelineState.activeTaskId !== id) return;
      const cur = data.current ?? 0;
      const tot = data.total ?? 0;
      const pct = tot > 0 ? Math.min(100, Math.round((cur / tot) * 100)) : 0;
      const prev = useTasks.getState().tasks.find((t) => t.id === id)?.progress;
      const prevPct = prev && prev.total === 100 ? prev.current : 0;
      if (pct < prevPct) return; // 不倒退
      setProgress(id, { current: pct, total: 100, preview: data.preview });
    }).then((fn) => (unsubT = fn));
    onProgress("correct", (data) => {
      const id = correctingIdRef.current;
      if (!id || correctionState.activeTaskId !== id) return;
      const cur = data.current ?? 0;
      const tot = data.total ?? 0;
      const pct = tot > 0 ? Math.min(100, Math.round((cur / tot) * 100)) : 0;
      const prev = useTasks.getState().tasks.find((t) => t.id === id)?.progress;
      const prevPct = prev && prev.total === 100 ? prev.current : 0;
      if (pct < prevPct) return;
      setProgress(id, { current: pct, total: 100 });
    }).then((fn) => (unsubC = fn));
    onProgress("asr_preflight_select", (data) => {
      const id = transcribingIdRef.current;
      if (!id || pipelineState.activeTaskId !== id) return;
      const cur = data.current ?? 0;
      const tot = data.total ?? 0;
      const pct = tot > 0 ? Math.min(15, Math.round((cur / tot) * 15)) : 0;
      setProgress(id, {
        current: pct,
        total: 100,
        preview: data.preview || "音频预检中",
      });
    }).then((fn) => (unsubP = fn));
    return () => {
      unsubT?.();
      unsubC?.();
      unsubP?.();
    };
  }, [setProgress]);

  // Pseudo-progress for MLX (which doesn't emit per-segment events). Estimates
  // expected runtime from audio duration × RTF and animates progress so the
  // UI doesn't sit at 0%. Real progress events override.
  useEffect(() => {
    const t = tasks.find((x) => x.stage === "transcribing");
    if (!t) return;
    let cancelled = false;
    let interval: number | null = null;

    (async () => {
      let estDurationS = 60;
      try {
        const probe = await ipc.probeAudio(t.audio);
        estDurationS = probe.duration || 60;
      } catch {
        // ignore — keep fallback
      }
      if (cancelled) return;
      // 估算总耗时 = 音频时长 × 0.025 (MLX RTF) + 1.5s 模型加载缓冲
      const estCostMs = estDurationS * 25 + 1500;
      const startTs = Date.now();
      interval = window.setInterval(() => {
        const cur = useTasks.getState().tasks.find((x) => x.id === t.id);
        if (!cur || cur.stage !== "transcribing") {
          if (interval) window.clearInterval(interval);
          interval = null;
          return;
        }
        const elapsedMs = Date.now() - startTs;
        // 95% asymptote — don't reach 100 before real result
        const fakeFraction = 1 - Math.exp(-elapsedMs / estCostMs);
        const fakePct = Math.min(95, Math.round(fakeFraction * 95));
        const realFracPct =
          cur.progress.total > 0
            ? Math.round((cur.progress.current / cur.progress.total) * 100)
            : 0;
        if (realFracPct >= fakePct) return;
        setProgress(t.id, {
          current: fakePct,
          total: 100,
          preview: cur.progress.preview,
        });
      }, 400);
    })();

    return () => {
      cancelled = true;
      if (interval) window.clearInterval(interval);
    };
  }, [tasks, setProgress]);

  // Auto-run transcription only — LLM stages are now opt-in via buttons.
  useEffect(() => {
    if (!settingsLoaded) return;
    if (pipelineState.activeTaskId) return;
    const next = [...tasks]
      .sort((a, b) => a.createdAt - b.createdAt)
      .find((t) => t.stage === "queued");
    if (!next) return;
    const taskId = next.id;
    pipelineState.activeTaskId = taskId;

    // 取消短路：ASR sidecar 会被定向终止；旧 owner 等终止完成后才释放队列。
    const isCancelled = () => pipelineState.cancelledIds.has(taskId);

    (async () => {
      try {
        const modelId = effectiveAsrModelId(settings.backend, settings.model_id);
        transcribingIdRef.current = taskId;
        setStage(taskId, "transcribing");
        setProgress(taskId, { current: 0, total: 100, preview: "准备转录" });
        let audioPreprocess: ASRPreprocessMode = "adaptive";
        const preprocessSetting = (settings.audio_preprocess || "adaptive") as ASRPreprocessSetting;
        const fixedMode = concretePreprocessMode(preprocessSetting);
        if (fixedMode) {
          audioPreprocess = fixedMode;
          setProgress(taskId, {
            current: 2,
            total: 100,
            preview: `音频预处理: ${audioPreprocess}`,
          });
        } else {
          try {
            setProgress(taskId, { current: 1, total: 100, preview: "音频预检中(较慢实验)" });
            const preflight = await ipc.asrPreflightSelect({
              audio: next.audio,
              backend: settings.backend,
              model_id: modelId,
              language: settings.language,
              hotwords: settings.asr_hotwords,
              preferred_mode: "adaptive",
              modes: ["adaptive", "ai_denoise", "enhance"],
              clip_seconds: 25,
              max_clips: 2,
            });
            if (isCancelled()) return;
            audioPreprocess = concretePreprocessMode(preflight.recommended_mode) || "adaptive";
            const risk = String(preflight.audio_quality?.risk_level || "unknown");
            const preview = preflight.skipped
              ? `音频风险 ${risk}，使用 ${audioPreprocess}`
              : `预检推荐 ${audioPreprocess}，音频风险 ${risk}`;
            setProgress(taskId, { current: 15, total: 100, preview });
          } catch (e) {
            console.warn("asr_preflight_select failed; fallback to adaptive", e);
            audioPreprocess = "adaptive";
            setProgress(taskId, {
              current: 2,
              total: 100,
              preview: "预检失败，使用通用安全模式",
            });
          }
        }
        if (isCancelled()) return;
        const result = await ipc.transcribe({
          audio: next.audio,
          backend: settings.backend,
          model_id: modelId,
          language: settings.language,
          hotwords: settings.asr_hotwords,
          asr_quality_mode: settings.asr_quality_mode || "standard",
          audio_preprocess: audioPreprocess,
          timing_align: (settings.transcript_sync || "precise") !== "fast",
        });
        if (isCancelled()) return;

        const stem = next.libraryStem ?? libraryStemFromFilename(next.filename);
        setStage(taskId, "saving");
        setProgress(taskId, { current: 98, total: 100, preview: "正在安全保存转录结果" });
        let stableAudio = next.audio;
        const pipelineAudio = selectedPipelineAudio(result, next.audio);
        const initialResult = { ...result, audio: pipelineAudio };
        try {
          const meta = await enqueueStemWrite(stem, () => ipc.librarySaveRaw({
            stem,
            audio_filename: next.filename,
            source_audio: pipelineAudio,
            txt: buildTxt(initialResult.segments, `${next.filename}\nbackend=${initialResult.backend} duration=${initialResult.duration.toFixed(1)}s segments=${initialResult.segments.length}`),
            srt: buildSrt(initialResult.segments),
            json: buildJson(initialResult),
            result: initialResult,
          }));
          if (isCancelled()) return;
          if (!meta.audio_path) {
            throw new Error("稳定音频副本未创建");
          }
          setLibraryStem(taskId, meta.stem || stem);
          stableAudio = meta.audio_path;
          result.audio = meta.audio_path;
          const audioStats = result.filter_stats?.audio_standardization as Record<string, unknown> | undefined;
          if (audioStats) {
            audioStats.library_path = meta.audio_path;
          }
          setAudio(taskId, meta.audio_path);
        } catch (e) {
          if (isCancelled()) return;
          setResult(taskId, initialResult);
          throw new Error(`转录已完成，但保存失败: ${String(e)}`);
        }
        if (isCancelled()) return;
        pendingCommittedResults.set(taskId, {
          ...result,
          segments: result.segments.map((segment) => ({ ...segment })),
        });

        // Keep the result private until optional diarization finishes so users
        // cannot start correction against a transcript that is still changing.

        // Optional diarization — run after source audio has been copied to a stable path.
        const diar = settings.diarization;
        let diarizationError: string | null = null;
        const timingReliable = result.filter_stats?.timing_reliable !== false;
        if (diar?.enabled && result.segments.length > 0 && timingReliable) {
          try {
            setStage(taskId, "diarizing");
            setProgress(taskId, { current: 0, total: 100, preview: "说话人分离中" });
            const dr = await ipc.diarize({
              audio: stableAudio,
              segments: result.segments,
              n_speakers: diar.n_speakers,
              engine: diar.engine || "auto",
              profiles: diar.speakers,
            });
            if (isCancelled()) return;
            if (dr.stats?.applied === false) {
              throw new Error(dr.stats.failure_reason || "说话人分离没有可应用结果");
            }
            const hasSpeakerLabels = dr.segments.every((s) => Boolean(s.speaker));
            if (!hasSpeakerLabels) {
              throw new Error("说话人分离没有返回 SPEAKER_A/B/C/D 标签");
            }
            assertDiarizationPreservesTranscript(result.segments, dr.segments);
            const diarizedResult = {
              ...result,
              segments: dr.segments,
              diarization_stats: dr.stats,
            };
            setStage(taskId, "saving");
            setProgress(taskId, { current: 99, total: 100, preview: "正在安全保存分人结果" });
            try {
              await enqueueStemWrite(stem, () => ipc.librarySaveRaw({
                stem,
                audio_filename: next.filename,
                source_audio: stableAudio,
                txt: buildTxt(diarizedResult.segments, `${next.filename}\nbackend=${diarizedResult.backend} duration=${diarizedResult.duration.toFixed(1)}s segments=${diarizedResult.segments.length}`),
                srt: buildSrt(diarizedResult.segments),
                json: buildJson(diarizedResult),
                result: diarizedResult,
              }));
              if (isCancelled()) return;
              result.segments = diarizedResult.segments;
              result.diarization_stats = diarizedResult.diarization_stats;
            } catch (e) {
              if (isCancelled()) return;
              throw new DiarizationPersistenceError(`说话人分离保存失败，已保留原始转录: ${String(e)}`);
            }
          } catch (e) {
            if (e instanceof DiarizationPersistenceError) throw e;
            diarizationError = `说话人分离失败: ${String(e)}`;
            console.warn("diarize failed (keeping transcript available)", e);
          }
        } else if (diar?.enabled && result.segments.length > 0 && !timingReliable) {
          diarizationError = "时间轴不可靠，已跳过说话人分离；转录文字已保留";
        }

        if (isCancelled()) return;
        setResult(taskId, result);
        pendingCommittedResults.delete(taskId);
        if (diarizationError) {
          setProgress(taskId, { current: 100, total: 100, preview: diarizationError });
        }
      } catch (e) {
        if (!isCancelled()) {
          const committed = pendingCommittedResults.get(taskId);
          if (committed) setResult(taskId, committed);
          setError(taskId, String(e));
        }
      } finally {
        const cancellation = pipelineState.cancelPromises.get(taskId);
        if (cancellation) await cancellation;
        pipelineState.cancelPromises.delete(taskId);
        pipelineState.cancelledIds.delete(taskId);
        pipelineState.diarizationCancelledIds.delete(taskId);
        pendingCommittedResults.delete(taskId);
        if (pipelineState.activeTaskId === taskId) {
          pipelineState.activeTaskId = null;
          if (transcribingIdRef.current === taskId) transcribingIdRef.current = null;
          wakeScheduler();
        }
      }
    })();
  }, [tasks, settings, settingsLoaded, schedulerTick, setStage, setProgress, setResult, setAudio, setLibraryStem, setError]);

  /** 触发对某个已转录任务的 LLM 校对。返回成功与否的 Promise。 */
  const runCorrection = useCallback(
    async (taskId: string) => {
      const task = useTasks.getState().tasks.find((t) => t.id === taskId);
      if (!task?.result) {
        throw new Error("任务尚未完成转录");
      }
      if (task.stage === "cancelled") {
        throw new Error("已取消任务不能直接重新校对，请重新导入或恢复任务");
      }
      if (correctionState.activeTaskId) {
        throw new Error("已有校对任务正在运行或取消收尾中");
      }
      const expectedCorrected = captureTaskCorrectedRevision(task);
      correctionState.activeTaskId = taskId;
      const isCorrectionCancelled = () => correctionState.cancelledIds.has(taskId);
      try {
        correctingIdRef.current = taskId;
        setStage(taskId, "correcting");
        setProgress(taskId, { current: 0, total: task.result.segments.length });
        const cor = await ipc.correctSegments({
          segments: task.result.segments,
          provider: settings.correction.provider,
          base_url: settings.correction.base_url,
          model: settings.correction.model,
          mode: settings.correction.mode,
          batch_size: settings.correction.batch_size,
          context_hint: settings.correction.context_hint,
          use_glossary: settings.correction.use_glossary,
          concurrency: settings.correction.concurrency,
          temperature: settings.correction.advanced.temperature,
          max_tokens: settings.correction.advanced.max_tokens,
          top_p: settings.correction.advanced.top_p,
          frequency_penalty: settings.correction.advanced.frequency_penalty,
          presence_penalty: settings.correction.advanced.presence_penalty,
          language: task.result.language || settings.language || undefined,
        });
        if (isCorrectionCancelled() || cor.cancelled) {
          setCancelled(taskId);
          return;
        }
        const corrected: CorrectedArtifact = {
          segments: cor.segments,
          changed: cor.changed,
          total: cor.total,
          model: cor.model,
          glossary: cor.glossary,
        };
        // Auto-persist corrected outputs through the shared per-stem CAS queue.
        try {
          await persistCorrectedArtifact({
            expectedRevision: expectedCorrected,
            artifact: corrected,
            buildPayload: buildPipelineCorrectedPayload,
            commitToStore: true,
          });
        } catch (e) {
          throw new Error(`校对已完成，但保存失败: ${String(e)}`);
        }
      } catch (e) {
        if (isCorrectionCancelled()) {
          setCancelled(taskId);
          return;
        }
        setError(taskId, String(e));
        throw e;
      } finally {
        correctionState.cancelledIds.delete(taskId);
        if (correctionState.activeTaskId === taskId) {
          correctionState.activeTaskId = null;
          if (correctingIdRef.current === taskId) correctingIdRef.current = null;
          wakeCorrectionScheduler();
        }
      }
    },
    [settings, setStage, setProgress, setError, setCancelled, wakeCorrectionScheduler],
  );

  /** 触发对某个任务的整篇排版。优先用校对后的 segments,没有就用原始转录。 */
  const runPolish = useCallback(
    async (taskId: string) => {
      const task = useTasks.getState().tasks.find((t) => t.id === taskId);
      if (!task?.result) {
        throw new Error("任务尚未完成转录");
      }
      const source: "corrected" | "raw" = task.corrected ? "corrected" : "raw";
      const segments = task.corrected?.segments ?? task.result.segments;
      try {
        setStage(taskId, "polishing");
        const pol = await ipc.polishArticle({
          segments,
          provider: settings.correction.provider,
          base_url: settings.correction.base_url,
          model: settings.polish.model,
          temperature: settings.polish.advanced.temperature,
          max_tokens: settings.polish.advanced.max_tokens,
          top_p: settings.polish.advanced.top_p,
          frequency_penalty: settings.polish.advanced.frequency_penalty,
          presence_penalty: settings.polish.advanced.presence_penalty,
        });
        setPolished(taskId, {
          text: pol.text,
          model: pol.model,
          source,
          truncated: pol.truncated,
          finish_reason: pol.finish_reason,
          input_chars: pol.input_chars,
        });
        const stem = task.libraryStem ?? libraryStemFromFilename(task.filename);
        try {
          await ipc.librarySavePolished({ stem, text: pol.text, model: pol.model, source });
        } catch (e) {
          throw new Error(`排版已完成，但保存失败: ${String(e)}`);
        }
      } catch (e) {
        setError(taskId, String(e));
        throw e;
      }
    },
    [settings, setStage, setPolished, setError],
  );

  /** 一键链式跑完 LLM 校对 → 整篇排版。校对失败/取消则不再排版。 */
  const runPipelineFull = useCallback(
    async (taskId: string) => {
      try {
        await runCorrection(taskId);
      } catch {
        return;
      }
      const after = useTasks.getState().tasks.find((t) => t.id === taskId);
      if (after?.stage !== "corrected") return;
      try {
        await runPolish(taskId);
      } catch {
        // already surfaces via stage="error"
      }
    },
    [runCorrection, runPolish],
  );

  // Auto-pipeline:转录完成后,如果设置开了"自动跑完整流水线"且 LLM 已启用,自动接力校对 + 排版。
  const autoTriggeredRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!settings.correction.enabled) return;
    if (!settings.correction.auto_pipeline) return;
    if (correctionState.activeTaskId) return;
    const t = tasks.find(
      (x) => x.stage === "transcribed" && !autoTriggeredRef.current.has(x.id),
    );
    if (!t) return;
    autoTriggeredRef.current.add(t.id);
    runPipelineFull(t.id).catch(() => {});
  }, [tasks, settings.correction.enabled, settings.correction.auto_pipeline, correctionSchedulerTick, runPipelineFull]);

  return { runCorrection, runPolish, runPipelineFull };
}

// Standalone control actions — safe to call from anywhere (no React state).
export async function pauseCorrection(taskId: string): Promise<void> {
  const cur = useTasks.getState().tasks.find((t) => t.id === taskId);
  if (cur?.stage !== "correcting") return;
  try {
    await ipc.correctPause();
    useTasks.getState().setStage(taskId, "correcting_paused");
  } catch (e) {
    console.warn("pause failed", e);
  }
}

export async function resumeCorrection(taskId: string): Promise<void> {
  const cur = useTasks.getState().tasks.find((t) => t.id === taskId);
  if (cur?.stage !== "correcting_paused") return;
  try {
    await ipc.correctResume();
    useTasks.getState().setStage(taskId, "correcting");
  } catch (e) {
    console.warn("resume failed", e);
  }
}

export async function cancelCorrection(taskId: string): Promise<void> {
  if (correctionState.activeTaskId !== taskId) return;
  correctionState.cancelledIds.add(taskId);
  useTasks.getState().setCancelled(taskId);
  try {
    await ipc.correctCancel();
  } catch (e) {
    console.warn("cancel failed", e);
  }
}

export function cancelTask(taskId: string): void {
  const cur = useTasks.getState().tasks.find((t) => t.id === taskId);
  if (!cur) return;

  if (cur.stage === "queued") {
    useTasks.getState().setCancelled(taskId);
    return;
  }

  if (cur.stage === "correcting" || cur.stage === "correcting_paused") {
    cancelCorrection(taskId);
    return;
  }

  if (cur.stage !== "transcribing" && cur.stage !== "diarizing") return;

  if (pipelineState.cancelPromises.has(taskId)) return;

  const pendingCommitted = pendingCommittedResults.get(taskId);
  const preserveTranscript = cur.stage === "diarizing" && Boolean(cur.result || pendingCommitted);
  pipelineState.cancelledIds.add(taskId);
  if (preserveTranscript) {
    if (!cur.result && pendingCommitted) {
      useTasks.getState().setResult(taskId, pendingCommitted);
      useTasks.getState().setStage(taskId, "diarizing");
    }
    pipelineState.diarizationCancelledIds.add(taskId);
    useTasks.getState().setProgress(taskId, {
      current: cur.progress.current,
      total: cur.progress.total || 100,
      preview: "正在取消分人，原始转录将保留",
    });
  } else {
    useTasks.getState().setCancelled(taskId);
  }

  const cancellation = (isTauriRuntime()
    ? ipc.asrCancel().then(() => undefined)
    : Promise.reject(new Error("浏览器调试模式不支持强制终止本地 ASR")))
    .then(() => {
      if (preserveTranscript) {
        useTasks.getState().setStage(taskId, "transcribed");
        useTasks.getState().setProgress(taskId, {
          current: 100,
          total: 100,
          preview: "已取消分人，原始转录已保留",
        });
      }
    })
    .catch((error) => {
      if (preserveTranscript) {
        useTasks.getState().setError(taskId, `取消分人失败: ${String(error)}`);
      } else {
        useTasks.getState().setCancelled(taskId, `取消请求失败，后台任务可能仍在退出: ${String(error)}`);
      }
    });
  pipelineState.cancelPromises.set(taskId, cancellation);
}

export async function retryTaskPersistence(taskId: string): Promise<void> {
  const task = useTasks.getState().tasks.find((item) => item.id === taskId);
  if (!task?.result) throw new Error("当前任务没有可保存的转录结果");
  if (task.stage === "cancelled") throw new Error("已取消任务不能重试保存");
  const taskResult = task.result;
  const retrySourceFingerprint = resultRevisionFingerprint(taskResult);
  const retryCorrectedFingerprint = correctedArtifactFingerprint(task.corrected);

  const stem = task.libraryStem ?? libraryStemFromFilename(task.filename);
  useTasks.getState().setStage(taskId, "saving");
  useTasks.getState().setProgress(taskId, { current: 98, total: 100, preview: "正在重试安全保存" });

  try {
    const retryRawRevision = committedRawRevisionForStem(stem);
    const rawOwnerIsCurrent = () => {
      const current = committedRawRevisionForStem(stem);
      return current.taskId === retryRawRevision.taskId
        && current.stem === retryRawRevision.stem
        && current.version === retryRawRevision.version
        && current.fingerprint === retryRawRevision.fingerprint;
    };
    const meta = await enqueueStemWrite(stem, async () => {
      const currentBefore = useTasks.getState().tasks.find((item) => item.id === taskId);
      if (
        retryRawRevision.taskId !== taskId
        || !rawOwnerIsCurrent()
        || !currentBefore?.result
        || libraryStemKey(currentBefore.libraryStem ?? libraryStemFromFilename(currentBefore.filename)) !== libraryStemKey(stem)
        || resultRevisionFingerprint(currentBefore.result) !== retrySourceFingerprint
        || correctedArtifactFingerprint(currentBefore.corrected) !== retryCorrectedFingerprint
      ) {
        throw new Error("重试保存结果在排队期间已过期，未写入磁盘");
      }
      const savedMeta = await ipc.librarySaveRawAndCorrected({
        raw: {
          stem,
          audio_filename: currentBefore.filename,
          source_audio: currentBefore.audio,
          txt: buildTxt(currentBefore.result.segments, `${currentBefore.filename}\nbackend=${currentBefore.result.backend} duration=${currentBefore.result.duration.toFixed(1)}s segments=${currentBefore.result.segments.length}`),
          srt: buildSrt(currentBefore.result.segments),
          json: buildJson(currentBefore.result),
          result: currentBefore.result,
        },
        corrected: currentBefore.corrected
          ? buildPipelineCorrectedPayload(stem, currentBefore.corrected)
          : undefined,
        clear_corrected: !currentBefore.corrected,
      });
      if (!savedMeta.audio_path) throw new Error("稳定音频副本未创建");

      const currentAfter = useTasks.getState().tasks.find((item) => item.id === taskId);
      if (
        !rawOwnerIsCurrent()
        || !currentAfter?.result
        || libraryStemKey(currentAfter.libraryStem ?? libraryStemFromFilename(currentAfter.filename)) !== libraryStemKey(stem)
        || resultRevisionFingerprint(currentAfter.result) !== retrySourceFingerprint
        || correctedArtifactFingerprint(currentAfter.corrected) !== retryCorrectedFingerprint
      ) {
        while (true) {
          const rawRevision = committedRawRevisionForStem(stem);
          const latestTask = authoritativePipelineTaskForStem(
            rawRevision.stem,
            rawRevision.taskId ?? undefined,
          );
          if (!latestTask?.result) {
            throw new Error("当前 stem 没有可恢复的 committed raw");
          }
          await ipc.librarySaveRawAndCorrected({
            raw: {
              stem: rawRevision.stem,
              audio_filename: latestTask.filename,
              source_audio: latestTask.result.audio || latestTask.audio,
              txt: buildTxt(latestTask.result.segments, `${latestTask.filename}\nbackend=${latestTask.result.backend} duration=${latestTask.result.duration.toFixed(1)}s segments=${latestTask.result.segments.length}`),
              srt: buildSrt(latestTask.result.segments),
              json: buildJson(latestTask.result),
              result: latestTask.result,
            },
            corrected: latestTask.corrected
              ? buildPipelineCorrectedPayload(rawRevision.stem, latestTask.corrected)
              : undefined,
            clear_corrected: !latestTask.corrected,
          });
          const after = committedRawRevisionForStem(stem);
          if (
            after.version === rawRevision.version
            && after.taskId === rawRevision.taskId
            && after.stem === rawRevision.stem
            && after.fingerprint === rawRevision.fingerprint
          ) {
            break;
          }
        }
        throw new Error("重试保存期间结果已变化；已恢复最新磁盘版本，本次旧结果未应用");
      }

      const nextPersistedResult = { ...currentBefore.result, audio: savedMeta.audio_path };
      useTasks.getState().setLibraryStem(taskId, savedMeta.stem || stem);
      useTasks.getState().setAudio(taskId, savedMeta.audio_path);
      useTasks.getState().setResult(taskId, nextPersistedResult);
      return savedMeta;
    });

    const latestAfterRaw = useTasks.getState().tasks.find((item) => item.id === taskId);
    if (!latestAfterRaw) throw new Error("原始结果保存后任务已不存在");

    if (task.polished) {
      await ipc.librarySavePolished({
        stem: meta.stem || stem,
        text: task.polished.text,
        model: task.polished.model,
        source: task.polished.source,
      });
    }

    useTasks.getState().clearError(taskId);
    const latest = useTasks.getState().tasks.find((item) => item.id === taskId) ?? latestAfterRaw;
    useTasks.getState().setStage(taskId, latest.polished ? "polished" : latest.corrected ? "corrected" : "transcribed");
    useTasks.getState().setProgress(taskId, { current: 100, total: 100, preview: "结果已安全保存" });
  } catch (error) {
    const message = `重试保存失败，内存结果仍保留: ${String(error)}`;
    useTasks.getState().setError(taskId, message);
    throw new Error(message);
  }
}

export type PipelineActions = ReturnType<typeof usePipeline>;
