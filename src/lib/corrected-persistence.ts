import { buildJson, buildSrt, buildTxt } from "./format";
import { ipc, libraryStemFromFilename, libraryStemKey } from "./ipc";
import { type Task, useTasks } from "../stores/tasks-store";

export type CorrectedArtifact = NonNullable<Task["corrected"]>;

type CorrectedSnapshot = Readonly<{
  artifact: CorrectedArtifact | undefined;
  fingerprint: string;
}>;

export type CorrectedRevision = CorrectedSnapshot & Readonly<{
  taskId: string;
  stem: string;
  stemKey: string;
  generation: number;
  rawTaskId: string;
  rawStem: string;
  rawFingerprint: string;
  rawVersion: number;
}>;

export type CorrectedSavePayload = Parameters<typeof ipc.librarySaveCorrected>[0];
export type CorrectedSavePayloadBuilder = (
  stem: string,
  artifact: CorrectedArtifact,
) => CorrectedSavePayload;

type CorrectedQueueState = {
  generation: number;
  headTaskId: string | null;
  headStem: string;
  headFingerprint: string;
  headArtifact: CorrectedArtifact | undefined;
  committedVersion: number;
  committedTaskId: string | null;
  committedStem: string;
  committedFingerprint: string;
  committedArtifact: CorrectedArtifact | undefined;
  committedRawVersion: number;
  committedRawTaskId: string | null;
  committedRawStem: string;
  committedRawFingerprint: string;
  committedTaskSnapshot: Task | undefined;
  lastPersistedTaskId: string | null;
  lastPersistedStem: string;
  lastPersistedFingerprint: string;
  lastPersistedArtifact: CorrectedArtifact | undefined;
};

type PersistCorrectedOptions = {
  expectedRevision: CorrectedRevision;
  artifact: CorrectedArtifact;
  buildPayload: CorrectedSavePayloadBuilder;
  commitToStore: boolean;
  requireActive?: boolean;
};

const correctedWriteTails = new Map<string, Promise<void>>();
const correctedWriteStates = new Map<string, CorrectedQueueState>();

export function enqueueStemWrite<T>(stem: string, operation: () => Promise<T>): Promise<T> {
  const stemKey = libraryStemKey(stem);
  const previous = correctedWriteTails.get(stemKey) ?? Promise.resolve();
  const run = previous.catch(() => undefined).then(operation);
  const settled = run.then(() => undefined, () => undefined);
  correctedWriteTails.set(stemKey, settled);
  void settled.finally(() => {
    if (correctedWriteTails.get(stemKey) === settled) {
      correctedWriteTails.delete(stemKey);
    }
  });
  return run;
}

function persistenceStem(task: Pick<Task, "libraryStem" | "filename">): string {
  return task.libraryStem ?? libraryStemFromFilename(task.filename);
}

function stringFingerprint(prefix: string, canonical: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < canonical.length; index += 1) {
    hash ^= canonical.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return `${prefix}:${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

export function correctedArtifactFingerprint(corrected: CorrectedArtifact | undefined): string {
  if (!corrected) return "corrected:none";
  return stringFingerprint("corrected-v1", JSON.stringify({
    model: corrected.model,
    changed: corrected.changed,
    total: corrected.total,
    glossary: corrected.glossary ?? null,
    segments: corrected.segments,
  }));
}

function rawResultFingerprint(result: Task["result"]): string {
  if (!result) return "raw:none";
  return stringFingerprint("raw-v1", JSON.stringify({
    segments: result.segments,
    diarization_stats: result.diarization_stats,
  }));
}

function correctedSnapshot(artifact: CorrectedArtifact | undefined): CorrectedSnapshot {
  return {
    artifact,
    fingerprint: correctedArtifactFingerprint(artifact),
  };
}

function correctedQueueState(
  stem: string,
  taskId: string | null,
  initial: CorrectedSnapshot,
): CorrectedQueueState {
  const key = libraryStemKey(stem);
  const existing = correctedWriteStates.get(key);
  if (existing) return existing;
  const currentStore = useTasks.getState();
  const rawOwner = authoritativeTaskForStem(currentStore.tasks, key, currentStore.activeId);
  const created: CorrectedQueueState = {
    generation: 0,
    headTaskId: taskId,
    headStem: stem,
    headFingerprint: initial.fingerprint,
    headArtifact: initial.artifact,
    committedVersion: 0,
    committedTaskId: taskId,
    committedStem: stem,
    committedFingerprint: initial.fingerprint,
    committedArtifact: initial.artifact,
    committedRawVersion: 0,
    committedRawTaskId: rawOwner?.id ?? taskId,
    committedRawStem: rawOwner ? persistenceStem(rawOwner) : stem,
    committedRawFingerprint: rawResultFingerprint(rawOwner?.result),
    committedTaskSnapshot: rawOwner,
    lastPersistedTaskId: null,
    lastPersistedStem: stem,
    lastPersistedFingerprint: "corrected:unknown",
    lastPersistedArtifact: undefined,
  };
  correctedWriteStates.set(key, created);
  return created;
}

function recordCorrectedStoreTransition(
  state: CorrectedQueueState,
  stem: string,
  taskId: string | null,
  latest: CorrectedSnapshot,
): void {
  state.generation += 1;
  state.committedVersion += 1;
  state.committedTaskId = taskId;
  state.committedStem = stem;
  state.committedFingerprint = latest.fingerprint;
  state.committedArtifact = latest.artifact;
  state.headTaskId = taskId;
  state.headStem = stem;
  state.headFingerprint = latest.fingerprint;
  state.headArtifact = latest.artifact;
}

function recordRawStoreTransition(
  state: CorrectedQueueState,
  stem: string,
  task: Task | undefined,
): void {
  state.committedRawVersion += 1;
  state.committedRawTaskId = task?.id ?? null;
  state.committedRawStem = stem;
  state.committedRawFingerprint = rawResultFingerprint(task?.result);
}

function reserveCorrectedQueueHead(
  state: CorrectedQueueState,
  revision: CorrectedRevision,
  next: CorrectedSnapshot,
): number {
  state.generation += 1;
  state.headTaskId = revision.taskId;
  state.headStem = revision.stem;
  state.headFingerprint = next.fingerprint;
  state.headArtifact = next.artifact;
  return state.generation;
}

function invalidateCorrectedReservation(state: CorrectedQueueState): void {
  state.generation += 1;
  state.headTaskId = state.committedTaskId;
  state.headStem = state.committedStem;
  state.headFingerprint = state.committedFingerprint;
  state.headArtifact = state.committedArtifact;
}

function authoritativeTaskForStem(tasks: Task[], key: string, activeId: string | null): Task | undefined {
  return tasks.find((task) => task.id === activeId && libraryStemKey(persistenceStem(task)) === key)
    ?? tasks.find((task) => libraryStemKey(persistenceStem(task)) === key);
}

// Module-lifetime observer: each stem has one authoritative task owner. Raw and
// corrected snapshots are advanced from that same owner so restore never mixes
// artifacts from different in-memory tasks sharing a library stem.
useTasks.subscribe((nextState) => {
  for (const [key, state] of correctedWriteStates) {
    const task = authoritativeTaskForStem(nextState.tasks, key, nextState.activeId);
    if (!task) continue;
    const stem = persistenceStem(task);
    state.committedTaskSnapshot = task;
    const corrected = correctedSnapshot(task?.corrected);
    if (
      state.committedTaskId !== (task?.id ?? null)
      || libraryStemKey(state.committedStem) !== key
      || state.committedFingerprint !== corrected.fingerprint
      || state.committedArtifact !== corrected.artifact
    ) {
      recordCorrectedStoreTransition(
        state,
        stem,
        task?.id ?? null,
        corrected,
      );
    }

    const rawFingerprint = rawResultFingerprint(task?.result);
    if (
      state.committedRawTaskId !== (task?.id ?? null)
      || libraryStemKey(state.committedRawStem) !== key
      || state.committedRawFingerprint !== rawFingerprint
    ) {
      recordRawStoreTransition(state, stem, task);
    }
  }
});

export function committedRawRevisionForStem(stem: string): {
  taskId: string | null;
  stem: string;
  fingerprint: string;
  version: number;
} {
  const key = libraryStemKey(stem);
  const store = useTasks.getState();
  const task = authoritativeTaskForStem(store.tasks, key, store.activeId);
  const state = correctedQueueState(
    task ? persistenceStem(task) : stem,
    task?.id ?? null,
    correctedSnapshot(task?.corrected),
  );
  const ownerStem = task ? persistenceStem(task) : stem;
  const ownerFingerprint = rawResultFingerprint(task?.result);
  if (
    state.committedRawTaskId !== (task?.id ?? null)
    || libraryStemKey(state.committedRawStem) !== key
    || state.committedRawFingerprint !== ownerFingerprint
  ) {
    recordRawStoreTransition(state, ownerStem, task);
  }
  return {
    taskId: state.committedRawTaskId,
    stem: state.committedRawStem,
    fingerprint: state.committedRawFingerprint,
    version: state.committedRawVersion,
  };
}

export function captureTaskCorrectedRevision(task: Pick<Task, "id" | "libraryStem" | "filename">): CorrectedRevision {
  const currentTask = useTasks.getState().tasks.find((candidate) => candidate.id === task.id);
  if (!currentTask) throw new Error("当前任务已不存在，无法保存校对稿");

  const stem = persistenceStem(currentTask);
  const stemKey = libraryStemKey(stem);
  const snapshot = correctedSnapshot(currentTask.corrected);
  const state = correctedQueueState(stem, currentTask.id, snapshot);
  if (
    state.committedTaskId !== currentTask.id
    || libraryStemKey(state.committedStem) !== stemKey
    || state.committedFingerprint !== snapshot.fingerprint
    || state.committedArtifact !== snapshot.artifact
  ) {
    recordCorrectedStoreTransition(state, stem, currentTask.id, snapshot);
  }
  const rawFingerprint = rawResultFingerprint(currentTask.result);
  if (
    state.committedRawTaskId !== currentTask.id
    || libraryStemKey(state.committedRawStem) !== stemKey
    || state.committedRawFingerprint !== rawFingerprint
  ) {
    recordRawStoreTransition(state, stem, currentTask);
  }
  state.committedTaskSnapshot = currentTask;
  return {
    ...snapshot,
    taskId: currentTask.id,
    stem,
    stemKey,
    generation: state.generation,
    rawTaskId: currentTask.id,
    rawStem: stem,
    rawFingerprint,
    rawVersion: state.committedRawVersion,
  };
}

function currentTaskSnapshot(revision: CorrectedRevision): {
  task: Task;
  stem: string;
  snapshot: CorrectedSnapshot;
} | null {
  const task = useTasks.getState().tasks.find((candidate) => candidate.id === revision.taskId);
  if (!task) return null;
  const stem = persistenceStem(task);
  if (libraryStemKey(stem) !== revision.stemKey) return null;
  return { task, stem, snapshot: correctedSnapshot(task.corrected) };
}

function reservationIsCurrent(
  state: CorrectedQueueState,
  revision: CorrectedRevision,
  reservationGeneration: number,
  next: CorrectedSnapshot,
): boolean {
  return state.generation === reservationGeneration
    && state.headTaskId === revision.taskId
    && libraryStemKey(state.headStem) === revision.stemKey
    && state.headFingerprint === next.fingerprint
    && state.headArtifact === next.artifact;
}

function taskRevisionIsCurrent(
  revision: CorrectedRevision,
  requireActive: boolean,
): boolean {
  const store = useTasks.getState();
  const current = currentTaskSnapshot(revision);
  return current != null
    && current.task.stage !== "cancelled"
    && (!requireActive || store.activeId === revision.taskId)
    && current.snapshot.fingerprint === revision.fingerprint
    && current.task.id === revision.rawTaskId
    && libraryStemKey(current.stem) === libraryStemKey(revision.rawStem)
    && rawResultFingerprint(current.task.result) === revision.rawFingerprint;
}

async function restoreLatestCommittedCorrected(
  state: CorrectedQueueState,
  buildPayload: CorrectedSavePayloadBuilder,
): Promise<void> {
  while (true) {
    const committedVersion = state.committedVersion;
    const committedRawVersion = state.committedRawVersion;
    const taskId = state.committedTaskId;
    const stem = state.committedStem;
    const latest = correctedSnapshot(state.committedArtifact);
    const task = state.committedTaskSnapshot;
    if (!task?.result) {
      throw new Error("保存期间内容已变化；当前没有可恢复的 committed raw");
    }
    const correctedPayload = latest.artifact
      ? buildPayload(stem, latest.artifact)
      : undefined;
    await ipc.librarySaveRawAndCorrected({
      raw: {
        stem,
        audio_filename: task.filename,
        source_audio: task.result.audio || task.audio,
        txt: buildTxt(task.result.segments, `${task.filename}\nbackend=${task.result.backend} duration=${task.result.duration.toFixed(1)}s segments=${task.result.segments.length}`),
        srt: buildSrt(task.result.segments),
        json: buildJson(task.result),
        result: task.result,
      },
      corrected: correctedPayload,
      clear_corrected: !correctedPayload,
    });
    if (
      state.committedVersion === committedVersion
      && state.committedRawVersion === committedRawVersion
      && state.committedTaskId === taskId
      && state.committedStem === stem
      && state.committedArtifact === latest.artifact
      && state.committedTaskSnapshot === task
    ) {
      state.lastPersistedTaskId = taskId;
      state.lastPersistedStem = stem;
      state.lastPersistedFingerprint = latest.fingerprint;
      state.lastPersistedArtifact = latest.artifact;
      return;
    }
  }
}

export function persistCorrectedArtifact({
  expectedRevision,
  artifact,
  buildPayload,
  commitToStore,
  requireActive = false,
}: PersistCorrectedOptions): Promise<CorrectedArtifact> {
  const state = correctedQueueState(
    expectedRevision.stem,
    expectedRevision.taskId,
    expectedRevision,
  );
  if (
    expectedRevision.generation !== state.generation
    || expectedRevision.taskId !== state.headTaskId
    || expectedRevision.fingerprint !== state.headFingerprint
    || expectedRevision.stemKey !== libraryStemKey(state.headStem)
    || expectedRevision.rawTaskId !== state.committedRawTaskId
    || expectedRevision.rawFingerprint !== state.committedRawFingerprint
    || libraryStemKey(expectedRevision.rawStem) !== libraryStemKey(state.committedRawStem)
  ) {
    return Promise.reject(new Error("校对稿已被同一文件的其他操作更新，本次旧写入已拒绝"));
  }

  const next = correctedSnapshot(artifact);
  if (!commitToStore && next.fingerprint !== expectedRevision.fingerprint) {
    return Promise.reject(new Error("仅持久化重试不能写入不同于当前 store 的校对稿"));
  }

  const reservationGeneration = reserveCorrectedQueueHead(state, expectedRevision, next);
  const previous = correctedWriteTails.get(expectedRevision.stemKey) ?? Promise.resolve();
  const run = previous.catch(() => undefined).then(async () => {
    if (
      !reservationIsCurrent(state, expectedRevision, reservationGeneration, next)
      || !taskRevisionIsCurrent(expectedRevision, requireActive)
    ) {
      throw new Error("校对稿在排队期间已变化，本次写入未执行，请基于最新内容重试");
    }

    await ipc.librarySaveCorrected(buildPayload(expectedRevision.stem, artifact));

    if (
      !reservationIsCurrent(state, expectedRevision, reservationGeneration, next)
      || !taskRevisionIsCurrent(expectedRevision, requireActive)
    ) {
      if (reservationIsCurrent(state, expectedRevision, reservationGeneration, next)) {
        const current = currentTaskSnapshot(expectedRevision);
        if (current) {
          recordCorrectedStoreTransition(
            state,
            current.stem,
            current.task.id,
            current.snapshot,
          );
          recordRawStoreTransition(state, current.stem, current.task);
          state.committedTaskSnapshot = current.task;
        } else {
          invalidateCorrectedReservation(state);
        }
      }
      try {
        await restoreLatestCommittedCorrected(state, buildPayload);
      } catch (restoreError) {
        throw new Error(`保存期间校对稿已变化，且恢复最新校对稿失败: ${String(restoreError)}`);
      }
      throw new Error("保存期间校对稿已变化；已把最新校对稿重新保存到磁盘，本次旧结果未应用");
    }

    if (commitToStore) {
      // CAS 后到 store 提交之间不再 await；observer 会同步推进 stem 世代。
      useTasks.getState().setCorrected(expectedRevision.taskId, artifact);
    }
    state.lastPersistedTaskId = expectedRevision.taskId;
    state.lastPersistedStem = expectedRevision.stem;
    state.lastPersistedFingerprint = next.fingerprint;
    state.lastPersistedArtifact = artifact;
    return artifact;
  });
  const settled = run.then(() => undefined, () => undefined);
  correctedWriteTails.set(expectedRevision.stemKey, settled);
  void settled.finally(() => {
    if (correctedWriteTails.get(expectedRevision.stemKey) === settled) {
      correctedWriteTails.delete(expectedRevision.stemKey);
    }
  });

  return run.catch((error) => {
    if (reservationIsCurrent(state, expectedRevision, reservationGeneration, next)) {
      invalidateCorrectedReservation(state);
    }
    throw error;
  });
}

type PersistCorrectedTransactionOptions = {
  expectedRevision: CorrectedRevision;
  artifact: CorrectedArtifact;
  persist: () => Promise<void>;
  restore: (revision: {
    taskId: string | null;
    stem: string;
    artifact: CorrectedArtifact | undefined;
  }) => Promise<void>;
  commit: (artifact: CorrectedArtifact) => void;
  isCurrent: () => boolean;
  requireActive?: boolean;
};

export function persistCorrectedTransaction({
  expectedRevision,
  artifact,
  persist,
  restore,
  commit,
  isCurrent,
  requireActive = false,
}: PersistCorrectedTransactionOptions): Promise<CorrectedArtifact> {
  const state = correctedQueueState(
    expectedRevision.stem,
    expectedRevision.taskId,
    expectedRevision,
  );
  if (
    expectedRevision.generation !== state.generation
    || expectedRevision.taskId !== state.headTaskId
    || expectedRevision.fingerprint !== state.headFingerprint
    || expectedRevision.stemKey !== libraryStemKey(state.headStem)
    || expectedRevision.rawTaskId !== state.committedRawTaskId
    || expectedRevision.rawFingerprint !== state.committedRawFingerprint
    || libraryStemKey(expectedRevision.rawStem) !== libraryStemKey(state.committedRawStem)
  ) {
    return Promise.reject(new Error("校对稿已被同一文件的其他操作更新，本次分人写入已拒绝"));
  }

  const next = correctedSnapshot(artifact);
  const reservationGeneration = reserveCorrectedQueueHead(state, expectedRevision, next);
  const previous = correctedWriteTails.get(expectedRevision.stemKey) ?? Promise.resolve();
  const run = previous.catch(() => undefined).then(async () => {
    if (
      !reservationIsCurrent(state, expectedRevision, reservationGeneration, next)
      || !taskRevisionIsCurrent(expectedRevision, requireActive)
      || !isCurrent()
    ) {
      throw new Error("分人结果在排队期间已过期，未写入磁盘");
    }

    await persist();

    if (
      !reservationIsCurrent(state, expectedRevision, reservationGeneration, next)
      || !taskRevisionIsCurrent(expectedRevision, requireActive)
      || !isCurrent()
    ) {
      while (true) {
        const committedVersion = state.committedVersion;
        const committedTaskId = state.committedTaskId;
        const committedStem = state.committedStem;
        const committedArtifact = state.committedArtifact;
        await restore({
          taskId: committedTaskId,
          stem: committedStem,
          artifact: committedArtifact,
        });
        if (
          state.committedVersion === committedVersion
          && state.committedTaskId === committedTaskId
          && state.committedStem === committedStem
          && state.committedArtifact === committedArtifact
        ) {
          break;
        }
      }
      throw new Error("分人保存期间正文或校对稿已变化；已恢复最新磁盘版本，本次旧结果未应用");
    }

    // No await between the final CAS and the store commit.
    commit(artifact);
    state.lastPersistedTaskId = expectedRevision.taskId;
    state.lastPersistedStem = expectedRevision.stem;
    state.lastPersistedFingerprint = next.fingerprint;
    state.lastPersistedArtifact = artifact;
    return artifact;
  });
  const settled = run.then(() => undefined, () => undefined);
  correctedWriteTails.set(expectedRevision.stemKey, settled);
  void settled.finally(() => {
    if (correctedWriteTails.get(expectedRevision.stemKey) === settled) {
      correctedWriteTails.delete(expectedRevision.stemKey);
    }
  });

  return run.catch((error) => {
    if (reservationIsCurrent(state, expectedRevision, reservationGeneration, next)) {
      invalidateCorrectedReservation(state);
    }
    throw error;
  });
}
