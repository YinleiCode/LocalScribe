//! Python sidecar manager.
//!
//! Lazy-initialised on first command call to avoid spawning child processes during
//! `applicationDidFinishLaunching` on macOS Tahoe (which causes a non-unwinding panic).

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot, Mutex, RwLock};
use tokio::time::{sleep, timeout_at, Instant};

use crate::ipc::{IpcEnvelope, ProgressEvent};

pub type ProgressReceiver = mpsc::UnboundedReceiver<ProgressEvent>;
pub type ProgressSender = mpsc::UnboundedSender<ProgressEvent>;

type PendingMap = StdMutex<HashMap<u64, oneshot::Sender<Result<Value>>>>;

const CONTROL_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const LONG_REQUEST_TIMEOUT: Duration = Duration::from_secs(6 * 60 * 60);
const TERMINATE_GRACE_PERIOD: Duration = Duration::from_millis(750);
const KILL_CONFIRM_TIMEOUT: Duration = Duration::from_secs(2);
const PROCESS_EXIT_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Every method has a finite deadline. Long-running media/LLM work gets a larger
/// budget, while control and inspection calls fail quickly when the sidecar is stuck.
fn request_timeout(method: &str) -> Duration {
    match method {
        "environment" | "check_model" | "probe_audio" | "correct_pause" | "correct_resume"
        | "correct_cancel" | "correct_status" => CONTROL_REQUEST_TIMEOUT,
        "asr_preflight_select"
        | "diarize"
        | "recommend_diarization"
        | "extract_voice_embedding"
        | "preflight_voiceprint_anchors"
        | "reidentify_speakers"
        | "transcribe"
        | "correct"
        | "polish"
        | "translate_article" => LONG_REQUEST_TIMEOUT,
        _ => DEFAULT_REQUEST_TIMEOUT,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SidecarLane {
    Asr,
    General,
}

impl SidecarLane {
    fn as_str(self) -> &'static str {
        match self {
            Self::Asr => "asr",
            Self::General => "general",
        }
    }
}

/// ASR and diarization work is deliberately isolated from general/LLM work so
/// cancelling a media job cannot fail an unrelated LLM request.
fn lane_for_method(method: &str) -> SidecarLane {
    match method {
        "asr_preflight_select"
        | "transcribe"
        | "diarize"
        | "recommend_diarization"
        | "extract_voice_embedding"
        | "preflight_voiceprint_anchors"
        | "reidentify_speakers" => SidecarLane::Asr,
        _ => SidecarLane::General,
    }
}

/// Configure Python as the leader of a fresh process group. ffmpeg and other
/// subprocesses inherit that group, allowing cancellation to terminate the whole tree.
#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    // `setpgid` is async-signal-safe and `pre_exec` is available on the pinned
    // Rust 1.77.2/toolchain-compatible Tokio API.
    unsafe {
        command.pre_exec(|| {
            if libc::setpgid(0, 0) == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        });
    }
}

/// Windows currently keeps the existing parent-process termination behaviour.
/// A Job Object would be required for Unix-equivalent descendant-tree semantics.
#[cfg(windows)]
fn configure_process_group(_command: &mut Command) {}

#[cfg(not(any(unix, windows)))]
fn configure_process_group(_command: &mut Command) {}

struct ChildProcess {
    child: Mutex<Child>,
    #[cfg(unix)]
    process_group: i32,
}

impl ChildProcess {
    fn new(child: Child) -> Result<Self> {
        #[cfg(unix)]
        let process_group = {
            let pid = child
                .id()
                .ok_or_else(|| anyhow!("spawned sidecar has no process id"))?;
            i32::try_from(pid).context("sidecar process id exceeds i32")?
        };

        Ok(Self {
            child: Mutex::new(child),
            #[cfg(unix)]
            process_group,
        })
    }

    /// Terminate and confirm exit. This method is idempotent and serialized so
    /// concurrent timeout/EOF/cancel paths cannot race process cleanup.
    async fn terminate(&self) -> Result<()> {
        let mut child = self.child.lock().await;

        #[cfg(unix)]
        {
            return terminate_unix_process_group(&mut child, self.process_group).await;
        }

        #[cfg(windows)]
        {
            return terminate_windows_parent(&mut child).await;
        }

        #[cfg(not(any(unix, windows)))]
        {
            terminate_other_parent(&mut child).await
        }
    }
}

#[cfg(unix)]
fn signal_process_group(process_group: i32, signal: i32) -> std::io::Result<()> {
    if process_group <= 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "process group id must be positive",
        ));
    }

    let result = unsafe { libc::kill(-process_group, signal) };
    if result == 0 {
        return Ok(());
    }

    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        // The group already exited, which satisfies termination.
        Ok(())
    } else {
        Err(error)
    }
}

#[cfg(unix)]
fn process_group_exists(process_group: i32) -> std::io::Result<bool> {
    let result = unsafe { libc::kill(-process_group, 0) };
    if result == 0 {
        return Ok(true);
    }

    let error = std::io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ESRCH) => Ok(false),
        // The group exists even if the current process cannot signal it.
        Some(libc::EPERM) => Ok(true),
        _ => Err(error),
    }
}

#[cfg(unix)]
fn unix_exit_confirmed(child: &mut Child, process_group: i32) -> Result<bool> {
    let parent_exited = child
        .try_wait()
        .context("failed to query sidecar parent process status")?
        .is_some();
    let group_exited = !process_group_exists(process_group)
        .context("failed to query sidecar process group status")?;
    Ok(parent_exited && group_exited)
}

#[cfg(unix)]
async fn wait_for_unix_exit(
    child: &mut Child,
    process_group: i32,
    wait_for: Duration,
) -> Result<bool> {
    let deadline = Instant::now() + wait_for;
    loop {
        if unix_exit_confirmed(child, process_group)? {
            return Ok(true);
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        sleep(PROCESS_EXIT_POLL_INTERVAL).await;
    }
}

#[cfg(unix)]
async fn terminate_unix_process_group(child: &mut Child, process_group: i32) -> Result<()> {
    if unix_exit_confirmed(child, process_group)? {
        return Ok(());
    }

    signal_process_group(process_group, libc::SIGTERM)
        .with_context(|| format!("failed to send SIGTERM to process group {process_group}"))?;
    if wait_for_unix_exit(child, process_group, TERMINATE_GRACE_PERIOD).await? {
        return Ok(());
    }

    signal_process_group(process_group, libc::SIGKILL)
        .with_context(|| format!("failed to send SIGKILL to process group {process_group}"))?;
    if wait_for_unix_exit(child, process_group, KILL_CONFIRM_TIMEOUT).await? {
        return Ok(());
    }

    Err(anyhow!(
        "sidecar process group {process_group} did not exit after SIGKILL"
    ))
}

#[cfg(windows)]
async fn terminate_windows_parent(child: &mut Child) -> Result<()> {
    if child
        .try_wait()
        .context("failed to query sidecar process status")?
        .is_some()
    {
        return Ok(());
    }

    match child.kill().await {
        Ok(()) => {}
        Err(error) => {
            if child
                .try_wait()
                .context("failed to query sidecar process after kill failure")?
                .is_none()
            {
                return Err(error).context("failed to terminate sidecar parent process");
            }
        }
    }

    if child
        .try_wait()
        .context("failed to confirm sidecar parent process exit")?
        .is_some()
    {
        Ok(())
    } else {
        Err(anyhow!(
            "sidecar parent process termination was not confirmed on Windows"
        ))
    }
}

#[cfg(not(any(unix, windows)))]
async fn terminate_other_parent(child: &mut Child) -> Result<()> {
    if child
        .try_wait()
        .context("failed to query sidecar process status")?
        .is_some()
    {
        return Ok(());
    }
    child
        .kill()
        .await
        .context("failed to terminate sidecar parent process")?;
    if child
        .try_wait()
        .context("failed to confirm sidecar parent process exit")?
        .is_some()
    {
        Ok(())
    } else {
        Err(anyhow!(
            "sidecar parent process termination was not confirmed"
        ))
    }
}

struct HandleState {
    alive: AtomicBool,
    pending: PendingMap,
}

impl HandleState {
    fn new() -> Self {
        Self {
            alive: AtomicBool::new(true),
            pending: StdMutex::new(HashMap::new()),
        }
    }

    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }

    fn pending(&self) -> std::sync::MutexGuard<'_, HashMap<u64, oneshot::Sender<Result<Value>>>> {
        self.pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn register(
        self: &Arc<Self>,
        id: u64,
        tx: oneshot::Sender<Result<Value>>,
    ) -> Result<PendingGuard> {
        if !self.is_alive() {
            return Err(anyhow!("sidecar handle is not alive"));
        }

        self.pending().insert(id, tx);
        let guard = PendingGuard {
            id,
            state: self.clone(),
        };

        // Close the race where stdout fails after the first liveness check but
        // before this request is inserted into the pending map.
        if !self.is_alive() {
            drop(guard);
            return Err(anyhow!("sidecar handle became unavailable"));
        }

        Ok(guard)
    }

    fn take_pending(&self, id: u64) -> Option<oneshot::Sender<Result<Value>>> {
        self.pending().remove(&id)
    }

    fn mark_dead(&self) {
        self.alive.store(false, Ordering::Release);
    }

    fn fail_pending(&self, reason: impl Into<String>) {
        let reason = reason.into();

        // Always drain, even if another task already marked the state dead: a
        // registering caller may have raced with the earlier drain.
        let senders: Vec<_> = self.pending().drain().map(|(_, tx)| tx).collect();
        for tx in senders {
            let _ = tx.send(Err(anyhow!(reason.clone())));
        }
    }

    fn invalidate(&self, reason: impl Into<String>) {
        self.mark_dead();
        self.fail_pending(reason);
    }

    #[cfg(test)]
    fn pending_len(&self) -> usize {
        self.pending().len()
    }
}

/// Synchronous drop cleanup makes cancellation safe: aborting/dropping a call
/// future cannot strand its entry in `pending`.
struct PendingGuard {
    id: u64,
    state: Arc<HandleState>,
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        self.state.take_pending(self.id);
    }
}

/// If a call future is cancelled after a write starts, the JSON line may be
/// incomplete. Invalidate the generation synchronously so it cannot be reused.
struct WriteIntegrityGuard<'a> {
    state: &'a HandleState,
    method: &'a str,
    id: u64,
    armed: bool,
}

impl<'a> WriteIntegrityGuard<'a> {
    fn new(state: &'a HandleState, method: &'a str, id: u64) -> Self {
        Self {
            state,
            method,
            id,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for WriteIntegrityGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.state.invalidate(format!(
                "sidecar stdin write cancelled for method={} id={}",
                self.method, self.id
            ));
        }
    }
}

async fn write_payload<W>(
    state: &HandleState,
    writer: &mut W,
    payload: &[u8],
    method: &str,
    id: u64,
) -> std::io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let mut integrity_guard = WriteIntegrityGuard::new(state, method, id);
    writer.write_all(payload).await?;
    writer.flush().await?;
    integrity_guard.disarm();
    Ok(())
}

async fn invalidate_and_terminate(
    state: &HandleState,
    process: &ChildProcess,
    reason: String,
) -> Result<()> {
    // Reject new requests immediately, but do not wake existing pending calls
    // until process-tree exit has been confirmed (or confirmation itself fails).
    state.mark_dead();
    match process.terminate().await {
        Ok(()) => {
            state.fail_pending(reason);
            Ok(())
        }
        Err(error) => {
            let pending_reason =
                format!("{reason}; sidecar termination was not confirmed: {error:#}");
            state.fail_pending(pending_reason);
            Err(error).with_context(|| format!("{reason}; sidecar termination was not confirmed"))
        }
    }
}

async fn write_with_deadline<F>(
    state: &HandleState,
    process: &ChildProcess,
    deadline: Instant,
    method: &str,
    id: u64,
    write: F,
) -> Result<()>
where
    F: Future<Output = std::io::Result<()>>,
{
    match timeout_at(deadline, write).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => {
            let message =
                format!("sidecar stdin write failed for method={method} id={id}: {error}");
            invalidate_and_terminate(state, process, message.clone()).await?;
            Err(anyhow!(message))
        }
        Err(_) => {
            let message = format!("sidecar stdin write timed out for method={method} id={id}");
            // The write may have been partial, so the protocol stream is no longer
            // safe to reuse. Termination is confirmed before this failure returns.
            invalidate_and_terminate(state, process, message.clone()).await?;
            Err(anyhow!(message))
        }
    }
}

async fn receive_with_deadline(
    state: &HandleState,
    process: &ChildProcess,
    rx: oneshot::Receiver<Result<Value>>,
    deadline: Instant,
    method: &str,
    id: u64,
) -> Result<Value> {
    match timeout_at(deadline, rx).await {
        // Application-level sidecar errors are returned without killing the lane.
        Ok(Ok(outcome)) => outcome,
        Ok(Err(_)) => {
            let message = format!("sidecar dropped response channel for method={method} id={id}");
            invalidate_and_terminate(state, process, message.clone()).await?;
            Err(anyhow!(message))
        }
        Err(_) => {
            let message = format!("sidecar request timed out for method={method} id={id}");
            // Treat a missed response deadline as generation-fatal. The request is
            // never replayed, and the process must exit before a later generation.
            invalidate_and_terminate(state, process, message.clone()).await?;
            Err(anyhow!(message))
        }
    }
}

fn dispatch_stdout_line(line: &str, state: &HandleState, progress_tx: &ProgressSender) {
    match serde_json::from_str::<IpcEnvelope>(line) {
        Ok(IpcEnvelope::Response { id, result, error }) => {
            let outcome = match (result, error) {
                (Some(value), None) => Ok(value),
                (_, Some(error)) => Err(anyhow!("sidecar error: {}", error.message)),
                (None, None) => Err(anyhow!("sidecar response missing both result and error")),
            };
            if let Some(tx) = state.take_pending(id) {
                let _ = tx.send(outcome);
            } else {
                tracing::warn!("orphan sidecar response id={id}");
            }
        }
        Ok(IpcEnvelope::Progress(event)) => {
            // Once a generation is invalidated, suppress late progress so a
            // stale process cannot leak events into its replacement.
            if state.is_alive() {
                let _ = progress_tx.send(event);
            }
        }
        Err(error) => {
            tracing::error!("malformed sidecar line: {error}; raw={line}");
        }
    }
}

async fn run_stdout_reader<R>(
    stdout: R,
    state: Arc<HandleState>,
    progress_tx: ProgressSender,
    process: Arc<ChildProcess>,
) -> Result<()>
where
    R: AsyncRead + Unpin,
{
    let mut reader = BufReader::new(stdout).lines();
    let reason = loop {
        match reader.next_line().await {
            Ok(Some(line)) => {
                if !line.is_empty() {
                    dispatch_stdout_line(&line, &state, &progress_tx);
                }
            }
            Ok(None) => {
                tracing::warn!("sidecar stdout reached EOF");
                break "sidecar stdout reached EOF".to_string();
            }
            Err(error) => {
                tracing::error!("sidecar stdout read failed: {error}");
                break format!("sidecar stdout read failed: {error}");
            }
        }
    };

    invalidate_and_terminate(&state, &process, reason).await
}

#[derive(Clone)]
pub struct SidecarHandle {
    next_id: Arc<AtomicU64>,
    state: Arc<HandleState>,
    stdin: Arc<Mutex<ChildStdin>>,
    progress_tx: ProgressSender,
    process: Arc<ChildProcess>,
}

impl SidecarHandle {
    /// Spawn `python -m scribe_py ipc` in an isolated process group on Unix.
    pub async fn spawn(
        python: PathBuf,
        scribe_py_dir: PathBuf,
    ) -> Result<(Self, ProgressReceiver)> {
        let mut cmd = Command::new(&python);
        let python_paths = vec![scribe_py_dir.join("src")];
        let maybe_repo_root = scribe_py_dir.parent().map(|path| path.to_path_buf());
        if let Some(root) = maybe_repo_root {
            let bundled_site =
                root.join("src-tauri/bundle-staging/python/lib/python3.12/site-packages");
            if bundled_site.is_dir() {
                cmd.env("LOCALSCRIBE_BUNDLED_SITE_PACKAGES", bundled_site);
            }
        }
        let pythonpath =
            std::env::join_paths(python_paths).context("failed to construct sidecar PYTHONPATH")?;
        // `-B` is stronger than PYTHONDONTWRITEBYTECODE: it prevents Python
        // from creating __pycache__ inside the signed application bundle.
        cmd.args(["-B", "-m", "scribe_py", "ipc"])
            .env("PYTHONPATH", pythonpath)
            .env("PYTHONUNBUFFERED", "1")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        configure_process_group(&mut cmd);

        // Ensure PATH is inherited so Python can find ffmpeg.
        if let Ok(path) = std::env::var("PATH") {
            cmd.env("PATH", path);
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to spawn sidecar at {}", python.display()))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("no stdin on sidecar"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("no stdout on sidecar"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("no stderr on sidecar"))?;

        let process = Arc::new(ChildProcess::new(child)?);
        let state = Arc::new(HandleState::new());
        let (progress_tx, progress_rx) = mpsc::unbounded_channel::<ProgressEvent>();

        // stderr → tracing
        tokio::spawn(async move {
            let mut reader = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                tracing::warn!(target: "scribe_py.stderr", "{}", line);
            }
        });

        // stdout → dispatch. EOF/read errors invalidate the generation, fail all
        // pending requests, and synchronously confirm process-tree termination.
        let reader_state = state.clone();
        let reader_progress_tx = progress_tx.clone();
        let reader_process = process.clone();
        tokio::spawn(async move {
            if let Err(error) =
                run_stdout_reader(stdout, reader_state, reader_progress_tx, reader_process).await
            {
                tracing::error!("sidecar stdout failure cleanup failed: {error:#}");
            }
        });

        Ok((
            SidecarHandle {
                next_id: Arc::new(AtomicU64::new(1)),
                state,
                stdin: Arc::new(Mutex::new(stdin)),
                progress_tx,
                process,
            },
            progress_rx,
        ))
    }

    fn is_alive(&self) -> bool {
        self.state.is_alive()
    }

    /// Send one request and wait for its response using that method's hard timeout.
    pub async fn call(&self, method: &str, params: Value) -> Result<Value> {
        self.call_with_timeout(method, params, request_timeout(method))
            .await
    }

    /// Send one request with an explicit hard timeout. The request is attempted
    /// exactly once; transport failures and timeouts are never automatically replayed.
    pub async fn call_with_timeout(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value> {
        if !self.is_alive() {
            return Err(anyhow!("sidecar handle is not alive"));
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let payload = serde_json::json!({
            "id": id,
            "method": method,
            "params": params,
        });
        // Serialize before registration so even a serialization failure cannot
        // create a pending entry.
        let line = serde_json::to_string(&payload)? + "\n";
        let deadline = Instant::now() + timeout;
        let (tx, rx) = oneshot::channel();
        let _pending_guard = self.state.register(id, tx)?;

        let write = async {
            let mut stdin = self.stdin.lock().await;
            write_payload(&self.state, &mut *stdin, line.as_bytes(), method, id).await
        };
        write_with_deadline(&self.state, &self.process, deadline, method, id, write).await?;

        receive_with_deadline(&self.state, &self.process, rx, deadline, method, id).await
    }

    /// Mark this generation unusable, fail its pending requests, and confirm the
    /// Python process tree has exited.
    async fn terminate(&self) -> Result<()> {
        invalidate_and_terminate(&self.state, &self.process, "sidecar terminated".to_string()).await
    }

    #[allow(dead_code)]
    pub fn progress_sender(&self) -> ProgressSender {
        self.progress_tx.clone()
    }
}

// ============================================================================
// Lazy-init wrapper
// ============================================================================

/// Resettable single-flight storage. Failed initialisation leaves the slot empty,
/// while failed stale cleanup retains the old slot and forbids a new generation.
struct RecoverableSlot<T> {
    current: RwLock<Option<Arc<T>>>,
    spawn_gate: Mutex<()>,
}

impl<T> RecoverableSlot<T> {
    fn new() -> Self {
        Self {
            current: RwLock::new(None),
            spawn_gate: Mutex::new(()),
        }
    }

    async fn get_or_try_init<IsValid, Cleanup, CleanupFuture, Init, InitFuture, Error>(
        &self,
        is_valid: IsValid,
        cleanup: Cleanup,
        init: Init,
    ) -> std::result::Result<Arc<T>, Error>
    where
        IsValid: Fn(&T) -> bool,
        Cleanup: FnOnce(Arc<T>) -> CleanupFuture,
        CleanupFuture: Future<Output = std::result::Result<(), Error>>,
        Init: FnOnce() -> InitFuture,
        InitFuture: Future<Output = std::result::Result<T, Error>>,
    {
        {
            let current = self.current.read().await;
            if let Some(value) = current.as_ref() {
                if is_valid(value.as_ref()) {
                    return Ok(value.clone());
                }
            }
        }

        let _spawn_guard = self.spawn_gate.lock().await;

        // Another caller may have completed recovery while this caller waited.
        {
            let current = self.current.read().await;
            if let Some(value) = current.as_ref() {
                if is_valid(value.as_ref()) {
                    return Ok(value.clone());
                }
            }
        }

        let stale = self.current.read().await.clone();
        if let Some(stale) = stale {
            // Cleanup is fallible. Keep the slot intact until exit is confirmed;
            // on failure, `?` returns without allowing `init` to spawn.
            cleanup(stale.clone()).await?;
            let mut current = self.current.write().await;
            if current
                .as_ref()
                .map(|value| Arc::ptr_eq(value, &stale))
                .unwrap_or(false)
            {
                current.take();
            }
        }

        let value = Arc::new(init().await?);
        *self.current.write().await = Some(value.clone());
        Ok(value)
    }

    /// Terminate the current value without removing it until termination succeeds.
    async fn terminate_current<Terminate, TerminateFuture, Error>(
        &self,
        terminate: Terminate,
    ) -> std::result::Result<(), Error>
    where
        Terminate: FnOnce(Arc<T>) -> TerminateFuture,
        TerminateFuture: Future<Output = std::result::Result<(), Error>>,
    {
        let _spawn_guard = self.spawn_gate.lock().await;
        let current = self.current.read().await.clone();
        if let Some(current) = current {
            terminate(current.clone()).await?;
            let mut slot = self.current.write().await;
            if slot
                .as_ref()
                .map(|value| Arc::ptr_eq(value, &current))
                .unwrap_or(false)
            {
                slot.take();
            }
        }
        Ok(())
    }
}

struct LaneSlots<T> {
    asr: RecoverableSlot<T>,
    general: RecoverableSlot<T>,
}

impl<T> LaneSlots<T> {
    fn new() -> Self {
        Self {
            asr: RecoverableSlot::new(),
            general: RecoverableSlot::new(),
        }
    }

    fn get(&self, lane: SidecarLane) -> &RecoverableSlot<T> {
        match lane {
            SidecarLane::Asr => &self.asr,
            SidecarLane::General => &self.general,
        }
    }
}

/// Holder put into Tauri state. Each lane is spawned independently on first access.
pub struct SidecarLazy {
    lanes: LaneSlots<SidecarHandle>,
    python: PathBuf,
    scribe_py_dir: PathBuf,
    /// AppHandle is needed to forward progress events once a sidecar starts.
    app: AppHandle,
}

impl SidecarLazy {
    pub fn new(app: AppHandle, python: PathBuf, scribe_py_dir: PathBuf) -> Self {
        Self {
            lanes: LaneSlots::new(),
            python,
            scribe_py_dir,
            app,
        }
    }

    async fn spawn_handle(&self, lane: SidecarLane) -> Result<SidecarHandle> {
        tracing::info!(
            "(lazy) spawning {} sidecar python={} dir={}",
            lane.as_str(),
            self.python.display(),
            self.scribe_py_dir.display()
        );
        let (handle, mut progress_rx) =
            SidecarHandle::spawn(self.python.clone(), self.scribe_py_dir.clone()).await?;

        let app = self.app.clone();
        tokio::spawn(async move {
            while let Some(event) = progress_rx.recv().await {
                let topic = format!("scribe://progress/{}", event.method);
                if let Err(error) = app.emit(&topic, &event.data) {
                    tracing::warn!("emit progress failed: {error}");
                }
            }
        });
        tracing::info!("{} sidecar handle ready (lazy)", lane.as_str());
        Ok(handle)
    }

    async fn handle_for_lane(&self, lane: SidecarLane) -> Result<Arc<SidecarHandle>> {
        self.lanes
            .get(lane)
            .get_or_try_init(
                SidecarHandle::is_alive,
                |stale| async move { stale.terminate().await },
                || self.spawn_handle(lane),
            )
            .await
    }

    /// Execute exactly one attempt on the lane selected by `method`. If that
    /// generation fails, a later independently initiated call may recover, but
    /// this request is never automatically replayed.
    pub async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let lane = lane_for_method(method);
        let handle = self.handle_for_lane(lane).await?;
        handle.call(method, params).await
    }

    /// Stop only the ASR/diarization generation. General/LLM pending work and
    /// its process are not touched. A later ASR call lazily starts a new generation.
    pub async fn terminate_asr(&self) -> Result<()> {
        self.lanes
            .get(SidecarLane::Asr)
            .terminate_current(|handle| async move { handle.terminate().await })
            .await
    }

    /// Backward-compatible alias used by the existing ASR cancel command.
    pub async fn terminate(&self) -> Result<()> {
        self.terminate_asr().await
    }

    /// Restart only the ASR lane, retaining the same isolation as cancellation.
    pub async fn restart(&self) -> Result<()> {
        let slot = self.lanes.get(SidecarLane::Asr);
        let _spawn_guard = slot.spawn_gate.lock().await;
        let current = slot.current.read().await.clone();
        if let Some(current) = current {
            current.terminate().await?;
            let mut stored = slot.current.write().await;
            if stored
                .as_ref()
                .map(|value| Arc::ptr_eq(value, &current))
                .unwrap_or(false)
            {
                stored.take();
            }
        }

        let handle = Arc::new(self.spawn_handle(SidecarLane::Asr).await?);
        *slot.current.write().await = Some(handle);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::task::{Context as TaskContext, Poll};
    use tokio::io::ReadBuf;
    use tokio::sync::Notify;

    struct ErrorReader;

    impl AsyncRead for ErrorReader {
        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut TaskContext<'_>,
            _buf: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            Poll::Ready(Err(io::Error::new(
                io::ErrorKind::Other,
                "synthetic read failure",
            )))
        }
    }

    struct PartialThenPendingWriter {
        wrote_prefix: bool,
        wrote_prefix_notify: Arc<Notify>,
    }

    impl AsyncWrite for PartialThenPendingWriter {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _cx: &mut TaskContext<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            if self.wrote_prefix {
                Poll::Pending
            } else {
                self.wrote_prefix = true;
                self.wrote_prefix_notify.notify_one();
                Poll::Ready(Ok(buf.len().min(1)))
            }
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
            Poll::Pending
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[test]
    fn method_timeouts_are_finite_and_method_specific() {
        let control = request_timeout("environment");
        let default = request_timeout("future_unknown_method");
        let long = request_timeout("transcribe");

        assert!(control > Duration::ZERO);
        assert!(control < default);
        assert!(default < long);
    }

    #[test]
    fn method_routing_isolates_asr_and_general_lanes() {
        for method in [
            "asr_preflight_select",
            "transcribe",
            "diarize",
            "recommend_diarization",
            "extract_voice_embedding",
            "reidentify_speakers",
        ] {
            assert_eq!(lane_for_method(method), SidecarLane::Asr, "{method}");
        }

        for method in [
            "environment",
            "probe_audio",
            "correct",
            "polish",
            "translate_article",
            "correct_cancel",
        ] {
            assert_eq!(lane_for_method(method), SidecarLane::General, "{method}");
        }
    }

    #[tokio::test]
    async fn receiver_drop_removes_pending_request() {
        let state = Arc::new(HandleState::new());
        let (tx, rx) = oneshot::channel();
        let guard = state.register(7, tx).unwrap();
        assert_eq!(state.pending_len(), 1);

        drop(rx);
        drop(guard);
        assert_eq!(state.pending_len(), 0);
    }

    #[tokio::test]
    async fn cancelled_partial_write_invalidates_and_cleans_pending() {
        let state = Arc::new(HandleState::new());
        let wrote_prefix_notify = Arc::new(Notify::new());
        let task_state = state.clone();
        let task_notify = wrote_prefix_notify.clone();

        let task = tokio::spawn(async move {
            let (tx, _rx) = oneshot::channel();
            let _pending_guard = task_state.register(13, tx).unwrap();
            let mut writer = PartialThenPendingWriter {
                wrote_prefix: false,
                wrote_prefix_notify: task_notify,
            };
            write_payload(&task_state, &mut writer, b"partial\n", "test", 13).await
        });

        wrote_prefix_notify.notified().await;
        task.abort();
        let _ = task.await;

        assert!(!state.is_alive());
        assert_eq!(state.pending_len(), 0);
    }

    #[tokio::test]
    async fn terminating_asr_slot_leaves_general_pending() {
        let lanes = LaneSlots::<HandleState>::new();
        let asr = Arc::new(HandleState::new());
        let general = Arc::new(HandleState::new());
        *lanes.asr.current.write().await = Some(asr.clone());
        *lanes.general.current.write().await = Some(general.clone());

        let (asr_tx, asr_rx) = oneshot::channel();
        let _asr_guard = asr.register(31, asr_tx).unwrap();
        let (general_tx, general_rx) = oneshot::channel();
        let _general_guard = general.register(32, general_tx).unwrap();

        lanes
            .asr
            .terminate_current(|state| async move {
                state.invalidate("ASR lane cancelled");
                Ok::<(), &'static str>(())
            })
            .await
            .unwrap();

        assert!(lanes.asr.current.read().await.is_none());
        let stored_general = lanes.general.current.read().await.clone().unwrap();
        assert!(Arc::ptr_eq(&stored_general, &general));
        assert!(general.is_alive());
        assert_eq!(general.pending_len(), 1);
        assert!(asr_rx
            .await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("ASR"));

        general
            .take_pending(32)
            .unwrap()
            .send(Ok(serde_json::json!({ "status": "still-running" })))
            .unwrap();
        assert_eq!(
            general_rx.await.unwrap().unwrap()["status"],
            "still-running"
        );
    }

    #[tokio::test]
    async fn recoverable_slot_spawns_once_for_concurrent_callers() {
        let slot = Arc::new(RecoverableSlot::<usize>::new());
        let spawn_count = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();

        for _ in 0..16 {
            let slot = slot.clone();
            let spawn_count = spawn_count.clone();
            tasks.push(tokio::spawn(async move {
                slot.get_or_try_init(
                    |_| true,
                    |_| async { Ok::<(), ()>(()) },
                    || async move {
                        spawn_count.fetch_add(1, AtomicOrdering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        Ok::<usize, ()>(42)
                    },
                )
                .await
                .unwrap()
            }));
        }

        let first = tasks.remove(0).await.unwrap();
        for task in tasks {
            let value = task.await.unwrap();
            assert!(Arc::ptr_eq(&first, &value));
        }
        assert_eq!(spawn_count.load(AtomicOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn recoverable_slot_retries_after_failed_spawn() {
        let slot = RecoverableSlot::<usize>::new();
        let first = slot
            .get_or_try_init(
                |_| true,
                |_| async { Ok::<(), &'static str>(()) },
                || async { Err::<usize, &'static str>("first spawn failed") },
            )
            .await;
        assert_eq!(first.unwrap_err(), "first spawn failed");

        let second = slot
            .get_or_try_init(
                |_| true,
                |_| async { Ok::<(), &'static str>(()) },
                || async { Ok::<usize, &'static str>(7) },
            )
            .await
            .unwrap();
        assert_eq!(*second, 7);
    }

    #[tokio::test]
    async fn stale_cleanup_failure_retains_slot_and_blocks_spawn() {
        let slot = RecoverableSlot::<usize>::new();
        let stale = Arc::new(41usize);
        *slot.current.write().await = Some(stale.clone());
        let spawn_count = Arc::new(AtomicUsize::new(0));

        let result = slot
            .get_or_try_init(
                |_| false,
                |_| async { Err::<(), &'static str>("cleanup not confirmed") },
                {
                    let spawn_count = spawn_count.clone();
                    move || async move {
                        spawn_count.fetch_add(1, AtomicOrdering::SeqCst);
                        Ok::<usize, &'static str>(42)
                    }
                },
            )
            .await;

        assert_eq!(result.unwrap_err(), "cleanup not confirmed");
        assert_eq!(spawn_count.load(AtomicOrdering::SeqCst), 0);
        let retained = slot.current.read().await.clone().unwrap();
        assert!(Arc::ptr_eq(&retained, &stale));
    }

    #[tokio::test]
    async fn terminate_failure_does_not_drop_current_slot() {
        let slot = RecoverableSlot::<usize>::new();
        let current = Arc::new(9usize);
        *slot.current.write().await = Some(current.clone());

        let result = slot
            .terminate_current(|_| async { Err::<(), &'static str>("kill failed") })
            .await;
        assert_eq!(result.unwrap_err(), "kill failed");

        let retained = slot.current.read().await.clone().unwrap();
        assert!(Arc::ptr_eq(&retained, &current));
    }

    #[cfg(unix)]
    struct TestGeneration {
        alive: AtomicBool,
        process: Arc<ChildProcess>,
        parent_pid: i32,
        child_pid: i32,
    }

    #[cfg(unix)]
    static TEST_PROCESS_SEQUENCE: AtomicU64 = AtomicU64::new(1);

    #[cfg(unix)]
    async fn spawn_test_generation() -> Result<TestGeneration> {
        spawn_test_generation_with_term_behavior(false).await
    }

    #[cfg(unix)]
    async fn spawn_test_generation_ignoring_sigterm() -> Result<TestGeneration> {
        spawn_test_generation_with_term_behavior(true).await
    }

    #[cfg(unix)]
    async fn spawn_test_generation_with_term_behavior(
        ignore_sigterm: bool,
    ) -> Result<TestGeneration> {
        let sequence = TEST_PROCESS_SEQUENCE.fetch_add(1, Ordering::SeqCst);
        let pid_file = std::env::temp_dir().join(format!(
            "localscribe-sidecar-test-{}-{sequence}.pid",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&pid_file);

        let script = if ignore_sigterm {
            "trap '' TERM; sleep 30 & echo $! > \"$1\"; wait"
        } else {
            "sleep 30 & echo $! > \"$1\"; wait"
        };
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(script)
            .arg("localscribe-sidecar-test")
            .arg(&pid_file)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        configure_process_group(&mut command);

        let child = command
            .spawn()
            .context("failed to spawn test process group")?;
        let parent_pid = i32::try_from(child.id().unwrap()).unwrap();
        let process = Arc::new(ChildProcess::new(child)?);
        let deadline = Instant::now() + Duration::from_secs(2);
        let child_pid = loop {
            match std::fs::read_to_string(&pid_file) {
                Ok(raw) => match raw.trim().parse::<i32>() {
                    Ok(pid) => break pid,
                    Err(_) => {}
                },
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => {
                    let _ = process.terminate().await;
                    return Err(error).context("failed to read test child pid file");
                }
            }

            if Instant::now() >= deadline {
                let _ = process.terminate().await;
                return Err(anyhow!("timed out waiting for test child pid"));
            }
            sleep(Duration::from_millis(10)).await;
        };
        let _ = std::fs::remove_file(&pid_file);

        Ok(TestGeneration {
            alive: AtomicBool::new(true),
            process,
            parent_pid,
            child_pid,
        })
    }

    #[cfg(unix)]
    fn unix_process_exists(pid: i32) -> bool {
        let result = unsafe { libc::kill(pid, 0) };
        if result == 0 {
            true
        } else {
            std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
        }
    }

    #[cfg(unix)]
    fn unix_process_group(pid: i32) -> i32 {
        unsafe { libc::getpgid(pid) }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn response_timeout_terminates_and_confirms_process_group() {
        let generation = spawn_test_generation().await.unwrap();
        let state = Arc::new(HandleState::new());
        let (tx, rx) = oneshot::channel();
        let _guard = state.register(14, tx).unwrap();
        let deadline = Instant::now() + Duration::from_millis(10);

        let result =
            receive_with_deadline(&state, &generation.process, rx, deadline, "test", 14).await;

        assert!(result.unwrap_err().to_string().contains("timed out"));
        assert!(!state.is_alive());
        assert_eq!(state.pending_len(), 0);
        assert!(!unix_process_exists(generation.parent_pid));
        assert!(!unix_process_exists(generation.child_pid));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stdout_eof_terminates_and_confirms_process_group() {
        let generation = spawn_test_generation().await.unwrap();
        let state = Arc::new(HandleState::new());
        let (tx, rx) = oneshot::channel();
        let _guard = state.register(21, tx).unwrap();
        let (progress_tx, _progress_rx) = mpsc::unbounded_channel();

        run_stdout_reader(
            tokio::io::empty(),
            state.clone(),
            progress_tx,
            generation.process.clone(),
        )
        .await
        .unwrap();

        assert!(!state.is_alive());
        assert_eq!(state.pending_len(), 0);
        assert!(rx.await.unwrap().unwrap_err().to_string().contains("EOF"));
        assert!(!unix_process_exists(generation.parent_pid));
        assert!(!unix_process_exists(generation.child_pid));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stdout_read_error_terminates_and_confirms_process_group() {
        let generation = spawn_test_generation().await.unwrap();
        let state = Arc::new(HandleState::new());
        let (tx, rx) = oneshot::channel();
        let _guard = state.register(22, tx).unwrap();
        let (progress_tx, _progress_rx) = mpsc::unbounded_channel();

        run_stdout_reader(
            ErrorReader,
            state.clone(),
            progress_tx,
            generation.process.clone(),
        )
        .await
        .unwrap();

        assert!(!state.is_alive());
        assert_eq!(state.pending_len(), 0);
        assert!(rx
            .await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("synthetic read failure"));
        assert!(!unix_process_exists(generation.parent_pid));
        assert!(!unix_process_exists(generation.child_pid));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn ignored_sigterm_escalates_to_sigkill_for_whole_group() {
        let generation = spawn_test_generation_ignoring_sigterm().await.unwrap();
        let started = Instant::now();

        generation.process.terminate().await.unwrap();

        assert!(started.elapsed() >= TERMINATE_GRACE_PERIOD);
        assert!(!unix_process_exists(generation.parent_pid));
        assert!(!unix_process_exists(generation.child_pid));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancel_kills_parent_and_child_before_next_generation() {
        let slot = RecoverableSlot::<TestGeneration>::new();
        let first = Arc::new(spawn_test_generation().await.unwrap());
        let first_parent = first.parent_pid;
        let first_child = first.child_pid;
        assert_eq!(unix_process_group(first_parent), first_parent);
        assert_eq!(unix_process_group(first_child), first_parent);
        first.alive.store(false, Ordering::Release);
        *slot.current.write().await = Some(first.clone());

        let spawn_count = Arc::new(AtomicUsize::new(0));
        let next = slot
            .get_or_try_init(
                |generation| generation.alive.load(Ordering::Acquire),
                |stale| async move { stale.process.terminate().await },
                {
                    let spawn_count = spawn_count.clone();
                    move || async move {
                        assert!(!unix_process_exists(first_parent));
                        assert!(!unix_process_exists(first_child));
                        spawn_count.fetch_add(1, AtomicOrdering::SeqCst);
                        spawn_test_generation().await
                    }
                },
            )
            .await
            .unwrap();

        assert_eq!(spawn_count.load(AtomicOrdering::SeqCst), 1);
        assert!(!unix_process_exists(first_parent));
        assert!(!unix_process_exists(first_child));
        assert!(unix_process_exists(next.parent_pid));
        assert!(unix_process_exists(next.child_pid));

        next.process.terminate().await.unwrap();
        assert!(!unix_process_exists(next.parent_pid));
        assert!(!unix_process_exists(next.child_pid));
    }
}
