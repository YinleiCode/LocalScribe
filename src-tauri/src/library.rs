//! Library persistence — saves transcribe/correct/polish outputs to
//! `<project_root>/transcripts/<stem>/` and lists them back to the frontend.
//!
//! Layout per task:
//!   transcripts/雅各书一章/
//!   ├── 雅各书一章.txt              (raw segments with timestamps)
//!   ├── 雅各书一章.srt
//!   ├── 雅各书一章.json             (full TranscribeResult)
//!   ├── 雅各书一章_corrected.txt
//!   ├── 雅各书一章_corrected.srt
//!   ├── 雅各书一章_corrected.json   (with original_text + diff metadata)
//!   ├── 雅各书一章_diff.txt
//!   ├── 雅各书一章_完整版.txt       (polished article)
//!   ├── asr_human_review.json       (human ASR review sidecar)
//!   └── task.json                    (cross-stage metadata)

use anyhow::{Context, Result};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

const LIBRARY_DIR_NAME: &str = "transcripts";
const ASR_REVIEW_STATUSES: [&str; 6] = [
    "pending",
    "confirmed_present",
    "confirmed_missing",
    "substitution",
    "noise",
    "resolved",
];

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
static STEM_LOCKS: Lazy<Mutex<HashMap<String, Weak<Mutex<()>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static LIBRARY_NAMESPACE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[cfg(test)]
static TEST_LIBRARY_ROOT: Lazy<Mutex<Option<PathBuf>>> = Lazy::new(|| Mutex::new(None));

/// Resolve a development-time project root (LocalScribe source tree).
///
/// Returns `Some` when we can find a tree containing both `package.json` and
/// `scribe-py/`. Returns `None` when running from a bundled `.app` — callers
/// must use `user_data_root()` for writable data and `crate::bundle_resources_dir()`
/// for embedded resources.
pub fn dev_project_root() -> Option<PathBuf> {
    fn looks_ok(p: &Path) -> bool {
        p.join("package.json").exists() && p.join("scribe-py").exists()
    }

    if let Ok(env) = std::env::var("LOCALSCRIBE_DEV_ROOT") {
        let r = PathBuf::from(env);
        if looks_ok(&r) {
            return Some(r);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut cur: Option<&Path> = exe.parent();
        while let Some(p) = cur {
            if looks_ok(p) {
                return Some(p.to_path_buf());
            }
            cur = p.parent();
        }
    }
    if let Some(parent) = std::env::current_dir()
        .ok()
        .and_then(|c| c.parent().map(|p| p.to_path_buf()))
    {
        if looks_ok(&parent) {
            return Some(parent);
        }
    }
    None
}

/// Writable user-data root.
///
/// - **Bundled `.app`** → `~/Library/Application Support/LocalScribe/`
///   (per macOS conventions; survives app upgrades/reinstalls)
/// - **Dev**            → source tree root (so editing the code keeps your
///   articles + transcripts visible inside `LocalScribe/`)
/// - **Fallback**       → cwd (last resort)
pub fn user_data_root() -> PathBuf {
    if crate::bundle_resources_dir().is_some() {
        if let Some(home) = dirs::home_dir() {
            let p = home.join("Library/Application Support/LocalScribe");
            let _ = std::fs::create_dir_all(&p);
            return p;
        }
    }
    if let Some(dev) = dev_project_root() {
        return dev;
    }
    if let Some(home) = dirs::home_dir() {
        let p = home.join("Library/Application Support/LocalScribe");
        let _ = std::fs::create_dir_all(&p);
        return p;
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Backwards-compat alias retained for callers that need the LocalScribe folder.
/// Equivalent to `user_data_root()`.
pub fn project_root() -> PathBuf {
    user_data_root()
}

pub fn library_root() -> PathBuf {
    #[cfg(test)]
    if let Some(root) = TEST_LIBRARY_ROOT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
    {
        return root;
    }
    user_data_root().join(LIBRARY_DIR_NAME)
}

pub fn sanitize_stem(stem: &str) -> String {
    let cleaned = stem
        .trim()
        .chars()
        .map(|ch| {
            if ch == '/' || ch == '\\' || ch.is_control() {
                '_'
            } else {
                ch
            }
        })
        .collect::<String>();
    let cleaned = cleaned.trim_matches(['.', ' ']).trim().to_string();
    if cleaned.is_empty() {
        "meeting".to_string()
    } else {
        cleaned
    }
}

fn task_dir(stem: &str) -> PathBuf {
    library_root().join(sanitize_stem(stem))
}

fn ensure_real_library_root() -> Result<PathBuf> {
    let root = library_root();
    match std::fs::symlink_metadata(&root) {
        Ok(metadata) if !metadata.file_type().is_dir() => {
            anyhow::bail!("library root is not a real directory: {}", root.display())
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => ensure_dir(&root)?,
        Err(error) => {
            return Err(error).with_context(|| format!("inspect library root {}", root.display()))
        }
    }
    let metadata = std::fs::symlink_metadata(&root)
        .with_context(|| format!("inspect library root {}", root.display()))?;
    if !metadata.file_type().is_dir() {
        anyhow::bail!("library root is not a real directory: {}", root.display());
    }
    root.canonicalize()
        .with_context(|| format!("canonicalize library root {}", root.display()))
}

fn validate_real_task_directory(dir: &Path) -> Result<PathBuf> {
    let root_real = ensure_real_library_root()?;
    let dir_metadata = std::fs::symlink_metadata(dir)
        .with_context(|| format!("inspect task directory {}", dir.display()))?;
    if !dir_metadata.file_type().is_dir() {
        anyhow::bail!("task path is not a real directory: {}", dir.display());
    }
    let dir_real = dir
        .canonicalize()
        .with_context(|| format!("canonicalize task directory {}", dir.display()))?;
    if !dir_real.starts_with(&root_real) {
        anyhow::bail!("task directory escapes library root: {}", dir.display());
    }
    Ok(dir_real)
}

fn validate_existing_task_layout(dir: &Path) -> Result<PathBuf> {
    let dir_real = validate_real_task_directory(dir)?;
    let audio = dir.join("audio");
    match std::fs::symlink_metadata(&audio) {
        Ok(metadata) if !metadata.file_type().is_dir() => {
            anyhow::bail!("audio path is not a real directory: {}", audio.display())
        }
        Ok(_) => {
            let audio_real = audio
                .canonicalize()
                .with_context(|| format!("canonicalize audio directory {}", audio.display()))?;
            if !audio_real.starts_with(&dir_real) {
                anyhow::bail!(
                    "audio directory escapes task directory: {}",
                    audio.display()
                );
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("inspect audio directory {}", audio.display()))
        }
    }
    Ok(dir_real)
}

fn ensure_dir(p: &Path) -> Result<()> {
    if p.is_dir() {
        return Ok(());
    }
    let mut missing = Vec::new();
    let mut cursor = Some(p);
    while let Some(path) = cursor {
        if path.exists() {
            break;
        }
        missing.push(path.to_path_buf());
        cursor = path.parent();
    }
    std::fs::create_dir_all(p).with_context(|| format!("create {}", p.display()))?;
    #[cfg(unix)]
    for created in missing.iter().rev() {
        if let Some(parent) = created.parent() {
            File::open(parent)?.sync_all()?;
        }
    }
    Ok(())
}

fn stem_lock(stem: &str) -> Arc<Mutex<()>> {
    let key = sanitize_stem(stem).to_lowercase();
    let mut locks = STEM_LOCKS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    locks.retain(|_, lock| lock.strong_count() > 0);
    if let Some(lock) = locks.get(&key).and_then(Weak::upgrade) {
        return lock;
    }
    let lock = Arc::new(Mutex::new(()));
    locks.insert(key, Arc::downgrade(&lock));
    lock
}

fn create_unique_temp_file(path: &Path) -> Result<(PathBuf, File)> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("path has no parent: {}", path.display()))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("path has no file name: {}", path.display()))?;

    for _ in 0..100 {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut temp_name = OsString::from(".");
        temp_name.push(file_name);
        temp_name.push(format!(".{}.{}.{}.tmp", std::process::id(), nonce, counter));
        let temp_path = parent.join(temp_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => return Ok((temp_path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("create temporary file for {}", path.display()))
            }
        }
    }

    anyhow::bail!(
        "could not create unique temporary file for {}",
        path.display()
    )
}

#[cfg(not(windows))]
fn replace_file(temp_path: &Path, path: &Path) -> io::Result<()> {
    std::fs::rename(temp_path, path)?;
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

#[cfg(windows)]
fn replace_file(temp_path: &Path, path: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;

    #[link(name = "Kernel32")]
    extern "system" {
        fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
    }

    let existing: Vec<u16> = temp_path.as_os_str().encode_wide().chain(Some(0)).collect();
    let replacement: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let result = unsafe {
        MoveFileExW(
            existing.as_ptr(),
            replacement.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn atomic_write_with<F>(path: &Path, writer: F) -> Result<()>
where
    F: FnOnce(&mut File) -> io::Result<()>,
{
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("path has no parent: {}", path.display()))?;
    ensure_dir(parent)?;
    let (temp_path, mut temp_file) = create_unique_temp_file(path)?;

    if let Err(error) = writer(&mut temp_file).and_then(|_| temp_file.sync_all()) {
        drop(temp_file);
        let _ = std::fs::remove_file(&temp_path);
        return Err(error).with_context(|| format!("write temporary file for {}", path.display()));
    }
    drop(temp_file);

    if let Err(error) = replace_file(&temp_path, path) {
        let _ = std::fs::remove_file(&temp_path);
        return Err(error).with_context(|| format!("replace {} atomically", path.display()));
    }
    Ok(())
}

fn write_file(path: &Path, contents: &str) -> Result<()> {
    atomic_write_with(path, |file| file.write_all(contents.as_bytes()))
}

struct StagedTaskWrite {
    target: PathBuf,
    staged: PathBuf,
}

impl Drop for StagedTaskWrite {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.staged);
    }
}

#[derive(Default)]
struct TaskFileTransaction {
    writes: Vec<StagedTaskWrite>,
}

struct CommittedTaskWrite {
    target: PathBuf,
    backup: Option<PathBuf>,
}

fn backup_existing_file(path: &Path) -> Result<Option<PathBuf>> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("inspect existing {}", path.display()))
        }
    };
    if !metadata.file_type().is_file() {
        anyhow::bail!(
            "transaction target is not a regular file: {}",
            path.display()
        );
    }

    let mut source =
        File::open(path).with_context(|| format!("open existing {}", path.display()))?;
    let (backup_path, mut backup_file) = create_unique_temp_file(path)?;
    if let Err(error) = io::copy(&mut source, &mut backup_file).and_then(|_| backup_file.sync_all())
    {
        drop(backup_file);
        let _ = std::fs::remove_file(&backup_path);
        return Err(error).with_context(|| format!("back up existing {}", path.display()));
    }
    drop(backup_file);
    Ok(Some(backup_path))
}

fn remove_file_and_sync(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error).with_context(|| format!("remove {}", path.display())),
    }
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

fn rollback_committed_writes(committed: &mut Vec<CommittedTaskWrite>) -> Vec<String> {
    let mut errors = Vec::new();
    for write in committed.drain(..).rev() {
        let result = match write.backup {
            Some(backup) => replace_file(&backup, &write.target)
                .with_context(|| format!("restore {}", write.target.display())),
            None => remove_file_and_sync(&write.target),
        };
        if let Err(error) = result {
            errors.push(format!("{}: {error:#}", write.target.display()));
        }
    }
    errors
}

impl TaskFileTransaction {
    fn stage_with<F>(&mut self, target: &Path, writer: F) -> Result<()>
    where
        F: FnOnce(&mut File) -> io::Result<()>,
    {
        if self.writes.iter().any(|write| write.target == target) {
            anyhow::bail!("duplicate transaction target: {}", target.display());
        }
        let parent = target
            .parent()
            .ok_or_else(|| anyhow::anyhow!("path has no parent: {}", target.display()))?;
        ensure_dir(parent)?;
        let (staged, mut file) = create_unique_temp_file(target)?;
        if let Err(error) = writer(&mut file).and_then(|_| file.sync_all()) {
            drop(file);
            let _ = std::fs::remove_file(&staged);
            return Err(error)
                .with_context(|| format!("stage transaction file for {}", target.display()));
        }
        drop(file);
        self.writes.push(StagedTaskWrite {
            target: target.to_path_buf(),
            staged,
        });
        Ok(())
    }

    fn stage_bytes(&mut self, target: &Path, contents: &[u8]) -> Result<()> {
        self.stage_with(target, |file| file.write_all(contents))
    }

    fn commit(self) -> Result<()> {
        let mut committed = Vec::with_capacity(self.writes.len());
        for write in &self.writes {
            let backup = match backup_existing_file(&write.target) {
                Ok(backup) => backup,
                Err(error) => {
                    let rollback_errors = rollback_committed_writes(&mut committed);
                    if rollback_errors.is_empty() {
                        return Err(error);
                    }
                    anyhow::bail!(
                        "transaction failed: {error:#}; rollback failed: {}",
                        rollback_errors.join("; ")
                    );
                }
            };

            if let Err(error) = replace_file(&write.staged, &write.target)
                .with_context(|| format!("commit transaction file {}", write.target.display()))
            {
                committed.push(CommittedTaskWrite {
                    target: write.target.clone(),
                    backup,
                });
                let rollback_errors = rollback_committed_writes(&mut committed);
                if rollback_errors.is_empty() {
                    return Err(error);
                }
                anyhow::bail!(
                    "transaction failed: {error:#}; rollback failed: {}",
                    rollback_errors.join("; ")
                );
            }
            committed.push(CommittedTaskWrite {
                target: write.target.clone(),
                backup,
            });
        }

        for write in committed {
            if let Some(backup) = write.backup {
                if let Err(error) = remove_file_and_sync(&backup) {
                    tracing::warn!(
                        "failed to clean transaction backup {}: {:#}",
                        backup.display(),
                        error
                    );
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Save APIs (called from Tauri commands)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct SaveRawArgs {
    pub stem: String,
    pub audio_filename: String,
    #[serde(default)]
    pub source_audio: Option<String>,
    pub txt: String,
    pub srt: String,
    pub json: String,
    /// Whole TranscribeResult so we can render task summaries on history list.
    pub result: Value,
}

#[derive(Debug, Deserialize)]
pub struct SaveAsrReviewArgs {
    pub stem: String,
    pub review: Value,
}

fn validate_asr_review(review: &Value) -> Result<()> {
    let object = review
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("ASR human review must be a JSON object"))?;
    if object.get("schema_version").and_then(Value::as_u64) != Some(1) {
        anyhow::bail!("ASR human review schema_version must be 1");
    }
    let items = object
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("ASR human review items must be an array"))?;
    let mut ids = HashSet::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let item = item.as_object().ok_or_else(|| {
            anyhow::anyhow!("ASR human review items[{index}] must be a JSON object")
        })?;
        let id = item.get("id").and_then(Value::as_str).ok_or_else(|| {
            anyhow::anyhow!("ASR human review items[{index}].id must be a string")
        })?;
        if id.trim().is_empty() {
            anyhow::bail!("ASR human review items[{index}].id must not be empty");
        }
        if !ids.insert(id) {
            anyhow::bail!("ASR human review item id must be unique: {id}");
        }

        let start = item.get("start").and_then(Value::as_f64).ok_or_else(|| {
            anyhow::anyhow!("ASR human review items[{index}].start must be a finite number")
        })?;
        let end = item.get("end").and_then(Value::as_f64).ok_or_else(|| {
            anyhow::anyhow!("ASR human review items[{index}].end must be a finite number")
        })?;
        if !start.is_finite() {
            anyhow::bail!("ASR human review items[{index}].start must be a finite number");
        }
        if !end.is_finite() {
            anyhow::bail!("ASR human review items[{index}].end must be a finite number");
        }
        if start < 0.0 || start >= end {
            anyhow::bail!("ASR human review items[{index}] must satisfy 0 <= start < end");
        }

        let status = item.get("status").and_then(Value::as_str).ok_or_else(|| {
            anyhow::anyhow!("ASR human review items[{index}].status must be a string")
        })?;
        if !ASR_REVIEW_STATUSES.contains(&status) {
            anyhow::bail!("ASR human review items[{index}].status is invalid: {status}");
        }

        for field in ["heard_text", "note", "replacement_text"] {
            if item.get(field).is_some_and(|value| !value.is_string()) {
                anyhow::bail!(
                    "ASR human review items[{index}].{field} must be a string when present"
                );
            }
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct SaveCorrectedArgs {
    pub stem: String,
    pub txt: String,
    pub srt: String,
    pub json: String,
    pub diff: String,
    pub model: String,
    pub changed: u64,
    pub total: u64,
    #[serde(default)]
    pub glossary: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct SaveRawAndCorrectedArgs {
    pub raw: SaveRawArgs,
    #[serde(default)]
    pub corrected: Option<SaveCorrectedArgs>,
    #[serde(default)]
    pub clear_corrected: bool,
}

#[derive(Debug, Deserialize)]
pub struct SavePolishedArgs {
    pub stem: String,
    pub text: String,
    pub model: String,
    /// "corrected" 或 "raw" — 表明排版输入用的是校对稿还是原始转录
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub struct SavedMeta {
    pub stem: String,
    pub audio_filename: String,
    #[serde(default)]
    pub raw_filename: String,
    #[serde(default)]
    pub audio_path: Option<String>,
    pub duration: f64,
    pub segments: u64,
    pub backend: String,
    pub model_id: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub has_corrected: bool,
    pub has_polished: bool,
    pub correction_model: Option<String>,
    pub correction_changed: Option<u64>,
    pub correction_glossary: Option<Value>,
    pub polish_model: Option<String>,
    pub polish_source: Option<String>,
    #[serde(default, flatten)]
    pub extras: Map<String, Value>,
}

#[derive(Debug, Serialize)]
pub struct PreparedMedia {
    pub path: String,
    pub optimized: bool,
}

#[derive(Debug, Default, Eq, PartialEq)]
struct Mp4Layout {
    mdat_offset: Option<u64>,
    moov_offset: Option<u64>,
    moov_size: Option<u64>,
    moof_offset: Option<u64>,
}

const ASSET_PROTOCOL_MAX_RANGE_BYTES: u64 = 1000 * 1024;
const MP4_FASTSTART_HEADER_ALLOWANCE_BYTES: u64 = 64 * 1024;
const PLAYBACK_FRAGMENT_DURATION_US: &str = "5000000";
const PLAYBACK_CACHE_MARKER: &str = ".localscribe-playback";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mp4PlaybackPreparation {
    FastStart,
    Fragmented,
}

fn read_mp4_layout(path: &Path) -> Result<Mp4Layout> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let len = file
        .metadata()
        .with_context(|| format!("inspect {}", path.display()))?
        .len();
    let mut offset = 0_u64;
    let mut layout = Mp4Layout::default();

    while offset.checked_add(8).is_some_and(|end| end <= len) {
        file.seek(SeekFrom::Start(offset))
            .with_context(|| format!("seek {}", path.display()))?;
        let mut header = [0_u8; 8];
        file.read_exact(&mut header)
            .with_context(|| format!("read MP4 atom header from {}", path.display()))?;

        let size32 = u32::from_be_bytes(header[..4].try_into().expect("four-byte atom size"));
        let kind: [u8; 4] = header[4..8].try_into().expect("four-byte atom kind");
        let (atom_size, header_size) = if size32 == 1 {
            let mut extended = [0_u8; 8];
            file.read_exact(&mut extended)
                .with_context(|| format!("read extended MP4 atom size from {}", path.display()))?;
            (u64::from_be_bytes(extended), 16_u64)
        } else if size32 == 0 {
            (len - offset, 8_u64)
        } else {
            (u64::from(size32), 8_u64)
        };

        if atom_size < header_size {
            anyhow::bail!(
                "invalid MP4 atom size at byte {offset} in {}",
                path.display()
            );
        }
        let next = offset
            .checked_add(atom_size)
            .filter(|next| *next <= len)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "MP4 atom at byte {offset} exceeds file length in {}",
                    path.display()
                )
            })?;

        match &kind {
            b"mdat" if layout.mdat_offset.is_none() => layout.mdat_offset = Some(offset),
            b"moov" if layout.moov_offset.is_none() => {
                layout.moov_offset = Some(offset);
                layout.moov_size = Some(atom_size);
            }
            b"moof" if layout.moof_offset.is_none() => layout.moof_offset = Some(offset),
            _ => {}
        }
        if atom_size == 0 || next == offset {
            break;
        }
        offset = next;
    }

    Ok(layout)
}

fn mp4_playback_preparation(layout: &Mp4Layout) -> Option<Mp4PlaybackPreparation> {
    let (mdat, moov, moov_size) = (layout.mdat_offset?, layout.moov_offset?, layout.moov_size?);
    if moov > mdat {
        let faststart_index_fits = moov_size
            .checked_add(MP4_FASTSTART_HEADER_ALLOWANCE_BYTES)
            .is_some_and(|end| end <= ASSET_PROTOCOL_MAX_RANGE_BYTES);
        if faststart_index_fits {
            Some(Mp4PlaybackPreparation::FastStart)
        } else {
            Some(Mp4PlaybackPreparation::Fragmented)
        }
    } else if !moov
        .checked_add(moov_size)
        .is_some_and(|end| end <= ASSET_PROTOCOL_MAX_RANGE_BYTES)
    {
        Some(Mp4PlaybackPreparation::Fragmented)
    } else {
        None
    }
}

fn ffmpeg_executable() -> PathBuf {
    if let Some(resources) = crate::bundle_resources_dir() {
        let bundled = resources.join("bin/ffmpeg");
        if bundled.is_file() {
            return bundled;
        }
    }
    PathBuf::from("ffmpeg")
}

fn playback_cache_path(source: &Path) -> Result<PathBuf> {
    let parent = source
        .parent()
        .ok_or_else(|| anyhow::anyhow!("media path has no parent: {}", source.display()))?;
    let file_name = source
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("media path has no file name: {}", source.display()))?;
    // Tauri's asset protocol rejects dotfiles by default on Unix, so the
    // WebKit-facing cache must not use a hidden filename.
    let mut cache_name = OsString::from(file_name);
    cache_name.push(PLAYBACK_CACHE_MARKER);
    if let Some(extension) = source.extension() {
        cache_name.push(".");
        cache_name.push(extension);
    }
    Ok(parent.join(cache_name))
}

fn playback_file_is_valid(path: &Path, preparation: Mp4PlaybackPreparation) -> bool {
    if !nonempty_regular_file(path) {
        return false;
    }
    let Ok(layout) = read_mp4_layout(path) else {
        return false;
    };
    let (Some(moov), Some(moov_size), Some(mdat)) =
        (layout.moov_offset, layout.moov_size, layout.mdat_offset)
    else {
        return false;
    };
    let Some(metadata_end) = moov.checked_add(moov_size) else {
        return false;
    };
    moov < mdat
        && metadata_end <= ASSET_PROTOCOL_MAX_RANGE_BYTES
        && (preparation != Mp4PlaybackPreparation::Fragmented || layout.moof_offset.is_some())
}

fn playback_cache_is_current(
    source: &Path,
    cache: &Path,
    preparation: Mp4PlaybackPreparation,
) -> bool {
    let (Ok(source_meta), Ok(cache_meta)) = (std::fs::metadata(source), std::fs::metadata(cache))
    else {
        return false;
    };
    let (Ok(source_modified), Ok(cache_modified)) = (source_meta.modified(), cache_meta.modified())
    else {
        return false;
    };
    cache_modified >= source_modified && playback_file_is_valid(cache, preparation)
}

fn internal_task_audio(path: &Path) -> Result<Option<(PathBuf, String)>> {
    if !nonempty_regular_file(path) {
        anyhow::bail!("media is not a nonempty regular file: {}", path.display());
    }
    let canonical = path
        .canonicalize()
        .with_context(|| format!("canonicalize media {}", path.display()))?;
    let root = ensure_real_library_root()?;
    let Ok(relative) = canonical.strip_prefix(&root) else {
        return Ok(None);
    };
    let components = relative.components().collect::<Vec<_>>();
    if components.len() != 3
        || components[1].as_os_str() != "audio"
        || !matches!(components[0], Component::Normal(_))
        || !matches!(components[2], Component::Normal(_))
    {
        return Ok(None);
    }
    let stem = components[0].as_os_str().to_string_lossy().into_owned();
    Ok(Some((canonical, stem)))
}

/// Prepare a separate WebKit playback cache next to LocalScribe's stable audio.
/// AAC samples are stream-copied; the ASR/diarization source is never rewritten.
pub fn prepare_media_for_playback(audio_path: &str) -> Result<PreparedMedia> {
    let requested = PathBuf::from(audio_path.trim());
    if audio_path.trim().is_empty() {
        anyhow::bail!("media path is empty");
    }

    let Some((initial_path, stem)) = internal_task_audio(&requested)? else {
        return Ok(PreparedMedia {
            path: requested.to_string_lossy().into_owned(),
            optimized: false,
        });
    };
    let extension = initial_path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if extension != "m4a" && extension != "mp4" {
        return Ok(PreparedMedia {
            path: initial_path.to_string_lossy().into_owned(),
            optimized: false,
        });
    }

    let lock = stem_lock(&stem);
    let _guard = lock
        .lock()
        .map_err(|_| anyhow::anyhow!("library lock poisoned for {stem}"))?;
    let Some((path, _)) = internal_task_audio(&requested)? else {
        anyhow::bail!("internal media path changed while preparing playback");
    };
    let layout = read_mp4_layout(&path)?;
    let Some(preparation) = mp4_playback_preparation(&layout) else {
        return Ok(PreparedMedia {
            path: path.to_string_lossy().into_owned(),
            optimized: false,
        });
    };

    let cache_path = playback_cache_path(&path)?;
    if playback_cache_is_current(&path, &cache_path, preparation) {
        return Ok(PreparedMedia {
            path: cache_path.to_string_lossy().into_owned(),
            optimized: true,
        });
    }

    let (temp_path, temp_file) = create_unique_temp_file(&cache_path)?;
    drop(temp_file);
    let mut command = std::process::Command::new(ffmpeg_executable());
    command
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&path)
        .args([
            "-map",
            "0",
            "-map_metadata",
            "0",
            "-map_chapters",
            "0",
            "-c",
            "copy",
        ]);
    match preparation {
        Mp4PlaybackPreparation::FastStart => {
            command.args(["-movflags", "+faststart"]);
        }
        Mp4PlaybackPreparation::Fragmented => {
            command.args([
                "-movflags",
                "+frag_keyframe+empty_moov+default_base_moof",
                "-frag_duration",
                PLAYBACK_FRAGMENT_DURATION_US,
            ]);
        }
    }
    let output = command.args(["-f", "mp4"]).arg(&temp_path).output();

    let output = match output {
        Ok(output) if output.status.success() => output,
        Ok(output) => {
            let _ = std::fs::remove_file(&temp_path);
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            anyhow::bail!("ffmpeg could not prepare media playback: {stderr}");
        }
        Err(error) => {
            let _ = std::fs::remove_file(&temp_path);
            return Err(error).context("start ffmpeg for media playback");
        }
    };
    drop(output);

    if !nonempty_regular_file(&temp_path) {
        let _ = std::fs::remove_file(&temp_path);
        anyhow::bail!("ffmpeg produced an empty playback file");
    }
    let verify_result = (|| -> Result<()> {
        if !playback_file_is_valid(&temp_path, preparation) {
            anyhow::bail!("ffmpeg output is not a valid WebKit playback file");
        }
        let original_permissions = std::fs::metadata(&path)
            .with_context(|| format!("inspect media permissions {}", path.display()))?
            .permissions();
        std::fs::set_permissions(&temp_path, original_permissions)
            .with_context(|| format!("preserve media permissions {}", temp_path.display()))?;
        File::open(&temp_path)
            .and_then(|file| file.sync_all())
            .with_context(|| format!("sync prepared media {}", temp_path.display()))?;
        Ok(())
    })();
    if let Err(error) = verify_result {
        let _ = std::fs::remove_file(&temp_path);
        return Err(error);
    }
    if let Err(error) = replace_file(&temp_path, &cache_path) {
        let _ = std::fs::remove_file(&temp_path);
        return Err(error).with_context(|| {
            format!("replace playback cache {} atomically", cache_path.display())
        });
    }

    Ok(PreparedMedia {
        path: cache_path.to_string_lossy().into_owned(),
        optimized: true,
    })
}

fn now_ts() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn meta_path(stem: &str) -> PathBuf {
    task_dir(stem).join("task.json")
}

fn load_existing_meta(stem: &str) -> Result<SavedMeta> {
    let stem = sanitize_stem(stem);
    let dir = task_dir(&stem);
    match std::fs::symlink_metadata(&dir) {
        Ok(_) => {
            validate_existing_task_layout(&dir)?;
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| format!("inspect {}", dir.display())),
    }
    let path = meta_path(&stem);
    let contents = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            anyhow::bail!("task metadata missing: {}", path.display())
        }
        Err(error) => {
            return Err(error).with_context(|| format!("read metadata {}", path.display()))
        }
    };
    let value: Value = serde_json::from_str(&contents)
        .with_context(|| format!("parse metadata {}", path.display()))?;
    if !value.is_object() {
        anyhow::bail!("task metadata must be a JSON object: {}", path.display());
    }
    let meta: SavedMeta = serde_json::from_value(value)
        .with_context(|| format!("parse metadata {}", path.display()))?;
    if meta.stem != stem {
        anyhow::bail!(
            "task metadata stem mismatch for {}: expected {stem:?}, found {:?}",
            path.display(),
            meta.stem
        );
    }
    Ok(meta)
}

#[cfg(test)]
fn save_meta(meta: &SavedMeta) -> Result<()> {
    let p = meta_path(&meta.stem);
    write_file(&p, &serde_json::to_string_pretty(meta)?)
}

fn nonempty_regular_file(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .map(|metadata| metadata.file_type().is_file() && metadata.len() > 0)
        .unwrap_or(false)
}

fn task_audio_relative_path(stem: &str, audio_path: &str) -> Option<PathBuf> {
    if audio_path.trim().is_empty() {
        return None;
    }
    let audio_dir = task_dir(stem).join("audio");
    if !std::fs::symlink_metadata(&audio_dir)
        .map(|metadata| metadata.file_type().is_dir())
        .unwrap_or(false)
    {
        return None;
    }
    let path = PathBuf::from(audio_path.trim());
    if !nonempty_regular_file(&path) {
        return None;
    }
    let audio_dir = audio_dir.canonicalize().ok()?;
    let path = path.canonicalize().ok()?;
    path.strip_prefix(audio_dir).ok().map(Path::to_path_buf)
}

fn reusable_task_audio(stem: &str, audio_path: &str) -> Option<PathBuf> {
    task_audio_relative_path(stem, audio_path)
        .map(|relative| task_dir(stem).join("audio").join(relative))
}

fn existing_stable_audio(stem: &str) -> Result<Option<String>> {
    let audio_dir = task_dir(stem).join("audio");
    if !std::fs::symlink_metadata(&audio_dir)
        .map(|metadata| metadata.file_type().is_dir())
        .unwrap_or(false)
    {
        return Ok(None);
    }
    let candidates: Vec<PathBuf> = std::fs::read_dir(&audio_dir)
        .with_context(|| format!("read {}", audio_dir.display()))?
        .filter_map(|entry| entry.ok().map(|item| item.path()))
        .filter(|path| {
            nonempty_regular_file(path)
                && !path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.starts_with('.') || name.ends_with(".tmp"))
                    .unwrap_or(true)
        })
        .collect();
    match candidates.as_slice() {
        [] => Ok(None),
        [path] => Ok(Some(path.to_string_lossy().into_owned())),
        _ => anyhow::bail!(
            "stable audio is ambiguous for {stem}: {} candidates",
            candidates.len()
        ),
    }
}

fn stage_source_audio(
    transaction: &mut TaskFileTransaction,
    stem: &str,
    source_audio: &Path,
    audio_filename: &str,
) -> Result<String> {
    if !nonempty_regular_file(source_audio) {
        anyhow::bail!(
            "source audio is not a nonempty regular file: {}",
            source_audio.display()
        );
    }
    let ext = source_audio
        .extension()
        .and_then(|s| s.to_str())
        .or_else(|| {
            Path::new(audio_filename)
                .extension()
                .and_then(|s| s.to_str())
        })
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("audio");
    let task_root = task_dir(stem);
    let dir = task_root.join("audio");
    ensure_dir(&dir)?;
    let metadata = std::fs::symlink_metadata(&dir)
        .with_context(|| format!("inspect audio directory {}", dir.display()))?;
    if !metadata.file_type().is_dir() {
        anyhow::bail!("audio path is not a real directory: {}", dir.display());
    }
    let task_root_real = validate_real_task_directory(&task_root)?;
    let dir_real = dir
        .canonicalize()
        .with_context(|| format!("canonicalize audio directory {}", dir.display()))?;
    if !dir_real.starts_with(&task_root_real) {
        anyhow::bail!("audio directory escapes task directory: {}", dir.display());
    }
    let dest = dir.join(format!("{stem}.{ext}"));
    if let (Ok(src_real), Ok(dest_real)) = (source_audio.canonicalize(), dest.canonicalize()) {
        if src_real == dest_real {
            return Ok(dest.to_string_lossy().into_owned());
        }
    }

    let mut source_file = File::open(source_audio)
        .with_context(|| format!("open source audio {}", source_audio.display()))?;
    let source_len = source_file
        .metadata()
        .with_context(|| format!("inspect source audio {}", source_audio.display()))?
        .len();
    if source_len == 0 {
        anyhow::bail!("source audio is empty: {}", source_audio.display());
    }

    transaction.stage_with(&dest, |temp_file| {
        let copied = io::copy(&mut source_file, temp_file)?;
        if copied != source_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "source audio changed while copying: expected {source_len} bytes, copied {copied}"
                ),
            ));
        }
        Ok(())
    })
    .with_context(|| {
        format!(
            "stage source audio {} -> {}",
            source_audio.display(),
            dest.display()
        )
    })?;
    Ok(dest.to_string_lossy().into_owned())
}

fn usable_audio_source(source_audio: Option<&str>) -> Option<PathBuf> {
    source_audio
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .filter(|path| nonempty_regular_file(path))
}

fn recoverable_uninitialized_task_dir(dir: &Path) -> Result<bool> {
    let entries = std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("read entry in {}", dir.display()))?;
        if entry.file_name() != "audio" {
            return Ok(false);
        }
        let file_type = entry
            .file_type()
            .with_context(|| format!("inspect {}", entry.path().display()))?;
        if !file_type.is_dir() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn load_or_initialize_raw_meta(stem: &str) -> Result<SavedMeta> {
    ensure_real_library_root()?;
    let dir = task_dir(stem);
    match std::fs::symlink_metadata(&dir) {
        Ok(metadata) if !metadata.file_type().is_dir() => {
            anyhow::bail!("task path is not a directory: {}", dir.display())
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            ensure_dir(&dir)?;
            validate_existing_task_layout(&dir)?;
            return Ok(SavedMeta {
                stem: stem.to_string(),
                ..SavedMeta::default()
            });
        }
        Err(error) => return Err(error).with_context(|| format!("inspect {}", dir.display())),
    }
    validate_existing_task_layout(&dir)?;

    match std::fs::symlink_metadata(meta_path(stem)) {
        Ok(_) => load_existing_meta(stem),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            if !recoverable_uninitialized_task_dir(&dir)? {
                anyhow::bail!(
                    "task metadata missing from nonempty directory: {}",
                    dir.display()
                );
            }
            Ok(SavedMeta {
                stem: stem.to_string(),
                ..SavedMeta::default()
            })
        }
        Err(error) => Err(error).with_context(|| format!("inspect metadata for {stem}")),
    }
}

pub fn save_raw(args: SaveRawArgs) -> Result<SavedMeta> {
    let stem = sanitize_stem(&args.stem);
    let lock = stem_lock(&stem);
    let _guard = lock
        .lock()
        .map_err(|_| anyhow::anyhow!("library lock poisoned for {stem}"))?;
    let dir = task_dir(&stem);
    let mut meta = load_or_initialize_raw_meta(&stem)?;
    let mut transaction = TaskFileTransaction::default();
    let copied_audio = if let Some(source) = usable_audio_source(args.source_audio.as_deref()) {
        Some(stage_source_audio(
            &mut transaction,
            &stem,
            &source,
            &args.audio_filename,
        )?)
    } else if let Some(path) = meta
        .audio_path
        .as_deref()
        .and_then(|path| reusable_task_audio(&stem, path))
    {
        Some(path.to_string_lossy().into_owned())
    } else if let Some(source) = usable_audio_source(meta.audio_path.as_deref()) {
        Some(stage_source_audio(
            &mut transaction,
            &stem,
            &source,
            &args.audio_filename,
        )?)
    } else {
        existing_stable_audio(&stem)?
    };

    let now = now_ts();
    if meta.created_at == 0 {
        meta.created_at = now;
    }
    meta.updated_at = now;
    meta.stem = stem.clone();
    meta.audio_filename = args.audio_filename.clone();
    meta.raw_filename = format!("{stem}.json");
    meta.audio_path = copied_audio;
    if let Some(d) = args.result.get("duration").and_then(|v| v.as_f64()) {
        meta.duration = d;
    }
    if let Some(s) = args.result.get("segments").and_then(|v| v.as_array()) {
        meta.segments = s.len() as u64;
    }
    if let Some(b) = args.result.get("backend").and_then(|v| v.as_str()) {
        meta.backend = b.into();
    }
    if let Some(m) = args.result.get("model_id").and_then(|v| v.as_str()) {
        meta.model_id = m.into();
    }
    let metadata = serde_json::to_vec_pretty(&meta)?;
    transaction.stage_bytes(&dir.join(format!("{}.txt", stem)), args.txt.as_bytes())?;
    transaction.stage_bytes(&dir.join(format!("{}.srt", stem)), args.srt.as_bytes())?;
    transaction.stage_bytes(&dir.join(format!("{}.json", stem)), args.json.as_bytes())?;
    transaction.stage_bytes(&meta_path(&stem), &metadata)?;
    transaction.commit()?;
    Ok(meta)
}

pub fn save_asr_review(args: SaveAsrReviewArgs) -> Result<()> {
    validate_asr_review(&args.review)?;

    let stem = sanitize_stem(&args.stem);
    let lock = stem_lock(&stem);
    let _guard = lock
        .lock()
        .map_err(|_| anyhow::anyhow!("library lock poisoned for {stem}"))?;
    load_existing_meta(&stem)?;
    let path = task_dir(&stem).join("asr_human_review.json");
    write_file(&path, &serde_json::to_string_pretty(&args.review)?)
}

pub fn save_corrected(args: SaveCorrectedArgs) -> Result<SavedMeta> {
    let stem = sanitize_stem(&args.stem);
    let lock = stem_lock(&stem);
    let _guard = lock
        .lock()
        .map_err(|_| anyhow::anyhow!("library lock poisoned for {stem}"))?;
    let dir = task_dir(&stem);
    let mut meta = load_existing_meta(&stem)?;
    meta.has_corrected = true;
    meta.correction_model = Some(args.model);
    meta.correction_changed = Some(args.changed);
    meta.correction_glossary = args.glossary;
    meta.updated_at = now_ts();
    let metadata = serde_json::to_vec_pretty(&meta)?;
    let mut transaction = TaskFileTransaction::default();
    transaction.stage_bytes(
        &dir.join(format!("{}_corrected.txt", stem)),
        args.txt.as_bytes(),
    )?;
    transaction.stage_bytes(
        &dir.join(format!("{}_corrected.srt", stem)),
        args.srt.as_bytes(),
    )?;
    transaction.stage_bytes(
        &dir.join(format!("{}_corrected.json", stem)),
        args.json.as_bytes(),
    )?;
    transaction.stage_bytes(
        &dir.join(format!("{}_diff.txt", stem)),
        args.diff.as_bytes(),
    )?;
    transaction.stage_bytes(&meta_path(&stem), &metadata)?;
    transaction.commit()?;
    Ok(meta)
}

pub fn save_raw_and_corrected(args: SaveRawAndCorrectedArgs) -> Result<SavedMeta> {
    let SaveRawAndCorrectedArgs {
        raw,
        corrected,
        clear_corrected,
    } = args;
    let stem = sanitize_stem(&raw.stem);
    if let Some(corrected_args) = corrected.as_ref() {
        let corrected_stem = sanitize_stem(&corrected_args.stem);
        if corrected_stem != stem {
            anyhow::bail!(
                "raw/corrected stem mismatch: raw={stem:?}, corrected={corrected_stem:?}"
            );
        }
    }

    let lock = stem_lock(&stem);
    let _guard = lock
        .lock()
        .map_err(|_| anyhow::anyhow!("library lock poisoned for {stem}"))?;
    let dir = task_dir(&stem);
    let mut meta = load_or_initialize_raw_meta(&stem)?;
    let mut transaction = TaskFileTransaction::default();
    let copied_audio = if let Some(source) = usable_audio_source(raw.source_audio.as_deref()) {
        Some(stage_source_audio(
            &mut transaction,
            &stem,
            &source,
            &raw.audio_filename,
        )?)
    } else if let Some(path) = meta
        .audio_path
        .as_deref()
        .and_then(|path| reusable_task_audio(&stem, path))
    {
        Some(path.to_string_lossy().into_owned())
    } else if let Some(source) = usable_audio_source(meta.audio_path.as_deref()) {
        Some(stage_source_audio(
            &mut transaction,
            &stem,
            &source,
            &raw.audio_filename,
        )?)
    } else {
        existing_stable_audio(&stem)?
    };

    let now = now_ts();
    if meta.created_at == 0 {
        meta.created_at = now;
    }
    meta.updated_at = now;
    meta.stem = stem.clone();
    meta.audio_filename = raw.audio_filename.clone();
    meta.raw_filename = format!("{stem}.json");
    meta.audio_path = copied_audio;
    if let Some(duration) = raw.result.get("duration").and_then(Value::as_f64) {
        meta.duration = duration;
    }
    if let Some(segments) = raw.result.get("segments").and_then(Value::as_array) {
        meta.segments = segments.len() as u64;
    }
    if let Some(backend) = raw.result.get("backend").and_then(Value::as_str) {
        meta.backend = backend.into();
    }
    if let Some(model_id) = raw.result.get("model_id").and_then(Value::as_str) {
        meta.model_id = model_id.into();
    }
    if let Some(corrected_args) = corrected.as_ref() {
        meta.has_corrected = true;
        meta.correction_model = Some(corrected_args.model.clone());
        meta.correction_changed = Some(corrected_args.changed);
        meta.correction_glossary = corrected_args.glossary.clone();
    } else if clear_corrected {
        meta.has_corrected = false;
        meta.correction_model = None;
        meta.correction_changed = None;
        meta.correction_glossary = None;
    }

    transaction.stage_bytes(&dir.join(format!("{stem}.txt")), raw.txt.as_bytes())?;
    transaction.stage_bytes(&dir.join(format!("{stem}.srt")), raw.srt.as_bytes())?;
    transaction.stage_bytes(&dir.join(format!("{stem}.json")), raw.json.as_bytes())?;
    if let Some(corrected_args) = corrected.as_ref() {
        transaction.stage_bytes(
            &dir.join(format!("{stem}_corrected.txt")),
            corrected_args.txt.as_bytes(),
        )?;
        transaction.stage_bytes(
            &dir.join(format!("{stem}_corrected.srt")),
            corrected_args.srt.as_bytes(),
        )?;
        transaction.stage_bytes(
            &dir.join(format!("{stem}_corrected.json")),
            corrected_args.json.as_bytes(),
        )?;
        transaction.stage_bytes(
            &dir.join(format!("{stem}_diff.txt")),
            corrected_args.diff.as_bytes(),
        )?;
    }
    let metadata = serde_json::to_vec_pretty(&meta)?;
    transaction.stage_bytes(&meta_path(&stem), &metadata)?;
    transaction.commit()?;
    Ok(meta)
}

pub fn save_polished(args: SavePolishedArgs) -> Result<SavedMeta> {
    let stem = sanitize_stem(&args.stem);
    let lock = stem_lock(&stem);
    let _guard = lock
        .lock()
        .map_err(|_| anyhow::anyhow!("library lock poisoned for {stem}"))?;
    let dir = task_dir(&stem);
    let mut meta = load_existing_meta(&stem)?;
    let body = format!(
        "# {} — 完整文字稿\n# 排版 {}\n\n{}\n",
        stem, args.model, args.text
    );
    meta.has_polished = true;
    meta.polish_model = Some(args.model);
    meta.polish_source = args.source;
    meta.updated_at = now_ts();
    let metadata = serde_json::to_vec_pretty(&meta)?;
    let mut transaction = TaskFileTransaction::default();
    transaction.stage_bytes(&dir.join(format!("{}_完整版.txt", stem)), body.as_bytes())?;
    transaction.stage_bytes(&meta_path(&stem), &metadata)?;
    transaction.commit()?;
    Ok(meta)
}

// ============================================================================
// List & load
// ============================================================================

pub fn list_library() -> Result<Vec<SavedMeta>> {
    let root = library_root();
    if !root.exists() {
        return Ok(vec![]);
    }
    let mut out = vec![];
    for entry in std::fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let lock = stem_lock(&name);
        let _guard = match lock.lock() {
            Ok(guard) => guard,
            Err(_) => {
                tracing::warn!("skipping corrupt library item {name}: library lock poisoned");
                continue;
            }
        };
        let meta =
            match load_existing_meta(&name).with_context(|| format!("load library item {name}")) {
                Ok(meta) => meta,
                Err(error) => {
                    tracing::warn!("skipping corrupt library item {name}: {error:#}");
                    continue;
                }
            };
        if !meta.stem.is_empty() && meta.created_at > 0 {
            out.push(meta);
        }
    }
    out.sort_by_key(|m| std::cmp::Reverse(m.updated_at));
    Ok(out)
}

fn regular_file(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .map(|metadata| metadata.file_type().is_file())
        .unwrap_or(false)
}

fn safe_raw_filename(filename: &str) -> bool {
    let path = Path::new(filename);
    let mut components = path.components();
    let Some(Component::Normal(_)) = components.next() else {
        return false;
    };
    if components.next().is_some()
        || path.extension().and_then(|value| value.to_str()) != Some("json")
    {
        return false;
    }
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    file_name != "task.json"
        && file_name != "asr_human_review.json"
        && !file_name.ends_with("_corrected.json")
}

fn configured_raw_path(dir: &Path, stem: &str, raw_filename: &str) -> Result<PathBuf> {
    if !safe_raw_filename(raw_filename) {
        anyhow::bail!("invalid raw_filename for {stem}: {raw_filename:?}");
    }
    let path = dir.join(raw_filename);
    if !regular_file(&path) {
        anyhow::bail!(
            "configured raw result missing for {stem}: {}",
            path.display()
        );
    }
    Ok(path)
}

fn is_legacy_raw_candidate(path: &Path) -> bool {
    if !regular_file(path) {
        return false;
    }
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    if path.extension().and_then(|value| value.to_str()) != Some("json")
        || file_name == "task.json"
        || file_name == "asr_human_review.json"
        || file_name.ends_with("_corrected.json")
    {
        return false;
    }
    std::fs::read_to_string(path)
        .ok()
        .and_then(|contents| serde_json::from_str::<Value>(&contents).ok())
        .map(|value| {
            value.get("segments").and_then(Value::as_array).is_some()
                && value.get("backend").and_then(Value::as_str).is_some()
                && value.get("model_id").and_then(Value::as_str).is_some()
        })
        .unwrap_or(false)
}

fn unique_legacy_raw_path(dir: &Path, stem: &str) -> Result<PathBuf> {
    let candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|item| item.path()))
        .filter(|path| is_legacy_raw_candidate(path))
        .collect();
    match candidates.as_slice() {
        [] => anyhow::bail!("library item not found: {stem}"),
        [path] => Ok(path.clone()),
        _ => anyhow::bail!(
            "legacy raw result is ambiguous for {stem}: {} candidates",
            candidates.len()
        ),
    }
}

fn unique_legacy_stage_path(
    dir: &Path,
    stem: &str,
    suffix: &str,
    description: &str,
) -> Result<PathBuf> {
    let candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|item| item.path()))
        .filter(|path| {
            regular_file(path)
                && path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .map(|name| name.ends_with(suffix))
                    .unwrap_or(false)
        })
        .collect();
    match candidates.as_slice() {
        [] => anyhow::bail!("committed {description} result missing: {stem}"),
        [path] => Ok(path.clone()),
        _ => anyhow::bail!(
            "committed {description} result is ambiguous for {stem}: {} candidates",
            candidates.len()
        ),
    }
}

#[derive(Debug, Serialize)]
pub struct LoadedTask {
    pub meta: SavedMeta,
    pub raw_json: Value,
    pub asr_human_review: Option<Value>,
    pub corrected_json: Option<Value>,
    pub polished_text: Option<String>,
}

pub fn load_task(stem: &str) -> Result<LoadedTask> {
    let stem = sanitize_stem(stem);
    let lock = stem_lock(&stem);
    let _guard = lock
        .lock()
        .map_err(|_| anyhow::anyhow!("library lock poisoned for {stem}"))?;
    let dir = task_dir(&stem);
    let meta = load_existing_meta(&stem)?;
    let raw_path = if meta.raw_filename.is_empty() {
        unique_legacy_raw_path(&dir, &stem)?
    } else {
        configured_raw_path(&dir, &stem, &meta.raw_filename)?
    };
    let raw_text = std::fs::read_to_string(&raw_path)
        .with_context(|| format!("read {}", raw_path.display()))?;
    let mut raw_json: Value =
        serde_json::from_str(&raw_text).with_context(|| format!("parse {}", raw_path.display()))?;

    let asr_human_review_path = dir.join("asr_human_review.json");
    let asr_human_review = match std::fs::read_to_string(&asr_human_review_path) {
        Ok(contents) => {
            let value: Value = serde_json::from_str(&contents).with_context(|| {
                format!(
                    "parse ASR human review sidecar {}",
                    asr_human_review_path.display()
                )
            })?;
            validate_asr_review(&value).with_context(|| {
                format!(
                    "validate ASR human review sidecar {}",
                    asr_human_review_path.display()
                )
            })?;
            Some(value)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => Err(error).with_context(|| {
            format!(
                "read ASR human review sidecar {}",
                asr_human_review_path.display()
            )
        })?,
    };

    let corrected_json = if meta.has_corrected {
        let expected_corrected_path = dir.join(format!("{stem}_corrected.json"));
        let corrected_path = if regular_file(&expected_corrected_path) {
            expected_corrected_path
        } else {
            unique_legacy_stage_path(&dir, &stem, "_corrected.json", "corrected")?
        };
        let value = std::fs::read_to_string(&corrected_path)
            .with_context(|| format!("read {}", corrected_path.display()))?;
        Some(
            serde_json::from_str(&value)
                .with_context(|| format!("parse {}", corrected_path.display()))?,
        )
    } else {
        None
    };

    let polished_text = if meta.has_polished {
        let expected_polished_path = dir.join(format!("{stem}_完整版.txt"));
        let polished_path = if regular_file(&expected_polished_path) {
            expected_polished_path
        } else {
            unique_legacy_stage_path(&dir, &stem, "_完整版.txt", "polished")?
        };
        let s = std::fs::read_to_string(&polished_path)
            .with_context(|| format!("read {}", polished_path.display()))?;
        Some({
            // Strip the two `# ...` header lines we wrote, return body only.
            let mut body_start = 0;
            let mut header_lines = 0;
            for (i, line) in s.lines().enumerate() {
                if line.starts_with('#') {
                    header_lines += 1;
                    continue;
                }
                if header_lines > 0 && line.trim().is_empty() {
                    continue;
                }
                body_start = s.lines().take(i).map(|l| l.len() + 1).sum::<usize>();
                break;
            }
            s[body_start..].trim_start().trim_end().to_string()
        })
    } else {
        None
    };

    if let (Some(audio_path), Some(object)) = (meta.audio_path.as_ref(), raw_json.as_object_mut()) {
        object.insert("audio".to_string(), Value::String(audio_path.clone()));
    }
    Ok(LoadedTask {
        meta,
        raw_json,
        asr_human_review,
        corrected_json,
        polished_text,
    })
}

pub fn delete_task(stem: &str) -> Result<()> {
    let stem = sanitize_stem(stem);
    let lock = stem_lock(&stem);
    let _guard = lock
        .lock()
        .map_err(|_| anyhow::anyhow!("library lock poisoned for {stem}"))?;
    ensure_real_library_root()?;
    let dir = task_dir(&stem);
    match std::fs::symlink_metadata(&dir) {
        Ok(_) => {
            validate_existing_task_layout(&dir)?;
            std::fs::remove_dir_all(&dir).with_context(|| format!("remove {}", dir.display()))?;
            if let Some(parent) = dir.parent() {
                File::open(parent)?.sync_all()?;
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| format!("inspect {}", dir.display())),
    }
    Ok(())
}

/// 把已存在的 `<stem>/` 重命名为 `<stem>-YYYYMMDD-HHMM/` 防覆盖,返回新路径。
pub fn archive_task(stem: &str) -> Result<Option<String>> {
    let stem = sanitize_stem(stem);
    let _namespace_guard = LIBRARY_NAMESPACE_LOCK
        .lock()
        .map_err(|_| anyhow::anyhow!("library namespace lock poisoned"))?;
    let source_lock = stem_lock(&stem);
    let _source_guard = source_lock
        .lock()
        .map_err(|_| anyhow::anyhow!("library lock poisoned for {stem}"))?;
    ensure_real_library_root()?;
    let dir = task_dir(&stem);
    match std::fs::symlink_metadata(&dir) {
        Ok(_) => {
            validate_existing_task_layout(&dir)?;
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("inspect {}", dir.display())),
    }
    let mut meta = load_existing_meta(&stem)?;
    let original_meta = std::fs::read_to_string(meta_path(&stem))
        .with_context(|| format!("read metadata before archiving {stem}"))?;
    let preferred_audio_relative = meta
        .audio_path
        .as_deref()
        .and_then(|path| task_audio_relative_path(&stem, path));

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let secs = now;
    let days = secs / 86400 + 719528;
    let (year, month, day) = days_to_ymd(days);
    let hour = (secs % 86400) / 3600;
    let minute = (secs % 3600) / 60;
    let tag = format!("{:04}{:02}{:02}-{:02}{:02}", year, month, day, hour, minute);

    let mut new_name = format!("{stem}-{tag}");
    let mut new_path = library_root().join(&new_name);
    let mut suffix = 0;
    while new_path.exists() {
        suffix += 1;
        new_name = format!("{stem}-{tag}-{suffix}");
        new_path = library_root().join(&new_name);
    }
    let destination_lock = stem_lock(&new_name);
    let _destination_guard = destination_lock
        .lock()
        .map_err(|_| anyhow::anyhow!("library lock poisoned for {new_name}"))?;
    std::fs::rename(&dir, &new_path)
        .with_context(|| format!("archive rename {} -> {}", dir.display(), new_path.display()))?;
    File::open(library_root())?.sync_all()?;

    let archived_meta_path = new_path.join("task.json");
    let migration = (|| -> Result<()> {
        let preferred_audio = preferred_audio_relative
            .as_ref()
            .map(|relative| new_path.join("audio").join(relative))
            .filter(|path| nonempty_regular_file(path));
        let audio_path = match preferred_audio {
            Some(path) => Some(path.to_string_lossy().into_owned()),
            None => existing_stable_audio(&new_name)?,
        };
        meta.stem = new_name.clone();
        meta.audio_path = audio_path;
        write_file(&archived_meta_path, &serde_json::to_string_pretty(&meta)?)?;
        Ok(())
    })();
    if let Err(error) = migration {
        let mut rollback_errors = Vec::new();
        if let Err(rollback_error) = write_file(&archived_meta_path, &original_meta) {
            rollback_errors.push(format!("restore metadata: {rollback_error:#}"));
        }
        if let Err(rollback_error) = std::fs::rename(&new_path, &dir) {
            rollback_errors.push(format!("restore directory: {rollback_error}"));
        }
        if !rollback_errors.is_empty() {
            anyhow::bail!(
                "archive migration failed: {error:#}; rollback failed: {}",
                rollback_errors.join("; ")
            );
        }
        return Err(error);
    }
    Ok(Some(new_path.to_string_lossy().into_owned()))
}

/// Convert days since year 0 (proleptic) to (year, month, day).
fn days_to_ymd(days: i64) -> (i64, i64, i64) {
    let mut d = days;
    let mut year = 0i64;
    loop {
        let y_days = if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
            366
        } else {
            365
        };
        if d < y_days {
            break;
        }
        d -= y_days;
        year += 1;
    }
    let month_days = if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 0;
    while d >= month_days[month] {
        d -= month_days[month];
        month += 1;
    }
    (year, (month + 1) as i64, (d + 1) as i64)
}

#[cfg(test)]
mod tests {
    use super::{
        archive_task, atomic_write_with, delete_task, list_library, load_existing_meta, load_task,
        mp4_playback_preparation, playback_cache_path, prepare_media_for_playback, read_mp4_layout,
        sanitize_stem, save_asr_review, save_corrected, save_meta, save_polished, save_raw,
        save_raw_and_corrected, write_file, Mp4Layout, Mp4PlaybackPreparation, SaveAsrReviewArgs,
        SaveCorrectedArgs, SavePolishedArgs, SaveRawAndCorrectedArgs, SaveRawArgs, SavedMeta,
        ASSET_PROTOCOL_MAX_RANGE_BYTES, TEST_LIBRARY_ROOT,
    };
    use once_cell::sync::Lazy;
    use serde_json::json;
    use std::io::{self, Write};
    use std::path::{Path, PathBuf};
    use std::sync::{Barrier, Mutex, MutexGuard};

    static TEST_ROOT_SERIAL: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    struct TestLibrary {
        root: PathBuf,
        _serial: MutexGuard<'static, ()>,
    }

    impl TestLibrary {
        fn new() -> Self {
            let serial = TEST_ROOT_SERIAL
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let root = std::env::temp_dir().join(format!(
                "localscribe-library-test-{}-{}-{}",
                std::process::id(),
                super::TEMP_FILE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                super::TEMP_FILE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&root).unwrap();
            *TEST_LIBRARY_ROOT
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(root.clone());
            Self {
                root,
                _serial: serial,
            }
        }

        fn task_dir(&self, stem: &str) -> PathBuf {
            self.root.join(stem)
        }
    }

    impl Drop for TestLibrary {
        fn drop(&mut self) {
            *TEST_LIBRARY_ROOT
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn temp_test_dir(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "localscribe-atomic-test-{label}-{}-{}",
            std::process::id(),
            super::TEMP_FILE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn temporary_files_for(path: &Path) -> Vec<PathBuf> {
        let prefix = format!(".{}.", path.file_name().unwrap().to_string_lossy());
        std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|entry| {
                entry
                    .file_name()
                    .map(|name| name.to_string_lossy().starts_with(&prefix))
                    .unwrap_or(false)
            })
            .collect()
    }

    fn write_test_atom(file: &mut std::fs::File, kind: &[u8; 4], payload_len: usize) {
        let size = u32::try_from(payload_len + 8).unwrap();
        file.write_all(&size.to_be_bytes()).unwrap();
        file.write_all(kind).unwrap();
        file.write_all(&vec![0_u8; payload_len]).unwrap();
    }

    fn create_oversized_index_m4a(path: &Path) -> bool {
        let output = std::process::Command::new(super::ffmpeg_executable())
            .args([
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=mono:sample_rate=16000",
                "-t",
                "0.2",
                "-c:a",
                "aac",
            ])
            .arg(path)
            .output();
        if !matches!(output, Ok(ref output) if output.status.success()) {
            return false;
        }

        let layout = read_mp4_layout(path).unwrap();
        let moov_offset = usize::try_from(layout.moov_offset.unwrap()).unwrap();
        let moov_size = usize::try_from(layout.moov_size.unwrap()).unwrap();
        let mut bytes = std::fs::read(path).unwrap();
        assert_eq!(moov_offset + moov_size, bytes.len());

        let target_size = usize::try_from(ASSET_PROTOCOL_MAX_RANGE_BYTES).unwrap() + 4096;
        let free_size = target_size - moov_size;
        assert!(free_size >= 8);
        bytes[moov_offset..moov_offset + 4]
            .copy_from_slice(&u32::try_from(target_size).unwrap().to_be_bytes());
        bytes.extend_from_slice(&u32::try_from(free_size).unwrap().to_be_bytes());
        bytes.extend_from_slice(b"free");
        bytes.resize(bytes.len() + free_size - 8, 0);
        std::fs::write(path, bytes).unwrap();
        true
    }

    fn encoded_packet_hash(path: &Path) -> Option<String> {
        let output = std::process::Command::new(super::ffmpeg_executable())
            .args(["-v", "error", "-i"])
            .arg(path)
            .args([
                "-map", "0:a:0", "-c", "copy", "-f", "hash", "-hash", "sha256", "-",
            ])
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    #[test]
    fn mp4_layout_detects_tail_index() {
        let dir = temp_test_dir("mp4-tail-index");
        let path = dir.join("long.m4a");
        let mut file = std::fs::File::create(&path).unwrap();
        write_test_atom(&mut file, b"ftyp", 8);
        write_test_atom(&mut file, b"mdat", 32);
        write_test_atom(&mut file, b"moov", 16);
        drop(file);

        let layout = read_mp4_layout(&path).unwrap();
        assert!(layout.moov_offset.unwrap() > layout.mdat_offset.unwrap());
        assert_eq!(
            mp4_playback_preparation(&layout),
            Some(Mp4PlaybackPreparation::FastStart)
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn mp4_layout_detects_faststart_index() {
        let dir = temp_test_dir("mp4-faststart-index");
        let path = dir.join("ready.m4a");
        let mut file = std::fs::File::create(&path).unwrap();
        write_test_atom(&mut file, b"ftyp", 8);
        write_test_atom(&mut file, b"moov", 16);
        write_test_atom(&mut file, b"mdat", 32);
        drop(file);

        let layout = read_mp4_layout(&path).unwrap();
        assert!(layout.moov_offset.unwrap() < layout.mdat_offset.unwrap());
        assert_eq!(mp4_playback_preparation(&layout), None);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn mp4_playback_fragments_front_index_that_exceeds_asset_range() {
        let layout = Mp4Layout {
            moov_offset: Some(28),
            moov_size: Some(ASSET_PROTOCOL_MAX_RANGE_BYTES),
            mdat_offset: Some(ASSET_PROTOCOL_MAX_RANGE_BYTES + 28),
            moof_offset: None,
        };

        assert_eq!(
            mp4_playback_preparation(&layout),
            Some(Mp4PlaybackPreparation::Fragmented)
        );
    }

    #[test]
    fn mp4_playback_faststarts_small_tail_index_even_in_large_file() {
        let layout = Mp4Layout {
            mdat_offset: Some(28),
            moov_offset: Some(150 * 1024 * 1024),
            moov_size: Some(100 * 1024),
            moof_offset: None,
        };

        assert_eq!(
            mp4_playback_preparation(&layout),
            Some(Mp4PlaybackPreparation::FastStart)
        );
    }

    #[test]
    fn mp4_playback_fragments_oversized_tail_index() {
        let layout = Mp4Layout {
            mdat_offset: Some(28),
            moov_offset: Some(150 * 1024 * 1024),
            moov_size: Some(ASSET_PROTOCOL_MAX_RANGE_BYTES),
            moof_offset: None,
        };

        assert_eq!(
            mp4_playback_preparation(&layout),
            Some(Mp4PlaybackPreparation::Fragmented)
        );
    }

    #[test]
    fn mp4_playback_accepts_small_fragmented_index() {
        let layout = Mp4Layout {
            moov_offset: Some(28),
            moov_size: Some(697),
            moof_offset: Some(725),
            mdat_offset: Some(2697),
        };

        assert_eq!(mp4_playback_preparation(&layout), None);
    }

    #[test]
    fn playback_cache_uses_a_separate_nonhidden_file_for_asset_protocol() {
        let source = Path::new("/tmp/meeting.m4a");
        assert_eq!(
            playback_cache_path(source).unwrap(),
            Path::new("/tmp/meeting.m4a.localscribe-playback.m4a")
        );
        assert!(!playback_cache_path(source)
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with('.'));
    }

    #[test]
    fn playback_preparation_preserves_analysis_audio_and_aac_packets() {
        let library = TestLibrary::new();
        let audio_dir = library.task_dir("long-meeting").join("audio");
        std::fs::create_dir_all(&audio_dir).unwrap();
        let source = audio_dir.join("long-meeting.m4a");
        if !create_oversized_index_m4a(&source) {
            return;
        }

        assert_eq!(
            mp4_playback_preparation(&read_mp4_layout(&source).unwrap()),
            Some(Mp4PlaybackPreparation::Fragmented)
        );
        let source_before = std::fs::read(&source).unwrap();
        let packet_hash_before = encoded_packet_hash(&source).unwrap();

        let prepared = prepare_media_for_playback(source.to_str().unwrap()).unwrap();
        let prepared_path = PathBuf::from(&prepared.path);

        assert!(prepared.optimized);
        assert_ne!(prepared_path, source);
        assert_eq!(
            prepared_path,
            playback_cache_path(&source.canonicalize().unwrap()).unwrap()
        );
        assert!(prepared_path.is_file());
        assert_eq!(std::fs::read(&source).unwrap(), source_before);
        assert_eq!(encoded_packet_hash(&source).unwrap(), packet_hash_before);
        assert_eq!(
            encoded_packet_hash(&prepared_path).unwrap(),
            packet_hash_before
        );
        assert!(read_mp4_layout(&prepared_path)
            .unwrap()
            .moof_offset
            .is_some());
    }

    fn write_raw_task(library: &TestLibrary, stem: &str, raw: &[u8]) -> PathBuf {
        let dir = library.task_dir(stem);
        std::fs::create_dir_all(&dir).unwrap();
        let raw_filename = format!("{stem}.json");
        let raw_path = dir.join(&raw_filename);
        std::fs::write(&raw_path, raw).unwrap();
        save_meta(&SavedMeta {
            stem: stem.to_string(),
            audio_filename: format!("{stem}.wav"),
            raw_filename,
            created_at: 1,
            updated_at: 1,
            ..SavedMeta::default()
        })
        .unwrap();
        raw_path
    }

    fn valid_review() -> serde_json::Value {
        json!({
            "schema_version": 1,
            "items": [{
                "id": "coverage:0-1000",
                "start": 0.0,
                "end": 1.0,
                "status": "pending",
                "heard_text": "",
                "note": "listen again",
                "replacement_text": ""
            }]
        })
    }

    #[test]
    fn sanitize_stem_replaces_path_separators_and_control_chars() {
        assert_eq!(sanitize_stem("../secret"), "_secret");
        assert_eq!(sanitize_stem("..\\secret"), "_secret");
        assert_eq!(sanitize_stem("meeting/name\npart"), "meeting_name_part");
    }

    #[test]
    fn sanitize_stem_trims_dot_space_edges_and_uses_safe_default() {
        assert_eq!(sanitize_stem("  .项目文件.  "), "项目文件");
        assert_eq!(sanitize_stem(" ... "), "meeting");
        assert_eq!(sanitize_stem("\n\t"), "meeting");
    }

    #[test]
    fn failed_atomic_write_keeps_old_file_and_cleans_temporary_file() {
        let dir = temp_test_dir("keep-old");
        let path = dir.join("task.json");
        std::fs::write(&path, "old complete contents").unwrap();

        let result = atomic_write_with(&path, |file| {
            file.write_all(b"partial new contents")?;
            Err(io::Error::new(io::ErrorKind::Other, "injected failure"))
        });

        assert!(result.is_err());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "old complete contents"
        );
        assert!(temporary_files_for(&path).is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn atomic_write_replaces_file_completely() {
        let dir = temp_test_dir("replace");
        let path = dir.join("result.json");
        std::fs::write(&path, "old trailing data that must disappear").unwrap();

        write_file(&path, "{\"complete\":true}").unwrap();

        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{\"complete\":true}"
        );
        assert!(temporary_files_for(&path).is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failed_replace_cleans_temporary_file_without_removing_destination() {
        let dir = temp_test_dir("cleanup");
        let destination = dir.join("existing-directory");
        std::fs::create_dir(&destination).unwrap();

        let result = write_file(&destination, "replacement");

        assert!(result.is_err());
        assert!(destination.is_dir());
        assert!(temporary_files_for(&destination).is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn save_raw_persists_submitted_json_bytes_and_overlays_stable_audio_in_memory() {
        let library = TestLibrary::new();
        let stem = "raw-bytes-verbatim";
        let source_audio = library.root.join("source audio.wav");
        std::fs::write(&source_audio, b"stable audio bytes").unwrap();
        let submitted = "{\r\n  \"audio\" : \"submitted/path.wav\",\r\n  \"segments\" : [], \"backend\":\"test\",\r\n  \"model_id\" : \"test-model\"\r\n}\n\t";

        let saved = save_raw(SaveRawArgs {
            stem: stem.to_string(),
            audio_filename: "source audio.wav".to_string(),
            source_audio: Some(source_audio.to_string_lossy().into_owned()),
            txt: "raw text".to_string(),
            srt: "raw srt".to_string(),
            json: submitted.to_string(),
            result: json!({
                "duration": 1.25,
                "segments": [],
                "backend": "test",
                "model_id": "test-model"
            }),
        })
        .unwrap();
        let raw_path = library.task_dir(stem).join(format!("{stem}.json"));

        assert_eq!(std::fs::read(&raw_path).unwrap(), submitted.as_bytes());
        assert_eq!(saved.raw_filename, format!("{stem}.json"));
        let stable_audio = saved.audio_path.clone().unwrap();
        assert_ne!(stable_audio, "submitted/path.wav");
        assert_eq!(
            load_task(stem).unwrap().raw_json["audio"],
            json!(stable_audio)
        );
        assert_eq!(std::fs::read(raw_path).unwrap(), submitted.as_bytes());
    }

    #[test]
    fn save_raw_and_corrected_commits_both_artifacts_together() {
        let library = TestLibrary::new();
        let stem = "diarization-atomic-success";
        let source_audio = library.root.join("source.wav");
        std::fs::write(&source_audio, b"audio").unwrap();

        let saved = save_raw_and_corrected(SaveRawAndCorrectedArgs {
            raw: SaveRawArgs {
                stem: stem.to_string(),
                audio_filename: "source.wav".to_string(),
                source_audio: Some(source_audio.to_string_lossy().into_owned()),
                txt: "raw new".to_string(),
                srt: "raw srt".to_string(),
                json: r#"{"segments":[{"start":0,"end":1,"text":"raw","speaker":"SPEAKER_A"}],"backend":"test","model_id":"test","duration":1}"#.to_string(),
                result: json!({
                    "duration": 1.0,
                    "segments": [{"start": 0.0, "end": 1.0, "text": "raw", "speaker": "SPEAKER_A"}],
                    "backend": "test",
                    "model_id": "test"
                }),
            },
            corrected: Some(SaveCorrectedArgs {
                stem: stem.to_string(),
                txt: "corrected new".to_string(),
                srt: "corrected srt".to_string(),
                json: r#"{"segments":[{"start":0,"end":1,"text":"corrected","speaker":"SPEAKER_A"}]}"#.to_string(),
                diff: "diff".to_string(),
                model: "test-corrector".to_string(),
                changed: 1,
                total: 1,
                glossary: None,
            }),
            clear_corrected: false,
        })
        .unwrap();

        let dir = library.task_dir(stem);
        assert_eq!(
            std::fs::read_to_string(dir.join(format!("{stem}.txt"))).unwrap(),
            "raw new"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join(format!("{stem}_corrected.txt"))).unwrap(),
            "corrected new"
        );
        assert!(saved.has_corrected);
        assert_eq!(saved.correction_model.as_deref(), Some("test-corrector"));
        assert_eq!(
            load_task(stem).unwrap().corrected_json.unwrap()["segments"][0]["text"],
            "corrected"
        );
    }

    #[test]
    fn save_raw_and_corrected_can_clear_corrected_metadata_without_deleting_files() {
        let library = TestLibrary::new();
        let stem = "diarization-clear-corrected";
        let source_audio = library.root.join("source.wav");
        std::fs::write(&source_audio, b"audio").unwrap();
        save_raw_and_corrected(SaveRawAndCorrectedArgs {
            raw: SaveRawArgs {
                stem: stem.to_string(),
                audio_filename: "source.wav".to_string(),
                source_audio: Some(source_audio.to_string_lossy().into_owned()),
                txt: "raw".to_string(),
                srt: "raw srt".to_string(),
                json: r#"{"segments":[],"backend":"test","model_id":"test","duration":1}"#.to_string(),
                result: json!({"duration": 1.0, "segments": [], "backend": "test", "model_id": "test"}),
            },
            corrected: Some(SaveCorrectedArgs {
                stem: stem.to_string(),
                txt: "corrected".to_string(),
                srt: "corrected srt".to_string(),
                json: r#"{"segments":[]}"#.to_string(),
                diff: "diff".to_string(),
                model: "test-corrector".to_string(),
                changed: 0,
                total: 0,
                glossary: None,
            }),
            clear_corrected: false,
        })
        .unwrap();

        let saved = save_raw_and_corrected(SaveRawAndCorrectedArgs {
            raw: SaveRawArgs {
                stem: stem.to_string(),
                audio_filename: "source.wav".to_string(),
                source_audio: None,
                txt: "raw restored".to_string(),
                srt: "raw restored srt".to_string(),
                json: r#"{"segments":[],"backend":"test","model_id":"test","duration":1}"#.to_string(),
                result: json!({"duration": 1.0, "segments": [], "backend": "test", "model_id": "test"}),
            },
            corrected: None,
            clear_corrected: true,
        })
        .unwrap();

        assert!(!saved.has_corrected);
        assert!(saved.correction_model.is_none());
        assert!(load_task(stem).unwrap().corrected_json.is_none());
        assert!(library
            .task_dir(stem)
            .join(format!("{stem}_corrected.json"))
            .is_file());
    }

    #[test]
    fn save_raw_and_corrected_rolls_back_raw_when_corrected_commit_fails() {
        let library = TestLibrary::new();
        let stem = "diarization-atomic-rollback";
        let source_audio = library.root.join("source.wav");
        std::fs::write(&source_audio, b"audio").unwrap();
        save_raw(SaveRawArgs {
            stem: stem.to_string(),
            audio_filename: "source.wav".to_string(),
            source_audio: Some(source_audio.to_string_lossy().into_owned()),
            txt: "raw old".to_string(),
            srt: "old srt".to_string(),
            json: r#"{"segments":[],"backend":"test","model_id":"test","duration":1}"#.to_string(),
            result: json!({"duration": 1.0, "segments": [], "backend": "test", "model_id": "test"}),
        })
        .unwrap();
        let dir = library.task_dir(stem);
        std::fs::create_dir(dir.join(format!("{stem}_corrected.json"))).unwrap();

        let result = save_raw_and_corrected(SaveRawAndCorrectedArgs {
            raw: SaveRawArgs {
                stem: stem.to_string(),
                audio_filename: "source.wav".to_string(),
                source_audio: None,
                txt: "raw new".to_string(),
                srt: "new srt".to_string(),
                json: r#"{"segments":[{"start":0,"end":1,"text":"new"}],"backend":"test","model_id":"test","duration":1}"#.to_string(),
                result: json!({
                    "duration": 1.0,
                    "segments": [{"start": 0.0, "end": 1.0, "text": "new"}],
                    "backend": "test",
                    "model_id": "test"
                }),
            },
            corrected: Some(SaveCorrectedArgs {
                stem: stem.to_string(),
                txt: "corrected new".to_string(),
                srt: "corrected srt".to_string(),
                json: r#"{"segments":[]}"#.to_string(),
                diff: "diff".to_string(),
                model: "test-corrector".to_string(),
                changed: 0,
                total: 0,
                glossary: None,
            }),
            clear_corrected: false,
        });

        assert!(result.is_err());
        assert_eq!(
            std::fs::read_to_string(dir.join(format!("{stem}.txt"))).unwrap(),
            "raw old"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join(format!("{stem}.json"))).unwrap(),
            r#"{"segments":[],"backend":"test","model_id":"test","duration":1}"#
        );
        assert!(!dir.join(format!("{stem}_corrected.txt")).exists());
        assert!(!load_existing_meta(stem).unwrap().has_corrected);
    }

    #[test]
    fn save_raw_recovers_from_empty_audio_directory_without_metadata() {
        let library = TestLibrary::new();
        let stem = "recover-empty-audio-dir";
        std::fs::create_dir_all(library.task_dir(stem).join("audio")).unwrap();

        let saved = save_raw(SaveRawArgs {
            stem: stem.to_string(),
            audio_filename: "source.wav".to_string(),
            source_audio: None,
            txt: "raw".to_string(),
            srt: "srt".to_string(),
            json: r#"{"segments":[],"backend":"test","model_id":"test","duration":1}"#.to_string(),
            result: json!({"duration": 1.0, "segments": [], "backend": "test", "model_id": "test"}),
        })
        .unwrap();

        assert_eq!(saved.stem, stem);
        assert!(library.task_dir(stem).join("task.json").is_file());
    }

    #[cfg(unix)]
    #[test]
    fn save_and_delete_reject_symlinked_library_root() {
        use std::os::unix::fs::symlink;

        let library = TestLibrary::new();
        let actual = library.root.join("actual-library");
        let linked = library.root.join("linked-library");
        std::fs::create_dir_all(actual.join("victim")).unwrap();
        std::fs::write(actual.join("victim/keep.txt"), b"keep").unwrap();
        symlink(&actual, &linked).unwrap();
        *TEST_LIBRARY_ROOT
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(linked);

        let save_result = save_raw(SaveRawArgs {
            stem: "new-task".to_string(),
            audio_filename: "source.wav".to_string(),
            source_audio: None,
            txt: "raw".to_string(),
            srt: "srt".to_string(),
            json: r#"{"segments":[],"backend":"test","model_id":"test","duration":1}"#.to_string(),
            result: json!({"duration": 1.0, "segments": [], "backend": "test", "model_id": "test"}),
        });
        let delete_result = delete_task("victim");

        *TEST_LIBRARY_ROOT
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(library.root.clone());
        assert!(save_result.is_err());
        assert!(delete_result.is_err());
        assert_eq!(
            std::fs::read(actual.join("victim/keep.txt")).unwrap(),
            b"keep"
        );
        assert!(!actual.join("new-task").exists());
    }

    #[cfg(unix)]
    #[test]
    fn save_archive_and_delete_reject_existing_audio_symlink() {
        use std::os::unix::fs::symlink;

        let library = TestLibrary::new();
        let stem = "reject-existing-audio-symlink";
        let task_dir = library.task_dir(stem);
        let outside = library.root.join("outside-existing-audio");
        std::fs::create_dir_all(&task_dir).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("keep.txt"), b"keep").unwrap();
        symlink(&outside, task_dir.join("audio")).unwrap();
        save_meta(&SavedMeta {
            stem: stem.to_string(),
            audio_filename: "source.wav".to_string(),
            raw_filename: format!("{stem}.json"),
            ..SavedMeta::default()
        })
        .unwrap();

        let save_result = save_raw(SaveRawArgs {
            stem: stem.to_string(),
            audio_filename: "source.wav".to_string(),
            source_audio: None,
            txt: "raw".to_string(),
            srt: "srt".to_string(),
            json: r#"{"segments":[],"backend":"test","model_id":"test","duration":1}"#.to_string(),
            result: json!({"duration": 1.0, "segments": [], "backend": "test", "model_id": "test"}),
        });
        let archive_result = archive_task(stem);
        let delete_result = delete_task(stem);

        assert!(save_result.is_err());
        assert!(archive_result.is_err());
        assert!(delete_result.is_err());
        assert_eq!(std::fs::read(outside.join("keep.txt")).unwrap(), b"keep");
    }

    #[cfg(unix)]
    #[test]
    fn save_raw_rejects_symlinked_task_directory() {
        use std::os::unix::fs::symlink;

        let library = TestLibrary::new();
        let stem = "reject-task-symlink";
        let outside = library.root.join("outside-task");
        std::fs::create_dir_all(&outside).unwrap();
        symlink(&outside, library.task_dir(stem)).unwrap();

        let result = save_raw(SaveRawArgs {
            stem: stem.to_string(),
            audio_filename: "source.wav".to_string(),
            source_audio: None,
            txt: "raw".to_string(),
            srt: "srt".to_string(),
            json: r#"{"segments":[],"backend":"test","model_id":"test","duration":1}"#.to_string(),
            result: json!({"duration": 1.0, "segments": [], "backend": "test", "model_id": "test"}),
        });

        assert!(result.is_err());
        assert!(std::fs::read_dir(&outside).unwrap().next().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn save_raw_rejects_symlinked_audio_directory() {
        use std::os::unix::fs::symlink;

        let library = TestLibrary::new();
        let stem = "reject-audio-symlink";
        let task_dir = library.task_dir(stem);
        let outside = library.root.join("outside-audio");
        std::fs::create_dir_all(&task_dir).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        symlink(&outside, task_dir.join("audio")).unwrap();
        let source_audio = library.root.join("source.wav");
        std::fs::write(&source_audio, b"audio").unwrap();

        let result = save_raw(SaveRawArgs {
            stem: stem.to_string(),
            audio_filename: "source.wav".to_string(),
            source_audio: Some(source_audio.to_string_lossy().into_owned()),
            txt: "raw".to_string(),
            srt: "srt".to_string(),
            json: r#"{"segments":[],"backend":"test","model_id":"test","duration":1}"#.to_string(),
            result: json!({"duration": 1.0, "segments": [], "backend": "test", "model_id": "test"}),
        });

        assert!(result.is_err());
        assert!(std::fs::read_dir(&outside).unwrap().next().is_none());
    }

    #[test]
    fn load_existing_meta_requires_object_shape_and_matching_stem() {
        let library = TestLibrary::new();

        let missing_error = load_existing_meta("missing-meta").unwrap_err().to_string();
        assert!(missing_error.contains("task metadata missing"));

        let non_object_stem = "non-object-meta";
        let non_object_dir = library.task_dir(non_object_stem);
        std::fs::create_dir_all(&non_object_dir).unwrap();
        std::fs::write(non_object_dir.join("task.json"), "[]").unwrap();
        let non_object_error = load_existing_meta(non_object_stem).unwrap_err().to_string();
        assert!(non_object_error.contains("must be a JSON object"));

        let invalid_stem = "invalid-object-meta";
        let invalid_dir = library.task_dir(invalid_stem);
        std::fs::create_dir_all(&invalid_dir).unwrap();
        std::fs::write(invalid_dir.join("task.json"), "{}").unwrap();
        let invalid_error = load_existing_meta(invalid_stem).unwrap_err().to_string();
        assert!(invalid_error.contains("parse metadata"));

        let mismatch_stem = "mismatch-meta";
        let mismatch_dir = library.task_dir(mismatch_stem);
        std::fs::create_dir_all(&mismatch_dir).unwrap();
        std::fs::write(
            mismatch_dir.join("task.json"),
            serde_json::to_vec_pretty(&SavedMeta {
                stem: "different-directory".to_string(),
                audio_filename: "source.wav".to_string(),
                created_at: 1,
                updated_at: 1,
                ..SavedMeta::default()
            })
            .unwrap(),
        )
        .unwrap();
        let mismatch_error = load_existing_meta(mismatch_stem).unwrap_err().to_string();
        assert!(mismatch_error.contains("stem mismatch"));
    }

    #[test]
    fn save_raw_initializes_an_existing_empty_directory_and_sets_raw_filename() {
        let library = TestLibrary::new();
        let stem = "empty-task-directory";
        std::fs::create_dir_all(library.task_dir(stem)).unwrap();

        let saved = save_raw(SaveRawArgs {
            stem: stem.to_string(),
            audio_filename: "source.wav".to_string(),
            source_audio: None,
            txt: "raw".to_string(),
            srt: "raw srt".to_string(),
            json: r#"{"segments":[],"backend":"test","model_id":"model"}"#.to_string(),
            result: json!({
                "segments": [],
                "backend": "test",
                "model_id": "model"
            }),
        })
        .unwrap();

        assert_eq!(saved.raw_filename, format!("{stem}.json"));
        assert_eq!(
            load_existing_meta(stem).unwrap().raw_filename,
            format!("{stem}.json")
        );
    }

    #[test]
    fn orphan_files_without_metadata_are_rejected_by_raw_stage_review_and_load() {
        let library = TestLibrary::new();
        let stem = "orphan-without-meta";
        let dir = library.task_dir(stem);
        std::fs::create_dir_all(&dir).unwrap();
        let orphan_path = dir.join("orphan.txt");
        std::fs::write(&orphan_path, b"keep me").unwrap();

        let raw_error = save_raw(SaveRawArgs {
            stem: stem.to_string(),
            audio_filename: "source.wav".to_string(),
            source_audio: None,
            txt: "raw".to_string(),
            srt: "raw srt".to_string(),
            json: r#"{"segments":[],"backend":"test","model_id":"model"}"#.to_string(),
            result: json!({"segments": [], "backend": "test", "model_id": "model"}),
        })
        .unwrap_err()
        .to_string();
        assert!(raw_error.contains("metadata missing from nonempty directory"));

        let corrected_error = save_corrected(SaveCorrectedArgs {
            stem: stem.to_string(),
            txt: "corrected".to_string(),
            srt: "corrected srt".to_string(),
            json: r#"{"corrected":true}"#.to_string(),
            diff: "diff".to_string(),
            model: "corrector".to_string(),
            changed: 1,
            total: 1,
            glossary: None,
        })
        .unwrap_err()
        .to_string();
        assert!(corrected_error.contains("task metadata missing"));

        let polished_error = save_polished(SavePolishedArgs {
            stem: stem.to_string(),
            text: "polished".to_string(),
            model: "polisher".to_string(),
            source: Some("raw".to_string()),
        })
        .unwrap_err()
        .to_string();
        assert!(polished_error.contains("task metadata missing"));

        let review_error = save_asr_review(SaveAsrReviewArgs {
            stem: stem.to_string(),
            review: valid_review(),
        })
        .unwrap_err()
        .to_string();
        assert!(review_error.contains("task metadata missing"));
        assert!(load_task(stem)
            .unwrap_err()
            .to_string()
            .contains("task metadata missing"));

        assert_eq!(std::fs::read(orphan_path).unwrap(), b"keep me");
        assert!(!dir.join("task.json").exists());
        assert!(!dir.join(format!("{stem}_corrected.txt")).exists());
        assert!(!dir.join(format!("{stem}_完整版.txt")).exists());
        assert!(!dir.join("asr_human_review.json").exists());
    }

    #[test]
    fn save_raw_migrates_external_metadata_audio_into_current_task_transaction() {
        let library = TestLibrary::new();
        let stem = "external-audio-migration";
        write_raw_task(
            &library,
            stem,
            br#"{"segments":[],"backend":"old","model_id":"old"}"#,
        );
        let external_audio = library.root.join("external-source.flac");
        std::fs::write(&external_audio, b"external audio bytes").unwrap();
        let mut meta = load_existing_meta(stem).unwrap();
        meta.audio_path = Some(external_audio.to_string_lossy().into_owned());
        save_meta(&meta).unwrap();

        let saved = save_raw(SaveRawArgs {
            stem: stem.to_string(),
            audio_filename: "external-source.flac".to_string(),
            source_audio: None,
            txt: "raw".to_string(),
            srt: "raw srt".to_string(),
            json: r#"{"segments":[],"backend":"new","model_id":"new"}"#.to_string(),
            result: json!({"segments": [], "backend": "new", "model_id": "new"}),
        })
        .unwrap();

        let migrated = library
            .task_dir(stem)
            .join("audio")
            .join(format!("{stem}.flac"));
        assert_eq!(saved.audio_path.as_deref(), migrated.to_str());
        assert_ne!(saved.audio_path.as_deref(), external_audio.to_str());
        assert_eq!(std::fs::read(migrated).unwrap(), b"external audio bytes");
        assert_eq!(
            std::fs::read(external_audio).unwrap(),
            b"external audio bytes"
        );
    }

    #[test]
    fn explicit_source_audio_wins_over_metadata_audio() {
        let library = TestLibrary::new();
        let stem = "source-audio-priority";
        write_raw_task(
            &library,
            stem,
            br#"{"segments":[],"backend":"old","model_id":"old"}"#,
        );
        let metadata_audio = library.root.join("metadata-source.wav");
        let explicit_audio = library.root.join("explicit-source.wav");
        std::fs::write(&metadata_audio, b"metadata audio").unwrap();
        std::fs::write(&explicit_audio, b"explicit audio").unwrap();
        let mut meta = load_existing_meta(stem).unwrap();
        meta.audio_path = Some(metadata_audio.to_string_lossy().into_owned());
        save_meta(&meta).unwrap();

        let saved = save_raw(SaveRawArgs {
            stem: stem.to_string(),
            audio_filename: "explicit-source.wav".to_string(),
            source_audio: Some(explicit_audio.to_string_lossy().into_owned()),
            txt: "raw".to_string(),
            srt: "raw srt".to_string(),
            json: r#"{"segments":[],"backend":"new","model_id":"new"}"#.to_string(),
            result: json!({"segments": [], "backend": "new", "model_id": "new"}),
        })
        .unwrap();

        assert_eq!(
            std::fs::read(saved.audio_path.unwrap()).unwrap(),
            b"explicit audio"
        );
    }

    #[test]
    fn raw_filename_is_preferred_and_legacy_multiple_candidates_are_ambiguous() {
        let library = TestLibrary::new();
        let stem = "raw-candidate-selection";
        let dir = library.task_dir(stem);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("preferred.json"),
            br#"{"segments":[],"backend":"preferred","model_id":"model"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("other.json"),
            br#"{"segments":[],"backend":"other","model_id":"model"}"#,
        )
        .unwrap();
        save_meta(&SavedMeta {
            stem: stem.to_string(),
            audio_filename: "source.wav".to_string(),
            raw_filename: "preferred.json".to_string(),
            created_at: 1,
            updated_at: 1,
            ..SavedMeta::default()
        })
        .unwrap();

        assert_eq!(load_task(stem).unwrap().raw_json["backend"], "preferred");

        let mut legacy_meta = load_existing_meta(stem).unwrap();
        legacy_meta.raw_filename.clear();
        save_meta(&legacy_meta).unwrap();
        let error = load_task(stem).unwrap_err().to_string();
        assert!(error.contains("ambiguous"), "unexpected error: {error}");
        assert!(error.contains("2 candidates"), "unexpected error: {error}");
    }

    #[test]
    fn corrected_save_rolls_back_earlier_files_when_a_later_target_fails() {
        let library = TestLibrary::new();
        let stem = "corrected-transaction-rollback";
        let dir = library.task_dir(stem);
        std::fs::create_dir_all(&dir).unwrap();
        let old_meta = SavedMeta {
            stem: stem.to_string(),
            audio_filename: "source.wav".to_string(),
            created_at: 1,
            updated_at: 2,
            ..SavedMeta::default()
        };
        save_meta(&old_meta).unwrap();
        let old_metadata_bytes = std::fs::read(dir.join("task.json")).unwrap();
        for (suffix, contents) in [
            ("_corrected.txt", b"old corrected text".as_slice()),
            ("_corrected.srt", b"old corrected srt".as_slice()),
            ("_corrected.json", br#"{"old":true}"#.as_slice()),
        ] {
            std::fs::write(dir.join(format!("{stem}{suffix}")), contents).unwrap();
        }
        let diff_path = dir.join(format!("{stem}_diff.txt"));
        std::fs::create_dir(&diff_path).unwrap();

        let error = save_corrected(SaveCorrectedArgs {
            stem: stem.to_string(),
            txt: "new corrected text".to_string(),
            srt: "new corrected srt".to_string(),
            json: r#"{"new":true}"#.to_string(),
            diff: "new diff".to_string(),
            model: "new-model".to_string(),
            changed: 1,
            total: 1,
            glossary: Some(json!({"term": "value"})),
        })
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("not a regular file"),
            "unexpected error: {error}"
        );
        assert_eq!(
            std::fs::read(dir.join(format!("{stem}_corrected.txt"))).unwrap(),
            b"old corrected text"
        );
        assert_eq!(
            std::fs::read(dir.join(format!("{stem}_corrected.srt"))).unwrap(),
            b"old corrected srt"
        );
        assert_eq!(
            std::fs::read(dir.join(format!("{stem}_corrected.json"))).unwrap(),
            br#"{"old":true}"#
        );
        assert!(diff_path.is_dir());
        assert_eq!(
            std::fs::read(dir.join("task.json")).unwrap(),
            old_metadata_bytes
        );
        assert!(!load_existing_meta(stem).unwrap().has_corrected);
        for entry in std::fs::read_dir(&dir).unwrap() {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            assert!(
                !name.ends_with(".tmp"),
                "left transaction temp file: {name}"
            );
        }
    }

    #[test]
    fn raw_save_rolls_back_new_files_and_staged_audio_when_commit_fails() {
        let library = TestLibrary::new();
        let stem = "raw-transaction-rollback";
        let dir = library.task_dir(stem);
        std::fs::create_dir_all(&dir).unwrap();
        save_meta(&SavedMeta {
            stem: stem.to_string(),
            audio_filename: "old.wav".to_string(),
            created_at: 1,
            updated_at: 2,
            ..SavedMeta::default()
        })
        .unwrap();
        let old_metadata_bytes = std::fs::read(dir.join("task.json")).unwrap();
        let raw_path = dir.join(format!("{stem}.json"));
        std::fs::create_dir(&raw_path).unwrap();
        let source_audio = library.root.join("rollback-source.wav");
        std::fs::write(&source_audio, b"audio that must be rolled back").unwrap();

        let error = save_raw(SaveRawArgs {
            stem: stem.to_string(),
            audio_filename: "rollback-source.wav".to_string(),
            source_audio: Some(source_audio.to_string_lossy().into_owned()),
            txt: "new raw text".to_string(),
            srt: "new raw srt".to_string(),
            json: r#"{"segments":[],"backend":"new","model_id":"new"}"#.to_string(),
            result: json!({
                "segments": [],
                "backend": "new",
                "model_id": "new"
            }),
        })
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("not a regular file"),
            "unexpected error: {error}"
        );
        assert!(raw_path.is_dir());
        assert!(!dir.join(format!("{stem}.txt")).exists());
        assert!(!dir.join(format!("{stem}.srt")).exists());
        assert_eq!(
            std::fs::read(dir.join("task.json")).unwrap(),
            old_metadata_bytes
        );
        let audio_dir = dir.join("audio");
        assert!(audio_dir.is_dir());
        assert_eq!(std::fs::read_dir(&audio_dir).unwrap().count(), 0);
        for entry in std::fs::read_dir(&dir).unwrap() {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            assert!(
                !name.ends_with(".tmp"),
                "left transaction temp file: {name}"
            );
        }
    }

    #[test]
    fn saving_asr_review_keeps_raw_json_bytes_unchanged() {
        let library = TestLibrary::new();
        let stem = "asr-review-raw-unchanged";
        let raw_path = write_raw_task(
            &library,
            stem,
            b"{\n  \"segments\": [],\n  \"backend\": \"test\",\n  \"model_id\": \"test-model\"\n}\n",
        );
        let before = std::fs::read(&raw_path).unwrap();

        save_asr_review(SaveAsrReviewArgs {
            stem: stem.to_string(),
            review: valid_review(),
        })
        .unwrap();

        assert_eq!(std::fs::read(raw_path).unwrap(), before);
    }

    #[test]
    fn load_task_loads_asr_human_review_sidecar() {
        let library = TestLibrary::new();
        let stem = "asr-review-load";
        write_raw_task(
            &library,
            stem,
            br#"{"segments":[],"backend":"test","model_id":"test-model"}"#,
        );
        let mut review = valid_review();
        review["reviewer"] = json!("human");
        review["extension"] = json!({"accepted": true});
        review["items"][0]["item_extension"] = json!(["preserved"]);
        save_asr_review(SaveAsrReviewArgs {
            stem: stem.to_string(),
            review: review.clone(),
        })
        .unwrap();

        let loaded = load_task(stem).unwrap();

        assert_eq!(loaded.asr_human_review, Some(review));
    }

    #[test]
    fn load_task_without_asr_human_review_is_backward_compatible() {
        let library = TestLibrary::new();
        let stem = "asr-review-legacy";
        write_raw_task(
            &library,
            stem,
            br#"{"segments":[],"backend":"test","model_id":"test-model"}"#,
        );

        let loaded = load_task(stem).unwrap();

        assert_eq!(loaded.asr_human_review, None);
    }

    #[test]
    fn saving_asr_review_twice_replaces_entire_sidecar() {
        let library = TestLibrary::new();
        let stem = "asr-review-replace";
        write_raw_task(
            &library,
            stem,
            br#"{"segments":[],"backend":"test","model_id":"model"}"#,
        );
        save_asr_review(SaveAsrReviewArgs {
            stem: stem.to_string(),
            review: json!({
                "schema_version": 1,
                "items": [{
                    "id": "old",
                    "start": 0,
                    "end": 1,
                    "status": "pending",
                    "obsolete": "this trailing data must disappear"
                }]
            }),
        })
        .unwrap();
        let replacement = valid_review();

        save_asr_review(SaveAsrReviewArgs {
            stem: stem.to_string(),
            review: replacement.clone(),
        })
        .unwrap();

        let sidecar_path = library.task_dir(stem).join("asr_human_review.json");
        assert_eq!(
            std::fs::read_to_string(sidecar_path).unwrap(),
            serde_json::to_string_pretty(&replacement).unwrap()
        );
    }

    #[test]
    fn corrupt_asr_human_review_sidecar_is_reported_clearly() {
        let library = TestLibrary::new();
        let stem = "asr-review-corrupt";
        write_raw_task(
            &library,
            stem,
            br#"{"segments":[],"backend":"test","model_id":"test-model"}"#,
        );
        std::fs::write(
            library.task_dir(stem).join("asr_human_review.json"),
            "{not valid json",
        )
        .unwrap();

        let error = load_task(stem).unwrap_err().to_string();

        assert!(error.contains("parse ASR human review sidecar"));
        assert!(error.contains("asr_human_review.json"));
    }

    #[test]
    fn asr_review_must_be_a_json_object() {
        let library = TestLibrary::new();
        let stem = "asr-review-object-only";

        let error = save_asr_review(SaveAsrReviewArgs {
            stem: stem.to_string(),
            review: json!(["not", "an", "object"]),
        })
        .unwrap_err()
        .to_string();

        assert!(error.contains("must be a JSON object"));
        assert!(!library
            .task_dir(stem)
            .join("asr_human_review.json")
            .exists());
    }

    #[test]
    fn asr_review_schema_validation_rejects_invalid_fields_on_save() {
        let library = TestLibrary::new();
        let stem = "asr-review-invalid-schema";
        let invalid_reviews = [
            (json!({"schema_version": 2, "items": []}), "schema_version"),
            (
                json!({"schema_version": 1, "items": {}}),
                "items must be an array",
            ),
            (
                json!({"schema_version": 1, "items": [{
                    "id": " ", "start": 0, "end": 1, "status": "pending"
                }]}),
                "id must not be empty",
            ),
            (
                json!({"schema_version": 1, "items": [
                    {"id": "duplicate", "start": 0, "end": 1, "status": "pending"},
                    {"id": "duplicate", "start": 1, "end": 2, "status": "resolved"}
                ]}),
                "id must be unique",
            ),
            (
                json!({"schema_version": 1, "items": [{
                    "id": "bad-range", "start": -1, "end": 1, "status": "pending"
                }]}),
                "0 <= start < end",
            ),
            (
                json!({"schema_version": 1, "items": [{
                    "id": "bad-status", "start": 0, "end": 1, "status": "approved"
                }]}),
                "status is invalid",
            ),
            (
                json!({"schema_version": 1, "items": [{
                    "id": "bad-text", "start": 0, "end": 1, "status": "pending",
                    "replacement_text": null
                }]}),
                "replacement_text must be a string",
            ),
            (
                json!({"schema_version": 1, "items": [{
                    "id": "bad-heard", "start": 0, "end": 1, "status": "pending",
                    "heard_text": 7
                }]}),
                "heard_text must be a string",
            ),
            (
                json!({"schema_version": 1, "items": [{
                    "id": "bad-note", "start": 0, "end": 1, "status": "pending",
                    "note": []
                }]}),
                "note must be a string",
            ),
        ];

        for (review, expected) in invalid_reviews {
            let error = save_asr_review(SaveAsrReviewArgs {
                stem: stem.to_string(),
                review,
            })
            .unwrap_err()
            .to_string();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
        assert!(!library
            .task_dir(stem)
            .join("asr_human_review.json")
            .exists());
    }

    #[test]
    fn asr_review_accepts_every_supported_status() {
        let library = TestLibrary::new();
        let stem = "asr-review-statuses";
        write_raw_task(
            &library,
            stem,
            br#"{"segments":[],"backend":"test","model_id":"model"}"#,
        );

        for status in super::ASR_REVIEW_STATUSES {
            let mut review = valid_review();
            review["items"][0]["status"] = json!(status);
            save_asr_review(SaveAsrReviewArgs {
                stem: stem.to_string(),
                review,
            })
            .unwrap();
        }

        assert!(library
            .task_dir(stem)
            .join("asr_human_review.json")
            .is_file());
    }

    #[test]
    fn load_task_validates_existing_asr_review_with_the_same_schema() {
        let library = TestLibrary::new();
        let stem = "asr-review-invalid-load";
        write_raw_task(
            &library,
            stem,
            br#"{"segments":[],"backend":"test","model_id":"test-model"}"#,
        );
        std::fs::write(
            library.task_dir(stem).join("asr_human_review.json"),
            r#"{"schema_version":1,"items":[{"id":"x","start":0,"end":1,"status":"approved"}]}"#,
        )
        .unwrap();

        let error = load_task(stem).unwrap_err().to_string();

        assert!(error.contains("validate ASR human review sidecar"));
        assert!(error.contains("asr_human_review.json"));
    }

    #[test]
    fn load_task_does_not_mistake_review_extensions_for_raw_json() {
        let library = TestLibrary::new();
        let stem = "missing-raw-with-rich-review";
        let dir = library.task_dir(stem);
        std::fs::create_dir_all(&dir).unwrap();
        save_meta(&SavedMeta {
            stem: stem.to_string(),
            audio_filename: "missing.wav".to_string(),
            created_at: 1,
            updated_at: 1,
            ..SavedMeta::default()
        })
        .unwrap();
        let mut review = valid_review();
        review["segments"] = json!([]);
        review["backend"] = json!("extension-backend");
        review["model_id"] = json!("extension-model");
        std::fs::write(
            dir.join("asr_human_review.json"),
            serde_json::to_string(&review).unwrap(),
        )
        .unwrap();

        let error = load_task(stem).unwrap_err().to_string();

        assert!(error.contains("library item not found"));
    }

    #[test]
    fn list_skips_stem_mismatch_without_rewriting_metadata() {
        let library = TestLibrary::new();
        let directory_stem = "mismatched-directory";
        let raw_path = write_raw_task(
            &library,
            directory_stem,
            br#"{"segments":[],"backend":"test","model_id":"model"}"#,
        );
        let metadata_path = library.task_dir(directory_stem).join("task.json");
        let mismatched_meta = SavedMeta {
            stem: "different-stem".to_string(),
            audio_filename: "source.wav".to_string(),
            raw_filename: format!("{directory_stem}.json"),
            created_at: 1,
            updated_at: 2,
            ..SavedMeta::default()
        };
        let metadata_bytes = serde_json::to_vec_pretty(&mismatched_meta).unwrap();
        std::fs::write(&metadata_path, &metadata_bytes).unwrap();
        let raw_bytes = std::fs::read(&raw_path).unwrap();

        assert!(list_library().unwrap().is_empty());
        assert_eq!(std::fs::read(metadata_path).unwrap(), metadata_bytes);
        assert_eq!(std::fs::read(raw_path).unwrap(), raw_bytes);
        let error = load_task(directory_stem).unwrap_err().to_string();
        assert!(error.contains("stem mismatch"), "unexpected error: {error}");
    }

    #[test]
    fn archive_only_updates_metadata_and_keeps_raw_bytes() {
        let library = TestLibrary::new();
        let stem = "archive-raw-unchanged";
        let raw_path = write_raw_task(
            &library,
            stem,
            b"{ \"audio\":\"old.wav\", \"segments\": [], \"backend\": \"test\", \"model_id\": \"model\" }\n",
        );
        let before = std::fs::read(&raw_path).unwrap();
        let audio_dir = library.task_dir(stem).join("audio");
        std::fs::create_dir_all(&audio_dir).unwrap();
        std::fs::write(audio_dir.join("stable.wav"), b"audio").unwrap();
        save_meta(&SavedMeta {
            stem: stem.to_string(),
            audio_filename: format!("{stem}.wav"),
            raw_filename: format!("{stem}.json"),
            audio_path: Some(audio_dir.join("stable.wav").to_string_lossy().into_owned()),
            created_at: 1,
            updated_at: 1,
            ..SavedMeta::default()
        })
        .unwrap();

        let archived_path = PathBuf::from(archive_task(stem).unwrap().unwrap());
        let archived_name = archived_path.file_name().unwrap().to_string_lossy();
        let archived_raw_path = archived_path.join(format!("{stem}.json"));
        let archived_meta: SavedMeta = serde_json::from_str(
            &std::fs::read_to_string(archived_path.join("task.json")).unwrap(),
        )
        .unwrap();

        assert_eq!(std::fs::read(archived_raw_path).unwrap(), before);
        assert_eq!(archived_meta.stem, archived_name);
        assert_eq!(archived_meta.raw_filename, format!("{stem}.json"));
        assert_eq!(
            load_task(&archived_name).unwrap().raw_json["backend"],
            "test"
        );
        assert!(archived_meta
            .audio_path
            .as_deref()
            .unwrap()
            .starts_with(archived_path.to_str().unwrap()));
    }

    #[test]
    fn metadata_unknown_fields_survive_all_saves_and_archive() {
        let library = TestLibrary::new();
        let stem = "metadata-extras-roundtrip";
        let dir = library.task_dir(stem);
        write_raw_task(
            &library,
            stem,
            br#"{"audio":"submitted.wav","segments":[],"backend":"old","model_id":"old"}"#,
        );
        let mut metadata = serde_json::to_value(SavedMeta {
            stem: stem.to_string(),
            audio_filename: "source.wav".to_string(),
            created_at: 1,
            updated_at: 1,
            ..SavedMeta::default()
        })
        .unwrap();
        metadata["future_config"] = json!({"nested": [1, {"enabled": true}]});
        metadata["future_flag"] = json!("preserve-me");
        std::fs::write(
            dir.join("task.json"),
            serde_json::to_vec_pretty(&metadata).unwrap(),
        )
        .unwrap();

        let loaded = load_existing_meta(stem).unwrap();
        assert_eq!(
            loaded.extras.get("future_config"),
            Some(&json!({"nested": [1, {"enabled": true}]}))
        );
        assert_eq!(
            loaded.extras.get("future_flag"),
            Some(&json!("preserve-me"))
        );

        save_raw(SaveRawArgs {
            stem: stem.to_string(),
            audio_filename: "source.wav".to_string(),
            source_audio: None,
            txt: "raw".to_string(),
            srt: "raw srt".to_string(),
            json: r#"{"audio":"submitted.wav","segments":[],"backend":"new","model_id":"new"}"#
                .to_string(),
            result: json!({
                "segments": [],
                "backend": "new",
                "model_id": "new"
            }),
        })
        .unwrap();
        save_corrected(SaveCorrectedArgs {
            stem: stem.to_string(),
            txt: "corrected".to_string(),
            srt: "corrected srt".to_string(),
            json: r#"{"corrected":true}"#.to_string(),
            diff: "diff".to_string(),
            model: "corrector".to_string(),
            changed: 1,
            total: 1,
            glossary: None,
        })
        .unwrap();
        save_polished(SavePolishedArgs {
            stem: stem.to_string(),
            text: "polished".to_string(),
            model: "polisher".to_string(),
            source: Some("corrected".to_string()),
        })
        .unwrap();

        let archived_path = PathBuf::from(archive_task(stem).unwrap().unwrap());
        let archived_meta: SavedMeta =
            serde_json::from_slice(&std::fs::read(archived_path.join("task.json")).unwrap())
                .unwrap();
        assert_eq!(
            archived_meta.extras.get("future_config"),
            Some(&json!({"nested": [1, {"enabled": true}]}))
        );
        assert_eq!(
            archived_meta.extras.get("future_flag"),
            Some(&json!("preserve-me"))
        );
    }

    #[test]
    fn list_skips_corrupt_task_but_explicit_load_reports_it() {
        let library = TestLibrary::new();
        let good_stem = "good-task";
        write_raw_task(
            &library,
            good_stem,
            br#"{"segments":[],"backend":"test","model_id":"model"}"#,
        );
        save_meta(&SavedMeta {
            stem: good_stem.to_string(),
            audio_filename: "good.wav".to_string(),
            created_at: 1,
            updated_at: 2,
            ..SavedMeta::default()
        })
        .unwrap();
        let corrupt_stem = "corrupt-task";
        let corrupt_dir = library.task_dir(corrupt_stem);
        std::fs::create_dir_all(&corrupt_dir).unwrap();
        std::fs::write(corrupt_dir.join("task.json"), "{not valid json").unwrap();

        let listed = list_library().unwrap();

        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].stem, good_stem);
        let error = load_task(corrupt_stem).unwrap_err().to_string();
        assert!(error.contains("parse metadata"));
    }

    #[test]
    fn corrupt_existing_metadata_is_reported_before_stage_files_change() {
        let library = TestLibrary::new();
        let stem = "corrupt-meta";
        let dir = library.task_dir(stem);
        std::fs::create_dir_all(&dir).unwrap();
        let metadata_path = dir.join("task.json");
        std::fs::write(&metadata_path, "{not valid json").unwrap();

        let error = load_existing_meta(stem).unwrap_err().to_string();
        assert!(error.contains("parse metadata"));
        let save_error = save_polished(SavePolishedArgs {
            stem: stem.to_string(),
            text: "must not be written".to_string(),
            model: "test-model".to_string(),
            source: Some("raw".to_string()),
        })
        .unwrap_err()
        .to_string();

        assert!(save_error.contains("parse metadata"));
        assert_eq!(
            std::fs::read_to_string(metadata_path).unwrap(),
            "{not valid json"
        );
        assert!(!dir.join(format!("{stem}_完整版.txt")).exists());
    }

    #[test]
    fn concurrent_corrected_and_polished_saves_preserve_both_flags() {
        let _library = TestLibrary::new();
        let stem = "concurrent-flags";
        save_meta(&SavedMeta {
            stem: stem.to_string(),
            audio_filename: "source.wav".to_string(),
            created_at: 1,
            updated_at: 1,
            ..SavedMeta::default()
        })
        .unwrap();

        let barrier = std::sync::Arc::new(Barrier::new(3));
        let corrected_barrier = barrier.clone();
        let corrected = std::thread::spawn(move || {
            corrected_barrier.wait();
            save_corrected(SaveCorrectedArgs {
                stem: stem.to_string(),
                txt: "corrected".to_string(),
                srt: "corrected srt".to_string(),
                json: "{\"corrected\":true}".to_string(),
                diff: "diff".to_string(),
                model: "corrector".to_string(),
                changed: 1,
                total: 1,
                glossary: None,
            })
        });
        let polished_barrier = barrier.clone();
        let polished = std::thread::spawn(move || {
            polished_barrier.wait();
            save_polished(SavePolishedArgs {
                stem: stem.to_string(),
                text: "polished".to_string(),
                model: "polisher".to_string(),
                source: Some("corrected".to_string()),
            })
        });

        barrier.wait();
        corrected.join().unwrap().unwrap();
        polished.join().unwrap().unwrap();

        let meta = load_existing_meta(stem).unwrap();
        assert!(meta.has_corrected);
        assert!(meta.has_polished);
        assert_eq!(meta.correction_model.as_deref(), Some("corrector"));
        assert_eq!(meta.polish_model.as_deref(), Some("polisher"));
    }
}
