//! Persistent app settings — `<app_data_dir>/settings.json`.
//!
//! Stores: model size, default language, output formats, output dir,
//! correction config (provider/base_url/model/mode — but NOT api_key, that's in keychain).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static SETTINGS_TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct Settings {
    pub model_id: String,
    pub backend: String, // "auto" | "mlx" | "ct2"
    pub language: String,
    pub asr_hotwords: String,
    pub asr_quality_mode: String,
    pub audio_preprocess: String,
    pub transcript_sync: String,
    pub output_formats: Vec<String>, // ["txt", "srt", "json"]
    pub output_dir: Option<String>,
    pub correction: CorrectionSettings,
    pub polish: PolishSettings,
    pub translation: TranslationSettings,
    pub diarization: DiarizationSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct LLMAdvanced {
    pub temperature: f64,
    pub max_tokens: u32,
    pub top_p: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct CorrectionSettings {
    pub enabled: bool,
    pub auto_pipeline: bool,
    pub provider: String,
    pub base_url: String,
    pub model: String,
    pub mode: String, // "light" | "medium" | "heavy"
    pub batch_size: u32,
    pub context_hint: String,
    pub use_glossary: bool,
    pub concurrency: u32,
    pub advanced: LLMAdvanced,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct PolishSettings {
    pub enabled: bool,
    pub model: String,
    pub advanced: LLMAdvanced,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct TranslationSettings {
    pub model: String,
    pub advanced: LLMAdvanced,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct DiarizationSettings {
    pub enabled: bool,
    pub n_speakers: u32,
    pub engine: String,
    pub speakers: Vec<SpeakerProfile>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SpeakerProfile {
    pub name: String,
    pub embedding: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub embeddings: Vec<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enrollment_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enrollment_ready: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub enrollment_reasons: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

fn is_reserved_speaker_profile_name(name: &str) -> bool {
    name.trim().to_ascii_uppercase().starts_with("SPEAKER_")
}

fn validate(settings: &Settings) -> Result<()> {
    if let Some(profile) = settings
        .diarization
        .speakers
        .iter()
        .find(|profile| is_reserved_speaker_profile_name(&profile.name))
    {
        anyhow::bail!(
            "SPEAKER_* is a per-recording placeholder and cannot be persisted as a voice profile: {}",
            profile.name
        );
    }
    Ok(())
}

impl Default for DiarizationSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            n_speakers: 0, // 0 = 自动检测(silhouette 扫描 K=2..8)
            engine: "auto".into(),
            speakers: Vec::new(),
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model_id: "iic/SenseVoiceSmall".into(),
            backend: "auto".into(),
            language: "zh".into(),
            asr_hotwords: String::new(),
            asr_quality_mode: "standard".into(),
            audio_preprocess: "adaptive".into(),
            transcript_sync: "precise".into(),
            output_formats: vec!["txt".into(), "srt".into(), "json".into()],
            output_dir: None,
            correction: CorrectionSettings::default(),
            polish: PolishSettings::default(),
            translation: TranslationSettings::default(),
            diarization: DiarizationSettings::default(),
        }
    }
}

impl Default for CorrectionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            auto_pipeline: false,
            provider: "deepseek".into(),
            base_url: "https://api.deepseek.com".into(),
            model: "deepseek-v4-flash".into(),
            mode: "medium".into(),
            batch_size: 30,
            context_hint: String::new(),
            use_glossary: true,
            concurrency: 15,
            advanced: LLMAdvanced {
                temperature: 0.1,
                max_tokens: 8192,
                top_p: 1.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
            },
        }
    }
}

impl Default for PolishSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "deepseek-v4-flash".into(),
            advanced: LLMAdvanced {
                temperature: 0.3,
                max_tokens: 384000,
                top_p: 1.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
            },
        }
    }
}

impl Default for TranslationSettings {
    fn default() -> Self {
        Self {
            model: "deepseek-v4-flash".into(),
            advanced: LLMAdvanced {
                temperature: 0.3,
                max_tokens: 384000,
                top_p: 1.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
            },
        }
    }
}

impl Default for LLMAdvanced {
    fn default() -> Self {
        Self {
            temperature: 0.1,
            max_tokens: 8192,
            top_p: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

pub fn load(path: &Path) -> Result<Settings> {
    if !path.exists() {
        return Ok(Settings::default());
    }
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read settings: {}", path.display()))?;
    let mut s: Settings = serde_json::from_str(&raw).with_context(|| "parse settings.json")?;
    let migrated = migrate(&mut s);
    if migrated {
        // 把迁移结果写回磁盘,下次启动直接读新值
        if let Err(e) = save(path, &s) {
            tracing::warn!("settings migration save failed: {e:#}");
        } else {
            tracing::info!("settings migrated to new defaults (concurrency/batch_size)");
        }
    }
    Ok(s)
}

/// 老版本默认值 → 新版本默认值的迁移。返回是否实际改了。
///
/// 仅在用户保留**旧默认值**时才迁移 — 如果用户曾手动改过(比如 concurrency=10),
/// 不动他的选择。
fn migrate(s: &mut Settings) -> bool {
    let mut changed = false;
    if matches!(s.backend.as_str(), "auto" | "sensevoice")
        && s.model_id == "mlx-community/whisper-large-v3-turbo"
    {
        s.model_id = "iic/SenseVoiceSmall".into();
        changed = true;
    }
    if s.correction.concurrency == 5 {
        s.correction.concurrency = 15;
        changed = true;
    }
    if s.correction.batch_size == 20 {
        s.correction.batch_size = 30;
        changed = true;
    }
    // v1.0.3: auto_preflight was an old default, but once AI denoise is bundled
    // it can spend extra time sampling denoise/enhance modes before transcription.
    // Keep the default customer path fast and fidelity-first; users can still
    // explicitly select auto_preflight from Settings when they want experiments.
    if s.audio_preprocess == "auto_preflight" {
        s.audio_preprocess = "adaptive".into();
        changed = true;
    }
    // v1.0 → v1.1:n_speakers=2 旧默认 → 0(自动)。仅当用户从未启用且声纹库为空时迁移。
    if !s.diarization.enabled && s.diarization.n_speakers == 2 && s.diarization.speakers.is_empty()
    {
        s.diarization.n_speakers = 0;
        changed = true;
    }
    let speaker_count = s.diarization.speakers.len();
    s.diarization
        .speakers
        .retain(|profile| !is_reserved_speaker_profile_name(&profile.name));
    if s.diarization.speakers.len() != speaker_count {
        changed = true;
    }
    changed
}

fn parent_directory(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn create_unique_temp_file(path: &Path) -> Result<(PathBuf, File)> {
    let parent = parent_directory(path);
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("settings path has no file name: {}", path.display()))?;

    for _ in 0..100 {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let counter = SETTINGS_TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
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
                return Err(error).with_context(|| {
                    format!("create temporary settings file for {}", path.display())
                })
            }
        }
    }

    anyhow::bail!(
        "could not create unique temporary settings file for {}",
        path.display()
    )
}

#[cfg(not(windows))]
fn replace_file(temp_path: &Path, path: &Path) -> io::Result<()> {
    std::fs::rename(temp_path, path)
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

#[cfg(unix)]
fn sync_parent_directory(path: &Path) -> io::Result<()> {
    File::open(parent_directory(path))?.sync_all()
}

#[cfg(not(unix))]
fn sync_parent_directory(_path: &Path) -> io::Result<()> {
    // MoveFileExW uses MOVEFILE_WRITE_THROUGH above. Rust's standard library
    // does not expose a portable directory fsync on non-Unix platforms.
    Ok(())
}

fn atomic_write_with<F>(path: &Path, writer: F) -> Result<()>
where
    F: FnOnce(&mut File) -> io::Result<()>,
{
    let parent = parent_directory(path);
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create settings directory: {}", parent.display()))?;
    let (temp_path, mut temp_file) = create_unique_temp_file(path)?;

    let write_result = writer(&mut temp_file)
        .and_then(|_| temp_file.flush())
        .and_then(|_| temp_file.sync_all());
    if let Err(error) = write_result {
        drop(temp_file);
        let _ = std::fs::remove_file(&temp_path);
        return Err(error)
            .with_context(|| format!("write temporary settings file for {}", path.display()));
    }
    drop(temp_file);

    if let Err(error) = replace_file(&temp_path, path) {
        let _ = std::fs::remove_file(&temp_path);
        return Err(error).with_context(|| format!("replace {} atomically", path.display()));
    }

    sync_parent_directory(path).with_context(|| {
        format!(
            "settings replacement committed but parent directory sync failed: {}",
            parent.display()
        )
    })?;
    Ok(())
}

pub fn save(path: &Path, settings: &Settings) -> Result<()> {
    validate(settings)?;
    let raw = serde_json::to_vec_pretty(settings).context("serialize settings")?;
    atomic_write_with(path, |file| file.write_all(&raw))
}

/// Resolve the settings file location: `<app_data_dir>/settings.json`.
/// `app_data` should be obtained from Tauri's path resolver in the command layer.
pub fn settings_path(app_data: &Path) -> PathBuf {
    app_data.join("settings.json")
}

#[cfg(test)]
mod tests {
    use super::{
        atomic_write_with, is_reserved_speaker_profile_name, save, Settings,
        SETTINGS_TEMP_FILE_COUNTER,
    };
    #[cfg(unix)]
    use std::io::Read;
    use std::io::{self, Write};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::Ordering;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "localscribe-settings-{label}-{}-{}",
                std::process::id(),
                SETTINGS_TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed),
            ));
            std::fs::create_dir_all(&path).expect("create settings test directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temporary_files_for(path: &Path) -> Vec<PathBuf> {
        let parent = path.parent().expect("test path parent");
        let prefix = format!(
            ".{}.",
            path.file_name()
                .expect("test path file name")
                .to_string_lossy()
        );
        std::fs::read_dir(parent)
            .expect("read settings test directory")
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|candidate| {
                let name = candidate
                    .file_name()
                    .map(|name| name.to_string_lossy())
                    .unwrap_or_default();
                name.starts_with(&prefix) && name.ends_with(".tmp")
            })
            .collect()
    }

    #[test]
    fn reserved_speaker_profile_names_cover_alpha_and_numeric_placeholders() {
        assert!(is_reserved_speaker_profile_name("SPEAKER_A"));
        assert!(is_reserved_speaker_profile_name("speaker_01"));
        assert!(!is_reserved_speaker_profile_name("张三"));
    }

    #[test]
    fn save_atomically_replaces_existing_settings_file() {
        let directory = TestDirectory::new("replace");
        let path = directory.path().join("settings.json");
        let old = "old settings contents that must remain readable through replacement";
        std::fs::write(&path, old).expect("write old settings");

        #[cfg(unix)]
        let mut old_handle = std::fs::File::open(&path).expect("open old settings handle");

        let mut settings = Settings::default();
        settings.language = "en".into();
        settings.asr_hotwords = "LocalScribe".into();
        let expected =
            serde_json::to_string_pretty(&settings).expect("serialize expected settings");

        save(&path, &settings).expect("atomically save settings");

        assert_eq!(
            std::fs::read_to_string(&path).expect("read replaced settings"),
            expected
        );
        #[cfg(unix)]
        {
            let mut still_old = String::new();
            old_handle
                .read_to_string(&mut still_old)
                .expect("read pre-replacement handle");
            assert_eq!(still_old, old);
        }
        assert!(temporary_files_for(&path).is_empty());
    }

    #[test]
    fn failed_atomic_write_preserves_existing_settings_file() {
        let directory = TestDirectory::new("write-failure");
        let path = directory.path().join("settings.json");
        let old = br#"{"language":"zh","old":true}"#;
        std::fs::write(&path, old).expect("write old settings");

        let result = atomic_write_with(&path, |file| {
            file.write_all(br#"{"language":"en","partial":"#)?;
            Err(io::Error::new(
                io::ErrorKind::Other,
                "injected settings write failure",
            ))
        });

        assert!(result.is_err());
        assert_eq!(std::fs::read(&path).expect("read preserved settings"), old);
        assert!(temporary_files_for(&path).is_empty());
    }
}
