//! LocalScribe Tauri backend library.
//!
//! Sidecar is **lazy-initialised** on first command call. Spawning the Python child
//! process during `applicationDidFinishLaunching` triggers a non-unwinding panic on
//! macOS Tahoe (26.x), so we defer it.

pub mod articles;
mod commands;
pub mod ipc;
pub mod library;
pub mod model_check;
pub mod secrets;
pub mod settings;
pub mod sidecar;

use std::path::{Path, PathBuf};
use tauri::Manager;
use tracing_subscriber::{fmt, EnvFilter};

use crate::sidecar::SidecarLazy;

/// Detect whether we're running from a bundled `.app/Contents/MacOS/<binary>`.
/// Returns `Some(Resources_dir)` if yes.
pub fn bundle_resources_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let macos_dir = exe.parent()?; // .../Contents/MacOS
    let contents = macos_dir.parent()?; // .../Contents
    if macos_dir.file_name()? != "MacOS" || contents.file_name()? != "Contents" {
        return None;
    }
    let resources = contents.join("Resources");
    if resources.is_dir() {
        Some(resources)
    } else {
        None
    }
}

/// Locate the Python interpreter and the scribe-py package directory.
///
/// Resolution order (first match wins):
///   1. **Bundled mode** — `Resources/python/bin/python3` + `Resources/scribe-py/`
///   2. `LOCALSCRIBE_DEV_ROOT` env var override
///   3. cwd's parent (works for `cargo run` / `tauri dev`)
///   4. exe's ancestors
///   5. Hardcoded fallback for dev convenience
fn resolve_python_paths() -> (PathBuf, PathBuf, Option<PathBuf>) {
    fn dev_ok(root: &Path) -> bool {
        root.join(".venv/bin/python3").exists() && root.join("scribe-py").exists()
    }
    // 1. Bundled .app
    if let Some(resources) = bundle_resources_dir() {
        let python = resources.join("python/bin/python3");
        let scribe = resources.join("scribe-py");
        if python.exists() && scribe.exists() {
            tracing::info!(
                "running in bundled mode (Resources={})",
                resources.display()
            );
            return (python, scribe, Some(resources));
        }
        tracing::warn!(
            "bundle Resources/ exists but python or scribe-py missing — falling back to dev mode"
        );
    }

    // 2. dev override
    if let Ok(p) = std::env::var("LOCALSCRIBE_DEV_ROOT") {
        let r = PathBuf::from(p);
        if dev_ok(&r) {
            return (r.join(".venv/bin/python3"), r.join("scribe-py"), None);
        }
    }
    // 3. cwd parent
    if let Some(p) = std::env::current_dir()
        .ok()
        .and_then(|c| c.parent().map(|p| p.to_path_buf()))
    {
        if dev_ok(&p) {
            return (p.join(".venv/bin/python3"), p.join("scribe-py"), None);
        }
    }
    // 4. exe ancestors
    if let Ok(exe) = std::env::current_exe() {
        let mut cur = exe.parent();
        while let Some(p) = cur {
            if dev_ok(p) {
                return (p.join(".venv/bin/python3"), p.join("scribe-py"), None);
            }
            cur = p.parent();
        }
    }
    // 5. dev hardcoded
    let hard = PathBuf::from("/Users/apple/gitCommit/SwarmPathAI/LocalScribe");
    (hard.join(".venv/bin/python3"), hard.join("scribe-py"), None)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(true)
        .init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            let app_handle = app.handle().clone();
            let (python, scribe_py_dir, resources) = resolve_python_paths();

            // If bundled, prepend Resources/bin to PATH so Python sees ffmpeg etc.
            if let Some(res) = resources.as_ref() {
                // The bundled Python runtime lives inside the signed App
                // resources.  Writing __pycache__ on first launch would alter
                // that sealed bundle and invalidate its signature.
                std::env::set_var("PYTHONDONTWRITEBYTECODE", "1");
                let bin = res.join("bin");
                if bin.is_dir() {
                    let cur = std::env::var("PATH").unwrap_or_default();
                    let new = format!("{}:{}", bin.display(), cur);
                    std::env::set_var("PATH", &new);
                    tracing::info!("PATH prepended with bundled bin: {}", bin.display());
                }
                std::env::set_var("LOCALSCRIBE_RESOURCES", res);
                let modelscope_cache = res.join("modelscope/hub");
                if modelscope_cache.is_dir() {
                    std::env::set_var("LOCALSCRIBE_MODELSCOPE_CACHE", &modelscope_cache);
                    std::env::set_var("MODELSCOPE_CACHE", &modelscope_cache);
                    tracing::info!(
                        "ModelScope cache set to bundled Resources: {}",
                        modelscope_cache.display()
                    );
                }
            }

            tracing::info!(
                "registering lazy sidecar (python={}, scribe_py_dir={})",
                python.display(),
                scribe_py_dir.display()
            );
            app.manage(SidecarLazy::new(app_handle, python, scribe_py_dir));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::environment,
            commands::check_model,
            commands::probe_audio,
            commands::asr_preflight_select,
            commands::diarize,
            commands::recommend_diarization,
            commands::extract_voice_embedding,
            commands::preflight_voiceprint_anchors,
            commands::reidentify_speakers,
            commands::transcribe,
            commands::asr_cancel,
            commands::correct_segments,
            commands::correct_pause,
            commands::correct_resume,
            commands::correct_cancel,
            commands::correct_status,
            commands::polish_article,
            commands::translate_article,
            commands::set_api_key,
            commands::has_api_key,
            commands::get_api_key,
            commands::delete_api_key,
            commands::load_settings,
            commands::save_settings,
            commands::check_model_cache,
            commands::reveal_models_dir,
            commands::open_url,
            commands::library_save_raw,
            commands::library_save_asr_review,
            commands::library_save_corrected,
            commands::library_save_raw_and_corrected,
            commands::library_save_polished,
            commands::library_list,
            commands::library_load,
            commands::library_delete,
            commands::library_archive,
            commands::library_root_path,
            commands::prepare_media_playback,
            commands::article_save,
            commands::article_list,
            commands::article_delete,
            commands::article_rename,
            commands::article_read,
            commands::articles_root_path,
        ])
        .run(tauri::generate_context!())
        .expect("error while running LocalScribe");
}
