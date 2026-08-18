//! Whisper model detection.
//!
//! 解析顺序(与 Python 端 ipc.handle_check_model / transcriber_mlx._resolve_model_path 保持一致):
//!   1. $LOCALSCRIBE_MODEL_DIR
//!   2. <project_root>/models/<basename>/weights.safetensors
//!   3. HF cache: ~/.cache/huggingface/hub/models--<org>--<name>/

use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::library;

pub fn hf_cache_root() -> PathBuf {
    if let Ok(p) = std::env::var("HF_HOME") {
        return PathBuf::from(p).join("hub");
    }
    if let Some(home) = dirs::home_dir() {
        return home.join(".cache").join("huggingface").join("hub");
    }
    PathBuf::from(".cache/huggingface/hub")
}

pub fn cache_dir_for(model_id: &str) -> PathBuf {
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    hf_cache_root().join(dir_name)
}

/// Recommended location for placing model weights.
///
/// - Bundled mode → `<.app>/Contents/Resources/models/<basename>/`
///   (read-only after install; build script puts them here)
/// - Dev mode    → `<repo>/models/<basename>/`
pub fn project_models_dir(model_id: &str) -> PathBuf {
    let basename = model_id.rsplit('/').next().unwrap_or(model_id);
    if let Some(resources) = crate::bundle_resources_dir() {
        return resources.join("models").join(basename);
    }
    if let Some(dev) = library::dev_project_root() {
        return dev.join("models").join(basename);
    }
    // last resort: user_data_root (so reveal_models_dir still has somewhere to open)
    library::user_data_root().join("models").join(basename)
}

fn weights_present(dir: &Path) -> bool {
    dir.join("weights.safetensors").exists()
}

#[derive(Debug, serde::Serialize)]
pub struct ModelStatus {
    pub model_id: String,
    pub exists: bool,
    pub source: Option<String>,
    pub path: Option<String>,
    /// 推荐用户放置文件的目标路径(便于前端引导)
    pub expected_local_path: String,
}

pub fn check(model_id: &str) -> Result<ModelStatus> {
    let project_path = project_models_dir(model_id);
    let expected = project_path.to_string_lossy().into_owned();

    // 1. env override
    if let Ok(env_dir) = std::env::var("LOCALSCRIBE_MODEL_DIR") {
        let p = PathBuf::from(&env_dir);
        let candidate = if weights_present(&p) {
            Some(p.clone())
        } else {
            let basename = model_id.rsplit('/').next().unwrap_or(model_id);
            let nested = p.join(basename);
            if weights_present(&nested) {
                Some(nested)
            } else {
                None
            }
        };
        if let Some(found) = candidate {
            return Ok(ModelStatus {
                model_id: model_id.to_string(),
                exists: true,
                source: Some("env".to_string()),
                path: Some(found.to_string_lossy().into_owned()),
                expected_local_path: expected,
            });
        }
    }

    // 2. project models/
    if weights_present(&project_path) {
        return Ok(ModelStatus {
            model_id: model_id.to_string(),
            exists: true,
            source: Some("project".to_string()),
            path: Some(project_path.to_string_lossy().into_owned()),
            expected_local_path: expected,
        });
    }

    // 3. HF cache
    let cache = cache_dir_for(model_id);
    if cache.exists() {
        return Ok(ModelStatus {
            model_id: model_id.to_string(),
            exists: true,
            source: Some("hf_cache".to_string()),
            path: Some(cache.to_string_lossy().into_owned()),
            expected_local_path: expected,
        });
    }

    Ok(ModelStatus {
        model_id: model_id.to_string(),
        exists: false,
        source: None,
        path: None,
        expected_local_path: expected,
    })
}
