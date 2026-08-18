//! Rust ↔ Python sidecar end-to-end smoke test.
//!
//! Run from the repo root:
//!   cd src-tauri && cargo run --example sidecar_smoke
//!
//! Validates: spawn → environment → check_model → probe_audio → 3 valid responses.

use anyhow::{anyhow, Result};
use localscribe_lib::sidecar::SidecarHandle;
use serde_json::{json, Value};
use std::path::PathBuf;

fn project_root() -> PathBuf {
    // examples runs with cwd = src-tauri/, so go one up.
    let cwd = std::env::current_dir().expect("cwd");
    cwd.parent().unwrap_or(&cwd).to_path_buf()
}

fn require_string(v: &Value, key: &str) -> Result<String> {
    v.get(key)
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("missing string field {key:?} in {v}"))
}

fn require_bool(v: &Value, key: &str) -> Result<bool> {
    v.get(key)
        .and_then(|x| x.as_bool())
        .ok_or_else(|| anyhow!("missing bool field {key:?} in {v}"))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("info"))
        .init();

    let root = project_root();
    let python = root.join(".venv/bin/python3");
    let scribe_py = root.join("scribe-py");
    println!("python   = {}", python.display());
    println!("scribe-py = {}", scribe_py.display());

    if !python.exists() {
        return Err(anyhow!("venv python not found at {}", python.display()));
    }

    let (handle, _progress_rx) = SidecarHandle::spawn(python, scribe_py).await?;
    println!("sidecar spawned ✅\n");

    // 1) environment
    let env = handle.call("environment", json!({})).await?;
    println!("[environment] {}", serde_json::to_string_pretty(&env)?);
    let backend = require_string(&env, "default_backend")?;
    let _ = require_bool(&env, "apple_silicon")?;
    let _ = require_string(&env, "default_model_id")?;

    // 2) check_model
    let model = handle
        .call("check_model", json!({"backend": "auto"}))
        .await?;
    println!("\n[check_model] {}", serde_json::to_string_pretty(&model)?);
    let exists = require_bool(&model, "exists")?;
    let model_id = require_string(&model, "model_id")?;

    // 3) probe_audio
    let audio = root.join("雅各书一章.m4a");
    let probe = handle
        .call("probe_audio", json!({"audio": audio.to_string_lossy()}))
        .await?;
    println!("\n[probe_audio] {}", serde_json::to_string_pretty(&probe)?);
    let duration = probe
        .get("duration")
        .and_then(|x| x.as_f64())
        .ok_or_else(|| anyhow!("probe_audio: duration missing"))?;

    println!("\n=== summary ===");
    println!("backend: {backend}");
    println!("model:   {model_id}  exists={exists}");
    println!("audio:   {} ({:.1}s)", audio.display(), duration);
    println!("\nall 3 sidecar methods responded with valid shape ✅");

    Ok(())
}
