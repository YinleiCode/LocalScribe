//! API-key storage.
//!
//! Two-tier resolution to avoid macOS keychain ACL prompts on each `cargo run`:
//!
//! 1. **Project-local dev file**: `<LocalScribe>/.dev-secrets.json` (plain
//!    text, 0600). Used when keychain fails or returns empty. Written by
//!    `set_api_key`.
//! 2. **System keychain**: `localscribe.<provider>` (Mac Keychain / Win Cred /
//!    libsecret) — the canonical store for release builds.
//!
//! For dev rebuilds, keychain ACLs reset on every new unsigned binary, so the
//! file fallback gives a no-prompt experience locally. Production builds with a
//! stable code signature should use keychain only.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

const SERVICE: &str = "ai.swarmpath.localscribe";

fn entry_for(provider: &str) -> Result<keyring::Entry> {
    let user = format!("api-key.{provider}");
    keyring::Entry::new(SERVICE, &user)
        .with_context(|| format!("failed to open keyring entry for provider={provider}"))
}

// ---- file fallback ----

#[derive(Default, Serialize, Deserialize)]
struct DevKeyStore {
    keys: HashMap<String, String>,
}

/// Resolve project root — identical contract to `library::project_root`.
fn project_root() -> PathBuf {
    crate::library::project_root()
}

fn dev_store_path() -> PathBuf {
    project_root().join(".dev-secrets.json")
}

fn read_dev_store() -> DevKeyStore {
    let p = dev_store_path();
    let raw = match std::fs::read_to_string(&p) {
        Ok(s) => s,
        Err(_) => return DevKeyStore::default(),
    };
    serde_json::from_str(&raw).unwrap_or_default()
}

fn write_dev_store(store: &DevKeyStore) -> Result<()> {
    let p = dev_store_path();
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let raw = serde_json::to_string_pretty(store)?;
    std::fs::write(&p, raw).with_context(|| format!("write {}", p.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

// ---- public API ----

pub fn set_api_key(provider: &str, key: &str) -> Result<()> {
    // Always write to dev file for instant no-prompt access.
    let mut store = read_dev_store();
    store.keys.insert(provider.to_string(), key.to_string());
    let _ = write_dev_store(&store);

    // Also try keychain (best-effort; ignore errors so dev never blocks).
    if let Ok(entry) = entry_for(provider) {
        let _ = entry.set_password(key);
    }
    Ok(())
}

pub fn get_api_key(provider: &str) -> Result<Option<String>> {
    // 1) try dev file first — no prompts, instant.
    let store = read_dev_store();
    if let Some(k) = store.keys.get(provider) {
        if !k.is_empty() {
            return Ok(Some(k.clone()));
        }
    }
    // 2) fall back to system keychain.
    match entry_for(provider)?.get_password() {
        Ok(v) => Ok(Some(v)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => {
            tracing::warn!("keychain get failed: {e:#}");
            Ok(None)
        }
    }
}

pub fn delete_api_key(provider: &str) -> Result<()> {
    // Remove from both stores.
    let mut store = read_dev_store();
    store.keys.remove(provider);
    let _ = write_dev_store(&store);

    if let Ok(entry) = entry_for(provider) {
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => {}
            Err(e) => tracing::warn!("keychain delete failed: {e:#}"),
        }
    }
    Ok(())
}
