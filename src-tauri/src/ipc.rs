//! JSON-RPC envelope types matching `scribe-py/src/scribe_py/ipc.py`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum IpcEnvelope {
    /// `{"id": 1, "result": ...}` or `{"id": 1, "error": {...}}`
    Response {
        id: u64,
        #[serde(default)]
        result: Option<Value>,
        #[serde(default)]
        error: Option<IpcError>,
    },
    /// `{"event": "progress", "method": "transcribe", "data": {...}}`
    Progress(ProgressEvent),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProgressEvent {
    pub event: String,
    pub method: String,
    pub data: Value,
}

#[derive(Debug, Deserialize)]
pub struct IpcError {
    #[allow(dead_code)]
    pub code: i32,
    pub message: String,
    #[allow(dead_code)]
    #[serde(default)]
    pub data: Option<Value>,
}
