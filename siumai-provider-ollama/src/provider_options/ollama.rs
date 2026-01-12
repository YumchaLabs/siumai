//! Ollama provider options.
//!
//! These typed option structs are owned by the Ollama provider crate and are serialized into
//! `providerOptions["ollama"]` (Vercel-aligned open options map).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Ollama-specific options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OllamaOptions {
    /// Keep model loaded in memory for this duration.
    pub keep_alive: Option<String>,
    /// Use raw mode (bypass templating).
    pub raw: Option<bool>,
    /// Format for structured outputs.
    pub format: Option<String>,
    /// Additional Ollama-specific parameters.
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl OllamaOptions {
    /// Create new Ollama options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set keep alive duration.
    pub fn with_keep_alive(mut self, duration: impl Into<String>) -> Self {
        self.keep_alive = Some(duration.into());
        self
    }

    /// Enable raw mode.
    pub fn with_raw_mode(mut self, raw: bool) -> Self {
        self.raw = Some(raw);
        self
    }

    /// Set output format.
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Add a custom parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}
