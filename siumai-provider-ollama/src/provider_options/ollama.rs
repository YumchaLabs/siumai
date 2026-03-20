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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    /// Use raw mode (bypass templating).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<bool>,
    /// Format for structured outputs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Additional Ollama-specific parameters.
    #[serde(flatten, skip_serializing_if = "HashMap::is_empty")]
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

pub(crate) fn parse_ollama_options_value(value: serde_json::Value) -> Option<OllamaOptions> {
    let mut options = serde_json::from_value::<OllamaOptions>(value).ok()?;

    if let Some(nested_extra_params) = options.extra_params.remove("extra_params") {
        if let Some(map) = nested_extra_params.as_object() {
            for (key, value) in map {
                options
                    .extra_params
                    .entry(key.clone())
                    .or_insert_with(|| value.clone());
            }
        } else {
            options
                .extra_params
                .entry("extra_params".to_string())
                .or_insert(nested_extra_params);
        }
    }

    Some(options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_options_serialize_sparse_fields() {
        let value = serde_json::to_value(
            OllamaOptions::new()
                .with_keep_alive("1m")
                .with_param("think", serde_json::json!(true)),
        )
        .expect("serialize options");

        assert_eq!(value["keep_alive"], serde_json::json!("1m"));
        assert_eq!(value["think"], serde_json::json!(true));
        assert!(value.get("raw").is_none());
        assert!(value.get("format").is_none());
    }

    #[test]
    fn ollama_options_parse_legacy_nested_extra_params() {
        let value = serde_json::json!({
            "keep_alive": "1m",
            "extra_params": {
                "think": true,
                "top_k": 40
            }
        });

        let parsed = parse_ollama_options_value(value).expect("parse legacy options");

        assert_eq!(parsed.keep_alive.as_deref(), Some("1m"));
        assert_eq!(
            parsed.extra_params.get("think"),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            parsed.extra_params.get("top_k"),
            Some(&serde_json::json!(40))
        );
        assert!(!parsed.extra_params.contains_key("extra_params"));
    }
}
