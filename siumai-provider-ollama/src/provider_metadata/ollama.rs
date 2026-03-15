//! `Ollama`-specific response metadata.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Ollama-specific metadata from chat responses.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OllamaMetadata {
    /// Output token generation rate, when Ollama exposes evaluation timing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f64>,

    /// Total request duration in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration_ms: Option<u64>,

    /// Model load duration in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration_ms: Option<u64>,

    /// Prompt evaluation duration in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration_ms: Option<u64>,

    /// Completion evaluation duration in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration_ms: Option<u64>,

    /// Preserve unknown provider-specific metadata for forward compatibility.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl crate::types::provider_metadata::FromMetadata for OllamaMetadata {
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self> {
        serde_json::from_value(serde_json::to_value(metadata).ok()?).ok()
    }
}

/// Typed helper for Ollama metadata extraction from `ChatResponse`.
pub trait OllamaChatResponseExt {
    fn ollama_metadata(&self) -> Option<OllamaMetadata>;
}

impl OllamaChatResponseExt for crate::types::ChatResponse {
    fn ollama_metadata(&self) -> Option<OllamaMetadata> {
        use crate::types::provider_metadata::FromMetadata;
        let meta = self.provider_metadata.as_ref()?.get("ollama")?;
        OllamaMetadata::from_metadata(meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_metadata_parses_timing_fields() {
        let mut resp = crate::types::ChatResponse::new(crate::types::MessageContent::Text(
            "hello".to_string(),
        ));

        let mut inner = HashMap::new();
        inner.insert("tokens_per_second".to_string(), serde_json::json!(28.5));
        inner.insert("total_duration_ms".to_string(), serde_json::json!(1250));
        inner.insert("load_duration_ms".to_string(), serde_json::json!(150));
        inner.insert(
            "prompt_eval_duration_ms".to_string(),
            serde_json::json!(200),
        );
        inner.insert("eval_duration_ms".to_string(), serde_json::json!(700));
        inner.insert("vendor_extra".to_string(), serde_json::json!(true));

        let mut outer = HashMap::new();
        outer.insert("ollama".to_string(), inner);
        resp.provider_metadata = Some(outer);

        let meta = resp.ollama_metadata().expect("ollama metadata");
        assert_eq!(meta.total_duration_ms, Some(1250));
        assert_eq!(meta.load_duration_ms, Some(150));
        assert_eq!(meta.prompt_eval_duration_ms, Some(200));
        assert_eq!(meta.eval_duration_ms, Some(700));
        assert_eq!(meta.tokens_per_second, Some(28.5));
        assert_eq!(
            meta.extra.get("vendor_extra"),
            Some(&serde_json::json!(true))
        );
    }
}
