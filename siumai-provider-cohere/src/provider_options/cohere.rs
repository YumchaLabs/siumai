//! Typed provider options for Cohere reranking.

use serde::{Deserialize, Serialize};

/// Typed options stored under `provider_options_map["cohere"]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CohereRerankOptions {
    /// Maximum tokens per document.
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxTokensPerDoc")]
    pub max_tokens_per_doc: Option<u32>,
    /// Request priority hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<u32>,
}

impl CohereRerankOptions {
    /// Create empty Cohere rerank options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `maxTokensPerDoc`.
    pub const fn with_max_tokens_per_doc(mut self, max_tokens_per_doc: u32) -> Self {
        self.max_tokens_per_doc = Some(max_tokens_per_doc);
        self
    }

    /// Set `priority`.
    pub const fn with_priority(mut self, priority: u32) -> Self {
        self.priority = Some(priority);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_matches_expected_shape() {
        let value = serde_json::to_value(
            CohereRerankOptions::new()
                .with_max_tokens_per_doc(1000)
                .with_priority(1),
        )
        .expect("serialize options");

        assert_eq!(
            value,
            serde_json::json!({
                "maxTokensPerDoc": 1000,
                "priority": 1
            })
        );
    }
}
