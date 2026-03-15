//! Typed provider options for TogetherAI reranking.

use serde::{Deserialize, Serialize};

/// Typed options stored under `provider_options_map["togetherai"]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TogetherAiRerankOptions {
    /// Rerank fields used for structured documents.
    #[serde(skip_serializing_if = "Option::is_none", rename = "rankFields")]
    pub rank_fields: Option<Vec<String>>,
}

impl TogetherAiRerankOptions {
    /// Create empty TogetherAI rerank options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `rankFields`.
    pub fn with_rank_fields(mut self, rank_fields: Vec<String>) -> Self {
        self.rank_fields = Some(rank_fields);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_matches_expected_shape() {
        let value = serde_json::to_value(
            TogetherAiRerankOptions::new().with_rank_fields(vec!["example".to_string()]),
        )
        .expect("serialize options");

        assert_eq!(
            value,
            serde_json::json!({
                "rankFields": ["example"]
            })
        );
    }
}
