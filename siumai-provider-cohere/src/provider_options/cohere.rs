//! Typed provider options for Cohere embedding and reranking.

use serde::{Deserialize, Serialize};
use siumai_core::{ModelFamily, ProviderOptionError, TypedProviderOptions};

const VALID_OUTPUT_DIMENSIONS: &[u32] = &[256, 512, 1024, 1536];

/// Input type used by Cohere embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CohereEmbeddingInputType {
    SearchDocument,
    SearchQuery,
    Classification,
    Clustering,
}

/// Truncation strategy used by Cohere embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CohereEmbeddingTruncate {
    None,
    Start,
    End,
}

/// Typed embedding options stored under `provider_options_map["cohere"]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CohereEmbeddingOptions {
    /// Input type hint for the embedding request.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "inputType",
        alias = "input_type"
    )]
    pub input_type: Option<CohereEmbeddingInputType>,
    /// Truncation strategy for oversized inputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate: Option<CohereEmbeddingTruncate>,
    /// Optional output dimension for Embed v4 models.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "outputDimension",
        alias = "output_dimension"
    )]
    pub output_dimension: Option<u32>,
}

impl CohereEmbeddingOptions {
    /// Create empty Cohere embedding options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the embedding input type.
    pub const fn with_input_type(mut self, input_type: CohereEmbeddingInputType) -> Self {
        self.input_type = Some(input_type);
        self
    }

    /// Set the truncation strategy.
    pub const fn with_truncate(mut self, truncate: CohereEmbeddingTruncate) -> Self {
        self.truncate = Some(truncate);
        self
    }

    /// Set the output dimension.
    pub const fn with_output_dimension(mut self, output_dimension: u32) -> Self {
        self.output_dimension = Some(output_dimension);
        self
    }
}

impl TypedProviderOptions for CohereEmbeddingOptions {
    const NAMESPACE: &'static str = "cohere";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Embedding;
    const API_MODE: Option<&'static str> = Some("v2");

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if let Some(output_dimension) = self.output_dimension
            && !VALID_OUTPUT_DIMENSIONS.contains(&output_dimension)
        {
            return Err(ProviderOptionError::Rejected {
                path: "outputDimension".to_string(),
                reason: "must be one of 256, 512, 1024, or 1536".to_string(),
            });
        }
        Ok(())
    }
}

/// Typed rerank options stored under `provider_options_map["cohere"]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CohereRerankOptions {
    /// Maximum tokens per document.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "maxTokensPerDoc",
        alias = "max_tokens_per_doc"
    )]
    pub max_tokens_per_doc: Option<u32>,
    /// Request priority hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
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

impl TypedProviderOptions for CohereRerankOptions {
    const NAMESPACE: &'static str = "cohere";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Rerank;
    const API_MODE: Option<&'static str> = Some("v2");

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if self.max_tokens_per_doc == Some(0) {
            return Err(ProviderOptionError::Rejected {
                path: "maxTokensPerDoc".to_string(),
                reason: "must be greater than zero".to_string(),
            });
        }
        if self.priority.is_some_and(|priority| priority > 999) {
            return Err(ProviderOptionError::Rejected {
                path: "priority".to_string(),
                reason: "must be between 0 and 999".to_string(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::ProviderOptions;

    #[test]
    fn embedding_options_serde_matches_expected_shape() {
        let value = serde_json::to_value(
            CohereEmbeddingOptions::new()
                .with_input_type(CohereEmbeddingInputType::SearchDocument)
                .with_truncate(CohereEmbeddingTruncate::End)
                .with_output_dimension(1024),
        )
        .expect("serialize options");

        assert_eq!(
            value,
            serde_json::json!({
                "inputType": "search_document",
                "truncate": "END",
                "outputDimension": 1024
            })
        );
    }

    #[test]
    fn rerank_options_serde_matches_expected_shape() {
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

    #[test]
    fn canonical_provider_options_reject_invalid_numeric_values() {
        assert!(
            ProviderOptions::typed(&CohereEmbeddingOptions::new().with_output_dimension(2048))
                .is_err()
        );
        assert!(
            ProviderOptions::typed(&CohereRerankOptions::new().with_max_tokens_per_doc(0)).is_err()
        );
        assert!(ProviderOptions::typed(&CohereRerankOptions::new().with_priority(999)).is_ok());
        assert!(ProviderOptions::typed(&CohereRerankOptions::new().with_priority(1000)).is_err());
    }
}
