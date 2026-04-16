//! Typed provider options for Cohere chat, embedding, and reranking.

use serde::{Deserialize, Serialize};

/// Thinking mode used by Cohere chat models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CohereThinkingType {
    Enabled,
    Disabled,
}

/// Typed chat reasoning configuration stored under `provider_options_map["cohere"]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CohereThinkingConfig {
    /// Thinking mode (`enabled` / `disabled`).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "type",
        alias = "thinking_type"
    )]
    pub thinking_type: Option<CohereThinkingType>,
    /// Maximum token budget available to the thinking phase.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "tokenBudget",
        alias = "token_budget"
    )]
    pub token_budget: Option<u32>,
}

impl CohereThinkingConfig {
    /// Create an empty thinking config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the thinking mode.
    pub const fn with_type(mut self, thinking_type: CohereThinkingType) -> Self {
        self.thinking_type = Some(thinking_type);
        self
    }

    /// Set the thinking token budget.
    pub const fn with_token_budget(mut self, token_budget: u32) -> Self {
        self.token_budget = Some(token_budget);
        self
    }
}

/// Typed chat options stored under `provider_options_map["cohere"]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CohereChatOptions {
    /// Optional thinking/reasoning configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<CohereThinkingConfig>,
}

impl CohereChatOptions {
    /// Create empty Cohere chat options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the thinking configuration.
    pub fn with_thinking(mut self, thinking: CohereThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }
}

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
    /// Optional output dimension for newer embedding models.
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

/// AI SDK-aligned alias for Cohere language-model options.
pub type CohereLanguageModelOptions = CohereChatOptions;

/// Deprecated AI SDK compatibility alias.
#[deprecated(
    since = "0.11.0-beta.6",
    note = "Use `CohereLanguageModelOptions` instead."
)]
pub type CohereChatModelOptions = CohereChatOptions;

/// AI SDK-aligned alias for Cohere embedding-model options.
pub type CohereEmbeddingModelOptions = CohereEmbeddingOptions;

/// AI SDK-aligned alias for Cohere reranking-model options.
pub type CohereRerankingModelOptions = CohereRerankOptions;

/// Deprecated AI SDK compatibility alias.
#[deprecated(
    since = "0.11.0-beta.6",
    note = "Use `CohereRerankingModelOptions` instead."
)]
pub type CohereRerankingOptions = CohereRerankOptions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_options_serde_matches_expected_shape() {
        let value = serde_json::to_value(
            CohereChatOptions::new().with_thinking(
                CohereThinkingConfig::new()
                    .with_type(CohereThinkingType::Enabled)
                    .with_token_budget(2048),
            ),
        )
        .expect("serialize options");

        assert_eq!(
            value,
            serde_json::json!({
                "thinking": {
                    "type": "enabled",
                    "tokenBudget": 2048
                }
            })
        );
    }

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
    #[allow(deprecated)]
    fn ai_sdk_style_aliases_resolve_to_same_types() {
        let language: CohereLanguageModelOptions = CohereChatOptions::new()
            .with_thinking(CohereThinkingConfig::new().with_type(CohereThinkingType::Enabled));
        let chat: CohereChatModelOptions = language.clone();
        let embedding: CohereEmbeddingModelOptions =
            CohereEmbeddingOptions::new().with_output_dimension(1024);
        let rerank: CohereRerankingModelOptions = CohereRerankOptions::new().with_priority(1);
        let reranking: CohereRerankingOptions = rerank.clone();

        assert_eq!(language.thinking, chat.thinking);
        assert_eq!(embedding.output_dimension, Some(1024));
        assert_eq!(rerank.priority, reranking.priority);
    }
}
