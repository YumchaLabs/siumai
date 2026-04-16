//! Typed provider options for Amazon Bedrock.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bedrock prompt-cache TTL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BedrockCacheTtl {
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

/// Bedrock cache-point kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BedrockCachePointType {
    Default,
}

/// Inner Bedrock cache-point payload stored under `providerOptions.bedrock.cachePoint`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BedrockCachePointConfig {
    /// Cache-point kind.
    #[serde(rename = "type")]
    pub r#type: BedrockCachePointType,
    /// Optional cache TTL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<BedrockCacheTtl>,
}

impl Default for BedrockCachePointConfig {
    fn default() -> Self {
        Self {
            r#type: BedrockCachePointType::Default,
            ttl: None,
        }
    }
}

impl BedrockCachePointConfig {
    /// Create the default Bedrock cache-point config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the cache TTL.
    pub fn with_ttl(mut self, ttl: BedrockCacheTtl) -> Self {
        self.ttl = Some(ttl);
        self
    }
}

/// Bedrock cache-point envelope aligned with the AI SDK Bedrock provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BedrockCachePoint {
    /// Cache-point payload.
    #[serde(rename = "cachePoint", alias = "cache_point")]
    pub cache_point: BedrockCachePointConfig,
}

impl Default for BedrockCachePoint {
    fn default() -> Self {
        Self {
            cache_point: BedrockCachePointConfig::default(),
        }
    }
}

impl BedrockCachePoint {
    /// Create the default Bedrock cache-point envelope.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the cache TTL.
    pub fn with_ttl(mut self, ttl: BedrockCacheTtl) -> Self {
        self.cache_point.ttl = Some(ttl);
        self
    }
}

/// Bedrock reasoning mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BedrockReasoningType {
    Enabled,
    Disabled,
    Adaptive,
}

/// Bedrock reasoning effort hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BedrockReasoningEffort {
    Low,
    Medium,
    High,
    Max,
}

/// Bedrock reasoning configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BedrockReasoningConfig {
    /// Bedrock reasoning mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<BedrockReasoningType>,
    /// Anthropic thinking budget in tokens.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "budgetTokens",
        alias = "budget_tokens"
    )]
    pub budget_tokens: Option<u32>,
    /// Provider-specific reasoning effort hint.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "maxReasoningEffort",
        alias = "max_reasoning_effort"
    )]
    pub max_reasoning_effort: Option<BedrockReasoningEffort>,
}

impl BedrockReasoningConfig {
    /// Create empty Bedrock reasoning config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the reasoning mode.
    pub fn with_type(mut self, reasoning_type: BedrockReasoningType) -> Self {
        self.r#type = Some(reasoning_type);
        self
    }

    /// Set the reasoning budget.
    pub fn with_budget_tokens(mut self, budget_tokens: u32) -> Self {
        self.budget_tokens = Some(budget_tokens);
        self
    }

    /// Set the maximum reasoning effort.
    pub fn with_max_reasoning_effort(
        mut self,
        max_reasoning_effort: BedrockReasoningEffort,
    ) -> Self {
        self.max_reasoning_effort = Some(max_reasoning_effort);
        self
    }
}

/// Bedrock request service tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BedrockServiceTier {
    Reserved,
    Priority,
    Default,
    Flex,
}

/// Bedrock document citation toggle for file parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BedrockFilePartCitations {
    /// Whether citations are enabled for this document part.
    pub enabled: bool,
}

/// Typed provider options stored under `ContentPart::File.provider_options["bedrock"]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BedrockFilePartProviderOptions {
    /// Citation configuration for this file/document part.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<BedrockFilePartCitations>,
    /// Preserve unknown Bedrock file-part passthrough fields.
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl BedrockFilePartProviderOptions {
    /// Create empty Bedrock file-part provider options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable Bedrock document citations.
    pub fn with_citations(mut self, enabled: bool) -> Self {
        self.citations = Some(BedrockFilePartCitations { enabled });
        self
    }

    /// Add a custom Bedrock file-part passthrough field.
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}

/// Typed chat options stored under `provider_options_map["bedrock"]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BedrockChatOptions {
    /// Additional provider-native inference fields passed through to Bedrock.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "additionalModelRequestFields",
        alias = "additional_model_request_fields"
    )]
    pub additional_model_request_fields: Option<serde_json::Value>,
    /// Typed Bedrock reasoning configuration.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "reasoningConfig",
        alias = "reasoning_config"
    )]
    pub reasoning_config: Option<BedrockReasoningConfig>,
    /// Anthropic beta feature flags routed through Bedrock request fields.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "anthropicBeta",
        alias = "anthropic_beta"
    )]
    pub anthropic_beta: Option<Vec<String>>,
    /// Bedrock service tier routed to the top-level request body.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "serviceTier",
        alias = "service_tier"
    )]
    pub service_tier: Option<BedrockServiceTier>,
    /// Preserve unknown top-level Bedrock passthrough fields like the upstream AI SDK.
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl BedrockChatOptions {
    /// Create empty Bedrock chat options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `additionalModelRequestFields`.
    pub fn with_additional_model_request_fields(
        mut self,
        additional_model_request_fields: serde_json::Value,
    ) -> Self {
        self.additional_model_request_fields = Some(additional_model_request_fields);
        self
    }

    /// Set `reasoningConfig`.
    pub fn with_reasoning_config(mut self, reasoning_config: BedrockReasoningConfig) -> Self {
        self.reasoning_config = Some(reasoning_config);
        self
    }

    /// Set `anthropicBeta`.
    pub fn with_anthropic_beta<I, S>(mut self, anthropic_beta: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let values = anthropic_beta
            .into_iter()
            .map(Into::into)
            .collect::<Vec<_>>();
        self.anthropic_beta = if values.is_empty() {
            None
        } else {
            Some(values)
        };
        self
    }

    /// Set `serviceTier`.
    pub fn with_service_tier(mut self, service_tier: BedrockServiceTier) -> Self {
        self.service_tier = Some(service_tier);
        self
    }

    /// Add a custom Bedrock request field passthrough.
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}

/// AI SDK-style alias for Bedrock language-model options.
pub type AmazonBedrockLanguageModelOptions = BedrockChatOptions;

/// Deprecated AI SDK compatibility alias for Bedrock chat options.
#[deprecated(note = "Use `AmazonBedrockLanguageModelOptions` instead.")]
pub type BedrockProviderOptions = AmazonBedrockLanguageModelOptions;

/// Cohere-on-Bedrock embedding input type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BedrockEmbeddingInputType {
    SearchDocument,
    SearchQuery,
    Classification,
    Clustering,
}

/// Nova embedding purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BedrockEmbeddingPurpose {
    GenericIndex,
    TextRetrieval,
    ImageRetrieval,
    VideoRetrieval,
    DocumentRetrieval,
    AudioRetrieval,
    GenericRetrieval,
    Classification,
    Clustering,
}

/// Truncation behavior for Nova and Cohere embedding models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BedrockEmbeddingTruncate {
    None,
    Start,
    End,
}

/// Typed embedding options stored under `provider_options_map["bedrock"]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BedrockEmbeddingOptions {
    /// Titan embedding dimensions. Supported values: `1024`, `512`, `256`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// Titan normalization toggle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,
    /// Nova embedding dimensions. Supported values: `256`, `384`, `1024`, `3072`.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "embeddingDimension",
        alias = "embedding_dimension"
    )]
    pub embedding_dimension: Option<u32>,
    /// Nova embedding purpose.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "embeddingPurpose",
        alias = "embedding_purpose"
    )]
    pub embedding_purpose: Option<BedrockEmbeddingPurpose>,
    /// Cohere embedding input type.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "inputType",
        alias = "input_type"
    )]
    pub input_type: Option<BedrockEmbeddingInputType>,
    /// Truncation behavior for Nova and Cohere embeddings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<BedrockEmbeddingTruncate>,
    /// Cohere output dimension. Supported values: `256`, `512`, `1024`, `1536`.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "outputDimension",
        alias = "output_dimension"
    )]
    pub output_dimension: Option<u32>,
}

impl BedrockEmbeddingOptions {
    /// Create empty Bedrock embedding options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set Titan dimensions.
    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Set Titan normalization.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = Some(normalize);
        self
    }

    /// Set Nova embedding dimensions.
    pub fn with_embedding_dimension(mut self, embedding_dimension: u32) -> Self {
        self.embedding_dimension = Some(embedding_dimension);
        self
    }

    /// Set Nova embedding purpose.
    pub fn with_embedding_purpose(mut self, embedding_purpose: BedrockEmbeddingPurpose) -> Self {
        self.embedding_purpose = Some(embedding_purpose);
        self
    }

    /// Set Cohere embedding input type.
    pub fn with_input_type(mut self, input_type: BedrockEmbeddingInputType) -> Self {
        self.input_type = Some(input_type);
        self
    }

    /// Set Nova/Cohere truncation behavior.
    pub fn with_truncate(mut self, truncate: BedrockEmbeddingTruncate) -> Self {
        self.truncate = Some(truncate);
        self
    }

    /// Set Cohere output dimension.
    pub fn with_output_dimension(mut self, output_dimension: u32) -> Self {
        self.output_dimension = Some(output_dimension);
        self
    }
}

/// AI SDK-style alias for Bedrock embedding options.
pub type AmazonBedrockEmbeddingModelOptions = BedrockEmbeddingOptions;

/// Typed rerank options stored under `provider_options_map["bedrock"]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BedrockRerankOptions {
    /// AWS region used to build the foundation model ARN.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Pagination token.
    #[serde(skip_serializing_if = "Option::is_none", rename = "nextToken")]
    pub next_token: Option<String>,
    /// Additional provider-native model fields passed through to Bedrock.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "additionalModelRequestFields"
    )]
    pub additional_model_request_fields: Option<serde_json::Value>,
}

impl BedrockRerankOptions {
    /// Create empty Bedrock rerank options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `region`.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set `nextToken`.
    pub fn with_next_token(mut self, next_token: impl Into<String>) -> Self {
        self.next_token = Some(next_token.into());
        self
    }

    /// Set `additionalModelRequestFields`.
    pub fn with_additional_model_request_fields(
        mut self,
        additional_model_request_fields: serde_json::Value,
    ) -> Self {
        self.additional_model_request_fields = Some(additional_model_request_fields);
        self
    }
}

/// AI SDK-style alias for Bedrock reranking options.
pub type AmazonBedrockRerankingModelOptions = BedrockRerankOptions;

/// Deprecated AI SDK compatibility alias for Bedrock reranking options.
#[deprecated(note = "Use `AmazonBedrockRerankingModelOptions` instead.")]
pub type BedrockRerankingOptions = AmazonBedrockRerankingModelOptions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_point_serializes_expected_shape() {
        let value =
            serde_json::to_value(BedrockCachePoint::new().with_ttl(BedrockCacheTtl::FiveMinutes))
                .expect("serialize BedrockCachePoint");

        assert_eq!(
            value,
            serde_json::json!({
                "cachePoint": {
                    "type": "default",
                    "ttl": "5m"
                }
            })
        );
    }

    #[test]
    fn chat_options_serialize_expected_shape() {
        let value = serde_json::to_value(
            BedrockChatOptions::new()
                .with_additional_model_request_fields(serde_json::json!({ "topK": 32 }))
                .with_reasoning_config(
                    BedrockReasoningConfig::new()
                        .with_type(BedrockReasoningType::Enabled)
                        .with_budget_tokens(2048)
                        .with_max_reasoning_effort(BedrockReasoningEffort::High),
                )
                .with_anthropic_beta(["context-1m-2025-08-07"])
                .with_service_tier(BedrockServiceTier::Priority)
                .with_param("guardrailConfig", serde_json::json!({ "id": "gr-1" })),
        )
        .expect("serialize options");

        assert_eq!(
            value,
            serde_json::json!({
                "additionalModelRequestFields": { "topK": 32 },
                "reasoningConfig": {
                    "type": "enabled",
                    "budgetTokens": 2048,
                    "maxReasoningEffort": "high"
                },
                "anthropicBeta": ["context-1m-2025-08-07"],
                "serviceTier": "priority",
                "guardrailConfig": { "id": "gr-1" }
            })
        );
    }

    #[test]
    fn chat_options_deserialize_snake_case_aliases() {
        let options: BedrockChatOptions = serde_json::from_value(serde_json::json!({
            "additional_model_request_fields": { "topK": 8 },
            "reasoning_config": {
                "type": "adaptive",
                "budget_tokens": 512,
                "max_reasoning_effort": "max"
            },
            "anthropic_beta": ["context-1m-2025-08-07"],
            "service_tier": "flex"
        }))
        .expect("deserialize options");

        assert_eq!(
            options.additional_model_request_fields,
            Some(serde_json::json!({ "topK": 8 }))
        );
        assert_eq!(
            options.reasoning_config,
            Some(
                BedrockReasoningConfig::new()
                    .with_type(BedrockReasoningType::Adaptive)
                    .with_budget_tokens(512)
                    .with_max_reasoning_effort(BedrockReasoningEffort::Max)
            )
        );
        assert_eq!(
            options.anthropic_beta,
            Some(vec!["context-1m-2025-08-07".to_string()])
        );
        assert_eq!(options.service_tier, Some(BedrockServiceTier::Flex));
    }

    #[test]
    fn file_part_provider_options_serialize_expected_shape() {
        let value = serde_json::to_value(
            BedrockFilePartProviderOptions::new()
                .with_citations(true)
                .with_param("vendorHint", serde_json::json!("abc")),
        )
        .expect("serialize file-part options");

        assert_eq!(
            value,
            serde_json::json!({
                "citations": { "enabled": true },
                "vendorHint": "abc"
            })
        );
    }

    #[test]
    fn rerank_options_serialize_expected_shape() {
        let value = serde_json::to_value(
            BedrockRerankOptions::new()
                .with_region("us-east-1")
                .with_next_token("token-1")
                .with_additional_model_request_fields(serde_json::json!({ "topK": 4 })),
        )
        .expect("serialize options");

        assert_eq!(
            value,
            serde_json::json!({
                "region": "us-east-1",
                "nextToken": "token-1",
                "additionalModelRequestFields": { "topK": 4 }
            })
        );
    }

    #[test]
    fn embedding_options_serialize_expected_shape() {
        let value = serde_json::to_value(
            BedrockEmbeddingOptions::new()
                .with_dimensions(512)
                .with_normalize(true)
                .with_embedding_dimension(256)
                .with_embedding_purpose(BedrockEmbeddingPurpose::GenericIndex)
                .with_input_type(BedrockEmbeddingInputType::SearchDocument)
                .with_truncate(BedrockEmbeddingTruncate::End)
                .with_output_dimension(1024),
        )
        .expect("serialize embedding options");

        assert_eq!(
            value,
            serde_json::json!({
                "dimensions": 512,
                "normalize": true,
                "embeddingDimension": 256,
                "embeddingPurpose": "GENERIC_INDEX",
                "inputType": "search_document",
                "truncate": "END",
                "outputDimension": 1024
            })
        );
    }

    #[test]
    fn embedding_options_deserialize_snake_case_aliases() {
        let options: BedrockEmbeddingOptions = serde_json::from_value(serde_json::json!({
            "dimensions": 256,
            "normalize": false,
            "embedding_dimension": 3072,
            "embedding_purpose": "TEXT_RETRIEVAL",
            "input_type": "search_query",
            "truncate": "START",
            "output_dimension": 1536
        }))
        .expect("deserialize embedding options");

        assert_eq!(
            options,
            BedrockEmbeddingOptions::new()
                .with_dimensions(256)
                .with_normalize(false)
                .with_embedding_dimension(3072)
                .with_embedding_purpose(BedrockEmbeddingPurpose::TextRetrieval)
                .with_input_type(BedrockEmbeddingInputType::SearchQuery)
                .with_truncate(BedrockEmbeddingTruncate::Start)
                .with_output_dimension(1536)
        );
    }
}
