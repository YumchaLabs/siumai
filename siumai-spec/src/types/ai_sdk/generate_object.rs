use crate::types::{FinishReason, ModelMessage, PromptInput, SystemPrompt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{
    CallWarning, JSONSchema7, JSONValue, LanguageModelRequestMetadata,
    LanguageModelResponseMetadata, LanguageModelUsage, ProviderMetadata, ProviderOptions,
};

/// Output strategy marker used by AI SDK `generateObject` and `streamObject` callbacks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum GenerateObjectOutputStrategy {
    /// Object output strategy.
    Object,
    /// Array output strategy.
    Array,
    /// Enum output strategy.
    Enum,
    /// Schema-less JSON output strategy.
    NoSchema,
}

/// AI SDK-style response metadata envelope for structured object generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateObjectResponseMetadata {
    /// Shared language-model response metadata.
    #[serde(flatten)]
    pub metadata: LanguageModelResponseMetadata,
    /// Raw response body when the provider exposes it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl GenerateObjectResponseMetadata {
    /// Create a structured-output response metadata envelope.
    pub fn new(metadata: LanguageModelResponseMetadata) -> Self {
        Self {
            metadata,
            body: None,
        }
    }

    /// Attach a raw response body.
    pub fn with_body(mut self, body: impl Into<JSONValue>) -> Self {
        self.body = Some(body.into());
        self
    }
}

/// Event payload for AI SDK `generateObject` / `streamObject` start callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateObjectStartEvent {
    /// Unique generation call id.
    pub call_id: String,
    /// Operation id, normally `ai.generateObject` or `ai.streamObject`.
    pub operation_id: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// System prompt input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    /// Prompt input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<PromptInput>,
    /// Message input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<ModelMessage>>,
    /// Maximum output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-k sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<f64>,
    /// Presence penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Frequency penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Deterministic seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Maximum number of retries.
    pub max_retries: u32,
    /// Additional HTTP headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, Option<String>>>,
    /// Provider-specific options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
    /// Output strategy.
    pub output: GenerateObjectOutputStrategy,
    /// JSON Schema used for object generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<JSONSchema7>,
    /// Optional schema name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_name: Option<String>,
    /// Optional schema description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_description: Option<String>,
}

/// Event payload for AI SDK structured-output step start callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateObjectStepStartEvent {
    /// Unique generation call id.
    pub call_id: String,
    /// Zero-based step index. Upstream object generation currently uses step `0`.
    pub step_number: u32,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Provider-specific options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
    /// Additional HTTP headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, Option<String>>>,
    /// Prompt messages in provider format, approximated by Siumai's shared model-message carrier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_messages: Option<Vec<ModelMessage>>,
}

/// Event payload for AI SDK structured-output step finish callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateObjectStepEndEvent {
    /// Unique generation call id.
    pub call_id: String,
    /// Zero-based step index. Upstream object generation currently uses step `0`.
    pub step_number: u32,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model_id: String,
    /// Unified finish reason.
    pub finish_reason: FinishReason,
    /// Token usage.
    pub usage: LanguageModelUsage,
    /// Raw object text before parsing/validation.
    pub object_text: String,
    /// Reasoning text when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Provider warnings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<CallWarning>>,
    /// Request metadata.
    pub request: LanguageModelRequestMetadata,
    /// Response metadata.
    pub response: GenerateObjectResponseMetadata,
    /// Provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Milliseconds from stream start to first chunk on streaming paths.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ms_to_first_chunk: Option<u64>,
}

/// Event payload for AI SDK structured-output finish callbacks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateObjectEndEvent<RESULT = JSONValue> {
    /// Unique generation call id.
    pub call_id: String,
    /// Generated object when parsing and validation succeeded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<RESULT>,
    /// Parse or validation error when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JSONValue>,
    /// Reasoning text when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Unified finish reason.
    pub finish_reason: FinishReason,
    /// Token usage.
    pub usage: LanguageModelUsage,
    /// Provider warnings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<CallWarning>>,
    /// Request metadata.
    pub request: LanguageModelRequestMetadata,
    /// Response metadata.
    pub response: GenerateObjectResponseMetadata,
    /// Provider-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<ProviderMetadata>,
}
