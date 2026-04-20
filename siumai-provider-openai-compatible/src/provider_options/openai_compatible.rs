//! Generic OpenAI-compatible typed option structs.
//!
//! These options target the shared `providerOptions["openaiCompatible"]` namespace and mirror the
//! AI SDK's generic `@ai-sdk/openai-compatible` option surface.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Typed generic OpenAI-compatible chat/language-model options stored under
/// `providerOptions["openaiCompatible"]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAiCompatibleLanguageModelChatOptions {
    /// A unique identifier representing the end user.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Reasoning effort for reasoning-capable models.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "reasoning_effort"
    )]
    pub reasoning_effort: Option<String>,
    /// Controls text verbosity when the provider supports it.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "text_verbosity"
    )]
    pub text_verbosity: Option<String>,
    /// Enables strict JSON-schema validation when structured outputs are supported.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "strict_json_schema"
    )]
    pub strict_json_schema: Option<bool>,
}

impl OpenAiCompatibleLanguageModelChatOptions {
    /// Create empty generic OpenAI-compatible chat options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the end-user id.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set the reasoning effort.
    pub fn with_reasoning_effort(mut self, reasoning_effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(reasoning_effort.into());
        self
    }

    /// Set the text verbosity.
    pub fn with_text_verbosity(mut self, text_verbosity: impl Into<String>) -> Self {
        self.text_verbosity = Some(text_verbosity.into());
        self
    }

    /// Enable or disable strict JSON schema validation.
    pub const fn with_strict_json_schema(mut self, strict_json_schema: bool) -> Self {
        self.strict_json_schema = Some(strict_json_schema);
        self
    }
}

/// Typed generic OpenAI-compatible completion options stored under
/// `providerOptions["openaiCompatible"]`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAiCompatibleLanguageModelCompletionOptions {
    /// Echo the prompt in addition to the completion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
    /// Modify token likelihoods by token id.
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "logit_bias")]
    pub logit_bias: Option<BTreeMap<String, f64>>,
    /// Append a suffix after inserted text completions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    /// A unique identifier representing the end user.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl OpenAiCompatibleLanguageModelCompletionOptions {
    /// Create empty generic OpenAI-compatible completion options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable prompt echoing.
    pub const fn with_echo(mut self, echo: bool) -> Self {
        self.echo = Some(echo);
        self
    }

    /// Replace the full logit-bias map.
    pub fn with_logit_bias(mut self, logit_bias: BTreeMap<String, f64>) -> Self {
        self.logit_bias = Some(logit_bias);
        self
    }

    /// Add or override a single logit-bias entry.
    pub fn with_logit_bias_token(mut self, token_id: impl Into<String>, bias: f64) -> Self {
        self.logit_bias
            .get_or_insert_with(BTreeMap::new)
            .insert(token_id.into(), bias);
        self
    }

    /// Set the completion suffix.
    pub fn with_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = Some(suffix.into());
        self
    }

    /// Set the end-user id.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}

/// Typed generic OpenAI-compatible embedding options stored under
/// `providerOptions["openaiCompatible"]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAiCompatibleEmbeddingModelOptions {
    /// Requested output dimensionality.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// A unique identifier representing the end user.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl OpenAiCompatibleEmbeddingModelOptions {
    /// Create empty generic OpenAI-compatible embedding options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the requested output dimensionality.
    pub const fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Set the end-user id.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}

/// AI SDK-exact-case alias for generic OpenAI-compatible chat options.
pub type OpenAICompatibleLanguageModelChatOptions = OpenAiCompatibleLanguageModelChatOptions;

/// AI SDK-exact-case alias for generic OpenAI-compatible completion options.
pub type OpenAICompatibleLanguageModelCompletionOptions =
    OpenAiCompatibleLanguageModelCompletionOptions;

/// AI SDK-exact-case alias for generic OpenAI-compatible embedding options.
pub type OpenAICompatibleEmbeddingModelOptions = OpenAiCompatibleEmbeddingModelOptions;

/// Deprecated AI SDK-exact-case alias for generic OpenAI-compatible chat options.
#[deprecated(note = "Use OpenAICompatibleLanguageModelChatOptions instead.")]
pub type OpenAICompatibleProviderOptions = OpenAICompatibleLanguageModelChatOptions;

/// Deprecated AI SDK-exact-case alias for generic OpenAI-compatible completion options.
#[deprecated(note = "Use OpenAICompatibleLanguageModelCompletionOptions instead.")]
pub type OpenAICompatibleCompletionProviderOptions = OpenAICompatibleLanguageModelCompletionOptions;

/// Deprecated AI SDK-exact-case alias for generic OpenAI-compatible embedding options.
#[deprecated(note = "Use OpenAICompatibleEmbeddingModelOptions instead.")]
pub type OpenAICompatibleEmbeddingProviderOptions = OpenAICompatibleEmbeddingModelOptions;

/// Deprecated AI SDK-compatible alias for generic OpenAI-compatible chat options.
#[deprecated(note = "Use OpenAICompatibleLanguageModelChatOptions instead.")]
pub type OpenAiCompatibleProviderOptions = OpenAiCompatibleLanguageModelChatOptions;

/// Deprecated AI SDK-compatible alias for generic OpenAI-compatible completion options.
#[deprecated(note = "Use OpenAICompatibleLanguageModelCompletionOptions instead.")]
pub type OpenAiCompatibleCompletionProviderOptions = OpenAiCompatibleLanguageModelCompletionOptions;

/// Deprecated AI SDK-compatible alias for generic OpenAI-compatible embedding options.
#[deprecated(note = "Use OpenAICompatibleEmbeddingModelOptions instead.")]
pub type OpenAiCompatibleEmbeddingProviderOptions = OpenAiCompatibleEmbeddingModelOptions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_compatible_chat_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            OpenAiCompatibleLanguageModelChatOptions::new()
                .with_user("user-123")
                .with_reasoning_effort("high")
                .with_text_verbosity("medium")
                .with_strict_json_schema(true),
        )
        .expect("options serialize");

        assert_eq!(
            value,
            serde_json::json!({
                "user": "user-123",
                "reasoningEffort": "high",
                "textVerbosity": "medium",
                "strictJsonSchema": true
            })
        );
    }

    #[test]
    fn openai_compatible_chat_options_accept_snake_case_aliases() {
        let options: OpenAiCompatibleLanguageModelChatOptions =
            serde_json::from_value(serde_json::json!({
                "user": "user-456",
                "reasoning_effort": "low",
                "text_verbosity": "high",
                "strict_json_schema": false
            }))
            .expect("options deserialize");

        assert_eq!(options.user.as_deref(), Some("user-456"));
        assert_eq!(options.reasoning_effort.as_deref(), Some("low"));
        assert_eq!(options.text_verbosity.as_deref(), Some("high"));
        assert_eq!(options.strict_json_schema, Some(false));
    }

    #[test]
    fn openai_compatible_completion_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            OpenAiCompatibleLanguageModelCompletionOptions::new()
                .with_echo(true)
                .with_logit_bias_token("42", 1.5)
                .with_suffix(" after")
                .with_user("user-789"),
        )
        .expect("options serialize");

        assert_eq!(
            value,
            serde_json::json!({
                "echo": true,
                "logitBias": {
                    "42": 1.5
                },
                "suffix": " after",
                "user": "user-789"
            })
        );
    }

    #[test]
    fn openai_compatible_completion_options_accept_snake_case_aliases() {
        let options: OpenAiCompatibleLanguageModelCompletionOptions =
            serde_json::from_value(serde_json::json!({
                "echo": true,
                "logit_bias": {
                    "42": 2.0
                },
                "suffix": " after",
                "user": "user-999"
            }))
            .expect("options deserialize");

        assert_eq!(options.echo, Some(true));
        assert_eq!(
            options
                .logit_bias
                .as_ref()
                .and_then(|value| value.get("42"))
                .copied(),
            Some(2.0)
        );
        assert_eq!(options.suffix.as_deref(), Some(" after"));
        assert_eq!(options.user.as_deref(), Some("user-999"));
    }

    #[test]
    fn openai_compatible_embedding_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            OpenAiCompatibleEmbeddingModelOptions::new()
                .with_dimensions(256)
                .with_user("user-321"),
        )
        .expect("options serialize");

        assert_eq!(
            value,
            serde_json::json!({
                "dimensions": 256,
                "user": "user-321"
            })
        );
    }

    #[test]
    #[allow(deprecated)]
    fn openai_compatible_option_aliases_remain_available() {
        let _: OpenAICompatibleLanguageModelChatOptions =
            OpenAiCompatibleLanguageModelChatOptions::new();
        let _: OpenAICompatibleProviderOptions = OpenAiCompatibleLanguageModelChatOptions::new();
        let _: OpenAICompatibleLanguageModelCompletionOptions =
            OpenAiCompatibleLanguageModelCompletionOptions::new();
        let _: OpenAICompatibleCompletionProviderOptions =
            OpenAiCompatibleLanguageModelCompletionOptions::new();
        let _: OpenAICompatibleEmbeddingModelOptions = OpenAiCompatibleEmbeddingModelOptions::new();
        let _: OpenAICompatibleEmbeddingProviderOptions =
            OpenAiCompatibleEmbeddingModelOptions::new();
        let _: OpenAiCompatibleLanguageModelChatOptions =
            OpenAiCompatibleLanguageModelChatOptions::new();
        let _: OpenAiCompatibleProviderOptions = OpenAiCompatibleLanguageModelChatOptions::new();
        let _: OpenAiCompatibleLanguageModelCompletionOptions =
            OpenAiCompatibleLanguageModelCompletionOptions::new();
        let _: OpenAiCompatibleCompletionProviderOptions =
            OpenAiCompatibleLanguageModelCompletionOptions::new();
        let _: OpenAiCompatibleEmbeddingModelOptions = OpenAiCompatibleEmbeddingModelOptions::new();
        let _: OpenAiCompatibleEmbeddingProviderOptions =
            OpenAiCompatibleEmbeddingModelOptions::new();
    }
}
