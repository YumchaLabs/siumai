//! Fireworks provider options.
//!
//! These typed option structs are owned by the OpenAI-compatible provider crate and are
//! serialized into `providerOptions["fireworks"]`.

use serde::{Deserialize, Serialize};

/// Fireworks reasoning mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FireworksThinkingType {
    /// Enable provider-side reasoning/thinking.
    Enabled,
    /// Disable provider-side reasoning/thinking.
    Disabled,
}

/// Fireworks reasoning history mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FireworksReasoningHistory {
    /// Do not preserve reasoning history.
    Disabled,
    /// Interleave reasoning with the visible response.
    Interleaved,
    /// Preserve reasoning history separately.
    Preserved,
}

/// Typed Fireworks thinking config stored under `providerOptions.fireworks.thinking`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FireworksThinkingConfig {
    /// Thinking mode (`enabled` / `disabled`).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "type",
        alias = "thinking_type"
    )]
    pub thinking_type: Option<FireworksThinkingType>,
    /// Maximum thinking token budget.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "budget_tokens"
    )]
    pub budget_tokens: Option<u32>,
}

impl FireworksThinkingConfig {
    /// Create an empty thinking config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the thinking mode.
    pub const fn with_type(mut self, thinking_type: FireworksThinkingType) -> Self {
        self.thinking_type = Some(thinking_type);
        self
    }

    /// Set the thinking token budget.
    pub const fn with_budget_tokens(mut self, budget_tokens: u32) -> Self {
        self.budget_tokens = Some(budget_tokens);
        self
    }
}

/// Typed Fireworks chat/language-model options stored under `providerOptions["fireworks"]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FireworksChatOptions {
    /// Optional thinking/reasoning configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<FireworksThinkingConfig>,
    /// Optional reasoning-history mode.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "reasoning_history"
    )]
    pub reasoning_history: Option<FireworksReasoningHistory>,
}

impl FireworksChatOptions {
    /// Create empty Fireworks chat options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the thinking configuration.
    pub fn with_thinking(mut self, thinking: FireworksThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Set the reasoning-history mode.
    pub const fn with_reasoning_history(
        mut self,
        reasoning_history: FireworksReasoningHistory,
    ) -> Self {
        self.reasoning_history = Some(reasoning_history);
        self
    }
}

/// Typed Fireworks embedding options stored under `providerOptions["fireworks"]`.
///
/// The current AI SDK package exports an empty object type for embedding options, so we keep the
/// same explicit struct on the Rust side for surface parity.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FireworksEmbeddingModelOptions {}

/// AI SDK-aligned alias for Fireworks chat options.
pub type FireworksLanguageModelOptions = FireworksChatOptions;

/// Deprecated AI SDK-compatible alias for Fireworks language-model options.
#[deprecated(note = "Use FireworksLanguageModelOptions instead.")]
pub type FireworksProviderOptions = FireworksLanguageModelOptions;

/// Deprecated AI SDK-compatible alias for Fireworks embedding options.
#[deprecated(note = "Use FireworksEmbeddingModelOptions instead.")]
pub type FireworksEmbeddingProviderOptions = FireworksEmbeddingModelOptions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fireworks_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            FireworksChatOptions::new()
                .with_thinking(
                    FireworksThinkingConfig::new()
                        .with_type(FireworksThinkingType::Enabled)
                        .with_budget_tokens(2048),
                )
                .with_reasoning_history(FireworksReasoningHistory::Interleaved),
        )
        .expect("options serialize");

        assert_eq!(
            value,
            serde_json::json!({
                "thinking": {
                    "type": "enabled",
                    "budgetTokens": 2048
                },
                "reasoningHistory": "interleaved"
            })
        );
    }

    #[test]
    fn fireworks_options_accept_snake_case_aliases() {
        let options: FireworksChatOptions = serde_json::from_value(serde_json::json!({
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1024
            },
            "reasoning_history": "preserved"
        }))
        .expect("options deserialize");

        assert_eq!(
            options.thinking.expect("thinking").budget_tokens,
            Some(1024)
        );
        assert_eq!(
            options.reasoning_history,
            Some(FireworksReasoningHistory::Preserved)
        );
    }

    #[test]
    fn fireworks_embedding_options_serialize_to_empty_object() {
        let value = serde_json::to_value(FireworksEmbeddingModelOptions::default())
            .expect("options serialize");

        assert_eq!(value, serde_json::json!({}));
    }

    #[test]
    #[allow(deprecated)]
    fn fireworks_option_aliases_remain_available() {
        let _: FireworksLanguageModelOptions = FireworksChatOptions::new();
        let _: FireworksProviderOptions = FireworksChatOptions::new();
        let _: FireworksEmbeddingModelOptions = FireworksEmbeddingModelOptions::default();
        let _: FireworksEmbeddingProviderOptions = FireworksEmbeddingModelOptions::default();
    }
}
