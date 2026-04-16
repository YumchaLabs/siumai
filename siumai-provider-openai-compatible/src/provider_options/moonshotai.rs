//! MoonshotAI provider options.
//!
//! These typed option structs are owned by the OpenAI-compatible provider crate and are
//! serialized into `providerOptions["moonshotai"]`.

use serde::{Deserialize, Serialize};

/// MoonshotAI thinking mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MoonshotAIThinkingType {
    /// Enable provider-side thinking.
    Enabled,
    /// Disable provider-side thinking.
    Disabled,
}

/// MoonshotAI reasoning-history mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MoonshotAIReasoningHistory {
    /// Do not preserve reasoning history.
    Disabled,
    /// Interleave reasoning with visible output.
    Interleaved,
    /// Preserve reasoning history separately.
    Preserved,
}

/// Typed MoonshotAI thinking config stored under `providerOptions.moonshotai.thinking`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoonshotAIThinkingConfig {
    /// Thinking mode (`enabled` / `disabled`).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "type",
        alias = "thinking_type"
    )]
    pub thinking_type: Option<MoonshotAIThinkingType>,
    /// Maximum thinking token budget.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "budget_tokens"
    )]
    pub budget_tokens: Option<u32>,
}

impl MoonshotAIThinkingConfig {
    /// Create an empty thinking config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the thinking mode.
    pub const fn with_type(mut self, thinking_type: MoonshotAIThinkingType) -> Self {
        self.thinking_type = Some(thinking_type);
        self
    }

    /// Set the thinking token budget.
    pub const fn with_budget_tokens(mut self, budget_tokens: u32) -> Self {
        self.budget_tokens = Some(budget_tokens);
        self
    }
}

/// Typed MoonshotAI chat/language-model options stored under `providerOptions["moonshotai"]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoonshotAIChatOptions {
    /// Optional thinking configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<MoonshotAIThinkingConfig>,
    /// Optional reasoning-history mode.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "reasoning_history"
    )]
    pub reasoning_history: Option<MoonshotAIReasoningHistory>,
}

impl MoonshotAIChatOptions {
    /// Create empty MoonshotAI chat options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the thinking configuration.
    pub fn with_thinking(mut self, thinking: MoonshotAIThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Set the reasoning-history mode.
    pub const fn with_reasoning_history(
        mut self,
        reasoning_history: MoonshotAIReasoningHistory,
    ) -> Self {
        self.reasoning_history = Some(reasoning_history);
        self
    }
}

/// AI SDK-aligned alias for MoonshotAI chat options.
pub type MoonshotAILanguageModelOptions = MoonshotAIChatOptions;

/// Deprecated AI SDK-compatible alias for MoonshotAI language-model options.
#[deprecated(note = "Use MoonshotAILanguageModelOptions instead.")]
pub type MoonshotAIProviderOptions = MoonshotAILanguageModelOptions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moonshotai_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            MoonshotAIChatOptions::new()
                .with_thinking(
                    MoonshotAIThinkingConfig::new()
                        .with_type(MoonshotAIThinkingType::Enabled)
                        .with_budget_tokens(2048),
                )
                .with_reasoning_history(MoonshotAIReasoningHistory::Interleaved),
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
    fn moonshotai_options_accept_snake_case_aliases() {
        let options: MoonshotAIChatOptions = serde_json::from_value(serde_json::json!({
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
            Some(MoonshotAIReasoningHistory::Preserved)
        );
    }

    #[test]
    #[allow(deprecated)]
    fn moonshotai_option_alias_remains_available() {
        let _: MoonshotAILanguageModelOptions = MoonshotAIChatOptions::new();
        let _: MoonshotAIProviderOptions = MoonshotAIChatOptions::new();
    }
}
