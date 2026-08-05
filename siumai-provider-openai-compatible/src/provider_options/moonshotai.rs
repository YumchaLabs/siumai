//! MoonshotAI provider options.
//!
//! These typed option structs are owned by the OpenAI-compatible provider crate and are
//! serialized into `providerOptions["moonshotai"]`.

use serde::{Deserialize, Serialize};
use siumai_core::{ProviderOptionError, TypedProviderOptions};

/// Kimi K3 reasoning effort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KimiReasoningEffort {
    Low,
    High,
    Max,
}

/// Thinking mode supported by current Kimi K2.x models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KimiThinkingMode {
    Enabled,
    Disabled,
}

/// Historical reasoning retention supported by current Kimi K2.x models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KimiThinkingRetention {
    All,
}

/// Current Kimi K2.x thinking configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiThinking {
    #[serde(rename = "type")]
    pub mode: KimiThinkingMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep: Option<KimiThinkingRetention>,
}

impl KimiThinking {
    pub const fn new(mode: KimiThinkingMode) -> Self {
        Self { mode, keep: None }
    }

    pub const fn with_preserved_history(mut self) -> Self {
        self.keep = Some(KimiThinkingRetention::All);
        self
    }
}

/// Typed options for the current Kimi Chat Completions profile.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiLanguageOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<KimiReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<KimiThinking>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
}

impl KimiLanguageOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_reasoning_effort(mut self, effort: KimiReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    pub fn with_thinking(mut self, thinking: KimiThinking) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub fn with_prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.prompt_cache_key = Some(key.into());
        self
    }

    pub fn with_safety_identifier(mut self, identifier: impl Into<String>) -> Self {
        self.safety_identifier = Some(identifier.into());
        self
    }
}

impl TypedProviderOptions for KimiLanguageOptions {
    const NAMESPACE: &'static str = "moonshotai";

    fn validate(&self) -> Result<(), ProviderOptionError> {
        for (path, value) in [
            ("prompt_cache_key", self.prompt_cache_key.as_deref()),
            ("safety_identifier", self.safety_identifier.as_deref()),
        ] {
            if value.is_some_and(|value| value.trim().is_empty()) {
                return Err(ProviderOptionError::Rejected {
                    path: path.to_string(),
                    reason: "must not be empty".to_string(),
                });
            }
        }
        Ok(())
    }
}

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
    use siumai_core::ProviderOptions;

    #[test]
    fn current_kimi_options_use_the_moonshotai_namespace_and_wire_names() {
        let options = ProviderOptions::typed(
            &KimiLanguageOptions::new()
                .with_reasoning_effort(KimiReasoningEffort::High)
                .with_thinking(
                    KimiThinking::new(KimiThinkingMode::Enabled).with_preserved_history(),
                )
                .with_prompt_cache_key("session-42")
                .with_safety_identifier("user-hash"),
        )
        .unwrap();

        assert_eq!(options.namespace().as_str(), "moonshotai");
        assert_eq!(
            options.value(),
            &serde_json::json!({
                "reasoning_effort": "high",
                "thinking": {"type": "enabled", "keep": "all"},
                "prompt_cache_key": "session-42",
                "safety_identifier": "user-hash"
            })
            .as_object()
            .unwrap()
            .clone()
        );
    }

    #[test]
    fn current_kimi_options_reject_empty_cache_and_safety_ids() {
        assert!(
            ProviderOptions::typed(&KimiLanguageOptions::new().with_prompt_cache_key("  "))
                .is_err()
        );
        assert!(
            ProviderOptions::typed(&KimiLanguageOptions::new().with_safety_identifier("")).is_err()
        );
    }

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
