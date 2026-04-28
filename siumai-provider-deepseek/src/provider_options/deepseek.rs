//! `DeepSeek` provider options.
//!
//! These typed option structs are owned by the DeepSeek provider crate and are serialized into
//! `providerOptions["deepseek"]` (Vercel-aligned open options map).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// DeepSeek thinking mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeepSeekThinkingType {
    /// Enable provider-side thinking.
    Enabled,
    /// Disable provider-side thinking.
    Disabled,
}

/// Typed DeepSeek thinking config stored under `providerOptions.deepseek.thinking`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeepSeekThinkingConfig {
    /// Thinking mode (`enabled` / `disabled`).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "type",
        alias = "thinking_type",
        alias = "thinkingType"
    )]
    pub thinking_type: Option<DeepSeekThinkingType>,
}

impl DeepSeekThinkingConfig {
    /// Create an empty thinking config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the thinking mode.
    pub const fn with_type(mut self, thinking_type: DeepSeekThinkingType) -> Self {
        self.thinking_type = Some(thinking_type);
        self
    }
}

/// DeepSeek-specific language-model options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeepSeekOptions {
    /// Optional thinking configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<DeepSeekThinkingConfig>,
    /// Additional provider-specific parameters.
    #[serde(flatten)]
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl DeepSeekOptions {
    /// Create new DeepSeek options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the thinking configuration.
    pub fn with_thinking(mut self, thinking: DeepSeekThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Enable provider-side thinking.
    pub fn with_thinking_enabled(self) -> Self {
        self.with_thinking(DeepSeekThinkingConfig::new().with_type(DeepSeekThinkingType::Enabled))
    }

    /// Disable provider-side thinking.
    pub fn with_thinking_disabled(self) -> Self {
        self.with_thinking(DeepSeekThinkingConfig::new().with_type(DeepSeekThinkingType::Disabled))
    }

    /// Backward-compatible alias for enabling/disabling thinking.
    pub fn with_reasoning(self, enable: bool) -> Self {
        if enable {
            self.with_thinking_enabled()
        } else {
            self.with_thinking_disabled()
        }
    }

    /// Backward-compatible alias that enables thinking. DeepSeek does not accept a token budget.
    pub fn with_reasoning_budget(self, _budget: i32) -> Self {
        self.with_thinking_enabled()
    }

    /// Add a custom DeepSeek parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}

/// AI SDK-style alias for the canonical DeepSeek language-model options surface.
pub type DeepSeekLanguageModelOptions = DeepSeekOptions;

/// Deprecated AI SDK-compatible alias retained for migration.
#[deprecated(note = "use `DeepSeekLanguageModelOptions` instead")]
pub type DeepSeekChatOptions = DeepSeekLanguageModelOptions;

/// Deprecated alias retained for migration.
#[deprecated(note = "use `DeepSeekLanguageModelOptions` instead")]
pub type DeepSeekProviderOptions = DeepSeekLanguageModelOptions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(DeepSeekOptions::new().with_thinking_enabled())
            .expect("serialize options");

        assert_eq!(
            value,
            serde_json::json!({
                "thinking": {
                    "type": "enabled"
                }
            })
        );
    }

    #[test]
    fn deepseek_reasoning_budget_alias_enables_thinking_without_budget() {
        let value = serde_json::to_value(DeepSeekOptions::new().with_reasoning_budget(2048))
            .expect("serialize options");

        assert_eq!(
            value,
            serde_json::json!({
                "thinking": {
                    "type": "enabled"
                }
            })
        );
        assert!(value.get("reasoningBudget").is_none());
        assert!(value.get("reasoning_budget").is_none());
    }

    #[test]
    fn deepseek_options_accept_snake_case_aliases() {
        let options: DeepSeekOptions = serde_json::from_value(serde_json::json!({
            "thinking": {
                "thinking_type": "disabled"
            }
        }))
        .expect("deserialize options");

        assert_eq!(
            options.thinking.expect("thinking").thinking_type,
            Some(DeepSeekThinkingType::Disabled)
        );
    }

    #[test]
    fn deepseek_options_support_extra_params() {
        let options = DeepSeekOptions::new().with_param("foo", serde_json::json!("bar"));
        let value = serde_json::to_value(options).expect("serialize options");
        assert_eq!(value["foo"], serde_json::json!("bar"));
    }

    #[test]
    #[allow(deprecated)]
    fn ai_sdk_style_aliases_resolve_to_same_type() {
        let _ = std::mem::size_of::<DeepSeekOptions>();
        let _ = std::mem::size_of::<DeepSeekLanguageModelOptions>();
        let _ = std::mem::size_of::<DeepSeekChatOptions>();
        let _ = std::mem::size_of::<DeepSeekProviderOptions>();
    }
}
