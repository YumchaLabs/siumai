//! DeepSeek provider options.
//!
//! These typed option structs are owned by the OpenAI-compatible provider crate and are
//! serialized into `providerOptions["deepseek"]`.

use serde::{Deserialize, Serialize};

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
        alias = "thinking_type"
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

/// Typed DeepSeek chat/language-model options.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeepSeekLanguageModelOptions {
    /// Optional thinking configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<DeepSeekThinkingConfig>,
}

impl DeepSeekLanguageModelOptions {
    /// Create empty DeepSeek language-model options.
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
}

/// Deprecated AI SDK-compatible alias for DeepSeek language-model options.
#[deprecated(note = "Use DeepSeekLanguageModelOptions instead.")]
pub type DeepSeekChatOptions = DeepSeekLanguageModelOptions;

/// Deprecated AI SDK-compatible alias for DeepSeek language-model options.
#[deprecated(note = "Use DeepSeekLanguageModelOptions instead.")]
pub type DeepSeekProviderOptions = DeepSeekLanguageModelOptions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek_options_serialize_to_ai_sdk_shape() {
        let value =
            serde_json::to_value(DeepSeekLanguageModelOptions::new().with_thinking_enabled())
                .expect("options serialize");

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
    fn deepseek_options_accept_snake_case_aliases() {
        let options: DeepSeekLanguageModelOptions = serde_json::from_value(serde_json::json!({
            "thinking": {
                "thinking_type": "disabled"
            }
        }))
        .expect("options deserialize");

        assert_eq!(
            options.thinking.expect("thinking").thinking_type,
            Some(DeepSeekThinkingType::Disabled)
        );
    }

    #[test]
    #[allow(deprecated)]
    fn deepseek_option_aliases_remain_available() {
        let _: DeepSeekChatOptions = DeepSeekLanguageModelOptions::new();
        let _: DeepSeekProviderOptions = DeepSeekLanguageModelOptions::new();
    }
}
