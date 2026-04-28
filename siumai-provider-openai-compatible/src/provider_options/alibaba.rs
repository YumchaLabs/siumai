//! Alibaba/Qwen provider options.
//!
//! These typed option structs mirror AI SDK `@ai-sdk/alibaba` language-model options.
//! The OpenAI-compatible preset historically uses provider id `qwen`, so both
//! `providerOptions["alibaba"]` and `providerOptions["qwen"]` are supported by the runtime.

use serde::{Deserialize, Serialize};

/// Typed Alibaba/Qwen chat/language-model options.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AlibabaChatOptions {
    /// Enable thinking/reasoning mode for supported Qwen models.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "enable_thinking"
    )]
    pub enable_thinking: Option<bool>,
    /// Maximum number of reasoning tokens to generate.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "thinking_budget"
    )]
    pub thinking_budget: Option<u32>,
    /// Whether to allow parallel tool calls.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "parallel_tool_calls"
    )]
    pub parallel_tool_calls: Option<bool>,
}

impl AlibabaChatOptions {
    /// Create empty Alibaba/Qwen chat options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable provider-side thinking.
    pub const fn with_enable_thinking(mut self, enable_thinking: bool) -> Self {
        self.enable_thinking = Some(enable_thinking);
        self
    }

    /// Set the thinking budget.
    pub const fn with_thinking_budget(mut self, thinking_budget: u32) -> Self {
        self.thinking_budget = Some(thinking_budget);
        self
    }

    /// Control parallel tool calls.
    pub const fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = Some(parallel_tool_calls);
        self
    }
}

/// AI SDK-aligned alias for Alibaba language-model options.
pub type AlibabaLanguageModelOptions = AlibabaChatOptions;

/// Deprecated AI SDK-compatible alias for Alibaba language-model options.
#[deprecated(note = "Use AlibabaLanguageModelOptions instead.")]
pub type AlibabaProviderOptions = AlibabaLanguageModelOptions;

/// Local preset alias for Qwen chat options.
pub type QwenChatOptions = AlibabaChatOptions;

/// Local preset alias for Qwen language-model options.
pub type QwenLanguageModelOptions = AlibabaLanguageModelOptions;

/// Deprecated local preset alias for Qwen provider options.
#[deprecated(note = "Use QwenLanguageModelOptions instead.")]
pub type QwenProviderOptions = QwenLanguageModelOptions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alibaba_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            AlibabaChatOptions::new()
                .with_enable_thinking(true)
                .with_thinking_budget(2048)
                .with_parallel_tool_calls(false),
        )
        .expect("options serialize");

        assert_eq!(
            value,
            serde_json::json!({
                "enableThinking": true,
                "thinkingBudget": 2048,
                "parallelToolCalls": false
            })
        );
    }

    #[test]
    fn alibaba_options_accept_snake_case_aliases() {
        let options: AlibabaChatOptions = serde_json::from_value(serde_json::json!({
            "enable_thinking": false,
            "thinking_budget": 1024,
            "parallel_tool_calls": true
        }))
        .expect("options deserialize");

        assert_eq!(options.enable_thinking, Some(false));
        assert_eq!(options.thinking_budget, Some(1024));
        assert_eq!(options.parallel_tool_calls, Some(true));
    }
}
