//! DeepSeek-owned language-model options.

use serde::{Deserialize, Serialize};
use siumai_core::{ModelFamily, ProviderOptionError, TypedProviderOptions};
use siumai_protocol_openai::chat_completions::API_MODE_ID as CHAT_API_MODE_ID;
use siumai_protocol_openai::responses::API_MODE_ID as RESPONSES_API_MODE_ID;

/// DeepSeek thinking mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeepSeekThinkingType {
    Enabled,
    Disabled,
}

/// DeepSeek `thinking` request object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepSeekThinkingConfig {
    #[serde(rename = "type")]
    pub kind: DeepSeekThinkingType,
}

impl DeepSeekThinkingConfig {
    pub const fn enabled() -> Self {
        Self {
            kind: DeepSeekThinkingType::Enabled,
        }
    }

    pub const fn disabled() -> Self {
        Self {
            kind: DeepSeekThinkingType::Disabled,
        }
    }
}

/// DeepSeek reasoning effort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeepSeekReasoningEffort {
    High,
    Max,
}

/// Provider-owned options for DeepSeek Chat Completions.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DeepSeekChatOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<DeepSeekThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<DeepSeekReasoningEffort>,
    /// Emit DeepSeek's strict function-tool declaration for every portable function tool.
    ///
    /// Strict mode is a provider policy control and is removed before the top-level request map is
    /// sent. Callers remain responsible for selecting an endpoint that enables strict tool mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict_tools: Option<bool>,
}

impl DeepSeekChatOptions {
    pub const fn new() -> Self {
        Self {
            thinking: None,
            reasoning_effort: None,
            strict_tools: None,
        }
    }

    pub const fn with_thinking(mut self, thinking: DeepSeekThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub const fn with_thinking_enabled(self) -> Self {
        self.with_thinking(DeepSeekThinkingConfig::enabled())
    }

    pub const fn with_thinking_disabled(self) -> Self {
        self.with_thinking(DeepSeekThinkingConfig::disabled())
    }

    pub const fn with_reasoning_effort(mut self, effort: DeepSeekReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    pub const fn with_strict_tools(mut self, enabled: bool) -> Self {
        self.strict_tools = Some(enabled);
        self
    }
}

impl TypedProviderOptions for DeepSeekChatOptions {
    const NAMESPACE: &'static str = "deepseek";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(CHAT_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if self
            .thinking
            .is_some_and(|thinking| thinking.kind == DeepSeekThinkingType::Disabled)
            && self.reasoning_effort.is_some()
        {
            return Err(ProviderOptionError::Rejected {
                path: "reasoning_effort".to_string(),
                reason: "reasoning effort cannot be set when thinking is disabled".to_string(),
            });
        }
        Ok(())
    }
}

/// Provider-owned options for DeepSeek's Responses API.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "snake_case")]
pub struct DeepSeekResponsesOptions {
    /// DeepSeek currently documents `high` and `max` for Responses reasoning effort.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<DeepSeekReasoningEffort>,
    /// Number of most likely tokens returned at each output token position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    /// Caller identity used by DeepSeek's documented isolation controls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Provider-executed tools supported by DeepSeek Responses.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub native_tools: Vec<DeepSeekResponsesTool>,
}

/// Provider-executed tools documented by DeepSeek Responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DeepSeekResponsesTool {
    WebSearch,
    ApplyPatch,
}

impl DeepSeekResponsesTool {
    pub(crate) fn as_wire_value(self) -> serde_json::Value {
        match self {
            Self::WebSearch => serde_json::json!({"type": "web_search"}),
            Self::ApplyPatch => serde_json::json!({
                "type": "custom",
                "name": "apply_patch"
            }),
        }
    }
}

impl DeepSeekResponsesOptions {
    pub const fn new() -> Self {
        Self {
            reasoning_effort: None,
            top_logprobs: None,
            user: None,
            native_tools: Vec::new(),
        }
    }

    pub const fn with_reasoning_effort(mut self, effort: DeepSeekReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    pub const fn with_top_logprobs(mut self, count: u8) -> Self {
        self.top_logprobs = Some(count);
        self
    }

    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub fn with_native_tool(mut self, tool: DeepSeekResponsesTool) -> Self {
        self.native_tools.push(tool);
        self
    }

    pub fn with_web_search(self) -> Self {
        self.with_native_tool(DeepSeekResponsesTool::WebSearch)
    }

    pub fn with_apply_patch(self) -> Self {
        self.with_native_tool(DeepSeekResponsesTool::ApplyPatch)
    }
}

impl TypedProviderOptions for DeepSeekResponsesOptions {
    const NAMESPACE: &'static str = "deepseek";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(RESPONSES_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if self.top_logprobs.is_some_and(|count| count > 20) {
            return Err(ProviderOptionError::Rejected {
                path: "top_logprobs".to_string(),
                reason: "must be between zero and 20".to_string(),
            });
        }
        if self
            .user
            .as_deref()
            .is_some_and(|user| user.trim().is_empty() || user.chars().any(char::is_control))
        {
            return Err(ProviderOptionError::Rejected {
                path: "user".to_string(),
                reason: "must not be empty or contain control characters".to_string(),
            });
        }
        let unique = self
            .native_tools
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        if unique.len() != self.native_tools.len() {
            return Err(ProviderOptionError::Rejected {
                path: "native_tools".to_string(),
                reason: "must not contain duplicate tools".to_string(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::{ApiModeId, ProviderOptions};

    #[test]
    fn chat_options_use_the_deepseek_wire_shape() {
        let options = DeepSeekChatOptions::new()
            .with_thinking_enabled()
            .with_reasoning_effort(DeepSeekReasoningEffort::High)
            .with_strict_tools(true);
        let value = serde_json::to_value(options).expect("serialize options");

        assert_eq!(
            value,
            serde_json::json!({
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
                "strict_tools": true
            })
        );
    }

    #[test]
    fn contradictory_chat_reasoning_is_rejected_before_erasure() {
        let options = DeepSeekChatOptions::new()
            .with_thinking_disabled()
            .with_reasoning_effort(DeepSeekReasoningEffort::High);
        assert!(ProviderOptions::typed(&options).is_err());
    }

    #[test]
    fn responses_options_are_bounded_and_use_the_responses_mode() {
        let options = DeepSeekResponsesOptions::new()
            .with_reasoning_effort(DeepSeekReasoningEffort::Max)
            .with_top_logprobs(20)
            .with_user("tenant-1")
            .with_web_search()
            .with_apply_patch();
        let options = ProviderOptions::typed(&options).expect("valid options");

        assert_eq!(options.api_mode().map(ApiModeId::as_str), Some("responses"));
        assert_eq!(options.value()["reasoning_effort"], "max");
        assert_eq!(options.value()["top_logprobs"], 20);
        assert_eq!(options.value()["user"], "tenant-1");
        assert_eq!(options.value()["native_tools"][0], "web_search");
        assert_eq!(options.value()["native_tools"][1], "apply_patch");
        assert!(
            ProviderOptions::typed(&DeepSeekResponsesOptions::new().with_top_logprobs(21)).is_err()
        );
    }
}
