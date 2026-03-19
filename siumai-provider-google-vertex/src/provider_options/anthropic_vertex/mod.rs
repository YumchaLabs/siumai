//! Provider-owned typed provider options for Anthropic on Vertex AI.

use serde::{Deserialize, Serialize};

/// Structured-output routing strategy for Anthropic-on-Vertex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum VertexAnthropicStructuredOutputMode {
    Auto,
    OutputFormat,
    JsonTool,
}

/// Thinking-mode configuration for Anthropic-on-Vertex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexAnthropicThinkingMode {
    /// Whether extended thinking is enabled.
    pub enabled: bool,
    /// Thinking budget in tokens.
    #[serde(skip_serializing_if = "Option::is_none", alias = "thinkingBudget")]
    pub thinking_budget: Option<u32>,
}

impl Default for VertexAnthropicThinkingMode {
    fn default() -> Self {
        Self {
            enabled: true,
            thinking_budget: None,
        }
    }
}

impl VertexAnthropicThinkingMode {
    /// Create an enabled thinking-mode config.
    pub fn enabled(thinking_budget: Option<u32>) -> Self {
        Self {
            enabled: true,
            thinking_budget,
        }
    }
}

/// Typed provider options for Anthropic-on-Vertex.
///
/// These values are serialized into `provider_options_map["anthropic"]` because the
/// Vertex wrapper reuses Anthropic's protocol-level option namespace.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VertexAnthropicOptions {
    /// Extended thinking mode (`thinking_mode` / `thinkingMode`).
    #[serde(skip_serializing_if = "Option::is_none", alias = "thinkingMode")]
    pub thinking_mode: Option<VertexAnthropicThinkingMode>,
    /// Structured-output routing strategy (`structured_output_mode` / `structuredOutputMode`).
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "structuredOutputMode"
    )]
    pub structured_output_mode: Option<VertexAnthropicStructuredOutputMode>,
    /// Disable Anthropic parallel tool use (`disable_parallel_tool_use` / `disableParallelToolUse`).
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "disableParallelToolUse"
    )]
    pub disable_parallel_tool_use: Option<bool>,
    /// Whether reasoning parts from prior turns should be forwarded (`send_reasoning` / `sendReasoning`).
    #[serde(skip_serializing_if = "Option::is_none", alias = "sendReasoning")]
    pub send_reasoning: Option<bool>,
}

impl VertexAnthropicOptions {
    /// Create empty Anthropic-on-Vertex options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure thinking mode.
    pub fn with_thinking_mode(mut self, thinking_mode: VertexAnthropicThinkingMode) -> Self {
        self.thinking_mode = Some(thinking_mode);
        self
    }

    /// Configure structured-output routing mode.
    pub fn with_structured_output_mode(
        mut self,
        mode: VertexAnthropicStructuredOutputMode,
    ) -> Self {
        self.structured_output_mode = Some(mode);
        self
    }

    /// Configure parallel tool-use behavior.
    pub fn with_disable_parallel_tool_use(mut self, disabled: bool) -> Self {
        self.disable_parallel_tool_use = Some(disabled);
        self
    }

    /// Configure reasoning replay behavior for prior turns.
    pub fn with_send_reasoning(mut self, send_reasoning: bool) -> Self {
        self.send_reasoning = Some(send_reasoning);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_mode_serializes_to_anthropic_provider_options_shape() {
        let value = serde_json::to_value(
            VertexAnthropicOptions::new()
                .with_thinking_mode(VertexAnthropicThinkingMode::enabled(Some(2048))),
        )
        .expect("serialize vertex anthropic options");

        assert_eq!(value["thinking_mode"]["enabled"], serde_json::json!(true));
        assert_eq!(
            value["thinking_mode"]["thinking_budget"],
            serde_json::json!(2048)
        );
    }

    #[test]
    fn structured_output_mode_serializes_to_provider_options_shape() {
        let value = serde_json::to_value(
            VertexAnthropicOptions::new()
                .with_structured_output_mode(VertexAnthropicStructuredOutputMode::JsonTool),
        )
        .expect("serialize vertex anthropic options");

        assert_eq!(
            value.get("structured_output_mode"),
            Some(&serde_json::json!("jsonTool"))
        );
    }

    #[test]
    fn options_deserialize_from_camel_case_shape() {
        let options: VertexAnthropicOptions = serde_json::from_value(serde_json::json!({
            "thinkingMode": {
                "enabled": true,
                "thinkingBudget": 3072
            },
            "structuredOutputMode": "outputFormat",
            "disableParallelToolUse": true,
            "sendReasoning": false
        }))
        .expect("deserialize vertex anthropic options");

        let thinking_mode = options.thinking_mode.expect("thinking mode present");
        assert!(thinking_mode.enabled);
        assert_eq!(thinking_mode.thinking_budget, Some(3072));
        assert_eq!(
            options.structured_output_mode,
            Some(VertexAnthropicStructuredOutputMode::OutputFormat)
        );
        assert_eq!(options.disable_parallel_tool_use, Some(true));
        assert_eq!(options.send_reasoning, Some(false));
    }

    #[test]
    fn options_serialization_omits_unset_fields() {
        let value = serde_json::to_value(
            VertexAnthropicOptions::new()
                .with_disable_parallel_tool_use(true)
                .with_send_reasoning(false),
        )
        .expect("serialize vertex anthropic options");

        let obj = value.as_object().expect("vertex anthropic options object");
        assert_eq!(
            obj.get("disable_parallel_tool_use"),
            Some(&serde_json::json!(true))
        );
        assert_eq!(obj.get("send_reasoning"), Some(&serde_json::json!(false)));
        assert!(!obj.contains_key("thinking_mode"));
        assert!(!obj.contains_key("structured_output_mode"));
    }
}
