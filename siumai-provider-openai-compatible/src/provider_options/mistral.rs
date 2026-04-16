//! Mistral provider options.
//!
//! These typed option structs are owned by the OpenAI-compatible provider crate and are
//! serialized into `providerOptions["mistral"]`.

use serde::{Deserialize, Serialize};

/// Mistral reasoning-effort level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MistralReasoningEffort {
    /// Enable provider-side reasoning.
    High,
    /// Disable provider-side reasoning.
    None,
}

/// Typed Mistral chat/language-model options stored under `providerOptions["mistral"]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MistralChatOptions {
    /// Whether to inject the provider safety prompt.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "safe_prompt"
    )]
    pub safe_prompt: Option<bool>,
    /// Maximum number of document images to process.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "document_image_limit"
    )]
    pub document_image_limit: Option<u32>,
    /// Maximum number of document pages to process.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "document_page_limit"
    )]
    pub document_page_limit: Option<u32>,
    /// Whether to preserve JSON Schema structured outputs.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "structured_outputs"
    )]
    pub structured_outputs: Option<bool>,
    /// Whether to mark `response_format.json_schema.strict`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "strict_json_schema"
    )]
    pub strict_json_schema: Option<bool>,
    /// Whether to allow parallel tool calls.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "parallel_tool_calls"
    )]
    pub parallel_tool_calls: Option<bool>,
    /// Optional provider-owned reasoning effort.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "reasoning_effort"
    )]
    pub reasoning_effort: Option<MistralReasoningEffort>,
}

impl MistralChatOptions {
    /// Create empty Mistral chat options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the safety-prompt flag.
    pub const fn with_safe_prompt(mut self, safe_prompt: bool) -> Self {
        self.safe_prompt = Some(safe_prompt);
        self
    }

    /// Set the document image limit.
    pub const fn with_document_image_limit(mut self, document_image_limit: u32) -> Self {
        self.document_image_limit = Some(document_image_limit);
        self
    }

    /// Set the document page limit.
    pub const fn with_document_page_limit(mut self, document_page_limit: u32) -> Self {
        self.document_page_limit = Some(document_page_limit);
        self
    }

    /// Control JSON Schema structured outputs.
    pub const fn with_structured_outputs(mut self, structured_outputs: bool) -> Self {
        self.structured_outputs = Some(structured_outputs);
        self
    }

    /// Control strict JSON Schema serialization.
    pub const fn with_strict_json_schema(mut self, strict_json_schema: bool) -> Self {
        self.strict_json_schema = Some(strict_json_schema);
        self
    }

    /// Control parallel tool calls.
    pub const fn with_parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = Some(parallel_tool_calls);
        self
    }

    /// Set provider-owned reasoning effort.
    pub const fn with_reasoning_effort(mut self, reasoning_effort: MistralReasoningEffort) -> Self {
        self.reasoning_effort = Some(reasoning_effort);
        self
    }
}

/// AI SDK-aligned alias for Mistral chat options.
pub type MistralLanguageModelOptions = MistralChatOptions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mistral_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            MistralChatOptions::new()
                .with_safe_prompt(true)
                .with_document_image_limit(8)
                .with_document_page_limit(16)
                .with_structured_outputs(false)
                .with_strict_json_schema(true)
                .with_parallel_tool_calls(false)
                .with_reasoning_effort(MistralReasoningEffort::High),
        )
        .expect("options serialize");

        assert_eq!(
            value,
            serde_json::json!({
                "safePrompt": true,
                "documentImageLimit": 8,
                "documentPageLimit": 16,
                "structuredOutputs": false,
                "strictJsonSchema": true,
                "parallelToolCalls": false,
                "reasoningEffort": "high"
            })
        );
    }

    #[test]
    fn mistral_options_accept_snake_case_aliases() {
        let options: MistralChatOptions = serde_json::from_value(serde_json::json!({
            "safe_prompt": true,
            "document_image_limit": 4,
            "document_page_limit": 10,
            "structured_outputs": true,
            "strict_json_schema": false,
            "parallel_tool_calls": true,
            "reasoning_effort": "none"
        }))
        .expect("options deserialize");

        assert_eq!(options.safe_prompt, Some(true));
        assert_eq!(options.document_image_limit, Some(4));
        assert_eq!(options.document_page_limit, Some(10));
        assert_eq!(options.structured_outputs, Some(true));
        assert_eq!(options.strict_json_schema, Some(false));
        assert_eq!(options.parallel_tool_calls, Some(true));
        assert_eq!(options.reasoning_effort, Some(MistralReasoningEffort::None));
    }
}
