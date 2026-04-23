//! AI SDK-style typed OpenAI provider option surfaces.
//!
//! These types mirror the named exports from `@ai-sdk/openai` while serializing into the
//! provider-owned `providerOptions["openai"]` JSON map that siumai already understands.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{
    PromptCacheRetention, ReasoningEffort, ResponsesApiConfig, ResponsesLogprobs, ServiceTier,
    SystemMessageMode, TextVerbosity, Truncation,
};

/// Typed OpenAI Responses context-management entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAIContextManagementConfig {
    #[serde(rename = "type")]
    pub kind: OpenAIContextManagementType,
    #[serde(
        rename = "compactThreshold",
        alias = "compact_threshold",
        skip_serializing_if = "Option::is_none"
    )]
    pub compact_threshold: Option<u32>,
}

impl OpenAIContextManagementConfig {
    pub fn compaction(compact_threshold: u32) -> Self {
        Self {
            kind: OpenAIContextManagementType::Compaction,
            compact_threshold: Some(compact_threshold),
        }
    }
}

/// Discriminator for OpenAI Responses context management.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIContextManagementType {
    Compaction,
}

/// AI SDK-style flat chat options stored under `providerOptions["openai"]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAILanguageModelChatOptions {
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "logitBias",
        alias = "logit_bias"
    )]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ResponsesLogprobs>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "parallelToolCalls",
        alias = "parallel_tool_calls"
    )]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "reasoningEffort",
        alias = "reasoning_effort"
    )]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "maxCompletionTokens",
        alias = "max_completion_tokens"
    )]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<serde_json::Value>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "serviceTier",
        alias = "service_tier"
    )]
    pub service_tier: Option<ServiceTier>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "strictJsonSchema",
        alias = "strict_json_schema"
    )]
    pub strict_json_schema: Option<bool>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "textVerbosity",
        alias = "text_verbosity"
    )]
    pub text_verbosity: Option<TextVerbosity>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "promptCacheKey",
        alias = "prompt_cache_key"
    )]
    pub prompt_cache_key: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "promptCacheRetention",
        alias = "prompt_cache_retention"
    )]
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "safetyIdentifier",
        alias = "safety_identifier"
    )]
    pub safety_identifier: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "forceReasoning",
        alias = "force_reasoning"
    )]
    pub force_reasoning: Option<bool>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "systemMessageMode",
        alias = "system_message_mode"
    )]
    pub system_message_mode: Option<SystemMessageMode>,
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl OpenAILanguageModelChatOptions {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Deprecated AI SDK compatibility alias for OpenAI chat options.
#[deprecated(note = "Use `OpenAILanguageModelChatOptions` instead.")]
pub type OpenAIChatLanguageModelOptions = OpenAILanguageModelChatOptions;

/// AI SDK-style flat Responses API options stored under `providerOptions[\"openai\"]`.
///
/// Internally this type always serializes an enabled `responsesApi` envelope so the request is
/// routed through `/responses` even when callers only set flat AI SDK-style fields.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAILanguageModelResponsesOptions {
    #[serde(skip, default = "ResponsesApiConfig::new")]
    responses_api: ResponsesApiConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ResponsesLogprobs>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "maxToolCalls",
        alias = "max_tool_calls"
    )]
    pub max_tool_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "parallelToolCalls",
        alias = "parallel_tool_calls"
    )]
    pub parallel_tool_calls: Option<bool>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "previousResponseId",
        alias = "previous_response_id"
    )]
    pub previous_response_id: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "promptCacheKey",
        alias = "prompt_cache_key"
    )]
    pub prompt_cache_key: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "promptCacheRetention",
        alias = "prompt_cache_retention"
    )]
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "reasoningEffort",
        alias = "reasoning_effort"
    )]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "reasoningSummary",
        alias = "reasoning_summary"
    )]
    pub reasoning_summary: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "responseFormat",
        alias = "response_format"
    )]
    pub response_format: Option<serde_json::Value>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "safetyIdentifier",
        alias = "safety_identifier"
    )]
    pub safety_identifier: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "serviceTier",
        alias = "service_tier"
    )]
    pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "strictJsonSchema",
        alias = "strict_json_schema"
    )]
    pub strict_json_schema: Option<bool>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "textVerbosity",
        alias = "text_verbosity"
    )]
    pub text_verbosity: Option<TextVerbosity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "maxCompletionTokens",
        alias = "max_completion_tokens"
    )]
    pub max_completion_tokens: Option<u32>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "systemMessageMode",
        alias = "system_message_mode"
    )]
    pub system_message_mode: Option<SystemMessageMode>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "forceReasoning",
        alias = "force_reasoning"
    )]
    pub force_reasoning: Option<bool>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "contextManagement",
        alias = "context_management"
    )]
    pub context_management: Option<Vec<OpenAIContextManagementConfig>>,
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl Default for OpenAILanguageModelResponsesOptions {
    fn default() -> Self {
        Self {
            responses_api: ResponsesApiConfig::new(),
            conversation: None,
            include: None,
            instructions: None,
            logprobs: None,
            max_tool_calls: None,
            metadata: None,
            parallel_tool_calls: None,
            previous_response_id: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            reasoning_effort: None,
            reasoning_summary: None,
            response_format: None,
            safety_identifier: None,
            service_tier: None,
            store: None,
            strict_json_schema: None,
            text_verbosity: None,
            truncation: None,
            user: None,
            background: None,
            max_completion_tokens: None,
            system_message_mode: None,
            force_reasoning: None,
            context_management: None,
            extra_fields: HashMap::new(),
        }
    }
}

impl OpenAILanguageModelResponsesOptions {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Serialize for OpenAILanguageModelResponsesOptions {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry(
            "responsesApi",
            &serde_json::json!({ "enabled": self.responses_api.enabled }),
        )?;

        if let Some(value) = &self.conversation {
            map.serialize_entry("conversation", value)?;
        }
        if let Some(value) = &self.include {
            map.serialize_entry("include", value)?;
        }
        if let Some(value) = &self.instructions {
            map.serialize_entry("instructions", value)?;
        }
        if let Some(value) = &self.logprobs {
            map.serialize_entry("logprobs", value)?;
        }
        if let Some(value) = &self.max_tool_calls {
            map.serialize_entry("maxToolCalls", value)?;
        }
        if let Some(value) = &self.metadata {
            map.serialize_entry("metadata", value)?;
        }
        if let Some(value) = &self.parallel_tool_calls {
            map.serialize_entry("parallelToolCalls", value)?;
        }
        if let Some(value) = &self.previous_response_id {
            map.serialize_entry("previousResponseId", value)?;
        }
        if let Some(value) = &self.prompt_cache_key {
            map.serialize_entry("promptCacheKey", value)?;
        }
        if let Some(value) = &self.prompt_cache_retention {
            map.serialize_entry("promptCacheRetention", value)?;
        }
        if let Some(value) = &self.reasoning_effort {
            map.serialize_entry("reasoningEffort", value)?;
        }
        if let Some(value) = &self.reasoning_summary {
            map.serialize_entry("reasoningSummary", value)?;
        }
        if let Some(value) = &self.response_format {
            map.serialize_entry("responseFormat", value)?;
        }
        if let Some(value) = &self.safety_identifier {
            map.serialize_entry("safetyIdentifier", value)?;
        }
        if let Some(value) = &self.service_tier {
            map.serialize_entry("serviceTier", value)?;
        }
        if let Some(value) = &self.store {
            map.serialize_entry("store", value)?;
        }
        if let Some(value) = &self.strict_json_schema {
            map.serialize_entry("strictJsonSchema", value)?;
        }
        if let Some(value) = &self.text_verbosity {
            map.serialize_entry("textVerbosity", value)?;
        }
        if let Some(value) = &self.truncation {
            map.serialize_entry("truncation", value)?;
        }
        if let Some(value) = &self.user {
            map.serialize_entry("user", value)?;
        }
        if let Some(value) = &self.background {
            map.serialize_entry("background", value)?;
        }
        if let Some(value) = &self.max_completion_tokens {
            map.serialize_entry("maxCompletionTokens", value)?;
        }
        if let Some(value) = &self.system_message_mode {
            map.serialize_entry("systemMessageMode", value)?;
        }
        if let Some(value) = &self.force_reasoning {
            map.serialize_entry("forceReasoning", value)?;
        }
        if let Some(value) = &self.context_management {
            map.serialize_entry("contextManagement", value)?;
        }
        for (key, value) in &self.extra_fields {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

/// Deprecated AI SDK compatibility alias for OpenAI Responses options.
#[deprecated(note = "Use `OpenAILanguageModelResponsesOptions` instead.")]
pub type OpenAIResponsesProviderOptions = OpenAILanguageModelResponsesOptions;

/// AI SDK-style completion options for the legacy `/completions` endpoint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAILanguageModelCompletionOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "logitBias",
        alias = "logit_bias"
    )]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ResponsesLogprobs>,
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl OpenAILanguageModelCompletionOptions {
    pub fn new() -> Self {
        Self::default()
    }
}

/// AI SDK-style alias for OpenAI embedding options.
#[cfg(feature = "openai")]
pub type OpenAIEmbeddingModelOptions = crate::providers::openai::types::OpenAiEmbeddingOptions;

/// AI SDK-style alias for OpenAI speech options.
#[cfg(feature = "openai")]
pub type OpenAISpeechModelOptions = crate::providers::openai::ext::audio_options::OpenAiTtsOptions;

/// AI SDK-style alias for OpenAI transcription options.
#[cfg(feature = "openai")]
pub type OpenAITranscriptionModelOptions =
    crate::providers::openai::ext::audio_options::OpenAiSttOptions;

/// AI SDK-style file upload options for OpenAI-compatible file endpoints.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAIFilesOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub purpose: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "expiresAfter",
        alias = "expires_after"
    )]
    pub expires_after: Option<u64>,
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

impl OpenAIFilesOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_purpose(mut self, purpose: impl Into<String>) -> Self {
        self.purpose = Some(purpose.into());
        self
    }

    pub fn with_expires_after(mut self, expires_after: u64) -> Self {
        self.expires_after = Some(expires_after);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_options_serialize_ai_sdk_keys() {
        let value = serde_json::to_value(OpenAILanguageModelChatOptions {
            force_reasoning: Some(true),
            system_message_mode: Some(SystemMessageMode::Developer),
            max_completion_tokens: Some(256),
            ..Default::default()
        })
        .expect("chat options serialize");

        assert_eq!(value["forceReasoning"], serde_json::json!(true));
        assert_eq!(value["systemMessageMode"], serde_json::json!("developer"));
        assert_eq!(value["maxCompletionTokens"], serde_json::json!(256));
    }

    #[test]
    fn responses_options_force_enabled_responses_api() {
        let value = serde_json::to_value(OpenAILanguageModelResponsesOptions {
            instructions: Some("be concise".to_string()),
            ..Default::default()
        })
        .expect("responses options serialize");

        assert_eq!(value["responsesApi"]["enabled"], serde_json::json!(true));
        assert_eq!(value["instructions"], serde_json::json!("be concise"));
    }

    #[test]
    fn responses_options_serialize_ai_sdk_control_fields() {
        let value = serde_json::to_value(OpenAILanguageModelResponsesOptions {
            system_message_mode: Some(SystemMessageMode::Developer),
            force_reasoning: Some(true),
            context_management: Some(vec![OpenAIContextManagementConfig::compaction(512)]),
            ..Default::default()
        })
        .expect("responses options serialize");

        assert_eq!(value["systemMessageMode"], serde_json::json!("developer"));
        assert_eq!(value["forceReasoning"], serde_json::json!(true));
        assert_eq!(
            value["contextManagement"][0],
            serde_json::json!({
                "type": "compaction",
                "compactThreshold": 512
            })
        );
    }

    #[test]
    fn completion_options_serialize_camel_case_keys() {
        let value = serde_json::to_value(OpenAILanguageModelCompletionOptions {
            logit_bias: Some(serde_json::json!({ "42": 1.5 })),
            ..Default::default()
        })
        .expect("completion options serialize");

        assert_eq!(value["logitBias"], serde_json::json!({ "42": 1.5 }));
    }

    #[test]
    fn file_options_serialize_expires_after() {
        let value = serde_json::to_value(OpenAIFilesOptions::new().with_expires_after(3600))
            .expect("file options serialize");

        assert_eq!(value["expiresAfter"], serde_json::json!(3600));
    }
}
