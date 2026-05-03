//! Azure-owned OpenAI-compatible provider option carriers.
//!
//! Azure exposes an OpenAI-compatible request surface, but provider crates must not depend on each
//! other. These types intentionally mirror the high-value AI SDK-style OpenAI option shapes used by
//! Azure callers while keeping the Azure crate dependent only on core/protocol crates.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Reasoning effort level for OpenAI-compatible reasoning models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    #[default]
    Medium,
    High,
    Xhigh,
}

/// Service tier preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    #[default]
    Auto,
    Flex,
    Priority,
    Default,
}

/// Prompt cache retention policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PromptCacheRetention {
    #[default]
    InMemory,
    #[serde(rename = "24h")]
    H24,
}

/// Text verbosity level for Responses API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TextVerbosity {
    Low,
    #[default]
    Medium,
    High,
}

/// Truncation strategy for Responses API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Truncation {
    Auto,
    Disabled,
}

/// System message mode for OpenAI-compatible Responses API requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SystemMessageMode {
    System,
    Developer,
    Remove,
}

/// Logprobs option accepted by OpenAI-compatible Responses/Chat options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesLogprobs {
    Bool(bool),
    Top(u32),
}

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

/// AI SDK-style flat Chat Completions options stored under `providerOptions["azure"]`.
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

/// AI SDK-style flat Responses API options stored under `providerOptions["azure"]`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OpenAILanguageModelResponsesOptions {
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
        map.serialize_entry("responsesApi", &serde_json::json!({ "enabled": true }))?;

        macro_rules! serialize_optional_entry {
            ($field:ident, $name:literal) => {
                if let Some(value) = &self.$field {
                    map.serialize_entry($name, value)?;
                }
            };
        }

        serialize_optional_entry!(conversation, "conversation");
        serialize_optional_entry!(include, "include");
        serialize_optional_entry!(instructions, "instructions");
        serialize_optional_entry!(logprobs, "logprobs");
        serialize_optional_entry!(max_tool_calls, "maxToolCalls");
        serialize_optional_entry!(metadata, "metadata");
        serialize_optional_entry!(parallel_tool_calls, "parallelToolCalls");
        serialize_optional_entry!(previous_response_id, "previousResponseId");
        serialize_optional_entry!(prompt_cache_key, "promptCacheKey");
        serialize_optional_entry!(prompt_cache_retention, "promptCacheRetention");
        serialize_optional_entry!(reasoning_effort, "reasoningEffort");
        serialize_optional_entry!(reasoning_summary, "reasoningSummary");
        serialize_optional_entry!(response_format, "responseFormat");
        serialize_optional_entry!(safety_identifier, "safetyIdentifier");
        serialize_optional_entry!(service_tier, "serviceTier");
        serialize_optional_entry!(store, "store");
        serialize_optional_entry!(strict_json_schema, "strictJsonSchema");
        serialize_optional_entry!(text_verbosity, "textVerbosity");
        serialize_optional_entry!(truncation, "truncation");
        serialize_optional_entry!(user, "user");
        serialize_optional_entry!(background, "background");
        serialize_optional_entry!(max_completion_tokens, "maxCompletionTokens");
        serialize_optional_entry!(system_message_mode, "systemMessageMode");
        serialize_optional_entry!(force_reasoning, "forceReasoning");
        serialize_optional_entry!(context_management, "contextManagement");

        for (key, value) in &self.extra_fields {
            map.serialize_entry(key, value)?;
        }

        map.end()
    }
}

/// Deprecated AI SDK compatibility alias for OpenAI Responses options.
#[deprecated(note = "Use `OpenAILanguageModelResponsesOptions` instead.")]
pub type OpenAIResponsesProviderOptions = OpenAILanguageModelResponsesOptions;
