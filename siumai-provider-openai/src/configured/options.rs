use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use siumai_core::{ModelFamily, ProviderOptionError, TypedProviderOptions};
use siumai_protocol_openai::chat_completions::API_MODE_ID as CHAT_API_MODE_ID;
use siumai_protocol_openai::responses_next::API_MODE_ID as RESPONSES_API_MODE_ID;
use siumai_protocol_openai::responses_next::{FunctionToolCaller, FunctionToolEncodingOptions};

const MAX_TOP_LOGPROBS: u8 = 20;
const MAX_PROMPT_CACHE_BREAKPOINTS: usize = 50;

/// GPT-5.6 reasoning effort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiReasoningEffort {
    None,
    /// Supported by selected earlier reasoning models, but not by GPT-5.6.
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

/// GPT-5.6 reasoning execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiReasoningMode {
    Standard,
    Pro,
}

/// Reasoning history made available to a Responses request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiReasoningContext {
    /// Use the selected model's current default.
    #[default]
    Auto,
    CurrentTurn,
    AllTurns,
}

/// Optional reasoning summary detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

/// Responses `reasoning` request object.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<OpenAiReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<OpenAiReasoningMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<OpenAiReasoningContext>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<OpenAiReasoningSummary>,
}

impl OpenAiReasoning {
    pub fn with_effort(mut self, effort: OpenAiReasoningEffort) -> Self {
        self.effort = Some(effort);
        self
    }

    pub fn with_mode(mut self, mode: OpenAiReasoningMode) -> Self {
        self.mode = Some(mode);
        self
    }

    pub fn with_context(mut self, context: OpenAiReasoningContext) -> Self {
        self.context = Some(context);
        self
    }

    pub fn with_summary(mut self, summary: OpenAiReasoningSummary) -> Self {
        self.summary = Some(summary);
        self
    }
}

/// Prompt-cache breakpoint placement behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiPromptCacheMode {
    Implicit,
    Explicit,
}

/// Currently supported explicit prompt-cache lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenAiPromptCacheTtl {
    #[serde(rename = "30m")]
    ThirtyMinutes,
}

/// GPT-5.6 prompt-cache controls.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiPromptCacheOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<OpenAiPromptCacheMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<OpenAiPromptCacheTtl>,
}

impl OpenAiPromptCacheOptions {
    pub const fn explicit() -> Self {
        Self {
            mode: Some(OpenAiPromptCacheMode::Explicit),
            ttl: None,
        }
    }

    pub const fn explicit_30_minutes() -> Self {
        Self {
            mode: Some(OpenAiPromptCacheMode::Explicit),
            ttl: Some(OpenAiPromptCacheTtl::ThirtyMinutes),
        }
    }
}

/// One canonical request content block that should carry an explicit cache breakpoint.
///
/// Indices are zero-based and refer to `LanguageRequest.messages[message_index]` and
/// that message's `content[content_index]` entry. The Responses codec validates that
/// every configured breakpoint identifies an encodable input block. Both
/// Responses and Chat Completions use these canonical coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiPromptCacheBreakpoint {
    pub message_index: usize,
    pub content_index: usize,
}

impl OpenAiPromptCacheBreakpoint {
    pub const fn new(message_index: usize, content_index: usize) -> Self {
        Self {
            message_index,
            content_index,
        }
    }
}

/// A validated OpenAI-native Responses tool definition.
///
/// Portable function tools continue to use `LanguageRequest.tools`. This type is
/// reserved for provider-executed or provider-native tools whose wire contract is
/// intentionally OpenAI-specific, such as web search, file search, MCP, shell, or
/// custom/programmatic tools. The object remains open so newly released tool types
/// do not require a core-trait revision.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct OpenAiProviderTool(Value);

impl OpenAiProviderTool {
    pub fn new(value: Value) -> Result<Self, ProviderOptionError> {
        validate_provider_tool(&value)?;
        Ok(Self(value))
    }

    /// Construct the default OpenAI web-search tool.
    pub fn web_search() -> Self {
        Self(json!({ "type": "web_search" }))
    }

    /// Enable OpenAI's provider-native programmatic tool-calling runtime.
    pub fn programmatic_tool_calling() -> Self {
        Self(json!({ "type": "programmatic_tool_calling" }))
    }

    pub fn as_value(&self) -> &Value {
        &self.0
    }

    pub(crate) fn into_value(self) -> Value {
        self.0
    }
}

impl<'de> Deserialize<'de> for OpenAiProviderTool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// OpenAI execution paths allowed to invoke a portable function tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiToolCaller {
    Direct,
    Programmatic,
}

/// Responses-only controls applied to one `LanguageRequest` function tool.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiFunctionToolOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_callers: Vec<OpenAiToolCaller>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
}

impl OpenAiFunctionToolOptions {
    pub fn programmatic() -> Self {
        Self::default().with_allowed_caller(OpenAiToolCaller::Programmatic)
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    pub fn with_defer_loading(mut self, defer_loading: bool) -> Self {
        self.defer_loading = Some(defer_loading);
        self
    }

    pub fn with_allowed_caller(mut self, caller: OpenAiToolCaller) -> Self {
        if !self.allowed_callers.contains(&caller) {
            self.allowed_callers.push(caller);
        }
        self
    }

    pub fn with_output_schema(mut self, output_schema: Value) -> Self {
        self.output_schema = Some(output_schema);
        self
    }

    fn validate(&self, tool_name: &str) -> Result<(), ProviderOptionError> {
        validate_tool_name(tool_name)?;
        if self
            .output_schema
            .as_ref()
            .is_some_and(|schema| !schema.is_object() && !schema.is_boolean())
        {
            return Err(rejected(
                format!("function_tool_options.{tool_name}.output_schema"),
                "output schema must be a JSON Schema object or boolean",
            ));
        }
        if self
            .allowed_callers
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != self.allowed_callers.len()
        {
            return Err(rejected(
                format!("function_tool_options.{tool_name}.allowed_callers"),
                "allowed callers must be unique",
            ));
        }
        Ok(())
    }

    fn into_encoding_options(self) -> FunctionToolEncodingOptions {
        let mut options = FunctionToolEncodingOptions::default();
        if let Some(strict) = self.strict {
            options = options.with_strict(strict);
        }
        if let Some(defer_loading) = self.defer_loading {
            options = options.with_defer_loading(defer_loading);
        }
        for caller in self.allowed_callers {
            options = options.with_allowed_caller(match caller {
                OpenAiToolCaller::Direct => FunctionToolCaller::Direct,
                OpenAiToolCaller::Programmatic => FunctionToolCaller::Programmatic,
            });
        }
        if let Some(output_schema) = self.output_schema {
            options = options.with_output_schema(output_schema);
        }
        options
    }
}

/// Responses context-management entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAiContextManagement {
    Compaction {
        #[serde(skip_serializing_if = "Option::is_none")]
        compact_threshold: Option<u64>,
    },
}

impl OpenAiContextManagement {
    pub fn compaction(compact_threshold: impl Into<Option<u64>>) -> Self {
        Self::Compaction {
            compact_threshold: compact_threshold.into(),
        }
    }

    fn validate(&self) -> Result<(), ProviderOptionError> {
        match self {
            Self::Compaction {
                compact_threshold: Some(0),
            } => Err(rejected(
                "context_management.compact_threshold",
                "compaction threshold must be greater than zero",
            )),
            Self::Compaction { .. } => Ok(()),
        }
    }
}

/// Additional Responses payloads requested from OpenAI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OpenAiResponseInclude {
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
    #[serde(rename = "file_search_call.results")]
    FileSearchResults,
    #[serde(rename = "web_search_call.action.sources")]
    WebSearchActionSources,
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterOutputs,
    #[serde(rename = "computer_call_output.output.image_url")]
    ComputerOutputImageUrl,
    #[serde(rename = "message.input_image.image_url")]
    InputImageUrl,
    #[serde(rename = "message.output_text.logprobs")]
    OutputTextLogprobs,
}

impl OpenAiResponseInclude {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReasoningEncryptedContent => "reasoning.encrypted_content",
            Self::FileSearchResults => "file_search_call.results",
            Self::WebSearchActionSources => "web_search_call.action.sources",
            Self::CodeInterpreterOutputs => "code_interpreter_call.outputs",
            Self::ComputerOutputImageUrl => "computer_call_output.output.image_url",
            Self::InputImageUrl => "message.input_image.image_url",
            Self::OutputTextLogprobs => "message.output_text.logprobs",
        }
    }
}

/// OpenAI request service tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiServiceTier {
    Auto,
    Default,
    Flex,
    Fast,
    Priority,
}

/// GPT-5 family output verbosity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiTextVerbosity {
    Low,
    Medium,
    High,
}

/// Responses automatic context truncation behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiTruncation {
    Auto,
    Disabled,
}

/// Typed OpenAI Responses options.
///
/// These are provider-owned options layered through [`siumai_core::CallOptions`].
/// Model, input, portable function tools, output schema, streaming, and common
/// generation fields remain owned by the canonical language request and cannot
/// be overridden here. `native_tools` is the explicit provider-specific lane for
/// OpenAI-executed tools that have no portable `ToolSpec` representation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiResponsesOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub include: Vec<OpenAiResponseInclude>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_options: Option<OpenAiPromptCacheOptions>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompt_cache_breakpoints: Vec<OpenAiPromptCacheBreakpoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<OpenAiReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<OpenAiServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_verbosity: Option<OpenAiTextVerbosity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<OpenAiTruncation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context_management: Vec<OpenAiContextManagement>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub native_tools: Vec<OpenAiProviderTool>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub function_tool_options: BTreeMap<String, OpenAiFunctionToolOptions>,
}

impl OpenAiResponsesOptions {
    pub fn with_reasoning(mut self, reasoning: OpenAiReasoning) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    pub fn with_prompt_cache(mut self, cache: OpenAiPromptCacheOptions) -> Self {
        self.prompt_cache_options = Some(cache);
        self
    }

    pub fn with_prompt_cache_breakpoint(mut self, breakpoint: OpenAiPromptCacheBreakpoint) -> Self {
        self.prompt_cache_breakpoints.push(breakpoint);
        self
    }

    pub fn with_native_tool(mut self, tool: OpenAiProviderTool) -> Self {
        self.native_tools.push(tool);
        self
    }

    pub fn with_function_tool_options(
        mut self,
        name: impl Into<String>,
        options: OpenAiFunctionToolOptions,
    ) -> Self {
        self.function_tool_options.insert(name.into(), options);
        self
    }

    pub(crate) fn validate_values(&self) -> Result<(), ProviderOptionError> {
        if self.conversation.is_some() && self.previous_response_id.is_some() {
            return Err(rejected(
                "conversation",
                "conversation and previous_response_id are mutually exclusive",
            ));
        }
        validate_optional_text("conversation", self.conversation.as_deref())?;
        validate_optional_text("instructions", self.instructions.as_deref())?;
        validate_optional_text("previous_response_id", self.previous_response_id.as_deref())?;
        validate_optional_text("prompt_cache_key", self.prompt_cache_key.as_deref())?;
        validate_optional_text("safety_identifier", self.safety_identifier.as_deref())?;
        validate_optional_text("user", self.user.as_deref())?;
        if self.max_tool_calls == Some(0) {
            return Err(rejected(
                "max_tool_calls",
                "maximum tool calls must be greater than zero",
            ));
        }
        if self
            .top_logprobs
            .is_some_and(|value| value > MAX_TOP_LOGPROBS)
        {
            return Err(rejected("top_logprobs", "top_logprobs must not exceed 20"));
        }
        for entry in &self.context_management {
            entry.validate()?;
        }
        validate_prompt_cache(
            self.prompt_cache_options.as_ref(),
            &self.prompt_cache_breakpoints,
        )?;
        for tool in &self.native_tools {
            validate_provider_tool(tool.as_value())?;
        }
        for (name, options) in &self.function_tool_options {
            options.validate(name)?;
        }
        if self.include.iter().copied().collect::<BTreeSet<_>>().len() != self.include.len() {
            return Err(rejected("include", "include values must be unique"));
        }
        Ok(())
    }

    pub(crate) fn into_request_options(
        mut self,
    ) -> Result<OpenAiResponsesRequestOptions, ProviderOptionError> {
        if self.top_logprobs.is_some()
            && !self
                .include
                .contains(&OpenAiResponseInclude::OutputTextLogprobs)
        {
            self.include.push(OpenAiResponseInclude::OutputTextLogprobs);
        }
        let mut value = object_from(self)?;
        let prompt_cache_breakpoints = value
            .remove("prompt_cache_breakpoints")
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?
            .unwrap_or_default();
        let native_tools = value
            .remove("native_tools")
            .map(serde_json::from_value::<Vec<OpenAiProviderTool>>)
            .transpose()
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?
            .unwrap_or_default()
            .into_iter()
            .map(OpenAiProviderTool::into_value)
            .collect();
        let function_tools = value
            .remove("function_tool_options")
            .map(serde_json::from_value::<BTreeMap<String, OpenAiFunctionToolOptions>>)
            .transpose()
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?
            .unwrap_or_default()
            .into_iter()
            .map(|(name, options)| (name, options.into_encoding_options()))
            .collect();
        Ok(OpenAiResponsesRequestOptions {
            wire: value.into_iter().collect(),
            prompt_cache_breakpoints,
            native_tools,
            function_tools,
        })
    }
}

pub(crate) struct OpenAiResponsesRequestOptions {
    pub(crate) wire: BTreeMap<String, Value>,
    pub(crate) prompt_cache_breakpoints: Vec<OpenAiPromptCacheBreakpoint>,
    pub(crate) native_tools: Vec<Value>,
    pub(crate) function_tools: BTreeMap<String, FunctionToolEncodingOptions>,
}

impl TypedProviderOptions for OpenAiResponsesOptions {
    const NAMESPACE: &'static str = "openai";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(RESPONSES_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        self.validate_values()
    }
}

/// Typed OpenAI Chat Completions options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiChatCompletionsOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<BTreeMap<String, i16>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<OpenAiReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<OpenAiServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_verbosity: Option<OpenAiTextVerbosity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_options: Option<OpenAiPromptCacheOptions>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompt_cache_breakpoints: Vec<OpenAiPromptCacheBreakpoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
}

impl OpenAiChatCompletionsOptions {
    pub fn with_prompt_cache(mut self, cache: OpenAiPromptCacheOptions) -> Self {
        self.prompt_cache_options = Some(cache);
        self
    }

    pub fn with_prompt_cache_breakpoint(mut self, breakpoint: OpenAiPromptCacheBreakpoint) -> Self {
        self.prompt_cache_breakpoints.push(breakpoint);
        self
    }

    pub(crate) fn validate_values(&self) -> Result<(), ProviderOptionError> {
        validate_optional_text("user", self.user.as_deref())?;
        validate_optional_text("prompt_cache_key", self.prompt_cache_key.as_deref())?;
        validate_optional_text("safety_identifier", self.safety_identifier.as_deref())?;
        if self
            .top_logprobs
            .is_some_and(|value| value > MAX_TOP_LOGPROBS)
        {
            return Err(rejected("top_logprobs", "top_logprobs must not exceed 20"));
        }
        if self.top_logprobs.is_some() && self.logprobs != Some(true) {
            return Err(rejected(
                "top_logprobs",
                "top_logprobs requires logprobs=true",
            ));
        }
        if self
            .logit_bias
            .as_ref()
            .is_some_and(|biases| biases.values().any(|value| !(-100..=100).contains(value)))
        {
            return Err(rejected(
                "logit_bias",
                "logit bias values must be between -100 and 100",
            ));
        }
        validate_prompt_cache(
            self.prompt_cache_options.as_ref(),
            &self.prompt_cache_breakpoints,
        )?;
        Ok(())
    }

    pub(crate) fn into_request_options(
        self,
    ) -> Result<OpenAiChatCompletionsRequestOptions, ProviderOptionError> {
        let mut value = object_from(self)?;
        let prompt_cache_breakpoints = value
            .remove("prompt_cache_breakpoints")
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?
            .unwrap_or_default();
        if let Some(verbosity) = value.remove("text_verbosity") {
            value.insert("verbosity".to_string(), verbosity);
        }
        Ok(OpenAiChatCompletionsRequestOptions {
            wire: value.into_iter().collect(),
            prompt_cache_breakpoints,
        })
    }
}

pub(crate) struct OpenAiChatCompletionsRequestOptions {
    pub(crate) wire: BTreeMap<String, Value>,
    pub(crate) prompt_cache_breakpoints: Vec<OpenAiPromptCacheBreakpoint>,
}

impl TypedProviderOptions for OpenAiChatCompletionsOptions {
    const NAMESPACE: &'static str = "openai";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(CHAT_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        self.validate_values()
    }
}

fn validate_prompt_cache(
    _options: Option<&OpenAiPromptCacheOptions>,
    breakpoints: &[OpenAiPromptCacheBreakpoint],
) -> Result<(), ProviderOptionError> {
    if breakpoints.len() > MAX_PROMPT_CACHE_BREAKPOINTS {
        return Err(rejected(
            "prompt_cache_breakpoints",
            "OpenAI accepts at most 50 prompt-cache breakpoints per request",
        ));
    }
    if breakpoints.iter().copied().collect::<BTreeSet<_>>().len() != breakpoints.len() {
        return Err(rejected(
            "prompt_cache_breakpoints",
            "prompt-cache breakpoints must be unique",
        ));
    }
    Ok(())
}

fn validate_optional_text(
    path: &'static str,
    value: Option<&str>,
) -> Result<(), ProviderOptionError> {
    if value.is_some_and(|value| value.trim().is_empty() || value != value.trim()) {
        return Err(rejected(path, "value must be non-empty and trimmed"));
    }
    Ok(())
}

fn object_from<T: Serialize>(value: T) -> Result<Map<String, Value>, ProviderOptionError> {
    match serde_json::to_value(value)
        .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?
    {
        Value::Object(value) => Ok(value),
        _ => Err(ProviderOptionError::ExpectedObject {
            namespace: "openai".to_string(),
        }),
    }
}

fn validate_provider_tool(value: &Value) -> Result<(), ProviderOptionError> {
    let Some(object) = value.as_object() else {
        return Err(rejected(
            "native_tools",
            "OpenAI-native tools must be JSON objects",
        ));
    };
    let Some(kind) = object.get("type").and_then(Value::as_str) else {
        return Err(rejected(
            "native_tools.type",
            "OpenAI-native tools require a string type",
        ));
    };
    if kind.trim().is_empty()
        || kind != kind.trim()
        || kind.len() > 128
        || kind.chars().any(char::is_control)
    {
        return Err(rejected(
            "native_tools.type",
            "OpenAI-native tool type must be non-empty, trimmed, and at most 128 bytes",
        ));
    }
    Ok(())
}

fn validate_tool_name(name: &str) -> Result<(), ProviderOptionError> {
    if name.is_empty()
        || name.len() > 128
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(rejected(
            "function_tool_options",
            "function tool names must be 1..=128 ASCII letters, digits, '-' or '_'",
        ));
    }
    Ok(())
}

fn rejected(path: impl Into<String>, reason: impl Into<String>) -> ProviderOptionError {
    ProviderOptionError::Rejected {
        path: path.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_modes_and_effort_encode_independently() {
        for mode in [OpenAiReasoningMode::Standard, OpenAiReasoningMode::Pro] {
            let options = OpenAiResponsesOptions::default().with_reasoning(
                OpenAiReasoning::default()
                    .with_mode(mode)
                    .with_effort(OpenAiReasoningEffort::Low),
            );
            let value = serde_json::to_value(options).unwrap();
            assert_eq!(value["reasoning"]["effort"], "low");
            assert_eq!(
                value["reasoning"]["mode"],
                match mode {
                    OpenAiReasoningMode::Standard => "standard",
                    OpenAiReasoningMode::Pro => "pro",
                }
            );
        }
    }

    #[test]
    fn prompt_cache_uses_current_wire_shape() {
        let options =
            OpenAiResponsesOptions::default().with_prompt_cache(OpenAiPromptCacheOptions {
                mode: Some(OpenAiPromptCacheMode::Explicit),
                ttl: Some(OpenAiPromptCacheTtl::ThirtyMinutes),
            });
        let value = serde_json::to_value(options).unwrap();

        assert_eq!(
            value["prompt_cache_options"],
            serde_json::json!({"mode": "explicit", "ttl": "30m"})
        );
        assert!(value.get("prompt_cache_retention").is_none());
    }

    #[test]
    fn conversation_and_previous_response_are_exclusive() {
        let options = OpenAiResponsesOptions {
            conversation: Some("conv_1".to_string()),
            previous_response_id: Some("resp_1".to_string()),
            ..OpenAiResponsesOptions::default()
        };

        assert!(options.validate_values().is_err());
    }

    #[test]
    fn cache_breakpoint_helpers_preserve_implicit_request_mode() {
        let breakpoint = OpenAiPromptCacheBreakpoint::new(0, 0);
        let responses = OpenAiResponsesOptions::default().with_prompt_cache_breakpoint(breakpoint);
        let chat = OpenAiChatCompletionsOptions::default().with_prompt_cache_breakpoint(breakpoint);

        assert!(responses.prompt_cache_options.is_none());
        assert!(chat.prompt_cache_options.is_none());

        let implicit_with_ttl = OpenAiResponsesOptions {
            prompt_cache_options: Some(OpenAiPromptCacheOptions {
                mode: Some(OpenAiPromptCacheMode::Implicit),
                ttl: Some(OpenAiPromptCacheTtl::ThirtyMinutes),
            }),
            prompt_cache_breakpoints: vec![breakpoint],
            ..OpenAiResponsesOptions::default()
        };
        assert!(implicit_with_ttl.validate_values().is_ok());
    }

    #[test]
    fn cache_breakpoints_are_unique_and_bounded() {
        let duplicate = OpenAiResponsesOptions {
            prompt_cache_options: Some(OpenAiPromptCacheOptions::explicit()),
            prompt_cache_breakpoints: vec![
                OpenAiPromptCacheBreakpoint::new(0, 0),
                OpenAiPromptCacheBreakpoint::new(0, 0),
            ],
            ..OpenAiResponsesOptions::default()
        };
        assert!(duplicate.validate_values().is_err());

        let excessive = OpenAiChatCompletionsOptions {
            prompt_cache_options: Some(OpenAiPromptCacheOptions::explicit()),
            prompt_cache_breakpoints: (0..51)
                .map(|content_index| OpenAiPromptCacheBreakpoint::new(0, content_index))
                .collect(),
            ..OpenAiChatCompletionsOptions::default()
        };
        assert!(excessive.validate_values().is_err());
    }

    #[test]
    fn fast_service_tier_uses_current_wire_value() {
        assert_eq!(
            serde_json::to_value(OpenAiServiceTier::Fast).unwrap(),
            serde_json::json!("fast")
        );
    }
}
