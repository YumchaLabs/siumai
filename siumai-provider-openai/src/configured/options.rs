use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
};

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use siumai_core::{ModelFamily, ProviderOptionError, TypedProviderOptions};
use siumai_protocol_openai::chat_completions::API_MODE_ID as CHAT_API_MODE_ID;
use siumai_protocol_openai::responses::API_MODE_ID as RESPONSES_API_MODE_ID;
use siumai_protocol_openai::responses::{FunctionToolCaller, FunctionToolEncodingOptions};

use super::tools::{OpenAiResponsesTool, OpenAiToolCaller, OpenAiToolNamespace};

const MAX_TOP_LOGPROBS: u8 = 20;
const MAX_METADATA_ENTRIES: usize = 16;
const MAX_METADATA_KEY_CHARS: usize = 64;
const MAX_METADATA_VALUE_CHARS: usize = 512;
const MAX_PROMPT_CACHE_KEY_CHARS: usize = 64;
const MAX_REASONING_MODE_BYTES: usize = 512;
const MAX_RESOURCE_ID_BYTES: usize = 512;
const MAX_SAFETY_IDENTIFIER_CHARS: usize = 64;

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

/// OpenAI reasoning execution mode.
///
/// Known values are exposed as constants while checked custom values keep this
/// provider-owned option open to future protocol additions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct OpenAiReasoningMode(Cow<'static, str>);

impl OpenAiReasoningMode {
    /// The provider's standard reasoning execution mode.
    pub const STANDARD: Self = Self(Cow::Borrowed("standard"));
    /// The provider's higher-work reasoning execution mode.
    pub const PRO: Self = Self(Cow::Borrowed("pro"));

    /// Construct a checked current or future provider mode.
    pub fn new(value: impl Into<String>) -> Result<Self, ProviderOptionError> {
        let value = value.into();
        validate_reasoning_mode(&value)?;
        Ok(Self(Cow::Owned(value)))
    }

    /// Return the provider wire value.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for OpenAiReasoningMode {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for OpenAiReasoningMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl TryFrom<String> for OpenAiReasoningMode {
    type Error = ProviderOptionError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<&str> for OpenAiReasoningMode {
    type Error = ProviderOptionError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl std::str::FromStr for OpenAiReasoningMode {
    type Err = ProviderOptionError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for OpenAiReasoningMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
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

/// Maximum prompt-cache retention policy.
///
/// This legacy field is deprecated by OpenAI in favor of
/// [`OpenAiPromptCacheOptions::ttl`], but the two controls have independent
/// semantics and may be sent together when the selected model supports them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiPromptCacheRetention {
    InMemory,
    #[serde(rename = "24h")]
    TwentyFourHours,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<OpenAiToolNamespace>,
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

    pub fn with_namespace(mut self, namespace: OpenAiToolNamespace) -> Self {
        self.namespace = Some(namespace);
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
        if let Some(namespace) = &self.namespace {
            namespace.validate()?;
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
        if let Some(namespace) = self.namespace {
            options = options.with_namespace(namespace.name, namespace.description);
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
    #[serde(rename = "web_search_call.results")]
    WebSearchResults,
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
            Self::WebSearchResults => "web_search_call.results",
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
    Scale,
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
/// be overridden here. `tools` is the explicit provider-specific lane for OpenAI
/// Responses tools that have no portable `ToolSpec` representation.
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<OpenAiPromptCacheRetention>,
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
    pub tools: Vec<OpenAiResponsesTool>,
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

    pub fn with_tool(mut self, tool: OpenAiResponsesTool) -> Self {
        self.tools.push(tool);
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
        validate_optional_resource_id("conversation", self.conversation.as_deref())?;
        validate_optional_resource_id(
            "previous_response_id",
            self.previous_response_id.as_deref(),
        )?;
        validate_prompt_cache_key(self.prompt_cache_key.as_deref())?;
        validate_safety_identifier(self.safety_identifier.as_deref())?;
        validate_metadata(self.metadata.as_ref())?;
        if self
            .top_logprobs
            .is_some_and(|value| value > MAX_TOP_LOGPROBS)
        {
            return Err(rejected("top_logprobs", "top_logprobs must not exceed 20"));
        }
        for tool in &self.tools {
            tool.validate()?;
        }
        for (name, options) in &self.function_tool_options {
            options.validate(name)?;
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
        let native_tools = std::mem::take(&mut self.tools)
            .into_iter()
            .map(OpenAiResponsesTool::into_value)
            .collect::<Result<Vec<_>, _>>()?;
        let function_tools = std::mem::take(&mut self.function_tool_options)
            .into_iter()
            .map(|(name, options)| (name, options.into_encoding_options()))
            .collect();
        let value = object_from(self)?;
        debug_assert!(!value.contains_key("tools"));
        debug_assert!(!value.contains_key("function_tool_options"));
        Ok(OpenAiResponsesRequestOptions {
            wire: value.into_iter().collect(),
            native_tools,
            function_tools,
        })
    }
}

pub(crate) struct OpenAiResponsesRequestOptions {
    pub(crate) wire: BTreeMap<String, Value>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<OpenAiPromptCacheRetention>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
}

impl OpenAiChatCompletionsOptions {
    pub fn with_prompt_cache(mut self, cache: OpenAiPromptCacheOptions) -> Self {
        self.prompt_cache_options = Some(cache);
        self
    }

    pub(crate) fn validate_values(&self) -> Result<(), ProviderOptionError> {
        validate_prompt_cache_key(self.prompt_cache_key.as_deref())?;
        validate_safety_identifier(self.safety_identifier.as_deref())?;
        validate_metadata(self.metadata.as_ref())?;
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
        Ok(())
    }

    pub(crate) fn into_request_options(
        self,
    ) -> Result<OpenAiChatCompletionsRequestOptions, ProviderOptionError> {
        let mut value = object_from(self)?;
        if let Some(verbosity) = value.remove("text_verbosity") {
            value.insert("verbosity".to_string(), verbosity);
        }
        Ok(OpenAiChatCompletionsRequestOptions {
            wire: value.into_iter().collect(),
        })
    }
}

pub(crate) struct OpenAiChatCompletionsRequestOptions {
    pub(crate) wire: BTreeMap<String, Value>,
}

impl TypedProviderOptions for OpenAiChatCompletionsOptions {
    const NAMESPACE: &'static str = "openai";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(CHAT_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        self.validate_values()
    }
}

fn validate_metadata(
    metadata: Option<&BTreeMap<String, String>>,
) -> Result<(), ProviderOptionError> {
    let Some(metadata) = metadata else {
        return Ok(());
    };
    if metadata.len() > MAX_METADATA_ENTRIES {
        return Err(rejected("metadata", "metadata must not exceed 16 entries"));
    }
    for (key, value) in metadata {
        if key.is_empty() || key.chars().count() > MAX_METADATA_KEY_CHARS {
            return Err(rejected(
                "metadata",
                "metadata keys must contain 1..=64 characters",
            ));
        }
        if value.chars().count() > MAX_METADATA_VALUE_CHARS {
            return Err(rejected(
                "metadata",
                "metadata values must not exceed 512 characters",
            ));
        }
    }
    Ok(())
}

fn validate_safety_identifier(value: Option<&str>) -> Result<(), ProviderOptionError> {
    if value.is_some_and(|value| {
        value.trim().is_empty()
            || value != value.trim()
            || value.chars().count() > MAX_SAFETY_IDENTIFIER_CHARS
            || value.chars().any(char::is_control)
    }) {
        return Err(rejected(
            "safety_identifier",
            "safety_identifier must not contain control characters or exceed 64 characters",
        ));
    }
    Ok(())
}

fn validate_prompt_cache_key(value: Option<&str>) -> Result<(), ProviderOptionError> {
    if value.is_some_and(|value| {
        value.chars().count() > MAX_PROMPT_CACHE_KEY_CHARS || value.chars().any(char::is_control)
    }) {
        return Err(rejected(
            "prompt_cache_key",
            "prompt_cache_key must not contain control characters or exceed 64 characters",
        ));
    }
    Ok(())
}

fn validate_reasoning_mode(value: &str) -> Result<(), ProviderOptionError> {
    if value.is_empty()
        || value.len() > MAX_REASONING_MODE_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(rejected(
            "reasoning.mode",
            "reasoning mode must be non-empty, at most 512 bytes, and contain no control characters",
        ));
    }
    Ok(())
}

fn validate_optional_resource_id(
    path: &'static str,
    value: Option<&str>,
) -> Result<(), ProviderOptionError> {
    if value.is_some_and(|value| {
        value.is_empty()
            || value.len() > MAX_RESOURCE_ID_BYTES
            || value.chars().any(char::is_control)
    }) {
        return Err(rejected(
            path,
            "resource identifier must be non-empty, bounded, and contain no control characters",
        ));
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
    fn reasoning_modes_round_trip_known_and_future_values() {
        for mode in [
            OpenAiReasoningMode::STANDARD,
            OpenAiReasoningMode::PRO,
            OpenAiReasoningMode::new("deliberate_v2").unwrap(),
        ] {
            let options = OpenAiResponsesOptions::default().with_reasoning(
                OpenAiReasoning::default()
                    .with_mode(mode.clone())
                    .with_effort(OpenAiReasoningEffort::Low),
            );
            let value = serde_json::to_value(&options).unwrap();
            assert_eq!(value["reasoning"]["effort"], "low");
            assert_eq!(value["reasoning"]["mode"], mode.as_str());

            let decoded: OpenAiResponsesOptions = serde_json::from_value(value).unwrap();
            assert_eq!(decoded.reasoning.unwrap().mode, Some(mode));
        }
    }

    #[test]
    fn reasoning_mode_rejects_invalid_values_without_disclosing_them() {
        for invalid in [
            String::new(),
            "private\nmode".to_string(),
            "x".repeat(MAX_REASONING_MODE_BYTES + 1),
        ] {
            let error = OpenAiReasoningMode::new(invalid.clone()).unwrap_err();
            let message = error.to_string();
            assert!(message.contains("reasoning.mode"));
            if !invalid.is_empty() {
                assert!(!message.contains(&invalid));
            }

            let encoded = serde_json::to_string(&invalid).unwrap();
            let error = serde_json::from_str::<OpenAiReasoningMode>(&encoded).unwrap_err();
            if !invalid.is_empty() {
                assert!(!error.to_string().contains(&invalid));
            }
        }
    }

    #[test]
    fn prompt_cache_uses_current_wire_shape() {
        let options = OpenAiResponsesOptions {
            prompt_cache_retention: Some(OpenAiPromptCacheRetention::TwentyFourHours),
            ..OpenAiResponsesOptions::default()
        };
        let value = serde_json::to_value(options).unwrap();
        assert_eq!(value["prompt_cache_retention"], "24h");

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

        let opaque = OpenAiResponsesOptions {
            conversation: Some("conv/part?view#fragment%2F资源".to_string()),
            ..OpenAiResponsesOptions::default()
        };
        assert!(opaque.validate_values().is_ok());
    }

    #[test]
    fn cache_lifetime_metadata_and_safety_values_fail_closed() {
        let independent_cache_controls = OpenAiResponsesOptions {
            prompt_cache_options: Some(OpenAiPromptCacheOptions::explicit_30_minutes()),
            prompt_cache_retention: Some(OpenAiPromptCacheRetention::TwentyFourHours),
            ..OpenAiResponsesOptions::default()
        };
        assert!(independent_cache_controls.validate_values().is_ok());
        let value = serde_json::to_value(independent_cache_controls).unwrap();
        assert_eq!(value["prompt_cache_options"]["ttl"], "30m");
        assert_eq!(value["prompt_cache_retention"], "24h");

        let excessive_metadata = OpenAiChatCompletionsOptions {
            metadata: Some(
                (0..17)
                    .map(|index| (format!("key-{index}"), "value".to_string()))
                    .collect(),
            ),
            ..OpenAiChatCompletionsOptions::default()
        };
        assert!(excessive_metadata.validate_values().is_err());

        let invalid_safety = OpenAiResponsesOptions {
            safety_identifier: Some("user\nidentifier".to_string()),
            ..OpenAiResponsesOptions::default()
        };
        assert!(invalid_safety.validate_values().is_err());

        let oversized_safety = OpenAiResponsesOptions {
            safety_identifier: Some("x".repeat(65)),
            ..OpenAiResponsesOptions::default()
        };
        assert!(oversized_safety.validate_values().is_err());

        let oversized_cache_key = OpenAiChatCompletionsOptions {
            prompt_cache_key: Some("x".repeat(65)),
            ..OpenAiChatCompletionsOptions::default()
        };
        assert!(oversized_cache_key.validate_values().is_err());

        let control_cache_key = OpenAiResponsesOptions {
            prompt_cache_key: Some("cache\nkey".to_string()),
            ..OpenAiResponsesOptions::default()
        };
        assert!(control_cache_key.validate_values().is_err());
    }

    #[test]
    fn current_response_include_service_tier_and_zero_values_are_typed() {
        assert_eq!(
            serde_json::to_value(OpenAiServiceTier::Fast).unwrap(),
            serde_json::json!("fast")
        );
        assert_eq!(
            serde_json::to_value(OpenAiServiceTier::Scale).unwrap(),
            serde_json::json!("scale")
        );
        assert_eq!(
            serde_json::to_value(OpenAiResponseInclude::WebSearchResults).unwrap(),
            serde_json::json!("web_search_call.results")
        );

        let zero_values = OpenAiResponsesOptions {
            instructions: Some(String::new()),
            max_tool_calls: Some(0),
            prompt_cache_key: Some(String::new()),
            user: Some(String::new()),
            context_management: vec![OpenAiContextManagement::compaction(0)],
            ..OpenAiResponsesOptions::default()
        };
        assert!(zero_values.validate_values().is_ok());
        let value = serde_json::to_value(zero_values).unwrap();
        assert_eq!(value["instructions"], "");
        assert_eq!(value["max_tool_calls"], 0);
        assert_eq!(value["prompt_cache_key"], "");
        assert_eq!(value["user"], "");
        assert_eq!(value["context_management"][0]["compact_threshold"], 0);

        let chat = OpenAiChatCompletionsOptions {
            prompt_cache_key: Some("  ".to_string()),
            user: Some(String::new()),
            ..OpenAiChatCompletionsOptions::default()
        };
        assert!(chat.validate_values().is_ok());
    }
}
