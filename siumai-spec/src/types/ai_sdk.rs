//! AI SDK-aligned shared surface aliases and metadata helpers.
//!
//! These names intentionally mirror the shared `packages/ai/src/types/*` contract where
//! Siumai already has a stable equivalent or can expose a passive data structure honestly
//! without pretending the runtime wiring is more complete than it is today.

use super::chat::{ContentPart, SourcePart};
use super::{
    EmbeddingUsage, HttpRequestInfo, HttpResponseInfo, ProviderMetadataMap, ProviderOptionsMap,
    ResponseMetadata, ToolResultOutput, Usage, Warning,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::time::Duration;
use tokio_util::sync::{CancellationToken, WaitForCancellationFuture};

/// AI SDK-style JSON value alias.
pub type JSONValue = serde_json::Value;

/// AI SDK-style JSON Schema draft-07 value alias.
pub type JSONSchema7 = serde_json::Value;

/// AI SDK-style shared warning alias.
pub type CallWarning = Warning;

/// AI SDK-style shared provider-metadata root.
pub type ProviderMetadata = ProviderMetadataMap;

/// AI SDK-style shared provider-options root.
pub type ProviderOptions = ProviderOptionsMap;

/// AI SDK-style shared execution context object.
pub type Context = HashMap<String, JSONValue>;

/// AI SDK-style single embedding vector.
pub type Embedding = Vec<f32>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceMarker {
    Source,
}

impl Default for SourceMarker {
    fn default() -> Self {
        Self::Source
    }
}

impl Serialize for SourceMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("source")
    }
}

impl<'de> Deserialize<'de> for SourceMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "source" {
            Ok(Self::Source)
        } else {
            Err(serde::de::Error::custom(format!(
                "expected source type marker `source`, got `{value}`"
            )))
        }
    }
}

/// AI SDK-style source citation used by language-model responses.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Source {
    /// Fixed AI SDK type marker. Serialized as `type: "source"`.
    #[serde(rename = "type", default)]
    kind: SourceMarker,
    /// Source id.
    pub id: String,
    /// Strict URL/document source union.
    #[serde(flatten)]
    pub source: SourcePart,
    /// Additional provider metadata for the source.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl Source {
    /// Create a URL-backed source without a title.
    pub fn url(id: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            kind: SourceMarker::Source,
            id: id.into(),
            source: SourcePart::Url {
                url: url.into(),
                title: None,
            },
            provider_metadata: None,
        }
    }

    /// Create a URL-backed source with a title.
    pub fn url_with_title(
        id: impl Into<String>,
        url: impl Into<String>,
        title: impl Into<String>,
    ) -> Self {
        Self {
            kind: SourceMarker::Source,
            id: id.into(),
            source: SourcePart::Url {
                url: url.into(),
                title: Some(title.into()),
            },
            provider_metadata: None,
        }
    }

    /// Create a document-backed source.
    pub fn document(
        id: impl Into<String>,
        media_type: impl Into<String>,
        title: impl Into<String>,
        filename: Option<String>,
    ) -> Self {
        Self {
            kind: SourceMarker::Source,
            id: id.into(),
            source: SourcePart::Document {
                media_type: media_type.into(),
                title: title.into(),
                filename,
            },
            provider_metadata: None,
        }
    }

    /// Return the fixed AI SDK source marker.
    pub const fn r#type(&self) -> &'static str {
        "source"
    }

    /// Return the source-type discriminator.
    pub fn source_type(&self) -> &'static str {
        self.source.source_type()
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }
}

impl From<Source> for ContentPart {
    fn from(value: Source) -> Self {
        Self::Source {
            id: value.id,
            source: value.source,
            provider_metadata: value.provider_metadata,
        }
    }
}

impl TryFrom<ContentPart> for Source {
    type Error = ContentPart;

    fn try_from(value: ContentPart) -> Result<Self, Self::Error> {
        match value {
            ContentPart::Source {
                id,
                source,
                provider_metadata,
            } => Ok(Self {
                kind: SourceMarker::Source,
                id,
                source,
                provider_metadata,
            }),
            other => Err(other),
        }
    }
}

impl TryFrom<&ContentPart> for Source {
    type Error = &'static str;

    fn try_from(value: &ContentPart) -> Result<Self, Self::Error> {
        match value {
            ContentPart::Source {
                id,
                source,
                provider_metadata,
            } => Ok(Self {
                kind: SourceMarker::Source,
                id: id.clone(),
                source: source.clone(),
                provider_metadata: provider_metadata.clone(),
            }),
            _ => Err("content part is not a source"),
        }
    }
}

/// AI SDK-style shared image-provider metadata root.
pub type ImageModelProviderMetadata = ProviderMetadata;

/// AI SDK-style shared video-provider metadata root.
pub type VideoModelProviderMetadata = ProviderMetadata;

/// AI SDK-style typed tool call view returned by higher-level text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall<NAME = String, INPUT = JSONValue> {
    /// ID of the tool call. This ID is used to match the tool call with the tool result.
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    /// Name of the tool being called.
    #[serde(rename = "toolName")]
    pub tool_name: NAME,
    /// Tool arguments.
    pub input: INPUT,
    /// Whether the tool call will be executed by the provider.
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
    /// Whether the tool is dynamic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
}

impl<NAME, INPUT> ToolCall<NAME, INPUT> {
    /// Create a typed tool call.
    pub fn new(tool_call_id: impl Into<String>, tool_name: NAME, input: INPUT) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name,
            input,
            provider_executed: None,
            dynamic: None,
        }
    }

    /// Mark whether the tool call is provider-executed.
    pub const fn with_provider_executed(mut self, provider_executed: bool) -> Self {
        self.provider_executed = Some(provider_executed);
        self
    }

    /// Mark whether the tool call is dynamic.
    pub const fn with_dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = Some(dynamic);
        self
    }
}

/// AI SDK-style typed tool result view returned by higher-level text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResult<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> {
    /// ID of the tool call. This ID is used to match the tool call with the tool result.
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    /// Name of the tool that was called.
    #[serde(rename = "toolName")]
    pub tool_name: NAME,
    /// Tool input.
    pub input: INPUT,
    /// Tool output.
    pub output: OUTPUT,
    /// Whether the tool result was executed by the provider.
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
    /// Whether the tool is dynamic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
}

impl<NAME, INPUT, OUTPUT> ToolResult<NAME, INPUT, OUTPUT> {
    /// Create a typed tool result.
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: NAME,
        input: INPUT,
        output: OUTPUT,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name,
            input,
            output,
            provider_executed: None,
            dynamic: None,
        }
    }

    /// Mark whether the tool result is provider-executed.
    pub const fn with_provider_executed(mut self, provider_executed: bool) -> Self {
        self.provider_executed = Some(provider_executed);
        self
    }

    /// Mark whether the tool is dynamic.
    pub const fn with_dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = Some(dynamic);
        self
    }
}

/// A cloneable cancellation handle for request-scoped abort semantics.
#[derive(Clone, Debug, Default)]
pub struct CancelHandle {
    token: CancellationToken,
}

impl CancelHandle {
    /// Create a new cancellation handle.
    pub fn new() -> Self {
        Self {
            token: CancellationToken::new(),
        }
    }

    /// Request cancellation.
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Whether cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    /// Future that resolves when cancellation is requested.
    pub fn cancelled(&self) -> WaitForCancellationFuture<'_> {
        self.token.cancelled()
    }

    /// Clone the underlying cancellation token for integrations that need it directly.
    pub fn token(&self) -> CancellationToken {
        self.token.clone()
    }
}

/// Structured timeout configuration details for request-facing controls.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct TimeoutConfigurationSettings {
    /// Total timeout in milliseconds.
    #[serde(rename = "totalMs", skip_serializing_if = "Option::is_none")]
    pub total_ms: Option<u64>,
    /// Per-step timeout in milliseconds.
    #[serde(rename = "stepMs", skip_serializing_if = "Option::is_none")]
    pub step_ms: Option<u64>,
    /// Timeout between stream chunks in milliseconds.
    #[serde(rename = "chunkMs", skip_serializing_if = "Option::is_none")]
    pub chunk_ms: Option<u64>,
    /// Default timeout for all tools in milliseconds.
    #[serde(rename = "toolMs", skip_serializing_if = "Option::is_none")]
    pub tool_ms: Option<u64>,
    /// Per-tool timeout overrides keyed by the AI SDK-style `{toolName}Ms` suffix form.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub tools: HashMap<String, u64>,
}

impl TimeoutConfigurationSettings {
    /// Create empty timeout settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set total timeout in milliseconds.
    pub const fn with_total_ms(mut self, total_ms: u64) -> Self {
        self.total_ms = Some(total_ms);
        self
    }

    /// Set per-step timeout in milliseconds.
    pub const fn with_step_ms(mut self, step_ms: u64) -> Self {
        self.step_ms = Some(step_ms);
        self
    }

    /// Set between-chunk timeout in milliseconds.
    pub const fn with_chunk_ms(mut self, chunk_ms: u64) -> Self {
        self.chunk_ms = Some(chunk_ms);
        self
    }

    /// Set default tool timeout in milliseconds.
    pub const fn with_tool_ms(mut self, tool_ms: u64) -> Self {
        self.tool_ms = Some(tool_ms);
        self
    }

    /// Set a tool-specific timeout in milliseconds.
    pub fn with_tool_timeout_ms(mut self, tool_name: impl AsRef<str>, timeout_ms: u64) -> Self {
        self.tools
            .insert(format!("{}Ms", tool_name.as_ref()), timeout_ms);
        self
    }

    /// Resolve the tool timeout for the given tool name.
    pub fn tool_timeout_ms(&self, tool_name: &str) -> Option<u64> {
        self.tools
            .get(&format!("{tool_name}Ms"))
            .copied()
            .or(self.tool_ms)
    }
}

/// AI SDK-style timeout configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum TimeoutConfiguration {
    /// A single total timeout in milliseconds.
    Millis(u64),
    /// Structured timeout configuration.
    Settings(TimeoutConfigurationSettings),
}

impl TimeoutConfiguration {
    /// Create a total-timeout configuration from milliseconds.
    pub const fn from_millis(total_ms: u64) -> Self {
        Self::Millis(total_ms)
    }

    /// Create a structured timeout configuration.
    pub fn settings(settings: TimeoutConfigurationSettings) -> Self {
        Self::Settings(settings)
    }

    /// Total timeout in milliseconds.
    pub const fn total_timeout_ms(&self) -> Option<u64> {
        match self {
            Self::Millis(total_ms) => Some(*total_ms),
            Self::Settings(settings) => settings.total_ms,
        }
    }

    /// Per-step timeout in milliseconds.
    pub const fn step_timeout_ms(&self) -> Option<u64> {
        match self {
            Self::Millis(_) => None,
            Self::Settings(settings) => settings.step_ms,
        }
    }

    /// Per-chunk timeout in milliseconds.
    pub const fn chunk_timeout_ms(&self) -> Option<u64> {
        match self {
            Self::Millis(_) => None,
            Self::Settings(settings) => settings.chunk_ms,
        }
    }

    /// Per-tool timeout in milliseconds.
    pub fn tool_timeout_ms(&self, tool_name: &str) -> Option<u64> {
        match self {
            Self::Millis(_) => None,
            Self::Settings(settings) => settings.tool_timeout_ms(tool_name),
        }
    }

    /// Total timeout as a `Duration`.
    pub fn total_timeout(&self) -> Option<Duration> {
        self.total_timeout_ms().map(Duration::from_millis)
    }

    /// Step timeout as a `Duration`.
    pub fn step_timeout(&self) -> Option<Duration> {
        self.step_timeout_ms().map(Duration::from_millis)
    }

    /// Chunk timeout as a `Duration`.
    pub fn chunk_timeout(&self) -> Option<Duration> {
        self.chunk_timeout_ms().map(Duration::from_millis)
    }
}

/// AI SDK-style helper for extracting the total timeout in milliseconds.
pub const fn get_total_timeout_ms(timeout: Option<&TimeoutConfiguration>) -> Option<u64> {
    match timeout {
        Some(timeout) => timeout.total_timeout_ms(),
        None => None,
    }
}

/// AI SDK-style helper for extracting the step timeout in milliseconds.
pub const fn get_step_timeout_ms(timeout: Option<&TimeoutConfiguration>) -> Option<u64> {
    match timeout {
        Some(timeout) => timeout.step_timeout_ms(),
        None => None,
    }
}

/// AI SDK-style helper for extracting the chunk timeout in milliseconds.
pub const fn get_chunk_timeout_ms(timeout: Option<&TimeoutConfiguration>) -> Option<u64> {
    match timeout {
        Some(timeout) => timeout.chunk_timeout_ms(),
        None => None,
    }
}

/// AI SDK-style helper for extracting one tool timeout in milliseconds.
pub fn get_tool_timeout_ms(timeout: Option<&TimeoutConfiguration>, tool_name: &str) -> Option<u64> {
    timeout.and_then(|timeout| timeout.tool_timeout_ms(tool_name))
}

/// AI SDK-style request-facing transport controls.
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    /// Maximum number of retries. `0` disables retries.
    pub max_retries: Option<u32>,
    /// Request-scoped abort signal.
    pub abort_signal: Option<CancelHandle>,
    /// Additional HTTP headers. `None` values are filtered when materialized.
    pub headers: HashMap<String, Option<String>>,
    /// Timeout configuration.
    pub timeout: Option<TimeoutConfiguration>,
}

impl RequestOptions {
    /// Create empty request options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max retries.
    pub const fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    /// Set the abort signal.
    pub fn with_abort_signal(mut self, abort_signal: CancelHandle) -> Self {
        self.abort_signal = Some(abort_signal);
        self
    }

    /// Set a request header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), Some(value.into()));
        self
    }

    /// Mark a header as intentionally omitted.
    pub fn without_header(mut self, key: impl Into<String>) -> Self {
        self.headers.insert(key.into(), None);
        self
    }

    /// Set timeout configuration.
    pub fn with_timeout(mut self, timeout: TimeoutConfiguration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Materialize only the headers with concrete values.
    pub fn effective_headers(&self) -> HashMap<String, String> {
        self.headers
            .iter()
            .filter_map(|(key, value)| value.clone().map(|value| (key.clone(), value)))
            .collect()
    }

    /// Total timeout as a `Duration`.
    pub fn total_timeout(&self) -> Option<Duration> {
        self.timeout
            .as_ref()
            .and_then(TimeoutConfiguration::total_timeout)
    }

    /// Step timeout as a `Duration`.
    pub fn step_timeout(&self) -> Option<Duration> {
        self.timeout
            .as_ref()
            .and_then(TimeoutConfiguration::step_timeout)
    }

    /// Chunk timeout as a `Duration`.
    pub fn chunk_timeout(&self) -> Option<Duration> {
        self.timeout
            .as_ref()
            .and_then(TimeoutConfiguration::chunk_timeout)
    }

    /// Per-tool timeout in milliseconds.
    pub fn tool_timeout_ms(&self, tool_name: &str) -> Option<u64> {
        self.timeout
            .as_ref()
            .and_then(|timeout| timeout.tool_timeout_ms(tool_name))
    }

    /// Convert retries into total attempts, where `0` retries means `1` attempt.
    pub fn max_attempts(&self) -> Option<u32> {
        self.max_retries.map(|retries| retries.saturating_add(1))
    }
}

/// AI SDK-style reasoning level for language-model call options.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum LanguageModelReasoning {
    /// Use the provider's default reasoning level.
    ProviderDefault,
    /// Disable reasoning when supported.
    None,
    /// Minimal reasoning.
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
    /// Extra-high reasoning effort.
    #[serde(rename = "xhigh")]
    XHigh,
}

/// AI SDK-style model-facing generation controls.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct LanguageModelCallOptions {
    /// Maximum output tokens requested by the caller.
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling.
    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-k sampling.
    #[serde(rename = "topK", skip_serializing_if = "Option::is_none")]
    pub top_k: Option<f64>,
    /// Presence penalty.
    #[serde(rename = "presencePenalty", skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Frequency penalty.
    #[serde(rename = "frequencyPenalty", skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Stop sequences.
    #[serde(rename = "stopSequences", skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Deterministic seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Cross-provider reasoning level.
    ///
    /// Siumai does not yet have a stable cross-provider request lane for this field, so
    /// `From<CommonParams>` leaves it empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<LanguageModelReasoning>,
}

impl From<super::CommonParams> for LanguageModelCallOptions {
    fn from(value: super::CommonParams) -> Self {
        Self::from(&value)
    }
}

impl From<&super::CommonParams> for LanguageModelCallOptions {
    fn from(value: &super::CommonParams) -> Self {
        Self {
            max_output_tokens: value.max_completion_tokens.or(value.max_tokens),
            temperature: value.temperature,
            top_p: value.top_p,
            top_k: value.top_k,
            presence_penalty: value.presence_penalty,
            frequency_penalty: value.frequency_penalty,
            stop_sequences: value.stop_sequences.clone(),
            seed: value.seed,
            reasoning: None,
        }
    }
}

/// Deprecated AI SDK-style combined call settings view.
///
/// Prefer using `LanguageModelCallOptions` together with `RequestOptions`.
#[deprecated(note = "Use `LanguageModelCallOptions` together with `RequestOptions` instead.")]
#[derive(Debug, Clone, Default)]
pub struct CallSettings {
    /// Maximum output tokens requested by the caller.
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Nucleus sampling.
    pub top_p: Option<f64>,
    /// Top-k sampling.
    pub top_k: Option<f64>,
    /// Presence penalty.
    pub presence_penalty: Option<f64>,
    /// Frequency penalty.
    pub frequency_penalty: Option<f64>,
    /// Stop sequences.
    pub stop_sequences: Option<Vec<String>>,
    /// Deterministic seed.
    pub seed: Option<u64>,
    /// Cross-provider reasoning level.
    pub reasoning: Option<LanguageModelReasoning>,
    /// Maximum number of retries. `0` disables retries.
    pub max_retries: Option<u32>,
    /// Request-scoped abort signal.
    pub abort_signal: Option<CancelHandle>,
    /// Additional HTTP headers. `None` values are filtered when materialized.
    pub headers: HashMap<String, Option<String>>,
}

#[allow(deprecated)]
impl CallSettings {
    /// Create empty call settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max output tokens.
    pub const fn with_max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    /// Set temperature.
    pub const fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set nucleus sampling.
    pub const fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set top-k sampling.
    pub const fn with_top_k(mut self, top_k: f64) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set presence penalty.
    pub const fn with_presence_penalty(mut self, presence_penalty: f64) -> Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }

    /// Set frequency penalty.
    pub const fn with_frequency_penalty(mut self, frequency_penalty: f64) -> Self {
        self.frequency_penalty = Some(frequency_penalty);
        self
    }

    /// Set stop sequences.
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(stop_sequences);
        self
    }

    /// Set deterministic seed.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set reasoning level.
    pub fn with_reasoning(mut self, reasoning: LanguageModelReasoning) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Set max retries.
    pub const fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    /// Set the abort signal.
    pub fn with_abort_signal(mut self, abort_signal: CancelHandle) -> Self {
        self.abort_signal = Some(abort_signal);
        self
    }

    /// Set a request header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), Some(value.into()));
        self
    }

    /// Mark a header as intentionally omitted.
    pub fn without_header(mut self, key: impl Into<String>) -> Self {
        self.headers.insert(key.into(), None);
        self
    }

    /// Project onto `LanguageModelCallOptions`.
    pub fn language_model_call_options(&self) -> LanguageModelCallOptions {
        self.into()
    }

    /// Project onto `RequestOptions`.
    pub fn request_options(&self) -> RequestOptions {
        self.into()
    }

    /// Materialize only the headers with concrete values.
    pub fn effective_headers(&self) -> HashMap<String, String> {
        self.headers
            .iter()
            .filter_map(|(key, value)| value.clone().map(|value| (key.clone(), value)))
            .collect()
    }

    /// Convert retries into total attempts, where `0` retries means `1` attempt.
    pub fn max_attempts(&self) -> Option<u32> {
        self.max_retries.map(|retries| retries.saturating_add(1))
    }
}

#[allow(deprecated)]
impl From<LanguageModelCallOptions> for CallSettings {
    fn from(value: LanguageModelCallOptions) -> Self {
        Self {
            max_output_tokens: value.max_output_tokens,
            temperature: value.temperature,
            top_p: value.top_p,
            top_k: value.top_k,
            presence_penalty: value.presence_penalty,
            frequency_penalty: value.frequency_penalty,
            stop_sequences: value.stop_sequences,
            seed: value.seed,
            reasoning: value.reasoning,
            max_retries: None,
            abort_signal: None,
            headers: HashMap::new(),
        }
    }
}

#[allow(deprecated)]
impl From<RequestOptions> for CallSettings {
    fn from(value: RequestOptions) -> Self {
        Self {
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            presence_penalty: None,
            frequency_penalty: None,
            stop_sequences: None,
            seed: None,
            reasoning: None,
            max_retries: value.max_retries,
            abort_signal: value.abort_signal,
            headers: value.headers,
        }
    }
}

#[allow(deprecated)]
impl From<&super::CommonParams> for CallSettings {
    fn from(value: &super::CommonParams) -> Self {
        LanguageModelCallOptions::from(value).into()
    }
}

#[allow(deprecated)]
impl From<super::CommonParams> for CallSettings {
    fn from(value: super::CommonParams) -> Self {
        Self::from(&value)
    }
}

#[allow(deprecated)]
impl From<&CallSettings> for LanguageModelCallOptions {
    fn from(value: &CallSettings) -> Self {
        Self {
            max_output_tokens: value.max_output_tokens,
            temperature: value.temperature,
            top_p: value.top_p,
            top_k: value.top_k,
            presence_penalty: value.presence_penalty,
            frequency_penalty: value.frequency_penalty,
            stop_sequences: value.stop_sequences.clone(),
            seed: value.seed,
            reasoning: value.reasoning.clone(),
        }
    }
}

#[allow(deprecated)]
impl From<CallSettings> for LanguageModelCallOptions {
    fn from(value: CallSettings) -> Self {
        Self::from(&value)
    }
}

#[allow(deprecated)]
impl From<&CallSettings> for RequestOptions {
    fn from(value: &CallSettings) -> Self {
        RequestOptions {
            max_retries: value.max_retries,
            abort_signal: value.abort_signal.clone(),
            headers: value.headers.clone(),
            timeout: None,
        }
    }
}

#[allow(deprecated)]
impl From<CallSettings> for RequestOptions {
    fn from(value: CallSettings) -> Self {
        Self::from(&value)
    }
}

/// AI SDK-style language-model input token detail shape.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct LanguageModelInputTokenDetails {
    /// Non-cached input tokens.
    #[serde(rename = "noCacheTokens", skip_serializing_if = "Option::is_none")]
    pub no_cache_tokens: Option<u32>,
    /// Cached input tokens read.
    #[serde(rename = "cacheReadTokens", skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    /// Cached input tokens written.
    #[serde(rename = "cacheWriteTokens", skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u32>,
}

impl LanguageModelInputTokenDetails {
    fn is_empty(&self) -> bool {
        self.no_cache_tokens.is_none()
            && self.cache_read_tokens.is_none()
            && self.cache_write_tokens.is_none()
    }
}

/// AI SDK-style language-model output token detail shape.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct LanguageModelOutputTokenDetails {
    /// Text/output-visible tokens.
    #[serde(rename = "textTokens", skip_serializing_if = "Option::is_none")]
    pub text_tokens: Option<u32>,
    /// Reasoning tokens.
    #[serde(rename = "reasoningTokens", skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

impl LanguageModelOutputTokenDetails {
    fn is_empty(&self) -> bool {
        self.text_tokens.is_none() && self.reasoning_tokens.is_none()
    }
}

/// AI SDK-style language-model usage shape.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct LanguageModelUsage {
    /// Total input tokens.
    #[serde(rename = "inputTokens", skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    /// Detailed input-token breakdown.
    #[serde(
        rename = "inputTokenDetails",
        default,
        skip_serializing_if = "LanguageModelInputTokenDetails::is_empty"
    )]
    pub input_token_details: LanguageModelInputTokenDetails,
    /// Total output tokens.
    #[serde(rename = "outputTokens", skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    /// Detailed output-token breakdown.
    #[serde(
        rename = "outputTokenDetails",
        default,
        skip_serializing_if = "LanguageModelOutputTokenDetails::is_empty"
    )]
    pub output_token_details: LanguageModelOutputTokenDetails,
    /// Total tokens across input and output.
    #[serde(rename = "totalTokens", skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
    /// Deprecated reasoning-token compatibility alias.
    #[serde(rename = "reasoningTokens", skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    /// Deprecated cached-input-token compatibility alias.
    #[serde(rename = "cachedInputTokens", skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u32>,
    /// Raw provider usage payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Map<String, JSONValue>>,
}

impl LanguageModelUsage {
    /// Merge another language-model usage payload into this one.
    pub fn merge(&mut self, other: &Self) {
        self.input_tokens = add_optional_u32(self.input_tokens, other.input_tokens);
        self.input_token_details.no_cache_tokens = add_optional_u32(
            self.input_token_details.no_cache_tokens,
            other.input_token_details.no_cache_tokens,
        );
        self.input_token_details.cache_read_tokens = add_optional_u32(
            self.input_token_details.cache_read_tokens,
            other.input_token_details.cache_read_tokens,
        );
        self.input_token_details.cache_write_tokens = add_optional_u32(
            self.input_token_details.cache_write_tokens,
            other.input_token_details.cache_write_tokens,
        );
        self.output_tokens = add_optional_u32(self.output_tokens, other.output_tokens);
        self.output_token_details.text_tokens = add_optional_u32(
            self.output_token_details.text_tokens,
            other.output_token_details.text_tokens,
        );
        self.output_token_details.reasoning_tokens = add_optional_u32(
            self.output_token_details.reasoning_tokens,
            other.output_token_details.reasoning_tokens,
        );
        self.total_tokens = add_optional_u32(self.total_tokens, other.total_tokens);
        self.reasoning_tokens = add_optional_u32(self.reasoning_tokens, other.reasoning_tokens);
        self.cached_input_tokens =
            add_optional_u32(self.cached_input_tokens, other.cached_input_tokens);
        self.raw = None;
    }
}

/// AI SDK-style helper for creating an empty language-model usage payload.
pub fn create_null_language_model_usage() -> LanguageModelUsage {
    LanguageModelUsage::default()
}

/// AI SDK-style helper for adding two language-model usage payloads.
///
/// Raw provider usage is intentionally dropped on the aggregated result, matching the
/// upstream helper's normalized aggregate shape.
pub fn add_language_model_usage(
    usage1: &LanguageModelUsage,
    usage2: &LanguageModelUsage,
) -> LanguageModelUsage {
    let mut merged = usage1.clone();
    merged.merge(usage2);
    merged
}

impl From<Usage> for LanguageModelUsage {
    fn from(value: Usage) -> Self {
        Self::from(&value)
    }
}

impl From<&Usage> for LanguageModelUsage {
    fn from(value: &Usage) -> Self {
        let input_tokens = value.normalized_input_tokens();
        let output_tokens = value.normalized_output_tokens();

        Self {
            input_tokens: input_tokens.total,
            input_token_details: LanguageModelInputTokenDetails {
                no_cache_tokens: input_tokens.no_cache,
                cache_read_tokens: input_tokens.cache_read,
                cache_write_tokens: input_tokens.cache_write,
            },
            output_tokens: output_tokens.total,
            output_token_details: LanguageModelOutputTokenDetails {
                text_tokens: output_tokens.text,
                reasoning_tokens: output_tokens.reasoning,
            },
            total_tokens: add_optional_u32(input_tokens.total, output_tokens.total),
            reasoning_tokens: output_tokens.reasoning,
            cached_input_tokens: input_tokens.cache_read,
            raw: value.raw.clone(),
        }
    }
}

/// AI SDK-style embedding usage shape.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingModelUsage {
    /// Tokens used by the embedding call.
    pub tokens: u32,
}

impl EmbeddingModelUsage {
    /// Create embedding usage from a token count.
    pub const fn new(tokens: u32) -> Self {
        Self { tokens }
    }
}

impl From<EmbeddingUsage> for EmbeddingModelUsage {
    fn from(value: EmbeddingUsage) -> Self {
        Self {
            tokens: value.prompt_tokens,
        }
    }
}

impl From<&EmbeddingUsage> for EmbeddingModelUsage {
    fn from(value: &EmbeddingUsage) -> Self {
        Self {
            tokens: value.prompt_tokens,
        }
    }
}

/// AI SDK-style image-model usage shape.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ImageModelUsage {
    /// Input prompt tokens when the provider reports them.
    #[serde(rename = "inputTokens", skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    /// Output tokens when the provider reports them.
    #[serde(rename = "outputTokens", skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    /// Total tokens when the provider reports them.
    #[serde(rename = "totalTokens", skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
}

impl ImageModelUsage {
    /// Create image-model usage from optional token totals.
    pub const fn new(
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
        total_tokens: Option<u32>,
    ) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens,
        }
    }

    /// Merge another image usage payload into this one.
    pub fn merge(&mut self, other: &Self) {
        self.input_tokens = add_optional_u32(self.input_tokens, other.input_tokens);
        self.output_tokens = add_optional_u32(self.output_tokens, other.output_tokens);
        self.total_tokens = add_optional_u32(self.total_tokens, other.total_tokens);
    }
}

/// AI SDK-style helper for adding two image-model usage payloads.
pub fn add_image_model_usage(
    usage1: &ImageModelUsage,
    usage2: &ImageModelUsage,
) -> ImageModelUsage {
    let mut merged = usage1.clone();
    merged.merge(usage2);
    merged
}

/// AI SDK-style request metadata for language-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct LanguageModelRequestMetadata {
    /// Serialized request body when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl From<HttpRequestInfo> for LanguageModelRequestMetadata {
    fn from(value: HttpRequestInfo) -> Self {
        Self {
            body: value.body.map(string_body_to_json_value),
        }
    }
}

impl From<&HttpRequestInfo> for LanguageModelRequestMetadata {
    fn from(value: &HttpRequestInfo) -> Self {
        Self {
            body: value.body.clone().map(string_body_to_json_value),
        }
    }
}

/// AI SDK-style response metadata for language-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageModelResponseMetadata {
    /// Response identifier.
    pub id: String,
    /// Timestamp for the start of the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// HTTP response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}

impl TryFrom<ResponseMetadata> for LanguageModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: ResponseMetadata) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&ResponseMetadata> for LanguageModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: &ResponseMetadata) -> Result<Self, Self::Error> {
        Ok(Self {
            id: value.id.clone().ok_or("missing response id")?,
            timestamp: value.created.ok_or("missing response timestamp")?,
            model_id: value.model.clone().ok_or("missing response model id")?,
            headers: value.headers.clone(),
        })
    }
}

/// AI SDK-style response metadata for image-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageModelResponseMetadata {
    /// Timestamp for the start of the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// HTTP response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}

impl TryFrom<HttpResponseInfo> for ImageModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: HttpResponseInfo) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&HttpResponseInfo> for ImageModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: &HttpResponseInfo) -> Result<Self, Self::Error> {
        Ok(Self {
            timestamp: value.timestamp,
            model_id: value.model_id.clone().ok_or("missing response model id")?,
            headers: (!value.headers.is_empty()).then_some(value.headers.clone()),
        })
    }
}

/// AI SDK-style response metadata for video-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VideoModelResponseMetadata {
    /// Timestamp for the start of the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// HTTP response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Provider-specific metadata for this call when available.
    #[serde(rename = "providerMetadata", skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<VideoModelProviderMetadata>,
}

impl VideoModelResponseMetadata {
    /// Attach provider metadata when it is not empty.
    pub fn with_provider_metadata(mut self, provider_metadata: VideoModelProviderMetadata) -> Self {
        if !provider_metadata.is_empty() {
            self.provider_metadata = Some(provider_metadata);
        }
        self
    }
}

impl TryFrom<HttpResponseInfo> for VideoModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: HttpResponseInfo) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&HttpResponseInfo> for VideoModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: &HttpResponseInfo) -> Result<Self, Self::Error> {
        Ok(Self {
            timestamp: value.timestamp,
            model_id: value.model_id.clone().ok_or("missing response model id")?,
            headers: (!value.headers.is_empty()).then_some(value.headers.clone()),
            provider_metadata: None,
        })
    }
}

/// AI SDK-style response metadata for speech-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeechModelResponseMetadata {
    /// Timestamp for the start of the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// HTTP response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Raw response body when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<JSONValue>,
}

impl TryFrom<HttpResponseInfo> for SpeechModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: HttpResponseInfo) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&HttpResponseInfo> for SpeechModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: &HttpResponseInfo) -> Result<Self, Self::Error> {
        Ok(Self {
            timestamp: value.timestamp,
            model_id: value.model_id.clone().ok_or("missing response model id")?,
            headers: (!value.headers.is_empty()).then_some(value.headers.clone()),
            body: None,
        })
    }
}

/// AI SDK-style response metadata for transcription-model helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranscriptionModelResponseMetadata {
    /// Timestamp for the start of the response.
    pub timestamp: DateTime<Utc>,
    /// Model identifier used for the response.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// HTTP response headers when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}

impl TryFrom<HttpResponseInfo> for TranscriptionModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: HttpResponseInfo) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&HttpResponseInfo> for TranscriptionModelResponseMetadata {
    type Error = &'static str;

    fn try_from(value: &HttpResponseInfo) -> Result<Self, Self::Error> {
        Ok(Self {
            timestamp: value.timestamp,
            model_id: value.model_id.clone().ok_or("missing response model id")?,
            headers: (!value.headers.is_empty()).then_some(value.headers.clone()),
        })
    }
}

fn add_optional_u32(left: Option<u32>, right: Option<u32>) -> Option<u32> {
    match (left, right) {
        (None, None) => None,
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (Some(left), Some(right)) => Some(left.saturating_add(right)),
    }
}

fn string_body_to_json_value(body: String) -> JSONValue {
    serde_json::from_str(&body).unwrap_or(JSONValue::String(body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_shape_matches_language_model_source_contract() {
        let mut provider_metadata = ProviderMetadataMap::new();
        provider_metadata.insert("anthropic".to_string(), serde_json::json!({ "foo": "bar" }));

        let source = Source::url_with_title("source-0", "https://example.com", "Example")
            .with_provider_metadata(provider_metadata);

        let value = serde_json::to_value(&source).expect("serialize source");
        assert_eq!(value["type"], serde_json::json!("source"));
        assert_eq!(value["sourceType"], serde_json::json!("url"));
        assert_eq!(value["id"], serde_json::json!("source-0"));
        assert_eq!(value["url"], serde_json::json!("https://example.com"));
        assert_eq!(value["title"], serde_json::json!("Example"));
        assert_eq!(
            value["providerMetadata"]["anthropic"],
            serde_json::json!({ "foo": "bar" })
        );

        let roundtrip: Source = serde_json::from_value(value).expect("deserialize source");
        assert_eq!(roundtrip.r#type(), "source");
        assert_eq!(roundtrip.source_type(), "url");
        assert_eq!(roundtrip, source);

        let content_part: ContentPart = source.clone().into();
        let converted = Source::try_from(&content_part).expect("convert source content part");
        assert_eq!(converted, source);
    }

    #[test]
    fn source_rejects_non_source_type_marker() {
        let error = serde_json::from_value::<Source>(serde_json::json!({
            "type": "text",
            "sourceType": "url",
            "id": "source-0",
            "url": "https://example.com"
        }))
        .expect_err("non-source marker should be rejected");

        assert!(error.to_string().contains("expected source type marker"));
    }

    #[test]
    fn language_model_request_metadata_parses_json_body_and_falls_back_to_string() {
        let json = LanguageModelRequestMetadata::from(HttpRequestInfo {
            body: Some("{\"ok\":true}".to_string()),
        });
        let text = LanguageModelRequestMetadata::from(HttpRequestInfo {
            body: Some("plain-text".to_string()),
        });

        assert_eq!(json.body, Some(serde_json::json!({ "ok": true })));
        assert_eq!(text.body, Some(JSONValue::String("plain-text".to_string())));
    }

    #[test]
    fn language_model_response_metadata_requires_main_fields() {
        let metadata = ResponseMetadata {
            id: Some("resp_123".to_string()),
            model: Some("gpt-4o".to_string()),
            created: Some(
                DateTime::parse_from_rfc3339("2026-04-21T09:00:00Z")
                    .expect("valid timestamp")
                    .with_timezone(&Utc),
            ),
            provider: "openai".to_string(),
            request_id: None,
            headers: Some(HashMap::from([(
                "x-request-id".to_string(),
                "req_123".to_string(),
            )])),
        };

        let converted =
            LanguageModelResponseMetadata::try_from(&metadata).expect("convert response metadata");

        assert_eq!(converted.id, "resp_123");
        assert_eq!(converted.model_id, "gpt-4o");
        assert_eq!(
            converted
                .headers
                .as_ref()
                .and_then(|headers| headers.get("x-request-id"))
                .map(String::as_str),
            Some("req_123")
        );
    }

    #[test]
    fn image_and_transcription_response_metadata_require_model_id() {
        let response = HttpResponseInfo {
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T09:00:00Z")
                .expect("valid timestamp")
                .with_timezone(&Utc),
            model_id: Some("model-1".to_string()),
            headers: HashMap::from([("x-test".to_string(), "1".to_string())]),
        };

        let image =
            ImageModelResponseMetadata::try_from(&response).expect("convert image metadata");
        let transcription = TranscriptionModelResponseMetadata::try_from(&response)
            .expect("convert transcription metadata");

        assert_eq!(image.model_id, "model-1");
        assert_eq!(transcription.model_id, "model-1");
    }

    #[test]
    fn cancel_handle_and_request_timeout_helpers_work() {
        let cancel = CancelHandle::new();
        assert!(!cancel.is_cancelled());
        cancel.cancel();
        assert!(cancel.is_cancelled());

        let timeout = TimeoutConfiguration::settings(
            TimeoutConfigurationSettings::new()
                .with_total_ms(3_000)
                .with_step_ms(1_000)
                .with_chunk_ms(250)
                .with_tool_ms(900)
                .with_tool_timeout_ms("weather", 1200),
        );

        let request = RequestOptions::new()
            .with_max_retries(2)
            .with_abort_signal(cancel)
            .with_header("x-test", "1")
            .without_header("x-drop")
            .with_timeout(timeout.clone());

        assert_eq!(timeout.total_timeout_ms(), Some(3_000));
        assert_eq!(timeout.step_timeout_ms(), Some(1_000));
        assert_eq!(timeout.chunk_timeout_ms(), Some(250));
        assert_eq!(timeout.tool_timeout_ms("weather"), Some(1_200));
        assert_eq!(timeout.tool_timeout_ms("other"), Some(900));
        assert_eq!(request.max_attempts(), Some(3));
        assert_eq!(
            request.effective_headers(),
            HashMap::from([("x-test".to_string(), "1".to_string())])
        );
        assert_eq!(request.total_timeout(), Some(Duration::from_millis(3_000)));
        assert_eq!(request.step_timeout(), Some(Duration::from_millis(1_000)));
        assert_eq!(request.chunk_timeout(), Some(Duration::from_millis(250)));
        assert_eq!(request.tool_timeout_ms("weather"), Some(1_200));
        assert!(
            request
                .abort_signal
                .as_ref()
                .is_some_and(CancelHandle::is_cancelled)
        );
    }

    #[test]
    fn embedding_and_image_usage_shared_shapes_are_available() {
        let embedding = EmbeddingModelUsage::from(EmbeddingUsage::new(12, 12));
        let mut image = ImageModelUsage::new(Some(3), Some(5), Some(8));
        image.merge(&ImageModelUsage::new(Some(2), None, Some(2)));
        let added_image =
            add_image_model_usage(&image, &ImageModelUsage::new(None, Some(1), Some(1)));
        let call_options = LanguageModelCallOptions::from(&super::super::CommonParams {
            model: "gpt-5".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(128),
            max_completion_tokens: Some(256),
            top_p: Some(0.9),
            top_k: Some(40.0),
            stop_sequences: Some(vec!["END".to_string()]),
            seed: Some(7),
            frequency_penalty: Some(0.2),
            presence_penalty: Some(0.1),
        });
        let mut language = LanguageModelUsage::from(
            Usage::builder()
                .with_input_tokens(super::super::UsageInputTokens {
                    total: Some(10),
                    no_cache: Some(7),
                    cache_read: Some(2),
                    cache_write: Some(1),
                })
                .with_output_tokens(super::super::UsageOutputTokens {
                    total: Some(6),
                    text: Some(4),
                    reasoning: Some(2),
                })
                .with_raw_usage_value(serde_json::json!({
                    "provider_total_tokens": 16
                }))
                .build(),
        );
        language.merge(&LanguageModelUsage {
            input_tokens: Some(2),
            input_token_details: LanguageModelInputTokenDetails {
                no_cache_tokens: Some(1),
                cache_read_tokens: Some(1),
                cache_write_tokens: None,
            },
            output_tokens: Some(1),
            output_token_details: LanguageModelOutputTokenDetails {
                text_tokens: Some(1),
                reasoning_tokens: None,
            },
            total_tokens: Some(3),
            reasoning_tokens: None,
            cached_input_tokens: Some(1),
            raw: Some(serde_json::Map::new()),
        });
        let added_language = add_language_model_usage(
            &create_null_language_model_usage(),
            &LanguageModelUsage {
                input_tokens: Some(4),
                input_token_details: LanguageModelInputTokenDetails::default(),
                output_tokens: Some(3),
                output_token_details: LanguageModelOutputTokenDetails::default(),
                total_tokens: Some(7),
                reasoning_tokens: None,
                cached_input_tokens: None,
                raw: Some(serde_json::Map::new()),
            },
        );

        assert_eq!(embedding.tokens, 12);
        assert_eq!(image.input_tokens, Some(5));
        assert_eq!(image.output_tokens, Some(5));
        assert_eq!(image.total_tokens, Some(10));
        assert_eq!(added_image.input_tokens, Some(5));
        assert_eq!(added_image.output_tokens, Some(6));
        assert_eq!(added_image.total_tokens, Some(11));
        assert_eq!(call_options.max_output_tokens, Some(256));
        assert_eq!(call_options.temperature, Some(0.7));
        assert_eq!(call_options.stop_sequences, Some(vec!["END".to_string()]));
        assert_eq!(call_options.reasoning, None);
        assert_eq!(language.input_tokens, Some(12));
        assert_eq!(language.output_tokens, Some(7));
        assert_eq!(language.total_tokens, Some(19));
        assert_eq!(language.input_token_details.no_cache_tokens, Some(8));
        assert_eq!(language.cached_input_tokens, Some(3));
        assert_eq!(language.output_token_details.reasoning_tokens, Some(2));
        assert_eq!(language.raw, None);
        assert_eq!(added_language.input_tokens, Some(4));
        assert_eq!(added_language.output_tokens, Some(3));
        assert_eq!(added_language.total_tokens, Some(7));
        assert_eq!(added_language.raw, None);
    }

    #[test]
    fn video_model_response_metadata_attaches_non_empty_provider_metadata() {
        let metadata = VideoModelResponseMetadata::try_from(&HttpResponseInfo {
            timestamp: Utc::now(),
            model_id: Some("video-model".to_string()),
            headers: HashMap::from([("x-video".to_string(), "1".to_string())]),
        })
        .expect("valid video response metadata")
        .with_provider_metadata(HashMap::from([(
            "fake-video".to_string(),
            serde_json::json!({ "taskId": "task-1" }),
        )]));

        assert_eq!(metadata.model_id, "video-model");
        assert_eq!(
            metadata
                .headers
                .as_ref()
                .and_then(|headers| headers.get("x-video")),
            Some(&"1".to_string())
        );
        assert_eq!(
            metadata
                .provider_metadata
                .as_ref()
                .and_then(|metadata| metadata.get("fake-video"))
                .and_then(|value| value.get("taskId"))
                .and_then(serde_json::Value::as_str),
            Some("task-1")
        );
    }

    #[test]
    fn timeout_helper_functions_follow_ai_sdk_semantics() {
        let timeout = TimeoutConfiguration::settings(
            TimeoutConfigurationSettings::new()
                .with_total_ms(1_000)
                .with_step_ms(200)
                .with_chunk_ms(50)
                .with_tool_ms(300)
                .with_tool_timeout_ms("search", 450),
        );

        assert_eq!(get_total_timeout_ms(Some(&timeout)), Some(1_000));
        assert_eq!(get_step_timeout_ms(Some(&timeout)), Some(200));
        assert_eq!(get_chunk_timeout_ms(Some(&timeout)), Some(50));
        assert_eq!(get_tool_timeout_ms(Some(&timeout), "search"), Some(450));
        assert_eq!(get_tool_timeout_ms(Some(&timeout), "other"), Some(300));
        assert_eq!(get_total_timeout_ms(None), None);
    }

    #[test]
    fn provider_utils_style_tool_and_context_types_are_available() {
        let mut context = Context::new();
        context.insert("tenant".to_string(), serde_json::json!("acme"));

        let mut provider_options = ProviderOptions::default();
        provider_options.insert("openai", serde_json::json!({ "serviceTier": "flex" }));

        let tool_call = ToolCall::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "q": "rust" }),
        )
        .with_provider_executed(true)
        .with_dynamic(true);

        let tool_result = ToolResult::new(
            "call_1",
            "search".to_string(),
            serde_json::json!({ "q": "rust" }),
            ToolResultOutput::json(serde_json::json!({ "ok": true })),
        )
        .with_provider_executed(true)
        .with_dynamic(true);

        assert_eq!(context.get("tenant"), Some(&serde_json::json!("acme")));
        assert_eq!(
            provider_options
                .get("openai")
                .and_then(|value| value.get("serviceTier"))
                .and_then(serde_json::Value::as_str),
            Some("flex")
        );
        assert_eq!(tool_call.tool_call_id, "call_1");
        assert_eq!(tool_call.provider_executed, Some(true));
        assert_eq!(tool_result.tool_call_id, "call_1");
        assert_eq!(tool_result.dynamic, Some(true));
    }

    #[test]
    #[allow(deprecated)]
    fn call_settings_projects_onto_call_and_request_options() {
        let abort_signal = CancelHandle::new();
        let settings = CallSettings::new()
            .with_max_output_tokens(256)
            .with_temperature(0.4)
            .with_top_p(0.8)
            .with_stop_sequences(vec!["END".to_string()])
            .with_seed(7)
            .with_reasoning(LanguageModelReasoning::Medium)
            .with_max_retries(2)
            .with_abort_signal(abort_signal.clone())
            .with_header("x-test", "1")
            .without_header("x-drop");

        let call_options = settings.language_model_call_options();
        let request_options = settings.request_options();

        assert_eq!(call_options.max_output_tokens, Some(256));
        assert_eq!(call_options.temperature, Some(0.4));
        assert_eq!(call_options.top_p, Some(0.8));
        assert_eq!(call_options.stop_sequences, Some(vec!["END".to_string()]));
        assert_eq!(call_options.seed, Some(7));
        assert_eq!(call_options.reasoning, Some(LanguageModelReasoning::Medium));
        assert_eq!(request_options.max_retries, Some(2));
        assert_eq!(request_options.max_attempts(), Some(3));
        assert!(
            request_options
                .abort_signal
                .as_ref()
                .is_some_and(|signal| !signal.is_cancelled())
        );
        assert_eq!(
            request_options.effective_headers().get("x-test"),
            Some(&"1".to_string())
        );
        assert!(!request_options.effective_headers().contains_key("x-drop"));
        assert!(request_options.timeout.is_none());
    }
}
