use crate::types::{
    CommonParams, LanguageModelV4Tool, LanguageModelV4ToolChoice, ModelMessage, ResponseFormat,
    Tool,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::{
    LanguageModelV4Prompt, ProviderOptions, TimeoutConfiguration, prepare_language_model_v4_prompt,
};

/// AI SDK V4 model-facing language-model call options.
///
/// This keeps the upstream provider-call shape as a single overlay while the ergonomic Rust API
/// may still expose `LanguageModelCallOptions` and `RequestOptions` as separate reusable pieces.
#[derive(Debug, Clone)]
pub struct LanguageModelV4CallOptions<ABORT = ()> {
    /// Standardized model prompt.
    pub prompt: LanguageModelV4Prompt,
    /// Maximum output tokens requested by the caller.
    pub max_output_tokens: Option<u64>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Stop sequences.
    pub stop_sequences: Option<Vec<String>>,
    /// Nucleus sampling.
    pub top_p: Option<f64>,
    /// Top-k sampling.
    pub top_k: Option<f64>,
    /// Presence penalty.
    pub presence_penalty: Option<f64>,
    /// Frequency penalty.
    pub frequency_penalty: Option<f64>,
    /// Response format hint.
    pub response_format: Option<ResponseFormat>,
    /// Deterministic seed.
    pub seed: Option<u64>,
    /// Model-facing tool definitions.
    pub tools: Option<Vec<LanguageModelV4Tool>>,
    /// Model-facing tool-choice object.
    pub tool_choice: Option<LanguageModelV4ToolChoice>,
    /// Include raw stream chunks when supported.
    pub include_raw_chunks: Option<bool>,
    /// Request-scoped abort signal.
    pub abort_signal: Option<ABORT>,
    /// Additional HTTP headers. `None` represents an explicitly undefined header.
    pub headers: HashMap<String, Option<String>>,
    /// Cross-provider reasoning level.
    pub reasoning: Option<LanguageModelReasoning>,
    /// Provider-specific options.
    pub provider_options: Option<ProviderOptions>,
}

impl<ABORT> Default for LanguageModelV4CallOptions<ABORT> {
    fn default() -> Self {
        Self {
            prompt: LanguageModelV4Prompt::default(),
            max_output_tokens: None,
            temperature: None,
            stop_sequences: None,
            top_p: None,
            top_k: None,
            presence_penalty: None,
            frequency_penalty: None,
            response_format: None,
            seed: None,
            tools: None,
            tool_choice: None,
            include_raw_chunks: None,
            abort_signal: None,
            headers: HashMap::new(),
            reasoning: None,
            provider_options: None,
        }
    }
}

impl<ABORT> LanguageModelV4CallOptions<ABORT> {
    /// Create call options for a standardized prompt.
    pub fn new(prompt: LanguageModelV4Prompt) -> Self {
        Self {
            prompt,
            ..Self::default()
        }
    }

    /// Create call options by projecting stable model messages to the V4 provider prompt.
    pub fn from_model_messages(messages: impl IntoIterator<Item = ModelMessage>) -> Self {
        Self::new(prepare_language_model_v4_prompt(messages))
    }

    /// Set the maximum output tokens requested by the caller.
    pub const fn with_max_output_tokens(mut self, max_output_tokens: u64) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    /// Attach already projected model-facing tools.
    pub fn with_tools(mut self, tools: Vec<LanguageModelV4Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Attach stable tools by projecting them to the model-facing V4 tool union.
    pub fn with_stable_tools(mut self, tools: impl IntoIterator<Item = Tool>) -> Self {
        self.tools = Some(tools.into_iter().map(LanguageModelV4Tool::from).collect());
        self
    }

    /// Attach a model-facing tool-choice object.
    pub fn with_tool_choice(mut self, tool_choice: impl Into<LanguageModelV4ToolChoice>) -> Self {
        self.tool_choice = Some(tool_choice.into());
        self
    }

    /// Set a request header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), Some(value.into()));
        self
    }

    /// Mark a header as intentionally undefined.
    pub fn without_header(mut self, key: impl Into<String>) -> Self {
        self.headers.insert(key.into(), None);
        self
    }

    /// Materialize only concrete headers.
    pub fn effective_headers(&self) -> HashMap<String, String> {
        self.headers
            .iter()
            .filter_map(|(key, value)| value.clone().map(|value| (key.clone(), value)))
            .collect()
    }
}

/// AI SDK-style request-facing transport controls.
#[derive(Debug, Clone)]
pub struct RequestOptions<ABORT = ()> {
    /// Maximum number of retries. `0` disables retries.
    pub max_retries: Option<u32>,
    /// Request-scoped abort signal.
    pub abort_signal: Option<ABORT>,
    /// Additional HTTP headers. `None` values are filtered when materialized.
    pub headers: HashMap<String, Option<String>>,
    /// Timeout configuration.
    pub timeout: Option<TimeoutConfiguration>,
}

impl<ABORT> Default for RequestOptions<ABORT> {
    fn default() -> Self {
        Self {
            max_retries: None,
            abort_signal: None,
            headers: HashMap::new(),
            timeout: None,
        }
    }
}

impl<ABORT> RequestOptions<ABORT> {
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
    pub fn with_abort_signal(mut self, abort_signal: ABORT) -> Self {
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

impl From<CommonParams> for LanguageModelCallOptions {
    fn from(value: CommonParams) -> Self {
        Self::from(&value)
    }
}

impl From<&CommonParams> for LanguageModelCallOptions {
    fn from(value: &CommonParams) -> Self {
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
#[derive(Debug, Clone)]
pub struct CallSettings<ABORT = ()> {
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
    pub abort_signal: Option<ABORT>,
    /// Additional HTTP headers. `None` values are filtered when materialized.
    pub headers: HashMap<String, Option<String>>,
}

#[allow(deprecated)]
impl<ABORT> Default for CallSettings<ABORT> {
    fn default() -> Self {
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
            max_retries: None,
            abort_signal: None,
            headers: HashMap::new(),
        }
    }
}

#[allow(deprecated)]
impl<ABORT> CallSettings<ABORT> {
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
    pub fn with_abort_signal(mut self, abort_signal: ABORT) -> Self {
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
    pub fn request_options(&self) -> RequestOptions<ABORT>
    where
        ABORT: Clone,
    {
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
impl<ABORT> From<LanguageModelCallOptions> for CallSettings<ABORT> {
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
impl<ABORT> From<RequestOptions<ABORT>> for CallSettings<ABORT> {
    fn from(value: RequestOptions<ABORT>) -> Self {
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
impl<ABORT> From<&CommonParams> for CallSettings<ABORT> {
    fn from(value: &CommonParams) -> Self {
        Self::from(LanguageModelCallOptions::from(value))
    }
}

#[allow(deprecated)]
impl<ABORT> From<CommonParams> for CallSettings<ABORT> {
    fn from(value: CommonParams) -> Self {
        Self::from(&value)
    }
}

#[allow(deprecated)]
impl<ABORT> From<&CallSettings<ABORT>> for LanguageModelCallOptions {
    fn from(value: &CallSettings<ABORT>) -> Self {
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
impl<ABORT> From<CallSettings<ABORT>> for LanguageModelCallOptions {
    fn from(value: CallSettings<ABORT>) -> Self {
        Self::from(&value)
    }
}

#[allow(deprecated)]
impl<ABORT: Clone> From<&CallSettings<ABORT>> for RequestOptions<ABORT> {
    fn from(value: &CallSettings<ABORT>) -> Self {
        RequestOptions {
            max_retries: value.max_retries,
            abort_signal: value.abort_signal.clone(),
            headers: value.headers.clone(),
            timeout: None,
        }
    }
}

#[allow(deprecated)]
impl<ABORT> From<CallSettings<ABORT>> for RequestOptions<ABORT> {
    fn from(value: CallSettings<ABORT>) -> Self {
        RequestOptions {
            max_retries: value.max_retries,
            abort_signal: value.abort_signal,
            headers: value.headers,
            timeout: None,
        }
    }
}
