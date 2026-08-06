//! Provider-owned Groq request options.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use siumai_core::{ModelFamily, ProviderOptionError, TypedProviderOptions};
use siumai_protocol_openai::chat_completions::API_MODE_ID as CHAT_API_MODE_ID;
use siumai_protocol_openai::responses_next::API_MODE_ID as RESPONSES_API_MODE_ID;

use crate::transcription::TRANSCRIPTION_API_MODE_ID;

/// Groq processing tier for one language request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GroqServiceTier {
    Auto,
    OnDemand,
    Performance,
    Flex,
}

/// Groq Responses processing tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GroqResponsesServiceTier {
    Auto,
    Default,
    Flex,
}

/// Groq reasoning effort for models that expose explicit reasoning controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GroqReasoningEffort {
    None,
    Default,
    Low,
    Medium,
    High,
}

/// Wire representation requested for model reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GroqReasoningFormat {
    Hidden,
    Raw,
    Parsed,
}

/// Typed Groq Chat Completions options.
///
/// Common generation controls stay on [`siumai_core::LanguageRequest`]. This type only carries
/// Groq-owned semantics and deliberately has no arbitrary JSON field bag.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "snake_case")]
pub struct GroqLanguageOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<GroqServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<GroqReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_format: Option<GroqReasoningFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_reasoning: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Select JSON Schema structured outputs (`true`) or JSON-object mode (`false`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_outputs: Option<bool>,
    /// Override the `strict` flag on an encoded JSON Schema response format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict_json_schema: Option<bool>,
    /// Add Groq's built-in browser-search tool when the selected model supports it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub browser_search: Option<bool>,
}

impl GroqLanguageOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_logprobs(mut self, enabled: bool) -> Self {
        self.logprobs = Some(enabled);
        self
    }

    pub const fn with_top_logprobs(mut self, count: u32) -> Self {
        self.top_logprobs = Some(count);
        self
    }

    pub const fn with_service_tier(mut self, tier: GroqServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    pub const fn with_reasoning_effort(mut self, effort: GroqReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    pub const fn with_reasoning_format(mut self, format: GroqReasoningFormat) -> Self {
        self.reasoning_format = Some(format);
        self
    }

    pub const fn with_include_reasoning(mut self, include: bool) -> Self {
        self.include_reasoning = Some(include);
        self
    }

    pub const fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub const fn with_structured_outputs(mut self, enabled: bool) -> Self {
        self.structured_outputs = Some(enabled);
        self
    }

    pub const fn with_strict_json_schema(mut self, enabled: bool) -> Self {
        self.strict_json_schema = Some(enabled);
        self
    }

    pub const fn with_browser_search(mut self, enabled: bool) -> Self {
        self.browser_search = Some(enabled);
        self
    }

    fn validate_values(&self) -> Result<(), ProviderOptionError> {
        if self.top_logprobs.is_some_and(|count| count > 20) {
            return Err(rejected("top_logprobs", "must be between zero and 20"));
        }
        if self.top_logprobs.is_some() && self.logprobs == Some(false) {
            return Err(rejected(
                "top_logprobs",
                "cannot be requested when logprobs is disabled",
            ));
        }
        if self.user.as_deref().is_some_and(invalid_text) {
            return Err(rejected(
                "user",
                "must not be empty or contain control characters",
            ));
        }
        if self.structured_outputs == Some(false) && self.strict_json_schema == Some(true) {
            return Err(rejected(
                "strict_json_schema",
                "cannot be enabled when structured outputs are disabled",
            ));
        }
        if self.reasoning_format.is_some() && self.include_reasoning.is_some() {
            return Err(rejected(
                "include_reasoning",
                "cannot be combined with reasoning_format",
            ));
        }
        Ok(())
    }
}

impl TypedProviderOptions for GroqLanguageOptions {
    const NAMESPACE: &'static str = "groq";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(CHAT_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        self.validate_values()
    }
}

/// Typed Groq Responses API options.
///
/// Portable messages, generation controls, function tools, and structured output stay on
/// [`siumai_core::LanguageRequest`]. Hosted tools and Groq execution controls remain provider
/// owned here.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "snake_case")]
pub struct GroqResponsesOptions {
    /// Groq currently exposes terminal Responses calls only. `true` is rejected before I/O.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<GroqResponsesServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<GroqReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
    /// Add Groq's built-in browser-search tool for verified GPT-OSS models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub browser_search: Option<bool>,
    /// Add Groq's built-in code-interpreter tool for verified GPT-OSS models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_execution: Option<bool>,
    /// Request detailed inference metrics through the Groq beta response metadata header.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_metrics: Option<bool>,
}

impl GroqResponsesOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_background(mut self, enabled: bool) -> Self {
        self.background = Some(enabled);
        self
    }

    pub const fn with_service_tier(mut self, tier: GroqResponsesServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    pub const fn with_reasoning_effort(mut self, effort: GroqReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    pub const fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub fn with_metadata<I, K, V>(mut self, metadata: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.metadata = metadata
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        self
    }

    pub const fn with_browser_search(mut self, enabled: bool) -> Self {
        self.browser_search = Some(enabled);
        self
    }

    pub const fn with_code_execution(mut self, enabled: bool) -> Self {
        self.code_execution = Some(enabled);
        self
    }

    pub const fn with_inference_metrics(mut self, enabled: bool) -> Self {
        self.inference_metrics = Some(enabled);
        self
    }

    fn validate_values(&self) -> Result<(), ProviderOptionError> {
        if self.background == Some(true) {
            return Err(rejected(
                "background",
                "Groq Responses does not support background execution",
            ));
        }
        if self.user.as_deref().is_some_and(invalid_text) {
            return Err(rejected(
                "user",
                "must not be empty or contain control characters",
            ));
        }
        if self.metadata.len() > 16 {
            return Err(rejected("metadata", "must contain at most 16 entries"));
        }
        if self
            .metadata
            .iter()
            .any(|(key, value)| invalid_text(key) || value.chars().any(char::is_control))
        {
            return Err(rejected(
                "metadata",
                "keys must be non-empty and metadata must not contain control characters",
            ));
        }
        Ok(())
    }
}

impl TypedProviderOptions for GroqResponsesOptions {
    const NAMESPACE: &'static str = "groq";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(RESPONSES_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        self.validate_values()
    }
}

/// Groq transcription response encoding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GroqTranscriptionResponseFormat {
    #[default]
    Json,
    VerboseJson,
    Text,
}

/// Timestamp detail requested from verbose Groq transcription responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GroqTimestampGranularity {
    Segment,
    Word,
}

/// Typed options for Groq's final-result audio transcription endpoint.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "snake_case")]
pub struct GroqTranscriptionOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<GroqTranscriptionResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub timestamp_granularities: Vec<GroqTimestampGranularity>,
}

impl GroqTranscriptionOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_response_format(mut self, format: GroqTranscriptionResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    pub const fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_timestamp_granularities<I>(mut self, granularities: I) -> Self
    where
        I: IntoIterator<Item = GroqTimestampGranularity>,
    {
        self.timestamp_granularities = granularities.into_iter().collect();
        self
    }

    fn validate_values(&self) -> Result<(), ProviderOptionError> {
        if self
            .temperature
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(rejected(
                "temperature",
                "must be a finite number between zero and one",
            ));
        }
        if !self.timestamp_granularities.is_empty()
            && self.response_format != Some(GroqTranscriptionResponseFormat::VerboseJson)
        {
            return Err(rejected(
                "timestamp_granularities",
                "requires response_format=verbose_json",
            ));
        }
        Ok(())
    }
}

impl TypedProviderOptions for GroqTranscriptionOptions {
    const NAMESPACE: &'static str = "groq";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Transcription;
    const API_MODE: Option<&'static str> = Some(TRANSCRIPTION_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        self.validate_values()
    }
}

fn invalid_text(value: &str) -> bool {
    value.trim().is_empty() || value.chars().any(char::is_control)
}

fn rejected(path: &str, reason: &str) -> ProviderOptionError {
    ProviderOptionError::Rejected {
        path: path.to_string(),
        reason: reason.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use siumai_core::ProviderOptions;

    use super::*;

    #[test]
    fn language_options_are_typed_and_snake_case() {
        let options = GroqLanguageOptions::new()
            .with_service_tier(GroqServiceTier::Performance)
            .with_reasoning_effort(GroqReasoningEffort::High)
            .with_reasoning_format(GroqReasoningFormat::Parsed)
            .with_browser_search(true);
        let erased = ProviderOptions::typed(&options).unwrap();

        assert_eq!(erased.namespace().as_str(), "groq");
        assert_eq!(erased.value()["service_tier"], "performance");
        assert_eq!(erased.value()["reasoning_effort"], "high");
        assert_eq!(erased.value()["reasoning_format"], "parsed");
        assert_eq!(erased.value()["browser_search"], true);
    }

    #[test]
    fn invalid_cross_field_combinations_fail_before_erasure() {
        let invalid = GroqLanguageOptions::new()
            .with_logprobs(false)
            .with_top_logprobs(2);
        assert!(ProviderOptions::typed(&invalid).is_err());

        let valid = GroqLanguageOptions::new()
            .with_logprobs(true)
            .with_top_logprobs(0);
        assert!(ProviderOptions::typed(&valid).is_ok());
        let invalid = GroqLanguageOptions::new()
            .with_logprobs(true)
            .with_top_logprobs(21);
        assert!(ProviderOptions::typed(&invalid).is_err());

        let invalid = GroqLanguageOptions::new()
            .with_reasoning_format(GroqReasoningFormat::Parsed)
            .with_include_reasoning(true);
        assert!(ProviderOptions::typed(&invalid).is_err());

        let invalid = GroqTranscriptionOptions::new()
            .with_timestamp_granularities([GroqTimestampGranularity::Word]);
        assert!(ProviderOptions::typed(&invalid).is_err());

        let invalid = GroqResponsesOptions::new().with_background(true);
        assert!(ProviderOptions::typed(&invalid).is_err());
        let valid = GroqResponsesOptions::new()
            .with_background(false)
            .with_browser_search(true)
            .with_inference_metrics(true);
        assert!(ProviderOptions::typed(&valid).is_ok());
    }
}
