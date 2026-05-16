use crate::types::{EmbeddingUsage, Usage, UsageInputTokens, UsageOutputTokens};
use serde::{Deserialize, Serialize};

use super::JSONValue;

/// AI SDK V4 input-token accounting.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct LanguageModelV4InputTokens {
    /// Total input tokens, including cache hits and cache writes when billed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    /// Non-cached input tokens.
    #[serde(rename = "noCache", skip_serializing_if = "Option::is_none")]
    pub no_cache: Option<u64>,
    /// Tokens read from cache.
    #[serde(rename = "cacheRead", skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<u64>,
    /// Tokens written to cache.
    #[serde(rename = "cacheWrite", skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<u64>,
}

impl LanguageModelV4InputTokens {
    /// Create input token usage with only the total populated.
    pub const fn with_total(total: u64) -> Self {
        Self {
            total: Some(total),
            no_cache: None,
            cache_read: None,
            cache_write: None,
        }
    }
}

impl From<&UsageInputTokens> for LanguageModelV4InputTokens {
    fn from(value: &UsageInputTokens) -> Self {
        Self {
            total: value.total.map(u64::from),
            no_cache: value.no_cache.map(u64::from),
            cache_read: value.cache_read.map(u64::from),
            cache_write: value.cache_write.map(u64::from),
        }
    }
}

impl From<UsageInputTokens> for LanguageModelV4InputTokens {
    fn from(value: UsageInputTokens) -> Self {
        Self::from(&value)
    }
}

/// AI SDK V4 output-token accounting.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct LanguageModelV4OutputTokens {
    /// Total output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    /// Text/output-visible tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<u64>,
    /// Reasoning tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
}

impl LanguageModelV4OutputTokens {
    /// Create output token usage with only the total populated.
    pub const fn with_total(total: u64) -> Self {
        Self {
            total: Some(total),
            text: None,
            reasoning: None,
        }
    }
}

impl From<&UsageOutputTokens> for LanguageModelV4OutputTokens {
    fn from(value: &UsageOutputTokens) -> Self {
        Self {
            total: value.total.map(u64::from),
            text: value.text.map(u64::from),
            reasoning: value.reasoning.map(u64::from),
        }
    }
}

impl From<UsageOutputTokens> for LanguageModelV4OutputTokens {
    fn from(value: UsageOutputTokens) -> Self {
        Self::from(&value)
    }
}

/// AI SDK V4 language-model usage payload.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct LanguageModelV4Usage {
    /// Input token accounting.
    #[serde(rename = "inputTokens")]
    pub input_tokens: LanguageModelV4InputTokens,
    /// Output token accounting.
    #[serde(rename = "outputTokens")]
    pub output_tokens: LanguageModelV4OutputTokens,
    /// Raw provider usage payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Map<String, JSONValue>>,
}

impl LanguageModelV4Usage {
    /// Create a V4 usage payload.
    pub fn new(
        input_tokens: impl Into<LanguageModelV4InputTokens>,
        output_tokens: impl Into<LanguageModelV4OutputTokens>,
    ) -> Self {
        Self {
            input_tokens: input_tokens.into(),
            output_tokens: output_tokens.into(),
            raw: None,
        }
    }

    /// Attach raw provider usage.
    pub fn with_raw(mut self, raw: serde_json::Map<String, JSONValue>) -> Self {
        self.raw = Some(raw);
        self
    }
}

impl From<&Usage> for LanguageModelV4Usage {
    fn from(value: &Usage) -> Self {
        Self {
            input_tokens: value.normalized_input_tokens().into(),
            output_tokens: value.normalized_output_tokens().into(),
            raw: value.raw.clone(),
        }
    }
}

impl From<Usage> for LanguageModelV4Usage {
    fn from(value: Usage) -> Self {
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
    /// Create a normalized usage payload from input and output token counts.
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens: Some(input_tokens),
            input_token_details: LanguageModelInputTokenDetails::default(),
            output_tokens: Some(output_tokens),
            output_token_details: LanguageModelOutputTokenDetails::default(),
            total_tokens: Some(input_tokens.saturating_add(output_tokens)),
            reasoning_tokens: None,
            cached_input_tokens: None,
            raw: None,
        }
    }

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

/// AI SDK-style helper for projecting shared provider usage onto `LanguageModelUsage`.
pub fn as_language_model_usage(usage: &Usage) -> LanguageModelUsage {
    LanguageModelUsage::from(usage)
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

fn add_optional_u32(left: Option<u32>, right: Option<u32>) -> Option<u32> {
    match (left, right) {
        (None, None) => None,
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (Some(left), Some(right)) => Some(left.saturating_add(right)),
    }
}
