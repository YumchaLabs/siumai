//! Usage statistics types.
//!
//! This module defines token usage structures shared across providers.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

fn sum_option(target: &mut Option<u32>, source: Option<u32>) {
    if let Some(value) = source {
        *target = Some(target.unwrap_or(0).saturating_add(value));
    }
}

fn merge_json_value(target: &mut Value, source: Value) {
    match (target, source) {
        (Value::Object(target_obj), Value::Object(source_obj)) => {
            for (key, value) in source_obj {
                if let Some(existing) = target_obj.get_mut(&key) {
                    merge_json_value(existing, value);
                } else {
                    target_obj.insert(key, value);
                }
            }
        }
        (target, source) => {
            *target = source;
        }
    }
}

fn merge_raw_usage(target: &mut Option<Map<String, Value>>, source: Option<&Map<String, Value>>) {
    let Some(source) = source else {
        return;
    };

    let target = target.get_or_insert_with(Map::new);
    for (key, value) in source {
        if let Some(existing) = target.get_mut(key) {
            merge_json_value(existing, value.clone());
        } else {
            target.insert(key.clone(), value.clone());
        }
    }
}

/// AI SDK v4-compatible input token accounting.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageInputTokens {
    /// Total input tokens, including cache hits and cache writes when billed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u32>,
    /// Non-cached input tokens.
    #[serde(rename = "noCache", skip_serializing_if = "Option::is_none")]
    pub no_cache: Option<u32>,
    /// Tokens read from cache.
    #[serde(rename = "cacheRead", skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<u32>,
    /// Tokens written to cache.
    #[serde(rename = "cacheWrite", skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<u32>,
}

impl UsageInputTokens {
    /// Create input token usage with only the total populated.
    pub const fn with_total(total: u32) -> Self {
        Self {
            total: Some(total),
            no_cache: None,
            cache_read: None,
            cache_write: None,
        }
    }

    /// Returns true when no v4 fields are populated.
    pub const fn is_empty(&self) -> bool {
        self.total.is_none()
            && self.no_cache.is_none()
            && self.cache_read.is_none()
            && self.cache_write.is_none()
    }

    fn normalize(mut self, fallback_total: Option<u32>, fallback_cache_read: Option<u32>) -> Self {
        if self.total.is_none() {
            self.total = fallback_total;
        }
        if self.cache_read.is_none() {
            self.cache_read = fallback_cache_read;
        }
        if self.no_cache.is_none() {
            self.no_cache = self.total.map(|total| {
                total
                    .saturating_sub(self.cache_read.unwrap_or(0))
                    .saturating_sub(self.cache_write.unwrap_or(0))
            });
        }
        self
    }

    fn merge(&mut self, other: &Self) {
        sum_option(&mut self.total, other.total);
        sum_option(&mut self.no_cache, other.no_cache);
        sum_option(&mut self.cache_read, other.cache_read);
        sum_option(&mut self.cache_write, other.cache_write);
    }
}

/// AI SDK v4-compatible output token accounting.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageOutputTokens {
    /// Total output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u32>,
    /// Text/output-visible tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<u32>,
    /// Reasoning tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u32>,
}

impl UsageOutputTokens {
    /// Create output token usage with only the total populated.
    pub const fn with_total(total: u32) -> Self {
        Self {
            total: Some(total),
            text: None,
            reasoning: None,
        }
    }

    /// Returns true when no v4 fields are populated.
    pub const fn is_empty(&self) -> bool {
        self.total.is_none() && self.text.is_none() && self.reasoning.is_none()
    }

    fn normalize(mut self, fallback_total: Option<u32>, fallback_reasoning: Option<u32>) -> Self {
        if self.total.is_none() {
            self.total = fallback_total;
        }
        if self.reasoning.is_none() {
            self.reasoning = fallback_reasoning;
        }
        if self.text.is_none() {
            self.text = self
                .total
                .map(|total| total.saturating_sub(self.reasoning.unwrap_or(0)));
        }
        self
    }

    fn merge(&mut self, other: &Self) {
        sum_option(&mut self.total, other.total);
        sum_option(&mut self.text, other.text);
        sum_option(&mut self.reasoning, other.reasoning);
    }
}

/// Usage statistics
#[derive(Debug, Clone, Default)]
pub struct Usage {
    /// Legacy input-token total kept only as a compatibility seed for accessors/serde.
    legacy_prompt_tokens: Option<u32>,
    /// Legacy output-token total kept only as a compatibility seed for accessors/serde.
    legacy_completion_tokens: Option<u32>,
    /// Legacy total-token count kept only as a compatibility seed for accessors/serde.
    legacy_total_tokens: Option<u32>,
    /// Cached tokens (if applicable)
    #[deprecated(
        since = "0.11.0",
        note = "Use prompt_tokens_details.cached_tokens or inputTokens.cacheRead instead"
    )]
    pub cached_tokens: Option<u32>,
    /// Reasoning tokens (for models like o1)
    #[deprecated(
        since = "0.11.0",
        note = "Use completion_tokens_details.reasoning_tokens or outputTokens.reasoning instead"
    )]
    pub reasoning_tokens: Option<u32>,
    /// Detailed breakdown of prompt tokens
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Detailed breakdown of completion tokens
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// AI SDK v4-compatible input token breakdown
    pub input_tokens: UsageInputTokens,
    /// AI SDK v4-compatible output token breakdown
    pub output_tokens: UsageOutputTokens,
    /// Raw provider usage payload in provider-native shape
    pub raw: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct UsageSerde {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prompt_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    completion_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    total_tokens: Option<u32>,
    #[deprecated(
        since = "0.11.0",
        note = "Use prompt_tokens_details.cached_tokens or inputTokens.cacheRead instead"
    )]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cached_tokens: Option<u32>,
    #[deprecated(
        since = "0.11.0",
        note = "Use completion_tokens_details.reasoning_tokens or outputTokens.reasoning instead"
    )]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(
        default,
        rename = "inputTokens",
        skip_serializing_if = "UsageInputTokens::is_empty"
    )]
    input_tokens: UsageInputTokens,
    #[serde(
        default,
        rename = "outputTokens",
        skip_serializing_if = "UsageOutputTokens::is_empty"
    )]
    output_tokens: UsageOutputTokens,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    raw: Option<Map<String, Value>>,
}

/// Breakdown of tokens used in the prompt
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PromptTokensDetails {
    /// Audio input tokens present in the prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
    /// Cached tokens present in the prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

/// Breakdown of tokens used in the completion
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompletionTokensDetails {
    /// Tokens generated by the model for reasoning
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    /// Audio output tokens generated by the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
    /// Accepted prediction tokens (when using Predicted Outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_prediction_tokens: Option<u32>,
    /// Rejected prediction tokens (when using Predicted Outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_prediction_tokens: Option<u32>,
}

impl Usage {
    /// Create new usage statistics.
    pub const fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            legacy_prompt_tokens: Some(prompt_tokens),
            legacy_completion_tokens: Some(completion_tokens),
            legacy_total_tokens: Some(prompt_tokens + completion_tokens),
            #[allow(deprecated)]
            cached_tokens: None,
            #[allow(deprecated)]
            reasoning_tokens: None,
            prompt_tokens_details: None,
            completion_tokens_details: None,
            input_tokens: UsageInputTokens {
                total: Some(prompt_tokens),
                no_cache: Some(prompt_tokens),
                cache_read: None,
                cache_write: None,
            },
            output_tokens: UsageOutputTokens {
                total: Some(completion_tokens),
                text: Some(completion_tokens),
                reasoning: None,
            },
            raw: None,
        }
    }

    /// Create an explicitly unknown usage payload.
    ///
    /// Legacy totals fall back to `0` for compatibility, but they are marked internally
    /// as unknown so normalized AI SDK-style views remain `None`.
    pub const fn unknown() -> Self {
        Self {
            legacy_prompt_tokens: None,
            legacy_completion_tokens: None,
            legacy_total_tokens: None,
            #[allow(deprecated)]
            cached_tokens: None,
            #[allow(deprecated)]
            reasoning_tokens: None,
            prompt_tokens_details: None,
            completion_tokens_details: None,
            input_tokens: UsageInputTokens {
                total: None,
                no_cache: None,
                cache_read: None,
                cache_write: None,
            },
            output_tokens: UsageOutputTokens {
                total: None,
                text: None,
                reasoning: None,
            },
            raw: None,
        }
    }

    /// Create a builder for constructing Usage with detailed token information.
    pub fn builder() -> UsageBuilder {
        UsageBuilder::default()
    }

    /// Create usage with all fields (for backward compatibility during migration).
    #[allow(deprecated)]
    pub fn with_legacy_fields(
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
        cached_tokens: Option<u32>,
        reasoning_tokens: Option<u32>,
    ) -> Self {
        let mut builder = Self::builder()
            .prompt_tokens(prompt_tokens)
            .completion_tokens(completion_tokens)
            .total_tokens(total_tokens);

        if let Some(cached_tokens) = cached_tokens {
            builder = builder.with_cached_tokens(cached_tokens);
        }
        if let Some(reasoning_tokens) = reasoning_tokens {
            builder = builder.with_reasoning_tokens(reasoning_tokens);
        }

        builder.build()
    }

    fn canonical_input_total(&self) -> Option<u32> {
        self.input_tokens.total.or_else(|| {
            if self.input_tokens.no_cache.is_none()
                && self.input_tokens.cache_read.is_none()
                && self.input_tokens.cache_write.is_none()
            {
                None
            } else {
                Some(
                    self.input_tokens
                        .no_cache
                        .unwrap_or(0)
                        .saturating_add(self.input_tokens.cache_read.unwrap_or(0))
                        .saturating_add(self.input_tokens.cache_write.unwrap_or(0)),
                )
            }
        })
    }

    fn canonical_output_total(&self) -> Option<u32> {
        self.output_tokens.total.or_else(|| {
            if self.output_tokens.text.is_none() && self.output_tokens.reasoning.is_none() {
                None
            } else {
                Some(
                    self.output_tokens
                        .text
                        .unwrap_or(0)
                        .saturating_add(self.output_tokens.reasoning.unwrap_or(0)),
                )
            }
        })
    }

    /// Return the compatibility legacy prompt token total when it is known or derivable.
    pub fn prompt_tokens_value(&self) -> Option<u32> {
        self.legacy_prompt_tokens
            .or_else(|| self.canonical_input_total())
    }

    /// Return the compatibility legacy completion token total when it is known or derivable.
    pub fn completion_tokens_value(&self) -> Option<u32> {
        self.legacy_completion_tokens
            .or_else(|| self.canonical_output_total())
    }

    /// Return the compatibility legacy total token count when it is known or derivable.
    pub fn total_tokens_value(&self) -> Option<u32> {
        self.legacy_total_tokens.or_else(|| {
            self.prompt_tokens_value()
                .zip(self.completion_tokens_value())
                .map(|(prompt, completion)| prompt.saturating_add(completion))
        })
    }

    /// Return the compatibility legacy prompt token total.
    pub fn prompt_tokens(&self) -> Option<u32> {
        self.prompt_tokens_value()
    }

    /// Return the compatibility legacy completion token total.
    pub fn completion_tokens(&self) -> Option<u32> {
        self.completion_tokens_value()
    }

    /// Return the compatibility legacy total token count.
    pub fn total_tokens(&self) -> Option<u32> {
        self.total_tokens_value()
    }

    /// Return a normalized AI SDK v4 input token summary.
    #[allow(deprecated)]
    pub fn normalized_input_tokens(&self) -> UsageInputTokens {
        let cached = self
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens)
            .or(self.cached_tokens);

        self.input_tokens
            .clone()
            .normalize(self.legacy_prompt_tokens, cached)
    }

    /// Return a normalized AI SDK v4 output token summary.
    #[allow(deprecated)]
    pub fn normalized_output_tokens(&self) -> UsageOutputTokens {
        let reasoning = self
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens)
            .or(self.reasoning_tokens);

        self.output_tokens
            .clone()
            .normalize(self.legacy_completion_tokens, reasoning)
    }

    /// Return the raw provider usage payload as a JSON value.
    pub fn raw_usage_value(&self) -> Option<Value> {
        self.raw.clone().map(Value::Object)
    }

    /// Merge usage statistics.
    pub fn merge(&mut self, other: &Usage) {
        let mut merged_input_tokens = self.normalized_input_tokens();
        let mut merged_output_tokens = self.normalized_output_tokens();
        let other_input_tokens = other.normalized_input_tokens();
        let other_output_tokens = other.normalized_output_tokens();

        let merged_prompt_tokens = self
            .prompt_tokens_value()
            .zip(other.prompt_tokens_value())
            .map(|(left, right)| left.saturating_add(right))
            .or_else(|| {
                self.prompt_tokens_value()
                    .or_else(|| other.prompt_tokens_value())
            });
        let merged_completion_tokens = self
            .completion_tokens_value()
            .zip(other.completion_tokens_value())
            .map(|(left, right)| left.saturating_add(right))
            .or_else(|| {
                self.completion_tokens_value()
                    .or_else(|| other.completion_tokens_value())
            });
        let merged_total_tokens = self
            .total_tokens_value()
            .zip(other.total_tokens_value())
            .map(|(left, right)| left.saturating_add(right))
            .or_else(|| {
                self.total_tokens_value()
                    .or_else(|| other.total_tokens_value())
            });

        self.legacy_prompt_tokens = merged_prompt_tokens;
        self.legacy_completion_tokens = merged_completion_tokens;
        self.legacy_total_tokens = merged_total_tokens;

        #[allow(deprecated)]
        {
            if let Some(cached) = other.cached_tokens {
                self.cached_tokens = Some(self.cached_tokens.unwrap_or(0).saturating_add(cached));
            }
            if let Some(reasoning) = other.reasoning_tokens {
                self.reasoning_tokens =
                    Some(self.reasoning_tokens.unwrap_or(0).saturating_add(reasoning));
            }
        }

        if let Some(ref other_details) = other.prompt_tokens_details {
            let self_details = self
                .prompt_tokens_details
                .get_or_insert_with(Default::default);
            if let Some(audio) = other_details.audio_tokens {
                self_details.audio_tokens =
                    Some(self_details.audio_tokens.unwrap_or(0).saturating_add(audio));
            }
            if let Some(cached) = other_details.cached_tokens {
                self_details.cached_tokens = Some(
                    self_details
                        .cached_tokens
                        .unwrap_or(0)
                        .saturating_add(cached),
                );
            }
        }
        if let Some(ref other_completion_details) = other.completion_tokens_details {
            let self_details = self
                .completion_tokens_details
                .get_or_insert_with(Default::default);
            if let Some(reasoning) = other_completion_details.reasoning_tokens {
                self_details.reasoning_tokens = Some(
                    self_details
                        .reasoning_tokens
                        .unwrap_or(0)
                        .saturating_add(reasoning),
                );
            }
            if let Some(audio) = other_completion_details.audio_tokens {
                self_details.audio_tokens =
                    Some(self_details.audio_tokens.unwrap_or(0).saturating_add(audio));
            }
            if let Some(accepted) = other_completion_details.accepted_prediction_tokens {
                self_details.accepted_prediction_tokens = Some(
                    self_details
                        .accepted_prediction_tokens
                        .unwrap_or(0)
                        .saturating_add(accepted),
                );
            }
            if let Some(rejected) = other_completion_details.rejected_prediction_tokens {
                self_details.rejected_prediction_tokens = Some(
                    self_details
                        .rejected_prediction_tokens
                        .unwrap_or(0)
                        .saturating_add(rejected),
                );
            }
        }

        merged_input_tokens.merge(&other_input_tokens);
        merged_output_tokens.merge(&other_output_tokens);
        self.input_tokens = merged_input_tokens;
        self.output_tokens = merged_output_tokens;
        merge_raw_usage(&mut self.raw, other.raw.as_ref());
    }
}

impl Serialize for Usage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[allow(deprecated)]
        let usage = UsageSerde {
            prompt_tokens: self.prompt_tokens_value(),
            completion_tokens: self.completion_tokens_value(),
            total_tokens: self.total_tokens_value(),
            cached_tokens: self.cached_tokens,
            reasoning_tokens: self.reasoning_tokens,
            prompt_tokens_details: self.prompt_tokens_details.clone(),
            completion_tokens_details: self.completion_tokens_details.clone(),
            input_tokens: self.input_tokens.clone(),
            output_tokens: self.output_tokens.clone(),
            raw: self.raw.clone(),
        };

        usage.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Usage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let usage = UsageSerde::deserialize(deserializer)?;

        let mut builder = Usage::builder();
        if let Some(prompt_tokens) = usage.prompt_tokens {
            builder = builder.prompt_tokens(prompt_tokens);
        }
        if let Some(completion_tokens) = usage.completion_tokens {
            builder = builder.completion_tokens(completion_tokens);
        }
        if let Some(total_tokens) = usage.total_tokens {
            builder = builder.total_tokens(total_tokens);
        }
        #[allow(deprecated)]
        {
            if let Some(cached_tokens) = usage.cached_tokens {
                builder = builder.with_cached_tokens(cached_tokens);
            }
            if let Some(reasoning_tokens) = usage.reasoning_tokens {
                builder = builder.with_reasoning_tokens(reasoning_tokens);
            }
        }
        if let Some(prompt_tokens_details) = usage.prompt_tokens_details {
            builder = builder.with_prompt_details(prompt_tokens_details);
        }
        if let Some(completion_tokens_details) = usage.completion_tokens_details {
            builder = builder.with_completion_details(completion_tokens_details);
        }
        if !usage.input_tokens.is_empty() {
            builder = builder.with_input_tokens(usage.input_tokens);
        }
        if !usage.output_tokens.is_empty() {
            builder = builder.with_output_tokens(usage.output_tokens);
        }
        if let Some(raw) = usage.raw {
            builder = builder.with_raw_usage(raw);
        }

        Ok(builder.build())
    }
}

/// Builder for constructing Usage with detailed token information.
#[derive(Default)]
pub struct UsageBuilder {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
    prompt_details: Option<PromptTokensDetails>,
    completion_details: Option<CompletionTokensDetails>,
    input_tokens: Option<UsageInputTokens>,
    output_tokens: Option<UsageOutputTokens>,
    raw: Option<Map<String, Value>>,
}

impl UsageBuilder {
    /// Set prompt tokens.
    pub fn prompt_tokens(mut self, tokens: u32) -> Self {
        self.prompt_tokens = Some(tokens);
        self
    }

    /// Set completion tokens.
    pub fn completion_tokens(mut self, tokens: u32) -> Self {
        self.completion_tokens = Some(tokens);
        self
    }

    /// Set total tokens (if not set, will be calculated as prompt + completion).
    pub fn total_tokens(mut self, tokens: u32) -> Self {
        self.total_tokens = Some(tokens);
        self
    }

    /// Add cached tokens to prompt details.
    pub fn with_cached_tokens(mut self, cached: u32) -> Self {
        let details = self.prompt_details.get_or_insert_with(Default::default);
        details.cached_tokens = Some(cached);
        self
    }

    /// Add audio input tokens to prompt details.
    pub fn with_prompt_audio_tokens(mut self, audio: u32) -> Self {
        let details = self.prompt_details.get_or_insert_with(Default::default);
        details.audio_tokens = Some(audio);
        self
    }

    /// Add reasoning tokens to completion details.
    pub fn with_reasoning_tokens(mut self, reasoning: u32) -> Self {
        let details = self.completion_details.get_or_insert_with(Default::default);
        details.reasoning_tokens = Some(reasoning);
        self
    }

    /// Add audio output tokens to completion details.
    pub fn with_completion_audio_tokens(mut self, audio: u32) -> Self {
        let details = self.completion_details.get_or_insert_with(Default::default);
        details.audio_tokens = Some(audio);
        self
    }

    /// Add accepted prediction tokens to completion details.
    pub fn with_accepted_prediction_tokens(mut self, accepted: u32) -> Self {
        let details = self.completion_details.get_or_insert_with(Default::default);
        details.accepted_prediction_tokens = Some(accepted);
        self
    }

    /// Add rejected prediction tokens to completion details.
    pub fn with_rejected_prediction_tokens(mut self, rejected: u32) -> Self {
        let details = self.completion_details.get_or_insert_with(Default::default);
        details.rejected_prediction_tokens = Some(rejected);
        self
    }

    /// Set prompt token details directly.
    pub fn with_prompt_details(mut self, details: PromptTokensDetails) -> Self {
        self.prompt_details = Some(details);
        self
    }

    /// Set completion token details directly.
    pub fn with_completion_details(mut self, details: CompletionTokensDetails) -> Self {
        self.completion_details = Some(details);
        self
    }

    /// Set the full AI SDK v4 input token summary.
    pub fn with_input_tokens(mut self, tokens: UsageInputTokens) -> Self {
        self.input_tokens = Some(tokens);
        self
    }

    /// Set the full AI SDK v4 output token summary.
    pub fn with_output_tokens(mut self, tokens: UsageOutputTokens) -> Self {
        self.output_tokens = Some(tokens);
        self
    }

    /// Set AI SDK v4 input total tokens.
    pub fn with_input_total_tokens(mut self, total: u32) -> Self {
        let tokens = self.input_tokens.get_or_insert_with(Default::default);
        tokens.total = Some(total);
        self
    }

    /// Set AI SDK v4 non-cached input tokens.
    pub fn with_input_no_cache_tokens(mut self, no_cache: u32) -> Self {
        let tokens = self.input_tokens.get_or_insert_with(Default::default);
        tokens.no_cache = Some(no_cache);
        self
    }

    /// Set AI SDK v4 cache-read input tokens.
    pub fn with_input_cache_read_tokens(mut self, cache_read: u32) -> Self {
        let tokens = self.input_tokens.get_or_insert_with(Default::default);
        tokens.cache_read = Some(cache_read);
        self
    }

    /// Set AI SDK v4 cache-write input tokens.
    pub fn with_input_cache_write_tokens(mut self, cache_write: u32) -> Self {
        let tokens = self.input_tokens.get_or_insert_with(Default::default);
        tokens.cache_write = Some(cache_write);
        self
    }

    /// Set AI SDK v4 output total tokens.
    pub fn with_output_total_tokens(mut self, total: u32) -> Self {
        let tokens = self.output_tokens.get_or_insert_with(Default::default);
        tokens.total = Some(total);
        self
    }

    /// Set AI SDK v4 output text tokens.
    pub fn with_output_text_tokens(mut self, text: u32) -> Self {
        let tokens = self.output_tokens.get_or_insert_with(Default::default);
        tokens.text = Some(text);
        self
    }

    /// Set AI SDK v4 output reasoning tokens.
    pub fn with_output_reasoning_tokens(mut self, reasoning: u32) -> Self {
        let tokens = self.output_tokens.get_or_insert_with(Default::default);
        tokens.reasoning = Some(reasoning);
        self
    }

    /// Attach the provider-native raw usage payload.
    pub fn with_raw_usage(mut self, raw: Map<String, Value>) -> Self {
        self.raw = Some(raw);
        self
    }

    /// Attach the provider-native raw usage payload from a JSON value.
    pub fn with_raw_usage_value(mut self, raw: Value) -> Self {
        if let Some(obj) = raw.as_object().cloned() {
            self.raw = Some(obj);
        }
        self
    }

    /// Build the Usage struct.
    #[allow(deprecated)]
    pub fn build(self) -> Usage {
        let cached_tokens = self.prompt_details.as_ref().and_then(|d| d.cached_tokens);
        let reasoning_tokens = self
            .completion_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens);

        let input_tokens_seed = self.input_tokens.unwrap_or_default();
        let output_tokens_seed = self.output_tokens.unwrap_or_default();

        let prompt_tokens = self.prompt_tokens.or(input_tokens_seed.total).or_else(|| {
            if input_tokens_seed.no_cache.is_none()
                && input_tokens_seed.cache_read.is_none()
                && input_tokens_seed.cache_write.is_none()
            {
                None
            } else {
                Some(
                    input_tokens_seed
                        .no_cache
                        .unwrap_or(0)
                        .saturating_add(input_tokens_seed.cache_read.unwrap_or(0))
                        .saturating_add(input_tokens_seed.cache_write.unwrap_or(0)),
                )
            }
        });
        let completion_tokens = self
            .completion_tokens
            .or(output_tokens_seed.total)
            .or_else(|| {
                if output_tokens_seed.text.is_none() && output_tokens_seed.reasoning.is_none() {
                    None
                } else {
                    Some(
                        output_tokens_seed
                            .text
                            .unwrap_or(0)
                            .saturating_add(output_tokens_seed.reasoning.unwrap_or(0)),
                    )
                }
            });
        let total_tokens = self.total_tokens.or_else(|| {
            prompt_tokens
                .zip(completion_tokens)
                .map(|(prompt, completion)| prompt.saturating_add(completion))
        });

        let input_tokens = input_tokens_seed.normalize(prompt_tokens, cached_tokens);
        let output_tokens = output_tokens_seed.normalize(completion_tokens, reasoning_tokens);

        Usage {
            legacy_prompt_tokens: prompt_tokens,
            legacy_completion_tokens: completion_tokens,
            legacy_total_tokens: total_tokens,
            cached_tokens,
            reasoning_tokens,
            prompt_tokens_details: self.prompt_details,
            completion_tokens_details: self.completion_details,
            input_tokens,
            output_tokens,
            raw: self.raw,
        }
    }
}

impl PromptTokensDetails {
    /// Create with only cached tokens.
    pub fn with_cached(cached: u32) -> Self {
        Self {
            audio_tokens: None,
            cached_tokens: Some(cached),
        }
    }

    /// Create with only audio tokens.
    pub fn with_audio(audio: u32) -> Self {
        Self {
            audio_tokens: Some(audio),
            cached_tokens: None,
        }
    }
}

impl CompletionTokensDetails {
    /// Create with only reasoning tokens.
    pub fn with_reasoning(reasoning: u32) -> Self {
        Self {
            reasoning_tokens: Some(reasoning),
            audio_tokens: None,
            accepted_prediction_tokens: None,
            rejected_prediction_tokens: None,
        }
    }

    /// Create with only audio tokens.
    pub fn with_audio(audio: u32) -> Self {
        Self {
            reasoning_tokens: None,
            audio_tokens: Some(audio),
            accepted_prediction_tokens: None,
            rejected_prediction_tokens: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_builder_populates_v4_usage_from_legacy_fields() {
        let usage = Usage::builder()
            .prompt_tokens(12)
            .completion_tokens(7)
            .total_tokens(19)
            .with_cached_tokens(3)
            .with_reasoning_tokens(2)
            .build();

        assert_eq!(usage.prompt_tokens(), Some(12));
        assert_eq!(usage.completion_tokens(), Some(7));
        assert_eq!(usage.total_tokens(), Some(19));
        assert_eq!(usage.normalized_input_tokens().total, Some(12));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(9));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(3));
        assert_eq!(usage.normalized_input_tokens().cache_write, None);
        assert_eq!(usage.normalized_output_tokens().total, Some(7));
        assert_eq!(usage.normalized_output_tokens().text, Some(5));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(2));
    }

    #[test]
    fn usage_builder_supports_explicit_v4_overrides_and_raw_usage() {
        let usage = Usage::builder()
            .prompt_tokens(17)
            .completion_tokens(5)
            .total_tokens(22)
            .with_input_tokens(UsageInputTokens {
                total: Some(32),
                no_cache: Some(17),
                cache_read: Some(5),
                cache_write: Some(10),
            })
            .with_output_tokens(UsageOutputTokens {
                total: Some(5),
                text: Some(5),
                reasoning: None,
            })
            .with_raw_usage_value(serde_json::json!({
                "input_tokens": 17,
                "output_tokens": 5,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 10,
            }))
            .build();

        assert_eq!(usage.prompt_tokens(), Some(17));
        assert_eq!(usage.completion_tokens(), Some(5));
        assert_eq!(usage.normalized_input_tokens().total, Some(32));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(17));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(5));
        assert_eq!(usage.normalized_input_tokens().cache_write, Some(10));
        assert_eq!(
            usage.raw_usage_value(),
            Some(serde_json::json!({
                "input_tokens": 17,
                "output_tokens": 5,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 10,
            }))
        );
    }

    #[test]
    fn usage_merge_accumulates_v4_breakdowns_and_merges_raw_objects() {
        let mut usage = Usage::builder()
            .prompt_tokens(10)
            .completion_tokens(4)
            .total_tokens(14)
            .with_cached_tokens(2)
            .with_reasoning_tokens(1)
            .with_raw_usage_value(serde_json::json!({
                "server_tool_use": {
                    "web_search_requests": 1
                }
            }))
            .build();

        let other = Usage::builder()
            .prompt_tokens(6)
            .completion_tokens(3)
            .total_tokens(9)
            .with_input_tokens(UsageInputTokens {
                total: Some(9),
                no_cache: Some(6),
                cache_read: Some(1),
                cache_write: Some(2),
            })
            .with_output_tokens(UsageOutputTokens {
                total: Some(3),
                text: Some(2),
                reasoning: Some(1),
            })
            .with_raw_usage_value(serde_json::json!({
                "server_tool_use": {
                    "web_fetch_requests": 2
                }
            }))
            .build();

        usage.merge(&other);

        assert_eq!(usage.prompt_tokens(), Some(16));
        assert_eq!(usage.completion_tokens(), Some(7));
        assert_eq!(usage.total_tokens(), Some(23));
        assert_eq!(usage.normalized_input_tokens().total, Some(19));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(14));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(3));
        assert_eq!(usage.normalized_input_tokens().cache_write, Some(2));
        assert_eq!(usage.normalized_output_tokens().total, Some(7));
        assert_eq!(usage.normalized_output_tokens().text, Some(5));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(2));
        assert_eq!(
            usage.raw_usage_value(),
            Some(serde_json::json!({
                "server_tool_use": {
                    "web_search_requests": 1,
                    "web_fetch_requests": 2
                }
            }))
        );
    }

    #[test]
    fn usage_builder_preserves_unknown_legacy_totals_when_only_raw_shape_exists() {
        let usage = Usage::builder()
            .with_raw_usage_value(serde_json::json!({
                "input_tokens": null,
                "output_tokens": null,
                "total_tokens": null,
            }))
            .build();

        assert_eq!(usage.prompt_tokens(), None);
        assert_eq!(usage.completion_tokens(), None);
        assert_eq!(usage.total_tokens(), None);
        assert_eq!(usage.prompt_tokens_value(), None);
        assert_eq!(usage.completion_tokens_value(), None);
        assert_eq!(usage.total_tokens_value(), None);
        assert_eq!(usage.normalized_input_tokens().total, None);
        assert_eq!(usage.normalized_output_tokens().total, None);
    }

    #[test]
    fn usage_serialization_omits_legacy_totals_when_unknown() {
        let usage = Usage::builder()
            .with_raw_usage_value(serde_json::json!({
                "input_tokens": null,
                "output_tokens": null,
                "total_tokens": null,
            }))
            .build();

        let value = serde_json::to_value(&usage).expect("serialize usage");
        assert!(value.get("prompt_tokens").is_none());
        assert!(value.get("completion_tokens").is_none());
        assert!(value.get("total_tokens").is_none());
        assert_eq!(value["raw"]["input_tokens"], serde_json::Value::Null);
    }
}
