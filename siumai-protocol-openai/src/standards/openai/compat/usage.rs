//! OpenAI-compatible usage policy.
//!
//! The generic OpenAI-compatible surface follows the AI SDK model: usage conversion is a provider
//! runtime policy. Most providers use the default OpenAI-compatible converter, while providers with
//! different token semantics install a provider-specific converter here.

use std::borrow::Cow;

use serde_json::{Map, Value};

use crate::standards::openai::utils;
use crate::types::{Usage, UsageInputTokens, UsageOutputTokens};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UsageConverter {
    OpenAiCompatibleChat,
    Alibaba,
    DeepSeek,
    Groq,
    MoonshotAi,
    XaiChat,
    DeepInfra,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UsageExtraction {
    StandardUsage,
    GroqStreaming,
}

/// Provider runtime policy for OpenAI-compatible usage extraction and conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenAiCompatibleUsagePolicy {
    converter: UsageConverter,
    extraction: UsageExtraction,
}

impl OpenAiCompatibleUsagePolicy {
    /// Resolve the usage policy for a built-in or custom OpenAI-compatible provider id.
    pub fn for_provider(provider_id: &str) -> Self {
        let normalized = provider_id.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "alibaba" | "qwen" => {
                Self::new(UsageConverter::Alibaba, UsageExtraction::StandardUsage)
            }
            "deepseek" => Self::new(UsageConverter::DeepSeek, UsageExtraction::StandardUsage),
            "groq" => Self::new(UsageConverter::Groq, UsageExtraction::GroqStreaming),
            "moonshot" | "moonshotai" => {
                Self::new(UsageConverter::MoonshotAi, UsageExtraction::StandardUsage)
            }
            "xai" => Self::new(UsageConverter::XaiChat, UsageExtraction::StandardUsage),
            "deepinfra" => Self::new(UsageConverter::DeepInfra, UsageExtraction::StandardUsage),
            _ => Self::new(
                UsageConverter::OpenAiCompatibleChat,
                UsageExtraction::StandardUsage,
            ),
        }
    }

    const fn new(converter: UsageConverter, extraction: UsageExtraction) -> Self {
        Self {
            converter,
            extraction,
        }
    }

    /// Convert a provider usage payload into Siumai's unified usage shape.
    pub fn convert_usage_value(&self, value: &Value) -> Option<Usage> {
        match self.converter {
            UsageConverter::OpenAiCompatibleChat => parse_openai_compatible_chat_usage_value(value),
            UsageConverter::Alibaba => parse_alibaba_usage_value(value),
            UsageConverter::DeepSeek => parse_deepseek_usage_value(value),
            UsageConverter::Groq => parse_groq_usage_value(value),
            UsageConverter::MoonshotAi => parse_moonshotai_usage_value(value),
            UsageConverter::XaiChat => parse_xai_chat_usage_value(value),
            UsageConverter::DeepInfra => {
                let normalized = normalize_deepinfra_usage_value(value);
                parse_openai_compatible_chat_usage_value(normalized.as_ref())
            }
        }
    }

    /// Extract and convert usage from a full OpenAI-compatible response or stream chunk.
    pub fn extract_usage(&self, raw: &Value) -> Option<Usage> {
        match self.extraction {
            UsageExtraction::StandardUsage => {
                non_null_usage(raw, &["usage"]).and_then(|usage| self.convert_usage_value(usage))
            }
            UsageExtraction::GroqStreaming => {
                for usage in [
                    non_null_usage(raw, &["x_groq", "usage"]),
                    non_null_usage(raw, &["usage"]),
                ]
                .into_iter()
                .flatten()
                {
                    if let Some(parsed) = self.convert_usage_value(usage) {
                        return Some(parsed);
                    }
                }

                None
            }
        }
    }
}

fn non_null_usage<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    let mut cursor = value;
    for key in keys {
        cursor = cursor.get(*key)?;
    }
    (!cursor.is_null()).then_some(cursor)
}

fn parse_normalized_openai_usage_with_raw(
    mut normalized: Map<String, Value>,
    raw: Map<String, Value>,
) -> Option<Usage> {
    normalized.insert("raw".to_string(), Value::Object(raw));
    utils::parse_openai_usage_value(&Value::Object(normalized))
}

struct OpenAiCompatibleChatUsageFields {
    raw: Map<String, Value>,
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: Option<u32>,
    cache_read_tokens: u32,
    cache_write_tokens: Option<u32>,
    no_cache_tokens: Option<u32>,
    reasoning_tokens: u32,
    output_text_tokens: Option<u32>,
    prompt_detail_cached_tokens: Option<u32>,
    completion_detail_reasoning_tokens: Option<u32>,
    prompt_audio_tokens: Option<u32>,
    completion_audio_tokens: Option<u32>,
    accepted_prediction_tokens: Option<u32>,
    rejected_prediction_tokens: Option<u32>,
}

fn build_openai_compatible_chat_usage(fields: OpenAiCompatibleChatUsageFields) -> Usage {
    let cache_write_for_no_cache = fields.cache_write_tokens.unwrap_or(0);
    let no_cache_tokens = fields.no_cache_tokens.unwrap_or_else(|| {
        fields
            .prompt_tokens
            .saturating_sub(fields.cache_read_tokens)
            .saturating_sub(cache_write_for_no_cache)
    });
    let output_text_tokens = fields.output_text_tokens.unwrap_or_else(|| {
        fields
            .completion_tokens
            .saturating_sub(fields.reasoning_tokens)
    });

    let mut builder = Usage::builder()
        .prompt_tokens(fields.prompt_tokens)
        .completion_tokens(fields.completion_tokens)
        .with_input_total_tokens(fields.prompt_tokens)
        .with_input_no_cache_tokens(no_cache_tokens)
        .with_input_cache_read_tokens(fields.cache_read_tokens)
        .with_output_total_tokens(fields.completion_tokens)
        .with_output_text_tokens(output_text_tokens)
        .with_output_reasoning_tokens(fields.reasoning_tokens)
        .with_raw_usage(fields.raw);

    if let Some(total_tokens) = fields.total_tokens {
        builder = builder.total_tokens(total_tokens);
    }
    if let Some(cache_write_tokens) = fields.cache_write_tokens {
        builder = builder.with_input_cache_write_tokens(cache_write_tokens);
    }
    if let Some(prompt_detail_cached_tokens) = fields.prompt_detail_cached_tokens {
        builder = builder.with_cached_tokens(prompt_detail_cached_tokens);
    }
    if let Some(completion_detail_reasoning_tokens) = fields.completion_detail_reasoning_tokens {
        builder = builder.with_reasoning_tokens(completion_detail_reasoning_tokens);
    }
    if let Some(prompt_audio_tokens) = fields.prompt_audio_tokens {
        builder = builder.with_prompt_audio_tokens(prompt_audio_tokens);
    }
    if let Some(completion_audio_tokens) = fields.completion_audio_tokens {
        builder = builder.with_completion_audio_tokens(completion_audio_tokens);
    }
    if let Some(accepted_prediction_tokens) = fields.accepted_prediction_tokens {
        builder = builder.with_accepted_prediction_tokens(accepted_prediction_tokens);
    }
    if let Some(rejected_prediction_tokens) = fields.rejected_prediction_tokens {
        builder = builder.with_rejected_prediction_tokens(rejected_prediction_tokens);
    }

    builder.build()
}

fn parse_openai_compatible_chat_usage_value(value: &Value) -> Option<Usage> {
    let object = value.as_object()?;
    let raw = object
        .get("raw")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_else(|| object.clone());

    let (input_total, input_no_cache, input_cache_read, input_cache_write) =
        utils::parse_input_tokens_value(object.get("inputTokens"));
    let (output_total, output_text, output_reasoning) =
        utils::parse_output_tokens_value(object.get("outputTokens"));

    let prompt_tokens = utils::usage_u32(utils::usage_value(
        object,
        &["prompt_tokens", "input_tokens"],
    ))
    .or(input_total)
    .unwrap_or(0);
    let completion_tokens = utils::usage_u32(utils::usage_value(
        object,
        &["completion_tokens", "output_tokens"],
    ))
    .or(output_total)
    .unwrap_or(0);
    let total_tokens =
        utils::usage_u32(utils::usage_value(object, &["total_tokens", "totalTokens"]));

    let prompt_details =
        utils::usage_object(object, &["prompt_tokens_details", "input_tokens_details"]);
    let completion_details = utils::usage_object(
        object,
        &["completion_tokens_details", "output_tokens_details"],
    );
    let cache_read_tokens = utils::usage_u32(
        prompt_details
            .and_then(|details| utils::usage_value(details, &["cached_tokens", "cachedTokens"])),
    )
    .or(input_cache_read);
    let reasoning_tokens = utils::usage_u32(utils::usage_value(
        object,
        &["reasoning_tokens", "reasoningTokens"],
    ))
    .or_else(|| {
        utils::usage_u32(completion_details.and_then(|details| {
            utils::usage_value(details, &["reasoning_tokens", "reasoningTokens"])
        }))
    })
    .or(output_reasoning);
    let prompt_audio_tokens = utils::usage_u32(
        prompt_details
            .and_then(|details| utils::usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let completion_audio_tokens = utils::usage_u32(
        completion_details
            .and_then(|details| utils::usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let accepted_prediction_tokens = utils::usage_u32(completion_details.and_then(|details| {
        utils::usage_value(
            details,
            &["accepted_prediction_tokens", "acceptedPredictionTokens"],
        )
    }));
    let rejected_prediction_tokens = utils::usage_u32(completion_details.and_then(|details| {
        utils::usage_value(
            details,
            &["rejected_prediction_tokens", "rejectedPredictionTokens"],
        )
    }));

    Some(build_openai_compatible_chat_usage(
        OpenAiCompatibleChatUsageFields {
            raw,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cache_read_tokens: cache_read_tokens.unwrap_or(0),
            cache_write_tokens: input_cache_write,
            no_cache_tokens: input_no_cache,
            reasoning_tokens: reasoning_tokens.unwrap_or(0),
            output_text_tokens: output_text,
            prompt_detail_cached_tokens: cache_read_tokens,
            completion_detail_reasoning_tokens: reasoning_tokens,
            prompt_audio_tokens,
            completion_audio_tokens,
            accepted_prediction_tokens,
            rejected_prediction_tokens,
        },
    ))
}

fn parse_deepseek_usage_value(value: &Value) -> Option<Usage> {
    let raw = value.as_object()?.clone();
    let mut normalized = raw.clone();

    if let Some(cache_read_tokens) =
        utils::usage_u32(utils::usage_value(&raw, &["prompt_cache_hit_tokens"]))
    {
        utils::set_usage_detail_number(
            &mut normalized,
            "prompt_tokens_details",
            "cached_tokens",
            cache_read_tokens,
        );
    }

    parse_normalized_openai_usage_with_raw(normalized, raw)
}

fn parse_groq_usage_value(value: &Value) -> Option<Usage> {
    let raw = value.as_object()?.clone();
    let mut normalized = raw.clone();

    if let Some(details) = normalized
        .get_mut("prompt_tokens_details")
        .and_then(Value::as_object_mut)
    {
        details.remove("cached_tokens");
        details.remove("cachedTokens");
        if details.is_empty() {
            normalized.remove("prompt_tokens_details");
        }
    }

    parse_normalized_openai_usage_with_raw(normalized, raw)
}

fn parse_moonshotai_usage_value(value: &Value) -> Option<Usage> {
    let raw = value.as_object()?.clone();

    let prompt_tokens =
        utils::usage_u32(utils::usage_value(&raw, &["prompt_tokens", "input_tokens"])).unwrap_or(0);
    let completion_tokens = utils::usage_u32(utils::usage_value(
        &raw,
        &["completion_tokens", "output_tokens"],
    ))
    .unwrap_or(0);
    let total_tokens = utils::usage_u32(utils::usage_value(&raw, &["total_tokens", "totalTokens"]));
    let prompt_details =
        utils::usage_object(&raw, &["prompt_tokens_details", "input_tokens_details"]);
    let completion_details = utils::usage_object(
        &raw,
        &["completion_tokens_details", "output_tokens_details"],
    );
    let cache_read_tokens =
        utils::usage_u32(utils::usage_value(&raw, &["cached_tokens", "cachedTokens"])).or_else(
            || {
                utils::usage_u32(prompt_details.and_then(|details| {
                    utils::usage_value(details, &["cached_tokens", "cachedTokens"])
                }))
            },
        );
    let reasoning_tokens =
        utils::usage_u32(completion_details.and_then(|details| {
            utils::usage_value(details, &["reasoning_tokens", "reasoningTokens"])
        }));
    let prompt_audio_tokens = utils::usage_u32(
        prompt_details
            .and_then(|details| utils::usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let completion_audio_tokens = utils::usage_u32(
        completion_details
            .and_then(|details| utils::usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let accepted_prediction_tokens = utils::usage_u32(completion_details.and_then(|details| {
        utils::usage_value(
            details,
            &["accepted_prediction_tokens", "acceptedPredictionTokens"],
        )
    }));
    let rejected_prediction_tokens = utils::usage_u32(completion_details.and_then(|details| {
        utils::usage_value(
            details,
            &["rejected_prediction_tokens", "rejectedPredictionTokens"],
        )
    }));

    Some(build_openai_compatible_chat_usage(
        OpenAiCompatibleChatUsageFields {
            raw,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cache_read_tokens: cache_read_tokens.unwrap_or(0),
            cache_write_tokens: None,
            no_cache_tokens: None,
            reasoning_tokens: reasoning_tokens.unwrap_or(0),
            output_text_tokens: None,
            prompt_detail_cached_tokens: cache_read_tokens,
            completion_detail_reasoning_tokens: reasoning_tokens,
            prompt_audio_tokens,
            completion_audio_tokens,
            accepted_prediction_tokens,
            rejected_prediction_tokens,
        },
    ))
}

fn parse_alibaba_usage_value(value: &Value) -> Option<Usage> {
    let raw = value.as_object()?.clone();

    let prompt_tokens =
        utils::usage_u32(utils::usage_value(&raw, &["prompt_tokens", "input_tokens"])).unwrap_or(0);
    let completion_tokens = utils::usage_u32(utils::usage_value(
        &raw,
        &["completion_tokens", "output_tokens"],
    ))
    .unwrap_or(0);
    let total_tokens = utils::usage_u32(utils::usage_value(&raw, &["total_tokens", "totalTokens"]));
    let prompt_details =
        utils::usage_object(&raw, &["prompt_tokens_details", "input_tokens_details"]);
    let completion_details = utils::usage_object(
        &raw,
        &["completion_tokens_details", "output_tokens_details"],
    );
    let cache_read_tokens = utils::usage_u32(
        prompt_details
            .and_then(|details| utils::usage_value(details, &["cached_tokens", "cachedTokens"])),
    );
    let cache_write_tokens = utils::usage_u32(prompt_details.and_then(|details| {
        utils::usage_value(
            details,
            &["cache_creation_input_tokens", "cacheCreationInputTokens"],
        )
    }))
    .unwrap_or(0);
    let reasoning_tokens =
        utils::usage_u32(completion_details.and_then(|details| {
            utils::usage_value(details, &["reasoning_tokens", "reasoningTokens"])
        }));
    let prompt_audio_tokens = utils::usage_u32(
        prompt_details
            .and_then(|details| utils::usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let completion_audio_tokens = utils::usage_u32(
        completion_details
            .and_then(|details| utils::usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let accepted_prediction_tokens = utils::usage_u32(completion_details.and_then(|details| {
        utils::usage_value(
            details,
            &["accepted_prediction_tokens", "acceptedPredictionTokens"],
        )
    }));
    let rejected_prediction_tokens = utils::usage_u32(completion_details.and_then(|details| {
        utils::usage_value(
            details,
            &["rejected_prediction_tokens", "rejectedPredictionTokens"],
        )
    }));

    Some(build_openai_compatible_chat_usage(
        OpenAiCompatibleChatUsageFields {
            raw,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cache_read_tokens: cache_read_tokens.unwrap_or(0),
            cache_write_tokens: Some(cache_write_tokens),
            no_cache_tokens: None,
            reasoning_tokens: reasoning_tokens.unwrap_or(0),
            output_text_tokens: None,
            prompt_detail_cached_tokens: cache_read_tokens,
            completion_detail_reasoning_tokens: reasoning_tokens,
            prompt_audio_tokens,
            completion_audio_tokens,
            accepted_prediction_tokens,
            rejected_prediction_tokens,
        },
    ))
}

fn parse_xai_chat_usage_value(value: &Value) -> Option<Usage> {
    let raw = value.as_object()?.clone();
    let mut normalized = raw.clone();

    let prompt_tokens =
        utils::usage_u32(utils::usage_value(&raw, &["prompt_tokens", "input_tokens"]));
    let cache_read_tokens =
        utils::usage_object(&raw, &["prompt_tokens_details", "input_tokens_details"])
            .and_then(|details| {
                utils::usage_u32(utils::usage_value(
                    details,
                    &["cached_tokens", "cachedTokens"],
                ))
            })
            .unwrap_or(0);

    utils::set_usage_detail_number(
        &mut normalized,
        "prompt_tokens_details",
        "cached_tokens",
        cache_read_tokens,
    );

    let normalized_input_total = prompt_tokens.map(|prompt_tokens| {
        let input_total = if cache_read_tokens > prompt_tokens {
            prompt_tokens.saturating_add(cache_read_tokens)
        } else {
            prompt_tokens
        };
        normalized.insert("prompt_tokens".to_string(), serde_json::json!(input_total));
        input_total
    });

    let completion_tokens = utils::usage_u32(utils::usage_value(
        &raw,
        &["completion_tokens", "output_tokens"],
    ));
    let reasoning_tokens = utils::usage_object(
        &raw,
        &["completion_tokens_details", "output_tokens_details"],
    )
    .and_then(|details| {
        utils::usage_u32(utils::usage_value(
            details,
            &["reasoning_tokens", "reasoningTokens"],
        ))
    })
    .unwrap_or(0);

    utils::set_usage_detail_number(
        &mut normalized,
        "completion_tokens_details",
        "reasoning_tokens",
        reasoning_tokens,
    );

    let normalized_output_total = completion_tokens.map(|completion_tokens| {
        let output_total = completion_tokens.saturating_add(reasoning_tokens);
        normalized.insert(
            "completion_tokens".to_string(),
            serde_json::json!(output_total),
        );
        output_total
    });

    if let Some(total_tokens) = normalized_input_total
        .zip(normalized_output_total)
        .map(|(input, output)| input.saturating_add(output))
    {
        normalized.insert("total_tokens".to_string(), serde_json::json!(total_tokens));
    }

    parse_normalized_openai_usage_with_raw(normalized, raw)
}

fn normalize_deepinfra_usage_value<'a>(value: &'a Value) -> Cow<'a, Value> {
    let Some(object) = value.as_object() else {
        return Cow::Borrowed(value);
    };

    let completion_tokens = utils::usage_u32(utils::usage_value(
        object,
        &["completion_tokens", "output_tokens"],
    ));
    let completion_details = utils::usage_object(
        object,
        &["completion_tokens_details", "output_tokens_details"],
    );
    let reasoning_tokens = utils::usage_u32(utils::usage_value(
        object,
        &["reasoning_tokens", "reasoningTokens"],
    ))
    .or_else(|| {
        utils::usage_u32(completion_details.and_then(|details| {
            utils::usage_value(details, &["reasoning_tokens", "reasoningTokens"])
        }))
    });

    let (Some(completion_tokens), Some(reasoning_tokens)) = (completion_tokens, reasoning_tokens)
    else {
        return Cow::Borrowed(value);
    };

    if reasoning_tokens <= completion_tokens {
        return Cow::Borrowed(value);
    }

    let corrected_completion_tokens = completion_tokens.saturating_add(reasoning_tokens);
    let mut fixed = object.clone();

    if fixed.contains_key("completion_tokens") {
        fixed.insert(
            "completion_tokens".to_string(),
            serde_json::json!(corrected_completion_tokens),
        );
    }
    if fixed.contains_key("output_tokens") {
        fixed.insert(
            "output_tokens".to_string(),
            serde_json::json!(corrected_completion_tokens),
        );
    }

    if let Some(total_tokens) =
        utils::usage_u32(utils::usage_value(object, &["total_tokens", "totalTokens"]))
    {
        let corrected_total_tokens = total_tokens.saturating_add(reasoning_tokens);
        if fixed.contains_key("total_tokens") {
            fixed.insert(
                "total_tokens".to_string(),
                serde_json::json!(corrected_total_tokens),
            );
        }
        if fixed.contains_key("totalTokens") {
            fixed.insert(
                "totalTokens".to_string(),
                serde_json::json!(corrected_total_tokens),
            );
        }
    }

    if let Some(output_tokens) = fixed.get_mut("outputTokens").and_then(Value::as_object_mut) {
        if !output_tokens.contains_key("text") && !output_tokens.contains_key("textTokens") {
            output_tokens.insert("text".to_string(), serde_json::json!(completion_tokens));
        }
        output_tokens.insert(
            if output_tokens.contains_key("totalTokens") {
                "totalTokens".to_string()
            } else {
                "total".to_string()
            },
            serde_json::json!(corrected_completion_tokens),
        );
    }

    Cow::Owned(Value::Object(fixed))
}

/// Parse xAI Responses usage payloads into the unified `Usage` shape.
pub fn parse_xai_responses_usage_value(value: &Value) -> Option<Usage> {
    let raw = value.as_object()?.clone();
    let mut normalized = raw.clone();

    let input_tokens =
        utils::usage_u32(utils::usage_value(&raw, &["input_tokens", "prompt_tokens"]));
    let cache_read_tokens =
        utils::usage_object(&raw, &["input_tokens_details", "prompt_tokens_details"])
            .and_then(|details| {
                utils::usage_u32(utils::usage_value(
                    details,
                    &["cached_tokens", "cachedTokens"],
                ))
            })
            .unwrap_or(0);

    utils::set_usage_detail_number(
        &mut normalized,
        "input_tokens_details",
        "cached_tokens",
        cache_read_tokens,
    );

    let normalized_input_total = input_tokens.map(|input_tokens| {
        let input_total = if cache_read_tokens > input_tokens {
            input_tokens.saturating_add(cache_read_tokens)
        } else {
            input_tokens
        };
        normalized.insert("input_tokens".to_string(), serde_json::json!(input_total));
        input_total
    });

    let reasoning_tokens = utils::usage_object(&raw, &["output_tokens_details"])
        .and_then(|details| {
            utils::usage_u32(utils::usage_value(
                details,
                &["reasoning_tokens", "reasoningTokens"],
            ))
        })
        .unwrap_or(0);

    utils::set_usage_detail_number(
        &mut normalized,
        "output_tokens_details",
        "reasoning_tokens",
        reasoning_tokens,
    );

    if let Some(total_tokens) = normalized_input_total
        .zip(utils::usage_u32(utils::usage_value(
            &raw,
            &["output_tokens"],
        )))
        .map(|(input, output)| input.saturating_add(output))
    {
        normalized.insert("total_tokens".to_string(), serde_json::json!(total_tokens));
    }

    parse_normalized_openai_usage_with_raw(normalized, raw)
}

/// Return the xAI Responses fallback usage when the provider omits `usage`.
///
/// AI SDK treats missing xAI Responses usage as explicit zero counts instead of unknown counts.
pub fn xai_responses_zero_usage() -> Usage {
    Usage::builder()
        .with_input_tokens(UsageInputTokens {
            total: Some(0),
            no_cache: Some(0),
            cache_read: Some(0),
            cache_write: Some(0),
        })
        .with_output_tokens(UsageOutputTokens {
            total: Some(0),
            text: Some(0),
            reasoning: Some(0),
        })
        .build()
}

/// Extract xAI Responses usage-owned provider metadata.
///
/// AI SDK exposes `usage.cost_in_usd_ticks` as
/// `providerMetadata.xai.costInUsdTicks` instead of folding it into unified usage.
pub fn xai_responses_usage_provider_metadata_value(value: &Value) -> Option<Value> {
    let object = value.as_object()?;
    let cost = utils::usage_value(object, &["cost_in_usd_ticks", "costInUsdTicks"])
        .filter(|value| !value.is_null())?;

    Some(serde_json::json!({ "costInUsdTicks": cost.clone() }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_policy_defaults_to_openai_compatible_chat_usage() {
        let usage = OpenAiCompatibleUsagePolicy::for_provider("openrouter")
            .convert_usage_value(&serde_json::json!({
                "prompt_tokens": 100,
                "completion_tokens": 50
            }))
            .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(100));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(100));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(0));
        assert_eq!(usage.normalized_input_tokens().cache_write, None);
        assert_eq!(usage.normalized_output_tokens().total, Some(50));
        assert_eq!(usage.normalized_output_tokens().text, Some(50));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(0));
    }

    #[test]
    fn groq_policy_reads_streaming_x_groq_usage() {
        let usage = OpenAiCompatibleUsagePolicy::for_provider("groq")
            .extract_usage(&serde_json::json!({
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }],
                "x_groq": {
                    "usage": {
                        "queue_time": 0.061348671,
                        "prompt_tokens": 18,
                        "completion_tokens": 439,
                        "total_tokens": 457
                    }
                }
            }))
            .expect("parse x_groq usage");

        assert_eq!(usage.prompt_tokens(), Some(18));
        assert_eq!(usage.completion_tokens(), Some(439));
        assert_eq!(usage.total_tokens(), Some(457));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["queue_time"],
            serde_json::json!(0.061348671)
        );
    }

    #[test]
    fn deepinfra_policy_fixes_reasoning_totals() {
        let usage = OpenAiCompatibleUsagePolicy::for_provider("deepinfra")
            .convert_usage_value(&serde_json::json!({
                "prompt_tokens": 21,
                "completion_tokens": 84,
                "total_tokens": 105,
                "completion_tokens_details": {
                    "reasoning_tokens": 1081
                }
            }))
            .expect("parse usage");

        assert_eq!(usage.prompt_tokens(), Some(21));
        assert_eq!(usage.completion_tokens(), Some(1165));
        assert_eq!(usage.total_tokens(), Some(1186));
        assert_eq!(usage.normalized_output_tokens().total, Some(1165));
        assert_eq!(usage.normalized_output_tokens().text, Some(84));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(1081));
    }

    #[test]
    fn alibaba_policy_maps_cache_creation_tokens() {
        let usage = OpenAiCompatibleUsagePolicy::for_provider("qwen")
            .convert_usage_value(&serde_json::json!({
                "prompt_tokens": 200,
                "completion_tokens": 75,
                "total_tokens": 275,
                "prompt_tokens_details": {
                    "cached_tokens": 120,
                    "cache_creation_input_tokens": 50
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 25
                }
            }))
            .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(200));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(30));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(120));
        assert_eq!(usage.normalized_input_tokens().cache_write, Some(50));
        assert_eq!(usage.normalized_output_tokens().total, Some(75));
        assert_eq!(usage.normalized_output_tokens().text, Some(50));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(25));
    }

    #[test]
    fn alibaba_policy_maps_missing_cache_write_to_zero() {
        let usage = OpenAiCompatibleUsagePolicy::for_provider("alibaba")
            .convert_usage_value(&serde_json::json!({
                "prompt_tokens": 200,
                "completion_tokens": 75,
                "prompt_tokens_details": {
                    "cached_tokens": 120
                }
            }))
            .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(200));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(80));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(120));
        assert_eq!(usage.normalized_input_tokens().cache_write, Some(0));
        assert_eq!(usage.normalized_output_tokens().total, Some(75));
        assert_eq!(usage.normalized_output_tokens().text, Some(75));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(0));
    }

    #[test]
    fn deepseek_policy_maps_prompt_cache_hits() {
        let usage = OpenAiCompatibleUsagePolicy::for_provider("deepseek")
            .convert_usage_value(&serde_json::json!({
                "prompt_tokens": 495,
                "completion_tokens": 157,
                "total_tokens": 652,
                "prompt_cache_hit_tokens": 320,
                "prompt_cache_miss_tokens": 175,
                "completion_tokens_details": {
                    "reasoning_tokens": 118
                }
            }))
            .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(495));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(175));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(320));
        assert_eq!(usage.normalized_output_tokens().total, Some(157));
        assert_eq!(usage.normalized_output_tokens().text, Some(39));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(118));
    }

    #[test]
    fn moonshot_policy_prefers_top_level_cached_tokens() {
        let usage = OpenAiCompatibleUsagePolicy::for_provider("moonshotai")
            .convert_usage_value(&serde_json::json!({
                "prompt_tokens": 100,
                "completion_tokens": 80,
                "cached_tokens": 35,
                "prompt_tokens_details": {
                    "cached_tokens": 25
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 30
                }
            }))
            .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(100));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(65));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(35));
        assert_eq!(usage.normalized_output_tokens().total, Some(80));
        assert_eq!(usage.normalized_output_tokens().text, Some(50));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(30));
    }

    #[test]
    fn moonshot_policy_maps_null_fields_to_zero() {
        let usage = OpenAiCompatibleUsagePolicy::for_provider("moonshotai")
            .convert_usage_value(&serde_json::json!({
                "prompt_tokens": null,
                "completion_tokens": null,
                "cached_tokens": null
            }))
            .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(0));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(0));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(0));
        assert_eq!(usage.normalized_input_tokens().cache_write, None);
        assert_eq!(usage.normalized_output_tokens().total, Some(0));
        assert_eq!(usage.normalized_output_tokens().text, Some(0));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(0));
    }

    #[test]
    fn xai_chat_policy_maps_reasoning_and_cache_semantics() {
        let usage = OpenAiCompatibleUsagePolicy::for_provider("xai")
            .convert_usage_value(&serde_json::json!({
                "prompt_tokens": 4142,
                "completion_tokens": 254,
                "total_tokens": 8734,
                "prompt_tokens_details": {
                    "cached_tokens": 4328
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 10
                }
            }))
            .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(8470));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(4142));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(4328));
        assert_eq!(usage.normalized_output_tokens().total, Some(264));
        assert_eq!(usage.normalized_output_tokens().text, Some(254));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(10));
        assert_eq!(usage.total_tokens(), Some(8734));
    }

    #[test]
    fn groq_policy_keeps_prompt_cache_details_raw_only() {
        let usage = OpenAiCompatibleUsagePolicy::for_provider("groq")
            .convert_usage_value(&serde_json::json!({
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "prompt_tokens_details": {
                    "cached_tokens": 5
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 3
                }
            }))
            .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(20));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(20));
        assert_eq!(usage.normalized_input_tokens().cache_read, None);
        assert_eq!(usage.normalized_output_tokens().total, Some(10));
        assert_eq!(usage.normalized_output_tokens().text, Some(7));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(3));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["prompt_tokens_details"]["cached_tokens"],
            serde_json::json!(5)
        );
    }

    #[test]
    fn xai_responses_usage_handles_noninclusive_cache_read_tokens() {
        let usage = parse_xai_responses_usage_value(&serde_json::json!({
            "input_tokens": 4142,
            "output_tokens": 254,
            "total_tokens": 4396,
            "input_tokens_details": {
                "cached_tokens": 4328
            },
            "output_tokens_details": {
                "reasoning_tokens": 10
            }
        }))
        .expect("parse usage");

        assert_eq!(usage.normalized_input_tokens().total, Some(8470));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(4142));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(4328));
        assert_eq!(usage.normalized_output_tokens().total, Some(254));
        assert_eq!(usage.normalized_output_tokens().text, Some(244));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(10));
        assert_eq!(usage.total_tokens(), Some(8724));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["input_tokens"],
            serde_json::json!(4142)
        );
    }

    #[test]
    fn xai_responses_zero_usage_matches_ai_sdk_missing_usage_fallback() {
        let usage = xai_responses_zero_usage();

        assert_eq!(usage.normalized_input_tokens().total, Some(0));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(0));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(0));
        assert_eq!(usage.normalized_input_tokens().cache_write, Some(0));
        assert_eq!(usage.normalized_output_tokens().total, Some(0));
        assert_eq!(usage.normalized_output_tokens().text, Some(0));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(0));
        assert!(usage.raw_usage_value().is_none());
    }

    #[test]
    fn xai_responses_usage_provider_metadata_maps_cost_ticks() {
        let metadata = xai_responses_usage_provider_metadata_value(&serde_json::json!({
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_in_usd_ticks": 113500
        }))
        .expect("provider metadata");

        assert_eq!(metadata["costInUsdTicks"], serde_json::json!(113500));
    }
}
