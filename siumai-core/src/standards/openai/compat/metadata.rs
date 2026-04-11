//! Shared provider metadata extraction helpers for OpenAI-compatible providers.

use serde_json::{Map, Value};
use std::collections::HashMap;

pub type NestedProviderMetadata = crate::types::ProviderMetadataMap;

pub fn nested_provider_metadata_to_map(
    metadata: NestedProviderMetadata,
) -> crate::types::ProviderMetadataMap {
    metadata
}

pub fn provider_options_key(provider_id: &str) -> String {
    provider_id
        .split('.')
        .next()
        .unwrap_or(provider_id)
        .trim()
        .to_ascii_lowercase()
}

pub fn to_camel_case(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut uppercase_next = false;

    for ch in value.chars() {
        if matches!(ch, '-' | '_') {
            uppercase_next = true;
            continue;
        }

        if uppercase_next {
            out.extend(ch.to_uppercase());
            uppercase_next = false;
        } else {
            out.push(ch);
        }
    }

    out
}

pub fn resolve_provider_metadata_key(
    provider_id: &str,
    provider_options: Option<&crate::types::ProviderOptionsMap>,
) -> String {
    let raw_name = provider_options_key(provider_id);
    let camel_name = to_camel_case(&raw_name);

    if camel_name != raw_name
        && provider_options
            .and_then(|options| options.get(&camel_name))
            .is_some()
    {
        camel_name
    } else {
        raw_name
    }
}

pub fn ensure_provider_metadata_namespace(
    metadata: Option<NestedProviderMetadata>,
    provider_key: &str,
    raw_provider_key: &str,
) -> NestedProviderMetadata {
    let mut metadata = metadata.unwrap_or_default();

    if provider_key != raw_provider_key
        && let Some(entries) = metadata.remove(raw_provider_key)
    {
        crate::types::provider_metadata::merge_provider_metadata(
            &mut metadata,
            HashMap::from([(provider_key.to_string(), entries)]),
        );
    }

    metadata
        .entry(provider_key.to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    metadata
}

fn namespaced_provider_metadata(
    provider_id: &str,
    meta: HashMap<String, serde_json::Value>,
) -> Option<NestedProviderMetadata> {
    if meta.is_empty() {
        None
    } else {
        Some(crate::types::provider_metadata::provider_metadata_from_object(provider_id, meta))
    }
}

pub(super) fn extract_openai_compatible_provider_metadata(
    provider_id: &str,
    raw: &serde_json::Value,
) -> Option<NestedProviderMetadata> {
    let mut meta = HashMap::<String, serde_json::Value>::new();

    if let Some(sources) = raw
        .get("sources")
        .filter(|value| !value.is_null())
        .filter(|value| value.as_array().is_some_and(|arr| !arr.is_empty()))
    {
        meta.insert("sources".to_string(), sources.clone());
    }

    if let Some(logprobs) = raw
        .get("choices")
        .and_then(|value| value.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("logprobs"))
        .and_then(|logprobs| logprobs.get("content"))
        .filter(|value| !value.is_null())
    {
        meta.insert("logprobs".to_string(), logprobs.clone());
    }

    if let Some(usage) = raw
        .get("usage")
        .and_then(crate::standards::openai::utils::parse_openai_usage_value)
        && let Some(details) = usage.completion_tokens_details.as_ref()
    {
        if let Some(accepted_prediction_tokens) = details.accepted_prediction_tokens {
            meta.insert(
                "acceptedPredictionTokens".to_string(),
                serde_json::json!(accepted_prediction_tokens),
            );
        }
        if let Some(rejected_prediction_tokens) = details.rejected_prediction_tokens {
            meta.insert(
                "rejectedPredictionTokens".to_string(),
                serde_json::json!(rejected_prediction_tokens),
            );
        }
    }

    namespaced_provider_metadata(provider_id, meta)
}

pub(super) fn extract_perplexity_provider_metadata(
    provider_id: &str,
    raw: &serde_json::Value,
) -> Option<NestedProviderMetadata> {
    // Perplexity extends the OpenAI-like response schema with extra fields such as
    // `search_results` and `videos` (see Perplexity OpenAPI spec). These are intentionally
    // exposed as provider metadata instead of being added to the unified surface.
    let mut meta = HashMap::<String, serde_json::Value>::new();
    for key in ["search_results", "videos", "citations"] {
        if let Some(value) = raw.get(key).filter(|value| !value.is_null()) {
            meta.insert(key.to_string(), value.clone());
        }
    }

    let usage = raw.get("usage").and_then(|value| value.as_object());
    meta.insert(
        "images".to_string(),
        map_perplexity_images_metadata(raw.get("images")),
    );
    meta.insert(
        "usage".to_string(),
        Value::Object(Map::from_iter([
            (
                "citationTokens".to_string(),
                perplexity_usage_field(usage, "citation_tokens", "citationTokens"),
            ),
            (
                "numSearchQueries".to_string(),
                perplexity_usage_field(usage, "num_search_queries", "numSearchQueries"),
            ),
            (
                "reasoningTokens".to_string(),
                perplexity_usage_field(usage, "reasoning_tokens", "reasoningTokens"),
            ),
        ])),
    );
    meta.insert(
        "cost".to_string(),
        map_perplexity_cost_metadata(
            usage
                .and_then(|usage| usage.get("cost"))
                .filter(|value| !value.is_null()),
        ),
    );

    namespaced_provider_metadata(provider_id, meta)
}

fn perplexity_usage_field(
    usage: Option<&Map<String, Value>>,
    snake_case: &str,
    camel_case: &str,
) -> Value {
    usage
        .and_then(|usage| usage.get(snake_case).or_else(|| usage.get(camel_case)))
        .cloned()
        .unwrap_or(Value::Null)
}

fn map_perplexity_images_metadata(images: Option<&Value>) -> Value {
    match images.filter(|value| !value.is_null()) {
        Some(Value::Array(images)) => Value::Array(
            images
                .iter()
                .map(|image| match image {
                    Value::Object(image) => {
                        let mut mapped = Map::new();

                        if let Some(value) =
                            image.get("image_url").or_else(|| image.get("imageUrl"))
                        {
                            mapped.insert("imageUrl".to_string(), value.clone());
                        }
                        if let Some(value) =
                            image.get("origin_url").or_else(|| image.get("originUrl"))
                        {
                            mapped.insert("originUrl".to_string(), value.clone());
                        }

                        for (key, value) in image {
                            if matches!(
                                key.as_str(),
                                "image_url" | "imageUrl" | "origin_url" | "originUrl"
                            ) {
                                continue;
                            }
                            mapped.insert(key.clone(), value.clone());
                        }

                        Value::Object(mapped)
                    }
                    value => value.clone(),
                })
                .collect(),
        ),
        Some(value) => value.clone(),
        None => Value::Null,
    }
}

fn map_perplexity_cost_metadata(cost: Option<&Value>) -> Value {
    match cost {
        Some(Value::Object(cost)) => Value::Object(Map::from_iter([
            (
                "inputTokensCost".to_string(),
                cost.get("input_tokens_cost")
                    .or_else(|| cost.get("inputTokensCost"))
                    .cloned()
                    .unwrap_or(Value::Null),
            ),
            (
                "outputTokensCost".to_string(),
                cost.get("output_tokens_cost")
                    .or_else(|| cost.get("outputTokensCost"))
                    .cloned()
                    .unwrap_or(Value::Null),
            ),
            (
                "requestCost".to_string(),
                cost.get("request_cost")
                    .or_else(|| cost.get("requestCost"))
                    .cloned()
                    .unwrap_or(Value::Null),
            ),
            (
                "totalCost".to_string(),
                cost.get("total_cost")
                    .or_else(|| cost.get("totalCost"))
                    .cloned()
                    .unwrap_or(Value::Null),
            ),
        ])),
        Some(value) => value.clone(),
        None => Value::Null,
    }
}

pub(super) fn merge_nested_provider_metadata(
    target: &mut NestedProviderMetadata,
    source: NestedProviderMetadata,
) {
    crate::types::provider_metadata::merge_provider_metadata(target, source);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perplexity_metadata_helper_extracts_hosted_search_fields() {
        let raw = serde_json::json!({
            "id": "resp_1",
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
                "citation_tokens": 4,
                "num_search_queries": 1,
                "reasoning_tokens": 6,
                "cost": {
                    "input_tokens_cost": 0.12,
                    "output_tokens_cost": 0.34,
                    "request_cost": 0.01,
                    "total_cost": 0.47
                }
            },
            "citations": ["https://example.com"],
            "images": [{
                "image_url": "https://images.example.com/rust.png",
                "origin_url": "https://example.com",
                "height": 900,
                "width": 1600
            }]
        });

        let meta =
            extract_perplexity_provider_metadata("perplexity", &raw).expect("metadata present");
        let perplexity = meta.get("perplexity").expect("perplexity namespace");
        assert_eq!(
            perplexity.get("citations"),
            Some(&serde_json::json!(["https://example.com"]))
        );
        assert_eq!(perplexity["usage"]["citationTokens"], serde_json::json!(4));
        assert_eq!(
            perplexity["usage"]["numSearchQueries"],
            serde_json::json!(1)
        );
        assert_eq!(perplexity["usage"]["reasoningTokens"], serde_json::json!(6));
        assert_eq!(
            perplexity["images"][0]["imageUrl"],
            serde_json::json!("https://images.example.com/rust.png")
        );
        assert_eq!(perplexity["cost"]["requestCost"], serde_json::json!(0.01));
    }

    #[test]
    fn xai_metadata_helper_extracts_sources_and_logprobs() {
        let raw = serde_json::json!({
            "id": "resp_1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello"
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [{
                        "token": "hello",
                        "logprob": -0.1,
                        "bytes": [104, 101, 108, 108, 111],
                        "top_logprobs": []
                    }]
                }
            }],
            "sources": [{
                "id": "src_1",
                "source_type": "url",
                "url": "https://example.com",
                "title": "Example"
            }]
        });

        let meta =
            extract_openai_compatible_provider_metadata("xai", &raw).expect("metadata present");
        let xai = meta.get("xai").expect("xai namespace");
        assert_eq!(
            xai["sources"][0]["url"],
            serde_json::json!("https://example.com")
        );
        assert_eq!(xai["logprobs"][0]["token"], serde_json::json!("hello"));
    }

    #[test]
    fn openai_metadata_helper_extracts_logprobs() {
        let raw = serde_json::json!({
            "id": "resp_1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello"
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [{
                        "token": "hello",
                        "logprob": -0.1,
                        "bytes": [104, 101, 108, 108, 111],
                        "top_logprobs": []
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 5,
                    "rejected_prediction_tokens": 6
                }
            }
        });

        let meta =
            extract_openai_compatible_provider_metadata("openai", &raw).expect("metadata present");
        let openai = meta.get("openai").expect("openai namespace");
        assert_eq!(openai["logprobs"][0]["token"], serde_json::json!("hello"));
        assert_eq!(
            openai.get("acceptedPredictionTokens"),
            Some(&serde_json::json!(5))
        );
        assert_eq!(
            openai.get("rejectedPredictionTokens"),
            Some(&serde_json::json!(6))
        );
    }

    #[test]
    fn openrouter_metadata_helper_extracts_sources_and_logprobs() {
        let raw = serde_json::json!({
            "id": "resp_1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello"
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [{
                        "token": "hello",
                        "logprob": -0.1,
                        "bytes": [104, 101, 108, 108, 111],
                        "top_logprobs": []
                    }]
                }
            }],
            "sources": [{
                "id": "src_1",
                "source_type": "url",
                "url": "https://example.com",
                "title": "Example"
            }]
        });

        let meta = extract_openai_compatible_provider_metadata("openrouter", &raw)
            .expect("metadata present");
        let openrouter = meta.get("openrouter").expect("openrouter namespace");
        assert_eq!(
            openrouter["sources"][0]["url"],
            serde_json::json!("https://example.com")
        );
        assert_eq!(
            openrouter["logprobs"][0]["token"],
            serde_json::json!("hello")
        );
    }
}
