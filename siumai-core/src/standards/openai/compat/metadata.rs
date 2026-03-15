//! Shared provider metadata extraction helpers for OpenAI-compatible providers.

use std::collections::HashMap;

pub(super) type NestedProviderMetadata = HashMap<String, HashMap<String, serde_json::Value>>;

fn extract_sources_and_logprobs(
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

    if meta.is_empty() {
        None
    } else {
        Some(HashMap::from([(provider_id.to_string(), meta)]))
    }
}

pub(super) fn extract_provider_metadata(
    provider_id: &str,
    raw: &serde_json::Value,
) -> Option<NestedProviderMetadata> {
    match provider_id {
        // Perplexity extends the OpenAI-like response schema with extra fields such as
        // `search_results` and `videos` (see Perplexity OpenAPI spec). These are intentionally
        // exposed as provider metadata instead of being added to the unified surface.
        "perplexity" => {
            let mut meta = HashMap::<String, serde_json::Value>::new();
            for key in ["search_results", "videos", "citations", "images"] {
                if let Some(value) = raw.get(key).filter(|value| !value.is_null()) {
                    meta.insert(key.to_string(), value.clone());
                }
            }
            if let Some(usage) = raw.get("usage").filter(|value| !value.is_null()) {
                meta.insert("usage".to_string(), usage.clone());
            }
            if meta.is_empty() {
                None
            } else {
                Some(HashMap::from([(provider_id.to_string(), meta)]))
            }
        }
        "openai" | "openrouter" | "xai" | "groq" | "deepseek" => {
            extract_sources_and_logprobs(provider_id, raw)
        }
        _ => None,
    }
}

pub(super) fn merge_nested_provider_metadata(
    target: &mut NestedProviderMetadata,
    source: NestedProviderMetadata,
) {
    for (provider, metadata) in source {
        target.entry(provider).or_default().extend(metadata);
    }
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
                "num_search_queries": 1
            },
            "citations": ["https://example.com"],
            "images": [{
                "image_url": "https://images.example.com/rust.png",
                "origin_url": "https://example.com",
                "height": 900,
                "width": 1600
            }]
        });

        let meta = extract_provider_metadata("perplexity", &raw).expect("metadata present");
        let perplexity = meta.get("perplexity").expect("perplexity namespace");
        assert_eq!(
            perplexity.get("citations"),
            Some(&serde_json::json!(["https://example.com"]))
        );
        assert_eq!(perplexity["usage"]["citation_tokens"], serde_json::json!(4));
        assert_eq!(
            perplexity["images"][0]["image_url"],
            serde_json::json!("https://images.example.com/rust.png")
        );
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

        let meta = extract_provider_metadata("xai", &raw).expect("metadata present");
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
            }]
        });

        let meta = extract_provider_metadata("openai", &raw).expect("metadata present");
        let openai = meta.get("openai").expect("openai namespace");
        assert_eq!(openai["logprobs"][0]["token"], serde_json::json!("hello"));
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

        let meta = extract_provider_metadata("openrouter", &raw).expect("metadata present");
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
