//! Shared completion response provider-metadata helpers for the OpenAI protocol family.

use serde_json::Value;
use std::collections::HashMap;

/// Extract provider-scoped metadata from an OpenAI-style `/completions` response chunk.
///
/// Completion logprobs are intentionally preserved as the raw `choices[0].logprobs` object. This
/// differs from Chat Completions metadata, where the AI SDK-compatible shape extracts
/// `choices[0].logprobs.content`.
pub fn extract_completion_provider_metadata(
    provider_id: &str,
    raw: &Value,
) -> Option<crate::types::ProviderMetadataMap> {
    let mut metadata = HashMap::new();

    if let Some(sources) = raw
        .get("sources")
        .filter(|value| !value.is_null())
        .filter(|value| value.as_array().is_some_and(|arr| !arr.is_empty()))
    {
        metadata.insert("sources".to_string(), sources.clone());
    }

    if let Some(logprobs) = raw
        .get("choices")
        .and_then(|value| value.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("logprobs"))
        .filter(|value| !value.is_null())
    {
        metadata.insert("logprobs".to_string(), logprobs.clone());
    }

    if metadata.is_empty() {
        None
    } else {
        Some(crate::types::provider_metadata::provider_metadata_from_object(provider_id, metadata))
    }
}

pub fn merge_completion_provider_metadata(
    target: &mut Option<crate::types::ProviderMetadataMap>,
    source: Option<crate::types::ProviderMetadataMap>,
) {
    let Some(source) = source else {
        return;
    };

    let target = target.get_or_insert_with(HashMap::new);
    crate::types::provider_metadata::merge_provider_metadata(target, source);
}

pub fn flatten_completion_stream_provider_metadata(
    nested: &Option<crate::types::ProviderMetadataMap>,
) -> Option<crate::types::ProviderMetadataMap> {
    nested.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_metadata_extracts_raw_logprobs_and_sources() {
        let raw = serde_json::json!({
            "choices": [{
                "text": " world",
                "logprobs": {
                    "tokens": ["world"],
                    "token_logprobs": [-0.2],
                    "top_logprobs": [{ "world": -0.2 }]
                }
            }],
            "sources": [{ "url": "https://example.com" }]
        });

        let metadata =
            extract_completion_provider_metadata("testProvider", &raw).expect("metadata");
        let provider = metadata
            .get("testProvider")
            .and_then(|value| value.as_object())
            .expect("provider metadata object");

        assert_eq!(
            provider.get("logprobs"),
            Some(&serde_json::json!({
                "tokens": ["world"],
                "token_logprobs": [-0.2],
                "top_logprobs": [{ "world": -0.2 }]
            }))
        );
        assert_eq!(
            provider.get("sources"),
            Some(&serde_json::json!([{ "url": "https://example.com" }]))
        );
    }

    #[test]
    fn completion_metadata_elides_empty_or_null_fields() {
        let raw = serde_json::json!({
            "choices": [{ "text": " world", "logprobs": null }],
            "sources": []
        });

        assert!(extract_completion_provider_metadata("openai", &raw).is_none());
    }

    #[test]
    fn completion_metadata_merge_shallow_merges_provider_objects() {
        let mut target = extract_completion_provider_metadata(
            "openai",
            &serde_json::json!({
                "choices": [{
                    "logprobs": {
                        "tokens": ["a"],
                        "token_logprobs": [-0.1]
                    }
                }]
            }),
        );
        let source = extract_completion_provider_metadata(
            "openai",
            &serde_json::json!({
                "sources": [{ "url": "https://example.com" }]
            }),
        );

        merge_completion_provider_metadata(&mut target, source);

        let provider = target
            .as_ref()
            .and_then(|metadata| metadata.get("openai"))
            .and_then(|value| value.as_object())
            .expect("merged provider metadata");

        assert!(provider.get("logprobs").is_some());
        assert_eq!(
            provider.get("sources"),
            Some(&serde_json::json!([{ "url": "https://example.com" }]))
        );
    }
}
