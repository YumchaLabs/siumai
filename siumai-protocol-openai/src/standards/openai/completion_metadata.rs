//! Shared completion response provider-metadata helpers for the OpenAI protocol family.

use chrono::{DateTime, TimeZone, Utc};
use reqwest::header::HeaderMap;
use serde_json::Value;
use std::collections::HashMap;

pub fn completion_request_id_from_headers(headers: &HeaderMap) -> Option<String> {
    for key in ["x-request-id", "request-id"] {
        if let Some(value) = headers.get(key)
            && let Ok(value) = value.to_str()
            && !value.trim().is_empty()
        {
            return Some(value.to_string());
        }
    }

    None
}

pub fn completion_created_at(raw: &Value) -> Option<DateTime<Utc>> {
    let created = raw
        .get("created")
        .and_then(|value| value.as_i64().or_else(|| value.as_u64().map(|v| v as i64)))?;
    Utc.timestamp_opt(created, 0).single()
}

pub fn completion_response_metadata(
    provider: impl Into<String>,
    raw: &Value,
    headers: &HeaderMap,
    include_body: bool,
) -> crate::types::ResponseMetadata {
    let request_id = completion_request_id_from_headers(headers);
    let headers = crate::execution::http::headers::headermap_to_hashmap(headers);

    crate::types::ResponseMetadata {
        id: raw
            .get("id")
            .and_then(|value| value.as_str())
            .map(ToString::to_string),
        model: raw
            .get("model")
            .and_then(|value| value.as_str())
            .map(ToString::to_string),
        created: completion_created_at(raw),
        provider: provider.into(),
        request_id,
        headers: (!headers.is_empty()).then_some(headers),
        body: include_body.then(|| raw.clone()),
    }
}

pub fn completion_stream_response_metadata(
    provider: impl Into<String>,
    id: Option<&str>,
    model: Option<&str>,
    created: Option<DateTime<Utc>>,
) -> crate::types::ResponseMetadata {
    crate::types::ResponseMetadata {
        id: id.map(ToString::to_string),
        model: model.map(ToString::to_string),
        created,
        provider: provider.into(),
        request_id: None,
        headers: None,
        body: None,
    }
}

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

    #[test]
    fn completion_response_metadata_reads_id_model_created_headers_and_body() {
        let raw = serde_json::json!({
            "id": "cmpl_1",
            "model": "gpt-3.5-turbo-instruct",
            "created": 1_718_345_013
        });
        let mut headers = HeaderMap::new();
        headers.insert("request-id", "req_1".parse().expect("request id header"));
        headers.insert("x-extra", "kept".parse().expect("extra header"));

        let metadata = completion_response_metadata("openai", &raw, &headers, true);

        assert_eq!(metadata.id.as_deref(), Some("cmpl_1"));
        assert_eq!(metadata.model.as_deref(), Some("gpt-3.5-turbo-instruct"));
        assert_eq!(metadata.provider, "openai");
        assert_eq!(metadata.request_id.as_deref(), Some("req_1"));
        assert_eq!(
            metadata
                .headers
                .as_ref()
                .and_then(|headers| headers.get("x-extra"))
                .map(String::as_str),
            Some("kept")
        );
        assert_eq!(metadata.body.as_ref(), Some(&raw));
        assert!(metadata.created.is_some());
    }

    #[test]
    fn completion_response_metadata_prefers_x_request_id_and_can_omit_body() {
        let raw = serde_json::json!({
            "created": 1_718_345_013
        });
        let mut headers = HeaderMap::new();
        headers.insert("request-id", "req_secondary".parse().expect("request id"));
        headers.insert("x-request-id", "req_primary".parse().expect("request id"));

        let metadata = completion_response_metadata("openrouter", &raw, &headers, false);

        assert_eq!(metadata.id, None);
        assert_eq!(metadata.model, None);
        assert_eq!(metadata.provider, "openrouter");
        assert_eq!(metadata.request_id.as_deref(), Some("req_primary"));
        assert_eq!(metadata.body, None);
    }

    #[test]
    fn completion_stream_response_metadata_builds_minimal_stream_envelope() {
        let created = completion_created_at(&serde_json::json!({ "created": 1_718_345_013 }));

        let metadata = completion_stream_response_metadata(
            "openai",
            Some("cmpl_stream"),
            Some("gpt-3.5-turbo-instruct"),
            created,
        );

        assert_eq!(metadata.id.as_deref(), Some("cmpl_stream"));
        assert_eq!(metadata.model.as_deref(), Some("gpt-3.5-turbo-instruct"));
        assert_eq!(metadata.provider, "openai");
        assert!(metadata.created.is_some());
        assert_eq!(metadata.request_id, None);
        assert_eq!(metadata.headers, None);
        assert_eq!(metadata.body, None);
    }
}
