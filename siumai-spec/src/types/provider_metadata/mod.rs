//! Provider metadata helpers.
//!
//! Provider-specific typed metadata types are intentionally owned by provider crates to
//! reduce coupling and compile cost in `siumai-core`.

use serde_json::{Map, Value};
use std::collections::HashMap;

// Provider-specific typed metadata types are intentionally owned by provider crates.

/// Provider-id keyed metadata map aligned with AI SDK `ProviderMetadata`.
///
/// Semantically this is `{ "provider_id": { ...provider fields... } }`.
/// We intentionally keep the inner payload as `serde_json::Value` for backward compatibility
/// while helper accessors expect object-shaped provider payloads.
///
/// This map is a public provider-scoped projection lane. It should contain fields that a provider
/// or protocol adapter intentionally exposes to application code, such as ids, cache metadata,
/// citations, safety ratings, or typed reasoning replay fields. It is not the right place to dump
/// raw HTTP bodies, response headers, whole SSE chunks, or provider debug events; keep those on
/// explicit raw/diagnostic carriers such as `ResponseMetadata::headers`, `ResponseMetadata::body`,
/// `ChatStreamPart::Raw`, or provider-specific diagnostics sinks.
pub type ProviderMetadataMap = HashMap<String, Value>;

/// Get the provider-scoped metadata object for `provider_id`.
pub fn provider_metadata_object<'a>(
    metadata: &'a ProviderMetadataMap,
    provider_id: &str,
) -> Option<&'a Map<String, Value>> {
    metadata.get(provider_id)?.as_object()
}

/// Get the first provider-scoped metadata object from a list of provider ids.
pub fn provider_metadata_object_any<'a>(
    metadata: &'a ProviderMetadataMap,
    provider_ids: &[&str],
) -> Option<&'a Map<String, Value>> {
    provider_ids
        .iter()
        .find_map(|provider_id| provider_metadata_object(metadata, provider_id))
}

/// Get one provider-scoped metadata value by provider id and key.
pub fn provider_metadata_value<'a>(
    metadata: &'a ProviderMetadataMap,
    provider_id: &str,
    key: &str,
) -> Option<&'a Value> {
    provider_metadata_object(metadata, provider_id)?.get(key)
}

/// Get one provider-scoped metadata value by trying multiple provider ids in order.
pub fn provider_metadata_value_any<'a>(
    metadata: &'a ProviderMetadataMap,
    provider_ids: &[&str],
    key: &str,
) -> Option<&'a Value> {
    provider_metadata_object_any(metadata, provider_ids)?.get(key)
}

/// Create a provider metadata map with one provider-scoped object entry.
pub fn provider_metadata_from_object(
    provider_id: impl Into<String>,
    object: impl IntoIterator<Item = (String, Value)>,
) -> ProviderMetadataMap {
    provider_metadata_public_projection(HashMap::from([(
        provider_id.into(),
        Value::Object(Map::from_iter(object)),
    )]))
}

/// Return whether a provider metadata key is reserved for private diagnostics.
///
/// Provider metadata is a public projection lane. Raw provider payloads, HTTP transport material,
/// and private/diagnostic debug fields belong in explicit diagnostics carriers instead.
pub fn provider_metadata_key_is_private_diagnostics(key: &str) -> bool {
    let normalized = key.trim().replace(['-', '.'], "_").to_ascii_lowercase();

    matches!(
        normalized.as_str(),
        "raw"
            | "rawitem"
            | "raw_item"
            | "rawvalue"
            | "raw_value"
            | "headers"
            | "body"
            | "request"
            | "response"
            | "httprequest"
            | "http_request"
            | "httpresponse"
            | "http_response"
            | "diagnostic"
            | "diagnostics"
            | "private"
    ) || normalized.starts_with("raw_")
        || normalized.starts_with("private_")
        || normalized.starts_with("diagnostic_")
        || normalized.starts_with("diagnostics_")
}

fn strip_private_diagnostics_from_value(value: &mut Value) {
    match value {
        Value::Object(obj) => {
            obj.retain(|key, _| !provider_metadata_key_is_private_diagnostics(key));
            for value in obj.values_mut() {
                strip_private_diagnostics_from_value(value);
            }
        }
        Value::Array(values) => {
            for value in values {
                strip_private_diagnostics_from_value(value);
            }
        }
        _ => {}
    }
}

/// Clone provider metadata with private/raw diagnostic fields removed.
pub fn provider_metadata_without_private_diagnostics(
    metadata: &ProviderMetadataMap,
) -> ProviderMetadataMap {
    provider_metadata_public_projection(metadata.clone())
}

/// Convert provider metadata into its public projection.
///
/// This function preserves provider namespaces and reviewed provider fields, but strips reserved
/// raw/private/diagnostic keys recursively from provider payloads.
pub fn provider_metadata_public_projection(metadata: ProviderMetadataMap) -> ProviderMetadataMap {
    metadata
        .into_iter()
        .filter_map(|(provider_id, mut value)| {
            strip_private_diagnostics_from_value(&mut value);
            match &value {
                Value::Object(obj) if obj.is_empty() => None,
                _ => Some((provider_id, value)),
            }
        })
        .collect()
}

/// Merge provider metadata maps.
///
/// When both sides contain object-shaped payloads for the same provider id, object keys are
/// merged shallowly with `source` winning on conflicts. Otherwise, the source entry replaces the
/// target entry.
pub fn merge_provider_metadata(target: &mut ProviderMetadataMap, source: ProviderMetadataMap) {
    *target = provider_metadata_public_projection(std::mem::take(target));

    for (provider_id, source_value) in provider_metadata_public_projection(source) {
        match (target.get_mut(&provider_id), source_value) {
            (Some(Value::Object(target_obj)), Value::Object(source_obj)) => {
                target_obj.extend(source_obj);
            }
            (_, value) => {
                target.insert(provider_id, value);
            }
        }
    }
}

/// Helper trait for converting HashMap metadata to typed structures
pub trait FromMetadata: Sized {
    /// Try to parse metadata from a HashMap
    fn from_metadata(metadata: &HashMap<String, Value>) -> Option<Self>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_metadata_object_reads_object_payloads() {
        let metadata = provider_metadata_from_object(
            "openai",
            Map::from_iter([("itemId".to_string(), Value::String("msg_1".to_string()))]),
        );

        assert_eq!(
            provider_metadata_value(&metadata, "openai", "itemId"),
            Some(&Value::String("msg_1".to_string()))
        );
    }

    #[test]
    fn merge_provider_metadata_shallow_merges_provider_objects() {
        let mut target = provider_metadata_from_object(
            "openai",
            Map::from_iter([("itemId".to_string(), Value::String("msg_1".to_string()))]),
        );
        let source = provider_metadata_from_object(
            "openai",
            Map::from_iter([("phase".to_string(), Value::String("done".to_string()))]),
        );

        merge_provider_metadata(&mut target, source);

        let openai = provider_metadata_object(&target, "openai").expect("openai metadata");
        assert_eq!(
            openai.get("itemId"),
            Some(&Value::String("msg_1".to_string()))
        );
        assert_eq!(
            openai.get("phase"),
            Some(&Value::String("done".to_string()))
        );
    }

    #[test]
    fn provider_metadata_object_any_reads_first_matching_alias() {
        let metadata = ProviderMetadataMap::from([
            (
                "google-vertex".to_string(),
                Value::Object(Map::from_iter([(
                    "thoughtSignature".to_string(),
                    Value::String("sig".to_string()),
                )])),
            ),
            (
                "vertex".to_string(),
                Value::Object(Map::from_iter([(
                    "thoughtSignature".to_string(),
                    Value::String("preferred".to_string()),
                )])),
            ),
        ]);

        let object = provider_metadata_object_any(&metadata, &["vertex", "google-vertex"])
            .expect("provider metadata alias object");
        assert_eq!(
            object.get("thoughtSignature"),
            Some(&Value::String("preferred".to_string()))
        );
        assert_eq!(
            provider_metadata_value_any(
                &metadata,
                &["missing", "google-vertex"],
                "thoughtSignature"
            ),
            Some(&Value::String("sig".to_string()))
        );
    }

    #[test]
    fn provider_metadata_public_projection_strips_private_fields_recursively() {
        let metadata = provider_metadata_public_projection(ProviderMetadataMap::from([(
            "openai".to_string(),
            serde_json::json!({
                "itemId": "item_1",
                "rawItem": { "private": true },
                "headers": { "authorization": "secret" },
                "nested": {
                    "body": { "secret": true },
                    "kept": 1
                },
                "items": [
                    { "raw_value": "secret", "kept": true }
                ]
            }),
        )]));

        let openai_value = metadata.get("openai").expect("openai metadata");
        let openai = openai_value.as_object().expect("openai metadata");
        assert_eq!(openai.get("itemId"), Some(&serde_json::json!("item_1")));
        assert!(openai.get("rawItem").is_none());
        assert!(openai.get("headers").is_none());
        assert_eq!(
            openai_value.pointer("/nested/kept"),
            Some(&serde_json::json!(1))
        );
        assert!(openai_value.pointer("/nested/body").is_none());
        assert_eq!(
            openai_value.pointer("/items/0/kept"),
            Some(&serde_json::json!(true))
        );
        assert!(openai_value.pointer("/items/0/raw_value").is_none());
    }

    #[test]
    fn merge_provider_metadata_projects_target_and_source() {
        let mut target = ProviderMetadataMap::from([(
            "openai".to_string(),
            serde_json::json!({
                "itemId": "item_1",
                "raw_item": { "secret": true }
            }),
        )]);
        let source = ProviderMetadataMap::from([(
            "openai".to_string(),
            serde_json::json!({
                "phase": "done",
                "diagnostics": { "secret": true }
            }),
        )]);

        merge_provider_metadata(&mut target, source);

        let openai = target
            .get("openai")
            .and_then(|value| value.as_object())
            .expect("openai metadata");
        assert_eq!(openai.get("itemId"), Some(&serde_json::json!("item_1")));
        assert_eq!(openai.get("phase"), Some(&serde_json::json!("done")));
        assert!(openai.get("raw_item").is_none());
        assert!(openai.get("diagnostics").is_none());
    }
}
