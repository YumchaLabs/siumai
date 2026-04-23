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
    HashMap::from([(provider_id.into(), Value::Object(Map::from_iter(object)))])
}

/// Merge provider metadata maps.
///
/// When both sides contain object-shaped payloads for the same provider id, object keys are
/// merged shallowly with `source` winning on conflicts. Otherwise, the source entry replaces the
/// target entry.
pub fn merge_provider_metadata(target: &mut ProviderMetadataMap, source: ProviderMetadataMap) {
    for (provider_id, source_value) in source {
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
}
