//! Open provider options map (Vercel-aligned)
//!
//! This is a provider-id keyed JSON map used as a pass-through channel for provider-specific
//! configuration. It intentionally avoids a closed enum so that new providers and new provider
//! knobs do not require changes in `siumai-core`.
//!
//! Provider-specific typed options are owned by provider crates and should be serialized into
//! this map under their provider id.

use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;

/// Provider-id keyed JSON map.
///
/// - Key: provider id (e.g. `"openai"`, `"google"`, `"anthropic"`)
/// - Value: provider-specific JSON object (recommended), but any JSON value is accepted.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderOptionsMap(pub BTreeMap<String, serde_json::Value>);

fn normalize_provider_id(provider_id: &str) -> String {
    provider_id.to_ascii_lowercase()
}

fn serialize_provider_id(provider_id: &str) -> &str {
    if provider_id.eq_ignore_ascii_case("openaicompatible") {
        "openaiCompatible"
    } else {
        provider_id
    }
}

impl Serialize for ProviderOptionsMap {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (provider_id, value) in &self.0 {
            map.serialize_entry(serialize_provider_id(provider_id), value)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for ProviderOptionsMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = BTreeMap::<String, serde_json::Value>::deserialize(deserializer)?;
        Ok(Self(
            raw.into_iter()
                .map(|(provider_id, value)| (normalize_provider_id(&provider_id), value))
                .collect(),
        ))
    }
}

impl ProviderOptionsMap {
    /// Create an empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true when the map has no entries.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Insert provider options under `provider_id`.
    ///
    /// Provider ids are normalized to lowercase for consistency.
    pub fn insert(&mut self, provider_id: impl AsRef<str>, value: serde_json::Value) {
        self.0
            .insert(normalize_provider_id(provider_id.as_ref()), value);
    }

    /// Get provider options for `provider_id`.
    pub fn get(&self, provider_id: impl AsRef<str>) -> Option<&serde_json::Value> {
        self.0.get(&normalize_provider_id(provider_id.as_ref()))
    }

    /// Get provider options for `provider_id` as a JSON object.
    pub fn get_object(
        &self,
        provider_id: impl AsRef<str>,
    ) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.get(provider_id)?.as_object()
    }

    /// Merge `overrides` into `self` (request overrides builder defaults).
    ///
    /// Rules:
    /// - If both sides are JSON objects, merge keys recursively (override wins).
    /// - Otherwise, the override replaces the base value.
    pub fn merge_overrides(&mut self, overrides: ProviderOptionsMap) {
        for (provider_id, override_value) in overrides.0 {
            match (self.0.get_mut(&provider_id), override_value) {
                (Some(base_value), serde_json::Value::Object(override_obj))
                    if base_value.is_object() =>
                {
                    merge_json_objects(base_value, override_obj);
                }
                (_, v) => {
                    self.0.insert(provider_id, v);
                }
            }
        }
    }
}

fn merge_json_objects(
    base: &mut serde_json::Value,
    override_obj: serde_json::Map<String, serde_json::Value>,
) {
    let Some(base_obj) = base.as_object_mut() else {
        *base = serde_json::Value::Object(override_obj);
        return;
    };

    for (k, override_value) in override_obj {
        match (base_obj.get_mut(&k), override_value) {
            (Some(serde_json::Value::Object(_)), serde_json::Value::Object(nested_override)) => {
                let base_nested = base_obj.get_mut(&k).expect("key just checked");
                merge_json_objects(base_nested, nested_override);
            }
            (_, v) => {
                base_obj.insert(k, v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ProviderOptionsMap;

    #[test]
    fn deserialize_normalizes_provider_ids() {
        let map: ProviderOptionsMap = serde_json::from_value(serde_json::json!({
            "openaiCompatible": { "cacheControl": { "type": "ephemeral" } },
            "OpenAI": { "store": false }
        }))
        .expect("deserialize provider options");

        assert!(map.get("openaiCompatible").is_some());
        assert!(map.get("openaicompatible").is_some());
        assert!(map.get("openai").is_some());
        assert!(map.0.contains_key("openaicompatible"));
        assert!(map.0.contains_key("openai"));
    }

    #[test]
    fn serialize_restores_openai_compatible_wire_key() {
        let mut map = ProviderOptionsMap::default();
        map.insert(
            "openaiCompatible",
            serde_json::json!({
                "cacheControl": { "type": "ephemeral" }
            }),
        );

        let value = serde_json::to_value(&map).expect("serialize provider options");
        assert!(value.get("openaiCompatible").is_some());
        assert!(value.get("openaicompatible").is_none());
    }
}
