//! Open provider options map (Vercel-aligned)
//!
//! This is a provider-id keyed JSON map used as a pass-through channel for provider-specific
//! configuration. It intentionally avoids a closed enum so that new providers and new provider
//! knobs do not require changes in `siumai-core`.
//!
//! The typed `ProviderOptions` enum remains as a compatibility layer during the fearless refactor.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Provider-id keyed JSON map.
///
/// - Key: provider id (e.g. `"openai"`, `"google"`, `"anthropic"`)
/// - Value: provider-specific JSON object (recommended), but any JSON value is accepted.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderOptionsMap(pub BTreeMap<String, serde_json::Value>);

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
            .insert(provider_id.as_ref().to_ascii_lowercase(), value);
    }

    /// Get provider options for `provider_id`.
    pub fn get(&self, provider_id: impl AsRef<str>) -> Option<&serde_json::Value> {
        self.0.get(&provider_id.as_ref().to_ascii_lowercase())
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

