use crate::types::{ProviderMetadataMap, ProviderOptionsMap};
use serde_json::{Value, json};

fn anthropic_container_id_from_provider_metadata(metadata: &ProviderMetadataMap) -> Option<String> {
    let provider = metadata
        .get("anthropic")
        .or_else(|| metadata.get("Anthropic"))?;

    let id = provider
        .get("container")
        .and_then(Value::as_object)
        .and_then(|container| container.get("id"))
        .and_then(Value::as_str)?;

    if id.trim().is_empty() {
        None
    } else {
        Some(id.to_string())
    }
}

/// Find the most recent Anthropic container id from prior step provider metadata.
///
/// This mirrors the package-level helper that `@ai-sdk/anthropic` exposes for `prepareStep`
/// workflows, but keeps the input generic over any history that can yield optional
/// `ProviderMetadataMap` references.
pub fn find_anthropic_container_id_from_last_step<'a, I>(steps: I) -> Option<String>
where
    I: IntoIterator<Item = Option<&'a ProviderMetadataMap>>,
    I::IntoIter: DoubleEndedIterator<Item = Option<&'a ProviderMetadataMap>>,
{
    steps
        .into_iter()
        .rev()
        .flatten()
        .find_map(anthropic_container_id_from_provider_metadata)
}

/// Build Anthropic `providerOptions` that forward the latest container id across steps.
///
/// This is the Rust package-surface counterpart to
/// `forwardAnthropicContainerIdFromLastStep(...)` from `@ai-sdk/anthropic`.
pub fn forward_anthropic_container_id_from_last_step<'a, I>(steps: I) -> Option<ProviderOptionsMap>
where
    I: IntoIterator<Item = Option<&'a ProviderMetadataMap>>,
    I::IntoIter: DoubleEndedIterator<Item = Option<&'a ProviderMetadataMap>>,
{
    let container_id = find_anthropic_container_id_from_last_step(steps)?;

    let mut provider_options = ProviderOptionsMap::new();
    provider_options.insert(
        "anthropic",
        json!({
            "container": { "id": container_id }
        }),
    );
    Some(provider_options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Map;

    fn provider_metadata_with_container(
        provider_id: &str,
        container_id: &str,
    ) -> ProviderMetadataMap {
        ProviderMetadataMap::from([(
            provider_id.to_string(),
            Value::Object(Map::from_iter([(
                "container".to_string(),
                json!({ "id": container_id }),
            )])),
        )])
    }

    #[test]
    fn find_anthropic_container_id_prefers_latest_non_empty_entry() {
        let older = provider_metadata_with_container("anthropic", "container_older");
        let empty = provider_metadata_with_container("anthropic", "");
        let newer = provider_metadata_with_container("Anthropic", "container_newer");

        assert_eq!(
            find_anthropic_container_id_from_last_step([
                Some(&older),
                Some(&empty),
                None,
                Some(&newer),
            ]),
            Some("container_newer".to_string())
        );
    }

    #[test]
    fn forward_anthropic_container_id_returns_provider_options_override() {
        let older = provider_metadata_with_container("anthropic", "container_older");
        let newer = provider_metadata_with_container("anthropic", "container_newer");
        let mut expected = ProviderOptionsMap::new();
        expected.insert(
            "anthropic",
            json!({ "container": { "id": "container_newer" } }),
        );

        assert_eq!(
            forward_anthropic_container_id_from_last_step([Some(&older), Some(&newer)]),
            Some(expected)
        );
    }
}
