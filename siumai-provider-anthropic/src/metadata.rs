use std::collections::BTreeMap;

use serde_json::Value;
use siumai_core::LanguageResponse;
use siumai_protocol_anthropic::messages::PROTOCOL_ID;
use thiserror::Error;

/// Service tier actually assigned to an Anthropic Messages response.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AnthropicAssignedServiceTier {
    Standard,
    Priority,
    Batch,
    Other(String),
}

/// Inference speed actually assigned to an Anthropic Messages response.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AnthropicAssignedSpeed {
    Standard,
    Fast,
    Other(String),
}

/// Inference geography reported by Anthropic for a Messages response.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AnthropicAssignedInferenceGeo {
    Global,
    Us,
    Other(String),
}

/// Anthropic-owned usage metadata that has no provider-neutral equivalent.
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicResponseUsage {
    assigned_service_tier: Option<AnthropicAssignedServiceTier>,
    speed: Option<AnthropicAssignedSpeed>,
    inference_geo: Option<AnthropicAssignedInferenceGeo>,
    raw: BTreeMap<String, Value>,
}

impl AnthropicResponseUsage {
    pub fn assigned_service_tier(&self) -> Option<&AnthropicAssignedServiceTier> {
        self.assigned_service_tier.as_ref()
    }

    pub fn speed(&self) -> Option<&AnthropicAssignedSpeed> {
        self.speed.as_ref()
    }

    pub fn inference_geo(&self) -> Option<&AnthropicAssignedInferenceGeo> {
        self.inference_geo.as_ref()
    }

    /// Inspect the bounded provider usage object retained by the protocol codec.
    pub fn raw(&self) -> &BTreeMap<String, Value> {
        &self.raw
    }
}

/// Typed view over Anthropic-specific response metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicResponseMetadata {
    usage: Option<AnthropicResponseUsage>,
    container: Option<Value>,
    context_management: Option<Value>,
    raw: BTreeMap<String, Value>,
}

impl AnthropicResponseMetadata {
    pub fn usage(&self) -> Option<&AnthropicResponseUsage> {
        self.usage.as_ref()
    }

    pub fn container(&self) -> Option<&Value> {
        self.container.as_ref()
    }

    pub fn context_management(&self) -> Option<&Value> {
        self.context_management.as_ref()
    }

    /// Inspect the bounded provider metadata retained by the protocol codec.
    pub fn raw(&self) -> &BTreeMap<String, Value> {
        &self.raw
    }
}

/// Typed Anthropic metadata access for a provider-neutral language response.
pub trait AnthropicLanguageResponseExt {
    fn anthropic_metadata(
        &self,
    ) -> Result<Option<AnthropicResponseMetadata>, AnthropicResponseMetadataError>;
}

impl AnthropicLanguageResponseExt for LanguageResponse {
    fn anthropic_metadata(
        &self,
    ) -> Result<Option<AnthropicResponseMetadata>, AnthropicResponseMetadataError> {
        let Some(value) = self.provider_metadata().get(PROTOCOL_ID) else {
            return Ok(None);
        };
        let raw = object(value, "provider_metadata.anthropic-messages")?;
        let usage = raw
            .get("usage")
            .map(|value| decode_usage(object(value, "provider_metadata.anthropic-messages.usage")?))
            .transpose()?;
        Ok(Some(AnthropicResponseMetadata {
            usage,
            container: raw.get("container").cloned(),
            context_management: raw.get("context_management").cloned(),
            raw,
        }))
    }
}

fn decode_usage(
    raw: BTreeMap<String, Value>,
) -> Result<AnthropicResponseUsage, AnthropicResponseMetadataError> {
    Ok(AnthropicResponseUsage {
        assigned_service_tier: decode_string_enum(
            &raw,
            "service_tier",
            "provider_metadata.anthropic-messages.usage.service_tier",
            |value| match value {
                "standard" => AnthropicAssignedServiceTier::Standard,
                "priority" => AnthropicAssignedServiceTier::Priority,
                "batch" => AnthropicAssignedServiceTier::Batch,
                other => AnthropicAssignedServiceTier::Other(other.to_string()),
            },
        )?,
        speed: decode_string_enum(
            &raw,
            "speed",
            "provider_metadata.anthropic-messages.usage.speed",
            |value| match value {
                "standard" => AnthropicAssignedSpeed::Standard,
                "fast" => AnthropicAssignedSpeed::Fast,
                other => AnthropicAssignedSpeed::Other(other.to_string()),
            },
        )?,
        inference_geo: decode_string_enum(
            &raw,
            "inference_geo",
            "provider_metadata.anthropic-messages.usage.inference_geo",
            |value| match value {
                "global" => AnthropicAssignedInferenceGeo::Global,
                "us" => AnthropicAssignedInferenceGeo::Us,
                other => AnthropicAssignedInferenceGeo::Other(other.to_string()),
            },
        )?,
        raw,
    })
}

fn decode_string_enum<T>(
    object: &BTreeMap<String, Value>,
    field: &'static str,
    path: &'static str,
    decode: impl FnOnce(&str) -> T,
) -> Result<Option<T>, AnthropicResponseMetadataError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(decode(value))),
        Some(_) => Err(AnthropicResponseMetadataError::InvalidField { path }),
    }
}

fn object(
    value: &Value,
    path: &'static str,
) -> Result<BTreeMap<String, Value>, AnthropicResponseMetadataError> {
    value
        .as_object()
        .map(|object| object.clone().into_iter().collect())
        .ok_or(AnthropicResponseMetadataError::InvalidField { path })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum AnthropicResponseMetadataError {
    #[error("Anthropic response metadata has an invalid field at {path}")]
    InvalidField { path: &'static str },
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;
    use siumai_core::{LanguageCompletionReason, LanguageResponse, Usage};

    use super::*;

    #[test]
    fn response_metadata_preserves_typed_and_future_usage_values() {
        let response = LanguageResponse::completed(
            Vec::new(),
            LanguageCompletionReason::Stop,
            Usage::default(),
        )
        .expect("response")
        .with_provider_metadata(BTreeMap::from([(
            PROTOCOL_ID.to_string(),
            json!({
                "usage": {
                    "service_tier": "priority",
                    "speed": "turbo",
                    "inference_geo": "global",
                    "future": 7
                },
                "container": { "id": "container_1" },
                "future": true
            }),
        )]));

        let metadata = response
            .anthropic_metadata()
            .expect("metadata")
            .expect("Anthropic metadata");
        let usage = metadata.usage().expect("usage");
        assert_eq!(
            usage.assigned_service_tier(),
            Some(&AnthropicAssignedServiceTier::Priority)
        );
        assert_eq!(
            usage.speed(),
            Some(&AnthropicAssignedSpeed::Other("turbo".to_string()))
        );
        assert_eq!(
            usage.inference_geo(),
            Some(&AnthropicAssignedInferenceGeo::Global)
        );
        assert_eq!(usage.raw().get("future"), Some(&json!(7)));
        assert_eq!(metadata.container(), Some(&json!({ "id": "container_1" })));
        assert_eq!(metadata.raw().get("future"), Some(&json!(true)));
    }
}
