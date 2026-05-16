//! `Bedrock`-specific response metadata.

use crate::types::ContentPart;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bedrock-specific metadata extracted from chat responses.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct BedrockMetadata {
    /// Whether the final response was synthesized from Bedrock's reserved `json` tool path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_json_response_from_tool: Option<bool>,

    /// Provider stop sequence metadata when Bedrock returns it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<serde_json::Value>,

    /// Preserve unknown provider-specific metadata for forward compatibility.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Bedrock-specific metadata attached to `ContentPart::Reasoning`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct BedrockReasoningContentPartMetadata {
    /// Reasoning signature returned by Bedrock for replay validation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,

    /// Redacted reasoning payload returned by Bedrock.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "redactedData",
        alias = "redacted_data"
    )]
    pub redacted_data: Option<String>,
}

impl crate::types::provider_metadata::FromMetadata for BedrockMetadata {
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self> {
        serde_json::from_value(serde_json::to_value(metadata).ok()?).ok()
    }
}

fn bedrock_reasoning_content_part_metadata(
    part: &ContentPart,
) -> Option<BedrockReasoningContentPartMetadata> {
    let ContentPart::Reasoning {
        provider_metadata: Some(provider_metadata),
        ..
    } = part
    else {
        return None;
    };

    let bedrock_metadata = provider_metadata.get("bedrock")?;
    serde_json::from_value(bedrock_metadata.clone()).ok()
}

/// Typed helper for Bedrock metadata extraction from `ChatResponse`.
pub trait BedrockChatResponseExt {
    fn bedrock_metadata(&self) -> Option<BedrockMetadata>;
}

impl BedrockChatResponseExt for crate::types::ChatResponse {
    fn bedrock_metadata(&self) -> Option<BedrockMetadata> {
        let meta = self
            .provider_metadata
            .as_ref()
            .and_then(|metadata| {
                crate::types::provider_metadata::provider_metadata_object(metadata, "bedrock")
            })?
            .clone();
        serde_json::from_value(serde_json::Value::Object(meta)).ok()
    }
}

/// Typed helper for Bedrock metadata extraction from `ContentPart::Reasoning`.
pub trait BedrockContentPartExt {
    fn bedrock_reasoning_metadata(&self) -> Option<BedrockReasoningContentPartMetadata>;
}

impl BedrockContentPartExt for ContentPart {
    fn bedrock_reasoning_metadata(&self) -> Option<BedrockReasoningContentPartMetadata> {
        bedrock_reasoning_content_part_metadata(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn production_source() -> &'static str {
        include_str!("bedrock.rs")
            .split_once("#[cfg(test)]")
            .expect("test marker should exist")
            .0
    }

    #[test]
    fn bedrock_provider_metadata_source_does_not_read_request_provider_options() {
        let source = production_source();

        assert!(
            !source.contains("providerOptions"),
            "Bedrock provider_metadata typed views must not read request-side providerOptions"
        );
        assert!(
            !source.contains("provider_options"),
            "Bedrock provider_metadata typed views must not read request-side provider_options"
        );
        assert!(
            !source.contains("provider_options_map"),
            "Bedrock provider_metadata typed views must not read request provider options maps"
        );
    }

    #[test]
    fn bedrock_metadata_parses_known_fields_and_preserves_extra() {
        let mut resp = crate::types::ChatResponse::new(crate::types::MessageContent::Text(
            "hello".to_string(),
        ));

        let mut inner = HashMap::new();
        inner.insert(
            "isJsonResponseFromTool".to_string(),
            serde_json::Value::Bool(true),
        );
        inner.insert("stopSequence".to_string(), serde_json::json!("END_OF_TURN"));
        inner.insert("vendor_extra".to_string(), serde_json::json!(42));

        let mut outer = HashMap::new();
        outer.insert(
            "bedrock".to_string(),
            serde_json::Value::Object(inner.into_iter().collect()),
        );
        resp.provider_metadata = Some(outer);

        let meta = resp.bedrock_metadata().expect("bedrock metadata");
        assert_eq!(meta.is_json_response_from_tool, Some(true));
        assert_eq!(meta.stop_sequence, Some(serde_json::json!("END_OF_TURN")));
        assert_eq!(meta.extra.get("vendor_extra"), Some(&serde_json::json!(42)));
    }

    #[test]
    fn bedrock_reasoning_metadata_parses_part_level_fields() {
        let part = crate::types::ContentPart::Reasoning {
            text: "internal".to_string(),
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: Some(HashMap::from([(
                "bedrock".to_string(),
                serde_json::json!({
                    "signature": "sig-1",
                    "redactedData": "blob"
                }),
            )])),
        };

        let meta = part
            .bedrock_reasoning_metadata()
            .expect("bedrock reasoning metadata");
        assert_eq!(meta.signature.as_deref(), Some("sig-1"));
        assert_eq!(meta.redacted_data.as_deref(), Some("blob"));
    }
}
