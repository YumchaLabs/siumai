//! `Bedrock`-specific response metadata.

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

impl crate::types::provider_metadata::FromMetadata for BedrockMetadata {
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self> {
        serde_json::from_value(serde_json::to_value(metadata).ok()?).ok()
    }
}

/// Typed helper for Bedrock metadata extraction from `ChatResponse`.
pub trait BedrockChatResponseExt {
    fn bedrock_metadata(&self) -> Option<BedrockMetadata>;
}

impl BedrockChatResponseExt for crate::types::ChatResponse {
    fn bedrock_metadata(&self) -> Option<BedrockMetadata> {
        use crate::types::provider_metadata::FromMetadata;
        let meta = self.provider_metadata.as_ref()?.get("bedrock")?;
        BedrockMetadata::from_metadata(meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        outer.insert("bedrock".to_string(), inner);
        resp.provider_metadata = Some(outer);

        let meta = resp.bedrock_metadata().expect("bedrock metadata");
        assert_eq!(meta.is_json_response_from_tool, Some(true));
        assert_eq!(meta.stop_sequence, Some(serde_json::json!("END_OF_TURN")));
        assert_eq!(meta.extra.get("vendor_extra"), Some(&serde_json::json!(42)));
    }
}
