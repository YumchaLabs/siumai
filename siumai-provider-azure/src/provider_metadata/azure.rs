//! Azure-specific response metadata.

/// Azure-specific content-part metadata currently reuses the OpenAI-compatible stable shape.
pub use siumai_protocol_openai::provider_metadata::openai::OpenAiContentPartMetadata as AzureContentPartMetadata;
/// Azure-specific typed chat metadata currently reuses the OpenAI-compatible stable shape.
pub use siumai_protocol_openai::provider_metadata::openai::OpenAiMetadata as AzureMetadata;
/// A normalized source entry reused from the OpenAI-compatible family.
pub use siumai_protocol_openai::provider_metadata::openai::OpenAiSource as AzureSource;
/// Azure-specific typed source metadata currently reuses the OpenAI-compatible stable shape.
pub use siumai_protocol_openai::provider_metadata::openai::OpenAiSourceMetadata as AzureSourceMetadata;

/// Typed helper for Azure metadata extraction from `ChatResponse`.
pub trait AzureChatResponseExt {
    /// Read Azure metadata from the default `provider_metadata["azure"]` root.
    fn azure_metadata(&self) -> Option<AzureMetadata>;

    /// Read Azure metadata from a custom provider metadata key.
    fn azure_metadata_with_key(&self, key: &str) -> Option<AzureMetadata>;
}

impl AzureChatResponseExt for crate::types::ChatResponse {
    fn azure_metadata(&self) -> Option<AzureMetadata> {
        self.azure_metadata_with_key("azure")
    }

    fn azure_metadata_with_key(&self, key: &str) -> Option<AzureMetadata> {
        use crate::types::provider_metadata::FromMetadata;
        let meta = self.provider_metadata.as_ref()?.get(key)?;
        AzureMetadata::from_metadata(meta)
    }
}

fn source_metadata_with_key(source: &AzureSource, key: &str) -> Option<AzureSourceMetadata> {
    let metadata = source.provider_metadata.clone()?;
    let inner = metadata.get(key).cloned().unwrap_or(metadata);
    serde_json::from_value(inner).ok()
}

/// Typed helper for Azure metadata extraction from `AzureSource`.
pub trait AzureSourceExt {
    /// Read Azure source metadata from the default `azure` envelope or direct source metadata.
    fn azure_metadata(&self) -> Option<AzureSourceMetadata>;

    /// Read Azure source metadata from a custom provider metadata key.
    fn azure_metadata_with_key(&self, key: &str) -> Option<AzureSourceMetadata>;
}

impl AzureSourceExt for AzureSource {
    fn azure_metadata(&self) -> Option<AzureSourceMetadata> {
        self.azure_metadata_with_key("azure")
    }

    fn azure_metadata_with_key(&self, key: &str) -> Option<AzureSourceMetadata> {
        source_metadata_with_key(self, key)
    }
}

fn content_part_metadata<'a>(
    part: &'a crate::types::ContentPart,
) -> Option<&'a std::collections::HashMap<String, serde_json::Value>> {
    use crate::types::ContentPart;

    match part {
        ContentPart::Text {
            provider_metadata, ..
        }
        | ContentPart::Image {
            provider_metadata, ..
        }
        | ContentPart::Audio {
            provider_metadata, ..
        }
        | ContentPart::File {
            provider_metadata, ..
        }
        | ContentPart::ToolCall {
            provider_metadata, ..
        }
        | ContentPart::ToolResult {
            provider_metadata, ..
        }
        | ContentPart::Reasoning {
            provider_metadata, ..
        } => provider_metadata.as_ref(),
        ContentPart::Source { .. }
        | ContentPart::ToolApprovalResponse { .. }
        | ContentPart::ToolApprovalRequest { .. } => None,
    }
}

/// Typed helper for Azure metadata extraction from `ContentPart`.
pub trait AzureContentPartExt {
    /// Read Azure content-part metadata from the default `provider_metadata["azure"]` root.
    fn azure_metadata(&self) -> Option<AzureContentPartMetadata>;

    /// Read Azure content-part metadata from a custom provider metadata key.
    fn azure_metadata_with_key(&self, key: &str) -> Option<AzureContentPartMetadata>;
}

impl AzureContentPartExt for crate::types::ContentPart {
    fn azure_metadata(&self) -> Option<AzureContentPartMetadata> {
        self.azure_metadata_with_key("azure")
    }

    fn azure_metadata_with_key(&self, key: &str) -> Option<AzureContentPartMetadata> {
        let provider_metadata = content_part_metadata(self)?;
        let meta = provider_metadata.get(key)?;
        serde_json::from_value(meta.clone()).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn azure_chat_metadata_reads_default_and_custom_keys() {
        let mut response =
            crate::types::ChatResponse::new(crate::types::MessageContent::Text("ok".to_string()));

        let mut default_inner = HashMap::new();
        default_inner.insert("service_tier".to_string(), serde_json::json!("default"));

        let mut custom_inner = HashMap::new();
        custom_inner.insert("service_tier".to_string(), serde_json::json!("priority"));

        let mut outer = HashMap::new();
        outer.insert("azure".to_string(), default_inner);
        outer.insert("openai".to_string(), custom_inner);
        response.provider_metadata = Some(outer);

        assert_eq!(
            response.azure_metadata().and_then(|meta| meta.service_tier),
            Some("default".to_string())
        );
        assert_eq!(
            response
                .azure_metadata_with_key("openai")
                .and_then(|meta| meta.service_tier),
            Some("priority".to_string())
        );
    }

    #[test]
    fn azure_source_metadata_reads_keyed_and_direct_shapes() {
        let keyed = AzureSource {
            id: "src_1".to_string(),
            source_type: "document".to_string(),
            url: "file_123".to_string(),
            title: None,
            tool_call_id: None,
            media_type: None,
            filename: None,
            provider_metadata: Some(serde_json::json!({
                "azure": {
                    "fileId": "file_123",
                    "containerId": "container_456",
                    "index": 2
                }
            })),
            snippet: None,
        };

        let direct = AzureSource {
            id: "src_2".to_string(),
            source_type: "document".to_string(),
            url: "file_789".to_string(),
            title: None,
            tool_call_id: None,
            media_type: None,
            filename: None,
            provider_metadata: Some(serde_json::json!({
                "fileId": "file_789",
                "index": 4
            })),
            snippet: None,
        };

        assert_eq!(
            keyed.azure_metadata().and_then(|meta| meta.file_id),
            Some("file_123".to_string())
        );
        assert_eq!(
            direct.azure_metadata().and_then(|meta| meta.file_id),
            Some("file_789".to_string())
        );
    }

    #[test]
    fn azure_content_part_metadata_reads_custom_key() {
        let part = crate::types::ContentPart::Reasoning {
            text: "thinking".to_string(),
            provider_metadata: Some(HashMap::from([(
                "openai".to_string(),
                serde_json::json!({
                    "itemId": "rs_1",
                    "reasoningEncryptedContent": "enc_123"
                }),
            )])),
        };

        assert!(part.azure_metadata().is_none());
        let meta = part
            .azure_metadata_with_key("openai")
            .expect("azure content-part metadata");
        assert_eq!(meta.item_id.as_deref(), Some("rs_1"));
        assert_eq!(meta.reasoning_encrypted_content.as_deref(), Some("enc_123"));
    }
}
