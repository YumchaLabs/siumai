use crate::provider_metadata::bedrock::BedrockContentPartExt;
use crate::provider_options::{
    BedrockCachePoint, BedrockChatOptions, BedrockEmbeddingOptions, BedrockFilePartProviderOptions,
    BedrockRerankOptions,
};
use crate::types::{
    ChatMessage, ChatMessageBuilder, ChatResponse, ContentPart, MessageContent, ProviderOptionsMap,
};

/// Typed chat request helpers for Amazon Bedrock.
pub trait BedrockChatRequestExt {
    /// Store typed options under `provider_options_map["bedrock"]`.
    fn with_bedrock_chat_options(self, options: BedrockChatOptions) -> Self;
}

impl BedrockChatRequestExt for crate::types::ChatRequest {
    fn with_bedrock_chat_options(mut self, options: BedrockChatOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize BedrockChatOptions");
        merge_bedrock_provider_option_map(&mut self.provider_options_map, value);
        self
    }
}

/// Typed rerank request helpers for Amazon Bedrock.
pub trait BedrockRerankRequestExt {
    /// Store typed options under `provider_options_map["bedrock"]`.
    fn with_bedrock_rerank_options(self, options: BedrockRerankOptions) -> Self;
}

impl BedrockRerankRequestExt for crate::types::RerankRequest {
    fn with_bedrock_rerank_options(mut self, options: BedrockRerankOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize BedrockRerankOptions");
        merge_bedrock_provider_option_map(&mut self.provider_options_map, value);
        self
    }
}

/// Typed embedding request helpers for Amazon Bedrock.
pub trait BedrockEmbeddingRequestExt {
    /// Store typed options under `provider_options_map["bedrock"]`.
    fn with_bedrock_embedding_options(self, options: BedrockEmbeddingOptions) -> Self;
}

impl BedrockEmbeddingRequestExt for crate::types::EmbeddingRequest {
    fn with_bedrock_embedding_options(mut self, options: BedrockEmbeddingOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize BedrockEmbeddingOptions");
        merge_bedrock_provider_option_map(&mut self.provider_options_map, value);
        self
    }
}

fn merge_bedrock_provider_option_map(
    provider_options: &mut ProviderOptionsMap,
    value: serde_json::Value,
) {
    match (provider_options.0.get_mut("bedrock"), value) {
        (Some(existing), serde_json::Value::Object(incoming)) if existing.is_object() => {
            if let Some(existing_obj) = existing.as_object_mut() {
                existing_obj.extend(incoming);
            }
        }
        (_, replacement) => {
            provider_options.insert("bedrock", replacement);
        }
    }
}

fn merge_bedrock_reasoning_options(
    part: &mut ContentPart,
    key: &str,
    value: serde_json::Value,
) -> bool {
    let Some(provider_options) = part.provider_options_mut() else {
        return false;
    };

    match provider_options.0.get_mut("bedrock") {
        Some(existing) if existing.is_object() => {
            if let Some(obj) = existing.as_object_mut() {
                obj.insert(key.to_string(), value);
            }
        }
        _ => {
            provider_options.insert("bedrock", serde_json::json!({ key: value }));
        }
    }

    true
}

/// Message-level Bedrock request helpers.
pub trait BedrockMessageExt {
    /// Attach a typed Bedrock prompt cache point under `providerOptions.bedrock.cachePoint`.
    fn with_bedrock_cache_point(self, cache_point: BedrockCachePoint) -> Self;
}

impl BedrockMessageExt for ChatMessageBuilder {
    fn with_bedrock_cache_point(self, cache_point: BedrockCachePoint) -> Self {
        let value = serde_json::to_value(cache_point).expect("serialize BedrockCachePoint");
        self.with_provider_option("bedrock", value)
    }
}

impl BedrockMessageExt for ChatMessage {
    fn with_bedrock_cache_point(mut self, cache_point: BedrockCachePoint) -> Self {
        let value = serde_json::to_value(cache_point).expect("serialize BedrockCachePoint");
        merge_bedrock_provider_option_map(&mut self.provider_options, value);
        self
    }
}

/// Request-side Bedrock content-part helpers.
pub trait BedrockRequestContentPartExt {
    /// Attach typed Bedrock file-part options under `ContentPart::File.providerOptions["bedrock"]`.
    fn with_bedrock_file_part_options(self, options: BedrockFilePartProviderOptions) -> Self;

    /// Convenience helper for Bedrock document citations on `ContentPart::File`.
    fn with_bedrock_document_citations(self, enabled: bool) -> Self;
}

impl BedrockRequestContentPartExt for ContentPart {
    fn with_bedrock_file_part_options(mut self, options: BedrockFilePartProviderOptions) -> Self {
        let value =
            serde_json::to_value(options).expect("serialize BedrockFilePartProviderOptions");
        if let Some(provider_options) = self.provider_options_mut() {
            merge_bedrock_provider_option_map(provider_options, value);
        }
        self
    }

    fn with_bedrock_document_citations(self, enabled: bool) -> Self {
        self.with_bedrock_file_part_options(
            BedrockFilePartProviderOptions::new().with_citations(enabled),
        )
    }
}

/// Convert a Bedrock response into a replayable assistant message with reasoning metadata.
///
/// Bedrock reasoning replay requires the request-side reasoning parts to carry
/// `providerOptions.bedrock.signature` or `providerOptions.bedrock.redactedData`.
/// This helper copies those fields from response-side `providerMetadata.bedrock` on
/// `ContentPart::Reasoning` into the corresponding request-side provider options.
pub fn assistant_message_with_reasoning_metadata(response: &ChatResponse) -> ChatMessage {
    let mut msg = response.to_assistant_message();

    if let MessageContent::MultiModal(parts) = &mut msg.content {
        for part in parts.iter_mut() {
            let Some(reasoning_meta) = part.bedrock_reasoning_metadata() else {
                continue;
            };

            if let Some(signature) = reasoning_meta.signature {
                let _ = merge_bedrock_reasoning_options(
                    part,
                    "signature",
                    serde_json::json!(signature),
                );
            }

            if let Some(redacted_data) = reasoning_meta.redacted_data {
                let _ = merge_bedrock_reasoning_options(
                    part,
                    "redactedData",
                    serde_json::json!(redacted_data),
                );
            }
        }
    }

    msg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::{
        BedrockCacheTtl, BedrockEmbeddingInputType, BedrockFilePartCitations,
    };

    #[test]
    fn assistant_message_with_reasoning_metadata_carries_part_fields_to_provider_options() {
        let response = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::text("visible"),
            ContentPart::Reasoning {
                text: "internal".to_string(),
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(std::collections::HashMap::from([(
                    "bedrock".to_string(),
                    serde_json::json!({
                        "signature": "sig-1",
                        "redactedData": "blob"
                    }),
                )])),
            },
        ]));

        let message = assistant_message_with_reasoning_metadata(&response);
        let MessageContent::MultiModal(parts) = message.content else {
            panic!("expected multimodal content");
        };

        let reasoning = parts
            .iter()
            .find(|part| matches!(part, ContentPart::Reasoning { .. }))
            .expect("reasoning part");
        let bedrock_options = reasoning
            .provider_options()
            .and_then(|provider_options| provider_options.get_object("bedrock"))
            .expect("bedrock provider options");

        assert_eq!(
            bedrock_options.get("signature"),
            Some(&serde_json::json!("sig-1"))
        );
        assert_eq!(
            bedrock_options.get("redactedData"),
            Some(&serde_json::json!("blob"))
        );
    }

    #[test]
    fn message_and_file_part_helpers_store_typed_bedrock_request_options() {
        let message = crate::types::ChatMessage::user("hello")
            .with_bedrock_cache_point(BedrockCachePoint::new().with_ttl(BedrockCacheTtl::OneHour))
            .build();

        assert_eq!(
            message.provider_options.get("bedrock"),
            Some(&serde_json::json!({
                "cachePoint": {
                    "type": "default",
                    "ttl": "1h"
                }
            }))
        );

        let part = ContentPart::file_base64(
            "AAECAw==",
            "application/pdf",
            Some("report.pdf".to_string()),
        )
        .with_bedrock_document_citations(true);
        let bedrock_options = part
            .provider_options()
            .and_then(|provider_options| provider_options.get_object("bedrock"))
            .expect("bedrock provider options");

        assert_eq!(
            serde_json::from_value::<BedrockFilePartProviderOptions>(serde_json::Value::Object(
                bedrock_options.clone()
            ))
            .expect("typed file part options")
            .citations,
            Some(BedrockFilePartCitations { enabled: true })
        );
    }

    #[test]
    fn request_helpers_merge_existing_bedrock_provider_options() {
        let chat_request =
            crate::types::ChatRequest::new(vec![crate::types::ChatMessage::user("hello").build()])
                .with_provider_option("bedrock", serde_json::json!({ "raw": true }))
                .with_bedrock_chat_options(
                    BedrockChatOptions::new()
                        .with_additional_model_request_fields(serde_json::json!({ "topK": 4 })),
                );
        assert_eq!(
            chat_request.provider_options_map.get("bedrock"),
            Some(&serde_json::json!({
                "raw": true,
                "additionalModelRequestFields": { "topK": 4 }
            }))
        );

        let rerank_request = crate::types::RerankRequest::new(
            "amazon.rerank-v1:0".to_string(),
            "query".to_string(),
            vec!["doc-1".to_string()],
        )
        .with_provider_option("bedrock", serde_json::json!({ "raw": true }))
        .with_bedrock_rerank_options(crate::provider_options::BedrockRerankOptions::new());
        assert_eq!(
            rerank_request.provider_options_map.get("bedrock"),
            Some(&serde_json::json!({ "raw": true }))
        );

        let embedding_request = crate::types::EmbeddingRequest::single("hello")
            .with_provider_option("bedrock", serde_json::json!({ "raw": true }))
            .with_bedrock_embedding_options(
                BedrockEmbeddingOptions::new()
                    .with_input_type(BedrockEmbeddingInputType::SearchDocument),
            );
        assert_eq!(
            embedding_request.provider_options_map.get("bedrock"),
            Some(&serde_json::json!({
                "raw": true,
                "inputType": "search_document"
            }))
        );
    }
}
