#[cfg(feature = "google")]
#[test]
fn gemini_metadata_helpers_accept_vertex_and_google_namespaces() {
    use siumai::prelude::unified::{ChatResponse, MessageContent};
    use siumai::provider_ext::google::GeminiChatResponseExt;
    use std::collections::HashMap;

    let mut google_response = ChatResponse::new(MessageContent::Text("google".to_string()));
    let mut google_inner = HashMap::new();
    google_inner.insert(
        "thoughtSignature".to_string(),
        serde_json::Value::String("google-sig".to_string()),
    );
    let mut google_outer = HashMap::new();
    google_outer.insert(
        "google".to_string(),
        serde_json::Value::Object(google_inner.into_iter().collect()),
    );
    google_response.provider_metadata = Some(google_outer);

    let google_meta = google_response.gemini_metadata().expect("google metadata");
    assert_eq!(google_meta.thought_signature.as_deref(), Some("google-sig"));

    let mut vertex_response = ChatResponse::new(MessageContent::Text("vertex".to_string()));
    let mut vertex_inner = HashMap::new();
    vertex_inner.insert(
        "thoughtSignature".to_string(),
        serde_json::Value::String("vertex-sig".to_string()),
    );
    let mut vertex_outer = HashMap::new();
    vertex_outer.insert(
        "vertex".to_string(),
        serde_json::Value::Object(vertex_inner.into_iter().collect()),
    );
    vertex_response.provider_metadata = Some(vertex_outer);

    let vertex_meta = vertex_response.gemini_metadata().expect("vertex metadata");
    assert_eq!(vertex_meta.thought_signature.as_deref(), Some("vertex-sig"));
}

#[cfg(feature = "google-vertex")]
#[test]
fn google_vertex_request_helpers_remain_vertex_options_led() {
    use siumai::prelude::unified::EmbeddingRequest;
    use siumai::provider_ext::google_vertex::{VertexEmbeddingOptions, VertexEmbeddingRequestExt};

    let request = EmbeddingRequest::new(vec!["hello from vertex".to_string()])
        .with_model("text-embedding-004")
        .with_vertex_embedding_options(VertexEmbeddingOptions {
            output_dimensionality: Some(256),
            ..Default::default()
        });

    assert_eq!(
        request.provider_options_map.get("vertex"),
        Some(&serde_json::json!({ "outputDimensionality": 256 }))
    );
    assert!(request.provider_options_map.get("google").is_none());
}

#[cfg(feature = "bedrock")]
#[test]
fn bedrock_request_helpers_remain_bedrock_request_helper_led() {
    use siumai::prelude::unified::{ChatMessage, ChatRequest, ContentPart, RerankRequest};
    use siumai::provider_ext::bedrock::{
        BedrockCachePoint, BedrockCacheTtl, BedrockChatOptions, BedrockChatRequestExt,
        BedrockMessageExt, BedrockReasoningConfig, BedrockReasoningEffort, BedrockReasoningType,
        BedrockRequestContentPartExt, BedrockRerankOptions, BedrockRerankRequestExt,
        BedrockServiceTier,
    };

    let chat_request = ChatRequest::new(vec![siumai::user!("hello bedrock")])
        .with_bedrock_chat_options(
            BedrockChatOptions::new()
                .with_additional_model_request_fields(serde_json::json!({ "topK": 16 }))
                .with_reasoning_config(
                    BedrockReasoningConfig::new()
                        .with_type(BedrockReasoningType::Enabled)
                        .with_budget_tokens(2048)
                        .with_max_reasoning_effort(BedrockReasoningEffort::High),
                )
                .with_anthropic_beta(["context-1m-2025-08-07"])
                .with_service_tier(BedrockServiceTier::Priority),
        );
    let rerank_request = RerankRequest::new(
        "amazon.rerank-v1:0".to_string(),
        "query".to_string(),
        vec!["doc-1".to_string()],
    )
    .with_bedrock_rerank_options(
        BedrockRerankOptions::new()
            .with_region("us-east-1")
            .with_next_token("token-1")
            .with_additional_model_request_fields(serde_json::json!({ "topK": 4 })),
    );

    assert_eq!(
        chat_request.provider_options_map.get("bedrock"),
        Some(&serde_json::json!({
            "additionalModelRequestFields": { "topK": 16 },
            "reasoningConfig": {
                "type": "enabled",
                "budgetTokens": 2048,
                "maxReasoningEffort": "high"
            },
            "anthropicBeta": ["context-1m-2025-08-07"],
            "serviceTier": "priority"
        }))
    );
    assert_eq!(
        rerank_request.provider_options_map.get("bedrock"),
        Some(&serde_json::json!({
            "region": "us-east-1",
            "nextToken": "token-1",
            "additionalModelRequestFields": { "topK": 4 }
        }))
    );

    let message = ChatMessage::user("cached")
        .with_bedrock_cache_point(BedrockCachePoint::new().with_ttl(BedrockCacheTtl::FiveMinutes))
        .build();
    assert_eq!(
        message.provider_options().get("bedrock"),
        Some(&serde_json::json!({
            "cachePoint": {
                "type": "default",
                "ttl": "5m"
            }
        }))
    );

    let part = ContentPart::file_base64("AAECAw==", "application/pdf", None)
        .with_bedrock_document_citations(true);
    assert_eq!(
        part.provider_options()
            .and_then(|provider_options| provider_options.get("bedrock")),
        Some(&serde_json::json!({
            "citations": { "enabled": true }
        }))
    );
}

#[cfg(feature = "bedrock")]
#[test]
fn bedrock_metadata_helpers_remain_bedrock_namespace_led() {
    use siumai::prelude::unified::{ChatResponse, ContentPart, MessageContent, ProviderOptionsMap};
    use siumai::provider_ext::bedrock::{
        BedrockChatResponseExt, BedrockContentPartExt, assistant_message_with_reasoning_metadata,
    };
    use std::collections::HashMap;

    let mut response = ChatResponse::new(MessageContent::Text("ok".to_string()));
    let mut bedrock = HashMap::new();
    bedrock.insert(
        "isJsonResponseFromTool".to_string(),
        serde_json::Value::Bool(true),
    );
    let mut metadata = HashMap::new();
    metadata.insert(
        "bedrock".to_string(),
        serde_json::Value::Object(bedrock.into_iter().collect()),
    );
    response.provider_metadata = Some(metadata);

    let parsed = response.bedrock_metadata().expect("bedrock metadata");
    assert_eq!(parsed.is_json_response_from_tool, Some(true));

    let reasoning_part = ContentPart::Reasoning {
        text: "internal".to_string(),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: Some(HashMap::from([(
            "bedrock".to_string(),
            serde_json::json!({
                "signature": "sig-1",
                "redactedData": "blob"
            }),
        )])),
    };
    let reasoning_metadata = reasoning_part
        .bedrock_reasoning_metadata()
        .expect("bedrock reasoning metadata");
    assert_eq!(reasoning_metadata.signature.as_deref(), Some("sig-1"));
    assert_eq!(reasoning_metadata.redacted_data.as_deref(), Some("blob"));

    let replay_response = ChatResponse::new(MessageContent::MultiModal(vec![reasoning_part]));
    let replay_message = assistant_message_with_reasoning_metadata(&replay_response);
    let MessageContent::MultiModal(parts) = replay_message.content else {
        panic!("expected multimodal replay content");
    };
    let bedrock_options = parts[0]
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
