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
    google_outer.insert("google".to_string(), google_inner);
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
    vertex_outer.insert("vertex".to_string(), vertex_inner);
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
    use siumai::prelude::unified::{ChatRequest, RerankRequest};
    use siumai::provider_ext::bedrock::{
        BedrockChatOptions, BedrockChatRequestExt, BedrockRerankOptions, BedrockRerankRequestExt,
    };

    let chat_request = ChatRequest::new(vec![siumai::user!("hello bedrock")])
        .with_bedrock_chat_options(
            BedrockChatOptions::new()
                .with_additional_model_request_fields(serde_json::json!({ "topK": 16 })),
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
            "additionalModelRequestFields": { "topK": 16 }
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
}

#[cfg(feature = "bedrock")]
#[test]
fn bedrock_metadata_helpers_remain_bedrock_namespace_led() {
    use siumai::prelude::unified::{ChatResponse, MessageContent};
    use siumai::provider_ext::bedrock::BedrockChatResponseExt;
    use std::collections::HashMap;

    let mut response = ChatResponse::new(MessageContent::Text("ok".to_string()));
    let mut bedrock = HashMap::new();
    bedrock.insert(
        "isJsonResponseFromTool".to_string(),
        serde_json::Value::Bool(true),
    );
    let mut metadata = HashMap::new();
    metadata.insert("bedrock".to_string(), bedrock);
    response.provider_metadata = Some(metadata);

    let parsed = response.bedrock_metadata().expect("bedrock metadata");
    assert_eq!(parsed.is_json_response_from_tool, Some(true));
}
