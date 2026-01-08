#![cfg(feature = "google-vertex")]

use siumai::experimental::core::{ProviderContext, ProviderSpec};
use std::collections::HashMap;

fn vertex_express_ctx() -> ProviderContext {
    ProviderContext::new(
        "vertex",
        "https://aiplatform.googleapis.com/v1/publishers/google".to_string(),
        Some("test-api-key".to_string()),
        HashMap::new(),
    )
}

#[test]
fn vertex_express_chat_url_appends_key_query_param() {
    let ctx = vertex_express_ctx();
    let spec = siumai::experimental::providers::google_vertex::standards::vertex_generative_ai::VertexGenerativeAiStandard::new().create_spec("vertex");

    let req = siumai::prelude::unified::ChatRequest::builder()
        .message(siumai::prelude::unified::ChatMessage::user("hi").build())
        .model("gemini-2.5-pro")
        .build();

    let url = spec.chat_url(false, &req, &ctx);
    assert_eq!(
        url,
        "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-pro:generateContent?key=test-api-key"
    );

    let stream_req = siumai::prelude::unified::ChatRequest::builder()
        .message(siumai::prelude::unified::ChatMessage::user("hi").build())
        .model("gemini-2.5-pro")
        .stream(true)
        .build();

    let url_stream = spec.chat_url(true, &stream_req, &ctx);
    assert_eq!(
        url_stream,
        "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-pro:streamGenerateContent?alt=sse&key=test-api-key"
    );

    let headers = spec.build_headers(&ctx).unwrap();
    assert_eq!(headers.get("content-type").unwrap(), "application/json");
    assert!(headers.get("x-goog-api-key").is_none());
}

#[test]
fn vertex_express_embedding_url_appends_key_query_param() {
    let ctx = vertex_express_ctx();
    let spec = siumai::experimental::providers::google_vertex::standards::vertex_embedding::VertexEmbeddingStandard::new().create_spec("vertex");

    let req = siumai::prelude::unified::EmbeddingRequest::single("hello")
        .with_model("textembedding-gecko@001");
    let url = spec.embedding_url(&req, &ctx);
    assert_eq!(
        url,
        "https://aiplatform.googleapis.com/v1/publishers/google/models/textembedding-gecko@001:predict?key=test-api-key"
    );
}

#[test]
fn vertex_enterprise_does_not_append_key_when_authorization_present() {
    let mut extra = HashMap::new();
    extra.insert("Authorization".to_string(), "Bearer token".to_string());
    let ctx = ProviderContext::new(
        "vertex",
        "https://aiplatform.googleapis.com/v1/publishers/google".to_string(),
        Some("test-api-key".to_string()),
        extra,
    );

    let spec = siumai::experimental::providers::google_vertex::standards::vertex_generative_ai::VertexGenerativeAiStandard::new().create_spec("vertex");
    let req = siumai::prelude::unified::ChatRequest::builder()
        .message(siumai::prelude::unified::ChatMessage::user("hi").build())
        .model("gemini-2.5-pro")
        .build();
    let url = spec.chat_url(false, &req, &ctx);
    assert_eq!(
        url,
        "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-pro:generateContent"
    );
}
