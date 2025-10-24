use std::sync::Arc;

use siumai::providers::openai_compatible::openai_config::OpenAiCompatibleConfig;
use siumai::providers::openai_compatible::registry::{ConfigurableAdapter, ProviderConfig, ProviderFieldMappings};
use siumai::traits::RerankCapability;
use siumai::types::RerankRequest;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn siliconflow_rerank_maps_meta_tokens_to_usage() {
    // Start mock server
    let server = MockServer::start().await;

    // SiliconFlow-style rerank response with meta.tokens
    let resp = serde_json::json!({
        "id": "rerank-123",
        "results": [
            {"index": 1, "relevance_score": 0.95, "document": "doc-b"},
            {"index": 0, "relevance_score": 0.35, "document": "doc-a"}
        ],
        "meta": {"tokens": {"input_tokens": 123, "output_tokens": 4}}
    });

    // Expect a POST to /rerank with auth header
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(resp))
        .mount(&server)
        .await;

    // Build adapter + client config
    let provider_config = ProviderConfig {
        id: "siliconflow".to_string(),
        name: "SiliconFlow".to_string(),
        base_url: server.uri(),
        field_mappings: ProviderFieldMappings::default(),
        capabilities: vec!["chat".to_string()],
        default_model: Some("bge-reranker-v2-m3".to_string()),
        supports_reasoning: false,
    };
    let adapter = Arc::new(ConfigurableAdapter::new(provider_config));

    let config = OpenAiCompatibleConfig::new(
        "siliconflow",
        "test-key",
        &server.uri(),
        adapter,
    )
    .with_model("bge-reranker-v2-m3");

    // Create client
    let client = siumai::providers::openai_compatible::openai_client::OpenAiCompatibleClient::new(config)
        .await
        .expect("client");

    // Make rerank request
    let request = RerankRequest {
        model: "bge-reranker-v2-m3".to_string(),
        query: "test query".to_string(),
        documents: vec!["doc-a".to_string(), "doc-b".to_string()],
        instruction: None,
        top_n: Some(2),
        return_documents: Some(true),
        max_chunks_per_doc: None,
        overlap_tokens: None,
    };

    let response = client.rerank(request).await.expect("rerank response");

    // Validate results ordering and token mapping
    assert_eq!(response.id, "rerank-123");
    assert_eq!(response.results.len(), 2);
    assert_eq!(response.tokens.input_tokens, 123);
    assert_eq!(response.tokens.output_tokens, 4);
}

