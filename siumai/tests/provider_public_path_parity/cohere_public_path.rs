use super::*;
use reqwest::header::AUTHORIZATION;
use siumai::experimental::client::LlmClient;
use siumai::prelude::unified::{
    EmbeddingExtensions, EmbeddingRequest, ResponseFormat, Tool, ToolChoice,
};
use siumai::provider_ext::cohere::{
    CohereChatOptions, CohereChatRequestExt, CohereClient, CohereConfig, CohereEmbeddingInputType,
    CohereEmbeddingOptions, CohereEmbeddingRequestExt, CohereEmbeddingTruncate,
    CohereProviderSettings, CohereRerankOptions, CohereRerankRequestExt, CohereThinkingConfig,
    CohereThinkingType,
};

fn cohere_registry_builder() -> siumai::registry::builder::RegistryBuilder {
    built_in_registry_builder("cohere", "cohere")
}

fn make_cohere_registry(
    api_key: &str,
    base_url: &str,
    transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    cohere_registry_builder()
        .with_api_key(api_key)
        .with_base_url(base_url)
        .fetch(transport)
        .auto_middleware(false)
        .build()
        .expect("build cohere registry")
}

fn make_cohere_override_registry(
    global_transport: Arc<dyn HttpTransport>,
    provider_transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    cohere_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/global")
        .fetch(global_transport)
        .with_provider_api_key_base_url_fetch(
            "cohere",
            "ctx-key",
            "https://example.com/cohere",
            provider_transport,
        )
        .auto_middleware(false)
        .build()
        .expect("build cohere override registry")
}

#[test]
fn cohere_package_settings_preserve_supported_provider_inputs() {
    let config = CohereProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/cohere")
        .with_header("x-test", "1")
        .into_config_for_model("command-a-03-2025")
        .expect("settings into config");

    assert_eq!(config.base_url, "https://example.com/cohere");
    assert_eq!(config.common_params.model, "command-a-03-2025");
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
}

#[tokio::test]
async fn cohere_public_builder_exposes_unified_capabilities() {
    let transport = CaptureTransport::default();

    let client = Provider::cohere()
        .api_key("test-key")
        .language_model("command-a-03-2025")
        .fetch(Arc::new(transport.clone()))
        .build()
        .expect("build provider client");

    assert_eq!(client.provider_id().as_ref(), "cohere");
    assert!(client.as_chat_capability().is_some());
    assert!(client.as_embedding_capability().is_some());
    assert!(client.as_rerank_capability().is_some());
    assert!(transport.take().is_none());
    assert!(transport.take_stream().is_none());
}

#[tokio::test]
async fn cohere_siumai_provider_config_chat_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let siumai_client = Siumai::builder()
        .cohere()
        .api_key("test-key")
        .base_url("https://example.com/cohere")
        .model("command-a-03-2025")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::cohere()
        .api_key("test-key")
        .base_url("https://example.com/cohere")
        .language_model("command-a-03-2025")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = CohereClient::from_config(
        CohereConfig::new("test-key")
            .with_base_url("https://example.com/cohere")
            .with_model("command-a-03-2025")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model("command-a-03-2025")
        .with_tools(vec![Tool::function(
            "lookup_weather",
            "Look up the weather",
            serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"],
                "additionalProperties": false
            }),
        )])
        .with_tool_choice(ToolChoice::None)
        .with_response_format(ResponseFormat::json_schema(schema.clone()))
        .with_cohere_options(
            CohereChatOptions::new().with_thinking(
                CohereThinkingConfig::new()
                    .with_type(CohereThinkingType::Enabled)
                    .with_token_budget(2048),
            ),
        );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(siumai_req.url, "https://example.com/cohere/chat");
    assert_eq!(
        siumai_req.body["model"],
        serde_json::json!("command-a-03-2025")
    );
    assert_eq!(siumai_req.body["tool_choice"], serde_json::json!("NONE"));
    assert_eq!(
        siumai_req.body["response_format"],
        serde_json::json!({
            "type": "json_object",
            "json_schema": schema
        })
    );
    assert_eq!(
        siumai_req.body["thinking"],
        serde_json::json!({
            "type": "enabled",
            "token_budget": 2048
        })
    );
}

#[tokio::test]
async fn cohere_registry_chat_request_matches_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let config_client = CohereClient::from_config(
        CohereConfig::new("test-key")
            .with_base_url("https://example.com/cohere")
            .with_model("command-a-03-2025")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_cohere_registry(
        "test-key",
        "https://example.com/cohere",
        Arc::new(registry_transport.clone()),
    );
    let registry_model = registry
        .language_model("cohere:command-a-03-2025")
        .expect("build registry language model");

    let request = make_chat_request_with_model("command-a-03-2025")
        .with_tools(vec![Tool::function(
            "lookup_weather",
            "Look up the weather",
            serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"],
                "additionalProperties": false
            }),
        )])
        .with_tool_choice(ToolChoice::None)
        .with_response_format(ResponseFormat::json_schema(schema.clone()))
        .with_cohere_options(
            CohereChatOptions::new().with_thinking(
                CohereThinkingConfig::new()
                    .with_type(CohereThinkingType::Enabled)
                    .with_token_budget(2048),
            ),
        );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(registry_req.url, "https://example.com/cohere/chat");
    assert_eq!(
        registry_req.body["model"],
        serde_json::json!("command-a-03-2025")
    );
    assert_eq!(registry_req.body["tool_choice"], serde_json::json!("NONE"));
    assert_eq!(
        registry_req.body["response_format"],
        serde_json::json!({
            "type": "json_object",
            "json_schema": schema
        })
    );
}

#[tokio::test]
async fn cohere_siumai_provider_config_embedding_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .cohere()
        .api_key("test-key")
        .base_url("https://example.com/cohere")
        .model("embed-v4.0")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::cohere()
        .api_key("test-key")
        .base_url("https://example.com/cohere")
        .embedding_model("embed-v4.0")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = CohereClient::from_config(
        CohereConfig::new("test-key")
            .with_base_url("https://example.com/cohere")
            .with_model("embed-v4.0")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = EmbeddingRequest::new(vec!["text-1".to_string(), "text-2".to_string()])
        .with_model("embed-v4.0")
        .with_dimensions(512)
        .with_cohere_options(
            CohereEmbeddingOptions::new()
                .with_input_type(CohereEmbeddingInputType::SearchDocument)
                .with_truncate(CohereEmbeddingTruncate::End)
                .with_output_dimension(1024),
        );

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(siumai_req.url, "https://example.com/cohere/embed");
    assert_eq!(siumai_req.body["model"], serde_json::json!("embed-v4.0"));
    assert_eq!(
        siumai_req.body["embedding_types"],
        serde_json::json!(["float"])
    );
    assert_eq!(
        siumai_req.body["texts"],
        serde_json::json!(["text-1", "text-2"])
    );
    assert_eq!(
        siumai_req.body["input_type"],
        serde_json::json!("search_document")
    );
    assert_eq!(siumai_req.body["truncate"], serde_json::json!("END"));
    assert_eq!(siumai_req.body["output_dimension"], serde_json::json!(1024));
}

#[tokio::test]
async fn cohere_registry_embedding_request_matches_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = CohereClient::from_config(
        CohereConfig::new("test-key")
            .with_base_url("https://example.com/cohere")
            .with_model("embed-v4.0")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_cohere_registry(
        "test-key",
        "https://example.com/cohere",
        Arc::new(registry_transport.clone()),
    );
    let registry_model = registry
        .embedding_model("cohere:embed-v4.0")
        .expect("build registry embedding model");

    let request = EmbeddingRequest::new(vec!["text-1".to_string(), "text-2".to_string()])
        .with_model("embed-v4.0")
        .with_dimensions(512)
        .with_cohere_options(
            CohereEmbeddingOptions::new()
                .with_input_type(CohereEmbeddingInputType::SearchDocument)
                .with_truncate(CohereEmbeddingTruncate::End)
                .with_output_dimension(1024),
        );

    let _ = config_client.embed_with_config(request.clone()).await;
    let _ = registry_model.embed_with_config(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(registry_req.url, "https://example.com/cohere/embed");
    assert_eq!(registry_req.body["model"], serde_json::json!("embed-v4.0"));
    assert_eq!(
        registry_req.body["output_dimension"],
        serde_json::json!(1024)
    );
}

#[tokio::test]
async fn cohere_siumai_provider_config_rerank_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .cohere()
        .api_key("test-key")
        .base_url("https://example.com/cohere")
        .model("rerank-v3.5")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::cohere()
        .api_key("test-key")
        .base_url("https://example.com/cohere")
        .reranking_model("rerank-v3.5")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .expect("build provider client");

    let config_client = CohereClient::from_config(
        CohereConfig::new("test-key")
            .with_base_url("https://example.com/cohere")
            .with_model("rerank-v3.5")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_rerank_request_with_model("rerank-v3.5")
        .with_top_n(1)
        .with_cohere_options(
            CohereRerankOptions::new()
                .with_max_tokens_per_doc(1000)
                .with_priority(1),
        );

    let _ = siumai_client.rerank(request.clone()).await;
    let _ = provider_client.rerank(request.clone()).await;
    let _ = config_client.rerank(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(siumai_req.url, "https://example.com/cohere/rerank");
    assert_eq!(siumai_req.body["model"], serde_json::json!("rerank-v3.5"));
}

#[tokio::test]
async fn cohere_registry_rerank_request_matches_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = CohereClient::from_config(
        CohereConfig::new("test-key")
            .with_base_url("https://example.com/cohere")
            .with_model("rerank-v3.5")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_cohere_registry(
        "test-key",
        "https://example.com/cohere",
        Arc::new(registry_transport.clone()),
    );
    let registry_model = registry
        .reranking_model("cohere:rerank-v3.5")
        .expect("build registry rerank model");

    let request = make_rerank_request_with_model("rerank-v3.5")
        .with_top_n(1)
        .with_cohere_options(
            CohereRerankOptions::new()
                .with_max_tokens_per_doc(1000)
                .with_priority(1),
        );

    let _ = config_client.rerank(request.clone()).await;
    let _ = registry_model.rerank(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(registry_req.url, "https://example.com/cohere/rerank");
    assert_eq!(registry_req.body["model"], serde_json::json!("rerank-v3.5"));
    assert_eq!(registry_req.body["top_n"], serde_json::json!(1));
    assert_eq!(
        registry_req.body["max_tokens_per_doc"],
        serde_json::json!(1000)
    );
    assert_eq!(registry_req.body["priority"], serde_json::json!(1));
}

#[tokio::test]
async fn cohere_registry_rerank_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let cohere_transport = CaptureTransport::default();

    let registry = make_cohere_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(cohere_transport.clone()),
    );

    let handle = registry
        .reranking_model("cohere:rerank-v3.5")
        .expect("build registry rerank model");

    let _ = handle
        .rerank(
            make_rerank_request_with_model("rerank-v3.5")
                .with_top_n(1)
                .with_cohere_options(
                    CohereRerankOptions::new()
                        .with_max_tokens_per_doc(1000)
                        .with_priority(1),
                ),
        )
        .await;

    let req = cohere_transport.take().expect("captured request");
    assert!(global_transport.take().is_none());
    assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
    assert_eq!(req.url, "https://example.com/cohere/rerank");
    assert_eq!(req.body["model"], serde_json::json!("rerank-v3.5"));
    assert_eq!(req.body["top_n"], serde_json::json!(1));
    assert_eq!(req.body["max_tokens_per_doc"], serde_json::json!(1000));
    assert_eq!(req.body["priority"], serde_json::json!(1));
}
