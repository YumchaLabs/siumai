use super::*;
use reqwest::header::AUTHORIZATION;
use siumai::experimental::client::LlmClient;
use siumai::provider_ext::togetherai::{
    TogetherAIImageModelOptions, TogetherAIProviderSettings, TogetherAIRerankingModelOptions,
    TogetherAiImageRequestExt, TogetherAiRerankRequestExt,
};

fn togetherai_registry_builder() -> siumai::registry::builder::RegistryBuilder {
    built_in_registry_builder("togetherai", "togetherai")
}

fn make_togetherai_registry(
    api_key: &str,
    base_url: &str,
    transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    togetherai_registry_builder()
        .with_api_key(api_key)
        .with_base_url(base_url)
        .fetch(transport)
        .auto_middleware(false)
        .build()
        .expect("build togetherai registry")
}

fn make_togetherai_override_registry(
    global_transport: Arc<dyn HttpTransport>,
    provider_transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    togetherai_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/global")
        .fetch(global_transport)
        .with_provider_api_key_base_url_fetch(
            "togetherai",
            "ctx-key",
            "https://example.com/together",
            provider_transport,
        )
        .auto_middleware(false)
        .build()
        .expect("build togetherai override registry")
}

#[test]
fn togetherai_package_settings_preserve_supported_provider_inputs() {
    let config = TogetherAIProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/together")
        .with_header("x-test", "1")
        .into_config_for_model("Salesforce/Llama-Rank-v1")
        .expect("settings into config");

    assert_eq!(config.base_url, "https://example.com/together");
    assert_eq!(config.common_params.model, "Salesforce/Llama-Rank-v1");
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
}

#[tokio::test]
async fn togetherai_public_builder_exposes_unified_capabilities() {
    let transport = CaptureTransport::default();

    let client = Provider::togetherai()
        .api_key("test-key")
        .base_url("https://example.com/together")
        .fetch(Arc::new(transport.clone()))
        .build()
        .await
        .expect("build togetherai unified client");

    assert_eq!(client.provider_id().as_ref(), "togetherai");
    assert!(client.as_chat_capability().is_some());
    assert!(client.as_completion_capability().is_some());
    assert!(client.as_embedding_capability().is_some());
    assert!(client.as_image_generation_capability().is_some());
    assert!(client.as_speech_capability().is_some());
    assert!(client.as_transcription_capability().is_some());
    assert!(client.as_rerank_capability().is_some());
    assert!(transport.take().is_none());
}

#[tokio::test]
async fn togetherai_siumai_provider_config_rerank_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .togetherai()
        .api_key("test-key")
        .base_url("https://example.com/together")
        .model("Salesforce/Llama-Rank-v1")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::togetherai()
        .api_key("test-key")
        .base_url("https://example.com/together")
        .model("Salesforce/Llama-Rank-v1")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::togetherai::TogetherAiClient::from_config(
        siumai::provider_ext::togetherai::TogetherAiConfig::new("test-key")
            .with_base_url("https://example.com/together")
            .with_model("Salesforce/Llama-Rank-v1")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_rerank_request_with_model("Salesforce/Llama-Rank-v1")
        .with_top_n(1)
        .with_togetherai_options(
            TogetherAIRerankingModelOptions::new().with_rank_fields(vec!["example".to_string()]),
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
    assert_eq!(siumai_req.url, "https://example.com/together/rerank");
}

#[tokio::test]
async fn togetherai_registry_rerank_request_matches_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = siumai::provider_ext::togetherai::TogetherAiClient::from_config(
        siumai::provider_ext::togetherai::TogetherAiConfig::new("test-key")
            .with_base_url("https://example.com/together")
            .with_model("Salesforce/Llama-Rank-v1")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_togetherai_registry(
        "test-key",
        "https://example.com/together",
        Arc::new(registry_transport.clone()),
    );
    let registry_model = registry
        .reranking_model("togetherai:Salesforce/Llama-Rank-v1")
        .expect("build registry rerank model");

    let request = make_rerank_request_with_model("Salesforce/Llama-Rank-v1")
        .with_top_n(1)
        .with_togetherai_options(
            TogetherAIRerankingModelOptions::new().with_rank_fields(vec!["example".to_string()]),
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
    assert_eq!(registry_req.url, "https://example.com/together/rerank");
    assert_eq!(
        registry_req.body["model"],
        serde_json::json!("Salesforce/Llama-Rank-v1")
    );
    assert_eq!(registry_req.body["top_n"], serde_json::json!(1));
    assert_eq!(
        registry_req.body["rank_fields"],
        serde_json::json!(["example"])
    );
}

#[tokio::test]
async fn togetherai_registry_rerank_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let together_transport = CaptureTransport::default();

    let registry = make_togetherai_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(together_transport.clone()),
    );

    let handle = registry
        .reranking_model("togetherai:Salesforce/Llama-Rank-v1")
        .expect("build registry rerank model");

    let _ = handle
        .rerank(
            make_rerank_request_with_model("Salesforce/Llama-Rank-v1")
                .with_top_n(1)
                .with_togetherai_options(
                    TogetherAIRerankingModelOptions::new()
                        .with_rank_fields(vec!["example".to_string()]),
                ),
        )
        .await;

    let req = together_transport.take().expect("captured request");
    assert!(global_transport.take().is_none());
    assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
    assert_eq!(req.url, "https://example.com/together/rerank");
    assert_eq!(req.body["top_n"], serde_json::json!(1));
    assert_eq!(req.body["rank_fields"], serde_json::json!(["example"]));
}

#[tokio::test]
async fn togetherai_image_request_alias_options_are_equivalent_across_public_paths() {
    let response_json = serde_json::json!({
        "data": [
            {
                "b64_json": "aGVsbG8="
            }
        ]
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "black-forest-labs/FLUX.1-schnell";
    let base_url = "https://example.com/together";

    let siumai_client = Siumai::builder()
        .togetherai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::togetherai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let registry =
        make_togetherai_registry("test-key", base_url, Arc::new(registry_transport.clone()));
    let registry_model = registry
        .image_model(&format!("togetherai:{model}"))
        .expect("build registry togetherai image model");

    let request = make_image_request_with_model(model).with_togetherai_image_options(
        TogetherAIImageModelOptions::new()
            .with_steps(12)
            .with_guidance(3.5)
            .with_disable_safety_checker(true),
    );

    let siumai_resp = siumai_client
        .generate_images(request.clone())
        .await
        .expect("siumai image generation ok");
    let provider_resp = provider_client
        .generate_images(request.clone())
        .await
        .expect("provider image generation ok");
    let registry_resp = registry_model
        .generate_images(request)
        .await
        .expect("registry image generation ok");

    assert_eq!(siumai_resp.images[0].b64_json.as_deref(), Some("aGVsbG8="));
    assert_eq!(
        provider_resp.images[0].b64_json.as_deref(),
        Some("aGVsbG8=")
    );
    assert_eq!(
        registry_resp.images[0].b64_json.as_deref(),
        Some("aGVsbG8=")
    );

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, format!("{base_url}/images/generations"));
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(siumai_req.body["steps"], serde_json::json!(12));
    assert_eq!(siumai_req.body["guidance"], serde_json::json!(3.5));
    assert_eq!(
        siumai_req.body["disable_safety_checker"],
        serde_json::json!(true)
    );
}

#[tokio::test]
async fn togetherai_siumai_provider_registry_chat_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .togetherai()
        .api_key("test-key")
        .base_url("https://example.com/together")
        .model("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::togetherai()
        .api_key("test-key")
        .base_url("https://example.com/together")
        .model("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");
    let registry = make_togetherai_registry(
        "test-key",
        "https://example.com/together",
        Arc::new(registry_transport.clone()),
    );

    let registry_model = registry
        .language_model("togetherai:meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        .expect("build registry togetherai language model");

    let request = make_chat_request_with_model("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo");

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = ChatCapability::chat_request(&registry_model, request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.com/together/chat/completions"
    );
    assert_eq!(
        siumai_req.body["model"],
        serde_json::json!("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    );
}

#[tokio::test]
async fn togetherai_completion_siumai_provider_registry_request_are_equivalent() {
    let response_json = serde_json::json!({
        "id": "cmpl-together-test",
        "object": "text_completion",
        "created": 1_718_345_013u64,
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "choices": [
            {
                "text": "done",
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "total_tokens": 9
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo";
    let base_url = "https://example.com/together";

    let siumai_client = Siumai::builder()
        .togetherai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::togetherai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let registry =
        make_togetherai_registry("test-key", base_url, Arc::new(registry_transport.clone()));

    let registry_model = registry
        .completion_model(&format!("togetherai:{model}"))
        .expect("build registry togetherai completion model");

    let request = CompletionRequest::from_prompt(vec![
        ChatMessage::system("Be terse.").build(),
        ChatMessage::user("Hello").build(),
        ChatMessage::assistant("Hi").build(),
        ChatMessage::user("Continue").build(),
    ])
    .with_model(model)
    .with_provider_option("togetherai", serde_json::json!({ "suffix": "!" }));

    let siumai_resp = siumai_client
        .as_completion_capability()
        .expect("siumai completion capability")
        .complete(request.clone())
        .await
        .expect("siumai completion ok");
    let provider_resp = provider_client
        .as_completion_capability()
        .expect("provider completion capability")
        .complete(request.clone())
        .await
        .expect("provider completion ok");
    let registry_resp = registry_model
        .as_completion_capability()
        .expect("registry completion capability")
        .complete(request)
        .await
        .expect("registry completion ok");

    assert_eq!(siumai_resp.text(), "done");
    assert_eq!(provider_resp.text(), "done");
    assert_eq!(registry_resp.text(), "done");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(siumai_req.url, "https://example.com/together/completions");
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(siumai_req.body["suffix"], serde_json::json!("!"));
    assert_eq!(
        siumai_req.body["prompt"],
        serde_json::json!(
            "Be terse.\n\nuser:\nHello\n\nassistant:\nHi\n\nuser:\nContinue\n\nassistant:\n"
        )
    );
    assert_eq!(siumai_req.body["stop"], serde_json::json!(["\nuser:"]));
}

#[tokio::test]
async fn togetherai_completion_stream_public_paths_keep_raw_chunks_runtime_only() {
    use futures_util::StreamExt;

    let model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo";
    let stream_body = concat!(
            "data: {\"id\":\"cmpl-together-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo\",\"choices\":[{\"text\":\"hello\",\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"cmpl-together-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo\",\"choices\":[{\"text\":\" world\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n",
            "data: [DONE]\n\n"
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let base_url = "https://example.com/together";

    let siumai_client = Siumai::builder()
        .togetherai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::togetherai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let registry =
        make_togetherai_registry("test-key", base_url, Arc::new(registry_transport.clone()));

    let registry_model = registry
        .completion_model(&format!("togetherai:{model}"))
        .expect("build registry togetherai completion model");

    let request = make_completion_request_with_model(model).with_include_raw_chunks(true);

    let mut siumai_stream = siumai_client
        .as_completion_capability()
        .expect("siumai completion capability")
        .complete_stream(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .as_completion_capability()
        .expect("provider completion capability")
        .complete_stream(request.clone())
        .await
        .expect("provider stream ok");
    let mut registry_stream = registry_model
        .as_completion_capability()
        .expect("registry completion capability")
        .complete_stream(request)
        .await
        .expect("registry stream ok");

    while siumai_stream.next().await.is_some() {}
    while provider_stream.next().await.is_some() {}
    while registry_stream.next().await.is_some() {}

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(siumai_req.url, "https://example.com/together/completions");
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert!(siumai_req.body.get("stream_options").is_none());
    assert!(siumai_req.body.get("includeRawChunks").is_none());
}
