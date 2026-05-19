use super::*;
use reqwest::header::AUTHORIZATION;
use siumai::prelude::unified::ResponseFormat;
use siumai::registry::builder::RegistryBuilder;

fn google_vertex_xai_registry_providers()
-> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
    built_in_registry_providers("google-vertex-xai", "google-vertex-xai")
}

fn google_vertex_xai_base_url(project: &str, location: &str) -> String {
    format!(
        "https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi"
    )
}

fn auth_http_config(token: &str) -> siumai::prelude::unified::HttpConfig {
    let mut http_config = siumai::prelude::unified::HttpConfig::empty();
    http_config
        .headers
        .insert("Authorization".to_string(), format!("Bearer {token}"));
    http_config
}

fn google_vertex_xai_registry(
    base_url: &str,
    transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    RegistryBuilder::new(google_vertex_xai_registry_providers())
        .with_provider_base_url_http_config_fetch(
            "google-vertex-xai",
            base_url,
            auth_http_config("test-token"),
            transport,
        )
        .auto_middleware(false)
        .build()
        .expect("build google-vertex-xai registry")
}

#[test]
fn google_vertex_xai_package_settings_preserve_supported_provider_inputs() {
    let transport = CaptureTransport::default();
    let config = siumai::provider_ext::google_vertex_xai::GoogleVertexXaiProviderSettings::new()
        .with_project("test-project")
        .with_location("us-central1")
        .with_header("Authorization", "Bearer test-token")
        .with_header("x-test", "1")
        .with_fetch(Arc::new(transport.clone()))
        .into_config_for_model("xai/grok-4.1-fast-reasoning")
        .expect("settings into config");

    assert_eq!(config.provider_id, "google-vertex-xai");
    assert_eq!(
        config.base_url,
        google_vertex_xai_base_url("test-project", "us-central1")
    );
    assert_eq!(config.common_params.model, "xai/grok-4.1-fast-reasoning");
    assert_eq!(config.include_usage, Some(true));
    assert_eq!(config.supports_structured_outputs, Some(true));
    assert!(config.request_body_transformer.is_some());
    assert_eq!(
        config
            .http_config
            .headers
            .get("Authorization")
            .map(String::as_str),
        Some("Bearer test-token")
    );
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
    assert!(config.http_transport.is_some());
    assert!(transport.take().is_none());
}

#[tokio::test]
async fn google_vertex_xai_public_builder_exposes_chat_only_capabilities() {
    let transport = CaptureTransport::default();

    let client = Provider::google_vertex_xai()
        .project("test-project")
        .location("us-central1")
        .model("xai/grok-4.1-fast-reasoning")
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(transport.clone()))
        .build()
        .await
        .expect("build google-vertex-xai unified client");

    assert_eq!(client.provider_id().as_ref(), "google-vertex-xai");
    assert!(client.as_chat_capability().is_some());
    assert!(client.as_completion_capability().is_none());
    assert!(client.as_embedding_capability().is_none());
    assert!(client.as_image_generation_capability().is_none());
    assert!(client.as_speech_capability().is_none());
    assert!(client.as_rerank_capability().is_none());
    assert!(transport.take().is_none());
}

#[tokio::test]
async fn google_vertex_xai_siumai_provider_config_registry_chat_request_are_equivalent() {
    let model = "xai/grok-4.1-fast-reasoning";
    let base_url = google_vertex_xai_base_url("test-project", "us-central1");
    let response_json = serde_json::json!({
        "id": "chatcmpl-google-vertex-xai-test",
        "object": "chat.completion",
        "created": 1_718_345_013u64,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "hello from google vertex xai"
            },
            "finish_reason": "stop"
        }]
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .google_vertex_xai()
        .project("test-project")
        .location("us-central1")
        .model(model)
        .reasoning(true)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::google_vertex_xai()
        .project("test-project")
        .location("us-central1")
        .model(model)
        .reasoning(true)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        siumai::provider_ext::google_vertex_xai::GoogleVertexXaiClient::from_config(
            siumai::provider_ext::google_vertex_xai::GoogleVertexXaiProviderSettings::new()
                .with_base_url(base_url.clone())
                .with_header("Authorization", "Bearer test-token")
                .with_fetch(Arc::new(config_transport.clone()))
                .into_config_for_model(model)
                .expect("settings into config"),
        )
        .await
        .expect("build config client");

    let registry = google_vertex_xai_registry(&base_url, Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model(&format!("googleVertex.xai:{model}"))
        .expect("build registry google-vertex-xai language model");

    let mut request = make_chat_request_with_model(model);
    request.provider_options_map.insert(
        "googleVertexXai",
        serde_json::json!({ "reasoning_effort": "high" }),
    );
    request.response_format = Some(ResponseFormat::json_schema(serde_json::json!({
        "type": "object",
        "properties": {
            "answer": { "type": "string" }
        },
        "required": ["answer"],
        "additionalProperties": false
    })));

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request.clone()).await;
    let _ = ChatCapability::chat_request(&registry_model, request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-token"
    );
    assert_eq!(siumai_req.url, format!("{base_url}/chat/completions"));
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert!(siumai_req.body.get("reasoning_effort").is_none());
    assert!(siumai_req.body.get("enable_reasoning").is_none());
    assert_eq!(
        siumai_req.body["response_format"]["type"],
        serde_json::json!("json_schema")
    );
    assert_eq!(
        siumai_req.body["response_format"]["json_schema"]["schema"]["properties"]["answer"]["type"],
        serde_json::json!("string")
    );
}

#[tokio::test]
async fn google_vertex_xai_stream_request_includes_usage() {
    let model = "xai/grok-4.20-reasoning";
    let base_url = google_vertex_xai_base_url("test-project", "us-central1");
    let stream_body = br#"data: {"id":"chatcmpl-google-vertex-xai-stream","object":"chat.completion.chunk","created":1718345013,"model":"xai/grok-4.20-reasoning","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":"stop"}]}

data: [DONE]

"#
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .google_vertex_xai()
        .project("test-project")
        .location("us-central1")
        .model(model)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::google_vertex_xai()
        .project("test-project")
        .location("us-central1")
        .model(model)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let registry = google_vertex_xai_registry(&base_url, Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model(&format!("google-vertex-xai:{model}"))
        .expect("build registry google-vertex-xai language model");

    let request = make_chat_request_with_model(model);

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    while futures_util::StreamExt::next(&mut siumai_stream)
        .await
        .is_some()
    {}
    while futures_util::StreamExt::next(&mut provider_stream)
        .await
        .is_some()
    {}
    while futures_util::StreamExt::next(&mut registry_stream)
        .await
        .is_some()
    {}

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
        "Bearer test-token"
    );
    assert_eq!(siumai_req.url, format!("{base_url}/chat/completions"));
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        siumai_req.body.get("stream_options"),
        Some(&serde_json::json!({ "include_usage": true }))
    );
}

#[tokio::test]
async fn google_vertex_xai_registry_non_chat_handles_remain_intentionally_unsupported() {
    let model = "xai/grok-4.1-fast-reasoning";
    let base_url = google_vertex_xai_base_url("test-project", "us-central1");
    let registry_transport = CaptureTransport::default();

    let registry = google_vertex_xai_registry(&base_url, Arc::new(registry_transport.clone()));

    let completion_err = match registry.completion_model(&format!("google-vertex-xai:{model}")) {
        Ok(_) => panic!("build google-vertex-xai completion model should be unsupported"),
        Err(err) => err,
    };
    let embedding_err = match registry.embedding_model(&format!("google-vertex-xai:{model}")) {
        Ok(_) => panic!("build google-vertex-xai embedding model should be unsupported"),
        Err(err) => err,
    };
    let image_err = match registry.image_model(&format!("google-vertex-xai:{model}")) {
        Ok(_) => panic!("build google-vertex-xai image model should be unsupported"),
        Err(err) => err,
    };
    let rerank_err = match registry.reranking_model(&format!("google-vertex-xai:{model}")) {
        Ok(_) => panic!("build google-vertex-xai rerank handle should be unsupported"),
        Err(err) => err,
    };
    let speech_err = match registry.speech_model(&format!("google-vertex-xai:{model}")) {
        Ok(_) => panic!("build google-vertex-xai speech model should be unsupported"),
        Err(err) => err,
    };
    let transcription_err =
        match registry.transcription_model(&format!("google-vertex-xai:{model}")) {
            Ok(_) => panic!("build google-vertex-xai transcription model should be unsupported"),
            Err(err) => err,
        };

    assert_unsupported_operation(&completion_err);
    assert_unsupported_operation(&embedding_err);
    assert_unsupported_operation(&image_err);
    assert_unsupported_operation(&rerank_err);
    assert_unsupported_operation(&speech_err);
    assert_unsupported_operation(&transcription_err);
    assert_capture_transports_unused(&[&registry_transport]);
}
