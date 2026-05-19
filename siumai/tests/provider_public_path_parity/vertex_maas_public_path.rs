use super::*;
use reqwest::header::AUTHORIZATION;
use siumai::experimental::client::LlmClient;
use siumai::registry::builder::RegistryBuilder;

fn vertex_maas_registry_providers() -> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
    built_in_registry_providers("vertex-maas", "vertex-maas")
}

fn vertex_maas_base_url(project: &str, location: &str) -> String {
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

fn vertex_maas_registry(
    base_url: &str,
    transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    RegistryBuilder::new(vertex_maas_registry_providers())
        .with_provider_base_url_http_config_fetch(
            "vertex-maas",
            base_url,
            auth_http_config("test-token"),
            transport,
        )
        .auto_middleware(false)
        .build()
        .expect("build vertex-maas registry")
}

#[test]
fn vertex_maas_package_settings_preserve_supported_provider_inputs() {
    let transport = CaptureTransport::default();
    let config = siumai::provider_ext::vertex_maas::GoogleVertexMaasProviderSettings::new()
        .with_project("test-project")
        .with_location("us-central1")
        .with_header("Authorization", "Bearer test-token")
        .with_header("x-test", "1")
        .with_fetch(Arc::new(transport.clone()))
        .into_config_for_model("deepseek-ai/deepseek-v3.2-maas")
        .expect("settings into config");

    assert_eq!(config.provider_id, "vertex-maas");
    assert_eq!(
        config.base_url,
        vertex_maas_base_url("test-project", "us-central1")
    );
    assert_eq!(config.common_params.model, "deepseek-ai/deepseek-v3.2-maas");
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
async fn vertex_maas_public_builder_exposes_chat_completion_embedding_capabilities() {
    let transport = CaptureTransport::default();

    let client = Provider::vertex_maas()
        .project("test-project")
        .location("us-central1")
        .model("deepseek-ai/deepseek-v3.2-maas")
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(transport.clone()))
        .build()
        .await
        .expect("build vertex-maas unified client");

    assert_eq!(client.provider_id().as_ref(), "vertex-maas");
    assert!(client.as_chat_capability().is_some());
    assert!(client.as_completion_capability().is_some());
    assert!(client.as_embedding_capability().is_some());
    assert!(client.as_image_generation_capability().is_none());
    assert!(client.as_speech_capability().is_none());
    assert!(client.as_rerank_capability().is_none());
    assert!(transport.take().is_none());
}

#[tokio::test]
async fn vertex_maas_siumai_provider_registry_chat_request_are_equivalent() {
    let model = "deepseek-ai/deepseek-v3.2-maas";
    let base_url = vertex_maas_base_url("test-project", "us-central1");
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .vertex_maas()
        .project("test-project")
        .location("us-central1")
        .model(model)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex_maas()
        .project("test-project")
        .location("us-central1")
        .model(model)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let registry = vertex_maas_registry(&base_url, Arc::new(registry_transport.clone()));

    let registry_model = registry
        .language_model(&format!("vertex-maas:{model}"))
        .expect("build registry vertex-maas language model");

    let request = make_chat_request_with_model(model);

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = ChatCapability::chat_request(&registry_model, request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-token"
    );
    assert_eq!(siumai_req.url, format!("{base_url}/chat/completions"));
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
}

#[tokio::test]
async fn vertex_maas_camel_case_provider_options_prefer_camel_passthrough_and_metadata_root() {
    let model = "deepseek-ai/deepseek-v3.2-maas";
    let base_url = vertex_maas_base_url("test-project", "us-central1");
    let response_json = serde_json::json!({
        "id": "chatcmpl-vertex-maas",
        "object": "chat.completion",
        "created": 1_718_345_013u64,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "hello from vertex maas"
            },
            "finish_reason": "stop"
        }]
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .vertex_maas()
        .project("test-project")
        .location("us-central1")
        .model(model)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex_maas()
        .project("test-project")
        .location("us-central1")
        .model(model)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let registry = vertex_maas_registry(&base_url, Arc::new(registry_transport.clone()));

    let registry_model = registry
        .language_model(&format!("vertex-maas:{model}"))
        .expect("build registry vertex-maas language model");

    let mut request = make_chat_request_with_model(model);
    request.provider_options_map.insert(
        "vertex-maas",
        serde_json::json!({ "customPassthrough": "raw-value" }),
    );
    request.provider_options_map.insert(
        "vertexMaas",
        serde_json::json!({ "customPassthrough": "camel-value" }),
    );

    let siumai_resp = siumai_client
        .chat_request(request.clone())
        .await
        .expect("siumai response ok");
    let provider_resp = provider_client
        .chat_request(request.clone())
        .await
        .expect("provider response ok");
    let registry_resp = ChatCapability::chat_request(&registry_model, request)
        .await
        .expect("registry response ok");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_eq!(
        siumai_req.body["customPassthrough"],
        serde_json::json!("camel-value")
    );
    assert_eq!(
        provider_req.body["customPassthrough"],
        serde_json::json!("camel-value")
    );
    assert_eq!(
        registry_req.body["customPassthrough"],
        serde_json::json!("camel-value")
    );

    for root in [
        siumai_resp
            .provider_metadata
            .as_ref()
            .expect("siumai provider metadata"),
        provider_resp
            .provider_metadata
            .as_ref()
            .expect("provider provider metadata"),
        registry_resp
            .provider_metadata
            .as_ref()
            .expect("registry provider metadata"),
    ] {
        assert!(root.contains_key("vertexMaas"));
        assert!(!root.contains_key("vertex-maas"));
    }
}

#[tokio::test]
async fn vertex_maas_stream_end_metadata_uses_camel_case_provider_key() {
    use futures_util::StreamExt;

    let model = "deepseek-ai/deepseek-v3.2-maas";
    let base_url = vertex_maas_base_url("test-project", "us-central1");
    let stream_body = br#"data: {"id":"chatcmpl-vertex-maas-stream","object":"chat.completion.chunk","created":1718345013,"model":"deepseek-ai/deepseek-v3.2-maas","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":"stop"}]}

data: [DONE]

"#
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .vertex_maas()
        .project("test-project")
        .location("us-central1")
        .model(model)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex_maas()
        .project("test-project")
        .location("us-central1")
        .model(model)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let registry = vertex_maas_registry(&base_url, Arc::new(registry_transport.clone()));

    let registry_model = registry
        .language_model(&format!("vertex-maas:{model}"))
        .expect("build registry vertex-maas language model");

    let mut request = make_chat_request_with_model(model);
    request.provider_options_map.insert(
        "vertexMaas",
        serde_json::json!({ "customPassthrough": "camel-value" }),
    );

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

    let mut siumai_end = None;
    let mut provider_end = None;
    let mut registry_end = None;

    while let Some(event) = siumai_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            siumai_end = Some(response);
            break;
        }
    }
    while let Some(event) = provider_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            provider_end = Some(response);
            break;
        }
    }
    while let Some(event) = registry_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            registry_end = Some(response);
            break;
        }
    }

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_eq!(
        siumai_req.body["customPassthrough"],
        serde_json::json!("camel-value")
    );
    assert_eq!(
        provider_req.body["customPassthrough"],
        serde_json::json!("camel-value")
    );
    assert_eq!(
        registry_req.body["customPassthrough"],
        serde_json::json!("camel-value")
    );

    for root in [
        siumai_end
            .expect("siumai stream end")
            .provider_metadata
            .expect("siumai stream provider metadata"),
        provider_end
            .expect("provider stream end")
            .provider_metadata
            .expect("provider stream provider metadata"),
        registry_end
            .expect("registry stream end")
            .provider_metadata
            .expect("registry stream provider metadata"),
    ] {
        assert!(root.contains_key("vertexMaas"));
        assert!(!root.contains_key("vertex-maas"));
    }
}

#[tokio::test]
async fn vertex_maas_completion_siumai_provider_registry_request_are_equivalent() {
    let model = "deepseek-ai/deepseek-v3.2-maas";
    let base_url = vertex_maas_base_url("test-project", "us-central1");
    let response_json = serde_json::json!({
        "id": "cmpl-vertex-maas-test",
        "object": "text_completion",
        "created": 1_718_345_013u64,
        "model": model,
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

    let siumai_client = Siumai::builder()
        .vertex_maas()
        .project("test-project")
        .location("us-central1")
        .model(model)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex_maas()
        .project("test-project")
        .location("us-central1")
        .model(model)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let registry = vertex_maas_registry(&base_url, Arc::new(registry_transport.clone()));

    let registry_model = registry
        .completion_model(&format!("vertex-maas:{model}"))
        .expect("build registry vertex-maas completion model");

    let request = CompletionRequest::from_prompt(vec![
        ChatMessage::system("Be terse.").build(),
        ChatMessage::user("Hello").build(),
        ChatMessage::assistant("Hi").build(),
        ChatMessage::user("Continue").build(),
    ])
    .with_model(model)
    .with_provider_option("vertex-maas", serde_json::json!({ "suffix": "!" }));

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
        "Bearer test-token"
    );
    assert_eq!(siumai_req.url, format!("{base_url}/completions"));
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
async fn vertex_maas_completion_stream_public_paths_keep_raw_chunks_runtime_only() {
    use futures_util::StreamExt;

    let model = "deepseek-ai/deepseek-v3.2-maas";
    let base_url = vertex_maas_base_url("test-project", "us-central1");
    let stream_body = concat!(
            "data: {\"id\":\"cmpl-vertex-maas-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"deepseek-ai/deepseek-v3.2-maas\",\"choices\":[{\"text\":\"hello\",\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"cmpl-vertex-maas-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"deepseek-ai/deepseek-v3.2-maas\",\"choices\":[{\"text\":\" world\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n",
            "data: [DONE]\n\n"
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .vertex_maas()
        .project("test-project")
        .location("us-central1")
        .model(model)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::vertex_maas()
        .project("test-project")
        .location("us-central1")
        .model(model)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let registry = vertex_maas_registry(&base_url, Arc::new(registry_transport.clone()));

    let registry_model = registry
        .completion_model(&format!("vertex-maas:{model}"))
        .expect("build registry vertex-maas completion model");

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
        "Bearer test-token"
    );
    assert_eq!(siumai_req.url, format!("{base_url}/completions"));
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert!(siumai_req.body.get("stream_options").is_none());
    assert!(siumai_req.body.get("includeRawChunks").is_none());
}

#[tokio::test]
async fn vertex_maas_builder_project_location_derive_openapi_base_url() {
    let model = "deepseek-ai/deepseek-v3.2-maas";
    let base_url = vertex_maas_base_url("test-project", "us-central1");
    let derived_transport = CaptureTransport::default();
    let explicit_transport = CaptureTransport::default();

    let derived_client = Siumai::builder()
        .vertex_maas()
        .project("test-project")
        .location("us-central1")
        .model(model)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(derived_transport.clone()))
        .build()
        .await
        .expect("build derived vertex-maas client");

    let explicit_client = Provider::vertex_maas()
        .base_url(base_url.clone())
        .model(model)
        .http_header("Authorization", "Bearer test-token")
        .fetch(Arc::new(explicit_transport.clone()))
        .build()
        .await
        .expect("build explicit base_url vertex-maas client");

    let request = make_chat_request_with_model(model);

    let _ = derived_client.chat_request(request.clone()).await;
    let _ = explicit_client.chat_request(request).await;

    let derived_req = derived_transport.take().expect("derived request");
    let explicit_req = explicit_transport.take().expect("explicit request");

    assert_requests_equivalent(&derived_req, &explicit_req);
    assert_eq!(derived_req.url, format!("{base_url}/chat/completions"));
}

#[tokio::test]
async fn vertex_maas_registry_non_text_handles_remain_intentionally_unsupported() {
    let model = "deepseek-ai/deepseek-v3.2-maas";
    let base_url = vertex_maas_base_url("test-project", "us-central1");
    let registry_transport = CaptureTransport::default();

    let registry = vertex_maas_registry(&base_url, Arc::new(registry_transport.clone()));

    let image_err = match registry.image_model(&format!("vertex-maas:{model}")) {
        Ok(_) => panic!("build vertex-maas registry image model should be unsupported"),
        Err(err) => err,
    };
    let rerank_err = match registry.reranking_model(&format!("vertex-maas:{model}")) {
        Ok(_) => panic!("build vertex-maas registry rerank handle should be unsupported"),
        Err(err) => err,
    };
    let speech_err = match registry.speech_model(&format!("vertex-maas:{model}")) {
        Ok(_) => panic!("build vertex-maas registry speech model should be unsupported"),
        Err(err) => err,
    };
    let transcription_err = match registry.transcription_model(&format!("vertex-maas:{model}")) {
        Ok(_) => {
            panic!("build vertex-maas registry transcription model should be unsupported")
        }
        Err(err) => err,
    };

    assert_unsupported_operation(&image_err);
    assert_unsupported_operation(&rerank_err);
    assert_unsupported_operation(&speech_err);
    assert_unsupported_operation(&transcription_err);
    assert_capture_transports_unused(&[&registry_transport]);
}
