use super::*;
use reqwest::header::AUTHORIZATION;
use siumai::experimental::client::LlmClient;
use siumai::experimental::execution::http::transport::HttpTransportMultipartRequest;
use siumai::extensions::types::{ImageEditInput, ImageEditRequest};

fn make_deepinfra_image_edit_request(model: &str) -> ImageEditRequest {
    let mut extra_params = std::collections::HashMap::new();
    extra_params.insert("strength".to_string(), serde_json::json!(0.35));

    let mut provider_options_map = siumai::prelude::unified::ProviderOptionsMap::default();
    provider_options_map.insert(
        "deepinfra",
        serde_json::json!({
            "guidance_scale": 6.5
        }),
    );

    ImageEditRequest {
        images: vec![ImageEditInput::file(b"image-one".to_vec())],
        mask: Some(ImageEditInput::file(b"mask-one".to_vec())),
        prompt: "replace the background with a neon skyline".to_string(),
        model: Some(model.to_string()),
        count: Some(1),
        size: Some("1024x1024".to_string()),
        aspect_ratio: Some("1:1".to_string()),
        seed: Some(7),
        response_format: Some("b64_json".to_string()),
        extra_params,
        provider_options_map,
        http_config: None,
    }
}

fn normalize_multipart_body(req: &HttpTransportMultipartRequest) -> String {
    let boundary = req
        .headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split("boundary=").nth(1))
        .expect("multipart boundary");
    String::from_utf8_lossy(&req.body).replace(boundary, "<boundary>")
}

fn deepinfra_registry_builder() -> siumai::registry::builder::RegistryBuilder {
    built_in_registry_builder("deepinfra", "deepinfra")
}

fn make_deepinfra_registry(
    api_key: &str,
    base_url: &str,
    transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    deepinfra_registry_builder()
        .with_api_key(api_key)
        .with_base_url(base_url)
        .fetch(transport)
        .auto_middleware(false)
        .build()
        .expect("build deepinfra registry")
}

#[test]
fn deepinfra_package_settings_preserve_supported_provider_inputs() {
    let config = siumai::provider_ext::deepinfra::DeepInfraProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/deepinfra")
        .with_header("x-test", "1")
        .into_config_for_model("meta-llama/Llama-3.3-70B-Instruct")
        .expect("settings into config");

    assert_eq!(config.provider_id, "deepinfra");
    assert_eq!(config.base_url, "https://example.com/deepinfra/openai");
    assert_eq!(
        config.common_params.model,
        "meta-llama/Llama-3.3-70B-Instruct"
    );
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
}
#[tokio::test]
async fn deepinfra_public_builder_exposes_unified_capabilities() {
    let transport = CaptureTransport::default();

    let client = Provider::deepinfra()
        .api_key("test-key")
        .base_url("https://example.com/deepinfra/v1")
        .fetch(Arc::new(transport.clone()))
        .build()
        .await
        .expect("build deepinfra unified client");

    assert_eq!(client.provider_id().as_ref(), "deepinfra");
    assert!(client.as_chat_capability().is_some());
    assert!(client.as_completion_capability().is_some());
    assert!(client.as_embedding_capability().is_some());
    assert!(client.as_image_generation_capability().is_some());
    assert!(client.as_image_extras().is_some());
    assert!(client.as_speech_capability().is_none());
    assert!(client.as_rerank_capability().is_none());
    assert!(transport.take().is_none());
}

#[tokio::test]
async fn deepinfra_siumai_provider_registry_chat_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .deepinfra()
        .api_key("test-key")
        .base_url("https://example.com/deepinfra/v1")
        .model("meta-llama/Llama-3.3-70B-Instruct")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepinfra()
        .api_key("test-key")
        .base_url("https://example.com/deepinfra/v1")
        .model("meta-llama/Llama-3.3-70B-Instruct")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let registry = make_deepinfra_registry(
        "test-key",
        "https://example.com/deepinfra/v1",
        Arc::new(registry_transport.clone()),
    );

    let registry_model = registry
        .language_model("deepinfra:meta-llama/Llama-3.3-70B-Instruct")
        .expect("build registry deepinfra language model");

    let request = make_chat_request_with_model("meta-llama/Llama-3.3-70B-Instruct");

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
        "Bearer test-key"
    );
    assert_eq!(
        siumai_req.url,
        "https://example.com/deepinfra/v1/openai/chat/completions"
    );
    assert_eq!(
        siumai_req.body["model"],
        serde_json::json!("meta-llama/Llama-3.3-70B-Instruct")
    );
}

#[tokio::test]
async fn deepinfra_image_generation_routes_to_provider_owned_inference_path() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .deepinfra()
        .api_key("test-key")
        .base_url("https://example.com/deepinfra/v1")
        .model("meta-llama/Llama-3.3-70B-Instruct")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepinfra()
        .api_key("test-key")
        .base_url("https://example.com/deepinfra/v1")
        .model("meta-llama/Llama-3.3-70B-Instruct")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let registry = make_deepinfra_registry(
        "test-key",
        "https://example.com/deepinfra/v1",
        Arc::new(registry_transport.clone()),
    );
    let registry_model = registry
        .image_model("deepinfra:black-forest-labs/FLUX-1-schnell")
        .expect("build registry deepinfra image model");

    let request = make_image_request_with_model("black-forest-labs/FLUX-1-schnell");

    let _ = siumai_client.generate_images(request.clone()).await;
    let _ = provider_client.generate_images(request.clone()).await;
    let _ = registry_model.generate_images(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(
        siumai_req.url,
        "https://example.com/deepinfra/v1/inference/black-forest-labs/FLUX-1-schnell"
    );
    assert_eq!(
        siumai_req.body["prompt"],
        serde_json::json!("a tiny purple robot")
    );
    assert_eq!(siumai_req.body["num_images"], serde_json::json!(1));
    assert_eq!(siumai_req.body["width"], serde_json::json!("1024"));
    assert_eq!(siumai_req.body["height"], serde_json::json!("1024"));
    assert!(siumai_req.body.get("negative_prompt").is_none());
    assert!(siumai_req.body.get("response_format").is_none());
}

#[tokio::test]
async fn deepinfra_image_edit_routes_to_provider_owned_openai_edit_path() {
    let siumai_transport = MixedCaptureTransport::default();
    let provider_transport = MixedCaptureTransport::default();
    let registry_transport = MixedCaptureTransport::default();

    let siumai_client = Siumai::builder()
        .deepinfra()
        .api_key("test-key")
        .base_url("https://example.com/deepinfra/v1")
        .model("meta-llama/Llama-3.3-70B-Instruct")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepinfra()
        .api_key("test-key")
        .base_url("https://example.com/deepinfra/v1")
        .model("meta-llama/Llama-3.3-70B-Instruct")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let registry = make_deepinfra_registry(
        "test-key",
        "https://example.com/deepinfra/v1",
        Arc::new(registry_transport.clone()),
    );
    let registry_model = registry
        .image_model("deepinfra:black-forest-labs/FLUX-1-schnell")
        .expect("build registry deepinfra image model");

    let request = make_deepinfra_image_edit_request("black-forest-labs/FLUX-1-schnell");

    let _ = siumai_client.edit_image(request.clone()).await;
    let _ = provider_client.edit_image(request.clone()).await;
    let _ = registry_model.edit_image(request).await;

    let siumai_req = siumai_transport.take_multipart().expect("siumai multipart");
    let provider_req = provider_transport
        .take_multipart()
        .expect("provider multipart");
    let registry_req = registry_transport
        .take_multipart()
        .expect("registry multipart");

    assert_eq!(siumai_req.url, provider_req.url);
    assert_eq!(siumai_req.url, registry_req.url);
    assert_eq!(
        siumai_req.headers.get(AUTHORIZATION).unwrap(),
        "Bearer test-key"
    );
    assert_eq!(
        siumai_req.url,
        "https://example.com/deepinfra/v1/openai/images/edits"
    );
    assert!(
        siumai_req
            .headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
    );

    let body_text = normalize_multipart_body(&siumai_req);
    let provider_body_text = normalize_multipart_body(&provider_req);
    let registry_body_text = normalize_multipart_body(&registry_req);

    assert_eq!(body_text, provider_body_text);
    assert_eq!(body_text, registry_body_text);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("black-forest-labs/FLUX-1-schnell"));
    assert!(body_text.contains("name=\"prompt\""));
    assert!(body_text.contains("replace the background with a neon skyline"));
    assert!(body_text.contains("name=\"image\"; filename=\"image-0.png\""));
    assert!(body_text.contains("name=\"mask\"; filename=\"mask-0.png\""));
    assert!(body_text.contains("name=\"strength\""));
    assert!(body_text.contains("0.35"));
    assert!(body_text.contains("name=\"guidance_scale\""));
    assert!(body_text.contains("6.5"));
}

#[tokio::test]
async fn deepinfra_completion_siumai_provider_registry_request_are_equivalent() {
    let response_json = serde_json::json!({
        "id": "cmpl-deepinfra-test",
        "object": "text_completion",
        "created": 1_718_345_013u64,
        "model": "meta-llama/Llama-3.3-70B-Instruct",
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

    let model = "meta-llama/Llama-3.3-70B-Instruct";
    let base_url = "https://example.com/deepinfra/v1";

    let siumai_client = Siumai::builder()
        .deepinfra()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepinfra()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let registry =
        make_deepinfra_registry("test-key", base_url, Arc::new(registry_transport.clone()));

    let registry_model = registry
        .completion_model(&format!("deepinfra:{model}"))
        .expect("build registry deepinfra completion model");

    let request = CompletionRequest::from_prompt(vec![
        ChatMessage::system("Be terse.").build(),
        ChatMessage::user("Hello").build(),
        ChatMessage::assistant("Hi").build(),
        ChatMessage::user("Continue").build(),
    ])
    .with_model(model)
    .with_provider_option("deepinfra", serde_json::json!({ "suffix": "!" }));

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
    assert_eq!(
        siumai_req.url,
        "https://example.com/deepinfra/v1/openai/completions"
    );
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
async fn deepinfra_completion_stream_public_paths_keep_raw_chunks_runtime_only() {
    use futures_util::StreamExt;

    let model = "meta-llama/Llama-3.3-70B-Instruct";
    let stream_body = concat!(
            "data: {\"id\":\"cmpl-deepinfra-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"meta-llama/Llama-3.3-70B-Instruct\",\"choices\":[{\"text\":\"hello\",\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"cmpl-deepinfra-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"meta-llama/Llama-3.3-70B-Instruct\",\"choices\":[{\"text\":\" world\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n",
            "data: [DONE]\n\n"
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let base_url = "https://example.com/deepinfra/v1";

    let siumai_client = Siumai::builder()
        .deepinfra()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::deepinfra()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let registry =
        make_deepinfra_registry("test-key", base_url, Arc::new(registry_transport.clone()));

    let registry_model = registry
        .completion_model(&format!("deepinfra:{model}"))
        .expect("build registry deepinfra completion model");

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
    assert_eq!(
        siumai_req.url,
        "https://example.com/deepinfra/v1/openai/completions"
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert!(siumai_req.body.get("stream_options").is_none());
    assert!(siumai_req.body.get("includeRawChunks").is_none());
}
