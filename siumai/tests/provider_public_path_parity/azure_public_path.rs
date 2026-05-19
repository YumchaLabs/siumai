use super::*;
use siumai::prelude::unified::{ResponseFormat, Tool, ToolChoice};
use siumai::provider_ext::azure::{AzureOpenAIProviderSettings, AzureUrlConfig};
use siumai::registry::builder::RegistryBuilder;

fn azure_responses_text_stream_body() -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
        .join("text")
        .join("openai-text-deltas.1.chunks.txt");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read azure responses text fixture failed: {path:?}: {err}"));

    let mut sse = String::new();
    for line in raw.lines().filter(|line| !line.trim().is_empty()) {
        sse.push_str("data: ");
        sse.push_str(line);
        sse.push_str("\n\n");
    }
    sse.push_str("data: [DONE]\n\n");
    sse.into_bytes()
}

fn azure_responses_response_body() -> serde_json::Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses")
        .join("response")
        .join("basic-text")
        .join("response.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read azure responses fixture failed: {path:?}: {err}"));
    serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("parse azure responses fixture failed: {path:?}: {err}"))
}

fn make_registry(
    transport: Arc<dyn HttpTransport>,
    base_url: &str,
) -> siumai::registry::ProviderRegistryHandle {
    make_registry_with_providers(transport, base_url, azure_registry_providers())
}

fn azure_registry_providers() -> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
    built_in_registry_providers("azure", "azure")
}

fn azure_registry_providers_with_options(
    url_config: AzureUrlConfig,
    provider_metadata_key: &'static str,
) -> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
    let mut providers = HashMap::new();
    providers.insert(
        "azure".to_string(),
        siumai::registry::azure_provider_factory_with_options(
            "azure",
            url_config,
            provider_metadata_key,
        )
        .expect("build azure registry factory with options"),
    );
    providers
}

fn azure_deployment_url_config(api_version: Option<&str>) -> AzureUrlConfig {
    let mut url_config = AzureUrlConfig::default();
    if let Some(api_version) = api_version {
        url_config.api_version = api_version.to_string();
    }
    url_config.use_deployment_based_urls = true;
    url_config
}

fn make_registry_with_providers(
    transport: Arc<dyn HttpTransport>,
    base_url: &str,
    providers: HashMap<String, Arc<dyn siumai::registry::ProviderFactory>>,
) -> siumai::registry::ProviderRegistryHandle {
    RegistryBuilder::new(providers)
        .with_provider_api_key_base_url_fetch("azure", "test-key", base_url, transport)
        .build()
        .expect("build azure registry")
}

#[test]
fn azure_package_settings_preserve_supported_provider_inputs() {
    let config = AzureOpenAIProviderSettings::new()
        .with_api_key("test-key")
        .with_resource_name("demo-resource")
        .with_header("x-test", "1")
        .with_api_version("2024-10-21")
        .with_use_deployment_based_urls(true)
        .into_config_for_model("deployment-id")
        .expect("settings into config");

    assert_eq!(
        config.base_url,
        "https://demo-resource.openai.azure.com/openai"
    );
    assert_eq!(config.common_params.model, "deployment-id");
    assert_eq!(config.url_config.api_version, "2024-10-21");
    assert!(config.url_config.use_deployment_based_urls);
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
}

fn custom_events_by_type(
    events: &[siumai::prelude::unified::ChatStreamEvent],
    ty: &str,
) -> Vec<serde_json::Value> {
    let stable_parts: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::Part { part }
            | siumai::prelude::unified::ChatStreamEvent::PartWithReplay { part, .. } => {
                serde_json::to_value(part).ok()
            }
            _ => None,
        })
        .filter(|data| data.get("type") == Some(&serde_json::Value::String(ty.to_string())))
        .collect();
    if !stable_parts.is_empty() {
        return stable_parts;
    }

    events
        .iter()
        .filter_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::Custom { data, .. } => Some(data.clone()),
            _ => None,
        })
        .filter(|data| data.get("type") == Some(&serde_json::Value::String(ty.to_string())))
        .collect()
}

async fn collect_stream_events(
    stream: &mut siumai::prelude::unified::ChatStream,
) -> Vec<siumai::prelude::unified::ChatStreamEvent> {
    use futures_util::StreamExt;

    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(event) => events.push(event),
            Err(err) => panic!("collect azure public-path stream event failed: {err:?}"),
        }
    }
    events
}

#[tokio::test]
async fn azure_siumai_provider_config_embedding_request_are_equivalent() {
    let response = serde_json::json!({
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2],
                "index": 0
            }
        ],
        "model": "embedding-deployment",
        "usage": {
            "prompt_tokens": 1,
            "total_tokens": 1
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response.clone());
    let provider_transport = JsonSuccessTransport::new(response.clone());
    let config_transport = JsonSuccessTransport::new(response.clone());
    let registry_transport = JsonSuccessTransport::new(response);

    let model = "embedding-deployment";
    let base_url = "https://example.invalid/openai";

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_embedding_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .embedding_model("azure:embedding-deployment")
        .expect("build registry embedding model");

    let siumai_resp = siumai_client
        .embed(vec!["hello azure embedding".to_string()])
        .await
        .expect("siumai embedding ok");
    let provider_resp = provider_client
        .embed(vec!["hello azure embedding".to_string()])
        .await
        .expect("provider embedding ok");
    let config_resp = EmbeddingModel::embed(
        &config_client,
        EmbeddingRequest::single("hello azure embedding"),
    )
    .await
    .expect("config embedding ok");
    let registry_resp = EmbeddingModel::embed(
        &registry_model,
        EmbeddingRequest::single("hello azure embedding"),
    )
    .await
    .expect("registry embedding ok");

    assert_eq!(siumai_resp.embeddings.len(), 1);
    assert_eq!(provider_resp.embeddings.len(), 1);
    assert_eq!(config_resp.embeddings.len(), 1);
    assert_eq!(registry_resp.embeddings.len(), 1);
    assert_eq!(siumai_resp.embeddings[0].len(), 2);
    assert_eq!(provider_resp.embeddings[0].len(), 2);
    assert_eq!(config_resp.embeddings[0].len(), 2);
    assert_eq!(registry_resp.embeddings[0].len(), 2);

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.invalid/openai/v1/embeddings?api-version=v1"
    );
    assert_eq!(
        header_value(&siumai_req, "api-key"),
        Some("test-key".to_string())
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!(["hello azure embedding"])
    );
    assert_eq!(
        siumai_req.body["encoding_format"],
        serde_json::json!("float")
    );
}

#[tokio::test]
async fn azure_siumai_provider_config_image_generation_request_are_equivalent() {
    let response = serde_json::json!({
        "created": 123,
        "data": [
            {
                "url": "https://example.com/generated.png",
                "revised_prompt": "a tiny purple robot"
            }
        ]
    });

    let siumai_transport = JsonSuccessTransport::new(response.clone());
    let provider_transport = JsonSuccessTransport::new(response.clone());
    let config_transport = JsonSuccessTransport::new(response.clone());
    let registry_transport = JsonSuccessTransport::new(response);

    let model = "image-deployment";
    let base_url = "https://example.invalid/openai";

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_image_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .image_model("azure:image-deployment")
        .expect("build registry image model");

    let request = make_image_request_with_model(model);

    let siumai_resp = siumai_client
        .generate_images(request.clone())
        .await
        .expect("siumai image ok");
    let provider_resp = provider_client
        .generate_images(request.clone())
        .await
        .expect("provider image ok");
    let config_resp = config_client
        .generate_images(request.clone())
        .await
        .expect("config image ok");
    let registry_resp = registry_model
        .generate_images(request)
        .await
        .expect("registry image ok");

    assert_eq!(
        siumai_resp.images[0].url.as_deref(),
        Some("https://example.com/generated.png")
    );
    assert_eq!(
        provider_resp.images[0].url.as_deref(),
        Some("https://example.com/generated.png")
    );
    assert_eq!(
        config_resp.images[0].url.as_deref(),
        Some("https://example.com/generated.png")
    );
    assert_eq!(
        registry_resp.images[0].url.as_deref(),
        Some("https://example.com/generated.png")
    );

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.invalid/openai/v1/images/generations?api-version=v1"
    );
    assert_eq!(
        header_value(&siumai_req, "api-key"),
        Some("test-key".to_string())
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["prompt"],
        serde_json::json!("a tiny purple robot")
    );
    assert_eq!(siumai_req.body["size"], serde_json::json!("1024x1024"));
    assert_eq!(siumai_req.body["n"], serde_json::json!(1));
    assert_eq!(siumai_req.body["response_format"], serde_json::json!("url"));
}

#[tokio::test]
async fn azure_registry_image_handle_prefers_provider_specific_build_overrides() {
    let response = serde_json::json!({
        "created": 123,
        "data": [
            {
                "url": "https://example.com/generated.png",
                "revised_prompt": "a tiny purple robot"
            }
        ]
    });

    let global_transport = JsonSuccessTransport::new(response.clone());
    let azure_transport = JsonSuccessTransport::new(response);

    let registry = RegistryBuilder::new(azure_registry_providers())
        .with_api_key("global-key")
        .with_base_url("https://example.invalid/global-openai")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "azure",
            "ctx-key",
            "https://example.invalid/custom-openai",
            Arc::new(azure_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build azure registry");

    let handle = registry
        .image_model("azure:image-deployment")
        .expect("build azure image model");

    let generated = handle
        .generate_images(make_image_request_with_model("image-deployment"))
        .await
        .expect("generate images through registry handle");

    assert_eq!(
        generated.images[0].url.as_deref(),
        Some("https://example.com/generated.png")
    );
    assert!(global_transport.take().is_none());

    let req = azure_transport.take().expect("captured azure request");
    assert_eq!(header_value(&req, "api-key"), Some("ctx-key".to_string()));
    assert_eq!(
        req.url,
        "https://example.invalid/custom-openai/v1/images/generations?api-version=v1"
    );
    assert_eq!(req.body["model"], serde_json::json!("image-deployment"));
    assert_eq!(req.body["n"], serde_json::json!(1));
}

#[tokio::test]
async fn azure_siumai_provider_config_tts_request_are_equivalent() {
    let siumai_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let provider_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let config_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let registry_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");

    let base_url = "https://example.invalid/openai";
    let model = "tts-deployment";

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_speech_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .speech_model("azure:tts-deployment")
        .expect("build registry speech model");

    let request = TtsRequest::new("hello from azure".to_string())
        .with_voice("alloy".to_string())
        .with_format("mp3".to_string())
        .with_speed(1.1);

    let siumai_resp = siumai_client
        .text_to_speech(request.clone())
        .await
        .expect("siumai tts ok");
    let provider_resp = provider_client
        .text_to_speech(request.clone())
        .await
        .expect("provider tts ok");
    let config_resp = config_client
        .text_to_speech(request.clone())
        .await
        .expect("config tts ok");
    let registry_resp = siumai::speech::SpeechModel::synthesize(&registry_model, request)
        .await
        .expect("registry tts ok");

    assert_eq!(siumai_resp.audio_data, vec![1, 2, 3, 4]);
    assert_eq!(provider_resp.audio_data, vec![1, 2, 3, 4]);
    assert_eq!(config_resp.audio_data, vec![1, 2, 3, 4]);
    assert_eq!(registry_resp.audio_data, vec![1, 2, 3, 4]);

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.invalid/openai/v1/audio/speech?api-version=v1"
    );
    assert_eq!(
        header_value(&siumai_req, "api-key"),
        Some("test-key".to_string())
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!("hello from azure")
    );
    assert_eq!(siumai_req.body["voice"], serde_json::json!("alloy"));
    assert_eq!(siumai_req.body["response_format"], serde_json::json!("mp3"));
    let speed = siumai_req.body["speed"]
        .as_f64()
        .expect("speed should serialize as number");
    assert!((speed - 1.1).abs() < 1e-6);
}

#[tokio::test]
async fn azure_siumai_provider_config_stt_request_are_equivalent() {
    let siumai_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from azure",
        "language": "en"
    }));
    let provider_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from azure",
        "language": "en"
    }));
    let config_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from azure",
        "language": "en"
    }));
    let registry_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from azure",
        "language": "en"
    }));

    let base_url = "https://example.invalid/openai";
    let model = "stt-deployment";

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_transcription_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .transcription_model("azure:stt-deployment")
        .expect("build registry transcription model");

    let request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg")
        .with_provider_option("azure", serde_json::json!({ "language": "en" }))
        .with_media_type("audio/mpeg".to_string());

    let siumai_resp = siumai_client
        .speech_to_text(request.clone())
        .await
        .expect("siumai stt ok");
    let provider_resp = provider_client
        .speech_to_text(request.clone())
        .await
        .expect("provider stt ok");
    let config_resp = config_client
        .speech_to_text(request.clone())
        .await
        .expect("config stt ok");
    let registry_resp = registry_model
        .speech_to_text(request)
        .await
        .expect("registry stt ok");

    assert_eq!(siumai_resp.text, "hello from azure");
    assert_eq!(provider_resp.text, "hello from azure");
    assert_eq!(config_resp.text, "hello from azure");
    assert_eq!(registry_resp.text, "hello from azure");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_multipart_requests_equivalent(&siumai_req, &provider_req);
    assert_multipart_requests_equivalent(&siumai_req, &config_req);
    assert_multipart_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.invalid/openai/v1/audio/transcriptions?api-version=v1"
    );

    let body_text = normalize_multipart_body(&siumai_req);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("stt-deployment"));
    assert!(body_text.contains("name=\"response_format\""));
    assert!(body_text.contains("json"));
    assert!(body_text.contains("name=\"language\""));
    assert!(body_text.contains("en"));
    assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
    assert!(body_text.contains("Content-Type: audio/mpeg"));
    assert!(body_text.contains("abc"));
}

#[tokio::test]
async fn azure_registry_speech_handle_prefers_provider_specific_build_overrides() {
    let global_transport = BinaryCaptureTransport::new(vec![9, 9, 9], "audio/mpeg");
    let azure_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let model = "tts-deployment";

    let registry = RegistryBuilder::new(azure_registry_providers())
        .with_api_key("global-key")
        .with_base_url("https://example.invalid/global-openai")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "azure",
            "ctx-key",
            "https://example.invalid/custom-openai",
            Arc::new(azure_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build azure registry");

    let registry_model = registry
        .speech_model("azure:tts-deployment")
        .expect("build azure speech model");

    let response = siumai::speech::SpeechModel::synthesize(
        &registry_model,
        TtsRequest::new("hello from azure".to_string())
            .with_voice("alloy".to_string())
            .with_format("mp3".to_string()),
    )
    .await
    .expect("registry tts ok");

    assert_eq!(response.audio_data, vec![1, 2, 3, 4]);

    let req = azure_transport
        .take()
        .expect("captured azure speech request");
    assert!(global_transport.take().is_none());
    assert_eq!(header_value(&req, "api-key"), Some("ctx-key".to_string()));
    assert_eq!(
        req.url,
        "https://example.invalid/custom-openai/v1/audio/speech?api-version=v1"
    );
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(req.body["input"], serde_json::json!("hello from azure"));
    assert_eq!(req.body["voice"], serde_json::json!("alloy"));
    assert_eq!(req.body["response_format"], serde_json::json!("mp3"));
}

#[tokio::test]
async fn azure_registry_transcription_handle_prefers_provider_specific_build_overrides() {
    let global_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from global",
        "language": "en"
    }));
    let azure_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from azure",
        "language": "en"
    }));
    let model = "stt-deployment";

    let registry = RegistryBuilder::new(azure_registry_providers())
        .with_api_key("global-key")
        .with_base_url("https://example.invalid/global-openai")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "azure",
            "ctx-key",
            "https://example.invalid/custom-openai",
            Arc::new(azure_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build azure registry");

    let registry_model = registry
        .transcription_model("azure:stt-deployment")
        .expect("build azure transcription model");

    let request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");

    let response = registry_model
        .speech_to_text(request)
        .await
        .expect("registry stt ok");

    assert_eq!(response.text, "hello from azure");
    assert_eq!(response.language.as_deref(), Some("en"));

    let req = azure_transport
        .take()
        .expect("captured azure transcription request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        req.headers
            .get("api-key")
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string),
        Some("ctx-key".to_string())
    );
    assert_eq!(
        req.url,
        "https://example.invalid/custom-openai/v1/audio/transcriptions?api-version=v1"
    );

    let body_text = normalize_multipart_body(&req);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains(model));
    assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
    assert!(body_text.contains("Content-Type: audio/mpeg"));
    assert!(body_text.contains("abc"));
}

#[tokio::test]
async fn azure_deployment_based_siumai_provider_config_embedding_request_are_equivalent() {
    let response = serde_json::json!({
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2],
                "index": 0
            }
        ],
        "model": "embedding-deployment",
        "usage": {
            "prompt_tokens": 1,
            "total_tokens": 1
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response.clone());
    let provider_transport = JsonSuccessTransport::new(response.clone());
    let config_transport = JsonSuccessTransport::new(response.clone());
    let registry_transport = JsonSuccessTransport::new(response);

    let model = "embedding-deployment";
    let base_url = "https://example.invalid/openai";
    let api_version = "2024-10-21";
    let registry_providers = azure_registry_providers_with_options(
        azure_deployment_url_config(Some(api_version)),
        "azure",
    );

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .api_version(api_version)
        .deployment_based_urls(true)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .api_version(api_version)
        .deployment_based_urls(true)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_embedding_model(model)
            .with_api_version(api_version)
            .with_deployment_based_urls(true)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry_with_providers(
        Arc::new(registry_transport.clone()),
        base_url,
        registry_providers,
    );
    let registry_model = registry
        .embedding_model("azure:embedding-deployment")
        .expect("build registry embedding model");

    let siumai_resp = siumai_client
        .embed(vec!["hello azure embedding".to_string()])
        .await
        .expect("siumai embedding ok");
    let provider_resp = provider_client
        .embed(vec!["hello azure embedding".to_string()])
        .await
        .expect("provider embedding ok");
    let config_resp = EmbeddingModel::embed(
        &config_client,
        EmbeddingRequest::single("hello azure embedding"),
    )
    .await
    .expect("config embedding ok");
    let registry_resp = EmbeddingModel::embed(
        &registry_model,
        EmbeddingRequest::single("hello azure embedding"),
    )
    .await
    .expect("registry embedding ok");

    assert_eq!(siumai_resp.embeddings[0].len(), 2);
    assert_eq!(provider_resp.embeddings[0].len(), 2);
    assert_eq!(config_resp.embeddings[0].len(), 2);
    assert_eq!(registry_resp.embeddings[0].len(), 2);

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.invalid/openai/deployments/embedding-deployment/embeddings?api-version=2024-10-21"
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
}

#[tokio::test]
async fn azure_deployment_based_siumai_provider_config_image_generation_request_are_equivalent() {
    let response = serde_json::json!({
        "created": 123,
        "data": [
            {
                "url": "https://example.com/generated.png",
                "revised_prompt": "a tiny purple robot"
            }
        ]
    });

    let siumai_transport = JsonSuccessTransport::new(response.clone());
    let provider_transport = JsonSuccessTransport::new(response.clone());
    let config_transport = JsonSuccessTransport::new(response.clone());
    let registry_transport = JsonSuccessTransport::new(response);

    let model = "image-deployment";
    let base_url = "https://example.invalid/openai";
    let registry_providers =
        azure_registry_providers_with_options(azure_deployment_url_config(None), "azure");

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .deployment_based_urls(true)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .deployment_based_urls(true)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_image_model(model)
            .with_deployment_based_urls(true)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry_with_providers(
        Arc::new(registry_transport.clone()),
        base_url,
        registry_providers,
    );
    let registry_model = registry
        .image_model("azure:image-deployment")
        .expect("build registry image model");

    let request = make_image_request_with_model(model);

    let siumai_resp = siumai_client
        .generate_images(request.clone())
        .await
        .expect("siumai image ok");
    let provider_resp = provider_client
        .generate_images(request.clone())
        .await
        .expect("provider image ok");
    let config_resp = config_client
        .generate_images(request.clone())
        .await
        .expect("config image ok");
    let registry_resp = registry_model
        .generate_images(request)
        .await
        .expect("registry image ok");

    assert_eq!(
        siumai_resp.images[0].url.as_deref(),
        Some("https://example.com/generated.png")
    );
    assert_eq!(
        provider_resp.images[0].url.as_deref(),
        Some("https://example.com/generated.png")
    );
    assert_eq!(
        config_resp.images[0].url.as_deref(),
        Some("https://example.com/generated.png")
    );
    assert_eq!(
        registry_resp.images[0].url.as_deref(),
        Some("https://example.com/generated.png")
    );

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.invalid/openai/deployments/image-deployment/images/generations?api-version=v1"
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
}

#[tokio::test]
async fn azure_deployment_based_siumai_provider_config_tts_request_are_equivalent() {
    let siumai_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let provider_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let config_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let registry_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");

    let base_url = "https://example.invalid/openai";
    let model = "tts-deployment";
    let registry_providers =
        azure_registry_providers_with_options(azure_deployment_url_config(None), "azure");

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .deployment_based_urls(true)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .deployment_based_urls(true)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_speech_model(model)
            .with_deployment_based_urls(true)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry_with_providers(
        Arc::new(registry_transport.clone()),
        base_url,
        registry_providers,
    );
    let registry_model = registry
        .speech_model("azure:tts-deployment")
        .expect("build registry speech model");

    let request = TtsRequest::new("hello from azure".to_string())
        .with_voice("alloy".to_string())
        .with_format("mp3".to_string());

    let siumai_resp = siumai_client
        .text_to_speech(request.clone())
        .await
        .expect("siumai tts ok");
    let provider_resp = provider_client
        .text_to_speech(request.clone())
        .await
        .expect("provider tts ok");
    let config_resp = config_client
        .text_to_speech(request.clone())
        .await
        .expect("config tts ok");
    let registry_resp = siumai::speech::SpeechModel::synthesize(&registry_model, request)
        .await
        .expect("registry tts ok");

    assert_eq!(siumai_resp.audio_data, vec![1, 2, 3, 4]);
    assert_eq!(provider_resp.audio_data, vec![1, 2, 3, 4]);
    assert_eq!(config_resp.audio_data, vec![1, 2, 3, 4]);
    assert_eq!(registry_resp.audio_data, vec![1, 2, 3, 4]);

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.invalid/openai/deployments/tts-deployment/audio/speech?api-version=v1"
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
}

#[tokio::test]
async fn azure_deployment_based_siumai_provider_config_stt_request_are_equivalent() {
    let siumai_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from azure",
        "language": "en"
    }));
    let provider_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from azure",
        "language": "en"
    }));
    let config_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from azure",
        "language": "en"
    }));
    let registry_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from azure",
        "language": "en"
    }));

    let base_url = "https://example.invalid/openai";
    let model = "stt-deployment";
    let registry_providers =
        azure_registry_providers_with_options(azure_deployment_url_config(None), "azure");

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .deployment_based_urls(true)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .deployment_based_urls(true)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_transcription_model(model)
            .with_deployment_based_urls(true)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry_with_providers(
        Arc::new(registry_transport.clone()),
        base_url,
        registry_providers,
    );
    let registry_model = registry
        .transcription_model("azure:stt-deployment")
        .expect("build registry transcription model");

    let request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");

    let siumai_resp = siumai_client
        .speech_to_text(request.clone())
        .await
        .expect("siumai stt ok");
    let provider_resp = provider_client
        .speech_to_text(request.clone())
        .await
        .expect("provider stt ok");
    let config_resp = config_client
        .speech_to_text(request.clone())
        .await
        .expect("config stt ok");
    let registry_resp = registry_model
        .speech_to_text(request)
        .await
        .expect("registry stt ok");

    assert_eq!(siumai_resp.text, "hello from azure");
    assert_eq!(provider_resp.text, "hello from azure");
    assert_eq!(config_resp.text, "hello from azure");
    assert_eq!(registry_resp.text, "hello from azure");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_multipart_requests_equivalent(&siumai_req, &provider_req);
    assert_multipart_requests_equivalent(&siumai_req, &config_req);
    assert_multipart_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.invalid/openai/deployments/stt-deployment/audio/transcriptions?api-version=v1"
    );
}

#[tokio::test]
async fn azure_siumai_provider_config_chat_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "deployment-id";
    let base_url = "https://example.invalid/openai";

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://example.invalid/openai/v1/responses?api-version=v1"
    );
    assert_eq!(
        header_value(&siumai_req, "api-key"),
        Some("test-key".to_string())
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert!(siumai_req.body.get("input").is_some());
}

#[tokio::test]
async fn azure_siumai_provider_config_stream_custom_events_keep_raw_provider_metadata() {
    let stream_body = azure_responses_text_stream_body();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body);

    let model = "deployment-id";
    let base_url = "https://example.invalid/openai";

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request)
        .await
        .expect("config stream ok");

    let siumai_events = collect_stream_events(&mut siumai_stream).await;
    let provider_events = collect_stream_events(&mut provider_stream).await;
    let config_events = collect_stream_events(&mut config_stream).await;

    let siumai_text_starts = custom_events_by_type(&siumai_events, "text-start");
    let provider_text_starts = custom_events_by_type(&provider_events, "text-start");
    let config_text_starts = custom_events_by_type(&config_events, "text-start");

    assert_eq!(
        siumai_text_starts.len(),
        1,
        "expected one siumai text-start"
    );
    assert_eq!(
        provider_text_starts.len(),
        1,
        "expected one provider text-start"
    );
    assert_eq!(
        config_text_starts.len(),
        1,
        "expected one config text-start"
    );

    let siumai_finishes = custom_events_by_type(&siumai_events, "finish");
    let provider_finishes = custom_events_by_type(&provider_events, "finish");
    let config_finishes = custom_events_by_type(&config_events, "finish");

    assert_eq!(siumai_finishes.len(), 1, "expected one siumai finish");
    assert_eq!(provider_finishes.len(), 1, "expected one provider finish");
    assert_eq!(config_finishes.len(), 1, "expected one config finish");

    for event in [
        &siumai_text_starts[0],
        &provider_text_starts[0],
        &config_text_starts[0],
        &siumai_finishes[0],
        &provider_finishes[0],
        &config_finishes[0],
    ] {
        assert!(
            event
                .get("providerMetadata")
                .and_then(|meta| meta.get("azure"))
                .is_some(),
            "expected raw providerMetadata.azure"
        );
        assert!(
            event
                .get("providerMetadata")
                .and_then(|meta| meta.get("openai"))
                .is_none(),
            "did not expect providerMetadata.openai"
        );
    }

    let siumai_end = siumai_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected siumai StreamEnd");
    let provider_end = provider_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected provider StreamEnd");
    let config_end = config_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected config StreamEnd");

    assert_eq!(siumai_end.content_text(), Some("Hello, World!"));
    assert_eq!(provider_end.content_text(), Some("Hello, World!"));
    assert_eq!(config_end.content_text(), Some("Hello, World!"));

    for response in [siumai_end, provider_end, config_end] {
        let provider_metadata = response
            .provider_metadata
            .as_ref()
            .expect("expected stream-end provider metadata");
        assert!(
            provider_metadata.contains_key("azure"),
            "expected raw stream-end provider_metadata.azure"
        );
        assert!(
            !provider_metadata.contains_key("openai"),
            "did not expect stream-end provider_metadata.openai"
        );
    }

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://example.invalid/openai/v1/responses?api-version=v1"
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn azure_registry_stream_end_metadata_match_config_path() {
    let stream_body = azure_responses_text_stream_body();

    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "deployment-id";
    let base_url = "https://example.invalid/openai";

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("azure:deployment-id")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();

    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    let config_events = collect_stream_events(&mut config_stream).await;
    let registry_events = collect_stream_events(&mut registry_stream).await;

    for events in [&config_events, &registry_events] {
        let text_starts = custom_events_by_type(events, "text-start");
        assert_eq!(text_starts.len(), 1, "expected one text-start");
        let finishes = custom_events_by_type(events, "finish");
        assert_eq!(finishes.len(), 1, "expected one finish");

        for event in [&text_starts[0], &finishes[0]] {
            assert!(
                event
                    .get("providerMetadata")
                    .and_then(|meta| meta.get("azure"))
                    .is_some(),
                "expected raw providerMetadata.azure"
            );
            assert!(
                event
                    .get("providerMetadata")
                    .and_then(|meta| meta.get("openai"))
                    .is_none(),
                "did not expect providerMetadata.openai"
            );
        }
    }

    let config_end = config_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected config StreamEnd");
    let registry_end = registry_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected registry StreamEnd");

    assert_eq!(config_end.content_text(), Some("Hello, World!"));
    assert_eq!(registry_end.content_text(), Some("Hello, World!"));

    for response in [config_end, registry_end] {
        let provider_metadata = response
            .provider_metadata
            .as_ref()
            .expect("expected stream-end provider metadata");
        assert!(
            provider_metadata.contains_key("azure"),
            "expected raw stream-end provider_metadata.azure"
        );
        assert!(
            !provider_metadata.contains_key("openai"),
            "did not expect stream-end provider_metadata.openai"
        );
    }

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.url,
        "https://example.invalid/openai/v1/responses?api-version=v1"
    );
    assert_eq!(
        header_value(&registry_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn azure_registry_chat_stream_request_with_explicit_request_model_match_config_path() {
    let stream_body = azure_responses_text_stream_body();

    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let default_model = "deployment-id";
    let request_model = "override-deployment-id";
    let base_url = "https://example.invalid/openai";

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(default_model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("azure:deployment-id")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(request_model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();

    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    use futures_util::StreamExt;
    let _ = config_stream.next().await;
    let _ = registry_stream.next().await;

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.body["model"], serde_json::json!(request_model));
    assert_eq!(registry_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        registry_req.url,
        "https://example.invalid/openai/v1/responses?api-version=v1"
    );
}

#[tokio::test]
async fn azure_registry_response_metadata_match_config_path() {
    let response_json = azure_responses_response_body();

    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "deployment-id";
    let base_url = "https://example.invalid/openai";

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("azure:deployment-id")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();

    let config_resp = config_client
        .chat_request(request.clone())
        .await
        .expect("config response ok");
    let registry_resp = registry_model
        .chat_request(request)
        .await
        .expect("registry response ok");

    assert_eq!(config_resp.content_text(), Some("answer text"));
    assert_eq!(registry_resp.content_text(), Some("answer text"));

    for response in [&config_resp, &registry_resp] {
        let provider_metadata = response
            .provider_metadata
            .as_ref()
            .expect("expected response provider metadata");
        assert!(
            provider_metadata.contains_key("azure"),
            "expected response provider_metadata.azure"
        );
        assert!(
            !provider_metadata.contains_key("openai"),
            "did not expect response provider_metadata.openai"
        );
    }

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.url,
        "https://example.invalid/openai/v1/responses?api-version=v1"
    );
}

#[tokio::test]
async fn azure_provider_metadata_key_openai_match_across_siumai_provider_config_registry() {
    let response_json = azure_responses_response_body();

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "deployment-id";
    let base_url = "https://example.invalid/openai";
    let metadata_key = "openai";
    let registry_providers =
        azure_registry_providers_with_options(AzureUrlConfig::default(), metadata_key);

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .provider_metadata_key(metadata_key)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .provider_metadata_key(metadata_key)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_provider_metadata_key(metadata_key)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry_with_providers(
        Arc::new(registry_transport.clone()),
        base_url,
        registry_providers,
    );
    let registry_model = registry
        .language_model("azure:deployment-id")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();

    let siumai_resp = siumai_client
        .chat_request(request.clone())
        .await
        .expect("siumai response ok");
    let provider_resp = provider_client
        .chat_request(request.clone())
        .await
        .expect("provider response ok");
    let config_resp = config_client
        .chat_request(request.clone())
        .await
        .expect("config response ok");
    let registry_resp = registry_model
        .chat_request(request)
        .await
        .expect("registry response ok");

    for response in [&siumai_resp, &provider_resp, &config_resp, &registry_resp] {
        let provider_metadata = response
            .provider_metadata
            .as_ref()
            .expect("expected response provider metadata");
        assert!(
            provider_metadata.contains_key("openai"),
            "expected response provider_metadata.openai"
        );
        assert!(
            !provider_metadata.contains_key("azure"),
            "did not expect response provider_metadata.azure"
        );
    }

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.invalid/openai/v1/responses?api-version=v1"
    );
}

#[tokio::test]
async fn azure_registry_chat_request_with_explicit_request_model_match_config_path() {
    let response_json = azure_responses_response_body();

    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let default_model = "deployment-id";
    let request_model = "override-deployment-id";
    let base_url = "https://example.invalid/openai";

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(default_model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("azure:deployment-id")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(request_model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.body["model"], serde_json::json!(request_model));
    assert_eq!(
        registry_req.url,
        "https://example.invalid/openai/v1/responses?api-version=v1"
    );
}

#[tokio::test]
async fn azure_chat_siumai_provider_config_chat_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "deployment-id";
    let base_url = "https://example.invalid/openai";

    let siumai_client = Siumai::builder()
        .azure_chat()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure_chat()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_chat_mode(siumai::provider_ext::azure::AzureChatMode::ChatCompletions)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://example.invalid/openai/v1/chat/completions?api-version=v1"
    );
    assert_eq!(
        header_value(&siumai_req, "api-key"),
        Some("test-key".to_string())
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["messages"][0]["content"],
        serde_json::json!("hi")
    );
}

#[tokio::test]
async fn azure_siumai_provider_config_stable_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "o3-mini";
    let base_url = "https://example.invalid/openai";
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "get_weather".to_string(),
            "Get weather".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
        )])
        .tool_choice(ToolChoice::None)
        .response_format(ResponseFormat::json_schema(schema.clone()))
        .build()
        .with_provider_option(
            "openai",
            serde_json::json!({
                "reasoningEffort": "low",
                "responsesApi": {
                    "reasoningSummary": "detailed"
                }
            }),
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
        siumai_req.url,
        "https://example.invalid/openai/v1/responses?api-version=v1"
    );
    assert_eq!(siumai_req.body["tool_choice"], serde_json::json!("none"));
    assert_eq!(
        siumai_req.body["reasoning"]["effort"],
        serde_json::json!("low")
    );
    assert_eq!(
        siumai_req.body["reasoning"]["summary"],
        serde_json::json!("detailed")
    );
    assert_eq!(
        siumai_req.body["text"]["format"],
        serde_json::json!({
            "type": "json_schema",
            "name": "response",
            "schema": schema,
            "strict": true
        })
    );
}

#[tokio::test]
async fn azure_chat_siumai_provider_config_stable_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "o3-mini";
    let base_url = "https://example.invalid/openai";
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let siumai_client = Siumai::builder()
        .azure_chat()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure_chat()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_chat_mode(siumai::provider_ext::azure::AzureChatMode::ChatCompletions)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "get_weather".to_string(),
            "Get weather".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
        )])
        .tool_choice(ToolChoice::None)
        .response_format(ResponseFormat::json_schema(schema.clone()))
        .build()
        .with_provider_option(
            "openai",
            serde_json::json!({
                "reasoningEffort": "low"
            }),
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
        siumai_req.url,
        "https://example.invalid/openai/v1/chat/completions?api-version=v1"
    );
    assert_eq!(siumai_req.body["tool_choice"], serde_json::json!("none"));
    assert_eq!(
        siumai_req.body["reasoning_effort"],
        serde_json::json!("low")
    );
    assert_eq!(
        siumai_req.body["response_format"],
        serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": true
            }
        })
    );
}

#[tokio::test]
async fn azure_completion_siumai_provider_config_registry_request_are_equivalent() {
    let response_json = serde_json::json!({
        "id": "cmpl-azure-test",
        "object": "text_completion",
        "created": 1_718_345_013u64,
        "model": "deployment-id",
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
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "deployment-id";
    let base_url = "https://example.invalid/openai";

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .completion_model("azure:deployment-id")
        .expect("build registry completion model");

    let request = CompletionRequest::from_prompt(vec![
        ChatMessage::system("Be terse.").build(),
        ChatMessage::user("Hello").build(),
        ChatMessage::assistant("Hi").build(),
        ChatMessage::user("Continue").build(),
    ])
    .with_model(model)
    .with_provider_option("azure", serde_json::json!({ "suffix": "!" }));

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
    let config_resp = config_client
        .complete(request.clone())
        .await
        .expect("config completion ok");
    let registry_resp = registry_model
        .as_completion_capability()
        .expect("registry completion capability")
        .complete(request)
        .await
        .expect("registry completion ok");

    assert_eq!(siumai_resp.text(), "done");
    assert_eq!(provider_resp.text(), "done");
    assert_eq!(config_resp.text(), "done");
    assert_eq!(registry_resp.text(), "done");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.invalid/openai/v1/completions?api-version=v1"
    );
    assert_eq!(
        header_value(&siumai_req, "api-key"),
        Some("test-key".to_string())
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
async fn azure_completion_stream_public_paths_keep_raw_chunks_runtime_only() {
    use futures_util::StreamExt;

    let model = "deployment-id";
    let stream_body = concat!(
            "data: {\"id\":\"cmpl-azure-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"deployment-id\",\"choices\":[{\"text\":\"hello\",\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"cmpl-azure-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"deployment-id\",\"choices\":[{\"text\":\" world\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n",
            "data: [DONE]\n\n"
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let base_url = "https://example.invalid/openai";

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .completion_model("azure:deployment-id")
        .expect("build registry completion model");

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
    let mut config_stream = config_client
        .complete_stream(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .as_completion_capability()
        .expect("registry completion capability")
        .complete_stream(request)
        .await
        .expect("registry stream ok");

    while siumai_stream.next().await.is_some() {}
    while provider_stream.next().await.is_some() {}
    while config_stream.next().await.is_some() {}
    while registry_stream.next().await.is_some() {}

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.invalid/openai/v1/completions?api-version=v1"
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        header_value(&siumai_req, "api-key"),
        Some("test-key".to_string())
    );
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        siumai_req.body["stream_options"],
        serde_json::json!({ "include_usage": true })
    );
    assert!(siumai_req.body.get("includeRawChunks").is_none());
    assert!(
        siumai_req.body["stream_options"]
            .get("includeRawChunks")
            .is_none()
    );
}
