use super::*;
use futures_util::StreamExt;
use siumai::experimental::client::LlmClient;
use siumai::extensions::TranscriptionExtras;
use siumai::prelude::unified::{
    EmbeddingExtensions, EmbeddingRequest, ResponseFormat, Tool, ToolChoice,
};
use siumai::provider_ext::mistral::{
    MistralChatOptions, MistralChatRequestExt, MistralReasoningEffort,
};
use siumai::provider_ext::moonshotai::{
    MoonshotAIChatOptions, MoonshotAIChatRequestExt, MoonshotAIReasoningHistory,
    MoonshotAIThinkingConfig, MoonshotAIThinkingType,
};
use siumai::provider_ext::openai_compatible::{
    OpenAICompatibleLanguageModelChatOptions, OpenAiCompatibleChatRequestExt,
};
use siumai::provider_ext::openrouter::{
    OpenRouterChatRequestExt, OpenRouterChatResponseExt, OpenRouterOptions, OpenRouterSourceExt,
    OpenRouterTransform,
};
use siumai::provider_ext::perplexity::{
    PerplexityChatRequestExt, PerplexityChatResponseExt, PerplexityOptions,
    PerplexitySearchContextSize, PerplexitySearchMode, PerplexitySearchRecencyFilter,
    PerplexityUserLocation,
};
use siumai_core::types::EmbeddingFormat;
use siumai_registry::registry::builder::RegistryBuilder;

fn make_audio_translation_request(model: &str) -> siumai_core::types::AudioTranslationRequest {
    let mut request =
        siumai_core::types::AudioTranslationRequest::from_audio(b"abc".to_vec(), "audio/mpeg")
            .with_media_type("audio/mpeg".to_string());
    request.model = Some(model.to_string());
    request
}

async fn make_config_client(
    provider_id: &str,
    model: &str,
    transport: Arc<dyn HttpTransport>,
) -> siumai::provider_ext::openai_compatible::OpenAiCompatibleClient {
    let provider = siumai::provider_ext::openai_compatible::get_provider_config(provider_id)
        .expect("builtin provider config");
    let adapter = Arc::new(
        siumai::provider_ext::openai_compatible::ConfigurableAdapter::new(provider.clone()),
    );

    let config = siumai::provider_ext::openai_compatible::OpenAiCompatibleConfig::new(
        provider_id,
        "test-key",
        &provider.base_url,
        adapter,
    )
    .with_model(model)
    .with_http_transport(transport);

    siumai::provider_ext::openai_compatible::OpenAiCompatibleClient::from_config(config)
        .await
        .expect("build config client")
}

fn openai_compatible_registry_providers(
    provider_id: &str,
) -> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
    let mut providers = HashMap::new();
    providers.insert(
        provider_id.to_string(),
        siumai::registry::openai_compatible_provider_factory(provider_id)
            .unwrap_or_else(|err| panic!("{provider_id} openai-compatible factory: {err:?}")),
    );
    providers
}

fn make_registry(
    provider_id: &str,
    transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    RegistryBuilder::new(openai_compatible_registry_providers(provider_id))
        .with_provider_api_key_fetch(provider_id, "test-key", transport)
        .build()
        .expect("build registry")
}

fn make_registry_with_global_reasoning_defaults(
    provider_id: &str,
    transport: Arc<dyn HttpTransport>,
    reasoning_enabled: bool,
    reasoning_budget: i32,
) -> siumai::registry::ProviderRegistryHandle {
    RegistryBuilder::new(openai_compatible_registry_providers(provider_id))
        .with_reasoning(reasoning_enabled)
        .with_reasoning_budget(reasoning_budget)
        .with_provider_api_key_fetch(provider_id, "test-key", transport)
        .build()
        .expect("build registry")
}

fn make_registry_builder_with_global_reasoning_defaults(
    provider_id: &str,
    transport: Arc<dyn HttpTransport>,
    reasoning_enabled: bool,
    reasoning_budget: i32,
) -> siumai::registry::ProviderRegistryHandle {
    RegistryBuilder::new(openai_compatible_registry_providers(provider_id))
        .with_api_key("test-key")
        .with_reasoning(reasoning_enabled)
        .with_reasoning_budget(reasoning_budget)
        .fetch(transport)
        .build()
        .expect("build registry")
}

#[tokio::test]
async fn together_siumai_provider_config_tts_request_are_equivalent() {
    let siumai_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let provider_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let config_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let registry_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");

    let siumai_client = Siumai::builder()
        .openai()
        .togetherai_openai_compatible()
        .api_key("test-key")
        .model("cartesia/sonic-2")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .togetherai_openai_compatible()
        .api_key("test-key")
        .model("cartesia/sonic-2")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "togetherai",
        "cartesia/sonic-2",
        Arc::new(config_transport.clone()),
    )
    .await;
    let registry = make_registry("togetherai", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .speech_model("togetherai:cartesia/sonic-2")
        .expect("build registry speech model");

    let request = TtsRequest::new("hello from together".to_string())
        .with_model("cartesia/sonic-2".to_string())
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
        .text_to_speech(request)
        .await
        .expect("config tts ok");
    let registry_resp = siumai::speech::SpeechModel::synthesize(
        &registry_model,
        TtsRequest::new("hello from together".to_string())
            .with_model("cartesia/sonic-2".to_string())
            .with_voice("alloy".to_string())
            .with_format("mp3".to_string()),
    )
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
    assert_eq!(siumai_req.url, "https://api.together.xyz/v1/audio/speech");
    assert_eq!(
        siumai_req.body["model"],
        serde_json::json!("cartesia/sonic-2")
    );
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!("hello from together")
    );
    assert_eq!(siumai_req.body["voice"], serde_json::json!("alloy"));
    assert_eq!(siumai_req.body["response_format"], serde_json::json!("mp3"));
}

#[tokio::test]
async fn together_siumai_provider_config_stt_request_are_equivalent() {
    let siumai_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from together",
        "language": "en"
    }));
    let provider_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from together",
        "language": "en"
    }));
    let config_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from together",
        "language": "en"
    }));
    let registry_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from together",
        "language": "en"
    }));

    let siumai_client = Siumai::builder()
        .openai()
        .togetherai_openai_compatible()
        .api_key("test-key")
        .model("openai/whisper-large-v3")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .togetherai_openai_compatible()
        .api_key("test-key")
        .model("openai/whisper-large-v3")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "togetherai",
        "openai/whisper-large-v3",
        Arc::new(config_transport.clone()),
    )
    .await;
    let registry = make_registry("togetherai", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .transcription_model("togetherai:openai/whisper-large-v3")
        .expect("build registry transcription model");

    let mut request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
    request.model = Some("openai/whisper-large-v3".to_string());
    request = request.with_media_type("audio/mpeg".to_string());

    let siumai_resp = siumai_client
        .speech_to_text(request.clone())
        .await
        .expect("siumai stt ok");
    let provider_resp = provider_client
        .speech_to_text(request.clone())
        .await
        .expect("provider stt ok");
    let config_resp = config_client
        .speech_to_text(request)
        .await
        .expect("config stt ok");
    let mut registry_request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
    registry_request.model = Some("openai/whisper-large-v3".to_string());
    registry_request = registry_request.with_media_type("audio/mpeg".to_string());

    let registry_resp = registry_model
        .speech_to_text(registry_request)
        .await
        .expect("registry stt ok");

    assert_eq!(siumai_resp.text, "hello from together");
    assert_eq!(provider_resp.text, "hello from together");
    assert_eq!(config_resp.text, "hello from together");
    assert_eq!(registry_resp.text, "hello from together");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_multipart_requests_equivalent(&siumai_req, &provider_req);
    assert_multipart_requests_equivalent(&siumai_req, &config_req);
    assert_multipart_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://api.together.xyz/v1/audio/transcriptions"
    );

    let body_text = normalize_multipart_body(&siumai_req);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("openai/whisper-large-v3"));
    assert!(body_text.contains("name=\"response_format\""));
    assert!(body_text.contains("json"));
    assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
    assert!(body_text.contains("Content-Type: audio/mpeg"));
    assert!(body_text.contains("abc"));
}

#[tokio::test]
async fn together_registry_speech_handle_prefers_provider_specific_build_overrides() {
    let global_transport = BinaryCaptureTransport::new(vec![9, 9, 9], "audio/mpeg");
    let together_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let model = "cartesia/sonic-2";

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("togetherai"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "togetherai",
            "ctx-key",
            "https://example.com/together/v1",
            Arc::new(together_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let registry_model = registry
        .speech_model("togetherai:cartesia/sonic-2")
        .expect("build together speech model");

    let response = siumai::speech::SpeechModel::synthesize(
        &registry_model,
        TtsRequest::new("hello from together".to_string())
            .with_model(model.to_string())
            .with_voice("alloy".to_string())
            .with_format("mp3".to_string()),
    )
    .await
    .expect("registry tts ok");

    assert_eq!(response.audio_data, vec![1, 2, 3, 4]);

    let req = together_transport
        .take()
        .expect("captured together speech request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/together/v1/audio/speech");
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(req.body["input"], serde_json::json!("hello from together"));
    assert_eq!(req.body["voice"], serde_json::json!("alloy"));
    assert_eq!(req.body["response_format"], serde_json::json!("mp3"));
}

#[tokio::test]
async fn together_registry_transcription_handle_prefers_provider_specific_build_overrides() {
    let global_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from global",
        "language": "en"
    }));
    let together_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from together",
        "language": "en"
    }));

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("togetherai"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "togetherai",
            "ctx-key",
            "https://example.com/together/v1",
            Arc::new(together_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let registry_model = registry
        .transcription_model("togetherai:openai/whisper-large-v3")
        .expect("build together transcription model");

    let mut request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
    request.model = Some("openai/whisper-large-v3".to_string());
    request = request.with_media_type("audio/mpeg".to_string());

    let response = registry_model
        .speech_to_text(request)
        .await
        .expect("registry stt ok");

    assert_eq!(response.text, "hello from together");
    assert_eq!(response.language.as_deref(), Some("en"));

    let req = together_transport
        .take()
        .expect("captured together transcription request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        req.headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(
        req.url,
        "https://example.com/together/v1/audio/transcriptions"
    );
    assert!(
        req.headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
    );

    let body_text = normalize_multipart_body(&req);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("openai/whisper-large-v3"));
    assert!(body_text.contains("name=\"response_format\""));
    assert!(body_text.contains("json"));
    assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
    assert!(body_text.contains("Content-Type: audio/mpeg"));
    assert!(body_text.contains("abc"));
}

#[tokio::test]
async fn together_siumai_provider_config_translation_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "openai/whisper-large-v3";

    let siumai_client = Siumai::builder()
        .openai()
        .togetherai_openai_compatible()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .togetherai_openai_compatible()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("togetherai", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("togetherai", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .transcription_model("togetherai:openai/whisper-large-v3")
        .expect("build registry transcription model");

    let request = make_audio_translation_request(model);

    let siumai_err = siumai_client
        .as_transcription_extras()
        .expect("siumai transcription extras")
        .audio_translate(request.clone())
        .await
        .expect_err("together translation should be unsupported");
    let provider_err = provider_client
        .as_transcription_extras()
        .expect("provider transcription extras")
        .audio_translate(request.clone())
        .await
        .expect_err("together translation should be unsupported");
    let config_err = config_client
        .as_transcription_extras()
        .expect("config transcription extras")
        .audio_translate(request.clone())
        .await
        .expect_err("together translation should be unsupported");
    let registry_err = registry_model
        .audio_translate(request)
        .await
        .expect_err("together registry translation should be unsupported");

    assert_unsupported_operation(&siumai_err);
    assert_unsupported_operation(&provider_err);
    assert_unsupported_operation(&config_err);
    assert_unsupported_operation(&registry_err);
    assert_capture_transports_unused(&[
        &siumai_transport,
        &provider_transport,
        &config_transport,
        &registry_transport,
    ]);
}

#[tokio::test]
async fn siliconflow_siumai_provider_config_rerank_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "BAAI/bge-reranker-v2-m3";

    let siumai_client = Siumai::builder()
        .openai()
        .siliconflow()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .siliconflow()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("siliconflow", model, Arc::new(config_transport.clone())).await;

    let request = make_rerank_request_with_model(model).with_top_n(1);

    let _ = siumai_client.rerank(request.clone()).await;
    let _ = provider_client.rerank(request.clone()).await;
    let _ = config_client.rerank(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://api.siliconflow.cn/v1/rerank");
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(siumai_req.body["query"], serde_json::json!("query"));
    assert_eq!(
        siumai_req.body["documents"],
        serde_json::json!(["doc-1", "doc-2"])
    );
    assert_eq!(siumai_req.body["top_n"], serde_json::json!(1));
}

#[tokio::test]
async fn jina_siumai_provider_config_rerank_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "jina-reranker-m0";

    let siumai_client = Siumai::builder()
        .openai()
        .jina()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .jina()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client("jina", model, Arc::new(config_transport.clone())).await;

    let request = make_rerank_request_with_model(model).with_top_n(1);

    let _ = siumai_client.rerank(request.clone()).await;
    let _ = provider_client.rerank(request.clone()).await;
    let _ = config_client.rerank(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://api.jina.ai/v1/rerank");
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(siumai_req.body["query"], serde_json::json!("query"));
    assert_eq!(
        siumai_req.body["documents"],
        serde_json::json!(["doc-1", "doc-2"])
    );
    assert_eq!(siumai_req.body["top_n"], serde_json::json!(1));
}

#[tokio::test]
async fn voyageai_siumai_provider_config_rerank_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "rerank-2";

    let siumai_client = Siumai::builder()
        .openai()
        .voyageai()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .voyageai()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("voyageai", model, Arc::new(config_transport.clone())).await;

    let request = make_rerank_request_with_model(model).with_top_n(1);

    let _ = siumai_client.rerank(request.clone()).await;
    let _ = provider_client.rerank(request.clone()).await;
    let _ = config_client.rerank(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://api.voyageai.com/v1/rerank");
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(siumai_req.body["query"], serde_json::json!("query"));
    assert_eq!(
        siumai_req.body["documents"],
        serde_json::json!(["doc-1", "doc-2"])
    );
    assert_eq!(siumai_req.body["top_n"], serde_json::json!(1));
}

#[tokio::test]
async fn jina_registry_rerank_request_matches_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "jina-reranker-m0";
    let config_client = make_config_client("jina", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("jina", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .reranking_model("jina:jina-reranker-m0")
        .expect("build registry rerank model");

    let request = make_rerank_request_with_model(model).with_top_n(1);

    let _ = config_client.rerank(request.clone()).await;
    let _ = registry_model.rerank(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.url, "https://api.jina.ai/v1/rerank");
    assert_eq!(registry_req.body["model"], serde_json::json!(model));
    assert_eq!(registry_req.body["query"], serde_json::json!("query"));
    assert_eq!(
        registry_req.body["documents"],
        serde_json::json!(["doc-1", "doc-2"])
    );
    assert_eq!(registry_req.body["top_n"], serde_json::json!(1));
}

#[tokio::test]
async fn voyageai_registry_rerank_request_matches_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "rerank-2";
    let config_client =
        make_config_client("voyageai", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("voyageai", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .reranking_model("voyageai:rerank-2")
        .expect("build registry rerank model");

    let request = make_rerank_request_with_model(model).with_top_n(1);

    let _ = config_client.rerank(request.clone()).await;
    let _ = registry_model.rerank(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.url, "https://api.voyageai.com/v1/rerank");
    assert_eq!(registry_req.body["model"], serde_json::json!(model));
    assert_eq!(registry_req.body["query"], serde_json::json!("query"));
    assert_eq!(
        registry_req.body["documents"],
        serde_json::json!(["doc-1", "doc-2"])
    );
    assert_eq!(registry_req.body["top_n"], serde_json::json!(1));
}

#[tokio::test]
async fn siliconflow_registry_rerank_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let siliconflow_transport = CaptureTransport::default();

    let registry = siumai::registry::builder::RegistryBuilder::new(
        openai_compatible_registry_providers("siliconflow"),
    )
    .with_api_key("global-key")
    .with_base_url("https://example.com/global/v1")
    .fetch(Arc::new(global_transport.clone()))
    .with_provider_api_key_base_url_fetch(
        "siliconflow",
        "ctx-key",
        "https://example.com/siliconflow/v1",
        Arc::new(siliconflow_transport.clone()),
    )
    .auto_middleware(false)
    .build()
    .expect("build registry");

    let handle = registry
        .reranking_model("siliconflow:BAAI/bge-reranker-v2-m3")
        .expect("build siliconflow rerank model");

    let _ = handle
        .rerank(make_rerank_request_with_model("BAAI/bge-reranker-v2-m3").with_top_n(1))
        .await;

    let req = siliconflow_transport
        .take()
        .expect("captured siliconflow rerank request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/siliconflow/v1/rerank");
    assert_eq!(
        req.body["model"],
        serde_json::json!("BAAI/bge-reranker-v2-m3")
    );
    assert_eq!(req.body["query"], serde_json::json!("query"));
    assert_eq!(req.body["documents"], serde_json::json!(["doc-1", "doc-2"]));
    assert_eq!(req.body["top_n"], serde_json::json!(1));
}

#[tokio::test]
async fn jina_registry_rerank_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let jina_transport = CaptureTransport::default();

    let registry = siumai::registry::builder::RegistryBuilder::new(
        openai_compatible_registry_providers("jina"),
    )
    .with_api_key("global-key")
    .with_base_url("https://example.com/global/v1")
    .fetch(Arc::new(global_transport.clone()))
    .with_provider_api_key_base_url_fetch(
        "jina",
        "ctx-key",
        "https://example.com/jina/v1",
        Arc::new(jina_transport.clone()),
    )
    .auto_middleware(false)
    .build()
    .expect("build registry");

    let handle = registry
        .reranking_model("jina:jina-reranker-m0")
        .expect("build jina rerank model");

    let _ = handle
        .rerank(make_rerank_request_with_model("jina-reranker-m0").with_top_n(1))
        .await;

    let req = jina_transport.take().expect("captured jina rerank request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/jina/v1/rerank");
    assert_eq!(req.body["model"], serde_json::json!("jina-reranker-m0"));
    assert_eq!(req.body["query"], serde_json::json!("query"));
    assert_eq!(req.body["documents"], serde_json::json!(["doc-1", "doc-2"]));
    assert_eq!(req.body["top_n"], serde_json::json!(1));
}

#[tokio::test]
async fn voyageai_registry_rerank_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let voyageai_transport = CaptureTransport::default();

    let registry = siumai::registry::builder::RegistryBuilder::new(
        openai_compatible_registry_providers("voyageai"),
    )
    .with_api_key("global-key")
    .with_base_url("https://example.com/global/v1")
    .fetch(Arc::new(global_transport.clone()))
    .with_provider_api_key_base_url_fetch(
        "voyageai",
        "ctx-key",
        "https://example.com/voyageai/v1",
        Arc::new(voyageai_transport.clone()),
    )
    .auto_middleware(false)
    .build()
    .expect("build registry");

    let handle = registry
        .reranking_model("voyageai:rerank-2")
        .expect("build voyageai rerank model");

    let _ = handle
        .rerank(make_rerank_request_with_model("rerank-2").with_top_n(1))
        .await;

    let req = voyageai_transport
        .take()
        .expect("captured voyageai rerank request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/voyageai/v1/rerank");
    assert_eq!(req.body["model"], serde_json::json!("rerank-2"));
    assert_eq!(req.body["query"], serde_json::json!("query"));
    assert_eq!(req.body["documents"], serde_json::json!(["doc-1", "doc-2"]));
    assert_eq!(req.body["top_n"], serde_json::json!(1));
}

#[tokio::test]
async fn jina_siumai_provider_config_registry_chat_request_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "jina-embeddings-v2-base-en";

    let siumai_client = Siumai::builder()
        .openai()
        .jina()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .jina()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client("jina", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("jina", Arc::new(registry_transport.clone()));

    let request = make_chat_request_with_model(model);

    let siumai_err = siumai_client
        .chat_request(request.clone())
        .await
        .expect_err("jina chat should be unsupported");
    let provider_err = provider_client
        .chat_request(request.clone())
        .await
        .expect_err("jina chat should be unsupported");
    let config_err = config_client
        .chat_request(request.clone())
        .await
        .expect_err("jina chat should be unsupported");
    let registry_err = match registry.language_model("jina:jina-embeddings-v2-base-en") {
        Ok(_) => panic!("jina registry language model construction should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&siumai_err);
    assert_unsupported_operation(&provider_err);
    assert_unsupported_operation(&config_err);
    assert_unsupported_operation(&registry_err);
    assert!(siumai_client.as_chat_capability().is_none());
    assert!(provider_client.as_chat_capability().is_none());
    assert!(config_client.as_chat_capability().is_none());
    assert_capture_transports_unused(&[
        &siumai_transport,
        &provider_transport,
        &config_transport,
        &registry_transport,
    ]);
}

#[tokio::test]
async fn voyageai_registry_language_model_construction_is_intentionally_unsupported() {
    let registry_transport = CaptureTransport::default();
    let registry = make_registry("voyageai", Arc::new(registry_transport.clone()));

    let registry_err = match registry.language_model("voyageai:rerank-2") {
        Ok(_) => panic!("voyageai registry language model construction should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&registry_err);
    assert_capture_transports_unused(&[&registry_transport]);
}

#[tokio::test]
async fn voyageai_siumai_provider_config_registry_chat_request_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "voyage-3";

    let siumai_client = Siumai::builder()
        .openai()
        .voyageai()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .voyageai()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("voyageai", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("voyageai", Arc::new(registry_transport.clone()));

    let request = make_chat_request_with_model(model);

    let siumai_err = siumai_client
        .chat_request(request.clone())
        .await
        .expect_err("voyageai chat should be unsupported");
    let provider_err = provider_client
        .chat_request(request.clone())
        .await
        .expect_err("voyageai chat should be unsupported");
    let config_err = config_client
        .chat_request(request)
        .await
        .expect_err("voyageai chat should be unsupported");
    let registry_err = match registry.language_model("voyageai:voyage-3") {
        Ok(_) => panic!("voyageai registry language model construction should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&siumai_err);
    assert_unsupported_operation(&provider_err);
    assert_unsupported_operation(&config_err);
    assert_unsupported_operation(&registry_err);
    assert!(siumai_client.as_chat_capability().is_none());
    assert!(provider_client.as_chat_capability().is_none());
    assert!(config_client.as_chat_capability().is_none());
    assert_capture_transports_unused(&[
        &siumai_transport,
        &provider_transport,
        &config_transport,
        &registry_transport,
    ]);
}

#[tokio::test]
async fn siliconflow_siumai_provider_config_tts_request_are_equivalent() {
    let siumai_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let provider_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let config_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let registry_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");

    let model = "FunAudioLLM/CosyVoice2-0.5B";

    let siumai_client = Siumai::builder()
        .openai()
        .siliconflow()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .siliconflow()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("siliconflow", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("siliconflow", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .speech_model("siliconflow:FunAudioLLM/CosyVoice2-0.5B")
        .expect("build registry speech model");

    let request = TtsRequest::new("hello from siliconflow".to_string())
        .with_model(model.to_string())
        .with_voice("FunAudioLLM/CosyVoice2-0.5B:diana".to_string())
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
        .text_to_speech(request)
        .await
        .expect("config tts ok");
    let registry_resp = siumai::speech::SpeechModel::synthesize(
        &registry_model,
        TtsRequest::new("hello from siliconflow".to_string())
            .with_model(model.to_string())
            .with_voice("FunAudioLLM/CosyVoice2-0.5B:diana".to_string())
            .with_format("mp3".to_string()),
    )
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
    assert_eq!(siumai_req.url, "https://api.siliconflow.cn/v1/audio/speech");
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!("hello from siliconflow")
    );
    assert_eq!(
        siumai_req.body["voice"],
        serde_json::json!("FunAudioLLM/CosyVoice2-0.5B:diana")
    );
    assert_eq!(siumai_req.body["response_format"], serde_json::json!("mp3"));
}

#[tokio::test]
async fn siliconflow_registry_speech_handle_prefers_provider_specific_build_overrides() {
    let global_transport = BinaryCaptureTransport::new(vec![9, 9, 9], "audio/mpeg");
    let siliconflow_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let model = "FunAudioLLM/CosyVoice2-0.5B";

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("siliconflow"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "siliconflow",
            "ctx-key",
            "https://example.com/siliconflow/v1",
            Arc::new(siliconflow_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let registry_model = registry
        .speech_model("siliconflow:FunAudioLLM/CosyVoice2-0.5B")
        .expect("build registry speech model");

    let response = siumai::speech::SpeechModel::synthesize(
        &registry_model,
        TtsRequest::new("hello from siliconflow".to_string())
            .with_model(model.to_string())
            .with_voice("FunAudioLLM/CosyVoice2-0.5B:diana".to_string())
            .with_format("mp3".to_string()),
    )
    .await
    .expect("registry tts ok");

    assert_eq!(response.audio_data, vec![1, 2, 3, 4]);

    let req = siliconflow_transport
        .take()
        .expect("captured siliconflow speech request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/siliconflow/v1/audio/speech");
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(
        req.body["input"],
        serde_json::json!("hello from siliconflow")
    );
    assert_eq!(
        req.body["voice"],
        serde_json::json!("FunAudioLLM/CosyVoice2-0.5B:diana")
    );
    assert_eq!(req.body["response_format"], serde_json::json!("mp3"));
}

#[tokio::test]
async fn siliconflow_siumai_provider_config_stt_request_are_equivalent() {
    let siumai_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from siliconflow",
        "language": "zh"
    }));
    let provider_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from siliconflow",
        "language": "zh"
    }));
    let config_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from siliconflow",
        "language": "zh"
    }));
    let registry_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from siliconflow",
        "language": "zh"
    }));

    let siumai_client = Siumai::builder()
        .openai()
        .siliconflow()
        .api_key("test-key")
        .model("FunAudioLLM/SenseVoiceSmall")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .siliconflow()
        .api_key("test-key")
        .model("FunAudioLLM/SenseVoiceSmall")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "siliconflow",
        "FunAudioLLM/SenseVoiceSmall",
        Arc::new(config_transport.clone()),
    )
    .await;
    let registry = make_registry("siliconflow", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .transcription_model("siliconflow:FunAudioLLM/SenseVoiceSmall")
        .expect("build registry transcription model");

    let mut request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
    request.model = Some("FunAudioLLM/SenseVoiceSmall".to_string());
    request = request.with_media_type("audio/mpeg".to_string());

    let siumai_resp = siumai_client
        .speech_to_text(request.clone())
        .await
        .expect("siumai stt ok");
    let provider_resp = provider_client
        .speech_to_text(request.clone())
        .await
        .expect("provider stt ok");
    let config_resp = config_client
        .speech_to_text(request)
        .await
        .expect("config stt ok");
    let mut registry_request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
    registry_request.model = Some("FunAudioLLM/SenseVoiceSmall".to_string());
    registry_request = registry_request.with_media_type("audio/mpeg".to_string());

    let registry_resp = registry_model
        .speech_to_text(registry_request)
        .await
        .expect("registry stt ok");

    assert_eq!(siumai_resp.text, "hello from siliconflow");
    assert_eq!(provider_resp.text, "hello from siliconflow");
    assert_eq!(config_resp.text, "hello from siliconflow");
    assert_eq!(registry_resp.text, "hello from siliconflow");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_multipart_requests_equivalent(&siumai_req, &provider_req);
    assert_multipart_requests_equivalent(&siumai_req, &config_req);
    assert_multipart_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://api.siliconflow.cn/v1/audio/transcriptions"
    );

    let body_text = normalize_multipart_body(&siumai_req);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("FunAudioLLM/SenseVoiceSmall"));
    assert!(body_text.contains("name=\"response_format\""));
    assert!(body_text.contains("json"));
    assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
    assert!(body_text.contains("Content-Type: audio/mpeg"));
    assert!(body_text.contains("abc"));
}

#[tokio::test]
async fn siliconflow_registry_transcription_handle_prefers_provider_specific_build_overrides() {
    let global_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from global",
        "language": "zh"
    }));
    let siliconflow_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from siliconflow",
        "language": "zh"
    }));

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("siliconflow"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "siliconflow",
            "ctx-key",
            "https://example.com/siliconflow/v1",
            Arc::new(siliconflow_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let registry_model = registry
        .transcription_model("siliconflow:FunAudioLLM/SenseVoiceSmall")
        .expect("build registry transcription model");

    let mut request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
    request.model = Some("FunAudioLLM/SenseVoiceSmall".to_string());
    request = request.with_media_type("audio/mpeg".to_string());

    let response = registry_model
        .speech_to_text(request)
        .await
        .expect("registry stt ok");

    assert_eq!(response.text, "hello from siliconflow");
    assert_eq!(response.language.as_deref(), Some("zh"));

    let req = siliconflow_transport
        .take()
        .expect("captured siliconflow transcription request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        req.headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(
        req.url,
        "https://example.com/siliconflow/v1/audio/transcriptions"
    );
    assert!(
        req.headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
    );

    let body_text = normalize_multipart_body(&req);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("FunAudioLLM/SenseVoiceSmall"));
    assert!(body_text.contains("name=\"response_format\""));
    assert!(body_text.contains("json"));
    assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
    assert!(body_text.contains("Content-Type: audio/mpeg"));
    assert!(body_text.contains("abc"));
}

#[tokio::test]
async fn siliconflow_siumai_provider_config_translation_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "FunAudioLLM/SenseVoiceSmall";

    let siumai_client = Siumai::builder()
        .openai()
        .siliconflow()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .siliconflow()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("siliconflow", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("siliconflow", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .transcription_model("siliconflow:FunAudioLLM/SenseVoiceSmall")
        .expect("build registry transcription model");

    let request = make_audio_translation_request(model);

    let siumai_err = siumai_client
        .as_transcription_extras()
        .expect("siumai transcription extras")
        .audio_translate(request.clone())
        .await
        .expect_err("siliconflow translation should be unsupported");
    let provider_err = provider_client
        .as_transcription_extras()
        .expect("provider transcription extras")
        .audio_translate(request.clone())
        .await
        .expect_err("siliconflow translation should be unsupported");
    let config_err = config_client
        .as_transcription_extras()
        .expect("config transcription extras")
        .audio_translate(request.clone())
        .await
        .expect_err("siliconflow translation should be unsupported");
    let registry_err = registry_model
        .audio_translate(request)
        .await
        .expect_err("siliconflow registry translation should be unsupported");

    assert_unsupported_operation(&siumai_err);
    assert_unsupported_operation(&provider_err);
    assert_unsupported_operation(&config_err);
    assert_unsupported_operation(&registry_err);
    assert_capture_transports_unused(&[
        &siumai_transport,
        &provider_transport,
        &config_transport,
        &registry_transport,
    ]);
}

#[tokio::test]
async fn fireworks_siumai_provider_config_stt_request_are_equivalent() {
    let siumai_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from fireworks",
        "language": "en"
    }));
    let provider_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from fireworks",
        "language": "en"
    }));
    let config_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from fireworks",
        "language": "en"
    }));
    let registry_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from fireworks",
        "language": "en"
    }));

    let siumai_client = Siumai::builder()
        .openai()
        .fireworks()
        .api_key("test-key")
        .model("whisper-v3")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .fireworks()
        .api_key("test-key")
        .model("whisper-v3")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "fireworks",
        "whisper-v3",
        Arc::new(config_transport.clone()),
    )
    .await;
    let registry = make_registry("fireworks", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .transcription_model("fireworks:whisper-v3")
        .expect("build registry transcription model");

    let mut request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
    request.model = Some("whisper-v3".to_string());
    request = request.with_media_type("audio/mpeg".to_string());

    let siumai_resp = siumai_client
        .speech_to_text(request.clone())
        .await
        .expect("siumai stt ok");
    let provider_resp = provider_client
        .speech_to_text(request.clone())
        .await
        .expect("provider stt ok");
    let config_resp = config_client
        .speech_to_text(request)
        .await
        .expect("config stt ok");
    let mut registry_request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
    registry_request.model = Some("whisper-v3".to_string());
    registry_request = registry_request.with_media_type("audio/mpeg".to_string());
    let registry_resp = registry_model
        .speech_to_text(registry_request)
        .await
        .expect("registry stt ok");

    assert_eq!(siumai_resp.text, "hello from fireworks");
    assert_eq!(provider_resp.text, "hello from fireworks");
    assert_eq!(config_resp.text, "hello from fireworks");
    assert_eq!(registry_resp.text, "hello from fireworks");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_multipart_requests_equivalent(&siumai_req, &provider_req);
    assert_multipart_requests_equivalent(&siumai_req, &config_req);
    assert_multipart_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://audio.fireworks.ai/v1/audio/transcriptions"
    );

    let body_text = normalize_multipart_body(&siumai_req);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("whisper-v3"));
    assert!(body_text.contains("name=\"response_format\""));
    assert!(body_text.contains("json"));
    assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
    assert!(body_text.contains("Content-Type: audio/mpeg"));
    assert!(body_text.contains("abc"));
}

#[tokio::test]
async fn fireworks_registry_transcription_handle_prefers_provider_specific_build_overrides() {
    let global_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from global",
        "language": "en"
    }));
    let fireworks_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from fireworks",
        "language": "en"
    }));

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("fireworks"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .with_provider_api_key_base_url_fetch(
            "fireworks",
            "ctx-key",
            "https://example.com/fireworks-audio/v1",
            Arc::new(fireworks_transport.clone()),
        )
        .fetch(Arc::new(global_transport.clone()))
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let registry_model = registry
        .transcription_model("fireworks:whisper-v3")
        .expect("build registry transcription model");

    let mut request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
    request.model = Some("whisper-v3".to_string());
    request = request.with_media_type("audio/mpeg".to_string());

    let response = registry_model
        .speech_to_text(request)
        .await
        .expect("registry stt ok");

    assert_eq!(response.text, "hello from fireworks");
    assert_eq!(response.language.as_deref(), Some("en"));

    let req = fireworks_transport
        .take()
        .expect("captured fireworks multipart request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        req.headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(
        req.url,
        "https://example.com/fireworks-audio/v1/audio/transcriptions"
    );
    assert!(
        req.headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
    );

    let body_text = normalize_multipart_body(&req);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("whisper-v3"));
    assert!(body_text.contains("name=\"response_format\""));
    assert!(body_text.contains("json"));
    assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
    assert!(body_text.contains("Content-Type: audio/mpeg"));
    assert!(body_text.contains("abc"));
}

#[tokio::test]
async fn fireworks_siumai_provider_config_translation_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "whisper-v3";

    let siumai_client = Siumai::builder()
        .openai()
        .fireworks()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .fireworks()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("fireworks", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("fireworks", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .transcription_model("fireworks:whisper-v3")
        .expect("build registry transcription model");

    let request = make_audio_translation_request(model);

    let siumai_err = siumai_client
        .as_transcription_extras()
        .expect("siumai transcription extras")
        .audio_translate(request.clone())
        .await
        .expect_err("fireworks translation should be unsupported");
    let provider_err = provider_client
        .as_transcription_extras()
        .expect("provider transcription extras")
        .audio_translate(request.clone())
        .await
        .expect_err("fireworks translation should be unsupported");
    let config_err = config_client
        .as_transcription_extras()
        .expect("config transcription extras")
        .audio_translate(request.clone())
        .await
        .expect_err("fireworks translation should be unsupported");
    let registry_err = registry_model
        .audio_translate(request)
        .await
        .expect_err("fireworks registry translation should be unsupported");

    assert_unsupported_operation(&siumai_err);
    assert_unsupported_operation(&provider_err);
    assert_unsupported_operation(&config_err);
    assert_unsupported_operation(&registry_err);
    assert_capture_transports_unused(&[
        &siumai_transport,
        &provider_transport,
        &config_transport,
        &registry_transport,
    ]);
}

#[tokio::test]
async fn fireworks_siumai_provider_config_tts_request_is_intentionally_unsupported() {
    let siumai_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let provider_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let config_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");

    let model = "whisper-v3";

    let siumai_client = Siumai::builder()
        .openai()
        .fireworks()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .fireworks()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("fireworks", model, Arc::new(config_transport.clone())).await;

    let request = TtsRequest::new("hello from fireworks".to_string())
        .with_model(model.to_string())
        .with_voice("alloy".to_string())
        .with_format("mp3".to_string());

    let siumai_err = siumai_client
        .text_to_speech(request.clone())
        .await
        .expect_err("fireworks speech should be unsupported");
    let provider_err = provider_client
        .text_to_speech(request.clone())
        .await
        .expect_err("fireworks speech should be unsupported");
    let config_err = config_client
        .text_to_speech(request)
        .await
        .expect_err("fireworks speech should be unsupported");

    assert_unsupported_operation(&siumai_err);
    assert_unsupported_operation(&provider_err);
    assert_unsupported_operation(&config_err);
    assert!(siumai_client.as_speech_capability().is_none());
    assert!(provider_client.as_speech_capability().is_none());
    assert!(config_client.as_speech_capability().is_none());
    assert!(siumai_transport.take().is_none());
    assert!(provider_transport.take().is_none());
    assert!(config_transport.take().is_none());
}

#[tokio::test]
async fn fireworks_registry_tts_request_is_intentionally_unsupported() {
    let registry_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let registry = make_registry("fireworks", Arc::new(registry_transport.clone()));
    let registry_err = match registry.speech_model("fireworks:whisper-v3") {
        Ok(_) => panic!("build registry speech model should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&registry_err);
    assert!(registry_transport.take().is_none());
}

#[tokio::test]
async fn siliconflow_siumai_provider_config_image_generation_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .siliconflow()
        .api_key("test-key")
        .model("stability-ai/sdxl")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .siliconflow()
        .api_key("test-key")
        .model("stability-ai/sdxl")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "siliconflow",
        "stability-ai/sdxl",
        Arc::new(config_transport.clone()),
    )
    .await;

    let request = ImageGenerationRequest {
        prompt: "a tiny orange robot".to_string(),
        negative_prompt: Some("blurry".to_string()),
        size: Some("1024x1024".to_string()),
        aspect_ratio: None,
        count: 1,
        model: Some("stability-ai/sdxl".to_string()),
        quality: None,
        style: None,
        seed: None,
        steps: None,
        guidance_scale: None,
        enhance_prompt: None,
        response_format: Some("url".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    };

    let _ = siumai_client.generate_images(request.clone()).await;
    let _ = provider_client.generate_images(request.clone()).await;
    let _ = config_client.generate_images(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://api.siliconflow.cn/v1/images/generations"
    );
    assert_eq!(
        siumai_req.body["model"],
        serde_json::json!("stability-ai/sdxl")
    );
    assert_eq!(
        siumai_req.body["prompt"],
        serde_json::json!("a tiny orange robot")
    );
    assert_eq!(siumai_req.body["size"], serde_json::json!("1024x1024"));
    assert_eq!(siumai_req.body["n"], serde_json::json!(1));
    assert_eq!(siumai_req.body["response_format"], serde_json::json!("url"));
}

#[tokio::test]
async fn together_siumai_provider_config_image_generation_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "black-forest-labs/FLUX.1-schnell";

    let siumai_client = Siumai::builder()
        .openai()
        .togetherai_openai_compatible()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .togetherai_openai_compatible()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("togetherai", model, Arc::new(config_transport.clone())).await;

    let request = ImageGenerationRequest {
        prompt: "a tiny blue robot".to_string(),
        negative_prompt: Some("blurry".to_string()),
        size: Some("1024x1024".to_string()),
        aspect_ratio: None,
        count: 1,
        model: Some(model.to_string()),
        quality: None,
        style: None,
        seed: None,
        steps: None,
        guidance_scale: None,
        enhance_prompt: None,
        response_format: Some("url".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    };

    let _ = siumai_client.generate_images(request.clone()).await;
    let _ = provider_client.generate_images(request.clone()).await;
    let _ = config_client.generate_images(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://api.together.xyz/v1/images/generations"
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["prompt"],
        serde_json::json!("a tiny blue robot")
    );
    assert_eq!(siumai_req.body["width"], serde_json::json!(1024));
    assert_eq!(siumai_req.body["height"], serde_json::json!(1024));
    assert!(siumai_req.body.get("size").is_none());
    assert!(siumai_req.body.get("n").is_none());
    assert_eq!(
        siumai_req.body["response_format"],
        serde_json::json!("base64")
    );
}

#[tokio::test]
async fn siliconflow_registry_image_generation_request_matches_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();
    let model = "stability-ai/sdxl";

    let config_client =
        make_config_client("siliconflow", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("siliconflow", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .image_model("siliconflow:stability-ai/sdxl")
        .expect("build registry image model");

    let request = ImageGenerationRequest {
        prompt: "a tiny orange robot".to_string(),
        negative_prompt: Some("blurry".to_string()),
        size: Some("1024x1024".to_string()),
        aspect_ratio: None,
        count: 1,
        model: Some(model.to_string()),
        quality: None,
        style: None,
        seed: None,
        steps: None,
        guidance_scale: None,
        enhance_prompt: None,
        response_format: Some("url".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    };

    let _ = config_client.generate_images(request.clone()).await;
    let _ = registry_model.generate_images(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.url,
        "https://api.siliconflow.cn/v1/images/generations"
    );
    assert_eq!(
        registry_req.body["model"],
        serde_json::json!("stability-ai/sdxl")
    );
    assert_eq!(
        registry_req.body["prompt"],
        serde_json::json!("a tiny orange robot")
    );
    assert_eq!(registry_req.body["size"], serde_json::json!("1024x1024"));
    assert_eq!(registry_req.body["n"], serde_json::json!(1));
    assert_eq!(
        registry_req.body["response_format"],
        serde_json::json!("url")
    );
}

#[tokio::test]
async fn together_registry_image_generation_request_matches_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();
    let model = "black-forest-labs/FLUX.1-schnell";

    let config_client =
        make_config_client("togetherai", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("togetherai", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .image_model("togetherai:black-forest-labs/FLUX.1-schnell")
        .expect("build registry image model");

    let request = ImageGenerationRequest {
        prompt: "a tiny blue robot".to_string(),
        negative_prompt: Some("blurry".to_string()),
        size: Some("1024x1024".to_string()),
        aspect_ratio: None,
        count: 1,
        model: Some(model.to_string()),
        quality: None,
        style: None,
        seed: None,
        steps: None,
        guidance_scale: None,
        enhance_prompt: None,
        response_format: Some("url".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    };

    let _ = config_client.generate_images(request.clone()).await;
    let _ = registry_model.generate_images(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.url,
        "https://api.together.xyz/v1/images/generations"
    );
    assert_eq!(registry_req.body["model"], serde_json::json!(model));
    assert_eq!(
        registry_req.body["prompt"],
        serde_json::json!("a tiny blue robot")
    );
    assert_eq!(registry_req.body["width"], serde_json::json!(1024));
    assert_eq!(registry_req.body["height"], serde_json::json!(1024));
    assert!(registry_req.body.get("size").is_none());
    assert!(registry_req.body.get("n").is_none());
    assert_eq!(
        registry_req.body["response_format"],
        serde_json::json!("base64")
    );
}

#[tokio::test]
async fn siliconflow_registry_image_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let siliconflow_transport = CaptureTransport::default();

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("siliconflow"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "siliconflow",
            "ctx-key",
            "https://example.com/siliconflow/v1",
            Arc::new(siliconflow_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .image_model("siliconflow:stability-ai/sdxl")
        .expect("build siliconflow image model");

    let _ = handle
        .generate_images(ImageGenerationRequest {
            prompt: "a tiny orange robot".to_string(),
            negative_prompt: Some("blurry".to_string()),
            size: Some("1024x1024".to_string()),
            aspect_ratio: None,
            count: 1,
            model: Some("stability-ai/sdxl".to_string()),
            quality: None,
            style: None,
            seed: None,
            steps: None,
            guidance_scale: None,
            enhance_prompt: None,
            response_format: Some("url".to_string()),
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        })
        .await;

    assert!(global_transport.take().is_none());

    let req = siliconflow_transport
        .take()
        .expect("captured siliconflow image request");
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(
        req.url,
        "https://example.com/siliconflow/v1/images/generations"
    );
    assert_eq!(req.body["model"], serde_json::json!("stability-ai/sdxl"));
    assert_eq!(req.body["prompt"], serde_json::json!("a tiny orange robot"));
    assert_eq!(req.body["size"], serde_json::json!("1024x1024"));
    assert_eq!(req.body["n"], serde_json::json!(1));
    assert_eq!(req.body["response_format"], serde_json::json!("url"));
}

#[tokio::test]
async fn together_registry_image_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let together_transport = CaptureTransport::default();

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("togetherai"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "togetherai",
            "ctx-key",
            "https://example.com/together/v1",
            Arc::new(together_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .image_model("togetherai:black-forest-labs/FLUX.1-schnell")
        .expect("build together image model");

    let _ = handle
        .generate_images(ImageGenerationRequest {
            prompt: "a tiny blue robot".to_string(),
            negative_prompt: Some("blurry".to_string()),
            size: Some("1024x1024".to_string()),
            aspect_ratio: None,
            count: 1,
            model: Some("black-forest-labs/FLUX.1-schnell".to_string()),
            quality: None,
            style: None,
            seed: None,
            steps: None,
            guidance_scale: None,
            enhance_prompt: None,
            response_format: Some("url".to_string()),
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        })
        .await;

    assert!(global_transport.take().is_none());

    let req = together_transport
        .take()
        .expect("captured together image request");
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(
        req.url,
        "https://example.com/together/v1/images/generations"
    );
    assert_eq!(
        req.body["model"],
        serde_json::json!("black-forest-labs/FLUX.1-schnell")
    );
    assert_eq!(req.body["prompt"], serde_json::json!("a tiny blue robot"));
    assert_eq!(req.body["width"], serde_json::json!(1024));
    assert_eq!(req.body["height"], serde_json::json!(1024));
    assert!(req.body.get("size").is_none());
    assert!(req.body.get("n").is_none());
    assert_eq!(req.body["response_format"], serde_json::json!("base64"));
}

#[test]
fn mistral_package_settings_preserve_supported_provider_inputs() {
    let config = siumai::provider_ext::mistral::MistralProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/mistral")
        .with_header("x-test", "1")
        .into_config_for_model("mistral-large-latest")
        .expect("settings into config");

    assert_eq!(config.provider_id, "mistral");
    assert_eq!(config.base_url, "https://example.com/mistral");
    assert_eq!(config.common_params.model, "mistral-large-latest");
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
}
#[tokio::test]
async fn mistral_siumai_provider_config_embedding_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .mistral()
        .api_key("test-key")
        .model("mistral-embed")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .mistral()
        .api_key("test-key")
        .model("mistral-embed")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "mistral",
        "mistral-embed",
        Arc::new(config_transport.clone()),
    )
    .await;

    let request =
        EmbeddingRequest::new(vec!["hello mistral".to_string()]).with_model("mistral-embed");

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://api.mistral.ai/v1/embeddings");
    assert_eq!(siumai_req.body["model"], serde_json::json!("mistral-embed"));
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!(["hello mistral"])
    );
}

#[tokio::test]
async fn mistral_registry_embedding_request_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = make_config_client(
        "mistral",
        "mistral-embed",
        Arc::new(config_transport.clone()),
    )
    .await;
    let registry = make_registry("mistral", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .embedding_model("mistral:mistral-embed")
        .expect("build registry embedding model");

    let request =
        EmbeddingRequest::new(vec!["hello mistral".to_string()]).with_model("mistral-embed");

    let _ = config_client.embed_with_config(request.clone()).await;
    let _ = registry_model.embed_with_config(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.url, "https://api.mistral.ai/v1/embeddings");
    assert_eq!(
        registry_req.body["model"],
        serde_json::json!("mistral-embed")
    );
    assert_eq!(
        registry_req.body["input"],
        serde_json::json!(["hello mistral"])
    );
}

#[tokio::test]
async fn mistral_registry_embedding_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let mistral_transport = CaptureTransport::default();

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("mistral"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "mistral",
            "ctx-key",
            "https://example.com/mistral/v1",
            Arc::new(mistral_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .embedding_model("mistral:mistral-embed")
        .expect("build mistral embedding handle");

    let _ = handle
        .embed_with_config(
            EmbeddingRequest::new(vec!["hello mistral".to_string()]).with_model("mistral-embed"),
        )
        .await;

    let req = mistral_transport
        .take()
        .expect("captured mistral embedding request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/mistral/v1/embeddings");
    assert_eq!(req.body["model"], serde_json::json!("mistral-embed"));
    assert_eq!(req.body["input"], serde_json::json!(["hello mistral"]));
}

#[tokio::test]
async fn mistral_public_completion_family_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "mistral-large-latest";

    let siumai_client = Siumai::builder()
        .mistral()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai mistral client");

    let provider_client = Provider::mistral()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider mistral client");

    let config_client =
        make_config_client("mistral", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("mistral", Arc::new(registry_transport.clone()));

    assert!(!siumai_client.capabilities().supports("completion"));
    assert!(!provider_client.capabilities().supports("completion"));
    assert!(!config_client.capabilities().supports("completion"));
    assert!(siumai_client.as_completion_capability().is_none());
    assert!(provider_client.as_completion_capability().is_none());
    assert!(config_client.as_completion_capability().is_none());

    let completion_err = match registry.completion_model(&format!("mistral:{model}")) {
        Ok(_) => panic!("mistral registry completion handle should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&completion_err);
    assert_capture_transports_unused(&[
        &siumai_transport,
        &provider_transport,
        &config_transport,
        &registry_transport,
    ]);
}

#[tokio::test]
async fn mistral_top_level_builder_chat_request_matches_config_registry_path() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "mistral-large-latest";

    let siumai_client = Siumai::builder()
        .mistral()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai mistral client");

    let provider_client = Provider::mistral()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider mistral client");

    let config_client =
        make_config_client("mistral", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("mistral", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model(&format!("mistral:{model}"))
        .expect("build registry mistral model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![
            ChatMessage::system("Be terse.").build(),
            ChatMessage::user("hi").build(),
        ])
        .build();

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "https://api.mistral.ai/v1/chat/completions");
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["messages"][0]["role"],
        serde_json::json!("system")
    );
    assert_eq!(
        siumai_req.body["messages"][0]["content"],
        serde_json::json!("Be terse.")
    );
    assert_eq!(
        siumai_req.body["messages"][1]["role"],
        serde_json::json!("user")
    );
    assert_eq!(
        siumai_req.body["messages"][1]["content"],
        serde_json::json!("hi")
    );
}

#[tokio::test]
async fn mistral_top_level_builder_embedding_request_matches_config_registry_path() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "mistral-embed";

    let siumai_client = Siumai::builder()
        .mistral()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai mistral client");

    let provider_client = Provider::mistral()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider mistral client");

    let config_client =
        make_config_client("mistral", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("mistral", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .embedding_model(&format!("mistral:{model}"))
        .expect("build registry mistral embedding model");

    let request = EmbeddingRequest::single("hello top-level mistral embedding")
        .with_model(model)
        .with_dimensions(1024)
        .with_encoding_format(EmbeddingFormat::Float);

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request.clone()).await;
    let _ = registry_model.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "https://api.mistral.ai/v1/embeddings");
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!(["hello top-level mistral embedding"])
    );
    assert_eq!(siumai_req.body["dimensions"], serde_json::json!(1024));
    assert_eq!(
        siumai_req.body["encoding_format"],
        serde_json::json!("float")
    );
}

#[tokio::test]
async fn mistral_top_level_builder_chat_stream_request_matches_config_registry_path() {
    use futures_util::StreamExt;

    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "mistral-large-latest";

    let siumai_client = Siumai::builder()
        .mistral()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai mistral client");

    let provider_client = Provider::mistral()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider mistral client");

    let config_client =
        make_config_client("mistral", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("mistral", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model(&format!("mistral:{model}"))
        .expect("build registry mistral model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("stream hi").build()])
        .build();

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai mistral stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider mistral stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config mistral stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry mistral stream ok");

    let _ = siumai_stream.next().await;
    let _ = provider_stream.next().await;
    let _ = config_stream.next().await;
    let _ = registry_stream.next().await;

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai mistral stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider mistral stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config mistral stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry mistral stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "https://api.mistral.ai/v1/chat/completions");
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["messages"][0]["content"],
        serde_json::json!("stream hi")
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn mistral_siumai_provider_config_chat_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "mistral-small-latest";

    let siumai_client = Siumai::builder()
        .mistral()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai mistral client");

    let provider_client = Provider::mistral()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider mistral client");

    let config_client =
        make_config_client("mistral", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("mistral", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model(&format!("mistral:{model}"))
        .expect("build registry mistral model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "lookup",
            "Lookup value",
            serde_json::json!({ "type": "object", "properties": {} }),
        )])
        .tool_choice(ToolChoice::None)
        .response_format(ResponseFormat::json_schema(schema).with_name("response"))
        .build()
        .with_mistral_options(
            MistralChatOptions::new()
                .with_safe_prompt(true)
                .with_document_image_limit(8)
                .with_document_page_limit(16)
                .with_structured_outputs(false)
                .with_strict_json_schema(false)
                .with_parallel_tool_calls(false)
                .with_reasoning_effort(MistralReasoningEffort::None),
        );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "https://api.mistral.ai/v1/chat/completions");
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(siumai_req.body["safe_prompt"], serde_json::json!(true));
    assert_eq!(
        siumai_req.body["document_image_limit"],
        serde_json::json!(8)
    );
    assert_eq!(
        siumai_req.body["document_page_limit"],
        serde_json::json!(16)
    );
    assert_eq!(
        siumai_req.body["reasoning_effort"],
        serde_json::json!("none")
    );
    assert_eq!(
        siumai_req.body["parallel_tool_calls"],
        serde_json::json!(false)
    );
    assert_eq!(siumai_req.body["tool_choice"], serde_json::json!("none"));
    assert_eq!(
        siumai_req.body["response_format"],
        serde_json::json!({ "type": "json_object" })
    );
    assert!(siumai_req.body.get("safePrompt").is_none());
    assert!(siumai_req.body.get("documentImageLimit").is_none());
    assert!(siumai_req.body.get("documentPageLimit").is_none());
    assert!(siumai_req.body.get("parallelToolCalls").is_none());
    assert!(siumai_req.body.get("structuredOutputs").is_none());
    assert!(siumai_req.body.get("strictJsonSchema").is_none());
}

#[tokio::test]
async fn fireworks_completion_siumai_provider_config_registry_request_are_equivalent() {
    let response_json = serde_json::json!({
        "id": "cmpl-fireworks-test",
        "object": "text_completion",
        "created": 1_718_345_013u64,
        "model": "accounts/fireworks/models/llama-v3-8b-instruct",
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

    let model = "accounts/fireworks/models/llama-v3-8b-instruct";
    let base_url = "https://api.fireworks.ai/inference/v1";

    let siumai_client = Siumai::builder()
        .openai()
        .fireworks()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .fireworks()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("fireworks", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("fireworks", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .completion_model(&format!("fireworks:{model}"))
        .expect("build registry fireworks completion model");

    let request = CompletionRequest::from_prompt(vec![
        ChatMessage::system("Be terse.").build(),
        ChatMessage::user("Hello").build(),
        ChatMessage::assistant("Hi").build(),
        ChatMessage::user("Continue").build(),
    ])
    .with_model(model)
    .with_provider_option("fireworks", serde_json::json!({ "suffix": "!" }));

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
        .as_completion_capability()
        .expect("config completion capability")
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
        "https://api.fireworks.ai/inference/v1/completions"
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
async fn fireworks_completion_stream_public_paths_keep_raw_chunks_runtime_only() {
    use futures_util::StreamExt;

    let model = "accounts/fireworks/models/llama-v3-8b-instruct";
    let stream_body = concat!(
            "data: {\"id\":\"cmpl-fireworks-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"accounts/fireworks/models/llama-v3-8b-instruct\",\"choices\":[{\"text\":\"hello\",\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"cmpl-fireworks-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"accounts/fireworks/models/llama-v3-8b-instruct\",\"choices\":[{\"text\":\" world\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n",
            "data: [DONE]\n\n"
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let base_url = "https://api.fireworks.ai/inference/v1";

    let siumai_client = Siumai::builder()
        .openai()
        .fireworks()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .fireworks()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("fireworks", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("fireworks", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .completion_model(&format!("fireworks:{model}"))
        .expect("build registry fireworks completion model");

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
        .as_completion_capability()
        .expect("config completion capability")
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
        "https://api.fireworks.ai/inference/v1/completions"
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert!(siumai_req.body.get("stream_options").is_none());
    assert!(siumai_req.body.get("includeRawChunks").is_none());
}

#[test]
fn fireworks_package_settings_preserve_supported_provider_inputs() {
    let config = siumai::provider_ext::fireworks::FireworksProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/fireworks")
        .with_header("x-test", "1")
        .into_config_for_model("accounts/fireworks/models/llama-v3p1-8b-instruct")
        .expect("settings into config");

    assert_eq!(config.provider_id, "fireworks");
    assert_eq!(config.base_url, "https://example.com/fireworks");
    assert_eq!(
        config.common_params.model,
        "accounts/fireworks/models/llama-v3p1-8b-instruct"
    );
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
}
#[tokio::test]
async fn fireworks_siumai_provider_config_embedding_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .fireworks()
        .api_key("test-key")
        .model("nomic-ai/nomic-embed-text-v1.5")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .fireworks()
        .api_key("test-key")
        .model("nomic-ai/nomic-embed-text-v1.5")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "fireworks",
        "nomic-ai/nomic-embed-text-v1.5",
        Arc::new(config_transport.clone()),
    )
    .await;

    let request = EmbeddingRequest::single("hello fireworks embedding")
        .with_model("nomic-ai/nomic-embed-text-v1.5")
        .with_dimensions(256)
        .with_encoding_format(EmbeddingFormat::Base64)
        .with_user("compat-user-1");

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://api.fireworks.ai/inference/v1/embeddings"
    );
    assert_eq!(
        siumai_req.body["model"],
        serde_json::json!("nomic-ai/nomic-embed-text-v1.5")
    );
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!(["hello fireworks embedding"])
    );
    assert_eq!(siumai_req.body["dimensions"], serde_json::json!(256));
    assert_eq!(
        siumai_req.body["encoding_format"],
        serde_json::json!("base64")
    );
    assert_eq!(siumai_req.body["user"], serde_json::json!("compat-user-1"));
}

#[tokio::test]
async fn fireworks_registry_embedding_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = make_config_client(
        "fireworks",
        "nomic-ai/nomic-embed-text-v1.5",
        Arc::new(config_transport.clone()),
    )
    .await;
    let registry = make_registry("fireworks", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .embedding_model("fireworks:nomic-ai/nomic-embed-text-v1.5")
        .expect("build registry embedding model");

    let request = EmbeddingRequest::single("hello fireworks embedding")
        .with_model("nomic-ai/nomic-embed-text-v1.5")
        .with_dimensions(256)
        .with_encoding_format(EmbeddingFormat::Base64)
        .with_user("compat-user-1");

    let _ = config_client.embed_with_config(request.clone()).await;
    let _ = registry_model.embed_with_config(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.url,
        "https://api.fireworks.ai/inference/v1/embeddings"
    );
    assert_eq!(
        registry_req.body["model"],
        serde_json::json!("nomic-ai/nomic-embed-text-v1.5")
    );
    assert_eq!(
        registry_req.body["input"],
        serde_json::json!(["hello fireworks embedding"])
    );
    assert_eq!(registry_req.body["dimensions"], serde_json::json!(256));
    assert_eq!(
        registry_req.body["encoding_format"],
        serde_json::json!("base64")
    );
    assert_eq!(
        registry_req.body["user"],
        serde_json::json!("compat-user-1")
    );
}

#[tokio::test]
async fn fireworks_registry_embedding_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let fireworks_transport = CaptureTransport::default();

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("fireworks"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "fireworks",
            "ctx-key",
            "https://example.com/fireworks/inference/v1",
            Arc::new(fireworks_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .embedding_model("fireworks:nomic-ai/nomic-embed-text-v1.5")
        .expect("build fireworks embedding handle");

    let _ = handle
        .embed_with_config(
            EmbeddingRequest::single("hello fireworks embedding")
                .with_model("nomic-ai/nomic-embed-text-v1.5")
                .with_dimensions(256)
                .with_encoding_format(EmbeddingFormat::Base64)
                .with_user("compat-user-1"),
        )
        .await;

    let req = fireworks_transport
        .take()
        .expect("captured fireworks embedding request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(
        req.url,
        "https://example.com/fireworks/inference/v1/embeddings"
    );
    assert_eq!(
        req.body["model"],
        serde_json::json!("nomic-ai/nomic-embed-text-v1.5")
    );
    assert_eq!(
        req.body["input"],
        serde_json::json!(["hello fireworks embedding"])
    );
    assert_eq!(req.body["dimensions"], serde_json::json!(256));
    assert_eq!(req.body["encoding_format"], serde_json::json!("base64"));
    assert_eq!(req.body["user"], serde_json::json!("compat-user-1"));
}

#[tokio::test]
async fn siliconflow_siumai_provider_config_embedding_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .siliconflow()
        .api_key("test-key")
        .model("BAAI/bge-large-zh-v1.5")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .siliconflow()
        .api_key("test-key")
        .model("BAAI/bge-large-zh-v1.5")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "siliconflow",
        "BAAI/bge-large-zh-v1.5",
        Arc::new(config_transport.clone()),
    )
    .await;

    let request = EmbeddingRequest::single("hello siliconflow embedding")
        .with_model("BAAI/bge-large-zh-v1.5")
        .with_dimensions(768)
        .with_encoding_format(EmbeddingFormat::Float)
        .with_user("compat-user-2");

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://api.siliconflow.cn/v1/embeddings");
    assert_eq!(
        siumai_req.body["model"],
        serde_json::json!("BAAI/bge-large-zh-v1.5")
    );
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!(["hello siliconflow embedding"])
    );
    assert_eq!(siumai_req.body["dimensions"], serde_json::json!(768));
    assert_eq!(
        siumai_req.body["encoding_format"],
        serde_json::json!("float")
    );
    assert_eq!(siumai_req.body["user"], serde_json::json!("compat-user-2"));
}

#[tokio::test]
async fn siliconflow_registry_embedding_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = make_config_client(
        "siliconflow",
        "BAAI/bge-large-zh-v1.5",
        Arc::new(config_transport.clone()),
    )
    .await;
    let registry = make_registry("siliconflow", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .embedding_model("siliconflow:BAAI/bge-large-zh-v1.5")
        .expect("build registry embedding model");

    let request = EmbeddingRequest::single("hello siliconflow embedding")
        .with_model("BAAI/bge-large-zh-v1.5")
        .with_dimensions(768)
        .with_encoding_format(EmbeddingFormat::Float)
        .with_user("compat-user-2");

    let _ = config_client.embed_with_config(request.clone()).await;
    let _ = registry_model.embed_with_config(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.url, "https://api.siliconflow.cn/v1/embeddings");
    assert_eq!(
        registry_req.body["model"],
        serde_json::json!("BAAI/bge-large-zh-v1.5")
    );
    assert_eq!(
        registry_req.body["input"],
        serde_json::json!(["hello siliconflow embedding"])
    );
    assert_eq!(registry_req.body["dimensions"], serde_json::json!(768));
    assert_eq!(
        registry_req.body["encoding_format"],
        serde_json::json!("float")
    );
    assert_eq!(
        registry_req.body["user"],
        serde_json::json!("compat-user-2")
    );
}

#[tokio::test]
async fn siliconflow_registry_embedding_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let siliconflow_transport = CaptureTransport::default();

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("siliconflow"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "siliconflow",
            "ctx-key",
            "https://example.com/siliconflow/v1",
            Arc::new(siliconflow_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .embedding_model("siliconflow:BAAI/bge-large-zh-v1.5")
        .expect("build siliconflow embedding handle");

    let _ = handle
        .embed_with_config(
            EmbeddingRequest::single("hello siliconflow embedding")
                .with_model("BAAI/bge-large-zh-v1.5")
                .with_dimensions(768)
                .with_encoding_format(EmbeddingFormat::Float)
                .with_user("compat-user-2"),
        )
        .await;

    let req = siliconflow_transport
        .take()
        .expect("captured siliconflow embedding request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/siliconflow/v1/embeddings");
    assert_eq!(
        req.body["model"],
        serde_json::json!("BAAI/bge-large-zh-v1.5")
    );
    assert_eq!(
        req.body["input"],
        serde_json::json!(["hello siliconflow embedding"])
    );
    assert_eq!(req.body["dimensions"], serde_json::json!(768));
    assert_eq!(req.body["encoding_format"], serde_json::json!("float"));
    assert_eq!(req.body["user"], serde_json::json!("compat-user-2"));
}

#[tokio::test]
async fn openrouter_siumai_provider_config_embedding_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .openrouter()
        .api_key("test-key")
        .model("openai/text-embedding-3-small")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .model("openai/text-embedding-3-small")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "openrouter",
        "openai/text-embedding-3-small",
        Arc::new(config_transport.clone()),
    )
    .await;

    let request = EmbeddingRequest::single("hello openrouter embedding")
        .with_model("openai/text-embedding-3-small")
        .with_dimensions(512)
        .with_encoding_format(EmbeddingFormat::Base64)
        .with_user("compat-user-3");

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://openrouter.ai/api/v1/embeddings");
    assert_eq!(
        siumai_req.body["model"],
        serde_json::json!("openai/text-embedding-3-small")
    );
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!(["hello openrouter embedding"])
    );
    assert_eq!(siumai_req.body["dimensions"], serde_json::json!(512));
    assert_eq!(
        siumai_req.body["encoding_format"],
        serde_json::json!("base64")
    );
    assert_eq!(siumai_req.body["user"], serde_json::json!("compat-user-3"));
}

#[tokio::test]
async fn openrouter_registry_embedding_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = make_config_client(
        "openrouter",
        "openai/text-embedding-3-small",
        Arc::new(config_transport.clone()),
    )
    .await;
    let registry = make_registry("openrouter", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .embedding_model("openrouter:openai/text-embedding-3-small")
        .expect("build registry embedding model");

    let request = EmbeddingRequest::single("hello openrouter embedding")
        .with_model("openai/text-embedding-3-small")
        .with_dimensions(512)
        .with_encoding_format(EmbeddingFormat::Base64)
        .with_user("compat-user-3");

    let _ = config_client.embed_with_config(request.clone()).await;
    let _ = registry_model.embed_with_config(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.url, "https://openrouter.ai/api/v1/embeddings");
    assert_eq!(
        registry_req.body["model"],
        serde_json::json!("openai/text-embedding-3-small")
    );
    assert_eq!(
        registry_req.body["input"],
        serde_json::json!(["hello openrouter embedding"])
    );
    assert_eq!(registry_req.body["dimensions"], serde_json::json!(512));
    assert_eq!(
        registry_req.body["encoding_format"],
        serde_json::json!("base64")
    );
    assert_eq!(
        registry_req.body["user"],
        serde_json::json!("compat-user-3")
    );
}

#[tokio::test]
async fn openrouter_registry_embedding_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let openrouter_transport = CaptureTransport::default();

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("openrouter"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "openrouter",
            "ctx-key",
            "https://example.com/openrouter/v1",
            Arc::new(openrouter_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .embedding_model("openrouter:openai/text-embedding-3-small")
        .expect("build openrouter embedding handle");

    let _ = handle
        .embed_with_config(
            EmbeddingRequest::single("hello openrouter embedding")
                .with_model("openai/text-embedding-3-small")
                .with_dimensions(512)
                .with_encoding_format(EmbeddingFormat::Base64)
                .with_user("compat-user-3"),
        )
        .await;

    let req = openrouter_transport
        .take()
        .expect("captured openrouter embedding request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/openrouter/v1/embeddings");
    assert_eq!(
        req.body["model"],
        serde_json::json!("openai/text-embedding-3-small")
    );
    assert_eq!(
        req.body["input"],
        serde_json::json!(["hello openrouter embedding"])
    );
    assert_eq!(req.body["dimensions"], serde_json::json!(512));
    assert_eq!(req.body["encoding_format"], serde_json::json!("base64"));
    assert_eq!(req.body["user"], serde_json::json!("compat-user-3"));
}

#[tokio::test]
async fn openrouter_registry_rerank_request_is_intentionally_unsupported() {
    let registry_transport = CaptureTransport::default();
    let registry = make_registry("openrouter", Arc::new(registry_transport.clone()));
    let err = match registry.reranking_model("openrouter:openai/gpt-4o") {
        Ok(_) => panic!("openrouter registry rerank handle should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&err);
    assert_capture_transports_unused(&[&registry_transport]);
}

#[tokio::test]
async fn openrouter_siumai_provider_config_audio_family_requests_are_intentionally_unsupported() {
    let siumai_transport = MixedCaptureTransport::default();
    let provider_transport = MixedCaptureTransport::default();
    let config_transport = MixedCaptureTransport::default();

    let model = "openai/gpt-4o";

    let siumai_client = Siumai::builder()
        .openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("openrouter", model, Arc::new(config_transport.clone())).await;

    let tts_request = TtsRequest::new("hello openrouter audio".to_string())
        .with_model(model.to_string())
        .with_voice("alloy".to_string())
        .with_format("mp3".to_string());

    let mut stt_request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
    stt_request.model = Some(model.to_string());
    stt_request = stt_request.with_media_type("audio/mpeg".to_string());

    let errs = vec![
        siumai_client
            .text_to_speech(tts_request.clone())
            .await
            .expect_err("openrouter siumai tts should be unsupported"),
        provider_client
            .text_to_speech(tts_request.clone())
            .await
            .expect_err("openrouter provider tts should be unsupported"),
        config_client
            .text_to_speech(tts_request)
            .await
            .expect_err("openrouter config tts should be unsupported"),
        siumai_client
            .speech_to_text(stt_request.clone())
            .await
            .expect_err("openrouter siumai stt should be unsupported"),
        provider_client
            .speech_to_text(stt_request.clone())
            .await
            .expect_err("openrouter provider stt should be unsupported"),
        config_client
            .speech_to_text(stt_request)
            .await
            .expect_err("openrouter config stt should be unsupported"),
    ];

    for err in errs {
        assert_unsupported_operation(&err);
    }

    assert!(siumai_client.as_speech_capability().is_none());
    assert!(provider_client.as_speech_capability().is_none());
    assert!(config_client.as_speech_capability().is_none());
    assert!(siumai_client.as_transcription_capability().is_none());
    assert!(provider_client.as_transcription_capability().is_none());
    assert!(config_client.as_transcription_capability().is_none());

    assert_mixed_capture_transports_unused(&[
        &siumai_transport,
        &provider_transport,
        &config_transport,
    ]);
}

#[tokio::test]
async fn openrouter_registry_audio_family_requests_are_intentionally_unsupported() {
    let registry_transport = MixedCaptureTransport::default();
    let registry = make_registry("openrouter", Arc::new(registry_transport.clone()));

    let tts_err = match registry.speech_model("openrouter:openai/gpt-4o") {
        Ok(_) => panic!("openrouter registry speech handle should be unsupported"),
        Err(err) => err,
    };
    let stt_err = match registry.transcription_model("openrouter:openai/gpt-4o") {
        Ok(_) => panic!("openrouter registry transcription handle should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&tts_err);
    assert_unsupported_operation(&stt_err);
    assert_mixed_capture_transports_unused(&[&registry_transport]);
}

#[tokio::test]
async fn perplexity_siumai_provider_config_non_text_family_requests_are_intentionally_unsupported()
{
    let siumai_transport = MixedCaptureTransport::default();
    let provider_transport = MixedCaptureTransport::default();
    let config_transport = MixedCaptureTransport::default();

    let model = "sonar";

    let siumai_client = Siumai::builder()
        .openai()
        .perplexity()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .perplexity()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("perplexity", model, Arc::new(config_transport.clone())).await;

    let rerank_err = siumai_client
        .rerank(make_rerank_request_with_model(model).with_top_n(1))
        .await
        .expect_err("perplexity rerank should be unsupported");

    let tts_request = TtsRequest::new("hello perplexity audio".to_string())
        .with_model(model.to_string())
        .with_voice("alloy".to_string())
        .with_format("mp3".to_string());

    let mut stt_request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
    stt_request.model = Some(model.to_string());
    stt_request = stt_request.with_media_type("audio/mpeg".to_string());

    let tts_err = siumai_client
        .text_to_speech(tts_request)
        .await
        .expect_err("perplexity tts should be unsupported");
    let stt_err = siumai_client
        .speech_to_text(stt_request)
        .await
        .expect_err("perplexity stt should be unsupported");

    for err in [rerank_err, tts_err, stt_err] {
        assert_unsupported_operation(&err);
    }

    assert_no_deferred_capability_leaks(&siumai_client);
    assert_no_deferred_capability_leaks(&provider_client);
    assert_no_deferred_capability_leaks(&config_client);
    assert!(siumai_client.as_speech_capability().is_none());
    assert!(provider_client.as_speech_capability().is_none());
    assert!(config_client.as_speech_capability().is_none());
    assert!(siumai_client.as_transcription_capability().is_none());
    assert!(provider_client.as_transcription_capability().is_none());
    assert!(config_client.as_transcription_capability().is_none());

    assert_mixed_capture_transports_unused(&[
        &siumai_transport,
        &provider_transport,
        &config_transport,
    ]);
}

#[tokio::test]
async fn perplexity_registry_non_text_family_requests_are_intentionally_unsupported() {
    let registry_transport = MixedCaptureTransport::default();
    let registry = make_registry("perplexity", Arc::new(registry_transport.clone()));

    let embedding_err = match registry.embedding_model("perplexity:sonar") {
        Ok(_) => panic!("perplexity registry embedding handle should be unsupported"),
        Err(err) => err,
    };
    let image_err = match registry.image_model("perplexity:sonar") {
        Ok(_) => panic!("perplexity registry image handle should be unsupported"),
        Err(err) => err,
    };
    let rerank_err = match registry.reranking_model("perplexity:sonar") {
        Ok(_) => panic!("perplexity registry rerank handle should be unsupported"),
        Err(err) => err,
    };
    let speech_err = match registry.speech_model("perplexity:sonar") {
        Ok(_) => panic!("perplexity registry speech handle should be unsupported"),
        Err(err) => err,
    };
    let transcription_err = match registry.transcription_model("perplexity:sonar") {
        Ok(_) => panic!("perplexity registry transcription handle should be unsupported"),
        Err(err) => err,
    };

    for err in [
        embedding_err,
        image_err,
        rerank_err,
        speech_err,
        transcription_err,
    ] {
        assert_unsupported_operation(&err);
    }

    assert_mixed_capture_transports_unused(&[&registry_transport]);
}

#[tokio::test]
async fn moonshotai_registry_non_text_family_requests_are_intentionally_unsupported() {
    let registry_transport = MixedCaptureTransport::default();
    let registry = make_registry("moonshotai", Arc::new(registry_transport.clone()));

    let embedding_err = match registry.embedding_model("moonshotai:kimi-k2.5") {
        Ok(_) => panic!("moonshotai registry embedding handle should be unsupported"),
        Err(err) => err,
    };
    let image_err = match registry.image_model("moonshotai:kimi-k2.5") {
        Ok(_) => panic!("moonshotai registry image handle should be unsupported"),
        Err(err) => err,
    };
    let rerank_err = match registry.reranking_model("moonshotai:kimi-k2.5") {
        Ok(_) => panic!("moonshotai registry rerank handle should be unsupported"),
        Err(err) => err,
    };
    let speech_err = match registry.speech_model("moonshotai:kimi-k2.5") {
        Ok(_) => panic!("moonshotai registry speech handle should be unsupported"),
        Err(err) => err,
    };
    let transcription_err = match registry.transcription_model("moonshotai:kimi-k2.5") {
        Ok(_) => panic!("moonshotai registry transcription handle should be unsupported"),
        Err(err) => err,
    };

    for err in [
        embedding_err,
        image_err,
        rerank_err,
        speech_err,
        transcription_err,
    ] {
        assert_unsupported_operation(&err);
    }

    assert_mixed_capture_transports_unused(&[&registry_transport]);
}

#[test]
fn moonshotai_package_settings_preserve_supported_provider_inputs() {
    let config = siumai::provider_ext::moonshotai::MoonshotAIProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/moonshot")
        .with_header("x-test", "1")
        .into_config_for_model("kimi-k2.5")
        .expect("settings into config");

    assert_eq!(config.provider_id, "moonshotai");
    assert_eq!(config.base_url, "https://example.com/moonshot");
    assert_eq!(config.common_params.model, "kimi-k2.5");
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
}
#[tokio::test]
async fn moonshotai_public_completion_family_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "kimi-k2.5";

    let siumai_client = Siumai::builder()
        .moonshotai()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai moonshotai client");

    let provider_client = Provider::moonshotai()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider moonshotai client");

    let config_client =
        make_config_client("moonshotai", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("moonshotai", Arc::new(registry_transport.clone()));

    assert!(!siumai_client.capabilities().supports("completion"));
    assert!(!provider_client.capabilities().supports("completion"));
    assert!(!config_client.capabilities().supports("completion"));
    assert!(siumai_client.as_completion_capability().is_none());
    assert!(provider_client.as_completion_capability().is_none());
    assert!(config_client.as_completion_capability().is_none());

    let completion_err = match registry.completion_model(&format!("moonshotai:{model}")) {
        Ok(_) => panic!("moonshotai registry completion handle should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&completion_err);
    assert_capture_transports_unused(&[
        &siumai_transport,
        &provider_transport,
        &config_transport,
        &registry_transport,
    ]);
}

#[tokio::test]
async fn moonshotai_top_level_builder_chat_request_matches_config_registry_path() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "kimi-k2-thinking";

    let siumai_client = Siumai::builder()
        .moonshotai()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai moonshotai client");

    let provider_client = Provider::moonshotai()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider moonshotai client");

    let config_client =
        make_config_client("moonshotai", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("moonshotai", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model(&format!("moonshotai:{model}"))
        .expect("build registry moonshotai model");

    let request = make_chat_request_with_model(model).with_moonshotai_options(
        MoonshotAIChatOptions::new()
            .with_thinking(
                MoonshotAIThinkingConfig::new()
                    .with_type(MoonshotAIThinkingType::Enabled)
                    .with_budget_tokens(2048),
            )
            .with_reasoning_history(MoonshotAIReasoningHistory::Interleaved),
    );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://api.moonshot.ai/v1/chat/completions"
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["reasoning_history"],
        serde_json::json!("interleaved")
    );
    assert_eq!(
        siumai_req.body["thinking"],
        serde_json::json!({
            "type": "enabled",
            "budget_tokens": 2048
        })
    );
    assert!(siumai_req.body.get("reasoningHistory").is_none());
    assert!(siumai_req.body["thinking"].get("budgetTokens").is_none());
}

#[tokio::test]
async fn perplexity_public_completion_family_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "sonar";

    let siumai_client = Siumai::builder()
        .perplexity()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai perplexity client");

    let provider_client = Provider::perplexity()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider perplexity client");

    let config_client =
        make_config_client("perplexity", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("perplexity", Arc::new(registry_transport.clone()));

    assert!(!siumai_client.capabilities().supports("completion"));
    assert!(!provider_client.capabilities().supports("completion"));
    assert!(!config_client.capabilities().supports("completion"));
    assert!(siumai_client.as_completion_capability().is_none());
    assert!(provider_client.as_completion_capability().is_none());
    assert!(config_client.as_completion_capability().is_none());

    let completion_err = match registry.completion_model(&format!("perplexity:{model}")) {
        Ok(_) => panic!("perplexity registry completion handle should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&completion_err);
    assert_capture_transports_unused(&[
        &siumai_transport,
        &provider_transport,
        &config_transport,
        &registry_transport,
    ]);
}

#[test]
fn perplexity_package_settings_preserve_supported_provider_inputs() {
    let config = siumai::provider_ext::perplexity::PerplexityProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/perplexity")
        .with_header("x-test", "1")
        .into_config_for_model("sonar")
        .expect("settings into config");

    assert_eq!(config.provider_id, "perplexity");
    assert_eq!(config.base_url, "https://example.com/perplexity");
    assert_eq!(config.common_params.model, "sonar");
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
}
#[tokio::test]
async fn perplexity_top_level_builder_chat_request_matches_config_registry_path() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "sonar";

    let siumai_client = Siumai::builder()
        .perplexity()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai perplexity client");

    let provider_client = Provider::perplexity()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider perplexity client");

    let config_client =
        make_config_client("perplexity", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("perplexity", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model(&format!("perplexity:{model}"))
        .expect("build registry perplexity model");

    let request = make_chat_request_with_model(model).with_perplexity_options(
        PerplexityOptions::new()
            .with_search_mode(PerplexitySearchMode::Academic)
            .with_search_recency_filter(PerplexitySearchRecencyFilter::Month)
            .with_return_images(true),
    );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "https://api.perplexity.ai/chat/completions");
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["search_mode"],
        serde_json::json!("academic")
    );
    assert_eq!(
        siumai_req.body["search_recency_filter"],
        serde_json::json!("month")
    );
    assert_eq!(siumai_req.body["return_images"], serde_json::json!(true));
}

#[tokio::test]
async fn perplexity_top_level_builder_chat_stream_request_matches_config_registry_path() {
    use futures_util::StreamExt;

    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "sonar";

    let siumai_client = Siumai::builder()
        .perplexity()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai perplexity client");

    let provider_client = Provider::perplexity()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider perplexity client");

    let config_client =
        make_config_client("perplexity", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("perplexity", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model(&format!("perplexity:{model}"))
        .expect("build registry perplexity model");

    let request = make_chat_request_with_model(model).with_perplexity_options(
        PerplexityOptions::new()
            .with_search_mode(PerplexitySearchMode::Academic)
            .with_search_recency_filter(PerplexitySearchRecencyFilter::Month)
            .with_return_images(true),
    );

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai perplexity stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider perplexity stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config perplexity stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry perplexity stream ok");

    let _ = siumai_stream.next().await;
    let _ = provider_stream.next().await;
    let _ = config_stream.next().await;
    let _ = registry_stream.next().await;

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai perplexity stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider perplexity stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config perplexity stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry perplexity stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "https://api.perplexity.ai/chat/completions");
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["search_mode"],
        serde_json::json!("academic")
    );
    assert_eq!(
        siumai_req.body["search_recency_filter"],
        serde_json::json!("month")
    );
    assert_eq!(siumai_req.body["return_images"], serde_json::json!(true));
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn perplexity_siumai_provider_config_chat_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .perplexity()
        .api_key("test-key")
        .model("sonar")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .perplexity()
        .api_key("test-key")
        .model("sonar")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("perplexity", "sonar", Arc::new(config_transport.clone())).await;
    let registry = make_registry("perplexity", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("perplexity:sonar")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let request = ChatRequest::builder()
        .model("sonar")
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({ "type": "object", "properties": {} }),
        )])
        .tool_choice(ToolChoice::None)
        .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
        .build()
        .with_perplexity_options(
            PerplexityOptions::new()
                .with_search_mode(PerplexitySearchMode::Academic)
                .with_search_recency_filter(PerplexitySearchRecencyFilter::Month)
                .with_return_images(true)
                .with_search_context_size(PerplexitySearchContextSize::High)
                .with_user_location(PerplexityUserLocation::new().with_country("US"))
                .with_param("someVendorParam", serde_json::json!(true)),
        );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "https://api.perplexity.ai/chat/completions");
    assert_eq!(siumai_req.body["model"], serde_json::json!("sonar"));
    assert_eq!(
        siumai_req.body["search_mode"],
        serde_json::json!("academic")
    );
    assert_eq!(
        siumai_req.body["search_recency_filter"],
        serde_json::json!("month")
    );
    assert_eq!(siumai_req.body["return_images"], serde_json::json!(true));
    assert_eq!(
        siumai_req.body["web_search_options"]["search_context_size"],
        serde_json::json!("high")
    );
    assert_eq!(
        siumai_req.body["web_search_options"]["user_location"]["country"],
        serde_json::json!("US")
    );
    assert_eq!(siumai_req.body["someVendorParam"], serde_json::json!(true));
    assert_eq!(siumai_req.body["tool_choice"], serde_json::json!("none"));
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
async fn perplexity_siumai_provider_config_chat_stream_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .perplexity()
        .api_key("test-key")
        .model("sonar")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .perplexity()
        .api_key("test-key")
        .model("sonar")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("perplexity", "sonar", Arc::new(config_transport.clone())).await;
    let registry = make_registry("perplexity", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("perplexity:sonar")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let request = ChatRequest::builder()
        .model("sonar")
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({ "type": "object", "properties": {} }),
        )])
        .tool_choice(ToolChoice::None)
        .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
        .build()
        .with_perplexity_options(
            PerplexityOptions::new()
                .with_search_mode(PerplexitySearchMode::Academic)
                .with_search_recency_filter(PerplexitySearchRecencyFilter::Month)
                .with_return_images(true)
                .with_search_context_size(PerplexitySearchContextSize::High)
                .with_user_location(PerplexityUserLocation::new().with_country("US"))
                .with_param("someVendorParam", serde_json::json!(true)),
        );

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    use futures_util::StreamExt;
    let _ = siumai_stream.next().await;
    let _ = provider_stream.next().await;
    let _ = config_stream.next().await;
    let _ = registry_stream.next().await;

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
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(siumai_req.url, "https://api.perplexity.ai/chat/completions");
    assert_eq!(siumai_req.body["model"], serde_json::json!("sonar"));
    assert_eq!(
        siumai_req.body["search_mode"],
        serde_json::json!("academic")
    );
    assert_eq!(
        siumai_req.body["search_recency_filter"],
        serde_json::json!("month")
    );
    assert_eq!(siumai_req.body["return_images"], serde_json::json!(true));
    assert_eq!(
        siumai_req.body["web_search_options"]["search_context_size"],
        serde_json::json!("high")
    );
    assert_eq!(
        siumai_req.body["web_search_options"]["user_location"]["country"],
        serde_json::json!("US")
    );
    assert_eq!(siumai_req.body["someVendorParam"], serde_json::json!(true));
    assert_eq!(siumai_req.body["tool_choice"], serde_json::json!("none"));
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
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn perplexity_siumai_provider_config_chat_response_metadata_are_equivalent() {
    let response_json = serde_json::json!({
        "id": "chatcmpl-perplexity-test",
        "object": "chat.completion",
        "created": 1_718_345_013,
        "model": "sonar",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Rust async tooling kept improving across the ecosystem."
                },
                "finish_reason": "stop"
            }
        ],
        "citations": ["https://example.com/rust"],
        "images": [
            {
                "image_url": "https://images.example.com/rust.png",
                "origin_url": "https://example.com/rust",
                "height": 900,
                "width": 1600
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 17,
            "total_tokens": 28,
            "citation_tokens": 7,
            "num_search_queries": 2,
            "reasoning_tokens": 3
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .openai()
        .perplexity()
        .api_key("test-key")
        .model("sonar")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .perplexity()
        .api_key("test-key")
        .model("sonar")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("perplexity", "sonar", Arc::new(config_transport.clone())).await;

    let request = ChatRequest::builder()
        .model("sonar")
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_perplexity_options(PerplexityOptions::new().with_return_images(true));

    let siumai_resp = siumai_client
        .chat_request(request.clone())
        .await
        .expect("siumai response ok");
    let provider_resp = provider_client
        .chat_request(request.clone())
        .await
        .expect("provider response ok");
    let config_resp = config_client
        .chat_request(request)
        .await
        .expect("config response ok");

    let siumai_root = siumai_resp
        .provider_metadata
        .as_ref()
        .expect("siumai provider metadata");
    let provider_root = provider_resp
        .provider_metadata
        .as_ref()
        .expect("provider provider metadata");
    let config_root = config_resp
        .provider_metadata
        .as_ref()
        .expect("config provider metadata");

    assert!(siumai_root.get("perplexity").is_some());
    assert!(provider_root.get("perplexity").is_some());
    assert!(config_root.get("perplexity").is_some());
    assert!(siumai_root.get("openai_compatible").is_none());
    assert!(provider_root.get("openai_compatible").is_none());
    assert!(config_root.get("openai_compatible").is_none());

    let siumai_meta = siumai_resp
        .perplexity_metadata()
        .expect("siumai perplexity metadata");
    let provider_meta = provider_resp
        .perplexity_metadata()
        .expect("provider perplexity metadata");
    let config_meta = config_resp
        .perplexity_metadata()
        .expect("config perplexity metadata");

    assert_eq!(
        siumai_resp.content_text(),
        Some("Rust async tooling kept improving across the ecosystem.")
    );
    assert_eq!(
        provider_resp.content_text(),
        Some("Rust async tooling kept improving across the ecosystem.")
    );
    assert_eq!(
        config_resp.content_text(),
        Some("Rust async tooling kept improving across the ecosystem.")
    );
    assert_eq!(
        siumai_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        provider_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        config_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );

    assert_eq!(
        siumai_meta.citations.as_ref(),
        Some(&vec!["https://example.com/rust".to_string()])
    );
    assert_eq!(
        provider_meta.citations.as_ref(),
        Some(&vec!["https://example.com/rust".to_string()])
    );
    assert_eq!(
        config_meta.citations.as_ref(),
        Some(&vec!["https://example.com/rust".to_string()])
    );
    assert_eq!(siumai_meta.images.as_ref().map(Vec::len), Some(1));
    assert_eq!(provider_meta.images.as_ref().map(Vec::len), Some(1));
    assert_eq!(config_meta.images.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        siumai_meta
            .images
            .as_ref()
            .and_then(|images| images.first())
            .map(|image| image.image_url.as_str()),
        Some("https://images.example.com/rust.png")
    );
    assert_eq!(
        provider_meta
            .images
            .as_ref()
            .and_then(|images| images.first())
            .map(|image| image.image_url.as_str()),
        Some("https://images.example.com/rust.png")
    );
    assert_eq!(
        config_meta
            .images
            .as_ref()
            .and_then(|images| images.first())
            .map(|image| image.image_url.as_str()),
        Some("https://images.example.com/rust.png")
    );
    assert_eq!(
        siumai_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.citation_tokens),
        Some(7)
    );
    assert_eq!(
        provider_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.citation_tokens),
        Some(7)
    );
    assert_eq!(
        config_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.citation_tokens),
        Some(7)
    );
    assert_eq!(
        siumai_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.num_search_queries),
        Some(2)
    );
    assert_eq!(
        provider_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.num_search_queries),
        Some(2)
    );
    assert_eq!(
        config_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.num_search_queries),
        Some(2)
    );
    assert_eq!(
        siumai_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.reasoning_tokens),
        Some(3)
    );
    assert_eq!(
        provider_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.reasoning_tokens),
        Some(3)
    );
    assert_eq!(
        config_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.reasoning_tokens),
        Some(3)
    );
    assert_eq!(siumai_meta.extra.get("citations"), None);
    assert_eq!(provider_meta.extra.get("citations"), None);
    assert_eq!(config_meta.extra.get("citations"), None);

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://api.perplexity.ai/chat/completions");
    assert_eq!(siumai_req.body["return_images"], serde_json::json!(true));
}

#[tokio::test]
async fn perplexity_siumai_provider_config_stream_end_metadata_are_equivalent() {
    let stream_body = br#"data: {"id":"1","model":"sonar","created":1718345013,"citations":["https://example.com/rust"],"choices":[{"index":0,"delta":{"content":"Rust","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"sonar","created":1718345013,"choices":[{"index":0,"delta":{"content":" ecosystem","role":null},"finish_reason":"stop"}],"images":[{"image_url":"https://images.example.com/rust.png","origin_url":"https://example.com/rust","height":900,"width":1600}],"usage":{"prompt_tokens":11,"completion_tokens":17,"total_tokens":28,"citation_tokens":7,"num_search_queries":2,"reasoning_tokens":3}}

data: [DONE]

"#
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .openai()
        .perplexity()
        .api_key("test-key")
        .model("sonar")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .perplexity()
        .api_key("test-key")
        .model("sonar")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("perplexity", "sonar", Arc::new(config_transport.clone())).await;

    let request = ChatRequest::builder()
        .model("sonar")
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_perplexity_options(PerplexityOptions::new().with_return_images(true));

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

    use futures_util::StreamExt;

    let mut siumai_end = None;
    let mut provider_end = None;
    let mut config_end = None;

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
    while let Some(event) = config_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            config_end = Some(response);
            break;
        }
    }

    let siumai_resp = siumai_end.expect("siumai stream end");
    let provider_resp = provider_end.expect("provider stream end");
    let config_resp = config_end.expect("config stream end");

    let siumai_root = siumai_resp
        .provider_metadata
        .as_ref()
        .expect("siumai provider metadata");
    let provider_root = provider_resp
        .provider_metadata
        .as_ref()
        .expect("provider provider metadata");
    let config_root = config_resp
        .provider_metadata
        .as_ref()
        .expect("config provider metadata");

    assert!(siumai_root.get("perplexity").is_some());
    assert!(provider_root.get("perplexity").is_some());
    assert!(config_root.get("perplexity").is_some());
    assert!(siumai_root.get("openai_compatible").is_none());
    assert!(provider_root.get("openai_compatible").is_none());
    assert!(config_root.get("openai_compatible").is_none());

    let siumai_meta = siumai_resp
        .perplexity_metadata()
        .expect("siumai perplexity metadata");
    let provider_meta = provider_resp
        .perplexity_metadata()
        .expect("provider perplexity metadata");
    let config_meta = config_resp
        .perplexity_metadata()
        .expect("config perplexity metadata");

    assert_eq!(siumai_resp.content_text(), Some("Rust ecosystem"));
    assert_eq!(provider_resp.content_text(), Some("Rust ecosystem"));
    assert_eq!(config_resp.content_text(), Some("Rust ecosystem"));
    assert_eq!(
        siumai_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        provider_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        config_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        siumai_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(28)
    );
    assert_eq!(
        provider_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(28)
    );
    assert_eq!(
        config_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(28)
    );

    assert_eq!(
        siumai_meta.citations.as_ref(),
        Some(&vec!["https://example.com/rust".to_string()])
    );
    assert_eq!(
        provider_meta.citations.as_ref(),
        Some(&vec!["https://example.com/rust".to_string()])
    );
    assert_eq!(
        config_meta.citations.as_ref(),
        Some(&vec!["https://example.com/rust".to_string()])
    );
    assert_eq!(siumai_meta.images.as_ref().map(Vec::len), Some(1));
    assert_eq!(provider_meta.images.as_ref().map(Vec::len), Some(1));
    assert_eq!(config_meta.images.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        siumai_meta
            .images
            .as_ref()
            .and_then(|images| images.first())
            .map(|image| image.image_url.as_str()),
        Some("https://images.example.com/rust.png")
    );
    assert_eq!(
        provider_meta
            .images
            .as_ref()
            .and_then(|images| images.first())
            .map(|image| image.image_url.as_str()),
        Some("https://images.example.com/rust.png")
    );
    assert_eq!(
        config_meta
            .images
            .as_ref()
            .and_then(|images| images.first())
            .map(|image| image.image_url.as_str()),
        Some("https://images.example.com/rust.png")
    );
    assert_eq!(
        siumai_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.citation_tokens),
        Some(7)
    );
    assert_eq!(
        provider_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.citation_tokens),
        Some(7)
    );
    assert_eq!(
        config_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.citation_tokens),
        Some(7)
    );
    assert_eq!(
        siumai_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.num_search_queries),
        Some(2)
    );
    assert_eq!(
        provider_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.num_search_queries),
        Some(2)
    );
    assert_eq!(
        config_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.num_search_queries),
        Some(2)
    );
    assert_eq!(
        siumai_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.reasoning_tokens),
        Some(3)
    );
    assert_eq!(
        provider_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.reasoning_tokens),
        Some(3)
    );
    assert_eq!(
        config_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.reasoning_tokens),
        Some(3)
    );
    assert_eq!(siumai_meta.extra.get("citations"), None);
    assert_eq!(provider_meta.extra.get("citations"), None);
    assert_eq!(config_meta.extra.get("citations"), None);

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
    assert_eq!(siumai_req.url, "https://api.perplexity.ai/chat/completions");
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(siumai_req.body["return_images"], serde_json::json!(true));
}

#[tokio::test]
async fn perplexity_registry_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let perplexity_transport = CaptureTransport::default();

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("perplexity"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "perplexity",
            "ctx-key",
            "https://example.com/perplexity",
            Arc::new(perplexity_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .language_model("perplexity:sonar")
        .expect("build perplexity handle");

    let _ = handle
        .chat_request(
            make_chat_request_with_model("sonar").with_perplexity_options(
                PerplexityOptions::new()
                    .with_search_mode(PerplexitySearchMode::Academic)
                    .with_param("someVendorParam", serde_json::json!(true)),
            ),
        )
        .await;

    let req = perplexity_transport
        .take()
        .expect("captured perplexity request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/perplexity/chat/completions");
    assert_eq!(req.body["model"], serde_json::json!("sonar"));
    assert_eq!(req.body["search_mode"], serde_json::json!("academic"));
    assert_eq!(req.body["someVendorParam"], serde_json::json!(true));
}

#[tokio::test]
async fn perplexity_registry_stream_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let perplexity_transport = CaptureTransport::default();

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("perplexity"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "perplexity",
            "ctx-key",
            "https://example.com/perplexity",
            Arc::new(perplexity_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .language_model("perplexity:sonar")
        .expect("build perplexity handle");

    let _ = handle
        .chat_stream_request(
            make_chat_request_with_model("sonar").with_perplexity_options(
                PerplexityOptions::new()
                    .with_search_mode(PerplexitySearchMode::Academic)
                    .with_param("someVendorParam", serde_json::json!(true)),
            ),
        )
        .await;

    let req = perplexity_transport
        .take_stream()
        .expect("captured perplexity stream request");
    assert!(global_transport.take().is_none());
    assert!(global_transport.take_stream().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(
        header_value(&req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(req.url, "https://example.com/perplexity/chat/completions");
    assert_eq!(req.body["model"], serde_json::json!("sonar"));
    assert_eq!(req.body["stream"], serde_json::json!(true));
    assert_eq!(req.body["search_mode"], serde_json::json!("academic"));
    assert_eq!(req.body["someVendorParam"], serde_json::json!(true));
}

#[tokio::test]
async fn perplexity_registry_override_stream_end_metadata_preserves_vendor_namespace() {
    let stream_body = br#"data: {"id":"1","model":"sonar","created":1718345013,"citations":["https://example.com/rust"],"choices":[{"index":0,"delta":{"content":"Rust","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"sonar","created":1718345013,"choices":[{"index":0,"delta":{"content":" ecosystem","role":null},"finish_reason":"stop"}],"images":[{"image_url":"https://images.example.com/rust.png","origin_url":"https://example.com/rust","height":900,"width":1600}],"usage":{"prompt_tokens":11,"completion_tokens":17,"total_tokens":28,"citation_tokens":7,"num_search_queries":2,"reasoning_tokens":3}}

data: [DONE]

"#
        .to_vec();

    let global_transport = CaptureTransport::default();
    let perplexity_transport = SseSuccessTransport::new(stream_body);

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("perplexity"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "perplexity",
            "ctx-key",
            "https://example.com/perplexity",
            Arc::new(perplexity_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let mut stream = registry
        .language_model("perplexity:sonar")
        .expect("build perplexity handle")
        .chat_stream_request(
            make_chat_request_with_model("sonar")
                .with_perplexity_options(PerplexityOptions::new().with_return_images(true)),
        )
        .await
        .expect("registry stream ok");

    use futures_util::StreamExt;
    let mut stream_end = None;
    while let Some(event) = stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            stream_end = Some(response);
            break;
        }
    }

    let response = stream_end.expect("registry stream end");
    let root = response
        .provider_metadata
        .as_ref()
        .expect("registry provider metadata");
    assert!(root.get("perplexity").is_some());
    assert!(root.get("openai_compatible").is_none());

    let metadata = response.perplexity_metadata().expect("perplexity metadata");
    assert_eq!(response.content_text(), Some("Rust ecosystem"));
    assert_eq!(
        response
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(28)
    );
    assert_eq!(
        metadata.citations.as_ref(),
        Some(&vec!["https://example.com/rust".to_string()])
    );
    assert_eq!(metadata.images.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        metadata
            .usage
            .as_ref()
            .and_then(|usage| usage.citation_tokens),
        Some(7)
    );

    let req = perplexity_transport
        .take_stream()
        .expect("captured stream request");
    assert!(global_transport.take().is_none());
    assert!(global_transport.take_stream().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/perplexity/chat/completions");
}

#[tokio::test]
async fn perplexity_registry_override_chat_response_metadata_preserves_vendor_namespace() {
    let response_json = serde_json::json!({
        "id": "chatcmpl-perplexity-test",
        "object": "chat.completion",
        "created": 1_718_345_013,
        "model": "sonar",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Rust async tooling kept improving across the ecosystem."
                },
                "finish_reason": "stop"
            }
        ],
        "citations": ["https://example.com/rust"],
        "images": [
            {
                "image_url": "https://images.example.com/rust.png",
                "origin_url": "https://example.com/rust",
                "height": 900,
                "width": 1600
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 17,
            "total_tokens": 28,
            "citation_tokens": 7,
            "num_search_queries": 2,
            "reasoning_tokens": 3
        }
    });

    let global_transport = CaptureTransport::default();
    let perplexity_transport = JsonSuccessTransport::new(response_json);

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("perplexity"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "perplexity",
            "ctx-key",
            "https://example.com/perplexity",
            Arc::new(perplexity_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let response = registry
        .language_model("perplexity:sonar")
        .expect("build perplexity handle")
        .chat_request(
            make_chat_request_with_model("sonar")
                .with_perplexity_options(PerplexityOptions::new().with_return_images(true)),
        )
        .await
        .expect("registry response ok");

    let root = response
        .provider_metadata
        .as_ref()
        .expect("registry provider metadata");
    assert!(root.get("perplexity").is_some());
    assert!(root.get("openai_compatible").is_none());

    let metadata = response.perplexity_metadata().expect("perplexity metadata");
    assert_eq!(
        response.content_text(),
        Some("Rust async tooling kept improving across the ecosystem.")
    );
    assert_eq!(
        response
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(28)
    );
    assert_eq!(
        metadata.citations.as_ref(),
        Some(&vec!["https://example.com/rust".to_string()])
    );
    assert_eq!(metadata.images.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        metadata
            .usage
            .as_ref()
            .and_then(|usage| usage.citation_tokens),
        Some(7)
    );

    let req = perplexity_transport.take().expect("captured request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/perplexity/chat/completions");
}

#[tokio::test]
async fn perplexity_registry_chat_request_with_explicit_request_model_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let default_model = "sonar";
    let request_model = "sonar-pro";

    let config_client = make_config_client(
        "perplexity",
        default_model,
        Arc::new(config_transport.clone()),
    )
    .await;
    let registry = make_registry("perplexity", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("perplexity:sonar")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(request_model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_perplexity_options(
            PerplexityOptions::new()
                .with_search_mode(PerplexitySearchMode::Academic)
                .with_return_images(true)
                .with_param("someVendorParam", serde_json::json!(true)),
        );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.body["model"], serde_json::json!(request_model));
    assert_eq!(
        registry_req.body["search_mode"],
        serde_json::json!("academic")
    );
    assert_eq!(registry_req.body["return_images"], serde_json::json!(true));
    assert_eq!(
        registry_req.body["someVendorParam"],
        serde_json::json!(true)
    );
}

#[tokio::test]
async fn perplexity_registry_chat_stream_request_with_explicit_request_model_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let default_model = "sonar";
    let request_model = "sonar-pro";

    let config_client = make_config_client(
        "perplexity",
        default_model,
        Arc::new(config_transport.clone()),
    )
    .await;
    let registry = make_registry("perplexity", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("perplexity:sonar")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(request_model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_perplexity_options(
            PerplexityOptions::new()
                .with_search_mode(PerplexitySearchMode::Academic)
                .with_return_images(true)
                .with_param("someVendorParam", serde_json::json!(true)),
        );

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
    assert_eq!(registry_req.body["return_images"], serde_json::json!(true));
    assert_eq!(
        registry_req.body["someVendorParam"],
        serde_json::json!(true)
    );
}

#[tokio::test]
async fn perplexity_registry_stream_end_metadata_match_config_path() {
    let stream_body = br#"data: {"id":"1","model":"sonar","created":1718345013,"citations":["https://example.com/rust"],"choices":[{"index":0,"delta":{"content":"Rust","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"sonar","created":1718345013,"choices":[{"index":0,"delta":{"content":" ecosystem","role":null},"finish_reason":"stop"}],"images":[{"image_url":"https://images.example.com/rust.png","origin_url":"https://example.com/rust","height":900,"width":1600}],"usage":{"prompt_tokens":11,"completion_tokens":17,"total_tokens":28,"citation_tokens":7,"num_search_queries":2,"reasoning_tokens":3}}

data: [DONE]

"#
        .to_vec();

    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let config_client =
        make_config_client("perplexity", "sonar", Arc::new(config_transport.clone())).await;
    let registry = make_registry("perplexity", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("perplexity:sonar")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model("sonar")
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_perplexity_options(PerplexityOptions::new().with_return_images(true));

    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    use futures_util::StreamExt;

    let mut config_end = None;
    let mut registry_end = None;

    while let Some(event) = config_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            config_end = Some(response);
            break;
        }
    }
    while let Some(event) = registry_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            registry_end = Some(response);
            break;
        }
    }

    let config_resp = config_end.expect("config stream end");
    let registry_resp = registry_end.expect("registry stream end");

    let config_root = config_resp
        .provider_metadata
        .as_ref()
        .expect("config provider metadata");
    let registry_root = registry_resp
        .provider_metadata
        .as_ref()
        .expect("registry provider metadata");

    assert!(config_root.get("perplexity").is_some());
    assert!(registry_root.get("perplexity").is_some());
    assert!(config_root.get("openai_compatible").is_none());
    assert!(registry_root.get("openai_compatible").is_none());

    let config_meta = config_resp
        .perplexity_metadata()
        .expect("config perplexity metadata");
    let registry_meta = registry_resp
        .perplexity_metadata()
        .expect("registry perplexity metadata");

    assert_eq!(config_resp.content_text(), Some("Rust ecosystem"));
    assert_eq!(registry_resp.content_text(), Some("Rust ecosystem"));
    assert_eq!(
        config_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        registry_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        config_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(28)
    );
    assert_eq!(
        registry_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(28)
    );
    assert_eq!(
        config_meta.citations.as_ref(),
        Some(&vec!["https://example.com/rust".to_string()])
    );
    assert_eq!(
        registry_meta.citations.as_ref(),
        Some(&vec!["https://example.com/rust".to_string()])
    );
    assert_eq!(config_meta.images.as_ref().map(Vec::len), Some(1));
    assert_eq!(registry_meta.images.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        config_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.citation_tokens),
        Some(7)
    );
    assert_eq!(
        registry_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.citation_tokens),
        Some(7)
    );

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(config_req.url, "https://api.perplexity.ai/chat/completions");
    assert_eq!(
        header_value(&config_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(config_req.body["return_images"], serde_json::json!(true));
}

#[tokio::test]
async fn perplexity_structured_output_synthetic_unknown_stream_end_extracts_across_public_paths() {
    let stream_body = concat!(
            "data: {\"id\":\"1\",\"model\":\"sonar\",\"created\":1718345013,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"{\\\"answer\\\":\\\"hel\",\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"1\",\"model\":\"sonar\",\"created\":1718345013,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\\\"}\"},\"finish_reason\":null}]}\n\n"
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .openai()
        .perplexity()
        .api_key("test-key")
        .model("sonar")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .perplexity()
        .api_key("test-key")
        .model("sonar")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("perplexity", "sonar", Arc::new(config_transport.clone())).await;
    let registry = make_registry("perplexity", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("perplexity:sonar")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let request = ChatRequest::builder()
        .model("sonar")
        .messages(vec![ChatMessage::user("hi").build()])
        .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
        .build()
        .with_perplexity_options(PerplexityOptions::new().with_return_images(true));

    let siumai_value = siumai::structured_output::extract_json_value_from_stream(
        siumai_client
            .chat_stream_request(request.clone())
            .await
            .expect("siumai stream ok"),
    )
    .await
    .expect("siumai structured output");
    let provider_value = siumai::structured_output::extract_json_value_from_stream(
        provider_client
            .chat_stream_request(request.clone())
            .await
            .expect("provider stream ok"),
    )
    .await
    .expect("provider structured output");
    let config_value = siumai::structured_output::extract_json_value_from_stream(
        config_client
            .chat_stream_request(request.clone())
            .await
            .expect("config stream ok"),
    )
    .await
    .expect("config structured output");
    let registry_value = siumai::structured_output::extract_json_value_from_stream(
        registry_model
            .chat_stream_request(request)
            .await
            .expect("registry stream ok"),
    )
    .await
    .expect("registry structured output");

    assert_eq!(siumai_value["answer"], "hello");
    assert_eq!(provider_value["answer"], "hello");
    assert_eq!(config_value["answer"], "hello");
    assert_eq!(registry_value["answer"], "hello");

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
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(siumai_req.url, "https://api.perplexity.ai/chat/completions");
    assert_eq!(siumai_req.body["return_images"], serde_json::json!(true));
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
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
async fn perplexity_structured_output_synthetic_unknown_stream_end_fails_consistently_across_public_paths()
 {
    let stream_body = "data: {\"id\":\"1\",\"model\":\"sonar\",\"created\":1718345013,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"{\\\"answer\\\":\" ,\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"
            .as_bytes()
            .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .openai()
        .perplexity()
        .api_key("test-key")
        .model("sonar")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .perplexity()
        .api_key("test-key")
        .model("sonar")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("perplexity", "sonar", Arc::new(config_transport.clone())).await;
    let registry = make_registry("perplexity", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("perplexity:sonar")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let request = ChatRequest::builder()
        .model("sonar")
        .messages(vec![ChatMessage::user("hi").build()])
        .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
        .build()
        .with_perplexity_options(PerplexityOptions::new().with_return_images(true));

    let siumai_err = siumai::structured_output::extract_json_value_from_stream(
        siumai_client
            .chat_stream_request(request.clone())
            .await
            .expect("siumai stream ok"),
    )
    .await
    .expect_err("siumai interrupted stream should fail");
    let provider_err = siumai::structured_output::extract_json_value_from_stream(
        provider_client
            .chat_stream_request(request.clone())
            .await
            .expect("provider stream ok"),
    )
    .await
    .expect_err("provider interrupted stream should fail");
    let config_err = siumai::structured_output::extract_json_value_from_stream(
        config_client
            .chat_stream_request(request.clone())
            .await
            .expect("config stream ok"),
    )
    .await
    .expect_err("config interrupted stream should fail");
    let registry_err = siumai::structured_output::extract_json_value_from_stream(
        registry_model
            .chat_stream_request(request)
            .await
            .expect("registry stream ok"),
    )
    .await
    .expect_err("registry interrupted stream should fail");

    for err in [siumai_err, provider_err, config_err, registry_err] {
        match err {
            LlmError::ParseError(message) => {
                assert!(message.contains("stream ended before a complete JSON value was produced"))
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
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
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(siumai_req.url, "https://api.perplexity.ai/chat/completions");
    assert_eq!(siumai_req.body["return_images"], serde_json::json!(true));
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
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
async fn perplexity_registry_chat_response_metadata_match_config_path() {
    let response_json = serde_json::json!({
        "id": "chatcmpl-perplexity-test",
        "object": "chat.completion",
        "created": 1_718_345_013,
        "model": "sonar",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Rust async tooling kept improving across the ecosystem."
                },
                "finish_reason": "stop"
            }
        ],
        "citations": ["https://example.com/rust"],
        "images": [
            {
                "image_url": "https://images.example.com/rust.png",
                "origin_url": "https://example.com/rust",
                "height": 900,
                "width": 1600
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 17,
            "total_tokens": 28,
            "citation_tokens": 7,
            "num_search_queries": 2,
            "reasoning_tokens": 3
        }
    });

    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let config_client =
        make_config_client("perplexity", "sonar", Arc::new(config_transport.clone())).await;
    let registry = make_registry("perplexity", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("perplexity:sonar")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model("sonar")
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_perplexity_options(PerplexityOptions::new().with_return_images(true));

    let config_resp = config_client
        .chat_request(request.clone())
        .await
        .expect("config response ok");
    let registry_resp = registry_model
        .chat_request(request)
        .await
        .expect("registry response ok");

    let config_root = config_resp
        .provider_metadata
        .as_ref()
        .expect("config provider metadata");
    let registry_root = registry_resp
        .provider_metadata
        .as_ref()
        .expect("registry provider metadata");

    assert!(config_root.get("perplexity").is_some());
    assert!(registry_root.get("perplexity").is_some());
    assert!(config_root.get("openai_compatible").is_none());
    assert!(registry_root.get("openai_compatible").is_none());

    let config_meta = config_resp
        .perplexity_metadata()
        .expect("config perplexity metadata");
    let registry_meta = registry_resp
        .perplexity_metadata()
        .expect("registry perplexity metadata");

    assert_eq!(
        config_resp.content_text(),
        Some("Rust async tooling kept improving across the ecosystem.")
    );
    assert_eq!(
        registry_resp.content_text(),
        Some("Rust async tooling kept improving across the ecosystem.")
    );
    assert_eq!(
        config_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        registry_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        config_meta.citations.as_ref(),
        Some(&vec!["https://example.com/rust".to_string()])
    );
    assert_eq!(
        registry_meta.citations.as_ref(),
        Some(&vec!["https://example.com/rust".to_string()])
    );
    assert_eq!(config_meta.images.as_ref().map(Vec::len), Some(1));
    assert_eq!(registry_meta.images.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        config_meta
            .images
            .as_ref()
            .and_then(|images| images.first())
            .map(|image| image.image_url.as_str()),
        Some("https://images.example.com/rust.png")
    );
    assert_eq!(
        registry_meta
            .images
            .as_ref()
            .and_then(|images| images.first())
            .map(|image| image.image_url.as_str()),
        Some("https://images.example.com/rust.png")
    );
    assert_eq!(
        config_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.citation_tokens),
        Some(7)
    );
    assert_eq!(
        registry_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.citation_tokens),
        Some(7)
    );
    assert_eq!(
        config_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.num_search_queries),
        Some(2)
    );
    assert_eq!(
        registry_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.num_search_queries),
        Some(2)
    );
    assert_eq!(
        config_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.reasoning_tokens),
        Some(3)
    );
    assert_eq!(
        registry_meta
            .usage
            .as_ref()
            .and_then(|usage| usage.reasoning_tokens),
        Some(3)
    );
    assert_eq!(config_meta.extra.get("citations"), None);
    assert_eq!(registry_meta.extra.get("citations"), None);

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(config_req.url, "https://api.perplexity.ai/chat/completions");
    assert_eq!(config_req.body["return_images"], serde_json::json!(true));
}

#[tokio::test]
async fn openrouter_siumai_provider_config_chat_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .openrouter()
        .api_key("test-key")
        .model("openai/gpt-4o")
        .reasoning(true)
        .reasoning_budget(2048)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .model("openai/gpt-4o")
        .reasoning(true)
        .reasoning_budget(2048)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let provider = siumai::provider_ext::openai_compatible::get_provider_config("openrouter")
        .expect("openrouter provider config");
    let adapter = Arc::new(
        siumai::provider_ext::openai_compatible::ConfigurableAdapter::new(provider.clone()),
    );
    let config = siumai::provider_ext::openai_compatible::OpenAiCompatibleConfig::new(
        "openrouter",
        "test-key",
        &provider.base_url,
        adapter,
    )
    .with_model("openai/gpt-4o")
    .with_reasoning(true)
    .with_reasoning_budget(2048)
    .with_http_transport(Arc::new(config_transport.clone()));
    let config_client =
        siumai::provider_ext::openai_compatible::OpenAiCompatibleClient::from_config(config)
            .await
            .expect("build config client");
    let registry = make_registry_with_global_reasoning_defaults(
        "openrouter",
        Arc::new(registry_transport.clone()),
        true,
        2048,
    );
    let registry_model = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let request = ChatRequest::builder()
        .model("openai/gpt-4o")
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({ "type": "object", "properties": {} }),
        )])
        .tool_choice(ToolChoice::None)
        .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
        .build()
        .with_openrouter_options(
            OpenRouterOptions::new()
                .with_transform(OpenRouterTransform::MiddleOut)
                .with_param("someVendorParam", serde_json::json!(true)),
        );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://openrouter.ai/api/v1/chat/completions"
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!("openai/gpt-4o"));
    assert_eq!(
        siumai_req.body["transforms"],
        serde_json::json!(["middle-out"])
    );
    assert_eq!(siumai_req.body["someVendorParam"], serde_json::json!(true));
    assert_eq!(siumai_req.body["tool_choice"], serde_json::json!("none"));
    assert_eq!(siumai_req.body["enable_reasoning"], serde_json::json!(true));
    assert_eq!(siumai_req.body["reasoning_budget"], serde_json::json!(2048));
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
async fn openrouter_public_paths_use_canonical_openai_compatible_provider_options() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "openai/gpt-4o";

    let siumai_client = Siumai::builder()
        .openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("openrouter", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("openrouter", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build registry language model");

    let mut system = ChatMessage::system("system prompt")
        .with_provider_option(
            "openaiCompatible",
            serde_json::json!({
                "systemTag": "canonical-system"
            }),
        )
        .build();
    system.metadata.custom.insert(
        "openaiCompatible".to_string(),
        serde_json::json!({
            "legacyOnlySystem": true,
            "systemTag": "legacy-system"
        }),
    );

    let user = siumai::prelude::unified::ChatMessage {
        role: siumai::prelude::unified::MessageRole::User,
        content: siumai::prelude::unified::MessageContent::MultiModal(vec![
            siumai::prelude::unified::ContentPart::Text {
                text: "hello".to_string(),
                provider_options: {
                    let mut options = siumai::prelude::unified::ProviderOptionsMap::default();
                    options.insert(
                        "openaiCompatible",
                        serde_json::json!({
                            "userTag": "canonical-user-part"
                        }),
                    );
                    options
                },
                provider_metadata: Some(std::collections::HashMap::from([(
                    "openaiCompatible".to_string(),
                    serde_json::json!({
                        "legacyOnlyUserPart": true,
                        "userTag": "legacy-user-part"
                    }),
                )])),
            },
        ]),
        provider_options: {
            let mut options = siumai::prelude::unified::ProviderOptionsMap::default();
            options.insert(
                "openaiCompatible",
                serde_json::json!({
                    "messageTagShouldBeIgnored": true
                }),
            );
            options
        },
        metadata: siumai::prelude::unified::MessageMetadata {
            custom: std::collections::HashMap::from([(
                "openaiCompatible".to_string(),
                serde_json::json!({
                    "legacyOnlyUserMessage": true,
                    "userTag": "legacy-user-message"
                }),
            )]),
            ..Default::default()
        },
    };

    let tool_call = siumai::prelude::unified::ContentPart::tool_call(
        "call_1",
        "get_weather",
        serde_json::json!({
            "city": "Tokyo"
        }),
        None,
    )
    .with_provider_option(
        "openaiCompatible",
        serde_json::json!({
            "toolCallTag": "canonical-tool-call"
        }),
    );

    let mut assistant = ChatMessage::assistant_with_content(vec![
        siumai::prelude::unified::ContentPart::reasoning("Count carefully. "),
        siumai::prelude::unified::ContentPart::text("There are three r characters."),
        tool_call,
    ])
    .with_provider_option(
        "openaiCompatible",
        serde_json::json!({
            "assistantTag": "canonical-assistant"
        }),
    )
    .build();
    assistant.metadata.custom.insert(
        "openaiCompatible".to_string(),
        serde_json::json!({
            "legacyOnlyAssistant": true,
            "assistantTag": "legacy-assistant"
        }),
    );

    let mut tool_result = ChatMessage::tool_result_json(
        "call_1",
        "get_weather",
        serde_json::json!({
            "temperature": 18
        }),
    )
    .build();
    if let siumai::prelude::unified::MessageContent::MultiModal(parts) = &mut tool_result.content
        && let Some(part) = parts.first_mut()
    {
        part.provider_options_mut()
            .expect("tool result provider options")
            .insert(
                "openaiCompatible",
                serde_json::json!({
                    "toolResultTag": "canonical-tool-result"
                }),
            );

        if let siumai::prelude::unified::ContentPart::ToolResult {
            provider_metadata, ..
        } = part
        {
            *provider_metadata = Some(std::collections::HashMap::from([(
                "openaiCompatible".to_string(),
                serde_json::json!({
                    "legacyOnlyToolResult": true
                }),
            )]));
        }
    }

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![system, user, assistant, tool_result])
        .build();

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://openrouter.ai/api/v1/chat/completions"
    );

    let messages = siumai_req.body["messages"]
        .as_array()
        .expect("expected messages array");
    assert_eq!(messages.len(), 4);

    assert_eq!(
        messages[0]["systemTag"],
        serde_json::json!("canonical-system")
    );
    assert!(messages[0].get("legacyOnlySystem").is_none());

    assert_eq!(
        messages[1]["userTag"],
        serde_json::json!("canonical-user-part")
    );
    assert!(messages[1].get("legacyOnlyUserPart").is_none());
    assert!(messages[1].get("legacyOnlyUserMessage").is_none());
    assert!(messages[1].get("messageTagShouldBeIgnored").is_none());

    assert_eq!(
        messages[2]["assistantTag"],
        serde_json::json!("canonical-assistant")
    );
    assert!(messages[2].get("legacyOnlyAssistant").is_none());
    assert_eq!(
        messages[2]["reasoning_content"],
        serde_json::json!("Count carefully. ")
    );
    assert_eq!(
        messages[2]["tool_calls"][0]["toolCallTag"],
        serde_json::json!("canonical-tool-call")
    );

    assert_eq!(
        messages[3]["toolResultTag"],
        serde_json::json!("canonical-tool-result")
    );
    assert!(messages[3].get("legacyOnlyToolResult").is_none());
}

#[tokio::test]
async fn openrouter_public_paths_preserve_typed_openai_compatible_chat_options() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "openai/gpt-4o";

    let siumai_client = Siumai::builder()
        .openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("openrouter", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("openrouter", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
        .build()
        .with_openai_compatible_options(
            OpenAICompatibleLanguageModelChatOptions::new()
                .with_user("compat-user-123")
                .with_reasoning_effort("high")
                .with_text_verbosity("medium")
                .with_strict_json_schema(false),
        );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://openrouter.ai/api/v1/chat/completions"
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["user"],
        serde_json::json!("compat-user-123")
    );
    assert_eq!(
        siumai_req.body["reasoning_effort"],
        serde_json::json!("high")
    );
    assert_eq!(siumai_req.body["verbosity"], serde_json::json!("medium"));
    assert_eq!(
        siumai_req.body["response_format"],
        serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": false
            }
        })
    );
    assert!(siumai_req.body.get("openaiCompatible").is_none());
    assert!(siumai_req.body.get("reasoningEffort").is_none());
    assert!(siumai_req.body.get("textVerbosity").is_none());
    assert!(siumai_req.body.get("strictJsonSchema").is_none());
}

#[tokio::test]
async fn openrouter_siumai_provider_config_chat_stream_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .openrouter()
        .api_key("test-key")
        .model("openai/gpt-4o")
        .reasoning(true)
        .reasoning_budget(2048)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .model("openai/gpt-4o")
        .reasoning(true)
        .reasoning_budget(2048)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let provider = siumai::provider_ext::openai_compatible::get_provider_config("openrouter")
        .expect("openrouter provider config");
    let adapter = Arc::new(
        siumai::provider_ext::openai_compatible::ConfigurableAdapter::new(provider.clone()),
    );
    let config = siumai::provider_ext::openai_compatible::OpenAiCompatibleConfig::new(
        "openrouter",
        "test-key",
        &provider.base_url,
        adapter,
    )
    .with_model("openai/gpt-4o")
    .with_reasoning(true)
    .with_reasoning_budget(2048)
    .with_http_transport(Arc::new(config_transport.clone()));
    let config_client =
        siumai::provider_ext::openai_compatible::OpenAiCompatibleClient::from_config(config)
            .await
            .expect("build config client");
    let registry = make_registry_with_global_reasoning_defaults(
        "openrouter",
        Arc::new(registry_transport.clone()),
        true,
        2048,
    );
    let registry_model = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let request = ChatRequest::builder()
        .model("openai/gpt-4o")
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({ "type": "object", "properties": {} }),
        )])
        .tool_choice(ToolChoice::None)
        .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
        .build()
        .with_openrouter_options(
            OpenRouterOptions::new()
                .with_transform(OpenRouterTransform::MiddleOut)
                .with_param("someVendorParam", serde_json::json!(true)),
        );

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    use futures_util::StreamExt;
    let _ = siumai_stream.next().await;
    let _ = provider_stream.next().await;
    let _ = config_stream.next().await;
    let _ = registry_stream.next().await;

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
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        siumai_req.url,
        "https://openrouter.ai/api/v1/chat/completions"
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!("openai/gpt-4o"));
    assert_eq!(
        siumai_req.body["transforms"],
        serde_json::json!(["middle-out"])
    );
    assert_eq!(siumai_req.body["someVendorParam"], serde_json::json!(true));
    assert_eq!(siumai_req.body["tool_choice"], serde_json::json!("none"));
    assert_eq!(siumai_req.body["enable_reasoning"], serde_json::json!(true));
    assert_eq!(siumai_req.body["reasoning_budget"], serde_json::json!(2048));
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
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn openrouter_siumai_provider_config_registry_chat_response_provider_metadata_are_equivalent()
{
    let model = "openai/gpt-4o";
    let response_json = serde_json::json!({
        "id": "chatcmpl-openrouter-test",
        "object": "chat.completion",
        "created": 1_718_345_013,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello from openrouter chat"
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "hello",
                            "logprob": -0.1,
                            "bytes": [104, 101, 108, 108, 111],
                            "top_logprobs": []
                        }
                    ]
                }
            }
        ],
        "sources": [
            {
                "id": "src_1",
                "source_type": "url",
                "url": "https://openrouter.ai/docs",
                "title": "OpenRouter Docs",
                "provider_metadata": {
                    "openrouter": {
                        "fileId": "file_123",
                        "containerId": "container_456",
                        "index": 1
                    }
                }
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "total_tokens": 14
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("openrouter", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("openrouter", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build registry language model");

    let mut request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();
    request
        .provider_options_map
        .insert("openrouter", serde_json::json!({ "logprobs": 3 }));

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

    let siumai_root = siumai_resp
        .provider_metadata
        .as_ref()
        .expect("siumai provider metadata");
    let provider_root = provider_resp
        .provider_metadata
        .as_ref()
        .expect("provider provider metadata");
    let config_root = config_resp
        .provider_metadata
        .as_ref()
        .expect("config provider metadata");
    let registry_root = registry_resp
        .provider_metadata
        .as_ref()
        .expect("registry provider metadata");

    assert!(siumai_root.get("openrouter").is_some());
    assert!(provider_root.get("openrouter").is_some());
    assert!(config_root.get("openrouter").is_some());
    assert!(registry_root.get("openrouter").is_some());
    assert!(siumai_root.get("openai_compatible").is_none());
    assert!(provider_root.get("openai_compatible").is_none());
    assert!(config_root.get("openai_compatible").is_none());
    assert!(registry_root.get("openai_compatible").is_none());

    let siumai_meta = siumai_resp
        .openrouter_metadata()
        .expect("siumai openrouter metadata");
    let provider_meta = provider_resp
        .openrouter_metadata()
        .expect("provider openrouter metadata");
    let config_meta = config_resp
        .openrouter_metadata()
        .expect("config openrouter metadata");
    let registry_meta = registry_resp
        .openrouter_metadata()
        .expect("registry openrouter metadata");

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.1,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(
        siumai_resp.content_text(),
        Some("hello from openrouter chat")
    );
    assert_eq!(
        provider_resp.content_text(),
        Some("hello from openrouter chat")
    );
    assert_eq!(
        config_resp.content_text(),
        Some("hello from openrouter chat")
    );
    assert_eq!(
        registry_resp.content_text(),
        Some("hello from openrouter chat")
    );
    assert_eq!(
        siumai_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(
        provider_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(
        config_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(
        registry_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(siumai_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(provider_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(config_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(registry_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        siumai_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://openrouter.ai/docs")
    );
    assert_eq!(
        registry_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://openrouter.ai/docs")
    );
    assert_eq!(
        siumai_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .and_then(|source| source.openrouter_metadata())
            .and_then(|meta| meta.file_id),
        Some("file_123".to_string())
    );
    assert_eq!(
        registry_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .and_then(|source| source.openrouter_metadata())
            .and_then(|meta| meta.container_id),
        Some("container_456".to_string())
    );
    assert_eq!(siumai_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(provider_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(config_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(registry_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(
        siumai_root
            .get("openrouter")
            .and_then(|meta| meta.get("logprobs")),
        Some(&expected_logprobs)
    );
    assert_eq!(
        provider_root
            .get("openrouter")
            .and_then(|meta| meta.get("logprobs")),
        Some(&expected_logprobs)
    );
    assert_eq!(
        config_root
            .get("openrouter")
            .and_then(|meta| meta.get("logprobs")),
        Some(&expected_logprobs)
    );
    assert_eq!(
        registry_root
            .get("openrouter")
            .and_then(|meta| meta.get("logprobs")),
        Some(&expected_logprobs)
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
        "https://openrouter.ai/api/v1/chat/completions"
    );
}

#[tokio::test]
async fn openrouter_siumai_provider_config_registry_stream_end_provider_metadata_are_equivalent() {
    let model = "openai/gpt-4o";
    let stream_body = br#"data: {"id":"1","model":"openai/gpt-4o","created":1718345013,"sources":[{"id":"src_1","source_type":"url","url":"https://openrouter.ai/docs","title":"OpenRouter Docs","provider_metadata":{"openrouter":{"fileId":"file_123","containerId":"container_456","index":1}}}],"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]},"finish_reason":null}]}

data: {"id":"1","model":"openai/gpt-4o","created":1718345013,"choices":[{"index":0,"delta":{"content":" from openrouter","role":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}}

data: [DONE]

"#
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("openrouter", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("openrouter", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build registry language model");

    let mut request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();
    request
        .provider_options_map
        .insert("openrouter", serde_json::json!({ "logprobs": 3 }));

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    use futures_util::StreamExt;

    let mut siumai_end = None;
    let mut provider_end = None;
    let mut config_end = None;
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
    while let Some(event) = config_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            config_end = Some(response);
            break;
        }
    }
    while let Some(event) = registry_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            registry_end = Some(response);
            break;
        }
    }

    let siumai_resp = siumai_end.expect("siumai stream end");
    let provider_resp = provider_end.expect("provider stream end");
    let config_resp = config_end.expect("config stream end");
    let registry_resp = registry_end.expect("registry stream end");

    let siumai_root = siumai_resp
        .provider_metadata
        .as_ref()
        .expect("siumai provider metadata");
    let provider_root = provider_resp
        .provider_metadata
        .as_ref()
        .expect("provider provider metadata");
    let config_root = config_resp
        .provider_metadata
        .as_ref()
        .expect("config provider metadata");
    let registry_root = registry_resp
        .provider_metadata
        .as_ref()
        .expect("registry provider metadata");

    assert!(siumai_root.get("openrouter").is_some());
    assert!(provider_root.get("openrouter").is_some());
    assert!(config_root.get("openrouter").is_some());
    assert!(registry_root.get("openrouter").is_some());
    assert!(siumai_root.get("openai_compatible").is_none());
    assert!(provider_root.get("openai_compatible").is_none());
    assert!(config_root.get("openai_compatible").is_none());
    assert!(registry_root.get("openai_compatible").is_none());

    let siumai_meta = siumai_resp
        .openrouter_metadata()
        .expect("siumai openrouter metadata");
    let provider_meta = provider_resp
        .openrouter_metadata()
        .expect("provider openrouter metadata");
    let config_meta = config_resp
        .openrouter_metadata()
        .expect("config openrouter metadata");
    let registry_meta = registry_resp
        .openrouter_metadata()
        .expect("registry openrouter metadata");

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.1,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(siumai_resp.content_text(), Some("hello from openrouter"));
    assert_eq!(provider_resp.content_text(), Some("hello from openrouter"));
    assert_eq!(config_resp.content_text(), Some("hello from openrouter"));
    assert_eq!(registry_resp.content_text(), Some("hello from openrouter"));
    assert_eq!(
        siumai_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        provider_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        config_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        registry_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        siumai_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(
        provider_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(
        config_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(
        registry_resp
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(siumai_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(provider_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(config_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(registry_meta.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        siumai_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://openrouter.ai/docs")
    );
    assert_eq!(
        registry_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .map(|source| source.url.as_str()),
        Some("https://openrouter.ai/docs")
    );
    assert_eq!(
        siumai_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .and_then(|source| source.openrouter_metadata())
            .and_then(|meta| meta.file_id),
        Some("file_123".to_string())
    );
    assert_eq!(
        registry_meta
            .sources
            .as_ref()
            .and_then(|sources| sources.first())
            .and_then(|source| source.openrouter_metadata())
            .and_then(|meta| meta.container_id),
        Some("container_456".to_string())
    );
    assert_eq!(siumai_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(provider_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(config_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(registry_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(
        siumai_root
            .get("openrouter")
            .and_then(|meta| meta.get("logprobs")),
        Some(&expected_logprobs)
    );
    assert_eq!(
        provider_root
            .get("openrouter")
            .and_then(|meta| meta.get("logprobs")),
        Some(&expected_logprobs)
    );
    assert_eq!(
        config_root
            .get("openrouter")
            .and_then(|meta| meta.get("logprobs")),
        Some(&expected_logprobs)
    );
    assert_eq!(
        registry_root
            .get("openrouter")
            .and_then(|meta| meta.get("logprobs")),
        Some(&expected_logprobs)
    );

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
        "https://openrouter.ai/api/v1/chat/completions"
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn openrouter_structured_output_synthetic_unknown_stream_end_extracts_across_public_paths() {
    let stream_body = concat!(
            "data: {\"id\":\"1\",\"model\":\"openai/gpt-4o\",\"created\":1718345013,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"{\\\"answer\\\":\\\"hel\",\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"1\",\"model\":\"openai/gpt-4o\",\"created\":1718345013,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\\\"}\"},\"finish_reason\":null}]}\n\n"
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "openai/gpt-4o";
    let siumai_client = Siumai::builder()
        .openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("openrouter", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("openrouter", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
        .build()
        .with_openrouter_options(
            OpenRouterOptions::new()
                .with_transform(OpenRouterTransform::MiddleOut)
                .with_param("someVendorParam", serde_json::json!(true)),
        );

    let siumai_value = siumai::structured_output::extract_json_value_from_stream(
        siumai_client
            .chat_stream_request(request.clone())
            .await
            .expect("siumai stream ok"),
    )
    .await
    .expect("siumai structured output");
    let provider_value = siumai::structured_output::extract_json_value_from_stream(
        provider_client
            .chat_stream_request(request.clone())
            .await
            .expect("provider stream ok"),
    )
    .await
    .expect("provider structured output");
    let config_value = siumai::structured_output::extract_json_value_from_stream(
        config_client
            .chat_stream_request(request.clone())
            .await
            .expect("config stream ok"),
    )
    .await
    .expect("config structured output");
    let registry_value = siumai::structured_output::extract_json_value_from_stream(
        registry_model
            .chat_stream_request(request)
            .await
            .expect("registry stream ok"),
    )
    .await
    .expect("registry structured output");

    assert_eq!(siumai_value["answer"], "hello");
    assert_eq!(provider_value["answer"], "hello");
    assert_eq!(config_value["answer"], "hello");
    assert_eq!(registry_value["answer"], "hello");

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
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        siumai_req.url,
        "https://openrouter.ai/api/v1/chat/completions"
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        siumai_req.body["transforms"],
        serde_json::json!(["middle-out"])
    );
    assert_eq!(siumai_req.body["someVendorParam"], serde_json::json!(true));
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
async fn openrouter_registry_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let openrouter_transport = CaptureTransport::default();

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("openrouter"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "openrouter",
            "ctx-key",
            "https://example.com/openrouter/v1",
            Arc::new(openrouter_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build openrouter handle");

    let _ = handle
        .chat_request(
            make_chat_request_with_model("openai/gpt-4o").with_openrouter_options(
                OpenRouterOptions::new().with_transform(OpenRouterTransform::MiddleOut),
            ),
        )
        .await;

    let req = openrouter_transport
        .take()
        .expect("captured openrouter request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(
        req.url,
        "https://example.com/openrouter/v1/chat/completions"
    );
    assert_eq!(req.body["model"], serde_json::json!("openai/gpt-4o"));
    assert_eq!(req.body["transforms"], serde_json::json!(["middle-out"]));
}

#[tokio::test]
async fn openrouter_registry_stream_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let openrouter_transport = CaptureTransport::default();

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("openrouter"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "openrouter",
            "ctx-key",
            "https://example.com/openrouter/v1",
            Arc::new(openrouter_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build openrouter handle");

    let _ = handle
        .chat_stream_request(
            make_chat_request_with_model("openai/gpt-4o").with_openrouter_options(
                OpenRouterOptions::new().with_transform(OpenRouterTransform::MiddleOut),
            ),
        )
        .await;

    let req = openrouter_transport
        .take_stream()
        .expect("captured openrouter stream request");
    assert!(global_transport.take().is_none());
    assert!(global_transport.take_stream().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(
        header_value(&req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        req.url,
        "https://example.com/openrouter/v1/chat/completions"
    );
    assert_eq!(req.body["model"], serde_json::json!("openai/gpt-4o"));
    assert_eq!(req.body["stream"], serde_json::json!(true));
    assert_eq!(req.body["transforms"], serde_json::json!(["middle-out"]));
}

#[tokio::test]
async fn openrouter_registry_override_chat_response_metadata_preserves_vendor_namespace() {
    let response_json = serde_json::json!({
        "id": "chatcmpl-openrouter-test",
        "object": "chat.completion",
        "created": 1_718_345_013,
        "model": "openai/gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello from openrouter chat"
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "hello",
                            "logprob": -0.1,
                            "bytes": [104, 101, 108, 108, 111],
                            "top_logprobs": []
                        }
                    ]
                }
            }
        ],
        "sources": [
            {
                "id": "src_1",
                "source_type": "url",
                "url": "https://openrouter.ai/docs",
                "title": "OpenRouter Docs",
                "provider_metadata": {
                    "openrouter": {
                        "fileId": "file_123",
                        "containerId": "container_456",
                        "index": 1
                    }
                }
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "total_tokens": 14
        }
    });

    let global_transport = CaptureTransport::default();
    let openrouter_transport = JsonSuccessTransport::new(response_json);

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("openrouter"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "openrouter",
            "ctx-key",
            "https://example.com/openrouter/v1",
            Arc::new(openrouter_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let response = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build openrouter handle")
        .chat_request(
            make_chat_request_with_model("openai/gpt-4o").with_openrouter_options(
                OpenRouterOptions::new().with_param("logprobs", serde_json::json!(3)),
            ),
        )
        .await
        .expect("registry response ok");

    let root = response
        .provider_metadata
        .as_ref()
        .expect("registry provider metadata");
    assert!(root.get("openrouter").is_some());
    assert!(root.get("openai_compatible").is_none());

    let metadata = response.openrouter_metadata().expect("openrouter metadata");
    assert_eq!(response.content_text(), Some("hello from openrouter chat"));
    assert_eq!(
        response
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(metadata.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        metadata
            .logprobs
            .as_ref()
            .and_then(|value| value.as_array().map(Vec::len)),
        Some(1)
    );

    let req = openrouter_transport.take().expect("captured request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(
        req.url,
        "https://example.com/openrouter/v1/chat/completions"
    );
}

#[tokio::test]
async fn openrouter_registry_chat_request_with_explicit_request_model_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();
    let default_model = "openai/gpt-4o";
    let request_model = "openai/gpt-4.1";

    let provider = siumai::provider_ext::openai_compatible::get_provider_config("openrouter")
        .expect("openrouter provider config");
    let adapter = Arc::new(
        siumai::provider_ext::openai_compatible::ConfigurableAdapter::new(provider.clone()),
    );
    let config = siumai::provider_ext::openai_compatible::OpenAiCompatibleConfig::new(
        "openrouter",
        "test-key",
        &provider.base_url,
        adapter,
    )
    .with_model(default_model)
    .with_reasoning(true)
    .with_reasoning_budget(2048)
    .with_http_transport(Arc::new(config_transport.clone()));
    let config_client =
        siumai::provider_ext::openai_compatible::OpenAiCompatibleClient::from_config(config)
            .await
            .expect("build config client");

    let registry = make_registry_with_global_reasoning_defaults(
        "openrouter",
        Arc::new(registry_transport.clone()),
        true,
        2048,
    );
    let registry_model = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(request_model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_openrouter_options(
            OpenRouterOptions::new()
                .with_transform(OpenRouterTransform::MiddleOut)
                .with_param("someVendorParam", serde_json::json!(true)),
        );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.body["model"], serde_json::json!(request_model));
    assert_eq!(
        registry_req.body["transforms"],
        serde_json::json!(["middle-out"])
    );
    assert_eq!(
        registry_req.body["someVendorParam"],
        serde_json::json!(true)
    );
    assert_eq!(
        registry_req.body["enable_reasoning"],
        serde_json::json!(true)
    );
    assert_eq!(
        registry_req.body["reasoning_budget"],
        serde_json::json!(2048)
    );
}

#[tokio::test]
async fn openrouter_registry_chat_stream_request_with_explicit_request_model_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();
    let default_model = "openai/gpt-4o";
    let request_model = "openai/gpt-4.1";

    let provider = siumai::provider_ext::openai_compatible::get_provider_config("openrouter")
        .expect("openrouter provider config");
    let adapter = Arc::new(
        siumai::provider_ext::openai_compatible::ConfigurableAdapter::new(provider.clone()),
    );
    let config = siumai::provider_ext::openai_compatible::OpenAiCompatibleConfig::new(
        "openrouter",
        "test-key",
        &provider.base_url,
        adapter,
    )
    .with_model(default_model)
    .with_reasoning(true)
    .with_reasoning_budget(2048)
    .with_http_transport(Arc::new(config_transport.clone()));
    let config_client =
        siumai::provider_ext::openai_compatible::OpenAiCompatibleClient::from_config(config)
            .await
            .expect("build config client");

    let registry = make_registry_with_global_reasoning_defaults(
        "openrouter",
        Arc::new(registry_transport.clone()),
        true,
        2048,
    );
    let registry_model = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(request_model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_openrouter_options(
            OpenRouterOptions::new()
                .with_transform(OpenRouterTransform::MiddleOut)
                .with_param("someVendorParam", serde_json::json!(true)),
        );

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
        registry_req.body["transforms"],
        serde_json::json!(["middle-out"])
    );
    assert_eq!(
        registry_req.body["someVendorParam"],
        serde_json::json!(true)
    );
    assert_eq!(
        registry_req.body["enable_reasoning"],
        serde_json::json!(true)
    );
    assert_eq!(
        registry_req.body["reasoning_budget"],
        serde_json::json!(2048)
    );
}

#[tokio::test]
async fn openrouter_registry_override_stream_end_metadata_preserves_vendor_namespace() {
    let stream_body = br#"data: {"id":"1","model":"openai/gpt-4o","created":1718345013,"sources":[{"id":"src_1","source_type":"url","url":"https://openrouter.ai/docs","title":"OpenRouter Docs","provider_metadata":{"openrouter":{"fileId":"file_123","containerId":"container_456","index":1}}}],"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]},"finish_reason":null}]}

data: {"id":"1","model":"openai/gpt-4o","created":1718345013,"choices":[{"index":0,"delta":{"content":" from openrouter","role":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}}

data: [DONE]

"#
        .to_vec();

    let global_transport = CaptureTransport::default();
    let openrouter_transport = SseSuccessTransport::new(stream_body);

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("openrouter"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "openrouter",
            "ctx-key",
            "https://example.com/openrouter/v1",
            Arc::new(openrouter_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let mut stream = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build openrouter handle")
        .chat_stream_request(
            make_chat_request_with_model("openai/gpt-4o").with_openrouter_options(
                OpenRouterOptions::new().with_param("logprobs", serde_json::json!(3)),
            ),
        )
        .await
        .expect("registry stream ok");

    use futures_util::StreamExt;
    let mut stream_end = None;
    while let Some(event) = stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            stream_end = Some(response);
            break;
        }
    }

    let response = stream_end.expect("registry stream end");
    let root = response
        .provider_metadata
        .as_ref()
        .expect("registry provider metadata");
    assert!(root.get("openrouter").is_some());
    assert!(root.get("openai_compatible").is_none());

    let metadata = response.openrouter_metadata().expect("openrouter metadata");
    assert_eq!(response.content_text(), Some("hello from openrouter"));
    assert_eq!(
        response
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(14)
    );
    assert_eq!(metadata.sources.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        metadata
            .logprobs
            .as_ref()
            .and_then(|value| value.as_array().map(Vec::len)),
        Some(1)
    );

    let req = openrouter_transport
        .take_stream()
        .expect("captured stream request");
    assert!(global_transport.take().is_none());
    assert!(global_transport.take_stream().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(
        req.url,
        "https://example.com/openrouter/v1/chat/completions"
    );
}

#[tokio::test]
async fn openrouter_structured_output_synthetic_unknown_stream_end_fails_consistently_across_public_paths()
 {
    let stream_body = "data: {\"id\":\"1\",\"model\":\"openai/gpt-4o\",\"created\":1718345013,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"{\\\"answer\\\":\" ,\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"
            .as_bytes()
            .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "openai/gpt-4o";
    let siumai_client = Siumai::builder()
        .openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("openrouter", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("openrouter", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build registry language model");

    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
        .build()
        .with_openrouter_options(
            OpenRouterOptions::new()
                .with_transform(OpenRouterTransform::MiddleOut)
                .with_param("someVendorParam", serde_json::json!(true)),
        );

    let siumai_err = siumai::structured_output::extract_json_value_from_stream(
        siumai_client
            .chat_stream_request(request.clone())
            .await
            .expect("siumai stream ok"),
    )
    .await
    .expect_err("siumai interrupted stream should fail");
    let provider_err = siumai::structured_output::extract_json_value_from_stream(
        provider_client
            .chat_stream_request(request.clone())
            .await
            .expect("provider stream ok"),
    )
    .await
    .expect_err("provider interrupted stream should fail");
    let config_err = siumai::structured_output::extract_json_value_from_stream(
        config_client
            .chat_stream_request(request.clone())
            .await
            .expect("config stream ok"),
    )
    .await
    .expect_err("config interrupted stream should fail");
    let registry_err = siumai::structured_output::extract_json_value_from_stream(
        registry_model
            .chat_stream_request(request)
            .await
            .expect("registry stream ok"),
    )
    .await
    .expect_err("registry interrupted stream should fail");

    for err in [siumai_err, provider_err, config_err, registry_err] {
        match err {
            LlmError::ParseError(message) => {
                assert!(message.contains("stream ended before a complete JSON value was produced"))
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
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
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        siumai_req.url,
        "https://openrouter.ai/api/v1/chat/completions"
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        siumai_req.body["transforms"],
        serde_json::json!(["middle-out"])
    );
    assert_eq!(siumai_req.body["someVendorParam"], serde_json::json!(true));
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
async fn openrouter_registry_global_reasoning_defaults_match_config_defaults() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let provider = siumai::provider_ext::openai_compatible::get_provider_config("openrouter")
        .expect("openrouter provider config");
    let adapter = Arc::new(
        siumai::provider_ext::openai_compatible::ConfigurableAdapter::new(provider.clone()),
    );
    let config = siumai::provider_ext::openai_compatible::OpenAiCompatibleConfig::new(
        "openrouter",
        "test-key",
        &provider.base_url,
        adapter,
    )
    .with_model("openai/gpt-4o")
    .with_reasoning(true)
    .with_reasoning_budget(1024)
    .with_http_transport(Arc::new(config_transport.clone()));
    let config_client =
        siumai::provider_ext::openai_compatible::OpenAiCompatibleClient::from_config(config)
            .await
            .expect("build config client");

    let registry = make_registry_with_global_reasoning_defaults(
        "openrouter",
        Arc::new(registry_transport.clone()),
        true,
        1024,
    );
    let registry_model = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model("openai/gpt-4o")
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_openrouter_options(
            OpenRouterOptions::new().with_transform(OpenRouterTransform::MiddleOut),
        );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.body["transforms"],
        serde_json::json!(["middle-out"])
    );
    assert_eq!(
        registry_req.body["enable_reasoning"],
        serde_json::json!(true)
    );
    assert_eq!(
        registry_req.body["reasoning_budget"],
        serde_json::json!(1024)
    );
}

#[tokio::test]
async fn openrouter_registry_builder_global_reasoning_defaults_match_config_defaults() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let provider = siumai::provider_ext::openai_compatible::get_provider_config("openrouter")
        .expect("openrouter provider config");
    let adapter = Arc::new(
        siumai::provider_ext::openai_compatible::ConfigurableAdapter::new(provider.clone()),
    );
    let config = siumai::provider_ext::openai_compatible::OpenAiCompatibleConfig::new(
        "openrouter",
        "test-key",
        &provider.base_url,
        adapter,
    )
    .with_model("openai/gpt-4o")
    .with_reasoning(true)
    .with_reasoning_budget(1024)
    .with_http_transport(Arc::new(config_transport.clone()));
    let config_client =
        siumai::provider_ext::openai_compatible::OpenAiCompatibleClient::from_config(config)
            .await
            .expect("build config client");

    let registry = make_registry_builder_with_global_reasoning_defaults(
        "openrouter",
        Arc::new(registry_transport.clone()),
        true,
        1024,
    );
    let registry_model = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model("openai/gpt-4o")
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_openrouter_options(
            OpenRouterOptions::new().with_transform(OpenRouterTransform::MiddleOut),
        );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.body["transforms"],
        serde_json::json!(["middle-out"])
    );
    assert_eq!(
        registry_req.body["enable_reasoning"],
        serde_json::json!(true)
    );
    assert_eq!(
        registry_req.body["reasoning_budget"],
        serde_json::json!(1024)
    );
}

#[tokio::test]
async fn openrouter_reasoning_response_is_equivalent_across_public_paths() {
    let model = "openai/gpt-4o";
    let response_json = serde_json::json!({
        "id": "chatcmpl-openrouter-reasoning",
        "object": "chat.completion",
        "created": 1_741_392_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "There are three letter r's in strawberry.",
                    "thinking": "Count the letters carefully. The word strawberry contains three r characters."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 18,
            "completion_tokens": 24,
            "total_tokens": 42,
            "completion_tokens_details": {
                "reasoning_tokens": 12
            }
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .reasoning(true)
        .reasoning_budget(1536)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .reasoning(true)
        .reasoning_budget(1536)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let provider = siumai::provider_ext::openai_compatible::get_provider_config("openrouter")
        .expect("openrouter provider config");
    let adapter = Arc::new(
        siumai::provider_ext::openai_compatible::ConfigurableAdapter::new(provider.clone()),
    );
    let config = siumai::provider_ext::openai_compatible::OpenAiCompatibleConfig::new(
        "openrouter",
        "test-key",
        &provider.base_url,
        adapter,
    )
    .with_model(model)
    .with_reasoning(true)
    .with_reasoning_budget(1536)
    .with_http_transport(Arc::new(config_transport.clone()));
    let config_client =
        siumai::provider_ext::openai_compatible::OpenAiCompatibleClient::from_config(config)
            .await
            .expect("build config client");

    let registry = make_registry_with_global_reasoning_defaults(
        "openrouter",
        Arc::new(registry_transport.clone()),
        true,
        1536,
    );
    let registry_model = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_openrouter_options(
            OpenRouterOptions::new().with_transform(OpenRouterTransform::MiddleOut),
        );

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

    let expected_reasoning =
        "Count the letters carefully. The word strawberry contains three r characters.".to_string();

    for response in [&siumai_resp, &provider_resp, &config_resp, &registry_resp] {
        assert_eq!(
            response.content_text(),
            Some("There are three letter r's in strawberry.")
        );
        assert_eq!(response.reasoning(), vec![expected_reasoning.clone()]);
        assert_eq!(
            response
                .usage
                .as_ref()
                .and_then(|usage| usage.completion_tokens_details.as_ref())
                .and_then(|details| details.reasoning_tokens),
            Some(12)
        );
        assert_eq!(
            response.finish_reason,
            Some(siumai::prelude::unified::FinishReason::Stop)
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
        "https://openrouter.ai/api/v1/chat/completions"
    );
    assert_eq!(
        siumai_req.body["transforms"],
        serde_json::json!(["middle-out"])
    );
    assert_eq!(siumai_req.body["enable_reasoning"], serde_json::json!(true));
    assert_eq!(siumai_req.body["reasoning_budget"], serde_json::json!(1536));
}

#[tokio::test]
async fn openrouter_tool_call_thought_signature_response_is_equivalent_across_public_paths() {
    let model = "google/gemini-2.0-flash-001";
    let response_json = serde_json::json!({
        "id": "chatcmpl-openrouter-thought-signature",
        "object": "chat.completion",
        "created": 1_741_392_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\":\"Tokyo\"}"
                            },
                            "extra_content": {
                                "google": {
                                    "thought_signature": "<Signature A>"
                                }
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 4,
            "total_tokens": 12
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("openrouter", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("openrouter", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model(&format!("openrouter:{model}"))
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_openrouter_options(
            OpenRouterOptions::new().with_transform(OpenRouterTransform::MiddleOut),
        );

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
        assert_eq!(
            response.finish_reason,
            Some(siumai::prelude::unified::FinishReason::ToolCalls)
        );
        assert_eq!(response.tool_calls().len(), 1);

        let response_root = response
            .provider_metadata
            .as_ref()
            .expect("response provider metadata");
        assert!(response_root.get("openrouter").is_some());
        assert!(response_root.get("openai_compatible").is_none());

        let tool_call = serde_json::to_value(response.tool_calls()[0]).expect("tool call json");
        assert_eq!(tool_call["toolCallId"], serde_json::json!("call_weather"));
        assert_eq!(tool_call["toolName"], serde_json::json!("get_weather"));
        assert_eq!(tool_call["input"], serde_json::json!({ "city": "Tokyo" }));
        assert_eq!(
            tool_call["providerMetadata"]["openrouter"]["thoughtSignature"],
            serde_json::json!("<Signature A>")
        );
        assert!(
            tool_call["providerMetadata"]
                .get("openai_compatible")
                .is_none()
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
        "https://openrouter.ai/api/v1/chat/completions"
    );
    assert_eq!(
        siumai_req.body["transforms"],
        serde_json::json!(["middle-out"])
    );
}

#[tokio::test]
async fn openrouter_reasoning_stream_is_equivalent_across_public_paths() {
    let model = "openai/gpt-4o";
    let stream_body = br#"data: {"id":"1","model":"openai/gpt-4o","created":1718345013,"choices":[{"index":0,"delta":{"thinking":"Count the letters carefully. ","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"openai/gpt-4o","created":1718345013,"choices":[{"index":0,"delta":{"thinking":"The word strawberry contains three r characters.","content":"There are three letter r's in strawberry.","role":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":18,"completion_tokens":24,"total_tokens":42,"completion_tokens_details":{"reasoning_tokens":12}}}

data: [DONE]

"#
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .reasoning(true)
        .reasoning_budget(1536)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .model(model)
        .reasoning(true)
        .reasoning_budget(1536)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let provider = siumai::provider_ext::openai_compatible::get_provider_config("openrouter")
        .expect("openrouter provider config");
    let adapter = Arc::new(
        siumai::provider_ext::openai_compatible::ConfigurableAdapter::new(provider.clone()),
    );
    let config = siumai::provider_ext::openai_compatible::OpenAiCompatibleConfig::new(
        "openrouter",
        "test-key",
        &provider.base_url,
        adapter,
    )
    .with_model(model)
    .with_reasoning(true)
    .with_reasoning_budget(1536)
    .with_http_transport(Arc::new(config_transport.clone()));
    let config_client =
        siumai::provider_ext::openai_compatible::OpenAiCompatibleClient::from_config(config)
            .await
            .expect("build config client");

    let registry = make_registry_with_global_reasoning_defaults(
        "openrouter",
        Arc::new(registry_transport.clone()),
        true,
        1536,
    );
    let registry_model = registry
        .language_model("openrouter:openai/gpt-4o")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_openrouter_options(
            OpenRouterOptions::new().with_transform(OpenRouterTransform::MiddleOut),
        );

    let collect_stream_summary = async |stream: &mut siumai::prelude::unified::ChatStream| {
        let mut end = None;
        let mut reasoning = String::new();
        while let Some(event) = stream.next().await {
            match event.expect("stream event ok") {
                event if event.reasoning_delta().is_some() => {
                    reasoning.push_str(event.reasoning_delta().expect("reasoning delta"));
                }
                siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => {
                    end = Some(response);
                    break;
                }
                _ => {}
            }
        }
        (reasoning, end)
    };

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    let (siumai_reasoning, siumai_resp) = collect_stream_summary(&mut siumai_stream).await;
    let (provider_reasoning, provider_resp) = collect_stream_summary(&mut provider_stream).await;
    let (config_reasoning, config_resp) = collect_stream_summary(&mut config_stream).await;
    let (registry_reasoning, registry_resp) = collect_stream_summary(&mut registry_stream).await;

    let siumai_resp = siumai_resp.expect("siumai stream end");
    let provider_resp = provider_resp.expect("provider stream end");
    let config_resp = config_resp.expect("config stream end");
    let registry_resp = registry_resp.expect("registry stream end");

    let expected_reasoning =
        "Count the letters carefully. The word strawberry contains three r characters.".to_string();

    for reasoning in [
        &siumai_reasoning,
        &provider_reasoning,
        &config_reasoning,
        &registry_reasoning,
    ] {
        assert_eq!(reasoning, &expected_reasoning);
    }

    for response in [&siumai_resp, &provider_resp, &config_resp, &registry_resp] {
        assert_eq!(
            response.content_text(),
            Some("There are three letter r's in strawberry.")
        );
        assert_eq!(
            response
                .usage
                .as_ref()
                .and_then(|usage| usage.completion_tokens_details.as_ref())
                .and_then(|details| details.reasoning_tokens),
            Some(12)
        );
        assert_eq!(
            response.finish_reason,
            Some(siumai::prelude::unified::FinishReason::Stop)
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
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        siumai_req.url,
        "https://openrouter.ai/api/v1/chat/completions"
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        siumai_req.body["transforms"],
        serde_json::json!(["middle-out"])
    );
    assert_eq!(siumai_req.body["enable_reasoning"], serde_json::json!(true));
    assert_eq!(siumai_req.body["reasoning_budget"], serde_json::json!(1536));
}

#[tokio::test]
async fn together_siumai_provider_config_embedding_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .togetherai_openai_compatible()
        .api_key("test-key")
        .model("togethercomputer/m2-bert-80M-8k-retrieval")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .togetherai_openai_compatible()
        .api_key("test-key")
        .model("togethercomputer/m2-bert-80M-8k-retrieval")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "togetherai",
        "togethercomputer/m2-bert-80M-8k-retrieval",
        Arc::new(config_transport.clone()),
    )
    .await;

    let request = EmbeddingRequest::single("hello together embedding")
        .with_model("togethercomputer/m2-bert-80M-8k-retrieval")
        .with_dimensions(384)
        .with_encoding_format(EmbeddingFormat::Float)
        .with_user("compat-user-4");

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://api.together.xyz/v1/embeddings");
    assert_eq!(
        siumai_req.body["model"],
        serde_json::json!("togethercomputer/m2-bert-80M-8k-retrieval")
    );
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!(["hello together embedding"])
    );
    assert_eq!(siumai_req.body["dimensions"], serde_json::json!(384));
    assert_eq!(
        siumai_req.body["encoding_format"],
        serde_json::json!("float")
    );
    assert_eq!(siumai_req.body["user"], serde_json::json!("compat-user-4"));
}

#[tokio::test]
async fn together_registry_embedding_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = make_config_client(
        "togetherai",
        "togethercomputer/m2-bert-80M-8k-retrieval",
        Arc::new(config_transport.clone()),
    )
    .await;
    let registry = make_registry("togetherai", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .embedding_model("togetherai:togethercomputer/m2-bert-80M-8k-retrieval")
        .expect("build registry embedding model");

    let request = EmbeddingRequest::single("hello together embedding")
        .with_model("togethercomputer/m2-bert-80M-8k-retrieval")
        .with_dimensions(384)
        .with_encoding_format(EmbeddingFormat::Float)
        .with_user("compat-user-4");

    let _ = config_client.embed_with_config(request.clone()).await;
    let _ = registry_model.embed_with_config(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.url, "https://api.together.xyz/v1/embeddings");
    assert_eq!(
        registry_req.body["model"],
        serde_json::json!("togethercomputer/m2-bert-80M-8k-retrieval")
    );
    assert_eq!(
        registry_req.body["input"],
        serde_json::json!(["hello together embedding"])
    );
    assert_eq!(registry_req.body["dimensions"], serde_json::json!(384));
    assert_eq!(
        registry_req.body["encoding_format"],
        serde_json::json!("float")
    );
    assert_eq!(
        registry_req.body["user"],
        serde_json::json!("compat-user-4")
    );
}

#[tokio::test]
async fn together_registry_embedding_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let together_transport = CaptureTransport::default();

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("togetherai"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "togetherai",
            "ctx-key",
            "https://example.com/together/v1",
            Arc::new(together_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .embedding_model("togetherai:togethercomputer/m2-bert-80M-8k-retrieval")
        .expect("build together embedding handle");

    let _ = handle
        .embed_with_config(
            EmbeddingRequest::single("hello together embedding")
                .with_model("togethercomputer/m2-bert-80M-8k-retrieval")
                .with_dimensions(384)
                .with_encoding_format(EmbeddingFormat::Float)
                .with_user("compat-user-4"),
        )
        .await;

    let req = together_transport
        .take()
        .expect("captured together embedding request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/together/v1/embeddings");
    assert_eq!(
        req.body["model"],
        serde_json::json!("togethercomputer/m2-bert-80M-8k-retrieval")
    );
    assert_eq!(
        req.body["input"],
        serde_json::json!(["hello together embedding"])
    );
    assert_eq!(req.body["dimensions"], serde_json::json!(384));
    assert_eq!(req.body["encoding_format"], serde_json::json!("float"));
    assert_eq!(req.body["user"], serde_json::json!("compat-user-4"));
}

#[tokio::test]
async fn jina_siumai_provider_config_embedding_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .jina()
        .api_key("test-key")
        .model("jina-embeddings-v2-base-en")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .jina()
        .api_key("test-key")
        .model("jina-embeddings-v2-base-en")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "jina",
        "jina-embeddings-v2-base-en",
        Arc::new(config_transport.clone()),
    )
    .await;

    let request = EmbeddingRequest::single("hello jina embedding")
        .with_model("jina-embeddings-v2-base-en")
        .with_dimensions(768)
        .with_encoding_format(EmbeddingFormat::Float)
        .with_user("compat-user-5");

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://api.jina.ai/v1/embeddings");
    assert_eq!(
        siumai_req.body["model"],
        serde_json::json!("jina-embeddings-v2-base-en")
    );
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!(["hello jina embedding"])
    );
    assert_eq!(siumai_req.body["dimensions"], serde_json::json!(768));
    assert_eq!(
        siumai_req.body["encoding_format"],
        serde_json::json!("float")
    );
    assert_eq!(siumai_req.body["user"], serde_json::json!("compat-user-5"));
}

#[tokio::test]
async fn jina_registry_embedding_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = make_config_client(
        "jina",
        "jina-embeddings-v2-base-en",
        Arc::new(config_transport.clone()),
    )
    .await;
    let registry = make_registry("jina", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .embedding_model("jina:jina-embeddings-v2-base-en")
        .expect("build registry embedding model");

    let request = EmbeddingRequest::single("hello jina embedding")
        .with_model("jina-embeddings-v2-base-en")
        .with_dimensions(768)
        .with_encoding_format(EmbeddingFormat::Float)
        .with_user("compat-user-5");

    let _ = config_client.embed_with_config(request.clone()).await;
    let _ = registry_model.embed_with_config(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.url, "https://api.jina.ai/v1/embeddings");
    assert_eq!(
        registry_req.body["model"],
        serde_json::json!("jina-embeddings-v2-base-en")
    );
    assert_eq!(
        registry_req.body["input"],
        serde_json::json!(["hello jina embedding"])
    );
    assert_eq!(registry_req.body["dimensions"], serde_json::json!(768));
    assert_eq!(
        registry_req.body["encoding_format"],
        serde_json::json!("float")
    );
    assert_eq!(
        registry_req.body["user"],
        serde_json::json!("compat-user-5")
    );
}

#[tokio::test]
async fn voyageai_siumai_provider_config_embedding_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .voyageai()
        .api_key("test-key")
        .model("voyage-3")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .voyageai()
        .api_key("test-key")
        .model("voyage-3")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("voyageai", "voyage-3", Arc::new(config_transport.clone())).await;

    let request = EmbeddingRequest::single("hello voyage embedding")
        .with_model("voyage-3")
        .with_dimensions(1024)
        .with_encoding_format(EmbeddingFormat::Base64)
        .with_user("compat-user-6");

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://api.voyageai.com/v1/embeddings");
    assert_eq!(siumai_req.body["model"], serde_json::json!("voyage-3"));
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!(["hello voyage embedding"])
    );
    assert_eq!(siumai_req.body["dimensions"], serde_json::json!(1024));
    assert_eq!(
        siumai_req.body["encoding_format"],
        serde_json::json!("base64")
    );
    assert_eq!(siumai_req.body["user"], serde_json::json!("compat-user-6"));
}

#[tokio::test]
async fn voyageai_registry_embedding_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client =
        make_config_client("voyageai", "voyage-3", Arc::new(config_transport.clone())).await;
    let registry = make_registry("voyageai", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .embedding_model("voyageai:voyage-3")
        .expect("build registry embedding model");

    let request = EmbeddingRequest::single("hello voyage embedding")
        .with_model("voyage-3")
        .with_dimensions(1024)
        .with_encoding_format(EmbeddingFormat::Base64)
        .with_user("compat-user-6");

    let _ = config_client.embed_with_config(request.clone()).await;
    let _ = registry_model.embed_with_config(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.url, "https://api.voyageai.com/v1/embeddings");
    assert_eq!(registry_req.body["model"], serde_json::json!("voyage-3"));
    assert_eq!(
        registry_req.body["input"],
        serde_json::json!(["hello voyage embedding"])
    );
    assert_eq!(registry_req.body["dimensions"], serde_json::json!(1024));
    assert_eq!(
        registry_req.body["encoding_format"],
        serde_json::json!("base64")
    );
    assert_eq!(
        registry_req.body["user"],
        serde_json::json!("compat-user-6")
    );
}

#[tokio::test]
async fn infini_siumai_provider_config_embedding_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .infini()
        .api_key("test-key")
        .model("text-embedding-3-small")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .infini()
        .api_key("test-key")
        .model("text-embedding-3-small")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = make_config_client(
        "infini",
        "text-embedding-3-small",
        Arc::new(config_transport.clone()),
    )
    .await;

    let request = EmbeddingRequest::single("hello infini embedding")
        .with_model("text-embedding-3-small")
        .with_dimensions(512)
        .with_encoding_format(EmbeddingFormat::Float)
        .with_user("compat-user-7");

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://cloud.infini-ai.com/maas/v1/embeddings"
    );
    assert_eq!(
        siumai_req.body["model"],
        serde_json::json!("text-embedding-3-small")
    );
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!(["hello infini embedding"])
    );
    assert_eq!(siumai_req.body["dimensions"], serde_json::json!(512));
    assert_eq!(
        siumai_req.body["encoding_format"],
        serde_json::json!("float")
    );
    assert_eq!(siumai_req.body["user"], serde_json::json!("compat-user-7"));
}

#[tokio::test]
async fn infini_registry_embedding_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let config_client = make_config_client(
        "infini",
        "text-embedding-3-small",
        Arc::new(config_transport.clone()),
    )
    .await;
    let registry = make_registry("infini", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .embedding_model("infini:text-embedding-3-small")
        .expect("build registry embedding model");

    let request = EmbeddingRequest::single("hello infini embedding")
        .with_model("text-embedding-3-small")
        .with_dimensions(512)
        .with_encoding_format(EmbeddingFormat::Float)
        .with_user("compat-user-7");

    let _ = config_client.embed_with_config(request.clone()).await;
    let _ = registry_model.embed_with_config(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.url,
        "https://cloud.infini-ai.com/maas/v1/embeddings"
    );
    assert_eq!(
        registry_req.body["model"],
        serde_json::json!("text-embedding-3-small")
    );
    assert_eq!(
        registry_req.body["input"],
        serde_json::json!(["hello infini embedding"])
    );
    assert_eq!(registry_req.body["dimensions"], serde_json::json!(512));
    assert_eq!(
        registry_req.body["encoding_format"],
        serde_json::json!("float")
    );
    assert_eq!(
        registry_req.body["user"],
        serde_json::json!("compat-user-7")
    );
}

#[tokio::test]
async fn jina_registry_embedding_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let jina_transport = CaptureTransport::default();

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("jina"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "jina",
            "ctx-key",
            "https://example.com/jina/v1",
            Arc::new(jina_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .embedding_model("jina:jina-embeddings-v2-base-en")
        .expect("build jina embedding handle");

    let _ = handle
        .embed_with_config(
            EmbeddingRequest::single("hello jina embedding")
                .with_model("jina-embeddings-v2-base-en")
                .with_dimensions(768)
                .with_encoding_format(EmbeddingFormat::Float)
                .with_user("compat-user-5"),
        )
        .await;

    let req = jina_transport
        .take()
        .expect("captured jina embedding request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/jina/v1/embeddings");
    assert_eq!(
        req.body["model"],
        serde_json::json!("jina-embeddings-v2-base-en")
    );
    assert_eq!(
        req.body["input"],
        serde_json::json!(["hello jina embedding"])
    );
    assert_eq!(req.body["dimensions"], serde_json::json!(768));
    assert_eq!(req.body["encoding_format"], serde_json::json!("float"));
    assert_eq!(req.body["user"], serde_json::json!("compat-user-5"));
}

#[tokio::test]
async fn voyageai_registry_embedding_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let voyageai_transport = CaptureTransport::default();

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("voyageai"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "voyageai",
            "ctx-key",
            "https://example.com/voyageai/v1",
            Arc::new(voyageai_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .embedding_model("voyageai:voyage-3")
        .expect("build voyageai embedding handle");

    let _ = handle
        .embed_with_config(
            EmbeddingRequest::single("hello voyage embedding")
                .with_model("voyage-3")
                .with_dimensions(1024)
                .with_encoding_format(EmbeddingFormat::Base64)
                .with_user("compat-user-6"),
        )
        .await;

    let req = voyageai_transport
        .take()
        .expect("captured voyageai embedding request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/voyageai/v1/embeddings");
    assert_eq!(req.body["model"], serde_json::json!("voyage-3"));
    assert_eq!(
        req.body["input"],
        serde_json::json!(["hello voyage embedding"])
    );
    assert_eq!(req.body["dimensions"], serde_json::json!(1024));
    assert_eq!(req.body["encoding_format"], serde_json::json!("base64"));
    assert_eq!(req.body["user"], serde_json::json!("compat-user-6"));
}

#[tokio::test]
async fn infini_registry_embedding_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let infini_transport = CaptureTransport::default();

    let registry = RegistryBuilder::new(openai_compatible_registry_providers("infini"))
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "infini",
            "ctx-key",
            "https://example.com/infini/maas/v1",
            Arc::new(infini_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .embedding_model("infini:text-embedding-3-small")
        .expect("build infini embedding handle");

    let _ = handle
        .embed_with_config(
            EmbeddingRequest::single("hello infini embedding")
                .with_model("text-embedding-3-small")
                .with_dimensions(512)
                .with_encoding_format(EmbeddingFormat::Float)
                .with_user("compat-user-7"),
        )
        .await;

    let req = infini_transport
        .take()
        .expect("captured infini embedding request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/infini/maas/v1/embeddings");
    assert_eq!(
        req.body["model"],
        serde_json::json!("text-embedding-3-small")
    );
    assert_eq!(
        req.body["input"],
        serde_json::json!(["hello infini embedding"])
    );
    assert_eq!(req.body["dimensions"], serde_json::json!(512));
    assert_eq!(req.body["encoding_format"], serde_json::json!("float"));
    assert_eq!(req.body["user"], serde_json::json!("compat-user-7"));
}

#[tokio::test]
async fn infini_siumai_provider_config_registry_chat_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "deepseek-chat";

    let siumai_client = Siumai::builder()
        .openai()
        .infini()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .infini()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("infini", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("infini", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("infini:deepseek-chat")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model);

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://cloud.infini-ai.com/maas/v1/chat/completions"
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["messages"],
        serde_json::json!([{ "role": "user", "content": "hi" }])
    );
}

#[tokio::test]
async fn infini_siumai_provider_config_registry_chat_stream_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "deepseek-chat";

    let siumai_client = Siumai::builder()
        .openai()
        .infini()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .infini()
        .api_key("test-key")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client =
        make_config_client("infini", model, Arc::new(config_transport.clone())).await;
    let registry = make_registry("infini", Arc::new(registry_transport.clone()));
    let registry_model = registry
        .language_model("infini:deepseek-chat")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model);

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    use futures_util::StreamExt;
    let _ = siumai_stream.next().await;
    let _ = provider_stream.next().await;
    let _ = config_stream.next().await;
    let _ = registry_stream.next().await;

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
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        siumai_req.url,
        "https://cloud.infini-ai.com/maas/v1/chat/completions"
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["messages"],
        serde_json::json!([{ "role": "user", "content": "hi" }])
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}
