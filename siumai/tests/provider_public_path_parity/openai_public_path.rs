use super::*;
use siumai::compat::content::ContentPart;
use siumai::experimental::client::LlmClient;
use siumai::extensions::{SpeechExtras, TranscriptionExtras};
use siumai::prelude::unified::{
    AudioStreamEvent, EmbeddingExtensions, EmbeddingRequest, ResponseFormat, Tool, ToolChoice,
};
use siumai::provider_ext::openai::{
    OpenAIProviderSettings, OpenAiChatRequestExt, OpenAiChatResponseExt, OpenAiContentPartExt,
    OpenAiOptions, OpenAiSourceExt, ReasoningEffort, ResponsesApiConfig,
};
use siumai_registry::registry::entry::BuildContext;

fn openai_builtin_factory() -> Arc<dyn siumai::registry::ProviderFactory> {
    built_in_registry_factory("openai")
}

fn openai_registry_builder(
    registry_provider_id: &str,
) -> siumai::registry::builder::RegistryBuilder {
    built_in_registry_builder(registry_provider_id, "openai")
}

fn make_openai_override_registry(
    global_transport: Arc<dyn HttpTransport>,
    provider_transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    openai_registry_builder("openai")
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(global_transport)
        .with_provider_api_key_base_url_fetch(
            "openai",
            "ctx-key",
            "https://example.com/openai/v1",
            provider_transport,
        )
        .auto_middleware(false)
        .build()
        .expect("build openai override registry")
}

fn make_registry(
    transport: Arc<dyn HttpTransport>,
    base_url: &str,
) -> siumai::registry::ProviderRegistryHandle {
    openai_registry_builder("openai")
        .with_provider_api_key_base_url_fetch("openai", "test-key", base_url, transport)
        .build()
        .expect("build openai registry")
}

fn make_chat_registry(
    transport: Arc<dyn HttpTransport>,
    base_url: &str,
) -> siumai::registry::ProviderRegistryHandle {
    openai_registry_builder("openai-chat")
        .with_provider_api_key_base_url_fetch("openai-chat", "test-key", base_url, transport)
        .build()
        .expect("build openai chat registry")
}

#[test]
fn openai_package_settings_preserve_supported_provider_inputs() {
    let config = OpenAIProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.invalid/openai")
        .with_organization("org-123")
        .with_project("proj-456")
        .with_header("x-test", "1")
        .into_config_for_model("gpt-4.1-mini")
        .expect("settings into config");

    assert_eq!(config.base_url, "https://example.invalid/openai");
    assert_eq!(config.organization.as_deref(), Some("org-123"));
    assert_eq!(config.project.as_deref(), Some("proj-456"));
    assert_eq!(config.common_params.model, "gpt-4.1-mini");
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
}

async fn collect_tts_audio(mut stream: siumai_core::types::AudioStream) -> Vec<u8> {
    use futures_util::StreamExt;

    let mut out = Vec::new();
    while let Some(item) = stream.next().await {
        match item.expect("tts stream event") {
            AudioStreamEvent::AudioDelta { data, .. } => out.extend(data),
            AudioStreamEvent::Done { .. } => break,
            _ => {}
        }
    }
    out
}

async fn collect_transcript_text(
    mut stream: siumai::provider_ext::openai::ext::transcription_streaming::OpenAiTranscriptionStream,
) -> String {
    use futures_util::StreamExt;

    let mut out = String::new();
    while let Some(item) = stream.next().await {
        match item.expect("transcription stream event") {
                siumai::provider_ext::openai::ext::transcription_streaming::OpenAiTranscriptionStreamEvent::TextDelta {
                    delta,
                    ..
                } => out.push_str(&delta),
                siumai::provider_ext::openai::ext::transcription_streaming::OpenAiTranscriptionStreamEvent::Done {
                    text,
                    ..
                } => {
                    if out.is_empty() {
                        return text.unwrap_or_default();
                    }
                    if let Some(text) = text {
                        assert_eq!(text, out);
                    }
                    return out;
                }
                _ => {}
            }
    }
    out
}

async fn collect_generic_transcript_text(mut stream: siumai_core::types::AudioStream) -> String {
    use futures_util::StreamExt;

    let mut out = String::new();
    while let Some(item) = stream.next().await {
        match item.expect("generic transcription stream event") {
            AudioStreamEvent::Metadata { metadata, .. } => {
                let event_type = metadata.get("event_type").and_then(|value| value.as_str());
                if event_type == Some("transcript.text.delta")
                    && let Some(delta) = metadata.get("text_delta").and_then(|value| value.as_str())
                {
                    out.push_str(delta);
                }
            }
            AudioStreamEvent::Done { metadata, .. } => {
                if let Some(text) = metadata.get("text").and_then(|value| value.as_str()) {
                    if out.is_empty() {
                        return text.to_string();
                    }
                    assert_eq!(text, out);
                }
                return out;
            }
            _ => {}
        }
    }
    out
}

fn openai_reasoning_response_json(model: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "resp_reasoning_1",
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "enc_payload_123",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": "Let me think."
                    }
                ]
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Final answer."
                    }
                ]
            }
        ],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 2,
            "output_tokens_details": {
                "reasoning_tokens": 1
            },
            "total_tokens": 3
        },
        "finish_reason": "stop"
    })
}

fn openai_reasoning_stream_body(model: &str) -> Vec<u8> {
    let response_json = openai_reasoning_response_json(model);
    let events = vec![
        (
            "response.created",
            serde_json::json!({
                "type": "response.created",
                "response": {
                    "id": "resp_reasoning_1",
                    "model": model,
                    "status": "in_progress",
                    "created_at": 1_735_689_600,
                    "output": []
                }
            }),
        ),
        (
            "response.output_item.added",
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "enc_payload_123",
                    "summary": []
                }
            }),
        ),
        (
            "response.reasoning_summary_part.added",
            serde_json::json!({
                "type": "response.reasoning_summary_part.added",
                "item_id": "rs_1",
                "summary_index": 0,
                "part": {
                    "type": "summary_text",
                    "text": ""
                }
            }),
        ),
        (
            "response.reasoning_summary_text.delta",
            serde_json::json!({
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "summary_index": 0,
                "delta": "Let me think."
            }),
        ),
        (
            "response.output_item.done",
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "enc_payload_123",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": "Let me think."
                        }
                    ]
                }
            }),
        ),
        (
            "response.completed",
            serde_json::json!({
                "type": "response.completed",
                "response": response_json
            }),
        ),
    ];

    events
        .into_iter()
        .map(|(event, data)| {
            format!(
                "event: {event}\ndata: {}\n\n",
                serde_json::to_string(&data).expect("serialize sse event")
            )
        })
        .collect::<String>()
        .into_bytes()
}

fn assert_openai_default_options_request(req: &HttpTransportRequest, base_url: &str, model: &str) {
    assert_eq!(req.url, format!("{base_url}/responses"));
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(
        req.body["previous_response_id"],
        serde_json::json!("resp_default")
    );
    assert_eq!(req.body["reasoning"]["effort"], serde_json::json!("low"));
    assert_eq!(
        req.body["reasoning"]["summary"],
        serde_json::json!("detailed")
    );
}

fn assert_openai_default_options_stream_request(
    req: &HttpTransportRequest,
    base_url: &str,
    model: &str,
) {
    assert_openai_default_options_request(req, base_url, model);
    assert_eq!(req.body["stream"], serde_json::json!(true));
    assert_eq!(
        header_value(req, "accept"),
        Some("text/event-stream".to_string())
    );
}

fn assert_openai_reasoning_response(response: &siumai::prelude::unified::ChatResponse) {
    assert_eq!(response.content_text(), Some("Final answer."));
    assert_eq!(response.reasoning(), vec!["Let me think.".to_string()]);
    assert_eq!(
        response.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        response
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens()),
        Some(3)
    );
    assert_eq!(
        response
            .usage
            .as_ref()
            .and_then(|usage| usage.completion_tokens_details.as_ref())
            .and_then(|details| details.reasoning_tokens),
        Some(1)
    );

    let parts = response
        .content
        .as_multimodal()
        .expect("expected multimodal content");
    let reasoning_part = parts
        .iter()
        .find(|part| matches!(part, ContentPart::Reasoning { .. }))
        .expect("expected reasoning content part");
    let part_meta = reasoning_part
        .openai_metadata()
        .expect("reasoning content metadata");
    assert_eq!(part_meta.item_id.as_deref(), Some("rs_1"));
    assert_eq!(
        part_meta.reasoning_encrypted_content.as_deref(),
        Some("enc_payload_123")
    );
}

#[tokio::test]
async fn openai_siumai_provider_config_embedding_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model("text-embedding-3-small")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model("text-embedding-3-small")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("text-embedding-3-small")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let mut request =
        EmbeddingRequest::single("hello embedding parity").with_model("text-embedding-3-small");
    request.provider_options_map.insert(
        "openai",
        serde_json::json!({
            "dimensions": 512,
            "user": "end_user_1"
        }),
    );

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://example.com/v1/embeddings");
    assert_eq!(
        siumai_req.body["model"],
        serde_json::json!("text-embedding-3-small")
    );
    assert_eq!(siumai_req.body["dimensions"], serde_json::json!(512));
    assert_eq!(siumai_req.body["user"], serde_json::json!("end_user_1"));
    assert_eq!(
        siumai_req.body["encoding_format"],
        serde_json::json!("float")
    );
}

#[tokio::test]
async fn openai_native_siumai_provider_config_rerank_request_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "gpt-4o";
    let base_url = "https://api.openai.com/v1";

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_rerank_request_with_model(model).with_top_n(1);

    let siumai_err = siumai_client
        .rerank(request.clone())
        .await
        .expect_err("native openai rerank should be unsupported");
    let provider_err =
            <siumai::provider_ext::openai::OpenAiClient as siumai::extensions::RerankCapability>::rerank(
                &provider_client,
                request.clone(),
            )
            .await
            .expect_err("native openai rerank should be unsupported");
    let config_err = <siumai::provider_ext::openai::OpenAiClient as siumai::extensions::RerankCapability>::rerank(
            &config_client,
            request,
        )
        .await
        .expect_err("native openai rerank should be unsupported");

    assert_unsupported_operation(&siumai_err);
    assert_unsupported_operation(&provider_err);
    assert_unsupported_operation(&config_err);
    assert!(siumai_client.as_rerank_capability().is_none());
    assert!(provider_client.as_rerank_capability().is_none());
    assert!(config_client.as_rerank_capability().is_none());
    assert_capture_transports_unused(&[&siumai_transport, &provider_transport, &config_transport]);
}

#[tokio::test]
async fn openai_native_registry_rerank_request_is_intentionally_unsupported() {
    let registry_transport = CaptureTransport::default();
    let registry = make_registry(
        Arc::new(registry_transport.clone()),
        "https://api.openai.com/v1",
    );
    let err = match registry.reranking_model("openai:gpt-4o") {
        Ok(_) => panic!("native openai registry rerank handle should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&err);
    assert_capture_transports_unused(&[&registry_transport]);
}

#[tokio::test]
async fn openai_siumai_provider_config_tts_request_are_equivalent() {
    let siumai_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let provider_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let config_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let registry_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");

    let base_url = "https://example.com/v1";
    let model = "gpt-4o-mini-tts";

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .speech_model("openai:gpt-4o-mini-tts")
        .expect("build registry speech model");

    let request = TtsRequest::new("hello from openai".to_string())
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
    assert_eq!(siumai_req.url, "https://example.com/v1/audio/speech");
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!("hello from openai")
    );
    assert_eq!(siumai_req.body["voice"], serde_json::json!("alloy"));
    assert_eq!(siumai_req.body["response_format"], serde_json::json!("mp3"));
    let speed = siumai_req.body["speed"]
        .as_f64()
        .expect("speed should serialize as number");
    assert!((speed - 1.1).abs() < 1e-6);
}

#[tokio::test]
async fn openai_siumai_provider_config_stt_request_are_equivalent() {
    let siumai_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from openai",
        "language": "en"
    }));
    let provider_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from openai",
        "language": "en"
    }));
    let config_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from openai",
        "language": "en"
    }));
    let registry_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from openai",
        "language": "en"
    }));

    let base_url = "https://example.com/v1";
    let model = "whisper-1";

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .transcription_model("openai:whisper-1")
        .expect("build registry transcription model");

    let request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg")
        .with_provider_option("openai", serde_json::json!({ "language": "en" }))
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

    assert_eq!(siumai_resp.text, "hello from openai");
    assert_eq!(provider_resp.text, "hello from openai");
    assert_eq!(config_resp.text, "hello from openai");
    assert_eq!(registry_resp.text, "hello from openai");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_multipart_requests_equivalent(&siumai_req, &provider_req);
    assert_multipart_requests_equivalent(&siumai_req, &config_req);
    assert_multipart_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.com/v1/audio/transcriptions"
    );

    let body_text = normalize_multipart_body(&siumai_req);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("whisper-1"));
    assert!(body_text.contains("name=\"response_format\""));
    assert!(body_text.contains("json"));
    assert!(body_text.contains("name=\"language\""));
    assert!(body_text.contains("en"));
    assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
    assert!(body_text.contains("Content-Type: audio/mpeg"));
    assert!(body_text.contains("abc"));
}

#[tokio::test]
async fn openai_siumai_provider_config_translation_request_are_equivalent() {
    let response = serde_json::json!({
        "text": "hello (translated)",
        "language": "en",
        "duration": 1.25,
        "usage": { "seconds": 1.25 },
        "segments": [{ "id": 0, "text": "hello (translated)" }]
    });

    let siumai_transport = MultipartCaptureTransport::new(response.clone());
    let provider_transport = MultipartCaptureTransport::new(response.clone());
    let config_transport = MultipartCaptureTransport::new(response.clone());
    let registry_transport = MultipartCaptureTransport::new(response);

    let base_url = "https://example.com/v1";
    let model = "whisper-1";

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .transcription_model("openai:whisper-1")
        .expect("build registry transcription model");

    let mut request =
        siumai_core::types::AudioTranslationRequest::from_audio(b"hello".to_vec(), "audio/mpeg")
            .with_media_type("audio/mpeg".to_string());
    request
        .extra_params
        .insert("prompt".to_string(), serde_json::json!("translate this"));
    request
        .extra_params
        .insert("temperature".to_string(), serde_json::json!(0));
    request
        .extra_params
        .insert("response_format".to_string(), serde_json::json!("json"));

    let siumai_resp = siumai_client
        .as_transcription_extras()
        .expect("siumai transcription extras")
        .audio_translate(request.clone())
        .await
        .expect("siumai translation ok");
    let provider_resp = provider_client
        .as_transcription_extras()
        .expect("provider transcription extras")
        .audio_translate(request.clone())
        .await
        .expect("provider translation ok");
    let config_resp = config_client
        .as_transcription_extras()
        .expect("config transcription extras")
        .audio_translate(request.clone())
        .await
        .expect("config translation ok");
    let registry_resp = registry_model
        .audio_translate(request)
        .await
        .expect("registry translation ok");

    assert_eq!(siumai_resp.text, "hello (translated)");
    assert_eq!(provider_resp.text, "hello (translated)");
    assert_eq!(config_resp.text, "hello (translated)");
    assert_eq!(registry_resp.text, "hello (translated)");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_multipart_requests_equivalent(&siumai_req, &provider_req);
    assert_multipart_requests_equivalent(&siumai_req, &config_req);
    assert_multipart_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "https://example.com/v1/audio/translations");

    let body_text = normalize_multipart_body(&siumai_req);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("whisper-1"));
    assert!(body_text.contains("name=\"response_format\""));
    assert!(body_text.contains("json"));
    assert!(body_text.contains("name=\"prompt\""));
    assert!(body_text.contains("translate this"));
    assert!(body_text.contains("name=\"temperature\""));
    assert!(body_text.contains("0"));
    assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
    assert!(body_text.contains("Content-Type: audio/mpeg"));
    assert!(body_text.contains("hello"));
}

#[tokio::test]
async fn openai_siumai_provider_config_image_generation_request_are_equivalent() {
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

    let base_url = "https://example.com/v1";
    let model = "dall-e-3";

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .image_model("openai:dall-e-3")
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
    assert_eq!(siumai_req.url, "https://example.com/v1/images/generations");
    assert_eq!(siumai_req.body["model"], serde_json::json!("dall-e-3"));
    assert_eq!(
        siumai_req.body["prompt"],
        serde_json::json!("a tiny purple robot")
    );
    assert_eq!(siumai_req.body["size"], serde_json::json!("1024x1024"));
    assert_eq!(siumai_req.body["n"], serde_json::json!(1));
    assert_eq!(siumai_req.body["response_format"], serde_json::json!("url"));
}

#[tokio::test]
async fn openai_siumai_provider_config_image_edit_data_url_request_are_equivalent() {
    let response = serde_json::json!({
        "created": 123,
        "data": [
            {
                "b64_json": "aW1hZ2UtMQ=="
            }
        ]
    });

    let siumai_transport = MultipartCaptureTransport::new(response.clone());
    let provider_transport = MultipartCaptureTransport::new(response.clone());
    let config_transport = MultipartCaptureTransport::new(response.clone());
    let registry_transport = MultipartCaptureTransport::new(response);

    let base_url = "https://example.com/v1";
    let model = "gpt-image-1";

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .image_model("openai:gpt-image-1")
        .expect("build registry image model");

    let request = make_data_url_image_edit_request_with_model(model);

    let siumai_resp = siumai_client
        .edit_image(request.clone())
        .await
        .expect("siumai image edit ok");
    let provider_resp = provider_client
        .edit_image(request.clone())
        .await
        .expect("provider image edit ok");
    let config_resp = config_client
        .edit_image(request.clone())
        .await
        .expect("config image edit ok");
    let registry_resp = registry_model
        .edit_image(request)
        .await
        .expect("registry image edit ok");

    assert_eq!(
        siumai_resp.images[0].b64_json.as_deref(),
        Some("aW1hZ2UtMQ==")
    );
    assert_eq!(
        provider_resp.images[0].b64_json.as_deref(),
        Some("aW1hZ2UtMQ==")
    );
    assert_eq!(
        config_resp.images[0].b64_json.as_deref(),
        Some("aW1hZ2UtMQ==")
    );
    assert_eq!(
        registry_resp.images[0].b64_json.as_deref(),
        Some("aW1hZ2UtMQ==")
    );

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_multipart_requests_equivalent(&siumai_req, &provider_req);
    assert_multipart_requests_equivalent(&siumai_req, &config_req);
    assert_multipart_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "https://example.com/v1/images/edits");

    let body_text = normalize_multipart_body(&siumai_req);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("gpt-image-1"));
    assert!(body_text.contains("name=\"prompt\""));
    assert!(body_text.contains("replace the background with a neon skyline"));
    assert!(body_text.contains("name=\"response_format\""));
    assert!(body_text.contains("b64_json"));
    assert!(body_text.contains("name=\"image\""));
    assert!(body_text.contains("filename=\"image-0\""));
    assert!(body_text.contains("name=\"mask\""));
    assert!(body_text.contains("filename=\"mask\""));
    assert!(body_text.contains("Content-Type: image/png"));
    assert!(body_text.contains("image-one"));
    assert!(body_text.contains("mask-one"));
}

#[tokio::test]
async fn openai_siumai_provider_config_tts_sse_request_are_equivalent() {
    let stream_body = concat!(
            "data: {\"type\":\"speech.audio.delta\",\"audio\":\"YWJj\"}\n\n",
            "data: {\"type\":\"speech.audio.delta\",\"audio\":\"ZA==\"}\n\n",
            "data: {\"type\":\"speech.audio.done\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}\n\n",
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "gpt-4o-mini-tts";
    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");
    let siumai_client = siumai_client
        .client()
        .as_any()
        .downcast_ref::<siumai::provider_ext::openai::OpenAiClient>()
        .expect("siumai wrapper should contain provider-owned OpenAiClient");

    let provider_client = Provider::openai()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry_factory = openai_builtin_factory();
    let registry_client = registry_factory
        .compat_speech_client_with_ctx(
            model,
            &BuildContext {
                provider_id: Some("openai".to_string()),
                api_key: Some("test-key".to_string()),
                base_url: Some("https://example.com/v1".to_string()),
                http_transport: Some(Arc::new(registry_transport.clone())),
                ..Default::default()
            },
        )
        .await
        .expect("build registry client");
    let registry_client = registry_client
        .as_any()
        .downcast_ref::<siumai::provider_ext::openai::OpenAiClient>()
        .expect("registry should build provider-owned OpenAiClient");

    let request = TtsRequest::new("hello from openai".to_string())
        .with_voice("alloy".to_string())
        .with_format("mp3".to_string())
        .with_speed(1.1);

    let siumai_audio = collect_tts_audio(
        siumai::provider_ext::openai::ext::speech_streaming::tts_sse_stream(
            siumai_client,
            request.clone(),
        )
        .await
        .expect("siumai tts sse ok"),
    )
    .await;
    let provider_audio = collect_tts_audio(
        siumai::provider_ext::openai::ext::speech_streaming::tts_sse_stream(
            &provider_client,
            request.clone(),
        )
        .await
        .expect("provider tts sse ok"),
    )
    .await;
    let config_audio = collect_tts_audio(
        siumai::provider_ext::openai::ext::speech_streaming::tts_sse_stream(
            &config_client,
            request.clone(),
        )
        .await
        .expect("config tts sse ok"),
    )
    .await;
    let registry_audio = collect_tts_audio(
        siumai::provider_ext::openai::ext::speech_streaming::tts_sse_stream(
            registry_client,
            request,
        )
        .await
        .expect("registry tts sse ok"),
    )
    .await;

    assert_eq!(siumai_audio, b"abcd");
    assert_eq!(provider_audio, b"abcd");
    assert_eq!(config_audio, b"abcd");
    assert_eq!(registry_audio, b"abcd");

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
    assert_eq!(siumai_req.url, "https://example.com/v1/audio/speech");
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!("hello from openai")
    );
    assert_eq!(siumai_req.body["voice"], serde_json::json!("alloy"));
    assert_eq!(siumai_req.body["response_format"], serde_json::json!("mp3"));
    assert_eq!(siumai_req.body["stream_format"], serde_json::json!("sse"));
    let speed = siumai_req.body["speed"]
        .as_f64()
        .expect("speed should serialize as number");
    assert!((speed - 1.1).abs() < 1e-6);
}

#[tokio::test]
async fn openai_siumai_provider_config_stt_sse_request_are_equivalent() {
    let stream_body = concat!(
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"hel\"}\n\n",
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"lo\"}\n\n",
            "data: {\"type\":\"transcript.text.done\",\"text\":\"hello\",\"usage\":{\"type\":\"tokens\",\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}\n\n",
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = MultipartSseSuccessTransport::new(stream_body.clone());
    let provider_transport = MultipartSseSuccessTransport::new(stream_body.clone());
    let config_transport = MultipartSseSuccessTransport::new(stream_body.clone());
    let registry_transport = MultipartSseSuccessTransport::new(stream_body);

    let model = "gpt-4o-mini-transcribe";
    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");
    let siumai_client = siumai_client
        .client()
        .as_any()
        .downcast_ref::<siumai::provider_ext::openai::OpenAiClient>()
        .expect("siumai wrapper should contain provider-owned OpenAiClient");

    let provider_client = Provider::openai()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry_factory = openai_builtin_factory();
    let registry_client = registry_factory
        .compat_transcription_client_with_ctx(
            model,
            &BuildContext {
                provider_id: Some("openai".to_string()),
                api_key: Some("test-key".to_string()),
                base_url: Some("https://example.com/v1".to_string()),
                http_transport: Some(Arc::new(registry_transport.clone())),
                ..Default::default()
            },
        )
        .await
        .expect("build registry client");
    let registry_client = registry_client
        .as_any()
        .downcast_ref::<siumai::provider_ext::openai::OpenAiClient>()
        .expect("registry should build provider-owned OpenAiClient");

    let request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg")
        .with_provider_option("openai", serde_json::json!({ "language": "en" }));

    let siumai_text = collect_transcript_text(
        siumai::provider_ext::openai::ext::transcription_streaming::stt_sse_stream(
            siumai_client,
            request.clone(),
        )
        .await
        .expect("siumai stt sse ok"),
    )
    .await;
    let provider_text = collect_transcript_text(
        siumai::provider_ext::openai::ext::transcription_streaming::stt_sse_stream(
            &provider_client,
            request.clone(),
        )
        .await
        .expect("provider stt sse ok"),
    )
    .await;
    let config_text = collect_transcript_text(
        siumai::provider_ext::openai::ext::transcription_streaming::stt_sse_stream(
            &config_client,
            request.clone(),
        )
        .await
        .expect("config stt sse ok"),
    )
    .await;
    let registry_text = collect_transcript_text(
        siumai::provider_ext::openai::ext::transcription_streaming::stt_sse_stream(
            registry_client,
            request,
        )
        .await
        .expect("registry stt sse ok"),
    )
    .await;

    assert_eq!(siumai_text, "hello");
    assert_eq!(provider_text, "hello");
    assert_eq!(config_text, "hello");
    assert_eq!(registry_text, "hello");

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai multipart stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider multipart stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config multipart stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry multipart stream request");

    assert_multipart_requests_equivalent(&siumai_req, &provider_req);
    assert_multipart_requests_equivalent(&siumai_req, &config_req);
    assert_multipart_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.com/v1/audio/transcriptions"
    );
    assert_eq!(
        siumai_req
            .headers
            .get("accept")
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string),
        Some("text/event-stream".to_string())
    );

    let body_text = normalize_multipart_body(&siumai_req);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("gpt-4o-mini-transcribe"));
    assert!(body_text.contains("name=\"language\""));
    assert!(body_text.contains("en"));
    assert!(body_text.contains("name=\"stream\""));
    assert!(body_text.contains("true"));
    assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
    assert!(body_text.contains("Content-Type: audio/mpeg"));
    assert!(body_text.contains("abc"));
}

#[tokio::test]
async fn openai_siumai_provider_config_speech_extras_tts_stream_are_equivalent() {
    let stream_body = concat!(
            "data: {\"type\":\"speech.audio.delta\",\"audio\":\"YWJj\"}\n\n",
            "data: {\"type\":\"speech.audio.delta\",\"audio\":\"ZA==\"}\n\n",
            "data: {\"type\":\"speech.audio.done\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}\n\n",
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "gpt-4o-mini-tts";
    let base_url = "https://example.com/v1";
    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .speech_model("openai:gpt-4o-mini-tts")
        .expect("build registry speech model");

    let request = TtsRequest::new("hello from openai".to_string())
        .with_voice("alloy".to_string())
        .with_format("mp3".to_string())
        .with_speed(1.1);

    let siumai_audio = collect_tts_audio(
        siumai_client
            .as_speech_extras()
            .expect("siumai speech extras")
            .tts_stream(request.clone())
            .await
            .expect("siumai generic tts stream ok"),
    )
    .await;
    let provider_audio = collect_tts_audio(
        provider_client
            .as_speech_extras()
            .expect("provider speech extras")
            .tts_stream(request.clone())
            .await
            .expect("provider generic tts stream ok"),
    )
    .await;
    let config_audio = collect_tts_audio(
        config_client
            .as_speech_extras()
            .expect("config speech extras")
            .tts_stream(request.clone())
            .await
            .expect("config generic tts stream ok"),
    )
    .await;
    let registry_audio = collect_tts_audio(
        registry_model
            .tts_stream(request)
            .await
            .expect("registry generic tts stream ok"),
    )
    .await;

    assert_eq!(siumai_audio, b"abcd");
    assert_eq!(provider_audio, b"abcd");
    assert_eq!(config_audio, b"abcd");
    assert_eq!(registry_audio, b"abcd");

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai speech extras stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider speech extras stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config speech extras stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry speech extras stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "https://example.com/v1/audio/speech");
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!("hello from openai")
    );
    assert_eq!(siumai_req.body["voice"], serde_json::json!("alloy"));
    assert_eq!(siumai_req.body["response_format"], serde_json::json!("mp3"));
    assert_eq!(siumai_req.body["stream_format"], serde_json::json!("sse"));
    let speed = siumai_req.body["speed"]
        .as_f64()
        .expect("speed should serialize as number");
    assert!((speed - 1.1).abs() < 1e-6);
}

#[tokio::test]
async fn openai_siumai_provider_config_transcription_extras_stt_stream_are_equivalent() {
    let stream_body = concat!(
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"hel\"}\n\n",
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"lo\"}\n\n",
            "data: {\"type\":\"transcript.text.done\",\"text\":\"hello\",\"usage\":{\"type\":\"tokens\",\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}\n\n",
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = MultipartSseSuccessTransport::new(stream_body.clone());
    let provider_transport = MultipartSseSuccessTransport::new(stream_body.clone());
    let config_transport = MultipartSseSuccessTransport::new(stream_body.clone());
    let registry_transport = MultipartSseSuccessTransport::new(stream_body);

    let model = "gpt-4o-mini-transcribe";
    let base_url = "https://example.com/v1";
    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .transcription_model("openai:gpt-4o-mini-transcribe")
        .expect("build registry transcription model");

    let request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg")
        .with_provider_option("openai", serde_json::json!({ "language": "en" }));

    let siumai_text = collect_generic_transcript_text(
        siumai_client
            .as_transcription_extras()
            .expect("siumai transcription extras")
            .stt_stream(request.clone())
            .await
            .expect("siumai generic stt stream ok"),
    )
    .await;
    let provider_text = collect_generic_transcript_text(
        provider_client
            .as_transcription_extras()
            .expect("provider transcription extras")
            .stt_stream(request.clone())
            .await
            .expect("provider generic stt stream ok"),
    )
    .await;
    let config_text = collect_generic_transcript_text(
        config_client
            .as_transcription_extras()
            .expect("config transcription extras")
            .stt_stream(request.clone())
            .await
            .expect("config generic stt stream ok"),
    )
    .await;
    let registry_text = collect_generic_transcript_text(
        registry_model
            .stt_stream(request)
            .await
            .expect("registry generic stt stream ok"),
    )
    .await;

    assert_eq!(siumai_text, "hello");
    assert_eq!(provider_text, "hello");
    assert_eq!(config_text, "hello");
    assert_eq!(registry_text, "hello");

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai transcription extras stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider transcription extras stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config transcription extras stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry transcription extras stream request");

    assert_multipart_requests_equivalent(&siumai_req, &provider_req);
    assert_multipart_requests_equivalent(&siumai_req, &config_req);
    assert_multipart_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.com/v1/audio/transcriptions"
    );
    assert_eq!(
        siumai_req
            .headers
            .get("accept")
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string),
        Some("text/event-stream".to_string())
    );

    let body_text = normalize_multipart_body(&siumai_req);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("gpt-4o-mini-transcribe"));
    assert!(body_text.contains("name=\"language\""));
    assert!(body_text.contains("en"));
    assert!(body_text.contains("name=\"stream\""));
    assert!(body_text.contains("true"));
    assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
    assert!(body_text.contains("Content-Type: audio/mpeg"));
    assert!(body_text.contains("abc"));
}

#[tokio::test]
async fn openai_responses_siumai_provider_config_chat_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "gpt-4o-mini";

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai_responses()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_use_responses_api(true)
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
    assert_eq!(siumai_req.url, "https://example.com/v1/responses");
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert!(siumai_req.body.get("input").is_some());
}

#[tokio::test]
async fn openai_default_options_match_public_request_shape() {
    let model = "o3-mini";
    let base_url = "https://example.com/v1";
    let response_json = openai_reasoning_response_json(model);

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_openai_options(OpenAiOptions::new().with_reasoning_effort(ReasoningEffort::Low))
        .with_openai_options(
            OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new()
                    .with_previous_response("resp_default".to_string())
                    .with_reasoning_summary("detailed"),
            ),
        )
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_openai_options(OpenAiOptions::new().with_reasoning_effort(ReasoningEffort::Low))
        .with_openai_options(
            OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new()
                    .with_previous_response("resp_default".to_string())
                    .with_reasoning_summary("detailed"),
            ),
        )
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_openai_options(OpenAiOptions::new().with_reasoning_effort(ReasoningEffort::Low))
            .with_openai_options(
                OpenAiOptions::new().with_responses_api(
                    ResponsesApiConfig::new()
                        .with_previous_response("resp_default".to_string())
                        .with_reasoning_summary("detailed"),
                ),
            )
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
    assert_openai_default_options_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn openai_chat_siumai_provider_config_chat_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "gpt-4o-mini";

    let siumai_client = Siumai::builder()
        .openai_chat()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai_chat()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_use_responses_api(false)
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
    assert_eq!(siumai_req.url, "https://example.com/v1/chat/completions");
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["messages"][0]["content"],
        serde_json::json!("hi")
    );
}

#[tokio::test]
async fn openai_completion_siumai_provider_config_registry_request_are_equivalent() {
    let response_json = serde_json::json!({
        "id": "cmpl-openai-test",
        "object": "text_completion",
        "created": 1_718_345_013u64,
        "model": "gpt-3.5-turbo-instruct",
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

    let model = "gpt-3.5-turbo-instruct";
    let base_url = "https://example.com/v1";

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .completion_model(&format!("openai:{model}"))
        .expect("build registry completion model");

    let request = siumai::prelude::unified::CompletionRequest::from_prompt(vec![
        ChatMessage::system("Be terse.").build(),
        ChatMessage::user("Hello").build(),
        ChatMessage::assistant("Hi").build(),
        ChatMessage::user("Continue").build(),
    ])
    .with_model(model)
    .with_provider_option("openai", serde_json::json!({ "suffix": "!" }));

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
    assert_eq!(siumai_req.url, "https://example.com/v1/completions");
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
async fn openai_completion_stream_public_paths_keep_raw_chunks_runtime_only() {
    use futures_util::StreamExt;

    let model = "gpt-3.5-turbo-instruct";
    let stream_body = concat!(
            "data: {\"id\":\"cmpl-openai-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"gpt-3.5-turbo-instruct\",\"choices\":[{\"text\":\"hello\",\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"cmpl-openai-stream\",\"object\":\"text_completion\",\"created\":1718345013,\"model\":\"gpt-3.5-turbo-instruct\",\"choices\":[{\"text\":\" world\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n",
            "data: [DONE]\n\n"
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let base_url = "https://example.com/v1";

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .completion_model(&format!("openai:{model}"))
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
    assert_eq!(siumai_req.url, "https://example.com/v1/completions");
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
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

#[tokio::test]
async fn openai_chat_public_paths_use_canonical_image_detail_provider_options() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "gpt-4o-mini";
    let base_url = "https://example.com/v1";

    let siumai_client = Siumai::builder()
        .openai_chat()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai_chat()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_use_responses_api(false)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_chat_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("openai-chat:gpt-4o-mini")
        .expect("build registry language model");

    let canonical_part = ContentPart::File {
        source: siumai::prelude::unified::FilePartSource::base64("AAEC"),
        media_type: "image/png".to_string(),
        filename: None,
        provider_options: {
            let mut options = siumai::prelude::unified::ProviderOptionsMap::default();
            options.insert(
                "openai",
                serde_json::json!({
                    "imageDetail": "high"
                }),
            );
            options
        },
        provider_metadata: Some(std::collections::HashMap::from([(
            "openai".to_string(),
            serde_json::json!({
                "imageDetail": "low"
            }),
        )])),
    };

    let legacy_only_part = ContentPart::File {
        source: siumai::prelude::unified::FilePartSource::base64("AQID"),
        media_type: "image/png".to_string(),
        filename: None,
        provider_options: siumai::prelude::unified::ProviderOptionsMap::default(),
        provider_metadata: Some(std::collections::HashMap::from([(
            "openai".to_string(),
            serde_json::json!({
                "imageDetail": "low"
            }),
        )])),
    };

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![siumai::prelude::unified::ChatMessage {
            role: siumai::prelude::unified::MessageRole::User,
            content: siumai::prelude::unified::MessageContent::MultiModal(vec![
                canonical_part,
                legacy_only_part,
            ]),
            provider_options: siumai::prelude::unified::ProviderOptionsMap::default(),
            metadata: siumai::prelude::unified::MessageMetadata::default(),
        }])
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
    assert_eq!(siumai_req.url, "https://example.com/v1/chat/completions");

    let parts = siumai_req.body["messages"][0]["content"]
        .as_array()
        .expect("expected multimodal content array");
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["type"], serde_json::json!("image_url"));
    assert_eq!(parts[0]["image_url"]["detail"], serde_json::json!("high"));
    assert!(parts[1]["image_url"].get("detail").is_none());
}

#[tokio::test]
async fn openai_registry_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let openai_transport = CaptureTransport::default();

    let registry = make_openai_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(openai_transport.clone()),
    );

    let handle = registry
        .language_model("openai:gpt-4o")
        .expect("build openai handle");

    let _ = handle
        .chat_request(make_chat_request_with_model("gpt-4o"))
        .await;

    let req = openai_transport.take().expect("captured openai request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/openai/v1/responses");
    assert_eq!(req.body["model"], serde_json::json!("gpt-4o"));
}

#[tokio::test]
async fn openai_registry_stream_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let openai_transport = CaptureTransport::default();

    let registry = make_openai_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(openai_transport.clone()),
    );

    let handle = registry
        .language_model("openai:gpt-4o")
        .expect("build openai handle");

    let _ = handle
        .chat_stream_request(make_chat_request_with_model("gpt-4o"))
        .await;

    let req = openai_transport
        .take_stream()
        .expect("captured openai stream request");
    assert!(global_transport.take().is_none());
    assert!(global_transport.take_stream().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/openai/v1/responses");
    assert_eq!(
        header_value(&req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(req.body["model"], serde_json::json!("gpt-4o"));
}

#[tokio::test]
async fn openai_registry_speech_handle_prefers_provider_specific_build_overrides() {
    let global_transport = BinaryCaptureTransport::new(vec![9, 9, 9], "audio/mpeg");
    let openai_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
    let model = "gpt-4o-mini-tts";

    let registry = make_openai_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(openai_transport.clone()),
    );

    let registry_model = registry
        .speech_model("openai:gpt-4o-mini-tts")
        .expect("build openai speech model");

    let response = siumai::speech::SpeechModel::synthesize(
        &registry_model,
        TtsRequest::new("hello from openai".to_string())
            .with_voice("alloy".to_string())
            .with_format("mp3".to_string()),
    )
    .await
    .expect("registry tts ok");

    assert_eq!(response.audio_data, vec![1, 2, 3, 4]);

    let req = openai_transport
        .take()
        .expect("captured openai speech request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/openai/v1/audio/speech");
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(req.body["input"], serde_json::json!("hello from openai"));
    assert_eq!(req.body["voice"], serde_json::json!("alloy"));
    assert_eq!(req.body["response_format"], serde_json::json!("mp3"));
}

#[tokio::test]
async fn openai_registry_transcription_handle_prefers_provider_specific_build_overrides() {
    let global_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from global",
        "language": "en"
    }));
    let openai_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from openai",
        "language": "en"
    }));

    let registry = make_openai_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(openai_transport.clone()),
    );

    let registry_model = registry
        .transcription_model("openai:whisper-1")
        .expect("build openai transcription model");

    let request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");

    let response = registry_model
        .speech_to_text(request)
        .await
        .expect("registry stt ok");

    assert_eq!(response.text, "hello from openai");
    assert_eq!(response.language.as_deref(), Some("en"));

    let req = openai_transport
        .take()
        .expect("captured openai transcription request");
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
        "https://example.com/openai/v1/audio/transcriptions"
    );
    assert!(
        req.headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
    );

    let body_text = normalize_multipart_body(&req);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("whisper-1"));
    assert!(body_text.contains("name=\"response_format\""));
    assert!(body_text.contains("json"));
    assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
    assert!(body_text.contains("Content-Type: audio/mpeg"));
    assert!(body_text.contains("abc"));
}

#[tokio::test]
async fn openai_registry_image_handle_prefers_provider_specific_build_overrides() {
    let image_response = serde_json::json!({
        "created": 123,
        "data": [
            {
                "url": "https://example.com/generated.png",
                "revised_prompt": "a tiny purple robot"
            }
        ]
    });

    let global_transport = JsonSuccessTransport::new(image_response.clone());
    let openai_transport = JsonSuccessTransport::new(image_response);

    let registry = make_openai_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(openai_transport.clone()),
    );

    let handle = registry
        .image_model("openai:dall-e-3")
        .expect("build openai image model");

    let generated = handle
        .generate_images(make_image_request_with_model("dall-e-3"))
        .await
        .expect("generate images through registry handle");

    assert_eq!(
        generated.images[0].url.as_deref(),
        Some("https://example.com/generated.png")
    );
    assert!(global_transport.take().is_none());

    let req = openai_transport.take().expect("captured request");
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/openai/v1/images/generations");
    assert_eq!(req.body["model"], serde_json::json!("dall-e-3"));
    assert_eq!(req.body["prompt"], serde_json::json!("a tiny purple robot"));
    assert_eq!(req.body["size"], serde_json::json!("1024x1024"));
    assert_eq!(req.body["n"], serde_json::json!(1));
    assert_eq!(req.body["response_format"], serde_json::json!("url"));
}

#[tokio::test]
async fn openai_responses_siumai_provider_config_stable_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "o3-mini";
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai_responses()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_use_responses_api(true)
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
        .response_format(siumai::prelude::unified::ResponseFormat::json_schema(
            schema.clone(),
        ))
        .build()
        .with_openai_options(
            OpenAiOptions::new()
                .with_reasoning_effort(ReasoningEffort::Low)
                .with_responses_api(ResponsesApiConfig::new().with_reasoning_summary("detailed")),
        );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://example.com/v1/responses");
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
async fn openai_chat_siumai_provider_config_stable_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "o3-mini";
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let siumai_client = Siumai::builder()
        .openai_chat()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai_chat()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_use_responses_api(false)
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
        .with_openai_options(OpenAiOptions::new().with_reasoning_effort(ReasoningEffort::Low));

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://example.com/v1/chat/completions");
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
async fn openai_chat_siumai_provider_config_chat_response_logprobs_metadata_are_equivalent() {
    let model = "gpt-4o-mini";
    let response_json = serde_json::json!({
        "id": "chatcmpl-openai-test",
        "object": "chat.completion",
        "created": 1_718_345_013,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello from openai chat"
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
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "total_tokens": 14,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 5,
                "rejected_prediction_tokens": 6
            }
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .openai_chat()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai_chat()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_use_responses_api(false)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let mut request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();
    request
        .provider_options_map
        .insert("openai", serde_json::json!({ "logprobs": 3 }));

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

    assert!(siumai_root.get("openai").is_some());
    assert!(provider_root.get("openai").is_some());
    assert!(config_root.get("openai").is_some());
    assert!(siumai_root.get("azure").is_none());
    assert!(provider_root.get("azure").is_none());
    assert!(config_root.get("azure").is_none());

    let siumai_meta = siumai_resp
        .openai_metadata()
        .expect("siumai openai metadata");
    let provider_meta = provider_resp
        .openai_metadata()
        .expect("provider openai metadata");
    let config_meta = config_resp
        .openai_metadata()
        .expect("config openai metadata");

    assert_eq!(siumai_resp.content_text(), Some("hello from openai chat"));
    assert_eq!(provider_resp.content_text(), Some("hello from openai chat"));
    assert_eq!(config_resp.content_text(), Some("hello from openai chat"));
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

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.1,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(siumai_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(provider_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(config_meta.logprobs, Some(expected_logprobs));
    assert_eq!(siumai_meta.accepted_prediction_tokens, Some(5));
    assert_eq!(provider_meta.accepted_prediction_tokens, Some(5));
    assert_eq!(config_meta.accepted_prediction_tokens, Some(5));
    assert_eq!(siumai_meta.rejected_prediction_tokens, Some(6));
    assert_eq!(provider_meta.rejected_prediction_tokens, Some(6));
    assert_eq!(config_meta.rejected_prediction_tokens, Some(6));

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://example.com/v1/chat/completions");
    assert_eq!(siumai_req.body["logprobs"], serde_json::json!(true));
    assert_eq!(siumai_req.body["top_logprobs"], serde_json::json!(3));
}

#[tokio::test]
async fn openai_chat_siumai_provider_config_stream_end_logprobs_metadata_are_equivalent() {
    let model = "gpt-4o-mini";
    let stream_body = br#"data: {"id":"1","model":"gpt-4o-mini","created":1718345013,"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"gpt-4o-mini","created":1718345013,"choices":[{"index":0,"delta":{"content":" from openai chat","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14,"completion_tokens_details":{"accepted_prediction_tokens":5,"rejected_prediction_tokens":6}}}

data: [DONE]

"#
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .openai_chat()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai_chat()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_use_responses_api(false)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let mut request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();
    request
        .provider_options_map
        .insert("openai", serde_json::json!({ "logprobs": 3 }));

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

    assert!(siumai_root.get("openai").is_some());
    assert!(provider_root.get("openai").is_some());
    assert!(config_root.get("openai").is_some());
    assert!(siumai_root.get("azure").is_none());
    assert!(provider_root.get("azure").is_none());
    assert!(config_root.get("azure").is_none());

    let siumai_meta = siumai_resp
        .openai_metadata()
        .expect("siumai openai metadata");
    let provider_meta = provider_resp
        .openai_metadata()
        .expect("provider openai metadata");
    let config_meta = config_resp
        .openai_metadata()
        .expect("config openai metadata");

    assert_eq!(siumai_resp.content_text(), Some("hello from openai chat"));
    assert_eq!(provider_resp.content_text(), Some("hello from openai chat"));
    assert_eq!(config_resp.content_text(), Some("hello from openai chat"));
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

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.1,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(siumai_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(provider_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(config_meta.logprobs, Some(expected_logprobs));
    assert_eq!(siumai_meta.accepted_prediction_tokens, Some(5));
    assert_eq!(provider_meta.accepted_prediction_tokens, Some(5));
    assert_eq!(config_meta.accepted_prediction_tokens, Some(5));
    assert_eq!(siumai_meta.rejected_prediction_tokens, Some(6));
    assert_eq!(provider_meta.rejected_prediction_tokens, Some(6));
    assert_eq!(config_meta.rejected_prediction_tokens, Some(6));

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
    assert_eq!(siumai_req.url, "https://example.com/v1/chat/completions");
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(siumai_req.body["logprobs"], serde_json::json!(true));
    assert_eq!(siumai_req.body["top_logprobs"], serde_json::json!(3));
}

#[tokio::test]
async fn openai_responses_siumai_provider_config_chat_response_source_metadata_are_equivalent() {
    let model = "gpt-4.1";
    let response_json = serde_json::json!({
        "id": "resp_sources_2",
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "See attached files.",
                        "annotations": [
                            {
                                "type": "container_file_citation",
                                "file_id": "file_container_1",
                                "container_id": "container_42",
                                "index": 3,
                                "filename": "bundle.txt",
                                "quote": "Bundle"
                            },
                            {
                                "type": "file_path",
                                "file_id": "file_path_9",
                                "index": 5,
                                "filename": "artifact.bin"
                            }
                        ]
                    }
                ]
            }
        ],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3
        },
        "finish_reason": "stop"
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai_responses()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_use_responses_api(true)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

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

    assert!(siumai_root.get("openai").is_some());
    assert!(provider_root.get("openai").is_some());
    assert!(config_root.get("openai").is_some());
    assert!(siumai_root.get("azure").is_none());
    assert!(provider_root.get("azure").is_none());
    assert!(config_root.get("azure").is_none());

    let siumai_meta = siumai_resp
        .openai_metadata()
        .expect("siumai openai metadata");
    let provider_meta = provider_resp
        .openai_metadata()
        .expect("provider openai metadata");
    let config_meta = config_resp
        .openai_metadata()
        .expect("config openai metadata");

    assert_eq!(siumai_resp.content_text(), Some("See attached files."));
    assert_eq!(provider_resp.content_text(), Some("See attached files."));
    assert_eq!(config_resp.content_text(), Some("See attached files."));
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

    let siumai_sources = siumai_meta.sources.expect("siumai sources");
    let provider_sources = provider_meta.sources.expect("provider sources");
    let config_sources = config_meta.sources.expect("config sources");

    assert_eq!(siumai_sources.len(), 2);
    assert_eq!(provider_sources.len(), 2);
    assert_eq!(config_sources.len(), 2);

    let siumai_container = siumai_sources
        .iter()
        .find(|source| source.url == "file_container_1")
        .expect("siumai container source");
    let provider_container = provider_sources
        .iter()
        .find(|source| source.url == "file_container_1")
        .expect("provider container source");
    let config_container = config_sources
        .iter()
        .find(|source| source.url == "file_container_1")
        .expect("config container source");

    let siumai_container_meta = siumai_container
        .openai_metadata()
        .expect("siumai container metadata");
    let provider_container_meta = provider_container
        .openai_metadata()
        .expect("provider container metadata");
    let config_container_meta = config_container
        .openai_metadata()
        .expect("config container metadata");

    assert_eq!(
        siumai_container_meta.file_id.as_deref(),
        Some("file_container_1")
    );
    assert_eq!(
        provider_container_meta.file_id.as_deref(),
        Some("file_container_1")
    );
    assert_eq!(
        config_container_meta.file_id.as_deref(),
        Some("file_container_1")
    );
    assert_eq!(
        siumai_container_meta.container_id.as_deref(),
        Some("container_42")
    );
    assert_eq!(
        provider_container_meta.container_id.as_deref(),
        Some("container_42")
    );
    assert_eq!(
        config_container_meta.container_id.as_deref(),
        Some("container_42")
    );
    assert_eq!(siumai_container_meta.index, Some(3));
    assert_eq!(provider_container_meta.index, Some(3));
    assert_eq!(config_container_meta.index, Some(3));

    let siumai_file_path = siumai_sources
        .iter()
        .find(|source| source.url == "file_path_9")
        .expect("siumai file path source");
    let provider_file_path = provider_sources
        .iter()
        .find(|source| source.url == "file_path_9")
        .expect("provider file path source");
    let config_file_path = config_sources
        .iter()
        .find(|source| source.url == "file_path_9")
        .expect("config file path source");

    let siumai_file_path_meta = siumai_file_path
        .openai_metadata()
        .expect("siumai file path metadata");
    let provider_file_path_meta = provider_file_path
        .openai_metadata()
        .expect("provider file path metadata");
    let config_file_path_meta = config_file_path
        .openai_metadata()
        .expect("config file path metadata");

    assert_eq!(
        siumai_file_path_meta.file_id.as_deref(),
        Some("file_path_9")
    );
    assert_eq!(
        provider_file_path_meta.file_id.as_deref(),
        Some("file_path_9")
    );
    assert_eq!(
        config_file_path_meta.file_id.as_deref(),
        Some("file_path_9")
    );
    assert!(siumai_file_path_meta.container_id.is_none());
    assert!(provider_file_path_meta.container_id.is_none());
    assert!(config_file_path_meta.container_id.is_none());
    assert_eq!(siumai_file_path_meta.index, Some(5));
    assert_eq!(provider_file_path_meta.index, Some(5));
    assert_eq!(config_file_path_meta.index, Some(5));

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://example.com/v1/responses");
}

#[tokio::test]
async fn openai_chat_registry_response_logprobs_metadata_match_config_path() {
    let model = "gpt-4o-mini";
    let response_json = serde_json::json!({
        "id": "chatcmpl-openai-test",
        "object": "chat.completion",
        "created": 1_718_345_013,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello from openai chat"
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
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "total_tokens": 14
        }
    });

    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_use_responses_api(false)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_chat_registry(
        Arc::new(registry_transport.clone()),
        "https://example.com/v1",
    );
    let registry_model = registry
        .language_model("openai-chat:gpt-4o-mini")
        .expect("build registry language model");

    let mut request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();
    request
        .provider_options_map
        .insert("openai", serde_json::json!({ "logprobs": 3 }));

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

    assert!(config_root.get("openai").is_some());
    assert!(registry_root.get("openai").is_some());
    assert!(config_root.get("azure").is_none());
    assert!(registry_root.get("azure").is_none());

    let config_meta = config_resp
        .openai_metadata()
        .expect("config openai metadata");
    let registry_meta = registry_resp
        .openai_metadata()
        .expect("registry openai metadata");

    assert_eq!(config_resp.content_text(), Some("hello from openai chat"));
    assert_eq!(registry_resp.content_text(), Some("hello from openai chat"));
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

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.1,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(config_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(registry_meta.logprobs, Some(expected_logprobs));

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.url, "https://example.com/v1/chat/completions");
    assert_eq!(registry_req.body["logprobs"], serde_json::json!(true));
    assert_eq!(registry_req.body["top_logprobs"], serde_json::json!(3));
}

#[tokio::test]
async fn openai_chat_registry_stream_end_logprobs_metadata_match_config_path() {
    let model = "gpt-4o-mini";
    let stream_body = br#"data: {"id":"1","model":"gpt-4o-mini","created":1718345013,"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"gpt-4o-mini","created":1718345013,"choices":[{"index":0,"delta":{"content":" from openai chat","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}}

data: [DONE]

"#
        .to_vec();

    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_use_responses_api(false)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_chat_registry(
        Arc::new(registry_transport.clone()),
        "https://example.com/v1",
    );
    let registry_model = registry
        .language_model("openai-chat:gpt-4o-mini")
        .expect("build registry language model");

    let mut request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();
    request
        .provider_options_map
        .insert("openai", serde_json::json!({ "logprobs": 3 }));

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

    assert!(config_root.get("openai").is_some());
    assert!(registry_root.get("openai").is_some());
    assert!(config_root.get("azure").is_none());
    assert!(registry_root.get("azure").is_none());

    let config_meta = config_resp
        .openai_metadata()
        .expect("config openai metadata");
    let registry_meta = registry_resp
        .openai_metadata()
        .expect("registry openai metadata");

    assert_eq!(config_resp.content_text(), Some("hello from openai chat"));
    assert_eq!(registry_resp.content_text(), Some("hello from openai chat"));
    assert_eq!(
        config_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        registry_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.1,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(config_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(registry_meta.logprobs, Some(expected_logprobs));

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.url, "https://example.com/v1/chat/completions");
    assert_eq!(
        header_value(&registry_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(registry_req.body["logprobs"], serde_json::json!(true));
    assert_eq!(registry_req.body["top_logprobs"], serde_json::json!(3));
}

#[tokio::test]
async fn openai_responses_registry_response_source_metadata_match_config_path() {
    let model = "gpt-4.1";
    let response_json = serde_json::json!({
        "id": "resp_sources_2",
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "See attached files.",
                        "annotations": [
                            {
                                "type": "container_file_citation",
                                "file_id": "file_container_1",
                                "container_id": "container_42",
                                "index": 3,
                                "filename": "bundle.txt",
                                "quote": "Bundle"
                            },
                            {
                                "type": "file_path",
                                "file_id": "file_path_9",
                                "index": 5,
                                "filename": "artifact.bin"
                            }
                        ]
                    }
                ]
            }
        ],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3
        },
        "finish_reason": "stop"
    });

    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_use_responses_api(true)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(
        Arc::new(registry_transport.clone()),
        "https://example.com/v1",
    );
    let registry_model = registry
        .language_model("openai:gpt-4.1")
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

    let config_root = config_resp
        .provider_metadata
        .as_ref()
        .expect("config provider metadata");
    let registry_root = registry_resp
        .provider_metadata
        .as_ref()
        .expect("registry provider metadata");

    assert!(config_root.get("openai").is_some());
    assert!(registry_root.get("openai").is_some());
    assert!(config_root.get("azure").is_none());
    assert!(registry_root.get("azure").is_none());

    let config_meta = config_resp
        .openai_metadata()
        .expect("config openai metadata");
    let registry_meta = registry_resp
        .openai_metadata()
        .expect("registry openai metadata");

    assert_eq!(config_resp.content_text(), Some("See attached files."));
    assert_eq!(registry_resp.content_text(), Some("See attached files."));
    assert_eq!(
        config_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        registry_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );

    let config_sources = config_meta.sources.expect("config sources");
    let registry_sources = registry_meta.sources.expect("registry sources");

    assert_eq!(config_sources.len(), 2);
    assert_eq!(registry_sources.len(), 2);

    let config_container = config_sources
        .iter()
        .find(|source| source.url == "file_container_1")
        .expect("config container source");
    let registry_container = registry_sources
        .iter()
        .find(|source| source.url == "file_container_1")
        .expect("registry container source");

    let config_container_meta = config_container
        .openai_metadata()
        .expect("config container metadata");
    let registry_container_meta = registry_container
        .openai_metadata()
        .expect("registry container metadata");

    assert_eq!(
        config_container_meta.file_id.as_deref(),
        Some("file_container_1")
    );
    assert_eq!(
        registry_container_meta.file_id.as_deref(),
        Some("file_container_1")
    );
    assert_eq!(
        config_container_meta.container_id.as_deref(),
        Some("container_42")
    );
    assert_eq!(
        registry_container_meta.container_id.as_deref(),
        Some("container_42")
    );
    assert_eq!(config_container_meta.index, Some(3));
    assert_eq!(registry_container_meta.index, Some(3));

    let config_file_path = config_sources
        .iter()
        .find(|source| source.url == "file_path_9")
        .expect("config file path source");
    let registry_file_path = registry_sources
        .iter()
        .find(|source| source.url == "file_path_9")
        .expect("registry file path source");

    let config_file_path_meta = config_file_path
        .openai_metadata()
        .expect("config file path metadata");
    let registry_file_path_meta = registry_file_path
        .openai_metadata()
        .expect("registry file path metadata");

    assert_eq!(
        config_file_path_meta.file_id.as_deref(),
        Some("file_path_9")
    );
    assert_eq!(
        registry_file_path_meta.file_id.as_deref(),
        Some("file_path_9")
    );
    assert!(config_file_path_meta.container_id.is_none());
    assert!(registry_file_path_meta.container_id.is_none());
    assert_eq!(config_file_path_meta.index, Some(5));
    assert_eq!(registry_file_path_meta.index, Some(5));

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.url, "https://example.com/v1/responses");
}

#[tokio::test]
async fn openai_responses_registry_stream_end_source_metadata_match_config_path() {
    let model = "gpt-4.1";
    let stream_body = concat!(
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_sources_2\",\"model\":\"gpt-4.1\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"See attached files.\",\"annotations\":[{\"type\":\"container_file_citation\",\"file_id\":\"file_container_1\",\"container_id\":\"container_42\",\"index\":3,\"filename\":\"bundle.txt\",\"quote\":\"Bundle\"},{\"type\":\"file_path\",\"file_id\":\"file_path_9\",\"index\":5,\"filename\":\"artifact.bin\"}]}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3},\"finish_reason\":\"stop\"}}\n\n"
        )
        .as_bytes()
        .to_vec();

    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_use_responses_api(true)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(
        Arc::new(registry_transport.clone()),
        "https://example.com/v1",
    );
    let registry_model = registry
        .language_model("openai:gpt-4.1")
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

    assert!(config_root.get("openai").is_some());
    assert!(registry_root.get("openai").is_some());
    assert!(config_root.get("azure").is_none());
    assert!(registry_root.get("azure").is_none());

    let config_meta = config_resp
        .openai_metadata()
        .expect("config openai metadata");
    let registry_meta = registry_resp
        .openai_metadata()
        .expect("registry openai metadata");

    assert_eq!(config_resp.content_text(), Some("See attached files."));
    assert_eq!(registry_resp.content_text(), Some("See attached files."));
    assert_eq!(
        config_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );
    assert_eq!(
        registry_resp.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );

    let config_sources = config_meta.sources.expect("config sources");
    let registry_sources = registry_meta.sources.expect("registry sources");

    assert_eq!(config_sources.len(), 2);
    assert_eq!(registry_sources.len(), 2);

    let config_container = config_sources
        .iter()
        .find(|source| source.url == "file_container_1")
        .expect("config container source");
    let registry_container = registry_sources
        .iter()
        .find(|source| source.url == "file_container_1")
        .expect("registry container source");

    let config_container_meta = config_container
        .openai_metadata()
        .expect("config container metadata");
    let registry_container_meta = registry_container
        .openai_metadata()
        .expect("registry container metadata");

    assert_eq!(
        config_container_meta.file_id.as_deref(),
        Some("file_container_1")
    );
    assert_eq!(
        registry_container_meta.file_id.as_deref(),
        Some("file_container_1")
    );
    assert_eq!(
        config_container_meta.container_id.as_deref(),
        Some("container_42")
    );
    assert_eq!(
        registry_container_meta.container_id.as_deref(),
        Some("container_42")
    );
    assert_eq!(config_container_meta.index, Some(3));
    assert_eq!(registry_container_meta.index, Some(3));

    let config_file_path = config_sources
        .iter()
        .find(|source| source.url == "file_path_9")
        .expect("config file path source");
    let registry_file_path = registry_sources
        .iter()
        .find(|source| source.url == "file_path_9")
        .expect("registry file path source");

    let config_file_path_meta = config_file_path
        .openai_metadata()
        .expect("config file path metadata");
    let registry_file_path_meta = registry_file_path
        .openai_metadata()
        .expect("registry file path metadata");

    assert_eq!(
        config_file_path_meta.file_id.as_deref(),
        Some("file_path_9")
    );
    assert_eq!(
        registry_file_path_meta.file_id.as_deref(),
        Some("file_path_9")
    );
    assert!(config_file_path_meta.container_id.is_none());
    assert!(registry_file_path_meta.container_id.is_none());
    assert_eq!(config_file_path_meta.index, Some(5));
    assert_eq!(registry_file_path_meta.index, Some(5));

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.url, "https://example.com/v1/responses");
    assert_eq!(
        header_value(&registry_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn openai_responses_siumai_provider_config_stream_end_source_metadata_are_equivalent() {
    let model = "gpt-4.1";
    let stream_body = concat!(
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_sources_2\",\"model\":\"gpt-4.1\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"See attached files.\",\"annotations\":[{\"type\":\"container_file_citation\",\"file_id\":\"file_container_1\",\"container_id\":\"container_42\",\"index\":3,\"filename\":\"bundle.txt\",\"quote\":\"Bundle\"},{\"type\":\"file_path\",\"file_id\":\"file_path_9\",\"index\":5,\"filename\":\"artifact.bin\"}]}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3},\"finish_reason\":\"stop\"}}\n\n"
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai_responses()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_use_responses_api(true)
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

    assert!(siumai_root.get("openai").is_some());
    assert!(provider_root.get("openai").is_some());
    assert!(config_root.get("openai").is_some());
    assert!(siumai_root.get("azure").is_none());
    assert!(provider_root.get("azure").is_none());
    assert!(config_root.get("azure").is_none());

    let siumai_meta = siumai_resp
        .openai_metadata()
        .expect("siumai openai metadata");
    let provider_meta = provider_resp
        .openai_metadata()
        .expect("provider openai metadata");
    let config_meta = config_resp
        .openai_metadata()
        .expect("config openai metadata");

    assert_eq!(siumai_resp.content_text(), Some("See attached files."));
    assert_eq!(provider_resp.content_text(), Some("See attached files."));
    assert_eq!(config_resp.content_text(), Some("See attached files."));
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

    let siumai_sources = siumai_meta.sources.expect("siumai sources");
    let provider_sources = provider_meta.sources.expect("provider sources");
    let config_sources = config_meta.sources.expect("config sources");

    assert_eq!(siumai_sources.len(), 2);
    assert_eq!(provider_sources.len(), 2);
    assert_eq!(config_sources.len(), 2);

    let siumai_container = siumai_sources
        .iter()
        .find(|source| source.url == "file_container_1")
        .expect("siumai container source");
    let provider_container = provider_sources
        .iter()
        .find(|source| source.url == "file_container_1")
        .expect("provider container source");
    let config_container = config_sources
        .iter()
        .find(|source| source.url == "file_container_1")
        .expect("config container source");

    let siumai_container_meta = siumai_container
        .openai_metadata()
        .expect("siumai container metadata");
    let provider_container_meta = provider_container
        .openai_metadata()
        .expect("provider container metadata");
    let config_container_meta = config_container
        .openai_metadata()
        .expect("config container metadata");

    assert_eq!(
        siumai_container_meta.file_id.as_deref(),
        Some("file_container_1")
    );
    assert_eq!(
        provider_container_meta.file_id.as_deref(),
        Some("file_container_1")
    );
    assert_eq!(
        config_container_meta.file_id.as_deref(),
        Some("file_container_1")
    );
    assert_eq!(
        siumai_container_meta.container_id.as_deref(),
        Some("container_42")
    );
    assert_eq!(
        provider_container_meta.container_id.as_deref(),
        Some("container_42")
    );
    assert_eq!(
        config_container_meta.container_id.as_deref(),
        Some("container_42")
    );
    assert_eq!(siumai_container_meta.index, Some(3));
    assert_eq!(provider_container_meta.index, Some(3));
    assert_eq!(config_container_meta.index, Some(3));

    let siumai_file_path = siumai_sources
        .iter()
        .find(|source| source.url == "file_path_9")
        .expect("siumai file path source");
    let provider_file_path = provider_sources
        .iter()
        .find(|source| source.url == "file_path_9")
        .expect("provider file path source");
    let config_file_path = config_sources
        .iter()
        .find(|source| source.url == "file_path_9")
        .expect("config file path source");

    let siumai_file_path_meta = siumai_file_path
        .openai_metadata()
        .expect("siumai file path metadata");
    let provider_file_path_meta = provider_file_path
        .openai_metadata()
        .expect("provider file path metadata");
    let config_file_path_meta = config_file_path
        .openai_metadata()
        .expect("config file path metadata");

    assert_eq!(
        siumai_file_path_meta.file_id.as_deref(),
        Some("file_path_9")
    );
    assert_eq!(
        provider_file_path_meta.file_id.as_deref(),
        Some("file_path_9")
    );
    assert_eq!(
        config_file_path_meta.file_id.as_deref(),
        Some("file_path_9")
    );
    assert!(siumai_file_path_meta.container_id.is_none());
    assert!(provider_file_path_meta.container_id.is_none());
    assert!(config_file_path_meta.container_id.is_none());
    assert_eq!(siumai_file_path_meta.index, Some(5));
    assert_eq!(provider_file_path_meta.index, Some(5));
    assert_eq!(config_file_path_meta.index, Some(5));

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
    assert_eq!(siumai_req.url, "https://example.com/v1/responses");
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn openai_default_options_match_public_stream_request_shape() {
    let model = "o3-mini";
    let base_url = "https://example.com/v1";
    let stream_body = openai_reasoning_stream_body(model);

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_openai_options(OpenAiOptions::new().with_reasoning_effort(ReasoningEffort::Low))
        .with_openai_options(
            OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new()
                    .with_previous_response("resp_default".to_string())
                    .with_reasoning_summary("detailed"),
            ),
        )
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_openai_options(OpenAiOptions::new().with_reasoning_effort(ReasoningEffort::Low))
        .with_openai_options(
            OpenAiOptions::new().with_responses_api(
                ResponsesApiConfig::new()
                    .with_previous_response("resp_default".to_string())
                    .with_reasoning_summary("detailed"),
            ),
        )
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_openai_options(OpenAiOptions::new().with_reasoning_effort(ReasoningEffort::Low))
            .with_openai_options(
                OpenAiOptions::new().with_responses_api(
                    ResponsesApiConfig::new()
                        .with_previous_response("resp_default".to_string())
                        .with_reasoning_summary("detailed"),
                ),
            )
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    use futures_util::StreamExt;

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

    while siumai_stream.next().await.is_some() {}
    while provider_stream.next().await.is_some() {}
    while config_stream.next().await.is_some() {}

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
    assert_openai_default_options_stream_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn openai_responses_reasoning_response_is_equivalent_across_public_paths() {
    let model = "o3-mini";
    let response_json = openai_reasoning_response_json(model);

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai_responses()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_use_responses_api(true)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(
        Arc::new(registry_transport.clone()),
        "https://example.com/v1",
    );
    let registry_model = registry
        .language_model("openai:o3-mini")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_openai_options(
            OpenAiOptions::new()
                .with_reasoning_effort(ReasoningEffort::Low)
                .with_responses_api(ResponsesApiConfig::new().with_reasoning_summary("detailed")),
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
        assert_openai_reasoning_response(response);
    }

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "https://example.com/v1/responses");
    assert_eq!(
        siumai_req.body["reasoning"]["effort"],
        serde_json::json!("low")
    );
    assert_eq!(
        siumai_req.body["reasoning"]["summary"],
        serde_json::json!("detailed")
    );
}

#[tokio::test]
async fn openai_responses_reasoning_stream_is_equivalent_across_public_paths() {
    let model = "o3-mini";
    let stream_body = openai_reasoning_stream_body(model);

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .openai()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::openai_responses()
        .api_key("test-key")
        .base_url("https://example.com/v1")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::openai::OpenAiClient::from_config(
        siumai::provider_ext::openai::OpenAiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model(model)
            .with_use_responses_api(true)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(
        Arc::new(registry_transport.clone()),
        "https://example.com/v1",
    );
    let registry_model = registry
        .language_model("openai:o3-mini")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_openai_options(
            OpenAiOptions::new()
                .with_reasoning_effort(ReasoningEffort::Low)
                .with_responses_api(ResponsesApiConfig::new().with_reasoning_summary("detailed")),
        );

    use futures_util::StreamExt;

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

    let (siumai_reasoning, siumai_end) = collect_stream_summary(&mut siumai_stream).await;
    let (provider_reasoning, provider_end) = collect_stream_summary(&mut provider_stream).await;
    let (config_reasoning, config_end) = collect_stream_summary(&mut config_stream).await;
    let (registry_reasoning, registry_end) = collect_stream_summary(&mut registry_stream).await;

    assert_eq!(siumai_reasoning, "Let me think.");
    assert_eq!(provider_reasoning, "Let me think.");
    assert_eq!(config_reasoning, "Let me think.");
    assert_eq!(registry_reasoning, "Let me think.");

    let siumai_resp = siumai_end.expect("siumai stream end");
    let provider_resp = provider_end.expect("provider stream end");
    let config_resp = config_end.expect("config stream end");
    let registry_resp = registry_end.expect("registry stream end");

    for response in [&siumai_resp, &provider_resp, &config_resp, &registry_resp] {
        assert_openai_reasoning_response(response);
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
    assert_eq!(siumai_req.url, "https://example.com/v1/responses");
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        siumai_req.body["reasoning"]["effort"],
        serde_json::json!("low")
    );
    assert_eq!(
        siumai_req.body["reasoning"]["summary"],
        serde_json::json!("detailed")
    );
}
