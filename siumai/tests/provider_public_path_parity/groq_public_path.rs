use super::*;
use futures_util::StreamExt;
use siumai::experimental::client::LlmClient;
use siumai::extensions::{SpeechExtras, TranscriptionExtras};
use siumai::prelude::unified::{
    EmbeddingExtensions, EmbeddingRequest, ResponseFormat, Tool, ToolChoice,
};
use siumai::provider_ext::groq::{
    GroqChatRequestExt, GroqChatResponseExt, GroqOptions, GroqReasoningEffort, GroqReasoningFormat,
    GroqServiceTier,
};
use siumai_registry::registry::builder::RegistryBuilder;

fn groq_registry_providers() -> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
    built_in_registry_providers("groq", "groq")
}

fn groq_registry_builder() -> RegistryBuilder {
    RegistryBuilder::new(groq_registry_providers())
}

fn make_registry(
    transport: Arc<dyn HttpTransport>,
    base_url: &str,
) -> siumai::registry::ProviderRegistryHandle {
    groq_registry_builder()
        .with_provider_api_key_base_url_fetch("groq", "test-key", base_url, transport)
        .build()
        .expect("build registry")
}

#[test]
fn groq_package_settings_preserve_supported_provider_inputs() {
    let config = siumai::provider_ext::groq::GroqProviderSettings::new()
        .with_api_key("test-key")
        .with_base_url("https://example.com/groq")
        .with_header("x-test", "1")
        .into_config_for_model("openai/gpt-oss-20b")
        .expect("settings into config");

    assert_eq!(config.base_url, "https://example.com/groq");
    assert_eq!(config.common_params.model, "openai/gpt-oss-20b");
    assert_eq!(
        config.http_config.headers.get("x-test").map(String::as_str),
        Some("1")
    );
}
fn make_groq_tool_call_request(model: &str) -> ChatRequest {
    ChatRequest::builder()
        .model(model)
        .messages(vec![
            ChatMessage::user("What's the weather in Tokyo?").build(),
        ])
        .tools(vec![Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
                "additionalProperties": false
            }),
        )])
        .build()
}

fn assert_groq_weather_tool_call_response(response: &siumai::prelude::unified::ChatResponse) {
    assert_eq!(
        response.finish_reason,
        Some(siumai::prelude::unified::FinishReason::ToolCalls)
    );
    assert_eq!(response.tool_calls().len(), 1);

    let call = response.tool_calls()[0].as_tool_call().expect("tool call");
    assert_eq!(call.tool_call_id, "call_1");
    assert_eq!(call.tool_name, "get_weather");
    assert_eq!(call.arguments, &serde_json::json!({ "city": "Tokyo" }));
}

fn assert_groq_default_options_request(req: &HttpTransportRequest, model: &str) {
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(req.body["logprobs"], serde_json::json!(true));
    assert_eq!(req.body["top_logprobs"], serde_json::json!(2));
    assert_eq!(req.body["service_tier"], serde_json::json!("flex"));
    assert_eq!(req.body["reasoning_effort"], serde_json::json!("default"));
    assert_eq!(req.body["reasoning_format"], serde_json::json!("parsed"));
}

fn assert_groq_default_options_stream_request(req: &HttpTransportRequest, model: &str) {
    assert_groq_default_options_request(req, model);
    assert_eq!(req.body["stream"], serde_json::json!(true));
    assert_eq!(
        header_value(req, "accept"),
        Some("text/event-stream".to_string())
    );
}

async fn collect_groq_streamed_tool_call(
    stream: &mut siumai::prelude::unified::ChatStream,
) -> (
    siumai::prelude::unified::ChatResponse,
    String,
    String,
    serde_json::Value,
) {
    let mut tool_call_id = None;
    let mut tool_name = None;
    let mut arguments = String::new();

    while let Some(event) = stream.next().await {
        match event {
            Ok(event) => {
                record_streamed_tool_part(
                    &event,
                    &mut tool_call_id,
                    &mut tool_name,
                    &mut arguments,
                );
                if let siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } = event {
                    return (
                        response,
                        tool_call_id.expect("stream tool call id"),
                        tool_name.expect("stream tool name"),
                        serde_json::from_str(&arguments).expect("stream tool arguments json"),
                    );
                }
            }
            Err(err) => panic!("stream error: {err}"),
        }
    }

    panic!("stream end event");
}

#[tokio::test]
async fn groq_siumai_provider_config_audio_extras_are_intentionally_unavailable() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let transcription_model = "whisper-large-v3";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(transcription_model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(transcription_model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(transcription_model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_speech_model = registry
        .speech_model("groq:playai-tts")
        .expect("build registry speech model");
    let registry_model = registry
        .transcription_model("groq:whisper-large-v3")
        .expect("build registry transcription model");

    assert!(siumai_client.as_speech_capability().is_some());
    assert!(provider_client.as_speech_capability().is_some());
    assert!(config_client.as_speech_capability().is_some());
    assert!(siumai_client.as_speech_extras().is_none());
    assert!(provider_client.as_speech_extras().is_none());
    assert!(config_client.as_speech_extras().is_none());

    assert!(siumai_client.as_transcription_capability().is_some());
    assert!(provider_client.as_transcription_capability().is_some());
    assert!(config_client.as_transcription_capability().is_some());
    assert!(siumai_client.as_transcription_extras().is_none());
    assert!(provider_client.as_transcription_extras().is_none());
    assert!(config_client.as_transcription_extras().is_none());

    let registry_speech_err = match registry_speech_model
        .tts_stream(
            TtsRequest::new("hello from groq".to_string())
                .with_voice("Fritz-PlayAI".to_string())
                .with_format("wav".to_string()),
        )
        .await
    {
        Ok(_) => panic!("groq registry speech extras should be unavailable"),
        Err(err) => err,
    };
    let registry_stt_stream_err = match registry_model
        .stt_stream(SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg"))
        .await
    {
        Ok(_) => panic!("groq registry transcription stream extras should be unavailable"),
        Err(err) => err,
    };
    let registry_translate_err = registry_model
        .audio_translate(
            siumai_core::types::AudioTranslationRequest::from_audio(b"abc".to_vec(), "audio/mpeg")
                .with_media_type("audio/mpeg".to_string()),
        )
        .await
        .expect_err("groq registry translation should be unsupported");

    assert_unsupported_operation(&registry_speech_err);
    assert_unsupported_operation(&registry_stt_stream_err);
    assert_unsupported_operation(&registry_translate_err);
    assert_capture_transports_unused(&[
        &siumai_transport,
        &provider_transport,
        &config_transport,
        &registry_transport,
    ]);
}

#[tokio::test]
async fn groq_siumai_provider_config_chat_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("llama-3.1-70b-versatile")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("llama-3.1-70b-versatile")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url("https://example.com/custom")
            .with_model("llama-3.1-70b-versatile")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_chat_request_with_model("llama-3.1-70b-versatile")
        .with_provider_option("groq", serde_json::json!({ "foo": "bar" }));

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.body["foo"], serde_json::json!("bar"));
}

#[tokio::test]
async fn groq_siumai_provider_config_stable_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "llama-3.1-70b-versatile";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_chat_request_with_model(model).with_groq_options(
        GroqOptions::new()
            .with_logprobs(true)
            .with_top_logprobs(2)
            .with_service_tier(GroqServiceTier::Flex)
            .with_reasoning_effort(GroqReasoningEffort::Default)
            .with_reasoning_format(GroqReasoningFormat::Parsed),
    );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.body["logprobs"], serde_json::json!(true));
    assert_eq!(siumai_req.body["top_logprobs"], serde_json::json!(2));
    assert_eq!(siumai_req.body["service_tier"], serde_json::json!("flex"));
    assert_eq!(
        siumai_req.body["reasoning_effort"],
        serde_json::json!("default")
    );
    assert_eq!(
        siumai_req.body["reasoning_format"],
        serde_json::json!("parsed")
    );
}

#[tokio::test]
async fn groq_default_options_match_public_request_shape() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "llama-3.1-70b-versatile";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_groq_options(GroqOptions::new().with_logprobs(true).with_top_logprobs(2))
        .with_groq_options(
            GroqOptions::new()
                .with_service_tier(GroqServiceTier::Flex)
                .with_reasoning_effort(GroqReasoningEffort::Default)
                .with_reasoning_format(GroqReasoningFormat::Parsed),
        )
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_groq_options(GroqOptions::new().with_logprobs(true).with_top_logprobs(2))
        .with_groq_options(
            GroqOptions::new()
                .with_service_tier(GroqServiceTier::Flex)
                .with_reasoning_effort(GroqReasoningEffort::Default)
                .with_reasoning_format(GroqReasoningFormat::Parsed),
        )
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_groq_options(GroqOptions::new().with_logprobs(true).with_top_logprobs(2))
            .with_groq_options(
                GroqOptions::new()
                    .with_service_tier(GroqServiceTier::Flex)
                    .with_reasoning_effort(GroqReasoningEffort::Default)
                    .with_reasoning_format(GroqReasoningFormat::Parsed),
            )
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_chat_request_with_model(model);

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_groq_default_options_request(&siumai_req, model);
}

#[tokio::test]
async fn groq_registry_stable_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "llama-3.1-70b-versatile";
    let base_url = "https://example.com/custom";

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("groq:llama-3.1-70b-versatile")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_groq_options(
        GroqOptions::new()
            .with_logprobs(true)
            .with_top_logprobs(2)
            .with_service_tier(GroqServiceTier::Flex)
            .with_reasoning_effort(GroqReasoningEffort::Default)
            .with_reasoning_format(GroqReasoningFormat::Parsed),
    );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.body["logprobs"], serde_json::json!(true));
    assert_eq!(registry_req.body["top_logprobs"], serde_json::json!(2));
    assert_eq!(registry_req.body["service_tier"], serde_json::json!("flex"));
    assert_eq!(
        registry_req.body["reasoning_effort"],
        serde_json::json!("default")
    );
    assert_eq!(
        registry_req.body["reasoning_format"],
        serde_json::json!("parsed")
    );
}

#[tokio::test]
async fn groq_reasoning_response_is_equivalent_across_public_paths() {
    let model = "llama-3.1-70b-versatile";
    let base_url = "https://example.com/custom";
    let response_json = serde_json::json!({
        "id": "chatcmpl-groq-reasoning",
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
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("groq:llama-3.1-70b-versatile")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_groq_options(
        GroqOptions::new()
            .with_service_tier(GroqServiceTier::Flex)
            .with_reasoning_effort(GroqReasoningEffort::Default)
            .with_reasoning_format(GroqReasoningFormat::Parsed),
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
    assert_eq!(siumai_req.body["service_tier"], serde_json::json!("flex"));
    assert_eq!(
        siumai_req.body["reasoning_effort"],
        serde_json::json!("default")
    );
    assert_eq!(
        siumai_req.body["reasoning_format"],
        serde_json::json!("parsed")
    );
}

#[tokio::test]
async fn groq_reasoning_stream_is_equivalent_across_public_paths() {
    let model = "llama-3.1-70b-versatile";
    let base_url = "https://example.com/custom";
    let stream_body = br#"data: {"id":"1","model":"llama-3.1-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"thinking":"Count the letters carefully. ","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"llama-3.1-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"thinking":"The word strawberry contains three r characters.","content":"There are three letter r's in strawberry.","role":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":18,"completion_tokens":24,"total_tokens":42,"completion_tokens_details":{"reasoning_tokens":12}}}

data: [DONE]

"#
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("groq:llama-3.1-70b-versatile")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_groq_options(
        GroqOptions::new()
            .with_service_tier(GroqServiceTier::Flex)
            .with_reasoning_effort(GroqReasoningEffort::Default)
            .with_reasoning_format(GroqReasoningFormat::Parsed),
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
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(siumai_req.body["service_tier"], serde_json::json!("flex"));
    assert_eq!(
        siumai_req.body["reasoning_effort"],
        serde_json::json!("default")
    );
    assert_eq!(
        siumai_req.body["reasoning_format"],
        serde_json::json!("parsed")
    );
}

#[tokio::test]
async fn groq_siumai_provider_config_tool_choice_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "llama-3.3-70b-versatile";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("groq:llama-3.3-70b-versatile")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .messages(vec![ChatMessage::user("hi").build()])
        .model(model)
        .tools(vec![Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"],
                "additionalProperties": false
            }),
        )])
        .tool_choice(ToolChoice::None)
        .build()
        .with_provider_option(
            "groq",
            serde_json::json!({
                "tool_choice": "auto",
                "service_tier": "flex"
            }),
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
    assert_eq!(siumai_req.body["tool_choice"], serde_json::json!("none"));
    assert_eq!(siumai_req.body["service_tier"], serde_json::json!("flex"));
    assert_eq!(
        siumai_req.body["tools"][0]["function"]["name"],
        serde_json::json!("get_weather")
    );
}

#[tokio::test]
async fn groq_structured_output_stream_is_equivalent_across_public_paths() {
    let stream_body = br#"data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"content":"{\"answer\":\"he","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"content":"llo\"}","role":null},"finish_reason":"stop"}]}

data: [DONE]

"#
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "llama-3.3-70b-versatile";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("groq:llama-3.3-70b-versatile")
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
        .with_provider_option(
            "groq",
            serde_json::json!({
                "response_format": { "type": "json_object" }
            }),
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
async fn groq_structured_output_synthetic_unknown_stream_end_fails_consistently_across_public_paths()
 {
    let stream_body = "data: {\"id\":\"1\",\"model\":\"llama-3.3-70b-versatile\",\"created\":1718345013,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"{\\\"answer\\\":\" ,\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"
            .as_bytes()
            .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "llama-3.3-70b-versatile";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("groq:llama-3.3-70b-versatile")
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
        .with_provider_option(
            "groq",
            serde_json::json!({
                "response_format": { "type": "json_object" }
            }),
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
async fn groq_tool_calls_stream_and_non_stream_are_equivalent_across_public_paths() {
    let model = "llama-3.3-70b-versatile";
    let base_url = "https://example.com/custom";

    let non_stream_response = serde_json::json!({
        "id": "chatcmpl-groq-tool-call",
        "object": "chat.completion",
        "created": 1_741_392_000,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Tokyo\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });

    let stream_body = br#"data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}

data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\""}}]},"finish_reason":null}]}

data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Tokyo\"}"}}]},"finish_reason":null}]}

data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

"#
        .to_vec();

    let siumai_non_stream_transport = JsonSuccessTransport::new(non_stream_response.clone());
    let provider_non_stream_transport = JsonSuccessTransport::new(non_stream_response.clone());
    let config_non_stream_transport = JsonSuccessTransport::new(non_stream_response.clone());
    let registry_non_stream_transport = JsonSuccessTransport::new(non_stream_response);

    let siumai_stream_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_stream_transport = SseSuccessTransport::new(stream_body.clone());
    let config_stream_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_stream_transport = SseSuccessTransport::new(stream_body);

    let siumai_non_stream_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_non_stream_transport.clone()))
        .build()
        .await
        .expect("build siumai non-stream client");

    let provider_non_stream_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_non_stream_transport.clone()))
        .build()
        .await
        .expect("build provider non-stream client");

    let config_non_stream_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_non_stream_transport.clone())),
    )
    .await
    .expect("build config non-stream client");

    let registry_non_stream =
        make_registry(Arc::new(registry_non_stream_transport.clone()), base_url);
    let registry_non_stream_model = registry_non_stream
        .language_model("groq:llama-3.3-70b-versatile")
        .expect("build registry non-stream language model");

    let siumai_stream_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_stream_transport.clone()))
        .build()
        .await
        .expect("build siumai stream client");

    let provider_stream_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_stream_transport.clone()))
        .build()
        .await
        .expect("build provider stream client");

    let config_stream_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_stream_transport.clone())),
    )
    .await
    .expect("build config stream client");

    let registry_stream = make_registry(Arc::new(registry_stream_transport.clone()), base_url);
    let registry_stream_model = registry_stream
        .language_model("groq:llama-3.3-70b-versatile")
        .expect("build registry stream language model");

    let request = make_groq_tool_call_request(model);

    let siumai_non_stream_resp = siumai_non_stream_client
        .chat_request(request.clone())
        .await
        .expect("siumai non-stream response");
    let provider_non_stream_resp = provider_non_stream_client
        .chat_request(request.clone())
        .await
        .expect("provider non-stream response");
    let config_non_stream_resp = config_non_stream_client
        .chat_request(request.clone())
        .await
        .expect("config non-stream response");
    let registry_non_stream_resp = registry_non_stream_model
        .chat_request(request.clone())
        .await
        .expect("registry non-stream response");

    let mut siumai_stream = siumai_stream_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream");
    let mut provider_stream = provider_stream_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream");
    let mut config_stream = config_stream_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream");
    let mut registry_stream_handle = registry_stream_model
        .chat_stream_request(request)
        .await
        .expect("registry stream");

    let (
        siumai_stream_resp,
        siumai_stream_tool_call_id,
        siumai_stream_tool_name,
        siumai_stream_arguments,
    ) = collect_groq_streamed_tool_call(&mut siumai_stream).await;
    let (
        provider_stream_resp,
        provider_stream_tool_call_id,
        provider_stream_tool_name,
        provider_stream_arguments,
    ) = collect_groq_streamed_tool_call(&mut provider_stream).await;
    let (
        config_stream_resp,
        config_stream_tool_call_id,
        config_stream_tool_name,
        config_stream_arguments,
    ) = collect_groq_streamed_tool_call(&mut config_stream).await;
    let (
        registry_stream_resp,
        registry_stream_tool_call_id,
        registry_stream_tool_name,
        registry_stream_arguments,
    ) = collect_groq_streamed_tool_call(&mut registry_stream_handle).await;

    for response in [
        &siumai_non_stream_resp,
        &provider_non_stream_resp,
        &config_non_stream_resp,
        &registry_non_stream_resp,
    ] {
        assert_groq_weather_tool_call_response(response);
    }

    for response in [
        &siumai_stream_resp,
        &provider_stream_resp,
        &config_stream_resp,
        &registry_stream_resp,
    ] {
        assert_eq!(
            response.finish_reason,
            Some(siumai::prelude::unified::FinishReason::ToolCalls)
        );
        assert!(response.tool_calls().is_empty());
    }

    let expected_call = siumai_non_stream_resp.tool_calls()[0]
        .as_tool_call()
        .expect("baseline tool call");

    for response in [
        &provider_non_stream_resp,
        &config_non_stream_resp,
        &registry_non_stream_resp,
    ] {
        let call = response.tool_calls()[0]
            .as_tool_call()
            .expect("matching tool call");
        assert_eq!(call.tool_call_id, expected_call.tool_call_id);
        assert_eq!(call.tool_name, expected_call.tool_name);
        assert_eq!(call.arguments, expected_call.arguments);
    }

    for (tool_call_id, tool_name, arguments) in [
        (
            &siumai_stream_tool_call_id,
            &siumai_stream_tool_name,
            &siumai_stream_arguments,
        ),
        (
            &provider_stream_tool_call_id,
            &provider_stream_tool_name,
            &provider_stream_arguments,
        ),
        (
            &config_stream_tool_call_id,
            &config_stream_tool_name,
            &config_stream_arguments,
        ),
        (
            &registry_stream_tool_call_id,
            &registry_stream_tool_name,
            &registry_stream_arguments,
        ),
    ] {
        assert_eq!(tool_call_id, expected_call.tool_call_id);
        assert_eq!(tool_name, expected_call.tool_name);
        assert_eq!(arguments, expected_call.arguments);
    }

    let siumai_non_stream_req = siumai_non_stream_transport
        .take()
        .expect("siumai non-stream request");
    let provider_non_stream_req = provider_non_stream_transport
        .take()
        .expect("provider non-stream request");
    let config_non_stream_req = config_non_stream_transport
        .take()
        .expect("config non-stream request");
    let registry_non_stream_req = registry_non_stream_transport
        .take()
        .expect("registry non-stream request");

    assert_requests_equivalent(&siumai_non_stream_req, &provider_non_stream_req);
    assert_requests_equivalent(&siumai_non_stream_req, &config_non_stream_req);
    assert_requests_equivalent(&siumai_non_stream_req, &registry_non_stream_req);
    assert_eq!(
        siumai_non_stream_req.body["tools"][0]["function"]["name"],
        serde_json::json!("get_weather")
    );

    let siumai_stream_req = siumai_stream_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_stream_req = provider_stream_transport
        .take_stream()
        .expect("provider stream request");
    let config_stream_req = config_stream_transport
        .take_stream()
        .expect("config stream request");
    let registry_stream_req = registry_stream_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&siumai_stream_req, &provider_stream_req);
    assert_requests_equivalent(&siumai_stream_req, &config_stream_req);
    assert_requests_equivalent(&siumai_stream_req, &registry_stream_req);
    assert_eq!(
        header_value(&siumai_stream_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(siumai_stream_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        siumai_stream_req.body["tools"][0]["function"]["name"],
        serde_json::json!("get_weather")
    );
}

#[tokio::test]
async fn groq_siumai_provider_config_chat_stream_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("llama-3.1-70b-versatile")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("llama-3.1-70b-versatile")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url("https://example.com/custom")
            .with_model("llama-3.1-70b-versatile")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_chat_request_with_model("llama-3.1-70b-versatile")
        .with_provider_option("groq", serde_json::json!({ "foo": "bar" }));

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
    let _ = siumai_stream.next().await;
    let _ = provider_stream.next().await;
    let _ = config_stream.next().await;

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
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn groq_siumai_provider_config_stable_stream_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "llama-3.1-70b-versatile";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_chat_request_with_model(model).with_groq_options(
        GroqOptions::new()
            .with_logprobs(true)
            .with_top_logprobs(2)
            .with_service_tier(GroqServiceTier::Flex)
            .with_reasoning_effort(GroqReasoningEffort::Default)
            .with_reasoning_format(GroqReasoningFormat::Parsed),
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
        .chat_stream_request(request)
        .await
        .expect("config stream ok");

    use futures_util::StreamExt;
    let _ = siumai_stream.next().await;
    let _ = provider_stream.next().await;
    let _ = config_stream.next().await;

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
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(siumai_req.body["logprobs"], serde_json::json!(true));
    assert_eq!(siumai_req.body["top_logprobs"], serde_json::json!(2));
    assert_eq!(siumai_req.body["service_tier"], serde_json::json!("flex"));
    assert_eq!(
        siumai_req.body["reasoning_effort"],
        serde_json::json!("default")
    );
    assert_eq!(
        siumai_req.body["reasoning_format"],
        serde_json::json!("parsed")
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn groq_default_options_match_public_stream_request_shape() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "llama-3.1-70b-versatile";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_groq_options(GroqOptions::new().with_logprobs(true).with_top_logprobs(2))
        .with_groq_options(
            GroqOptions::new()
                .with_service_tier(GroqServiceTier::Flex)
                .with_reasoning_effort(GroqReasoningEffort::Default)
                .with_reasoning_format(GroqReasoningFormat::Parsed),
        )
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_groq_options(GroqOptions::new().with_logprobs(true).with_top_logprobs(2))
        .with_groq_options(
            GroqOptions::new()
                .with_service_tier(GroqServiceTier::Flex)
                .with_reasoning_effort(GroqReasoningEffort::Default)
                .with_reasoning_format(GroqReasoningFormat::Parsed),
        )
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_groq_options(GroqOptions::new().with_logprobs(true).with_top_logprobs(2))
            .with_groq_options(
                GroqOptions::new()
                    .with_service_tier(GroqServiceTier::Flex)
                    .with_reasoning_effort(GroqReasoningEffort::Default)
                    .with_reasoning_format(GroqReasoningFormat::Parsed),
            )
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

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
        .chat_stream_request(request)
        .await
        .expect("config stream ok");

    let _ = siumai_stream.next().await;
    let _ = provider_stream.next().await;
    let _ = config_stream.next().await;

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
    assert_groq_default_options_stream_request(&siumai_req, model);
}

#[tokio::test]
async fn groq_registry_stable_stream_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "llama-3.1-70b-versatile";
    let base_url = "https://example.com/custom";

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("groq:llama-3.1-70b-versatile")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_groq_options(
        GroqOptions::new()
            .with_logprobs(true)
            .with_top_logprobs(2)
            .with_service_tier(GroqServiceTier::Flex)
            .with_reasoning_effort(GroqReasoningEffort::Default)
            .with_reasoning_format(GroqReasoningFormat::Parsed),
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
    assert_eq!(registry_req.body["stream"], serde_json::json!(true));
    assert_eq!(registry_req.body["logprobs"], serde_json::json!(true));
    assert_eq!(registry_req.body["top_logprobs"], serde_json::json!(2));
    assert_eq!(registry_req.body["service_tier"], serde_json::json!("flex"));
    assert_eq!(
        registry_req.body["reasoning_effort"],
        serde_json::json!("default")
    );
    assert_eq!(
        registry_req.body["reasoning_format"],
        serde_json::json!("parsed")
    );
    assert_eq!(
        header_value(&registry_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn groq_registry_chat_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let groq_transport = CaptureTransport::default();

    let registry = groq_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/global")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "groq",
            "ctx-key",
            "https://example.com",
            Arc::new(groq_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .language_model("groq:llama-3.1-70b-versatile")
        .expect("build registry language model");

    let _ = handle
        .chat_request(
            make_chat_request_with_model("llama-3.1-70b-versatile").with_groq_options(
                GroqOptions::new()
                    .with_logprobs(true)
                    .with_top_logprobs(2)
                    .with_service_tier(GroqServiceTier::Flex)
                    .with_reasoning_effort(GroqReasoningEffort::Default)
                    .with_reasoning_format(GroqReasoningFormat::Parsed),
            ),
        )
        .await;

    let req = groq_transport.take().expect("captured groq request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/openai/v1/chat/completions");
    assert_eq!(
        req.body["model"],
        serde_json::json!("llama-3.1-70b-versatile")
    );
    assert_eq!(req.body["logprobs"], serde_json::json!(true));
    assert_eq!(req.body["top_logprobs"], serde_json::json!(2));
    assert_eq!(req.body["service_tier"], serde_json::json!("flex"));
    assert_eq!(req.body["reasoning_effort"], serde_json::json!("default"));
    assert_eq!(req.body["reasoning_format"], serde_json::json!("parsed"));
}

#[tokio::test]
async fn groq_registry_chat_stream_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let groq_transport = CaptureTransport::default();

    let registry = groq_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/global")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "groq",
            "ctx-key",
            "https://example.com",
            Arc::new(groq_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let handle = registry
        .language_model("groq:llama-3.1-70b-versatile")
        .expect("build registry language model");

    let _ = handle
        .chat_stream_request(
            make_chat_request_with_model("llama-3.1-70b-versatile").with_groq_options(
                GroqOptions::new()
                    .with_logprobs(true)
                    .with_top_logprobs(2)
                    .with_service_tier(GroqServiceTier::Flex)
                    .with_reasoning_effort(GroqReasoningEffort::Default)
                    .with_reasoning_format(GroqReasoningFormat::Parsed),
            ),
        )
        .await;

    let req = groq_transport
        .take_stream()
        .expect("captured groq stream request");
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
    assert_eq!(req.url, "https://example.com/openai/v1/chat/completions");
    assert_eq!(req.body["stream"], serde_json::json!(true));
    assert_eq!(req.body["logprobs"], serde_json::json!(true));
    assert_eq!(req.body["top_logprobs"], serde_json::json!(2));
    assert_eq!(req.body["service_tier"], serde_json::json!("flex"));
    assert_eq!(req.body["reasoning_effort"], serde_json::json!("default"));
    assert_eq!(req.body["reasoning_format"], serde_json::json!("parsed"));
}

#[tokio::test]
async fn groq_siumai_provider_config_chat_response_metadata_are_equivalent() {
    let response_json = serde_json::json!({
        "id": "chatcmpl-groq-test",
        "object": "chat.completion",
        "created": 1_741_392_000,
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello from groq"
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "hello",
                            "logprob": -0.2,
                            "bytes": [104, 101, 108, 108, 111],
                            "top_logprobs": []
                        }
                    ]
                }
            }
        ]
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json);

    let model = "llama-3.3-70b-versatile";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_chat_request_with_model(model);

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

    let siumai_meta = siumai_resp.groq_metadata().expect("siumai groq metadata");
    let provider_meta = provider_resp
        .groq_metadata()
        .expect("provider groq metadata");
    let config_meta = config_resp.groq_metadata().expect("config groq metadata");

    assert_eq!(siumai_resp.content_text(), Some("hello from groq"));
    assert_eq!(provider_resp.content_text(), Some("hello from groq"));
    assert_eq!(config_resp.content_text(), Some("hello from groq"));

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.2,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(siumai_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(provider_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(config_meta.logprobs, Some(expected_logprobs));

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
}

#[tokio::test]
async fn groq_siumai_provider_config_stream_end_metadata_are_equivalent() {
    let stream_body = br#"data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"content":" from groq","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.2,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}]}

data: [DONE]

"#
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body);

    let model = "llama-3.3-70b-versatile";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

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

    let siumai_meta = siumai_resp.groq_metadata().expect("siumai groq metadata");
    let provider_meta = provider_resp
        .groq_metadata()
        .expect("provider groq metadata");
    let config_meta = config_resp.groq_metadata().expect("config groq metadata");

    assert_eq!(siumai_resp.content_text(), Some("hello from groq"));
    assert_eq!(provider_resp.content_text(), Some("hello from groq"));
    assert_eq!(config_resp.content_text(), Some("hello from groq"));
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

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.2,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(siumai_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(provider_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(config_meta.logprobs, Some(expected_logprobs));

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
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn groq_registry_chat_request_with_explicit_request_model_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let default_model = "llama-3.1-70b-versatile";
    let request_model = "llama-3.3-70b-versatile";
    let base_url = "https://example.com/custom";

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(default_model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("groq:llama-3.1-70b-versatile")
        .expect("build registry language model");

    let request = make_chat_request_with_model(request_model)
        .with_provider_option("groq", serde_json::json!({ "foo": "bar" }));

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.body["model"], serde_json::json!(request_model));
    assert_eq!(registry_req.body["foo"], serde_json::json!("bar"));
}

#[tokio::test]
async fn groq_registry_chat_stream_request_with_explicit_request_model_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let default_model = "llama-3.1-70b-versatile";
    let request_model = "llama-3.3-70b-versatile";
    let base_url = "https://example.com/custom";

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(default_model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("groq:llama-3.1-70b-versatile")
        .expect("build registry language model");

    let request = make_chat_request_with_model(request_model)
        .with_provider_option("groq", serde_json::json!({ "foo": "bar" }));

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
    assert_eq!(registry_req.body["foo"], serde_json::json!("bar"));
}

#[tokio::test]
async fn groq_registry_chat_response_metadata_match_config_path() {
    let response_json = serde_json::json!({
        "id": "chatcmpl-groq-test",
        "object": "chat.completion",
        "created": 1_741_392_000,
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello from groq"
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "hello",
                            "logprob": -0.2,
                            "bytes": [104, 101, 108, 108, 111],
                            "top_logprobs": []
                        }
                    ]
                }
            }
        ]
    });

    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "llama-3.3-70b-versatile";
    let base_url = "https://example.com/custom";

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("groq:llama-3.3-70b-versatile")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model);

    let config_resp = config_client
        .chat_request(request.clone())
        .await
        .expect("config response ok");
    let registry_resp = registry_model
        .chat_request(request)
        .await
        .expect("registry response ok");

    let config_meta = config_resp.groq_metadata().expect("config groq metadata");
    let registry_meta = registry_resp
        .groq_metadata()
        .expect("registry groq metadata");

    assert_eq!(config_resp.content_text(), Some("hello from groq"));
    assert_eq!(registry_resp.content_text(), Some("hello from groq"));
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
            "logprob": -0.2,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(config_meta.logprobs, Some(expected_logprobs.clone()));
    assert_eq!(registry_meta.logprobs, Some(expected_logprobs));

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
}

#[tokio::test]
async fn groq_registry_stream_end_metadata_match_config_path() {
    let stream_body = br#"data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"content":" from groq","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.2,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}]}

data: [DONE]

"#
        .to_vec();

    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "llama-3.3-70b-versatile";
    let base_url = "https://example.com/custom";

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("groq:llama-3.3-70b-versatile")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model);

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

    let config_meta = config_resp.groq_metadata().expect("config groq metadata");
    let registry_meta = registry_resp
        .groq_metadata()
        .expect("registry groq metadata");

    assert_eq!(config_resp.content_text(), Some("hello from groq"));
    assert_eq!(registry_resp.content_text(), Some("hello from groq"));
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
            "logprob": -0.2,
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
    assert_eq!(
        header_value(&config_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn groq_registry_override_chat_response_metadata_preserves_vendor_namespace() {
    let response_json = serde_json::json!({
        "id": "chatcmpl-groq-test",
        "object": "chat.completion",
        "created": 1_741_392_000,
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello from groq"
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "hello",
                            "logprob": -0.2,
                            "bytes": [104, 101, 108, 108, 111],
                            "top_logprobs": []
                        }
                    ]
                }
            }
        ]
    });

    let global_transport = CaptureTransport::default();
    let groq_transport = JsonSuccessTransport::new(response_json);

    let registry = groq_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "groq",
            "ctx-key",
            "https://example.com",
            Arc::new(groq_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let response = registry
        .language_model("groq:llama-3.3-70b-versatile")
        .expect("build groq handle")
        .chat_request(make_chat_request_with_model("llama-3.3-70b-versatile"))
        .await
        .expect("registry response ok");

    let root = response
        .provider_metadata
        .as_ref()
        .expect("registry provider metadata");
    assert!(root.get("groq").is_some());

    let metadata = response.groq_metadata().expect("groq metadata");
    assert_eq!(response.content_text(), Some("hello from groq"));
    assert_eq!(
        response.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.2,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(metadata.logprobs, Some(expected_logprobs));

    let req = groq_transport.take().expect("captured groq request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/openai/v1/chat/completions");
}

#[tokio::test]
async fn groq_registry_override_stream_end_metadata_preserves_vendor_namespace() {
    let stream_body = br#"data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"content":" from groq","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.2,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}]}

data: [DONE]

"#
        .to_vec();

    let global_transport = CaptureTransport::default();
    let groq_transport = SseSuccessTransport::new(stream_body);

    let registry = groq_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/global/v1")
        .fetch(Arc::new(global_transport.clone()))
        .with_provider_api_key_base_url_fetch(
            "groq",
            "ctx-key",
            "https://example.com",
            Arc::new(groq_transport.clone()),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry");

    let mut stream = registry
        .language_model("groq:llama-3.3-70b-versatile")
        .expect("build groq handle")
        .chat_stream_request(make_chat_request_with_model("llama-3.3-70b-versatile"))
        .await
        .expect("registry stream ok");

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
    assert!(root.get("groq").is_some());

    let metadata = response.groq_metadata().expect("groq metadata");
    assert_eq!(response.content_text(), Some("hello from groq"));
    assert_eq!(
        response.finish_reason,
        Some(siumai::prelude::unified::FinishReason::Stop)
    );

    let expected_logprobs = serde_json::json!([
        {
            "token": "hello",
            "logprob": -0.2,
            "bytes": [104, 101, 108, 108, 111],
            "top_logprobs": []
        }
    ]);
    assert_eq!(metadata.logprobs, Some(expected_logprobs));

    let req = groq_transport
        .take_stream()
        .expect("captured groq stream request");
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
    assert_eq!(req.url, "https://example.com/openai/v1/chat/completions");
}

#[tokio::test]
async fn groq_siumai_provider_config_tts_request_are_equivalent() {
    let siumai_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/wav");
    let provider_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/wav");
    let config_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/wav");
    let registry_transport = BinaryCaptureTransport::new(vec![1, 2, 3, 4], "audio/wav");

    let model = "playai-tts";
    let base_url = "https://example.com";

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");
    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .speech_model("groq:playai-tts")
        .expect("build registry speech model");

    let request = TtsRequest::new("hello from groq".to_string())
        .with_voice("Fritz-PlayAI".to_string())
        .with_format("wav".to_string())
        .with_speed(1.25);

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
        TtsRequest::new("hello from groq".to_string())
            .with_voice("Fritz-PlayAI".to_string())
            .with_format("wav".to_string())
            .with_speed(1.25),
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
    assert_eq!(siumai_req.url, "https://example.com/openai/v1/audio/speech");
    assert_eq!(siumai_req.body["model"], serde_json::json!(model));
    assert_eq!(
        siumai_req.body["input"],
        serde_json::json!("hello from groq")
    );
    assert_eq!(siumai_req.body["voice"], serde_json::json!("Fritz-PlayAI"));
    assert_eq!(siumai_req.body["response_format"], serde_json::json!("wav"));
    assert_eq!(siumai_req.body["speed"], serde_json::json!(1.25));
}

#[tokio::test]
async fn groq_siumai_provider_config_stt_request_are_equivalent() {
    let siumai_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from groq"
    }));
    let provider_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from groq"
    }));
    let config_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from groq"
    }));
    let registry_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "hello from groq"
    }));

    let model = "whisper-large-v3";
    let base_url = "https://example.com";

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");
    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .transcription_model("groq:whisper-large-v3")
        .expect("build registry transcription model");

    let mut request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
    request = request.with_media_type("audio/wav".to_string());

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
    let registry_request = SttRequest::from_audio(b"abc".to_vec(), "audio/wav");

    let registry_resp = registry_model
        .speech_to_text(registry_request)
        .await
        .expect("registry stt ok");

    assert_eq!(siumai_resp.text, "hello from groq");
    assert_eq!(provider_resp.text, "hello from groq");
    assert_eq!(config_resp.text, "hello from groq");
    assert_eq!(registry_resp.text, "hello from groq");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_multipart_requests_equivalent(&siumai_req, &provider_req);
    assert_multipart_requests_equivalent(&siumai_req, &config_req);
    assert_multipart_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.com/openai/v1/audio/transcriptions"
    );

    let body_text = normalize_multipart_body(&siumai_req);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("whisper-large-v3"));
    assert!(body_text.contains("name=\"file\"; filename=\"audio.wav\""));
    assert!(body_text.contains("Content-Type: audio/wav"));
    assert!(body_text.contains("abc"));
    assert!(!body_text.contains("name=\"response_format\""));
}

#[tokio::test]
async fn groq_siumai_provider_embedding_request_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("groq-embedding-test")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("groq-embedding-test")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url("https://example.com/custom")
            .with_model("groq-embedding-test")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request =
        EmbeddingRequest::single("hello groq embedding").with_model("groq-embedding-test");

    let siumai_err = siumai_client
        .embed_with_config(request.clone())
        .await
        .expect_err("groq embedding should be unsupported");

    assert_unsupported_operation(&siumai_err);
    assert_no_deferred_capability_leaks(&provider_client);
    assert_no_deferred_capability_leaks(&config_client);
    assert_capture_transports_unused(&[&siumai_transport, &provider_transport, &config_transport]);
}

#[tokio::test]
async fn groq_siumai_provider_rerank_request_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "groq-rerank-test";

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url("https://example.com/custom")
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_rerank_request_with_model(model).with_top_n(1);

    let siumai_err = siumai_client
        .rerank(request)
        .await
        .expect_err("groq rerank should be unsupported");

    assert_unsupported_operation(&siumai_err);
    assert_no_deferred_capability_leaks(&provider_client);
    assert_no_deferred_capability_leaks(&config_client);
    assert_capture_transports_unused(&[&siumai_transport, &provider_transport, &config_transport]);
}

#[tokio::test]
async fn groq_siumai_provider_image_request_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "groq-image-test";

    let siumai_client = Siumai::builder()
        .groq()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::groq()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::groq::GroqClient::from_config(
        siumai::provider_ext::groq::GroqConfig::new("test-key")
            .with_base_url("https://example.com/custom")
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .await
    .expect("build config client");

    let request = make_image_request_with_model(model);

    let siumai_err = siumai_client
        .generate_images(request)
        .await
        .expect_err("groq image generation should be unsupported");

    assert_unsupported_operation(&siumai_err);
    assert_no_deferred_capability_leaks(&provider_client);
    assert_no_deferred_capability_leaks(&config_client);
    assert_capture_transports_unused(&[&siumai_transport, &provider_transport, &config_transport]);
}
