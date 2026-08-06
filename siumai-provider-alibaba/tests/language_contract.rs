use futures_util::StreamExt;
use serde_json::{Value, json};
use siumai_core::{
    CallOptions, ContentPart, ErrorKind, LanguageModel, LanguageStreamEvent, Message, MessageRole,
    Model, ModelFamily, ProviderOptions, StreamTerminal, UsageValue, WarningKind,
};
use siumai_provider_alibaba::{
    ALIBABA_SESSION_CACHE_HEADER, AlibabaChatOptions, AlibabaCredential,
    AlibabaPromptCacheBreakpoint, AlibabaProvider, AlibabaReasoningEffort, AlibabaResponsesOptions,
    AlibabaResponsesTool, AlibabaSearchOptions,
};
use siumai_transport::EndpointConfig;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn request(text: &str) -> siumai_core::LanguageRequest {
    siumai_core::LanguageRequest::new(vec![Message::text(MessageRole::User, text)])
}

fn provider(server: &MockServer) -> AlibabaProvider {
    AlibabaProvider::builder(AlibabaCredential::api_key("test-key"))
        .with_language_endpoint(
            EndpointConfig::local_explicit(format!("{}/v1", server.uri())).unwrap(),
        )
        .build()
        .unwrap()
}

#[test]
fn one_public_provider_exposes_both_language_modes_for_open_model_ids() {
    let endpoint = EndpointConfig::local_explicit("http://127.0.0.1:9/v1").unwrap();
    let provider = AlibabaProvider::builder(AlibabaCredential::unauthenticated())
        .with_language_endpoint(endpoint)
        .build()
        .unwrap();

    let responses = provider.responses("future-qwen-responses").unwrap();
    let chat = provider.chat_completions("future-qwen-chat").unwrap();
    assert_eq!(responses.provider_id().as_str(), "alibaba");
    assert_eq!(chat.provider_id().as_str(), "alibaba");
    assert_eq!(responses.descriptor().platform(), Some("local"));
    assert_eq!(responses.descriptor().api_mode(), Some("responses"));
    assert_eq!(chat.descriptor().api_mode(), Some("chat-completions"));

    let responses_registration = provider.responses_registration().unwrap();
    let chat_registration = provider.chat_completions_registration().unwrap();
    assert_eq!(responses_registration.provider_id().as_str(), "alibaba");
    assert_eq!(chat_registration.provider_id().as_str(), "alibaba");
    assert_ne!(
        responses_registration.api_mode(ModelFamily::Language),
        chat_registration.api_mode(ModelFamily::Language)
    );
}

#[tokio::test]
async fn chat_options_use_alibaba_namespace_and_decode_reasoning_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chat-alibaba",
            "model": "qwen3-coder-plus",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "thought"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
                "cache_read_input_tokens": 1,
                "cache_creation_input_tokens": 2
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let options = AlibabaChatOptions::new()
        .with_enable_thinking(true)
        .with_thinking_budget(512)
        .with_enable_search(true)
        .with_search_options(AlibabaSearchOptions::agent())
        .with_prompt_cache_breakpoint(AlibabaPromptCacheBreakpoint::new(0, 0));
    let response = provider(&server)
        .chat_completions("qwen3-coder-plus")
        .unwrap()
        .generate(
            request("hello"),
            CallOptions::default().with_provider_options(ProviderOptions::typed(&options).unwrap()),
        )
        .await
        .unwrap();

    assert!(
        response
            .content()
            .iter()
            .any(|part| matches!(part, ContentPart::Reasoning { text } if text == "thought"))
    );
    assert!(matches!(
        response.warnings().first().map(siumai_core::Warning::kind),
        Some(&WarningKind::UnknownModel)
    ));
    assert_eq!(response.usage().cache_read_tokens, UsageValue::Known(1));
    assert_eq!(response.usage().cache_write_tokens, UsageValue::Known(2));

    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["model"], json!("qwen3-coder-plus"));
    assert_eq!(body["enable_thinking"], json!(true));
    assert_eq!(body["thinking_budget"], json!(512));
    assert_eq!(body["enable_search"], json!(true));
    assert_eq!(body["search_options"], json!({"search_strategy": "agent"}));
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
    assert!(
        body["messages"][0]["content"][0]
            .get("prompt_cache_breakpoint")
            .is_none()
    );
    assert!(body.get("prompt_cache_breakpoints").is_none());
    assert_eq!(body["stream"], json!(false));
}

#[tokio::test]
async fn responses_options_map_native_tools_reasoning_and_session_cache_to_wire() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header(ALIBABA_SESSION_CACHE_HEADER, "enable"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp-alibaba",
            "object": "response",
            "created_at": 1_785_811_200_i64,
            "model": "future-qwen-responses",
            "status": "completed",
            "output": [{
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "answer",
                    "annotations": []
                }]
            }],
            "usage": {
                "input_tokens": 1,
                "input_tokens_details": {"cached_tokens": 1},
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 1},
                "total_tokens": 3
            },
            "error": null,
            "incomplete_details": null,
            "reasoning": null
        })))
        .expect(1)
        .mount(&server)
        .await;

    let options = AlibabaResponsesOptions::new()
        .with_store(true)
        .with_reasoning_effort(AlibabaReasoningEffort::High)
        .with_session_cache(true)
        .with_native_tool(AlibabaResponsesTool::web_search());
    let response = provider(&server)
        .responses("future-qwen-responses")
        .unwrap()
        .generate(
            request("search"),
            CallOptions::default().with_provider_options(ProviderOptions::typed(&options).unwrap()),
        )
        .await
        .unwrap();

    assert_eq!(response.id(), Some("resp-alibaba"));
    assert_eq!(response.usage().cache_read_tokens, UsageValue::Known(1));
    assert_eq!(response.usage().reasoning_tokens, UsageValue::Known(1));

    let requests = server.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["model"], json!("future-qwen-responses"));
    assert_eq!(body["reasoning"], json!({"effort": "high"}));
    assert_eq!(body["store"], json!(true));
    assert_eq!(body["tools"], json!([{"type": "web_search"}]));
    assert!(body.get("reasoning_effort").is_none());
    assert!(body.get("session_cache").is_none());
    assert!(body.get("native_tools").is_none());
}

#[tokio::test]
async fn responses_stream_emits_one_terminal_and_preserves_zero_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    "data: {\"type\":\"response.completed\",\"sequence_number\":0,\"response\":{\"id\":\"resp-stream\",\"created_at\":1785811200,\"model\":\"future-qwen-stream\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens\":0,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":0},\"error\":null,\"incomplete_details\":null,\"reasoning\":null}}\n\n",
                    "text/event-stream",
                ),
        )
        .expect(1)
        .mount(&server)
        .await;

    let events = provider(&server)
        .responses("future-qwen-stream")
        .unwrap()
        .stream(request("hello"), CallOptions::default())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        events
            .iter()
            .filter(|event| event.terminal().is_some())
            .count(),
        1
    );
    assert!(events.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::Terminal(StreamTerminal::Completed { response })
            if response.usage().input_tokens == UsageValue::Known(0)
                && response.usage().output_tokens == UsageValue::Known(0)
    )));
}

#[tokio::test]
async fn invalid_native_tool_fails_before_wire_and_wrong_mode_options_are_rejected() {
    let server = MockServer::start().await;
    let provider = provider(&server);
    let invalid =
        AlibabaResponsesOptions::new().with_native_tool(AlibabaResponsesTool::WebExtractor);
    let error = provider
        .responses("future-qwen")
        .unwrap()
        .generate(
            request("extract"),
            CallOptions::default().with_provider_options(ProviderOptions::typed(&invalid).unwrap()),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);

    let wrong_mode = AlibabaResponsesOptions::new().with_store(true);
    let error = provider
        .chat_completions("future-qwen")
        .unwrap()
        .generate(
            request("hello"),
            CallOptions::default()
                .with_provider_options(ProviderOptions::typed(&wrong_mode).unwrap()),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);

    let mut too_many_breakpoints = AlibabaChatOptions::new();
    for content_index in 0..5 {
        too_many_breakpoints = too_many_breakpoints
            .with_prompt_cache_breakpoint(AlibabaPromptCacheBreakpoint::new(0, content_index));
    }
    assert!(ProviderOptions::typed(&too_many_breakpoints).is_err());
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn provider_errors_keep_remote_canaries_out_of_default_diagnostics() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(400)
                .insert_header("x-request-id", "alibaba-request-42")
                .insert_header("x-private-canary", "canary-header-secret")
                .set_body_string("canary-body-secret"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let error = provider(&server)
        .responses("future-qwen")
        .unwrap()
        .generate(request("hello"), CallOptions::default())
        .await
        .unwrap_err();
    let debug = format!("{error:?}");
    let display = error.to_string();
    assert!(!debug.contains("canary-header-secret"));
    assert!(!debug.contains("canary-body-secret"));
    assert!(!display.contains("canary-header-secret"));
    assert!(!display.contains("canary-body-secret"));
    assert_eq!(
        error.diagnostics().and_then(|value| value.status()),
        Some(400)
    );
    assert_eq!(
        error.diagnostics().and_then(|value| value.request_id()),
        Some("alibaba-request-42")
    );
}
