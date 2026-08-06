use futures_util::StreamExt;
use serde_json::{Value, json};
use siumai_core::{
    ApiModeId, CallOptions, ContentPart, ErrorKind, ExecutionOwner, LanguageModel, LanguageRequest,
    LanguageStreamEvent, MediaData, MediaPart, Message, MessageRole, Model, ModelFamily,
    ProviderOptions, StreamTerminal, StructuredOutputSpec, ToolCall, ToolOutcome, ToolResult,
    ToolSpec, UsageValue, WarningKind,
};
use siumai_provider_deepseek::{
    DeepSeekChatOptions, DeepSeekCredential, DeepSeekLanguageApi, DeepSeekProvider,
    DeepSeekReasoningEffort, DeepSeekResponsesOptions,
};
use siumai_transport::EndpointConfig;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> DeepSeekProvider {
    DeepSeekProvider::builder(DeepSeekCredential::api_key("test-key"))
        .with_endpoint(
            EndpointConfig::local_explicit(format!("{}/v1", server.uri())).expect("local endpoint"),
        )
        .build()
        .expect("provider")
}

fn request(text: &str) -> LanguageRequest {
    LanguageRequest::new(vec![Message::text(MessageRole::User, text)])
}

fn strict_tool() -> ToolSpec {
    ToolSpec::new(
        "lookup",
        Some("Look up a value".to_string()),
        json!({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
            "additionalProperties": false
        }),
    )
    .expect("tool")
}

#[test]
fn provider_exposes_open_chat_and_responses_model_handles() {
    let provider = DeepSeekProvider::builder(DeepSeekCredential::unauthenticated())
        .with_endpoint(
            EndpointConfig::local_explicit("http://127.0.0.1:9/v1").expect("local endpoint"),
        )
        .build()
        .expect("provider");

    let chat = provider
        .chat_completions("future-deepseek-chat")
        .expect("chat");
    let responses = provider
        .responses("future-deepseek-responses")
        .expect("responses");
    assert_eq!(chat.provider_id().as_str(), "deepseek");
    assert_eq!(chat.api(), DeepSeekLanguageApi::ChatCompletions);
    assert_eq!(responses.api(), DeepSeekLanguageApi::Responses);
    assert_eq!(chat.descriptor().platform(), Some("local"));
    assert_eq!(chat.descriptor().api_mode(), Some("chat-completions"));
    assert_eq!(responses.descriptor().api_mode(), Some("responses"));
    assert_eq!(provider.registration().provider_id().as_str(), "deepseek");
    assert_eq!(
        provider
            .responses_registration()
            .api_mode(ModelFamily::Language)
            .map(ApiModeId::as_str),
        Some("responses")
    );
}

#[tokio::test]
async fn chat_replays_all_reasoning_and_preserves_strict_json_cache_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chat-deepseek",
            "model": "deepseek-v4-flash",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "final thought"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_cache_hit_tokens": 4,
                "prompt_cache_miss_tokens": 6,
                "reasoning_tokens": 2
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let request = LanguageRequest {
        messages: vec![
            Message::text(MessageRole::User, "first"),
            Message::new(
                MessageRole::Assistant,
                [
                    ContentPart::Reasoning {
                        text: "reason one".to_string(),
                    },
                    ContentPart::ToolCall(ToolCall {
                        id: "call-1".to_string(),
                        name: "lookup".to_string(),
                        arguments: json!({"q": "one"}),
                        owner: ExecutionOwner::Local,
                    }),
                ],
            ),
            Message::new(
                MessageRole::Tool,
                [ContentPart::ToolResult(ToolResult {
                    call_id: "call-1".to_string(),
                    name: "lookup".to_string(),
                    outcome: ToolOutcome::Success {
                        value: json!({"answer": 1}),
                    },
                })],
            ),
            Message::text(MessageRole::User, "second"),
            Message::new(
                MessageRole::Assistant,
                [
                    ContentPart::Reasoning {
                        text: "reason two".to_string(),
                    },
                    ContentPart::ToolCall(ToolCall {
                        id: "call-2".to_string(),
                        name: "lookup".to_string(),
                        arguments: json!({"q": "two"}),
                        owner: ExecutionOwner::Local,
                    }),
                ],
            ),
            Message::new(
                MessageRole::Tool,
                [ContentPart::ToolResult(ToolResult {
                    call_id: "call-2".to_string(),
                    name: "lookup".to_string(),
                    outcome: ToolOutcome::Success {
                        value: json!({"answer": 2}),
                    },
                })],
            ),
            Message::text(MessageRole::User, "final"),
        ],
        generation: Default::default(),
        tools: vec![strict_tool()],
        tool_choice: None,
        structured_output: Some(StructuredOutputSpec {
            name: "answer".to_string(),
            description: None,
            schema: json!({
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": false
            }),
            strict: true,
        }),
    };
    let options = DeepSeekChatOptions::new()
        .with_thinking_enabled()
        .with_reasoning_effort(DeepSeekReasoningEffort::High)
        .with_strict_tools(true);
    let response = provider(&server)
        .chat_completions("deepseek-v4-flash")
        .expect("model")
        .generate(
            request,
            CallOptions::default()
                .with_provider_options(ProviderOptions::typed(&options).expect("options")),
        )
        .await
        .expect("response");

    assert!(
        response
            .content()
            .iter()
            .any(|part| matches!(part, ContentPart::Reasoning { text } if text == "final thought"))
    );
    assert_eq!(response.usage().cache_read_tokens, UsageValue::Known(4));
    assert_eq!(response.usage().reasoning_tokens, UsageValue::Known(2));
    assert_eq!(
        response.provider_metadata()["deepseek"]["prompt_cache_miss_tokens"],
        6
    );
    assert!(response.warnings().iter().any(|warning| matches!(
        warning.kind(),
        WarningKind::Provider { code } if code.as_str() == "structured_output_fallback"
    )));

    let requests = server.received_requests().await.expect("requests");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("request body");
    assert_eq!(body["thinking"], json!({"type": "enabled"}));
    assert_eq!(body["reasoning_effort"], "high");
    assert!(body.get("strict_tools").is_none());
    assert_eq!(body["tools"][0]["function"]["strict"], true);
    assert_eq!(body["response_format"], json!({"type": "json_object"}));
    assert!(
        body["messages"][0]["content"]
            .as_str()
            .is_some_and(|value| value.contains("Return JSON"))
    );
    assert_eq!(body["messages"][2]["reasoning_content"], "reason one");
    assert_eq!(body["messages"][5]["reasoning_content"], "reason two");
}

#[tokio::test]
async fn media_fails_before_wire() {
    let server = MockServer::start().await;
    let provider = provider(&server);
    let media_request = LanguageRequest::new(vec![Message::new(
        MessageRole::User,
        [ContentPart::Media(MediaPart {
            media_type: "image/png".to_string(),
            data: MediaData::Url("https://example.com/image.png".to_string()),
            name: None,
        })],
    )]);
    let error = provider
        .chat_completions("deepseek-v4-flash")
        .expect("model")
        .generate(media_request, CallOptions::default())
        .await
        .expect_err("media must be rejected");
    assert_eq!(error.kind(), ErrorKind::Unsupported);

    assert!(
        server
            .received_requests()
            .await
            .expect("requests")
            .is_empty()
    );
}

#[tokio::test]
async fn responses_maps_supported_controls_and_rejects_known_pro() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp-deepseek",
            "object": "response",
            "created_at": 1_785_811_200_i64,
            "model": "deepseek-v4-flash",
            "status": "completed",
            "output": [{
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "answer", "annotations": []}]
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

    let options = DeepSeekResponsesOptions::new()
        .with_reasoning_effort(DeepSeekReasoningEffort::Max)
        .with_top_logprobs(20)
        .with_user("tenant-1")
        .with_web_search()
        .with_apply_patch();
    let response = provider(&server)
        .responses("deepseek-v4-flash")
        .expect("model")
        .generate(
            request("continue"),
            CallOptions::default()
                .with_provider_options(ProviderOptions::typed(&options).expect("options")),
        )
        .await
        .expect("response");
    assert_eq!(response.id(), Some("resp-deepseek"));
    assert_eq!(response.usage().cache_read_tokens, UsageValue::Known(1));

    let requests = server.received_requests().await.expect("requests");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("request body");
    assert_eq!(body["reasoning"], json!({"effort": "max"}));
    assert_eq!(body["top_logprobs"], 20);
    assert_eq!(body["user"], "tenant-1");
    assert_eq!(body["tools"][0], json!({"type": "web_search"}));
    assert_eq!(
        body["tools"][1],
        json!({"type": "custom", "name": "apply_patch"})
    );
    assert!(body.get("reasoning_effort").is_none());
    assert!(body.get("native_tools").is_none());

    let error = provider(&server)
        .responses("deepseek-v4-pro")
        .expect("model")
        .generate(request("hello"), CallOptions::default())
        .await
        .expect_err("known Pro model must not use Responses");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(server.received_requests().await.expect("requests").len(), 1);
}

#[tokio::test]
async fn chat_stream_emits_one_terminal_and_preserves_known_zero_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    concat!(
                        "data: {\"id\":\"chat-stream\",\"model\":\"deepseek-v4-flash\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"why\",\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
                        "data: {\"id\":\"chat-stream\",\"model\":\"deepseek-v4-flash\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0,\"prompt_cache_hit_tokens\":0,\"prompt_cache_miss_tokens\":0,\"reasoning_tokens\":0}}\n\n",
                        "data: [DONE]\n\n"
                    ),
                    "text/event-stream",
                ),
        )
        .expect(1)
        .mount(&server)
        .await;

    let events = provider(&server)
        .chat_completions("deepseek-v4-flash")
        .expect("model")
        .stream(request("hello"), CallOptions::default())
        .await
        .expect("stream")
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
        LanguageStreamEvent::Usage(usage)
            if usage.cache_read_tokens == UsageValue::Known(0)
                && usage.reasoning_tokens == UsageValue::Known(0)
                && usage.provider["deepseek"]["prompt_cache_miss_tokens"] == 0
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::Terminal(StreamTerminal::Completed { response })
            if response.usage().input_tokens == UsageValue::Known(0)
                && response.usage().output_tokens == UsageValue::Known(0)
                && response.usage().cache_read_tokens == UsageValue::Known(0)
    )));

    let requests = server.received_requests().await.expect("requests");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("request body");
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"], json!({"include_usage": true}));
}
