use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use serde_json::{Value, json};
use siumai_core::{
    ApiModeId, CallOptions, ContentPart, ErrorKind, LanguageModel, LanguageRequest,
    LanguageStreamEvent, MediaData, MediaPart, Message, MessageRole, Model, ModelFamily,
    ReplayDomain, ReplayDomainId, StreamTerminal, StructuredOutputSpec, ToolCall, ToolOutcome,
    ToolResult, ToolSpec, UsageValue, WarningKind,
};
use siumai_provider_deepseek::{
    DeepSeekAssistantPrefix, DeepSeekChatOptions, DeepSeekConfigError, DeepSeekCredential,
    DeepSeekLanguageApi, DeepSeekProvider, DeepSeekReasoningEffort, DeepSeekResponsesOptions,
};
use siumai_transport::{
    EndpointConfig, OfficialOrigin, ProviderHttpTransportSettings, TransportEvent,
    TransportObserver,
};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Default)]
struct RecordingObserver {
    events: Mutex<Vec<TransportEvent>>,
}

impl TransportObserver for RecordingObserver {
    fn observe(&self, event: &TransportEvent) {
        self.events.lock().unwrap().push(event.clone());
    }
}

fn test_replay_domain() -> ReplayDomain {
    ReplayDomain::custom(ReplayDomainId::new("test-endpoint").expect("replay domain"))
}

fn provider(server: &MockServer) -> DeepSeekProvider {
    DeepSeekProvider::builder(DeepSeekCredential::api_key("test-key"))
        .with_endpoint(
            EndpointConfig::local_explicit(format!("{}/v1", server.uri())).expect("local endpoint"),
        )
        .with_replay_domain(test_replay_domain())
        .build()
        .expect("provider")
}

fn messages_provider(server: &MockServer) -> DeepSeekProvider {
    DeepSeekProvider::builder(DeepSeekCredential::api_key("test-key"))
        .with_messages_endpoint(
            EndpointConfig::local_explicit(format!("{}/anthropic", server.uri()))
                .expect("local Messages endpoint"),
        )
        .with_messages_replay_domain(test_replay_domain())
        .build()
        .expect("provider")
}

fn beta_provider(server: &MockServer) -> DeepSeekProvider {
    DeepSeekProvider::builder(DeepSeekCredential::api_key("test-key"))
        .with_beta_endpoint(
            EndpointConfig::local_explicit(format!("{}/beta", server.uri()))
                .expect("local beta endpoint"),
        )
        .with_beta_replay_domain(test_replay_domain())
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
fn provider_exposes_open_chat_beta_responses_and_messages_model_handles() {
    let provider = DeepSeekProvider::builder(DeepSeekCredential::unauthenticated())
        .with_endpoint(
            EndpointConfig::local_explicit("http://127.0.0.1:9/v1").expect("local endpoint"),
        )
        .with_replay_domain(test_replay_domain())
        .build()
        .expect("provider");

    let chat = provider
        .chat_completions("future-deepseek-chat")
        .expect("chat");
    let responses = provider
        .responses("future-deepseek-responses")
        .expect("responses");
    let beta = provider
        .beta_chat_completions("future-deepseek-beta")
        .expect("beta Chat");
    let messages = provider
        .messages("future-deepseek-messages")
        .expect("Messages");
    assert_eq!(chat.provider_id().as_str(), "deepseek");
    assert_eq!(chat.api(), DeepSeekLanguageApi::ChatCompletions);
    assert_eq!(beta.api(), DeepSeekLanguageApi::BetaChatCompletions);
    assert_eq!(responses.api(), DeepSeekLanguageApi::Responses);
    assert_eq!(messages.api(), DeepSeekLanguageApi::Messages);
    assert_eq!(chat.descriptor().platform(), Some("local"));
    assert_eq!(chat.descriptor().api_mode(), Some("chat-completions"));
    assert_eq!(responses.descriptor().api_mode(), Some("responses"));
    assert_eq!(beta.descriptor().api_mode(), Some("chat-completions"));
    assert_eq!(messages.descriptor().api_mode(), Some("messages"));
    assert_eq!(provider.registration().provider_id().as_str(), "deepseek");
    assert_eq!(
        provider
            .responses_registration()
            .api_mode(ModelFamily::Language)
            .map(ApiModeId::as_str),
        Some("responses")
    );
    assert_eq!(
        provider
            .beta_chat_completions_registration()
            .api_mode(ModelFamily::Language)
            .map(ApiModeId::as_str),
        Some("chat-completions")
    );
    assert_eq!(
        provider
            .messages_registration()
            .expect("Messages registration")
            .api_mode(ModelFamily::Language)
            .map(ApiModeId::as_str),
        Some("messages")
    );
}

#[test]
fn custom_endpoints_require_a_matching_replay_domain() {
    let missing = DeepSeekProvider::builder(DeepSeekCredential::unauthenticated())
        .with_endpoint(
            EndpointConfig::local_explicit("http://127.0.0.1:9/v1").expect("local endpoint"),
        )
        .build()
        .expect_err("custom endpoint must require a replay domain");
    assert!(matches!(
        missing,
        DeepSeekConfigError::CustomEndpointRequiresReplayDomain
    ));

    let mismatched = DeepSeekProvider::builder(DeepSeekCredential::unauthenticated())
        .with_endpoint(
            EndpointConfig::local_explicit("http://127.0.0.1:9/v1").expect("local endpoint"),
        )
        .with_replay_domain(ReplayDomain::official(
            ReplayDomainId::new("test-endpoint").expect("replay domain"),
        ))
        .build()
        .expect_err("custom endpoint must reject an official replay audience");
    assert!(matches!(
        mismatched,
        DeepSeekConfigError::ReplayAudienceMismatch
    ));

    let missing_beta = DeepSeekProvider::builder(DeepSeekCredential::unauthenticated())
        .with_beta_endpoint(
            EndpointConfig::local_explicit("http://127.0.0.1:9/beta").expect("local beta endpoint"),
        )
        .build()
        .expect_err("custom beta endpoint must require a replay domain");
    assert!(matches!(
        missing_beta,
        DeepSeekConfigError::CustomBetaEndpointRequiresReplayDomain
    ));

    let missing_messages = DeepSeekProvider::builder(DeepSeekCredential::unauthenticated())
        .with_messages_endpoint(
            EndpointConfig::local_explicit("http://127.0.0.1:9/anthropic")
                .expect("local Messages endpoint"),
        )
        .build()
        .expect_err("custom Messages endpoint must require a replay domain");
    assert!(matches!(
        missing_messages,
        DeepSeekConfigError::CustomMessagesEndpointRequiresReplayDomain
    ));

    let mismatched_messages = DeepSeekProvider::builder(DeepSeekCredential::unauthenticated())
        .with_messages_endpoint(
            EndpointConfig::local_explicit("http://127.0.0.1:9/anthropic")
                .expect("local Messages endpoint"),
        )
        .with_messages_replay_domain(ReplayDomain::official(
            ReplayDomainId::new("test-messages-endpoint").expect("replay domain"),
        ))
        .build()
        .expect_err("custom Messages endpoint must reject an official replay audience");
    assert!(matches!(
        mismatched_messages,
        DeepSeekConfigError::MessagesReplayAudienceMismatch
    ));
}

#[test]
fn caller_supplied_official_policy_does_not_gain_official_identity() {
    let endpoint = EndpointConfig::official(
        "https://relay.example/v1",
        OfficialOrigin::new("https://relay.example").expect("origin"),
    )
    .expect("caller endpoint");

    let missing = DeepSeekProvider::builder(DeepSeekCredential::unauthenticated())
        .with_endpoint(endpoint.clone())
        .build()
        .expect_err("caller endpoint must require a custom replay domain");
    assert!(matches!(
        missing,
        DeepSeekConfigError::CustomEndpointRequiresReplayDomain
    ));

    let mismatched = DeepSeekProvider::builder(DeepSeekCredential::unauthenticated())
        .with_endpoint(endpoint.clone())
        .with_replay_domain(ReplayDomain::official(
            ReplayDomainId::new("deepseek-public-api").expect("replay domain"),
        ))
        .build()
        .expect_err("caller endpoint must reject an official replay audience");
    assert!(matches!(
        mismatched,
        DeepSeekConfigError::ReplayAudienceMismatch
    ));

    let provider = DeepSeekProvider::builder(DeepSeekCredential::unauthenticated())
        .with_endpoint(endpoint)
        .with_replay_domain(test_replay_domain())
        .build()
        .expect("caller endpoint with custom replay domain");
    assert!(provider.profile().generic_claims().is_some());
    assert!(provider.profile().verified_claims().is_none());
    assert!(
        !provider
            .language("future-model")
            .expect("model")
            .descriptor()
            .replay_domain()
            .expect("replay domain")
            .audience()
            .is_official()
    );
}

#[tokio::test]
async fn messages_direct_uses_x_api_key_and_preserves_open_model_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg-deepseek",
            "type": "message",
            "role": "assistant",
            "model": "private-deepseek-alias",
            "content": [{"type": "text", "text": "answer"}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 3, "output_tokens": 2}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let mut request = request("hello");
    request.generation.max_output_tokens = Some(128);
    let response = messages_provider(&server)
        .messages("private-deepseek-alias")
        .expect("model")
        .generate(request, CallOptions::default())
        .await
        .expect("response");

    assert!(
        response
            .content()
            .iter()
            .any(|part| matches!(part, ContentPart::Text { text } if text == "answer"))
    );
    assert_eq!(response.usage().input_tokens, UsageValue::Known(3));
    assert_eq!(response.usage().output_tokens, UsageValue::Known(2));

    let requests = server.received_requests().await.expect("requests");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("request body");
    assert_eq!(body["model"], json!("private-deepseek-alias"));
    assert_eq!(body["max_tokens"], json!(128));
}

#[tokio::test]
async fn branded_compatible_branch_installs_the_shared_attempt_observer() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-observed",
            "object": "chat.completion",
            "created": 1_787_000_000_i64,
            "model": "future-deepseek-chat",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let observer = Arc::new(RecordingObserver::default());
    let provider = DeepSeekProvider::builder(DeepSeekCredential::api_key("test-key"))
        .with_endpoint(EndpointConfig::local_explicit(format!("{}/v1", server.uri())).unwrap())
        .with_replay_domain(test_replay_domain())
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default().with_observer(observer.clone()),
        )
        .build()
        .unwrap();

    provider
        .chat_completions("future-deepseek-chat")
        .unwrap()
        .generate(request("hello"), CallOptions::default())
        .await
        .unwrap();

    let events = observer.events.lock().unwrap();
    assert!(matches!(
        events.as_slice(),
        [
            TransportEvent::AttemptBudgetResolved { .. },
            TransportEvent::AttemptStarted { .. },
            TransportEvent::ResponseHeadReceived { .. },
            TransportEvent::AttemptLoopFinished { .. }
        ]
    ));
    assert!(
        events
            .iter()
            .all(|event| event.call_id() == events[0].call_id())
    );
}

#[tokio::test]
async fn messages_stream_emits_one_terminal_with_usage() {
    let server = MockServer::start().await;
    let frames = [
        json!({
            "type": "message_start",
            "message": {
                "id": "msg-stream",
                "type": "message",
                "role": "assistant",
                "model": "deepseek-v4-flash",
                "usage": {"input_tokens": 2}
            }
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"}
        }),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 1}
        }),
        json!({"type": "message_stop"}),
    ];
    let sse = frames
        .into_iter()
        .map(|frame| format!("data: {frame}\n\n"))
        .collect::<String>();
    Mock::given(method("POST"))
        .and(path("/anthropic/v1/messages"))
        .and(header("accept", "text/event-stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .expect(1)
        .mount(&server)
        .await;

    let mut request = request("hello");
    request.generation.max_output_tokens = Some(64);
    let events = messages_provider(&server)
        .messages("deepseek-v4-flash")
        .expect("model")
        .stream(request, CallOptions::default())
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
        LanguageStreamEvent::Terminal(StreamTerminal::Completed { response })
            if response.usage().input_tokens == UsageValue::Known(2)
                && response.usage().output_tokens == UsageValue::Known(1)
    )));
}

#[tokio::test]
async fn messages_reject_media_before_transport_submission() {
    let server = MockServer::start().await;
    let media_request = LanguageRequest::new(vec![Message::new(
        MessageRole::User,
        [ContentPart::Media(MediaPart {
            media_type: "image/png".to_string(),
            data: MediaData::Url("https://example.com/image.png".to_string()),
            name: None,
        })],
    )]);
    let error = messages_provider(&server)
        .messages("deepseek-v4-flash")
        .expect("model")
        .generate(media_request, CallOptions::default())
        .await
        .expect_err("Messages media must be rejected");

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
async fn chat_replays_all_reasoning_and_preserves_json_cache_usage() {
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
                    ContentPart::ToolCall(
                        ToolCall::local("call-1", "lookup", json!({"q": "one"})).unwrap(),
                    ),
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
                    ContentPart::ToolCall(
                        ToolCall::local("call-2", "lookup", json!({"q": "two"})).unwrap(),
                    ),
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
        .with_reasoning_effort(DeepSeekReasoningEffort::High);
    let model = provider(&server)
        .chat_completions("deepseek-v4-flash")
        .expect("model");
    let call_options = CallOptions::default()
        .with_provider_options_for(&model, &options)
        .expect("call options");
    let response = model
        .generate(request, call_options)
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
    assert!(body["tools"][0]["function"].get("strict").is_none());
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
async fn stable_chat_rejects_beta_strict_tools_before_transport() {
    let server = MockServer::start().await;
    let strict_request = LanguageRequest {
        messages: vec![Message::text(MessageRole::User, "lookup")],
        generation: Default::default(),
        tools: vec![strict_tool()],
        tool_choice: None,
        structured_output: None,
    };
    let options = DeepSeekChatOptions::new().with_strict_tools(true);
    let model = provider(&server)
        .chat_completions("deepseek-v4-flash")
        .expect("model");
    let call_options = CallOptions::default()
        .with_provider_options_for(&model, &options)
        .expect("call options");
    let error = model
        .generate(strict_request, call_options)
        .await
        .expect_err("stable endpoint must reject beta-only strict tools");

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
async fn beta_chat_projects_strict_tools_and_final_assistant_prefix() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/beta/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chat-beta",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek-v4-flash",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let prefix = Message::assistant("{\"answer\":")
        .with_provider_annotation(&DeepSeekAssistantPrefix::new())
        .expect("prefix annotation");
    let tool = ToolSpec::new(
        "lookup",
        None,
        json!({
            "type": "object",
            "properties": {
                "filter": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                    "additionalProperties": false
                }
            },
            "required": ["filter"],
            "additionalProperties": false
        }),
    )
    .expect("tool");
    let request = LanguageRequest {
        messages: vec![Message::user("complete this prefix"), prefix],
        generation: Default::default(),
        tools: vec![tool],
        tool_choice: None,
        structured_output: None,
    };
    let options = DeepSeekChatOptions::new().with_strict_tools(true);
    let model = beta_provider(&server)
        .beta_chat_completions("deepseek-v4-flash")
        .expect("beta model");
    let call_options = CallOptions::default()
        .with_provider_options_for(&model, &options)
        .expect("call options");
    model
        .generate(request, call_options)
        .await
        .expect("beta response");

    let requests = server.received_requests().await.expect("requests");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("request body");
    assert!(body.get("strict_tools").is_none());
    assert_eq!(body["tools"][0]["function"]["strict"], true);
    assert_eq!(body["messages"][1]["role"], "assistant");
    assert_eq!(body["messages"][1]["prefix"], true);
}

#[tokio::test]
async fn stable_chat_rejects_beta_prefix_before_transport() {
    let server = MockServer::start().await;
    let prefix = Message::assistant("prefix")
        .with_provider_annotation(&DeepSeekAssistantPrefix::new())
        .expect("prefix annotation");
    let error = provider(&server)
        .chat_completions("deepseek-v4-flash")
        .expect("model")
        .generate(
            LanguageRequest::new(vec![Message::user("continue"), prefix]),
            CallOptions::default(),
        )
        .await
        .expect_err("stable endpoint must reject prefix completion");

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
async fn beta_strict_tools_validate_nested_object_schemas_before_transport() {
    let server = MockServer::start().await;
    let invalid_tool = ToolSpec::new(
        "lookup",
        None,
        json!({
            "type": "object",
            "properties": {
                "filter": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"]
                }
            },
            "required": ["filter"],
            "additionalProperties": false
        }),
    )
    .expect("tool");
    let request = LanguageRequest {
        messages: vec![Message::user("lookup")],
        generation: Default::default(),
        tools: vec![invalid_tool],
        tool_choice: None,
        structured_output: None,
    };
    let options = DeepSeekChatOptions::new().with_strict_tools(true);
    let model = beta_provider(&server)
        .beta_chat_completions("deepseek-v4-flash")
        .expect("beta model");
    let call_options = CallOptions::default()
        .with_provider_options_for(&model, &options)
        .expect("call options");
    let error = model
        .generate(request, call_options)
        .await
        .expect_err("invalid nested schema must fail before transport");

    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(
        server
            .received_requests()
            .await
            .expect("requests")
            .is_empty()
    );
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
async fn responses_maps_supported_controls_for_pro() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp-deepseek",
            "object": "response",
            "created_at": 1_785_811_200_i64,
            "model": "deepseek-v4-pro",
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
    let model = provider(&server)
        .responses("deepseek-v4-pro")
        .expect("model");
    let call_options = CallOptions::default()
        .with_provider_options_for(&model, &options)
        .expect("call options");
    let response = model
        .generate(request("continue"), call_options)
        .await
        .expect("response");
    assert_eq!(response.id(), Some("resp-deepseek"));
    assert_eq!(response.usage().cache_read_tokens, UsageValue::Known(1));

    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 1);
    let pro_body: Value = serde_json::from_slice(&requests[0].body).expect("Pro request body");
    assert_eq!(pro_body["model"], "deepseek-v4-pro");
    assert_eq!(pro_body["reasoning"], json!({"effort": "max"}));
    assert_eq!(pro_body["top_logprobs"], 20);
    assert_eq!(pro_body["user"], "tenant-1");
    assert_eq!(pro_body["tools"][0], json!({"type": "web_search"}));
    assert_eq!(
        pro_body["tools"][1],
        json!({"type": "custom", "name": "apply_patch"})
    );
    assert!(pro_body.get("reasoning_effort").is_none());
    assert!(pro_body.get("native_tools").is_none());
}

#[tokio::test]
async fn responses_preserve_future_model_ids_on_final_wire() {
    const FUTURE_MODEL: &str = "private-deepseek-responses";

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp-deepseek-future",
            "object": "response",
            "created_at": 1_785_811_200_i64,
            "model": FUTURE_MODEL,
            "status": "completed",
            "output": [{
                "id": "msg-future",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "answer", "annotations": []}]
            }],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2
            },
            "error": null,
            "incomplete_details": null,
            "reasoning": null
        })))
        .expect(1)
        .mount(&server)
        .await;

    let response = provider(&server)
        .responses(FUTURE_MODEL)
        .expect("future model")
        .generate(request("future"), CallOptions::default())
        .await
        .expect("future response");
    assert_eq!(response.id(), Some("resp-deepseek-future"));

    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 1);
    let body: Value = serde_json::from_slice(&requests[0].body).expect("future request body");
    assert_eq!(body["model"], FUTURE_MODEL);
}

#[tokio::test]
async fn responses_reject_unreviewed_raw_options_before_submission() {
    let server = MockServer::start().await;
    let model = provider(&server)
        .responses("private-deepseek-responses")
        .expect("model");
    let options = CallOptions::default()
        .with_raw_provider_options_for(
            &model,
            json!({"previous_response_id": "resp-not-supported"}),
        )
        .expect("raw provider options");
    let error = model
        .generate(request("continue"), options)
        .await
        .expect_err("unreviewed raw Responses options must fail locally");

    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(
        server
            .received_requests()
            .await
            .expect("requests")
            .is_empty()
    );
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
        LanguageStreamEvent::Usage(update)
            if update.usage().cache_read_tokens == UsageValue::Known(0)
                && update.usage().reasoning_tokens == UsageValue::Known(0)
                && update.usage().provider["deepseek"]["prompt_cache_miss_tokens"] == 0
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
