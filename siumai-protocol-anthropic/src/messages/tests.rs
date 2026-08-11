use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use siumai_core::{
    ApiModeId, ContentAnnotationTarget, ContentAnnotations, ContentPart, ErrorKind,
    LanguageCompletionReason, LanguageIncompleteReason, LanguageRequest, LanguageStreamDecoder,
    LanguageStreamEvent, LanguageTermination, MediaData, MediaPart, Message,
    MessageAnnotationTarget, MessageAnnotations, MessagePart, MessageRole, ModelId, ProtocolId,
    ProviderId, ProviderScope, ReplayDomain, ReplayDomainId, ResponseDiagnostics, StreamTerminal,
    ToolAnnotationTarget, ToolAnnotations, ToolCall, ToolChoice, ToolOutcome, ToolResult, ToolSpec,
    TypedProviderAnnotation,
};

use super::*;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestMessageCache {
    ttl: CacheTtl,
}

impl TypedProviderAnnotation for TestMessageCache {
    type Target = MessageAnnotationTarget;

    const NAMESPACE: &'static str = "test-anthropic";
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestContentCache {
    ttl: CacheTtl,
}

impl TypedProviderAnnotation for TestContentCache {
    type Target = ContentAnnotationTarget;

    const NAMESPACE: &'static str = "test-anthropic";
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestToolCache {
    ttl: CacheTtl,
}

impl TypedProviderAnnotation for TestToolCache {
    type Target = ToolAnnotationTarget;

    const NAMESPACE: &'static str = "test-anthropic";
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ForeignContentAnnotation {
    label: String,
}

impl TypedProviderAnnotation for ForeignContentAnnotation {
    type Target = ContentAnnotationTarget;

    const NAMESPACE: &'static str = "foreign";
    const API_MODE: Option<&'static str> = Some("responses");
}

#[derive(Debug, Clone, Copy)]
struct TestResolver;

impl MessagesAnnotationResolver for TestResolver {
    fn resolve_message(
        &self,
        annotations: &MessageAnnotations,
    ) -> Result<MessageNodeOptions, MessagesCodecError> {
        annotations
            .decode::<TestMessageCache>()
            .map_err(|source| MessagesCodecError::InvalidAnnotation {
                node: "test message",
                source,
            })
            .map(|annotation| {
                annotation.map_or_else(MessageNodeOptions::default, |annotation| {
                    MessageNodeOptions::default()
                        .with_cache_control(CacheControl::new(annotation.ttl))
                })
            })
    }

    fn resolve_content(
        &self,
        annotations: &ContentAnnotations,
    ) -> Result<ContentNodeOptions, MessagesCodecError> {
        annotations
            .decode::<TestContentCache>()
            .map_err(|source| MessagesCodecError::InvalidAnnotation {
                node: "test content",
                source,
            })
            .map(|annotation| {
                annotation.map_or_else(ContentNodeOptions::default, |annotation| {
                    ContentNodeOptions::default()
                        .with_cache_control(CacheControl::new(annotation.ttl))
                })
            })
    }

    fn resolve_tool(
        &self,
        annotations: &ToolAnnotations,
    ) -> Result<ToolNodeOptions, MessagesCodecError> {
        annotations
            .decode::<TestToolCache>()
            .map_err(|source| MessagesCodecError::InvalidAnnotation {
                node: "test tool",
                source,
            })
            .map(|annotation| {
                annotation.map_or_else(ToolNodeOptions::default, |annotation| {
                    ToolNodeOptions::default().with_cache_control(CacheControl::new(annotation.ttl))
                })
            })
    }
}

fn model() -> ModelId {
    ModelId::new("claude-fable-5").unwrap()
}

fn scope() -> ProviderScope {
    ProviderScope::new(ProviderId::new("anthropic").unwrap())
        .with_protocol(ProtocolId::new(PROTOCOL_ID).unwrap())
        .with_api_mode(ApiModeId::new(API_MODE_ID).unwrap())
        .with_replay_domain(ReplayDomain::official(
            ReplayDomainId::new("anthropic-test").unwrap(),
        ))
}

fn request(messages: Vec<Message>) -> LanguageRequest {
    let mut request = LanguageRequest::new(messages);
    request.generation.max_output_tokens = Some(2_048);
    request
}

#[test]
fn encodes_system_user_media_and_generation_controls() {
    let mut request = request(vec![
        Message::text(MessageRole::System, "Be concise."),
        Message::new(
            MessageRole::User,
            [
                MessagePart::text("Describe this image."),
                MessagePart::new(ContentPart::Media(MediaPart {
                    media_type: "image/png".to_string(),
                    data: MediaData::Bytes(vec![1, 2, 3].into()),
                    name: None,
                })),
            ],
        ),
    ]);
    request.generation.temperature = Some(0.4);
    request.generation.stop_sequences = vec!["END".to_string()];

    let encoded = encode_request(&model(), &request, &MessagesRequestOptions::new(false)).unwrap();

    assert_eq!(encoded["model"], "claude-fable-5");
    assert_eq!(encoded["system"][0]["text"], "Be concise.");
    assert_eq!(encoded["messages"][0]["role"], "user");
    assert_eq!(encoded["messages"][0]["content"][1]["type"], "image");
    assert_eq!(encoded["temperature"], 0.4);
    assert_eq!(encoded["stop_sequences"], json!(["END"]));
}

#[test]
fn encodes_function_tools_calls_and_results() {
    let tool = ToolSpec::new(
        "lookup",
        Some("Look up inventory".to_string()),
        json!({"type": "object", "properties": {"sku": {"type": "string"}}}),
    )
    .unwrap();
    let call = ToolCall::local("toolu_1", "lookup", json!({"sku": "A-1"})).unwrap();
    let result = ToolResult {
        call_id: "toolu_1".to_string(),
        name: "lookup".to_string(),
        outcome: ToolOutcome::Success {
            value: json!({"count": 4}),
        },
    };
    let mut request = request(vec![
        Message::text(MessageRole::User, "Check inventory."),
        Message::new(MessageRole::Assistant, [ContentPart::ToolCall(call)]),
        Message::new(MessageRole::Tool, [ContentPart::ToolResult(result)]),
    ]);
    request.tools.push(tool);
    request.tool_choice = Some(ToolChoice::Named {
        name: "lookup".to_string(),
    });

    let encoded = encode_request(&model(), &request, &MessagesRequestOptions::default()).unwrap();

    assert_eq!(encoded["tools"][0]["name"], "lookup");
    assert_eq!(
        encoded["tool_choice"],
        json!({"type": "tool", "name": "lookup"})
    );
    assert_eq!(encoded["messages"][1]["content"][0]["type"], "tool_use");
    assert_eq!(encoded["messages"][2]["content"][0]["type"], "tool_result");
}

#[test]
fn resolver_projects_cache_in_prefix_order_and_ignores_foreign_annotations() {
    let tool = ToolSpec::new("lookup", None, json!({"type": "object", "properties": {}}))
        .unwrap()
        .with_provider_annotation(&TestToolCache {
            ttl: CacheTtl::OneHour,
        })
        .unwrap();
    let system = Message::text(MessageRole::System, "Stable policy")
        .with_provider_annotation(&TestMessageCache {
            ttl: CacheTtl::OneHour,
        })
        .unwrap();
    let content = MessagePart::text("Hello")
        .with_provider_annotation(&ForeignContentAnnotation {
            label: "keep-me".to_string(),
        })
        .unwrap()
        .with_provider_annotation(&TestContentCache {
            ttl: CacheTtl::FiveMinutes,
        })
        .unwrap();
    let mut request = request(vec![system, Message::new(MessageRole::User, [content])]);
    request.tools.push(tool);

    let encoded = encode_request_with_resolver(
        &model(),
        &request,
        &MessagesRequestOptions::default(),
        &TestResolver,
    )
    .unwrap();

    assert_eq!(encoded["tools"][0]["cache_control"]["ttl"], "1h");
    assert_eq!(encoded["system"][0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        encoded["messages"][0]["content"][0]["cache_control"]["ttl"],
        "5m"
    );
    assert!(encoded.to_string().find("keep-me").is_none());
}

#[test]
fn rejects_cache_breakpoint_limit_and_invalid_ttl_order() {
    let messages = (0..5)
        .map(|index| {
            let part = MessagePart::text(format!("part-{index}"))
                .with_provider_annotation(&TestContentCache {
                    ttl: CacheTtl::FiveMinutes,
                })
                .unwrap();
            Message::new(MessageRole::User, [part])
        })
        .collect();
    let error = encode_request_with_resolver(
        &model(),
        &request(messages),
        &MessagesRequestOptions::default(),
        &TestResolver,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        MessagesCodecError::TooManyCacheBreakpoints {
            actual: 5,
            maximum: 4
        }
    ));

    let tool = ToolSpec::new("lookup", None, json!({"type": "object"}))
        .unwrap()
        .with_provider_annotation(&TestToolCache {
            ttl: CacheTtl::FiveMinutes,
        })
        .unwrap();
    let system = Message::text(MessageRole::System, "Policy")
        .with_provider_annotation(&TestMessageCache {
            ttl: CacheTtl::OneHour,
        })
        .unwrap();
    let mut invalid = request(vec![system, Message::text(MessageRole::User, "Hello")]);
    invalid.tools.push(tool);
    assert!(matches!(
        encode_request_with_resolver(
            &model(),
            &invalid,
            &MessagesRequestOptions::default(),
            &TestResolver,
        ),
        Err(MessagesCodecError::InvalidCacheTtlOrder)
    ));
}

#[test]
fn rejects_protected_options() {
    for (field, value) in [
        ("api_key", "secret"),
        ("service_tier", "auto"),
        ("cache_control", "ephemeral"),
        ("mcp_servers", "untyped"),
    ] {
        let mut extra = BTreeMap::new();
        extra.insert(field.to_string(), Value::String(value.to_string()));
        let options = MessagesRequestOptions::default().with_extra(extra);
        assert!(matches!(
            encode_request(
                &model(),
                &request(vec![Message::text(MessageRole::User, "Hello")]),
                &options,
            ),
            Err(MessagesCodecError::ProtectedOptionField { .. })
        ));
    }
}

#[test]
fn encodes_all_explicit_thinking_modes_and_structured_output() {
    let base = request(vec![Message::text(MessageRole::User, "Return JSON")]);
    for (thinking, expected) in [
        (ThinkingConfig::Disabled, "disabled"),
        (ThinkingConfig::adaptive(), "adaptive"),
        (ThinkingConfig::enabled(1_024), "enabled"),
    ] {
        let encoded = encode_request(
            &model(),
            &base,
            &MessagesRequestOptions::default().with_thinking(thinking),
        )
        .unwrap();
        assert_eq!(encoded["thinking"]["type"], expected);
    }

    let mut structured = base;
    structured.structured_output = Some(siumai_core::StructuredOutputSpec {
        name: "answer".to_string(),
        description: None,
        schema: json!({"type": "object", "properties": {"answer": {"type": "string"}}}),
        strict: true,
    });
    let encoded =
        encode_request(&model(), &structured, &MessagesRequestOptions::default()).unwrap();
    assert_eq!(encoded["output_config"]["format"]["type"], "json_schema");
}

fn response_fixture(stop_reason: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello"}],
        "model": "claude-fable-5",
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 3,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 4
        }
    }))
    .unwrap()
}

#[test]
fn maps_current_stop_reasons_without_collapsing_unknown_values() {
    let cases = [
        (
            "end_turn",
            LanguageTermination::Completed(LanguageCompletionReason::Stop),
        ),
        (
            "max_tokens",
            LanguageTermination::Incomplete(LanguageIncompleteReason::MaxOutputTokens),
        ),
        (
            "stop_sequence",
            LanguageTermination::Completed(LanguageCompletionReason::Stop),
        ),
        (
            "tool_use",
            LanguageTermination::Completed(LanguageCompletionReason::ToolCalls),
        ),
        (
            "pause_turn",
            LanguageTermination::Incomplete(LanguageIncompleteReason::Other(
                "pause_turn".to_string(),
            )),
        ),
        (
            "refusal",
            LanguageTermination::Completed(LanguageCompletionReason::Refusal),
        ),
        (
            "model_context_window_exceeded",
            LanguageTermination::Incomplete(LanguageIncompleteReason::Other(
                "model_context_window_exceeded".to_string(),
            )),
        ),
        (
            "compaction",
            LanguageTermination::Incomplete(LanguageIncompleteReason::Other(
                "compaction".to_string(),
            )),
        ),
        (
            "unknown",
            LanguageTermination::Incomplete(LanguageIncompleteReason::Other("unknown".to_string())),
        ),
        (
            "future_reason",
            LanguageTermination::Incomplete(LanguageIncompleteReason::Other(
                "future_reason".to_string(),
            )),
        ),
    ];
    for (wire, expected) in cases {
        let response = decode_response(&response_fixture(wire), &scope(), &model()).unwrap();
        assert_eq!(response.termination(), &expected);
    }
}

#[test]
fn response_preserves_signed_thinking_as_replayable_opaque_state() {
    let body = serde_json::to_vec(&json!({
        "id": "msg_thinking",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "inspect", "signature": "signed"},
            {"type": "text", "text": "done"}
        ],
        "model": "claude-fable-5",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 4, "output_tokens": 2}
    }))
    .unwrap();

    let response = decode_response(&body, &scope(), &model()).unwrap();

    assert!(response.content().iter().any(|part| matches!(
        part,
        ContentPart::Reasoning { text } if text == "inspect"
    )));
    assert!(response.content().iter().any(|part| matches!(
        part,
        ContentPart::ProviderOpaque(item)
            if item.kind() == OPAQUE_CONTENT_BLOCK_KIND
                && item.data()["signature"] == "signed"
    )));
}

#[test]
fn automatic_cache_without_an_eligible_target_keeps_the_request_valid() {
    let body = serde_json::to_vec(&json!({
        "id": "msg_thinking_only",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "inspect", "signature": "signed"}
        ],
        "model": "claude-fable-5",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 4, "output_tokens": 1}
    }))
    .unwrap();
    let response = decode_response(&body, &scope(), &model()).unwrap();
    let history = response
        .project_assistant_history()
        .into_parts()
        .0
        .expect("response projects assistant history");

    let encoded = encode_request_for_scope(
        &scope(),
        &model(),
        &request(vec![history]),
        &MessagesRequestOptions::default()
            .with_cache_control(CacheControl::new(CacheTtl::FiveMinutes)),
    )
    .unwrap();

    assert_eq!(
        encoded["cache_control"],
        json!({"type": "ephemeral", "ttl": "5m"})
    );
    assert_eq!(encoded["messages"][0]["content"][0]["type"], "thinking");
}

#[test]
fn response_rejects_non_object_tool_input_before_execution() {
    for input in [json!(["value"]), json!("value"), Value::Null] {
        let body = serde_json::to_vec(&json!({
            "id": "msg_tool_input",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "call_1",
                "name": "lookup",
                "input": input
            }],
            "model": "claude-fable-5",
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {"input_tokens": 4, "output_tokens": 2}
        }))
        .unwrap();

        assert!(matches!(
            decode_response(&body, &scope(), &model()),
            Err(MessagesCodecError::ProtocolViolation {
                reason: "tool_use input must be a JSON object",
            })
        ));
    }
}

#[test]
fn request_rejects_reasoning_that_disagrees_with_native_replay() {
    let body = serde_json::to_vec(&json!({
        "id": "msg_thinking",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "inspect", "signature": "signed"},
            {"type": "text", "text": "done"}
        ],
        "model": "claude-fable-5",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 4, "output_tokens": 2}
    }))
    .unwrap();
    let response = decode_response(&body, &scope(), &model()).unwrap();
    let mut history = response
        .project_assistant_history()
        .into_parts()
        .0
        .expect("response projects assistant history");
    let reasoning = history
        .content_mut()
        .iter_mut()
        .find(|part| matches!(part.content(), ContentPart::Reasoning { .. }))
        .expect("projected reasoning");
    let ContentPart::Reasoning { text } = reasoning.content_mut() else {
        unreachable!("selected reasoning part")
    };
    *text = "different".to_string();

    let error = encode_request_for_scope(
        &scope(),
        &model(),
        &request(vec![history]),
        &MessagesRequestOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        MessagesCodecError::InvalidOption {
            field: "messages.reasoning",
            ..
        }
    ));
}

#[test]
fn stream_emits_one_terminal_and_rejects_unexpected_eof() {
    let mut decoder = MessagesStreamDecoder::new(scope(), model());
    let frames = [
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_stream",
                "type": "message",
                "role": "assistant",
                "model": "claude-fable-5",
                "usage": {"input_tokens": 5}
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
    let mut terminal_count = 0;
    for frame in frames {
        let events = decoder.decode(&frame.to_string()).unwrap();
        terminal_count += events
            .iter()
            .filter(|event| matches!(event, LanguageStreamEvent::Terminal(_)))
            .count();
    }
    assert_eq!(terminal_count, 1);
    assert!(decoder.terminal_seen());
    assert!(decoder.finish().unwrap().is_empty());

    let mut incomplete = MessagesStreamDecoder::new(scope(), model());
    incomplete
        .decode(
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_incomplete",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-fable-5",
                    "usage": {}
                }
            })
            .to_string(),
        )
        .unwrap();
    let error = incomplete.finish().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::UnexpectedEof);
}

#[test]
fn streamed_tool_input_is_bounded_before_json_normalization() {
    let mut decoder = MessagesStreamDecoder::new(scope(), model());
    decoder
        .decode(
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_bounded",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-fable-5",
                    "usage": {}
                }
            })
            .to_string(),
        )
        .unwrap();
    decoder
        .decode(
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_bounded",
                    "name": "lookup",
                    "input": {}
                }
            })
            .to_string(),
        )
        .unwrap();
    let oversized = json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "input_json_delta",
            "partial_json": " ".repeat(siumai_core::DEFAULT_TOOL_INPUT_BYTE_LIMIT + 1)
        }
    })
    .to_string();

    let error = decoder.decode(&oversized).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResponseLimit);
}

#[test]
fn stream_error_event_is_a_canonical_failed_terminal() {
    let mut decoder = MessagesStreamDecoder::new(scope(), model()).with_response_diagnostics(
        ResponseDiagnostics::default()
            .with_status(200)
            .with_retry_after(Duration::from_secs(7)),
    );
    let events = decoder
        .decode(
            &json!({
                "type": "error",
                "request_id": "req_in_band",
                "error": {"type": "overloaded_error", "message": "private body"}
            })
            .to_string(),
        )
        .unwrap();
    assert!(matches!(
        events.as_slice(),
        [LanguageStreamEvent::Terminal(StreamTerminal::Failed { error, partial: None })]
            if error.kind() == ErrorKind::Unavailable
                && error.diagnostics().is_some_and(|diagnostics|
                    diagnostics.status() == Some(200)
                        && diagnostics.provider_type() == Some("overloaded_error")
                        && diagnostics.request_id() == Some("req_in_band")
                        && diagnostics.retry_after() == Some(Duration::from_secs(7)))
                && error.sensitive_response().is_some_and(|response|
                    response.expose().1.windows("private body".len()).any(|window|
                        window == "private body".as_bytes()))
    ));
    assert!(!format!("{:?}", events).contains("private body"));
}
