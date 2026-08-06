use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use siumai_core::{
    ApiModeId, ContentAnnotationTarget, ContentAnnotations, ContentPart, ErrorKind, ExecutionOwner,
    FinishReason, LanguageRequest, LanguageStreamDecoder, LanguageStreamEvent, MediaData,
    MediaPart, Message, MessageAnnotationTarget, MessageAnnotations, MessagePart, MessageRole,
    ModelId, ProtocolId, ProviderId, ProviderScope, StreamTerminal, ToolAnnotationTarget,
    ToolAnnotations, ToolCall, ToolChoice, ToolOutcome, ToolResult, ToolSpec,
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
    let call = ToolCall {
        id: "toolu_1".to_string(),
        name: "lookup".to_string(),
        arguments: json!({"sku": "A-1"}),
        owner: ExecutionOwner::Local,
    };
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
    for (field, value) in [("api_key", "secret"), ("service_tier", "priority")] {
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
        ("end_turn", FinishReason::Stop),
        ("max_tokens", FinishReason::Length),
        ("stop_sequence", FinishReason::Stop),
        ("tool_use", FinishReason::ToolCalls),
        ("pause_turn", FinishReason::Other("pause_turn".to_string())),
        ("refusal", FinishReason::Refusal),
        (
            "model_context_window_exceeded",
            FinishReason::Other("model_context_window_exceeded".to_string()),
        ),
        ("unknown", FinishReason::Other("unknown".to_string())),
        (
            "future_reason",
            FinishReason::Other("future_reason".to_string()),
        ),
    ];
    for (wire, expected) in cases {
        let response = decode_response(&response_fixture(wire), &scope(), &model()).unwrap();
        assert_eq!(response.finish_reason(), &expected);
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
fn stream_error_event_is_a_canonical_failed_terminal() {
    let mut decoder = MessagesStreamDecoder::new(scope(), model());
    let events = decoder
        .decode(
            &json!({
                "type": "error",
                "error": {"type": "overloaded_error", "message": "private body"}
            })
            .to_string(),
        )
        .unwrap();
    assert!(matches!(
        events.as_slice(),
        [LanguageStreamEvent::Terminal(StreamTerminal::Failed { error, response: None })]
            if error.kind() == ErrorKind::Provider
    ));
    assert!(!format!("{:?}", events).contains("private body"));
}
