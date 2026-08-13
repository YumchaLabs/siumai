use serde_json::{Value, json};
use siumai_core::{
    ApiModeId, ContentPart, ErrorKind, ExecutionOwner, LanguageCompletionReason,
    LanguageIncompleteReason, LanguageRequest, LanguageStreamDecoder, LanguageStreamEvent,
    LanguageTermination, Message, MessageRole, ModelId, PlatformId, ProtocolId, ProviderId,
    ProviderScope, ReplayDomain, ReplayDomainId, StreamTerminal, ToolCall, UsageValue,
};

use super::*;

fn model() -> ModelId {
    ModelId::new("claude-fable-5").expect("valid test model")
}

fn scope() -> ProviderScope {
    ProviderScope::new(ProviderId::new("anthropic").expect("valid provider"))
        .with_protocol(ProtocolId::new(PROTOCOL_ID).expect("valid protocol"))
        .with_api_mode(ApiModeId::new(API_MODE_ID).expect("valid API mode"))
        .with_replay_domain(ReplayDomain::official(
            ReplayDomainId::new("anthropic-test").expect("valid replay domain"),
        ))
}

fn scoped_replay_domain(caller_scope: &str) -> ProviderScope {
    ProviderScope::new(ProviderId::new("anthropic").expect("valid provider"))
        .with_platform(PlatformId::new("anthropic-api").expect("valid platform"))
        .with_protocol(ProtocolId::new(PROTOCOL_ID).expect("valid protocol"))
        .with_api_mode(ApiModeId::new(API_MODE_ID).expect("valid API mode"))
        .with_replay_domain(
            ReplayDomain::official(
                ReplayDomainId::new("anthropic-compaction").expect("valid replay domain"),
            )
            .with_caller_scope(
                ReplayDomainId::new(caller_scope).expect("valid caller replay scope"),
            ),
        )
}

fn compaction_response_body(summary: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "id": "msg_compaction",
        "type": "message",
        "role": "assistant",
        "model": "claude-fable-5",
        "content": [{
            "type": "compaction",
            "content": summary,
            "encrypted_content": "encrypted-compaction-state"
        }],
        "stop_reason": "compaction",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 120,
            "output_tokens": 4,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens_details": {"thinking_tokens": 0},
            "iterations": [
                {
                    "type": "compaction",
                    "input_tokens": 30,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 7
                },
                {"type": "message", "input_tokens": 120, "output_tokens": 4}
            ]
        }
    }))
    .expect("serialize compaction fixture")
}

fn response_with_content(content: serde_json::Value, stop_reason: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "id": "msg_native_tools",
        "type": "message",
        "role": "assistant",
        "model": "claude-fable-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {"input_tokens": 12, "output_tokens": 6}
    }))
    .expect("serialize native tool fixture")
}

#[test]
fn direct_hosted_tools_remain_provider_owned_and_replay_exactly_in_scope() {
    let source_scope = scoped_replay_domain("workspace-a");
    let blocks = json!([
        {
            "type": "server_tool_use",
            "id": "srv_web",
            "name": "web_search",
            "input": {"query": "siumai"},
            "future_state": {"retained": true}
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srv_web",
            "content": [{
                "type": "web_search_result",
                "title": "Siumai",
                "url": "https://example.invalid/siumai"
            }]
        },
        {
            "type": "web_fetch_tool_result",
            "tool_use_id": "srv_fetch",
            "content": {"type": "web_fetch_result", "future": "preserved"}
        },
        {
            "type": "code_execution_tool_result",
            "tool_use_id": "srv_code",
            "content": {"type": "code_execution_result", "stdout": "ok"}
        },
        {
            "type": "bash_code_execution_tool_result",
            "tool_use_id": "srv_bash",
            "content": {"type": "bash_code_execution_result", "stdout": "ok"}
        },
        {
            "type": "text_editor_code_execution_tool_result",
            "tool_use_id": "srv_editor",
            "content": {"type": "text_editor_code_execution_result", "path": "notes.md"}
        },
        {
            "type": "tool_search_tool_result",
            "tool_use_id": "srv_search",
            "content": {"type": "tool_search_tool_result", "names": ["lookup"]}
        },
        {
            "type": "advisor_tool_result",
            "tool_use_id": "srv_advisor",
            "content": {"type": "advisor_result", "answer": "bounded"}
        },
        {
            "type": "mcp_tool_use",
            "id": "mcp_use",
            "name": "lookup",
            "server_name": "knowledge",
            "input": {
                "query": "siumai",
                "headers": {"authorization": "hosted-secret-sentinel"}
            }
        },
        {
            "type": "mcp_tool_result",
            "tool_use_id": "mcp_use",
            "content": "MCP result",
            "is_error": false
        }
    ]);
    let response = decode_response(
        &response_with_content(blocks.clone(), "end_turn"),
        &source_scope,
        &model(),
    )
    .expect("decode hosted-tool response");

    assert!(
        response
            .content()
            .iter()
            .all(|part| !matches!(part, ContentPart::ToolCall(_) | ContentPart::ToolResult(_)))
    );
    let native = response
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::ProviderOpaque(item) => Some(item),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        native.len(),
        blocks.as_array().expect("fixture array").len()
    );
    assert!(matches!(
        native[0]
            .anthropic_hosted_tool()
            .expect("inspect server tool"),
        Some(AnthropicHostedToolBlockRef::ServerToolUse(value))
            if value.id() == "srv_web"
                && value.name() == "web_search"
                && value.input() == &json!({"query": "siumai"})
    ));
    assert!(matches!(
        native[8]
            .anthropic_hosted_tool()
            .expect("inspect MCP tool"),
        Some(AnthropicHostedToolBlockRef::McpToolUse(value))
            if value.id() == "mcp_use"
                && value.server_name() == "knowledge"
    ));
    assert!(matches!(
        native[9]
            .anthropic_hosted_tool()
            .expect("inspect MCP result"),
        Some(AnthropicHostedToolBlockRef::Result(value))
            if value.kind() == AnthropicHostedToolResultKind::Mcp
                && value.tool_use_id() == "mcp_use"
                && value.content() == &json!("MCP result")
                && value.is_error() == Some(false)
    ));
    let expected_results = [
        AnthropicHostedToolResultKind::WebSearch,
        AnthropicHostedToolResultKind::WebFetch,
        AnthropicHostedToolResultKind::CodeExecution,
        AnthropicHostedToolResultKind::BashCodeExecution,
        AnthropicHostedToolResultKind::TextEditorCodeExecution,
        AnthropicHostedToolResultKind::ToolSearch,
        AnthropicHostedToolResultKind::Advisor,
    ];
    for (item, expected) in native[1..8].iter().zip(expected_results) {
        assert!(matches!(
            item.anthropic_hosted_tool()
                .expect("inspect hosted-tool result"),
            Some(AnthropicHostedToolBlockRef::Result(value)) if value.kind() == expected
        ));
    }
    assert_eq!(native[1].relations()[0].kind(), "related_item");
    assert_eq!(native[1].relations()[0].target_id(), "srv_web");
    let opaque_debug = format!("{:?}", native[8]);
    let inspection_debug = format!(
        "{:?}",
        native[8]
            .anthropic_hosted_tool()
            .expect("inspect for diagnostics")
    );
    let response_debug = format!("{response:?}");
    for sensitive in ["mcp_use", "knowledge", "siumai", "hosted-secret-sentinel"] {
        assert!(!opaque_debug.contains(sensitive));
        assert!(!inspection_debug.contains(sensitive));
        assert!(!response_debug.contains(sensitive));
    }

    let history = response
        .project_assistant_history()
        .into_message()
        .expect("hosted tools project to assistant history");
    let mut continuation = LanguageRequest::new(vec![history, Message::user("Continue")]);
    continuation.generation.max_output_tokens = Some(2_048);
    let encoded = encode_request_for_scope(
        &source_scope,
        &model(),
        &continuation,
        &MessagesRequestOptions::default(),
    )
    .expect("replay hosted tools in the source scope");
    assert_eq!(encoded["messages"][0]["content"], blocks);

    assert!(matches!(
        encode_request_for_scope(
            &scoped_replay_domain("workspace-b"),
            &model(),
            &continuation,
            &MessagesRequestOptions::default(),
        ),
        Err(MessagesCodecError::Unsupported { .. })
    ));
}

#[test]
fn arbitrary_json_tool_inputs_preserve_direct_and_replay_semantics() {
    let source_scope = scoped_replay_domain("workspace-a");
    for (index, input) in [
        Value::Null,
        json!(["array", 1]),
        json!("scalar"),
        json!(7),
        json!(true),
    ]
    .into_iter()
    .enumerate()
    {
        let mcp_id = format!("mcp_{index}");
        let call_id = format!("call_{index}");
        let blocks = json!([
            {
                "type": "mcp_tool_use",
                "id": mcp_id,
                "name": "lookup",
                "server_name": "knowledge",
                "input": input.clone()
            },
            {
                "type": "tool_use",
                "id": call_id,
                "name": "local_lookup",
                "input": input.clone(),
                "caller": {
                    "type": "code_execution_20250825",
                    "tool_id": "srv_code"
                }
            }
        ]);
        let response = decode_response(
            &response_with_content(blocks.clone(), "tool_use"),
            &source_scope,
            &model(),
        )
        .expect("decode arbitrary tool inputs");

        assert!(response.content().iter().any(|part| matches!(
            part,
            ContentPart::ToolCall(call)
                if call.id() == call_id && call.arguments() == &input
        )));
        assert!(response.content().iter().any(|part| matches!(
            part,
            ContentPart::ProviderOpaque(item)
                if matches!(
                    item.anthropic_hosted_tool().expect("inspect MCP input"),
                    Some(AnthropicHostedToolBlockRef::McpToolUse(value))
                        if value.id() == mcp_id && value.input() == &input
                )
        )));

        let history = response
            .project_assistant_history()
            .into_message()
            .expect("arbitrary tool inputs project to history");
        let mut continuation = LanguageRequest::new(vec![history, Message::user("Continue")]);
        continuation.generation.max_output_tokens = Some(2_048);
        let encoded = encode_request_for_scope(
            &source_scope,
            &model(),
            &continuation,
            &MessagesRequestOptions::default(),
        )
        .expect("replay arbitrary tool inputs");
        assert_eq!(encoded["messages"][0]["content"], blocks);
    }
}

#[test]
fn ordinary_local_tool_inputs_round_trip_direct_history_and_request() {
    let source_scope = scoped_replay_domain("workspace-a");
    for (index, input) in [
        Value::Null,
        json!(["array", 1]),
        json!("scalar"),
        json!(7),
        json!(true),
    ]
    .into_iter()
    .enumerate()
    {
        let block = json!({
            "type": "tool_use",
            "id": format!("local_{index}"),
            "name": "lookup",
            "input": input.clone()
        });
        let response = decode_response(
            &response_with_content(json!([block.clone()]), "tool_use"),
            &source_scope,
            &model(),
        )
        .expect("decode ordinary local tool input");
        assert!(matches!(
            response.content(),
            [ContentPart::ToolCall(call)] if call.arguments() == &input
        ));

        let history = response
            .project_assistant_history()
            .into_message()
            .expect("local tool input projects to history");
        let mut continuation = LanguageRequest::new(vec![history, Message::user("Continue")]);
        continuation.generation.max_output_tokens = Some(2_048);
        let encoded = encode_request_for_scope(
            &source_scope,
            &model(),
            &continuation,
            &MessagesRequestOptions::default(),
        )
        .expect("replay ordinary local tool input");
        assert_eq!(encoded["messages"][0]["content"], json!([block]));
    }
}

#[test]
fn hosted_replay_rejects_incomplete_foreign_and_unknown_native_state() {
    let source_scope = scoped_replay_domain("workspace-a");
    let response = decode_response(
        &response_with_content(
            json!([{
                "type": "server_tool_use",
                "id": "srv_replay",
                "name": "web_search",
                "input": {"query": "siumai"}
            }]),
            "end_turn",
        ),
        &source_scope,
        &model(),
    )
    .expect("decode hosted replay fixture");
    let history = response
        .project_assistant_history()
        .into_message()
        .expect("hosted replay history");
    let mut continuation = LanguageRequest::new(vec![history, Message::user("Continue")]);
    continuation.generation.max_output_tokens = Some(2_048);

    let base_scope = || {
        ProviderScope::new(ProviderId::new("anthropic").expect("provider"))
            .with_platform(PlatformId::new("anthropic-api").expect("platform"))
            .with_protocol(ProtocolId::new(PROTOCOL_ID).expect("protocol"))
            .with_api_mode(ApiModeId::new(API_MODE_ID).expect("API mode"))
    };
    let foreign_scopes = [
        base_scope().with_replay_domain(
            ReplayDomain::official(
                ReplayDomainId::new("different-audience").expect("replay domain"),
            )
            .with_caller_scope(ReplayDomainId::new("workspace-a").expect("caller scope")),
        ),
        base_scope().with_replay_domain(
            ReplayDomain::official(
                ReplayDomainId::new("anthropic-compaction").expect("replay domain"),
            )
            .with_caller_scope(ReplayDomainId::new("workspace-b").expect("caller scope")),
        ),
        base_scope()
            .with_protocol(ProtocolId::new("other-protocol").expect("protocol"))
            .with_replay_domain(
                ReplayDomain::official(
                    ReplayDomainId::new("anthropic-compaction").expect("replay domain"),
                )
                .with_caller_scope(ReplayDomainId::new("workspace-a").expect("caller scope")),
            ),
        base_scope()
            .with_api_mode(ApiModeId::new("other-mode").expect("API mode"))
            .with_replay_domain(
                ReplayDomain::official(
                    ReplayDomainId::new("anthropic-compaction").expect("replay domain"),
                )
                .with_caller_scope(ReplayDomainId::new("workspace-a").expect("caller scope")),
            ),
        base_scope().with_replay_domain(ReplayDomain::official(
            ReplayDomainId::new("anthropic-compaction").expect("replay domain"),
        )),
        base_scope(),
    ];
    for foreign_scope in foreign_scopes {
        assert!(matches!(
            encode_request_for_scope(
                &foreign_scope,
                &model(),
                &continuation,
                &MessagesRequestOptions::default(),
            ),
            Err(MessagesCodecError::Unsupported { .. })
        ));
    }

    let future = decode_response(
        &response_with_content(
            json!([{
                "type": "future_hosted_tool_result",
                "tool_use_id": "future_use",
                "content": {"future": true}
            }]),
            "end_turn",
        ),
        &source_scope,
        &model(),
    )
    .expect("unknown future native output remains observable");
    let future_native = future
        .content()
        .iter()
        .find_map(|part| match part {
            ContentPart::ProviderOpaque(item) => Some(item),
            _ => None,
        })
        .expect("future native block");
    assert!(
        future_native
            .anthropic_hosted_tool()
            .expect("inspect future block")
            .is_none()
    );
    let history = future
        .project_assistant_history()
        .into_message()
        .expect("future block projects to bounded history");
    let mut request = LanguageRequest::new(vec![history, Message::user("Continue")]);
    request.generation.max_output_tokens = Some(2_048);
    assert!(matches!(
        encode_request_for_scope(
            &source_scope,
            &model(),
            &request,
            &MessagesRequestOptions::default(),
        ),
        Err(MessagesCodecError::Unsupported { .. })
    ));
}

#[test]
fn hosted_replay_rejects_tampered_retained_identity_metadata() {
    let source_scope = scoped_replay_domain("workspace-a");
    let response = decode_response(
        &response_with_content(
            json!([{
                "type": "web_search_tool_result",
                "tool_use_id": "srv_expected",
                "content": [{"type": "web_search_result", "title": "Siumai"}]
            }]),
            "end_turn",
        ),
        &source_scope,
        &model(),
    )
    .expect("decode hosted result");
    let native = response
        .content()
        .iter()
        .find_map(|part| match part {
            ContentPart::ProviderOpaque(item) => Some(item.clone()),
            _ => None,
        })
        .expect("native result");
    let mut wire = serde_json::to_value(native).expect("serialize retained item");
    wire["relations"][0]["target_id"] = json!("srv_tampered");
    let tampered = serde_json::from_value::<siumai_core::OpaqueProviderItem>(wire)
        .expect("structurally valid tampered item");
    let mut request = LanguageRequest::new(vec![
        Message::new(
            MessageRole::Assistant,
            [ContentPart::ProviderOpaque(tampered)],
        ),
        Message::user("Continue"),
    ]);
    request.generation.max_output_tokens = Some(2_048);

    assert!(matches!(
        encode_request_for_scope(
            &source_scope,
            &model(),
            &request,
            &MessagesRequestOptions::default(),
        ),
        Err(MessagesCodecError::InvalidOption {
            field: "messages.native.relations",
            ..
        })
    ));

    let response = decode_response(
        &response_with_content(
            json!([{
                "type": "server_tool_use",
                "id": "srv_without_caller",
                "name": "web_search",
                "input": {"query": "siumai"}
            }]),
            "end_turn",
        ),
        &source_scope,
        &model(),
    )
    .expect("decode hosted use without caller");
    let native = response
        .content()
        .iter()
        .find_map(|part| match part {
            ContentPart::ProviderOpaque(item) => Some(item.clone()),
            _ => None,
        })
        .expect("native hosted use");
    let mut wire = serde_json::to_value(native).expect("serialize retained use");
    wire["relations"] = json!([{"kind": "caller", "target_id": "forged-caller"}]);
    let tampered = serde_json::from_value::<siumai_core::OpaqueProviderItem>(wire)
        .expect("structurally valid forged caller relation");
    let mut request = LanguageRequest::new(vec![
        Message::new(
            MessageRole::Assistant,
            [ContentPart::ProviderOpaque(tampered)],
        ),
        Message::user("Continue"),
    ]);
    request.generation.max_output_tokens = Some(2_048);
    assert!(matches!(
        encode_request_for_scope(
            &source_scope,
            &model(),
            &request,
            &MessagesRequestOptions::default(),
        ),
        Err(MessagesCodecError::InvalidOption {
            field: "messages.native.relations",
            ..
        })
    ));

    assert!(matches!(
        decode_response(
            &response_with_content(
                json!([{
                    "type": "tool_use",
                    "id": "missing_caller_tool_id",
                    "name": "lookup",
                    "input": null,
                    "caller": {"type": "code_execution_20250825"}
                }]),
                "tool_use",
            ),
            &source_scope,
            &model(),
        ),
        Err(MessagesCodecError::ProtocolViolation { .. })
    ));
}

#[test]
fn caller_linked_tool_use_keeps_one_local_call_and_one_native_replay_block() {
    let source_scope = scoped_replay_domain("workspace-a");
    let raw = json!({
        "type": "tool_use",
        "id": "local_call",
        "name": "lookup",
        "input": {"query": "siumai"},
        "caller": {"type": "code_execution_20250825", "tool_id": "srv_code"},
        "future_state": {"retained": true}
    });
    let response = decode_response(
        &response_with_content(json!([raw.clone()]), "tool_use"),
        &source_scope,
        &model(),
    )
    .expect("decode caller-linked tool use");

    let local = response
        .content()
        .iter()
        .find_map(|part| match part {
            ContentPart::ToolCall(call) => Some(call),
            _ => None,
        })
        .expect("local tool call");
    assert_eq!(local.owner(), &ExecutionOwner::Local);
    assert_eq!(local.id(), "local_call");
    let native = response
        .content()
        .iter()
        .find_map(|part| match part {
            ContentPart::ProviderOpaque(item) => Some(item),
            _ => None,
        })
        .expect("native caller replay");
    assert_eq!(native.item_id(), Some("local_call"));
    assert_eq!(native.relations()[0].kind(), "caller");
    assert_eq!(native.relations()[0].target_id(), "srv_code");

    let history = response
        .project_assistant_history()
        .into_message()
        .expect("caller-linked call projects to history");
    let mut continuation = LanguageRequest::new(vec![history.clone(), Message::user("Continue")]);
    continuation.generation.max_output_tokens = Some(2_048);
    let encoded = encode_request_for_scope(
        &source_scope,
        &model(),
        &continuation,
        &MessagesRequestOptions::default(),
    )
    .expect("replay caller-linked tool use");
    assert_eq!(encoded["messages"][0]["content"], json!([raw]));

    let mismatched = Message::new(
        MessageRole::Assistant,
        [
            ContentPart::ToolCall(
                ToolCall::local("local_call", "different", json!({"query": "siumai"}))
                    .expect("local call"),
            ),
            history.content()[1].content().clone(),
        ],
    );
    let mut invalid = LanguageRequest::new(vec![mismatched, Message::user("Continue")]);
    invalid.generation.max_output_tokens = Some(2_048);
    assert!(matches!(
        encode_request_for_scope(
            &source_scope,
            &model(),
            &invalid,
            &MessagesRequestOptions::default(),
        ),
        Err(MessagesCodecError::InvalidOption {
            field: "messages.tool_use",
            ..
        })
    ));
}

#[test]
fn unknown_native_callers_remain_observable_but_are_not_executable_or_replayable() {
    let source_scope = scoped_replay_domain("workspace-a");
    for (raw, stop_reason) in [
        (
            json!({
                "type": "tool_use",
                "id": "future_local_caller",
                "name": "lookup",
                "input": ["future"],
                "caller": {"type": "future_caller", "tool_id": "future_parent"}
            }),
            "tool_use",
        ),
        (
            json!({
                "type": "server_tool_use",
                "id": "future_hosted_caller",
                "name": "web_search",
                "input": "future",
                "caller": {"type": "future_caller", "tool_id": "future_parent"}
            }),
            "end_turn",
        ),
    ] {
        let response = decode_response(
            &response_with_content(json!([raw]), stop_reason),
            &source_scope,
            &model(),
        )
        .expect("unknown caller remains observable");
        assert!(
            response
                .content()
                .iter()
                .all(|part| !matches!(part, ContentPart::ToolCall(_)))
        );
        let history = response
            .project_assistant_history()
            .into_message()
            .expect("unknown caller projects as opaque history");
        let mut continuation = LanguageRequest::new(vec![history, Message::user("Continue")]);
        continuation.generation.max_output_tokens = Some(2_048);
        assert!(matches!(
            encode_request_for_scope(
                &source_scope,
                &model(),
                &continuation,
                &MessagesRequestOptions::default(),
            ),
            Err(MessagesCodecError::Unsupported { .. })
        ));
    }
}

#[test]
fn streamed_unknown_tool_caller_stays_opaque_without_local_events() {
    let source_scope = scoped_replay_domain("workspace-a");
    let mut decoder = MessagesStreamDecoder::new(source_scope, model());
    let frames = [
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_future_caller",
                "type": "message",
                "role": "assistant",
                "model": "claude-fable-5",
                "usage": {"input_tokens": 3}
            }
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "future_local_caller",
                "name": "lookup",
                "input": {},
                "caller": {"type": "future_caller", "tool_id": "future_parent"}
            }
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "[\"future\"]"}
        }),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": null},
            "usage": {"output_tokens": 2}
        }),
        json!({"type": "message_stop"}),
    ];

    let mut events = Vec::new();
    for frame in frames {
        events.extend(decoder.decode(&frame.to_string()).expect("decode frame"));
    }
    assert!(events.iter().all(|event| !matches!(
        event,
        LanguageStreamEvent::ToolInputStart { .. }
            | LanguageStreamEvent::ToolInputDelta { .. }
            | LanguageStreamEvent::ToolCall(_)
    )));
    let native = events
        .iter()
        .find_map(|event| match event {
            LanguageStreamEvent::ProviderOpaque(item) => Some(item),
            _ => None,
        })
        .expect("unknown caller remains observable as native content");
    assert_eq!(native.data()["input"], json!(["future"]));

    let terminal = events
        .iter()
        .find_map(|event| match event {
            LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }) => Some(response),
            _ => None,
        })
        .expect("terminal response");
    assert!(matches!(
        terminal.content(),
        [ContentPart::ProviderOpaque(item)] if item.data()["input"] == json!(["future"])
    ));
}

#[test]
fn streamed_known_tool_caller_is_validated_before_local_events() {
    let mut valid = MessagesStreamDecoder::new(scoped_replay_domain("workspace-a"), model());
    valid
        .decode(
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_valid_caller",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-fable-5",
                    "usage": {}
                }
            })
            .to_string(),
        )
        .expect("decode message start");
    let events = valid
        .decode(
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "local_from_code",
                    "name": "lookup",
                    "input": {},
                    "caller": {
                        "type": "code_execution_20250825",
                        "tool_id": "srv_code"
                    }
                }
            })
            .to_string(),
        )
        .expect("known caller is executable");
    assert!(matches!(
        events.as_slice(),
        [LanguageStreamEvent::ToolInputStart {
            id,
            name,
            owner: ExecutionOwner::Local,
        }] if id == "local_from_code" && name == "lookup"
    ));
    let events = valid
        .decode(
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "{\"query\":\"siumai\"}"
                }
            })
            .to_string(),
        )
        .expect("known caller input delta remains local");
    assert!(matches!(
        events.as_slice(),
        [LanguageStreamEvent::ToolInputDelta { id, .. }] if id == "local_from_code"
    ));
    let events = valid
        .decode(&json!({"type": "content_block_stop", "index": 0}).to_string())
        .expect("known caller completes as a local call");
    assert!(events.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::ToolCall(call)
            if call.id() == "local_from_code"
                && call.arguments() == &json!({"query": "siumai"})
    )));

    let mut invalid = MessagesStreamDecoder::new(scoped_replay_domain("workspace-a"), model());
    invalid
        .decode(
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_invalid_caller",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-fable-5",
                    "usage": {}
                }
            })
            .to_string(),
        )
        .expect("decode message start");
    let error = invalid
        .decode(
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "invalid_code_caller",
                    "name": "lookup",
                    "input": {},
                    "caller": {"type": "code_execution_20250825"}
                }
            })
            .to_string(),
        )
        .expect_err("known caller without tool_id must fail before publication");
    assert_eq!(error.kind(), ErrorKind::Protocol);
}

#[test]
fn streamed_hosted_tool_input_never_enters_the_local_tool_event_lane() {
    let source_scope = scoped_replay_domain("workspace-a");
    let mut decoder = MessagesStreamDecoder::new(source_scope, model());
    let frames = [
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_hosted_stream",
                "type": "message",
                "role": "assistant",
                "model": "claude-fable-5",
                "usage": {"input_tokens": 3}
            }
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srv_stream",
                "name": "web_search",
                "input": {}
            }
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{\"query\":"}
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "\"siumai\"}"}
        }),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 2}
        }),
        json!({"type": "message_stop"}),
    ];

    let mut events = Vec::new();
    for frame in frames {
        events.extend(decoder.decode(&frame.to_string()).expect("decode frame"));
    }
    assert!(events.iter().all(|event| !matches!(
        event,
        LanguageStreamEvent::ToolInputStart { .. }
            | LanguageStreamEvent::ToolInputDelta { .. }
            | LanguageStreamEvent::ToolCall(_)
    )));
    let native = events
        .iter()
        .find_map(|event| match event {
            LanguageStreamEvent::ProviderOpaque(item) => Some(item),
            _ => None,
        })
        .expect("provider-owned streamed tool item");
    assert_eq!(native.data()["input"], json!({"query": "siumai"}));
    assert!(matches!(
        native
            .anthropic_hosted_tool()
            .expect("inspect streamed hosted tool"),
        Some(AnthropicHostedToolBlockRef::ServerToolUse(value))
            if value.id() == "srv_stream"
    ));

    let terminal = events
        .iter()
        .find_map(|event| match event {
            LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }) => Some(response),
            _ => None,
        })
        .expect("terminal response");
    assert!(
        terminal
            .content()
            .iter()
            .all(|part| !matches!(part, ContentPart::ToolCall(_) | ContentPart::ToolResult(_)))
    );
}

#[test]
fn streamed_hosted_tool_inputs_preserve_arbitrary_json_values() {
    for (index, input) in [
        Value::Null,
        json!(["array", 1]),
        json!("scalar"),
        json!(7),
        json!(true),
    ]
    .into_iter()
    .enumerate()
    {
        for use_delta in [false, true] {
            let mut decoder =
                MessagesStreamDecoder::new(scoped_replay_domain("workspace-a"), model());
            let mut frames = vec![
                json!({
                    "type": "message_start",
                    "message": {
                        "id": format!("msg_arbitrary_{index}_{use_delta}"),
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-fable-5",
                        "usage": {"input_tokens": 1}
                    }
                }),
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "mcp_tool_use",
                        "id": format!("mcp_arbitrary_{index}_{use_delta}"),
                        "name": "lookup",
                        "server_name": "knowledge",
                        "input": if use_delta { json!({}) } else { input.clone() }
                    }
                }),
            ];
            if use_delta {
                frames.push(json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": serde_json::to_string(&input).expect("encode input delta")
                    }
                }));
            }
            frames.extend([
                json!({"type": "content_block_stop", "index": 0}),
                json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                    "usage": {"output_tokens": 1}
                }),
                json!({"type": "message_stop"}),
            ]);

            let mut events = Vec::new();
            for frame in frames {
                events.extend(
                    decoder
                        .decode(&frame.to_string())
                        .expect("decode arbitrary input"),
                );
            }
            let native = events
                .iter()
                .find_map(|event| match event {
                    LanguageStreamEvent::ProviderOpaque(item) => Some(item),
                    _ => None,
                })
                .expect("streamed MCP item");
            assert_eq!(native.data()["input"], input);
            assert!(events.iter().all(|event| !matches!(
                event,
                LanguageStreamEvent::ToolInputStart { .. }
                    | LanguageStreamEvent::ToolInputDelta { .. }
                    | LanguageStreamEvent::ToolCall(_)
            )));
        }
    }
}

#[test]
fn streamed_hosted_tool_input_is_bounded_before_native_publication() {
    let mut decoder = MessagesStreamDecoder::new(scoped_replay_domain("workspace-a"), model());
    decoder
        .decode(
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_hosted_bounded",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-fable-5",
                    "usage": {}
                }
            })
            .to_string(),
        )
        .expect("decode message start");
    decoder
        .decode(
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "mcp_tool_use",
                    "id": "mcp_bounded",
                    "name": "lookup",
                    "server_name": "knowledge",
                    "input": {}
                }
            })
            .to_string(),
        )
        .expect("decode hosted tool start");
    let error = decoder
        .decode(
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": " ".repeat(siumai_core::DEFAULT_TOOL_INPUT_BYTE_LIMIT + 1)
                }
            })
            .to_string(),
        )
        .expect_err("oversized hosted input must fail before publication");
    assert_eq!(error.kind(), ErrorKind::ResponseLimit);
}

#[test]
fn streamed_native_items_are_bounded_and_redacted_before_publication() {
    let sentinel = "private-hosted-input-sentinel";
    let mut decoder = MessagesStreamDecoder::new(scoped_replay_domain("workspace-a"), model());
    decoder
        .decode(
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_native_budget",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-fable-5",
                    "usage": {}
                }
            })
            .to_string(),
        )
        .expect("decode message start");
    decoder
        .decode(
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "mcp_tool_use",
                    "id": "mcp_redacted",
                    "name": "lookup",
                    "server_name": "knowledge",
                    "input": sentinel
                }
            })
            .to_string(),
        )
        .expect("decode redacted block start");
    assert!(!format!("{decoder:?}").contains(sentinel));
    decoder
        .decode(&json!({"type": "content_block_stop", "index": 0}).to_string())
        .expect("decode first native block");

    for index in 1..=siumai_core::DEFAULT_OPAQUE_ITEM_COUNT_LIMIT {
        decoder
            .decode(
                &json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "future_native_block",
                        "value": index
                    }
                })
                .to_string(),
            )
            .expect("decode native block start");
        let result =
            decoder.decode(&json!({"type": "content_block_stop", "index": index}).to_string());
        if index < siumai_core::DEFAULT_OPAQUE_ITEM_COUNT_LIMIT {
            assert!(matches!(
                result.as_deref(),
                Ok([LanguageStreamEvent::ProviderOpaque(_)])
            ));
        } else {
            let error = result.expect_err("aggregate opaque budget must fail before publication");
            assert_eq!(error.kind(), ErrorKind::Protocol);
        }
    }
}

#[test]
fn response_projects_refusal_and_preserves_current_usage_metadata() {
    let body = serde_json::to_vec(&json!({
        "id": "msg_refusal",
        "type": "message",
        "role": "assistant",
        "model": "claude-fable-5",
        "content": [{"type": "text", "text": "provisional text"}],
        "stop_reason": "refusal",
        "stop_sequence": null,
        "stop_details": {
            "type": "refusal",
            "category": "cyber",
            "explanation": "blocked by policy",
            "recommended_model": "claude-fable-5"
        },
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "output_tokens_details": {"thinking_tokens": 3, "future_tokens": 2},
            "cache_read_input_tokens": 4,
            "iterations": [{"type": "message", "input_tokens": 10, "output_tokens": 5}],
            "service_tier": "standard",
            "speed": "fast",
            "inference_geo": "us"
        },
        "container": {"id": "container-1"}
    }))
    .expect("serialize fixture");

    let response = decode_response(&body, &scope(), &model()).expect("decode response");

    assert_eq!(
        response.termination(),
        &LanguageTermination::Completed(LanguageCompletionReason::Refusal)
    );
    assert!(matches!(
        response.content(),
        [ContentPart::Refusal { reason: Some(reason) }] if reason == "blocked by policy"
    ));
    assert_eq!(response.usage().reasoning_tokens, UsageValue::Known(3));

    let metadata = response
        .provider_metadata()
        .get(PROTOCOL_ID)
        .and_then(serde_json::Value::as_object)
        .expect("Anthropic response metadata");
    assert_eq!(metadata["stop_details"]["category"], "cyber");
    assert_eq!(metadata["iterations"][0]["type"], "message");
    assert_eq!(
        metadata["usage"]["output_tokens_details"]["future_tokens"],
        2
    );
    assert_eq!(metadata["usage"]["service_tier"], "standard");
    assert_eq!(metadata["usage"]["speed"], "fast");
    assert_eq!(metadata["usage"]["inference_geo"], "us");
    assert_eq!(metadata["container"]["id"], "container-1");
}

#[test]
fn unknown_assigned_service_tier_remains_forward_compatible_metadata() {
    let body = serde_json::to_vec(&json!({
        "id": "msg_future_tier",
        "type": "message",
        "role": "assistant",
        "model": "claude-fable-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "service_tier": "future_tier"
        }
    }))
    .expect("serialize fixture");

    let response = decode_response(&body, &scope(), &model()).expect("decode response");
    let metadata = response
        .provider_metadata()
        .get(PROTOCOL_ID)
        .and_then(serde_json::Value::as_object)
        .expect("Anthropic response metadata");
    assert_eq!(metadata["usage"]["service_tier"], "future_tier");
}

#[test]
fn assigned_service_tier_uses_response_only_values() {
    let assigned = serde_json::from_value::<MessagesAssignedServiceTier>(json!("batch"))
        .expect("decode assigned tier");
    assert_eq!(assigned, MessagesAssignedServiceTier::Batch);
    assert!(
        serde_json::from_value::<MessagesAssignedServiceTier>(json!("auto")).is_err(),
        "request-only preferences must not decode as assigned tiers"
    );
}

#[test]
fn direct_compaction_is_incomplete_and_replays_only_in_the_same_scope() {
    let source_scope = scoped_replay_domain("workspace-a");
    let response = decode_response(
        &compaction_response_body("Keep the verified decisions."),
        &source_scope,
        &model(),
    )
    .expect("decode direct compaction");

    assert_eq!(
        response.termination(),
        &LanguageTermination::Incomplete(LanguageIncompleteReason::Other("compaction".to_string()))
    );
    let native = response
        .content()
        .iter()
        .find_map(|part| match part {
            ContentPart::ProviderOpaque(item) if item.data()["type"] == "compaction" => Some(item),
            _ => None,
        })
        .expect("retained compaction block");
    assert_eq!(native.data()["content"], "Keep the verified decisions.");
    assert_eq!(
        native.data()["encrypted_content"],
        "encrypted-compaction-state"
    );
    assert_eq!(response.usage().input_tokens, UsageValue::Known(150));
    assert_eq!(response.usage().output_tokens, UsageValue::Known(6));
    assert_eq!(response.usage().total_tokens, UsageValue::Known(156));
    assert_eq!(response.usage().reasoning_tokens, UsageValue::Known(0));
    assert_eq!(response.usage().cache_read_tokens, UsageValue::Known(5));
    assert_eq!(response.usage().cache_write_tokens, UsageValue::Known(7));

    let history = response
        .project_assistant_history()
        .into_message()
        .expect("compaction projects to assistant history");
    let mut continuation = LanguageRequest::new(vec![history, Message::user("Continue")]);
    continuation.generation.max_output_tokens = Some(2_048);
    let encoded = encode_request_for_scope(
        &source_scope,
        &model(),
        &continuation,
        &MessagesRequestOptions::default(),
    )
    .expect("replay compaction in the source scope");
    assert_eq!(
        encoded["messages"][0]["content"][0],
        json!({
            "type": "compaction",
            "content": "Keep the verified decisions.",
            "encrypted_content": "encrypted-compaction-state"
        })
    );

    let foreign_scope = scoped_replay_domain("workspace-b");
    assert!(matches!(
        encode_request_for_scope(
            &foreign_scope,
            &model(),
            &continuation,
            &MessagesRequestOptions::default(),
        ),
        Err(MessagesCodecError::Unsupported { .. })
    ));
}

#[test]
fn null_compaction_state_round_trips_only_in_the_same_scope() {
    let source_scope = scoped_replay_domain("workspace-a");
    let body = serde_json::to_vec(&json!({
        "id": "msg_failed_compaction",
        "type": "message",
        "role": "assistant",
        "model": "claude-fable-5",
        "content": [{"type": "compaction", "content": null, "encrypted_content": null}],
        "stop_reason": "compaction",
        "stop_sequence": null,
        "usage": {"input_tokens": 120, "output_tokens": 0}
    }))
    .expect("serialize failed compaction fixture");
    let response = decode_response(&body, &source_scope, &model())
        .expect("failed compaction remains an inspectable incomplete response");
    assert_eq!(
        response.termination(),
        &LanguageTermination::Incomplete(LanguageIncompleteReason::Other("compaction".to_string()))
    );
    let native = response
        .content()
        .iter()
        .find_map(|part| match part {
            ContentPart::ProviderOpaque(item) if item.data()["type"] == "compaction" => Some(item),
            _ => None,
        })
        .expect("null compaction block is retained");
    assert!(native.data()["content"].is_null());

    let history = response
        .project_assistant_history()
        .into_message()
        .expect("native failure remains inspectable in history projection");
    let mut continuation = LanguageRequest::new(vec![history, Message::user("Continue")]);
    continuation.generation.max_output_tokens = Some(2_048);
    let encoded = encode_request_for_scope(
        &source_scope,
        &model(),
        &continuation,
        &MessagesRequestOptions::default(),
    )
    .expect("replay null compaction state in the source scope");
    assert_eq!(
        encoded["messages"][0]["content"][0],
        json!({"type": "compaction", "content": null, "encrypted_content": null})
    );

    assert!(matches!(
        encode_request_for_scope(
            &scoped_replay_domain("workspace-b"),
            &model(),
            &continuation,
            &MessagesRequestOptions::default(),
        ),
        Err(MessagesCodecError::Unsupported { .. })
    ));
}

#[test]
fn streamed_compaction_delta_matches_direct_and_keeps_terminal_lifecycle_strict() {
    let source_scope = scoped_replay_domain("workspace-a");
    let direct = decode_response(
        &compaction_response_body("Keep the verified decisions."),
        &source_scope,
        &model(),
    )
    .expect("decode direct compaction");
    let mut decoder = MessagesStreamDecoder::new(source_scope, model());
    let frames = [
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_compaction",
                "type": "message",
                "role": "assistant",
                "model": "claude-fable-5",
                "usage": {"input_tokens": 120}
            }
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "compaction",
                "content": null,
                "encrypted_content": null
            }
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "compaction_delta"}
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "compaction_delta",
                "content": null,
                "encrypted_content": null
            }
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "compaction_delta",
                "content": "Keep the verified ",
                "encrypted_content": "superseded-encrypted-state"
            }
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "compaction_delta",
                "content": "decisions.",
                "encrypted_content": "encrypted-compaction-state"
            }
        }),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "compaction", "stop_sequence": null},
            "usage": {
                "input_tokens": 120,
                "output_tokens": 4,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "output_tokens_details": {"thinking_tokens": 0},
                "iterations": [
                    {
                        "type": "compaction",
                        "input_tokens": 30,
                        "output_tokens": 2,
                        "cache_read_input_tokens": 5,
                        "cache_creation_input_tokens": 7
                    },
                    {"type": "message", "input_tokens": 120, "output_tokens": 4}
                ]
            }
        }),
        json!({"type": "message_stop"}),
    ];
    let mut events = Vec::new();
    for frame in frames {
        events.extend(
            decoder
                .decode(&frame.to_string())
                .expect("decode compaction frame"),
        );
    }

    let opaque_events = events
        .iter()
        .filter_map(|event| match event {
            LanguageStreamEvent::ProviderOpaque(item) if item.data()["type"] == "compaction" => {
                Some(item)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(opaque_events.len(), 1, "compaction closes exactly once");
    assert_eq!(
        opaque_events[0].data()["content"],
        "Keep the verified decisions."
    );
    assert_eq!(
        opaque_events[0].data()["encrypted_content"],
        "encrypted-compaction-state"
    );

    let terminal = events
        .iter()
        .find_map(|event| match event {
            LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }) => Some(response),
            _ => None,
        })
        .expect("completed stream terminal");
    assert_eq!(terminal.termination(), direct.termination());
    assert_eq!(terminal.usage(), direct.usage());
    let terminal_compaction = terminal
        .content()
        .iter()
        .find_map(|part| match part {
            ContentPart::ProviderOpaque(item) if item.data()["type"] == "compaction" => Some(item),
            _ => None,
        })
        .expect("terminal retained compaction block");
    assert_eq!(terminal_compaction, opaque_events[0]);
    let direct_compaction = direct
        .content()
        .iter()
        .find_map(|part| match part {
            ContentPart::ProviderOpaque(item) if item.data()["type"] == "compaction" => Some(item),
            _ => None,
        })
        .expect("direct response retained compaction block");
    assert_eq!(terminal_compaction, direct_compaction);

    let duplicate_terminal = decoder
        .decode(&json!({"type": "message_stop"}).to_string())
        .expect_err("duplicate terminal frame must fail");
    assert_eq!(duplicate_terminal.kind(), ErrorKind::Protocol);
    let event_after_terminal = decoder
        .decode(&json!({"type": "ping"}).to_string())
        .expect_err("event after terminal must fail");
    assert_eq!(event_after_terminal.kind(), ErrorKind::Protocol);
    assert!(decoder.finish().expect("finish after terminal").is_empty());
    assert_eq!(
        decoder
            .finish()
            .expect_err("duplicate finish must fail")
            .kind(),
        ErrorKind::Protocol
    );

    let mut eof = MessagesStreamDecoder::new(scoped_replay_domain("workspace-a"), model());
    eof.decode(
        &json!({
            "type": "message_start",
            "message": {
                "id": "msg_compaction_eof",
                "type": "message",
                "role": "assistant",
                "model": "claude-fable-5",
                "usage": {"input_tokens": 120}
            }
        })
        .to_string(),
    )
    .expect("decode message start");
    eof.decode(
        &json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "compaction", "content": null}
        })
        .to_string(),
    )
    .expect("decode compaction start");
    assert_eq!(
        eof.finish()
            .expect_err("EOF before message_stop must fail")
            .kind(),
        ErrorKind::UnexpectedEof
    );
}

#[test]
fn streamed_null_encrypted_compaction_state_clears_and_replays_in_scope() {
    let source_scope = scoped_replay_domain("workspace-a");
    let mut decoder = MessagesStreamDecoder::new(source_scope.clone(), model());
    let frames = [
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_null_encrypted_compaction",
                "type": "message",
                "role": "assistant",
                "model": "claude-fable-5",
                "usage": {"input_tokens": 120}
            }
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "compaction",
                "content": null,
                "encrypted_content": null
            }
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "compaction_delta",
                "encrypted_content": "superseded-encrypted-state"
            }
        }),
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "compaction_delta", "encrypted_content": null}
        }),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "compaction", "stop_sequence": null},
            "usage": {"output_tokens": 0}
        }),
        json!({"type": "message_stop"}),
    ];

    let mut events = Vec::new();
    for frame in frames {
        events.extend(decoder.decode(&frame.to_string()).expect("decode frame"));
    }
    let terminal = events
        .iter()
        .find_map(|event| match event {
            LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }) => Some(response),
            _ => None,
        })
        .expect("completed stream terminal");
    let native = terminal
        .content()
        .iter()
        .find_map(|part| match part {
            ContentPart::ProviderOpaque(item) if item.data()["type"] == "compaction" => Some(item),
            _ => None,
        })
        .expect("retained compaction block");
    assert!(native.data()["content"].is_null());
    assert!(native.data()["encrypted_content"].is_null());

    let history = terminal
        .project_assistant_history()
        .into_message()
        .expect("project compaction history");
    let mut continuation = LanguageRequest::new(vec![history, Message::user("Continue")]);
    continuation.generation.max_output_tokens = Some(2_048);
    encode_request_for_scope(
        &source_scope,
        &model(),
        &continuation,
        &MessagesRequestOptions::default(),
    )
    .expect("replay streamed null compaction state in source scope");
    assert!(matches!(
        encode_request_for_scope(
            &scoped_replay_domain("workspace-b"),
            &model(),
            &continuation,
            &MessagesRequestOptions::default(),
        ),
        Err(MessagesCodecError::Unsupported { .. })
    ));
}

#[test]
fn streamed_compaction_content_is_bounded_before_native_retention() {
    let mut decoder = MessagesStreamDecoder::new(scoped_replay_domain("workspace-a"), model());
    decoder
        .decode(
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_compaction_bounded",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-fable-5",
                    "usage": {}
                }
            })
            .to_string(),
        )
        .expect("decode message start");
    decoder
        .decode(
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "compaction", "content": null}
            })
            .to_string(),
        )
        .expect("decode compaction start");

    let oversized = json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "compaction_delta",
            "content": "x".repeat(siumai_core::DEFAULT_OPAQUE_ITEM_LIMIT + 1)
        }
    })
    .to_string();
    assert_eq!(
        decoder
            .decode(&oversized)
            .expect_err("oversized compaction delta must fail")
            .kind(),
        ErrorKind::Protocol
    );
}

#[test]
fn streamed_refusal_clears_provisional_content_from_terminal_response() {
    let mut decoder = MessagesStreamDecoder::new(scope(), model());
    let frames = [
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_stream_refusal",
                "type": "message",
                "role": "assistant",
                "model": "claude-fable-5",
                "usage": {"input_tokens": 10}
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
            "delta": {"type": "text_delta", "text": "provisional text"}
        }),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "refusal",
                "stop_sequence": null,
                "stop_details": {"type": "refusal", "explanation": "blocked"}
            },
            "usage": {
                "output_tokens": 5,
                "output_tokens_details": {"thinking_tokens": 2}
            }
        }),
        json!({"type": "message_stop"}),
    ];

    let mut events = Vec::new();
    for frame in frames {
        events.extend(decoder.decode(&frame.to_string()).expect("decode frame"));
    }

    assert!(events.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::Refusal { reason: Some(reason) } if reason == "blocked"
    )));
    let terminal = events
        .iter()
        .find_map(|event| match event {
            LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }) => Some(response),
            _ => None,
        })
        .expect("completed terminal");
    assert_eq!(
        terminal.termination(),
        &LanguageTermination::Completed(LanguageCompletionReason::Refusal)
    );
    assert!(matches!(
        terminal.content(),
        [ContentPart::Refusal { reason: Some(reason) }] if reason == "blocked"
    ));
    assert_eq!(terminal.usage().reasoning_tokens, UsageValue::Known(2));
}

#[test]
fn streamed_fallback_updates_served_model_and_keeps_iteration_metadata() {
    let mut decoder = MessagesStreamDecoder::new(scope(), model());
    let frames = [
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_stream_fallback",
                "type": "message",
                "role": "assistant",
                "model": "claude-fable-5",
                "usage": {"input_tokens": 408}
            }
        }),
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "fallback",
                "from": {"model": "claude-fable-5"},
                "to": {"model": "claude-opus-5"}
            }
        }),
        json!({"type": "content_block_stop", "index": 0}),
        json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""}
        }),
        json!({
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "served"}
        }),
        json!({"type": "content_block_stop", "index": 1}),
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {
                "input_tokens": 412,
                "output_tokens": 264,
                "speed": "standard",
                "inference_geo": "global",
                "iterations": [
                    {"type": "message", "model": "claude-fable-5", "input_tokens": 408, "output_tokens": 0},
                    {"type": "fallback_message", "model": "claude-opus-5", "input_tokens": 412, "output_tokens": 264}
                ]
            }
        }),
        json!({"type": "message_stop"}),
    ];

    let mut events = Vec::new();
    for frame in frames {
        events.extend(decoder.decode(&frame.to_string()).expect("decode frame"));
    }
    let terminal = events
        .iter()
        .find_map(|event| match event {
            LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }) => Some(response),
            _ => None,
        })
        .expect("completed terminal");
    assert_eq!(terminal.model().map(ModelId::as_str), Some("claude-opus-5"));
    assert_eq!(terminal.usage().input_tokens, UsageValue::Known(412));
    assert_eq!(terminal.usage().output_tokens, UsageValue::Known(264));
    assert_eq!(terminal.usage().total_tokens, UsageValue::Known(676));
    assert!(terminal.content().iter().any(|part| matches!(
        part,
        ContentPart::ProviderOpaque(item) if item.data()["type"] == "fallback"
    )));
    let metadata = terminal
        .provider_metadata()
        .get(PROTOCOL_ID)
        .and_then(serde_json::Value::as_object)
        .expect("Anthropic response metadata");
    assert_eq!(metadata["iterations"][1]["type"], "fallback_message");
    assert_eq!(metadata["usage"]["speed"], "standard");
    assert_eq!(metadata["usage"]["inference_geo"], "global");
}

#[test]
fn advisor_iterations_do_not_duplicate_top_level_usage() {
    let body = serde_json::to_vec(&json!({
        "id": "msg_advisor_usage",
        "type": "message",
        "role": "assistant",
        "model": "claude-fable-5",
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 20,
            "output_tokens": 3,
            "iterations": [
                {"type": "advisor_message", "input_tokens": 7, "output_tokens": 2},
                {"type": "message", "input_tokens": 20, "output_tokens": 3}
            ]
        }
    }))
    .expect("serialize advisor usage fixture");

    let response = decode_response(&body, &scope(), &model()).expect("decode response");
    assert_eq!(response.usage().input_tokens, UsageValue::Known(20));
    assert_eq!(response.usage().output_tokens, UsageValue::Known(3));
    assert_eq!(response.usage().total_tokens, UsageValue::Known(23));
}

#[test]
fn malformed_stop_details_are_rejected() {
    let body = json!({
        "id": "msg_bad_stop_details",
        "type": "message",
        "role": "assistant",
        "model": "claude-fable-5",
        "content": [],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "stop_details": "not-an-object",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let error = decode_response(
        &serde_json::to_vec(&body).expect("serialize fixture"),
        &scope(),
        &model(),
    )
    .expect_err("malformed stop details must fail");
    assert_eq!(error.error_kind(), ErrorKind::Protocol);
}
