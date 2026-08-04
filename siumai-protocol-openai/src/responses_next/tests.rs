use std::collections::BTreeMap;

use serde_json::{Value, json};
use siumai_core::{
    ApiModeId, ContentPart, ErrorKind, FinishReason, LanguageIncompleteReason, LanguageRequest,
    LanguageResponseStatus, LanguageStreamEvent, MediaData, MediaPart, Message, MessageRole,
    ModelId, PlatformId, ProtocolId, ProviderId, ProviderScope, StreamTerminal,
    StructuredOutputSpec, ToolOutcome, ToolResult, ToolSpec, UsageValue,
};

use super::*;

fn scope() -> ProviderScope {
    ProviderScope::new(ProviderId::new("openai").unwrap())
        .with_platform(PlatformId::new("public-api").unwrap())
        .with_protocol(ProtocolId::new(OPENAI_RESPONSES_PROTOCOL).unwrap())
        .with_api_mode(ApiModeId::new("responses").unwrap())
}

fn model() -> ModelId {
    ModelId::new("gpt-5.6").unwrap()
}

fn fidelity_response() -> Value {
    json!({
        "id": "resp_fidelity",
        "object": "response",
        "created_at": 1785811200,
        "model": "gpt-5.6",
        "status": "completed",
        "output": [
            {
                "id": "rs_1",
                "type": "reasoning",
                "encrypted_content": "encrypted-reasoning",
                "summary": [{"type": "summary_text", "text": "checked inventory"}],
                "content": []
            },
            {
                "id": "cm_1",
                "type": "program",
                "call_id": "call_program",
                "code": "const inventory = await tools.inventory({ sku: '1' });",
                "fingerprint": "program-fingerprint"
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "status": "in_progress",
                "call_id": "call_inventory",
                "name": "inventory",
                "namespace": "warehouse",
                "arguments": "{\"sku\":\"1\"}",
                "caller": {"type": "program", "caller_id": "call_program"}
            },
            {
                "id": "cmo_1",
                "type": "program_output",
                "call_id": "call_program",
                "result": "{\"available\":42}",
                "status": "completed"
            },
            {
                "id": "ws_1",
                "type": "web_search_call",
                "status": "completed",
                "action": {"type": "search", "query": "inventory docs"}
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": "final_answer",
                "content": [
                    {
                        "type": "output_text",
                        "text": "There are 42 units.",
                        "annotations": [{
                            "type": "url_citation",
                            "url": "https://example.com/inventory",
                            "title": "Inventory",
                            "start_index": 10,
                            "end_index": 18
                        }]
                    },
                    {"type": "refusal", "refusal": "restricted detail omitted"}
                ]
            }
        ],
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {
                "cached_tokens": 80,
                "cache_write_tokens": 10,
                "orchestration_input_tokens": 3,
                "orchestration_input_cached_tokens": null
            },
            "output_tokens": 25,
            "output_tokens_details": {
                "reasoning_tokens": 7,
                "orchestration_output_tokens": 2
            },
            "total_tokens": 125
        },
        "error": null,
        "incomplete_details": null,
        "reasoning": {
            "effort": "medium",
            "mode": "standard",
            "context": "all_turns",
            "summary": null
        }
    })
}

#[test]
fn non_streaming_decode_preserves_native_items_identity_citations_and_usage() {
    let fixture = fidelity_response();
    let body = serde_json::to_vec(&fixture).unwrap();
    let decoded = decode_response(&body, &scope(), &model()).unwrap();

    assert_eq!(decoded.status(), &ResponseStatus::Completed);
    assert_eq!(decoded.canonical().id(), Some("resp_fidelity"));
    assert!(matches!(
        decoded.canonical().status(),
        LanguageResponseStatus::Completed
    ));
    assert_eq!(
        decoded.canonical().usage().cache_read_tokens,
        UsageValue::Known(80)
    );
    assert_eq!(
        decoded.canonical().usage().cache_write_tokens,
        UsageValue::Known(10)
    );
    assert_eq!(
        decoded.canonical().usage().reasoning_tokens,
        UsageValue::Known(7)
    );
    assert_eq!(
        decoded.canonical().usage().orchestration_tokens,
        UsageValue::Known(5)
    );
    assert!(decoded.canonical().content().iter().any(|part| matches!(
        part,
        ContentPart::Citation(citation)
            if citation.url.as_deref() == Some("https://example.com/inventory")
    )));
    assert!(decoded.canonical().content().iter().any(|part| matches!(
        part,
        ContentPart::Refusal { reason: Some(reason) }
            if reason == "restricted detail omitted"
    )));

    let opaque = decoded
        .canonical()
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::ProviderOpaque(item) => Some(item),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(opaque.len(), fixture["output"].as_array().unwrap().len());
    let function = opaque
        .iter()
        .find(|item| item.item_id() == Some("fc_1"))
        .unwrap();
    assert!(
        function.relations().iter().any(|relation| {
            relation.kind() == "call" && relation.target_id() == "call_inventory"
        })
    );
    assert!(
        function.relations().iter().any(|relation| {
            relation.kind() == "caller" && relation.target_id() == "call_program"
        })
    );
    assert_eq!(function.data()["caller"]["caller_id"], "call_program");

    assert_eq!(serde_json::to_value(decoded.native()).unwrap(), fixture);
}

#[test]
fn encrypted_reasoning_larger_than_legacy_limit_remains_replayable() {
    let mut response = fidelity_response();
    response["output"][0]["encrypted_content"] = Value::String("e".repeat(128 * 1024));
    let body = serde_json::to_vec(&response).unwrap();

    let decoded = decode_response(&body, &scope(), &model()).unwrap();
    let native = decoded
        .canonical()
        .content()
        .iter()
        .find_map(|part| match part {
            ContentPart::ProviderOpaque(item) if item.item_id() == Some("rs_1") => Some(item),
            _ => None,
        })
        .unwrap();

    assert!(native.encoded_json_bytes() > 64 * 1024);
    assert!(native.encoded_json_bytes() < siumai_core::DEFAULT_OPAQUE_ITEM_LIMIT);
}

#[test]
fn request_replays_native_program_history_and_copies_caller_to_tool_output() {
    let decoded = decode_response(
        &serde_json::to_vec(&fidelity_response()).unwrap(),
        &scope(),
        &model(),
    )
    .unwrap();
    let user = Message::text(MessageRole::User, "Check inventory");
    let assistant = Message {
        role: MessageRole::Assistant,
        content: decoded.canonical().content().to_vec(),
    };
    let tool = Message {
        role: MessageRole::Tool,
        content: vec![ContentPart::ToolResult(ToolResult {
            call_id: "call_inventory".to_string(),
            name: "inventory".to_string(),
            outcome: ToolOutcome::Success {
                value: json!({"available": 42}),
            },
        })],
    };
    let mut request = LanguageRequest::new(vec![user, assistant, tool]);
    request.structured_output = Some(StructuredOutputSpec {
        name: "inventory_result".to_string(),
        description: None,
        schema: json!({"type": "object"}),
        strict: true,
    });
    let options = RequestEncodingOptions::new(false)
        .with_extra(BTreeMap::from([
            (
                "prompt_cache_options".to_string(),
                json!({"mode": "explicit", "ttl": "30m"}),
            ),
            (TEXT_VERBOSITY_OPTION.to_string(), json!("low")),
        ]))
        .with_native_tool(json!({"type": "web_search"}))
        .with_prompt_cache_breakpoint(PromptCacheBlock::new(0, 0));

    let body = encode_request_with_options(&scope(), &model(), &request, &options).unwrap();
    assert_eq!(
        body["input"][0]["content"][0]["prompt_cache_breakpoint"],
        json!({"mode": "explicit"})
    );
    assert_eq!(body["prompt_cache_options"]["ttl"], "30m");
    assert_eq!(body["text"]["verbosity"], "low");
    assert_eq!(body["text"]["format"]["type"], "json_schema");
    assert_eq!(body["tools"].as_array().unwrap().len(), 1);

    let input = body["input"].as_array().unwrap();
    let fidelity = fidelity_response();
    for native in fidelity["output"].as_array().unwrap() {
        assert!(
            input.contains(native),
            "native item was not replayed: {native}"
        );
    }
    let output = input
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .unwrap();
    assert_eq!(output["call_id"], "call_inventory");
    assert_eq!(output["caller"]["type"], "program");
    assert_eq!(output["caller"]["caller_id"], "call_program");
    assert!(output.get("name").is_none());
    assert!(output.get("namespace").is_none());
}

#[test]
fn opaque_history_cannot_cross_provider_platform_boundaries() {
    let decoded = decode_response(
        &serde_json::to_vec(&fidelity_response()).unwrap(),
        &scope(),
        &model(),
    )
    .unwrap();
    let request = LanguageRequest::new(vec![Message {
        role: MessageRole::Assistant,
        content: decoded.canonical().content().to_vec(),
    }]);
    let azure_scope = ProviderScope::new(ProviderId::new("openai").unwrap())
        .with_platform(PlatformId::new("azure").unwrap())
        .with_protocol(ProtocolId::new(OPENAI_RESPONSES_PROTOCOL).unwrap())
        .with_api_mode(ApiModeId::new("responses").unwrap());

    let error =
        encode_request(&azure_scope, &model(), &request, false, &BTreeMap::new()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
}

#[test]
fn protected_fields_cannot_be_overridden_by_extra_options() {
    let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]);
    let extra = BTreeMap::from([("input".to_string(), json!([]))]);
    let error = encode_request(&scope(), &model(), &request, false, &extra).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(is_protected_option_field("tools"));
    assert!(!is_protected_option_field("reasoning"));
}

#[test]
fn failed_and_cancelled_non_streaming_resources_remain_canonical_responses() {
    for status in ["failed", "cancelled"] {
        let response = json!({
            "id": format!("resp_{status}"),
            "created_at": 1785811200,
            "model": "gpt-5.6",
            "status": status,
            "output": [],
            "usage": null,
            "error": if status == "failed" {
                json!({"code": "server_error", "message": "provider detail"})
            } else {
                Value::Null
            },
            "incomplete_details": null,
            "reasoning": null
        });
        let canonical =
            decode_response(&serde_json::to_vec(&response).unwrap(), &scope(), &model())
                .unwrap()
                .into_result()
                .unwrap();
        assert!(matches!(
            (status, canonical.status(), canonical.finish_reason()),
            (
                "failed",
                LanguageResponseStatus::Failed,
                FinishReason::Error
            ) | (
                "cancelled",
                LanguageResponseStatus::Cancelled,
                FinishReason::Cancelled
            )
        ));
    }
}

#[test]
fn background_resource_decode_remains_native_until_it_is_terminal() {
    let response = json!({
        "id": "resp_background",
        "created_at": 1785811200,
        "model": "gpt-5.6",
        "status": "queued",
        "output": [],
        "usage": null,
        "error": null,
        "incomplete_details": null,
        "reasoning": null
    });
    let body = serde_json::to_vec(&response).unwrap();
    let native = decode_response_resource(&body).unwrap();
    assert_eq!(native.status, ResponseStatus::Queued);
    assert_eq!(
        decode_response(&body, &scope(), &model())
            .unwrap_err()
            .kind(),
        ErrorKind::Protocol
    );
}

#[test]
fn explicit_prompt_cache_breakpoints_cover_text_image_and_file_blocks() {
    let request = LanguageRequest::new(vec![Message {
        role: MessageRole::User,
        content: vec![
            ContentPart::Text {
                text: "inspect these inputs".to_string(),
            },
            ContentPart::Media(MediaPart {
                media_type: "image/png".to_string(),
                data: MediaData::Url("https://example.com/input.png".to_string()),
                name: None,
            }),
            ContentPart::Media(MediaPart {
                media_type: "application/pdf".to_string(),
                data: MediaData::Url("https://example.com/input.pdf".to_string()),
                name: Some("input.pdf".to_string()),
            }),
        ],
    }]);
    let options = RequestEncodingOptions::new(false)
        .with_prompt_cache_breakpoint(PromptCacheBlock::new(0, 0))
        .with_prompt_cache_breakpoint(PromptCacheBlock::new(0, 1))
        .with_prompt_cache_breakpoint(PromptCacheBlock::new(0, 2));

    let body = encode_request_with_options(&scope(), &model(), &request, &options).unwrap();
    let blocks = body["input"][0]["content"].as_array().unwrap();
    assert_eq!(blocks[0]["type"], "input_text");
    assert_eq!(blocks[1]["type"], "input_image");
    assert_eq!(blocks[2]["type"], "input_file");
    assert!(
        blocks
            .iter()
            .all(|block| { block["prompt_cache_breakpoint"] == json!({"mode": "explicit"}) })
    );
}

#[test]
fn explicit_prompt_cache_breakpoints_are_bounded() {
    let request = LanguageRequest::new(vec![Message {
        role: MessageRole::User,
        content: (0..51)
            .map(|index| ContentPart::Text {
                text: format!("cache block {index}"),
            })
            .collect(),
    }]);
    let mut options = RequestEncodingOptions::new(false);
    for content_index in 0..51 {
        options = options.with_prompt_cache_breakpoint(PromptCacheBlock::new(0, content_index));
    }

    let error = encode_request_with_options(&scope(), &model(), &request, &options).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
}

#[test]
fn function_tool_options_enable_programmatic_callers_losslessly() {
    let mut request = LanguageRequest::new(vec![Message::text(MessageRole::User, "inventory")]);
    request.tools.push(
        ToolSpec::new(
            "get_inventory",
            Some("Read inventory".to_string()),
            json!({
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"]
            }),
        )
        .unwrap(),
    );
    let options = RequestEncodingOptions::new(false).with_function_tool_options(
        "get_inventory",
        FunctionToolEncodingOptions::default()
            .with_strict(true)
            .with_defer_loading(true)
            .with_allowed_caller(FunctionToolCaller::Programmatic)
            .with_output_schema(json!({"type": "object"})),
    );

    let body = encode_request_with_options(&scope(), &model(), &request, &options).unwrap();
    assert_eq!(body["tools"][0]["strict"], true);
    assert_eq!(body["tools"][0]["defer_loading"], true);
    assert_eq!(body["tools"][0]["allowed_callers"], json!(["programmatic"]));
    assert_eq!(body["tools"][0]["output_schema"], json!({"type": "object"}));
}

#[test]
fn function_tool_options_reject_unknown_tool_names() {
    let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "inventory")]);
    let options = RequestEncodingOptions::new(false).with_function_tool_options(
        "missing_tool",
        FunctionToolEncodingOptions::default()
            .with_allowed_caller(FunctionToolCaller::Programmatic),
    );

    let error = encode_request_with_options(&scope(), &model(), &request, &options).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
}

#[test]
fn stream_waits_for_complete_tool_json_and_emits_one_terminal() {
    let mut decoder = ResponsesStreamDecoder::new(scope(), model());
    decoder
        .decode(
            &json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": progress_response("in_progress")
            })
            .to_string(),
        )
        .unwrap();
    let added = decoder
        .decode(
            &json!({
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "id": "fc_stream",
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": "call_stream",
                    "name": "lookup",
                    "arguments": ""
                }
            })
            .to_string(),
        )
        .unwrap();
    assert!(added.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::ToolInputStart { id, .. } if id == "call_stream"
    )));

    let first_half = decoder
        .decode(
            &json!({
                "type": "response.function_call_arguments.delta",
                "sequence_number": 2,
                "item_id": "fc_stream",
                "output_index": 0,
                "delta": "{\"q\":"
            })
            .to_string(),
        )
        .unwrap();
    assert!(
        !first_half
            .iter()
            .any(|event| matches!(event, LanguageStreamEvent::ToolCall(_)))
    );
    decoder
        .decode(
            &json!({
                "type": "response.function_call_arguments.delta",
                "sequence_number": 3,
                "item_id": "fc_stream",
                "output_index": 0,
                "delta": "\"tea\"}"
            })
            .to_string(),
        )
        .unwrap();
    decoder
        .decode(
            &json!({
                "type": "response.function_call_arguments.done",
                "sequence_number": 4,
                "item_id": "fc_stream",
                "output_index": 0,
                "arguments": "{\"q\":\"tea\"}"
            })
            .to_string(),
        )
        .unwrap();
    let done = decoder
        .decode(
            &json!({
                "type": "response.output_item.done",
                "sequence_number": 5,
                "output_index": 0,
                "item": {
                    "id": "fc_stream",
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": "call_stream",
                    "name": "lookup",
                    "arguments": "{\"q\":\"tea\"}"
                }
            })
            .to_string(),
        )
        .unwrap();
    assert!(done.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::ToolCall(call)
            if call.id == "call_stream" && call.arguments == json!({"q": "tea"})
    )));
    assert!(done.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::ProviderOpaque(item) if item.item_id() == Some("fc_stream")
    )));

    let terminal = decoder
        .decode(
            &json!({
                "type": "response.completed",
                "sequence_number": 6,
                "response": {
                    "id": "resp_stream",
                    "created_at": 1785811200,
                    "model": "gpt-5.6",
                    "status": "completed",
                    "output": [{
                        "id": "fc_stream",
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": "call_stream",
                        "name": "lookup",
                        "arguments": "{\"q\":\"tea\"}"
                    }],
                    "usage": {
                        "input_tokens": 0,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 2,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 2
                    },
                    "error": null,
                    "incomplete_details": null,
                    "reasoning": null
                }
            })
            .to_string(),
        )
        .unwrap();
    assert_eq!(
        terminal
            .iter()
            .filter(|event| event.terminal().is_some())
            .count(),
        1
    );
    assert!(terminal.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::Usage(usage) if usage.input_tokens == UsageValue::Known(0)
    )));
    assert!(matches!(
        terminal.last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }))
            if response.finish_reason() == &FinishReason::ToolCalls
    ));
    assert!(decoder.decode("{}").is_err());
    assert!(decoder.finish().unwrap().is_empty());
    assert!(decoder.finish().is_err());
}

#[test]
fn usage_only_terminal_still_emits_started_usage_and_one_terminal() {
    let mut decoder = ResponsesStreamDecoder::new(scope(), model());
    let events = decoder
        .decode(
            &json!({
                "type": "response.completed",
                "sequence_number": 0,
                "response": {
                    "id": "resp_usage_only",
                    "created_at": 1785811200,
                    "model": "gpt-5.6",
                    "status": "completed",
                    "output": [],
                    "usage": {
                        "input_tokens": 0,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 0,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 0
                    },
                    "error": null,
                    "incomplete_details": null,
                    "reasoning": null
                }
            })
            .to_string(),
        )
        .unwrap();
    assert!(matches!(
        events.first(),
        Some(LanguageStreamEvent::Started { id: Some(id), .. }) if id == "resp_usage_only"
    ));
    assert!(events.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::Usage(usage)
            if usage.total_tokens == UsageValue::Known(0)
    )));
    assert_eq!(
        events
            .iter()
            .filter(|event| event.terminal().is_some())
            .count(),
        1
    );
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }))
            if response.content().is_empty()
    ));
}

#[test]
fn terminal_status_matrix_preserves_partial_failed_and_cancelled_responses() {
    let mut incomplete = ResponsesStreamDecoder::new(scope(), model());
    let events = incomplete
        .decode(&terminal_frame(
            "response.incomplete",
            "incomplete",
            json!({
                "reason": "max_output_tokens"
            }),
            Value::Null,
        ))
        .unwrap();
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }))
            if matches!(
                response.status(),
                LanguageResponseStatus::Incomplete {
                    reason: Some(LanguageIncompleteReason::MaxOutputTokens)
                }
            ) && response.finish_reason() == &FinishReason::Length
    ));

    let mut cancelled = ResponsesStreamDecoder::new(scope(), model());
    let events = cancelled
        .decode(&terminal_frame(
            "response.cancelled",
            "cancelled",
            Value::Null,
            Value::Null,
        ))
        .unwrap();
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
            response: Some(response), ..
        })) if matches!(response.status(), LanguageResponseStatus::Cancelled)
    ));

    let mut failed = ResponsesStreamDecoder::new(scope(), model());
    let events = failed
        .decode(&terminal_frame(
            "response.failed",
            "failed",
            Value::Null,
            json!({"code": "server_error", "message": "sensitive provider message"}),
        ))
        .unwrap();
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Failed {
            response: Some(response), error
        })) if matches!(response.status(), LanguageResponseStatus::Failed)
            && error.message() == "OpenAI Responses generation failed"
    ));
}

#[test]
fn early_error_and_eof_are_terminal_failures_not_success() {
    let mut early_error = ResponsesStreamDecoder::new(scope(), model());
    let events = early_error
        .decode(
            &json!({
                "type": "error",
                "sequence_number": 0,
                "code": "rate_limit_exceeded",
                "message": "provider detail",
                "param": null
            })
            .to_string(),
        )
        .unwrap();
    assert!(matches!(
        events.as_slice(),
        [LanguageStreamEvent::Terminal(StreamTerminal::Failed {
            response: None,
            ..
        })]
    ));
    assert!(early_error.decode("{}").is_err());

    let mut eof = ResponsesStreamDecoder::new(scope(), model());
    eof.decode(
        &json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": progress_response("in_progress")
        })
        .to_string(),
    )
    .unwrap();
    let error = eof.finish().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::UnexpectedEof);
    assert!(!eof.terminal_seen());
    assert!(eof.finish().is_err());
}

#[test]
fn no_output_response_and_unknown_provider_tool_remain_valid() {
    let response = json!({
        "id": "resp_empty",
        "created_at": 1785811200,
        "model": "gpt-5.6",
        "status": "completed",
        "output": [],
        "usage": null,
        "error": null,
        "incomplete_details": null,
        "reasoning": null
    });
    let decoded =
        decode_response(&serde_json::to_vec(&response).unwrap(), &scope(), &model()).unwrap();
    assert!(decoded.canonical().content().is_empty());
    assert_eq!(decoded.canonical().finish_reason(), &FinishReason::Stop);

    let future: OutputItem = serde_json::from_value(json!({
        "id": "future_1",
        "type": "future_provider_tool_call",
        "status": "completed",
        "payload": {"new": true}
    }))
    .unwrap();
    assert!(matches!(&future, OutputItem::Unknown(_)));
    assert_eq!(future.status(), Some("completed"));
    assert_eq!(
        future.to_value().unwrap()["payload"]["new"],
        Value::Bool(true)
    );
}

#[test]
fn incomplete_program_output_is_not_projected_as_success() {
    let response = json!({
        "id": "resp_program_incomplete",
        "created_at": 1785811200,
        "model": "gpt-5.6",
        "status": "completed",
        "output": [{
            "id": "cmo_incomplete",
            "type": "program_output",
            "call_id": "call_program",
            "result": "partial output",
            "status": "incomplete"
        }],
        "usage": null,
        "error": null,
        "incomplete_details": null,
        "reasoning": null
    });
    let decoded =
        decode_response(&serde_json::to_vec(&response).unwrap(), &scope(), &model()).unwrap();
    assert!(decoded.canonical().content().iter().any(|part| matches!(
        part,
        ContentPart::ToolResult(ToolResult {
            outcome: ToolOutcome::ExecutionFailed { .. },
            ..
        })
    )));
    assert!(decoded.canonical().content().iter().any(|part| matches!(
        part,
        ContentPart::ProviderOpaque(item) if item.item_id() == Some("cmo_incomplete")
    )));
}

fn progress_response(status: &str) -> Value {
    json!({
        "id": "resp_stream",
        "created_at": 1785811200,
        "model": "gpt-5.6",
        "status": status,
        "output": [],
        "usage": null,
        "error": null,
        "incomplete_details": null,
        "reasoning": null
    })
}

#[test]
fn repository_response_fixtures_round_trip_native_items_losslessly() {
    for (name, body) in [
        (
            "encrypted reasoning",
            include_str!(
                "../../../siumai/tests/fixtures/openai/responses/response/reasoning-encrypted-content.1/response.json"
            ),
        ),
        (
            "web search",
            include_str!(
                "../../../siumai/tests/fixtures/openai/responses/response/web-search-tool.1/response.json"
            ),
        ),
        (
            "apply patch",
            include_str!(
                "../../../siumai/tests/fixtures/openai/responses/response/apply-patch-tool.1/response.json"
            ),
        ),
    ] {
        let original = serde_json::from_str::<Value>(body).unwrap();
        let requested_model = ModelId::new(original["model"].as_str().unwrap()).unwrap();
        let decoded = decode_response(body.as_bytes(), &scope(), &requested_model)
            .unwrap_or_else(|error| panic!("{name} fixture failed: {error}"));
        let (native, canonical) = decoded.into_parts();
        let round_trip = serde_json::to_value(&native).unwrap();

        assert_eq!(round_trip["output"], original["output"], "{name}");
        assert!(canonical.content().iter().any(|part| {
            matches!(part, ContentPart::ProviderOpaque(item) if item.kind() == OPENAI_RESPONSES_OPAQUE_KIND)
        }));
    }
}

fn terminal_frame(
    event_type: &str,
    status: &str,
    incomplete_details: Value,
    error: Value,
) -> String {
    json!({
        "type": event_type,
        "sequence_number": 0,
        "response": {
            "id": format!("resp_{status}"),
            "created_at": 1785811200,
            "model": "gpt-5.6",
            "status": status,
            "output": [],
            "usage": null,
            "error": error,
            "incomplete_details": incomplete_details,
            "reasoning": null
        }
    })
    .to_string()
}
