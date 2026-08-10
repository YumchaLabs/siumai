use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use siumai_core::{
    ApiModeId, ContentAnnotationTarget, ContentAnnotations, ContentPart, Error, ErrorKind,
    FinishReason, LanguageIncompleteReason, LanguageRequest, LanguageResponseStatus,
    LanguageStreamEvent, MediaData, MediaPart, Message, MessagePart, MessageRole, ModelId,
    PlatformId, ProtocolId, ProviderId, ProviderScope, ReplayDomain, ReplayDomainId,
    ResponseDiagnostics, StreamTerminal, StructuredOutputSpec, ToolCall, ToolOutcome, ToolResult,
    ToolSpec, TypedProviderAnnotation, UsageValue,
};

use crate::{PromptCacheAnnotationResolver, PromptCacheNodeOptions};

use super::*;

fn scope() -> ProviderScope {
    ProviderScope::new(ProviderId::new("openai").unwrap())
        .with_platform(PlatformId::new("public-api").unwrap())
        .with_protocol(ProtocolId::new(OPENAI_RESPONSES_PROTOCOL).unwrap())
        .with_api_mode(ApiModeId::new("responses").unwrap())
        .with_replay_domain(ReplayDomain::official(
            ReplayDomainId::new("openai-test").unwrap(),
        ))
}

fn model() -> ModelId {
    ModelId::new("gpt-5.6").unwrap()
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestPromptCacheAnnotation {
    marked: bool,
}

impl TypedProviderAnnotation for TestPromptCacheAnnotation {
    type Target = ContentAnnotationTarget;

    const NAMESPACE: &'static str = "test-openai-cache";
}

#[derive(Debug, Default)]
struct TestPromptCacheResolver;

impl PromptCacheAnnotationResolver for TestPromptCacheResolver {
    fn resolve_content(
        &self,
        annotations: &ContentAnnotations,
    ) -> Result<PromptCacheNodeOptions, Error> {
        annotations
            .decode::<TestPromptCacheAnnotation>()
            .map(|annotation| {
                PromptCacheNodeOptions::new()
                    .with_explicit_breakpoint(annotation.is_some_and(|value| value.marked))
            })
            .map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "invalid test prompt-cache annotation",
                )
                .with_source(source)
            })
    }
}

fn annotated_part(content: ContentPart) -> MessagePart {
    let annotation = TestPromptCacheAnnotation { marked: true };
    MessagePart::from_parts(
        content,
        ContentAnnotations::default()
            .with(&annotation)
            .expect("test annotation"),
    )
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

fn opaque_output_part(value: Value) -> ContentPart {
    let item = serde_json::from_value::<OutputItem>(value).expect("native output item");
    ContentPart::ProviderOpaque(
        super::response::opaque_item(&item, &scope(), &model()).expect("opaque output item"),
    )
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
    let local_calls = decoded
        .canonical()
        .content()
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(local_calls.len(), 1);
    assert_eq!(local_calls[0].id(), "call_inventory");
    assert_eq!(local_calls[0].name(), "inventory");

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
fn direct_function_input_is_bounded_before_json_normalization() {
    let mut response = fidelity_response();
    response["output"][2]["arguments"] = Value::String(format!(
        "{}{{}}",
        " ".repeat(siumai_core::DEFAULT_TOOL_INPUT_BYTE_LIMIT)
    ));
    let body = serde_json::to_vec(&response).unwrap();

    let error = decode_response(&body, &scope(), &model()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResponseLimit);
}

#[test]
fn request_replays_native_program_history_and_copies_caller_to_tool_output() {
    let decoded = decode_response(
        &serde_json::to_vec(&fidelity_response()).unwrap(),
        &scope(),
        &model(),
    )
    .unwrap();
    let user = Message::new(
        MessageRole::User,
        [annotated_part(ContentPart::Text {
            text: "Check inventory".to_string(),
        })],
    );
    let projection = decoded.canonical().project_assistant_history();
    assert_eq!(projection.omissions().len(), 2);
    let assistant = projection
        .into_message()
        .expect("native replay items remain in assistant history");
    let tool = Message::new(
        MessageRole::Tool,
        [ContentPart::ToolResult(ToolResult {
            call_id: "call_inventory".to_string(),
            name: "inventory".to_string(),
            outcome: ToolOutcome::Success {
                value: json!({"available": 42}),
            },
        })],
    );
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
        .with_native_tool(json!({"type": "web_search"}));

    let body = encode_request_with_options_and_resolver(
        &scope(),
        &model(),
        &request,
        &options,
        &TestPromptCacheResolver,
    )
    .unwrap();
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
fn request_suppresses_only_semantically_equal_native_projection_siblings() {
    let assistant = Message::new(
        MessageRole::Assistant,
        [
            ContentPart::Text {
                text: "same answer".to_string(),
            },
            opaque_output_part(json!({
                "id": "msg_equal",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "same answer"}]
            })),
            ContentPart::Reasoning {
                text: "same reasoning".to_string(),
            },
            opaque_output_part(json!({
                "id": "reason_equal",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "same reasoning"}],
                "content": []
            })),
            opaque_output_part(json!({
                "id": "program_parent",
                "type": "program",
                "call_id": "call_program",
                "code": "const result = await tools.lookup({ a: 1, b: 2 });",
                "fingerprint": "program-fingerprint"
            })),
            ContentPart::ToolCall(
                ToolCall::local("call_equal", "lookup", json!({"a": 1, "b": 2}))
                    .expect("local tool call"),
            ),
            opaque_output_part(json!({
                "id": "function_equal",
                "type": "function_call",
                "call_id": "call_equal",
                "name": "lookup",
                "arguments": "{ \"b\": 2, \"a\": 1 }",
                "caller": {"type": "program", "caller_id": "call_program"}
            })),
        ],
    );
    let request = LanguageRequest::new(vec![assistant]);

    let body = encode_request(&scope(), &model(), &request, false, &BTreeMap::new()).unwrap();
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 4);
    assert_eq!(
        input
            .iter()
            .map(|item| item["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["message", "reasoning", "program", "function_call"]
    );
    assert_eq!(input[3]["caller"]["type"], "program");
    assert_eq!(input[3]["caller"]["caller_id"], "call_program");
}

#[test]
fn request_rejects_message_and_reasoning_projection_mismatches() {
    let cases = [
        (
            ContentPart::Text {
                text: "portable text".to_string(),
            },
            json!({
                "id": "msg_mismatch",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "native text"}]
            }),
        ),
        (
            ContentPart::Reasoning {
                text: "portable reasoning".to_string(),
            },
            json!({
                "id": "reason_mismatch",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "native reasoning"}],
                "content": []
            }),
        ),
        (
            ContentPart::Media(MediaPart {
                media_type: "image/png".to_string(),
                data: MediaData::Url("https://example.com/portable.png".to_string()),
                name: None,
            }),
            json!({
                "id": "msg_media_mismatch",
                "type": "message",
                "role": "assistant",
                "content": []
            }),
        ),
    ];

    for (portable, native) in cases {
        let request = LanguageRequest::new(vec![Message::new(
            MessageRole::Assistant,
            [portable, opaque_output_part(native)],
        )]);
        let error = encode_request(&scope(), &model(), &request, false, &BTreeMap::new())
            .expect_err("native projection mismatch");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}

#[test]
fn request_rejects_native_function_name_and_argument_mismatches() {
    let native = opaque_output_part(json!({
        "id": "function_mismatch",
        "type": "function_call",
        "call_id": "call_mismatch",
        "name": "lookup",
        "arguments": "{\"value\":1}"
    }));
    let portable = [
        ToolCall::local("call_mismatch", "other_lookup", json!({"value": 1})).unwrap(),
        ToolCall::local("call_mismatch", "lookup", json!({"value": 2})).unwrap(),
    ];

    for call in portable {
        let request = LanguageRequest::new(vec![Message::new(
            MessageRole::Assistant,
            [ContentPart::ToolCall(call), native.clone()],
        )]);
        let error = encode_request(&scope(), &model(), &request, false, &BTreeMap::new())
            .expect_err("native function mismatch");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}

#[test]
fn request_rejects_program_and_custom_call_id_collisions_with_local_tools() {
    let native = [
        json!({
            "id": "program_collision",
            "type": "program",
            "call_id": "call_collision",
            "code": "return 1;",
            "fingerprint": "program-fingerprint"
        }),
        json!({
            "id": "custom_collision",
            "type": "custom_tool_call",
            "call_id": "call_collision",
            "name": "custom_lookup",
            "input": "opaque input"
        }),
    ];

    for native in native {
        let request = LanguageRequest::new(vec![Message::new(
            MessageRole::Assistant,
            [
                ContentPart::ToolCall(
                    ToolCall::local("call_collision", "lookup", json!({"value": 1})).unwrap(),
                ),
                opaque_output_part(native),
            ],
        )]);
        let error = encode_request(&scope(), &model(), &request, false, &BTreeMap::new())
            .expect_err("provider-owned call ID collision");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}

#[test]
fn opaque_history_requires_an_exact_replay_domain() {
    let decoded = decode_response(
        &serde_json::to_vec(&fidelity_response()).unwrap(),
        &scope(),
        &model(),
    )
    .unwrap();
    let assistant = decoded
        .canonical()
        .project_assistant_history()
        .into_message()
        .expect("native replay content remains");
    let request = LanguageRequest::new(vec![assistant]);
    let azure_scope = ProviderScope::new(ProviderId::new("openai").unwrap())
        .with_platform(PlatformId::new("azure").unwrap())
        .with_protocol(ProtocolId::new(OPENAI_RESPONSES_PROTOCOL).unwrap())
        .with_api_mode(ApiModeId::new("responses").unwrap())
        .with_replay_domain(ReplayDomain::official(
            ReplayDomainId::new("azure-openai-test").unwrap(),
        ));
    let custom_scope = ProviderScope::new(ProviderId::new("openai").unwrap())
        .with_platform(PlatformId::new("public-api").unwrap())
        .with_protocol(ProtocolId::new(OPENAI_RESPONSES_PROTOCOL).unwrap())
        .with_api_mode(ApiModeId::new("responses").unwrap())
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("openai-test").unwrap(),
        ));
    let account_scope = ProviderScope::new(ProviderId::new("openai").unwrap())
        .with_platform(PlatformId::new("public-api").unwrap())
        .with_protocol(ProtocolId::new(OPENAI_RESPONSES_PROTOCOL).unwrap())
        .with_api_mode(ApiModeId::new("responses").unwrap())
        .with_replay_domain(
            ReplayDomain::official(ReplayDomainId::new("openai-test").unwrap())
                .with_caller_scope(ReplayDomainId::new("account-b").unwrap()),
        );

    for target in [azure_scope, custom_scope, account_scope] {
        let error = encode_request(&target, &model(), &request, false, &BTreeMap::new())
            .expect_err("foreign replay domains must fail before transport");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}

#[test]
fn protected_fields_cannot_be_overridden_by_extra_options() {
    let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]);
    let extra = BTreeMap::from([("input".to_string(), json!([]))]);
    let error = encode_request(&scope(), &model(), &request, false, &extra).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(is_protected_option_field("tools"));
    assert!(is_protected_option_field("background"));
    assert!(is_protected_option_field("type"));
    assert!(is_protected_option_field("generate"));
    assert!(!is_protected_option_field("reasoning"));
}

#[test]
fn websocket_request_body_omits_http_transport_fields() {
    let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]);
    let body = encode_request_with_options(
        &scope(),
        &model(),
        &request,
        &RequestEncodingOptions::websocket(),
    )
    .unwrap();

    assert_eq!(body["model"], "gpt-5.6");
    assert!(body.get("stream").is_none());
    assert!(body.get("background").is_none());
    assert!(body.get("type").is_none());
    assert!(body.get("generate").is_none());
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
fn resolver_projects_annotated_text_image_and_file_blocks() {
    let request = LanguageRequest::new(vec![Message::new(
        MessageRole::User,
        [
            annotated_part(ContentPart::Text {
                text: "inspect these inputs".to_string(),
            }),
            annotated_part(ContentPart::Media(MediaPart {
                media_type: "image/png".to_string(),
                data: MediaData::Url("https://example.com/input.png".to_string()),
                name: None,
            })),
            annotated_part(ContentPart::Media(MediaPart {
                media_type: "application/pdf".to_string(),
                data: MediaData::Url("https://example.com/input.pdf".to_string()),
                name: Some("input.pdf".to_string()),
            })),
        ],
    )]);
    let options = RequestEncodingOptions::new(false);

    let body = encode_request_with_options_and_resolver(
        &scope(),
        &model(),
        &request,
        &options,
        &TestPromptCacheResolver,
    )
    .unwrap();
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
fn compatible_media_dialect_encodes_video_without_enabling_generic_files() {
    let video = ContentPart::Media(MediaPart {
        media_type: "video/mp4".to_string(),
        data: MediaData::Url("https://example.com/input.mp4".to_string()),
        name: None,
    });
    let request = LanguageRequest::new(vec![Message::new(MessageRole::User, [video.clone()])]);

    let native_error = encode_request_with_options(
        &scope(),
        &model(),
        &request,
        &RequestEncodingOptions::new(false),
    )
    .unwrap_err();
    assert_eq!(native_error.kind(), ErrorKind::Unsupported);

    let dialect = ResponsesMediaDialect::native()
        .with_video_input(true)
        .with_file_input(false);
    let options = RequestEncodingOptions::new(false).with_media_dialect(dialect);
    let body = encode_request_with_options(&scope(), &model(), &request, &options).unwrap();
    assert_eq!(body["input"][0]["content"][0]["type"], "input_video");
    assert_eq!(
        body["input"][0]["content"][0]["video_url"],
        "https://example.com/input.mp4"
    );

    let file_request = LanguageRequest::new(vec![Message::new(
        MessageRole::User,
        [ContentPart::Media(MediaPart {
            media_type: "application/pdf".to_string(),
            data: MediaData::Url("https://example.com/input.pdf".to_string()),
            name: Some("input.pdf".to_string()),
        })],
    )]);
    let file_error =
        encode_request_with_options(&scope(), &model(), &file_request, &options).unwrap_err();
    assert_eq!(file_error.kind(), ErrorKind::Unsupported);
}

#[test]
fn resolver_rejects_annotated_non_content_nodes() {
    let request = LanguageRequest::new(vec![Message::new(
        MessageRole::Assistant,
        [annotated_part(ContentPart::Reasoning {
            text: "private state".to_string(),
        })],
    )]);
    let error = encode_request_with_options_and_resolver(
        &scope(),
        &model(),
        &request,
        &RequestEncodingOptions::new(false),
        &TestPromptCacheResolver,
    )
    .unwrap_err();
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
        .decode_native(
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
    assert_eq!(
        done.native().kind(),
        &ResponsesStreamEventKind::OutputItemDone
    );
    assert!(
        matches!(done.native().item(), Some(OutputItem::FunctionCall(call))
        if call.call_id == "call_stream")
    );
    assert!(done.portable_events().iter().any(|event| matches!(
        event,
        LanguageStreamEvent::ToolCall(call)
            if call.id() == "call_stream" && call.arguments() == &json!({"q": "tea"})
    )));
    assert!(
        !done
            .portable_events()
            .iter()
            .any(|event| matches!(event, LanguageStreamEvent::ProviderOpaque(_)))
    );

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
fn provider_owned_custom_tool_stream_stays_out_of_portable_tool_events() {
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
                    "id": "ctc_1",
                    "type": "custom_tool_call",
                    "status": "in_progress",
                    "call_id": "call_custom",
                    "name": "sql",
                    "input": ""
                }
            })
            .to_string(),
        )
        .unwrap();
    assert!(added.is_empty());

    let delta = decoder
        .decode(
            &json!({
                "type": "response.custom_tool_call_input.delta",
                "sequence_number": 2,
                "item_id": "ctc_1",
                "output_index": 0,
                "delta": "SELECT 1"
            })
            .to_string(),
        )
        .unwrap();
    assert!(delta.is_empty());
    let input_done = decoder
        .decode(
            &json!({
                "type": "response.custom_tool_call_input.done",
                "sequence_number": 3,
                "item_id": "ctc_1",
                "output_index": 0,
                "input": "SELECT 1"
            })
            .to_string(),
        )
        .unwrap();
    assert!(input_done.is_empty());

    let item_done = decoder
        .decode(
            &json!({
                "type": "response.output_item.done",
                "sequence_number": 4,
                "output_index": 0,
                "item": {
                    "id": "ctc_1",
                    "type": "custom_tool_call",
                    "status": "completed",
                    "call_id": "call_custom",
                    "name": "sql",
                    "input": "SELECT 1"
                }
            })
            .to_string(),
        )
        .unwrap();
    assert!(item_done.is_empty());
}

#[test]
fn streamed_function_input_is_bounded_before_json_normalization() {
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
    decoder
        .decode(
            &json!({
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "id": "fc_bounded",
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": "call_bounded",
                    "name": "lookup",
                    "arguments": "{"
                }
            })
            .to_string(),
        )
        .unwrap();
    let oversized = json!({
        "type": "response.function_call_arguments.delta",
        "sequence_number": 2,
        "item_id": "fc_bounded",
        "output_index": 0,
        "delta": " ".repeat(siumai_core::DEFAULT_TOOL_INPUT_BYTE_LIMIT)
    })
    .to_string();

    let error = decoder.decode(&oversized).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResponseLimit);
}

#[test]
fn completed_function_input_is_bounded_without_deltas() {
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
    decoder
        .decode(
            &json!({
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "id": "fc_final_bounded",
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": "call_final_bounded",
                    "name": "lookup",
                    "arguments": ""
                }
            })
            .to_string(),
        )
        .unwrap();
    let oversized = json!({
        "type": "response.output_item.done",
        "sequence_number": 2,
        "output_index": 0,
        "item": {
            "id": "fc_final_bounded",
            "type": "function_call",
            "status": "completed",
            "call_id": "call_final_bounded",
            "name": "lookup",
            "arguments": format!(
                "{}{{}}",
                " ".repeat(siumai_core::DEFAULT_TOOL_INPUT_BYTE_LIMIT)
            )
        }
    })
    .to_string();

    let error = decoder.decode(&oversized).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ResponseLimit);
}

#[test]
fn usage_only_terminal_still_emits_started_usage_and_one_terminal() {
    let mut decoder = ResponsesStreamDecoder::new(scope(), model());
    assert!(decoder.replay_status().is_pending());
    assert!(!decoder.replay_status().is_available());
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
    let mut early_error = ResponsesStreamDecoder::new(scope(), model()).with_response_diagnostics(
        ResponseDiagnostics::default()
            .with_status(200)
            .with_retry_after(Duration::from_secs(2)),
    );
    let frame = early_error
        .decode_native(
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
    assert_eq!(frame.native().kind(), &ResponsesStreamEventKind::Error);
    assert_eq!(
        frame
            .native()
            .error()
            .and_then(|error| error.code.as_deref()),
        Some("rate_limit_exceeded")
    );
    let [
        LanguageStreamEvent::Terminal(StreamTerminal::Failed {
            error,
            response: None,
        }),
    ] = frame.portable_events()
    else {
        panic!("expected one canonical failed terminal");
    };
    assert_eq!(error.kind(), ErrorKind::RateLimited);
    assert_eq!(
        error.diagnostics().and_then(|value| value.retry_after()),
        Some(Duration::from_secs(2))
    );
    assert!(!format!("{error:?}").contains("provider detail"));
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
fn terminal_reconciliation_compares_function_calls_semantically() {
    let mut matching = completed_function_decoder("{\"q\":\"tea\"}");
    let events = matching
        .decode(&completed_function_response(
            3,
            json!([{
                "id": "fc_reconcile",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_reconcile",
                "name": "lookup",
                "arguments": "{ \"q\": \"tea\" }",
                "terminal_metadata": "preserved"
            }]),
        ))
        .unwrap();
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }))
            if response.content().iter().any(|part| matches!(
                part,
                ContentPart::ToolCall(call) if call.arguments() == &json!({"q": "tea"})
            ))
    ));

    let mut caller_metadata = completed_function_decoder_with_caller(
        "{\"q\":\"tea\"}",
        Some(json!({
            "type": "program",
            "caller_id": "program_1",
            "phase": "stream"
        })),
    );
    caller_metadata
        .decode(&completed_function_response(
            3,
            json!([{
                "id": "fc_reconcile",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_reconcile",
                "name": "lookup",
                "arguments": "{\"q\":\"tea\"}",
                "caller": {
                    "type": "program",
                    "caller_id": "program_1",
                    "phase": "terminal",
                    "provider_metadata": true
                }
            }]),
        ))
        .unwrap();

    let mut mismatching = completed_function_decoder("{\"q\":\"tea\"}");
    let error = mismatching
        .decode(&completed_function_response(
            3,
            json!([{
                "id": "fc_reconcile",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_reconcile",
                "name": "lookup",
                "arguments": "{\"q\":\"coffee\"}"
            }]),
        ))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Protocol);
}

#[test]
fn wire_dialect_normalizes_optional_function_identity_and_message_fields() {
    let mut function = completed_function_decoder("{\"q\":\"tea\"}");
    let events = function
        .decode(&completed_function_response(
            3,
            json!([{
                "type": "function_call",
                "call_id": "call_reconcile",
                "name": "lookup",
                "arguments": "{\"q\":\"tea\"}"
            }]),
        ))
        .unwrap();
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }))
            if response.content().iter().any(|part| matches!(
                part,
                ContentPart::ToolCall(call) if call.id() == "call_reconcile"
            ))
    ));

    let mut message = completed_message_decoder();
    let events = message
        .decode(&completed_function_response(
            3,
            json!([{
                "id": "msg_reconcile",
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "done",
                    "annotations": []
                }]
            }]),
        ))
        .unwrap();
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }))
            if response.content().iter().any(
                |part| matches!(part, ContentPart::Text { text } if text == "done")
            )
    ));
}

#[test]
fn wire_dialect_controls_missing_message_identity_recovery() {
    let terminal = completed_function_response(
        3,
        json!([{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "done",
                "annotations": []
            }]
        }]),
    );

    let mut strict = completed_message_decoder();
    assert_eq!(
        strict.decode(&terminal).unwrap_err().kind(),
        ErrorKind::Protocol
    );

    let mut compatible =
        completed_message_decoder().with_wire_dialect(ResponsesWireDialect::compatible());
    let events = compatible.decode(&terminal).unwrap();
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }))
            if response.content().iter().any(
                |part| matches!(part, ContentPart::Text { text } if text == "done")
            )
    ));
    assert_eq!(
        compatible.terminal_response().unwrap().output[0].id(),
        Some("msg_reconcile")
    );
}

#[test]
fn compatible_terminal_preserves_native_metadata_boundaries() {
    let terminal = completed_function_response(
        3,
        json!([{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "done"
            }]
        }]),
    );

    let mut compatible = completed_message_decoder_with_provider_metadata()
        .with_wire_dialect(ResponsesWireDialect::compatible());
    compatible.decode(&terminal).unwrap();
    let OutputItem::Message(message) = &compatible.terminal_response().unwrap().output[0] else {
        panic!("expected one reconciled message item");
    };
    let OutputContentPart::Text(text) = &message.content[0] else {
        panic!("expected one reconciled text part");
    };
    assert_eq!(text.text, "done");
    assert_eq!(text.annotations.len(), 1);
    assert!(text.logprobs.is_none());
    assert_eq!(message.phase, None);
}

#[test]
fn terminal_reconciliation_rejects_present_message_semantic_conflicts() {
    let mut decoder = completed_message_decoder();
    let error = decoder
        .decode(&completed_function_response(
            3,
            json!([{
                "id": "msg_reconcile",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "changed",
                    "annotations": []
                }]
            }]),
        ))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Protocol);

    let mut compatible =
        completed_message_decoder().with_wire_dialect(ResponsesWireDialect::compatible());
    let error = compatible
        .decode(&completed_function_response(
            3,
            json!([{
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "changed"
                }]
            }]),
        ))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Protocol);
}

#[test]
fn terminal_reconciliation_rejects_executable_identity_mutations() {
    let base = json!({
        "id": "fc_reconcile",
        "type": "function_call",
        "status": "completed",
        "call_id": "call_reconcile",
        "name": "lookup",
        "arguments": "{\"q\":\"tea\"}"
    });
    let mut changed_item_id = base.clone();
    changed_item_id["id"] = json!("fc_changed");
    let mut changed_provider_identity = completed_function_decoder("{\"q\":\"tea\"}");
    let events = changed_provider_identity
        .decode(&completed_function_response(3, json!([changed_item_id])))
        .unwrap();
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(
            StreamTerminal::Completed { .. }
        ))
    ));
    assert_eq!(
        changed_provider_identity
            .replay_status()
            .item_identity_conflicts(),
        1
    );

    let mut variants = Vec::new();
    let mut changed_call_id = base.clone();
    changed_call_id["call_id"] = json!("call_changed");
    variants.push(changed_call_id);
    let mut changed_name = base.clone();
    changed_name["name"] = json!("other_tool");
    variants.push(changed_name);
    let mut changed_caller = base.clone();
    changed_caller["caller"] = json!({"type": "program", "caller_id": "program_1"});
    variants.push(changed_caller);
    let mut changed_kind = base;
    changed_kind["type"] = json!("custom_tool_call");
    changed_kind["input"] = json!("{\"q\":\"tea\"}");
    variants.push(changed_kind);

    for terminal_item in variants {
        let mut decoder = completed_function_decoder("{\"q\":\"tea\"}");
        let error = decoder
            .decode(&completed_function_response(3, json!([terminal_item])))
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Protocol);
    }

    let mut compatible = completed_function_decoder("{\"q\":\"tea\"}")
        .with_wire_dialect(ResponsesWireDialect::compatible());
    let error = compatible
        .decode(&completed_function_response(
            3,
            json!([{
                "id": "fc_changed",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_changed",
                "name": "lookup",
                "arguments": "{\"q\":\"tea\"}"
            }]),
        ))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Protocol);
}

#[test]
fn wire_dialects_produce_the_same_portable_result_for_a_complete_sequence() {
    let terminal = completed_function_response(
        3,
        json!([{
            "id": "msg_reconcile",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "done",
                "annotations": []
            }]
        }]),
    );
    let mut openai = completed_message_decoder();
    let mut compatible =
        completed_message_decoder().with_wire_dialect(ResponsesWireDialect::compatible());

    let openai = openai.decode(&terminal).unwrap();
    let compatible = compatible.decode(&terminal).unwrap();
    let Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response: openai })) =
        openai.last()
    else {
        panic!("expected an OpenAI terminal response");
    };
    let Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed {
        response: compatible,
    })) = compatible.last()
    else {
        panic!("expected a compatible terminal response");
    };
    assert_eq!(openai, compatible);
}

#[test]
fn terminal_whole_item_completion_finishes_a_partial_text_lane() {
    let mut decoder = partial_message_decoder("hel");
    let events = decoder
        .decode(&completed_function_response(
            3,
            json!([{
                "id": "msg_partial",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "hello",
                    "annotations": []
                }]
            }]),
        ))
        .unwrap();
    assert!(events.iter().any(|event| matches!(
        event,
        LanguageStreamEvent::TextDelta { delta, .. } if delta == "lo"
    )));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, LanguageStreamEvent::TextEnd { .. }))
    );
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }))
            if response.content().iter().any(
                |part| matches!(part, ContentPart::Text { text } if text == "hello")
            )
    ));

    let mut conflicting = partial_message_decoder("hel");
    let error = conflicting
        .decode(&completed_function_response(
            3,
            json!([{
                "id": "msg_partial",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "hallo",
                    "annotations": []
                }]
            }]),
        ))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Protocol);
}

#[test]
fn incremental_ordering_rejects_backward_sequence_and_events_after_lane_completion() {
    let mut sequence = ResponsesStreamDecoder::new(scope(), model());
    sequence
        .decode(
            &json!({
                "type": "response.created",
                "sequence_number": 1,
                "response": progress_response("in_progress")
            })
            .to_string(),
        )
        .unwrap();
    let error = sequence
        .decode(
            &json!({
                "type": "response.in_progress",
                "sequence_number": 1,
                "response": progress_response("in_progress")
            })
            .to_string(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Protocol);

    let mut lane = partial_message_decoder("hel");
    let done = json!({
        "type": "response.output_text.done",
        "item_id": "msg_partial",
        "output_index": 0,
        "content_index": 0,
        "text": "hel"
    })
    .to_string();
    lane.decode(&done).unwrap();
    assert_eq!(lane.decode(&done).unwrap_err().kind(), ErrorKind::Protocol);
    let error = lane
        .decode(
            &json!({
                "type": "response.output_text.delta",
                "item_id": "msg_partial",
                "output_index": 0,
                "content_index": 0,
                "delta": "lo"
            })
            .to_string(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Protocol);
}

#[test]
fn reasoning_replay_conflict_is_structural_and_does_not_fail_portable_output() {
    const STREAM_SECRET: &str = "stream-encrypted-sentinel";
    const TERMINAL_SECRET: &str = "terminal-encrypted-sentinel";

    let mut decoder = completed_reasoning_decoder(STREAM_SECRET);
    let frame = decoder
        .decode_native(&completed_function_response(
            3,
            json!([{
                "id": "reasoning_reconcile",
                "type": "reasoning",
                "status": "completed",
                "summary": [{
                    "type": "summary_text",
                    "text": "checked inventory",
                    "provider_marker": "terminal-metadata-sentinel"
                }],
                "content": [],
                "encrypted_content": TERMINAL_SECRET
            }]),
        ))
        .unwrap();
    assert!(frame.is_terminal());
    assert_eq!(frame.replay_status().reasoning_state_conflicts(), 1);
    assert!(!frame.replay_status().is_available());
    let debug = format!("{frame:?}");
    for sentinel in [STREAM_SECRET, TERMINAL_SECRET, "terminal-metadata-sentinel"] {
        assert!(!debug.contains(sentinel));
    }
    assert!(matches!(
        frame.portable_events().last(),
        Some(LanguageStreamEvent::Terminal(
            StreamTerminal::Completed { .. }
        ))
    ));
    let Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response })) =
        frame.portable_events().last()
    else {
        panic!("expected a completed terminal response");
    };
    assert!(
        !response
            .content()
            .iter()
            .any(|part| matches!(part, ContentPart::ProviderOpaque(_)))
    );
}

#[test]
fn terminal_alignment_keeps_portable_message_semantics_when_identity_drifts() {
    let mut decoder = completed_message_decoder();
    let events = decoder
        .decode(&completed_function_response(
            3,
            json!([{
                "id": "msg_other",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "done",
                    "annotations": []
                }]
            }]),
        ))
        .unwrap();
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }))
            if response.content().iter().filter(|part| matches!(part, ContentPart::Text { text } if text == "done")).count() == 1
    ));
    assert_eq!(decoder.replay_status().item_identity_conflicts(), 1);
    assert!(!decoder.replay_status().is_available());
}

#[test]
fn terminal_alignment_rejects_uncovered_partial_message_lanes() {
    let mut decoder = partial_message_decoder("hel");
    let error = decoder
        .decode(&completed_function_response(
            3,
            json!([{
                "id": "msg_partial",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": []
            }]),
        ))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Protocol);
}

#[test]
fn terminal_sequence_drift_is_not_an_incremental_ordering_error() {
    let mut decoder = completed_message_decoder();
    let events = decoder
        .decode(&completed_function_response(
            0,
            json!([{
                "id": "msg_reconcile",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "done",
                    "annotations": []
                }]
            }]),
        ))
        .unwrap();
    assert!(events.iter().any(|event| event.terminal().is_some()));
}

#[test]
fn function_arguments_use_canonical_json_equality_at_terminal_boundaries() {
    let mut decoder = completed_function_decoder("{\"q\":\"tea\"}");
    let events = decoder
        .decode(&completed_function_response(
            3,
            json!([{
                "id": "fc_reconcile",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_reconcile",
                "name": "lookup",
                "arguments": "{ \"q\" : \"tea\" }"
            }]),
        ))
        .unwrap();
    assert!(events.iter().any(|event| event.terminal().is_some()));
}

#[test]
fn terminal_reconciliation_merges_a_completed_stream_item_missing_from_snapshot() {
    let mut decoder = completed_function_decoder("{\"q\":\"tea\"}");
    let frame = decoder
        .decode_native(&completed_function_response(3, json!([])))
        .unwrap();
    assert!(frame.is_terminal());
    assert_eq!(
        frame.native().kind(),
        &ResponsesStreamEventKind::ResponseCompleted
    );
    assert_eq!(frame.native().raw_response().unwrap()["output"], json!([]));
    assert!(matches!(
        frame.portable_events().last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }))
            if response.content().iter().any(|part| matches!(
                part,
                ContentPart::ToolCall(call) if call.id() == "call_reconcile"
            ))
    ));
    assert_eq!(decoder.terminal_response().unwrap().output.len(), 1);

    let mut terminal_only = ResponsesStreamDecoder::new(scope(), model());
    terminal_only
        .decode(
            &json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": progress_response("in_progress")
            })
            .to_string(),
        )
        .unwrap();
    let events = terminal_only
        .decode(&completed_function_response(
            1,
            json!([{
                "id": "fc_terminal_only",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_terminal_only",
                "name": "lookup",
                "arguments": "{\"q\":\"tea\"}"
            }]),
        ))
        .unwrap();
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }))
            if response.content().iter().any(|part| matches!(
                part,
                ContentPart::ToolCall(call) if call.id() == "call_terminal_only"
            ))
    ));
}

#[test]
fn terminal_reconciliation_merges_a_function_omitted_before_a_retained_item() {
    let mut decoder = completed_function_decoder("{\"q\":\"tea\"}")
        .with_wire_dialect(ResponsesWireDialect::compatible());
    decoder
        .decode(
            &json!({
                "type": "response.output_item.added",
                "sequence_number": 3,
                "output_index": 1,
                "item": {
                    "id": "msg_after_call",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": []
                }
            })
            .to_string(),
        )
        .unwrap();
    let terminal_message = json!({
        "id": "msg_after_call",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{
            "type": "output_text",
            "text": "done",
            "annotations": []
        }]
    });
    decoder
        .decode(
            &json!({
                "type": "response.output_item.done",
                "sequence_number": 4,
                "output_index": 1,
                "item": terminal_message.clone()
            })
            .to_string(),
        )
        .unwrap();

    let events = decoder
        .decode(&completed_function_response(5, json!([terminal_message])))
        .unwrap();
    let Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response })) = events.last()
    else {
        panic!("expected a completed terminal response");
    };
    let tool_index = response
        .content()
        .iter()
        .position(
            |part| matches!(part, ContentPart::ToolCall(call) if call.id() == "call_reconcile"),
        )
        .unwrap();
    let text_index = response
        .content()
        .iter()
        .position(|part| matches!(part, ContentPart::Text { text } if text == "done"))
        .unwrap();
    assert!(tool_index < text_index);
}

#[test]
fn terminal_reconciliation_does_not_synthesize_provider_native_items() {
    let mut decoder = completed_native_item_decoder(
        json!({
            "id": "future_1",
            "type": "future_provider_tool_call",
            "status": "in_progress",
            "payload": {"phase": "stream"}
        }),
        json!({
            "id": "future_1",
            "type": "future_provider_tool_call",
            "status": "completed",
            "payload": {"phase": "stream"}
        }),
    );

    let events = decoder
        .decode(&completed_function_response(3, json!([])))
        .unwrap();
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }))
            if response.content().is_empty()
    ));
}

#[test]
fn terminal_reconciliation_accepts_provider_native_metadata_changes() {
    let mut decoder = completed_native_item_decoder(
        json!({
            "id": "future_1",
            "type": "future_provider_tool_call",
            "status": "in_progress",
            "payload": {"phase": "stream"}
        }),
        json!({
            "id": "future_1",
            "type": "future_provider_tool_call",
            "status": "completed",
            "payload": {"phase": "stream"}
        }),
    );

    let events = decoder
        .decode(&completed_function_response(
            3,
            json!([{
                "id": "future_1",
                "type": "future_provider_tool_call",
                "status": "completed",
                "payload": {"phase": "terminal", "provider_metadata": true}
            }]),
        ))
        .unwrap();
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(
            StreamTerminal::Completed { .. }
        ))
    ));
    assert!(!decoder.replay_status().is_available());

    let mut message = completed_message_decoder_with_provider_metadata();
    let events = message
        .decode(&completed_function_response(
            3,
            json!([{
                "id": "msg_reconcile",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "phase": "provider_changed_phase",
                "content": [{
                    "type": "output_text",
                    "text": "done",
                    "annotations": [{
                        "type": "url_citation",
                        "url": "https://example.com/source",
                        "title": "Source",
                        "start_index": 0,
                        "end_index": 4,
                        "provider_marker": "terminal-only"
                    }],
                    "logprobs": [{"token": "done", "logprob": -9.0}]
                }]
            }]),
        ))
        .unwrap();
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(
            StreamTerminal::Completed { .. }
        ))
    ));
    assert_eq!(message.replay_status().provider_item_conflicts(), 1);
    assert!(!message.replay_status().is_available());
}

#[test]
fn terminal_reconciliation_marks_typed_native_mutations_non_replayable() {
    for (added, done, terminal) in [
        (
            json!({
                "id": "custom_1",
                "type": "custom_tool_call",
                "status": "in_progress",
                "call_id": "call_custom",
                "name": "grammar",
                "input": "opaque"
            }),
            json!({
                "id": "custom_1",
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": "call_custom",
                "name": "grammar",
                "input": "opaque"
            }),
            json!({
                "id": "custom_1",
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": "call_custom",
                "name": "grammar",
                "input": "changed"
            }),
        ),
        (
            json!({
                "id": "program_1",
                "type": "program",
                "call_id": "call_program",
                "code": "return 1",
                "fingerprint": "fp_1"
            }),
            json!({
                "id": "program_1",
                "type": "program",
                "call_id": "call_program",
                "code": "return 1",
                "fingerprint": "fp_1"
            }),
            json!({
                "id": "program_1",
                "type": "program",
                "call_id": "call_program",
                "code": "return 2",
                "fingerprint": "fp_1"
            }),
        ),
        (
            json!({
                "id": "program_output_1",
                "type": "program_output",
                "status": "in_progress",
                "call_id": "call_program",
                "result": "pending"
            }),
            json!({
                "id": "program_output_1",
                "type": "program_output",
                "status": "completed",
                "call_id": "call_program",
                "result": "done"
            }),
            json!({
                "id": "program_output_1",
                "type": "program_output",
                "status": "completed",
                "call_id": "call_program",
                "result": "changed"
            }),
        ),
    ] {
        let mut decoder = completed_native_item_decoder(added, done);
        let events = decoder
            .decode(&completed_function_response(3, json!([terminal])))
            .unwrap();
        assert!(matches!(
            events.last(),
            Some(LanguageStreamEvent::Terminal(
                StreamTerminal::Completed { .. }
            ))
        ));
        assert!(!decoder.replay_status().is_available());
        assert_eq!(decoder.replay_status().provider_item_conflicts(), 1);
    }
}

#[test]
fn terminal_reconciliation_keeps_related_native_items_out_of_function_identity() {
    let function = json!({
        "id": "fc_reconcile",
        "type": "function_call",
        "status": "completed",
        "call_id": "call_reconcile",
        "name": "lookup",
        "arguments": "{\"q\":\"tea\"}"
    });
    let related_output = json!({
        "id": "fco_related",
        "type": "function_call_output",
        "status": "completed",
        "call_id": "call_reconcile",
        "output": "{\"result\":\"ok\"}"
    });

    let mut retained = completed_function_decoder("{\"q\":\"tea\"}");
    retained
        .decode(&completed_function_response(
            3,
            json!([function, related_output.clone()]),
        ))
        .unwrap();

    let mut omitted = completed_function_decoder("{\"q\":\"tea\"}")
        .with_wire_dialect(ResponsesWireDialect::compatible());
    let events = omitted
        .decode(&completed_function_response(3, json!([related_output])))
        .unwrap();
    assert!(matches!(
        events.last(),
        Some(LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }))
            if response.content().iter().any(|part| matches!(
                part,
                ContentPart::ToolCall(call) if call.id() == "call_reconcile"
            ))
    ));
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
    assert!(matches!(
        decoded.native().output.as_slice(),
        [OutputItem::ProgramOutput(output)] if output.status.as_str() == "incomplete"
    ));
    assert!(
        !decoded
            .canonical()
            .content()
            .iter()
            .any(|part| matches!(part, ContentPart::ToolResult(_)))
    );
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

fn completed_function_decoder(arguments: &str) -> ResponsesStreamDecoder {
    completed_function_decoder_with_caller(arguments, None)
}

fn partial_message_decoder(prefix: &str) -> ResponsesStreamDecoder {
    let mut decoder = ResponsesStreamDecoder::new(scope(), model());
    for event in [
        json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": progress_response("in_progress")
        }),
        json!({
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "id": "msg_partial",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": []
            }
        }),
        json!({
            "type": "response.output_text.delta",
            "sequence_number": 2,
            "item_id": "msg_partial",
            "output_index": 0,
            "content_index": 0,
            "delta": prefix
        }),
    ] {
        decoder.decode(&event.to_string()).unwrap();
    }
    decoder
}

fn completed_reasoning_decoder(encrypted_content: &str) -> ResponsesStreamDecoder {
    let mut decoder = ResponsesStreamDecoder::new(scope(), model());
    for event in [
        json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": progress_response("in_progress")
        }),
        json!({
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "id": "reasoning_reconcile",
                "type": "reasoning",
                "status": "in_progress",
                "summary": [],
                "content": []
            }
        }),
        json!({
            "type": "response.output_item.done",
            "sequence_number": 2,
            "output_index": 0,
            "item": {
                "id": "reasoning_reconcile",
                "type": "reasoning",
                "status": "completed",
                "summary": [{
                    "type": "summary_text",
                    "text": "checked inventory",
                    "provider_marker": "stream-metadata-sentinel"
                }],
                "content": [],
                "encrypted_content": encrypted_content
            }
        }),
    ] {
        decoder.decode(&event.to_string()).unwrap();
    }
    decoder
}

fn completed_message_decoder() -> ResponsesStreamDecoder {
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
    decoder
        .decode(
            &json!({
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "id": "msg_reconcile",
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        )
        .unwrap();
    decoder
        .decode(
            &json!({
                "type": "response.output_item.done",
                "sequence_number": 2,
                "output_index": 0,
                "item": {
                    "id": "msg_reconcile",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": "done",
                        "annotations": []
                    }]
                }
            })
            .to_string(),
        )
        .unwrap();
    decoder
}

fn completed_message_decoder_with_provider_metadata() -> ResponsesStreamDecoder {
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
    decoder
        .decode(
            &json!({
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "id": "msg_reconcile",
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        )
        .unwrap();
    decoder
        .decode(
            &json!({
                "type": "response.output_item.done",
                "sequence_number": 2,
                "output_index": 0,
                "item": {
                    "id": "msg_reconcile",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{
                        "type": "output_text",
                        "text": "done",
                        "annotations": [{
                            "type": "url_citation",
                            "url": "https://example.com/source",
                            "title": "Source",
                            "start_index": 0,
                            "end_index": 4
                        }],
                        "logprobs": [{"token": "done", "logprob": -0.1}]
                    }]
                }
            })
            .to_string(),
        )
        .unwrap();
    decoder
}

fn completed_function_decoder_with_caller(
    arguments: &str,
    caller: Option<Value>,
) -> ResponsesStreamDecoder {
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
    let mut added = json!({
        "id": "fc_reconcile",
        "type": "function_call",
        "status": "in_progress",
        "call_id": "call_reconcile",
        "name": "lookup",
        "arguments": ""
    });
    let mut done = json!({
        "id": "fc_reconcile",
        "type": "function_call",
        "status": "completed",
        "call_id": "call_reconcile",
        "name": "lookup",
        "arguments": arguments
    });
    if let Some(caller) = caller {
        added["caller"] = caller.clone();
        done["caller"] = caller;
    }
    decoder
        .decode(
            &json!({
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": added
            })
            .to_string(),
        )
        .unwrap();
    decoder
        .decode(
            &json!({
                "type": "response.output_item.done",
                "sequence_number": 2,
                "output_index": 0,
                "item": done
            })
            .to_string(),
        )
        .unwrap();
    decoder
}

fn completed_native_item_decoder(added: Value, done: Value) -> ResponsesStreamDecoder {
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
    decoder
        .decode(
            &json!({
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": added
            })
            .to_string(),
        )
        .unwrap();
    decoder
        .decode(
            &json!({
                "type": "response.output_item.done",
                "sequence_number": 2,
                "output_index": 0,
                "item": done
            })
            .to_string(),
        )
        .unwrap();
    decoder
}

fn completed_function_response(sequence_number: u64, output: Value) -> String {
    json!({
        "type": "response.completed",
        "sequence_number": sequence_number,
        "response": {
            "id": "resp_stream",
            "created_at": 1785811200,
            "model": "gpt-5.6",
            "status": "completed",
            "output": output,
            "usage": null,
            "error": null,
            "incomplete_details": null,
            "reasoning": null
        }
    })
    .to_string()
}

#[test]
fn repository_response_fixtures_round_trip_native_items_losslessly() {
    for (name, body) in [
        (
            "encrypted reasoning",
            include_str!("../../tests/fixtures/responses/reasoning-encrypted-content.json"),
        ),
        (
            "web search",
            include_str!("../../tests/fixtures/responses/web-search-tool.json"),
        ),
        (
            "apply patch",
            include_str!("../../tests/fixtures/responses/apply-patch-tool.json"),
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

#[test]
fn native_response_debug_redacts_provider_payloads() {
    let sentinel = "response-debug-sentinel";
    let body = serde_json::to_vec(&json!({
        "id": "resp_debug",
        "created_at": 1,
        "model": "gpt-5.6",
        "status": "completed",
        "output": [
            {
                "id": "msg_debug",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": sentinel,
                    "annotations": []
                }]
            },
            {
                "id": "future_debug",
                "type": "future_provider_tool_call",
                "private_payload": sentinel
            }
        ],
        "usage": null,
        "error": null,
        "incomplete_details": null,
        "reasoning": null,
        "private_top_level": sentinel
    }))
    .unwrap();

    let decoded = decode_response(&body, &scope(), &model()).unwrap();
    assert!(!format!("{decoded:?}").contains(sentinel));
    assert!(!format!("{:?}", decoded.native()).contains(sentinel));
    for item in &decoded.native().output {
        assert!(!format!("{item:?}").contains(sentinel));
    }

    let wire = StreamEventWire {
        kind: "response.future".to_string(),
        sequence_number: Some(1),
        fields: BTreeMap::from([("private_payload".to_string(), json!(sentinel))]),
    };
    assert!(!format!("{wire:?}").contains(sentinel));
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
