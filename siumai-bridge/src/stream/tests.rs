#![allow(dead_code, unused_imports)]

use std::sync::Arc;

#[cfg(feature = "anthropic")]
use crate::StreamBridgeHook;
use crate::{
    BridgeCustomization, BridgeLossAction, BridgeLossPolicy, BridgeMode, BridgeOptions,
    BridgePrimitiveContext, BridgePrimitiveRemapper, BridgeTarget, RequestBridgeContext,
    ResponseBridgeContext, StreamBridgeContext,
};
use futures_util::{StreamExt, stream};
use siumai_core::streaming::ChatByteStream;
use siumai_core::types::{
    ChatResponse, ChatStreamEvent, ChatStreamFinishInfo, ChatStreamPart, ChatStreamReplay,
    ChatStreamToolCall, ChatStreamToolResult, ContentPart, FinishReason, MessageContent,
    ResponseMetadata, Usage,
};

#[cfg(feature = "anthropic")]
use super::{
    bridge_chat_stream_to_anthropic_messages_sse,
    bridge_chat_stream_to_anthropic_messages_sse_with_options,
};
#[cfg(feature = "openai")]
use super::{
    bridge_chat_stream_to_openai_chat_completions_sse, bridge_chat_stream_to_openai_responses_sse,
    transform_chat_stream_with_bridge_options,
};

#[cfg(feature = "anthropic")]
struct UppercaseStreamHook;

fn uppercase_textual_part(part: ChatStreamPart) -> ChatStreamPart {
    match part {
        ChatStreamPart::TextDelta {
            id,
            delta,
            provider_metadata,
        } => ChatStreamPart::TextDelta {
            id,
            delta: delta.to_uppercase(),
            provider_metadata,
        },
        ChatStreamPart::ReasoningDelta {
            id,
            delta,
            provider_metadata,
        } => ChatStreamPart::ReasoningDelta {
            id,
            delta: delta.to_uppercase(),
            provider_metadata,
        },
        other => other,
    }
}

fn uppercase_textual_event(event: ChatStreamEvent) -> Vec<ChatStreamEvent> {
    match event {
        ChatStreamEvent::Part { part } => vec![ChatStreamEvent::Part {
            part: uppercase_textual_part(part),
        }],
        ChatStreamEvent::PartWithReplay { part, replay } => vec![ChatStreamEvent::PartWithReplay {
            part: uppercase_textual_part(part),
            replay,
        }],
        other => vec![other],
    }
}

#[cfg(feature = "anthropic")]
impl StreamBridgeHook for UppercaseStreamHook {
    fn map_event(
        &self,
        _ctx: &StreamBridgeContext,
        event: ChatStreamEvent,
    ) -> Vec<ChatStreamEvent> {
        uppercase_textual_event(event)
    }
}

#[cfg(feature = "openai")]
struct ContinueLossyPolicy;

#[cfg(feature = "openai")]
impl BridgeLossPolicy for ContinueLossyPolicy {
    fn request_action(
        &self,
        _ctx: &RequestBridgeContext,
        _report: &crate::BridgeReport,
    ) -> BridgeLossAction {
        BridgeLossAction::Continue
    }

    fn response_action(
        &self,
        _ctx: &ResponseBridgeContext,
        _report: &crate::BridgeReport,
    ) -> BridgeLossAction {
        BridgeLossAction::Continue
    }

    fn stream_action(
        &self,
        _ctx: &StreamBridgeContext,
        _report: &crate::BridgeReport,
    ) -> BridgeLossAction {
        BridgeLossAction::Continue
    }
}

struct PrefixStreamRemapper;

impl BridgePrimitiveRemapper for PrefixStreamRemapper {
    fn remap_tool_name(&self, _ctx: &BridgePrimitiveContext, name: &str) -> Option<String> {
        Some(format!("gw_{name}"))
    }

    fn remap_tool_call_id(&self, _ctx: &BridgePrimitiveContext, id: &str) -> Option<String> {
        Some(format!("gw_{id}"))
    }
}

struct CompositeStreamCustomization;

impl BridgeCustomization for CompositeStreamCustomization {
    fn map_stream_event(
        &self,
        ctx: &StreamBridgeContext,
        event: ChatStreamEvent,
    ) -> Vec<ChatStreamEvent> {
        assert_eq!(ctx.source, Some(BridgeTarget::OpenAiResponses));
        assert_eq!(ctx.target, BridgeTarget::AnthropicMessages);
        assert_eq!(
            ctx.route_label.as_deref(),
            Some("tests.stream.customization")
        );
        assert_eq!(ctx.path_label.as_deref(), Some("tests.stream.custom-path"));

        match event {
            ChatStreamEvent::Part {
                part: ChatStreamPart::TextDelta { .. } | ChatStreamPart::ReasoningDelta { .. },
            }
            | ChatStreamEvent::PartWithReplay {
                part: ChatStreamPart::TextDelta { .. } | ChatStreamPart::ReasoningDelta { .. },
                ..
            } => uppercase_textual_event(event),
            ChatStreamEvent::Part {
                part:
                    ChatStreamPart::ToolInputStart {
                        ref id,
                        ref tool_name,
                        ..
                    },
            }
            | ChatStreamEvent::PartWithReplay {
                part:
                    ChatStreamPart::ToolInputStart {
                        ref id,
                        ref tool_name,
                        ..
                    },
                ..
            } => {
                assert_eq!(id, "bundle_call_1");
                assert_eq!(tool_name, "bundle_weather");
                vec![event]
            }
            ChatStreamEvent::Part {
                part: ChatStreamPart::ToolCall(ref call),
            }
            | ChatStreamEvent::PartWithReplay {
                part: ChatStreamPart::ToolCall(ref call),
                ..
            } => {
                assert_eq!(call.tool_call_id, "bundle_call_1");
                assert_eq!(call.tool_name, "bundle_weather");
                vec![event]
            }
            other => vec![other],
        }
    }

    fn remap_tool_name(&self, ctx: &BridgePrimitiveContext, name: &str) -> Option<String> {
        assert_eq!(ctx.source, Some(BridgeTarget::OpenAiResponses));
        assert_eq!(ctx.target, BridgeTarget::AnthropicMessages);
        Some(format!("bundle_{name}"))
    }

    fn remap_tool_call_id(&self, ctx: &BridgePrimitiveContext, id: &str) -> Option<String> {
        assert_eq!(ctx.source, Some(BridgeTarget::OpenAiResponses));
        assert_eq!(ctx.target, BridgeTarget::AnthropicMessages);
        Some(format!("bundle_{id}"))
    }
}

async fn collect_bytes(mut stream: ChatByteStream) -> String {
    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("stream chunk");
        out.extend_from_slice(&chunk);
    }
    String::from_utf8(out).expect("utf8")
}

#[cfg(feature = "openai")]
fn parse_sse_frames(body: &str) -> Vec<(String, serde_json::Value)> {
    body.split("\n\n")
        .filter_map(|frame| {
            let mut event = None;
            let mut data_lines = Vec::new();

            for line in frame.lines() {
                if let Some(value) = line.strip_prefix("event: ") {
                    event = Some(value.to_string());
                } else if let Some(value) = line.strip_prefix("data: ") {
                    data_lines.push(value);
                }
            }

            let event = event?;
            let data = data_lines.join("\n");
            if data == "[DONE]" || data.is_empty() {
                return None;
            }

            Some((event, serde_json::from_str(&data).expect("json sse data")))
        })
        .collect()
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[tokio::test]
async fn openai_responses_stream_bridge_rewrites_anthropic_custom_parts() {
    let response = ChatResponse {
        id: Some("resp_1".to_string()),
        model: Some("gpt-4.1-mini".to_string()),
        content: MessageContent::Text("Hello".to_string()),
        usage: None,
        finish_reason: Some(FinishReason::Stop),
        raw_finish_reason: None,
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
        request: None,
        response: None,
        provider_metadata: None,
    };

    let stream = stream::iter(vec![
        Ok(ChatStreamEvent::Custom {
            event_type: "anthropic:stream-start".to_string(),
            data: serde_json::json!({
                "type": "stream-start",
                "warnings": [],
            }),
        }),
        Ok(ChatStreamEvent::Custom {
            event_type: "anthropic:response-metadata".to_string(),
            data: serde_json::json!({
                "type": "response-metadata",
                "id": "resp_1",
                "modelId": "gpt-4.1-mini",
            }),
        }),
        Ok(ChatStreamEvent::Custom {
            event_type: "anthropic:text-start".to_string(),
            data: serde_json::json!({
                "type": "text-start",
                "id": "0",
            }),
        }),
        Ok(ChatStreamEvent::Custom {
            event_type: "anthropic:text-delta".to_string(),
            data: serde_json::json!({
                "type": "text-delta",
                "id": "0",
                "delta": "Hello",
            }),
        }),
        Ok(ChatStreamEvent::Custom {
            event_type: "anthropic:text-end".to_string(),
            data: serde_json::json!({
                "type": "text-end",
                "id": "0",
            }),
        }),
        Ok(ChatStreamEvent::StreamEnd { response }),
    ]);

    let bridged = bridge_chat_stream_to_openai_responses_sse(
        stream,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    let body = collect_bytes(bridged.value.expect("byte stream")).await;

    assert!(
        body.contains("event: response.created"),
        "expected response.created frame"
    );
    assert!(
        body.contains("event: response.output_text.delta"),
        "expected output_text.delta frame"
    );
    assert!(
        body.contains("\"finish_reason\":\"stop\""),
        "expected response.completed finish_reason stop"
    );
    assert!(
        body.contains("data: [DONE]"),
        "expected terminal done frame"
    );
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_stream_bridge_serializes_standard_events() {
    let response = ChatResponse {
        id: Some("msg_1".to_string()),
        model: Some("claude-sonnet-4-5".to_string()),
        content: MessageContent::Text("Hello".to_string()),
        usage: None,
        finish_reason: Some(FinishReason::Stop),
        raw_finish_reason: None,
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
        request: None,
        response: None,
        provider_metadata: None,
    };

    let stream = stream::iter(vec![
        Ok(ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("msg_1".to_string()),
                model: Some("claude-sonnet-4-5".to_string()),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
                headers: None,
                body: None,
            },
        }),
        Ok(ChatStreamEvent::text_delta_part("0", "Hello")),
        Ok(ChatStreamEvent::StreamEnd { response }),
    ]);

    let bridged = bridge_chat_stream_to_anthropic_messages_sse(
        stream,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    let body = collect_bytes(bridged.value.expect("byte stream")).await;

    assert!(body.contains("event: message_start"));
    assert!(body.contains("event: content_block_delta"));
    assert!(body.contains("event: message_stop"));
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn stream_bridge_options_can_transform_events() {
    let response = ChatResponse {
        id: Some("msg_1".to_string()),
        model: Some("claude-sonnet-4-5".to_string()),
        content: MessageContent::Text("HELLO".to_string()),
        usage: None,
        finish_reason: Some(FinishReason::Stop),
        raw_finish_reason: None,
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
        request: None,
        response: None,
        provider_metadata: None,
    };

    let stream = stream::iter(vec![
        Ok(ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("msg_1".to_string()),
                model: Some("claude-sonnet-4-5".to_string()),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
                headers: None,
                body: None,
            },
        }),
        Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::TextDelta {
                id: "txt_1".to_string(),
                delta: "hello".to_string(),
                provider_metadata: None,
            },
        }),
        Ok(ChatStreamEvent::StreamEnd { response }),
    ]);

    let bridged = bridge_chat_stream_to_anthropic_messages_sse_with_options(
        stream,
        Some(BridgeTarget::OpenAiResponses),
        BridgeOptions::new(BridgeMode::BestEffort)
            .with_route_label("tests.stream.transform")
            .with_stream_hook(Arc::new(UppercaseStreamHook)),
    )
    .expect("bridge");

    let body = collect_bytes(bridged.value.expect("byte stream")).await;

    assert!(body.contains("HELLO"));
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_stream_bridge_finalizes_clean_eof_without_stream_end() {
    let stream = stream::iter(vec![
        Ok(ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("msg_1".to_string()),
                model: Some("claude-sonnet-4-5".to_string()),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
                headers: None,
                body: None,
            },
        }),
        Ok(ChatStreamEvent::text_delta_part("0", "Hello")),
    ]);

    let bridged = bridge_chat_stream_to_anthropic_messages_sse(
        stream,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    let body = collect_bytes(bridged.value.expect("byte stream")).await;

    assert!(body.contains("event: content_block_delta"));
    assert!(body.contains("event: message_stop"));
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[tokio::test]
async fn anthropic_stream_bridge_splits_interleaved_blocks_into_ordered_output() {
    let response = ChatResponse {
        id: Some("msg_1".to_string()),
        model: Some("claude-sonnet-4-5".to_string()),
        content: MessageContent::Text("Hello world".to_string()),
        usage: None,
        finish_reason: Some(FinishReason::Stop),
        raw_finish_reason: None,
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
        request: None,
        response: None,
        provider_metadata: None,
    };

    let stream = stream::iter(vec![
        Ok(ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("msg_1".to_string()),
                model: Some("claude-sonnet-4-5".to_string()),
                created: None,
                provider: "anthropic".to_string(),
                request_id: None,
                headers: None,
                body: None,
            },
        }),
        Ok(ChatStreamEvent::text_delta_part("text-0", "Hello")),
        Ok(ChatStreamEvent::reasoning_delta_part(
            "reasoning-0",
            "Thinking",
        )),
        Ok(ChatStreamEvent::text_delta_part("text-1", " world")),
        Ok(ChatStreamEvent::StreamEnd { response }),
    ]);

    let bridged = bridge_chat_stream_to_anthropic_messages_sse(
        stream,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    let body = collect_bytes(bridged.value.expect("byte stream")).await;

    assert_eq!(body.matches("event: content_block_start").count(), 3);
    assert_eq!(body.matches("event: content_block_stop").count(), 3);
    assert!(body.contains("\"index\":0"));
    assert!(body.contains("\"index\":1"));
    assert!(body.contains("\"index\":2"));
    assert!(body.contains("event: message_stop"));
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_responses_stream_bridge_finalizes_clean_eof_without_stream_end() {
    let stream = stream::iter(vec![
        Ok(ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("resp_1".to_string()),
                model: Some("gpt-4.1-mini".to_string()),
                created: None,
                provider: "openai".to_string(),
                request_id: None,
                headers: None,
                body: None,
            },
        }),
        Ok(ChatStreamEvent::text_delta_part("0", "Hello")),
    ]);

    let bridged = bridge_chat_stream_to_openai_responses_sse(
        stream,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    let body = collect_bytes(bridged.value.expect("byte stream")).await;

    assert!(body.contains("event: response.completed"));
    assert!(body.contains("data: [DONE]"));
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_chat_completions_stream_bridge_prefers_stream_end_terminal_envelope() {
    let response = ChatResponse {
        id: Some("chatcmpl_1".to_string()),
        model: Some("gpt-4o-mini".to_string()),
        content: MessageContent::Text("Hello".to_string()),
        usage: Some(Usage::new(4, 2)),
        finish_reason: Some(FinishReason::Stop),
        raw_finish_reason: None,
        audio: None,
        system_fingerprint: Some("fp_terminal_1".to_string()),
        service_tier: Some("priority".to_string()),
        warnings: None,
        request: None,
        response: None,
        provider_metadata: None,
    };

    let stream = stream::iter(vec![
        Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::Finish {
                usage: Usage::new(4, 2),
                finish_reason: ChatStreamFinishInfo {
                    unified: FinishReason::Stop,
                    raw: Some("stop".to_string()),
                },
                provider_metadata: None,
            },
        }),
        Ok(ChatStreamEvent::StreamEnd { response }),
    ]);

    let bridged = bridge_chat_stream_to_openai_chat_completions_sse(
        stream,
        Some(BridgeTarget::OpenAiChatCompletions),
        BridgeMode::Strict,
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    let body = collect_bytes(bridged.value.expect("byte stream")).await;

    assert_eq!(
        body.matches("\"finish_reason\":\"stop\"").count(),
        1,
        "expected a single terminal chunk"
    );
    assert!(body.contains("\"system_fingerprint\":\"fp_terminal_1\""));
    assert!(body.contains("\"service_tier\":\"priority\""));
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn cross_protocol_stream_bridge_rejects_in_strict_mode() {
    let stream = stream::iter(vec![Ok(ChatStreamEvent::text_delta_part("0", "Hello"))]);

    let bridged = bridge_chat_stream_to_anthropic_messages_sse(
        stream,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::Strict,
    )
    .expect("bridge");

    assert!(
        bridged.is_rejected(),
        "strict cross-protocol stream should reject"
    );
    assert!(bridged.report.is_rejected());
    assert!(
        bridged
            .report
            .lossy_fields
            .iter()
            .any(|field| field == "stream.protocol")
    );
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn best_effort_cross_protocol_stream_bridge_allows_lossy_route() {
    let stream = stream::iter(vec![Ok(ChatStreamEvent::text_delta_part("0", "Hello"))]);

    let bridged = bridge_chat_stream_to_anthropic_messages_sse(
        stream,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_lossy());
    assert!(
        bridged
            .report
            .lossy_fields
            .iter()
            .any(|field| field == "stream.protocol")
    );

    let body = collect_bytes(bridged.value.expect("byte stream")).await;
    assert!(body.contains("event: message_stop"));
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn custom_stream_loss_policy_can_allow_cross_protocol_strict_mode() {
    let stream = stream::iter(vec![Ok(ChatStreamEvent::text_delta_part("0", "Hello"))]);

    let bridged = super::bridge_chat_stream_to_openai_responses_sse_with_options(
        stream,
        Some(BridgeTarget::AnthropicMessages),
        BridgeOptions::new(BridgeMode::Strict).with_loss_policy(Arc::new(ContinueLossyPolicy)),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_lossy());

    let body = collect_bytes(bridged.value.expect("byte stream")).await;
    assert!(body.contains("event: response.completed"));
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_responses_stream_bridge_maps_gemini_tool_events_to_output_items() {
    let stream = stream::iter(vec![
        Ok(ChatStreamEvent::Custom {
            event_type: "gemini:tool".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "call_1",
                "toolName": "code_execution",
                "providerExecuted": true,
                "input": { "language": "PYTHON", "code": "print(1)" }
            }),
        }),
        Ok(ChatStreamEvent::Custom {
            event_type: "gemini:tool".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "call_1",
                "toolName": "code_execution",
                "providerExecuted": true,
                "result": { "outcome": "OUTCOME_OK", "output": "1" }
            }),
        }),
    ]);

    let bridged = bridge_chat_stream_to_openai_responses_sse(
        stream,
        Some(BridgeTarget::GeminiGenerateContent),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    let body = collect_bytes(bridged.value.expect("byte stream")).await;
    let frames = parse_sse_frames(&body);

    let added = frames
        .iter()
        .find(|(ev, _)| ev == "response.output_item.added");
    let done = frames
        .iter()
        .find(|(ev, _)| ev == "response.output_item.done");
    assert!(
        added.is_some(),
        "tool-call should produce output_item.added"
    );
    assert!(
        done.is_some(),
        "tool-result should produce output_item.done"
    );

    let added_output_index = added
        .and_then(|(_, v)| v.get("output_index").and_then(|v| v.as_u64()))
        .unwrap_or_default();
    let done_output_index = done
        .and_then(|(_, v)| v.get("output_index").and_then(|v| v.as_u64()))
        .unwrap_or_default();

    assert_eq!(added_output_index, done_output_index);
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_responses_stream_bridge_maps_stable_provider_tool_parts_to_output_items() {
    let stream = stream::iter(vec![
        Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::ToolInputStart {
                id: "call_1".to_string(),
                tool_name: "code_execution".to_string(),
                provider_metadata: None,
                provider_executed: Some(true),
                dynamic: Some(false),
                title: None,
            },
        }),
        Ok(ChatStreamEvent::tool_input_delta_part(
            "call_1",
            r#"{"language":"PYTHON","#,
        )),
        Ok(ChatStreamEvent::tool_input_delta_part(
            "call_1",
            r#""code":"print(1)"}"#,
        )),
        Ok(ChatStreamEvent::tool_input_end_part("call_1")),
        Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                tool_call_id: "call_1".to_string(),
                tool_name: "code_execution".to_string(),
                input: String::new(),
                provider_executed: Some(true),
                dynamic: Some(false),
                provider_metadata: None,
            }),
        }),
        Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::ToolResult(ChatStreamToolResult {
                tool_call_id: "call_1".to_string(),
                tool_name: "code_execution".to_string(),
                result: serde_json::json!({ "outcome": "OUTCOME_OK", "output": "1" }),
                is_error: Some(false),
                preliminary: None,
                dynamic: Some(false),
                provider_metadata: None,
            }),
        }),
    ]);

    let bridged = bridge_chat_stream_to_openai_responses_sse(
        stream,
        Some(BridgeTarget::GeminiGenerateContent),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    let body = collect_bytes(bridged.value.expect("byte stream")).await;
    let frames = parse_sse_frames(&body);

    let added = frames
        .iter()
        .find(|(ev, value)| ev == "response.output_item.added" && value["item"]["id"] == "call_1")
        .expect("stable provider tool-call should produce output_item.added");
    assert_eq!(
        added.1["item"]["input"],
        serde_json::json!(r#"{"language":"PYTHON","code":"print(1)"}"#)
    );

    let done = frames
        .iter()
        .find(|(ev, value)| ev == "response.output_item.done" && value["item"]["id"] == "call_1")
        .expect("stable provider tool-result should produce output_item.done");
    assert_eq!(done.1["item"]["status"], serde_json::json!("completed"));
    assert_eq!(done.1["item"]["output"]["output"], serde_json::json!("1"));

    let added_output_index = added
        .1
        .get("output_index")
        .and_then(|value| value.as_u64())
        .unwrap_or_default();
    let done_output_index = done
        .1
        .get("output_index")
        .and_then(|value| value.as_u64())
        .unwrap_or_default();

    assert_eq!(added_output_index, done_output_index);
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_responses_stream_bridge_synthesizes_tool_call_when_only_result_is_available() {
    let stream = stream::iter(vec![Ok(ChatStreamEvent::Custom {
        event_type: "anthropic:tool-result".to_string(),
        data: serde_json::json!({
            "type": "tool-result",
            "toolCallId": "call_2",
            "toolName": "web_search",
            "providerExecuted": true,
            "isError": false,
            "result": [{ "type": "web_search_result", "url": "https://example.com", "title": "Example" }]
        }),
    })]);

    let bridged = bridge_chat_stream_to_openai_responses_sse(
        stream,
        Some(BridgeTarget::AnthropicMessages),
        BridgeMode::BestEffort,
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    let body = collect_bytes(bridged.value.expect("byte stream")).await;
    let frames = parse_sse_frames(&body);

    let added = frames
        .iter()
        .find(|(ev, _)| ev == "response.output_item.added");
    let done = frames
        .iter()
        .find(|(ev, _)| ev == "response.output_item.done");
    assert!(
        added.is_some(),
        "bridged tool-result should synthesize output_item.added"
    );
    assert!(
        done.is_some(),
        "bridged tool-result should produce output_item.done"
    );

    let added_output_index = added
        .and_then(|(_, v)| v.get("output_index").and_then(|v| v.as_u64()))
        .unwrap_or_default();
    let done_output_index = done
        .and_then(|(_, v)| v.get("output_index").and_then(|v| v.as_u64()))
        .unwrap_or_default();

    assert_eq!(added_output_index, done_output_index);
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn stream_bridge_remapper_rewrites_tool_input_parts_and_final_response() {
    let response = ChatResponse {
        id: Some("resp_1".to_string()),
        model: Some("gpt-4.1-mini".to_string()),
        content: MessageContent::MultiModal(vec![ContentPart::tool_call(
            "call_1",
            "weather",
            serde_json::json!({ "city": "Tokyo" }),
            None,
        )]),
        usage: None,
        finish_reason: Some(FinishReason::ToolCalls),
        raw_finish_reason: None,
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
        request: None,
        response: None,
        provider_metadata: None,
    };

    let stream = stream::iter(vec![
        Ok(ChatStreamEvent::tool_input_start_part("call_1", "weather")),
        Ok(ChatStreamEvent::tool_input_delta_part(
            "call_1",
            r#"{"city":"Tokyo"}"#,
        )),
        Ok(ChatStreamEvent::StreamEnd { response }),
    ]);

    let transformed = transform_chat_stream_with_bridge_options(
        stream,
        Some(BridgeTarget::OpenAiResponses),
        BridgeTarget::AnthropicMessages,
        &BridgeOptions::new(BridgeMode::BestEffort)
            .with_route_label("tests.stream.remap")
            .with_primitive_remapper(Arc::new(PrefixStreamRemapper)),
        Some("tests.stream.remap-path".to_string()),
    );

    let events = transformed.collect::<Vec<_>>().await;

    let ChatStreamEvent::Part {
        part: ChatStreamPart::ToolInputStart { id, tool_name, .. },
    } = events[0].as_ref().expect("tool input start")
    else {
        panic!("expected tool input start");
    };
    assert_eq!(id, "gw_call_1");
    assert_eq!(tool_name, "gw_weather");

    let ChatStreamEvent::Part {
        part: ChatStreamPart::ToolInputDelta { id, delta, .. },
    } = events[1].as_ref().expect("tool input delta")
    else {
        panic!("expected tool input delta");
    };
    assert_eq!(id, "gw_call_1");
    assert_eq!(delta, r#"{"city":"Tokyo"}"#);

    let ChatStreamEvent::StreamEnd { response } = events[2].as_ref().expect("stream end") else {
        panic!("expected stream end");
    };
    let MessageContent::MultiModal(parts) = &response.content else {
        panic!("expected multimodal response");
    };
    let ContentPart::ToolCall {
        tool_call_id,
        tool_name,
        ..
    } = &parts[0]
    else {
        panic!("expected tool call part");
    };
    assert_eq!(tool_call_id, "gw_call_1");
    assert_eq!(tool_name, "gw_weather");
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn stream_bridge_remapper_rewrites_stable_part_events_and_drops_stale_replay_raw_item() {
    let response = ChatResponse {
        id: Some("resp_1".to_string()),
        model: Some("gpt-4.1-mini".to_string()),
        content: MessageContent::MultiModal(vec![ContentPart::tool_call(
            "call_1",
            "weather",
            serde_json::json!({ "city": "Tokyo" }),
            None,
        )]),
        usage: None,
        finish_reason: Some(FinishReason::ToolCalls),
        raw_finish_reason: None,
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
        request: None,
        response: None,
        provider_metadata: None,
    };

    let stream = stream::iter(vec![
        Ok(ChatStreamEvent::PartWithReplay {
            part: ChatStreamPart::ToolCall(ChatStreamToolCall {
                tool_call_id: "call_1".to_string(),
                tool_name: "weather".to_string(),
                input: r#"{"city":"Tokyo"}"#.to_string(),
                provider_executed: Some(true),
                dynamic: Some(false),
                provider_metadata: None,
            }),
            replay: ChatStreamReplay::openai_responses(
                Some(7),
                Some(serde_json::json!({
                    "id": "call_1",
                    "type": "custom_tool_call",
                    "status": "in_progress",
                    "name": "weather",
                    "input": "{\"city\":\"Tokyo\"}"
                })),
            )
            .expect("replay"),
        }),
        Ok(ChatStreamEvent::StreamEnd { response }),
    ]);

    let transformed = transform_chat_stream_with_bridge_options(
        stream,
        Some(BridgeTarget::OpenAiResponses),
        BridgeTarget::AnthropicMessages,
        &BridgeOptions::new(BridgeMode::BestEffort)
            .with_route_label("tests.stream.part-remap")
            .with_primitive_remapper(Arc::new(PrefixStreamRemapper)),
        Some("tests.stream.part-remap-path".to_string()),
    );

    let events = transformed.collect::<Vec<_>>().await;

    let ChatStreamEvent::PartWithReplay { part, replay } =
        events[0].as_ref().expect("part with replay")
    else {
        panic!("expected part with replay");
    };

    let ChatStreamPart::ToolCall(call) = part else {
        panic!("expected tool-call part");
    };
    assert_eq!(call.tool_call_id, "gw_call_1");
    assert_eq!(call.tool_name, "gw_weather");

    let openai_replay = replay
        .openai_responses_ref()
        .expect("openai responses replay");
    assert_eq!(openai_replay.output_index, Some(7));
    assert!(
        openai_replay.raw_item.is_none(),
        "stale raw_item should be dropped after semantic remap"
    );

    let ChatStreamEvent::StreamEnd { response } = events[1].as_ref().expect("stream end") else {
        panic!("expected stream end");
    };
    let MessageContent::MultiModal(parts) = &response.content else {
        panic!("expected multimodal response");
    };
    let ContentPart::ToolCall {
        tool_call_id,
        tool_name,
        ..
    } = &parts[0]
    else {
        panic!("expected tool call part");
    };
    assert_eq!(tool_call_id, "gw_call_1");
    assert_eq!(tool_name, "gw_weather");
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn stream_bridge_customization_bundle_can_transform_events_and_remap_tools() {
    let response = ChatResponse {
        id: Some("resp_1".to_string()),
        model: Some("gpt-4.1-mini".to_string()),
        content: MessageContent::MultiModal(vec![ContentPart::tool_call(
            "call_1",
            "weather",
            serde_json::json!({ "city": "Tokyo" }),
            None,
        )]),
        usage: None,
        finish_reason: Some(FinishReason::ToolCalls),
        raw_finish_reason: None,
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
        request: None,
        response: None,
        provider_metadata: None,
    };

    let stream = stream::iter(vec![
        Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::TextDelta {
                id: "txt_1".to_string(),
                delta: "hello".to_string(),
                provider_metadata: None,
            },
        }),
        Ok(ChatStreamEvent::tool_call_part(
            "call_1",
            "weather",
            r#"{"city":"Tokyo"}"#,
        )),
        Ok(ChatStreamEvent::StreamEnd { response }),
    ]);

    let transformed = transform_chat_stream_with_bridge_options(
        stream,
        Some(BridgeTarget::OpenAiResponses),
        BridgeTarget::AnthropicMessages,
        &BridgeOptions::new(BridgeMode::BestEffort)
            .with_route_label("tests.stream.customization")
            .with_customization(Arc::new(CompositeStreamCustomization)),
        Some("tests.stream.custom-path".to_string()),
    );

    let events = transformed.collect::<Vec<_>>().await;

    let ChatStreamEvent::Part { part } = events[0].as_ref().expect("text delta part") else {
        panic!("expected text delta part");
    };
    let ChatStreamPart::TextDelta { delta, .. } = part else {
        panic!("expected text delta part");
    };
    assert_eq!(delta, "HELLO");

    let ChatStreamEvent::Part {
        part: ChatStreamPart::ToolCall(call),
    } = events[1].as_ref().expect("tool call part")
    else {
        panic!("expected tool call part");
    };
    assert_eq!(call.tool_call_id, "bundle_call_1");
    assert_eq!(call.tool_name, "bundle_weather");

    let ChatStreamEvent::StreamEnd { response } = events[2].as_ref().expect("stream end") else {
        panic!("expected stream end");
    };
    let MessageContent::MultiModal(parts) = &response.content else {
        panic!("expected multimodal response");
    };
    let ContentPart::ToolCall {
        tool_call_id,
        tool_name,
        ..
    } = &parts[0]
    else {
        panic!("expected tool call part");
    };
    assert_eq!(tool_call_id, "bundle_call_1");
    assert_eq!(tool_name, "bundle_weather");
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn same_protocol_stream_bridge_allows_strict_mode() {
    let stream = stream::iter(vec![
        Ok(ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("resp_1".to_string()),
                model: Some("gpt-4.1-mini".to_string()),
                created: None,
                provider: "openai".to_string(),
                request_id: None,
                headers: None,
                body: None,
            },
        }),
        Ok(ChatStreamEvent::text_delta_part("0", "Hello")),
    ]);

    let bridged = bridge_chat_stream_to_openai_responses_sse(
        stream,
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::Strict,
    )
    .expect("bridge");

    assert!(
        !bridged.is_rejected(),
        "same-protocol stream should remain allowed in strict mode"
    );
    let body = collect_bytes(bridged.value.expect("byte stream")).await;
    assert!(body.contains("event: response.completed"));
    assert!(body.contains("data: [DONE]"));
}
