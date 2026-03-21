use std::sync::Arc;

use futures_util::{StreamExt, stream};
use siumai_core::bridge::{
    BridgeLossAction, BridgeLossPolicy, BridgeMode, BridgeOptions, BridgePrimitiveContext,
    BridgePrimitiveRemapper, BridgeTarget, RequestBridgeContext, ResponseBridgeContext,
    StreamBridgeContext, StreamBridgeHook,
};
use siumai_core::streaming::ChatByteStream;
use siumai_core::types::{
    ChatResponse, ChatStreamEvent, ContentPart, FinishReason, MessageContent, ResponseMetadata,
};

#[cfg(feature = "anthropic")]
use super::{
    bridge_chat_stream_to_anthropic_messages_sse,
    bridge_chat_stream_to_anthropic_messages_sse_with_options,
};
#[cfg(feature = "openai")]
use super::{
    bridge_chat_stream_to_openai_responses_sse, transform_chat_stream_with_bridge_options,
};

struct UppercaseStreamHook;

impl StreamBridgeHook for UppercaseStreamHook {
    fn map_event(
        &self,
        _ctx: &StreamBridgeContext,
        event: ChatStreamEvent,
    ) -> Vec<ChatStreamEvent> {
        match event {
            ChatStreamEvent::ContentDelta { delta, index } => vec![ChatStreamEvent::ContentDelta {
                delta: delta.to_uppercase(),
                index,
            }],
            other => vec![other],
        }
    }
}

struct ContinueLossyPolicy;

impl BridgeLossPolicy for ContinueLossyPolicy {
    fn request_action(
        &self,
        _ctx: &RequestBridgeContext,
        _report: &siumai_core::bridge::BridgeReport,
    ) -> BridgeLossAction {
        BridgeLossAction::Continue
    }

    fn response_action(
        &self,
        _ctx: &ResponseBridgeContext,
        _report: &siumai_core::bridge::BridgeReport,
    ) -> BridgeLossAction {
        BridgeLossAction::Continue
    }

    fn stream_action(
        &self,
        _ctx: &StreamBridgeContext,
        _report: &siumai_core::bridge::BridgeReport,
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

async fn collect_bytes(mut stream: ChatByteStream) -> String {
    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("stream chunk");
        out.extend_from_slice(&chunk);
    }
    String::from_utf8(out).expect("utf8")
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
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
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
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
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
            },
        }),
        Ok(ChatStreamEvent::ContentDelta {
            delta: "Hello".to_string(),
            index: None,
        }),
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
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
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
            },
        }),
        Ok(ChatStreamEvent::ContentDelta {
            delta: "hello".to_string(),
            index: None,
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
            },
        }),
        Ok(ChatStreamEvent::ContentDelta {
            delta: "Hello".to_string(),
            index: None,
        }),
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
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
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
            },
        }),
        Ok(ChatStreamEvent::ContentDelta {
            delta: "Hello".to_string(),
            index: None,
        }),
        Ok(ChatStreamEvent::ThinkingDelta {
            delta: "Thinking".to_string(),
        }),
        Ok(ChatStreamEvent::ContentDelta {
            delta: " world".to_string(),
            index: None,
        }),
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
            },
        }),
        Ok(ChatStreamEvent::ContentDelta {
            delta: "Hello".to_string(),
            index: None,
        }),
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

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn cross_protocol_stream_bridge_rejects_in_strict_mode() {
    let stream = stream::iter(vec![Ok(ChatStreamEvent::ContentDelta {
        delta: "Hello".to_string(),
        index: None,
    })]);

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
    let stream = stream::iter(vec![Ok(ChatStreamEvent::ContentDelta {
        delta: "Hello".to_string(),
        index: None,
    })]);

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
    let stream = stream::iter(vec![Ok(ChatStreamEvent::ContentDelta {
        delta: "Hello".to_string(),
        index: None,
    })]);

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
async fn stream_bridge_remapper_rewrites_tool_delta_and_final_response() {
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
        audio: None,
        system_fingerprint: None,
        service_tier: None,
        warnings: None,
        provider_metadata: None,
    };

    let stream = stream::iter(vec![
        Ok(ChatStreamEvent::ToolCallDelta {
            id: "call_1".to_string(),
            function_name: Some("weather".to_string()),
            arguments_delta: Some(r#"{"city":"Tokyo"}"#.to_string()),
            index: Some(0),
        }),
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

    let ChatStreamEvent::ToolCallDelta {
        id, function_name, ..
    } = events[0].as_ref().expect("tool delta")
    else {
        panic!("expected tool delta");
    };
    assert_eq!(id, "gw_call_1");
    assert_eq!(function_name.as_deref(), Some("gw_weather"));

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
async fn same_protocol_stream_bridge_allows_strict_mode() {
    let stream = stream::iter(vec![
        Ok(ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: Some("resp_1".to_string()),
                model: Some("gpt-4.1-mini".to_string()),
                created: None,
                provider: "openai".to_string(),
                request_id: None,
            },
        }),
        Ok(ChatStreamEvent::ContentDelta {
            delta: "Hello".to_string(),
            index: None,
        }),
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
