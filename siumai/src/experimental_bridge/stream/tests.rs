use std::sync::Arc;

use futures_util::{StreamExt, stream};
use siumai_core::bridge::{
    BridgeMode, BridgeOptions, BridgeTarget, StreamBridgeContext, StreamBridgeHook,
};
use siumai_core::streaming::ChatByteStream;
use siumai_core::types::{
    ChatResponse, ChatStreamEvent, FinishReason, MessageContent, ResponseMetadata,
};

#[cfg(feature = "openai")]
use super::bridge_chat_stream_to_openai_responses_sse;
#[cfg(feature = "anthropic")]
use super::{
    bridge_chat_stream_to_anthropic_messages_sse,
    bridge_chat_stream_to_anthropic_messages_sse_with_options,
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
