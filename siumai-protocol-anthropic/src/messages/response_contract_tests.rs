use serde_json::json;
use siumai_core::{
    ApiModeId, ContentPart, ErrorKind, FinishReason, LanguageStreamDecoder, LanguageStreamEvent,
    ModelId, ProtocolId, ProviderId, ProviderScope, ReplayDomain, ReplayDomainId, StreamTerminal,
    UsageValue,
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

    assert_eq!(response.finish_reason(), &FinishReason::Refusal);
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
    assert_eq!(terminal.finish_reason(), &FinishReason::Refusal);
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
