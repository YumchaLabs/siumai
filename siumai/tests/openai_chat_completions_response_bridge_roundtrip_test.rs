#![cfg(feature = "openai")]

use serde_json::Value;
use siumai::experimental::bridge::{
    BridgeMode, BridgeTarget, bridge_chat_response_to_openai_chat_completions_json_value,
};
use siumai::experimental::encoding::JsonEncodeOptions;
use siumai::experimental::execution::transformers::response::ResponseTransformer;

fn normalize_json(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for inner in map.values_mut() {
                normalize_json(inner);
            }

            let keys: Vec<String> = map
                .iter()
                .filter_map(|(key, inner)| {
                    if inner.is_null() {
                        return Some(key.clone());
                    }
                    if let Value::Object(obj) = inner
                        && obj.is_empty()
                    {
                        return Some(key.clone());
                    }
                    None
                })
                .collect();
            for key in keys {
                map.remove(&key);
            }
        }
        Value::Array(items) => {
            for inner in items.iter_mut() {
                normalize_json(inner);
            }
        }
        _ => {}
    }
}

fn normalize_chat_response_json(response: &siumai::prelude::unified::ChatResponse) -> Value {
    let mut value = serde_json::to_value(response).expect("serialize chat response");
    normalize_json(&mut value);
    value
}

#[test]
fn openai_chat_completions_response_bridge_roundtrip_preserves_normalized_projection() {
    let raw = serde_json::json!({
        "id": "chatcmpl_123",
        "object": "chat.completion",
        "created": 1730000000u64,
        "model": "gpt-4.1-mini",
        "system_fingerprint": "fp_123",
        "service_tier": "priority",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": "{\"city\":\"Tokyo\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "prompt_tokens_details": {
                "cached_tokens": 3,
                "audio_tokens": 2
            },
            "completion_tokens_details": {
                "reasoning_tokens": 4,
                "audio_tokens": 1,
                "accepted_prediction_tokens": 5,
                "rejected_prediction_tokens": 6
            }
        }
    });

    let tx =
        siumai::experimental::standards::openai::transformers::response::OpenAiResponseTransformer;
    let response = tx
        .transform_chat_response(&raw)
        .expect("transform raw chat completion");

    assert_eq!(response.system_fingerprint.as_deref(), Some("fp_123"));
    assert_eq!(response.service_tier.as_deref(), Some("priority"));

    let bridged = bridge_chat_response_to_openai_chat_completions_json_value(
        &response,
        Some(BridgeTarget::OpenAiChatCompletions),
        BridgeMode::Strict,
        JsonEncodeOptions::default(),
    )
    .expect("bridge");

    assert!(!bridged.is_rejected());
    assert!(bridged.report.is_exact());

    let bridged_json = bridged.value.expect("bridged json");
    assert_eq!(
        bridged_json["system_fingerprint"],
        serde_json::json!("fp_123")
    );
    assert_eq!(bridged_json["service_tier"], serde_json::json!("priority"));
    assert_eq!(
        bridged_json["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        serde_json::json!("weather")
    );
    assert_eq!(
        bridged_json["usage"]["completion_tokens_details"]["accepted_prediction_tokens"],
        serde_json::json!(5)
    );

    let roundtripped = tx
        .transform_chat_response(&bridged_json)
        .expect("transform bridged chat completion");

    assert_eq!(
        normalize_chat_response_json(&roundtripped),
        normalize_chat_response_json(&response)
    );
}
