#![cfg(feature = "google")]

use eventsource_stream::Event;
use siumai_protocol_gemini::standards::gemini::streaming::GeminiEventConverter;
use siumai_protocol_gemini::standards::gemini::types::GeminiConfig;
use siumai_protocol_gemini::streaming::{ChatStreamEvent, SseEventConverter};

fn create_test_config() -> GeminiConfig {
    GeminiConfig::new("test-key")
}

fn parse_sse_json_frames(bytes: &[u8]) -> Vec<serde_json::Value> {
    let text = String::from_utf8_lossy(bytes);
    text.split("\n\n")
        .filter_map(|chunk| {
            let line = chunk
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .map(str::trim)?;
            if line.is_empty() || line == "[DONE]" {
                return None;
            }
            serde_json::from_str::<serde_json::Value>(line).ok()
        })
        .collect()
}

async fn decode_frames(
    converter: &GeminiEventConverter,
    frames: &[serde_json::Value],
) -> Vec<ChatStreamEvent> {
    let mut events = Vec::new();

    for (index, frame) in frames.iter().enumerate() {
        let out = converter
            .convert_event(Event {
                event: String::new(),
                data: serde_json::to_string(frame).expect("serialize frame"),
                id: index.to_string(),
                retry: None,
            })
            .await;

        for item in out {
            events.push(item.expect("decode stream frame"));
        }
    }

    events
}

#[tokio::test]
async fn gemini_public_feature_surface_roundtrips_provider_executed_code_execution_parts() {
    let encoder = GeminiEventConverter::new(create_test_config());

    let tool_call_bytes = encoder
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "call_1",
                "toolName": "code_execution",
                "providerExecuted": true,
                "input": {
                    "language": "PYTHON",
                    "code": "print(1)"
                }
            }),
        })
        .expect("serialize tool-call");
    let tool_result_bytes = encoder
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "call_1",
                "toolName": "code_execution",
                "providerExecuted": true,
                "result": {
                    "outcome": "OUTCOME_OK",
                    "output": "1"
                }
            }),
        })
        .expect("serialize tool-result");

    let call_frames = parse_sse_json_frames(&tool_call_bytes);
    let result_frames = parse_sse_json_frames(&tool_result_bytes);

    assert_eq!(call_frames.len(), 1);
    assert_eq!(
        call_frames[0]["candidates"][0]["content"]["parts"][0]["executableCode"]["language"],
        serde_json::json!("PYTHON")
    );
    assert_eq!(
        call_frames[0]["candidates"][0]["content"]["parts"][0]["executableCode"]["code"],
        serde_json::json!("print(1)")
    );
    assert_eq!(result_frames.len(), 1);
    assert_eq!(
        result_frames[0]["candidates"][0]["content"]["parts"][0]["codeExecutionResult"]["outcome"],
        serde_json::json!("OUTCOME_OK")
    );
    assert_eq!(
        result_frames[0]["candidates"][0]["content"]["parts"][0]["codeExecutionResult"]["output"],
        serde_json::json!("1")
    );

    let decoder = GeminiEventConverter::new(create_test_config());
    let events = decode_frames(
        &decoder,
        &call_frames
            .into_iter()
            .chain(result_frames.into_iter())
            .collect::<Vec<_>>(),
    )
    .await;

    let provider_tool_calls = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                ChatStreamEvent::Custom { event_type, data }
                    if event_type == "gemini:tool"
                        && data.get("type") == Some(&serde_json::json!("tool-call"))
                        && data.get("toolName") == Some(&serde_json::json!("code_execution"))
                        && data.get("providerExecuted") == Some(&serde_json::json!(true))
            )
        })
        .count();
    let provider_tool_results = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                ChatStreamEvent::Custom { event_type, data }
                    if event_type == "gemini:tool"
                        && data.get("type") == Some(&serde_json::json!("tool-result"))
                        && data.get("toolName") == Some(&serde_json::json!("code_execution"))
                        && data.get("providerExecuted") == Some(&serde_json::json!(true))
            )
        })
        .count();

    assert_eq!(provider_tool_calls, 1);
    assert_eq!(provider_tool_results, 1);
}

#[tokio::test]
async fn gemini_public_feature_surface_preserves_reasoning_metadata_without_duplicate_chunk() {
    let encoder = GeminiEventConverter::new(create_test_config());

    let start_bytes = encoder
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "openai:reasoning-start".to_string(),
            data: serde_json::json!({
                "type": "reasoning-start",
                "id": "rs_1",
                "providerMetadata": {
                    "google": {
                        "thoughtSignature": "stream_sig"
                    }
                }
            }),
        })
        .expect("serialize reasoning-start");
    let delta_bytes = encoder
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "openai:reasoning-delta".to_string(),
            data: serde_json::json!({
                "type": "reasoning-delta",
                "id": "rs_1",
                "delta": "thinking...",
                "providerMetadata": {
                    "google": {
                        "thoughtSignature": "stream_sig"
                    }
                }
            }),
        })
        .expect("serialize reasoning-delta");
    let thinking_bytes = encoder
        .serialize_event(&ChatStreamEvent::ThinkingDelta {
            delta: "thinking...".to_string(),
        })
        .expect("serialize ThinkingDelta");

    assert!(
        start_bytes.is_empty(),
        "reasoning-start should not emit a standalone chunk"
    );
    assert!(
        delta_bytes.is_empty(),
        "reasoning-delta should wait for the paired ThinkingDelta"
    );

    let frames = parse_sse_json_frames(&thinking_bytes);
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0]["candidates"][0]["content"]["parts"][0]["thought"],
        serde_json::json!(true)
    );
    assert_eq!(
        frames[0]["candidates"][0]["content"]["parts"][0]["thoughtSignature"],
        serde_json::json!("stream_sig")
    );
    assert_eq!(
        frames[0]["candidates"][0]["content"]["parts"][0]["text"],
        serde_json::json!("thinking...")
    );

    let decoder = GeminiEventConverter::new(create_test_config());
    let events = decode_frames(&decoder, &frames).await;

    let thinking_deltas = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                ChatStreamEvent::ThinkingDelta { delta } if delta == "thinking..."
            )
        })
        .count();
    let reasoning_deltas = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                ChatStreamEvent::Custom { event_type, data }
                    if event_type == "gemini:reasoning"
                        && data.get("type") == Some(&serde_json::json!("reasoning-delta"))
                        && data.get("providerMetadata")
                            == Some(&serde_json::json!({
                                "google": {
                                    "thoughtSignature": "stream_sig"
                                }
                            }))
            )
        })
        .count();

    assert_eq!(thinking_deltas, 1);
    assert_eq!(reasoning_deltas, 1);
}
