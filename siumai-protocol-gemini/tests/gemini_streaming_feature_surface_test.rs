#![cfg(feature = "google")]

use eventsource_stream::Event;
use siumai_protocol_gemini::standards::gemini::streaming::GeminiEventConverter;
use siumai_protocol_gemini::standards::gemini::types::GeminiConfig;
use siumai_protocol_gemini::streaming::{ChatStreamEvent, SseEventConverter};

fn create_test_config() -> GeminiConfig {
    GeminiConfig::new("test-key")
}

fn stream_part(
    event: &ChatStreamEvent,
) -> Option<siumai_protocol_gemini::streaming::LanguageModelV3StreamPart> {
    siumai_protocol_gemini::streaming::LanguageModelV3StreamPart::try_from_chat_event(event)
}

fn google_provider_metadata(
    value: serde_json::Value,
) -> siumai_protocol_gemini::types::StreamProviderMetadata {
    let mut provider_metadata = siumai_protocol_gemini::types::StreamProviderMetadata::new();
    provider_metadata.insert("google".to_string(), value);
    provider_metadata
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
        .serialize_event(&ChatStreamEvent::Part {
            part: siumai_protocol_gemini::types::ChatStreamPart::ToolCall(
                siumai_protocol_gemini::types::ChatStreamToolCall {
                    tool_call_id: "call_1".to_string(),
                    tool_name: "code_execution".to_string(),
                    input: r#"{"language":"PYTHON","code":"print(1)"}"#.to_string(),
                    provider_executed: Some(true),
                    dynamic: None,
                    provider_metadata: None,
                },
            ),
        })
        .expect("serialize tool-call");
    let tool_result_bytes = encoder
        .serialize_event(&ChatStreamEvent::Part {
            part: siumai_protocol_gemini::types::ChatStreamPart::ToolResult(
                siumai_protocol_gemini::types::ChatStreamToolResult {
                    tool_call_id: "call_1".to_string(),
                    tool_name: "code_execution".to_string(),
                    result: serde_json::json!({
                        "outcome": "OUTCOME_OK",
                        "output": "1"
                    }),
                    is_error: None,
                    preliminary: None,
                    dynamic: None,
                    provider_metadata: None,
                },
            ),
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
                stream_part(event),
                Some(siumai_protocol_gemini::streaming::LanguageModelV3StreamPart::ToolCall(call))
                    if call.tool_name == "code_execution"
                        && call.provider_executed == Some(true)
            )
        })
        .count();
    let provider_tool_results = events
        .iter()
        .filter(|event| {
            matches!(
                stream_part(event),
                Some(siumai_protocol_gemini::streaming::LanguageModelV3StreamPart::ToolResult(result))
                    if result.tool_name == "code_execution"
            )
        })
        .count();

    assert_eq!(provider_tool_calls, 1);
    assert_eq!(provider_tool_results, 1);
}

#[tokio::test]
async fn gemini_public_feature_surface_preserves_reasoning_metadata_from_typed_delta() {
    let encoder = GeminiEventConverter::new(create_test_config());

    let start_bytes = encoder
        .serialize_event(&ChatStreamEvent::Part {
            part: siumai_protocol_gemini::types::ChatStreamPart::ReasoningStart {
                id: "rs_1".to_string(),
                provider_metadata: Some(google_provider_metadata(serde_json::json!({
                    "thoughtSignature": "stream_sig"
                }))),
            },
        })
        .expect("serialize reasoning-start");
    let delta_bytes = encoder
        .serialize_event(&ChatStreamEvent::Part {
            part: siumai_protocol_gemini::types::ChatStreamPart::ReasoningDelta {
                id: "rs_1".to_string(),
                delta: "thinking...".to_string(),
                provider_metadata: Some(google_provider_metadata(serde_json::json!({
                    "thoughtSignature": "stream_sig"
                }))),
            },
        })
        .expect("serialize reasoning-delta");
    assert!(
        start_bytes.is_empty(),
        "reasoning-start should not emit a standalone chunk"
    );

    let frames = parse_sse_json_frames(&delta_bytes);
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
                stream_part(event),
                Some(siumai_protocol_gemini::streaming::LanguageModelV3StreamPart::ReasoningDelta {
                    delta,
                    provider_metadata,
                    ..
                }) if delta == "thinking..."
                    && provider_metadata.as_ref()
                        .and_then(|meta| meta.get("google"))
                        .and_then(|meta| meta.get("thoughtSignature"))
                        == Some(&serde_json::json!("stream_sig"))
            )
        })
        .count();

    assert_eq!(thinking_deltas, 0);
    assert_eq!(reasoning_deltas, 1);
}
