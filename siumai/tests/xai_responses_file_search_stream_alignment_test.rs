#![cfg(feature = "xai")]

use siumai::experimental::streaming::SseEventConverter;
use siumai::prelude::unified::*;

fn stream_events_by_type(events: &[ChatStreamEvent], kind: &str) -> Vec<serde_json::Value> {
    let stable_parts: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            ChatStreamEvent::Part { part } | ChatStreamEvent::PartWithReplay { part, .. } => {
                Some(serde_json::to_value(part).expect("serialize stream part"))
            }
            _ => None,
        })
        .filter(|value| value.get("type").and_then(|v| v.as_str()) == Some(kind))
        .collect();
    if !stable_parts.is_empty() {
        return stable_parts;
    }

    events
        .iter()
        .filter_map(|event| match event {
            ChatStreamEvent::Custom { data, .. } => Some(data.clone()),
            _ => None,
        })
        .filter(|value| value.get("type").and_then(|v| v.as_str()) == Some(kind))
        .collect()
}

#[test]
fn xai_responses_file_search_stream_emits_vercel_aligned_tool_call_and_result() {
    let tools = vec![Tool::ProviderDefined(ProviderDefinedTool::new(
        "xai.file_search",
        "file_search",
    ))];

    let conv =
        siumai_provider_xai::standards::xai::responses_sse::XaiResponsesEventConverter::new()
            .with_request_tools(&tools);

    let lines = vec![
        serde_json::json!({
            "type": "response.created",
            "response": {
                "id": "resp_123",
                "object": "response",
                "model": "grok-4-fast-non-reasoning",
                "status": "in_progress",
                "output": []
            }
        })
        .to_string(),
        serde_json::json!({
            "type": "response.output_item.added",
            "item": {
                "type": "file_search_call",
                "id": "fs_stream_123",
                "status": "in_progress",
                "queries": ["search query"],
                "results": null
            },
            "output_index": 0
        })
        .to_string(),
        serde_json::json!({
            "type": "response.output_item.done",
            "item": {
                "type": "file_search_call",
                "id": "fs_stream_123",
                "status": "completed",
                "queries": ["search query"],
                "results": [
                    {
                        "file_id": "file_abc",
                        "filename": "doc.txt",
                        "score": 0.9,
                        "text": "Found text content"
                    }
                ]
            },
            "output_index": 0
        })
        .to_string(),
        serde_json::json!({
            "type": "response.done",
            "response": {
                "id": "resp_123",
                "object": "response",
                "status": "completed",
                "output": [],
                "usage": { "input_tokens": 10, "output_tokens": 5 }
            }
        })
        .to_string(),
    ];

    let mut events = Vec::new();
    for (i, line) in lines.into_iter().enumerate() {
        let ev = eventsource_stream::Event {
            event: String::new(),
            data: line,
            id: i.to_string(),
            retry: None,
        };

        for item in futures::executor::block_on(conv.convert_event(ev)) {
            events.push(item.expect("convert xai file search stream chunk"));
        }
    }

    while let Some(item) = conv.handle_stream_end() {
        events.push(item.expect("finalize xai file search stream"));
    }

    let tool_inputs_start = stream_events_by_type(&events, "tool-input-start");
    let tool_inputs_delta = stream_events_by_type(&events, "tool-input-delta");
    let tool_inputs_end = stream_events_by_type(&events, "tool-input-end");
    let tool_calls = stream_events_by_type(&events, "tool-call");
    let tool_results = stream_events_by_type(&events, "tool-result");

    assert_eq!(tool_inputs_start.len(), 1, "expected one tool-input-start");
    assert_eq!(tool_inputs_delta.len(), 1, "expected one tool-input-delta");
    assert_eq!(tool_inputs_end.len(), 1, "expected one tool-input-end");
    assert_eq!(tool_calls.len(), 1, "expected one tool-call");
    assert_eq!(tool_results.len(), 1, "expected one tool-result");

    let tool_id = tool_inputs_start[0]["id"]
        .as_str()
        .expect("tool-input-start id");
    assert_eq!(
        tool_inputs_start[0]["toolName"],
        serde_json::json!("file_search")
    );
    assert!(
        tool_inputs_start[0].get("providerExecuted").is_none(),
        "xAI file_search tool-input-start should omit providerExecuted"
    );

    assert_eq!(tool_inputs_delta[0]["id"], serde_json::json!(tool_id));
    assert_eq!(tool_inputs_delta[0]["delta"], serde_json::json!(""));
    assert_eq!(tool_inputs_end[0]["id"], serde_json::json!(tool_id));

    assert_eq!(tool_calls[0]["toolCallId"], serde_json::json!(tool_id));
    assert_eq!(tool_calls[0]["toolName"], serde_json::json!("file_search"));
    assert_eq!(tool_calls[0]["input"], serde_json::json!(""));
    assert_eq!(tool_calls[0]["providerExecuted"], serde_json::json!(true));

    assert_eq!(tool_results[0]["toolCallId"], serde_json::json!(tool_id));
    assert_eq!(
        tool_results[0]["toolName"],
        serde_json::json!("file_search")
    );
    assert_eq!(
        tool_results[0]["result"]["queries"],
        serde_json::json!(["search query"])
    );
    assert_eq!(
        tool_results[0]["result"]["results"][0]["fileId"],
        serde_json::json!("file_abc")
    );
    assert!(
        tool_results[0]["result"]["results"][0]
            .get("file_id")
            .is_none(),
        "xAI file_search stream results should normalize file_id to fileId"
    );
    assert!(
        tool_results[0]["result"].get("status").is_none(),
        "xAI file_search stream result should omit status"
    );
}
