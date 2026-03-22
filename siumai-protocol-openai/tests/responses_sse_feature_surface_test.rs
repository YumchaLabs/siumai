#![cfg(all(feature = "openai-standard", feature = "openai-responses"))]

use eventsource_stream::Event;
use siumai_protocol_openai::standards::openai::responses_sse::OpenAiResponsesEventConverter;
use siumai_protocol_openai::streaming::{ChatStreamEvent, SseEventConverter};

fn parse_sse_frames(bytes: &[u8]) -> Vec<(String, serde_json::Value)> {
    let text = String::from_utf8_lossy(bytes);
    text.split("\n\n")
        .filter_map(|chunk| {
            let mut event_name: Option<String> = None;
            let mut data_line: Option<String> = None;
            for line in chunk.lines() {
                if let Some(v) = line.strip_prefix("event: ") {
                    event_name = Some(v.trim().to_string());
                } else if let Some(v) = line.strip_prefix("data: ") {
                    data_line = Some(v.trim().to_string());
                }
            }
            let event_name = event_name?;
            let data_line = data_line?;
            if data_line == "[DONE]" {
                return None;
            }
            let json = serde_json::from_str::<serde_json::Value>(&data_line).ok()?;
            Some((event_name, json))
        })
        .collect()
}

#[test]
fn openai_responses_public_feature_surface_roundtrips_mcp_tool_stream_parts() {
    let encoder = OpenAiResponsesEventConverter::new();

    let tool_call_bytes = encoder
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "mcp_1",
                "toolName": "mcp.web_search_exa",
                "providerExecuted": true,
                "dynamic": true,
                "input": "{\"query\":\"nyc mayor\"}",
            }),
        })
        .expect("serialize mcp tool-call");
    let tool_result_bytes = encoder
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "mcp_1",
                "toolName": "mcp.web_search_exa",
                "providerExecuted": true,
                "dynamic": true,
                "result": {
                    "type": "call",
                    "serverLabel": "exa",
                    "name": "web_search_exa",
                    "arguments": "{\"query\":\"nyc mayor\"}",
                    "output": { "hits": 3 }
                }
            }),
        })
        .expect("serialize mcp tool-result");

    let call_frames = parse_sse_frames(&tool_call_bytes);
    let result_frames = parse_sse_frames(&tool_result_bytes);

    assert!(call_frames.iter().any(|(ev, payload)| {
        ev == "response.output_item.added"
            && payload["item"]["type"] == serde_json::json!("mcp_call")
    }));
    assert!(call_frames.iter().any(|(ev, payload)| {
        ev == "response.mcp_call_arguments.done"
            && payload["arguments"] == serde_json::json!("{\"query\":\"nyc mayor\"}")
    }));
    assert!(result_frames.iter().any(|(ev, payload)| {
        ev == "response.output_item.done"
            && payload["item"]["type"] == serde_json::json!("mcp_call")
            && payload["item"]["output"]["hits"] == serde_json::json!(3)
    }));

    let decoder = OpenAiResponsesEventConverter::new();
    let mut events = Vec::new();
    for (index, (event_name, payload)) in call_frames
        .into_iter()
        .chain(result_frames.into_iter())
        .enumerate()
    {
        let out = futures::executor::block_on(decoder.convert_event(Event {
            event: event_name,
            data: serde_json::to_string(&payload).expect("serialize payload"),
            id: index.to_string(),
            retry: None,
        }));

        for item in out {
            events.push(item.expect("decode stream frame"));
        }
    }

    let provider_tool_calls = events
        .iter()
        .filter_map(|event| match event {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::json!("tool-call"))
                    && data.get("toolName") == Some(&serde_json::json!("mcp.web_search_exa"))
                    && data.get("providerExecuted") == Some(&serde_json::json!(true)) =>
            {
                Some(data)
            }
            _ => None,
        })
        .count();
    let provider_tool_results = events
        .iter()
        .filter_map(|event| match event {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::json!("tool-result"))
                    && data.get("toolName") == Some(&serde_json::json!("mcp.web_search_exa"))
                    && data.get("providerExecuted") == Some(&serde_json::json!(true)) =>
            {
                Some(data)
            }
            _ => None,
        })
        .count();

    assert_eq!(provider_tool_calls, 1);
    assert_eq!(provider_tool_results, 1);
}
