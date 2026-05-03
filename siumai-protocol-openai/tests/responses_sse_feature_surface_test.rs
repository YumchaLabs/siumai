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
    for (index, (event_name, payload)) in call_frames.into_iter().chain(result_frames).enumerate() {
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
        .filter_map(|event| match event.part_ref() {
            Some(siumai_protocol_openai::types::ChatStreamPart::ToolCall(call))
                if call.tool_name == "mcp.web_search_exa"
                    && call.provider_executed == Some(true) =>
            {
                Some(call)
            }
            _ => None,
        })
        .count();
    let provider_tool_results = events
        .iter()
        .filter_map(|event| match event.part_ref() {
            Some(siumai_protocol_openai::types::ChatStreamPart::ToolResult(result))
                if result.tool_name == "mcp.web_search_exa" =>
            {
                Some(result)
            }
            _ => None,
        })
        .count();

    assert_eq!(provider_tool_calls, 1);
    assert_eq!(provider_tool_results, 1);
}

#[test]
fn openai_responses_public_feature_surface_roundtrips_tool_search_without_raw_items() {
    let encoder = OpenAiResponsesEventConverter::new();

    let tool_call_bytes = encoder
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "openai:tool-call".to_string(),
            data: serde_json::json!({
                "type": "tool-call",
                "toolCallId": "call_final",
                "toolName": "toolSearch",
                "input": "{\"arguments\":{\"goal\":\"Find weather\"},\"call_id\":\"call_final\"}",
                "providerMetadata": {
                    "openai": { "itemId": "tsc_client_1" }
                }
            }),
        })
        .expect("serialize tool_search tool-call");
    let tool_result_bytes = encoder
        .serialize_event(&ChatStreamEvent::Custom {
            event_type: "openai:tool-result".to_string(),
            data: serde_json::json!({
                "type": "tool-result",
                "toolCallId": "call_final",
                "toolName": "toolSearch",
                "result": {
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_weather",
                            "parameters": { "type": "object" }
                        }
                    ]
                },
                "providerMetadata": {
                    "openai": { "itemId": "tso_client_1" }
                }
            }),
        })
        .expect("serialize tool_search tool-result");

    let call_frames = parse_sse_frames(&tool_call_bytes);
    let result_frames = parse_sse_frames(&tool_result_bytes);

    assert!(call_frames.iter().any(|(ev, payload)| {
        ev == "response.output_item.done"
            && payload["item"]["type"] == serde_json::json!("tool_search_call")
            && payload["item"]["id"] == serde_json::json!("tsc_client_1")
            && payload["item"]["execution"] == serde_json::json!("client")
            && payload["item"]["call_id"] == serde_json::json!("call_final")
    }));
    assert!(result_frames.iter().any(|(ev, payload)| {
        ev == "response.output_item.done"
            && payload["item"]["type"] == serde_json::json!("tool_search_output")
            && payload["item"]["id"] == serde_json::json!("tso_client_1")
            && payload["item"]["execution"] == serde_json::json!("client")
            && payload["item"]["tools"][0]["name"] == serde_json::json!("get_weather")
    }));

    let decoder = OpenAiResponsesEventConverter::new();
    let mut events = Vec::new();
    for (index, (event_name, payload)) in call_frames.into_iter().chain(result_frames).enumerate() {
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

    let tool_search_calls = events
        .iter()
        .filter_map(|event| match event.part_ref() {
            Some(siumai_protocol_openai::types::ChatStreamPart::ToolCall(call))
                if call.tool_name == "toolSearch" && call.tool_call_id == "call_final" =>
            {
                Some(call)
            }
            _ => None,
        })
        .count();
    let tool_search_results = events
        .iter()
        .filter_map(|event| match event.part_ref() {
            Some(siumai_protocol_openai::types::ChatStreamPart::ToolResult(result))
                if result.tool_name == "toolSearch"
                    && result.result["tools"][0]["name"] == serde_json::json!("get_weather") =>
            {
                Some(result)
            }
            _ => None,
        })
        .count();

    assert_eq!(tool_search_calls, 1);
    assert_eq!(tool_search_results, 1);
}
