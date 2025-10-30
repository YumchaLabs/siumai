//! OpenAI Responses request mapping tests aligned with official API semantics.
//!
//! These tests verify message-to-input item conversions according to the
//! Responses API: tool role → function_call_output, and assistant tool_calls →
//! tool_use parts with structured input.
//!
//! References:
//! - OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses

use siumai::execution::transformers::request::RequestTransformer;

#[test]
fn tool_role_message_maps_to_function_call_output_item() {
    use siumai::providers::openai::transformers::OpenAiResponsesRequestTransformer;
    use siumai::types::{ChatMessage, ChatRequest, CommonParams};

    // Tool role message with tool result produces a function_call_output item
    let msg = ChatMessage::tool_result_text("call_1", "some_tool", "result: 42").build();

    // Build request
    let req = ChatRequest::builder()
        .messages(vec![msg.clone()])
        .common_params(CommonParams {
            model: "gpt-4o-mini".into(),
            ..Default::default()
        })
        .build();
    let json = OpenAiResponsesRequestTransformer
        .transform_chat(&req)
        .expect("map");
    let input = json["input"].as_array().expect("input array");
    assert_eq!(input.len(), 1);
    let item = &input[0];
    assert_eq!(item["type"], "function_call_output");
    assert_eq!(item["call_id"], "call_1");
    assert_eq!(item["output"], "result: 42");
}

#[test]
fn assistant_tool_calls_map_to_tool_use_parts_and_text() {
    use serde_json::json;
    use siumai::providers::openai::transformers::OpenAiResponsesRequestTransformer;
    use siumai::types::{ChatMessage, ChatRequest, CommonParams, ContentPart};

    // Assistant message with both text and tool_calls
    let msg = ChatMessage::assistant_with_content(vec![
        ContentPart::text("Here is a tool call"),
        ContentPart::tool_call("t1", "lookup", json!({"q": "x"}), None),
    ])
    .build();

    let req = ChatRequest::builder()
        .messages(vec![msg])
        .common_params(CommonParams {
            model: "gpt-5-mini".into(),
            ..Default::default()
        })
        .build();

    let json = OpenAiResponsesRequestTransformer
        .transform_chat(&req)
        .expect("map");
    let input = json["input"].as_array().expect("input array");
    assert_eq!(input.len(), 1);
    let message = &input[0];
    assert_eq!(message["role"], "assistant");
    let parts = message["content"].as_array().expect("content parts");
    // Should contain an input_text and a tool_use part
    assert!(
        parts
            .iter()
            .any(|p| p["type"] == "input_text" && p["text"] == "Here is a tool call")
    );
    let tool_part = parts
        .iter()
        .find(|p| p["type"] == "tool_use")
        .expect("tool_use part");
    assert_eq!(tool_part["id"], "t1");
    assert_eq!(tool_part["name"], "lookup");
    assert!(tool_part["input"].is_object());
}
