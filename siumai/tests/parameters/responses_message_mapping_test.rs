//! OpenAI Responses request mapping tests aligned with official API semantics.
//!
//! These tests verify message-to-input item conversions according to the
//! Responses API: tool role → function_call_output, and assistant tool_calls →
//! tool_use parts with structured input.
//!
//! References:
//! - OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses

use siumai::transformers::request::RequestTransformer;

#[test]
fn tool_role_message_maps_to_function_call_output_item() {
    use siumai::providers::openai::transformers::OpenAiResponsesRequestTransformer;
    use siumai::types::{ChatMessage, ChatRequest, CommonParams, MessageContent, MessageRole};

    // Tool role message with tool_call_id produces a function_call_output item
    let msg = ChatMessage {
        role: MessageRole::Tool,
        content: MessageContent::Text("result: 42".into()),
        metadata: Default::default(),
        tool_calls: None,
        tool_call_id: Some("call_1".into()),
    };
    // Build request
    let req = ChatRequest {
        messages: vec![msg.clone()],
        tools: None,
        common_params: CommonParams {
            model: "gpt-4o-mini".into(),
            ..Default::default()
        },
        provider_params: None,
        http_config: None,
        web_search: None,
        stream: false,
    };
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
    use siumai::providers::openai::transformers::OpenAiResponsesRequestTransformer;
    use siumai::types::{
        ChatMessage, ChatRequest, CommonParams, FunctionCall, MessageContent, MessageRole, ToolCall,
    };

    // Assistant message with both text and tool_calls
    let tool_call = ToolCall {
        id: "t1".into(),
        r#type: "function".into(),
        function: Some(FunctionCall {
            name: "lookup".into(),
            arguments: "{\"q\":\"x\"}".into(),
        }),
    };
    let msg = ChatMessage {
        role: MessageRole::Assistant,
        content: MessageContent::Text("Here is a tool call".into()),
        metadata: Default::default(),
        tool_calls: Some(vec![tool_call]),
        tool_call_id: None,
    };

    let req = ChatRequest {
        messages: vec![msg],
        tools: None,
        common_params: CommonParams {
            model: "gpt-5-mini".into(),
            ..Default::default()
        },
        provider_params: None,
        http_config: None,
        web_search: None,
        stream: false,
    };

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
