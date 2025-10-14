use siumai::providers::openai::transformers::OpenAiResponsesRequestTransformer;
use siumai::types::*;

#[test]
fn transform_basic_non_stream() {
    let tx = OpenAiResponsesRequestTransformer;
    let req = ChatRequest {
        messages: vec![ChatMessage::user("hello")],
        tools: None,
        common_params: CommonParams {
            model: "gpt-5-mini".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(256),
            top_p: None,
            stop_sequences: None,
            seed: Some(42),
        },
        provider_params: None,
        http_config: None,
        web_search: None,
        stream: false,
    };
    let body = tx.transform_chat(&req).expect("ok");
    assert_eq!(body.get("model").and_then(|v| v.as_str()).unwrap(), "gpt-5-mini");
    assert_eq!(body.get("stream").and_then(|v| v.as_bool()).unwrap(), false);
    assert_eq!(body.get("temperature").and_then(|v| v.as_f64()).unwrap(), 0.7);
    assert_eq!(body.get("max_output_tokens").and_then(|v| v.as_u64()).unwrap(), 256);
    assert_eq!(body.get("seed").and_then(|v| v.as_u64()).unwrap(), 42);
    let input = body.get("input").and_then(|v| v.as_array()).unwrap();
    assert_eq!(input.len(), 1);
}

#[test]
fn transform_with_tool_and_stream() {
    let tx = OpenAiResponsesRequestTransformer;
    // Assistant suggesting a tool call
    let mut assistant = ChatMessage::assistant("");
    assistant.tool_calls = Some(vec![ToolCall {
        id: "call_1".to_string(),
        r#type: "function".to_string(),
        function: Some(FunctionCall { name: "lookup".to_string(), arguments: "{\"q\":\"rust\"}".to_string() }),
    }]);

    let req = ChatRequest {
        messages: vec![ChatMessage::user("please lookup"), assistant],
        tools: Some(vec![Tool { r#type: "function".to_string(), function: ToolFunction { name: "lookup".to_string(), description: Some("search".to_string()), parameters: serde_json::json!({"type":"object","properties":{"q":{"type":"string"}}}) } }]),
        common_params: CommonParams { model: "gpt-5-mini".to_string(), temperature: None, max_tokens: None, top_p: None, stop_sequences: None, seed: None },
        provider_params: None,
        http_config: None,
        web_search: None,
        stream: true,
    };
    let body = tx.transform_chat(&req).expect("ok");
    assert_eq!(body.get("stream").and_then(|v| v.as_bool()).unwrap(), true);
    let tools = body.get("tools").and_then(|v| v.as_array()).unwrap();
    assert_eq!(tools.len(), 1);
    let input = body.get("input").and_then(|v| v.as_array()).unwrap();
    assert_eq!(input.len(), 2);
}

