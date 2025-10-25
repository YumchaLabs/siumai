use siumai::providers::openai::transformers::OpenAiResponsesRequestTransformer;
use siumai::types::*;

#[test]
fn transform_basic_non_stream() {
    let tx = OpenAiResponsesRequestTransformer;
    let req = ChatRequest::builder()
        .message(ChatMessage::user("hello").build())
        .common_params(CommonParams {
            model: "gpt-5-mini".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(256),
            top_p: None,
            stop_sequences: None,
            seed: Some(42),
        })
        .build();
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
    let assistant = ChatMessage::assistant_with_content(vec![
        ContentPart::tool_call("call_1", "lookup", serde_json::json!({"q":"rust"}), None),
    ])
    .build();

    let req = ChatRequest::builder()
        .messages(vec![
            ChatMessage::user("please lookup").build(),
            assistant,
        ])
        .tools(vec![Tool::function(
            "lookup".to_string(),
            "search".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {"q": {"type": "string"}}
            }),
        )])
        .common_params(CommonParams {
            model: "gpt-5-mini".to_string(),
            ..Default::default()
        })
        .stream(true)
        .build();
    let body = tx.transform_chat(&req).expect("ok");
    assert_eq!(body.get("stream").and_then(|v| v.as_bool()).unwrap(), true);
    let tools = body.get("tools").and_then(|v| v.as_array()).unwrap();
    assert_eq!(tools.len(), 1);
    let input = body.get("input").and_then(|v| v.as_array()).unwrap();
    assert_eq!(input.len(), 2);
}
