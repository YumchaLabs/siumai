use siumai::providers::openai::transformers::OpenAiResponsesResponseTransformer;
use siumai::types::ChatResponse;

#[test]
fn parse_tool_calls_with_nested_function_object() {
    let tx = OpenAiResponsesResponseTransformer;
    let raw = serde_json::json!({
        "response": {
            "id": "r1",
            "model": "gpt-5-mini",
            "output": [
                {
                    "content": [{"type": "output_text", "text": "hello"}],
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "lookup", "arguments": "{\"q\":\"rust\"}"}}
                    ]
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        }
    });

    let resp: ChatResponse = tx.transform_chat_response(&raw).expect("ok");
    assert_eq!(resp.content_text().unwrap_or_default(), "hello");
    let calls = resp.get_tool_calls().expect("has tool calls");
    assert_eq!(calls.len(), 1);
    let call = calls[0];
    let info = call.as_tool_call().expect("tool call info");
    assert_eq!(info.tool_name, "lookup");
    assert_eq!(info.tool_call_id, "call_1");
    assert_eq!(
        info.arguments,
        &serde_json::json!({"q": "rust"})
    );
}

#[test]
fn parse_tool_calls_with_flattened_fields() {
    let tx = OpenAiResponsesResponseTransformer;
    let raw = serde_json::json!({
        "id": "r2",
        "model": "gpt-5-mini",
        "output": [
            {
                "content": [{"type": "output_text", "text": "world"}],
                "tool_calls": [
                    {"id": "call_2", "name": "search", "arguments": "{\"k\":\"v\"}"}
                ]
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    });

    let resp: ChatResponse = tx.transform_chat_response(&raw).expect("ok");
    assert_eq!(resp.content_text().unwrap_or_default(), "world");
    let calls = resp.get_tool_calls().expect("has tool calls");
    assert_eq!(calls.len(), 1);
    let call = calls[0];
    let info = call.as_tool_call().expect("tool call info");
    assert_eq!(info.tool_name, "search");
    assert_eq!(info.tool_call_id, "call_2");
    assert_eq!(
        info.arguments,
        &serde_json::json!({"k": "v"})
    );
}

#[test]
fn parse_usage_with_camel_case_fields() {
    let tx = OpenAiResponsesResponseTransformer;
    let raw = serde_json::json!({
        "response": {
            "id": "r3",
            "model": "gpt-5-mini",
            "output": [
                { "content": [{"type": "output_text", "text": "ok"}] }
            ],
            "usage": {
                "inputTokens": 7,
                "outputTokens": 9,
                "totalTokens": 16,
                "reasoningTokens": 2
            },
            "stop_reason": "length"
        }
    });

    let resp: ChatResponse = tx.transform_chat_response(&raw).expect("ok");
    assert_eq!(resp.content_text().unwrap_or_default(), "ok");
    let usage = resp.usage.expect("usage present");
    assert_eq!(usage.prompt_tokens, 7);
    assert_eq!(usage.completion_tokens, 9);
    assert_eq!(usage.total_tokens, 16);
    assert_eq!(usage.reasoning_tokens, Some(2));
    assert_eq!(resp.finish_reason.unwrap().to_string(), "length");
}

#[test]
fn missing_tool_calls_results_in_none() {
    let tx = OpenAiResponsesResponseTransformer;
    let raw = serde_json::json!({
        "id": "r4",
        "model": "gpt-5-mini",
        "output": [ { "content": [{"type": "output_text", "text": "hello"}] } ]
    });
    let resp: ChatResponse = tx.transform_chat_response(&raw).expect("ok");
    assert!(resp.get_tool_calls().is_none());
}

#[test]
fn openai_metadata_is_populated_from_responses() {
    let tx = OpenAiResponsesResponseTransformer;
    let raw = serde_json::json!({
        "response": {
            "id": "r5",
            "model": "gpt-5-mini",
            "output": [
                { "content": [{"type": "output_text", "text": "meta"}] }
            ],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "completion_tokens_details": {
                    "reasoning_tokens": 11
                }
            },
            "system_fingerprint": "fp_test",
            "service_tier": "scale"
        }
    });

    let resp: ChatResponse = tx.transform_chat_response(&raw).expect("ok");

    // Top-level fields should be populated
    assert_eq!(resp.system_fingerprint.as_deref(), Some("fp_test"));
    assert_eq!(resp.service_tier.as_deref(), Some("scale"));

    // Provider-specific metadata should mirror these values
    let meta = resp.openai_metadata().expect("openai metadata");
    assert_eq!(meta.reasoning_tokens, Some(11));
    assert_eq!(meta.system_fingerprint.as_deref(), Some("fp_test"));
    assert_eq!(meta.service_tier.as_deref(), Some("scale"));
}
