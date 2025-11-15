use siumai_core::types::FinishReasonCore;
use siumai_std_openai::openai::responses::OpenAiResponsesStandard;

#[test]
fn openai_responses_response_transformer_parses_snake_case_usage() {
    let standard = OpenAiResponsesStandard::new();
    let tx = standard.create_response_transformer("openai_responses");

    let raw = serde_json::json!({
        "response": {
            "id": "r1",
            "model": "gpt-5-mini",
            "output": [],
            "usage": {
                "input_tokens": 7,
                "output_tokens": 9,
                "total_tokens": 16
            }
        }
    });

    let result = tx
        .transform_responses_response(&raw)
        .expect("transform_responses_response should succeed");

    let usage = result.usage.expect("usage should be present");
    assert_eq!(usage.prompt_tokens, 7);
    assert_eq!(usage.completion_tokens, 9);
    assert_eq!(usage.total_tokens, 16);
}

#[test]
fn openai_responses_response_transformer_parses_camel_case_usage() {
    let standard = OpenAiResponsesStandard::new();
    let tx = standard.create_response_transformer("openai_responses");

    let raw = serde_json::json!({
        "response": {
            "id": "r2",
            "model": "gpt-5-mini",
            "output": [],
            "usage": {
                "inputTokens": 5,
                "outputTokens": 11,
                "totalTokens": 16
            }
        }
    });

    let result = tx
        .transform_responses_response(&raw)
        .expect("transform_responses_response should succeed");

    let usage = result.usage.expect("usage should be present");
    assert_eq!(usage.prompt_tokens, 5);
    assert_eq!(usage.completion_tokens, 11);
    assert_eq!(usage.total_tokens, 16);
}

#[test]
fn openai_responses_response_transformer_maps_finish_reason_max_tokens_to_length() {
    let standard = OpenAiResponsesStandard::new();
    let tx = standard.create_response_transformer("openai_responses");

    let raw = serde_json::json!({
        "response": {
            "id": "r3",
            "model": "gpt-5-mini",
            "output": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3
            },
            "stop_reason": "max_tokens"
        }
    });

    let result = tx
        .transform_responses_response(&raw)
        .expect("transform_responses_response should succeed");

    assert_eq!(result.finish_reason, Some(FinishReasonCore::Length));
}

#[test]
fn openai_responses_response_transformer_maps_finish_reason_tool_use_to_tool_calls() {
    let standard = OpenAiResponsesStandard::new();
    let tx = standard.create_response_transformer("openai_responses");

    let raw = serde_json::json!({
        "response": {
            "id": "r4",
            "model": "gpt-5-mini",
            "output": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3
            },
            "finish_reason": "tool_use"
        }
    });

    let result = tx
        .transform_responses_response(&raw)
        .expect("transform_responses_response should succeed");

    assert_eq!(result.finish_reason, Some(FinishReasonCore::ToolCalls));
}
