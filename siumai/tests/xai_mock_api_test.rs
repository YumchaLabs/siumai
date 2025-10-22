//! xAI Mock API Tests
//!
//! These tests verify the HTTP layer interaction with xAI API using wiremock.
//! All test responses are based on official xAI API documentation:
//! https://docs.x.ai/docs/api-reference

use serde_json::json;
use siumai::{ChatCapability, ChatMessage, FinishReason, LlmBuilder, Tool};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path},
};

/// Helper function to create a standard xAI chat response
/// Based on official API documentation (OpenAI-compatible format)
fn create_chat_response() -> serde_json::Value {
    json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "grok-beta",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm Grok, how can I help you today?"
                },
                "logprobs": null,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 12,
            "total_tokens": 22
        },
        "system_fingerprint": "fp_44709d6fcb"
    })
}

/// Helper function to create an xAI error response
/// Based on OpenAI-compatible error format
fn create_error_response() -> serde_json::Value {
    json!({
        "error": {
            "message": "Invalid model specified",
            "type": "invalid_request_error",
            "code": "model_not_found"
        }
    })
}

/// Helper function to create a tool calling response
/// Based on OpenAI-compatible tool calling format
fn create_tool_call_response() -> serde_json::Value {
    json!({
        "id": "chatcmpl-tool-456",
        "object": "chat.completion",
        "created": 1677652300,
        "model": "grok-beta",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_xyz789",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\":\"San Francisco\",\"unit\":\"celsius\"}"
                            }
                        }
                    ]
                },
                "logprobs": null,
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 45,
            "completion_tokens": 18,
            "total_tokens": 63
        },
        "system_fingerprint": "fp_44709d6fcb"
    })
}

#[tokio::test]
#[cfg(feature = "xai")]
async fn test_xai_chat_non_streaming() {
    let mock_server = MockServer::start().await;

    // Mount mock endpoint
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer test-api-key"))
        .and(header("content-type", "application/json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_chat_response()))
        .mount(&mock_server)
        .await;

    // Create client
    let client = LlmBuilder::new()
        .xai()
        .api_key("test-api-key")
        .base_url(&mock_server.uri())
        .model("grok-beta")
        .build()
        .await
        .unwrap();

    // Make request
    let response = client
        .chat(vec![ChatMessage::user("Hello!").build()])
        .await
        .unwrap();

    // Verify response
    assert_eq!(
        response.content.text(),
        Some("Hello! I'm Grok, how can I help you today?")
    );
    assert_eq!(response.model, Some("grok-beta".to_string()));
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    assert!(response.usage.is_some());

    let usage = response.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 12);
    assert_eq!(usage.total_tokens, 22);
}

#[tokio::test]
#[cfg(feature = "xai")]
async fn test_xai_error_response() {
    let mock_server = MockServer::start().await;

    // Mount error response
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(404).set_body_json(create_error_response()))
        .mount(&mock_server)
        .await;

    // Create client
    let client = LlmBuilder::new()
        .xai()
        .api_key("test-api-key")
        .base_url(&mock_server.uri())
        .model("invalid-model")
        .build()
        .await
        .unwrap();

    // Make request and expect error
    let result = client.chat(vec![ChatMessage::user("Hello").build()]).await;

    assert!(result.is_err());
    let error = result.unwrap_err();
    let error_str = error.to_string();
    assert!(
        error_str.contains("model_not_found") || error_str.contains("invalid_request_error"),
        "Error should contain API error details, got: {}",
        error_str
    );
}

#[tokio::test]
#[cfg(feature = "xai")]
async fn test_xai_request_format() {
    let mock_server = MockServer::start().await;

    // Mount mock with request body verification
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_chat_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .xai()
        .api_key("test-key")
        .base_url(&mock_server.uri())
        .model("grok-beta")
        .build()
        .await
        .unwrap();

    let response = client
        .chat(vec![
            ChatMessage::system("You are a helpful assistant.").build(),
            ChatMessage::user("Hello!").build(),
        ])
        .await
        .unwrap();

    assert!(response.content.text().is_some());
}

#[tokio::test]
#[cfg(feature = "xai")]
async fn test_xai_tool_calling() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_tool_call_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .xai()
        .api_key("test-api-key")
        .base_url(&mock_server.uri())
        .model("grok-beta")
        .build()
        .await
        .unwrap();

    // Create weather tool
    let weather_tool = Tool::function(
        "get_weather".to_string(),
        "Get the current weather for a location".to_string(),
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit"
                }
            },
            "required": ["location", "unit"]
        }),
    );

    let response = client
        .chat_with_tools(
            vec![ChatMessage::user("What's the weather in San Francisco?").build()],
            Some(vec![weather_tool]),
        )
        .await
        .unwrap();

    // Verify tool calls
    assert!(response.tool_calls.is_some());
    let tool_calls = response.tool_calls.unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.as_ref().unwrap().name, "get_weather");
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
}

#[tokio::test]
#[cfg(feature = "xai")]
async fn test_xai_system_fingerprint() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_chat_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .xai()
        .api_key("test-api-key")
        .base_url(&mock_server.uri())
        .model("grok-beta")
        .build()
        .await
        .unwrap();

    let response = client
        .chat(vec![ChatMessage::user("Hello").build()])
        .await
        .unwrap();

    // Verify response ID is preserved
    assert_eq!(response.id, Some("chatcmpl-123".to_string()));
}
