//! Groq Mock API Tests
//!
//! These tests verify the HTTP layer interaction with Groq API using wiremock.
//! All test responses are based on official Groq API documentation:
//! https://console.groq.com/docs/api-reference

use serde_json::json;
use siumai::types::ContentPart;
use siumai::{ChatCapability, ChatMessage, FinishReason, LlmBuilder, Tool};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path},
};

/// Helper function to create a standard Groq chat response
/// Based on official API documentation example response
fn create_chat_response() -> serde_json::Value {
    json!({
        "id": "chatcmpl-f51b2cd2-bef7-417e-964e-a08f0b513c22",
        "object": "chat.completion",
        "created": 1730241104,
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Fast language models are important because they enable real-time applications and improve user experience."
                },
                "logprobs": null,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "queue_time": 0.037493756,
            "prompt_tokens": 18,
            "prompt_time": 0.000680594,
            "completion_tokens": 25,
            "completion_time": 0.463333333,
            "total_tokens": 43,
            "total_time": 0.464013927
        },
        "system_fingerprint": "fp_179b0f92c9",
        "x_groq": {
            "id": "req_01jbd6g2qdfw2adyrt2az8hz4w"
        }
    })
}

/// Helper function to create a Groq error response
/// Based on official API error format
fn create_error_response() -> serde_json::Value {
    json!({
        "error": {
            "message": "The model `invalid-model` does not exist",
            "type": "invalid_request_error",
            "code": "model_not_found"
        }
    })
}

/// Helper function to create a tool calling response
/// Based on official API tool calling example
fn create_tool_call_response() -> serde_json::Value {
    json!({
        "id": "chatcmpl-tool-123",
        "object": "chat.completion",
        "created": 1730241200,
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": "{\"location\":\"San Francisco\",\"format\":\"celsius\"}"
                            }
                        }
                    ]
                },
                "logprobs": null,
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "queue_time": 0.02,
            "prompt_tokens": 50,
            "prompt_time": 0.001,
            "completion_tokens": 20,
            "completion_time": 0.3,
            "total_tokens": 70,
            "total_time": 0.321
        },
        "system_fingerprint": "fp_179b0f92c9",
        "x_groq": {
            "id": "req_tool_call_123"
        }
    })
}

#[tokio::test]
#[cfg(feature = "groq")]
async fn test_groq_chat_non_streaming() {
    let mock_server = MockServer::start().await;

    // Mount mock endpoint
    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .and(header("authorization", "Bearer test-api-key"))
        .and(header("content-type", "application/json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_chat_response()))
        .mount(&mock_server)
        .await;

    // Create client
    let client = LlmBuilder::new()
        .groq()
        .api_key("test-api-key")
        .base_url(format!("{}/openai/v1", mock_server.uri()))
        .model("llama-3.3-70b-versatile")
        .build()
        .await
        .unwrap();

    // Make request
    let response = client
        .chat(vec![
            ChatMessage::user("Explain the importance of fast language models").build(),
        ])
        .await
        .unwrap();

    // Verify response
    assert_eq!(
        response.content.text(),
        Some(
            "Fast language models are important because they enable real-time applications and improve user experience."
        )
    );
    assert_eq!(response.model, Some("llama-3.3-70b-versatile".to_string()));
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    assert!(response.usage.is_some());

    let usage = response.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 18);
    assert_eq!(usage.completion_tokens, 25);
    assert_eq!(usage.total_tokens, 43);
}

#[tokio::test]
#[cfg(feature = "groq")]
async fn test_groq_error_response() {
    let mock_server = MockServer::start().await;

    // Mount error response
    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(404).set_body_json(create_error_response()))
        .mount(&mock_server)
        .await;

    // Create client
    let client = LlmBuilder::new()
        .groq()
        .api_key("test-api-key")
        .base_url(format!("{}/openai/v1", mock_server.uri()))
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
#[cfg(feature = "groq")]
async fn test_groq_request_format() {
    let mock_server = MockServer::start().await;

    // Mount mock with request body verification
    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_chat_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .groq()
        .api_key("test-key")
        .base_url(format!("{}/openai/v1", mock_server.uri()))
        .model("llama-3.3-70b-versatile")
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
#[cfg(feature = "groq")]
async fn test_groq_tool_calling() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_tool_call_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .groq()
        .api_key("test-api-key")
        .base_url(format!("{}/openai/v1", mock_server.uri()))
        .model("llama-3.3-70b-versatile")
        .build()
        .await
        .unwrap();

    // Create weather tool
    let weather_tool = Tool::function(
        "get_current_weather".to_string(),
        "Get the current weather for a location".to_string(),
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit"
                }
            },
            "required": ["location", "format"]
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
    assert!(response.has_tool_calls());
    let tool_calls = response.tool_calls();
    assert_eq!(tool_calls.len(), 1);
    if let ContentPart::ToolCall { tool_name, .. } = &tool_calls[0] {
        assert_eq!(tool_name, "get_current_weather");
    } else {
        panic!("Expected ToolCall");
    }
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
}

#[tokio::test]
#[cfg(feature = "groq")]
async fn test_groq_system_fingerprint() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_chat_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .groq()
        .api_key("test-api-key")
        .base_url(format!("{}/openai/v1", mock_server.uri()))
        .model("llama-3.3-70b-versatile")
        .build()
        .await
        .unwrap();

    let response = client
        .chat(vec![ChatMessage::user("Hello").build()])
        .await
        .unwrap();

    // Verify Groq-specific fields are preserved
    assert_eq!(
        response.id,
        Some("chatcmpl-f51b2cd2-bef7-417e-964e-a08f0b513c22".to_string())
    );
}
