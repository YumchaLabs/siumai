#![cfg(feature = "anthropic")]
//! Mock API tests for Anthropic provider
//!
//! These tests use wiremock to simulate Anthropic API responses based on official documentation.
//! Response formats are based on Anthropic's official API reference:
//! https://docs.anthropic.com/en/api/messages

use serde_json::json;
use siumai::builder::LlmBuilder;
use siumai::prelude::*;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Official Anthropic Messages response format
/// Based on: https://docs.anthropic.com/en/api/messages
fn create_messages_response() -> serde_json::Value {
    json!({
        "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello! How can I help you today?"
            }
        ],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 15,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0
        }
    })
}

/// Official Anthropic error response format
/// Based on: https://docs.anthropic.com/en/api/errors
fn create_error_response() -> serde_json::Value {
    json!({
        "type": "error",
        "error": {
            "type": "authentication_error",
            "message": "invalid x-api-key"
        }
    })
}

/// Official Anthropic tool use response format
fn create_tool_use_response() -> serde_json::Value {
    json!({
        "id": "msg_01T1x1fJ34qw5pBdwsBKZEu",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01T1x1fJ34qw5pBdwsBKZEu",
                "name": "get_weather",
                "input": {
                    "location": "San Francisco, CA"
                }
            }
        ],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "tool_use",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 20,
            "output_tokens": 10,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0
        }
    })
}

#[tokio::test]
#[cfg(feature = "anthropic")]
async fn test_anthropic_messages_non_streaming() {
    let mock_server = MockServer::start().await;

    // Setup mock response
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-api-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_messages_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .anthropic()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("claude-3-5-sonnet-20241022")
        .build()
        .await
        .unwrap();

    // Send request
    let response = client
        .chat(vec![ChatMessage::user("Hello").build()])
        .await
        .unwrap();

    // Verify response
    assert_eq!(
        response.model,
        Some("claude-3-5-sonnet-20241022".to_string())
    );
    assert_eq!(
        response.content.text(),
        Some("Hello! How can I help you today?")
    );
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));

    // Verify usage
    assert!(response.usage.is_some());
    let usage = response.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 15);
    assert_eq!(usage.total_tokens, 25);
}

#[tokio::test]
#[cfg(feature = "anthropic")]
async fn test_anthropic_error_response() {
    let mock_server = MockServer::start().await;

    // Setup error response
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_json(create_error_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .anthropic()
        .api_key("invalid-key")
        .base_url(mock_server.uri())
        .model("claude-3-5-sonnet-20241022")
        .build()
        .await
        .unwrap();

    // Send request and expect error
    let result = client.chat(vec![ChatMessage::user("Hello").build()]).await;

    assert!(result.is_err());
}

#[tokio::test]
#[cfg(feature = "anthropic")]
async fn test_anthropic_request_headers() {
    let mock_server = MockServer::start().await;

    // Verify all required headers are sent
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-api-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(header("content-type", "application/json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_messages_response()))
        .expect(1) // Verify the request was made exactly once
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .anthropic()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("claude-3-5-sonnet-20241022")
        .build()
        .await
        .unwrap();

    // Send request
    let _response = client
        .chat(vec![ChatMessage::user("Hello").build()])
        .await
        .unwrap();

    // Mock server will verify headers automatically
}

#[tokio::test]
#[cfg(feature = "anthropic")]
async fn test_anthropic_tool_use_response() {
    let mock_server = MockServer::start().await;

    // Setup tool use response
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_tool_use_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .anthropic()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("claude-3-5-sonnet-20241022")
        .build()
        .await
        .unwrap();

    // Create a simple tool
    let tool = Tool::function(
        "get_weather".to_string(),
        "Get the weather for a location".to_string(),
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }),
    );

    // Send request with tools
    let response = client
        .chat_with_tools(
            vec![ChatMessage::user("What's the weather in San Francisco?").build()],
            Some(vec![tool]),
        )
        .await
        .unwrap();

    // Verify response
    use siumai::types::ContentPart;

    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    assert!(response.has_tool_calls());

    let tool_calls = response.tool_calls();
    assert_eq!(tool_calls.len(), 1);

    // Extract tool call details
    if let ContentPart::ToolCall {
        tool_name,
        arguments,
        ..
    } = &tool_calls[0]
    {
        assert_eq!(tool_name, "get_weather");
        assert_eq!(arguments["location"], "San Francisco, CA");
    } else {
        panic!("Expected ToolCall content part");
    }
}
