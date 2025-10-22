//! Ollama Mock API tests
//!
//! These tests verify that the Ollama provider correctly interacts with the Ollama API
//! using mock responses based on the official Ollama API documentation.
//! Reference: https://github.com/ollama/ollama/blob/main/docs/api.md

use serde_json::json;
use siumai::builder::LlmBuilder;
use siumai::prelude::*;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Create a non-streaming chat response based on official Ollama API docs
fn create_chat_response() -> serde_json::Value {
    json!({
        "model": "llama3.2",
        "created_at": "2023-12-12T14:13:43.416799Z",
        "message": {
            "role": "assistant",
            "content": "Hello! How are you today?"
        },
        "done": true,
        "total_duration": 5191566416u64,
        "load_duration": 2154458u64,
        "prompt_eval_count": 26,
        "prompt_eval_duration": 383809000u64,
        "eval_count": 298,
        "eval_duration": 4799921000u64
    })
}

/// Create an error response based on Ollama API behavior
fn create_error_response() -> serde_json::Value {
    json!({
        "error": "model 'invalid-model' not found, try pulling it first"
    })
}

#[tokio::test]
#[cfg(feature = "ollama")]
async fn test_ollama_chat_non_streaming() {
    let mock_server = MockServer::start().await;

    // Setup mock response
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(header("content-type", "application/json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_chat_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .ollama()
        .base_url(&mock_server.uri())
        .model("llama3.2")
        .build()
        .await
        .unwrap();

    let response = client
        .chat(vec![ChatMessage::user("Hello!").build()])
        .await
        .unwrap();

    // Verify response
    assert_eq!(response.content.text(), Some("Hello! How are you today?"));
    assert_eq!(response.model, Some("llama3.2".to_string()));
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));

    // Verify usage
    let usage = response.usage.expect("Usage should be present");
    assert_eq!(usage.prompt_tokens, 26);
    assert_eq!(usage.completion_tokens, 298);
    assert_eq!(usage.total_tokens, 324);
}

#[tokio::test]
#[cfg(feature = "ollama")]
async fn test_ollama_error_response() {
    let mock_server = MockServer::start().await;

    // Setup mock error response
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(404).set_body_json(create_error_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .ollama()
        .base_url(&mock_server.uri())
        .model("invalid-model")
        .build()
        .await
        .unwrap();

    let result = client.chat(vec![ChatMessage::user("Hello!").build()]).await;

    // Verify error
    assert!(result.is_err());
    let err = result.unwrap_err();
    // Should be ApiError after classify_http_error fix
    assert!(matches!(err, LlmError::ApiError { .. }));
}

#[tokio::test]
#[cfg(feature = "ollama")]
async fn test_ollama_request_format() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(header("content-type", "application/json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_chat_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .ollama()
        .base_url(&mock_server.uri())
        .model("llama3.2")
        .build()
        .await
        .unwrap();

    let response = client
        .chat(vec![ChatMessage::user("Test message").build()])
        .await
        .unwrap();

    assert!(response.content.text().is_some());
}

#[tokio::test]
#[cfg(feature = "ollama")]
async fn test_ollama_tool_calling() {
    let mock_server = MockServer::start().await;

    // Create tool calling response based on official docs
    let tool_response = json!({
        "model": "llama3.2",
        "created_at": "2024-07-22T20:33:28.123648Z",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_current_weather",
                        "arguments": {
                            "format": "celsius",
                            "location": "Paris, FR"
                        }
                    }
                }
            ]
        },
        "done_reason": "stop",
        "done": true,
        "total_duration": 885095291u64,
        "load_duration": 3753500u64,
        "prompt_eval_count": 122,
        "prompt_eval_duration": 328493000u64,
        "eval_count": 33,
        "eval_duration": 552222000u64
    });

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tool_response))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .ollama()
        .base_url(&mock_server.uri())
        .model("llama3.2")
        .build()
        .await
        .unwrap();

    let weather_tool = Tool::function(
        "get_current_weather".to_string(),
        "Get the current weather for a location".to_string(),
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the weather for"
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The format to return the weather in"
                }
            },
            "required": ["location", "format"]
        }),
    );

    let response = client
        .chat_with_tools(
            vec![ChatMessage::user("What is the weather today in Paris?").build()],
            Some(vec![weather_tool]),
        )
        .await
        .unwrap();

    // Verify tool call
    assert!(response.tool_calls.is_some());
    let tool_calls = response.tool_calls.unwrap();
    assert_eq!(tool_calls.len(), 1);

    let function = tool_calls[0].function.as_ref().unwrap();
    assert_eq!(function.name, "get_current_weather");

    // Parse arguments JSON string
    let args: serde_json::Value = serde_json::from_str(&function.arguments).unwrap();
    assert_eq!(
        args.get("location").and_then(|v| v.as_str()),
        Some("Paris, FR")
    );
    assert_eq!(args.get("format").and_then(|v| v.as_str()), Some("celsius"));
}

#[tokio::test]
#[cfg(feature = "ollama")]
async fn test_ollama_json_mode() {
    let mock_server = MockServer::start().await;

    // Create JSON mode response
    let json_response = json!({
        "model": "llama3.2",
        "created_at": "2023-11-09T21:07:55.186497Z",
        "message": {
            "role": "assistant",
            "content": "{\"morning\":{\"color\":\"blue\"},\"noon\":{\"color\":\"blue-gray\"},\"afternoon\":{\"color\":\"warm gray\"},\"evening\":{\"color\":\"orange\"}}"
        },
        "done": true,
        "total_duration": 4648158584u64,
        "load_duration": 4071084u64,
        "prompt_eval_count": 36,
        "prompt_eval_duration": 439038000u64,
        "eval_count": 180,
        "eval_duration": 4196918000u64
    });

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json_response))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .ollama()
        .base_url(&mock_server.uri())
        .model("llama3.2")
        .format("json")
        .build()
        .await
        .unwrap();

    let response = client
        .chat(vec![
            ChatMessage::user(
                "What color is the sky at different times of the day? Respond using JSON",
            )
            .build(),
        ])
        .await
        .unwrap();

    // Verify JSON response
    let content = response.content.text().unwrap();
    assert!(content.contains("morning"));
    assert!(content.contains("blue"));

    // Verify it's valid JSON
    let parsed: serde_json::Value = serde_json::from_str(content).unwrap();
    assert!(parsed.get("morning").is_some());
}
