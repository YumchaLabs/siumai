//! Mock API tests for Gemini provider
//!
//! These tests use wiremock to simulate Gemini API responses based on official documentation.
//! Response formats are based on Google's official Gemini API reference:
//! https://ai.google.dev/api/generate-content

use serde_json::json;
use siumai::builder::LlmBuilder;
use siumai::prelude::*;
use wiremock::matchers::{header, method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Official Gemini generateContent response format
/// Based on: https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse
fn create_generate_content_response() -> serde_json::Value {
    json!({
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "Hello! How can I help you today?"
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE"
                    }
                ]
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 10,
            "totalTokenCount": 15
        },
        "modelVersion": "gemini-1.5-flash",
        "responseId": "resp_abc123"
    })
}

/// Official Gemini error response format
fn create_error_response() -> serde_json::Value {
    json!({
        "error": {
            "code": 401,
            "message": "API key not valid. Please pass a valid API key.",
            "status": "UNAUTHENTICATED"
        }
    })
}

/// Official Gemini function calling response format
fn create_function_calling_response() -> serde_json::Value {
    json!({
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {
                                    "location": "San Francisco, CA"
                                }
                            }
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "safetyRatings": []
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 10,
            "totalTokenCount": 30
        },
        "modelVersion": "gemini-1.5-flash"
    })
}

#[tokio::test]
#[cfg(feature = "google")]
async fn test_gemini_generate_content_non_streaming() {
    let mock_server = MockServer::start().await;

    // Setup mock response
    // Gemini API path: /models/{model}:generateContent
    // Gemini uses x-goog-api-key header for authentication
    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_generate_content_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .gemini()
        .api_key("test-api-key")
        .base_url(&mock_server.uri())
        .model("gemini-1.5-flash")
        .build()
        .await
        .unwrap();

    // Send request
    let response = client
        .chat(vec![ChatMessage::user("Hello").build()])
        .await
        .unwrap();

    // Verify response content
    assert_eq!(response.content.text(), Some("Hello! How can I help you today?"));

    // Verify response metadata (now correctly parsed)
    assert_eq!(response.id, Some("resp_abc123".to_string()));
    assert_eq!(response.model, Some("gemini-1.5-flash".to_string()));

    // Verify usage metadata (now correctly parsed)
    let usage = response.usage.expect("Usage should be present");
    assert_eq!(usage.prompt_tokens, 5);
    assert_eq!(usage.completion_tokens, 10);
    assert_eq!(usage.total_tokens, 15);

    // Verify finish reason (now correctly parsed)
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
}

#[tokio::test]
#[cfg(feature = "google")]
async fn test_gemini_error_response() {
    let mock_server = MockServer::start().await;

    // Setup error response
    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent"))
        .respond_with(ResponseTemplate::new(401).set_body_json(create_error_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .gemini()
        .api_key("invalid-key")
        .base_url(&mock_server.uri())
        .model("gemini-1.5-flash")
        .build()
        .await
        .unwrap();

    // Send request and expect error
    let result = client
        .chat(vec![ChatMessage::user("Hello").build()])
        .await;

    assert!(result.is_err());
}

#[tokio::test]
#[cfg(feature = "google")]
async fn test_gemini_request_headers() {
    let mock_server = MockServer::start().await;

    // Verify API key is sent as x-goog-api-key header
    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(header("content-type", "application/json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_generate_content_response()))
        .expect(1) // Verify the request was made exactly once
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .gemini()
        .api_key("test-api-key")
        .base_url(&mock_server.uri())
        .model("gemini-1.5-flash")
        .build()
        .await
        .unwrap();

    // Send request
    let _response = client
        .chat(vec![ChatMessage::user("Hello").build()])
        .await
        .unwrap();

    // Mock server will verify query parameters automatically
}

#[tokio::test]
#[cfg(feature = "google")]
async fn test_gemini_function_calling_response() {
    let mock_server = MockServer::start().await;

    // Setup function calling response
    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_function_calling_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .gemini()
        .api_key("test-api-key")
        .base_url(&mock_server.uri())
        .model("gemini-1.5-flash")
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
    assert!(response.tool_calls.is_some());

    let tool_calls = response.tool_calls.unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.as_ref().unwrap().name, "get_weather");

    // Verify tool arguments
    let args: serde_json::Value = serde_json::from_str(&tool_calls[0].function.as_ref().unwrap().arguments).unwrap();
    assert_eq!(args["location"], "San Francisco, CA");

    // Note: Gemini transformer may not set finish_reason to ToolCalls
}

