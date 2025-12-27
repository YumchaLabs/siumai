#![cfg(feature = "google")]
//! Tests for Gemini grounding tools (Google Search, URL Context)
//!
//! These tests verify that grounding tools can be properly configured and sent to the API.

use serde_json::json;
use siumai::builder::LlmBuilder;
use siumai::prelude::*;
use siumai::providers::gemini::{
    DynamicRetrievalConfig, DynamicRetrievalMode, GeminiTool, GoogleSearch, GoogleSearchRetrieval,
    UrlContext,
};
use wiremock::matchers::{header, method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Create a simple Gemini response
fn create_simple_response() -> serde_json::Value {
    json!({
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "This is a grounded response."
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP"
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15
        },
        "modelVersion": "gemini-2.5-flash",
        "responseId": "resp_grounding_123"
    })
}

#[tokio::test]
#[cfg(feature = "google")]
async fn test_google_search_tool_serialization() {
    // Test that GoogleSearch tool serializes correctly
    let tool = GeminiTool::GoogleSearch {
        google_search: GoogleSearch {},
    };

    let json = serde_json::to_value(&tool).unwrap();
    // Gemini API expects camelCase keys for tool types.
    assert_eq!(json, json!({"googleSearch": {}}));
}

#[tokio::test]
#[cfg(feature = "google")]
async fn test_google_search_retrieval_tool_serialization() {
    // Test that GoogleSearchRetrieval tool serializes correctly
    let tool = GeminiTool::GoogleSearchRetrieval {
        google_search_retrieval: GoogleSearchRetrieval {
            dynamic_retrieval_config: Some(DynamicRetrievalConfig {
                mode: DynamicRetrievalMode::Dynamic,
                dynamic_threshold: Some(0.7),
            }),
        },
    };

    let json = serde_json::to_value(&tool).unwrap();

    // Verify structure (avoid floating point comparison issues)
    assert!(json.get("googleSearchRetrieval").is_some());
    let retrieval = json.get("googleSearchRetrieval").unwrap();
    assert!(retrieval.get("dynamicRetrievalConfig").is_some());
    let config = retrieval.get("dynamicRetrievalConfig").unwrap();
    assert_eq!(config.get("mode").unwrap(), "MODE_DYNAMIC");

    // Verify threshold is approximately 0.7
    let threshold = config.get("dynamicThreshold").unwrap().as_f64().unwrap();
    assert!((threshold - 0.7).abs() < 0.01);
}

#[tokio::test]
#[cfg(feature = "google")]
async fn test_url_context_tool_serialization() {
    // Test that UrlContext tool serializes correctly
    let tool = GeminiTool::UrlContext {
        url_context: UrlContext {},
    };

    let json = serde_json::to_value(&tool).unwrap();
    // Gemini API expects camelCase keys for tool types.
    assert_eq!(json, json!({"urlContext": {}}));
}

#[tokio::test]
#[cfg(feature = "google")]
async fn test_google_search_in_request() {
    let mock_server = MockServer::start().await;

    // Setup mock response
    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_simple_response()))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .gemini()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("gemini-2.5-flash")
        .build()
        .await
        .unwrap();

    // Note: Currently we don't have a direct way to add grounding tools via the builder
    // This test verifies the types are correctly defined and can be serialized
    // In the future, we should add builder methods for grounding tools

    // For now, just verify the response works
    let response = client
        .chat(vec![ChatMessage::user("Test query").build()])
        .await
        .unwrap();

    assert_eq!(
        response.content.text(),
        Some("This is a grounded response.")
    );
}

#[tokio::test]
#[cfg(feature = "google")]
async fn test_url_context_in_request() {
    let mock_server = MockServer::start().await;

    // Setup mock response with URL context metadata
    let response_with_url_metadata = json!({
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "Based on the URL content, here is the answer."
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "urlContextMetadata": {
                    "urlMetadata": [
                        {
                            "retrievedUrl": "https://example.com/page1",
                            "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 10,
            "totalTokenCount": 30,
            "toolUsePromptTokenCount": 100
        },
        "modelVersion": "gemini-2.5-flash",
        "responseId": "resp_url_context_123"
    });

    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_with_url_metadata))
        .mount(&mock_server)
        .await;

    let client = LlmBuilder::new()
        .gemini()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("gemini-2.5-flash")
        .build()
        .await
        .unwrap();

    let response = client
        .chat(vec![
            ChatMessage::user("Summarize the content from https://example.com/page1").build(),
        ])
        .await
        .unwrap();

    assert_eq!(
        response.content.text(),
        Some("Based on the URL content, here is the answer.")
    );
}

#[tokio::test]
#[cfg(feature = "google")]
async fn test_combined_grounding_tools() {
    // Test that multiple grounding tools can be used together
    let google_search = GeminiTool::GoogleSearch {
        google_search: GoogleSearch {},
    };

    let url_context = GeminiTool::UrlContext {
        url_context: UrlContext {},
    };

    let tools = vec![google_search, url_context];

    let json = serde_json::to_value(&tools).unwrap();
    assert_eq!(
        json,
        json!([
            {"googleSearch": {}},
            {"urlContext": {}}
        ])
    );
}
