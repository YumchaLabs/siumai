//! Tool warning parity tests for Gemini (Vercel AI SDK aligned).

use serde_json::json;
use siumai::prelude::*;
use siumai_core::types::Warning;
use wiremock::matchers::{header, method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn create_simple_response() -> serde_json::Value {
    json!({
        "candidates": [
            {
                "content": {
                    "parts": [
                        { "text": "ok" }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP"
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 1,
            "totalTokenCount": 2
        },
        "modelVersion": "gemini-2.5-flash",
        "responseId": "resp_123"
    })
}

#[tokio::test]
#[cfg(feature = "google")]
async fn gemini_warns_on_mixed_function_and_provider_tools() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_simple_response()))
        .mount(&mock_server)
        .await;

    let client = Siumai::builder()
        .gemini()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("gemini-2.5-flash")
        .build()
        .await
        .unwrap();

    let tools = vec![
        Tool::function("testFunction".to_string(), "Test".to_string(), json!({})),
        tools::google::google_search(),
    ];

    let response = client
        .chat_with_tools(vec![ChatMessage::user("hi").build()], Some(tools))
        .await
        .unwrap();

    let warnings = response.warnings.unwrap_or_default();
    assert!(
        warnings.iter().any(|w| matches!(
            w,
            Warning::UnsupportedSetting { setting, details: Some(d) }
                if setting == "tools" && d == "combination of function and provider-defined tools"
        )),
        "expected mixed-tools warning, got: {warnings:?}"
    );
}

#[tokio::test]
#[cfg(feature = "google")]
async fn gemini_warns_on_unsupported_url_context_tool_for_non_gemini2_models() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_simple_response()))
        .mount(&mock_server)
        .await;

    let client = Siumai::builder()
        .gemini()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("gemini-1.5-pro")
        .build()
        .await
        .unwrap();

    let response = client
        .chat_with_tools(
            vec![ChatMessage::user("hi").build()],
            Some(vec![tools::google::url_context()]),
        )
        .await
        .unwrap();

    let warnings = response.warnings.unwrap_or_default();
    assert!(
        warnings.iter().any(|w| matches!(
            w,
            Warning::UnsupportedTool { tool_name, details: Some(d) }
                if tool_name == "google.url_context"
                    && d == "The URL context tool is not supported with other Gemini models than Gemini 2."
        )),
        "expected url_context unsupported warning, got: {warnings:?}"
    );
}

#[tokio::test]
#[cfg(feature = "google")]
async fn gemini_warns_on_unsupported_file_search_tool_for_non_gemini_2_5_models() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_simple_response()))
        .mount(&mock_server)
        .await;

    let client = Siumai::builder()
        .gemini()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("gemini-2.0-pro")
        .build()
        .await
        .unwrap();

    let response = client
        .chat_with_tools(
            vec![ChatMessage::user("hi").build()],
            Some(vec![tools::google::file_search()]),
        )
        .await
        .unwrap();

    let warnings = response.warnings.unwrap_or_default();
    assert!(
        warnings.iter().any(|w| matches!(
            w,
            Warning::UnsupportedTool { tool_name, details: Some(d) }
                if tool_name == "google.file_search"
                    && d == "The file search tool is only supported with Gemini 2.5 models."
        )),
        "expected file_search unsupported warning, got: {warnings:?}"
    );
}

#[tokio::test]
#[cfg(feature = "google")]
async fn gemini_warns_on_unsupported_code_execution_tool_for_non_gemini2_models() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_simple_response()))
        .mount(&mock_server)
        .await;

    let client = Siumai::builder()
        .gemini()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("gemini-1.5-pro")
        .build()
        .await
        .unwrap();

    let response = client
        .chat_with_tools(
            vec![ChatMessage::user("hi").build()],
            Some(vec![tools::google::code_execution()]),
        )
        .await
        .unwrap();

    let warnings = response.warnings.unwrap_or_default();
    assert!(
        warnings.iter().any(|w| matches!(
            w,
            Warning::UnsupportedTool { tool_name, details: Some(d) }
                if tool_name == "google.code_execution"
                    && d == "The code execution tools is not supported with other Gemini models than Gemini 2."
        )),
        "expected code_execution unsupported warning, got: {warnings:?}"
    );
}

#[tokio::test]
#[cfg(feature = "google")]
async fn gemini_warns_on_unknown_provider_tool() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_simple_response()))
        .mount(&mock_server)
        .await;

    let client = Siumai::builder()
        .gemini()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("gemini-2.5-flash")
        .build()
        .await
        .unwrap();

    let response = client
        .chat_with_tools(
            vec![ChatMessage::user("hi").build()],
            Some(vec![Tool::provider_defined(
                "google.unknown_tool",
                "unknown_tool",
            )]),
        )
        .await
        .unwrap();

    let warnings = response.warnings.unwrap_or_default();
    assert!(
        warnings.iter().any(|w| matches!(
            w,
            Warning::UnsupportedTool { tool_name, details: None }
                if tool_name == "google.unknown_tool"
        )),
        "expected unknown tool warning, got: {warnings:?}"
    );
}

#[tokio::test]
#[cfg(feature = "google")]
async fn gemini_warns_on_unsupported_enterprise_web_search_tool_for_non_gemini2_models() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_simple_response()))
        .mount(&mock_server)
        .await;

    let client = Siumai::builder()
        .gemini()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("gemini-1.5-pro")
        .build()
        .await
        .unwrap();

    let response = client
        .chat_with_tools(
            vec![ChatMessage::user("hi").build()],
            Some(vec![tools::google::enterprise_web_search()]),
        )
        .await
        .unwrap();

    let warnings = response.warnings.unwrap_or_default();
    assert!(
        warnings.iter().any(|w| matches!(
            w,
            Warning::UnsupportedTool { tool_name, details: Some(d) }
                if tool_name == "google.enterprise_web_search"
                    && d == "Enterprise Web Search requires Gemini 2.0 or newer."
        )),
        "expected enterprise_web_search unsupported warning, got: {warnings:?}"
    );
}

#[tokio::test]
#[cfg(feature = "google")]
async fn gemini_warns_on_unsupported_vertex_rag_store_tool_for_non_gemini2_models() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_simple_response()))
        .mount(&mock_server)
        .await;

    let client = Siumai::builder()
        .gemini()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("gemini-1.5-pro")
        .build()
        .await
        .unwrap();

    let response = client
        .chat_with_tools(
            vec![ChatMessage::user("hi").build()],
            Some(vec![tools::google::vertex_rag_store().with_args(json!({
                "ragCorpus": "projects/my-project/locations/us-central1/ragCorpora/my-rag-corpus",
                "topK": 5
            }))]),
        )
        .await
        .unwrap();

    let warnings = response.warnings.unwrap_or_default();
    assert!(
        warnings.iter().any(|w| matches!(
            w,
            Warning::UnsupportedTool { tool_name, details: Some(d) }
                if tool_name == "google.vertex_rag_store"
                    && d == "The RAG store tool is not supported with other Gemini models than Gemini 2."
        )),
        "expected vertex_rag_store unsupported warning, got: {warnings:?}"
    );
}

#[tokio::test]
#[cfg(feature = "google")]
async fn gemini_warns_on_unsupported_google_maps_tool_for_non_gemini2_models() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r"/models/.*:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(create_simple_response()))
        .mount(&mock_server)
        .await;

    let client = Siumai::builder()
        .gemini()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("gemini-1.5-pro")
        .build()
        .await
        .unwrap();

    let response = client
        .chat_with_tools(
            vec![ChatMessage::user("hi").build()],
            Some(vec![tools::google::google_maps()]),
        )
        .await
        .unwrap();

    let warnings = response.warnings.unwrap_or_default();
    assert!(
        warnings.iter().any(|w| matches!(
            w,
            Warning::UnsupportedTool { tool_name, details: Some(d) }
                if tool_name == "google.google_maps"
                    && d == "The Google Maps grounding tool is not supported with Gemini models other than Gemini 2 or newer."
        )),
        "expected google_maps unsupported warning, got: {warnings:?}"
    );
}
