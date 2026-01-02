//! Test JSON repair functionality in non-streaming responses
//!
//! This test verifies that malformed JSON in HTTP responses is automatically
//! repaired when the `json-repair` feature is enabled.

#[cfg(feature = "json-repair")]
#[tokio::test]
async fn test_json_repair_in_non_streaming_response() {
    use siumai::prelude::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // Start mock server
    let mock_server = MockServer::start().await;

    // Mock response with malformed JSON (unquoted keys, trailing comma)
    let malformed_json = r#"{
        id: "chatcmpl-123",
        object: "chat.completion",
        created: 1677652288,
        model: "gpt-4",
        choices: [{
            index: 0,
            message: {
                role: "assistant",
                content: "Hello! How can I help you today?",
            },
            finish_reason: "stop",
        }],
        usage: {
            prompt_tokens: 10,
            completion_tokens: 9,
            total_tokens: 19,
        },
    }"#;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_string(malformed_json))
        .mount(&mock_server)
        .await;

    // Create client pointing to mock server
    let client = Siumai::builder()
        .openai()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("gpt-4")
        .build()
        .await
        .expect("Failed to build client");

    // Make request - should succeed with JSON repair
    let messages = vec![user!("Hello")];
    let response = client.chat(messages).await;

    // Verify response was successfully parsed despite malformed JSON
    assert!(
        response.is_ok(),
        "Expected successful response with JSON repair, got error: {:?}",
        response.err()
    );

    let response = response.unwrap();
    assert_eq!(
        response.content_text().unwrap(),
        "Hello! How can I help you today?"
    );
}

#[cfg(not(feature = "json-repair"))]
#[tokio::test]
async fn test_malformed_json_fails_without_repair() {
    use siumai::prelude::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // Start mock server
    let mock_server = MockServer::start().await;

    // Mock response with malformed JSON
    let malformed_json = r#"{
        id: "chatcmpl-123",
        object: "chat.completion",
        choices: [{message: {content: "test"}}]
    }"#;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_string(malformed_json))
        .mount(&mock_server)
        .await;

    // Create client
    let client = Siumai::builder()
        .openai()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("gpt-4")
        .build()
        .await
        .expect("Failed to build client");

    // Make request - should fail without JSON repair
    let messages = vec![user!("Hello")];
    let response = client.chat(messages).await;

    // Verify response fails due to malformed JSON
    assert!(
        response.is_err(),
        "Expected error without JSON repair feature"
    );
}

#[cfg(feature = "json-repair")]
#[tokio::test]
async fn test_valid_json_has_zero_overhead() {
    use siumai::prelude::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // Start mock server
    let mock_server = MockServer::start().await;

    // Mock response with valid JSON
    let valid_json = r#"{
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Valid JSON response"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "total_tokens": 13
        }
    }"#;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_string(valid_json))
        .mount(&mock_server)
        .await;

    // Create client
    let client = Siumai::builder()
        .openai()
        .api_key("test-api-key")
        .base_url(mock_server.uri())
        .model("gpt-4")
        .build()
        .await
        .expect("Failed to build client");

    // Make request - should succeed with fast path (no repair needed)
    let messages = vec![user!("Hello")];
    let response = client.chat(messages).await;

    assert!(response.is_ok());
    let response = response.unwrap();
    assert_eq!(response.content_text().unwrap(), "Valid JSON response");
}
