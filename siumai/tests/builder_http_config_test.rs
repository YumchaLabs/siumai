//! Tests for the unified `SiumaiBuilder` HTTP client configuration (basic subset)

#![cfg(feature = "openai")]

use siumai::provider::Siumai;
use std::time::Duration;

/// Test that custom HTTP client takes precedence
#[tokio::test]
async fn test_llm_builder_custom_client_precedence() {
    let custom_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .expect("Failed to build custom client");

    let result = Siumai::builder()
        .with_http_client(custom_client.clone())
        .with_timeout(Duration::from_secs(30)) // Should be ignored
        .openai()
        .api_key("test-key-custom-client")
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with custom client");
}

/// Test that builder works without any advanced options (fallback to simple path)
#[tokio::test]
async fn test_llm_builder_simple_config() {
    let result = Siumai::builder()
        .with_timeout(Duration::from_secs(30))
        .with_user_agent("test-agent/1.0")
        .openai()
        .api_key("test-key-simple")
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with simple config");
}

/// Test that default builder can build client
#[tokio::test]
async fn test_llm_builder_default() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-default")
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with default config");
}

/// Test that builder works with basic configuration
#[tokio::test]
async fn test_llm_builder_basic_config() {
    let result = Siumai::builder()
        .with_timeout(Duration::from_secs(30))
        .with_connect_timeout(Duration::from_secs(10))
        .openai()
        .api_key("test-key-basic")
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with basic config");
}
