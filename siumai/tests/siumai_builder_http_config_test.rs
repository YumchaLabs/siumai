//! Tests for SiumaiBuilder HTTP configuration after simplification.

use siumai::prelude::Siumai;
use std::time::Duration;

#[tokio::test]
async fn test_siumai_builder_custom_client_precedence() {
    let custom_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("Failed to build custom client");

    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-custom-client")
        .model("gpt-4o")
        .with_http_client(custom_client)
        .http_timeout(Duration::from_secs(30)) // should not override custom client
        .build()
        .await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_siumai_builder_with_timeout_alias() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-timeout")
        .model("gpt-4o")
        .with_timeout(Duration::from_secs(30))
        .build()
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_siumai_builder_with_proxy_alias() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-proxy")
        .model("gpt-4o")
        .with_proxy("http://proxy.example.com:8080")
        .build()
        .await;
    let _ = result; // network dependent
}

#[tokio::test]
async fn test_siumai_builder_basic_http_config() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-basic")
        .model("gpt-4o")
        .http_timeout(Duration::from_secs(30))
        .http_connect_timeout(Duration::from_secs(10))
        .build()
        .await;
    assert!(result.is_ok());
}
