//! Tests for LlmBuilder HTTP client configuration
//!
//! This test suite verifies that all HTTP configuration options in LlmBuilder
//! are correctly applied when building provider clients.

#![allow(unsafe_code)]

use siumai::builder::LlmBuilder;
use std::time::Duration;

// Helper functions for safe environment variable manipulation in tests
fn set_test_api_key() {
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "test-key-llm-builder");
    }
}

fn remove_test_api_key() {
    unsafe {
        std::env::remove_var("OPENAI_API_KEY");
    }
}

/// Test that gzip compression is correctly applied
#[tokio::test]
#[cfg(feature = "gzip")]
async fn test_llm_builder_gzip_enabled() {
    set_test_api_key();

    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_gzip(true)
        .openai()
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with gzip enabled");
    remove_test_api_key();
}

/// Test that gzip compression can be disabled
#[tokio::test]
#[cfg(feature = "gzip")]
async fn test_llm_builder_gzip_disabled() {
    set_test_api_key();

    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_gzip(false)
        .openai()
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with gzip disabled");
    remove_test_api_key();
}

/// Test that brotli compression is correctly applied
#[tokio::test]
#[cfg(feature = "brotli")]
async fn test_llm_builder_brotli_enabled() {
    set_test_api_key();

    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_brotli(true)
        .openai()
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with brotli enabled");
    remove_test_api_key();
}

/// Test that brotli compression can be disabled
#[tokio::test]
#[cfg(feature = "brotli")]
async fn test_llm_builder_brotli_disabled() {
    set_test_api_key();

    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_brotli(false)
        .openai()
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with brotli disabled"
    );
    remove_test_api_key();
}

/// Test that cookie store is correctly applied
#[tokio::test]
#[cfg(feature = "cookies")]
async fn test_llm_builder_cookie_store_enabled() {
    set_test_api_key();

    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_cookie_store(true)
        .openai()
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with cookie store enabled"
    );
    remove_test_api_key();
}

/// Test that cookie store can be disabled
#[tokio::test]
#[cfg(feature = "cookies")]
async fn test_llm_builder_cookie_store_disabled() {
    set_test_api_key();

    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_cookie_store(false)
        .openai()
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with cookie store disabled"
    );
    remove_test_api_key();
}

/// Test that HTTP/2 prior knowledge is correctly applied
#[tokio::test]
#[cfg(feature = "http2")]
async fn test_llm_builder_http2_prior_knowledge() {
    set_test_api_key();

    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_http2_prior_knowledge(true)
        .openai()
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with HTTP/2 prior knowledge"
    );
    remove_test_api_key();
}

/// Test that multiple compression options can be combined
#[tokio::test]
#[cfg(all(feature = "gzip", feature = "brotli"))]
async fn test_llm_builder_multiple_compression() {
    set_test_api_key();

    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_gzip(true)
        .with_brotli(true)
        .openai()
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with multiple compression options"
    );
    remove_test_api_key();
}

/// Test that all advanced HTTP options can be combined
#[tokio::test]
#[cfg(all(
    feature = "gzip",
    feature = "brotli",
    feature = "http2",
    feature = "cookies"
))]
async fn test_llm_builder_all_advanced_options() {
    set_test_api_key();

    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_connect_timeout(Duration::from_secs(10))
        .with_user_agent("test-agent/1.0")
        .with_gzip(true)
        .with_brotli(true)
        .with_http2_prior_knowledge(true)
        .with_cookie_store(true)
        .with_header("X-Custom-Header", "test-value")
        .openai()
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with all advanced options"
    );
    remove_test_api_key();
}

/// Test that proxy configuration works with advanced options
#[tokio::test]
#[cfg(all(feature = "gzip", feature = "brotli"))]
async fn test_llm_builder_proxy_with_compression() {
    set_test_api_key();

    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_proxy("http://proxy.example.com:8080")
        .with_gzip(true)
        .with_brotli(true)
        .openai()
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with proxy and compression"
    );
    remove_test_api_key();
}

/// Test that custom HTTP client takes precedence
#[tokio::test]
async fn test_llm_builder_custom_client_precedence() {
    set_test_api_key();

    let custom_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .expect("Failed to build custom client");

    let result = LlmBuilder::new()
        .with_http_client(custom_client.clone())
        .with_timeout(Duration::from_secs(30)) // Should be ignored
        .openai()
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with custom client");
    remove_test_api_key();
}

/// Test that builder works without any advanced options (fallback to simple path)
#[tokio::test]
async fn test_llm_builder_simple_config() {
    set_test_api_key();

    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_user_agent("test-agent/1.0")
        .openai()
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with simple config");
    remove_test_api_key();
}

/// Test that default builder can build client
#[tokio::test]
async fn test_llm_builder_default() {
    set_test_api_key();

    let result = LlmBuilder::new().openai().build().await;

    assert!(result.is_ok(), "Failed to build client with default config");
    remove_test_api_key();
}

/// Test with_defaults() factory method
#[tokio::test]
#[cfg(all(feature = "gzip", feature = "brotli"))]
async fn test_llm_builder_with_defaults() {
    set_test_api_key();

    let result = LlmBuilder::with_defaults().openai().build().await;

    assert!(
        result.is_ok(),
        "Failed to build client with defaults factory"
    );
    remove_test_api_key();
}

/// Test fast() factory method
#[tokio::test]
async fn test_llm_builder_fast() {
    set_test_api_key();

    let result = LlmBuilder::fast().openai().build().await;

    assert!(result.is_ok(), "Failed to build client with fast factory");
    remove_test_api_key();
}

/// Test long_running() factory method
#[tokio::test]
async fn test_llm_builder_long_running() {
    set_test_api_key();

    let result = LlmBuilder::long_running().openai().build().await;

    assert!(
        result.is_ok(),
        "Failed to build client with long_running factory"
    );
    remove_test_api_key();
}

/// Test that builder works with basic configuration
#[tokio::test]
async fn test_llm_builder_basic_config() {
    set_test_api_key();

    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_connect_timeout(Duration::from_secs(10))
        .openai()
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with basic config");
    remove_test_api_key();
}
