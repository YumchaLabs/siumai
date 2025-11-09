//! Tests for LlmBuilder HTTP client configuration
//!
//! This test suite verifies that all HTTP configuration options in LlmBuilder
//! are correctly applied when building provider clients.

use siumai::builder::LlmBuilder;
use std::time::Duration;

/// Test that gzip compression is correctly applied
#[tokio::test]
#[cfg(feature = "gzip")]
async fn test_llm_builder_gzip_enabled() {
    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_gzip(true)
        .openai()
        .api_key("test-key-gzip-enabled")
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with gzip enabled");
}

/// Test that gzip compression can be disabled
#[tokio::test]
#[cfg(feature = "gzip")]
async fn test_llm_builder_gzip_disabled() {
    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_gzip(false)
        .openai()
        .api_key("test-key-gzip-disabled")
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with gzip disabled");
}

/// Test that brotli compression is correctly applied
#[tokio::test]
#[cfg(feature = "brotli")]
async fn test_llm_builder_brotli_enabled() {
    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_brotli(true)
        .openai()
        .api_key("test-key-brotli-enabled")
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with brotli enabled");
}

/// Test that brotli compression can be disabled
#[tokio::test]
#[cfg(feature = "brotli")]
async fn test_llm_builder_brotli_disabled() {
    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_brotli(false)
        .openai()
        .api_key("test-key-brotli-disabled")
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with brotli disabled"
    );
}

/// Test that cookie store is correctly applied
#[tokio::test]
#[cfg(feature = "cookies")]
async fn test_llm_builder_cookie_store_enabled() {
    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_cookie_store(true)
        .openai()
        .api_key("test-key-cookie-enabled")
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with cookie store enabled"
    );
}

/// Test that cookie store can be disabled
#[tokio::test]
#[cfg(feature = "cookies")]
async fn test_llm_builder_cookie_store_disabled() {
    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_cookie_store(false)
        .openai()
        .api_key("test-key-cookie-disabled")
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with cookie store disabled"
    );
}

/// Test that HTTP/2 prior knowledge is correctly applied
#[tokio::test]
#[cfg(feature = "http2")]
async fn test_llm_builder_http2_prior_knowledge() {
    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_http2_prior_knowledge(true)
        .openai()
        .api_key("test-key-http2")
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with HTTP/2 prior knowledge"
    );
}

/// Test that multiple compression options can be combined
#[tokio::test]
#[cfg(all(feature = "gzip", feature = "brotli"))]
async fn test_llm_builder_multiple_compression() {
    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_gzip(true)
        .with_brotli(true)
        .openai()
        .api_key("test-key-multi-compression")
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with multiple compression options"
    );
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
        .api_key("test-key-all-options")
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with all advanced options"
    );
}

/// Test that proxy configuration works with advanced options
#[tokio::test]
#[cfg(all(feature = "gzip", feature = "brotli"))]
async fn test_llm_builder_proxy_with_compression() {
    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_proxy("http://proxy.example.com:8080")
        .with_gzip(true)
        .with_brotli(true)
        .openai()
        .api_key("test-key-proxy")
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with proxy and compression"
    );
}

/// Test that custom HTTP client takes precedence
#[tokio::test]
async fn test_llm_builder_custom_client_precedence() {
    let custom_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .expect("Failed to build custom client");

    let result = LlmBuilder::new()
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
    let result = LlmBuilder::new()
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
    let result = LlmBuilder::new()
        .openai()
        .api_key("test-key-default")
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with default config");
}

/// Test with_defaults() factory method
#[tokio::test]
#[cfg(all(feature = "gzip", feature = "brotli"))]
async fn test_llm_builder_with_defaults() {
    let result = LlmBuilder::with_defaults()
        .openai()
        .api_key("test-key-with-defaults")
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with defaults factory"
    );
}

/// Test fast() factory method
#[tokio::test]
async fn test_llm_builder_fast() {
    let result = LlmBuilder::fast()
        .openai()
        .api_key("test-key-fast")
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with fast factory");
}

/// Test long_running() factory method
#[tokio::test]
async fn test_llm_builder_long_running() {
    let result = LlmBuilder::long_running()
        .openai()
        .api_key("test-key-long-running")
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build client with long_running factory"
    );
}

/// Test that builder works with basic configuration
#[tokio::test]
async fn test_llm_builder_basic_config() {
    let result = LlmBuilder::new()
        .with_timeout(Duration::from_secs(30))
        .with_connect_timeout(Duration::from_secs(10))
        .openai()
        .api_key("test-key-basic")
        .build()
        .await;

    assert!(result.is_ok(), "Failed to build client with basic config");
}
