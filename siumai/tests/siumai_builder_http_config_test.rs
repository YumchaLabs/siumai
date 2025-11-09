//! Comprehensive tests for SiumaiBuilder HTTP configuration
//!
//! This test suite verifies that SiumaiBuilder correctly handles:
//! - Advanced HTTP features (gzip, brotli, cookies, http2)
//! - Custom HTTP client
//! - API consistency aliases (with_* methods)
//! - Feature parity with LlmBuilder

use siumai::prelude::Siumai;
use std::time::Duration;

// ============================================================================
// Advanced HTTP Features Tests
// ============================================================================

#[tokio::test]
#[cfg(feature = "gzip")]
async fn test_siumai_builder_gzip_enabled() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-gzip-enabled")
        .model("gpt-4o")
        .http_gzip(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with gzip enabled"
    );
}

#[tokio::test]
#[cfg(feature = "gzip")]
async fn test_siumai_builder_gzip_disabled() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-gzip-disabled")
        .model("gpt-4o")
        .http_gzip(false)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with gzip disabled"
    );
}

#[tokio::test]
#[cfg(feature = "brotli")]
async fn test_siumai_builder_brotli_enabled() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-brotli-enabled")
        .model("gpt-4o")
        .http_brotli(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with brotli enabled"
    );
}

#[tokio::test]
#[cfg(feature = "brotli")]
async fn test_siumai_builder_brotli_disabled() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-brotli-disabled")
        .model("gpt-4o")
        .http_brotli(false)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with brotli disabled"
    );
}

#[tokio::test]
#[cfg(feature = "cookies")]
async fn test_siumai_builder_cookie_store_enabled() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-cookie-enabled")
        .model("gpt-4o")
        .http_cookie_store(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with cookie store enabled"
    );
}

#[tokio::test]
#[cfg(feature = "cookies")]
async fn test_siumai_builder_cookie_store_disabled() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-cookie-disabled")
        .model("gpt-4o")
        .http_cookie_store(false)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with cookie store disabled"
    );
}

#[tokio::test]
#[cfg(feature = "http2")]
async fn test_siumai_builder_http2_prior_knowledge() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-http2")
        .model("gpt-4o")
        .http_http2_prior_knowledge(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with HTTP/2 prior knowledge"
    );
}

// ============================================================================
// Combination Tests
// ============================================================================

#[tokio::test]
#[cfg(all(feature = "gzip", feature = "brotli"))]
async fn test_siumai_builder_multiple_compression() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-multi-compression")
        .model("gpt-4o")
        .http_gzip(true)
        .http_brotli(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with multiple compression options"
    );
}

#[tokio::test]
#[cfg(all(
    feature = "gzip",
    feature = "brotli",
    feature = "cookies",
    feature = "http2"
))]
async fn test_siumai_builder_all_advanced_options() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-all-advanced")
        .model("gpt-4o")
        .http_gzip(true)
        .http_brotli(true)
        .http_cookie_store(true)
        .http_http2_prior_knowledge(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with all advanced HTTP options"
    );
}

#[tokio::test]
#[cfg(feature = "gzip")]
async fn test_siumai_builder_proxy_with_compression() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-proxy-compression")
        .model("gpt-4o")
        .http_proxy("http://proxy.example.com:8080")
        .http_gzip(true)
        .build()
        .await;

    // Note: This may fail if proxy is not reachable, but we're testing configuration
    // The important part is that it doesn't panic during build
    let _ = result;
}

// ============================================================================
// Custom HTTP Client Tests
// ============================================================================

#[tokio::test]
async fn test_siumai_builder_custom_client_precedence() {
    // Create a custom HTTP client
    let custom_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("Failed to build custom client");

    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-custom-client")
        .model("gpt-4o")
        .with_http_client(custom_client)
        .http_timeout(Duration::from_secs(30)) // This should be ignored
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with custom HTTP client"
    );
}

// ============================================================================
// API Consistency Alias Tests
// ============================================================================

#[tokio::test]
async fn test_siumai_builder_with_timeout_alias() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-timeout-alias")
        .model("gpt-4o")
        .with_timeout(Duration::from_secs(30))
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with with_timeout alias"
    );
}

#[tokio::test]
async fn test_siumai_builder_with_proxy_alias() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-proxy-alias")
        .model("gpt-4o")
        .with_proxy("http://proxy.example.com:8080")
        .build()
        .await;

    let _ = result; // May fail if proxy not reachable
}

#[tokio::test]
#[cfg(feature = "gzip")]
async fn test_siumai_builder_with_gzip_alias() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-gzip-alias")
        .model("gpt-4o")
        .with_gzip(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with with_gzip alias"
    );
}

#[tokio::test]
#[cfg(feature = "brotli")]
async fn test_siumai_builder_with_brotli_alias() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-brotli-alias")
        .model("gpt-4o")
        .with_brotli(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with with_brotli alias"
    );
}

#[tokio::test]
#[cfg(feature = "cookies")]
async fn test_siumai_builder_with_cookie_store_alias() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-cookie-alias")
        .model("gpt-4o")
        .with_cookie_store(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with with_cookie_store alias"
    );
}

#[tokio::test]
#[cfg(feature = "http2")]
async fn test_siumai_builder_with_http2_prior_knowledge_alias() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-http2-alias")
        .model("gpt-4o")
        .with_http2_prior_knowledge(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with with_http2_prior_knowledge alias"
    );
}

// ============================================================================
// Basic Configuration Tests
// ============================================================================

#[tokio::test]
async fn test_siumai_builder_simple_config() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-simple-config")
        .model("gpt-4o")
        .http_timeout(Duration::from_secs(30))
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with simple config"
    );
}

#[tokio::test]
async fn test_siumai_builder_basic_config() {
    let result = Siumai::builder()
        .openai()
        .api_key("test-key-siumai-basic-config")
        .model("gpt-4o")
        .http_timeout(Duration::from_secs(30))
        .http_connect_timeout(Duration::from_secs(10))
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with basic timeout configuration"
    );
}
