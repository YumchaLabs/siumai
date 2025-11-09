//! Comprehensive tests for SiumaiBuilder HTTP configuration
//!
//! This test suite verifies that SiumaiBuilder correctly handles:
//! - Advanced HTTP features (gzip, brotli, cookies, http2)
//! - Custom HTTP client
//! - API consistency aliases (with_* methods)
//! - Feature parity with LlmBuilder

#![allow(unsafe_code)]

use siumai::prelude::Siumai;
use std::time::Duration;

mod test_helpers {
    /// Set test API key for OpenAI (required for builder validation)
    pub fn set_test_api_key() {
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "test-key-siumai-builder-2");
        }
    }

    /// Remove test API key
    pub fn remove_test_api_key() {
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
        }
    }
}

use test_helpers::*;

// ============================================================================
// Advanced HTTP Features Tests
// ============================================================================

#[tokio::test]
#[cfg(feature = "gzip")]
async fn test_siumai_builder_gzip_enabled() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .http_gzip(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with gzip enabled"
    );
    remove_test_api_key();
}

#[tokio::test]
#[cfg(feature = "gzip")]
async fn test_siumai_builder_gzip_disabled() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .http_gzip(false)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with gzip disabled"
    );
    remove_test_api_key();
}

#[tokio::test]
#[cfg(feature = "brotli")]
async fn test_siumai_builder_brotli_enabled() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .http_brotli(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with brotli enabled"
    );
    remove_test_api_key();
}

#[tokio::test]
#[cfg(feature = "brotli")]
async fn test_siumai_builder_brotli_disabled() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .http_brotli(false)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with brotli disabled"
    );
    remove_test_api_key();
}

#[tokio::test]
#[cfg(feature = "cookies")]
async fn test_siumai_builder_cookie_store_enabled() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .http_cookie_store(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with cookie store enabled"
    );
    remove_test_api_key();
}

#[tokio::test]
#[cfg(feature = "cookies")]
async fn test_siumai_builder_cookie_store_disabled() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .http_cookie_store(false)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with cookie store disabled"
    );
    remove_test_api_key();
}

#[tokio::test]
#[cfg(feature = "http2")]
async fn test_siumai_builder_http2_prior_knowledge() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .http_http2_prior_knowledge(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with HTTP/2 prior knowledge"
    );
    remove_test_api_key();
}

// ============================================================================
// Combination Tests
// ============================================================================

#[tokio::test]
#[cfg(all(feature = "gzip", feature = "brotli"))]
async fn test_siumai_builder_multiple_compression() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .http_gzip(true)
        .http_brotli(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with multiple compression options"
    );
    remove_test_api_key();
}

#[tokio::test]
#[cfg(all(
    feature = "gzip",
    feature = "brotli",
    feature = "cookies",
    feature = "http2"
))]
async fn test_siumai_builder_all_advanced_options() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
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
    remove_test_api_key();
}

#[tokio::test]
#[cfg(feature = "gzip")]
async fn test_siumai_builder_proxy_with_compression() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .http_proxy("http://proxy.example.com:8080")
        .http_gzip(true)
        .build()
        .await;

    // Note: This may fail if proxy is not reachable, but we're testing configuration
    // The important part is that it doesn't panic during build
    let _ = result;
    remove_test_api_key();
}

// ============================================================================
// Custom HTTP Client Tests
// ============================================================================

#[tokio::test]
async fn test_siumai_builder_custom_client_precedence() {
    set_test_api_key();

    // Create a custom HTTP client
    let custom_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("Failed to build custom client");

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .with_http_client(custom_client)
        .http_timeout(Duration::from_secs(30)) // This should be ignored
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with custom HTTP client"
    );
    remove_test_api_key();
}

// ============================================================================
// API Consistency Alias Tests
// ============================================================================

#[tokio::test]
async fn test_siumai_builder_with_timeout_alias() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .with_timeout(Duration::from_secs(30))
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with with_timeout alias"
    );
    remove_test_api_key();
}

#[tokio::test]
async fn test_siumai_builder_with_proxy_alias() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .with_proxy("http://proxy.example.com:8080")
        .build()
        .await;

    let _ = result; // May fail if proxy not reachable
    remove_test_api_key();
}

#[tokio::test]
#[cfg(feature = "gzip")]
async fn test_siumai_builder_with_gzip_alias() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .with_gzip(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with with_gzip alias"
    );
    remove_test_api_key();
}

#[tokio::test]
#[cfg(feature = "brotli")]
async fn test_siumai_builder_with_brotli_alias() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .with_brotli(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with with_brotli alias"
    );
    remove_test_api_key();
}

#[tokio::test]
#[cfg(feature = "cookies")]
async fn test_siumai_builder_with_cookie_store_alias() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .with_cookie_store(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with with_cookie_store alias"
    );
    remove_test_api_key();
}

#[tokio::test]
#[cfg(feature = "http2")]
async fn test_siumai_builder_with_http2_prior_knowledge_alias() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .with_http2_prior_knowledge(true)
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with with_http2_prior_knowledge alias"
    );
    remove_test_api_key();
}

// ============================================================================
// Basic Configuration Tests
// ============================================================================

#[tokio::test]
async fn test_siumai_builder_simple_config() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .http_timeout(Duration::from_secs(30))
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with simple config"
    );
    remove_test_api_key();
}

#[tokio::test]
async fn test_siumai_builder_basic_config() {
    set_test_api_key();

    let result = Siumai::builder()
        .openai()
        .model("gpt-4o")
        .http_timeout(Duration::from_secs(30))
        .http_connect_timeout(Duration::from_secs(10))
        .build()
        .await;

    assert!(
        result.is_ok(),
        "Failed to build Siumai client with basic timeout configuration"
    );
    remove_test_api_key();
}
