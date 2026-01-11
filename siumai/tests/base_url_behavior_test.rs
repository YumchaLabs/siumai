#![cfg(any(feature = "openai", feature = "google"))]
//! Base URL behavior tests for unified builder (`SiumaiBuilder`).
//!
//! These tests ensure that when a custom `base_url` is provided to the
//! unified `Siumai::builder()`, it is treated as the full API prefix
//! and no provider-default path segments like `/v1` or `/v1beta` are
//! automatically appended. This mirrors the behavior of the Vercel AI SDK.

use siumai::provider::Siumai;

/// OpenAI: custom base_url without trailing slash should be used as-is.
#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_custom_base_url_is_used_as_full_prefix() {
    let client = Siumai::builder()
        .openai()
        .api_key("test-key-openai-base-url")
        .base_url("https://example.com")
        .model("gpt-4o-mini")
        .build()
        .await
        .expect("failed to build OpenAI client");

    let inner = client.client();
    let any = inner.as_any();
    let openai = any
        .downcast_ref::<siumai::provider_ext::openai::OpenAiClient>()
        .expect("expected OpenAiClient");

    assert_eq!(
        openai.base_url(),
        "https://example.com",
        "custom base_url should fully override default OpenAI prefix without appending /v1",
    );
}

/// OpenAI: custom base_url with trailing slash should be normalized by trimming
/// the trailing `/` but still not append `/v1`.
#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_custom_base_url_trailing_slash_is_trimmed() {
    let client = Siumai::builder()
        .openai()
        .api_key("test-key-openai-base-url-trim")
        .base_url("https://example.com/")
        .model("gpt-4o-mini")
        .build()
        .await
        .expect("failed to build OpenAI client");

    let inner = client.client();
    let any = inner.as_any();
    let openai = any
        .downcast_ref::<siumai::provider_ext::openai::OpenAiClient>()
        .expect("expected OpenAiClient");

    assert_eq!(
        openai.base_url(),
        "https://example.com",
        "custom base_url with trailing slash should be normalized without adding /v1",
    );
}

/// Gemini: custom base_url without trailing slash should be used as-is.
#[cfg(feature = "google")]
#[tokio::test]
async fn gemini_custom_base_url_is_used_as_full_prefix() {
    let client = Siumai::builder()
        .gemini()
        .api_key("test-key-gemini-base-url")
        .base_url("https://example.org")
        .model("gemini-1.5-pro")
        .build()
        .await
        .expect("failed to build Gemini client");

    let inner = client.client();
    let any = inner.as_any();
    let gemini = any
        .downcast_ref::<siumai::provider_ext::gemini::GeminiClient>()
        .expect("expected GeminiClient");

    assert_eq!(
        gemini.base_url(),
        "https://example.org",
        "custom base_url should fully override default Gemini prefix without appending /v1beta",
    );
}

/// Gemini: custom base_url with trailing slash should be normalized by trimming
/// the trailing `/` but still not append `/v1beta`.
#[cfg(feature = "google")]
#[tokio::test]
async fn gemini_custom_base_url_trailing_slash_is_trimmed() {
    let client = Siumai::builder()
        .gemini()
        .api_key("test-key-gemini-base-url-trim")
        .base_url("https://example.org/")
        .model("gemini-1.5-pro")
        .build()
        .await
        .expect("failed to build Gemini client");

    let inner = client.client();
    let any = inner.as_any();
    let gemini = any
        .downcast_ref::<siumai::provider_ext::gemini::GeminiClient>()
        .expect("expected GeminiClient");

    assert_eq!(
        gemini.base_url(),
        "https://example.org",
        "custom base_url with trailing slash should be normalized without adding /v1beta",
    );
}
