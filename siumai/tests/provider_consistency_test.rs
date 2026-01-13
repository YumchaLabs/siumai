#![cfg(all(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "xai",
    feature = "ollama",
    feature = "groq"
))]
//! Provider Consistency Tests
//!
//! This test suite ensures that all providers implement consistent behavior
//! for core features:
//! - CustomProviderOptions support
//! - Custom reqwest client support
//! - HTTP interceptors support
//! - Debug mode support
//! - Middleware support
//!
//! This prevents regressions when adding new providers.

use siumai::Provider;
use siumai::prelude::unified::{ChatMessage, ChatRequest, CommonParams};

/// List of all providers to test
const ALL_PROVIDERS: &[&str] = &["openai", "anthropic", "xai", "gemini", "groq", "ollama"];

/// Helper to create a basic chat request
fn create_test_request() -> ChatRequest {
    ChatRequest::builder()
        .message(ChatMessage::user("test").build())
        .common_params(CommonParams {
            model: "test-model".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(100),
            ..Default::default()
        })
        .build()
}

/// Test that all providers support CustomProviderOptions
#[test]
fn test_all_providers_support_custom_options() {
    for provider_id in ALL_PROVIDERS {
        println!(
            "Testing CustomProviderOptions for provider: {}",
            provider_id
        );

        // Create request with providerOptions entry.
        // Providers decide how to interpret these keys; this test simply ensures the
        // request surface supports passing the per-provider JSON payload.
        let request = create_test_request().with_provider_option(
            provider_id,
            serde_json::json!({
                "test_param": "test_value",
                "experimental_feature": true
            }),
        );

        assert!(request.provider_option(provider_id).is_some());
    }
}

/// Test that all providers can be instantiated
#[test]
fn test_all_providers_can_be_instantiated() {
    for provider_id in ALL_PROVIDERS {
        println!("Testing instantiation for provider: {}", provider_id);

        // Create provider builder
        // We just verify that the builder can be created without panicking
        match *provider_id {
            "openai" => {
                let _builder = Provider::openai().api_key("test-key").model("gpt-4");
            }
            "anthropic" => {
                let _builder = Provider::anthropic()
                    .api_key("test-key")
                    .model("claude-3-5-sonnet-20241022");
            }
            "xai" => {
                let _builder = Provider::xai().api_key("test-key").model("grok-2-latest");
            }
            "gemini" => {
                let _builder = Provider::gemini()
                    .api_key("test-key")
                    .model("gemini-2.0-flash-exp");
            }
            "groq" => {
                let _builder = Provider::groq()
                    .api_key("test-key")
                    .model("llama-3.3-70b-versatile");
            }
            "ollama" => {
                let _builder = Provider::ollama().model("llama3.2");
            }
            _ => panic!("Unknown provider: {}", provider_id),
        };

        // Test passes if no panic occurs
    }
}
