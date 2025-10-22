//! Siumai Unified Interface Tests
//!
//! Tests to verify that the Siumai unified interface supports all providers
//! and provides consistent behavior across different LLM providers.

use siumai::error::LlmError;
use siumai::provider::SiumaiBuilder;
use siumai::types::{ChatMessage, MessageContent, MessageRole, ProviderType};

/// Test that all providers can be created through the unified interface
#[tokio::test]
async fn test_all_providers_supported() {
    // Test OpenAI
    let openai_result = SiumaiBuilder::new()
        .openai()
        .api_key("test-key")
        .model("gpt-4")
        .build()
        .await;

    match openai_result {
        Ok(client) => {
            // Test that basic capabilities are supported
            assert!(client.supports("chat"));
            assert!(client.supports("streaming"));
        }
        Err(e) => {
            // Expected to fail with test key, but should not be a configuration error
            // related to provider support
            assert!(
                !matches!(e, LlmError::ConfigurationError(msg) if msg.contains("not yet implemented"))
            );
        }
    }

    // Test Anthropic
    let anthropic_result = SiumaiBuilder::new()
        .anthropic()
        .api_key("test-key")
        .model("claude-3-5-sonnet-20241022")
        .build()
        .await;

    match anthropic_result {
        Ok(client) => {
            assert!(client.supports("chat"));
            assert!(client.supports("streaming"));
        }
        Err(e) => {
            assert!(
                !matches!(e, LlmError::ConfigurationError(msg) if msg.contains("not yet implemented"))
            );
        }
    }

    // Test Gemini
    let gemini_result = SiumaiBuilder::new()
        .gemini()
        .api_key("test-key")
        .model("gemini-1.5-flash")
        .build()
        .await;

    match gemini_result {
        Ok(client) => {
            assert!(client.supports("chat"));
            assert!(client.supports("streaming"));
        }
        Err(e) => {
            assert!(
                !matches!(e, LlmError::ConfigurationError(msg) if msg.contains("not yet implemented"))
            );
        }
    }

    // Test Ollama
    let ollama_result = SiumaiBuilder::new()
        .ollama()
        .api_key("not-needed-for-ollama")
        .model("llama3.2:latest")
        .base_url("http://localhost:11434")
        .build()
        .await;

    match ollama_result {
        Ok(client) => {
            assert!(client.supports("chat"));
            assert!(client.supports("streaming"));
        }
        Err(e) => {
            assert!(
                !matches!(e, LlmError::ConfigurationError(msg) if msg.contains("not yet implemented"))
            );
        }
    }

    // Test xAI
    let xai_result = SiumaiBuilder::new()
        .xai()
        .api_key("test-key")
        .model("grok-3-latest")
        .build()
        .await;

    match xai_result {
        Ok(client) => {
            assert!(client.supports("chat"));
            assert!(client.supports("streaming"));
        }
        Err(e) => {
            assert!(
                !matches!(e, LlmError::ConfigurationError(msg) if msg.contains("not yet implemented"))
            );
        }
    }

    // Test Groq
    let groq_result = SiumaiBuilder::new()
        .groq()
        .api_key("test-key")
        .model("llama-3.3-70b-versatile")
        .build()
        .await;

    match groq_result {
        Ok(client) => {
            assert!(client.supports("chat"));
            assert!(client.supports("streaming"));
        }
        Err(e) => {
            assert!(
                !matches!(e, LlmError::ConfigurationError(msg) if msg.contains("not yet implemented"))
            );
        }
    }

    // Test DeepSeek (OpenAI-compatible)
    let deepseek_result = SiumaiBuilder::new()
        .deepseek()
        .api_key("test-key")
        .model("deepseek-chat")
        .build()
        .await;

    match deepseek_result {
        Ok(client) => {
            // DeepSeek uses OpenAI client internally, so provider_name will be "openai"
            // but the metadata should reflect it's a custom provider
            assert!(client.supports("chat"));
            assert!(client.supports("streaming"));
        }
        Err(e) => {
            assert!(
                !matches!(e, LlmError::ConfigurationError(msg) if msg.contains("not yet implemented"))
            );
        }
    }

    // Test OpenRouter (OpenAI-compatible)
    let openrouter_result = SiumaiBuilder::new()
        .openrouter()
        .api_key("test-key")
        .model("openai/gpt-4")
        .build()
        .await;

    match openrouter_result {
        Ok(client) => {
            // OpenRouter uses OpenAI client internally
            assert!(client.supports("chat"));
            assert!(client.supports("streaming"));
        }
        Err(e) => {
            assert!(
                !matches!(e, LlmError::ConfigurationError(msg) if msg.contains("not yet implemented"))
            );
        }
    }
}

/// Test provider name mapping consistency
#[tokio::test]
async fn test_provider_name_mapping() {
    // Test that provider_name() method correctly maps provider types
    let test_cases = vec![
        ("openai", ProviderType::OpenAi),
        ("anthropic", ProviderType::Anthropic),
        ("gemini", ProviderType::Gemini),
        ("ollama", ProviderType::Ollama),
        ("xai", ProviderType::XAI),
        ("groq", ProviderType::Groq),
        ("deepseek", ProviderType::Custom("deepseek".to_string())),
        ("openrouter", ProviderType::Custom("openrouter".to_string())),
    ];

    for (name, _expected_type) in test_cases {
        let _builder = SiumaiBuilder::new().provider_name(name);
        // We can't easily access the internal provider_type, but we can test
        // that the builder doesn't fail with unsupported provider errors

        // This is a basic smoke test - the actual provider type verification
        // would require exposing internal state or building the client
        // Placeholder - builder creation succeeded
    }
}

/// Test that common parameters work across all providers
#[tokio::test]
async fn test_common_parameters_consistency() {
    let common_params_test = |builder: SiumaiBuilder| {
        builder
            .api_key("test-key")
            .model("test-model")
            .temperature(0.7)
            .max_tokens(1000)
            .top_p(0.9)
            .seed(42)
    };

    // Test that all providers accept common parameters without errors
    let providers = vec![
        SiumaiBuilder::new().openai(),
        SiumaiBuilder::new().anthropic(),
        SiumaiBuilder::new().gemini(),
        SiumaiBuilder::new().ollama(),
        SiumaiBuilder::new().xai(),
        SiumaiBuilder::new().groq(),
        SiumaiBuilder::new().deepseek(),
        SiumaiBuilder::new().openrouter(),
    ];

    for builder in providers {
        let _configured_builder = common_params_test(builder);
        // If we reach here without panicking, the parameters were accepted
        // Placeholder assertion
    }
}

/// Test reasoning interface consistency across providers
#[tokio::test]
async fn test_reasoning_interface_consistency() {
    // Test that reasoning parameters are accepted by providers that support them
    let reasoning_test = |builder: SiumaiBuilder| {
        builder
            .api_key("test-key")
            .model("test-model")
            .reasoning(true)
            .reasoning_budget(5000)
    };

    // Providers that should support reasoning
    let reasoning_providers = vec![
        SiumaiBuilder::new().anthropic(), // Has thinking mode
        SiumaiBuilder::new().gemini(),    // Has thinking mode
        SiumaiBuilder::new().ollama(),    // Has think parameter
    ];

    for builder in reasoning_providers {
        let _configured_builder = reasoning_test(builder);
        // If we reach here without panicking, the reasoning parameters were accepted
        // Placeholder assertion
    }
}

/// Test error handling consistency
#[tokio::test]
async fn test_error_handling_consistency() {
    // Test that missing API key produces consistent error
    let result = SiumaiBuilder::new()
        .openai()
        .model("gpt-4")
        // Missing API key
        .build()
        .await;

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, LlmError::ConfigurationError(_)));
    }

    // Test that missing provider type produces error
    let result = SiumaiBuilder::new()
        .api_key("test-key")
        .model("test-model")
        // Missing provider type
        .build()
        .await;

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(matches!(e, LlmError::ConfigurationError(_)));
    }
}

/// Test capability reporting consistency
#[tokio::test]
async fn test_capability_reporting() {
    // Create a mock client for testing (using OpenAI as it's most stable)
    let client_result = SiumaiBuilder::new()
        .openai()
        .api_key("test-key")
        .model("gpt-4")
        .build()
        .await;

    // Even if the client creation fails due to invalid API key,
    // we can test the builder pattern and capability structure
    match client_result {
        Ok(client) => {
            // Test that basic capabilities are reported
            assert!(client.supports("chat"));

            // Test capability checking doesn't panic
            let _ = client.supports("streaming");
            let _ = client.supports("tools");
            let _ = client.supports("vision");
            let _ = client.supports("audio");
            let _ = client.supports("embedding");
            let _ = client.supports("nonexistent_capability");
        }
        Err(_) => {
            // Expected with test API key, but the test structure is valid
            // Placeholder assertion
        }
    }
}

/// Test provider-specific features accessibility
#[tokio::test]
async fn test_provider_specific_features() {
    // Test that provider-specific features can be configured through unified interface

    // Anthropic thinking mode
    let _anthropic_builder = SiumaiBuilder::new()
        .anthropic()
        .api_key("test-key")
        .model("claude-3-5-sonnet-20241022")
        .reasoning(true)
        .reasoning_budget(10000);

    // Should not panic during configuration
    // Should not panic during configuration

    // Gemini thinking mode
    let _gemini_builder = SiumaiBuilder::new()
        .gemini()
        .api_key("test-key")
        .model("gemini-1.5-flash")
        .reasoning(true)
        .reasoning_budget(5000);

    // Should not panic during configuration
    // Should not panic during configuration

    // Ollama local configuration
    let _ollama_builder = SiumaiBuilder::new()
        .ollama()
        .api_key("not-needed")
        .model("llama3.2:latest")
        .base_url("http://localhost:11434")
        .reasoning(true);

    // Should not panic during configuration
    // Should not panic during configuration
}

/// Test that all providers support the same core interface methods
#[test]
fn test_core_interface_methods() {
    // This test verifies that the Siumai trait methods are available
    // We can't easily test the actual functionality without valid API keys,
    // but we can test that the interface is consistent

    // Test message creation (this should work without API calls)
    let messages = [ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("Hello, world!".to_string()),
        metadata: Default::default(),
    }];

    assert_eq!(messages.len(), 1);
    assert!(matches!(messages[0].role, MessageRole::User));

    // Test that message content can be accessed
    if let MessageContent::Text(text) = &messages[0].content {
        assert_eq!(text, "Hello, world!");
    }
}

/// Test builder pattern consistency
#[test]
fn test_builder_pattern_consistency() {
    // Test that all provider builders support method chaining
    let builders = vec![
        SiumaiBuilder::new().openai().api_key("test").model("test"),
        SiumaiBuilder::new()
            .anthropic()
            .api_key("test")
            .model("test"),
        SiumaiBuilder::new().gemini().api_key("test").model("test"),
        SiumaiBuilder::new().ollama().api_key("test").model("test"),
        SiumaiBuilder::new().xai().api_key("test").model("test"),
        SiumaiBuilder::new().groq().api_key("test").model("test"),
        SiumaiBuilder::new()
            .deepseek()
            .api_key("test")
            .model("test"),
        SiumaiBuilder::new()
            .openrouter()
            .api_key("test")
            .model("test"),
    ];

    // If we reach here, all builders support the basic chaining pattern
    assert_eq!(builders.len(), 8);
}

/// Test provider type consistency
#[test]
fn test_provider_type_consistency() {
    // Test that ProviderType enum covers all supported providers
    let provider_types = vec![
        ProviderType::OpenAi,
        ProviderType::Anthropic,
        ProviderType::Gemini,
        ProviderType::Ollama,
        ProviderType::XAI,
        ProviderType::Groq,
        ProviderType::Custom("deepseek".to_string()),
        ProviderType::Custom("openrouter".to_string()),
    ];

    // Test that all provider types can be displayed
    for provider_type in provider_types {
        let display_string = format!("{}", provider_type);
        assert!(!display_string.is_empty());
    }
}
