//! Tests for Debug trait implementations
//!
//! This module tests that all client types properly implement Debug trait
//! with appropriate security measures for sensitive information.

use siumai::experimental::client::ClientWrapper;
use siumai::provider::SiumaiBuilder;

#[cfg(test)]
mod debug_tests {
    use super::*;

    #[test]
    fn test_siumai_builder_debug() {
        let builder = SiumaiBuilder::new()
            .api_key("sk-test-key-1234567890abcdef")
            .model("gpt-4o-mini")
            .temperature(0.7)
            .max_tokens(1000);

        let debug_output = format!("{:?}", builder);

        // Should contain useful information
        assert!(debug_output.contains("SiumaiBuilder"));
        assert!(debug_output.contains("model"));
        assert!(debug_output.contains("gpt-4o-mini"));
        assert!(debug_output.contains("temperature"));
        assert!(debug_output.contains("0.7"));

        // Should show existence of API key without exposing it
        assert!(debug_output.contains("has_api_key"));
        assert!(debug_output.contains("true"));
        assert!(!debug_output.contains("sk-test-key-1234567890abcdef"));
        assert!(!debug_output.contains("[MASKED]")); // No longer using MASKED
    }

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn test_openai_client_debug() {
        use siumai::provider_ext::openai::{OpenAiClient, OpenAiConfig};

        let config = OpenAiConfig::new("sk-test-key-1234567890abcdef".to_string())
            .with_model("gpt-4o-mini")
            .with_temperature(0.8);

        let client = OpenAiClient::new(config, reqwest::Client::new());
        let debug_output = format!("{:?}", client);

        // Should contain useful information
        assert!(debug_output.contains("OpenAiClient"));
        assert!(debug_output.contains("provider_id"));
        assert!(debug_output.contains("openai"));
        assert!(debug_output.contains("model"));
        assert!(debug_output.contains("gpt-4o-mini"));

        // Should not expose sensitive information
        assert!(!debug_output.contains("sk-test-key-1234567890abcdef"));
        assert!(!debug_output.contains("[MASKED]")); // No longer using MASKED
    }

    #[cfg(feature = "anthropic")]
    #[test]
    fn test_anthropic_client_debug() {
        use siumai::prelude::unified::{CommonParams, HttpConfig};
        use siumai::provider_ext::anthropic::AnthropicClient;
        use siumai::provider_ext::anthropic::AnthropicParams;

        let common_params = CommonParams {
            model: "claude-3-5-sonnet-20241022".to_string(),
            temperature: Some(0.5),
            ..Default::default()
        };

        let client = AnthropicClient::new(
            "sk-ant-test-key-1234567890abcdef".to_string(),
            "https://api.anthropic.com".to_string(),
            reqwest::Client::new(),
            common_params,
            AnthropicParams::default(),
            HttpConfig::default(),
        );

        let debug_output = format!("{:?}", client);

        // Should contain useful information
        assert!(debug_output.contains("AnthropicClient"));
        assert!(debug_output.contains("provider_id"));
        assert!(debug_output.contains("anthropic"));
        assert!(debug_output.contains("model"));
        assert!(debug_output.contains("claude-3-5-sonnet-20241022"));

        // Should not expose sensitive information
        assert!(!debug_output.contains("sk-ant-test-key-1234567890abcdef"));
        assert!(!debug_output.contains("[MASKED]")); // No longer using MASKED
    }

    #[cfg(feature = "groq")]
    #[tokio::test]
    async fn test_groq_client_debug() {
        use siumai::Provider;

        let client = Provider::groq()
            .api_key("gsk-test-key-1234567890abcdef")
            .model("llama-3.3-70b-versatile")
            .temperature(0.6)
            .build()
            .await
            .expect("build groq client");
        let debug_output = format!("{:?}", client);

        // Should contain useful information
        assert!(debug_output.contains("provider_id"));
        assert!(debug_output.contains("groq"));
        assert!(debug_output.contains("model"));
        assert!(debug_output.contains("llama-3.3-70b-versatile"));

        // Should not expose sensitive information
        assert!(!debug_output.contains("gsk-test-key-1234567890abcdef"));
        assert!(!debug_output.contains("[MASKED]")); // No longer using MASKED
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn test_ollama_client_debug() {
        use siumai::prelude::unified::CommonParams;
        use siumai::provider_ext::ollama::{OllamaClient, OllamaConfig};

        let common_params = CommonParams {
            model: "llama3.2:3b".to_string(),
            ..Default::default()
        };

        let mut config = OllamaConfig::new();
        config.common_params = common_params;
        config.base_url = "http://localhost:11434".to_string();

        let client = OllamaClient::new(config, reqwest::Client::new());
        let debug_output = format!("{:?}", client);

        // Should contain useful information
        assert!(debug_output.contains("OllamaClient"));
        assert!(debug_output.contains("provider_id"));
        assert!(debug_output.contains("ollama"));
        assert!(debug_output.contains("model"));
        assert!(debug_output.contains("llama3.2:3b"));
        assert!(debug_output.contains("base_url"));
        assert!(debug_output.contains("localhost:11434"));
    }

    #[cfg(feature = "openai")]
    #[test]
    fn test_client_wrapper_debug() {
        use siumai::provider_ext::openai::{OpenAiClient, OpenAiConfig};

        let config = OpenAiConfig::new("test-key".to_string()).with_model("gpt-4o-mini");

        let client = OpenAiClient::new(config, reqwest::Client::new());
        let wrapper = ClientWrapper::openai(Box::new(client));
        let debug_output = format!("{:?}", wrapper);

        // Should show wrapper type without exposing internal client details
        assert!(debug_output.contains("ClientWrapper::Client"));
        assert!(debug_output.contains("[LlmClient]"));
    }

    #[test]
    fn test_debug_consistency() {
        // Test that all debug implementations follow consistent patterns
        let builder = SiumaiBuilder::new().api_key("test-key").model("test-model");

        let debug_output = format!("{:?}", builder);

        // All debug implementations should:
        // 1. Show the struct/enum name
        assert!(debug_output.contains("SiumaiBuilder"));

        // 2. Show existence of sensitive fields without exposing values
        assert!(debug_output.contains("has_api_key"));

        // 3. Show useful configuration information
        assert!(debug_output.contains("model"));
        assert!(debug_output.contains("test-model"));
    }

    #[test]
    fn test_no_sensitive_data_leakage() {
        // Test various API key formats to ensure they're not exposed
        let test_keys = vec![
            "sk-1234567890abcdef",
            "sk-ant-1234567890abcdef",
            "gsk-1234567890abcdef",
            "Bearer sk-1234567890abcdef",
            "very-long-api-key-that-should-be-masked",
        ];

        for key in test_keys {
            let builder = SiumaiBuilder::new().api_key(key);
            let debug_output = format!("{:?}", builder);

            // Should not contain the actual key
            assert!(
                !debug_output.contains(key),
                "Debug output should not contain API key: {}",
                key
            );

            // Should show existence of API key
            assert!(
                debug_output.contains("has_api_key"),
                "Debug output should show has_api_key for key: {}",
                key
            );
        }
    }
}
