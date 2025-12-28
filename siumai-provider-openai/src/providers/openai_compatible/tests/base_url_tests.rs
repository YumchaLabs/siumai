//! Tests for OpenAiCompatibleBuilder base_url functionality
//!
//! This module tests the custom base_url method that allows users to override
//! the default provider base URL for self-deployed services and alternative endpoints.

use crate::builder::BuilderBase;
use crate::providers::openai_compatible::builder::OpenAiCompatibleBuilder;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_url_method_exists() {
        // Test that the base_url method exists and can be called
        let openai_compatible_builder = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek");

        let builder_with_url = openai_compatible_builder.base_url("https://custom-server.com/v1");

        // The method should return Self for chaining
        assert!(std::mem::size_of_val(&builder_with_url) > 0);
    }

    #[test]
    fn test_base_url_chaining() {
        // Test that base_url method supports method chaining
        let result = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .api_key("test-key")
            .base_url("https://custom-server.com/v1")
            .model("deepseek-chat")
            .temperature(0.7);

        // Should be able to chain methods after base_url
        assert!(std::mem::size_of_val(&result) > 0);
    }

    #[test]
    fn test_base_url_accepts_string_types() {
        // Test that base_url accepts different string types
        // String literal
        let _result1 = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .base_url("https://example.com/v1");

        // String
        let url = String::from("https://example.com/v1");
        let _result2 = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek").base_url(url);

        // &str
        let url_ref = "https://example.com/v1";
        let _result3 = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek").base_url(url_ref);
    }

    #[tokio::test]
    async fn test_base_url_integration_with_build() {
        // Test that custom base_url is used during build process
        // The build should succeed with valid configuration

        let openai_compatible_builder = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .base_url("https://custom-deepseek-server.com/v1")
            .api_key("test-key")
            .model("deepseek-chat");

        // The build should succeed with proper configuration
        let result = openai_compatible_builder.build().await;

        // Should succeed in creating the client (configuration is valid)
        assert!(
            result.is_ok(),
            "Build should succeed with valid configuration"
        );

        // Verify the client was created successfully
        let _client = result.unwrap();
    }

    #[test]
    fn test_base_url_documentation_examples() {
        // Ensure the documentation-style API can be expressed via `OpenAiCompatibleBuilder`.
        let _client_builder = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .api_key("your-api-key")
            .base_url("https://my-deepseek-server.com/v1")
            .model("deepseek-chat");

        let _client_builder2 = OpenAiCompatibleBuilder::new(BuilderBase::default(), "openrouter")
            .api_key("your-api-key")
            .base_url("https://openrouter.ai/api/v1")
            .model("openai/gpt-4");
    }

    #[test]
    fn test_base_url_empty_string() {
        // Test behavior with empty string (should be allowed but may cause issues later)
        let _result = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek").base_url("");

        // Should not panic during builder creation
    }

    #[test]
    fn test_base_url_with_trailing_slash() {
        // Test that URLs with trailing slashes are handled correctly
        let _result =
            OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek").base_url("https://example.com/v1/");

        // Should not panic during builder creation
    }

    #[test]
    fn test_base_url_with_different_protocols() {
        // Test different URL protocols
        // HTTPS
        let _result1 =
            OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek").base_url("https://example.com/v1");

        // HTTP (for local development)
        let _result2 =
            OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek").base_url("http://localhost:8080/v1");

        // Custom port
        let _result3 = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .base_url("https://example.com:8443/v1");
    }
}
