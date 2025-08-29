//! Integration Tests for Siumai LLM Library
//!
//! These tests verify the core functionality of the unified LLM interface

use siumai::prelude::*;
use siumai::stream::{ProcessedEvent, StreamProcessor};
use siumai::types::{ChatRequest, ContentPart, ProviderParams};
use siumai::user_builder;
use std::time::Duration;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation_with_macros() {
        // Test basic message creation using convenience macros - now returns ChatMessage directly
        let user_msg = user!("Hello, world!");
        assert_eq!(user_msg.role, MessageRole::User);

        let system_msg = system!("You are a helpful assistant");
        assert_eq!(system_msg.role, MessageRole::System);

        let assistant_msg = assistant!("I'm here to help");
        assert_eq!(assistant_msg.role, MessageRole::Assistant);

        // Test message content extraction
        match user_msg.content {
            MessageContent::Text(text) => assert_eq!(text, "Hello, world!"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_multimodal_message_creation() {
        // Test creating multimodal messages with images - use user_builder for complex messages
        let multimodal_msg = user_builder!("Analyze this image")
            .with_image(
                "https://example.com/image.jpg".to_string(),
                Some("high".to_string()),
            )
            .build();

        match multimodal_msg.content {
            MessageContent::MultiModal(parts) => {
                assert_eq!(parts.len(), 2);

                // Check text part
                if let ContentPart::Text { text } = &parts[0] {
                    assert_eq!(text, "Analyze this image");
                } else {
                    panic!("Expected text part");
                }

                // Check image part
                if let ContentPart::Image { image_url, detail } = &parts[1] {
                    assert_eq!(image_url, "https://example.com/image.jpg");
                    assert_eq!(detail.as_ref().unwrap(), "high");
                } else {
                    panic!("Expected image part");
                }
            }
            _ => panic!("Expected multimodal content"),
        }
    }

    #[test]
    fn test_chat_request_builder() {
        // Test building a complete chat request - simple messages use macros directly
        let messages = vec![
            system!("You are a helpful assistant"),
            user!("What is the capital of France?"),
        ];

        let request = ChatRequest::builder()
            .messages(messages)
            .common_params(CommonParams {
                model: "gpt-4".to_string(),
                temperature: Some(0.7),
                max_tokens: Some(1000),
                top_p: Some(0.9),
                stop_sequences: None,
                seed: Some(42),
            })
            .build();

        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.common_params.model, "gpt-4");
        assert_eq!(request.common_params.temperature, Some(0.7));
        assert_eq!(request.common_params.max_tokens, Some(1000));
    }

    #[test]
    fn test_provider_params() {
        // Test provider-specific parameter handling
        let provider_params = ProviderParams::new()
            .with_param("temperature", 0.8)
            .with_param("max_tokens", 2000)
            .with_param("custom_setting", "value");

        // Test parameter retrieval
        let temp: Option<f64> = provider_params.get("temperature");
        assert_eq!(temp, Some(0.8));

        let tokens: Option<u32> = provider_params.get("max_tokens");
        assert_eq!(tokens, Some(2000));

        let custom: Option<String> = provider_params.get("custom_setting");
        assert_eq!(custom, Some("value".to_string()));

        // Test non-existent parameter
        let missing: Option<String> = provider_params.get("non_existent");
        assert_eq!(missing, None);
    }

    #[test]
    fn test_provider_capabilities() {
        // Test provider capability configuration
        let capabilities = ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_custom_feature("structured_output", true);

        assert!(capabilities.supports("chat"));
        assert!(capabilities.supports("streaming"));
        assert!(capabilities.supports("tools"));
        assert!(capabilities.supports("vision"));
        assert!(capabilities.supports("structured_output"));
        assert!(!capabilities.supports("audio"));
        assert!(!capabilities.supports("non_existent"));
    }

    #[test]
    fn test_error_handling() {
        // Test error creation and properties
        let api_error = LlmError::api_error(404, "Not found");
        assert_eq!(api_error.status_code(), Some(404));
        assert!(!api_error.is_retryable());
        assert!(!api_error.is_auth_error());

        let auth_error = LlmError::AuthenticationError("Invalid API key".to_string());
        assert!(auth_error.is_auth_error());
        assert!(!auth_error.is_retryable());

        let rate_limit_error = LlmError::RateLimitError("Too many requests".to_string());
        assert!(rate_limit_error.is_retryable());
        assert!(rate_limit_error.is_rate_limit_error());

        let server_error = LlmError::api_error(500, "Internal server error");
        assert!(server_error.is_retryable());
        assert_eq!(server_error.status_code(), Some(500));
    }

    #[test]
    fn test_http_config() {
        // Test HTTP configuration
        let mut http_config = HttpConfig {
            timeout: Some(Duration::from_secs(60)),
            ..Default::default()
        };
        http_config
            .headers
            .insert("Custom-Header".to_string(), "value".to_string());

        assert_eq!(http_config.timeout, Some(Duration::from_secs(60)));
        assert_eq!(
            http_config.headers.get("Custom-Header"),
            Some(&"value".to_string())
        );
        assert!(http_config.user_agent.is_some());
    }

    #[test]
    fn test_usage_statistics() {
        // Test usage statistics merging
        let mut usage1 = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            reasoning_tokens: None,
            cached_tokens: None,
        };

        let usage2 = Usage {
            prompt_tokens: 200,
            completion_tokens: 75,
            total_tokens: 275,
            reasoning_tokens: Some(25),
            cached_tokens: Some(10),
        };

        usage1.merge(&usage2);

        assert_eq!(usage1.prompt_tokens, 300);
        assert_eq!(usage1.completion_tokens, 125);
        assert_eq!(usage1.total_tokens, 425);
        assert_eq!(usage1.reasoning_tokens, Some(25));
    }

    #[test]
    fn test_stream_processor() {
        // Test stream event processing
        let mut processor = StreamProcessor::new();

        // Process content delta
        let content_event = ChatStreamEvent::ContentDelta {
            delta: "Hello".to_string(),
            index: None,
        };

        let processed = processor.process_event(content_event);
        match processed {
            ProcessedEvent::ContentUpdate {
                delta, accumulated, ..
            } => {
                assert_eq!(delta, "Hello");
                assert_eq!(accumulated, "Hello");
            }
            _ => panic!("Expected ContentUpdate"),
        }

        // Process another content delta
        let content_event2 = ChatStreamEvent::ContentDelta {
            delta: " World".to_string(),
            index: None,
        };

        let processed2 = processor.process_event(content_event2);
        match processed2 {
            ProcessedEvent::ContentUpdate {
                delta, accumulated, ..
            } => {
                assert_eq!(delta, " World");
                assert_eq!(accumulated, "Hello World");
            }
            _ => panic!("Expected ContentUpdate"),
        }
    }

    // Commented out until provider info functions are implemented
    // #[test]
    // fn test_provider_info() {
    //     use siumai::providers::{get_provider_info, get_supported_providers, is_model_supported};
    //     // Test implementation when available
    // }

    // Commented out until parameter mapping is implemented
    // #[test]
    // fn test_parameter_mapping() {
    //     // Test implementation when available
    // }

    #[test]
    fn test_enhanced_parameter_validation() {
        use siumai::params::EnhancedParameterValidator;

        let params = CommonParams {
            model: "gpt-4".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(1000),
            top_p: Some(0.9),
            stop_sequences: None,
            seed: Some(42),
        };

        let result =
            EnhancedParameterValidator::validate_for_provider(&params, &ProviderType::OpenAi);

        assert!(result.is_ok());
        let report = result.unwrap();
        assert!(!report.has_errors());
        assert!(!report.valid_params.is_empty());
    }

    #[test]
    fn test_retry_policy() {
        use siumai::retry::RetryPolicy;
        use std::time::Duration;

        let policy = RetryPolicy::new()
            .with_max_attempts(5)
            .with_initial_delay(Duration::from_millis(100))
            .with_backoff_multiplier(2.0)
            .with_jitter(false);

        assert_eq!(policy.max_attempts, 5);
        assert_eq!(policy.initial_delay, Duration::from_millis(100));
        assert_eq!(policy.backoff_multiplier, 2.0);

        // Test delay calculation
        assert_eq!(policy.calculate_delay(0), Duration::from_millis(100));
        assert_eq!(policy.calculate_delay(1), Duration::from_millis(200));
        assert_eq!(policy.calculate_delay(2), Duration::from_millis(400));
    }

    #[test]
    fn test_error_classification() {
        use siumai::error::ErrorCategory;

        let error = LlmError::ApiError {
            code: 429,
            message: "Rate limit exceeded".to_string(),
            details: None,
        };

        // Test the category method that exists on LlmError
        assert_eq!(error.category(), ErrorCategory::RateLimit);

        // Test retryability
        assert!(error.is_retryable());

        // Test max retry attempts
        assert_eq!(error.max_retry_attempts(), 3);
    }
}

#[cfg(test)]
mod builder_tests {
    use super::*;

    #[test]
    fn test_llm_builder_creation() {
        // Test basic builder creation
        let builder = LlmBuilder::new();

        // Test OpenAI builder
        let _openai_builder = builder
            .openai()
            .model("gpt-4")
            .temperature(0.7)
            .max_tokens(1000);

        // We can't actually build without API key, but we can test the builder pattern
        // Placeholder assertion
    }

    #[test]
    fn test_builder_with_http_client() {
        // Test builder with custom HTTP client
        let custom_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();

        let builder = LlmBuilder::new()
            .with_http_client(custom_client)
            .with_timeout(Duration::from_secs(30));

        let _openai_builder = builder.openai().model("gpt-4").temperature(0.8);

        // Placeholder assertion
    }

    /// Test OpenAI-compatible HTTP configuration
    #[tokio::test]
    async fn test_openai_compatible_http_config() {
        println!("🧪 Testing OpenAI-compatible HTTP configuration");

        // Create HTTP config with various settings
        let http_config = HttpConfig {
            timeout: Some(Duration::from_secs(60)),
            connect_timeout: Some(Duration::from_secs(10)),
            user_agent: Some("test-agent/1.0".to_string()),
            ..Default::default()
        };

        // Add custom headers
        let mut http_config = http_config;
        let mut headers = std::collections::HashMap::new();
        headers.insert("X-Custom-Header".to_string(), "test-value".to_string());
        headers.insert("X-Request-ID".to_string(), "test-123".to_string());
        http_config.headers = headers;

        // Test SiliconFlow with HTTP config
        let result = LlmBuilder::new()
            .siliconflow()
            .api_key("test-key")
            .model("deepseek-chat")
            .with_http_config(http_config.clone())
            .build()
            .await;

        match result {
            Ok(client) => {
                println!("    ✅ SiliconFlow client created with HTTP config");
                assert_eq!(client.provider_id(), "siliconflow");
                assert_eq!(client.model(), "deepseek-chat");
            }
            Err(e) => {
                // Expected to fail with invalid API key, but should not fail due to HTTP config
                println!(
                    "    ✅ Failed with expected error (not HTTP config related): {}",
                    e
                );
                assert!(!e.to_string().contains("HTTP"));
                assert!(!e.to_string().contains("timeout"));
            }
        }

        // Test DeepSeek with HTTP config
        let result = LlmBuilder::new()
            .deepseek()
            .api_key("test-key")
            .with_http_config(http_config)
            .build()
            .await;

        match result {
            Ok(client) => {
                println!("    ✅ DeepSeek client created with HTTP config");
                assert_eq!(client.provider_id(), "deepseek");
            }
            Err(e) => {
                println!(
                    "    ✅ Failed with expected error (not HTTP config related): {}",
                    e
                );
                assert!(!e.to_string().contains("HTTP"));
            }
        }
    }

    /// Test default model configuration for OpenAI-compatible providers
    #[tokio::test]
    async fn test_openai_compatible_default_models() {
        println!("🧪 Testing default model configuration for OpenAI-compatible providers");

        // Test SiliconFlow default model
        let result = LlmBuilder::new()
            .siliconflow()
            .api_key("test-key")
            .build()
            .await;

        match result {
            Ok(client) => {
                println!("    ✅ SiliconFlow client created with default model");
                assert_eq!(client.model(), "deepseek-chat");
            }
            Err(e) => {
                println!("    ✅ Failed with expected error: {}", e);
                // Should not fail due to model configuration
                assert!(!e.to_string().contains("model"));
            }
        }

        // Test OpenRouter default model
        let result = LlmBuilder::new()
            .openrouter()
            .api_key("test-key")
            .build()
            .await;

        match result {
            Ok(client) => {
                println!("    ✅ OpenRouter client created with default model");
                assert_eq!(client.model(), "openai/gpt-4o");
            }
            Err(e) => {
                println!("    ✅ Failed with expected error: {}", e);
                assert!(!e.to_string().contains("model"));
            }
        }
    }
}
