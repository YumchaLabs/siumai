//! Parameter Mapping Consistency Tests
//!
//! Tests to verify that parameter mapping is consistent across providers
//! and that the new provider-local parameter mapping works correctly.

use siumai::params::{AnthropicParams, GeminiParams, OpenAiParams};
use siumai::providers::anthropic::request::{AnthropicParameterMapper, AnthropicRequestBuilder};
use siumai::providers::gemini::request::{GeminiParameterMapper, GeminiRequestBuilder};
use siumai::providers::openai::request::{OpenAiParameterMapper, OpenAiRequestBuilder};
use siumai::request_factory::{RequestBuilder, RequestBuilderConfig};
use siumai::types::{CommonParams, MessageContent, MessageRole};

/// Test that OpenAI parameter mapping works correctly
#[test]
fn test_openai_parameter_mapping() {
    let common_params = CommonParams {
        model: "gpt-4".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        top_p: Some(0.9),
        stop_sequences: Some(vec!["STOP".to_string(), "END".to_string()]),
        seed: Some(42),
    };

    let openai_params = OpenAiParams::default();
    let builder = OpenAiRequestBuilder::new(common_params.clone(), openai_params);

    // Test parameter mapping
    let mapped = builder.map_common_to_openai(&common_params);

    // Verify OpenAI-specific mappings
    assert_eq!(mapped["model"], "gpt-4");
    assert!((mapped["temperature"].as_f64().unwrap() - 0.7).abs() < 0.001);
    assert_eq!(mapped["max_tokens"], 1000);
    assert!((mapped["top_p"].as_f64().unwrap() - 0.9).abs() < 0.001);
    assert_eq!(mapped["seed"], 42);

    // OpenAI uses "stop" instead of "stop_sequences"
    assert_eq!(mapped["stop"], serde_json::json!(["STOP", "END"]));
    assert!(mapped.get("stop_sequences").is_none());
}

/// Test that Anthropic parameter mapping works correctly
#[test]
fn test_anthropic_parameter_mapping() {
    let common_params = CommonParams {
        model: "claude-3-5-sonnet-20241022".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        top_p: Some(0.9),
        stop_sequences: Some(vec!["STOP".to_string(), "END".to_string()]),
        seed: Some(42), // Should be ignored by Anthropic
    };

    let anthropic_params = AnthropicParams::default();
    let builder = AnthropicRequestBuilder::new(common_params.clone(), anthropic_params);

    // Test parameter mapping
    let mapped = builder.map_common_to_anthropic(&common_params);

    // Verify Anthropic-specific mappings
    assert_eq!(mapped["model"], "claude-3-5-sonnet-20241022");
    assert!((mapped["temperature"].as_f64().unwrap() - 0.7).abs() < 0.001);
    assert_eq!(mapped["max_tokens"], 1000);
    assert!((mapped["top_p"].as_f64().unwrap() - 0.9).abs() < 0.001);

    // Anthropic uses "stop_sequences" (same as common)
    assert_eq!(mapped["stop_sequences"], serde_json::json!(["STOP", "END"]));

    // Anthropic doesn't support seed - should not be present
    assert!(mapped.get("seed").is_none());
}

/// Test that Gemini parameter mapping works correctly
#[test]
fn test_gemini_parameter_mapping() {
    let common_params = CommonParams {
        model: "gemini-1.5-pro".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1000),
        top_p: Some(0.9),
        stop_sequences: Some(vec!["STOP".to_string(), "END".to_string()]),
        seed: Some(42), // Should be ignored by Gemini
    };

    let gemini_params = GeminiParams::default();
    let builder = GeminiRequestBuilder::new(common_params.clone(), gemini_params);

    // Test parameter mapping
    let mapped = builder.map_common_to_gemini(&common_params);

    // Verify Gemini-specific parameter name mappings
    assert_eq!(mapped["model"], "gemini-1.5-pro");
    assert!((mapped["temperature"].as_f64().unwrap() - 0.7).abs() < 0.001);
    assert_eq!(mapped["maxOutputTokens"], 1000); // Different name from common
    assert!((mapped["topP"].as_f64().unwrap() - 0.9).abs() < 0.001); // Different name
    assert_eq!(mapped["stopSequences"], serde_json::json!(["STOP", "END"])); // Different name

    // Gemini doesn't support seed - should not be present
    assert!(mapped.get("seed").is_none());
}

/// Test that Anthropic sets default max_tokens when not provided
#[test]
fn test_anthropic_default_max_tokens() {
    let common_params = CommonParams {
        model: "claude-3-5-sonnet-20241022".to_string(),
        temperature: Some(0.7),
        max_tokens: None, // Not provided
        ..Default::default()
    };

    let anthropic_params = AnthropicParams::default();
    let builder = AnthropicRequestBuilder::new(common_params.clone(), anthropic_params);

    let mapped = builder.map_common_to_anthropic(&common_params);

    // Should have default max_tokens for Anthropic
    assert_eq!(mapped["max_tokens"], 4096);
}

/// Test parameter validation configuration for OpenAI based on official API spec
#[test]
fn test_openai_parameter_validation() {
    let messages = vec![siumai::types::ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("Hello".to_string()),
        metadata: Default::default(),
        tool_calls: None,
        tool_call_id: None,
    }];

    let validation_config = RequestBuilderConfig {
        strict_validation: false,
        provider_validation: true,
    };

    // Test invalid temperature (> 2.0)
    let invalid_temp_params = CommonParams {
        model: "gpt-4".to_string(),
        temperature: Some(2.5), // Invalid: > 2.0
        ..Default::default()
    };
    let builder = OpenAiRequestBuilder::new(invalid_temp_params, OpenAiParams::default());
    let result =
        builder.build_chat_request_with_config(messages.clone(), None, false, &validation_config);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("temperature must be between 0.0 and 2.0")
    );

    // Test invalid top_p (> 1.0)
    let invalid_top_p_params = CommonParams {
        model: "gpt-4".to_string(),
        top_p: Some(1.5), // Invalid: > 1.0
        ..Default::default()
    };
    let builder = OpenAiRequestBuilder::new(invalid_top_p_params, OpenAiParams::default());
    let result =
        builder.build_chat_request_with_config(messages.clone(), None, false, &validation_config);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("top_p must be between 0.0 and 1.0")
    );

    // Test invalid frequency_penalty (> 2.0)
    let invalid_freq_penalty_params = CommonParams {
        model: "gpt-4".to_string(),
        ..Default::default()
    };
    let invalid_openai_params = OpenAiParams {
        frequency_penalty: Some(3.0), // Invalid: > 2.0
        ..Default::default()
    };
    let builder = OpenAiRequestBuilder::new(invalid_freq_penalty_params, invalid_openai_params);
    let result =
        builder.build_chat_request_with_config(messages.clone(), None, false, &validation_config);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("frequency_penalty must be between -2.0 and 2.0")
    );

    // Test invalid presence_penalty (< -2.0)
    let invalid_pres_penalty_params = CommonParams {
        model: "gpt-4".to_string(),
        ..Default::default()
    };
    let invalid_openai_params = OpenAiParams {
        presence_penalty: Some(-3.0), // Invalid: < -2.0
        ..Default::default()
    };
    let builder = OpenAiRequestBuilder::new(invalid_pres_penalty_params, invalid_openai_params);
    let result =
        builder.build_chat_request_with_config(messages.clone(), None, false, &validation_config);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("presence_penalty must be between -2.0 and 2.0")
    );

    // Test with validation disabled (default) - should succeed even with invalid values
    let default_config = RequestBuilderConfig::default();
    let invalid_params = CommonParams {
        model: "gpt-4".to_string(),
        temperature: Some(5.0), // Invalid but should pass when validation disabled
        ..Default::default()
    };
    let builder = OpenAiRequestBuilder::new(invalid_params, OpenAiParams::default());
    let result =
        builder.build_chat_request_with_config(messages.clone(), None, false, &default_config);
    assert!(result.is_ok()); // Should pass when validation disabled

    // Test with valid parameters and validation enabled - should succeed
    let valid_params = CommonParams {
        model: "gpt-4".to_string(),
        temperature: Some(0.7), // Valid: 0.0 <= 0.7 <= 2.0
        top_p: Some(0.9),       // Valid: 0.0 <= 0.9 <= 1.0
        ..Default::default()
    };
    let valid_openai_params = OpenAiParams {
        frequency_penalty: Some(0.5), // Valid: -2.0 <= 0.5 <= 2.0
        presence_penalty: Some(-0.3), // Valid: -2.0 <= -0.3 <= 2.0
        ..Default::default()
    };
    let builder = OpenAiRequestBuilder::new(valid_params, valid_openai_params);
    let result = builder.build_chat_request_with_config(messages, None, false, &validation_config);
    assert!(result.is_ok()); // Should pass with valid parameters
}

/// Test parameter validation configuration for Anthropic based on official API spec
#[test]
fn test_anthropic_parameter_validation() {
    let messages = vec![siumai::types::ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("Hello".to_string()),
        metadata: Default::default(),
        tool_calls: None,
        tool_call_id: None,
    }];

    let validation_config = RequestBuilderConfig {
        strict_validation: false,
        provider_validation: true,
    };

    // Test invalid temperature (> 1.0 for Anthropic)
    let invalid_temp_params = CommonParams {
        model: "claude-3-5-sonnet-20241022".to_string(),
        temperature: Some(1.5), // Invalid: > 1.0 for Anthropic
        ..Default::default()
    };
    let builder = AnthropicRequestBuilder::new(invalid_temp_params, AnthropicParams::default());
    let result =
        builder.build_chat_request_with_config(messages.clone(), None, false, &validation_config);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("temperature must be between 0.0 and 1.0")
    );

    // Test invalid top_p (> 1.0)
    let invalid_top_p_params = CommonParams {
        model: "claude-3-5-sonnet-20241022".to_string(),
        top_p: Some(1.5), // Invalid: > 1.0
        ..Default::default()
    };
    let builder = AnthropicRequestBuilder::new(invalid_top_p_params, AnthropicParams::default());
    let result =
        builder.build_chat_request_with_config(messages.clone(), None, false, &validation_config);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("top_p must be between 0.0 and 1.0")
    );

    // Test with validation disabled (default) - should succeed even with invalid values
    let default_config = RequestBuilderConfig::default();
    let invalid_params = CommonParams {
        model: "claude-3-5-sonnet-20241022".to_string(),
        temperature: Some(2.0), // Invalid but should pass when validation disabled
        ..Default::default()
    };
    let builder = AnthropicRequestBuilder::new(invalid_params, AnthropicParams::default());
    let result =
        builder.build_chat_request_with_config(messages.clone(), None, false, &default_config);
    assert!(result.is_ok()); // Should pass when validation disabled

    // Test with valid parameters and validation enabled - should succeed
    let valid_params = CommonParams {
        model: "claude-3-5-sonnet-20241022".to_string(),
        temperature: Some(0.7), // Valid: 0.0 <= 0.7 <= 1.0
        top_p: Some(0.9),       // Valid: 0.0 <= 0.9 <= 1.0
        max_tokens: Some(1000), // Valid: > 0
        ..Default::default()
    };
    let builder = AnthropicRequestBuilder::new(valid_params, AnthropicParams::default());
    let result = builder.build_chat_request_with_config(messages, None, false, &validation_config);
    assert!(result.is_ok()); // Should pass with valid parameters
}

/// Test that provider-specific parameters are merged correctly
#[test]
fn test_provider_specific_parameter_merging() {
    let common_params = CommonParams {
        model: "gpt-4".to_string(),
        temperature: Some(0.7),
        ..Default::default()
    };

    let openai_params = OpenAiParams {
        frequency_penalty: Some(0.5),
        presence_penalty: Some(0.3),
        ..Default::default()
    };

    let builder = OpenAiRequestBuilder::new(common_params.clone(), openai_params.clone());

    // Map common parameters first
    let base = builder.map_common_to_openai(&common_params);

    // Then merge OpenAI-specific parameters
    let merged = builder.merge_openai_params(base, &openai_params);

    // Should have both common and provider-specific parameters
    assert_eq!(merged["model"], "gpt-4");
    assert!((merged["temperature"].as_f64().unwrap() - 0.7).abs() < 0.001);
    assert!((merged["frequency_penalty"].as_f64().unwrap() - 0.5).abs() < 0.001);
    assert!((merged["presence_penalty"].as_f64().unwrap() - 0.3).abs() < 0.001);
}

/// Test xAI OpenAI compatibility for parameter validation
#[test]
fn test_xai_openai_compatibility() {
    // xAI uses OpenAI-compatible parameters, so it should behave identically to OpenAI
    let common_params = CommonParams {
        model: "grok-3-latest".to_string(),
        temperature: Some(0.8),
        max_tokens: Some(2000),
        top_p: Some(0.95),
        stop_sequences: Some(vec!["HALT".to_string()]),
        seed: Some(123),
    };

    let openai_params = OpenAiParams::default();
    let openai_builder = OpenAiRequestBuilder::new(common_params.clone(), openai_params);
    let openai_mapped = openai_builder.map_common_to_openai(&common_params);

    // xAI should produce identical parameter mapping to OpenAI
    // since it uses OpenAI-compatible format
    assert_eq!(openai_mapped["model"], "grok-3-latest");
    assert!((openai_mapped["temperature"].as_f64().unwrap() - 0.8).abs() < 0.001);
    assert_eq!(openai_mapped["max_tokens"], 2000);
    assert!((openai_mapped["top_p"].as_f64().unwrap() - 0.95).abs() < 0.001);
    assert_eq!(openai_mapped["stop"], serde_json::json!(["HALT"]));
    assert_eq!(openai_mapped["seed"], 123);

    // Test validation compatibility - xAI should use same validation rules as OpenAI
    let validation_config = RequestBuilderConfig {
        strict_validation: false,
        provider_validation: true,
    };

    // Test invalid temperature (> 2.0) - should fail for both OpenAI and xAI
    let invalid_temp_params = CommonParams {
        model: "grok-3-latest".to_string(),
        temperature: Some(2.5), // Invalid: > 2.0
        ..Default::default()
    };

    let messages = vec![siumai::types::ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("Hello".to_string()),
        metadata: Default::default(),
        tool_calls: None,
        tool_call_id: None,
    }];

    let invalid_builder = OpenAiRequestBuilder::new(invalid_temp_params, OpenAiParams::default());
    let result =
        invalid_builder.build_chat_request_with_config(messages, None, false, &validation_config);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("temperature must be between 0.0 and 2.0")
    );
}

/// Test Gemini parameter validation based on official API documentation
#[test]
fn test_gemini_parameter_validation() {
    let messages = vec![siumai::types::ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("Hello".to_string()),
        metadata: Default::default(),
        tool_calls: None,
        tool_call_id: None,
    }];

    let validation_config = RequestBuilderConfig {
        strict_validation: false,
        provider_validation: true,
    };

    // Test invalid temperature (> 2.0 for Gemini)
    let invalid_temp_params = CommonParams {
        model: "gemini-1.5-pro".to_string(),
        temperature: Some(2.5), // Invalid: > 2.0 for Gemini
        ..Default::default()
    };
    let builder = GeminiRequestBuilder::new(invalid_temp_params, GeminiParams::default());
    let result =
        builder.build_chat_request_with_config(messages.clone(), None, false, &validation_config);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("temperature must be between 0.0 and 2.0")
    );

    // Test invalid topP (> 1.0)
    let invalid_top_p_params = CommonParams {
        model: "gemini-1.5-pro".to_string(),
        top_p: Some(1.5), // Invalid: > 1.0
        ..Default::default()
    };
    let builder = GeminiRequestBuilder::new(invalid_top_p_params, GeminiParams::default());
    let result =
        builder.build_chat_request_with_config(messages.clone(), None, false, &validation_config);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("topP must be between 0.0 and 1.0")
    );

    // Test with validation disabled (default) - should succeed even with invalid values
    let default_config = RequestBuilderConfig::default();
    let invalid_params = CommonParams {
        model: "gemini-1.5-pro".to_string(),
        temperature: Some(3.0), // Invalid but should pass when validation disabled
        ..Default::default()
    };
    let builder = GeminiRequestBuilder::new(invalid_params, GeminiParams::default());
    let result =
        builder.build_chat_request_with_config(messages.clone(), None, false, &default_config);
    assert!(result.is_ok()); // Should pass when validation disabled

    // Test with valid parameters and validation enabled - should succeed
    let valid_params = CommonParams {
        model: "gemini-1.5-pro".to_string(),
        temperature: Some(0.7), // Valid: 0.0 <= 0.7 <= 2.0
        top_p: Some(0.9),       // Valid: 0.0 <= 0.9 <= 1.0
        max_tokens: Some(1000), // Valid: > 0
        ..Default::default()
    };
    let builder = GeminiRequestBuilder::new(valid_params, GeminiParams::default());
    let result = builder.build_chat_request_with_config(messages, None, false, &validation_config);
    assert!(result.is_ok()); // Should pass with valid parameters
}

/// Test Groq OpenAI compatibility for parameter validation
#[test]
fn test_groq_openai_compatibility() {
    // Groq uses OpenAI-compatible parameters, so it should behave identically to OpenAI
    let common_params = CommonParams {
        model: "llama-3.3-70b-versatile".to_string(),
        temperature: Some(0.8),
        max_tokens: Some(2000),
        top_p: Some(0.95),
        stop_sequences: Some(vec!["HALT".to_string()]),
        seed: Some(123),
    };

    let openai_params = OpenAiParams::default();
    let openai_builder = OpenAiRequestBuilder::new(common_params.clone(), openai_params);
    let openai_mapped = openai_builder.map_common_to_openai(&common_params);

    // Groq should produce identical parameter mapping to OpenAI
    // since it uses OpenAI-compatible format
    assert_eq!(openai_mapped["model"], "llama-3.3-70b-versatile");
    assert!((openai_mapped["temperature"].as_f64().unwrap() - 0.8).abs() < 0.001);
    assert_eq!(openai_mapped["max_tokens"], 2000);
    assert!((openai_mapped["top_p"].as_f64().unwrap() - 0.95).abs() < 0.001);
    assert_eq!(openai_mapped["stop"], serde_json::json!(["HALT"]));
    assert_eq!(openai_mapped["seed"], 123);

    // Test validation compatibility - Groq should use same validation rules as OpenAI
    let validation_config = RequestBuilderConfig {
        strict_validation: false,
        provider_validation: true,
    };

    // Test invalid temperature (> 2.0) - should fail for both OpenAI and Groq
    let invalid_temp_params = CommonParams {
        model: "llama-3.3-70b-versatile".to_string(),
        temperature: Some(2.5), // Invalid: > 2.0
        ..Default::default()
    };

    let messages = vec![siumai::types::ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("Hello".to_string()),
        metadata: Default::default(),
        tool_calls: None,
        tool_call_id: None,
    }];

    let invalid_builder = OpenAiRequestBuilder::new(invalid_temp_params, OpenAiParams::default());
    let result =
        invalid_builder.build_chat_request_with_config(messages, None, false, &validation_config);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("temperature must be between 0.0 and 2.0")
    );
}

/// Test Ollama native API parameter handling
#[test]
fn test_ollama_native_api_parameters() {
    // Ollama uses a completely different API structure from OpenAI
    // It uses an "options" object for model parameters and has unique local parameters

    // Test that Ollama parameters are structured correctly
    let ollama_params = siumai::providers::ollama::config::OllamaParams {
        keep_alive: Some("10m".to_string()),
        raw: Some(false),
        format: Some("json".to_string()),
        numa: Some(true),
        num_ctx: Some(4096),
        num_gpu: Some(1),
        num_thread: Some(8),
        think: Some(true),
        options: Some({
            let mut options = std::collections::HashMap::new();
            options.insert(
                "temperature".to_string(),
                serde_json::Value::Number(serde_json::Number::from_f64(0.7).unwrap()),
            );
            options.insert(
                "top_p".to_string(),
                serde_json::Value::Number(serde_json::Number::from_f64(0.9).unwrap()),
            );
            options
        }),
        ..Default::default()
    };

    // Verify Ollama-specific parameters
    assert_eq!(ollama_params.keep_alive, Some("10m".to_string()));
    assert_eq!(ollama_params.raw, Some(false));
    assert_eq!(ollama_params.format, Some("json".to_string()));
    assert_eq!(ollama_params.numa, Some(true));
    assert_eq!(ollama_params.num_ctx, Some(4096));
    assert_eq!(ollama_params.num_gpu, Some(1));
    assert_eq!(ollama_params.num_thread, Some(8));
    assert_eq!(ollama_params.think, Some(true));

    // Verify options structure
    if let Some(options) = &ollama_params.options {
        assert!((options["temperature"].as_f64().unwrap() - 0.7).abs() < 0.001);
        assert!((options["top_p"].as_f64().unwrap() - 0.9).abs() < 0.001);
    }

    // Test Ollama config builder
    let config = siumai::providers::ollama::config::OllamaConfig::builder()
        .base_url("http://localhost:11434")
        .model("llama3.2")
        .keep_alive("5m")
        .format("json")
        .think(true)
        .option(
            "temperature",
            serde_json::Value::Number(serde_json::Number::from_f64(0.8).unwrap()),
        )
        .build();

    assert!(config.is_ok());
    let config = config.unwrap();
    assert_eq!(config.base_url, "http://localhost:11434");
    assert_eq!(config.model, Some("llama3.2".to_string()));
    assert_eq!(config.ollama_params.keep_alive, Some("5m".to_string()));
    assert_eq!(config.ollama_params.format, Some("json".to_string()));
    assert_eq!(config.ollama_params.think, Some(true));
}

/// Test OpenAI-compatible providers using new adapter system
#[test]
fn test_openai_compatible_providers_validation() {
    use siumai::providers::openai_compatible::{
        adapter::ProviderAdapter,
        providers::siliconflow::SiliconFlowAdapter,
        types::RequestType,
    };

    // Test SiliconFlow adapter
    let adapter = SiliconFlowAdapter;

    // Test provider properties
    assert_eq!(adapter.provider_id(), "siliconflow");
    assert_eq!(adapter.base_url(), "https://api.siliconflow.cn/v1");

    // Test model validation
    assert!(adapter.validate_model("deepseek-chat").is_ok());
    assert!(adapter.validate_model("").is_err());

    // Test field mappings for DeepSeek models
    let deepseek_mappings = adapter.get_field_mappings("deepseek-chat");
    assert_eq!(deepseek_mappings.thinking_fields, vec!["reasoning_content", "thinking"]);

    // Test field mappings for other models
    let standard_mappings = adapter.get_field_mappings("qwen-turbo");
    assert_eq!(standard_mappings.thinking_fields, vec!["thinking"]);

    // Test parameter transformation
    let mut params = serde_json::json!({
        "model": "deepseek-chat",
        "messages": [],
        "thinking_budget": 1000,
        "temperature": 0.7
    });

    // Test parameter transformation
    assert!(adapter.transform_request_params(&mut params, "deepseek-chat", RequestType::Chat).is_ok());

    // Verify transformation worked
    assert!(params.get("reasoning_effort").is_some());
    assert!(params.get("thinking_budget").is_none());

    // Test capabilities
    let caps = adapter.capabilities();
    assert!(caps.chat);
    assert!(caps.streaming);
    assert!(caps.thinking);
    assert!(caps.rerank);
}

/// Test parameter mapping consistency across different scenarios
#[test]
fn test_parameter_mapping_consistency() {
    let common_params = CommonParams {
        model: "test-model".to_string(),
        temperature: Some(0.8),
        max_tokens: Some(2000),
        top_p: Some(0.95),
        stop_sequences: Some(vec!["HALT".to_string()]),
        seed: Some(123),
    };

    // Test OpenAI mapping
    let openai_params = OpenAiParams::default();
    let openai_builder = OpenAiRequestBuilder::new(common_params.clone(), openai_params);
    let openai_mapped = openai_builder.map_common_to_openai(&common_params);

    // Test Anthropic mapping
    let anthropic_params = AnthropicParams::default();
    let anthropic_builder = AnthropicRequestBuilder::new(common_params.clone(), anthropic_params);
    let anthropic_mapped = anthropic_builder.map_common_to_anthropic(&common_params);

    // Both should have the same basic parameters
    assert_eq!(openai_mapped["model"], anthropic_mapped["model"]);
    assert_eq!(
        openai_mapped["temperature"],
        anthropic_mapped["temperature"]
    );
    assert_eq!(openai_mapped["max_tokens"], anthropic_mapped["max_tokens"]);
    assert_eq!(openai_mapped["top_p"], anthropic_mapped["top_p"]);

    // But different stop sequence formats
    assert_eq!(openai_mapped["stop"], serde_json::json!(["HALT"]));
    assert_eq!(
        anthropic_mapped["stop_sequences"],
        serde_json::json!(["HALT"])
    );

    // And different seed handling
    assert_eq!(openai_mapped["seed"], 123);
    assert!(anthropic_mapped.get("seed").is_none());
}
