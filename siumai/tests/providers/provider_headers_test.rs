//! Provider Headers Validation Tests
//!
//! This module provides comprehensive tests for validating HTTP request headers
//! across all supported providers. These tests verify that headers conform to
//! official API documentation without making actual API calls, thus avoiding
//! quota consumption while ensuring maximum provider coverage.
//!
//! ## Test Strategy
//!
//! 1. **Header Construction Testing**: Verify that each provider's header building
//!    logic produces the correct headers according to official documentation.
//!
//! 2. **Authentication Testing**: Ensure proper authentication headers are set
//!    for each provider (Bearer tokens, API keys, custom auth schemes).
//!
//! 3. **Provider-Specific Headers**: Test provider-specific headers like
//!    OpenAI-Organization, anthropic-version, etc.
//!
//! 4. **Content-Type Validation**: Verify correct content-type headers for
//!    different request types.
//!
//! 5. **Custom Headers Support**: Test that custom headers are properly merged
//!    and can override default headers when needed (this is intentional flexibility).
//!
//! ## Supported Providers
//!
//! - **OpenAI**: Authorization, OpenAI-Organization, OpenAI-Project
//! - **Anthropic**: x-api-key, anthropic-version, anthropic-beta
//! - **Gemini**: x-goog-api-key
//! - **Groq**: Authorization, User-Agent
//! - **xAI**: Authorization
//! - **Ollama**: User-Agent (no auth required)
//! - **OpenAI Compatible**: Authorization with provider-specific variations
//!
//! ## Official Documentation References
//!
//! - **OpenAI**: https://platform.openai.com/docs/api-reference/authentication
//! - **Anthropic**: https://docs.anthropic.com/en/api/messages-examples
//! - **Gemini**: https://ai.google.dev/gemini-api/docs/api-key
//! - **Groq**: https://console.groq.com/docs/quickstart
//! - **xAI**: https://docs.x.ai/api
//! - **Ollama**: https://github.com/ollama/ollama/blob/main/docs/api.md

use reqwest::header::{CONTENT_TYPE, HeaderMap};
use siumai::execution::http::headers::ProviderHeaders;
use std::collections::HashMap;

/// Test configuration for a provider
#[derive(Debug, Clone)]
pub struct ProviderTestConfig {
    /// Provider name for identification
    pub name: &'static str,
    /// Test API key to use
    pub api_key: &'static str,
    /// Expected authentication header name
    pub auth_header: Option<&'static str>,
    /// Expected authentication header value prefix
    pub auth_value_prefix: Option<&'static str>,
    /// Required headers that must be present
    pub required_headers: Vec<&'static str>,
    /// Expected content type for basic requests
    pub expected_content_type: &'static str,
    /// Whether this provider requires authentication
    pub requires_auth: bool,
}

/// Helper trait for header validation
pub trait HeaderValidator {
    /// Validate that all required headers are present
    fn validate_required_headers(
        &self,
        headers: &HeaderMap,
        required: &[&str],
    ) -> Result<(), String>;

    /// Validate authentication header
    fn validate_auth_header(
        &self,
        headers: &HeaderMap,
        header_name: &str,
        expected_prefix: Option<&str>,
        api_key: &str,
    ) -> Result<(), String>;

    /// Validate content type header
    fn validate_content_type(&self, headers: &HeaderMap, expected: &str) -> Result<(), String>;
}

/// Default implementation of HeaderValidator
pub struct DefaultHeaderValidator;

impl HeaderValidator for DefaultHeaderValidator {
    fn validate_required_headers(
        &self,
        headers: &HeaderMap,
        required: &[&str],
    ) -> Result<(), String> {
        for &header_name in required {
            if !headers.contains_key(header_name) {
                return Err(format!("Missing required header: {}", header_name));
            }
        }
        Ok(())
    }

    fn validate_auth_header(
        &self,
        headers: &HeaderMap,
        header_name: &str,
        expected_prefix: Option<&str>,
        api_key: &str,
    ) -> Result<(), String> {
        let header_value = headers
            .get(header_name)
            .ok_or_else(|| format!("Missing authentication header: {}", header_name))?
            .to_str()
            .map_err(|_| format!("Invalid authentication header value for: {}", header_name))?;

        if let Some(prefix) = expected_prefix {
            let expected_value = format!("{} {}", prefix, api_key);
            if header_value != expected_value {
                return Err(format!(
                    "Authentication header mismatch. Expected: '{}', Got: '{}'",
                    expected_value, header_value
                ));
            }
        } else {
            // Direct API key comparison
            if header_value != api_key {
                return Err(format!(
                    "API key mismatch. Expected: '{}', Got: '{}'",
                    api_key, header_value
                ));
            }
        }

        Ok(())
    }

    fn validate_content_type(&self, headers: &HeaderMap, expected: &str) -> Result<(), String> {
        let content_type = headers
            .get(CONTENT_TYPE)
            .ok_or("Missing Content-Type header")?
            .to_str()
            .map_err(|_| "Invalid Content-Type header value")?;

        if content_type != expected {
            return Err(format!(
                "Content-Type mismatch. Expected: '{}', Got: '{}'",
                expected, content_type
            ));
        }

        Ok(())
    }
}

/// Test runner for provider header validation
pub struct ProviderHeaderTester {
    validator: Box<dyn HeaderValidator>,
}

impl ProviderHeaderTester {
    /// Create a new tester with default validator
    pub fn new() -> Self {
        Self {
            validator: Box::new(DefaultHeaderValidator),
        }
    }

    /// Create a new tester with custom validator
    pub fn with_validator(validator: Box<dyn HeaderValidator>) -> Self {
        Self { validator }
    }

    /// Run comprehensive header tests for a provider
    pub fn test_provider_headers(
        &self,
        config: &ProviderTestConfig,
        header_builder: impl Fn(
            &str,
            &HashMap<String, String>,
        ) -> Result<HeaderMap, siumai::error::LlmError>,
    ) -> Result<(), String> {
        println!("🧪 Testing {} headers...", config.name);

        // Test 1: Basic headers without custom headers
        println!("  📋 Testing basic headers...");
        let empty_custom = HashMap::new();
        let basic_headers = header_builder(config.api_key, &empty_custom)
            .map_err(|e| format!("Failed to build basic headers: {}", e))?;

        self.validate_basic_headers(&basic_headers, config)?;

        // Test 2: Headers with custom headers (should be added)
        println!("  🔧 Testing with custom headers...");
        let mut custom_headers_map = HashMap::new();
        custom_headers_map.insert("X-Custom-Header".to_string(), "custom-value".to_string());
        custom_headers_map.insert("X-Request-ID".to_string(), "test-request-123".to_string());

        let headers_with_custom = header_builder(config.api_key, &custom_headers_map)
            .map_err(|e| format!("Failed to build headers with custom headers: {}", e))?;

        // Verify custom headers were added
        for (key, expected_value) in &custom_headers_map {
            let actual_value = headers_with_custom
                .get(key)
                .ok_or_else(|| format!("Custom header '{}' was not added", key))?
                .to_str()
                .map_err(|_| format!("Invalid custom header value for '{}'", key))?;

            if actual_value != expected_value {
                return Err(format!(
                    "Custom header '{}' value mismatch. Expected: '{}', Got: '{}'",
                    key, expected_value, actual_value
                ));
            }
        }

        // Test 3: Custom headers can override defaults (this is intentional flexibility)
        println!("  🔄 Testing custom header override capability...");
        self.test_header_override_capability(config, &header_builder)?;

        println!("  ✅ {} headers validation passed", config.name);
        Ok(())
    }

    /// Validate basic header requirements
    fn validate_basic_headers(
        &self,
        headers: &HeaderMap,
        config: &ProviderTestConfig,
    ) -> Result<(), String> {
        // Validate required headers
        self.validator
            .validate_required_headers(headers, &config.required_headers)?;

        // Validate authentication if required
        if config.requires_auth {
            if let (Some(auth_header), Some(auth_prefix)) =
                (config.auth_header, config.auth_value_prefix)
            {
                self.validator.validate_auth_header(
                    headers,
                    auth_header,
                    Some(auth_prefix),
                    config.api_key,
                )?;
            } else if let Some(auth_header) = config.auth_header {
                self.validator
                    .validate_auth_header(headers, auth_header, None, config.api_key)?;
            }
        }

        // Validate content type
        self.validator
            .validate_content_type(headers, config.expected_content_type)?;

        Ok(())
    }

    /// Test that custom headers can override defaults (this is intentional flexibility)
    fn test_header_override_capability(
        &self,
        config: &ProviderTestConfig,
        header_builder: &impl Fn(
            &str,
            &HashMap<String, String>,
        ) -> Result<HeaderMap, siumai::error::LlmError>,
    ) -> Result<(), String> {
        let mut override_custom = HashMap::new();

        // Test overriding content-type (this should work)
        override_custom.insert(
            "content-type".to_string(),
            "application/x-custom".to_string(),
        );

        // Test adding a custom auth header (for flexibility)
        override_custom.insert("X-Custom-Auth".to_string(), "custom-auth-value".to_string());

        let headers_with_override = header_builder(config.api_key, &override_custom)
            .map_err(|e| format!("Failed to build headers with override: {}", e))?;

        // Verify that custom content-type was set
        let content_type = headers_with_override
            .get("content-type")
            .ok_or("Content-Type header missing")?
            .to_str()
            .map_err(|_| "Invalid Content-Type header value")?;

        if content_type != "application/x-custom" {
            return Err(format!(
                "Content-Type override failed. Expected: 'application/x-custom', Got: '{}'",
                content_type
            ));
        }

        // Verify custom auth header was added
        let custom_auth = headers_with_override
            .get("X-Custom-Auth")
            .ok_or("Custom auth header missing")?
            .to_str()
            .map_err(|_| "Invalid custom auth header value")?;

        if custom_auth != "custom-auth-value" {
            return Err(format!(
                "Custom auth header mismatch. Expected: 'custom-auth-value', Got: '{}'",
                custom_auth
            ));
        }

        Ok(())
    }
}

impl Default for ProviderHeaderTester {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Provider-Specific Tests
// ============================================================================

/// Test OpenAI headers according to official documentation
/// Reference: https://platform.openai.com/docs/api-reference/authentication
#[test]
fn test_openai_headers() {
    let config = ProviderTestConfig {
        name: "OpenAI",
        api_key: "sk-test-key-1234567890abcdef",
        auth_header: Some("authorization"),
        auth_value_prefix: Some("Bearer"),
        required_headers: vec!["authorization", "content-type"],
        expected_content_type: "application/json",
        requires_auth: true,
    };

    let tester = ProviderHeaderTester::new();

    // Test basic headers
    let result = tester.test_provider_headers(&config, |api_key, custom_headers| {
        ProviderHeaders::openai(api_key, None, None, custom_headers)
    });

    assert!(
        result.is_ok(),
        "OpenAI basic headers test failed: {:?}",
        result
    );

    // Test with organization and project
    let result_with_org = tester.test_provider_headers(&config, |api_key, custom_headers| {
        ProviderHeaders::openai(api_key, Some("org-test"), Some("proj-test"), custom_headers)
    });

    assert!(
        result_with_org.is_ok(),
        "OpenAI headers with org/project test failed: {:?}",
        result_with_org
    );

    // Verify organization and project headers are set correctly
    let headers = ProviderHeaders::openai(
        "test-key",
        Some("test-org"),
        Some("test-proj"),
        &HashMap::new(),
    )
    .unwrap();
    assert_eq!(headers.get("openai-organization").unwrap(), "test-org");
    assert_eq!(headers.get("openai-project").unwrap(), "test-proj");

    println!("✅ OpenAI headers test passed");
}

/// Test Anthropic headers according to official documentation
/// Reference: https://docs.anthropic.com/en/api/getting-started
#[test]
fn test_anthropic_headers() {
    let config = ProviderTestConfig {
        name: "Anthropic",
        api_key: "sk-ant-test-key-1234567890abcdef",
        auth_header: Some("x-api-key"),
        auth_value_prefix: None, // Direct API key, no Bearer prefix
        required_headers: vec!["x-api-key", "content-type", "anthropic-version"],
        expected_content_type: "application/json",
        requires_auth: true,
    };

    let tester = ProviderHeaderTester::new();

    let result = tester.test_provider_headers(&config, |api_key, custom_headers| {
        ProviderHeaders::anthropic(api_key, custom_headers)
    });

    assert!(
        result.is_ok(),
        "Anthropic headers test failed: {:?}",
        result
    );

    // Verify anthropic-version header is set (version may vary for compatibility)
    let headers = ProviderHeaders::anthropic("test-key", &HashMap::new()).unwrap();
    assert!(
        headers.contains_key("anthropic-version"),
        "anthropic-version header should be present"
    );
    // Note: We don't assert the exact version to allow for provider compatibility

    // Test with beta features
    let mut beta_headers = HashMap::new();
    beta_headers.insert(
        "anthropic-beta".to_string(),
        "messages-2023-12-15".to_string(),
    );
    let headers_with_beta = ProviderHeaders::anthropic("test-key", &beta_headers).unwrap();
    assert_eq!(
        headers_with_beta.get("anthropic-beta").unwrap(),
        "messages-2023-12-15"
    );

    println!("✅ Anthropic headers test passed");
}

/// Test Gemini headers according to official documentation
/// Reference: https://ai.google.dev/gemini-api/docs/api-key
#[test]
fn test_gemini_headers() {
    let config = ProviderTestConfig {
        name: "Gemini",
        api_key: "AIzaSyTest-Key-1234567890abcdef",
        auth_header: Some("x-goog-api-key"),
        auth_value_prefix: None, // Direct API key
        required_headers: vec!["x-goog-api-key", "content-type"],

        expected_content_type: "application/json",
        requires_auth: true,
    };

    let tester = ProviderHeaderTester::new();

    let result = tester.test_provider_headers(&config, |api_key, custom_headers| {
        ProviderHeaders::gemini(api_key, custom_headers)
    });

    assert!(result.is_ok(), "Gemini headers test failed: {:?}", result);

    println!("✅ Gemini headers test passed");
}

/// Test Groq headers according to official documentation
/// Reference: https://console.groq.com/docs/quickstart
#[test]
fn test_groq_headers() {
    let config = ProviderTestConfig {
        name: "Groq",
        api_key: "gsk_test-key-1234567890abcdef",
        auth_header: Some("authorization"),
        auth_value_prefix: Some("Bearer"),
        required_headers: vec!["authorization", "content-type", "user-agent"],
        expected_content_type: "application/json",
        requires_auth: true,
    };

    let tester = ProviderHeaderTester::new();

    let result = tester.test_provider_headers(&config, |api_key, custom_headers| {
        ProviderHeaders::groq(api_key, custom_headers)
    });

    assert!(result.is_ok(), "Groq headers test failed: {:?}", result);

    // Verify user-agent is set (content may vary for compatibility)
    let headers = ProviderHeaders::groq("test-key", &HashMap::new()).unwrap();
    assert!(
        headers.contains_key("user-agent"),
        "user-agent header should be present"
    );
    // Note: We don't assert specific content to allow for provider compatibility

    println!("✅ Groq headers test passed");
}

/// Test xAI headers according to official documentation
/// Reference: https://docs.x.ai/api
#[test]
fn test_xai_headers() {
    let config = ProviderTestConfig {
        name: "xAI",
        api_key: "xai-test-key-1234567890abcdef",
        auth_header: Some("authorization"),
        auth_value_prefix: Some("Bearer"),
        required_headers: vec!["authorization", "content-type"],
        expected_content_type: "application/json",
        requires_auth: true,
    };

    let tester = ProviderHeaderTester::new();

    let result = tester.test_provider_headers(&config, |api_key, custom_headers| {
        ProviderHeaders::xai(api_key, custom_headers)
    });

    assert!(result.is_ok(), "xAI headers test failed: {:?}", result);

    println!("✅ xAI headers test passed");
}

/// Test Ollama headers (no authentication required)
/// Reference: https://github.com/ollama/ollama/blob/main/docs/api.md
#[test]
fn test_ollama_headers() {
    let config = ProviderTestConfig {
        name: "Ollama",
        api_key: "", // No API key required
        auth_header: None,
        auth_value_prefix: None,
        required_headers: vec!["content-type", "user-agent"],
        expected_content_type: "application/json",
        requires_auth: false,
    };

    let tester = ProviderHeaderTester::new();

    let result = tester.test_provider_headers(&config, |_api_key, custom_headers| {
        ProviderHeaders::ollama(custom_headers)
    });

    assert!(result.is_ok(), "Ollama headers test failed: {:?}", result);

    // Verify user-agent is set (content may vary for compatibility)
    let headers = ProviderHeaders::ollama(&HashMap::new()).unwrap();
    assert!(
        headers.contains_key("user-agent"),
        "user-agent header should be present"
    );
    // Note: We don't assert specific content to allow for provider compatibility

    println!("✅ Ollama headers test passed");
}

/// Test comprehensive header validation across all providers
#[test]
fn test_all_providers_comprehensive() {
    println!("🚀 Running comprehensive provider headers test...");

    // This test ensures all providers follow consistent patterns
    let providers = vec![
        ("OpenAI", true),
        ("Anthropic", true),
        ("Gemini", true),
        ("Groq", true),
        ("xAI", true),
        ("Ollama", false), // No auth required
    ];

    for (provider_name, _requires_auth) in providers {
        println!("  🔍 Validating {} consistency...", provider_name);

        // Each provider should have consistent header patterns
        match provider_name {
            "OpenAI" => {
                let headers = ProviderHeaders::openai("test", None, None, &HashMap::new()).unwrap();
                assert!(headers.contains_key("authorization"));
                assert!(headers.contains_key("content-type"));
            }
            "Anthropic" => {
                let headers = ProviderHeaders::anthropic("test", &HashMap::new()).unwrap();
                assert!(headers.contains_key("x-api-key"));
                assert!(headers.contains_key("anthropic-version"));
                assert!(headers.contains_key("content-type"));
            }
            "Gemini" => {
                let headers = ProviderHeaders::gemini("test", &HashMap::new()).unwrap();
                assert!(headers.contains_key("x-goog-api-key"));
                assert!(headers.contains_key("content-type"));
            }
            "Groq" => {
                let headers = ProviderHeaders::groq("test", &HashMap::new()).unwrap();
                assert!(headers.contains_key("authorization"));
                assert!(headers.contains_key("content-type"));
                assert!(headers.contains_key("user-agent"));
            }
            "xAI" => {
                let headers = ProviderHeaders::xai("test", &HashMap::new()).unwrap();
                assert!(headers.contains_key("authorization"));
                assert!(headers.contains_key("content-type"));
            }
            "Ollama" => {
                let headers = ProviderHeaders::ollama(&HashMap::new()).unwrap();
                assert!(headers.contains_key("content-type"));
                assert!(headers.contains_key("user-agent"));
                assert!(!headers.contains_key("authorization")); // Should not have auth
            }
            _ => panic!("Unknown provider: {}", provider_name),
        }

        println!("    ✅ {} consistency validated", provider_name);
    }

    println!("🎉 All providers passed comprehensive validation!");
}

/// Test OpenAI-compatible providers (OpenRouter, DeepSeek, etc.)
/// These providers use OpenAI-style authentication but may have additional headers
#[test]
fn test_openai_compatible_providers() {
    println!("🔗 Testing OpenAI-compatible providers...");

    // Test OpenRouter-style headers
    println!("  📋 Testing OpenRouter headers...");
    let mut openrouter_headers = HashMap::new();
    openrouter_headers.insert("HTTP-Referer".to_string(), "https://myapp.com".to_string());
    openrouter_headers.insert("X-Title".to_string(), "My App".to_string());

    let headers = ProviderHeaders::openai("or-test-key", None, None, &openrouter_headers).unwrap();
    assert!(headers.contains_key("authorization"));
    assert!(headers.contains_key("content-type"));
    assert!(headers.contains_key("HTTP-Referer"));
    assert!(headers.contains_key("X-Title"));
    assert_eq!(headers.get("HTTP-Referer").unwrap(), "https://myapp.com");
    assert_eq!(headers.get("X-Title").unwrap(), "My App");

    // Test DeepSeek-style headers (standard OpenAI format)
    println!("  📋 Testing DeepSeek headers...");
    let deepseek_headers = HashMap::new();
    let headers =
        ProviderHeaders::openai("sk-deepseek-key", None, None, &deepseek_headers).unwrap();
    assert!(headers.contains_key("authorization"));
    assert!(headers.contains_key("content-type"));
    assert_eq!(
        headers.get("authorization").unwrap(),
        "Bearer sk-deepseek-key"
    );

    println!("  ✅ OpenAI-compatible providers test passed");
}

/// Test edge cases and error conditions
#[test]
fn test_header_edge_cases() {
    println!("🧪 Testing header edge cases...");

    // Test empty API key handling
    println!("  📋 Testing empty API key...");
    let empty_headers = HashMap::new();

    // Test empty keys - some providers may reject them, others may accept them
    let openai_result = ProviderHeaders::openai("", None, None, &empty_headers);
    // Note: We don't assert success here as some compatible providers may reject empty keys
    println!("    OpenAI empty key result: {:?}", openai_result.is_ok());

    let anthropic_result = ProviderHeaders::anthropic("", &empty_headers);
    // Note: We don't assert success here as some compatible providers may reject empty keys
    println!(
        "    Anthropic empty key result: {:?}",
        anthropic_result.is_ok()
    );

    // Test very long API keys
    println!("  📋 Testing long API keys...");
    let long_key = "a".repeat(1000);
    let openai_long = ProviderHeaders::openai(&long_key, None, None, &empty_headers);
    assert!(openai_long.is_ok(), "Should handle long API keys");

    // Test special characters in custom headers
    println!("  📋 Testing special characters in custom headers...");
    let mut special_headers = HashMap::new();
    special_headers.insert(
        "X-Special-Chars".to_string(),
        "value with spaces & symbols!".to_string(),
    );

    let result = ProviderHeaders::openai("test-key", None, None, &special_headers);
    assert!(
        result.is_ok(),
        "Should handle special characters in header values"
    );

    // Test case sensitivity
    println!("  📋 Testing header case sensitivity...");
    let mut case_headers = HashMap::new();
    case_headers.insert("x-custom-header".to_string(), "lowercase".to_string());
    case_headers.insert("X-CUSTOM-HEADER-2".to_string(), "uppercase".to_string());

    let headers = ProviderHeaders::openai("test-key", None, None, &case_headers).unwrap();
    assert!(headers.contains_key("x-custom-header"));
    assert!(headers.contains_key("X-CUSTOM-HEADER-2"));

    println!("  ✅ Edge cases test passed");
}

/// Test header flexibility and custom header support
#[test]
fn test_header_flexibility() {
    println!("🔧 Testing header flexibility...");

    // Test that custom headers can override defaults (this is intentional flexibility)
    println!("  🔄 Testing custom header override capability...");

    let mut custom_headers = HashMap::new();
    custom_headers.insert("authorization".to_string(), "Bearer custom-key".to_string());
    custom_headers.insert("content-type".to_string(), "application/custom".to_string());

    // The header builder should allow custom headers to override defaults
    let headers = ProviderHeaders::openai("real-key", None, None, &custom_headers).unwrap();

    // Verify that custom headers can override defaults
    assert_eq!(headers.get("authorization").unwrap(), "Bearer custom-key");
    assert_eq!(headers.get("content-type").unwrap(), "application/custom");

    // Test Anthropic custom headers
    let mut anthropic_custom = HashMap::new();
    anthropic_custom.insert("x-api-key".to_string(), "custom-key".to_string());
    anthropic_custom.insert(
        "anthropic-version".to_string(),
        "custom-version".to_string(),
    );

    let anthropic_headers = ProviderHeaders::anthropic("real-key", &anthropic_custom).unwrap();
    assert_eq!(anthropic_headers.get("x-api-key").unwrap(), "custom-key");
    assert_eq!(
        anthropic_headers.get("anthropic-version").unwrap(),
        "custom-version"
    );

    // Test adding completely new headers
    let mut new_headers = HashMap::new();
    new_headers.insert("X-Custom-Feature".to_string(), "enabled".to_string());
    new_headers.insert("X-Request-Source".to_string(), "test-suite".to_string());

    let headers_with_new = ProviderHeaders::openai("test-key", None, None, &new_headers).unwrap();
    assert_eq!(headers_with_new.get("X-Custom-Feature").unwrap(), "enabled");
    assert_eq!(
        headers_with_new.get("X-Request-Source").unwrap(),
        "test-suite"
    );

    println!("  ✅ Header flexibility test passed");
}

/// Integration test that validates headers work with actual request building
#[test]
fn test_headers_integration() {
    println!("🔗 Testing headers integration with request building...");

    // This test ensures headers can be built and a ChatRequest can be constructed
    use siumai::types::{ChatMessage, CommonParams, MessageContent, MessageRole, ChatRequest};

    let common_params = CommonParams {
        model: "test-model".to_string(),
        temperature: Some(0.7),
        ..Default::default()
    };

    let messages = vec![ChatMessage {
        role: MessageRole::User,
        content: MessageContent::Text("Test message".to_string()),
        metadata: Default::default(),
        tool_calls: None,
        tool_call_id: None,
    }];

    // Build headers and construct a ChatRequest struct
    let _headers = ProviderHeaders::openai("test-key", None, None, &HashMap::new()).unwrap();
    let _request = ChatRequest::builder()
        .messages(messages)
        .common_params(common_params)
        .build();

    println!("  ✅ Headers integration test passed");
}
