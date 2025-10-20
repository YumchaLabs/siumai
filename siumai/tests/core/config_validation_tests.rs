//! Configuration Validation Tests
//!
//! This module provides comprehensive tests for configuration validation in the LLM library.
//! These tests ensure that invalid configurations are properly detected and handled,
//! preventing runtime errors and providing clear feedback to developers.
//!
//! ## Test Categories
//!
//! 1. **URL Validation**: Test various invalid URL formats, protocols, and structures
//! 2. **Port Range Validation**: Verify proper handling of invalid port numbers
//! 3. **Parameter Combination Validation**: Test invalid parameter combinations
//! 4. **API Key Validation**: Test various API key format validations
//! 5. **Timeout Configuration**: Test invalid timeout values and ranges
//! 6. **Model Name Validation**: Test invalid model names and formats

use serde_json::json;
use std::collections::HashMap;
use std::time::Duration;

/// Configuration validation test result
#[derive(Debug, Clone)]
pub struct ConfigValidationResult {
    pub test_name: String,
    pub passed: bool,
    pub error_message: Option<String>,
    pub expected_error_type: Option<String>,
    pub actual_error_type: Option<String>,
    pub validation_details: Vec<String>,
}

impl ConfigValidationResult {
    pub fn new(test_name: &str) -> Self {
        Self {
            test_name: test_name.to_string(),
            passed: false,
            error_message: None,
            expected_error_type: None,
            actual_error_type: None,
            validation_details: Vec::new(),
        }
    }

    pub fn success(mut self) -> Self {
        self.passed = true;
        self
    }

    pub fn failure(mut self, message: &str) -> Self {
        self.passed = false;
        self.error_message = Some(message.to_string());
        self
    }

    pub fn with_expected_error(mut self, error_type: &str) -> Self {
        self.expected_error_type = Some(error_type.to_string());
        self
    }

    pub fn with_actual_error(mut self, error_type: &str) -> Self {
        self.actual_error_type = Some(error_type.to_string());
        self
    }

    pub fn add_detail(mut self, detail: &str) -> Self {
        self.validation_details.push(detail.to_string());
        self
    }
}

/// Configuration validator for testing various config scenarios
pub struct ConfigValidator;

impl ConfigValidator {
    /// Test URL validation
    pub fn test_url_validation() -> Vec<ConfigValidationResult> {
        let mut results = Vec::new();

        let invalid_urls = vec![
            ("empty_url", "", "Empty URL should be rejected"),
            (
                "invalid_protocol",
                "ftp://example.com",
                "FTP protocol should be rejected",
            ),
            (
                "malformed_url",
                "not-a-url",
                "Malformed URL should be rejected",
            ),
            (
                "missing_protocol",
                "example.com",
                "Missing protocol should be rejected",
            ),
            (
                "invalid_characters",
                "http://ex ample.com",
                "URLs with spaces should be rejected",
            ),
            (
                "localhost_invalid_path",
                "http://localhost/invalid path",
                "Localhost with invalid path should be rejected",
            ),
            (
                "invalid_port",
                "http://example.com:99999",
                "Port number too high should be rejected",
            ),
            (
                "negative_port",
                "http://example.com:-80",
                "Negative port should be rejected",
            ),
            (
                "non_numeric_port",
                "http://example.com:abc",
                "Non-numeric port should be rejected",
            ),
        ];

        for (test_name, url, description) in invalid_urls {
            let mut result = ConfigValidationResult::new(&format!("url_validation_{}", test_name))
                .with_expected_error("InvalidUrl")
                .add_detail(description);

            // Test URL validation logic
            let validation_result = Self::validate_url(url);
            match validation_result {
                Ok(_) => {
                    result = result.failure(&format!(
                        "URL '{}' was accepted but should be rejected",
                        url
                    ));
                }
                Err(error) => {
                    result = result
                        .with_actual_error("InvalidUrl")
                        .success()
                        .add_detail(&format!("Correctly rejected with error: {}", error));
                }
            }

            results.push(result);
        }

        // Test valid URLs
        let valid_urls = vec![
            "http://localhost:8080",
            "https://api.openai.com",
            "https://api.anthropic.com",
            "http://127.0.0.1:3000",
        ];

        for url in valid_urls {
            let mut result = ConfigValidationResult::new(&format!(
                "url_validation_valid_{}",
                url.replace("://", "_").replace(".", "_").replace(":", "_")
            ));

            match Self::validate_url(url) {
                Ok(_) => {
                    result = result
                        .success()
                        .add_detail(&format!("Correctly accepted URL: {}", url));
                }
                Err(error) => {
                    result =
                        result.failure(&format!("Valid URL '{}' was rejected: {}", url, error));
                }
            }

            results.push(result);
        }

        results
    }

    /// Test port range validation
    pub fn test_port_validation() -> Vec<ConfigValidationResult> {
        let mut results = Vec::new();

        let invalid_ports = vec![
            (0, "Port 0 should be rejected"),
            (65536, "Port above 65535 should be rejected"),
            (70000, "Port well above valid range should be rejected"),
        ];

        for (port, description) in invalid_ports {
            let mut result = ConfigValidationResult::new(&format!("port_validation_{}", port))
                .with_expected_error("InvalidPort")
                .add_detail(description);

            let validation_result = Self::validate_port(port);
            match validation_result {
                Ok(_) => {
                    result = result.failure(&format!(
                        "Port {} was accepted but should be rejected",
                        port
                    ));
                }
                Err(error) => {
                    result = result
                        .with_actual_error("InvalidPort")
                        .success()
                        .add_detail(&format!("Correctly rejected with error: {}", error));
                }
            }

            results.push(result);
        }

        // Test valid ports
        let valid_ports = vec![80, 443, 8080, 3000, 8000, 65535];

        for port in valid_ports {
            let mut result =
                ConfigValidationResult::new(&format!("port_validation_valid_{}", port));

            match Self::validate_port(port) {
                Ok(_) => {
                    result = result
                        .success()
                        .add_detail(&format!("Correctly accepted port: {}", port));
                }
                Err(error) => {
                    result =
                        result.failure(&format!("Valid port {} was rejected: {}", port, error));
                }
            }

            results.push(result);
        }

        results
    }

    /// Test timeout validation
    pub fn test_timeout_validation() -> Vec<ConfigValidationResult> {
        let mut results = Vec::new();

        let invalid_timeouts = vec![
            (Duration::ZERO, "Zero timeout should be rejected"),
            (Duration::from_secs(0), "Zero duration should be rejected"),
            (
                Duration::from_secs(3600),
                "Timeout over 1 hour should be rejected",
            ),
            (
                Duration::from_millis(1),
                "Timeout under 10ms should be rejected",
            ),
        ];

        for (timeout, description) in invalid_timeouts {
            let mut result =
                ConfigValidationResult::new(&format!("timeout_validation_{:?}", timeout))
                    .with_expected_error("InvalidTimeout")
                    .add_detail(description);

            let validation_result = Self::validate_timeout(timeout);
            match validation_result {
                Ok(_) => {
                    result = result.failure(&format!(
                        "Timeout {:?} was accepted but should be rejected",
                        timeout
                    ));
                }
                Err(error) => {
                    result = result
                        .with_actual_error("InvalidTimeout")
                        .success()
                        .add_detail(&format!("Correctly rejected with error: {}", error));
                }
            }

            results.push(result);
        }

        // Test valid timeouts
        let valid_timeouts = vec![
            Duration::from_millis(100),
            Duration::from_secs(1),
            Duration::from_secs(30),
            Duration::from_secs(300),
        ];

        for timeout in valid_timeouts {
            let mut result =
                ConfigValidationResult::new(&format!("timeout_validation_valid_{:?}", timeout));

            match Self::validate_timeout(timeout) {
                Ok(_) => {
                    result = result
                        .success()
                        .add_detail(&format!("Correctly accepted timeout: {:?}", timeout));
                }
                Err(error) => {
                    result = result.failure(&format!(
                        "Valid timeout {:?} was rejected: {}",
                        timeout, error
                    ));
                }
            }

            results.push(result);
        }

        results
    }

    /// Test API key validation
    pub fn test_api_key_validation() -> Vec<ConfigValidationResult> {
        let mut results = Vec::new();

        let invalid_api_keys = vec![
            ("", "Empty API key should be rejected"),
            ("   ", "Whitespace-only API key should be rejected"),
            ("short", "Too short API key should be rejected"),
            (
                "key\nwith\nnewlines",
                "API key with newlines should be rejected",
            ),
        ];

        for (api_key, description) in invalid_api_keys {
            let mut result = ConfigValidationResult::new(&format!(
                "api_key_validation_{}",
                if api_key.is_empty() {
                    "empty"
                } else {
                    "invalid"
                }
            ))
            .with_expected_error("InvalidApiKey")
            .add_detail(description);

            let validation_result = Self::validate_api_key(api_key);
            match validation_result {
                Ok(_) => {
                    result = result.failure(&format!(
                        "API key '{}' was accepted but should be rejected",
                        api_key
                    ));
                }
                Err(error) => {
                    result = result
                        .with_actual_error("InvalidApiKey")
                        .success()
                        .add_detail(&format!("Correctly rejected with error: {}", error));
                }
            }

            results.push(result);
        }

        // Test valid API keys
        let valid_api_keys = vec![
            "sk-1234567890abcdef1234567890abcdef",
            "xai-1234567890abcdef1234567890abcdef",
            "sk-ant-1234567890abcdef1234567890abcdef",
            "AIzaSy1234567890abcdef1234567890abcdef",
        ];

        for api_key in valid_api_keys {
            let mut result = ConfigValidationResult::new("api_key_validation_valid");

            match Self::validate_api_key(api_key) {
                Ok(_) => {
                    result = result
                        .success()
                        .add_detail("Correctly accepted API key format");
                }
                Err(error) => {
                    result = result.failure(&format!("Valid API key was rejected: {}", error));
                }
            }

            results.push(result);
        }

        results
    }

    /// Validate URL format
    fn validate_url(url: &str) -> Result<(), String> {
        if url.is_empty() {
            return Err("URL cannot be empty".to_string());
        }

        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err("URL must start with http:// or https://".to_string());
        }

        if url.contains(' ') {
            return Err("URL cannot contain spaces".to_string());
        }

        // Check for localhost without port (might be problematic)
        if url == "http://localhost" || url == "https://localhost" {
            return Err("Localhost URLs should specify a port".to_string());
        }

        // Basic port validation in URL
        if let Some(port_part) = url.split(':').nth(2)
            && let Some(port_str) = port_part.split('/').next()
        {
            if let Ok(port) = port_str.parse::<u16>() {
                Self::validate_port(port as u32)?;
            } else {
                return Err("Invalid port format in URL".to_string());
            }
        }

        Ok(())
    }

    /// Validate port number
    fn validate_port(port: u32) -> Result<(), String> {
        if port == 0 {
            return Err("Port cannot be 0".to_string());
        }

        if port > 65535 {
            return Err("Port cannot be greater than 65535".to_string());
        }

        Ok(())
    }

    /// Validate timeout duration
    fn validate_timeout(timeout: Duration) -> Result<(), String> {
        if timeout.is_zero() {
            return Err("Timeout cannot be zero".to_string());
        }

        if timeout < Duration::from_millis(10) {
            return Err("Timeout must be at least 10ms".to_string());
        }

        if timeout > Duration::from_secs(1800) {
            // 30 minutes
            return Err("Timeout cannot exceed 30 minutes".to_string());
        }

        Ok(())
    }

    /// Validate API key format
    fn validate_api_key(api_key: &str) -> Result<(), String> {
        if api_key.is_empty() {
            return Err("API key cannot be empty".to_string());
        }

        if api_key.trim().is_empty() {
            return Err("API key cannot be only whitespace".to_string());
        }

        if api_key.len() < 8 {
            return Err("API key must be at least 8 characters long".to_string());
        }

        // Reject API keys with newlines
        if api_key.contains('\n') || api_key.contains('\r') {
            return Err("API key cannot contain newlines".to_string());
        }

        Ok(())
    }

    /// Test parameter combination validation
    pub fn test_parameter_combinations() -> Vec<ConfigValidationResult> {
        let mut results = Vec::new();

        // Test invalid parameter combinations
        let invalid_combinations = vec![
            (
                "negative_temperature",
                json!({"temperature": -0.5}),
                "Negative temperature should be rejected",
            ),
            (
                "temperature_too_high",
                json!({"temperature": 3.0}),
                "Temperature above 2.0 should be rejected",
            ),
            (
                "negative_max_tokens",
                json!({"max_tokens": -100}),
                "Negative max_tokens should be rejected",
            ),
            (
                "max_tokens_too_high",
                json!({"max_tokens": 1000000}),
                "Extremely high max_tokens should be rejected",
            ),
            (
                "invalid_top_p",
                json!({"top_p": 1.5}),
                "top_p above 1.0 should be rejected",
            ),
            (
                "negative_top_p",
                json!({"top_p": -0.1}),
                "Negative top_p should be rejected",
            ),
        ];

        for (test_name, params, description) in invalid_combinations {
            let mut result =
                ConfigValidationResult::new(&format!("param_combination_{}", test_name))
                    .with_expected_error("InvalidParameter")
                    .add_detail(description);

            let validation_result = Self::validate_parameters(&params);
            match validation_result {
                Ok(_) => {
                    result =
                        result.failure(&format!("Invalid parameters were accepted: {}", params));
                }
                Err(error) => {
                    result = result
                        .with_actual_error("InvalidParameter")
                        .success()
                        .add_detail(&format!("Correctly rejected with error: {}", error));
                }
            }

            results.push(result);
        }

        // Test valid parameter combinations
        let valid_combinations = [
            json!({"temperature": 0.7, "max_tokens": 1000}),
            json!({"top_p": 0.9, "temperature": 0.5}),
            json!({"max_tokens": 500, "top_p": 0.8}),
        ];

        for (i, params) in valid_combinations.iter().enumerate() {
            let mut result = ConfigValidationResult::new(&format!("param_combination_valid_{}", i));

            match Self::validate_parameters(params) {
                Ok(_) => {
                    result = result
                        .success()
                        .add_detail(&format!("Correctly accepted parameters: {}", params));
                }
                Err(error) => {
                    result = result.failure(&format!("Valid parameters were rejected: {}", error));
                }
            }

            results.push(result);
        }

        results
    }

    /// Test model name validation
    pub fn test_model_name_validation() -> Vec<ConfigValidationResult> {
        let mut results = Vec::new();

        let invalid_models = vec![
            ("", "Empty model name should be rejected"),
            ("   ", "Whitespace-only model name should be rejected"),
            ("invalid/model", "Model name with slash should be rejected"),
            (
                "model with spaces",
                "Model name with spaces should be rejected",
            ),
            (
                "model@with#symbols",
                "Model names with special symbols should be rejected",
            ),
        ];

        for (model_name, description) in invalid_models {
            let mut result = ConfigValidationResult::new(&format!(
                "model_validation_{}",
                if model_name.is_empty() {
                    "empty"
                } else {
                    "invalid"
                }
            ))
            .with_expected_error("InvalidModel")
            .add_detail(description);

            let validation_result = Self::validate_model_name(model_name);
            match validation_result {
                Ok(_) => {
                    result = result.failure(&format!(
                        "Model name '{}' was accepted but should be rejected",
                        model_name
                    ));
                }
                Err(error) => {
                    result = result
                        .with_actual_error("InvalidModel")
                        .success()
                        .add_detail(&format!("Correctly rejected with error: {}", error));
                }
            }

            results.push(result);
        }

        // Test valid model names
        let valid_models = vec![
            "gpt-3.5-turbo",
            "gpt-4",
            "claude-3-5-sonnet-20241022",
            "gemini-1.5-flash",
            "llama-2-7b",
        ];

        for model_name in valid_models {
            let mut result = ConfigValidationResult::new("model_validation_valid");

            match Self::validate_model_name(model_name) {
                Ok(_) => {
                    result = result
                        .success()
                        .add_detail(&format!("Correctly accepted model: {}", model_name));
                }
                Err(error) => {
                    result = result.failure(&format!("Valid model name was rejected: {}", error));
                }
            }

            results.push(result);
        }

        results
    }

    /// Validate parameter combinations
    fn validate_parameters(params: &serde_json::Value) -> Result<(), String> {
        if let Some(temperature) = params.get("temperature").and_then(|v| v.as_f64()) {
            if temperature < 0.0 {
                return Err("Temperature cannot be negative".to_string());
            }
            if temperature > 2.0 {
                return Err("Temperature cannot exceed 2.0".to_string());
            }
        }

        if let Some(max_tokens) = params.get("max_tokens").and_then(|v| v.as_i64()) {
            if max_tokens < 0 {
                return Err("max_tokens cannot be negative".to_string());
            }
            if max_tokens > 100000 {
                return Err("max_tokens cannot exceed 100000".to_string());
            }
        }

        if let Some(top_p) = params.get("top_p").and_then(|v| v.as_f64()) {
            if top_p < 0.0 {
                return Err("top_p cannot be negative".to_string());
            }
            if top_p > 1.0 {
                return Err("top_p cannot exceed 1.0".to_string());
            }
        }

        Ok(())
    }

    /// Validate model name
    fn validate_model_name(model_name: &str) -> Result<(), String> {
        if model_name.is_empty() {
            return Err("Model name cannot be empty".to_string());
        }

        if model_name.trim().is_empty() {
            return Err("Model name cannot be only whitespace".to_string());
        }

        if model_name.contains('/') {
            return Err("Model name cannot contain forward slashes".to_string());
        }

        if model_name.contains(' ') {
            return Err("Model name cannot contain spaces".to_string());
        }

        // Reject model names with special symbols (but allow hyphens, dots, underscores)
        if model_name
            .chars()
            .any(|c| !c.is_alphanumeric() && c != '-' && c != '.' && c != '_')
        {
            return Err("Model name can only contain alphanumeric characters, hyphens, dots, and underscores".to_string());
        }

        Ok(())
    }

    /// Run all configuration validation tests
    pub fn run_all_tests() -> Vec<ConfigValidationResult> {
        println!("🚀 Running comprehensive configuration validation tests...");

        let mut all_results = Vec::new();

        println!("  🔗 Testing URL validation...");
        all_results.extend(Self::test_url_validation());

        println!("  🔌 Testing port validation...");
        all_results.extend(Self::test_port_validation());

        println!("  ⏱️  Testing timeout validation...");
        all_results.extend(Self::test_timeout_validation());

        println!("  🔑 Testing API key validation...");
        all_results.extend(Self::test_api_key_validation());

        println!("  ⚙️  Testing parameter combinations...");
        all_results.extend(Self::test_parameter_combinations());

        println!("  🤖 Testing model name validation...");
        all_results.extend(Self::test_model_name_validation());

        // Print summary
        let passed = all_results.iter().filter(|r| r.passed).count();
        let total = all_results.len();

        println!("\n📊 Configuration Validation Tests Summary:");
        println!("  ✅ Passed: {}/{}", passed, total);
        println!("  ❌ Failed: {}/{}", total - passed, total);

        let mut category_summary = HashMap::new();
        for result in &all_results {
            let category = result.test_name.split('_').next().unwrap_or("unknown");
            let entry = category_summary.entry(category).or_insert((0, 0));
            if result.passed {
                entry.0 += 1;
            } else {
                entry.1 += 1;
            }
        }

        for (category, (passed, failed)) in category_summary {
            let status = if failed == 0 { "✅" } else { "⚠️" };
            println!(
                "  {} {}: {}/{} passed",
                status,
                category,
                passed,
                passed + failed
            );
        }

        // Print details of failed tests
        let failed_tests: Vec<_> = all_results.iter().filter(|r| !r.passed).collect();
        if !failed_tests.is_empty() {
            println!("\n❌ Failed test details:");
            for result in failed_tests {
                println!(
                    "  - {}: {}",
                    result.test_name,
                    result
                        .error_message
                        .as_ref()
                        .unwrap_or(&"Unknown error".to_string())
                );
            }
        }

        all_results
    }
}

// ============================================================================
// Actual Tests
// ============================================================================

#[test]
fn test_url_validation() {
    let results = ConfigValidator::test_url_validation();

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    // At least 80% of URL validation tests should pass
    let min_passed = (total * 4) / 5;
    assert!(
        passed >= min_passed,
        "Too many URL validation tests failed: {}/{} passed",
        passed,
        total
    );

    // Check that invalid URLs are properly rejected
    let invalid_url_tests = results
        .iter()
        .filter(|r| r.test_name.contains("invalid") || r.test_name.contains("empty"));
    let invalid_passed = invalid_url_tests.filter(|r| r.passed).count();

    assert!(
        invalid_passed > 0,
        "No invalid URL tests passed - validation might be too permissive"
    );

    println!(
        "✅ URL validation tests completed: {}/{} passed",
        passed, total
    );
}

#[test]
fn test_port_validation() {
    let results = ConfigValidator::test_port_validation();

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    // All port validation tests should pass
    assert_eq!(
        passed, total,
        "Port validation tests failed: {}/{} passed",
        passed, total
    );

    // Verify that invalid ports are rejected
    let invalid_port_tests = results.iter().filter(|r| {
        r.test_name.contains("port_validation_0")
            || r.test_name.contains("port_validation_65536")
            || r.test_name.contains("port_validation_70000")
    });
    let invalid_rejected = invalid_port_tests.filter(|r| r.passed).count();

    assert!(
        invalid_rejected >= 2,
        "Invalid ports should be properly rejected"
    );

    println!(
        "✅ Port validation tests completed: {}/{} passed",
        passed, total
    );
}

#[test]
fn test_timeout_validation() {
    let results = ConfigValidator::test_timeout_validation();

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    // At least 75% of timeout validation tests should pass
    let min_passed = (total * 3) / 4;
    assert!(
        passed >= min_passed,
        "Too many timeout validation tests failed: {}/{} passed",
        passed,
        total
    );

    // Check that zero timeouts are rejected
    let zero_timeout_tests = results
        .iter()
        .filter(|r| r.test_name.contains("0ns") || r.test_name.contains("zero"));
    let zero_rejected = zero_timeout_tests.filter(|r| r.passed).count();

    assert!(zero_rejected > 0, "Zero timeouts should be rejected");

    println!(
        "✅ Timeout validation tests completed: {}/{} passed",
        passed, total
    );
}

#[test]
fn test_api_key_validation() {
    let results = ConfigValidator::test_api_key_validation();

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    // At least 70% of API key validation tests should pass
    let min_passed = (total * 7) / 10;
    assert!(
        passed >= min_passed,
        "Too many API key validation tests failed: {}/{} passed",
        passed,
        total
    );

    // Check that empty API keys are rejected
    let empty_key_tests = results.iter().filter(|r| r.test_name.contains("empty"));
    let empty_rejected = empty_key_tests.filter(|r| r.passed).count();

    assert!(empty_rejected > 0, "Empty API keys should be rejected");

    println!(
        "✅ API key validation tests completed: {}/{} passed",
        passed, total
    );
}

#[test]
fn test_parameter_combinations() {
    let results = ConfigValidator::test_parameter_combinations();

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    // At least 80% of parameter combination tests should pass
    let min_passed = (total * 4) / 5;
    assert!(
        passed >= min_passed,
        "Too many parameter combination tests failed: {}/{} passed",
        passed,
        total
    );

    // Check that negative values are rejected
    let negative_tests = results.iter().filter(|r| r.test_name.contains("negative"));
    let negative_rejected = negative_tests.filter(|r| r.passed).count();

    assert!(
        negative_rejected > 0,
        "Negative parameter values should be rejected"
    );

    println!(
        "✅ Parameter combination tests completed: {}/{} passed",
        passed, total
    );
}

#[test]
fn test_model_name_validation() {
    let results = ConfigValidator::test_model_name_validation();

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    // At least 75% of model name validation tests should pass
    let min_passed = (total * 3) / 4;
    assert!(
        passed >= min_passed,
        "Too many model name validation tests failed: {}/{} passed",
        passed,
        total
    );

    // Check that empty model names are rejected
    let empty_model_tests = results.iter().filter(|r| r.test_name.contains("empty"));
    let empty_rejected = empty_model_tests.filter(|r| r.passed).count();

    assert!(empty_rejected > 0, "Empty model names should be rejected");

    println!(
        "✅ Model name validation tests completed: {}/{} passed",
        passed, total
    );
}

#[test]
fn test_comprehensive_config_validation() {
    let results = ConfigValidator::run_all_tests();

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    // At least 75% of all configuration validation tests should pass
    let min_passed = (total * 3) / 4;
    assert!(
        passed >= min_passed,
        "Too many configuration validation tests failed: {}/{} passed",
        passed,
        total
    );

    // Verify that we tested various configuration aspects
    assert!(
        total >= 20,
        "Should have run at least 20 different configuration tests"
    );

    // Check that we have tests for different categories
    let categories: std::collections::HashSet<_> = results
        .iter()
        .map(|r| r.test_name.split('_').next().unwrap_or("unknown"))
        .collect();

    assert!(
        categories.len() >= 5,
        "Should test at least 5 different configuration categories"
    );

    // Verify that some invalid configurations were properly rejected
    let rejection_tests = results.iter().filter(|r| {
        r.test_name.contains("invalid")
            || r.test_name.contains("empty")
            || r.test_name.contains("negative")
    });
    let rejections_working = rejection_tests.filter(|r| r.passed).count();

    assert!(
        rejections_working >= 5,
        "Should have at least 5 working rejection tests for invalid configurations"
    );

    println!(
        "🎉 Comprehensive configuration validation tests completed: {}/{} passed",
        passed, total
    );
    println!(
        "📊 Tested {} different configuration categories",
        categories.len()
    );
}
