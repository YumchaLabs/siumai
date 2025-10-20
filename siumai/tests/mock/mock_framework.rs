//! Mock Testing Framework for LLM Providers
//!
//! This module provides a comprehensive mock testing framework that allows testing
//! LLM provider interactions without making real API calls. It supports:
//!
//! - HTTP mock server with configurable responses
//! - Error injection and network failure simulation
//! - Latency and timeout testing
//! - Rate limiting simulation
//! - Authentication failure testing
//! - Malformed response testing

use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use wiremock::{
    Mock, MockServer, Request, ResponseTemplate,
    matchers::{header, method, path, query_param},
};

/// Configuration for mock server behavior
#[derive(Debug, Clone, Default)]
pub struct MockConfig {
    /// Artificial delay to add to responses
    pub response_delay: Option<Duration>,
    /// Whether to simulate network failures
    pub simulate_network_failure: bool,
    /// Rate limit configuration
    pub rate_limit: Option<RateLimitConfig>,
    /// Custom error responses
    #[allow(dead_code)]
    pub error_responses: HashMap<String, ErrorResponse>,
    /// Whether to return malformed JSON
    pub return_malformed_json: bool,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per window
    pub max_requests: u32,
    /// Time window for rate limiting
    pub window: Duration,
    /// Current request count (shared across threads)
    pub current_count: Arc<Mutex<u32>>,
    /// Window start time
    pub window_start: Arc<Mutex<Instant>>,
}

impl RateLimitConfig {
    pub fn new(max_requests: u32, window: Duration) -> Self {
        Self {
            max_requests,
            window,
            current_count: Arc::new(Mutex::new(0)),
            window_start: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Check if request should be rate limited
    pub fn should_rate_limit(&self) -> bool {
        let mut count = self.current_count.lock().unwrap();
        let mut start = self.window_start.lock().unwrap();
        let now = Instant::now();

        // Reset window if expired
        if now.duration_since(*start) >= self.window {
            *count = 0;
            *start = now;
        }

        if *count >= self.max_requests {
            true
        } else {
            *count += 1;
            false
        }
    }
}

/// Custom error response configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ErrorResponse {
    pub status_code: u16,
    pub body: Value,
    pub headers: HashMap<String, String>,
}

#[allow(dead_code)]
impl ErrorResponse {
    pub fn new(status_code: u16, message: &str) -> Self {
        Self {
            status_code,
            body: json!({
                "error": {
                    "message": message,
                    "type": "mock_error",
                    "code": status_code
                }
            }),
            headers: HashMap::new(),
        }
    }

    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }
}

/// Mock LLM server for testing
pub struct MockLlmServer {
    server: MockServer,
    config: Arc<Mutex<MockConfig>>,
}

#[allow(dead_code)]
impl MockLlmServer {
    /// Create a new mock server
    pub async fn new() -> Self {
        let server = MockServer::start().await;
        Self {
            server,
            config: Arc::new(Mutex::new(MockConfig::default())),
        }
    }

    /// Get the base URL of the mock server
    pub fn base_url(&self) -> String {
        self.server.uri()
    }

    /// Update mock configuration
    pub fn configure(&self, config: MockConfig) {
        *self.config.lock().unwrap() = config;
    }

    /// Setup OpenAI-compatible chat endpoint
    pub async fn setup_openai_chat(&self) {
        let config = self.config.clone();

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(move |req: &Request| {
                let config = config.lock().unwrap();
                Self::handle_chat_request(req, &config)
            })
            .with_priority(1) // Lower priority than auth failures
            .mount(&self.server)
            .await;
    }

    /// Setup Anthropic-compatible messages endpoint
    pub async fn setup_anthropic_messages(&self) {
        let config = self.config.clone();

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(move |req: &Request| {
                let config = config.lock().unwrap();
                Self::handle_anthropic_request(req, &config)
            })
            .mount(&self.server)
            .await;
    }

    /// Setup Gemini-compatible endpoint
    pub async fn setup_gemini_chat(&self) {
        let config = self.config.clone();

        Mock::given(method("POST"))
            .and(path("/v1/models/gemini-1.5-flash:generateContent"))
            .respond_with(move |req: &Request| {
                let config = config.lock().unwrap();
                Self::handle_gemini_request(req, &config)
            })
            .mount(&self.server)
            .await;
    }

    /// Handle chat request with mock logic
    fn handle_chat_request(req: &Request, config: &MockConfig) -> ResponseTemplate {
        // Simulate network failure
        if config.simulate_network_failure {
            return ResponseTemplate::new(500).set_body_string("Network error");
        }

        // Check rate limiting
        if let Some(rate_limit) = &config.rate_limit
            && rate_limit.should_rate_limit()
        {
            return ResponseTemplate::new(429)
                .set_body_json(json!({
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded"
                    }
                }))
                .insert_header("retry-after", "60");
        }

        // Authentication is handled by specific mocks with higher priority

        // Check for malformed JSON response
        if config.return_malformed_json {
            return ResponseTemplate::new(200).set_body_string("{ invalid json");
        }

        // Parse request body
        let body: Value = match serde_json::from_slice(&req.body) {
            Ok(body) => body,
            Err(_) => {
                return ResponseTemplate::new(400).set_body_json(json!({
                    "error": {
                        "message": "Invalid JSON in request body",
                        "type": "invalid_request_error"
                    }
                }));
            }
        };

        // Extract model and messages
        let model = body
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or("gpt-3.5-turbo");
        let messages = body.get("messages").and_then(|m| m.as_array());

        if messages.is_none() {
            return ResponseTemplate::new(400).set_body_json(json!({
                "error": {
                    "message": "Missing 'messages' field",
                    "type": "invalid_request_error"
                }
            }));
        }

        // Create mock response
        let response = json!({
            "id": "chatcmpl-mock123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response from the test server."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 12,
                "total_tokens": 22
            }
        });

        let mut template = ResponseTemplate::new(200).set_body_json(response);

        // Add artificial delay if configured
        if let Some(delay) = config.response_delay {
            template = template.set_delay(delay);
        }

        template
    }

    /// Handle Anthropic messages request
    fn handle_anthropic_request(req: &Request, config: &MockConfig) -> ResponseTemplate {
        // Similar logic to OpenAI but with Anthropic-specific format
        if config.simulate_network_failure {
            return ResponseTemplate::new(500).set_body_string("Network error");
        }

        if !req.headers.contains_key("x-api-key") {
            return ResponseTemplate::new(401).set_body_json(json!({
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "Missing x-api-key header"
                }
            }));
        }

        let response = json!({
            "id": "msg_mock123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "This is a mock response from Anthropic test server."
            }],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 12
            }
        });

        let mut template = ResponseTemplate::new(200).set_body_json(response);
        if let Some(delay) = config.response_delay {
            template = template.set_delay(delay);
        }
        template
    }

    /// Handle Gemini request
    fn handle_gemini_request(req: &Request, config: &MockConfig) -> ResponseTemplate {
        if config.simulate_network_failure {
            return ResponseTemplate::new(500).set_body_string("Network error");
        }

        if !req.headers.contains_key("x-goog-api-key") {
            return ResponseTemplate::new(401).set_body_json(json!({
                "error": {
                    "code": 401,
                    "message": "Missing x-goog-api-key header",
                    "status": "UNAUTHENTICATED"
                }
            }));
        }

        let response = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "This is a mock response from Gemini test server."
                    }],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 12,
                "totalTokenCount": 22
            }
        });

        let mut template = ResponseTemplate::new(200).set_body_json(response);
        if let Some(delay) = config.response_delay {
            template = template.set_delay(delay);
        }
        template
    }

    /// Setup authentication failure scenarios
    pub async fn setup_auth_failures(&self) {
        // Invalid API key - higher priority than general chat endpoint
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer invalid-key"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error"
                }
            })))
            .with_priority(10) // Higher priority than general endpoint
            .mount(&self.server)
            .await;

        // Expired token - higher priority than general chat endpoint
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer expired-token"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": {
                    "message": "Token has expired",
                    "type": "authentication_error"
                }
            })))
            .with_priority(10) // Higher priority than general endpoint
            .mount(&self.server)
            .await;

        // Missing authorization header - higher priority than general chat endpoint
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(|req: &Request| !req.headers.contains_key("authorization"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": {
                    "message": "Missing authorization header",
                    "type": "authentication_error"
                }
            })))
            .with_priority(10) // Higher priority than general endpoint
            .mount(&self.server)
            .await;
    }

    /// Setup various error scenarios
    pub async fn setup_error_scenarios(&self) {
        // Model not found
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(|req: &Request| {
                if let Ok(body) = serde_json::from_slice::<Value>(&req.body) {
                    body.get("model").and_then(|m| m.as_str()) == Some("nonexistent-model")
                } else {
                    false
                }
            })
            .respond_with(ResponseTemplate::new(404).set_body_json(json!({
                "error": {
                    "message": "Model not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            })))
            .mount(&self.server)
            .await;

        // Server error
        Mock::given(method("POST"))
            .and(query_param("trigger_error", "server_error"))
            .respond_with(ResponseTemplate::new(500).set_body_json(json!({
                "error": {
                    "message": "Internal server error",
                    "type": "server_error"
                }
            })))
            .mount(&self.server)
            .await;
    }
}

/// Test utilities for mock framework
pub struct MockTestUtils;

#[allow(dead_code)]
impl MockTestUtils {
    /// Create a mock config with network failure simulation
    pub fn network_failure_config() -> MockConfig {
        MockConfig {
            simulate_network_failure: true,
            ..Default::default()
        }
    }

    /// Create a mock config with rate limiting
    pub fn rate_limited_config(max_requests: u32, window: Duration) -> MockConfig {
        MockConfig {
            rate_limit: Some(RateLimitConfig::new(max_requests, window)),
            ..Default::default()
        }
    }

    /// Create a mock config with artificial delay
    pub fn delayed_response_config(delay: Duration) -> MockConfig {
        MockConfig {
            response_delay: Some(delay),
            ..Default::default()
        }
    }

    /// Create a mock config that returns malformed JSON
    pub fn malformed_json_config() -> MockConfig {
        MockConfig {
            return_malformed_json: true,
            ..Default::default()
        }
    }

    /// Create a comprehensive test scenario config
    pub fn comprehensive_test_config() -> MockConfig {
        let mut error_responses = HashMap::new();
        error_responses.insert(
            "quota_exceeded".to_string(),
            ErrorResponse::new(429, "Quota exceeded").with_header("retry-after", "3600"),
        );
        error_responses.insert(
            "model_overloaded".to_string(),
            ErrorResponse::new(503, "Model is currently overloaded"),
        );

        MockConfig {
            response_delay: Some(Duration::from_millis(100)),
            rate_limit: Some(RateLimitConfig::new(5, Duration::from_secs(60))),
            error_responses,
            ..Default::default()
        }
    }
}

/// Mock provider factory for creating test clients
#[allow(dead_code)]
pub struct MockProviderFactory {
    server: Arc<MockLlmServer>,
}

#[allow(dead_code)]
impl MockProviderFactory {
    pub async fn new() -> Self {
        let server = Arc::new(MockLlmServer::new().await);
        Self { server }
    }

    pub fn server(&self) -> Arc<MockLlmServer> {
        self.server.clone()
    }

    /// Create a mock OpenAI client configuration
    pub async fn create_openai_config(&self) -> HashMap<String, String> {
        self.server.setup_openai_chat().await;
        self.server.setup_auth_failures().await;
        self.server.setup_error_scenarios().await;

        let mut config = HashMap::new();
        config.insert("api_key".to_string(), "test-api-key".to_string());
        config.insert("base_url".to_string(), self.server.base_url());
        config
    }

    /// Create a mock Anthropic client configuration
    pub async fn create_anthropic_config(&self) -> HashMap<String, String> {
        self.server.setup_anthropic_messages().await;
        self.server.setup_auth_failures().await;

        let mut config = HashMap::new();
        config.insert("api_key".to_string(), "test-api-key".to_string());
        config.insert("base_url".to_string(), self.server.base_url());
        config
    }

    /// Create a mock Gemini client configuration
    pub async fn create_gemini_config(&self) -> HashMap<String, String> {
        self.server.setup_gemini_chat().await;
        self.server.setup_auth_failures().await;

        let mut config = HashMap::new();
        config.insert("api_key".to_string(), "test-api-key".to_string());
        config.insert("base_url".to_string(), self.server.base_url());
        config
    }
}

/// Test scenarios for comprehensive testing
#[allow(dead_code)]
pub enum TestScenario {
    /// Normal successful operation
    Success,
    /// Network connectivity issues
    NetworkFailure,
    /// Authentication failures
    AuthFailure,
    /// Rate limiting
    RateLimit,
    /// Server errors
    ServerError,
    /// Malformed responses
    MalformedResponse,
    /// Timeout scenarios
    Timeout,
    /// Model not available
    ModelNotFound,
}

#[allow(dead_code)]
impl TestScenario {
    /// Get mock config for this scenario
    pub fn get_config(&self) -> MockConfig {
        match self {
            TestScenario::Success => MockConfig::default(),
            TestScenario::NetworkFailure => MockTestUtils::network_failure_config(),
            TestScenario::AuthFailure => MockConfig::default(), // Handled by specific endpoints
            TestScenario::RateLimit => {
                MockTestUtils::rate_limited_config(1, Duration::from_secs(60))
            }
            TestScenario::ServerError => MockConfig::default(), // Handled by specific endpoints
            TestScenario::MalformedResponse => MockTestUtils::malformed_json_config(),
            TestScenario::Timeout => {
                MockTestUtils::delayed_response_config(Duration::from_secs(30))
            }
            TestScenario::ModelNotFound => MockConfig::default(), // Handled by specific endpoints
        }
    }

    /// Get expected error type for this scenario
    pub fn expected_error_type(&self) -> Option<&'static str> {
        match self {
            TestScenario::Success => None,
            TestScenario::NetworkFailure => Some("HttpError"),
            TestScenario::AuthFailure => Some("AuthenticationError"),
            TestScenario::RateLimit => Some("RateLimitError"),
            TestScenario::ServerError => Some("ApiError"),
            TestScenario::MalformedResponse => Some("ParseError"),
            TestScenario::Timeout => Some("TimeoutError"),
            TestScenario::ModelNotFound => Some("ApiError"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_server_basic_functionality() {
        let server = MockLlmServer::new().await;
        server.setup_openai_chat().await;

        // Test that server is running and responds
        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/v1/chat/completions", server.base_url()))
            .header("authorization", "Bearer test-key")
            .json(&json!({
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), 200);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["object"], "chat.completion");
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let server = MockLlmServer::new().await;
        server.configure(MockTestUtils::rate_limited_config(
            2,
            Duration::from_secs(60),
        ));
        server.setup_openai_chat().await;

        let client = reqwest::Client::new();
        let base_url = server.base_url();

        // First two requests should succeed
        for i in 0..2 {
            let response = client
                .post(format!("{}/v1/chat/completions", base_url))
                .header("authorization", "Bearer test-key")
                .json(&json!({
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": format!("Hello {}", i)}]
                }))
                .send()
                .await
                .unwrap();

            assert_eq!(response.status(), 200, "Request {} should succeed", i);
        }

        // Third request should be rate limited
        let response = client
            .post(format!("{}/v1/chat/completions", base_url))
            .header("authorization", "Bearer test-key")
            .json(&json!({
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello 3"}]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), 429);
        assert!(response.headers().contains_key("retry-after"));
    }

    #[tokio::test]
    async fn test_authentication_failures() {
        let server = MockLlmServer::new().await;

        // Setup only auth failure mocks, no general chat endpoint
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(|req: &Request| !req.headers.contains_key("authorization"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": {
                    "message": "Missing authorization header",
                    "type": "authentication_error"
                }
            })))
            .mount(&server.server)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer invalid-key"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error"
                }
            })))
            .mount(&server.server)
            .await;

        let client = reqwest::Client::new();
        let base_url = server.base_url();

        // Test missing auth header
        let response = client
            .post(format!("{}/v1/chat/completions", base_url))
            .json(&json!({
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), 401);

        // Test invalid API key
        let response = client
            .post(format!("{}/v1/chat/completions", base_url))
            .header("authorization", "Bearer invalid-key")
            .json(&json!({
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), 401);
    }

    #[tokio::test]
    async fn test_network_failure_simulation() {
        let server = MockLlmServer::new().await;
        server.configure(MockTestUtils::network_failure_config());
        server.setup_openai_chat().await;

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/v1/chat/completions", server.base_url()))
            .header("authorization", "Bearer test-key")
            .json(&json!({
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), 500);
    }

    #[tokio::test]
    async fn test_malformed_json_response() {
        let server = MockLlmServer::new().await;
        server.configure(MockTestUtils::malformed_json_config());
        server.setup_openai_chat().await;

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/v1/chat/completions", server.base_url()))
            .header("authorization", "Bearer test-key")
            .json(&json!({
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), 200);
        let text = response.text().await.unwrap();
        assert_eq!(text, "{ invalid json");
    }

    #[tokio::test]
    async fn test_response_delay() {
        let server = MockLlmServer::new().await;
        server.configure(MockTestUtils::delayed_response_config(
            Duration::from_millis(500),
        ));
        server.setup_openai_chat().await;

        let client = reqwest::Client::new();
        let start = Instant::now();

        let response = client
            .post(format!("{}/v1/chat/completions", server.base_url()))
            .header("authorization", "Bearer test-key")
            .json(&json!({
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}]
            }))
            .send()
            .await
            .unwrap();

        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(500));
        assert_eq!(response.status(), 200);
    }
}
