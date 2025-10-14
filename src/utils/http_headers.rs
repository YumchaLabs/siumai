//! HTTP Headers Utility
//!
//! Common utilities for building HTTP headers across all providers.

use crate::error::LlmError;
use reqwest::header::{
    AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue, USER_AGENT,
};
use std::collections::HashMap;

/// HTTP header builder for API requests
pub struct HttpHeaderBuilder {
    headers: HeaderMap,
}

impl HttpHeaderBuilder {
    /// Create a new header builder
    pub fn new() -> Self {
        Self {
            headers: HeaderMap::new(),
        }
    }

    /// Add Bearer token authorization
    pub fn with_bearer_auth(mut self, token: &str) -> Result<Self, LlmError> {
        let auth_value = format!("Bearer {token}");
        self.headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&auth_value).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid API key format: {e}"))
            })?,
        );
        Ok(self)
    }

    /// Add custom authorization header (e.g., x-api-key for Anthropic)
    pub fn with_custom_auth(mut self, header_name: &str, value: &str) -> Result<Self, LlmError> {
        let header_name = HeaderName::from_bytes(header_name.as_bytes()).map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid header name '{header_name}': {e}"))
        })?;
        self.headers.insert(
            header_name,
            HeaderValue::from_str(value)
                .map_err(|e| LlmError::ConfigurationError(format!("Invalid header value: {e}")))?,
        );
        Ok(self)
    }

    /// Add JSON content type
    pub fn with_json_content_type(mut self) -> Self {
        self.headers
            .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        self
    }

    /// Add user agent
    pub fn with_user_agent(mut self, user_agent: &str) -> Result<Self, LlmError> {
        self.headers.insert(
            USER_AGENT,
            HeaderValue::from_str(user_agent)
                .map_err(|e| LlmError::ConfigurationError(format!("Invalid user agent: {e}")))?,
        );
        Ok(self)
    }

    /// Add a custom header
    pub fn with_header(mut self, name: &str, value: &str) -> Result<Self, LlmError> {
        let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
            LlmError::ConfigurationError(format!("Invalid header name '{name}': {e}"))
        })?;
        self.headers.insert(
            header_name,
            HeaderValue::from_str(value).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid header value '{value}': {e}"))
            })?,
        );
        Ok(self)
    }

    /// Add multiple custom headers from a HashMap
    pub fn with_custom_headers(
        mut self,
        custom_headers: &HashMap<String, String>,
    ) -> Result<Self, LlmError> {
        for (key, value) in custom_headers {
            let header_name = HeaderName::from_bytes(key.as_bytes()).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid header name '{key}': {e}"))
            })?;
            self.headers.insert(
                header_name,
                HeaderValue::from_str(value).map_err(|e| {
                    LlmError::ConfigurationError(format!("Invalid header value '{value}': {e}"))
                })?,
            );
        }
        Ok(self)
    }

    /// Build the final HeaderMap
    pub fn build(self) -> HeaderMap {
        self.headers
    }
}

impl Default for HttpHeaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Provider-specific header builders
pub struct ProviderHeaders;

impl ProviderHeaders {
    /// Build headers for OpenAI API
    pub fn openai(
        api_key: &str,
        organization: Option<&str>,
        project: Option<&str>,
        custom_headers: &HashMap<String, String>,
    ) -> Result<HeaderMap, LlmError> {
        let mut builder = HttpHeaderBuilder::new()
            .with_bearer_auth(api_key)?
            .with_json_content_type();

        // Add OpenAI-specific headers
        if let Some(org) = organization {
            builder = builder.with_header("OpenAI-Organization", org)?;
        }

        if let Some(proj) = project {
            builder = builder.with_header("OpenAI-Project", proj)?;
        }

        builder = builder.with_custom_headers(custom_headers)?;
        Ok(builder.build())
    }

    /// Build headers for Anthropic API
    pub fn anthropic(
        api_key: &str,
        custom_headers: &HashMap<String, String>,
    ) -> Result<HeaderMap, LlmError> {
        let mut builder = HttpHeaderBuilder::new()
            .with_custom_auth("x-api-key", api_key)?
            .with_json_content_type()
            .with_header("anthropic-version", "2023-06-01")?;

        // Handle anthropic-beta header specially
        if let Some(beta_features) = custom_headers.get("anthropic-beta") {
            builder = builder.with_header("anthropic-beta", beta_features)?;
        }

        // Add other custom headers (excluding anthropic-beta which we handled above)
        let filtered_headers: HashMap<String, String> = custom_headers
            .iter()
            .filter(|(k, _)| k.as_str() != "anthropic-beta")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        builder = builder.with_custom_headers(&filtered_headers)?;
        Ok(builder.build())
    }

    /// Build headers for Groq API
    pub fn groq(
        api_key: &str,
        custom_headers: &HashMap<String, String>,
    ) -> Result<HeaderMap, LlmError> {
        let builder = HttpHeaderBuilder::new()
            .with_bearer_auth(api_key)?
            .with_json_content_type()
            .with_user_agent("siumai/0.1.0 (groq-provider)")?
            .with_custom_headers(custom_headers)?;

        Ok(builder.build())
    }

    /// Build headers for xAI API
    pub fn xai(
        api_key: &str,
        custom_headers: &HashMap<String, String>,
    ) -> Result<HeaderMap, LlmError> {
        let builder = HttpHeaderBuilder::new()
            .with_bearer_auth(api_key)?
            .with_json_content_type()
            .with_custom_headers(custom_headers)?;

        Ok(builder.build())
    }

    /// Build headers for Ollama API (no auth required)
    pub fn ollama(custom_headers: &HashMap<String, String>) -> Result<HeaderMap, LlmError> {
        let version = env!("CARGO_PKG_VERSION");
        let builder = HttpHeaderBuilder::new()
            .with_json_content_type()
            .with_user_agent(&format!("siumai/{version}"))?
            .with_custom_headers(custom_headers)?;

        Ok(builder.build())
    }

    /// Build headers for Gemini API
    pub fn gemini(
        api_key: &str,
        custom_headers: &HashMap<String, String>,
    ) -> Result<HeaderMap, LlmError> {
        let builder = HttpHeaderBuilder::new()
            .with_custom_auth("x-goog-api-key", api_key)?
            .with_json_content_type()
            .with_custom_headers(custom_headers)?;

        Ok(builder.build())
    }
}

/// Inject tracing headers into a HeaderMap.
/// Always injects `X-Trace-Id` and `X-Span-Id`.
/// If W3C trace is enabled (via env or config), also injects `traceparent`.
pub fn inject_tracing_headers(headers: &mut HeaderMap) {
    let tid = crate::tracing::TraceId::new().to_string();
    let sid = crate::tracing::SpanId::new().to_string();
    if let Ok(v) = HeaderValue::from_str(&tid) {
        let _ = headers.insert("X-Trace-Id", v);
    }
    if let Ok(v) = HeaderValue::from_str(&sid) {
        let _ = headers.insert("X-Span-Id", v);
    }
    if crate::tracing::w3c_trace_enabled() {
        let tp = crate::tracing::create_w3c_traceparent();
        if let Ok(v) = HeaderValue::from_str(&tp) {
            let _ = headers.insert("traceparent", v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_builder() {
        let headers = HttpHeaderBuilder::new()
            .with_bearer_auth("test-token")
            .unwrap()
            .with_json_content_type()
            .with_user_agent("test-agent")
            .unwrap()
            .build();

        assert_eq!(headers.get(AUTHORIZATION).unwrap(), "Bearer test-token");
        assert_eq!(headers.get(CONTENT_TYPE).unwrap(), "application/json");
        assert_eq!(headers.get(USER_AGENT).unwrap(), "test-agent");
    }

    #[test]
    fn test_openai_headers() {
        let custom_headers = HashMap::new();
        let headers =
            ProviderHeaders::openai("test-key", Some("org"), Some("proj"), &custom_headers)
                .unwrap();

        assert_eq!(headers.get(AUTHORIZATION).unwrap(), "Bearer test-key");
        assert_eq!(headers.get("OpenAI-Organization").unwrap(), "org");
        assert_eq!(headers.get("OpenAI-Project").unwrap(), "proj");
    }

    #[test]
    fn test_anthropic_headers() {
        let custom_headers = HashMap::new();
        let headers = ProviderHeaders::anthropic("test-key", &custom_headers).unwrap();

        assert_eq!(headers.get("x-api-key").unwrap(), "test-key");
        assert_eq!(headers.get("anthropic-version").unwrap(), "2023-06-01");
    }

    #[test]
    fn test_inject_tracing_headers_basic() {
        // Ensure W3C trace is disabled for this test (tests run in parallel)
        crate::tracing::set_w3c_trace_enabled(false);
        let mut headers = HeaderMap::new();
        super::inject_tracing_headers(&mut headers);
        assert!(headers.get("X-Trace-Id").is_some());
        assert!(headers.get("X-Span-Id").is_some());
        // Note: Other tests may toggle W3C concurrently; only assert absence if flag is false now
        if !crate::tracing::w3c_trace_enabled() {
            assert!(headers.get("traceparent").is_none());
        }
    }

    #[test]
    fn test_inject_tracing_headers_w3c() {
        // Enable W3C trace headers for this test
        crate::tracing::set_w3c_trace_enabled(true);
        let mut headers = HeaderMap::new();
        super::inject_tracing_headers(&mut headers);
        assert!(headers.get("traceparent").is_some());
        // Reset after test
        crate::tracing::set_w3c_trace_enabled(false);
    }
}
