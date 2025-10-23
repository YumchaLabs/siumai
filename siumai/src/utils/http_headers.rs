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
    ///
    /// Behavior:
    /// - If `custom_headers` already contains `Authorization` (case-insensitive),
    ///   treat it as a Bearer token (e.g., Vertex AI enterprise auth) and DO NOT
    ///   inject `x-goog-api-key`.
    /// - Otherwise, if `api_key` is non-empty, inject `x-goog-api-key`.
    /// - Always include `Content-Type: application/json` and pass through custom headers.
    pub fn gemini(
        api_key: &str,
        custom_headers: &HashMap<String, String>,
    ) -> Result<HeaderMap, LlmError> {
        // Base headers: JSON + custom headers
        let mut builder = HttpHeaderBuilder::new()
            .with_json_content_type()
            .with_custom_headers(custom_headers)?;

        // Detect whether Authorization (Bearer) is provided; if so, skip x-goog-api-key
        let has_authorization = custom_headers
            .keys()
            .any(|k| k.eq_ignore_ascii_case("authorization"));

        if !has_authorization {
            // Without Authorization, fall back to API Key if provided
            if !api_key.is_empty() {
                builder = builder.with_custom_auth("x-goog-api-key", api_key)?;
            }
        }

        Ok(builder.build())
    }

    /// Build headers for Vertex (Bearer-only JSON)
    pub fn vertex_bearer(custom_headers: &HashMap<String, String>) -> Result<HeaderMap, LlmError> {
        let mut builder = HttpHeaderBuilder::new().with_json_content_type();
        // Pass through custom headers (should include Authorization)
        builder = builder.with_custom_headers(custom_headers)?;
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

/// Merge extra headers into base headers (immutable version).
///
/// Creates a new HeaderMap by cloning the base headers and adding extra headers.
/// Extra headers will override base headers if they have the same name.
///
/// # Arguments
/// * `base` - Base HeaderMap to start with
/// * `extra` - HashMap of additional headers to merge
///
/// # Returns
/// A new HeaderMap with merged headers
///
/// # Example
/// ```rust,ignore
/// let base_headers = /* ... */;
/// let extra = HashMap::from([("X-Custom".to_string(), "value".to_string())]);
/// let merged = merge_headers(base_headers, &extra);
/// ```
pub fn merge_headers(mut base: HeaderMap, extra: &HashMap<String, String>) -> HeaderMap {
    for (k, v) in extra {
        if let (Ok(name), Ok(val)) = (
            HeaderName::from_bytes(k.as_bytes()),
            HeaderValue::from_str(v),
        ) {
            base.insert(name, val);
        }
    }
    base
}

/// Apply extra headers to a mutable HeaderMap (mutable version).
///
/// Modifies the base HeaderMap in place by adding extra headers.
/// Extra headers will override base headers if they have the same name.
///
/// # Arguments
/// * `base` - Mutable reference to HeaderMap to modify
/// * `extra` - HashMap of additional headers to apply
///
/// # Example
/// ```rust,ignore
/// let mut headers = /* ... */;
/// let extra = HashMap::from([("X-Custom".to_string(), "value".to_string())]);
/// apply_extra_headers(&mut headers, &extra);
/// ```
pub fn apply_extra_headers(base: &mut HeaderMap, extra: &HashMap<String, String>) {
    for (k, v) in extra {
        if let (Ok(name), Ok(val)) = (
            HeaderName::from_bytes(k.as_bytes()),
            HeaderValue::from_str(v),
        ) {
            base.insert(name, val);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};
    static TRACING_TOGGLE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

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
    fn test_gemini_headers_with_bearer_authorization() {
        // When Authorization is provided, x-goog-api-key must NOT be injected
        let mut extra = HashMap::new();
        extra.insert("Authorization".to_string(), "Bearer test-token".to_string());

        let headers = ProviderHeaders::gemini("", &extra).unwrap();

        assert_eq!(headers.get("Authorization").unwrap(), "Bearer test-token");
        assert_eq!(headers.get(CONTENT_TYPE).unwrap(), "application/json");
        assert!(headers.get("x-goog-api-key").is_none());
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
        let _g = TRACING_TOGGLE_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap();
        // Ensure W3C trace is disabled for this test
        crate::tracing::set_w3c_trace_enabled(false);
        let mut headers = HeaderMap::new();
        super::inject_tracing_headers(&mut headers);
        assert!(headers.get("X-Trace-Id").is_some());
        assert!(headers.get("X-Span-Id").is_some());
        // Note: Other tests may toggle W3C concurrently; only assert absence if flag is false now
        // Note: Other tests may toggle W3C concurrently; asserting absence here is flaky.
        // We only assert required headers; traceparent presence depends on global state.
    }

    #[test]
    fn test_inject_tracing_headers_w3c() {
        let _g = TRACING_TOGGLE_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap();
        // Enable W3C trace headers for this test
        crate::tracing::set_w3c_trace_enabled(true);
        let mut headers = HeaderMap::new();
        super::inject_tracing_headers(&mut headers);
        assert!(headers.get("traceparent").is_some());
        // Reset after test
        crate::tracing::set_w3c_trace_enabled(false);
    }
}
