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

/// Convert reqwest HeaderMap to HashMap<String, String>
///
/// This is a utility function to convert HTTP headers from reqwest's HeaderMap
/// format to a standard HashMap. Invalid UTF-8 header values are filtered out.
///
/// # Arguments
/// * `headers` - The reqwest HeaderMap to convert
///
/// # Returns
/// A HashMap containing all valid UTF-8 headers as String key-value pairs
///
/// # Example
/// ```rust,ignore
/// use reqwest::header::HeaderMap;
/// use siumai::experimental::execution::http::headers::headermap_to_hashmap;
///
/// let headers = HeaderMap::new();
/// let map = headermap_to_hashmap(&headers);
/// ```
pub fn headermap_to_hashmap(headers: &HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(k, v)| {
            v.to_str()
                .ok()
                .map(|v_str| (k.as_str().to_string(), v_str.to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};
    static _TRACING_TOGGLE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

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
    fn merge_headers_overrides_existing_values() {
        let mut base = HeaderMap::new();
        base.insert(
            HeaderName::from_bytes(b"anthropic-beta").unwrap(),
            HeaderValue::from_str("a,b").unwrap(),
        );

        let mut extra = HashMap::new();
        extra.insert("Anthropic-Beta".to_string(), "c".to_string());

        let merged = merge_headers(base, &extra);
        let value = merged
            .get("anthropic-beta")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert_eq!(value, "c");
    }
}
