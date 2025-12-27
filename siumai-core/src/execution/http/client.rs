//! HTTP client builder utilities
//!
//! This module provides unified HTTP client construction logic to avoid code duplication
//! across different providers and builders.

use crate::error::LlmError;
use crate::types::HttpConfig;

/// Build an HTTP client from HttpConfig
///
/// This is a unified function used by all providers and builders to construct
/// reqwest::Client instances with consistent configuration.
///
/// # Arguments
/// * `config` - HTTP configuration containing timeout, proxy, headers, etc.
///
/// # Returns
/// * `Ok(reqwest::Client)` - Configured HTTP client
/// * `Err(LlmError)` - Configuration or build error
///
/// # Example
/// ```rust,no_run
/// use siumai::types::HttpConfig;
/// use siumai::execution::http::client::build_http_client_from_config;
///
/// let config = HttpConfig::default();
/// let client = build_http_client_from_config(&config)?;
/// # Ok::<(), siumai::LlmError>(())
/// ```
pub fn build_http_client_from_config(config: &HttpConfig) -> Result<reqwest::Client, LlmError> {
    let mut builder = reqwest::Client::builder();

    // Apply timeout settings
    if let Some(timeout) = config.timeout {
        builder = builder.timeout(timeout);
    }

    if let Some(connect_timeout) = config.connect_timeout {
        builder = builder.connect_timeout(connect_timeout);
    }

    // Apply proxy settings
    if let Some(proxy_url) = &config.proxy {
        let proxy = reqwest::Proxy::all(proxy_url)
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid proxy URL: {e}")))?;
        builder = builder.proxy(proxy);
    }

    // Apply user agent
    if let Some(user_agent) = &config.user_agent {
        builder = builder.user_agent(user_agent);
    }

    // Apply default headers
    if !config.headers.is_empty() {
        let mut headers = reqwest::header::HeaderMap::new();
        for (k, v) in &config.headers {
            let name = reqwest::header::HeaderName::from_bytes(k.as_bytes()).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid header name '{k}': {e}"))
            })?;
            let value = reqwest::header::HeaderValue::from_str(v).map_err(|e| {
                LlmError::ConfigurationError(format!("Invalid header value for '{k}': {e}"))
            })?;
            headers.insert(name, value);
        }
        builder = builder.default_headers(headers);
    }

    // Build the client
    builder
        .build()
        .map_err(|e| LlmError::HttpError(format!("Failed to create HTTP client: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_build_http_client_default() {
        let config = HttpConfig::default();
        let result = build_http_client_from_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_http_client_with_timeout() {
        let config = HttpConfig {
            timeout: Some(Duration::from_secs(30)),
            connect_timeout: Some(Duration::from_secs(10)),
            ..Default::default()
        };

        let result = build_http_client_from_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_http_client_with_user_agent() {
        let config = HttpConfig {
            user_agent: Some("test-agent/1.0".to_string()),
            ..Default::default()
        };

        let result = build_http_client_from_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_http_client_with_headers() {
        let mut config = HttpConfig::default();
        config
            .headers
            .insert("X-Custom-Header".to_string(), "custom-value".to_string());

        let result = build_http_client_from_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_http_client_with_invalid_header_name() {
        let mut config = HttpConfig::default();
        config
            .headers
            .insert("Invalid Header Name".to_string(), "value".to_string());

        let result = build_http_client_from_config(&config);
        assert!(result.is_err());
    }
}
