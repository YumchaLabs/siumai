//! HTTP configuration types.
//!
//! This module defines `HttpConfig` and its builder, used to configure HTTP
//! behavior for all providers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// HTTP response metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpResponseInfo {
    /// Timestamp when the response was received.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Model id used for the request (if known).
    #[serde(rename = "modelId", skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Response headers (lowercased keys).
    pub headers: HashMap<String, String>,
}

/// HTTP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    /// Request timeout
    #[serde(with = "duration_option_serde")]
    pub timeout: Option<Duration>,
    /// Connection timeout
    #[serde(with = "duration_option_serde")]
    pub connect_timeout: Option<Duration>,
    /// Custom headers
    pub headers: HashMap<String, String>,
    /// Proxy settings
    pub proxy: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
    /// Whether to disable compression for streaming (SSE) requests.
    ///
    /// When `true`, streaming requests explicitly set `Accept-Encoding: identity`
    /// to avoid intermediary/proxy compression which can break long-lived SSE
    /// connections. Default is `true` for stability.
    pub stream_disable_compression: bool,
}

/// Builder for `HttpConfig` to construct configuration in a unified and safe way
#[derive(Debug, Clone, Default)]
pub struct HttpConfigBuilder {
    timeout: Option<Duration>,
    connect_timeout: Option<Duration>,
    headers: HashMap<String, String>,
    proxy: Option<String>,
    user_agent: Option<String>,
    stream_disable_compression: Option<bool>,
}

impl HttpConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    pub fn timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }
    pub fn connect_timeout(mut self, connect_timeout: Option<Duration>) -> Self {
        self.connect_timeout = connect_timeout;
        self
    }
    pub fn user_agent<S: Into<String>>(mut self, user_agent: Option<S>) -> Self {
        self.user_agent = user_agent.map(|s| s.into());
        self
    }
    pub fn proxy<S: Into<String>>(mut self, proxy: Option<S>) -> Self {
        self.proxy = proxy.map(|s| s.into());
        self
    }
    pub fn header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }
    pub fn headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers.extend(headers);
        self
    }
    pub fn stream_disable_compression(mut self, val: bool) -> Self {
        self.stream_disable_compression = Some(val);
        self
    }

    /// Build the configuration
    pub fn build(self) -> HttpConfig {
        let default_sdc = HttpConfig::default().stream_disable_compression;
        HttpConfig {
            timeout: self.timeout,
            connect_timeout: self.connect_timeout,
            headers: self.headers,
            proxy: self.proxy,
            user_agent: self.user_agent,
            stream_disable_compression: self.stream_disable_compression.unwrap_or(default_sdc),
        }
    }
}

impl HttpConfig {
    /// Returns a builder for constructing `HttpConfig`
    pub fn builder() -> HttpConfigBuilder {
        HttpConfigBuilder::new()
    }
}

// Helper module for Duration serialization
mod duration_option_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match duration {
            Some(d) => d.as_secs().serialize(serializer),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs: Option<u64> = Option::deserialize(deserializer)?;
        Ok(secs.map(Duration::from_secs))
    }
}

impl Default for HttpConfig {
    fn default() -> Self {
        // Determine default for stream_disable_compression from env var (default: true)
        let sdc = match std::env::var("SIUMAI_STREAM_DISABLE_COMPRESSION") {
            Ok(val) => {
                let v = val.trim().to_lowercase();
                !(v == "false" || v == "0" || v == "off" || v == "no")
            }
            Err(_) => true,
        };
        Self {
            timeout: Some(crate::defaults::http::REQUEST_TIMEOUT),
            connect_timeout: Some(crate::defaults::http::CONNECT_TIMEOUT),
            headers: HashMap::new(),
            proxy: None,
            user_agent: Some(crate::defaults::http::USER_AGENT.to_string()),
            stream_disable_compression: sdc,
        }
    }
}
