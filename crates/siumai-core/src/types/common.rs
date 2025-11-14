//! Common types used across crates (subset)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// HTTP configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HttpConfig {
    #[serde(with = "duration_option_serde")]
    pub timeout: Option<Duration>,
    #[serde(with = "duration_option_serde")]
    pub connect_timeout: Option<Duration>,
    pub headers: HashMap<String, String>,
    pub proxy: Option<String>,
    pub user_agent: Option<String>,
    /// Whether to disable compression for streaming (SSE) requests.
    pub stream_disable_compression: bool,
}

/// Builder for `HttpConfig`
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
    pub fn new() -> Self {
        Self::default()
    }
    pub fn timeout(mut self, v: Option<Duration>) -> Self {
        self.timeout = v;
        self
    }
    pub fn connect_timeout(mut self, v: Option<Duration>) -> Self {
        self.connect_timeout = v;
        self
    }
    pub fn user_agent<S: Into<String>>(mut self, v: Option<S>) -> Self {
        self.user_agent = v.map(Into::into);
        self
    }
    pub fn proxy<S: Into<String>>(mut self, v: Option<S>) -> Self {
        self.proxy = v.map(Into::into);
        self
    }
    pub fn header<K: Into<String>, V: Into<String>>(mut self, k: K, v: V) -> Self {
        self.headers.insert(k.into(), v.into());
        self
    }
    pub fn headers(mut self, map: HashMap<String, String>) -> Self {
        self.headers.extend(map);
        self
    }
    pub fn stream_disable_compression(mut self, val: bool) -> Self {
        self.stream_disable_compression = Some(val);
        self
    }
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
    pub fn builder() -> HttpConfigBuilder {
        HttpConfigBuilder::new()
    }
}

// Helper module for Duration serialization
mod duration_option_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;
    pub fn serialize<S>(duration: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match duration {
            Some(d) => serializer.serialize_u64(d.as_millis() as u64),
            None => serializer.serialize_none(),
        }
    }
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Some(Duration::from_millis(u64::deserialize(deserializer)?)))
    }
}
