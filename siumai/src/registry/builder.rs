//! Registry builder for configuring ProviderRegistryHandle instances.
//!
//! This builder provides a fluent API for constructing a provider
//! registry with middlewares, HTTP interceptors, retry options, and
//! cache settings. It wraps `create_provider_registry` and keeps the
//! external API backward compatible.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::registry::{ProviderFactory, ProviderRegistryHandle, create_provider_registry};
use crate::retry_api::RetryOptions;
use crate::types::HttpConfig;

use super::RegistryOptions;

/// Fluent builder for constructing a provider registry.
///
/// This is a convenience wrapper around `create_provider_registry` that
/// makes it easier to configure middlewares, interceptors, retry options,
/// and cache settings without having to manually construct `RegistryOptions`.
pub struct RegistryBuilder {
    providers: HashMap<String, Arc<dyn ProviderFactory>>,
    separator: char,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    http_config: Option<HttpConfig>,
    retry_options: Option<RetryOptions>,
    max_cache_entries: Option<usize>,
    client_ttl: Option<Duration>,
    auto_middleware: bool,
}

impl RegistryBuilder {
    /// Create a new registry builder with defaults matching
    /// `create_provider_registry(providers, None)`.
    pub fn new(providers: HashMap<String, Arc<dyn ProviderFactory>>) -> Self {
        Self {
            providers,
            separator: ':',
            middlewares: Vec::new(),
            http_interceptors: Vec::new(),
            http_config: None,
            retry_options: None,
            max_cache_entries: None,
            client_ttl: None,
            auto_middleware: true,
        }
    }

    /// Set the provider/model separator (default: `':'`).
    pub fn separator(mut self, sep: char) -> Self {
        self.separator = sep;
        self
    }

    /// Add a language model middleware that applies to all models.
    pub fn with_middleware(mut self, mw: Arc<dyn LanguageModelMiddleware>) -> Self {
        self.middlewares.push(mw);
        self
    }

    /// Replace the language model middleware list.
    pub fn with_middlewares(mut self, mws: Vec<Arc<dyn LanguageModelMiddleware>>) -> Self {
        self.middlewares = mws;
        self
    }

    /// Add an HTTP interceptor applied to all clients created via the registry.
    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    /// Replace the HTTP interceptor list.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Set unified HTTP configuration applied to all clients created via the registry.
    ///
    /// This mirrors the HttpConfig used by `SiumaiBuilder` and provider builders.
    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = Some(config);
        self
    }

    /// Set request timeout for all clients created via the registry.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        let mut cfg = self.http_config.unwrap_or_else(HttpConfig::default);
        cfg.timeout = Some(timeout);
        self.http_config = Some(cfg);
        self
    }

    /// Set connection timeout for all clients created via the registry.
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        let mut cfg = self.http_config.unwrap_or_else(HttpConfig::default);
        cfg.connect_timeout = Some(timeout);
        self.http_config = Some(cfg);
        self
    }

    /// Set a custom User-Agent header for all registry-created clients.
    pub fn with_user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        let mut cfg = self.http_config.unwrap_or_else(HttpConfig::default);
        cfg.user_agent = Some(user_agent.into());
        self.http_config = Some(cfg);
        self
    }

    /// Set a proxy URL for all registry-created clients.
    pub fn with_proxy<S: Into<String>>(mut self, proxy_url: S) -> Self {
        let mut cfg = self.http_config.unwrap_or_else(HttpConfig::default);
        cfg.proxy = Some(proxy_url.into());
        self.http_config = Some(cfg);
        self
    }

    /// Add a default header that will be sent with all registry-created requests.
    pub fn with_header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        let mut cfg = self.http_config.unwrap_or_else(HttpConfig::default);
        cfg.headers.insert(key.into(), value.into());
        self.http_config = Some(cfg);
        self
    }

    /// Set unified retry options applied to all clients created via the registry.
    pub fn with_retry_options(mut self, opts: RetryOptions) -> Self {
        self.retry_options = Some(opts);
        self
    }

    /// Set the maximum number of cached clients (LRU capacity).
    pub fn with_max_cache_entries(mut self, max: usize) -> Self {
        self.max_cache_entries = Some(max);
        self
    }

    /// Set the client TTL for cached clients.
    pub fn with_client_ttl(mut self, ttl: Duration) -> Self {
        self.client_ttl = Some(ttl);
        self
    }

    /// Enable or disable automatic model-specific middlewares.
    pub fn auto_middleware(mut self, enabled: bool) -> Self {
        self.auto_middleware = enabled;
        self
    }

    /// Build the `ProviderRegistryHandle`.
    pub fn build(self) -> Result<ProviderRegistryHandle, LlmError> {
        let opts = RegistryOptions {
            separator: self.separator,
            language_model_middleware: self.middlewares,
            http_interceptors: self.http_interceptors,
            http_config: self.http_config,
            retry_options: self.retry_options,
            max_cache_entries: self.max_cache_entries,
            client_ttl: self.client_ttl,
            auto_middleware: self.auto_middleware,
        };
        Ok(create_provider_registry(self.providers, Some(opts)))
    }
}
