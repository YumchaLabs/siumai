//! Provider Builder Core
//!
//! This module provides shared functionality for all provider builders using composition.
//! Following the principle "composition over inheritance", this module extracts common
//! builder fields and methods into a reusable `ProviderCore` structure.
//!
//! # Design Philosophy
//!
//! Instead of inheritance, we use composition:
//! - Each provider builder contains a `ProviderCore` instance
//! - `ProviderCore` handles all common HTTP/tracing/retry/interceptor configuration
//! - Provider builders focus only on provider-specific parameters
//!
//! # Benefits
//!
//! - **Code Reuse**: ~500 lines of duplicate code eliminated
//! - **Consistency**: All providers behave identically for common features
//! - **Maintainability**: Changes to common behavior only need to be made once
//! - **Extensibility**: New providers can be added with minimal boilerplate

use crate::builder::LlmBuilder;
use crate::error::LlmError;
use crate::execution::http::interceptor::{
    HttpInterceptor, HttpTracingInterceptor, LoggingInterceptor,
};
use crate::observability::tracing::TracingConfig;
use crate::retry_api::RetryOptions;
use crate::types::HttpConfig;
use std::sync::Arc;
use std::time::Duration;

/// Core configuration shared by all provider builders
///
/// This structure contains all the common fields that every provider builder needs:
/// - HTTP configuration (timeouts, client, headers)
/// - Tracing configuration
/// - Retry options
/// - HTTP interceptors
///
/// # Example
///
/// ```rust,ignore
/// pub struct OpenAiBuilder {
///     core: ProviderCore,
///     api_key: Option<String>,
///     model: Option<String>,
///     // ... provider-specific fields
/// }
///
/// impl OpenAiBuilder {
///     pub fn new(base: LlmBuilder) -> Self {
///         Self {
///             core: ProviderCore::new(base),
///             api_key: None,
///             model: None,
///         }
///     }
///
///     // Delegate common methods to core
///     pub fn timeout(mut self, timeout: Duration) -> Self {
///         self.core = self.core.timeout(timeout);
///         self
///     }
/// }
/// ```
#[derive(Clone)]
pub struct ProviderCore {
    /// Base builder with HTTP client configuration
    pub base: LlmBuilder,
    /// HTTP configuration (timeout, connect_timeout, headers, proxy, etc.)
    pub http_config: HttpConfig,
    /// Tracing configuration
    pub tracing_config: Option<TracingConfig>,
    /// Retry options for chat operations
    pub retry_options: Option<RetryOptions>,
    /// HTTP interceptors applied to requests
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Enable built-in logging interceptor for debugging
    pub http_debug: bool,
}

impl ProviderCore {
    /// Create a new ProviderCore from a base LlmBuilder
    ///
    /// This automatically inherits interceptors and debug settings from the base builder.
    pub fn new(base: LlmBuilder) -> Self {
        let inherited_interceptors = base.http_interceptors.clone();
        let inherited_debug = base.http_debug;

        // Derive initial HttpConfig from the base LlmBuilder so that
        // LlmBuilder-level HTTP settings (timeout, proxy, headers, UA)
        // are honored by provider builders. ProviderCore-specific methods
        // (timeout/connect_timeout/headers/...) can then override these
        // values per-provider.
        let mut http_config = HttpConfig::default();
        if let Some(timeout) = base.timeout {
            http_config.timeout = Some(timeout);
        }
        if let Some(connect_timeout) = base.connect_timeout {
            http_config.connect_timeout = Some(connect_timeout);
        }
        if let Some(ref proxy) = base.proxy {
            http_config.proxy = Some(proxy.clone());
        }
        if let Some(ref ua) = base.user_agent {
            http_config.user_agent = Some(ua.clone());
        }
        if !base.default_headers.is_empty() {
            http_config.headers.extend(base.default_headers.clone());
        }

        Self {
            base,
            http_config,
            tracing_config: None,
            retry_options: None,
            http_interceptors: inherited_interceptors,
            http_debug: inherited_debug,
        }
    }

    // ========================================================================
    // HTTP Configuration Methods
    // ========================================================================

    /// Set request timeout
    ///
    /// This timeout applies to the entire request, including connection establishment,
    /// sending the request, and receiving the response.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }

    /// Set connection timeout
    ///
    /// This timeout only applies to establishing the TCP connection.
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }

    /// Set custom HTTP client
    ///
    /// This allows you to provide a pre-configured reqwest::Client with custom settings.
    /// The client will be used for all HTTP requests made by this provider.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.base = self.base.with_http_client(client);
        self
    }

    /// Control whether to disable compression for streaming (SSE) requests
    ///
    /// When `true`, streaming requests explicitly set `Accept-Encoding: identity`
    /// to avoid intermediary/proxy compression which can break long-lived SSE connections.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
        self
    }

    // ========================================================================
    // Tracing Configuration Methods
    // ========================================================================

    /// Set custom tracing configuration
    pub fn tracing(mut self, config: TracingConfig) -> Self {
        self.tracing_config = Some(config);
        self
    }

    /// Enable debug tracing (development-friendly configuration)
    ///
    /// This enables detailed logging suitable for development and debugging.
    pub fn debug_tracing(self) -> Self {
        self.tracing(TracingConfig::development())
    }

    /// Enable minimal tracing (info level, LLM only)
    ///
    /// This enables minimal logging suitable for production environments.
    pub fn minimal_tracing(self) -> Self {
        self.tracing(TracingConfig::minimal())
    }

    /// Enable production-ready JSON tracing
    ///
    /// This enables structured JSON logging suitable for production log aggregation.
    pub fn json_tracing(self) -> Self {
        self.tracing(TracingConfig::json_production())
    }

    /// Enable pretty-printed formatting for JSON bodies and headers in tracing
    ///
    /// This makes JSON output more readable but increases log size.
    pub fn pretty_json(mut self, pretty: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(TracingConfig::development)
            .with_pretty_json(pretty);
        self.tracing_config = Some(config);
        self
    }

    /// Control masking of sensitive values (API keys, tokens) in tracing logs
    ///
    /// When `true` (default), sensitive values are masked as "sk-1...cdef".
    /// Set to `false` only in secure development environments.
    pub fn mask_sensitive_values(mut self, mask: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(TracingConfig::development)
            .with_mask_sensitive_values(mask);
        self.tracing_config = Some(config);
        self
    }

    // ========================================================================
    // Retry Configuration Methods
    // ========================================================================

    /// Set unified retry options for chat operations
    ///
    /// This enables automatic retries with exponential backoff for transient failures.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use siumai::retry_api::RetryOptions;
    ///
    /// let builder = builder.with_retry(RetryOptions::backoff());
    /// ```
    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.retry_options = Some(options);
        self
    }

    // ========================================================================
    // HTTP Interceptor Methods
    // ========================================================================

    /// Add a custom HTTP interceptor
    ///
    /// Interceptors can observe and modify HTTP requests and responses.
    /// Multiple interceptors can be added and will be executed in order.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::sync::Arc;
    /// use siumai::execution::http::interceptor::LoggingInterceptor;
    ///
    /// let builder = builder.with_http_interceptor(Arc::new(LoggingInterceptor));
    /// ```
    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    /// Enable a built-in logging interceptor for HTTP debugging
    ///
    /// This is a convenience method that adds a `LoggingInterceptor` which logs
    /// HTTP requests and responses without exposing sensitive data.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let builder = builder.http_debug(true);
    /// ```
    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.http_debug = enabled;
        self
    }

    // ========================================================================
    // Build Helper Methods
    // ========================================================================

    /// Build the HTTP client from configuration
    ///
    /// This method creates a reqwest::Client using either:
    /// 1. The custom client provided via `with_http_client()`, or
    /// 2. A new client built from `http_config` settings
    ///
    /// This is used internally by provider builders during the build process.
    pub fn build_http_client(&self) -> Result<reqwest::Client, LlmError> {
        // Prefer a custom client when provided at the base builder level.
        if let Some(client) = &self.base.http_client {
            return Ok(client.clone());
        }

        // Otherwise, build from the effective HttpConfig derived from
        // LlmBuilder + ProviderCore overrides.
        crate::execution::http::client::build_http_client_from_config(&self.http_config)
    }

    /// Get HTTP interceptors including debug interceptor if enabled
    ///
    /// This method returns all configured HTTP interceptors, automatically
    /// adding a `LoggingInterceptor` if `http_debug` is enabled.
    pub fn get_http_interceptors(&self) -> Vec<Arc<dyn HttpInterceptor>> {
        let mut interceptors = self.http_interceptors.clone();
        if let Some(cfg) = &self.tracing_config
            && cfg.enable_http_tracing
            && cfg.sampling_rate > 0.0
        {
            interceptors.push(Arc::new(HttpTracingInterceptor::new(cfg.clone())));
        }
        if self.http_debug {
            interceptors.push(Arc::new(LoggingInterceptor));
        }
        interceptors
    }

    /// Get automatic middlewares for a provider and model
    ///
    /// This method returns the automatically configured middlewares based on
    /// the provider and model identifiers.
    pub fn get_auto_middlewares(
        &self,
        provider_id: &str,
        model_id: &str,
    ) -> Vec<Arc<dyn crate::execution::middleware::LanguageModelMiddleware>> {
        crate::execution::middleware::build_auto_middlewares_vec(provider_id, model_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_core_creation() {
        let base = LlmBuilder::new();
        let core = ProviderCore::new(base);

        assert!(core.tracing_config.is_none());
        assert!(core.retry_options.is_none());
        assert!(core.http_interceptors.is_empty());
        assert!(!core.http_debug);
    }

    #[test]
    fn test_provider_core_timeout_configuration() {
        let base = LlmBuilder::new();
        let core = ProviderCore::new(base)
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10));

        assert_eq!(core.http_config.timeout, Some(Duration::from_secs(30)));
        assert_eq!(
            core.http_config.connect_timeout,
            Some(Duration::from_secs(10))
        );
    }

    #[test]
    fn test_provider_core_tracing_configuration() {
        let base = LlmBuilder::new();
        let core = ProviderCore::new(base)
            .debug_tracing()
            .pretty_json(true)
            .mask_sensitive_values(false);

        assert!(core.tracing_config.is_some());
        let config = core.tracing_config.unwrap();
        assert!(config.pretty_json);
        assert!(!config.mask_sensitive_values);
    }

    #[test]
    fn test_provider_core_http_debug() {
        let base = LlmBuilder::new();
        let core = ProviderCore::new(base).http_debug(true);

        assert!(core.http_debug);
    }

    #[test]
    fn test_provider_core_inherits_from_base() {
        let base = LlmBuilder::new().http_debug(true);
        let core = ProviderCore::new(base);

        // Should inherit http_debug from base
        assert!(core.http_debug);
    }
}
