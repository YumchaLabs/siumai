//! Builder base + provider builder core utilities.
//!
//! This module is provider-agnostic and exists to enable provider implementations to live
//! in separate crates while still sharing a consistent, ergonomic builder experience.
//!
//! Notes:
//! - The user-facing unified builder lives in `siumai-registry` (`provider::SiumaiBuilder`) and is
//!   re-exported by the `siumai` facade as `Siumai::builder()`.
//! - Provider builders should consume `BuilderBase` (a provider-agnostic snapshot of unified HTTP
//!   config/interceptor settings) to avoid circular dependencies.

use crate::error::LlmError;
use crate::execution::http::interceptor::{
    HttpInterceptor, HttpTracingInterceptor, LoggingInterceptor,
};
use crate::execution::http::transport::HttpTransport;
use crate::observability::tracing::TracingConfig;
use crate::retry_api::RetryOptions;
use crate::types::HttpConfig;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// A provider-agnostic snapshot of the user-facing unified builder.
///
/// Provider crates use this type to inherit common HTTP/interceptor/debug configuration
/// without depending on the facade or registry crates.
#[derive(Clone, Default)]
pub struct BuilderBase {
    /// Custom reqwest client (takes precedence over all other HTTP settings).
    pub http_client: Option<reqwest::Client>,
    /// Request timeout.
    pub timeout: Option<Duration>,
    /// Connection timeout.
    pub connect_timeout: Option<Duration>,
    /// User-Agent header value.
    pub user_agent: Option<String>,
    /// Default headers applied to all requests (header-name -> value).
    pub default_headers: HashMap<String, String>,
    /// Optional HTTP interceptors applied to requests.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Enable lightweight HTTP debug logging interceptor.
    pub http_debug: bool,
    /// Proxy URL.
    pub proxy: Option<String>,
}

/// Provider builder core (composition over inheritance).
///
/// This is shared by provider builders to ensure consistent behavior for:
/// - HTTP configuration / client construction
/// - tracing
/// - retry options
/// - interceptors and debug logging
#[derive(Clone)]
pub struct ProviderCore {
    /// The inherited base configuration from the unified `LlmBuilder`.
    pub base: BuilderBase,
    /// Effective HTTP config for this provider builder.
    pub http_config: HttpConfig,
    /// Tracing configuration.
    pub tracing_config: Option<TracingConfig>,
    /// Retry options for chat operations.
    pub retry_options: Option<RetryOptions>,
    /// Optional custom HTTP transport.
    pub http_transport: Option<Arc<dyn HttpTransport>>,
    /// HTTP interceptors applied to requests.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Enable built-in logging interceptor for debugging.
    pub http_debug: bool,
}

impl ProviderCore {
    /// Create a new ProviderCore from a base config snapshot.
    pub fn new(base: BuilderBase) -> Self {
        let inherited_interceptors = base.http_interceptors.clone();
        let inherited_debug = base.http_debug;

        // Derive initial HttpConfig from the base builder so that global HTTP settings
        // (timeout, proxy, headers, UA) are honored by provider builders.
        let mut http_config = HttpConfig {
            timeout: base.timeout,
            connect_timeout: base.connect_timeout,
            proxy: base.proxy.clone(),
            user_agent: base.user_agent.clone(),
            ..Default::default()
        };
        if !base.default_headers.is_empty() {
            http_config.headers.extend(base.default_headers.clone());
        }

        Self {
            base,
            http_config,
            tracing_config: None,
            retry_options: None,
            http_transport: None,
            http_interceptors: inherited_interceptors,
            http_debug: inherited_debug,
        }
    }

    // ========================================================================
    // HTTP Configuration Methods
    // ========================================================================

    /// Set request timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }

    /// Set connection timeout.
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }

    /// Set custom HTTP client.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.base.http_client = Some(client);
        self
    }

    /// Control whether to disable compression for streaming (SSE) requests.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
        self
    }

    // ========================================================================
    // Tracing Configuration Methods
    // ========================================================================

    /// Set custom tracing configuration.
    pub fn tracing(mut self, config: TracingConfig) -> Self {
        self.tracing_config = Some(config);
        self
    }

    /// Enable debug tracing (development-friendly configuration).
    pub fn debug_tracing(self) -> Self {
        self.tracing(TracingConfig::development())
    }

    /// Enable minimal tracing (info level, LLM only).
    pub fn minimal_tracing(self) -> Self {
        self.tracing(TracingConfig::minimal())
    }

    /// Enable production-ready JSON tracing.
    pub fn json_tracing(self) -> Self {
        self.tracing(TracingConfig::json_production())
    }

    /// Enable pretty-printed formatting for JSON bodies and headers in tracing.
    pub fn pretty_json(mut self, pretty: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(TracingConfig::development)
            .with_pretty_json(pretty);
        self.tracing_config = Some(config);
        self
    }

    /// Control masking of sensitive values (API keys, tokens) in tracing logs.
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

    /// Set unified retry options for chat operations.
    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.retry_options = Some(options);
        self
    }

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_http_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.http_transport = Some(transport);
        self
    }

    // ========================================================================
    // HTTP Interceptor Methods
    // ========================================================================

    /// Add a custom HTTP interceptor.
    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    /// Enable a built-in logging interceptor for HTTP debugging.
    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.http_debug = enabled;
        self
    }

    // ========================================================================
    // Build Helper Methods
    // ========================================================================

    /// Build the HTTP client from configuration.
    pub fn build_http_client(&self) -> Result<reqwest::Client, LlmError> {
        // Prefer a custom client when provided at the base builder level.
        if let Some(client) = &self.base.http_client {
            return Ok(client.clone());
        }
        crate::execution::http::client::build_http_client_from_config(&self.http_config)
    }

    /// Get HTTP interceptors including debug interceptor if enabled.
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

    /// Get automatic middlewares for a provider and model.
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
    fn provider_core_creation_inherits_defaults() {
        let base = BuilderBase::default();
        let core = ProviderCore::new(base);
        assert!(core.tracing_config.is_none());
        assert!(core.retry_options.is_none());
        assert!(core.http_interceptors.is_empty());
        assert!(!core.http_debug);
    }

    #[test]
    fn provider_core_timeout_configuration() {
        let base = BuilderBase::default();
        let core = ProviderCore::new(base)
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10));
        assert_eq!(core.http_config.timeout, Some(Duration::from_secs(30)));
        assert_eq!(
            core.http_config.connect_timeout,
            Some(Duration::from_secs(10))
        );
    }
}
