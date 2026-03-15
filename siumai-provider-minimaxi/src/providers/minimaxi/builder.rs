//! MiniMaxi Builder Implementation
//!
//! Builder pattern implementation for creating MiniMaxi clients.

use std::time::Duration;

use crate::builder::{BuilderBase, ProviderCore};
use crate::error::LlmError;
use crate::provider_options::{MinimaxiOptions, MinimaxiThinkingModeConfig};
use crate::retry_api::RetryOptions;

use super::client::MinimaxiClient;
use super::config::MinimaxiConfig;

/// MiniMaxi client builder
///
/// Retry: call `.with_retry(RetryOptions::backoff())` to enable unified retry
/// for chat operations.
#[derive(Clone)]
pub struct MinimaxiBuilder {
    /// Core provider configuration (composition)
    pub(crate) core: ProviderCore,
    pub(crate) config: MinimaxiConfig,
}

impl MinimaxiBuilder {
    /// Create a new MiniMaxi builder
    pub fn new(base: BuilderBase) -> Self {
        Self {
            core: ProviderCore::new(base),
            config: MinimaxiConfig::default(),
        }
    }

    /// Set the API key
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.config.api_key = api_key.into();
        self
    }

    /// Set the base URL
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.config.base_url = base_url.into();
        self
    }

    /// Set the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.config.common_params.model = model.into();
        self
    }

    /// Set request timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.core = self.core.timeout(timeout);
        self
    }

    /// Set connection timeout
    pub fn connect_timeout(mut self, connect_timeout: Duration) -> Self {
        self.core = self.core.connect_timeout(connect_timeout);
        self
    }

    /// Set a custom HTTP client.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.core = self.core.with_http_client(client);
        self
    }

    /// Control whether to disable compression for streaming (SSE) requests.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.core = self.core.http_stream_disable_compression(disable);
        self
    }

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_http_transport(
        mut self,
        transport: std::sync::Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.core = self.core.with_http_transport(transport);
        self
    }

    /// Alias for `with_http_transport(...)` (Vercel-aligned: `fetch`).
    pub fn fetch(
        self,
        transport: std::sync::Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.with_http_transport(transport)
    }

    /// Enable built-in HTTP debug logging.
    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.core = self.core.http_debug(enabled);
        self
    }

    // === Tracing Configuration ===

    /// Set custom tracing configuration
    pub fn tracing(mut self, config: crate::observability::tracing::TracingConfig) -> Self {
        self.core = self.core.tracing(config);
        self
    }

    /// Enable debug tracing (development-friendly configuration)
    pub fn debug_tracing(mut self) -> Self {
        self.core = self.core.debug_tracing();
        self
    }

    /// Enable minimal tracing (info level, LLM only)
    pub fn minimal_tracing(mut self) -> Self {
        self.core = self.core.minimal_tracing();
        self
    }

    /// Enable production-ready JSON tracing
    pub fn json_tracing(mut self) -> Self {
        self.core = self.core.json_tracing();
        self
    }

    /// Enable pretty-printed formatting for JSON bodies and headers in tracing
    pub fn pretty_json(mut self, pretty: bool) -> Self {
        self.core = self.core.pretty_json(pretty);
        self
    }

    /// Control masking of sensitive values (API keys, tokens) in tracing logs
    pub fn mask_sensitive_values(mut self, mask: bool) -> Self {
        self.core = self.core.mask_sensitive_values(mask);
        self
    }

    // === Retry Configuration ===

    /// Enable unified retry for chat operations
    pub fn with_retry(mut self, retry_options: RetryOptions) -> Self {
        self.core = self.core.with_retry(retry_options);
        self
    }

    // === HTTP Interceptors ===

    /// Add an HTTP interceptor
    pub fn with_http_interceptor(
        mut self,
        interceptor: std::sync::Arc<dyn crate::execution::http::interceptor::HttpInterceptor>,
    ) -> Self {
        self.core = self.core.with_http_interceptor(interceptor);
        self
    }

    /// Merge MiniMaxi default provider options into the builder config.
    pub fn with_minimaxi_options(mut self, options: MinimaxiOptions) -> Self {
        self.config = self.config.with_minimaxi_options(options);
        self
    }

    /// Configure MiniMaxi thinking mode defaults.
    pub fn with_thinking_mode(mut self, config: MinimaxiThinkingModeConfig) -> Self {
        self.config = self.config.with_thinking_mode(config);
        self
    }

    /// Configure MiniMaxi reasoning enablement defaults.
    pub fn with_reasoning_enabled(mut self, enabled: bool) -> Self {
        self.config = self.config.with_reasoning_enabled(enabled);
        self
    }

    /// Configure MiniMaxi reasoning budget defaults.
    pub fn with_reasoning_budget(mut self, budget: u32) -> Self {
        self.config = self.config.with_reasoning_budget(budget);
        self
    }

    /// Configure MiniMaxi JSON-object structured output defaults.
    pub fn with_json_object(mut self) -> Self {
        self.config = self.config.with_json_object();
        self
    }

    /// Configure MiniMaxi JSON-schema structured output defaults.
    pub fn with_json_schema(
        mut self,
        name: impl Into<String>,
        schema: serde_json::Value,
        strict: bool,
    ) -> Self {
        self.config = self.config.with_json_schema(name, schema, strict);
        self
    }

    /// Convert the builder into the canonical MiniMaxi config.
    pub fn into_config(self) -> Result<MinimaxiConfig, LlmError> {
        let api_key = if self.config.api_key.trim().is_empty() {
            std::env::var("MINIMAXI_API_KEY").map_err(|_| {
                LlmError::ConfigurationError(
                    "MiniMaxi API key not provided. Set MINIMAXI_API_KEY environment variable or use .api_key()".to_string(),
                )
            })?
        } else {
            self.config.api_key.trim().to_string()
        };

        let model_id = self.config.common_params.model.clone();
        let http_interceptors = self.core.get_http_interceptors();
        let model_middlewares = self.core.get_auto_middlewares("minimaxi", &model_id);

        let mut config = self.config;
        config.api_key = api_key;
        config.http_config = self.core.http_config.clone();
        config.http_transport = self.core.http_transport.clone();
        config.http_interceptors = http_interceptors;
        config.model_middlewares = model_middlewares;
        config.validate()?;

        Ok(config)
    }

    /// Build the MiniMaxi client
    pub async fn build(self) -> Result<MinimaxiClient, LlmError> {
        let http_client_override = self.core.base.http_client.clone();
        let tracing_config = self.core.tracing_config.clone();
        let retry_options = self.core.retry_options.clone();
        let config = self.into_config()?;

        let mut client = if let Some(http_client) = http_client_override {
            MinimaxiClient::with_http_client(config, http_client)?
        } else {
            MinimaxiClient::from_config(config)?
        };

        if let Some(tracing_config) = tracing_config {
            client = client.with_tracing(tracing_config);
        }

        if let Some(retry_options) = retry_options {
            client = client.with_retry(retry_options);
        }

        Ok(client)
    }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod config_first_tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl crate::execution::http::interceptor::HttpInterceptor for NoopInterceptor {}

    #[derive(Clone, Default)]
    struct NoopTransport;

    #[async_trait]
    impl crate::execution::http::transport::HttpTransport for NoopTransport {
        async fn execute_json(
            &self,
            _request: crate::execution::http::transport::HttpTransportRequest,
        ) -> Result<crate::execution::http::transport::HttpTransportResponse, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "noop transport should not execute during builder tests".to_string(),
            ))
        }
    }

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(v) => unsafe {
                    std::env::set_var(self.key, v);
                },
                None => unsafe {
                    std::env::remove_var(self.key);
                },
            }
        }
    }

    #[test]
    fn into_config_resolves_env_key_and_preserves_core_options() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = EnvGuard::set("MINIMAXI_API_KEY", "env-key");

        let cfg = MinimaxiBuilder::new(BuilderBase::default())
            .model("MiniMax-M2")
            .timeout(Duration::from_secs(12))
            .with_http_interceptor(Arc::new(NoopInterceptor))
            .with_http_transport(Arc::new(NoopTransport))
            .http_debug(true)
            .into_config()
            .expect("into_config");

        assert_eq!(cfg.api_key, "env-key");
        assert_eq!(cfg.common_params.model, "MiniMax-M2");
        assert_eq!(cfg.http_config.timeout, Some(Duration::from_secs(12)));
        assert!(cfg.http_transport.is_some());
        assert_eq!(cfg.http_interceptors.len(), 2);
        assert!(!cfg.model_middlewares.is_empty());
    }

    #[tokio::test]
    async fn build_and_into_config_converge_on_config_first_client_construction() {
        let builder = MinimaxiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .base_url("https://example.com/anthropic")
            .model("MiniMax-M2")
            .timeout(Duration::from_secs(9))
            .with_http_interceptor(Arc::new(NoopInterceptor));

        let cfg = builder.clone().into_config().expect("into_config");
        let built = builder.build().await.expect("build client");
        let from_config = MinimaxiClient::from_config(cfg).expect("from_config client");

        assert_eq!(built.config().api_key, from_config.config().api_key);
        assert_eq!(built.config().base_url, from_config.config().base_url);
        assert_eq!(
            built.config().common_params.model,
            from_config.config().common_params.model
        );
        assert_eq!(
            built.config().http_config.timeout,
            from_config.config().http_config.timeout
        );
        assert_eq!(
            built.config().http_interceptors.len(),
            from_config.config().http_interceptors.len()
        );
        assert_eq!(
            built.config().model_middlewares.len(),
            from_config.config().model_middlewares.len()
        );
    }

    #[test]
    fn into_config_preserves_default_minimaxi_provider_options() {
        let cfg = MinimaxiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .model("MiniMax-M2")
            .with_reasoning_budget(1024)
            .with_json_object()
            .into_config()
            .expect("into_config");

        let value = cfg
            .default_provider_options_map
            .get("minimaxi")
            .expect("minimaxi defaults");
        assert_eq!(value["thinking_mode"]["enabled"], serde_json::json!(true));
        assert_eq!(
            value["thinking_mode"]["thinking_budget"],
            serde_json::json!(1024)
        );
        assert_eq!(value["response_format"], serde_json::json!("JsonObject"));
    }
}
