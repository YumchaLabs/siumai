use super::client::AnthropicClient;
use crate::builder::{BuilderBase, ProviderCore};
use crate::params::AnthropicParams;
use crate::retry_api::RetryOptions;
use crate::{CommonParams, LlmError};
use secrecy::ExposeSecret;
use std::collections::HashMap;
use std::sync::Arc;

/// Anthropic-specific builder
///
/// Retry: call `.with_retry(RetryOptions::backoff())` to enable unified retry
/// for chat operations.
pub struct AnthropicBuilder {
    /// Core provider configuration (composition)
    pub(crate) core: ProviderCore,
    api_key: Option<String>,
    base_url: Option<String>,
    common_params: CommonParams,
    anthropic_params: AnthropicParams,
}

impl AnthropicBuilder {
    pub fn new(base: BuilderBase) -> Self {
        Self {
            core: ProviderCore::new(base),
            api_key: None,
            base_url: None,
            common_params: CommonParams::default(),
            anthropic_params: AnthropicParams::default(),
        }
    }

    /// Sets the API key
    pub fn api_key<S: Into<String>>(mut self, key: S) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the base URL
    pub fn base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Sets the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Common parameter setting methods (similar to `OpenAI`)
    pub const fn temperature(mut self, temp: f64) -> Self {
        self.common_params.temperature = Some(temp);
        self
    }

    pub const fn max_tokens(mut self, tokens: u32) -> Self {
        self.common_params.max_tokens = Some(tokens);
        self
    }

    pub const fn top_p(mut self, top_p: f64) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    // ========================================================================
    // Common Configuration Methods (delegated to ProviderCore)
    // ========================================================================

    // === HTTP Basic Configuration ===

    /// Set request timeout
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.core = self.core.timeout(timeout);
        self
    }

    /// Set connection timeout
    pub fn connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.core = self.core.connect_timeout(timeout);
        self
    }

    /// Set custom HTTP client
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.core = self.core.with_http_client(client);
        self
    }

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_http_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.core = self.core.with_http_transport(transport);
        self
    }

    /// Alias for `with_http_transport(...)` (Vercel-aligned: `fetch`).
    pub fn fetch(
        self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.with_http_transport(transport)
    }

    // === HTTP Advanced Configuration ===

    /// Control whether to disable compression for streaming (SSE) requests.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.core = self.core.http_stream_disable_compression(disable);
        self
    }

    /// Add a custom HTTP interceptor (builder collects and installs them on build).
    pub fn with_http_interceptor(
        mut self,
        interceptor: std::sync::Arc<dyn crate::execution::http::interceptor::HttpInterceptor>,
    ) -> Self {
        self.core = self.core.with_http_interceptor(interceptor);
        self
    }

    /// Enable a built-in logging interceptor for HTTP debugging (no sensitive data).
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

    /// Set unified retry options for chat operations
    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.core = self.core.with_retry(options);
        self
    }

    // ========================================================================
    // Provider-Specific Configuration
    // ========================================================================

    /// Sets cache control
    pub fn cache_control(mut self, cache: crate::params::anthropic::CacheControl) -> Self {
        self.anthropic_params.cache_control = Some(cache);
        self
    }

    /// Sets the thinking budget
    pub const fn thinking_budget(mut self, budget: u32) -> Self {
        self.anthropic_params.thinking_budget = Some(budget);
        self
    }

    /// Enable thinking mode with default budget (10k tokens)
    pub const fn with_thinking_enabled(mut self) -> Self {
        self.anthropic_params.thinking_budget = Some(10000);
        self
    }

    /// Enable thinking mode with specified budget tokens
    pub const fn with_thinking_mode(mut self, budget_tokens: Option<u32>) -> Self {
        self.anthropic_params.thinking_budget = budget_tokens;
        self
    }

    /// Sets the system message
    pub fn system_message<S: Into<String>>(mut self, system: S) -> Self {
        self.anthropic_params.system = Some(system.into());
        self
    }

    /// Adds metadata
    pub fn metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        if self.anthropic_params.metadata.is_none() {
            self.anthropic_params.metadata = Some(HashMap::new());
        }
        self.anthropic_params
            .metadata
            .as_mut()
            .unwrap()
            .insert(key.into(), value.into());
        self
    }

    /// Builds the Anthropic client
    pub fn into_config(self) -> Result<crate::providers::anthropic::AnthropicConfig, LlmError> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .ok_or(LlmError::MissingApiKey(
                "Anthropic API key not provided".to_string(),
            ))?;

        let base_url = self
            .base_url
            .unwrap_or_else(|| "https://api.anthropic.com".to_string());

        let model_id = self.common_params.model.clone();
        let http_interceptors = self.core.get_http_interceptors();
        let model_middlewares = self.core.get_auto_middlewares("anthropic", &model_id);

        Ok(crate::providers::anthropic::AnthropicConfig {
            api_key: secrecy::SecretString::from(api_key),
            base_url,
            common_params: self.common_params,
            anthropic_params: self.anthropic_params,
            http_config: self.core.http_config.clone(),
            http_transport: self.core.http_transport.clone(),
            http_interceptors,
            model_middlewares,
        })
    }

    pub async fn build(self) -> Result<AnthropicClient, LlmError> {
        let http_client_override = self.core.base.http_client.clone();
        let tracing_config = self.core.tracing_config.clone();
        let retry_options = self.core.retry_options.clone();
        let config = self.into_config()?;

        let mut client = if let Some(http_client) = http_client_override {
            let specific_params = crate::providers::anthropic::specific_params_from_legacy_params(
                &config.anthropic_params,
            );
            let http_interceptors = config.http_interceptors.clone();
            let model_middlewares = config.model_middlewares.clone();
            let mut client = AnthropicClient::new(
                config.api_key.expose_secret().to_string(),
                config.base_url,
                http_client,
                config.common_params,
                config.anthropic_params,
                config.http_config,
            )
            .with_specific_params(specific_params);

            if let Some(transport) = config.http_transport {
                client = client.with_http_transport(transport);
            }

            client
                .with_http_interceptors(http_interceptors)
                .with_model_middlewares(model_middlewares)
        } else {
            AnthropicClient::from_config(config)?
        };

        if let Some(tracing_config) = tracing_config {
            client.set_tracing_config(Some(tracing_config));
        }
        if let Some(retry_options) = retry_options {
            client.set_retry_options(Some(retry_options));
        }

        Ok(client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn anthropic_builder_into_config_converges_on_anthropic_config() {
        let config = AnthropicBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .base_url("https://example.com")
            .model("claude-3-5-sonnet-20241022")
            .temperature(0.2)
            .max_tokens(512)
            .top_p(0.9)
            .cache_control(crate::params::anthropic::CacheControl::ephemeral())
            .thinking_budget(2048)
            .system_message("You are helpful")
            .metadata("team", "core")
            .timeout(Duration::from_secs(18))
            .http_debug(true)
            .into_config()
            .expect("into_config ok");

        assert_eq!(config.base_url, "https://example.com");
        assert_eq!(config.common_params.model, "claude-3-5-sonnet-20241022");
        assert_eq!(config.common_params.temperature, Some(0.2));
        assert_eq!(config.common_params.max_tokens, Some(512));
        assert_eq!(config.common_params.top_p, Some(0.9));
        assert_eq!(
            config
                .anthropic_params
                .cache_control
                .as_ref()
                .map(|c| c.r#type.as_str()),
            Some("ephemeral")
        );
        assert_eq!(config.anthropic_params.thinking_budget, Some(2048));
        assert_eq!(
            config.anthropic_params.system.as_deref(),
            Some("You are helpful")
        );
        assert_eq!(
            config
                .anthropic_params
                .metadata
                .as_ref()
                .and_then(|m| m.get("team"))
                .map(String::as_str),
            Some("core")
        );
        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(18)));
        assert_eq!(config.http_interceptors.len(), 1);
    }

    #[test]
    fn anthropic_builder_into_config_matches_manual_anthropic_config() {
        let builder_config = AnthropicBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .base_url("https://example.com")
            .model("claude-3-5-sonnet-20241022")
            .temperature(0.2)
            .max_tokens(512)
            .top_p(0.9)
            .cache_control(crate::params::anthropic::CacheControl::ephemeral())
            .thinking_budget(2048)
            .system_message("You are helpful")
            .metadata("team", "core")
            .timeout(Duration::from_secs(18))
            .http_debug(true)
            .into_config()
            .expect("builder config");

        let mut http_config = crate::types::HttpConfig::default();
        http_config.timeout = Some(Duration::from_secs(18));
        let manual_config = crate::providers::anthropic::AnthropicConfig::new("test-key")
            .with_base_url("https://example.com")
            .with_model("claude-3-5-sonnet-20241022")
            .with_temperature(0.2)
            .with_max_tokens(512)
            .with_top_p(0.9)
            .with_cache_control(crate::params::anthropic::CacheControl::ephemeral())
            .with_thinking_budget(2048)
            .with_system_message("You are helpful")
            .add_metadata("team", "core")
            .with_http_config(http_config)
            .with_http_interceptors(vec![Arc::new(
                crate::execution::http::interceptor::LoggingInterceptor,
            )])
            .with_model_middlewares(crate::execution::middleware::build_auto_middlewares_vec(
                "anthropic",
                "claude-3-5-sonnet-20241022",
            ));

        assert_eq!(builder_config.base_url, manual_config.base_url);
        assert_eq!(
            builder_config.common_params.model,
            manual_config.common_params.model
        );
        assert_eq!(
            builder_config.common_params.temperature,
            manual_config.common_params.temperature
        );
        assert_eq!(
            builder_config.common_params.max_tokens,
            manual_config.common_params.max_tokens
        );
        assert_eq!(
            builder_config.common_params.top_p,
            manual_config.common_params.top_p
        );
        assert_eq!(
            serde_json::to_value(&builder_config.anthropic_params)
                .expect("serialize builder params"),
            serde_json::to_value(&manual_config.anthropic_params).expect("serialize manual params")
        );
        assert_eq!(
            builder_config.http_config.timeout,
            manual_config.http_config.timeout
        );
        assert_eq!(
            builder_config.http_interceptors.len(),
            manual_config.http_interceptors.len()
        );
        assert_eq!(
            builder_config.model_middlewares.len(),
            manual_config.model_middlewares.len()
        );
    }
}
