use crate::params::AnthropicParams;
use crate::provider_core::builder_core::ProviderCore;
use crate::providers::AnthropicClient;
use crate::retry_api::RetryOptions;
use crate::{CommonParams, LlmBuilder, LlmError};
use std::collections::HashMap;

/// Anthropic-specific builder
///
/// Retry: call `.with_retry(RetryOptions::backoff())` to enable unified retry
/// for chat operations.
pub struct AnthropicBuilder {
    /// Core provider configuration (composition)
    pub(crate) core: ProviderCore,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    common_params: CommonParams,
    anthropic_params: AnthropicParams,
}

impl AnthropicBuilder {
    pub fn new(base: LlmBuilder) -> Self {
        Self {
            core: ProviderCore::new(base),
            api_key: None,
            base_url: None,
            model: None,
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
        let model_str = model.into();
        self.model = Some(model_str.clone());
        self.common_params.model = model_str;
        self
    }

    /// Common parameter setting methods (similar to `OpenAI`)
    pub const fn temperature(mut self, temp: f32) -> Self {
        self.common_params.temperature = Some(temp);
        self
    }

    pub const fn max_tokens(mut self, tokens: u32) -> Self {
        self.common_params.max_tokens = Some(tokens);
        self
    }

    pub const fn top_p(mut self, top_p: f32) -> Self {
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

    // === HTTP Advanced Configuration ===

    /// Control whether to disable compression for streaming (SSE) requests.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.core = self.core.http_stream_disable_compression(disable);
        self
    }

    /// Add a custom HTTP interceptor (builder collects and installs them on build).
    pub fn with_http_interceptor(
        mut self,
        interceptor: std::sync::Arc<dyn crate::utils::http_interceptor::HttpInterceptor>,
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
    pub fn tracing(mut self, config: crate::tracing::TracingConfig) -> Self {
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
    pub async fn build(self) -> Result<AnthropicClient, LlmError> {
        // Step 1: Get API key (priority: parameter > environment variable)
        let api_key = self
            .api_key
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .ok_or(LlmError::MissingApiKey(
                "Anthropic API key not provided".to_string(),
            ))?;

        // Step 2: Get base URL (priority: parameter > default)
        let base_url = self
            .base_url
            .unwrap_or_else(|| "https://api.anthropic.com".to_string());

        // Note: Tracing initialization has been moved to siumai-extras.
        // Users should initialize tracing manually using siumai_extras::telemetry
        // or tracing_subscriber directly before creating the client.

        // Step 3: Build configuration
        let model_id = self.common_params.model.clone();

        let specific_params = crate::providers::anthropic::types::AnthropicSpecificParams {
            beta_features: self
                .anthropic_params
                .beta_features
                .clone()
                .unwrap_or_default(),
            cache_control: self.anthropic_params.cache_control.as_ref().map(|_cc| {
                crate::providers::anthropic::cache::CacheControl::ephemeral() // Convert from params::CacheControl
            }),
            thinking_config: self.anthropic_params.thinking_budget.map(|budget| {
                crate::providers::anthropic::thinking::ThinkingConfig::enabled(budget)
            }),
            metadata: self.anthropic_params.metadata.as_ref().map(|m| {
                // Convert HashMap<String, String> to serde_json::Value
                let mut json_map = serde_json::Map::new();
                for (k, v) in m {
                    json_map.insert(k.clone(), serde_json::Value::String(v.clone()));
                }
                serde_json::Value::Object(json_map)
            }),
        };

        // Step 4: Build HTTP client from core
        let http_client = self.core.build_http_client()?;

        // Step 5: Create client instance
        let mut client = AnthropicClient::new(
            api_key,
            base_url,
            http_client,
            self.common_params,
            self.anthropic_params,
            self.core.http_config.clone(),
        );

        // Step 6: Apply tracing and retry configuration from core
        client = client.with_specific_params(specific_params);
        if let Some(ref tracing_config) = self.core.tracing_config {
            client.set_tracing_config(Some(tracing_config.clone()));
        }
        if let Some(ref retry_options) = self.core.retry_options {
            client.set_retry_options(Some(retry_options.clone()));
        }

        // Step 7: Install HTTP interceptors
        let interceptors = self.core.get_http_interceptors();
        if !interceptors.is_empty() {
            client = client.with_http_interceptors(interceptors);
        }

        // Step 8: Install automatic middlewares
        let middlewares = self.core.get_auto_middlewares("anthropic", &model_id);
        if !middlewares.is_empty() {
            client = client.with_model_middlewares(middlewares);
        }

        Ok(client)
    }
}
