use crate::params::AnthropicParams;
use crate::providers::AnthropicClient;
use crate::retry_api::RetryOptions;
use crate::{CommonParams, HttpConfig, LlmBuilder, LlmError};
use std::collections::HashMap;
use std::time::Duration;

/// Anthropic-specific builder
///
/// Retry: call `.with_retry(RetryOptions::backoff())` to enable unified retry
/// for chat operations.
pub struct AnthropicBuilder {
    pub(crate) base: LlmBuilder,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    common_params: CommonParams,
    anthropic_params: AnthropicParams,
    http_config: HttpConfig,
    tracing_config: Option<crate::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
}

impl AnthropicBuilder {
    pub fn new(base: LlmBuilder) -> Self {
        Self {
            base,
            api_key: None,
            base_url: None,
            model: None,
            common_params: CommonParams::default(),
            anthropic_params: AnthropicParams::default(),
            http_config: HttpConfig::default(),
            tracing_config: None,
            retry_options: None,
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

    /// Control whether to disable compression for streaming (SSE) requests.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
        self
    }

    // Anthropic-specific parameters

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

    // === Tracing Configuration ===

    /// Set custom tracing configuration
    pub fn tracing(mut self, config: crate::tracing::TracingConfig) -> Self {
        self.tracing_config = Some(config);
        self
    }

    /// Enable debug tracing (development-friendly configuration)
    pub fn debug_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::development())
    }

    /// Enable minimal tracing (info level, LLM only)
    pub fn minimal_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::minimal())
    }

    /// Enable production-ready JSON tracing
    pub fn json_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::json_production())
    }

    /// Enable pretty-printed formatting for JSON bodies and headers in tracing
    pub fn pretty_json(mut self, pretty: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(crate::tracing::TracingConfig::development)
            .with_pretty_json(pretty);
        self.tracing_config = Some(config);
        self
    }

    /// Control masking of sensitive values (API keys, tokens) in tracing logs
    pub fn mask_sensitive_values(mut self, mask: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(crate::tracing::TracingConfig::development)
            .with_mask_sensitive_values(mask);
        self.tracing_config = Some(config);
        self
    }

    /// Set unified retry options for chat operations
    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.retry_options = Some(options);
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
        let api_key = self
            .api_key
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .ok_or(LlmError::MissingApiKey(
                "Anthropic API key not provided".to_string(),
            ))?;

        let base_url = self
            .base_url
            .unwrap_or_else(|| "https://api.anthropic.com".to_string());

        // Note: Tracing initialization has been moved to siumai-extras.
        // Users should initialize tracing manually using siumai_extras::telemetry
        // or tracing_subscriber directly before creating the client.

        let http_client = self.base.http_client.unwrap_or_else(|| {
            reqwest::Client::builder()
                .timeout(self.base.timeout.unwrap_or(Duration::from_secs(30)))
                .build()
                .unwrap()
        });

        // Convert AnthropicParams to AnthropicSpecificParams
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

        // Create AnthropicClient with the converted specific_params
        let mut client = AnthropicClient::new(
            api_key,
            base_url,
            http_client,
            self.common_params,
            self.anthropic_params,
            self.http_config,
        );

        // Update the client with the specific params and tracing
        client = client.with_specific_params(specific_params);
        client.set_tracing_config(self.tracing_config);
        client.set_retry_options(self.retry_options.clone());

        Ok(client)
    }
}
