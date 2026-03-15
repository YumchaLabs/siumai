use crate::LlmError;
use crate::builder::{BuilderBase, ProviderCore};
use crate::retry_api::RetryOptions;
use crate::types::CommonParams;
use std::sync::Arc;

/// Gemini-specific builder for configuring Gemini clients.
///
/// This builder provides Gemini-specific configuration options while
/// inheriting common HTTP and timeout settings from the base `LlmBuilder`.
///
/// Retry: call `.with_retry(RetryOptions::backoff())` to enable unified retry
/// for chat operations.
///
/// # Example
/// ```rust,no_run
/// use siumai::builder::LlmBuilder;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let client = LlmBuilder::new()
///         .gemini()
///         .api_key("your-api-key")
///         .model("gemini-1.5-flash")
///         .temperature(0.7)
///         .max_tokens(8192)
///         .build()
///         .await?;
///
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct GeminiBuilder {
    /// Core provider configuration (composition)
    pub(crate) core: ProviderCore,
    /// Gemini API key
    api_key: Option<String>,
    /// Base URL for Gemini API
    base_url: Option<String>,
    /// Common params (unified: model, temperature, max_tokens, top_p, stop_sequences)
    common_params: CommonParams,
    /// Top-k setting
    top_k: Option<i32>,
    /// Stop sequences are provided via common_params
    /// Candidate count
    candidate_count: Option<i32>,
    /// Safety settings
    safety_settings: Option<Vec<crate::providers::gemini::SafetySetting>>,
    /// JSON schema for structured output
    json_schema: Option<serde_json::Value>,
    /// Thinking configuration
    thinking_config: Option<crate::providers::gemini::ThinkingConfig>,
}

impl GeminiBuilder {
    /// Create a new Gemini builder
    pub fn new(base: BuilderBase) -> Self {
        Self {
            core: ProviderCore::new(base),
            api_key: None,
            base_url: None,
            common_params: CommonParams::default(),
            top_k: None,
            candidate_count: None,
            safety_settings: None,
            json_schema: None,
            thinking_config: None,
        }
    }

    /// Set the API key
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the base URL
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the model (unified via common_params)
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Set temperature (0.0 to 2.0)
    pub const fn temperature(mut self, temperature: f64) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set maximum output tokens
    pub fn max_tokens(mut self, max_tokens: i32) -> Self {
        let v = if max_tokens < 0 { 0 } else { max_tokens as u32 };
        self.common_params.max_tokens = Some(v);
        self
    }

    /// Set top-p (0.0 to 1.0)
    pub const fn top_p(mut self, top_p: f64) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Set top-k
    pub const fn top_k(mut self, top_k: i32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(sequences);
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

    /// Set candidate count
    pub const fn candidate_count(mut self, count: i32) -> Self {
        self.candidate_count = Some(count);
        self
    }

    /// Set safety settings
    pub fn safety_settings(
        mut self,
        settings: Vec<crate::providers::gemini::SafetySetting>,
    ) -> Self {
        self.safety_settings = Some(settings);
        self
    }

    /// Enable structured output with JSON schema
    pub fn json_schema(mut self, schema: serde_json::Value) -> Self {
        self.json_schema = Some(schema);
        self
    }

    /// Set thinking budget in tokens
    ///
    /// - Use -1 for dynamic thinking (model decides)
    /// - Use 0 to attempt to disable thinking (may not work on all models)
    /// - Use positive values to set a specific token budget
    ///
    /// The actual supported range depends on the model being used.
    /// Note: This automatically enables thought summaries in the response.
    pub fn thinking_budget(mut self, budget: i32) -> Self {
        if self.thinking_config.is_none() {
            self.thinking_config = Some(crate::providers::gemini::ThinkingConfig::new());
        }
        if let Some(ref mut config) = self.thinking_config {
            config.thinking_budget = Some(budget);
            // Automatically enable thought summaries when setting a budget
            // This is required by Gemini API to actually receive thinking content
            if budget != 0 {
                config.include_thoughts = Some(true);
            } else {
                config.include_thoughts = Some(false);
            }
        }
        self
    }

    /// Enable or disable thought summaries in response
    ///
    /// This controls whether thinking summaries are included in the response,
    /// not whether the model thinks internally.
    pub fn thought_summaries(mut self, include: bool) -> Self {
        if self.thinking_config.is_none() {
            self.thinking_config = Some(crate::providers::gemini::ThinkingConfig::new());
        }
        if let Some(ref mut config) = self.thinking_config {
            config.include_thoughts = Some(include);
        }
        self
    }

    /// Enable dynamic thinking (model decides when and how much to think)
    pub fn thinking(mut self) -> Self {
        self.thinking_config = Some(crate::providers::gemini::ThinkingConfig::dynamic());
        self
    }

    /// Enable or disable reasoning mode (unified interface)
    ///
    /// This is the unified reasoning interface that works across all providers.
    /// For Gemini, this maps to thinking configuration with thought summaries.
    ///
    /// # Arguments
    /// * `enable` - Whether to enable reasoning/thinking mode
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Siumai::builder()
    ///         .gemini()
    ///         .api_key("your-api-key")
    ///         .model("gemini-2.5-pro")
    ///         .reasoning(true)  // Unified interface
    ///         .build()
    ///         .await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub fn reasoning(mut self, enable: bool) -> Self {
        if enable {
            // Enable dynamic thinking with thought summaries
            self.thinking_config = Some(crate::providers::gemini::ThinkingConfig::dynamic());
        } else {
            // Attempt to disable thinking
            self.thinking_config = Some(crate::providers::gemini::ThinkingConfig::disabled());
        }
        self
    }

    /// Set reasoning budget (unified interface)
    ///
    /// This is the unified reasoning budget interface that works across all providers.
    /// For Gemini, this maps directly to thinking budget configuration.
    ///
    /// # Arguments
    /// * `budget` - Number of tokens for reasoning (-1 for dynamic, 0 for disabled)
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = Siumai::builder()
    ///         .gemini()
    ///         .api_key("your-api-key")
    ///         .model("gemini-2.5-pro")
    ///         .reasoning_budget(2048)  // Unified interface
    ///         .build()
    ///         .await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub fn reasoning_budget(self, budget: i32) -> Self {
        // Use the existing thinking_budget method
        self.thinking_budget(budget)
    }

    /// Attempt to disable thinking
    ///
    /// Note: Not all models support disabling thinking. If the model doesn't
    /// support it, the API will return an appropriate error.
    pub fn disable_thinking(mut self) -> Self {
        self.thinking_config = Some(crate::providers::gemini::ThinkingConfig::disabled());
        self
    }

    /// Convert the builder into the canonical Gemini config.
    pub fn into_config(self) -> Result<crate::providers::gemini::GeminiConfig, LlmError> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("GEMINI_API_KEY").ok())
            .ok_or_else(|| {
                LlmError::ConfigurationError("API key is required for Gemini".to_string())
            })?;

        let mut config = crate::providers::gemini::GeminiConfig::new(api_key);

        if let Some(base_url) = self.base_url {
            config = config.with_base_url(base_url);
        }

        config = config.with_common_params(self.common_params.clone());
        if !config.common_params.model.is_empty() {
            let model = config.common_params.model.clone();
            config = config.with_model(model);
        }

        let mut generation_config = crate::providers::gemini::GenerationConfig::new();
        let mut has_generation_config = false;

        if let Some(top_k) = self.top_k {
            generation_config = generation_config.with_top_k(top_k);
            has_generation_config = true;
        }

        if let Some(count) = self.candidate_count {
            generation_config = generation_config.with_candidate_count(count);
            has_generation_config = true;
        }

        if let Some(schema) = self.json_schema {
            generation_config = generation_config.with_response_schema(schema);
            generation_config =
                generation_config.with_response_mime_type("application/json".to_string());
            has_generation_config = true;
        }

        if let Some(thinking_config) = &self.thinking_config {
            generation_config = generation_config.with_thinking_config(thinking_config.clone());
            has_generation_config = true;
        }

        if has_generation_config {
            config = config.with_generation_config(generation_config);
        }

        if let Some(safety_settings) = self.safety_settings {
            config = config.with_safety_settings(safety_settings);
        }

        config = config.with_http_config(self.core.http_config.clone());
        if let Some(transport) = self.core.http_transport.clone() {
            config = config.with_http_transport(transport);
        }

        let model_id = if !config.common_params.model.is_empty() {
            config.common_params.model.clone()
        } else {
            config.model.clone()
        };

        let mut model_middlewares = self.core.get_auto_middlewares("gemini", &model_id);
        model_middlewares.push(Arc::new(
            crate::providers::gemini::middleware::GeminiToolWarningsMiddleware::new(),
        ));

        Ok(config
            .with_http_interceptors(self.core.get_http_interceptors())
            .with_model_middlewares(model_middlewares))
    }

    /// Build the Gemini client
    pub async fn build(self) -> Result<crate::providers::gemini::GeminiClient, LlmError> {
        let http_client_override = self.core.base.http_client.clone();
        let tracing_config = self.core.tracing_config.clone();
        let retry_options = self.core.retry_options.clone();
        let config = self.into_config()?;

        let mut client = if let Some(http_client) = http_client_override {
            crate::providers::gemini::GeminiClient::with_http_client(config, http_client)?
        } else {
            crate::providers::gemini::GeminiClient::from_config(config)?
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
    fn gemini_builder_into_config_converges_on_gemini_config() {
        let config = GeminiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .base_url("https://example.com")
            .model("gemini-2.5-flash")
            .temperature(0.3)
            .max_tokens(1024)
            .top_k(8)
            .candidate_count(2)
            .json_schema(serde_json::json!({ "type": "object" }))
            .reasoning_budget(2048)
            .timeout(Duration::from_secs(20))
            .http_debug(true)
            .into_config()
            .expect("into_config ok");

        assert_eq!(config.base_url, "https://example.com");
        assert_eq!(config.model, "gemini-2.5-flash");
        assert_eq!(config.common_params.model, "gemini-2.5-flash");
        assert_eq!(config.common_params.temperature, Some(0.3));
        assert_eq!(config.common_params.max_tokens, Some(1024));
        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(20)));
        let generation_config = config.generation_config.expect("generation config");
        assert_eq!(generation_config.top_k, Some(8));
        assert_eq!(generation_config.candidate_count, Some(2));
        assert_eq!(
            generation_config.response_mime_type.as_deref(),
            Some("application/json")
        );
        assert_eq!(
            generation_config.response_schema,
            Some(serde_json::json!({ "type": "object" }))
        );
        let thinking = generation_config.thinking_config.expect("thinking config");
        assert_eq!(thinking.thinking_budget, Some(2048));
        assert_eq!(thinking.include_thoughts, Some(true));
        assert_eq!(config.http_interceptors.len(), 1);
        assert!(!config.model_middlewares.is_empty());
    }

    #[test]
    fn gemini_builder_into_config_matches_manual_gemini_config() {
        let builder_config = GeminiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .base_url("https://example.com")
            .model("gemini-2.5-flash")
            .temperature(0.3)
            .max_tokens(1024)
            .top_k(8)
            .candidate_count(2)
            .json_schema(serde_json::json!({ "type": "object" }))
            .reasoning_budget(2048)
            .timeout(Duration::from_secs(20))
            .http_debug(true)
            .into_config()
            .expect("builder config");

        let mut http_config = crate::types::HttpConfig::default();
        http_config.timeout = Some(Duration::from_secs(20));
        let mut manual_middlewares =
            crate::execution::middleware::build_auto_middlewares_vec("gemini", "gemini-2.5-flash");
        manual_middlewares.push(Arc::new(
            crate::providers::gemini::middleware::GeminiToolWarningsMiddleware::new(),
        ));
        let manual_config = crate::providers::gemini::GeminiConfig::new("test-key")
            .with_base_url("https://example.com".to_string())
            .with_model("gemini-2.5-flash".to_string())
            .with_temperature(0.3)
            .with_max_tokens(1024)
            .with_top_k(8)
            .with_candidate_count(2)
            .with_json_schema(serde_json::json!({ "type": "object" }))
            .with_reasoning_budget(2048)
            .with_http_config(http_config)
            .with_http_interceptors(vec![Arc::new(
                crate::execution::http::interceptor::LoggingInterceptor,
            )])
            .with_model_middlewares(manual_middlewares);

        assert_eq!(builder_config.base_url, manual_config.base_url);
        assert_eq!(builder_config.model, manual_config.model);
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
            serde_json::to_value(&builder_config.generation_config)
                .expect("serialize builder generation config"),
            serde_json::to_value(&manual_config.generation_config)
                .expect("serialize manual generation config")
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
