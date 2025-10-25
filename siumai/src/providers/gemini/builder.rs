use crate::core::builder_core::ProviderCore;
use crate::retry_api::RetryOptions;
use crate::{LlmBuilder, LlmError};

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
    /// Model to use
    model: Option<String>,
    /// Temperature setting
    temperature: Option<f32>,
    /// Maximum output tokens
    max_tokens: Option<i32>,
    /// Top-p setting
    top_p: Option<f32>,
    /// Top-k setting
    top_k: Option<i32>,
    /// Stop sequences
    stop_sequences: Option<Vec<String>>,
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
    pub fn new(base: LlmBuilder) -> Self {
        Self {
            core: ProviderCore::new(base),
            api_key: None,
            base_url: None,
            model: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
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

    /// Set the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set temperature (0.0 to 2.0)
    pub const fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set maximum output tokens
    pub const fn max_tokens(mut self, max_tokens: i32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set top-p (0.0 to 1.0)
    pub const fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set top-k
    pub const fn top_k(mut self, top_k: i32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
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
    pub const fn thinking_budget(mut self, budget: i32) -> Self {
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
    pub const fn thought_summaries(mut self, include: bool) -> Self {
        if self.thinking_config.is_none() {
            self.thinking_config = Some(crate::providers::gemini::ThinkingConfig::new());
        }
        if let Some(ref mut config) = self.thinking_config {
            config.include_thoughts = Some(include);
        }
        self
    }

    /// Enable dynamic thinking (model decides when and how much to think)
    pub const fn thinking(mut self) -> Self {
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
    pub const fn reasoning(mut self, enable: bool) -> Self {
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
    pub const fn reasoning_budget(self, budget: i32) -> Self {
        // Use the existing thinking_budget method
        self.thinking_budget(budget)
    }

    /// Attempt to disable thinking
    ///
    /// Note: Not all models support disabling thinking. If the model doesn't
    /// support it, the API will return an appropriate error.
    pub const fn disable_thinking(mut self) -> Self {
        self.thinking_config = Some(crate::providers::gemini::ThinkingConfig::disabled());
        self
    }

    /// Build the Gemini client
    pub async fn build(self) -> Result<crate::providers::gemini::GeminiClient, LlmError> {
        // Step 1: Get API key (from parameter)
        let api_key = self.api_key.ok_or_else(|| {
            LlmError::ConfigurationError("API key is required for Gemini".to_string())
        })?;

        // Step 2: Get base URL (from parameter or default)
        // Note: Default is set in GeminiConfig::new()

        // Note: Tracing initialization has been moved to siumai-extras.
        // Users should initialize tracing manually using siumai_extras::telemetry
        // or tracing_subscriber directly before creating the client.

        // Step 3: Build configuration
        let mut config = crate::providers::gemini::GeminiConfig::new(api_key);

        if let Some(base_url) = self.base_url {
            config = config.with_base_url(base_url);
        }

        // Basic validation of thinking configuration
        if let Some(_thinking_config) = &self.thinking_config {
            // ThinkingConfig validation is now handled by the API
            // No client-side validation needed
        }

        // Save model before potentially moving it
        let model_for_middleware = self.model.clone();

        if let Some(model) = self.model {
            config = config.with_model(model);
        }

        // Set common parameters (temperature, max_tokens, top_p, stop_sequences)
        if let Some(temp) = self.temperature {
            config = config.with_temperature(temp);
        }

        if let Some(max_tokens) = self.max_tokens {
            config = config.with_max_tokens(max_tokens as u32);
        }

        if let Some(top_p) = self.top_p {
            config = config.with_top_p(top_p);
        }

        if let Some(stop_sequences) = self.stop_sequences {
            config = config.with_stop_sequences(stop_sequences);
        }

        // Build generation config for Gemini-specific parameters (top_k, candidate_count, etc.)
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

        // Apply thinking configuration to generation config
        if let Some(thinking_config) = &self.thinking_config {
            generation_config = generation_config.with_thinking_config(thinking_config.clone());
            has_generation_config = true;
        }

        // Only set generation_config if it has Gemini-specific parameters
        if has_generation_config {
            config = config.with_generation_config(generation_config);
        }

        if let Some(safety_settings) = self.safety_settings {
            config = config.with_safety_settings(safety_settings);
        }

        // Step 4: Build HTTP client from core
        let http_client = self.core.build_http_client()?;

        // Save model from config before moving it
        let model_id = config.model.clone();

        // Step 5: Create client instance
        let mut client =
            crate::providers::gemini::GeminiClient::with_http_client(config, http_client)?;

        // Step 6: Apply tracing and retry configuration from core
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
        let final_model_id = model_for_middleware.as_deref().unwrap_or(&model_id);
        let middlewares = self.core.get_auto_middlewares("gemini", final_model_id);
        if !middlewares.is_empty() {
            client = client.with_model_middlewares(middlewares);
        }

        Ok(client)
    }
}
