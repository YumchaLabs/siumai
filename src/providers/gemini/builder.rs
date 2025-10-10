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
#[derive(Debug, Clone)]
pub struct GeminiBuilder {
    /// Base builder with HTTP configuration
    pub(crate) base: LlmBuilder,
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
    /// Tracing configuration
    tracing_config: Option<crate::tracing::TracingConfig>,
    /// Unified retry options
    retry_options: Option<RetryOptions>,
}

impl GeminiBuilder {
    /// Create a new Gemini builder
    pub const fn new(base: LlmBuilder) -> Self {
        Self {
            base,
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
            tracing_config: None,
            retry_options: None,
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

    /// Build the Gemini client
    pub async fn build(self) -> Result<crate::providers::gemini::GeminiClient, LlmError> {
        let api_key = self.api_key.ok_or_else(|| {
            LlmError::ConfigurationError("API key is required for Gemini".to_string())
        })?;

        // Initialize tracing if configured
        let _tracing_guard = if let Some(ref tracing_config) = self.tracing_config {
            crate::tracing::init_tracing(tracing_config.clone())?
        } else {
            None
        };

        let mut config = crate::providers::gemini::GeminiConfig::new(api_key);

        if let Some(base_url) = self.base_url {
            config = config.with_base_url(base_url);
        }

        // Basic validation of thinking configuration
        if let Some(thinking_config) = &self.thinking_config {
            thinking_config.validate().map_err(|e| {
                crate::error::LlmError::ConfigurationError(format!(
                    "Invalid thinking configuration: {e}"
                ))
            })?;
        }

        if let Some(model) = self.model {
            config = config.with_model(model);
        }

        // Build generation config
        let mut generation_config = crate::providers::gemini::GenerationConfig::new();

        if let Some(temp) = self.temperature {
            generation_config = generation_config.with_temperature(temp);
        }

        if let Some(max_tokens) = self.max_tokens {
            generation_config = generation_config.with_max_output_tokens(max_tokens);
        }

        if let Some(top_p) = self.top_p {
            generation_config = generation_config.with_top_p(top_p);
        }

        if let Some(top_k) = self.top_k {
            generation_config = generation_config.with_top_k(top_k);
        }

        if let Some(stop_sequences) = self.stop_sequences {
            generation_config = generation_config.with_stop_sequences(stop_sequences);
        }

        if let Some(count) = self.candidate_count {
            generation_config = generation_config.with_candidate_count(count);
        }

        if let Some(schema) = self.json_schema {
            generation_config = generation_config.with_response_schema(schema);
            generation_config =
                generation_config.with_response_mime_type("application/json".to_string());
        }

        // Apply thinking configuration to generation config
        if let Some(thinking_config) = &self.thinking_config {
            generation_config = generation_config.with_thinking_config(thinking_config.clone());
        }

        config = config.with_generation_config(generation_config);

        if let Some(safety_settings) = self.safety_settings {
            config = config.with_safety_settings(safety_settings);
        }

        // Apply HTTP configuration from base builder
        if let Some(timeout) = self.base.timeout {
            config = config.with_timeout(timeout.as_secs());
        }

        // Build HTTP client from base builder to inherit unified HTTP config
        let http_client = self.base.build_http_client()?;

        let mut client =
            crate::providers::gemini::GeminiClient::with_http_client(config, http_client)?;
        client.set_tracing_guard(_tracing_guard);
        client.set_tracing_config(self.tracing_config);
        client.set_retry_options(self.retry_options.clone());

        Ok(client)
    }
}
