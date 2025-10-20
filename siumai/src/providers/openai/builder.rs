//! `OpenAI` Provider Builder
//!
//! This module provides the OpenAI-specific builder implementation that follows
//! the design pattern established in the main builder module.

use crate::builder::LlmBuilder;
use crate::error::LlmError;
use crate::params::{OpenAiParams, ResponseFormat, ToolChoice};
use crate::retry_api::RetryOptions;
use crate::types::*;

use super::OpenAiClient;
use crate::utils::http_interceptor::{HttpInterceptor, LoggingInterceptor};
use std::sync::Arc;

/// OpenAI-specific builder for configuring `OpenAI` clients.
///
/// This builder provides OpenAI-specific configuration options while
/// inheriting common HTTP and timeout settings from the base `LlmBuilder`.
///
/// Retry: call `.with_retry(RetryOptions::backoff())` to enable automatic retries
/// for chat operations via the unified retry facade.
///
/// # Example
/// ```rust,no_run
/// use siumai::builder::LlmBuilder;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let client = LlmBuilder::new()
///         .openai()
///         .api_key("your-api-key")
///         .model("gpt-4")
///         .temperature(0.7)
///         .max_tokens(1000)
///         .build()
///         .await?;
///
///     Ok(())
/// }
/// ```
pub struct OpenAiBuilder {
    pub(crate) base: LlmBuilder,
    api_key: Option<String>,
    base_url: Option<String>,
    organization: Option<String>,
    project: Option<String>,
    model: Option<String>,
    common_params: CommonParams,
    openai_params: OpenAiParams,
    http_config: HttpConfig,
    tracing_config: Option<crate::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    /// Whether to use the OpenAI Responses API instead of Chat Completions
    use_responses_api: bool,
    /// Optional HTTP interceptors applied to chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Enable lightweight HTTP debug logging interceptor
    http_debug: bool,
    /// Responses API chaining id
    responses_previous_response_id: Option<String>,
    /// Responses API built-in tools
    responses_built_in_tools: Vec<crate::types::OpenAiBuiltInTool>,
}

#[cfg(feature = "openai")]
impl OpenAiBuilder {
    pub fn new(base: LlmBuilder) -> Self {
        // Inherit interceptors/debug from unified builder
        let inherited_interceptors = base.http_interceptors.clone();
        let inherited_debug = base.http_debug;
        Self {
            base,
            api_key: None,
            base_url: None,
            organization: None,
            project: None,
            model: None,
            common_params: CommonParams::default(),
            openai_params: OpenAiParams::default(),
            http_config: HttpConfig::default(),
            tracing_config: None,
            retry_options: None,
            use_responses_api: false,
            http_interceptors: inherited_interceptors,
            http_debug: inherited_debug,
            responses_previous_response_id: None,
            responses_built_in_tools: Vec::new(),
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

    /// Sets the organization ID
    pub fn organization<S: Into<String>>(mut self, org: S) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Sets the project ID
    pub fn project<S: Into<String>>(mut self, project: S) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Sets the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        let model_str = model.into();
        self.model = Some(model_str.clone());
        self.common_params.model = model_str;
        self
    }

    /// Sets the temperature
    pub const fn temperature(mut self, temp: f32) -> Self {
        self.common_params.temperature = Some(temp);
        self
    }

    /// Sets the maximum number of tokens
    pub const fn max_tokens(mut self, tokens: u32) -> Self {
        self.common_params.max_tokens = Some(tokens);
        self
    }

    /// Sets `top_p`
    pub const fn top_p(mut self, top_p: f32) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Sets the stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(sequences);
        self
    }

    /// Sets the random seed
    pub const fn seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    // OpenAI-specific parameters

    /// Sets the response format
    pub fn response_format(mut self, format: ResponseFormat) -> Self {
        self.openai_params.response_format = Some(format);
        self
    }

    /// Sets the tool choice strategy
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.openai_params.tool_choice = Some(choice);
        self
    }

    /// Sets the frequency penalty
    pub const fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.openai_params.frequency_penalty = Some(penalty);
        self
    }

    /// Sets the presence penalty
    pub const fn presence_penalty(mut self, penalty: f32) -> Self {
        self.openai_params.presence_penalty = Some(penalty);
        self
    }

    /// Sets the user ID
    pub fn user<S: Into<String>>(mut self, user: S) -> Self {
        self.openai_params.user = Some(user.into());
        self
    }

    /// Enables parallel tool calls
    pub const fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.openai_params.parallel_tool_calls = Some(enabled);
        self
    }

    /// Sets the HTTP configuration
    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = config;
        self
    }

    /// Control whether to disable compression for streaming (SSE) requests.
    /// Default is true for stability. Set to false to allow compression.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
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

    /// Enable simple tracing (uses debug configuration)
    pub fn enable_tracing(self) -> Self {
        self.debug_tracing()
    }

    /// Disable tracing explicitly
    pub fn disable_tracing(self) -> Self {
        self.tracing(crate::tracing::TracingConfig::disabled())
    }

    /// Enable pretty-printed formatting for JSON bodies and headers in tracing
    ///
    /// This enables multi-line, indented JSON formatting and organized header display
    /// in debug logs, making them more human-readable for debugging purposes.
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Provider::openai()
    ///     .api_key("your-key")
    ///     .model("gpt-4o-mini")
    ///     .debug_tracing()
    ///     .pretty_json(true)  // Enable pretty formatting
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn pretty_json(mut self, pretty: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(crate::tracing::TracingConfig::development);

        let updated_config = crate::tracing::TracingConfigBuilder::from_config(config)
            .pretty_json(pretty)
            .build();

        self.tracing_config = Some(updated_config);
        self
    }

    /// Control masking of sensitive values (API keys, tokens) in tracing logs
    ///
    /// When enabled (default), sensitive values like API keys and authorization tokens
    /// are automatically masked in logs for security. Only the first and last few
    /// characters are shown.
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Provider::openai()
    ///     .api_key("your-key")
    ///     .model("gpt-4o-mini")
    ///     .debug_tracing()
    ///     .mask_sensitive_values(false)  // Disable masking (not recommended for production)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn mask_sensitive_values(mut self, mask: bool) -> Self {
        let config = self
            .tracing_config
            .take()
            .unwrap_or_else(crate::tracing::TracingConfig::development);

        let updated_config = crate::tracing::TracingConfigBuilder::from_config(config)
            .mask_sensitive_values(mask)
            .build();

        self.tracing_config = Some(updated_config);
        self
    }

    /// Set unified retry options for chat operations
    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.retry_options = Some(options);
        self
    }

    /// Add a custom HTTP interceptor (builder collects and installs them on build).
    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    /// Enable a built-in logging interceptor for HTTP debugging (no sensitive data).
    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.http_debug = enabled;
        self
    }

    /// Use the OpenAI Responses API instead of Chat Completions.
    ///
    /// When enabled, the client routes chat requests to `/responses` and sets
    /// the required beta header automatically.
    pub fn use_responses_api(mut self, enabled: bool) -> Self {
        self.use_responses_api = enabled;
        self
    }

    /// Set previous response id for Responses API chaining.
    pub fn responses_previous_response_id<S: Into<String>>(mut self, id: S) -> Self {
        self.responses_previous_response_id = Some(id.into());
        self
    }

    /// Add a built-in tool for Responses API.
    pub fn responses_built_in_tool(mut self, tool: crate::types::OpenAiBuiltInTool) -> Self {
        self.responses_built_in_tools.push(tool);
        self
    }

    /// Add multiple built-in tools for Responses API.
    pub fn responses_built_in_tools(mut self, tools: Vec<crate::types::OpenAiBuiltInTool>) -> Self {
        self.responses_built_in_tools.extend(tools);
        self
    }

    /// Builds the `OpenAI` client
    pub async fn build(self) -> Result<OpenAiClient, LlmError> {
        // Step 1: Get API key (priority: parameter > environment variable)
        let api_key = self
            .api_key
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or(LlmError::MissingApiKey(
                "OpenAI API key not provided".to_string(),
            ))?;

        // Step 2: Get base URL (priority: parameter > default)
        let base_url = self
            .base_url
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

        // Note: Tracing initialization has been moved to siumai-extras.
        // Users should initialize tracing manually using siumai_extras::telemetry
        // or tracing_subscriber directly before creating the client.

        // Step 3: Save model_id before moving common_params
        let model_id = self
            .model
            .clone()
            .unwrap_or_else(|| self.common_params.model.clone());

        // Step 4: Build HTTP client (inherits all base configuration)
        let http_client = self.base.build_http_client()?;

        // Step 5: Create client via unified OpenAiConfig (Spec-based path)
        let mut cfg = crate::providers::openai::OpenAiConfig {
            api_key: secrecy::SecretString::from(api_key),
            base_url,
            organization: self.organization,
            project: self.project,
            common_params: self.common_params,
            openai_params: self.openai_params,
            http_config: self.http_config,
            web_search_config: crate::types::WebSearchConfig::default(),
        };

        // Ensure model is set if provided separately
        if let Some(m) = &self.model {
            cfg.common_params.model = m.clone();
        }

        let mut client = OpenAiClient::new(cfg, http_client);

        // Step 6: Set retry
        client.set_retry_options(self.retry_options.clone());

        // Step 7: Install HTTP interceptors
        let mut interceptors = self.http_interceptors;
        if self.http_debug {
            interceptors.push(Arc::new(LoggingInterceptor));
        }
        client = client.with_http_interceptors(interceptors);

        // Step 8: Install automatic middlewares
        let middlewares = crate::middleware::build_auto_middlewares_vec("openai", &model_id);
        client = client.with_model_middlewares(middlewares);

        Ok(client)
    }
}
