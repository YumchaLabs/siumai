//! `OpenAI` Provider Builder
//!
//! This module provides the OpenAI-specific builder implementation that follows
//! the design pattern established in the main builder module.
//!
//! This builder uses composition with `ProviderCore` to eliminate code duplication
//! and ensure consistency across all provider builders.

use crate::builder::LlmBuilder;
use crate::error::LlmError;
use crate::params::{OpenAiParams, ResponseFormat, ToolChoice};
use crate::provider_core::builder_core::ProviderCore;
use crate::retry_api::RetryOptions;
use crate::types::*;

use super::OpenAiClient;

/// OpenAI-specific builder for configuring `OpenAI` clients.
///
/// This builder provides OpenAI-specific configuration options while
/// using `ProviderCore` for common HTTP, tracing, and retry configuration.
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
    /// Core configuration shared across all providers (composition over inheritance)
    pub(crate) core: ProviderCore,
    /// OpenAI API key
    api_key: Option<String>,
    /// Base URL for OpenAI API
    base_url: Option<String>,
    /// Organization ID
    organization: Option<String>,
    /// Project ID
    project: Option<String>,
    /// Model name
    model: Option<String>,
    /// Common parameters (temperature, max_tokens, etc.)
    common_params: CommonParams,
    /// OpenAI-specific parameters
    openai_params: OpenAiParams,
    /// Whether to use the OpenAI Responses API instead of Chat Completions
    use_responses_api: bool,
    /// Responses API chaining id
    responses_previous_response_id: Option<String>,
    /// Responses API built-in tools
    responses_built_in_tools: Vec<crate::types::OpenAiBuiltInTool>,
}

#[cfg(feature = "openai")]
impl OpenAiBuilder {
    pub fn new(base: LlmBuilder) -> Self {
        Self {
            core: ProviderCore::new(base),
            api_key: None,
            base_url: None,
            organization: None,
            project: None,
            model: None,
            common_params: CommonParams::default(),
            openai_params: OpenAiParams::default(),
            use_responses_api: false,
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

    // ========================================================================
    // Common Configuration Methods (delegated to ProviderCore)
    // ========================================================================

    /// Control whether to disable compression for streaming (SSE) requests.
    /// Default is true for stability. Set to false to allow compression.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.core = self.core.http_stream_disable_compression(disable);
        self
    }

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

    /// Set unified retry options for chat operations
    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.core = self.core.with_retry(options);
        self
    }

    /// Add a custom HTTP interceptor
    pub fn with_http_interceptor(
        mut self,
        interceptor: std::sync::Arc<dyn crate::utils::http_interceptor::HttpInterceptor>,
    ) -> Self {
        self.core = self.core.with_http_interceptor(interceptor);
        self
    }

    /// Enable a built-in logging interceptor for HTTP debugging
    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.core = self.core.http_debug(enabled);
        self
    }

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

        // Step 3: Build configuration
        let model_id = self
            .model
            .clone()
            .unwrap_or_else(|| self.common_params.model.clone());

        let mut cfg = crate::providers::openai::OpenAiConfig {
            api_key: secrecy::SecretString::from(api_key),
            base_url,
            organization: self.organization,
            project: self.project,
            common_params: self.common_params,
            openai_params: self.openai_params,
            http_config: self.core.http_config.clone(),
            web_search_config: crate::types::WebSearchConfig::default(),
        };

        // Ensure model is set if provided separately
        if let Some(m) = &self.model {
            cfg.common_params.model = m.clone();
        }

        // Step 4: Build HTTP client from core
        let http_client = self.core.build_http_client()?;

        // Step 5: Create client instance
        let mut client = OpenAiClient::new(cfg, http_client);

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
        let middlewares = self.core.get_auto_middlewares("openai", &model_id);
        if !middlewares.is_empty() {
            client = client.with_model_middlewares(middlewares);
        }

        Ok(client)
    }
}
