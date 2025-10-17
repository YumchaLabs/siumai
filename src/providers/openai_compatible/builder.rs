use super::registry::get_provider_adapter;
use crate::retry_api::RetryOptions;
use crate::utils::http_interceptor::{HttpInterceptor, LoggingInterceptor};
use crate::{LlmBuilder, LlmError};
use std::sync::Arc;

/// OpenAI-compatible builder for configuring OpenAI-compatible providers.
///
/// Retry: call `.with_retry(RetryOptions::backoff())` to enable unified retry
/// for chat operations across OpenAI-compatible providers.
///
/// This unified builder supports all OpenAI-compatible providers (SiliconFlow, DeepSeek,
/// OpenRouter, etc.) using the adapter system for proper parameter handling and field mapping.
///
/// # Example
/// ```rust,no_run
/// use siumai::builder::LlmBuilder;
/// use std::time::Duration;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // SiliconFlow
///     let client = LlmBuilder::new()
///         .with_timeout(Duration::from_secs(60))
///         .siliconflow()
///         .api_key("your-api-key")
///         .model("deepseek-chat")
///         .temperature(0.7)
///         .build()
///         .await?;
///
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct OpenAiCompatibleBuilder {
    /// Base builder with HTTP configuration
    base: LlmBuilder,
    /// Provider identifier (siliconflow, deepseek, openrouter, etc.)
    provider_id: String,
    /// API key for the provider
    api_key: Option<String>,
    /// Custom base URL (overrides provider default)
    base_url: Option<String>,
    /// Model to use
    model: Option<String>,
    /// Common parameters
    common_params: crate::types::CommonParams,
    /// HTTP configuration
    http_config: crate::types::HttpConfig,
    /// Provider-specific configuration
    provider_specific_config: std::collections::HashMap<String, serde_json::Value>,
    /// Unified retry options
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Enable lightweight HTTP debug logging interceptor
    http_debug: bool,
}

impl OpenAiCompatibleBuilder {
    /// Create a new OpenAI-compatible builder
    pub fn new(base: LlmBuilder, provider_id: &str) -> Self {
        // Get default model from registry
        let default_model =
            crate::providers::openai_compatible::default_models::get_default_chat_model(
                provider_id,
            )
            .map(|model| model.to_string());

        Self {
            base: base.clone(),
            provider_id: provider_id.to_string(),
            api_key: None,
            base_url: None,
            model: default_model,
            common_params: crate::types::CommonParams::default(),
            http_config: crate::types::HttpConfig::default(),
            provider_specific_config: std::collections::HashMap::new(),
            retry_options: None,
            // Inherit interceptors/debug from unified builder
            http_interceptors: base.http_interceptors.clone(),
            http_debug: base.http_debug,
        }
    }

    /// Set the API key
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set a custom base URL
    ///
    /// This allows you to override the default provider base URL, which is useful for:
    /// - Self-deployed OpenAI-compatible servers
    /// - Providers with multiple service endpoints
    /// - Custom proxy or gateway configurations
    ///
    /// # Arguments
    /// * `base_url` - The custom base URL to use (e.g., "https://my-server.com/v1")
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     // Use a self-deployed OpenAI-compatible server
    ///     let client = LlmBuilder::new()
    ///         .deepseek()
    ///         .api_key("your-api-key")
    ///         .base_url("https://my-deepseek-server.com/v1")
    ///         .model("deepseek-chat")
    ///         .build()
    ///         .await?;
    ///
    ///     // Use an alternative endpoint for a provider
    ///     let client2 = LlmBuilder::new()
    ///         .openrouter()
    ///         .api_key("your-api-key")
    ///         .base_url("https://openrouter.ai/api/v1")
    ///         .model("openai/gpt-4")
    ///         .build()
    ///         .await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set top_p
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Set request timeout
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }

    /// Set connection timeout
    pub fn connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }

    /// Set user agent
    pub fn user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        self.http_config.user_agent = Some(user_agent.into());
        self
    }

    /// Set proxy URL
    pub fn proxy<S: Into<String>>(mut self, proxy: S) -> Self {
        self.http_config.proxy = Some(proxy.into());
        self
    }

    /// Set custom headers
    pub fn custom_headers(mut self, headers: std::collections::HashMap<String, String>) -> Self {
        self.http_config.headers = headers;
        self
    }

    /// Add a custom header
    pub fn header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.http_config.headers.insert(key.into(), value.into());
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

    /// Set stop sequences
    pub fn stop<S: Into<String>>(mut self, stop: Vec<S>) -> Self {
        self.common_params.stop_sequences = Some(stop.into_iter().map(|s| s.into()).collect());
        self
    }

    /// Set seed for deterministic outputs
    pub fn seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    /// Set HTTP configuration
    pub fn with_http_config(mut self, config: crate::types::HttpConfig) -> Self {
        self.http_config = config;
        self
    }

    /// Set custom HTTP client
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.base = self.base.with_http_client(client);
        self
    }

    /// Enable thinking mode for supported models (SiliconFlow only)
    ///
    /// When enabled, models that support thinking (like DeepSeek V3.1, Qwen 3, etc.)
    /// will include their reasoning process in the response.
    ///
    /// # Arguments
    /// * `enable` - Whether to enable thinking mode
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = LlmBuilder::new()
    ///         .siliconflow()
    ///         .api_key("your-api-key")
    ///         .model("deepseek-ai/DeepSeek-V3.1")
    ///         .with_thinking(true)
    ///         .build()
    ///         .await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub fn with_thinking(mut self, enable: bool) -> Self {
        // Store thinking preference for later use in adapter creation
        self.provider_specific_config.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(enable),
        );
        self
    }

    /// Set the thinking budget (maximum tokens for reasoning) for supported models (SiliconFlow only)
    ///
    /// This controls how many tokens the model can use for its internal reasoning process.
    /// Higher values allow for more detailed reasoning but consume more tokens.
    ///
    /// # Arguments
    /// * `budget` - Number of tokens (128-32768, default varies by model size)
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = LlmBuilder::new()
    ///         .siliconflow()
    ///         .api_key("your-api-key")
    ///         .model("deepseek-ai/DeepSeek-V3.1")
    ///         .with_thinking_budget(8192)  // 8K tokens for complex reasoning
    ///         .build()
    ///         .await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub fn with_thinking_budget(mut self, budget: u32) -> Self {
        // Clamp to reasonable range and store for later use in adapter creation
        let clamped_budget = budget.clamp(128, 32768);
        self.provider_specific_config.insert(
            "thinking_budget".to_string(),
            serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
        );
        self
    }

    /// Enable reasoning mode for supported models
    ///
    /// This is the unified reasoning interface that works across all OpenAI-compatible providers.
    /// Different providers handle reasoning differently:
    /// - DeepSeek: Uses `reasoning_content` field for thinking output
    /// - OpenRouter: Passes through to underlying model's reasoning capabilities
    /// - SiliconFlow: Maps to `enable_thinking` parameter
    ///
    /// # Arguments
    /// * `enable` - Whether to enable reasoning output
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = LlmBuilder::new()
    ///         .deepseek()
    ///         .api_key("your-api-key")
    ///         .model("deepseek-reasoner")
    ///         .reasoning(true)  // Unified interface
    ///         .build()
    ///         .await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub fn reasoning(mut self, enable: bool) -> Self {
        // Map unified reasoning to provider-specific parameters
        match self.provider_id.as_str() {
            "siliconflow" => {
                // SiliconFlow uses "enable_thinking"
                self.provider_specific_config.insert(
                    "enable_thinking".to_string(),
                    serde_json::Value::Bool(enable),
                );
            }
            "deepseek" | "openrouter" => {
                // DeepSeek and OpenRouter use "enable_reasoning"
                self.provider_specific_config.insert(
                    "enable_reasoning".to_string(),
                    serde_json::Value::Bool(enable),
                );
            }
            _ => {
                // Default to "enable_reasoning" for unknown providers
                self.provider_specific_config.insert(
                    "enable_reasoning".to_string(),
                    serde_json::Value::Bool(enable),
                );
            }
        }
        self
    }

    /// Set reasoning budget
    ///
    /// This controls how many tokens the model can use for its internal reasoning process.
    /// Different providers interpret this differently:
    /// - SiliconFlow: Maps to `thinking_budget` parameter
    /// - DeepSeek: Ignored (uses boolean reasoning mode)
    /// - OpenRouter: Passed through to underlying model
    ///
    /// # Arguments
    /// * `budget` - Number of tokens for reasoning (128-32768)
    ///
    /// # Example
    /// ```rust,no_run
    /// use siumai::prelude::*;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let client = LlmBuilder::new()
    ///         .siliconflow()
    ///         .api_key("your-api-key")
    ///         .model("deepseek-ai/DeepSeek-V3.1")
    ///         .reasoning_budget(8192)  // Unified interface
    ///         .build()
    ///         .await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub fn reasoning_budget(mut self, budget: i32) -> Self {
        // Clamp to reasonable range
        let clamped_budget = budget.clamp(128, 32768) as u32;

        // Map unified reasoning_budget to provider-specific parameters
        match self.provider_id.as_str() {
            "siliconflow" => {
                // SiliconFlow uses "thinking_budget"
                self.provider_specific_config.insert(
                    "thinking_budget".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                );
                // Also enable thinking when budget is set
                self.provider_specific_config
                    .insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
            }
            "deepseek" | "openrouter" => {
                // DeepSeek and OpenRouter: store budget but mainly use boolean mode
                self.provider_specific_config.insert(
                    "reasoning_budget".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                );
                // Also enable reasoning when budget is set
                self.provider_specific_config.insert(
                    "enable_reasoning".to_string(),
                    serde_json::Value::Bool(true),
                );
            }
            _ => {
                // Default behavior for unknown providers
                self.provider_specific_config.insert(
                    "reasoning_budget".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                );
            }
        }
        self
    }

    /// Build the OpenAI-compatible client
    pub async fn build(
        self,
    ) -> Result<crate::providers::openai_compatible::OpenAiCompatibleClient, LlmError> {
        let api_key = self.api_key.ok_or_else(|| {
            LlmError::ConfigurationError(format!("API key is required for {}", self.provider_id))
        })?;

        // Create adapter using the registry (much simpler!)
        let adapter = get_provider_adapter(&self.provider_id)?;

        // Use custom base URL if provided, otherwise use adapter's default
        let base_url = self
            .base_url
            .unwrap_or_else(|| adapter.base_url().to_string());

        // Create configuration
        let mut config = crate::providers::openai_compatible::OpenAiCompatibleConfig::new(
            &self.provider_id,
            &api_key,
            &base_url,
            adapter,
        );

        // Set model if provided
        if let Some(model) = self.model {
            config = config.with_model(&model);
        }

        // Set common parameters
        config = config.with_common_params(self.common_params);

        // Merge HTTP configurations
        let mut final_http_config = self.http_config;

        // Apply base builder HTTP settings
        if let Some(timeout) = self.base.timeout {
            final_http_config.timeout = Some(timeout);
        }
        if let Some(connect_timeout) = self.base.connect_timeout {
            final_http_config.connect_timeout = Some(connect_timeout);
        }
        if let Some(user_agent) = self.base.user_agent {
            final_http_config.user_agent = Some(user_agent);
        }
        if let Some(proxy) = self.base.proxy {
            final_http_config.proxy = Some(proxy);
        }

        // Merge headers from base builder
        for (key, value) in self.base.default_headers {
            final_http_config.headers.insert(key, value);
        }

        config = config.with_http_config(final_http_config);

        // Create client with or without custom HTTP client
        let mut client = if let Some(http_client) = self.base.http_client {
            crate::providers::openai_compatible::OpenAiCompatibleClient::with_http_client(
                config,
                http_client,
            )
            .await?
        } else {
            crate::providers::openai_compatible::OpenAiCompatibleClient::new(config).await?
        };

        client.set_retry_options(self.retry_options.clone());
        // Install interceptors
        let mut interceptors = self.http_interceptors;
        if self.http_debug {
            interceptors.push(Arc::new(LoggingInterceptor::default()));
        }
        if !interceptors.is_empty() {
            client = client.with_http_interceptors(interceptors);
        }
        Ok(client)
    }
}
