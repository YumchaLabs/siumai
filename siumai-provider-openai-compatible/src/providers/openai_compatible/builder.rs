use crate::LlmError;
use crate::builder::BuilderBase;
use crate::execution::http::interceptor::{HttpInterceptor, LoggingInterceptor};
use crate::execution::http::transport::HttpTransport;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
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
    base: BuilderBase,
    /// Provider identifier (siliconflow, deepseek, openrouter, etc.)
    provider_id: String,
    /// API key for the provider
    api_key: Option<String>,
    /// Custom base URL (overrides provider default)
    base_url: Option<String>,
    /// Common parameters
    common_params: crate::types::CommonParams,
    /// HTTP configuration
    http_config: crate::types::HttpConfig,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    http_transport: Option<Arc<dyn HttpTransport>>,
    /// Provider-specific configuration
    provider_specific_config: std::collections::HashMap<String, serde_json::Value>,
    /// Unified retry options
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Additional model middlewares appended after provider auto-middlewares.
    extra_model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Enable lightweight HTTP debug logging interceptor
    http_debug: bool,
}

impl OpenAiCompatibleBuilder {
    /// Create a new OpenAI-compatible builder
    pub fn new(base: BuilderBase, provider_id: &str) -> Self {
        Self {
            base: base.clone(),
            provider_id: provider_id.to_string(),
            api_key: None,
            base_url: None,
            common_params: {
                let mut cp = crate::types::CommonParams::default();
                if let Some(m) =
                    crate::providers::openai_compatible::default_models::get_default_chat_model(
                        provider_id,
                    )
                {
                    cp.model = m.to_string();
                }
                cp
            },
            http_config: crate::types::HttpConfig::default(),
            http_transport: None,
            provider_specific_config: std::collections::HashMap::new(),
            retry_options: None,
            // Inherit interceptors/debug from unified builder
            http_interceptors: base.http_interceptors.clone(),
            extra_model_middlewares: Vec::new(),
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
        self.common_params.model = model.into();
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set top_p
    pub fn top_p(mut self, top_p: f64) -> Self {
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

    /// Control whether to disable compression for streaming (SSE) requests.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
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
        self.base.http_client = Some(client);
        self
    }

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_http_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.http_transport = Some(transport);
        self
    }

    /// Append extra model middlewares after provider auto-middlewares.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.extra_model_middlewares = middlewares;
        self
    }

    /// Alias for `with_http_transport(...)` (Vercel-aligned: `fetch`).
    pub fn fetch(self, transport: Arc<dyn HttpTransport>) -> Self {
        self.with_http_transport(transport)
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

    /// Convert the builder into the canonical OpenAI-compatible config.
    pub fn into_config(
        self,
    ) -> Result<crate::providers::openai_compatible::OpenAiCompatibleConfig, LlmError> {
        let provider_config =
            crate::providers::openai_compatible::config::get_provider_config(&self.provider_id)
                .ok_or_else(|| {
                    LlmError::ConfigurationError(format!(
                        "Unknown OpenAI-compatible provider id: {}",
                        self.provider_id
                    ))
                })?;

        let api_key = crate::utils::builder_helpers::get_api_key_with_envs(
            self.api_key,
            &self.provider_id,
            provider_config.api_key_env.as_deref(),
            &provider_config.api_key_env_aliases,
        )?;
        let mut adapter: Box<dyn crate::providers::openai_compatible::ProviderAdapter> = Box::new(
            crate::standards::openai::compat::provider_registry::ConfigurableAdapter::new(
                provider_config,
            ),
        );
        if !self.provider_specific_config.is_empty() {
            adapter = Box::new(
                crate::standards::openai::compat::adapter::ParamMergingAdapter::new(
                    adapter,
                    self.provider_specific_config,
                ),
            );
        }
        let adapter: Arc<dyn crate::providers::openai_compatible::ProviderAdapter> =
            Arc::from(adapter);

        let base_url = crate::utils::builder_helpers::resolve_base_url(
            self.base_url.clone(),
            adapter.base_url(),
        );

        let mut config = crate::providers::openai_compatible::OpenAiCompatibleConfig::new(
            &self.provider_id,
            &api_key,
            &base_url,
            adapter,
        );

        let effective_model_raw = crate::utils::builder_helpers::get_effective_model(
            &self.common_params.model,
            &self.provider_id,
        );
        let effective_model = crate::utils::builder_helpers::normalize_model_id(
            &self.provider_id,
            &effective_model_raw,
        );

        if !effective_model.is_empty() {
            config = config.with_model(&effective_model);
        }

        config = config.with_common_params(self.common_params);

        let mut final_http_config = self.http_config;
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
        for (key, value) in self.base.default_headers {
            final_http_config.headers.insert(key, value);
        }

        config = config.with_http_config(final_http_config);
        if let Some(transport) = self.http_transport.clone() {
            config = config.with_http_transport(transport);
        }

        let model_id = config.model.clone();
        let mut interceptors = self.http_interceptors;
        if self.http_debug {
            interceptors.push(Arc::new(LoggingInterceptor));
        }
        let mut middlewares =
            crate::execution::middleware::build_auto_middlewares_vec(&self.provider_id, &model_id);
        middlewares.extend(self.extra_model_middlewares);

        Ok(config
            .with_http_interceptors(interceptors)
            .with_model_middlewares(middlewares))
    }

    /// Build the OpenAI-compatible client
    pub async fn build(
        self,
    ) -> Result<crate::providers::openai_compatible::OpenAiCompatibleClient, LlmError> {
        let http_client_override = self.base.http_client.clone();
        let retry_options = self.retry_options.clone();
        let config = self.into_config()?;

        let mut client = if let Some(http_client) = http_client_override {
            let http_interceptors = config.http_interceptors.clone();
            let model_middlewares = config.model_middlewares.clone();
            crate::providers::openai_compatible::OpenAiCompatibleClient::with_http_client(
                config,
                http_client,
            )
            .await?
            .with_http_interceptors(http_interceptors)
            .with_model_middlewares(model_middlewares)
        } else {
            crate::providers::openai_compatible::OpenAiCompatibleClient::from_config(config).await?
        };

        client.set_retry_options(retry_options);
        Ok(client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::middleware::language_model::LanguageModelMiddleware;
    use std::sync::Arc;
    use std::time::Duration;

    #[derive(Clone, Default)]
    struct NoopMiddleware;

    impl LanguageModelMiddleware for NoopMiddleware {}

    #[test]
    fn openai_compatible_builder_into_config_converges() {
        let config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .api_key("test-key")
            .model("deepseek-chat")
            .temperature(0.4)
            .max_tokens(256)
            .top_p(0.9)
            .stop(vec!["END"])
            .seed(7)
            .reasoning(true)
            .reasoning_budget(2048)
            .timeout(Duration::from_secs(15))
            .connect_timeout(Duration::from_secs(5))
            .http_stream_disable_compression(true)
            .user_agent("siumai-test/1.0")
            .proxy("http://127.0.0.1:8080")
            .custom_headers(std::collections::HashMap::from([(
                "x-one".to_string(),
                "1".to_string(),
            )]))
            .header("x-two", "2")
            .with_http_interceptor(Arc::new(
                crate::execution::http::interceptor::LoggingInterceptor,
            ))
            .http_debug(true)
            .into_config()
            .expect("into_config ok");

        assert_eq!(config.provider_id, "deepseek");
        assert_eq!(config.model, "deepseek-chat");
        assert_eq!(config.common_params.temperature, Some(0.4));
        assert_eq!(config.common_params.max_tokens, Some(256));
        assert_eq!(config.common_params.top_p, Some(0.9));
        assert_eq!(
            config.common_params.stop_sequences,
            Some(vec!["END".to_string()])
        );
        assert_eq!(config.common_params.seed, Some(7));
        let mut params = serde_json::json!({});
        config
            .adapter
            .transform_request_params(
                &mut params,
                &config.model,
                crate::providers::openai_compatible::RequestType::Chat,
            )
            .expect("transform request params");
        assert_eq!(params["enable_reasoning"], serde_json::json!(true));
        assert_eq!(params["reasoning_budget"], serde_json::json!(2048));
        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(15)));
        assert_eq!(
            config.http_config.connect_timeout,
            Some(Duration::from_secs(5))
        );
        assert!(config.http_config.stream_disable_compression);
        assert_eq!(
            config.http_config.user_agent.as_deref(),
            Some("siumai-test/1.0")
        );
        assert_eq!(
            config.http_config.proxy.as_deref(),
            Some("http://127.0.0.1:8080")
        );
        assert_eq!(
            config.http_config.headers.get("x-one"),
            Some(&"1".to_string())
        );
        assert_eq!(
            config.http_config.headers.get("x-two"),
            Some(&"2".to_string())
        );
        assert_eq!(config.http_interceptors.len(), 2);
    }

    #[test]
    fn openai_compatible_builder_into_config_matches_manual_compatible_config() {
        let builder_config = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .api_key("test-key")
            .model("deepseek-chat")
            .temperature(0.4)
            .max_tokens(256)
            .top_p(0.9)
            .stop(vec!["END"])
            .seed(7)
            .reasoning(true)
            .reasoning_budget(2048)
            .timeout(Duration::from_secs(15))
            .connect_timeout(Duration::from_secs(5))
            .http_stream_disable_compression(true)
            .user_agent("siumai-test/1.0")
            .proxy("http://127.0.0.1:8080")
            .custom_headers(std::collections::HashMap::from([(
                "x-one".to_string(),
                "1".to_string(),
            )]))
            .header("x-two", "2")
            .with_http_interceptor(Arc::new(
                crate::execution::http::interceptor::LoggingInterceptor,
            ))
            .http_debug(true)
            .with_model_middlewares(vec![Arc::new(NoopMiddleware)])
            .into_config()
            .expect("builder config");

        let provider = crate::providers::openai_compatible::get_provider_config("deepseek")
            .expect("provider config");
        let adapter =
            Arc::new(crate::providers::openai_compatible::ConfigurableAdapter::new(provider));
        let manual_config = crate::providers::openai_compatible::OpenAiCompatibleConfig::new(
            "deepseek",
            "test-key",
            "https://api.deepseek.com",
            adapter,
        )
        .with_model("deepseek-chat")
        .with_temperature(0.4)
        .with_max_tokens(256)
        .with_top_p(0.9)
        .with_stop_sequences(vec!["END".to_string()])
        .with_seed(7)
        .with_reasoning(true)
        .with_reasoning_budget(2048)
        .with_timeout(Duration::from_secs(15))
        .with_connect_timeout(Duration::from_secs(5))
        .with_http_stream_disable_compression(true)
        .with_user_agent("siumai-test/1.0")
        .with_proxy("http://127.0.0.1:8080")
        .with_http_headers(std::collections::HashMap::from([(
            "x-one".to_string(),
            "1".to_string(),
        )]))
        .with_http_header("x-two", "2")
        .with_http_interceptor(Arc::new(
            crate::execution::http::interceptor::LoggingInterceptor,
        ))
        .with_http_interceptor(Arc::new(
            crate::execution::http::interceptor::LoggingInterceptor,
        ))
        .with_model_middlewares({
            let mut middlewares = crate::execution::middleware::build_auto_middlewares_vec(
                "deepseek",
                "deepseek-chat",
            );
            middlewares.push(Arc::new(NoopMiddleware));
            middlewares
        });

        assert_eq!(builder_config.provider_id, manual_config.provider_id);
        assert_eq!(builder_config.base_url, manual_config.base_url);
        assert_eq!(builder_config.model, manual_config.model);
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
            builder_config.common_params.stop_sequences,
            manual_config.common_params.stop_sequences
        );
        assert_eq!(
            builder_config.common_params.seed,
            manual_config.common_params.seed
        );
        let mut builder_params = serde_json::json!({});
        builder_config
            .adapter
            .transform_request_params(
                &mut builder_params,
                &builder_config.model,
                crate::providers::openai_compatible::RequestType::Chat,
            )
            .expect("builder transform request params");
        let mut manual_params = serde_json::json!({});
        manual_config
            .adapter
            .transform_request_params(
                &mut manual_params,
                &manual_config.model,
                crate::providers::openai_compatible::RequestType::Chat,
            )
            .expect("manual transform request params");
        assert_eq!(builder_params, manual_params);
        assert_eq!(
            builder_config.http_config.timeout,
            manual_config.http_config.timeout
        );
        assert_eq!(
            builder_config.http_config.connect_timeout,
            manual_config.http_config.connect_timeout
        );
        assert_eq!(
            builder_config.http_config.stream_disable_compression,
            manual_config.http_config.stream_disable_compression
        );
        assert_eq!(
            builder_config.http_config.user_agent,
            manual_config.http_config.user_agent
        );
        assert_eq!(
            builder_config.http_config.proxy,
            manual_config.http_config.proxy
        );
        assert_eq!(
            builder_config.http_config.headers,
            manual_config.http_config.headers
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

    #[tokio::test]
    async fn openai_compatible_builder_build_preserves_http_client_override_and_retry_options() {
        let client = OpenAiCompatibleBuilder::new(BuilderBase::default(), "deepseek")
            .api_key("test-key")
            .model("deepseek-chat")
            .with_http_config(crate::types::HttpConfig {
                proxy: Some("not-a-url".to_string()),
                ..Default::default()
            })
            .with_http_client(reqwest::Client::new())
            .with_retry(RetryOptions::default())
            .build()
            .await
            .expect("build client with explicit http client");

        assert!(client.retry_options().is_some());
    }

    #[test]
    fn openai_compatible_builder_falls_back_to_provider_config_default_model() {
        let mistral = OpenAiCompatibleBuilder::new(BuilderBase::default(), "mistral")
            .api_key("test-key")
            .into_config()
            .expect("mistral config");
        let cohere = OpenAiCompatibleBuilder::new(BuilderBase::default(), "cohere")
            .api_key("test-key")
            .into_config()
            .expect("cohere config");

        assert_eq!(mistral.model, "mistral-large-latest");
        assert_eq!(mistral.common_params.model, mistral.model);
        assert_eq!(cohere.model, "command-r-plus");
        assert_eq!(cohere.common_params.model, cohere.model);
    }
}
