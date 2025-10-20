use crate::providers::ollama::config::OllamaParams;
use crate::retry_api::RetryOptions;
use crate::utils::http_interceptor::{HttpInterceptor, LoggingInterceptor};
use crate::{CommonParams, HttpConfig, LlmBuilder, LlmError};
use std::sync::Arc;

/// Ollama-specific builder
///
/// Retry: call `.with_retry(RetryOptions::backoff())` to enable unified retry
/// for chat operations.
pub struct OllamaBuilder {
    pub(crate) base: LlmBuilder,
    base_url: Option<String>,
    model: Option<String>,
    common_params: CommonParams,
    ollama_params: OllamaParams,
    http_config: HttpConfig,
    tracing_config: Option<crate::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Enable lightweight HTTP debug logging interceptor
    http_debug: bool,
}

impl OllamaBuilder {
    /// Create a new Ollama builder
    pub fn new(base: LlmBuilder) -> Self {
        // Inherit interceptors/debug from unified builder
        let inherited_interceptors = base.http_interceptors.clone();
        let inherited_debug = base.http_debug;

        Self {
            base,
            base_url: None,
            model: None,
            common_params: CommonParams::default(),
            ollama_params: OllamaParams::default(),
            http_config: HttpConfig::default(),
            tracing_config: None,
            retry_options: None,
            http_interceptors: inherited_interceptors,
            http_debug: inherited_debug,
        }
    }

    /// Set the base URL for Ollama API
    ///
    /// # Arguments
    /// * `url` - The base URL (e.g., "<http://localhost:11434>")
    pub fn base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the model to use
    ///
    /// # Arguments
    /// * `model` - The model name (e.g., "llama3.2", "mistral:7b")
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the temperature for generation
    ///
    /// # Arguments
    /// * `temperature` - Temperature value (0.0 to 2.0)
    pub const fn temperature(mut self, temperature: f32) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set the maximum number of tokens to generate
    ///
    /// # Arguments
    /// * `max_tokens` - Maximum tokens to generate
    pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top-p value for nucleus sampling
    ///
    /// # Arguments
    /// * `top_p` - Top-p value (0.0 to 1.0)
    pub const fn top_p(mut self, top_p: f32) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Set how long to keep the model loaded in memory
    ///
    /// # Arguments
    /// * `duration` - Duration string (e.g., "5m", "1h", "30s")
    pub fn keep_alive<S: Into<String>>(mut self, duration: S) -> Self {
        self.ollama_params.keep_alive = Some(duration.into());
        self
    }

    /// Enable or disable raw mode (bypass templating)
    ///
    /// # Arguments
    /// * `raw` - Whether to enable raw mode
    pub const fn raw(mut self, raw: bool) -> Self {
        self.ollama_params.raw = Some(raw);
        self
    }

    /// Set the output format
    ///
    /// # Arguments
    /// * `format` - Format string ("json" or JSON schema)
    pub fn format<S: Into<String>>(mut self, format: S) -> Self {
        self.ollama_params.format = Some(format.into());
        self
    }

    /// Add a model option
    ///
    /// # Arguments
    /// * `key` - Option key
    /// * `value` - Option value
    pub fn option<K: Into<String>>(mut self, key: K, value: serde_json::Value) -> Self {
        let mut options = self.ollama_params.options.unwrap_or_default();
        options.insert(key.into(), value);
        self.ollama_params.options = Some(options);
        self
    }

    /// Set multiple model options at once
    ///
    /// # Arguments
    /// * `options` - `HashMap` of options
    pub fn options(
        mut self,
        options: std::collections::HashMap<String, serde_json::Value>,
    ) -> Self {
        self.ollama_params.options = Some(options);
        self
    }

    /// Enable or disable NUMA support
    ///
    /// # Arguments
    /// * `numa` - Whether to enable NUMA support
    pub const fn numa(mut self, numa: bool) -> Self {
        self.ollama_params.numa = Some(numa);
        self
    }

    /// Set the context window size
    ///
    /// # Arguments
    /// * `num_ctx` - Context window size
    pub const fn num_ctx(mut self, num_ctx: u32) -> Self {
        self.ollama_params.num_ctx = Some(num_ctx);
        self
    }

    /// Set the number of GPU layers to use
    ///
    /// # Arguments
    /// * `num_gpu` - Number of GPU layers
    pub const fn num_gpu(mut self, num_gpu: u32) -> Self {
        self.ollama_params.num_gpu = Some(num_gpu);
        self
    }

    /// Set the batch size for processing
    ///
    /// # Arguments
    /// * `num_batch` - Batch size
    pub const fn num_batch(mut self, num_batch: u32) -> Self {
        self.ollama_params.num_batch = Some(num_batch);
        self
    }

    /// Set the main GPU to use
    ///
    /// # Arguments
    /// * `main_gpu` - Main GPU index
    pub const fn main_gpu(mut self, main_gpu: u32) -> Self {
        self.ollama_params.main_gpu = Some(main_gpu);
        self
    }

    /// Enable or disable memory mapping
    ///
    /// # Arguments
    /// * `use_mmap` - Whether to use memory mapping
    pub const fn use_mmap(mut self, use_mmap: bool) -> Self {
        self.ollama_params.use_mmap = Some(use_mmap);
        self
    }

    /// Set the number of threads to use
    ///
    /// # Arguments
    /// * `num_thread` - Number of threads
    pub const fn num_thread(mut self, num_thread: u32) -> Self {
        self.ollama_params.num_thread = Some(num_thread);
        self
    }

    /// Enable reasoning mode for reasoning models
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable reasoning mode
    pub const fn reasoning(mut self, enabled: bool) -> Self {
        self.ollama_params.think = Some(enabled);
        self
    }

    /// Enable thinking mode for thinking models (alias for reasoning)
    ///
    /// # Arguments
    /// * `think` - Whether to enable thinking mode
    ///
    /// # Deprecated
    /// Use `reasoning()` instead for consistency with other providers
    #[deprecated(since = "0.7.1", note = "Use `reasoning()` instead for consistency")]
    pub const fn think(self, think: bool) -> Self {
        self.reasoning(think)
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

    /// Build the Ollama client
    pub async fn build(self) -> Result<crate::providers::ollama::OllamaClient, LlmError> {
        let base_url = self
            .base_url
            .unwrap_or_else(|| "http://localhost:11434".to_string());

        // Note: Tracing initialization has been moved to siumai-extras.
        // Users should initialize tracing manually using siumai_extras::telemetry
        // or tracing_subscriber directly before creating the client.

        // Save model before moving common_params
        let model_for_middleware = self.model.clone();
        let model_from_common_params = self.common_params.model.clone();

        let mut config = crate::providers::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .common_params(self.common_params)
            .http_config(self.http_config)
            .ollama_params(self.ollama_params);

        if let Some(model) = self.model {
            config = config.model(model);
        }

        let config = config.build()?;
        let http_client = self.base.build_http_client()?;

        let mut client = crate::providers::ollama::OllamaClient::new(config, http_client);
        client.set_tracing_config(self.tracing_config);
        client.set_retry_options(self.retry_options.clone());

        // Step 7: Install HTTP interceptors
        let mut interceptors = self.http_interceptors;
        if self.http_debug {
            interceptors.push(Arc::new(LoggingInterceptor));
        }
        if !interceptors.is_empty() {
            client = client.with_http_interceptors(interceptors);
        }

        // Step 8: Install automatic middlewares based on provider and model
        let model_id = model_for_middleware
            .as_deref()
            .unwrap_or(&model_from_common_params);
        let middlewares = crate::middleware::build_auto_middlewares_vec("ollama", model_id);
        client = client.with_model_middlewares(middlewares);

        Ok(client)
    }
}
