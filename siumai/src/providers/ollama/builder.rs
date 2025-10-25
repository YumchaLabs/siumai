use crate::core::builder_core::ProviderCore;
use crate::providers::ollama::config::OllamaParams;
use crate::retry_api::RetryOptions;
use crate::{CommonParams, LlmBuilder, LlmError};

/// Ollama-specific builder
///
/// Retry: call `.with_retry(RetryOptions::backoff())` to enable unified retry
/// for chat operations.
pub struct OllamaBuilder {
    /// Core provider configuration (composition)
    pub(crate) core: ProviderCore,
    base_url: Option<String>,
    model: Option<String>,
    common_params: CommonParams,
    ollama_params: OllamaParams,
}

impl OllamaBuilder {
    /// Create a new Ollama builder
    pub fn new(base: LlmBuilder) -> Self {
        Self {
            core: ProviderCore::new(base),
            base_url: None,
            model: None,
            common_params: CommonParams::default(),
            ollama_params: OllamaParams::default(),
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

    // === HTTP Configuration ===

    /// Control whether to disable compression for streaming (SSE) requests.
    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.core = self.core.http_stream_disable_compression(disable);
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

    /// Set unified retry options for chat operations
    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.core = self.core.with_retry(options);
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

    /// Build the Ollama client
    pub async fn build(self) -> Result<crate::providers::ollama::OllamaClient, LlmError> {
        // Step 1: Get API key (not required for Ollama)
        // Note: Ollama typically doesn't require an API key

        // Step 2: Get base URL (from parameter or default)
        let base_url = self
            .base_url
            .unwrap_or_else(|| "http://localhost:11434".to_string());

        // Note: Tracing initialization has been moved to siumai-extras.
        // Users should initialize tracing manually using siumai_extras::telemetry
        // or tracing_subscriber directly before creating the client.

        // Step 3: Build configuration
        let model_for_middleware = self.model.clone();
        let model_from_common_params = self.common_params.model.clone();

        let http_config_clone = self.core.http_config.clone();

        let mut config = crate::providers::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .common_params(self.common_params)
            .http_config(http_config_clone.clone())
            .ollama_params(self.ollama_params);

        if let Some(model) = self.model {
            config = config.model(model);
        }

        let config = config.build()?;

        // Step 4: Build HTTP client from core
        let http_client = self.core.build_http_client()?;

        // Step 5: Create client instance
        let mut client = crate::providers::ollama::OllamaClient::new(config, http_client);

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
        let model_id = model_for_middleware
            .as_deref()
            .unwrap_or(&model_from_common_params);
        let middlewares = self.core.get_auto_middlewares("ollama", model_id);
        if !middlewares.is_empty() {
            client = client.with_model_middlewares(middlewares);
        }

        Ok(client)
    }
}
