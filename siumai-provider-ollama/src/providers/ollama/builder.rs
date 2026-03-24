use crate::builder::{BuilderBase, ProviderCore};
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::provider_options::ollama::OllamaOptions;
use crate::providers::ollama::config::OllamaParams;
use crate::retry_api::RetryOptions;
use crate::{CommonParams, LlmError};
use std::sync::Arc;

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
    extra_model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl OllamaBuilder {
    /// Create a new Ollama builder
    pub fn new(base: BuilderBase) -> Self {
        Self {
            core: ProviderCore::new(base),
            base_url: None,
            model: None,
            common_params: CommonParams::default(),
            ollama_params: OllamaParams::default(),
            extra_model_middlewares: Vec::new(),
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
        let m = model.into();
        // Keep legacy field for compatibility
        self.model = Some(m.clone());
        // Unify: store in common_params as the single source of truth
        self.common_params.model = m;
        self
    }

    /// Seed the builder from a full common-params snapshot.
    pub fn common_params(mut self, params: CommonParams) -> Self {
        if !params.model.is_empty() {
            self.model = Some(params.model.clone());
        }
        self.common_params = params;
        self
    }

    /// Alias for `common_params(...)` on the config-first parity surface.
    pub fn with_common_params(self, params: CommonParams) -> Self {
        self.common_params(params)
    }

    /// Set the temperature for generation
    ///
    /// # Arguments
    /// * `temperature` - Temperature value (0.0 to 2.0)
    pub const fn temperature(mut self, temperature: f64) -> Self {
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
    pub const fn top_p(mut self, top_p: f64) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Replace the full Ollama provider params snapshot.
    pub fn ollama_params(mut self, params: OllamaParams) -> Self {
        self.ollama_params = params;
        self
    }

    /// Alias for `ollama_params(...)` on the config-first parity surface.
    pub fn with_ollama_params(self, params: OllamaParams) -> Self {
        self.ollama_params(params)
    }

    /// Apply typed Ollama default options on the provider-owned builder surface.
    pub fn with_ollama_options(mut self, options: OllamaOptions) -> Self {
        if let Some(keep_alive) = options.keep_alive {
            self.ollama_params.keep_alive = Some(keep_alive);
        }
        if let Some(raw) = options.raw {
            self.ollama_params.raw = Some(raw);
        }
        if let Some(format) = options.format {
            self.ollama_params.format = Some(format);
        }
        for (key, value) in options.extra_params {
            if key == "think"
                && let Some(think) = value.as_bool()
            {
                self.ollama_params.think = Some(think);
                continue;
            }
            let mut merged = self.ollama_params.options.unwrap_or_default();
            merged.insert(key, value);
            self.ollama_params.options = Some(merged);
        }
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

    /// Set provider-native stop sequences (distinct from unified `stop_sequences`).
    pub fn stop(mut self, stop: Vec<String>) -> Self {
        self.ollama_params.stop = Some(stop);
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

    /// Enable or disable provider-native thinking mode.
    pub const fn think(mut self, think: bool) -> Self {
        self.ollama_params.think = Some(think);
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

    /// Set unified retry options for chat operations
    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.core = self.core.with_retry(options);
        self
    }

    /// Replace the full HTTP config snapshot.
    pub fn http_config(mut self, config: crate::types::HttpConfig) -> Self {
        self.core.http_config = config;
        self
    }

    /// Alias for `http_config(...)` on the config-first parity surface.
    pub fn with_http_config(mut self, config: crate::types::HttpConfig) -> Self {
        self.core.http_config = config;
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

    /// Set a custom HTTP transport (Vercel-style `fetch` parity).
    pub fn with_http_transport(
        mut self,
        transport: std::sync::Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.core = self.core.with_http_transport(transport);
        self
    }

    /// Alias for `with_http_transport(...)` (Vercel-aligned: `fetch`).
    pub fn fetch(
        self,
        transport: std::sync::Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.with_http_transport(transport)
    }

    /// Append custom model middlewares after provider auto-middlewares.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.extra_model_middlewares = middlewares;
        self
    }

    /// Convert the builder into the canonical Ollama config.
    pub fn into_config(self) -> Result<crate::providers::ollama::OllamaConfig, LlmError> {
        let base_url = self
            .base_url
            .unwrap_or_else(|| "http://localhost:11434".to_string());
        let model_id = self.common_params.model.clone();

        let mut config = crate::providers::ollama::OllamaConfig::builder()
            .base_url(base_url)
            .model(self.model.unwrap_or(model_id.clone()))
            .common_params(self.common_params)
            .http_config(self.core.http_config.clone())
            .ollama_params(self.ollama_params)
            .http_transport_opt(self.core.http_transport.clone())
            .http_interceptors(self.core.get_http_interceptors())
            .model_middlewares(self.core.get_auto_middlewares("ollama", &model_id))
            .build()?;

        if !self.extra_model_middlewares.is_empty() {
            let mut middlewares = config.model_middlewares.clone();
            middlewares.extend(self.extra_model_middlewares);
            config.model_middlewares = middlewares;
        }

        Ok(config)
    }

    /// Build the Ollama client
    pub async fn build(self) -> Result<crate::providers::ollama::OllamaClient, LlmError> {
        let http_client_override = self.core.base.http_client.clone();
        let tracing_config = self.core.tracing_config.clone();
        let retry_options = self.core.retry_options.clone();
        let config = self.into_config()?;

        let http_interceptors = config.http_interceptors.clone();
        let model_middlewares = config.model_middlewares.clone();

        let mut client = if let Some(http_client) = http_client_override {
            crate::providers::ollama::OllamaClient::new(config, http_client)
                .with_http_interceptors(http_interceptors)
                .with_model_middlewares(model_middlewares)
        } else {
            crate::providers::ollama::OllamaClient::from_config(config)?
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

    #[derive(Clone, Default)]
    struct NoopMiddleware;

    impl LanguageModelMiddleware for NoopMiddleware {}

    #[test]
    fn ollama_builder_into_config_converges_on_ollama_config() {
        let config = OllamaBuilder::new(BuilderBase::default())
            .base_url("http://localhost:11434")
            .model("llama3.2")
            .temperature(0.5)
            .max_tokens(256)
            .timeout(Duration::from_secs(13))
            .connect_timeout(Duration::from_secs(3))
            .http_stream_disable_compression(true)
            .http_debug(true)
            .into_config()
            .expect("into_config ok");

        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model.as_deref(), Some("llama3.2"));
        assert_eq!(config.common_params.model, "llama3.2");
        assert_eq!(config.common_params.temperature, Some(0.5));
        assert_eq!(config.common_params.max_tokens, Some(256));
        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(13)));
        assert_eq!(
            config.http_config.connect_timeout,
            Some(Duration::from_secs(3))
        );
        assert!(config.http_config.stream_disable_compression);
        assert_eq!(config.http_interceptors.len(), 1);
    }

    #[test]
    fn ollama_builder_into_config_matches_manual_ollama_config() {
        let builder_config = OllamaBuilder::new(BuilderBase::default())
            .base_url("http://localhost:11434")
            .model("llama3.2")
            .temperature(0.5)
            .max_tokens(256)
            .keep_alive("5m")
            .timeout(Duration::from_secs(13))
            .connect_timeout(Duration::from_secs(3))
            .http_stream_disable_compression(true)
            .http_debug(true)
            .into_config()
            .expect("builder config");

        let manual_config = crate::providers::ollama::OllamaConfig::builder()
            .base_url("http://localhost:11434")
            .model("llama3.2")
            .common_params(crate::types::CommonParams {
                model: "llama3.2".to_string(),
                temperature: Some(0.5),
                max_tokens: Some(256),
                ..Default::default()
            })
            .timeout(Duration::from_secs(13))
            .connect_timeout(Duration::from_secs(3))
            .http_stream_disable_compression(true)
            .keep_alive("5m")
            .http_interceptor(Arc::new(
                crate::execution::http::interceptor::LoggingInterceptor,
            ))
            .model_middlewares(crate::execution::middleware::build_auto_middlewares_vec(
                "ollama", "llama3.2",
            ))
            .build()
            .expect("manual config");

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
            builder_config.ollama_params.keep_alive,
            manual_config.ollama_params.keep_alive
        );
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
            builder_config.http_interceptors.len(),
            manual_config.http_interceptors.len()
        );
        assert_eq!(
            builder_config.model_middlewares.len(),
            manual_config.model_middlewares.len()
        );
    }

    #[test]
    fn ollama_builder_bulk_seed_and_overrides_converge_on_provider_config() {
        let mut options = std::collections::HashMap::new();
        options.insert("temperature".to_string(), serde_json::json!(0.7));

        let config = OllamaBuilder::new(BuilderBase::default())
            .common_params(crate::types::CommonParams {
                model: "llama3.2".to_string(),
                temperature: Some(0.2),
                max_tokens: Some(128),
                top_p: Some(0.8),
                stop_sequences: Some(vec!["END".to_string()]),
                ..Default::default()
            })
            .ollama_params(crate::providers::ollama::config::OllamaParams {
                keep_alive: Some("5m".to_string()),
                raw: Some(true),
                think: Some(true),
                stop: Some(vec!["HALT".to_string()]),
                options: Some(options),
                ..Default::default()
            })
            .model("llama3.2:latest")
            .top_p(0.9)
            .think(false)
            .option("top_k", serde_json::json!(40))
            .http_config(crate::types::HttpConfig {
                timeout: Some(Duration::from_secs(9)),
                ..Default::default()
            })
            .http_debug(true)
            .with_model_middlewares(vec![Arc::new(NoopMiddleware)])
            .into_config()
            .expect("into_config ok");

        assert_eq!(config.model.as_deref(), Some("llama3.2:latest"));
        assert_eq!(config.common_params.model, "llama3.2:latest");
        assert_eq!(config.common_params.temperature, Some(0.2));
        assert_eq!(config.common_params.max_tokens, Some(128));
        assert_eq!(config.common_params.top_p, Some(0.9));
        assert_eq!(
            config.common_params.stop_sequences,
            Some(vec!["END".to_string()])
        );
        assert_eq!(config.ollama_params.keep_alive.as_deref(), Some("5m"));
        assert_eq!(config.ollama_params.raw, Some(true));
        assert_eq!(config.ollama_params.think, Some(false));
        assert_eq!(config.ollama_params.stop, Some(vec!["HALT".to_string()]));
        assert_eq!(
            config
                .ollama_params
                .options
                .as_ref()
                .and_then(|map| map.get("temperature")),
            Some(&serde_json::json!(0.7))
        );
        assert_eq!(
            config
                .ollama_params
                .options
                .as_ref()
                .and_then(|map| map.get("top_k")),
            Some(&serde_json::json!(40))
        );
        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(9)));
        assert_eq!(config.http_interceptors.len(), 1);
        assert!(!config.model_middlewares.is_empty());
        assert!(
            config.model_middlewares.len()
                > crate::execution::middleware::build_auto_middlewares_vec(
                    "ollama",
                    "llama3.2:latest"
                )
                .len()
        );
    }

    #[test]
    fn ollama_builder_with_ollama_options_merges_into_params() {
        let config = OllamaBuilder::new(BuilderBase::default())
            .model("llama3.2")
            .with_ollama_options(
                OllamaOptions::new()
                    .with_keep_alive("1m")
                    .with_raw_mode(true)
                    .with_format("json")
                    .with_param("think", serde_json::json!(true))
                    .with_param("num_ctx", serde_json::json!(4096)),
            )
            .into_config()
            .expect("into_config ok");

        assert_eq!(config.ollama_params.keep_alive, Some("1m".to_string()));
        assert_eq!(config.ollama_params.raw, Some(true));
        assert_eq!(config.ollama_params.format, Some("json".to_string()));
        assert_eq!(config.ollama_params.think, Some(true));
        assert_eq!(
            config
                .ollama_params
                .options
                .as_ref()
                .and_then(|options| options.get("num_ctx")),
            Some(&serde_json::json!(4096))
        );
    }
}
