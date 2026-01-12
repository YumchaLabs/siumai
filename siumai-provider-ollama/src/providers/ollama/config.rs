//! Ollama Configuration
//!
//! Configuration structures and builders for Ollama provider.

use crate::error::LlmError;
use crate::types::{CommonParams, HttpConfig};
use std::sync::Arc;

pub use crate::standards::ollama::params::OllamaParams;

/// Ollama provider configuration
#[derive(Clone)]
pub struct OllamaConfig {
    /// Base URL for Ollama API (default: <http://localhost:11434>)
    pub base_url: String,
    /// Default model to use
    pub model: Option<String>,
    /// Common parameters shared across providers
    pub common_params: CommonParams,
    /// HTTP configuration
    pub http_config: HttpConfig,
    /// Ollama-specific parameters
    pub ollama_params: OllamaParams,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
}

impl std::fmt::Debug for OllamaConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaConfig")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("has_http_transport", &self.http_transport.is_some())
            .finish()
    }
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: None,
            common_params: CommonParams::default(),
            http_config: HttpConfig::default(),
            ollama_params: OllamaParams::default(),
            http_transport: None,
        }
    }
}

impl OllamaConfig {
    /// Create a new Ollama configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for Ollama configuration
    pub fn builder() -> OllamaConfigBuilder {
        OllamaConfigBuilder::new()
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), LlmError> {
        if self.base_url.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Base URL cannot be empty".to_string(),
            ));
        }

        // Validate URL format
        if !self.base_url.starts_with("http://") && !self.base_url.starts_with("https://") {
            return Err(LlmError::ConfigurationError(
                "Base URL must start with http:// or https://".to_string(),
            ));
        }

        Ok(())
    }
}

/// Builder for Ollama configuration
#[derive(Default)]
pub struct OllamaConfigBuilder {
    base_url: Option<String>,
    model: Option<String>,
    common_params: Option<CommonParams>,
    http_config: Option<HttpConfig>,
    ollama_params: Option<OllamaParams>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
}

impl std::fmt::Debug for OllamaConfigBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaConfigBuilder")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("common_params", &self.common_params)
            .field("http_config", &self.http_config)
            .field("ollama_params", &self.ollama_params)
            .field("has_http_transport", &self.http_transport.is_some())
            .finish()
    }
}

impl OllamaConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the base URL
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the default model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set common parameters
    pub fn common_params(mut self, params: CommonParams) -> Self {
        self.common_params = Some(params);
        self
    }

    /// Set HTTP configuration
    pub fn http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = Some(config);
        self
    }

    /// Set Ollama-specific parameters
    pub fn ollama_params(mut self, params: OllamaParams) -> Self {
        self.ollama_params = Some(params);
        self
    }

    /// Set keep alive duration
    pub fn keep_alive<S: Into<String>>(mut self, duration: S) -> Self {
        let mut params = self.ollama_params.unwrap_or_default();
        params.keep_alive = Some(duration.into());
        self.ollama_params = Some(params);
        self
    }

    /// Enable raw mode
    pub fn raw(mut self, raw: bool) -> Self {
        let mut params = self.ollama_params.unwrap_or_default();
        params.raw = Some(raw);
        self.ollama_params = Some(params);
        self
    }

    /// Set output format
    pub fn format<S: Into<String>>(mut self, format: S) -> Self {
        let mut params = self.ollama_params.unwrap_or_default();
        params.format = Some(format.into());
        self.ollama_params = Some(params);
        self
    }

    /// Set stop sequences
    pub fn stop(mut self, stop: Vec<String>) -> Self {
        let mut params = self.ollama_params.unwrap_or_default();
        params.stop = Some(stop);
        self.ollama_params = Some(params);
        self
    }

    /// Enable or disable NUMA support
    pub fn numa(mut self, numa: bool) -> Self {
        let mut params = self.ollama_params.unwrap_or_default();
        params.numa = Some(numa);
        self.ollama_params = Some(params);
        self
    }

    /// Set context window size
    pub fn num_ctx(mut self, num_ctx: u32) -> Self {
        let mut params = self.ollama_params.unwrap_or_default();
        params.num_ctx = Some(num_ctx);
        self.ollama_params = Some(params);
        self
    }

    /// Set batch size for processing
    pub fn num_batch(mut self, num_batch: u32) -> Self {
        let mut params = self.ollama_params.unwrap_or_default();
        params.num_batch = Some(num_batch);
        self.ollama_params = Some(params);
        self
    }

    /// Set number of GPU layers to use
    pub fn num_gpu(mut self, num_gpu: u32) -> Self {
        let mut params = self.ollama_params.unwrap_or_default();
        params.num_gpu = Some(num_gpu);
        self.ollama_params = Some(params);
        self
    }

    /// Set main GPU to use
    pub fn main_gpu(mut self, main_gpu: u32) -> Self {
        let mut params = self.ollama_params.unwrap_or_default();
        params.main_gpu = Some(main_gpu);
        self.ollama_params = Some(params);
        self
    }

    /// Enable or disable memory mapping
    pub fn use_mmap(mut self, use_mmap: bool) -> Self {
        let mut params = self.ollama_params.unwrap_or_default();
        params.use_mmap = Some(use_mmap);
        self.ollama_params = Some(params);
        self
    }

    /// Set number of threads to use
    pub fn num_thread(mut self, num_thread: u32) -> Self {
        let mut params = self.ollama_params.unwrap_or_default();
        params.num_thread = Some(num_thread);
        self.ollama_params = Some(params);
        self
    }

    /// Enable thinking mode for thinking models
    pub fn think(mut self, think: bool) -> Self {
        let mut params = self.ollama_params.unwrap_or_default();
        params.think = Some(think);
        self.ollama_params = Some(params);
        self
    }

    /// Add model option
    pub fn option<K: Into<String>>(mut self, key: K, value: serde_json::Value) -> Self {
        let mut params = self.ollama_params.unwrap_or_default();
        let mut options = params.options.unwrap_or_default();
        options.insert(key.into(), value);
        params.options = Some(options);
        self.ollama_params = Some(params);
        self
    }

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn http_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.http_transport = Some(transport);
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<OllamaConfig, LlmError> {
        let mut common_params = self.common_params.unwrap_or_default();

        // Sync model from config.model to common_params.model if set
        if let Some(ref model) = self.model {
            common_params.model = model.clone();
        }

        let config = OllamaConfig {
            base_url: self
                .base_url
                .unwrap_or_else(|| "http://localhost:11434".to_string()),
            model: self.model,
            common_params,
            http_config: self.http_config.unwrap_or_default(),
            ollama_params: self.ollama_params.unwrap_or_default(),
            http_transport: self.http_transport,
        };

        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert!(config.model.is_none());
    }

    #[test]
    fn test_config_builder() {
        let config = OllamaConfig::builder()
            .base_url("http://localhost:11434")
            .model("llama3.2")
            .keep_alive("10m")
            .raw(true)
            .format("json")
            .think(true)
            .option(
                "temperature",
                serde_json::Value::Number(serde_json::Number::from_f64(0.7).unwrap()),
            )
            .build()
            .unwrap();

        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, Some("llama3.2".to_string()));
        assert_eq!(config.ollama_params.keep_alive, Some("10m".to_string()));
        assert_eq!(config.ollama_params.raw, Some(true));
        assert_eq!(config.ollama_params.format, Some("json".to_string()));
        assert_eq!(config.ollama_params.think, Some(true));
    }

    #[test]
    fn test_thinking_model_config() {
        let config = OllamaConfig::builder()
            .base_url("http://localhost:11434")
            .model("deepseek-r1:latest")
            .think(true)
            .num_ctx(4096)
            .option(
                "temperature",
                serde_json::Value::Number(serde_json::Number::from_f64(0.7).unwrap()),
            )
            .build()
            .unwrap();

        assert_eq!(config.ollama_params.think, Some(true));
        assert_eq!(config.ollama_params.num_ctx, Some(4096));
        assert_eq!(config.model, Some("deepseek-r1:latest".to_string()));
    }

    #[test]
    fn test_config_validation() {
        let config = OllamaConfig::builder().base_url("").build();
        assert!(config.is_err());

        let config = OllamaConfig::builder().base_url("invalid-url").build();
        assert!(config.is_err());
    }
}
