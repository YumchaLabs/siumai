//! Ollama Parameter Mapping
//!
//! Maps common parameters to Ollama-specific format.

use crate::error::LlmError;
use crate::types::CommonParams;
use serde_json::{Value, json};
use std::collections::HashMap;

/// Ollama parameter mapper
pub struct OllamaParameterMapper;

impl OllamaParameterMapper {
    #[cfg(test)]
    fn provider_type(&self) -> crate::types::ProviderType {
        crate::types::ProviderType::Ollama
    }

    pub fn map_common_params(&self, params: &CommonParams) -> Value {
        let mut ollama_params = json!({});

        // Map model
        if !params.model.is_empty() {
            ollama_params["model"] = json!(params.model);
        }

        // Map temperature
        if let Some(temperature) = params.temperature {
            ollama_params["temperature"] = json!(temperature);
        }

        // Map max_tokens to num_predict
        if let Some(max_tokens) = params.max_tokens {
            ollama_params["num_predict"] = json!(max_tokens);
        }

        // Map top_p
        if let Some(top_p) = params.top_p {
            ollama_params["top_p"] = json!(top_p);
        }

        // Map stop sequences
        if let Some(stop_sequences) = &params.stop_sequences {
            ollama_params["stop"] = json!(stop_sequences);
        }

        // Map seed
        if let Some(seed) = params.seed {
            ollama_params["seed"] = json!(seed);
        }

        ollama_params
    }

    // merge_provider_params has been removed in v0.12.0
    // Use OllamaOptions instead

    pub fn validate_params(&self, params: &Value) -> Result<(), LlmError> {
        // Validate temperature
        if let Some(temp) = params
            .get("temperature")
            .and_then(serde_json::Value::as_f64)
            && !(0.0..=2.0).contains(&temp)
        {
            return Err(LlmError::InvalidParameter(
                "Temperature must be between 0.0 and 2.0".to_string(),
            ));
        }

        // Validate top_p
        if let Some(top_p) = params.get("top_p").and_then(serde_json::Value::as_f64)
            && !(0.0..=1.0).contains(&top_p)
        {
            return Err(LlmError::InvalidParameter(
                "top_p must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Validate num_predict (max_tokens)
        if let Some(num_predict) = params
            .get("num_predict")
            .and_then(serde_json::Value::as_u64)
            && num_predict == 0
        {
            return Err(LlmError::InvalidParameter(
                "num_predict must be greater than 0".to_string(),
            ));
        }

        // Validate num_ctx
        if let Some(num_ctx) = params.get("num_ctx").and_then(serde_json::Value::as_u64)
            && num_ctx == 0
        {
            return Err(LlmError::InvalidParameter(
                "num_ctx must be greater than 0".to_string(),
            ));
        }

        // Validate num_batch
        if let Some(num_batch) = params.get("num_batch").and_then(serde_json::Value::as_u64)
            && num_batch == 0
        {
            return Err(LlmError::InvalidParameter(
                "num_batch must be greater than 0".to_string(),
            ));
        }

        // Validate num_gpu
        if let Some(num_gpu) = params.get("num_gpu").and_then(serde_json::Value::as_u64)
            && num_gpu > 64
        {
            return Err(LlmError::InvalidParameter(
                "num_gpu should not exceed 64".to_string(),
            ));
        }

        // Validate num_thread
        if let Some(num_thread) = params.get("num_thread").and_then(serde_json::Value::as_u64)
            && (num_thread == 0 || num_thread > 256)
        {
            return Err(LlmError::InvalidParameter(
                "num_thread must be between 1 and 256".to_string(),
            ));
        }

        Ok(())
    }

    // Note: Parameter constraints are validated in Transformers/validator; no per-provider accessor here.
}

/// Ollama-specific provider parameters
#[derive(Debug, Clone, Default)]
pub struct OllamaProviderParams {
    /// Keep model loaded in memory for this duration
    pub keep_alive: Option<String>,
    /// Use raw mode (bypass templating)
    pub raw: Option<bool>,
    /// Format for structured outputs
    pub format: Option<String>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Enable/disable NUMA support
    pub numa: Option<bool>,
    /// Context window size
    pub num_ctx: Option<u32>,
    /// Batch size for processing
    pub num_batch: Option<u32>,
    /// Number of GPU layers to use
    pub num_gpu: Option<u32>,
    /// Main GPU to use
    pub main_gpu: Option<u32>,
    /// Use memory mapping
    pub use_mmap: Option<bool>,
    /// Number of threads to use
    pub num_thread: Option<u32>,
    /// Additional model options
    pub options: Option<HashMap<String, Value>>,
}

impl OllamaProviderParams {
    /// Create new Ollama provider parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set keep alive duration
    pub fn keep_alive<S: Into<String>>(mut self, duration: S) -> Self {
        self.keep_alive = Some(duration.into());
        self
    }

    /// Enable raw mode
    pub const fn raw(mut self, raw: bool) -> Self {
        self.raw = Some(raw);
        self
    }

    /// Set output format
    pub fn format<S: Into<String>>(mut self, format: S) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Set stop sequences
    pub fn stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Enable NUMA support
    pub const fn numa(mut self, numa: bool) -> Self {
        self.numa = Some(numa);
        self
    }

    /// Set context window size
    pub const fn num_ctx(mut self, num_ctx: u32) -> Self {
        self.num_ctx = Some(num_ctx);
        self
    }

    /// Set batch size
    pub const fn num_batch(mut self, num_batch: u32) -> Self {
        self.num_batch = Some(num_batch);
        self
    }

    /// Set number of GPU layers
    pub const fn num_gpu(mut self, num_gpu: u32) -> Self {
        self.num_gpu = Some(num_gpu);
        self
    }

    /// Set main GPU
    pub const fn main_gpu(mut self, main_gpu: u32) -> Self {
        self.main_gpu = Some(main_gpu);
        self
    }

    /// Enable memory mapping
    pub const fn use_mmap(mut self, use_mmap: bool) -> Self {
        self.use_mmap = Some(use_mmap);
        self
    }

    /// Set number of threads
    pub const fn num_thread(mut self, num_thread: u32) -> Self {
        self.num_thread = Some(num_thread);
        self
    }

    /// Add custom option
    pub fn option<K: Into<String>>(mut self, key: K, value: Value) -> Self {
        let mut options = self.options.unwrap_or_default();
        options.insert(key.into(), value);
        self.options = Some(options);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ProviderType;

    #[test]
    fn test_ollama_parameter_mapper() {
        let mapper = OllamaParameterMapper;
        assert_eq!(mapper.provider_type(), ProviderType::Ollama);

        let common_params = CommonParams {
            model: "llama3.2".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(1000),
            top_p: Some(0.9),
            stop_sequences: Some(vec!["\\n".to_string()]),
            seed: Some(42),
        };

        let mapped_params = mapper.map_common_params(&common_params);
        assert_eq!(mapped_params["model"], "llama3.2");
        assert!((mapped_params["temperature"].as_f64().unwrap() - 0.7).abs() < 0.001);
        assert_eq!(mapped_params["num_predict"], 1000);
        assert!((mapped_params["top_p"].as_f64().unwrap() - 0.9).abs() < 0.001);
        assert_eq!(mapped_params["seed"], 42);
    }

    #[test]
    fn test_ollama_provider_params() {
        let params = OllamaProviderParams::new()
            .keep_alive("10m".to_string())
            .raw(true)
            .numa(false)
            .num_ctx(2048)
            .num_gpu(1)
            .num_thread(8);

        assert_eq!(params.keep_alive, Some("10m".to_string()));
        assert_eq!(params.raw, Some(true));
        assert_eq!(params.numa, Some(false));
        assert_eq!(params.num_ctx, Some(2048));
        assert_eq!(params.num_gpu, Some(1));
        assert_eq!(params.num_thread, Some(8));
    }
}
