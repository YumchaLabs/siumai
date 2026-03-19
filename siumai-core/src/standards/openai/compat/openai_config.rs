//! OpenAI-compatible configuration (protocol layer)
//!
//! This module provides configuration types for OpenAI-compatible providers.

use super::adapter::ProviderAdapter;
use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::types::{CommonParams, HttpConfig};
use std::sync::Arc;

/// Configuration for OpenAI-compatible providers
#[derive(Clone)]
pub struct OpenAiCompatibleConfig {
    /// Provider identifier
    pub provider_id: String,
    /// API key for authentication
    pub api_key: String,
    /// Base URL for the provider
    pub base_url: String,
    /// Model to use
    pub model: String,
    /// Common parameters shared across providers
    pub common_params: CommonParams,
    /// HTTP configuration (timeout, proxy, etc.)
    pub http_config: HttpConfig,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub http_transport: Option<Arc<dyn HttpTransport>>,
    /// Custom headers for requests
    pub custom_headers: reqwest::header::HeaderMap,
    /// Provider adapter for handling specifics
    pub adapter: Arc<dyn ProviderAdapter>,
    /// Optional HTTP interceptors applied to all requests built from this config.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping (chat only).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl std::fmt::Debug for OpenAiCompatibleConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("OpenAiCompatibleConfig");
        ds.field("provider_id", &self.provider_id)
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("common_params", &self.common_params)
            .field("http_config", &self.http_config);

        if !self.api_key.is_empty() {
            ds.field("has_api_key", &true);
        }
        if self.http_transport.is_some() {
            ds.field("has_http_transport", &true);
        }

        ds.finish()
    }
}

impl OpenAiCompatibleConfig {
    /// Create a new configuration
    pub fn new(
        provider_id: &str,
        api_key: &str,
        base_url: &str,
        adapter: Arc<dyn ProviderAdapter>,
    ) -> Self {
        Self {
            provider_id: provider_id.to_string(),
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
            model: String::new(),
            common_params: CommonParams::default(),
            http_config: HttpConfig::default(),
            http_transport: None,
            custom_headers: reqwest::header::HeaderMap::new(),
            adapter,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        }
    }

    /// Set the model
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self.common_params.model = model.to_string();
        self
    }

    /// Set common parameters
    pub fn with_common_params(mut self, params: CommonParams) -> Self {
        self.common_params = params;
        self
    }

    /// Set HTTP configuration
    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = config;
        self
    }

    /// Set request timeout.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }

    /// Set connection timeout.
    pub fn with_connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }

    /// Control whether to disable compression for streaming requests.
    pub fn with_http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
        self
    }

    /// Set user agent.
    pub fn with_user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        self.http_config.user_agent = Some(user_agent.into());
        self
    }

    /// Set proxy URL.
    pub fn with_proxy<S: Into<String>>(mut self, proxy: S) -> Self {
        self.http_config.proxy = Some(proxy.into());
        self
    }

    /// Replace HTTP headers stored on the config-level HTTP template.
    pub fn with_http_headers(mut self, headers: std::collections::HashMap<String, String>) -> Self {
        self.http_config.headers = headers;
        self
    }

    /// Add a single HTTP header to the config-level HTTP template.
    pub fn with_http_header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.http_config.headers.insert(key.into(), value.into());
        self
    }

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_http_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.http_transport = Some(transport);
        self
    }

    /// Install a single HTTP interceptor for requests created by clients built from this config.
    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    /// Install HTTP interceptors for requests created by clients built from this config.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Install model-level middlewares for chat requests created by clients built from this config.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    /// Add a custom header
    pub fn with_header(mut self, key: &str, value: &str) -> Result<Self, LlmError> {
        let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid header name: {}", e)))?;
        let header_value = reqwest::header::HeaderValue::from_str(value)
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid header value: {}", e)))?;

        self.custom_headers.insert(header_name, header_value);
        Ok(self)
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    /// Set top-p.
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    /// Set stop sequences.
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(stop_sequences);
        self
    }

    /// Set deterministic seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    /// Merge provider-specific request defaults through the adapter layer.
    pub fn with_provider_specific_config(
        mut self,
        params: std::collections::HashMap<String, serde_json::Value>,
    ) -> Self {
        if params.is_empty() {
            return self;
        }
        let adapter = self.adapter.clone_adapter();
        self.adapter = Arc::from(
            Box::new(super::adapter::ParamMergingAdapter::new(adapter, params))
                as Box<dyn ProviderAdapter>,
        );
        self
    }

    /// Add a single provider-specific request default.
    pub fn with_provider_specific_param(
        mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        let mut params = std::collections::HashMap::new();
        params.insert(key.into(), value);
        self = self.with_provider_specific_config(params);
        self
    }

    /// Enable provider-native thinking mode when supported.
    pub fn with_thinking(mut self, enable: bool) -> Self {
        match self.provider_id.as_str() {
            "siliconflow" => {
                self = self.with_provider_specific_param(
                    "enable_thinking",
                    serde_json::Value::Bool(enable),
                );
            }
            _ => {
                self = self.with_provider_specific_param(
                    "enable_reasoning",
                    serde_json::Value::Bool(enable),
                );
            }
        }
        self
    }

    /// Set thinking budget for providers that expose it.
    pub fn with_thinking_budget(mut self, budget: u32) -> Self {
        let clamped_budget = budget.clamp(128, 32768);
        match self.provider_id.as_str() {
            "siliconflow" => {
                self = self
                    .with_provider_specific_param(
                        "thinking_budget",
                        serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                    )
                    .with_provider_specific_param("enable_thinking", serde_json::Value::Bool(true));
            }
            _ => {
                self = self
                    .with_provider_specific_param(
                        "reasoning_budget",
                        serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                    )
                    .with_provider_specific_param(
                        "enable_reasoning",
                        serde_json::Value::Bool(true),
                    );
            }
        }
        self
    }

    /// Unified reasoning toggle.
    pub fn with_reasoning(mut self, enable: bool) -> Self {
        match self.provider_id.as_str() {
            "siliconflow" => {
                self = self.with_provider_specific_param(
                    "enable_thinking",
                    serde_json::Value::Bool(enable),
                );
            }
            _ => {
                self = self.with_provider_specific_param(
                    "enable_reasoning",
                    serde_json::Value::Bool(enable),
                );
            }
        }
        self
    }

    /// Unified reasoning budget.
    pub fn with_reasoning_budget(mut self, budget: i32) -> Self {
        let clamped_budget = budget.clamp(128, 32768) as u32;
        match self.provider_id.as_str() {
            "siliconflow" => {
                self = self
                    .with_provider_specific_param(
                        "thinking_budget",
                        serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                    )
                    .with_provider_specific_param("enable_thinking", serde_json::Value::Bool(true));
            }
            _ => {
                self = self
                    .with_provider_specific_param(
                        "reasoning_budget",
                        serde_json::Value::Number(serde_json::Number::from(clamped_budget)),
                    )
                    .with_provider_specific_param(
                        "enable_reasoning",
                        serde_json::Value::Bool(true),
                    );
            }
        }
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), LlmError> {
        if self.provider_id.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Provider ID cannot be empty".to_string(),
            ));
        }

        if self.api_key.is_empty() {
            return Err(LlmError::ConfigurationError(
                "API key cannot be empty".to_string(),
            ));
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ProviderCapabilities;

    #[test]
    fn test_config_creation() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }

            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }

            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }

            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }

            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }

            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }

            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        );

        assert_eq!(config.provider_id, "test");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, "https://api.test.com/v1");
    }

    #[test]
    fn test_config_with_model() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        )
        .with_model("test-model");

        assert_eq!(config.model, "test-model");
    }

    #[test]
    fn test_config_validation() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        // Valid config
        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        );
        assert!(config.validate().is_ok());

        // Empty provider ID
        let config = OpenAiCompatibleConfig::new(
            "",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        );
        assert!(config.validate().is_err());

        // Empty API key
        let config = OpenAiCompatibleConfig::new(
            "test",
            "",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        );
        assert!(config.validate().is_err());

        // Invalid URL
        let config =
            OpenAiCompatibleConfig::new("test", "test-key", "invalid-url", Arc::new(DummyAdapter));
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_with_header() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        )
        .with_header("X-Custom", "test-value")
        .unwrap();

        assert!(config.custom_headers.contains_key("X-Custom"));
    }

    #[test]
    fn test_config_http_convenience_helpers() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(DummyAdapter),
        )
        .with_timeout(std::time::Duration::from_secs(15))
        .with_connect_timeout(std::time::Duration::from_secs(3))
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
        ));

        assert_eq!(
            config.http_config.timeout,
            Some(std::time::Duration::from_secs(15))
        );
        assert_eq!(
            config.http_config.connect_timeout,
            Some(std::time::Duration::from_secs(3))
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
        assert_eq!(config.http_interceptors.len(), 1);
    }

    #[test]
    fn test_config_fluent_common_params_and_reasoning_defaults() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "deepseek",
            "test-key",
            "https://api.deepseek.com",
            Arc::new(DummyAdapter),
        )
        .with_model("deepseek-chat")
        .with_temperature(0.4)
        .with_max_tokens(256)
        .with_top_p(0.9)
        .with_stop_sequences(vec!["END".to_string()])
        .with_seed(7)
        .with_reasoning(true)
        .with_reasoning_budget(2048);

        assert_eq!(config.model, "deepseek-chat");
        assert_eq!(config.common_params.model, "deepseek-chat");
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
                super::super::types::RequestType::Chat,
            )
            .expect("transform request params");
        assert_eq!(params["enable_reasoning"], serde_json::json!(true));
        assert_eq!(params["reasoning_budget"], serde_json::json!(2048));
    }

    #[test]
    fn test_config_thinking_budget_maps_for_siliconflow() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://api.test.com/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "siliconflow",
            "test-key",
            "https://api.siliconflow.cn/v1",
            Arc::new(DummyAdapter),
        )
        .with_model("deepseek-ai/DeepSeek-V3.1")
        .with_thinking_budget(8192);

        let mut params = serde_json::json!({});
        config
            .adapter
            .transform_request_params(
                &mut params,
                &config.model,
                super::super::types::RequestType::Chat,
            )
            .expect("transform request params");
        assert_eq!(params["enable_thinking"], serde_json::json!(true));
        assert_eq!(params["thinking_budget"], serde_json::json!(8192));
    }

    #[test]
    fn test_config_reasoning_maps_for_openrouter() {
        #[derive(Debug, Clone)]
        struct DummyAdapter;
        impl super::super::adapter::ProviderAdapter for DummyAdapter {
            fn provider_id(&self) -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed("test")
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _request_type: super::super::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> super::super::types::FieldMappings {
                super::super::types::FieldMappings::standard()
            }
            fn get_model_config(&self, _model: &str) -> super::super::types::ModelConfig {
                super::super::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new().with_chat()
            }
            fn base_url(&self) -> &str {
                "https://openrouter.ai/api/v1"
            }
            fn clone_adapter(&self) -> Box<dyn super::super::adapter::ProviderAdapter> {
                Box::new(self.clone())
            }
        }

        let config = OpenAiCompatibleConfig::new(
            "openrouter",
            "test-key",
            "https://openrouter.ai/api/v1",
            Arc::new(DummyAdapter),
        )
        .with_model("openai/gpt-4o")
        .with_reasoning(true)
        .with_reasoning_budget(2048);

        let mut params = serde_json::json!({});
        config
            .adapter
            .transform_request_params(
                &mut params,
                &config.model,
                super::super::types::RequestType::Chat,
            )
            .expect("transform request params");
        assert_eq!(params["enable_reasoning"], serde_json::json!(true));
        assert_eq!(params["reasoning_budget"], serde_json::json!(2048));
    }
}
