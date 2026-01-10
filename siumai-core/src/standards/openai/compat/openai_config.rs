//! OpenAI-compatible configuration (protocol layer)
//!
//! This module provides configuration types for OpenAI-compatible providers.

use super::adapter::ProviderAdapter;
use crate::error::LlmError;
use crate::execution::http::transport::HttpTransport;
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

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_http_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.http_transport = Some(transport);
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
}
