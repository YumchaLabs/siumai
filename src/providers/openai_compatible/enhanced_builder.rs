//! Enhanced OpenAI-Compatible Builder with Configuration Support
//!
//! This module provides an enhanced builder that integrates the configuration-driven
//! adapter system with our existing LlmBuilder and HTTP configuration system.

use super::adapter::ProviderAdapter;
use super::config_adapter::ConfigurableAdapter;
use super::openai_client::OpenAiCompatibleClient;
use super::openai_config::OpenAiCompatibleConfig;
use crate::builder::LlmBuilder;
use crate::error::LlmError;
use crate::types::{CommonParams, HttpConfig};
use std::sync::Arc;

/// Enhanced builder for OpenAI-compatible providers with configuration support
#[derive(Debug, Clone)]
pub struct EnhancedOpenAiCompatibleBuilder {
    /// Base builder with HTTP configuration
    pub(crate) base: LlmBuilder,
    /// API key
    api_key: Option<String>,
    /// Model name
    model: Option<String>,
    /// Common parameters
    common_params: CommonParams,
    /// Configurable adapter
    adapter: Option<ConfigurableAdapter>,
    /// Provider configuration file path
    config_file: Option<std::path::PathBuf>,
}

impl EnhancedOpenAiCompatibleBuilder {
    /// Create a new enhanced builder
    pub fn new(base: LlmBuilder) -> Self {
        Self {
            base,
            api_key: None,
            model: None,
            common_params: CommonParams::default(),
            adapter: None,
            config_file: None,
        }
    }

    /// Set the API key
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the model
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        let model_str = model.into();
        self.model = Some(model_str.clone());
        self.common_params.model = model_str;
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

    /// Use a pre-configured adapter
    pub fn with_adapter(mut self, adapter: ConfigurableAdapter) -> Self {
        self.adapter = Some(adapter);
        self
    }

    /// Load adapter configuration from file
    pub fn with_config_file<P: Into<std::path::PathBuf>>(mut self, path: P) -> Self {
        self.config_file = Some(path.into());
        self
    }

    /// Use DeepSeek configuration
    pub fn deepseek_config(mut self) -> Self {
        self.adapter = Some(ConfigurableAdapter::deepseek());
        self
    }

    /// Use SiliconFlow configuration
    pub fn siliconflow_config(mut self) -> Self {
        self.adapter = Some(ConfigurableAdapter::siliconflow());
        self
    }

    /// Create a custom adapter configuration
    pub fn custom_config<F>(mut self, configure: F) -> Self
    where
        F: FnOnce(ConfigurableAdapter) -> ConfigurableAdapter,
    {
        let base_adapter = self.adapter.unwrap_or_else(|| {
            ConfigurableAdapter::new("custom", "Custom Provider", "https://api.example.com")
        });
        self.adapter = Some(configure(base_adapter));
        self
    }

    /// Build the client with enhanced configuration support
    pub async fn build(self) -> Result<OpenAiCompatibleClient, LlmError> {
        let api_key = self
            .api_key
            .ok_or_else(|| LlmError::ConfigurationError("API key is required".to_string()))?;

        // Load adapter from file if specified
        let adapter = if let Some(config_path) = &self.config_file {
            ConfigurableAdapter::from_json_file(config_path)?
        } else if let Some(adapter) = self.adapter {
            adapter
        } else {
            return Err(LlmError::ConfigurationError(
                "Either adapter or config file must be specified".to_string(),
            ));
        };

        // Get base HTTP configuration from LlmBuilder
        let mut http_config = HttpConfig::default();

        // Apply LlmBuilder HTTP settings
        if let Some(timeout) = self.base.timeout {
            http_config.timeout = Some(timeout);
        }
        if let Some(connect_timeout) = self.base.connect_timeout {
            http_config.connect_timeout = Some(connect_timeout);
        }
        if let Some(user_agent) = &self.base.user_agent {
            http_config.user_agent = Some(user_agent.clone());
        }
        if let Some(proxy) = &self.base.proxy {
            http_config.proxy = Some(proxy.clone());
        }

        // Merge default headers from LlmBuilder
        for (key, value) in &self.base.default_headers {
            http_config.headers.insert(key.clone(), value.clone());
        }

        // Apply adapter-specific HTTP configuration
        http_config = ProviderAdapter::apply_http_config(&adapter, http_config);

        // Create configuration
        let provider_id = adapter.provider_id.clone();
        let base_url = adapter.base_url.clone();
        let mut config =
            OpenAiCompatibleConfig::new(&provider_id, &api_key, &base_url, Arc::new(adapter));

        config = config
            .with_common_params(self.common_params)
            .with_http_config(http_config);

        if let Some(model) = self.model {
            config = config.with_model(&model);
        }

        // Build HTTP client using LlmBuilder's method (supports custom clients)
        let http_client = self.base.build_http_client()?;

        // Create the client
        OpenAiCompatibleClient::with_http_client(config, http_client).await
    }
}

/// Integration with existing LlmBuilder
impl LlmBuilder {
    /// Create an enhanced OpenAI-compatible builder
    pub fn enhanced_openai_compatible(self) -> EnhancedOpenAiCompatibleBuilder {
        EnhancedOpenAiCompatibleBuilder::new(self)
    }

    /// Create a DeepSeek client with enhanced configuration
    pub fn enhanced_deepseek(self) -> EnhancedOpenAiCompatibleBuilder {
        self.enhanced_openai_compatible().deepseek_config()
    }

    /// Create a SiliconFlow client with enhanced configuration
    pub fn enhanced_siliconflow(self) -> EnhancedOpenAiCompatibleBuilder {
        self.enhanced_openai_compatible().siliconflow_config()
    }

    /// Create a client from configuration file
    pub fn from_config_file<P: Into<std::path::PathBuf>>(
        self,
        path: P,
    ) -> EnhancedOpenAiCompatibleBuilder {
        self.enhanced_openai_compatible().with_config_file(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_enhanced_builder_with_http_config() {
        // Test that HTTP configuration is properly inherited and enhanced
        let builder = LlmBuilder::new()
            .with_timeout(Duration::from_secs(60))
            .with_proxy("http://proxy.example.com:8080")
            .with_user_agent("test-agent/1.0")
            .with_header("X-Test-Header", "test-value")
            .enhanced_deepseek()
            .api_key("test-key")
            .model("deepseek-chat")
            .temperature(0.7);

        // Verify configuration is properly set
        assert_eq!(builder.base.timeout, Some(Duration::from_secs(60)));
        assert_eq!(
            builder.base.proxy,
            Some("http://proxy.example.com:8080".to_string())
        );
        assert_eq!(builder.base.user_agent, Some("test-agent/1.0".to_string()));
        assert!(builder.base.default_headers.contains_key("X-Test-Header"));
        assert_eq!(builder.common_params.temperature, Some(0.7));
    }

    #[test]
    fn test_custom_adapter_configuration() {
        let builder = LlmBuilder::new()
            .enhanced_openai_compatible()
            .custom_config(|mut adapter| {
                adapter.provider_id = "custom-provider".to_string();
                adapter.base_url = "https://api.custom.com".to_string();
                adapter.compatibility.supports_array_content = false;
                adapter
                    .custom_headers
                    .insert("X-Custom-Header".to_string(), "custom-value".to_string());
                adapter
            })
            .api_key("test-key")
            .model("custom-model");

        // Verify custom configuration
        let adapter = builder.adapter.unwrap();
        assert_eq!(adapter.provider_id, "custom-provider");
        assert_eq!(adapter.base_url, "https://api.custom.com");
        assert!(!adapter.compatibility.supports_array_content);
        assert_eq!(
            adapter.custom_headers.get("X-Custom-Header"),
            Some(&"custom-value".to_string())
        );
    }
}
