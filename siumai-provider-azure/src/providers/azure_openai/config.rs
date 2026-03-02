//! Azure OpenAI client configuration.
//!
//! This config is intentionally minimal and is expected to be built by higher-level
//! builders (e.g., `siumai-registry` factories) in most cases.

use super::{AzureChatMode, AzureUrlConfig};
use crate::types::{CommonParams, HttpConfig};
use std::sync::Arc;

use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;

/// Configuration for Azure OpenAI (OpenAI-compatible endpoints).
#[derive(Clone)]
pub struct AzureOpenAiConfig {
    pub api_key: String,
    /// Base URL prefix, typically `https://{resource}.openai.azure.com/openai`.
    pub base_url: String,
    /// Default model id (Azure deployment id).
    pub common_params: CommonParams,
    pub http_config: HttpConfig,
    pub url_config: AzureUrlConfig,
    pub chat_mode: AzureChatMode,
    /// Provider metadata key used by OpenAI Responses mapping.
    pub provider_metadata_key: &'static str,
    /// Optional custom HTTP transport (Vercel-style "custom fetch" parity).
    pub http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    /// Optional HTTP interceptors applied to all requests built from this config.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping (chat only).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl std::fmt::Debug for AzureOpenAiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenAiConfig")
            .field("base_url", &self.base_url)
            .field("model", &self.common_params.model)
            .field("api_version", &self.url_config.api_version)
            .field(
                "use_deployment_based_urls",
                &self.url_config.use_deployment_based_urls,
            )
            .field("chat_mode", &self.chat_mode)
            .field("provider_metadata_key", &self.provider_metadata_key)
            .field("has_http_transport", &self.http_transport.is_some())
            .finish()
    }
}

impl AzureOpenAiConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: String::new(),
            common_params: CommonParams::default(),
            http_config: HttpConfig::default(),
            url_config: AzureUrlConfig::default(),
            chat_mode: AzureChatMode::default(),
            provider_metadata_key: "azure",
            http_transport: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.common_params.model = model.into();
        self
    }

    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = http_config;
        self
    }

    pub fn with_url_config(mut self, url_config: AzureUrlConfig) -> Self {
        self.url_config = url_config;
        self
    }

    pub fn with_chat_mode(mut self, mode: AzureChatMode) -> Self {
        self.chat_mode = mode;
        self
    }

    pub fn with_provider_metadata_key(mut self, key: &'static str) -> Self {
        self.provider_metadata_key = key;
        self
    }

    /// Set a custom HTTP transport (Vercel-style "custom fetch" parity).
    pub fn with_http_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.http_transport = Some(transport);
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

    pub fn validate(&self) -> Result<(), crate::error::LlmError> {
        if self.api_key.trim().is_empty() {
            return Err(crate::error::LlmError::InvalidParameter(
                "Azure OpenAI api_key cannot be empty".to_string(),
            ));
        }
        if self.base_url.trim().is_empty() {
            return Err(crate::error::LlmError::InvalidParameter(
                "Azure OpenAI base_url cannot be empty".to_string(),
            ));
        }
        if self.common_params.model.trim().is_empty() {
            return Err(crate::error::LlmError::InvalidParameter(
                "Azure OpenAI model (deployment id) cannot be empty".to_string(),
            ));
        }
        Ok(())
    }
}
