//! `Cohere` configuration.

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::types::{CommonParams, HttpConfig};
use secrecy::{ExposeSecret, SecretString};
use std::sync::Arc;

/// Provider-owned config-first surface for Cohere reranking.
#[derive(Clone)]
pub struct CohereConfig {
    /// API key (securely stored).
    pub api_key: SecretString,
    /// Base URL for the Cohere API.
    pub base_url: String,
    /// Shared request defaults.
    pub common_params: CommonParams,
    /// HTTP configuration.
    pub http_config: HttpConfig,
    /// Optional custom HTTP transport.
    pub http_transport: Option<Arc<dyn HttpTransport>>,
    /// Optional HTTP interceptors applied to all requests.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
}

impl std::fmt::Debug for CohereConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("CohereConfig");
        ds.field("base_url", &self.base_url)
            .field("common_params", &self.common_params)
            .field("http_config", &self.http_config);

        if !self.api_key.expose_secret().is_empty() {
            ds.field("has_api_key", &true);
        }
        if self.http_transport.is_some() {
            ds.field("has_http_transport", &true);
        }
        if !self.http_interceptors.is_empty() {
            ds.field("http_interceptors_len", &self.http_interceptors.len());
        }

        ds.finish()
    }
}

impl CohereConfig {
    /// Default Cohere API base URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.cohere.com/v2";
    /// Default Cohere rerank model.
    pub const DEFAULT_MODEL: &'static str = "rerank-english-v3.0";

    /// Create a new config with the given API key.
    pub fn new<S: Into<String>>(api_key: S) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            common_params: CommonParams {
                model: Self::DEFAULT_MODEL.to_string(),
                ..Default::default()
            },
            http_config: HttpConfig::default(),
            http_transport: None,
            http_interceptors: Vec::new(),
        }
    }

    /// Create config from `COHERE_API_KEY`.
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("COHERE_API_KEY")
            .map_err(|_| LlmError::MissingApiKey("Cohere API key not provided".to_string()))?;
        Ok(Self::new(api_key))
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = SecretString::from(api_key.into());
        self
    }

    pub fn with_base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    pub fn with_http_config(mut self, http: HttpConfig) -> Self {
        self.http_config = http;
        self
    }

    pub fn with_http_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.http_transport = Some(transport);
        self
    }

    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    pub fn validate(&self) -> Result<(), LlmError> {
        if self.api_key.expose_secret().trim().is_empty() {
            return Err(LlmError::MissingApiKey(
                "Cohere API key not provided".to_string(),
            ));
        }
        if self.base_url.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Cohere base_url cannot be empty".to_string(),
            ));
        }
        if self.common_params.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Cohere requires a non-empty rerank model id".to_string(),
            ));
        }
        Ok(())
    }
}
