use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::types::{CommonParams, HttpConfig};
use secrecy::{ExposeSecret, SecretString};
use std::collections::HashMap;
use std::sync::Arc;

/// Provider-owned config-first surface for Vercel AI Gateway.
#[derive(Clone)]
pub struct GatewayConfig {
    pub api_key: SecretString,
    pub base_url: String,
    pub team_id_or_slug: Option<String>,
    pub common_params: CommonParams,
    pub http_config: HttpConfig,
    pub http_transport: Option<Arc<dyn HttpTransport>>,
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
}

impl std::fmt::Debug for GatewayConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatewayConfig")
            .field("base_url", &self.base_url)
            .field("team_id_or_slug", &self.team_id_or_slug)
            .field("common_params", &self.common_params)
            .field("http_config", &self.http_config)
            .field("has_api_key", &!self.api_key.expose_secret().is_empty())
            .field("has_http_transport", &self.http_transport.is_some())
            .field("http_interceptors_len", &self.http_interceptors.len())
            .finish()
    }
}

impl GatewayConfig {
    pub const DEFAULT_BASE_URL: &'static str = "https://ai-gateway.vercel.sh/v4/ai";

    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            team_id_or_slug: None,
            common_params: CommonParams::default(),
            http_config: crate::defaults::http::config_default(),
            http_transport: None,
            http_interceptors: Vec::new(),
        }
    }

    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("AI_GATEWAY_API_KEY").map_err(|_| {
            LlmError::MissingApiKey("AI_GATEWAY_API_KEY is not configured".to_string())
        })?;
        Ok(Self::new(api_key))
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = SecretString::from(api_key.into());
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.common_params.model = model.into();
        self
    }

    pub fn with_team_id_or_slug(mut self, team_id_or_slug: impl Into<String>) -> Self {
        self.team_id_or_slug = Some(team_id_or_slug.into());
        self
    }

    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = http_config;
        self
    }

    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.http_config.headers.extend(headers);
        self
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.http_config.headers.insert(name.into(), value.into());
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
                "Vercel AI Gateway API key not provided".to_string(),
            ));
        }
        if self.base_url.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Gateway base_url cannot be empty".to_string(),
            ));
        }
        Ok(())
    }
}
