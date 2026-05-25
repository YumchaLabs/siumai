use super::{GatewayClient, GatewayConfig};
use crate::builder::{BuilderBase, ProviderCore};
use crate::error::LlmError;
use crate::retry_api::RetryOptions;
use secrecy::ExposeSecret;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct GatewayBuilder {
    pub(crate) core: ProviderCore,
    config: GatewayConfig,
}

impl GatewayBuilder {
    pub fn new(base: BuilderBase) -> Self {
        Self {
            core: ProviderCore::new(base),
            config: GatewayConfig::new(""),
        }
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.config = self.config.with_api_key(api_key);
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.config = self.config.with_base_url(base_url);
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config = self.config.with_model(model);
        self
    }

    pub fn language_model(self, model: impl Into<String>) -> Self {
        self.model(model)
    }

    pub fn embedding_model(self, model: impl Into<String>) -> Self {
        self.model(model)
    }

    pub fn team_id_or_slug(mut self, team_id_or_slug: impl Into<String>) -> Self {
        self.config = self.config.with_team_id_or_slug(team_id_or_slug);
        self
    }

    pub fn headers(mut self, headers: HashMap<String, String>) -> Self {
        self.core.http_config.headers.extend(headers);
        self
    }

    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.core
            .http_config
            .headers
            .insert(name.into(), value.into());
        self
    }

    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.core = self.core.timeout(timeout);
        self
    }

    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.core = self.core.with_http_client(client);
        self
    }

    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.core = self.core.with_retry(options);
        self
    }

    pub fn with_http_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.core = self.core.with_http_transport(transport);
        self
    }

    pub fn fetch(
        self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.with_http_transport(transport)
    }

    pub fn into_config(mut self) -> Result<GatewayConfig, LlmError> {
        if self.config.api_key.expose_secret().trim().is_empty()
            && let Ok(api_key) = std::env::var("AI_GATEWAY_API_KEY")
        {
            self.config = self.config.with_api_key(api_key);
        }

        let mut config = self.config;
        config.http_config = self.core.http_config.clone();
        config.http_transport = self.core.http_transport.clone();
        config.http_interceptors = self.core.get_http_interceptors();
        config.validate()?;
        if config.common_params.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Gateway requires an explicit model id".to_string(),
            ));
        }
        Ok(config)
    }

    pub fn build(self) -> Result<GatewayClient, LlmError> {
        let http_client_override = self.core.base.http_client.clone();
        let retry_options = self.core.retry_options.clone();
        let config = self.into_config()?;

        let mut client = if let Some(http_client) = http_client_override {
            GatewayClient::with_http_client(config, http_client)?
        } else {
            GatewayClient::from_config(config)?
        };

        if let Some(retry_options) = retry_options {
            client = client.with_retry_options(retry_options);
        }

        Ok(client)
    }
}
