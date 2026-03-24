//! `TogetherAI` builder.

use super::{client::TogetherAiClient, config::TogetherAiConfig};
use crate::builder::{BuilderBase, ProviderCore};
use crate::error::LlmError;
use crate::retry_api::RetryOptions;
use secrecy::ExposeSecret;
use std::sync::Arc;

/// Provider-owned builder for TogetherAI rerank clients.
#[derive(Clone)]
pub struct TogetherAiBuilder {
    pub(crate) core: ProviderCore,
    config: TogetherAiConfig,
}

impl TogetherAiBuilder {
    pub fn new(base: BuilderBase) -> Self {
        Self {
            core: ProviderCore::new(base),
            config: TogetherAiConfig::new(""),
        }
    }

    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.config = self.config.with_api_key(api_key);
        self
    }

    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.config = self.config.with_base_url(base_url);
        self
    }

    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.config = self.config.with_model(model);
        self
    }

    /// Alias for `model(...)` on rerank-only providers.
    pub fn reranking_model<S: Into<String>>(self, model: S) -> Self {
        self.model(model)
    }

    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.core = self.core.timeout(timeout);
        self
    }

    pub fn connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.core = self.core.connect_timeout(timeout);
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

    pub fn with_http_interceptor(
        mut self,
        interceptor: Arc<dyn crate::execution::http::interceptor::HttpInterceptor>,
    ) -> Self {
        self.core = self.core.with_http_interceptor(interceptor);
        self
    }

    pub fn with_http_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.core = self.core.with_http_transport(transport);
        self
    }

    /// Alias for `with_http_transport(...)` (Vercel-style `fetch`).
    pub fn fetch(
        self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.with_http_transport(transport)
    }

    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.core = self.core.http_debug(enabled);
        self
    }

    pub fn tracing(mut self, config: crate::observability::tracing::TracingConfig) -> Self {
        self.core = self.core.tracing(config);
        self
    }

    pub fn debug_tracing(mut self) -> Self {
        self.core = self.core.debug_tracing();
        self
    }

    pub fn minimal_tracing(mut self) -> Self {
        self.core = self.core.minimal_tracing();
        self
    }

    pub fn json_tracing(mut self) -> Self {
        self.core = self.core.json_tracing();
        self
    }

    pub fn into_config(mut self) -> Result<TogetherAiConfig, LlmError> {
        if self.config.api_key.expose_secret().trim().is_empty()
            && let Ok(api_key) = std::env::var("TOGETHER_API_KEY")
        {
            self.config = self.config.with_api_key(api_key);
        }

        let mut config = self.config;
        config.http_config = self.core.http_config.clone();
        config.http_transport = self.core.http_transport.clone();
        config.http_interceptors = self.core.get_http_interceptors();
        config.validate()?;
        Ok(config)
    }

    pub fn build(self) -> Result<TogetherAiClient, LlmError> {
        let http_client_override = self.core.base.http_client.clone();
        let retry_options = self.core.retry_options.clone();
        let config = self.into_config()?;

        let mut client = if let Some(http_client) = http_client_override {
            TogetherAiClient::with_http_client(config, http_client)?
        } else {
            TogetherAiClient::from_config(config)?
        };

        if let Some(retry_options) = retry_options {
            client = client.with_retry_options(retry_options);
        }

        Ok(client)
    }
}

#[cfg(test)]
mod config_first_tests {
    use super::*;
    use std::sync::Arc;

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl crate::execution::http::interceptor::HttpInterceptor for NoopInterceptor {}

    #[test]
    fn into_config_preserves_explicit_api_key_and_http_interceptors() {
        let cfg = TogetherAiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .model("Salesforce/Llama-Rank-v1")
            .timeout(std::time::Duration::from_secs(15))
            .connect_timeout(std::time::Duration::from_secs(3))
            .with_http_interceptor(Arc::new(NoopInterceptor))
            .into_config()
            .expect("into_config");

        assert_eq!(cfg.api_key.expose_secret(), "test-key");
        assert_eq!(cfg.common_params.model, "Salesforce/Llama-Rank-v1");
        assert_eq!(
            cfg.http_config.timeout,
            Some(std::time::Duration::from_secs(15))
        );
        assert_eq!(
            cfg.http_config.connect_timeout,
            Some(std::time::Duration::from_secs(3))
        );
        assert_eq!(cfg.http_interceptors.len(), 1);
    }

    #[test]
    fn into_config_matches_manual_config_for_http_conveniences() {
        let builder_cfg = TogetherAiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .model("Salesforce/Llama-Rank-v1")
            .timeout(std::time::Duration::from_secs(15))
            .connect_timeout(std::time::Duration::from_secs(3))
            .with_http_interceptor(Arc::new(NoopInterceptor))
            .into_config()
            .expect("builder into_config");

        let manual_cfg = TogetherAiConfig::new("test-key")
            .with_model("Salesforce/Llama-Rank-v1")
            .with_timeout(std::time::Duration::from_secs(15))
            .with_connect_timeout(std::time::Duration::from_secs(3))
            .with_http_interceptor(Arc::new(NoopInterceptor));

        assert_eq!(
            builder_cfg.api_key.expose_secret(),
            manual_cfg.api_key.expose_secret()
        );
        assert_eq!(builder_cfg.base_url, manual_cfg.base_url);
        assert_eq!(
            builder_cfg.common_params.model,
            manual_cfg.common_params.model
        );
        assert_eq!(
            builder_cfg.http_config.timeout,
            manual_cfg.http_config.timeout
        );
        assert_eq!(
            builder_cfg.http_config.connect_timeout,
            manual_cfg.http_config.connect_timeout
        );
        assert_eq!(
            builder_cfg.http_config.headers,
            manual_cfg.http_config.headers
        );
        assert_eq!(
            builder_cfg.http_interceptors.len(),
            manual_cfg.http_interceptors.len()
        );
    }

    #[test]
    fn build_and_into_config_converge_on_config_first_client_construction() {
        let builder = TogetherAiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .model("Salesforce/Llama-Rank-v1");

        let cfg = builder.clone().into_config().expect("into_config");
        let built = builder.build().expect("build client");
        let from_config = TogetherAiClient::from_config(cfg).expect("from_config client");

        assert_eq!(crate::client::LlmClient::provider_id(&built), "togetherai");
        assert_eq!(
            crate::client::LlmClient::provider_id(&from_config),
            "togetherai"
        );
        assert_eq!(
            crate::client::LlmClient::supported_models(&built),
            crate::client::LlmClient::supported_models(&from_config)
        );
    }
}
