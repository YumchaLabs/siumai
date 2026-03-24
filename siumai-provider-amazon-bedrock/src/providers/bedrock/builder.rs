//! `Bedrock` builder.

use super::{client::BedrockClient, config::BedrockConfig};
use crate::builder::{BuilderBase, ProviderCore};
use crate::error::LlmError;
use crate::retry_api::RetryOptions;
use std::sync::Arc;

/// Provider-owned builder for Amazon Bedrock clients.
#[derive(Clone)]
pub struct BedrockBuilder {
    pub(crate) core: ProviderCore,
    config: BedrockConfig,
}

impl BedrockBuilder {
    pub fn new(base: BuilderBase) -> Self {
        Self {
            core: ProviderCore::new(base),
            config: BedrockConfig::new(),
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

    pub fn runtime_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.config = self.config.with_runtime_base_url(base_url);
        self
    }

    pub fn agent_runtime_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.config = self.config.with_agent_runtime_base_url(base_url);
        self
    }

    pub fn region<S: Into<String>>(mut self, region: S) -> Self {
        self.config = self.config.with_region(region);
        self
    }

    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.config = self.config.with_model(model);
        self
    }

    pub fn rerank_model<S: Into<String>>(mut self, model: S) -> Self {
        self.config = self.config.with_rerank_model(model);
        self
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

    pub fn into_config(mut self) -> Result<BedrockConfig, LlmError> {
        if self.config.api_key.is_none()
            && let Ok(api_key) = std::env::var("BEDROCK_API_KEY")
        {
            self.config = self.config.with_api_key(api_key);
        }

        let default_runtime =
            BedrockConfig::runtime_base_url_for_region(BedrockConfig::DEFAULT_REGION);
        let default_agent =
            BedrockConfig::agent_runtime_base_url_for_region(BedrockConfig::DEFAULT_REGION);
        if self.config.region == BedrockConfig::DEFAULT_REGION
            && self.config.runtime_base_url == default_runtime
            && self.config.agent_runtime_base_url == default_agent
        {
            if let Ok(region) = std::env::var("AWS_REGION") {
                if !region.trim().is_empty() {
                    self.config = self.config.with_region(region);
                }
            } else if let Ok(region) = std::env::var("AWS_DEFAULT_REGION")
                && !region.trim().is_empty()
            {
                self.config = self.config.with_region(region);
            }
        }

        let mut config = self.config;
        config.http_config = self.core.http_config.clone();
        config.http_transport = self.core.http_transport.clone();
        config.http_interceptors = self.core.get_http_interceptors();
        config.validate()?;
        Ok(config)
    }

    pub fn build(self) -> Result<BedrockClient, LlmError> {
        let http_client_override = self.core.base.http_client.clone();
        let retry_options = self.core.retry_options.clone();
        let config = self.into_config()?;

        let mut client = if let Some(http_client) = http_client_override {
            BedrockClient::with_http_client(config, http_client)?
        } else {
            BedrockClient::from_config(config)?
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
        let cfg = BedrockBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .base_url("https://bedrock-runtime.us-east-1.amazonaws.com")
            .model("anthropic.claude-3-haiku-20240307-v1:0")
            .rerank_model("amazon.rerank-v1:0")
            .with_http_interceptor(Arc::new(NoopInterceptor))
            .into_config()
            .expect("into_config");

        assert!(cfg.api_key.is_some());
        assert_eq!(
            cfg.common_params.model,
            "anthropic.claude-3-haiku-20240307-v1:0"
        );
        assert_eq!(
            cfg.default_rerank_model.as_deref(),
            Some("amazon.rerank-v1:0")
        );
        assert_eq!(cfg.http_interceptors.len(), 1);
        assert_eq!(
            cfg.agent_runtime_base_url,
            "https://bedrock-agent-runtime.us-east-1.amazonaws.com"
        );
    }

    #[test]
    fn build_and_into_config_converge_on_config_first_client_construction() {
        let builder = BedrockBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .base_url("https://bedrock-runtime.us-east-1.amazonaws.com")
            .model("anthropic.claude-3-haiku-20240307-v1:0")
            .rerank_model("amazon.rerank-v1:0");

        let cfg = builder.clone().into_config().expect("into_config");
        let built = builder.build().expect("build client");
        let from_config = BedrockClient::from_config(cfg).expect("from_config client");

        assert_eq!(crate::client::LlmClient::provider_id(&built), "bedrock");
        assert_eq!(
            crate::client::LlmClient::provider_id(&from_config),
            "bedrock"
        );
        assert_eq!(
            crate::client::LlmClient::supported_models(&built),
            crate::client::LlmClient::supported_models(&from_config)
        );
    }
}
