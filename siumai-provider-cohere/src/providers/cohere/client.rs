//! `Cohere` rerank client.

use super::config::CohereConfig;
use crate::client::LlmClient;
use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::executors::rerank::{RerankExecutor, RerankExecutorBuilder};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::retry_api::RetryOptions;
use crate::standards::cohere::rerank::CohereRerankStandard;
use crate::traits::{ModelMetadata, ProviderCapabilities, RerankCapability};
use crate::types::{RerankRequest, RerankResponse};
use async_trait::async_trait;
use secrecy::ExposeSecret;
use std::borrow::Cow;
use std::sync::Arc;

/// Provider-owned Cohere client (rerank-only).
#[derive(Clone)]
pub struct CohereClient {
    config: CohereConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
}

impl std::fmt::Debug for CohereClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CohereClient")
            .field("config", &self.config)
            .field("retry_options", &self.retry_options)
            .finish()
    }
}

impl CohereClient {
    /// Build a client from config using an HTTP client derived from `http_config`.
    pub fn from_config(config: CohereConfig) -> Result<Self, LlmError> {
        config.validate()?;
        let http_client =
            crate::execution::http::client::build_http_client_from_config(&config.http_config)?;
        Self::with_http_client(config, http_client)
    }

    /// Build a client from config with an explicit `reqwest::Client`.
    pub fn with_http_client(
        config: CohereConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        config.validate()?;
        Ok(Self {
            config,
            http_client,
            retry_options: None,
        })
    }

    /// Set retry options.
    pub fn with_retry_options(mut self, retry_options: RetryOptions) -> Self {
        self.retry_options = Some(retry_options);
        self
    }

    /// Alias for `with_retry_options(...)`.
    pub fn with_retry(self, retry_options: RetryOptions) -> Self {
        self.with_retry_options(retry_options)
    }

    fn provider_spec(&self) -> Arc<dyn ProviderSpec> {
        Arc::new(CohereRerankStandard::new().create_spec("cohere"))
    }

    fn build_context(&self) -> ProviderContext {
        ProviderContext::new(
            "cohere",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        )
    }

    /// Get the normalized provider context used by execution helpers.
    pub fn provider_context(&self) -> ProviderContext {
        self.build_context()
    }

    /// Get the configured base URL.
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Get the underlying HTTP client.
    pub fn http_client(&self) -> reqwest::Client {
        self.http_client.clone()
    }

    /// Get installed retry options.
    pub fn retry_options(&self) -> Option<RetryOptions> {
        self.retry_options.clone()
    }

    /// Get installed HTTP interceptors.
    pub fn http_interceptors(&self) -> Vec<Arc<dyn HttpInterceptor>> {
        self.config.http_interceptors.clone()
    }

    /// Get the installed custom HTTP transport.
    pub fn http_transport(&self) -> Option<Arc<dyn HttpTransport>> {
        self.config.http_transport.clone()
    }

    /// Set retry options.
    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options;
    }

    #[cfg(test)]
    pub(crate) fn _debug_base_url(&self) -> &str {
        &self.config.base_url
    }
}

#[async_trait]
impl RerankCapability for CohereClient {
    async fn rerank(&self, mut request: RerankRequest) -> Result<RerankResponse, LlmError> {
        if request.model.trim().is_empty() {
            request.model = self.config.common_params.model.clone();
        }
        if request.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Cohere rerank request requires a non-empty model id".to_string(),
            ));
        }

        let mut builder = RerankExecutorBuilder::new("cohere", self.http_client.clone())
            .with_spec(self.provider_spec())
            .with_context(self.build_context())
            .with_interceptors(self.config.http_interceptors.clone());

        if let Some(retry_options) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry_options);
        }
        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        let exec = builder.build_for_request(&request);
        RerankExecutor::execute(&*exec, request).await
    }

    fn supported_models(&self) -> Vec<String> {
        vec![self.config.common_params.model.clone()]
    }
}

impl ModelMetadata for CohereClient {
    fn provider_id(&self) -> &str {
        "cohere"
    }

    fn model_id(&self) -> &str {
        &self.config.common_params.model
    }
}

impl LlmClient for CohereClient {
    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("cohere")
    }

    fn supported_models(&self) -> Vec<String> {
        vec![self.config.common_params.model.clone()]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_rerank()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_rerank_capability(&self) -> Option<&dyn RerankCapability> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::cohere::CohereConfig;
    use async_trait::async_trait;
    use std::sync::Arc;

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl HttpInterceptor for NoopInterceptor {}

    #[derive(Clone, Default)]
    struct NoopTransport;

    #[async_trait]
    impl HttpTransport for NoopTransport {
        async fn execute_json(
            &self,
            _request: crate::execution::http::transport::HttpTransportRequest,
        ) -> Result<crate::execution::http::transport::HttpTransportResponse, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "NoopTransport does not execute requests".to_string(),
            ))
        }
    }

    #[test]
    fn cohere_client_exposes_provider_context_and_runtime_helpers() {
        let transport = Arc::new(NoopTransport);
        let interceptor = Arc::new(NoopInterceptor);
        let config = CohereConfig::new("test-key")
            .with_base_url("https://example.com/cohere")
            .with_model("rerank-v3.5")
            .with_http_transport(transport.clone())
            .with_http_interceptors(vec![interceptor]);
        let mut client = CohereClient::from_config(config).expect("client");

        client.set_retry_options(Some(RetryOptions::backoff()));

        let ctx = client.provider_context();
        assert_eq!(ctx.base_url, "https://example.com/cohere");
        assert_eq!(ctx.api_key.as_deref(), Some("test-key"));
        assert_eq!(client.base_url(), "https://example.com/cohere");
        assert!(client.retry_options().is_some());
        assert!(client.http_transport().is_some());
        assert_eq!(client.http_interceptors().len(), 1);
        let _http_client = client.http_client();
    }
}
