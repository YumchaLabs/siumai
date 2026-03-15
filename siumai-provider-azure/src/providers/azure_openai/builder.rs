//! `Azure OpenAI` builder.

use super::{AzureChatMode, AzureOpenAiClient, AzureOpenAiConfig, AzureUrlConfig};
use crate::builder::{BuilderBase, ProviderCore};
use crate::error::LlmError;
use crate::retry_api::RetryOptions;
use std::sync::Arc;

/// Provider-owned builder for Azure OpenAI clients.
#[derive(Clone)]
pub struct AzureOpenAiBuilder {
    pub(crate) core: ProviderCore,
    config: AzureOpenAiConfig,
}

impl AzureOpenAiBuilder {
    pub fn new(base: BuilderBase) -> Self {
        Self {
            core: ProviderCore::new(base),
            config: AzureOpenAiConfig::new(""),
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

    pub fn language_model<S: Into<String>>(self, model: S) -> Self {
        self.model(model)
    }

    pub fn embedding_model<S: Into<String>>(self, model: S) -> Self {
        self.model(model)
    }

    pub fn image_model<S: Into<String>>(self, model: S) -> Self {
        self.model(model)
    }

    pub fn speech_model<S: Into<String>>(self, model: S) -> Self {
        self.model(model)
    }

    pub fn transcription_model<S: Into<String>>(self, model: S) -> Self {
        self.model(model)
    }

    pub fn chat_mode(mut self, mode: AzureChatMode) -> Self {
        self.config = self.config.with_chat_mode(mode);
        self
    }

    pub fn use_responses_api(self, enabled: bool) -> Self {
        if enabled {
            self.chat_mode(AzureChatMode::Responses)
        } else {
            self.chat_mode(AzureChatMode::ChatCompletions)
        }
    }

    pub fn responses(self) -> Self {
        self.chat_mode(AzureChatMode::Responses)
    }

    pub fn chat_completions(self) -> Self {
        self.chat_mode(AzureChatMode::ChatCompletions)
    }

    pub fn api_version<S: Into<String>>(mut self, api_version: S) -> Self {
        let mut url_config = self.config.url_config.clone();
        url_config.api_version = api_version.into();
        self.config = self.config.with_url_config(url_config);
        self
    }

    pub fn deployment_based_urls(mut self, enabled: bool) -> Self {
        let mut url_config = self.config.url_config.clone();
        url_config.use_deployment_based_urls = enabled;
        self.config = self.config.with_url_config(url_config);
        self
    }

    pub fn url_config(mut self, url_config: AzureUrlConfig) -> Self {
        self.config = self.config.with_url_config(url_config);
        self
    }

    pub fn provider_metadata_key(mut self, key: &'static str) -> Self {
        self.config = self.config.with_provider_metadata_key(key);
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

    pub fn with_http_config(mut self, http_config: crate::types::HttpConfig) -> Self {
        self.core.http_config = http_config;
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

    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<
            Arc<dyn crate::execution::middleware::language_model::LanguageModelMiddleware>,
        >,
    ) -> Self {
        self.config = self.config.with_model_middlewares(middlewares);
        self
    }

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

    pub fn into_config(mut self) -> Result<AzureOpenAiConfig, LlmError> {
        if self.config.api_key.trim().is_empty()
            && let Ok(api_key) = std::env::var("AZURE_API_KEY")
        {
            self.config = self.config.with_api_key(api_key);
        }

        if self.config.base_url.trim().is_empty()
            && let Ok(resource) = std::env::var("AZURE_RESOURCE_NAME")
        {
            let resource = resource.trim();
            if !resource.is_empty() {
                self.config = self
                    .config
                    .with_base_url(format!("https://{resource}.openai.azure.com/openai"));
            }
        }

        let mut config = self.config;
        config.http_config = self.core.http_config.clone();
        config.http_transport = self.core.http_transport.clone();
        config.http_interceptors = self.core.get_http_interceptors();
        config.validate()?;
        Ok(config)
    }

    pub fn build(self) -> Result<AzureOpenAiClient, LlmError> {
        let http_client_override = self.core.base.http_client.clone();
        let retry_options = self.core.retry_options.clone();
        let config = self.into_config()?;

        let client = if let Some(http_client) = http_client_override {
            AzureOpenAiClient::with_http_client(config, http_client)?
        } else {
            AzureOpenAiClient::from_config(config)?
        };

        Ok(client.with_retry_options(retry_options))
    }
}

#[cfg(test)]
mod config_first_tests {
    use super::*;
    use crate::execution::middleware::language_model::LanguageModelMiddleware;
    use crate::types::HttpConfig;
    use std::sync::Arc;
    use std::time::Duration;

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl crate::execution::http::interceptor::HttpInterceptor for NoopInterceptor {}

    #[derive(Clone, Default)]
    struct NoopMiddleware;

    impl LanguageModelMiddleware for NoopMiddleware {}

    #[test]
    fn into_config_preserves_explicit_fields_and_http_interceptors() {
        let mut http_config = HttpConfig {
            timeout: Some(Duration::from_secs(3)),
            ..Default::default()
        };
        http_config.stream_disable_compression = true;

        let cfg = AzureOpenAiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .base_url("https://example.openai.azure.com/openai")
            .model("deployment-id")
            .chat_completions()
            .api_version("2024-10-21")
            .with_http_config(http_config)
            .with_http_interceptor(Arc::new(NoopInterceptor))
            .with_model_middlewares(vec![Arc::new(NoopMiddleware)])
            .into_config()
            .expect("into_config");

        assert_eq!(cfg.api_key, "test-key");
        assert_eq!(cfg.base_url, "https://example.openai.azure.com/openai");
        assert_eq!(cfg.common_params.model, "deployment-id");
        assert_eq!(cfg.chat_mode, AzureChatMode::ChatCompletions);
        assert_eq!(cfg.url_config.api_version, "2024-10-21");
        assert_eq!(cfg.http_config.timeout, Some(Duration::from_secs(3)));
        assert!(cfg.http_config.stream_disable_compression);
        assert_eq!(cfg.http_interceptors.len(), 1);
        assert_eq!(cfg.model_middlewares.len(), 1);
    }

    #[test]
    fn build_and_into_config_converge_on_config_first_client_construction() {
        let builder = AzureOpenAiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .base_url("https://example.openai.azure.com/openai")
            .model("deployment-id");

        let cfg = builder.clone().into_config().expect("into_config");
        let built = builder.build().expect("build client");
        let from_config = AzureOpenAiClient::from_config(cfg).expect("from_config client");

        assert_eq!(crate::client::LlmClient::provider_id(&built), "azure");
        assert_eq!(crate::client::LlmClient::provider_id(&from_config), "azure");
        assert_eq!(
            crate::client::LlmClient::supported_models(&built),
            crate::client::LlmClient::supported_models(&from_config)
        );
    }
}
