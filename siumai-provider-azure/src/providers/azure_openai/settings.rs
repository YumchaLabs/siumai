use crate::LlmError;
use crate::builder::BuilderBase;
use crate::execution::http::transport::HttpTransport;
use std::collections::HashMap;
use std::sync::Arc;

use super::{AzureOpenAiBuilder, AzureOpenAiConfig};

/// Package-level provider settings aligned with the supported subset of
/// `repo-ref/ai/packages/azure/src/azure-openai-provider.ts`.
///
/// Unlike `AzureOpenAiConfig`, this carrier intentionally does not require a deployment/model id.
/// Model selection happens later through `into_builder_for_model(...)`.
#[derive(Clone, Default)]
pub struct AzureOpenAIProviderSettings {
    pub resource_name: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub headers: HashMap<String, String>,
    pub fetch: Option<Arc<dyn HttpTransport>>,
    pub api_version: Option<String>,
    pub use_deployment_based_urls: Option<bool>,
}

impl AzureOpenAIProviderSettings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_resource_name<S: Into<String>>(mut self, resource_name: S) -> Self {
        self.resource_name = Some(resource_name.into());
        self
    }

    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers.extend(headers);
        self
    }

    pub fn with_header<K: Into<String>, V: Into<String>>(mut self, name: K, value: V) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    pub fn with_fetch(mut self, fetch: Arc<dyn HttpTransport>) -> Self {
        self.fetch = Some(fetch);
        self
    }

    pub fn with_api_version<S: Into<String>>(mut self, api_version: S) -> Self {
        self.api_version = Some(api_version.into());
        self
    }

    pub fn with_use_deployment_based_urls(mut self, enabled: bool) -> Self {
        self.use_deployment_based_urls = Some(enabled);
        self
    }

    /// Convert package-level provider settings into the provider-owned builder surface.
    pub fn into_builder(self) -> AzureOpenAiBuilder {
        let mut builder = AzureOpenAiBuilder::new(BuilderBase::default());

        if let Some(api_key) = self.api_key {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = self.base_url {
            builder = builder.base_url(base_url);
        } else if let Some(resource_name) = self.resource_name {
            builder = builder.resource_name(resource_name);
        }
        if !self.headers.is_empty() {
            builder = builder.headers(self.headers);
        }
        if let Some(fetch) = self.fetch {
            builder = builder.fetch(fetch);
        }
        if let Some(api_version) = self.api_version {
            builder = builder.api_version(api_version);
        }
        if let Some(use_deployment_based_urls) = self.use_deployment_based_urls {
            builder = builder.deployment_based_urls(use_deployment_based_urls);
        }

        builder
    }

    /// Convert package-level provider settings into a builder with a selected deployment/model id.
    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> AzureOpenAiBuilder {
        self.into_builder().model(model)
    }

    /// Convert package-level provider settings into the config-first carrier for a specific model.
    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<AzureOpenAiConfig, LlmError> {
        self.into_builder_for_model(model).into_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::execution::http::transport::{
        HttpTransportGetRequest, HttpTransportRequest, HttpTransportResponse,
    };
    use reqwest::header::HeaderMap;

    #[derive(Clone, Default)]
    struct NoopTransport;

    #[async_trait]
    impl HttpTransport for NoopTransport {
        async fn execute_json(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            Ok(HttpTransportResponse {
                status: 200,
                headers: HeaderMap::new(),
                body: b"{}".to_vec(),
            })
        }

        async fn execute_get(
            &self,
            _request: HttpTransportGetRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            Ok(HttpTransportResponse {
                status: 200,
                headers: HeaderMap::new(),
                body: b"{}".to_vec(),
            })
        }
    }

    #[test]
    fn azure_provider_settings_resource_name_and_http_inputs_flow_into_config() {
        let config = AzureOpenAIProviderSettings::new()
            .with_api_key("test-key")
            .with_resource_name("demo-resource")
            .with_header("x-test", "1")
            .with_api_version("2024-10-21")
            .with_use_deployment_based_urls(true)
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("deployment-id")
            .expect("settings into config");

        assert_eq!(
            config.base_url,
            "https://demo-resource.openai.azure.com/openai"
        );
        assert_eq!(config.common_params.model, "deployment-id");
        assert_eq!(config.url_config.api_version, "2024-10-21");
        assert!(config.url_config.use_deployment_based_urls);
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn azure_provider_settings_explicit_base_url_wins_over_resource_name() {
        let config = AzureOpenAIProviderSettings::new()
            .with_api_key("test-key")
            .with_resource_name("ignored-resource")
            .with_base_url("https://example.openai.azure.com/openai")
            .into_config_for_model("deployment-id")
            .expect("settings into config");

        assert_eq!(config.base_url, "https://example.openai.azure.com/openai");
    }
}
