use crate::LlmError;
use crate::builder::BuilderBase;
use crate::execution::http::transport::HttpTransport;
use std::collections::HashMap;
use std::sync::Arc;

use super::{OpenAiBuilder, OpenAiConfig};

/// Package-level provider settings aligned with the supported subset of
/// `repo-ref/ai/packages/openai/src/openai-provider.ts`.
///
/// Unlike `OpenAiConfig`, this carrier intentionally does not require a model id.
/// Model selection happens later through `into_builder_for_model(...)`.
///
/// Note: the upstream `name` field is intentionally deferred here. Siumai currently keeps the
/// canonical `openai` provider identity fixed across `provider_id`, `providerMetadata`, and
/// registry/factory routing, so exposing a partially effective display-only alias would be
/// misleading.
#[derive(Clone, Default)]
pub struct OpenAIProviderSettings {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub organization: Option<String>,
    pub project: Option<String>,
    pub headers: HashMap<String, String>,
    pub fetch: Option<Arc<dyn HttpTransport>>,
}

impl OpenAIProviderSettings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn with_organization<S: Into<String>>(mut self, organization: S) -> Self {
        self.organization = Some(organization.into());
        self
    }

    pub fn with_project<S: Into<String>>(mut self, project: S) -> Self {
        self.project = Some(project.into());
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

    /// Convert package-level provider settings into the provider-owned builder surface.
    pub fn into_builder(self) -> OpenAiBuilder {
        let mut builder = OpenAiBuilder::new(BuilderBase::default());

        if let Some(api_key) = self.api_key {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = self.base_url {
            builder = builder.base_url(base_url);
        }
        if let Some(organization) = self.organization {
            builder = builder.organization(organization);
        }
        if let Some(project) = self.project {
            builder = builder.project(project);
        }
        if !self.headers.is_empty() {
            builder = builder.headers(self.headers);
        }
        if let Some(fetch) = self.fetch {
            builder = builder.fetch(fetch);
        }

        builder
    }

    /// Convert package-level provider settings into a builder with a selected model id.
    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> OpenAiBuilder {
        self.into_builder().model(model)
    }

    /// Convert package-level provider settings into the config-first carrier for a specific model.
    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<OpenAiConfig, LlmError> {
        self.into_builder_for_model(model).into_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransportGetRequest, HttpTransportRequest, HttpTransportResponse,
    };
    use async_trait::async_trait;
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
    fn openai_provider_settings_into_builder_for_model_preserves_package_level_inputs() {
        let config = OpenAIProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/custom")
            .with_organization("org-123")
            .with_project("proj-456")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("gpt-4.1-mini")
            .expect("settings into config");

        assert_eq!(config.base_url, "https://example.com/custom");
        assert_eq!(config.organization.as_deref(), Some("org-123"));
        assert_eq!(config.project.as_deref(), Some("proj-456"));
        assert_eq!(config.common_params.model, "gpt-4.1-mini");
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn openai_provider_settings_can_defer_model_selection_until_builder_time() {
        let config = OpenAIProviderSettings::new()
            .with_api_key("test-key")
            .into_builder()
            .model("gpt-4.1")
            .into_config()
            .expect("settings into config");

        assert_eq!(config.common_params.model, "gpt-4.1");
    }
}
