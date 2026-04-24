use crate::LlmError;
use crate::builder::BuilderBase;
use crate::execution::http::transport::HttpTransport;
use std::collections::HashMap;
use std::sync::Arc;

use super::{CohereBuilder, CohereConfig};

/// Package-level provider settings aligned with the supported subset of
/// `repo-ref/ai/packages/cohere/src/cohere-provider.ts`.
///
/// Unlike `CohereConfig`, this carrier intentionally does not require a model id.
/// Model selection happens later through `into_builder_for_model(...)`.
///
/// Note: upstream `generateId` is intentionally deferred because the current Cohere runtime does
/// not own a comparable provider-level stable ID hook.
#[derive(Clone, Default)]
pub struct CohereProviderSettings {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub headers: HashMap<String, String>,
    pub fetch: Option<Arc<dyn HttpTransport>>,
}

impl CohereProviderSettings {
    pub fn new() -> Self {
        Self::default()
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

    /// Convert package-level provider settings into the provider-owned builder surface.
    pub fn into_builder(self) -> CohereBuilder {
        let mut builder = CohereBuilder::new(BuilderBase::default());

        if let Some(api_key) = self.api_key {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = self.base_url {
            builder = builder.base_url(base_url);
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
    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> CohereBuilder {
        self.into_builder().model(model)
    }

    /// Convert package-level provider settings into the config-first carrier for a specific model.
    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<CohereConfig, LlmError> {
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
    fn cohere_provider_settings_into_config_preserve_supported_inputs() {
        let config = CohereProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/cohere")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("command-a-03-2025")
            .expect("settings into config");

        assert_eq!(config.base_url, "https://example.com/cohere");
        assert_eq!(config.common_params.model, "command-a-03-2025");
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn cohere_provider_settings_can_defer_model_selection_until_builder_time() {
        let config = CohereProviderSettings::new()
            .with_api_key("test-key")
            .into_builder()
            .reranking_model("rerank-v3.5")
            .into_config()
            .expect("settings into config");

        assert_eq!(config.common_params.model, "rerank-v3.5");
    }
}
