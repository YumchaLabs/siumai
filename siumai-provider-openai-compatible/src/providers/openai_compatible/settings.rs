use crate::LlmError;
use crate::builder::BuilderBase;
use crate::execution::http::transport::HttpTransport;
use std::collections::HashMap;
use std::sync::Arc;

use super::{MistralConfig, OpenAiCompatibleBuilder};

/// Package-level Mistral provider settings aligned with
/// `repo-ref/ai/packages/mistral/src/mistral-provider.ts`.
///
/// This carrier is model-agnostic. Model selection happens later through
/// `into_builder_for_model(...)` or `into_config_for_model(...)`.
///
/// Note: upstream `generateId` is intentionally deferred until the shared compat runtime owns an
/// honest provider-level stable-id hook.
#[derive(Clone, Default)]
pub struct MistralProviderSettings {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub headers: HashMap<String, String>,
    pub fetch: Option<Arc<dyn HttpTransport>>,
}

impl MistralProviderSettings {
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

    pub fn into_builder(self) -> OpenAiCompatibleBuilder {
        let mut builder = OpenAiCompatibleBuilder::new(BuilderBase::default(), "mistral");

        if let Some(api_key) = self.api_key {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = self.base_url {
            builder = builder.base_url(base_url);
        }
        if !self.headers.is_empty() {
            builder = builder.custom_headers(self.headers);
        }
        if let Some(fetch) = self.fetch {
            builder = builder.fetch(fetch);
        }

        builder
    }

    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> OpenAiCompatibleBuilder {
        self.into_builder().model(model)
    }

    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<MistralConfig, LlmError> {
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
    fn mistral_provider_settings_into_config_preserve_supported_inputs() {
        let config = MistralProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/mistral")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("mistral-large-latest")
            .expect("settings into config");

        assert_eq!(config.provider_id, "mistral");
        assert_eq!(config.base_url, "https://example.com/mistral");
        assert_eq!(config.common_params.model, "mistral-large-latest");
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }
}
