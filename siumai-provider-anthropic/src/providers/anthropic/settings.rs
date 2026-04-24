use crate::builder::BuilderBase;
use crate::error::LlmError;
use crate::execution::http::transport::HttpTransport;
use std::collections::HashMap;
use std::sync::Arc;

use super::{AnthropicBuilder, AnthropicConfig};

/// Package-level provider settings aligned with `@ai-sdk/anthropic`.
///
/// This carrier is model-agnostic. Model selection happens later through
/// `into_builder_for_model(...)` or `into_config_for_model(...)`.
#[derive(Clone, Default)]
pub struct AnthropicProviderSettings {
    /// Optional Anthropic API key for `x-api-key` authentication.
    pub api_key: Option<String>,
    /// Optional Anthropic auth token for `Authorization: Bearer ...` authentication.
    pub auth_token: Option<String>,
    /// Optional base URL override.
    pub base_url: Option<String>,
    /// Default headers applied to requests built from this settings object.
    pub headers: HashMap<String, String>,
    /// Optional custom HTTP transport, mirroring AI SDK `fetch`.
    pub fetch: Option<Arc<dyn HttpTransport>>,
}

impl AnthropicProviderSettings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_auth_token<S: Into<String>>(mut self, auth_token: S) -> Self {
        self.auth_token = Some(auth_token.into());
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

    pub fn into_builder(self) -> AnthropicBuilder {
        let mut builder = AnthropicBuilder::new(BuilderBase::default());

        if let Some(api_key) = self.api_key {
            builder = builder.api_key(api_key);
        }
        if let Some(auth_token) = self.auth_token {
            builder = builder.auth_token(auth_token);
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

    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> AnthropicBuilder {
        self.into_builder().model(model)
    }

    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<AnthropicConfig, LlmError> {
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
    use secrecy::ExposeSecret;

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
    fn anthropic_provider_settings_into_config_preserve_supported_inputs() {
        let config = AnthropicProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/anthropic")
            .with_header("x-test", "1")
            .with_fetch(Arc::new(NoopTransport))
            .into_config_for_model("claude-sonnet-4-5-20250929")
            .expect("settings into config");

        assert_eq!(config.api_key.expose_secret(), "test-key");
        assert_eq!(config.base_url, "https://example.com/anthropic");
        assert_eq!(config.common_params.model, "claude-sonnet-4-5-20250929");
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
        assert!(config.http_transport.is_some());
    }

    #[test]
    fn anthropic_provider_settings_support_auth_token() {
        let config = AnthropicProviderSettings::new()
            .with_auth_token("test-token")
            .with_base_url("https://example.com/anthropic")
            .into_config_for_model("claude-sonnet-4-5-20250929")
            .expect("settings into config");

        assert!(config.api_key.expose_secret().is_empty());
        assert_eq!(
            config
                .http_config
                .headers
                .get("Authorization")
                .map(String::as_str),
            Some("Bearer test-token")
        );
    }
}
