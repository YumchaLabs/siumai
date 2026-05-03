use crate::auth::TokenProvider;
use crate::builder::BuilderBase;
use crate::error::LlmError;
use crate::execution::http::transport::HttpTransport;
use std::collections::HashMap;
use std::sync::Arc;

use super::{VertexAnthropicBuilder, VertexAnthropicConfig};

/// Package-level provider settings aligned with `@ai-sdk/google-vertex/anthropic`.
///
/// Unlike `VertexAnthropicConfig`, this settings carrier intentionally does not require a model id.
/// Model selection happens later through `into_builder_for_model(...)` or the builder aliases.
#[derive(Clone, Default)]
pub struct GoogleVertexAnthropicProviderSettings {
    /// Optional Google Cloud project id.
    pub project: Option<String>,
    /// Optional Google Cloud location / region.
    pub location: Option<String>,
    /// Default headers applied to requests built from this settings object.
    pub headers: HashMap<String, String>,
    /// Optional custom HTTP transport, mirroring AI SDK `fetch`.
    pub fetch: Option<Arc<dyn HttpTransport>>,
    /// Optional base URL override.
    pub base_url: Option<String>,
    /// Rust-side auth analogue for the AI SDK's fetch/runtime auth path.
    pub token_provider: Option<Arc<dyn TokenProvider>>,
}

impl GoogleVertexAnthropicProviderSettings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_project<S: Into<String>>(mut self, project: S) -> Self {
        self.project = Some(project.into());
        self
    }

    pub fn with_location<S: Into<String>>(mut self, location: S) -> Self {
        self.location = Some(location.into());
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

    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn with_token_provider(mut self, token_provider: Arc<dyn TokenProvider>) -> Self {
        self.token_provider = Some(token_provider);
        self
    }

    /// Convert package-level provider settings into the provider-owned builder surface.
    pub fn into_builder(self) -> VertexAnthropicBuilder {
        let base = BuilderBase {
            default_headers: self.headers,
            ..Default::default()
        };

        let mut builder = VertexAnthropicBuilder::new(base);

        if let Some(project) = self.project {
            builder = builder.project(project);
        }
        if let Some(location) = self.location {
            builder = builder.location(location);
        }
        if let Some(base_url) = self.base_url {
            builder = builder.base_url(base_url);
        }
        if let Some(fetch) = self.fetch {
            builder = builder.fetch(fetch);
        }
        if let Some(token_provider) = self.token_provider {
            builder = builder.token_provider(token_provider);
        }

        builder
    }

    /// Convert package-level provider settings into a builder with a selected model id.
    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> VertexAnthropicBuilder {
        self.into_builder().model(model)
    }

    /// Convert package-level provider settings into the config-first carrier for a specific model.
    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<VertexAnthropicConfig, LlmError> {
        self.into_builder_for_model(model).into_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::StaticTokenProvider;
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
    fn provider_settings_derive_vertex_base_url_from_project_and_location() {
        let config = GoogleVertexAnthropicProviderSettings::new()
            .with_project("demo-project")
            .with_location("global")
            .with_header("x-test", "1")
            .into_config_for_model("claude-sonnet-4-5-latest")
            .expect("settings into config");

        assert_eq!(
            config.base_url,
            crate::auth::vertex::google_vertex_anthropic_base_url("demo-project", "global")
        );
        assert_eq!(config.model, "claude-sonnet-4-5-latest");
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
    }

    #[test]
    fn provider_settings_base_url_override_wins_and_trims_trailing_slash() {
        let config = GoogleVertexAnthropicProviderSettings::new()
            .with_project("demo-project")
            .with_location("global")
            .with_base_url("https://example.com/custom/")
            .into_config_for_model("claude-sonnet-4-5-latest")
            .expect("settings into config");

        assert_eq!(config.base_url, "https://example.com/custom");
    }

    #[test]
    fn provider_settings_preserve_fetch_and_token_provider() {
        let config = GoogleVertexAnthropicProviderSettings::new()
            .with_base_url("https://example.com/custom")
            .with_fetch(Arc::new(NoopTransport))
            .with_token_provider(Arc::new(StaticTokenProvider::new("token")))
            .into_config_for_model("claude-sonnet-4-5-latest")
            .expect("settings into config");

        assert!(config.http_transport.is_some());
        assert!(config.token_provider.is_some());
    }

    #[test]
    fn provider_settings_can_defer_model_selection_until_builder_time() {
        let config = GoogleVertexAnthropicProviderSettings::new()
            .with_project("demo-project")
            .with_location("us-central1")
            .into_builder()
            .language_model("claude-sonnet-4-5-latest")
            .into_config()
            .expect("settings into config");

        assert_eq!(config.model, "claude-sonnet-4-5-latest");
        assert_eq!(
            config.base_url,
            crate::auth::vertex::google_vertex_anthropic_base_url("demo-project", "us-central1")
        );
    }
}
