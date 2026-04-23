use crate::auth::TokenProvider;
use crate::builder::BuilderBase;
use crate::error::LlmError;
use crate::execution::http::transport::HttpTransport;
use std::collections::HashMap;
use std::sync::Arc;

use super::{GoogleVertexBuilder, GoogleVertexConfig, SharedIdGenerator};

/// Provider-construction settings aligned with `@ai-sdk/google-vertex` package input.
///
/// Unlike `GoogleVertexConfig`, this type intentionally does not require a model id.
/// It mirrors the upstream package-level construction story, where model selection
/// happens on the provider member (`languageModel`, `imageModel`, `videoModel`, etc.).
#[derive(Clone, Default)]
pub struct GoogleVertexProviderSettings {
    /// Optional express-mode API key.
    pub api_key: Option<String>,
    /// Optional Vertex location used in enterprise mode.
    pub location: Option<String>,
    /// Optional Vertex project used in enterprise mode.
    pub project: Option<String>,
    /// Default headers applied to requests built from this provider settings object.
    pub headers: HashMap<String, String>,
    /// Optional custom HTTP transport, mirroring AI SDK `fetch`.
    pub fetch: Option<Arc<dyn HttpTransport>>,
    /// Optional stable ID generator aligned with AI SDK `generateId`.
    pub generate_id: Option<SharedIdGenerator>,
    /// Optional Vertex base URL override.
    pub base_url: Option<String>,
    /// Rust-side auth analogue for the Node-only `googleAuthOptions` path.
    pub token_provider: Option<Arc<dyn TokenProvider>>,
}

impl GoogleVertexProviderSettings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_location<S: Into<String>>(mut self, location: S) -> Self {
        self.location = Some(location.into());
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

    pub fn with_generate_id<F>(mut self, generate_id: F) -> Self
    where
        F: Fn() -> String + Send + Sync + 'static,
    {
        self.generate_id = Some(Arc::new(generate_id));
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
    pub fn into_builder(self) -> GoogleVertexBuilder {
        let api_key = self.api_key.clone();
        let mut builder = GoogleVertexBuilder::new(BuilderBase::default());

        if let Some(api_key) = self.api_key {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = self.base_url {
            builder = builder.base_url(base_url);
        }
        if let Some(project) = self.project {
            builder = builder.project(project);
        }
        if let Some(location) = self.location {
            builder = builder.location(location);
        }
        if !self.headers.is_empty() {
            builder = builder.headers(self.headers);
        }
        if let Some(fetch) = self.fetch {
            builder = builder.fetch(fetch);
        }
        if let Some(generate_id) = self.generate_id {
            builder = builder.with_shared_generate_id(generate_id);
        }
        if api_key.is_none()
            && let Some(token_provider) = self.token_provider
        {
            builder = builder.token_provider(token_provider);
        }

        builder
    }

    /// Convert package-level provider settings into a builder with a selected model id.
    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> GoogleVertexBuilder {
        self.into_builder().model(model)
    }

    /// Convert package-level provider settings into the config-first carrier for a specific model.
    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<GoogleVertexConfig, LlmError> {
        self.into_builder_for_model(model).into_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::StaticTokenProvider;

    #[test]
    fn provider_settings_into_builder_for_model_preserves_package_level_inputs() {
        let config = GoogleVertexProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/custom")
            .with_project("demo-project")
            .with_location("us-central1")
            .with_header("x-test", "1")
            .into_builder_for_model("gemini-2.5-flash")
            .into_config()
            .expect("settings into config");

        assert_eq!(config.base_url, "https://example.com/custom");
        assert_eq!(config.api_key.as_deref(), Some("test-key"));
        assert_eq!(config.model, "gemini-2.5-flash");
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
    }

    #[test]
    fn provider_settings_token_provider_is_ignored_when_api_key_is_present() {
        let config = GoogleVertexProviderSettings::new()
            .with_api_key("test-key")
            .with_token_provider(Arc::new(StaticTokenProvider::new("token")))
            .into_config_for_model("gemini-2.5-flash")
            .expect("settings into config");

        assert!(config.api_key.is_some());
        assert!(config.token_provider.is_none());
    }

    #[test]
    fn provider_settings_preserve_custom_generate_id() {
        let config = GoogleVertexProviderSettings::new()
            .with_api_key("test-key")
            .with_generate_id(|| "custom-vertex-id".to_string())
            .into_config_for_model("gemini-2.5-flash")
            .expect("settings into config");

        assert_eq!(
            config.generate_id.as_ref().map(|generate_id| generate_id()),
            Some("custom-vertex-id".to_string())
        );
    }

    #[test]
    fn provider_settings_support_enterprise_mode_without_model_until_selection_time() {
        let config = GoogleVertexProviderSettings::new()
            .with_project("demo-project")
            .with_location("global")
            .into_builder()
            .model("gemini-2.5-pro")
            .into_config()
            .expect("settings into config");

        assert_eq!(
            config.base_url,
            crate::auth::vertex::google_vertex_base_url("demo-project", "global")
        );
        assert_eq!(config.model, "gemini-2.5-pro");
    }
}
