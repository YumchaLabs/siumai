use crate::LlmError;
use crate::builder::BuilderBase;
use crate::execution::http::transport::HttpTransport;
use std::collections::HashMap;
use std::sync::Arc;

use super::{GeminiBuilder, GeminiConfig, SharedIdGenerator};

/// Provider-construction settings aligned with the package-level `@ai-sdk/google` input.
///
/// Unlike `GeminiConfig`, this type intentionally does not require a model id.
/// It mirrors the upstream package-level construction story, where model selection
/// happens on the provider member (`languageModel`, `embeddingModel`, `imageModel`, etc.).
#[derive(Clone, Default)]
pub struct GoogleProviderSettings {
    /// Optional API key for Google Generative AI.
    pub api_key: Option<String>,
    /// Optional base URL override.
    pub base_url: Option<String>,
    /// Default headers applied to requests built from this provider settings object.
    pub headers: HashMap<String, String>,
    /// Optional custom HTTP transport, mirroring AI SDK `fetch`.
    pub fetch: Option<Arc<dyn HttpTransport>>,
    /// Optional stable ID generator aligned with AI SDK `generateId`.
    pub generate_id: Option<SharedIdGenerator>,
    /// Optional provider-facing display name aligned with AI SDK `name`.
    pub name: Option<String>,
}

impl GoogleProviderSettings {
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

    pub fn with_generate_id<F>(mut self, generate_id: F) -> Self
    where
        F: Fn() -> String + Send + Sync + 'static,
    {
        self.generate_id = Some(Arc::new(generate_id));
        self
    }

    pub fn with_name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Convert package-level provider settings into the provider-owned builder surface.
    pub fn into_builder(self) -> GeminiBuilder {
        let mut builder = GeminiBuilder::new(BuilderBase::default());

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
        if let Some(generate_id) = self.generate_id {
            builder = builder.with_shared_generate_id(generate_id);
        }
        builder = builder.name(
            self.name
                .unwrap_or_else(|| "google.generative-ai".to_string()),
        );

        builder
    }

    /// Convert package-level provider settings into a builder with a selected model id.
    pub fn into_builder_for_model<S: Into<String>>(self, model: S) -> GeminiBuilder {
        self.into_builder().model(model)
    }

    /// Convert package-level provider settings into the config-first carrier for a specific model.
    pub fn into_config_for_model<S: Into<String>>(
        self,
        model: S,
    ) -> Result<GeminiConfig, LlmError> {
        self.into_builder_for_model(model).into_config()
    }
}

/// Deprecated Google Generative AI settings alias preserved for package-surface parity.
#[allow(deprecated)]
#[deprecated(note = "Use `GoogleProviderSettings` instead.")]
pub type GoogleGenerativeAIProviderSettings = GoogleProviderSettings;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn google_provider_settings_into_builder_for_model_preserves_package_level_inputs() {
        let config = GoogleProviderSettings::new()
            .with_api_key("test-key")
            .with_base_url("https://example.com/custom")
            .with_header("x-test", "1")
            .into_builder_for_model("gemini-2.5-flash")
            .into_config()
            .expect("settings into config");

        assert_eq!(config.base_url, "https://example.com/custom");
        assert_eq!(config.model, "gemini-2.5-flash");
        assert_eq!(
            config.http_config.headers.get("x-test").map(String::as_str),
            Some("1")
        );
    }

    #[test]
    fn google_provider_settings_preserve_custom_generate_id() {
        let config = GoogleProviderSettings::new()
            .with_api_key("test-key")
            .with_generate_id(|| "custom-google-id".to_string())
            .into_config_for_model("gemini-2.5-flash")
            .expect("settings into config");

        assert_eq!(
            config.generate_id.as_ref().map(|generate_id| generate_id()),
            Some("custom-google-id".to_string())
        );
    }

    #[test]
    fn google_provider_settings_default_to_google_provider_name() {
        let config = GoogleProviderSettings::new()
            .with_api_key("test-key")
            .into_config_for_model("gemini-2.5-flash")
            .expect("settings into config");

        assert_eq!(config.provider_name(), "google.generative-ai");
    }

    #[test]
    fn google_provider_settings_preserve_custom_name() {
        let config = GoogleProviderSettings::new()
            .with_api_key("test-key")
            .with_name("my-gemini-proxy")
            .into_config_for_model("gemini-2.5-flash")
            .expect("settings into config");

        assert_eq!(config.provider_name(), "my-gemini-proxy");
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_google_generative_ai_provider_settings_alias_stays_compatible() {
        let config = GoogleGenerativeAIProviderSettings::new()
            .with_api_key("test-key")
            .into_config_for_model("gemini-2.5-pro")
            .expect("settings into config");

        assert_eq!(config.model, "gemini-2.5-pro");
    }
}
