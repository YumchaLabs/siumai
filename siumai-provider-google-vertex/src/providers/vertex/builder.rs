//! Google Vertex provider builder.
//!
//! This builder aims to mirror Vercel AI SDK's `createVertex()` base URL behavior:
//! - Express mode (API key): `https://aiplatform.googleapis.com/v1/publishers/google`
//! - Enterprise mode (project+location): `https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/publishers/google`

use crate::auth::TokenProvider;
use crate::builder::{BuilderBase, ProviderCore};
use crate::error::LlmError;
use crate::retry_api::RetryOptions;
use crate::types::{CommonParams, HttpConfig};
use std::sync::Arc;

use super::{GoogleVertexClient, GoogleVertexConfig};

#[derive(Clone)]
pub struct GoogleVertexBuilder {
    pub(crate) core: ProviderCore,
    api_key: Option<String>,
    base_url: Option<String>,
    project: Option<String>,
    location: Option<String>,
    token_provider: Option<Arc<dyn TokenProvider>>,
    common_params: CommonParams,
}

impl GoogleVertexBuilder {
    pub fn new(base: BuilderBase) -> Self {
        Self {
            core: ProviderCore::new(base),
            api_key: None,
            base_url: None,
            project: None,
            location: None,
            token_provider: None,
            common_params: CommonParams::default(),
        }
    }

    /// Express mode API key (query param `?key=...`).
    pub fn api_key<S: Into<String>>(mut self, key: S) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override base URL (full prefix).
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set Vertex project (enterprise mode base URL).
    pub fn project<S: Into<String>>(mut self, project: S) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Set Vertex location (enterprise mode base URL; supports `global`).
    pub fn location<S: Into<String>>(mut self, location: S) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Set Bearer token provider (enterprise mode).
    pub fn token_provider(mut self, provider: Arc<dyn TokenProvider>) -> Self {
        self.token_provider = Some(provider);
        self
    }

    /// Set the model id (unified via common_params).
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    /// Alias for `model(...)` (Vercel-aligned naming: `languageModel(modelId)`).
    pub fn language_model<S: Into<String>>(self, model: S) -> Self {
        self.model(model)
    }

    /// Set the embedding model id (Vercel-aligned naming: `embeddingModel(modelId)`).
    ///
    /// Note: Siumai's Vertex client is multi-capability; this method simply sets the
    /// default model id used by `EmbeddingCapability::embed(...)` when the request does
    /// not specify a model.
    pub fn embedding_model<S: Into<String>>(self, model: S) -> Self {
        self.model(model)
    }

    /// Deprecated alias for `embedding_model(...)` (Vercel-aligned naming: `textEmbeddingModel`).
    #[deprecated(note = "Use `embedding_model(...)` instead.")]
    pub fn text_embedding_model<S: Into<String>>(self, model: S) -> Self {
        self.embedding_model(model)
    }

    // === Common configuration (delegated to ProviderCore) ===

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

    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.core = self.core.http_stream_disable_compression(disable);
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

    /// Alias for `with_http_transport(...)` (Vercel-aligned: `fetch`).
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

    pub fn build(self) -> Result<GoogleVertexClient, LlmError> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("GOOGLE_VERTEX_API_KEY").ok())
            .and_then(|k| {
                let trimmed = k.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            });

        let base_url = if let Some(b) = self.base_url {
            b
        } else if api_key.is_some() {
            crate::auth::vertex::GOOGLE_VERTEX_EXPRESS_BASE_URL.to_string()
        } else {
            let project = self
                .project
                .or_else(|| std::env::var("GOOGLE_VERTEX_PROJECT").ok());
            let location = self
                .location
                .or_else(|| std::env::var("GOOGLE_VERTEX_LOCATION").ok());
            let Some(project) = project else {
                return Err(LlmError::ConfigurationError(
                    "Google Vertex requires `base_url`, `api_key` (express mode), or a `project` (GOOGLE_VERTEX_PROJECT)".to_string(),
                ));
            };
            let Some(location) = location else {
                return Err(LlmError::ConfigurationError(
                    "Google Vertex requires `base_url`, `api_key` (express mode), or a `location` (GOOGLE_VERTEX_LOCATION)".to_string(),
                ));
            };
            crate::auth::vertex::google_vertex_base_url(project.trim(), location.trim())
        };

        let model_id = if self.common_params.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Google Vertex requires a non-empty model id".to_string(),
            ));
        } else {
            self.common_params.model.clone()
        };

        let http_client = self.core.build_http_client()?;

        // Vercel AI SDK alignment (`@ai-sdk/google-vertex` exports the node variant by default):
        // - If no API key is provided (enterprise mode) and the user didn't supply an explicit
        //   Authorization header or custom token provider, auto-enable ADC-based auth (when available).
        let token_provider = {
            #[cfg(feature = "gcp")]
            {
                fn has_auth_header(headers: &std::collections::HashMap<String, String>) -> bool {
                    headers
                        .keys()
                        .any(|k| k.eq_ignore_ascii_case("authorization"))
                }

                let mut token_provider = self.token_provider;
                if api_key.is_none()
                    && token_provider.is_none()
                    && !has_auth_header(&self.core.http_config.headers)
                {
                    token_provider = Some(std::sync::Arc::new(
                        crate::auth::adc::AdcTokenProvider::default_client(),
                    ));
                }
                token_provider
            }
            #[cfg(not(feature = "gcp"))]
            {
                self.token_provider
            }
        };

        let cfg = GoogleVertexConfig {
            base_url,
            model: model_id.clone(),
            api_key,
            http_config: HttpConfig {
                ..self.core.http_config.clone()
            },
            http_transport: self.core.http_transport.clone(),
            token_provider,
        };

        let mut client =
            GoogleVertexClient::new(cfg, http_client).with_common_params(self.common_params);

        if let Some(opts) = self.core.retry_options.clone() {
            client = client.with_retry_options(opts);
        }

        let interceptors = self.core.get_http_interceptors();
        if !interceptors.is_empty() {
            client = client.with_interceptors(interceptors);
        }

        Ok(client)
    }
}

#[cfg(all(test, feature = "gcp"))]
mod tests {
    use super::*;
    use crate::builder::BuilderBase;

    #[test]
    fn build_auto_enables_adc_token_provider_when_missing_auth() {
        let client = GoogleVertexBuilder::new(BuilderBase::default())
            .project("p")
            .location("us-central1")
            .model("gemini-2.0-flash")
            .build()
            .expect("build");

        assert!(!client._debug_has_api_key());
        assert!(client._debug_has_token_provider());
    }

    #[test]
    fn build_does_not_override_user_authorization_header() {
        let mut base = BuilderBase::default();
        base.default_headers
            .insert("Authorization".to_string(), "Bearer user".to_string());

        let client = GoogleVertexBuilder::new(base)
            .project("p")
            .location("us-central1")
            .model("gemini-2.0-flash")
            .build()
            .expect("build");

        assert!(!client._debug_has_api_key());
        assert!(
            !client._debug_has_token_provider(),
            "user Authorization should suppress auto ADC"
        );
    }

    #[test]
    fn build_does_not_enable_adc_in_express_mode() {
        let client = GoogleVertexBuilder::new(BuilderBase::default())
            .api_key("k")
            .model("gemini-2.0-flash")
            .build()
            .expect("build");

        assert!(client._debug_has_api_key());
        assert!(!client._debug_has_token_provider());
    }
}
