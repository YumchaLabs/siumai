//! Provider factory implementations.

use super::*;

/// Google Vertex provider factory (Imagen via Vertex AI).
#[cfg(feature = "google-vertex")]
pub struct GoogleVertexProviderFactory;

#[cfg(feature = "google-vertex")]
#[async_trait::async_trait]
impl ProviderFactory for GoogleVertexProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = crate::native_provider_metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == "vertex")
            .map(|m| m.capabilities)
            .unwrap_or_else(ProviderCapabilities::new)
    }

    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Resolve HTTP configuration and client.
        let http_config = ctx.http_config.clone().unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        let api_key = ctx.api_key.clone().or_else(|| {
            std::env::var("GOOGLE_VERTEX_API_KEY").ok().and_then(|k| {
                let trimmed = k.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            })
        });

        let base_url = if let Some(b) = ctx.base_url.clone() {
            b
        } else if api_key.is_some() {
            crate::utils::vertex::GOOGLE_VERTEX_EXPRESS_BASE_URL.to_string()
        } else {
            let project = std::env::var("GOOGLE_VERTEX_PROJECT").ok().and_then(|v| {
                let trimmed = v.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            });
            let location = std::env::var("GOOGLE_VERTEX_LOCATION").ok().and_then(|v| {
                let trimmed = v.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            });

            let (project, location) = match (project, location) {
                (Some(p), Some(l)) => (p, l),
                _ => {
                    return Err(LlmError::ConfigurationError(
                        "Google Vertex requires `base_url`, `api_key` (express mode), or env vars GOOGLE_VERTEX_PROJECT + GOOGLE_VERTEX_LOCATION".to_string(),
                    ))
                }
            };

            crate::utils::vertex::google_vertex_base_url(&project, &location)
        };

        // Resolve common parameters.
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        crate::registry::factory::build_google_vertex_client(
            base_url,
            api_key,
            http_client,
            common_params,
            http_config,
            ctx.gemini_token_provider.clone(),
            ctx.tracing_config.clone(),
            ctx.retry_options.clone(),
            ctx.http_interceptors.clone(),
            ctx.model_middlewares.clone(),
            ctx.http_transport.clone(),
        )
        .await
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.language_model_with_ctx(model_id, ctx).await
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("vertex")
    }
}
