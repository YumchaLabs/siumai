//! Provider factory implementations.

use super::*;
use crate::provider::ids;

fn normalize_non_empty(value: impl Into<String>) -> Option<String> {
    let value = value.into();
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn load_optional_env_var(name: &str) -> Option<String> {
    std::env::var(name).ok().and_then(normalize_non_empty)
}

#[cfg(feature = "google-vertex")]
async fn build_typed_client_with_ctx(
    model_id: &str,
    ctx: &BuildContext,
) -> Result<
    siumai_provider_google_vertex::providers::anthropic_vertex::client::VertexAnthropicClient,
    LlmError,
> {
    let http_config = ctx.http_config.clone().unwrap_or_default();
    let http_client = if let Some(client) = &ctx.http_client {
        client.clone()
    } else {
        build_http_client_from_config(&http_config)?
    };

    let common_params =
        crate::utils::builder_helpers::resolve_common_params(ctx.common_params.clone(), model_id);

    let base_url = if let Some(base_url) = ctx.base_url.clone().and_then(normalize_non_empty) {
        base_url.trim_end_matches('/').to_string()
    } else {
        let project = ctx
            .project
            .clone()
            .and_then(normalize_non_empty)
            .or_else(|| load_optional_env_var("GOOGLE_VERTEX_PROJECT"));
        let location = ctx
            .location
            .clone()
            .and_then(normalize_non_empty)
            .or_else(|| load_optional_env_var("GOOGLE_VERTEX_LOCATION"));

        let (project, location) = match (project, location) {
            (Some(project), Some(location)) => (project, location),
            _ => {
                return Err(LlmError::ConfigurationError(
                    "Anthropic on Vertex requires `base_url`, explicit project+location, or env vars GOOGLE_VERTEX_PROJECT + GOOGLE_VERTEX_LOCATION".to_string(),
                ))
            }
        };

        siumai_provider_google_vertex::auth::vertex::google_vertex_anthropic_base_url(
            &project, &location,
        )
    };

    crate::registry::factory::build_anthropic_vertex_typed_client(
        base_url,
        http_client,
        common_params,
        http_config,
        ctx.resolved_google_token_provider(),
        ctx.tracing_config.clone(),
        ctx.retry_options.clone(),
        ctx.http_interceptors.clone(),
        ctx.model_middlewares.clone(),
        ctx.http_transport.clone(),
    )
    .await
}

/// Anthropic on Vertex AI provider factory
///
/// This factory builds `anthropic-vertex` clients that communicate with
/// Anthropic models hosted on Vertex AI. Authentication is handled via
/// `Authorization: Bearer` headers configured on the HTTP client.
#[cfg(feature = "google-vertex")]
pub struct AnthropicVertexProviderFactory;

#[cfg(feature = "google-vertex")]
#[async_trait::async_trait]
impl ProviderFactory for AnthropicVertexProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = crate::native_provider_metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == ids::ANTHROPIC_VERTEX)
            .map(|m| m.capabilities)
            .unwrap_or_else(ProviderCapabilities::new)
    }

    async fn compat_language_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Delegate to the context-aware implementation with default context.
        let ctx = BuildContext::default();
        self.compat_language_client_with_ctx(model_id, &ctx).await
    }

    async fn compat_language_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(build_typed_client_with_ctx(model_id, ctx).await?))
    }

    async fn language_model_text_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn crate::text::LanguageModel>, LlmError> {
        Ok(Arc::new(build_typed_client_with_ctx(model_id, ctx).await?))
    }

    async fn compat_embedding_client_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Anthropic Vertex does not currently expose a provider-owned embedding family path"
                .to_string(),
        ))
    }

    async fn compat_image_client_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Anthropic Vertex does not currently expose a provider-owned image family path"
                .to_string(),
        ))
    }

    async fn compat_speech_client_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Anthropic Vertex does not currently expose a provider-owned speech family path"
                .to_string(),
        ))
    }

    async fn compat_transcription_client_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Anthropic Vertex does not currently expose a provider-owned transcription family path"
                .to_string(),
        ))
    }

    async fn compat_reranking_client_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Anthropic Vertex does not currently expose a provider-owned reranking family path"
                .to_string(),
        ))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(ids::ANTHROPIC_VERTEX)
    }
}
