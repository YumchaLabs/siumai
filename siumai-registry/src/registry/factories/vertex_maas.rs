//! Google Vertex MaaS provider factory.
//!
//! AI SDK exposes Vertex MaaS as a first-class provider surface backed by
//! Vertex's OpenAI-compatible `/endpoints/openapi` routes.

use super::*;
use crate::embedding::EmbeddingModel as FamilyEmbeddingModel;
use crate::provider::ids;
use crate::text::LanguageModel as FamilyLanguageModel;
use siumai_core::completion::CompletionModel as FamilyCompletionModel;
use std::borrow::Cow;

const DEFAULT_LOCATION: &str = "global";

#[cfg(feature = "google-vertex")]
fn vertex_maas_capabilities() -> ProviderCapabilities {
    crate::native_provider_metadata::native_providers_metadata()
        .into_iter()
        .find(|meta| meta.id == ids::VERTEX_MAAS)
        .map(|meta| meta.capabilities)
        .unwrap_or_else(|| {
            ProviderCapabilities::new()
                .with_chat()
                .with_completion()
                .with_streaming()
                .with_tools()
                .with_vision()
                .with_embedding()
        })
}

#[cfg(feature = "google-vertex")]
fn non_empty(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}

#[cfg(feature = "google-vertex")]
fn env_non_empty(name: &str) -> Option<String> {
    non_empty(std::env::var(name).ok())
}

#[cfg(feature = "google-vertex")]
fn http_config_has_authorization(http_config: &crate::types::HttpConfig) -> bool {
    http_config
        .headers
        .keys()
        .any(|key| key.eq_ignore_ascii_case("authorization"))
}

#[cfg(feature = "google-vertex")]
fn resolve_base_url(ctx: &BuildContext) -> Result<String, LlmError> {
    if let Some(base_url) = non_empty(ctx.base_url.clone()) {
        return Ok(base_url);
    }

    let project = non_empty(ctx.project.clone())
        .or_else(|| env_non_empty("GOOGLE_VERTEX_PROJECT"))
        .ok_or_else(|| {
            LlmError::ConfigurationError(
                "Google Vertex MaaS requires `project`, `base_url`, or GOOGLE_VERTEX_PROJECT"
                    .to_string(),
            )
        })?;
    let location = non_empty(ctx.location.clone())
        .or_else(|| env_non_empty("GOOGLE_VERTEX_LOCATION"))
        .unwrap_or_else(|| DEFAULT_LOCATION.to_string());

    Ok(crate::utils::vertex::google_vertex_maas_base_url(
        &project, &location,
    ))
}

#[cfg(feature = "google-vertex")]
fn resolve_auth(
    ctx: &BuildContext,
    http_config: &crate::types::HttpConfig,
) -> Result<
    (
        String,
        Option<std::sync::Arc<dyn crate::auth::TokenProvider>>,
    ),
    LlmError,
> {
    if let Some(token_provider) = ctx.resolved_google_token_provider() {
        return Ok((String::new(), Some(token_provider)));
    }

    if http_config_has_authorization(http_config) {
        return Ok((String::new(), None));
    }

    if let Some(api_key) = non_empty(ctx.api_key.clone()) {
        return Ok((api_key, None));
    }

    #[cfg(feature = "gcp")]
    {
        return Ok((
            String::new(),
            Some(std::sync::Arc::new(
                crate::auth::adc::AdcTokenProvider::default_client(),
            )),
        ));
    }

    #[cfg(not(feature = "gcp"))]
    {
        Err(LlmError::ConfigurationError(
            "Google Vertex MaaS requires an Authorization header, explicit api_key bearer token, or a Google token provider".to_string(),
        ))
    }
}

#[cfg(feature = "google-vertex")]
async fn build_text_client_with_ctx(
    model_id: &str,
    ctx: &BuildContext,
) -> Result<
    siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient,
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
    let (api_key, token_provider) = resolve_auth(ctx, &http_config)?;

    crate::registry::factory::build_openai_compatible_typed_client(
        ids::VERTEX_MAAS.to_string(),
        api_key,
        Some(resolve_base_url(ctx)?),
        http_client,
        common_params,
        ctx.reasoning_enabled,
        ctx.reasoning_budget,
        http_config,
        token_provider,
        None,
        ctx.tracing_config.clone(),
        ctx.retry_options.clone(),
        ctx.http_interceptors.clone(),
        ctx.model_middlewares.clone(),
        ctx.http_transport.clone(),
    )
    .await
}

/// Google Vertex MaaS provider factory.
#[cfg(feature = "google-vertex")]
pub struct VertexMaasProviderFactory;

#[cfg(feature = "google-vertex")]
#[async_trait::async_trait]
impl ProviderFactory for VertexMaasProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        vertex_maas_capabilities()
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
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn language_model_text_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyLanguageModel>, LlmError> {
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn completion_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn completion_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyCompletionModel>, LlmError> {
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    async fn embedding_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn FamilyEmbeddingModel>, LlmError> {
        let client = build_text_client_with_ctx(model_id, ctx).await?;
        Ok(Arc::new(client))
    }

    fn provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed(ids::VERTEX_MAAS)
    }
}
