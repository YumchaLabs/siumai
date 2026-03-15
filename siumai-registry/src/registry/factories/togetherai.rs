//! TogetherAI provider factory (rerank-only).

use super::*;

#[cfg(feature = "togetherai")]
fn resolve_api_key(ctx: &BuildContext) -> Result<String, LlmError> {
    if let Some(api_key) = &ctx.api_key {
        return Ok(api_key.clone());
    }

    std::env::var("TOGETHER_API_KEY").map_err(|_| {
        LlmError::ConfigurationError(
            "Missing TOGETHER_API_KEY or explicit api_key in BuildContext".to_string(),
        )
    })
}

#[cfg(feature = "togetherai")]
fn build_typed_client_with_ctx(
    model_id: &str,
    ctx: &BuildContext,
) -> Result<siumai_provider_togetherai::providers::togetherai::TogetherAiClient, LlmError> {
    let http_config = ctx.http_config.clone().unwrap_or_default();
    let http_client = if let Some(client) = &ctx.http_client {
        client.clone()
    } else {
        build_http_client_from_config(&http_config)?
    };

    let mut cfg = siumai_provider_togetherai::providers::togetherai::TogetherAiConfig::new(
        resolve_api_key(ctx)?,
    )
    .with_base_url(crate::utils::builder_helpers::resolve_base_url(
        ctx.base_url.clone(),
        siumai_provider_togetherai::providers::togetherai::TogetherAiConfig::DEFAULT_BASE_URL,
    ))
    .with_model(model_id)
    .with_http_config(http_config)
    .with_http_interceptors(ctx.http_interceptors.clone());

    if let Some(http_transport) = ctx.http_transport.clone() {
        cfg = cfg.with_http_transport(http_transport);
    }

    let mut client =
        siumai_provider_togetherai::providers::togetherai::TogetherAiClient::with_http_client(
            cfg,
            http_client,
        )?;

    if let Some(retry_options) = ctx.retry_options.clone() {
        client = client.with_retry_options(retry_options);
    }

    Ok(client)
}

/// TogetherAI provider factory (rerank-only).
#[cfg(feature = "togetherai")]
pub struct TogetherAiProviderFactory;

#[cfg(feature = "togetherai")]
#[async_trait::async_trait]
impl ProviderFactory for TogetherAiProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = crate::native_provider_metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == crate::provider::ids::TOGETHERAI)
            .map(|m| m.capabilities)
            .unwrap_or_else(|| ProviderCapabilities::new().with_rerank())
    }

    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        _model_id: &str,
        _ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "TogetherAI does not expose the language_model/chat family path; use reranking_model instead"
                .to_string(),
        ))
    }

    async fn reranking_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(build_typed_client_with_ctx(model_id, ctx)?))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(crate::provider::ids::TOGETHERAI)
    }
}
