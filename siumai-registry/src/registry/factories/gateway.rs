//! Vercel AI Gateway provider factory.

use super::*;
use siumai_provider_gateway::providers::gateway::{GatewayClient, GatewayConfig};

#[cfg(feature = "gateway")]
fn resolve_api_key(ctx: &BuildContext) -> Option<String> {
    if let Some(api_key) = &ctx.api_key {
        return Some(api_key.clone());
    }

    std::env::var("AI_GATEWAY_API_KEY").ok()
}

#[cfg(feature = "gateway")]
fn build_typed_client_with_ctx(
    model_id: &str,
    ctx: &BuildContext,
) -> Result<GatewayClient, LlmError> {
    let http_config = ctx.http_config.clone().unwrap_or_default();
    let http_client = if let Some(client) = &ctx.http_client {
        client.clone()
    } else {
        build_http_client_from_config(&http_config)?
    };

    let mut common_params = ctx.common_params.clone().unwrap_or_default();
    common_params.model = model_id.to_string();

    let mut cfg = GatewayConfig::new(resolve_api_key(ctx).unwrap_or_default())
        .with_http_config(http_config)
        .with_http_interceptors(ctx.http_interceptors.clone());
    cfg.common_params = common_params;

    if let Some(base_url) = ctx.base_url.clone() {
        cfg = cfg.with_base_url(base_url);
    }
    if let Some(http_transport) = ctx.http_transport.clone() {
        cfg = cfg.with_http_transport(http_transport);
    }

    let mut client = GatewayClient::with_http_client(cfg, http_client)?;

    if let Some(retry_options) = ctx.retry_options.clone() {
        client = client.with_retry_options(retry_options);
    }

    Ok(client)
}

#[cfg(feature = "gateway")]
fn build_typed_client_arc(
    model_id: &str,
    ctx: &BuildContext,
) -> Result<Arc<GatewayClient>, LlmError> {
    let client = build_typed_client_with_ctx(model_id, ctx)?;
    Ok(Arc::new(client))
}

/// Vercel AI Gateway provider factory.
#[cfg(feature = "gateway")]
pub struct GatewayProviderFactory;

#[cfg(feature = "gateway")]
#[async_trait::async_trait]
impl ProviderFactory for GatewayProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = crate::native_provider_metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == crate::provider::ids::GATEWAY)
            .map(|m| m.capabilities)
            .unwrap_or_else(|| {
                ProviderCapabilities::new()
                    .with_chat()
                    .with_embedding()
                    .with_streaming()
                    .with_tools()
            })
    }

    async fn compat_language_client(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let ctx = BuildContext::default();
        self.compat_language_client_with_ctx(model_id, &ctx).await
    }

    async fn compat_language_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_typed_client_arc(model_id, ctx)?;
        Ok(client)
    }

    async fn compat_embedding_client_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = build_typed_client_arc(model_id, ctx)?;
        Ok(client)
    }

    async fn language_model_text_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn crate::text::LanguageModel>, LlmError> {
        let client = build_typed_client_arc(model_id, ctx)?;
        Ok(client)
    }

    async fn embedding_model_family_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn crate::embedding::EmbeddingModel>, LlmError> {
        let client = build_typed_client_arc(model_id, ctx)?;
        Ok(client)
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(crate::provider::ids::GATEWAY)
    }
}
