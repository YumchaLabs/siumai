//! Amazon Bedrock provider factory.

use super::*;

#[cfg(feature = "bedrock")]
fn resolve_api_key(ctx: &BuildContext) -> Option<String> {
    if let Some(api_key) = &ctx.api_key {
        return Some(api_key.clone());
    }

    std::env::var("BEDROCK_API_KEY").ok()
}

#[cfg(feature = "bedrock")]
fn build_typed_client_with_ctx(
    model_id: &str,
    ctx: &BuildContext,
) -> Result<siumai_provider_amazon_bedrock::providers::bedrock::BedrockClient, LlmError> {
    let http_config = ctx.http_config.clone().unwrap_or_default();
    let http_client = if let Some(client) = &ctx.http_client {
        client.clone()
    } else {
        build_http_client_from_config(&http_config)?
    };

    let mut cfg = siumai_provider_amazon_bedrock::providers::bedrock::BedrockConfig::from_env()
        .with_model(model_id)
        .with_http_config(http_config)
        .with_http_interceptors(ctx.http_interceptors.clone());

    if let Some(api_key) = resolve_api_key(ctx) {
        cfg = cfg.with_api_key(api_key);
    }
    if let Some(base_url) = ctx.base_url.clone() {
        cfg = cfg.with_base_url(base_url);
    }
    if let Some(http_transport) = ctx.http_transport.clone() {
        cfg = cfg.with_http_transport(http_transport);
    }

    let mut client =
        siumai_provider_amazon_bedrock::providers::bedrock::BedrockClient::with_http_client(
            cfg,
            http_client,
        )?;

    if let Some(retry_options) = ctx.retry_options.clone() {
        client = client.with_retry_options(retry_options);
    }

    Ok(client)
}

/// Amazon Bedrock provider factory.
#[cfg(feature = "bedrock")]
pub struct BedrockProviderFactory;

#[cfg(feature = "bedrock")]
#[async_trait::async_trait]
impl ProviderFactory for BedrockProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = crate::native_provider_metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == crate::provider::ids::BEDROCK)
            .map(|m| m.capabilities)
            .unwrap_or_else(|| {
                ProviderCapabilities::new()
                    .with_chat()
                    .with_streaming()
                    .with_tools()
                    .with_rerank()
            })
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
        Ok(Arc::new(build_typed_client_with_ctx(model_id, ctx)?))
    }

    async fn reranking_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(build_typed_client_with_ctx(model_id, ctx)?))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(crate::provider::ids::BEDROCK)
    }
}
