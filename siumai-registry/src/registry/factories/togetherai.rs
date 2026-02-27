//! TogetherAI provider factory (rerank-only).

use super::*;

#[cfg(feature = "togetherai")]
#[derive(Clone)]
struct TogetherAiRerankClient {
    provider_id: String,
    model_id: String,
    http_client: reqwest::Client,
    http_interceptors:
        Vec<std::sync::Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    retry_options: Option<crate::retry_api::RetryOptions>,
    http_transport: Option<std::sync::Arc<dyn crate::execution::http::transport::HttpTransport>>,
    provider_spec: std::sync::Arc<dyn crate::core::ProviderSpec>,
    provider_context: crate::core::ProviderContext,
}

#[cfg(feature = "togetherai")]
#[async_trait::async_trait]
impl crate::traits::ChatCapability for TogetherAiRerankClient {
    async fn chat_with_tools(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::types::ChatResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "TogetherAI client (built-in) currently supports rerank only".to_string(),
        ))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<crate::types::ChatMessage>,
        _tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<crate::streaming::ChatStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "TogetherAI client (built-in) does not support chat streaming".to_string(),
        ))
    }
}

#[cfg(feature = "togetherai")]
#[async_trait::async_trait]
impl crate::traits::RerankCapability for TogetherAiRerankClient {
    async fn rerank(
        &self,
        request: crate::types::RerankRequest,
    ) -> Result<crate::types::RerankResponse, LlmError> {
        use crate::execution::executors::rerank::{RerankExecutor, RerankExecutorBuilder};

        let mut builder =
            RerankExecutorBuilder::new(self.provider_id.clone(), self.http_client.clone())
                .with_spec(self.provider_spec.clone())
                .with_context(self.provider_context.clone())
                .with_interceptors(self.http_interceptors.clone());

        if let Some(opts) = self.retry_options.clone() {
            builder = builder.with_retry_options(opts);
        }
        if let Some(transport) = self.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        let exec = builder.build_for_request(&request);
        RerankExecutor::execute(&*exec, request).await
    }
}

#[cfg(feature = "togetherai")]
impl LlmClient for TogetherAiRerankClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Owned(self.provider_id.clone())
    }

    fn supported_models(&self) -> Vec<String> {
        vec![self.model_id.clone()]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_rerank()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_rerank_capability(&self) -> Option<&dyn crate::traits::RerankCapability> {
        Some(self)
    }
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
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.reranking_model_with_ctx(model_id, ctx).await
    }

    async fn reranking_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Resolve HTTP configuration and client (prefer provided client).
        let http_config = ctx.http_config.clone().unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        // Resolve API key: context override -> environment variable.
        let api_key = if let Some(key) = &ctx.api_key {
            key.clone()
        } else {
            std::env::var("TOGETHER_API_KEY").map_err(|_| {
                LlmError::ConfigurationError(
                    "Missing TOGETHER_API_KEY or explicit api_key in BuildContext".to_string(),
                )
            })?
        };

        // Resolve base URL (context override -> default).
        let base_url = crate::utils::builder_helpers::resolve_base_url(
            ctx.base_url.clone(),
            "https://api.together.xyz/v1",
        );

        let provider_id = crate::provider::ids::TOGETHERAI.to_string();
        let provider_context = crate::core::ProviderContext::new(
            provider_id.clone(),
            base_url,
            Some(api_key),
            http_config.headers.clone(),
        )
        .with_org_project(ctx.organization.clone(), ctx.project.clone());

        let standard = siumai_provider_togetherai::standards::togetherai::rerank::TogetherAiRerankStandard::new();
        let provider_spec: std::sync::Arc<dyn crate::core::ProviderSpec> =
            std::sync::Arc::new(standard.create_spec(crate::provider::ids::TOGETHERAI));

        Ok(Arc::new(TogetherAiRerankClient {
            provider_id,
            model_id: model_id.to_string(),
            http_client,
            http_interceptors: ctx.http_interceptors.clone(),
            retry_options: ctx.retry_options.clone(),
            http_transport: ctx.http_transport.clone(),
            provider_spec,
            provider_context,
        }))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(crate::provider::ids::TOGETHERAI)
    }
}
