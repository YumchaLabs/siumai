use super::OpenAiClient;
use crate::error::LlmError;
use crate::traits::RerankCapability;
use crate::types::{RerankRequest, RerankResponse};
use async_trait::async_trait;

#[async_trait]
impl RerankCapability for OpenAiClient {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        use crate::execution::executors::rerank::{RerankExecutor, RerankExecutorBuilder};

        let mut request = request;
        self.merge_default_provider_options_map(&mut request.provider_options_map);

        // OpenAI's public API does not expose a rerank endpoint. This capability is intended
        // for OpenAI-compatible providers that implement `/rerank` behind an OpenAI-like surface.
        //
        // Prefer `providers::openai_compatible::siliconflow` for rerank usage.
        if self.base_url.to_lowercase().contains("api.openai.com") {
            return Err(LlmError::UnsupportedOperation(
                "Rerank is not supported by OpenAI. Use OpenAI-compatible providers (e.g., siliconflow) instead."
                    .to_string(),
            ));
        }

        let ctx = self.build_context();
        let spec = std::sync::Arc::new(crate::providers::openai::spec::OpenAiSpecWithRerank::new());

        let mut builder = RerankExecutorBuilder::new("openai", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        let exec = builder.build_for_request(&request);
        RerankExecutor::execute(&*exec, request).await
    }

    fn max_documents(&self) -> Option<u32> {
        self.rerank_capability.max_documents()
    }

    fn supported_models(&self) -> Vec<String> {
        self.rerank_capability.supported_models()
    }
}
