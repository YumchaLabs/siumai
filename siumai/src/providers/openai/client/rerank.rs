use super::OpenAiClient;
use crate::error::LlmError;
use crate::traits::RerankCapability;
use crate::types::{RerankRequest, RerankResponse};
use async_trait::async_trait;
use secrecy::ExposeSecret;

#[async_trait]
impl RerankCapability for OpenAiClient {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        use crate::execution::executors::rerank::{HttpRerankExecutor, RerankExecutor};
        use crate::execution::http::headers::ProviderHeaders;
        use crate::standards::openai::rerank::OpenAiRerankStandard;

        let standard = OpenAiRerankStandard::new();
        let transformers = standard.create_transformers("openai");

        let url = format!("{}/rerank", self.base_url.trim_end_matches('/'));

        let headers = ProviderHeaders::openai(
            self.api_key.expose_secret(),
            self.organization.as_deref(),
            self.project.as_deref(),
            &self.http_config.headers,
        )?;

        let exec = HttpRerankExecutor {
            provider_id: "openai".to_string(),
            http_client: self.http_client.clone(),
            request_transformer: transformers.request,
            response_transformer: transformers.response,
            policy: crate::execution::ExecutionPolicy::new()
                .with_interceptors(self.http_interceptors.clone())
                .with_retry_options(self.retry_options.clone()),
            url,
            headers,
            before_send: None,
        };
        RerankExecutor::execute(&exec, request).await
    }

    fn max_documents(&self) -> Option<u32> {
        self.rerank_capability.max_documents()
    }

    fn supported_models(&self) -> Vec<String> {
        self.rerank_capability.supported_models()
    }
}
