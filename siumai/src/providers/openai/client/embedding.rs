use super::OpenAiClient;
use crate::error::LlmError;
use crate::traits::{EmbeddingCapability, EmbeddingExtensions};
use crate::types::{EmbeddingRequest, EmbeddingResponse};
use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
impl EmbeddingCapability for OpenAiClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        use crate::execution::executors::embedding::EmbeddingExecutor;

        let request = EmbeddingRequest::new(texts).with_model(self.common_params.model.clone());
        let exec = self.build_embedding_executor();

        EmbeddingExecutor::execute(&*exec, request).await
    }

    fn embedding_dimension(&self) -> usize {
        let model = if !self.common_params.model.is_empty() {
            &self.common_params.model
        } else {
            "text-embedding-3-small"
        };

        match model {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536,
        }
    }

    fn max_tokens_per_embedding(&self) -> usize {
        8192
    }

    fn supported_embedding_models(&self) -> Vec<String> {
        vec![
            "text-embedding-3-small".to_string(),
            "text-embedding-3-large".to_string(),
            "text-embedding-ada-002".to_string(),
        ]
    }
}

// Extended embedding APIs that accept EmbeddingRequest directly
#[async_trait]
impl EmbeddingExtensions for OpenAiClient {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        use crate::execution::executors::embedding::{EmbeddingExecutor, HttpEmbeddingExecutor};

        if let Some(opts) = &self.retry_options {
            let http0 = self.http_client.clone();
            let base0 = self.base_url.clone();
            let api_key0 = self.api_key.clone();
            let org0 = self.organization.clone();
            let proj0 = self.project.clone();
            crate::retry_api::retry_with(
                || {
                    let rq = request.clone();
                    let http = http0.clone();
                    let base = base0.clone();
                    let api_key = api_key0.clone();
                    let org = org0.clone();
                    let proj = proj0.clone();
                    async move {
                        use crate::core::{ProviderContext, ProviderSpec};
                        use secrecy::ExposeSecret;

                        let spec = crate::providers::openai::spec::OpenAiSpec::new();
                        let ctx = ProviderContext::new(
                            "openai",
                            base,
                            Some(api_key.expose_secret().to_string()),
                            self.http_config.headers.clone(),
                        )
                        .with_org_project(org, proj);
                        let bundle = spec.choose_embedding_transformers(&rq, &ctx);
                        let spec_arc = Arc::new(spec);

                        let exec = HttpEmbeddingExecutor {
                            provider_id: "openai".to_string(),
                            http_client: http,
                            request_transformer: bundle.request,
                            response_transformer: bundle.response,
                            provider_spec: spec_arc,
                            provider_context: ctx,
                            policy: crate::execution::ExecutionPolicy::new(),
                        };
                        EmbeddingExecutor::execute(&exec, rq).await
                    }
                },
                opts.clone(),
            )
            .await
        } else {
            use crate::core::{ProviderContext, ProviderSpec};
            use secrecy::ExposeSecret;

            let spec = crate::providers::openai::spec::OpenAiSpec::new();
            let ctx = ProviderContext::new(
                "openai",
                self.base_url.clone(),
                Some(self.api_key.expose_secret().to_string()),
                self.http_config.headers.clone(),
            )
            .with_org_project(self.organization.clone(), self.project.clone());
            let bundle = spec.choose_embedding_transformers(&request, &ctx);
            let spec_arc = Arc::new(spec);

            let exec = HttpEmbeddingExecutor {
                provider_id: "openai".to_string(),
                http_client: self.http_client.clone(),
                request_transformer: bundle.request,
                response_transformer: bundle.response,
                provider_spec: spec_arc,
                provider_context: ctx,
                policy: crate::execution::ExecutionPolicy::new(),
            };
            exec.execute(request).await
        }
    }
}
