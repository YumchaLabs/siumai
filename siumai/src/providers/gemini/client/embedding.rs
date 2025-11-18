use super::GeminiClient;
use crate::error::LlmError;
use crate::traits::{EmbeddingCapability, EmbeddingExtensions};
use crate::types::{EmbeddingRequest, EmbeddingResponse};
use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
impl EmbeddingCapability for GeminiClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        use crate::core::{ProviderContext, ProviderSpec};
        use crate::execution::executors::embedding::{EmbeddingExecutor, HttpEmbeddingExecutor};
        use secrecy::ExposeSecret;

        let req = EmbeddingRequest::new(texts).with_model(self.config.model.clone());

        // Merge Authorization from token_provider if present
        let mut extra_headers = self.config.http_config.headers.clone();
        if let Some(ref tp) = self.config.token_provider
            && let Ok(tok) = tp.token().await
        {
            extra_headers.insert("Authorization".to_string(), format!("Bearer {tok}"));
        }

        let ctx = ProviderContext::new(
            "gemini",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            extra_headers,
        );

        // Use unified GeminiSpec + std-gemini embedding transformers.
        let spec: Arc<dyn ProviderSpec> = Arc::new(crate::providers::gemini::spec::GeminiSpec);

        if let Some(opts) = &self.retry_options {
            let http = self.http_client.clone();
            let spec_clone = spec.clone();
            let ctx_clone = ctx.clone();
            let interceptors = self.http_interceptors.clone();
            crate::retry_api::retry_with(
                || {
                    let rq = req.clone();
                    let http = http.clone();
                    let spec = spec_clone.clone();
                    let ctx = ctx_clone.clone();
                    let interceptors = interceptors.clone();
                    async move {
                        let bundle = spec.choose_embedding_transformers(&rq, &ctx);
                        let exec = HttpEmbeddingExecutor {
                            provider_id: "gemini".to_string(),
                            http_client: http,
                            request_transformer: bundle.request,
                            response_transformer: bundle.response,
                            provider_spec: spec,
                            provider_context: ctx,
                            policy: crate::execution::ExecutionPolicy::new()
                                .with_interceptors(interceptors),
                        };
                        EmbeddingExecutor::execute(&exec, rq).await
                    }
                },
                opts.clone(),
            )
            .await
        } else {
            let bundle = spec.choose_embedding_transformers(&req, &ctx);
            let exec = HttpEmbeddingExecutor {
                provider_id: "gemini".to_string(),
                http_client: self.http_client.clone(),
                request_transformer: bundle.request,
                response_transformer: bundle.response,
                provider_spec: spec,
                provider_context: ctx,
                policy: crate::execution::ExecutionPolicy::new()
                    .with_interceptors(self.http_interceptors.clone())
                    .with_retry_options(self.retry_options.clone()),
            };
            EmbeddingExecutor::execute(&exec, req).await
        }
    }

    fn embedding_dimension(&self) -> usize {
        3072
    }
    fn max_tokens_per_embedding(&self) -> usize {
        2048
    }
    fn supported_embedding_models(&self) -> Vec<String> {
        vec!["gemini-embedding-001".to_string()]
    }
}

#[async_trait]
impl EmbeddingExtensions for GeminiClient {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        use crate::core::{ProviderContext, ProviderSpec};
        use crate::execution::executors::embedding::{EmbeddingExecutor, HttpEmbeddingExecutor};
        if request.input.len() > 2048 {
            return Err(LlmError::InvalidParameter(format!(
                "Too many values for a single embedding call. The Gemini model \"{}\" can only embed up to 2048 values per call, but {} values were provided.",
                self.config.model,
                request.input.len()
            )));
        }
        use secrecy::ExposeSecret;

        // Merge Authorization from token_provider if present
        let mut extra_headers = self.config.http_config.headers.clone();
        if let Some(ref tp) = self.config.token_provider
            && let Ok(tok) = tp.token().await
        {
            extra_headers.insert("Authorization".to_string(), format!("Bearer {tok}"));
        }

        let ctx = ProviderContext::new(
            "gemini",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            extra_headers,
        );

        // Use unified GeminiSpec + std-gemini embedding transformers.
        let spec: Arc<dyn ProviderSpec> = Arc::new(crate::providers::gemini::spec::GeminiSpec);

        if let Some(opts) = &self.retry_options {
            let http = self.http_client.clone();
            let spec_clone = spec.clone();
            let ctx_clone = ctx.clone();
            crate::retry_api::retry_with(
                || {
                    let rq = request.clone();
                    let http = http.clone();
                    let spec = spec_clone.clone();
                    let ctx = ctx_clone.clone();
                    async move {
                        let bundle = spec.choose_embedding_transformers(&rq, &ctx);
                        let exec = HttpEmbeddingExecutor {
                            provider_id: "gemini".to_string(),
                            http_client: http,
                            request_transformer: bundle.request,
                            response_transformer: bundle.response,
                            provider_spec: spec,
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
            let bundle = spec.choose_embedding_transformers(&request, &ctx);
            let exec = HttpEmbeddingExecutor {
                provider_id: "gemini".to_string(),
                http_client: self.http_client.clone(),
                request_transformer: bundle.request,
                response_transformer: bundle.response,
                provider_spec: spec,
                provider_context: ctx,
                policy: crate::execution::ExecutionPolicy::new(),
            };
            EmbeddingExecutor::execute(&exec, request).await
        }
    }
}
