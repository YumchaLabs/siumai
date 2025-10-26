use super::GeminiClient;
use crate::error::LlmError;
use crate::traits::{EmbeddingCapability, EmbeddingExtensions};
use crate::types::{EmbeddingRequest, EmbeddingResponse};
use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
impl EmbeddingCapability for GeminiClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        use crate::execution::executors::embedding::{EmbeddingExecutor, HttpEmbeddingExecutor};
        let req = EmbeddingRequest::new(texts).with_model(self.config.model.clone());
        use crate::core::ProviderContext;
        use secrecy::ExposeSecret;

        // Merge Authorization from token_provider if present
        let mut extra_headers = self.config.http_config.headers.clone();
        if let Some(ref tp) = self.config.token_provider {
            if let Ok(tok) = tp.token().await {
                extra_headers.insert("Authorization".to_string(), format!("Bearer {tok}"));
            }
        }

        let spec = crate::providers::gemini::spec::create_embedding_wrapper(
            self.config.base_url.clone(),
            self.config.model.clone(),
        );

        let ctx = ProviderContext::new(
            "gemini",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            extra_headers.clone(),
        );

        let req_tx = super::super::transformers::GeminiRequestTransformer {
            config: self.config.clone(),
        };
        let resp_tx = super::super::transformers::GeminiResponseTransformer {
            config: self.config.clone(),
        };

        if let Some(opts) = &self.retry_options {
            let http = self.http_client.clone();
            let spec_clone = spec.clone();
            let ctx_clone = ctx.clone();
            let config = self.config.clone();
            crate::retry_api::retry_with(
                || {
                    let rq = req.clone();
                    let http = http.clone();
                    let spec = spec_clone.clone();
                    let ctx = ctx_clone.clone();
                    let req_tx = super::super::transformers::GeminiRequestTransformer {
                        config: config.clone(),
                    };
                    let resp_tx = super::super::transformers::GeminiResponseTransformer {
                        config: config.clone(),
                    };
                    async move {
                        let exec = HttpEmbeddingExecutor {
                            provider_id: "gemini".to_string(),
                            http_client: http,
                            request_transformer: Arc::new(req_tx),
                            response_transformer: Arc::new(resp_tx),
                            provider_spec: spec,
                            provider_context: ctx,
                            policy: crate::execution::ExecutionPolicy::new()
                                .with_interceptors(self.http_interceptors.clone()),
                        };
                        EmbeddingExecutor::execute(&exec, rq).await
                    }
                },
                opts.clone(),
            )
            .await
        } else {
            let exec = HttpEmbeddingExecutor {
                provider_id: "gemini".to_string(),
                http_client: self.http_client.clone(),
                request_transformer: Arc::new(req_tx),
                response_transformer: Arc::new(resp_tx),
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
        use crate::execution::executors::embedding::{EmbeddingExecutor, HttpEmbeddingExecutor};
        if request.input.len() > 2048 {
            return Err(LlmError::InvalidParameter(format!(
                "Too many values for a single embedding call. The Gemini model \"{}\" can only embed up to 2048 values per call, but {} values were provided.",
                self.config.model,
                request.input.len()
            )));
        }
        use crate::core::ProviderContext;
        use secrecy::ExposeSecret;

        // Merge Authorization from token_provider if present
        let mut extra_headers = self.config.http_config.headers.clone();
        if let Some(ref tp) = self.config.token_provider {
            if let Ok(tok) = tp.token().await {
                extra_headers.insert("Authorization".to_string(), format!("Bearer {tok}"));
            }
        }

        let spec = crate::providers::gemini::spec::create_embedding_wrapper(
            self.config.base_url.clone(),
            self.config.model.clone(),
        );
        let ctx = ProviderContext::new(
            "gemini",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            extra_headers.clone(),
        );
        let req_tx = super::super::transformers::GeminiRequestTransformer {
            config: self.config.clone(),
        };
        let resp_tx = super::super::transformers::GeminiResponseTransformer {
            config: self.config.clone(),
        };

        if let Some(opts) = &self.retry_options {
            let http = self.http_client.clone();
            let spec_clone = spec.clone();
            let ctx_clone = ctx.clone();
            let config = self.config.clone();
            crate::retry_api::retry_with(
                || {
                    let rq = request.clone();
                    let http = http.clone();
                    let spec = spec_clone.clone();
                    let ctx = ctx_clone.clone();
                    let req_tx = super::super::transformers::GeminiRequestTransformer {
                        config: config.clone(),
                    };
                    let resp_tx = super::super::transformers::GeminiResponseTransformer {
                        config: config.clone(),
                    };
                    async move {
                        let exec = HttpEmbeddingExecutor {
                            provider_id: "gemini".to_string(),
                            http_client: http,
                            request_transformer: Arc::new(req_tx),
                            response_transformer: Arc::new(resp_tx),
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
            let exec = HttpEmbeddingExecutor {
                provider_id: "gemini".to_string(),
                http_client: self.http_client.clone(),
                request_transformer: Arc::new(req_tx),
                response_transformer: Arc::new(resp_tx),
                provider_spec: spec,
                provider_context: ctx,
                policy: crate::execution::ExecutionPolicy::new(),
            };
            EmbeddingExecutor::execute(&exec, request).await
        }
    }
}
