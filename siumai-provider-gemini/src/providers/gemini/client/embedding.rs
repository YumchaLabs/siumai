use super::GeminiClient;
use crate::error::LlmError;
use crate::traits::{EmbeddingCapability, EmbeddingExtensions};
use crate::types::{EmbeddingRequest, EmbeddingResponse};
use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
impl EmbeddingCapability for GeminiClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        use crate::execution::executors::embedding::{EmbeddingExecutor, EmbeddingExecutorBuilder};
        let req = EmbeddingRequest::new(texts).with_model(self.config.model.clone());
        let ctx = super::super::context::build_context(&self.config).await;
        let spec = Arc::new(crate::providers::gemini::spec::GeminiSpecWithConfig::new(
            self.config.clone(),
        ));

        let exec = EmbeddingExecutorBuilder::new("gemini", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());
        let exec = if let Some(transport) = self.config.http_transport.clone() {
            exec.with_transport(transport)
        } else {
            exec
        };

        let exec = if let Some(retry) = self.retry_options.clone() {
            exec.with_retry_options(retry).build_for_request(&req)
        } else {
            exec.build_for_request(&req)
        };

        EmbeddingExecutor::execute(&*exec, req).await
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
        use crate::execution::executors::embedding::{EmbeddingExecutor, EmbeddingExecutorBuilder};
        let mut request = request;
        if request.input.len() > 2048 {
            return Err(LlmError::InvalidParameter(format!(
                "Too many values for a single embedding call. The Gemini model \"{}\" can only embed up to 2048 values per call, but {} values were provided.",
                self.config.model,
                request.input.len()
            )));
        }
        if request.model.is_none() {
            request.model = Some(self.config.model.clone());
        }
        let ctx = super::super::context::build_context(&self.config).await;
        let spec = Arc::new(crate::providers::gemini::spec::GeminiSpecWithConfig::new(
            self.config.clone(),
        ));

        let exec = EmbeddingExecutorBuilder::new("gemini", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors.clone());
        let exec = if let Some(transport) = self.config.http_transport.clone() {
            exec.with_transport(transport)
        } else {
            exec
        };

        let exec = if let Some(retry) = self.retry_options.clone() {
            exec.with_retry_options(retry).build_for_request(&request)
        } else {
            exec.build_for_request(&request)
        };

        EmbeddingExecutor::execute(&*exec, request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ProviderSpec;

    #[test]
    fn spec_with_config_uses_request_model_for_embedding_body() {
        let base = crate::providers::gemini::GeminiConfig::default()
            .with_model("chat-model".to_string())
            .with_common_params(crate::types::CommonParams {
                model: "chat-model".to_string(),
                ..Default::default()
            });
        let spec = crate::providers::gemini::spec::GeminiSpecWithConfig::new(base);
        let ctx = crate::core::ProviderContext::new(
            "gemini",
            "https://example".to_string(),
            Some("KEY".to_string()),
            std::collections::HashMap::new(),
        );

        let req =
            EmbeddingRequest::new(vec!["hi".to_string()]).with_model("embed-model".to_string());
        let bundle = spec.choose_embedding_transformers(&req, &ctx);
        let body = bundle.request.transform_embedding(&req).unwrap();
        assert_eq!(body["model"], serde_json::json!("models/embed-model"));
    }
}
