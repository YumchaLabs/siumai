use super::OpenAiClient;
use crate::error::LlmError;
use crate::traits::{EmbeddingCapability, EmbeddingExtensions};
use crate::types::{EmbeddingRequest, EmbeddingResponse};
use async_trait::async_trait;

#[async_trait]
impl EmbeddingCapability for OpenAiClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        use crate::execution::executors::embedding::EmbeddingExecutor;

        let mut request = EmbeddingRequest::new(texts).with_model(self.common_params.model.clone());
        self.merge_default_provider_options_map(&mut request.provider_options_map);
        let exec = self.build_embedding_executor(&request);

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
        use crate::execution::executors::embedding::EmbeddingExecutor;

        let mut request = request;
        if request.model.as_deref().unwrap_or("").is_empty() {
            request.model = Some(self.common_params.model.clone());
        }
        self.merge_default_provider_options_map(&mut request.provider_options_map);

        let exec = self.build_embedding_executor(&request);
        EmbeddingExecutor::execute(&*exec, request).await
    }
}
