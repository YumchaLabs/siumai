//! Embedding capability traits and extensions

use crate::error::LlmError;
use crate::types::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingModelInfo, EmbeddingRequest,
    EmbeddingResponse,
};
use async_trait::async_trait;
use std::collections::HashMap;

#[async_trait]
pub trait EmbeddingCapability: Send + Sync {
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError>;

    fn embedding_dimension(&self) -> usize;

    fn max_tokens_per_embedding(&self) -> usize {
        8192
    }

    fn supported_embedding_models(&self) -> Vec<String> {
        vec!["default".to_string()]
    }
}

#[async_trait]
pub trait EmbeddingExtensions: EmbeddingCapability {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        self.embed(request.input).await
    }

    async fn embed_batch(
        &self,
        requests: BatchEmbeddingRequest,
    ) -> Result<BatchEmbeddingResponse, LlmError> {
        let mut responses = Vec::new();
        for request in requests.requests {
            let result = self
                .embed_with_config(request)
                .await
                .map_err(|e| e.to_string());
            responses.push(result);
            if requests.batch_options.fail_fast && responses.last().unwrap().is_err() {
                break;
            }
        }
        Ok(BatchEmbeddingResponse {
            responses,
            metadata: HashMap::new(),
        })
    }

    async fn list_embedding_models(&self) -> Result<Vec<EmbeddingModelInfo>, LlmError> {
        let models = self.supported_embedding_models();
        let model_infos = models
            .into_iter()
            .map(|id| {
                EmbeddingModelInfo::new(
                    id.clone(),
                    id,
                    self.embedding_dimension(),
                    self.max_tokens_per_embedding(),
                )
            })
            .collect();
        Ok(model_infos)
    }

    fn calculate_similarity(
        &self,
        embedding1: &[f32],
        embedding2: &[f32],
    ) -> Result<f32, LlmError> {
        if embedding1.len() != embedding2.len() {
            return Err(LlmError::InvalidInput(
                "Embedding vectors must have the same dimension".to_string(),
            ));
        }
        let dot_product: f32 = embedding1
            .iter()
            .zip(embedding2.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm1 == 0.0 || norm2 == 0.0 {
            return Err(LlmError::InvalidInput(
                "Cannot calculate similarity for zero vectors".to_string(),
            ));
        }
        Ok(dot_product / (norm1 * norm2))
    }
}
