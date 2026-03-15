//! Embedding model family (V3).
//!
//! This module provides a Rust-first, family-oriented abstraction for embeddings.
//! In V3-M2 it is intentionally implemented as an adapter over the existing
//! `EmbeddingCapability` so we can ship the new surface quickly, then iterate
//! towards a fully decoupled family-first foundation.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::error::LlmError;
use crate::traits::{EmbeddingCapability, ModelMetadata};
use crate::types::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingRequest, EmbeddingResponse,
};

/// V3 interface for embedding models.
#[async_trait]
pub trait EmbeddingModelV3: Send + Sync {
    /// Generate embeddings for a single request.
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError>;

    /// Generate embeddings for a batch of requests.
    async fn embed_many(
        &self,
        requests: BatchEmbeddingRequest,
    ) -> Result<BatchEmbeddingResponse, LlmError>;
}

/// Stable embedding-model contract for the V4 refactor spike.
pub trait EmbeddingModel: EmbeddingModelV3 + ModelMetadata + Send + Sync {}

impl<T> EmbeddingModel for T where T: EmbeddingModelV3 + ModelMetadata + Send + Sync + ?Sized {}

/// Adapter: any `EmbeddingCapability` can be used as an `EmbeddingModelV3`.
#[async_trait]
impl<T> EmbeddingModelV3 for T
where
    T: EmbeddingCapability + Send + Sync + ?Sized,
{
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        EmbeddingCapability::embed(self, request.input).await
    }

    async fn embed_many(
        &self,
        requests: BatchEmbeddingRequest,
    ) -> Result<BatchEmbeddingResponse, LlmError> {
        let mut responses = Vec::new();
        for request in requests.requests {
            let result = EmbeddingCapability::embed(self, request.input)
                .await
                .map_err(|e| e.to_string());
            responses.push(result);
            if requests.batch_options.fail_fast && responses.last().is_some_and(|r| r.is_err()) {
                break;
            }
        }
        Ok(BatchEmbeddingResponse {
            responses,
            metadata: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ModelSpecVersion;

    struct FakeEmbedding {
        dim: usize,
    }

    impl crate::traits::ModelMetadata for FakeEmbedding {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "fake-embedding"
        }
    }

    #[async_trait]
    impl EmbeddingCapability for FakeEmbedding {
        async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
            Ok(EmbeddingResponse::new(
                input
                    .into_iter()
                    .map(|s| vec![s.len() as f32; self.dim])
                    .collect(),
                "fake".to_string(),
            ))
        }

        fn embedding_dimension(&self) -> usize {
            self.dim
        }
    }

    #[tokio::test]
    async fn adapter_embed_uses_capability() {
        let model = FakeEmbedding { dim: 3 };
        let resp = EmbeddingModelV3::embed(
            &model,
            EmbeddingRequest {
                input: vec!["a".to_string(), "abcd".to_string()],
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(resp.embeddings.len(), 2);
        assert_eq!(resp.embeddings[0], vec![1.0, 1.0, 1.0]);
        assert_eq!(resp.embeddings[1], vec![4.0, 4.0, 4.0]);
    }

    #[tokio::test]
    async fn adapter_embed_many_respects_fail_fast() {
        struct FailOnSecond;

        impl crate::traits::ModelMetadata for FailOnSecond {
            fn provider_id(&self) -> &str {
                "fake"
            }

            fn model_id(&self) -> &str {
                "fail-on-second"
            }
        }

        #[async_trait]
        impl EmbeddingCapability for FailOnSecond {
            async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
                let first = input.first().cloned().unwrap_or_default();
                if first == "boom" {
                    return Err(LlmError::InternalError("boom".to_string()));
                }
                Ok(EmbeddingResponse::new(vec![vec![1.0]], "fake".to_string()))
            }

            fn embedding_dimension(&self) -> usize {
                1
            }
        }

        let model = FailOnSecond;
        let resp = EmbeddingModelV3::embed_many(
            &model,
            BatchEmbeddingRequest {
                requests: vec![
                    EmbeddingRequest {
                        input: vec!["ok".to_string()],
                        ..Default::default()
                    },
                    EmbeddingRequest {
                        input: vec!["boom".to_string()],
                        ..Default::default()
                    },
                    EmbeddingRequest {
                        input: vec!["late".to_string()],
                        ..Default::default()
                    },
                ],
                batch_options: crate::types::BatchOptions {
                    fail_fast: true,
                    ..Default::default()
                },
            },
        )
        .await
        .unwrap();

        assert_eq!(resp.responses.len(), 2);
        assert!(resp.responses[0].is_ok());
        assert!(resp.responses[1].is_err());
    }

    #[test]
    fn embedding_model_trait_includes_metadata() {
        let model = FakeEmbedding { dim: 3 };

        fn assert_embedding_model<M>(model: &M)
        where
            M: EmbeddingModel + ?Sized,
        {
            assert_eq!(crate::traits::ModelMetadata::provider_id(model), "fake");
            assert_eq!(
                crate::traits::ModelMetadata::model_id(model),
                "fake-embedding"
            );
            assert_eq!(
                crate::traits::ModelMetadata::specification_version(model),
                ModelSpecVersion::V1
            );
        }

        assert_embedding_model(&model);
    }
}
