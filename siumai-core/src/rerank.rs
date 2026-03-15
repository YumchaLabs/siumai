//! Rerank model family (V3).
//!
//! This module provides a Rust-first, family-oriented abstraction for reranking.
//! In V3-M2 it is implemented as an adapter over `RerankCapability`.

use async_trait::async_trait;

use crate::error::LlmError;
use crate::traits::{ModelMetadata, RerankCapability};
use crate::types::{RerankRequest, RerankResponse};

/// V3 interface for reranking models.
#[async_trait]
pub trait RerankModelV3: Send + Sync {
    /// Rerank candidates for a query.
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError>;
}

/// Stable reranking-model contract for the V4 refactor spike.
pub trait RerankingModel: RerankModelV3 + ModelMetadata + Send + Sync {}

impl<T> RerankingModel for T where T: RerankModelV3 + ModelMetadata + Send + Sync + ?Sized {}

/// Adapter: any `RerankCapability` can be used as a `RerankModelV3`.
#[async_trait]
impl<T> RerankModelV3 for T
where
    T: RerankCapability + Send + Sync + ?Sized,
{
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        RerankCapability::rerank(self, request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ModelSpecVersion;

    struct FakeRerank;

    impl crate::traits::ModelMetadata for FakeRerank {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "fake-rerank"
        }
    }

    #[async_trait]
    impl RerankCapability for FakeRerank {
        async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
            let docs = request.documents.to_strings_lossy();
            Ok(RerankResponse {
                id: "fake".to_string(),
                results: docs
                    .into_iter()
                    .enumerate()
                    .map(|(idx, _)| crate::types::RerankResult {
                        document: None,
                        index: idx as u32,
                        relevance_score: 1.0 / (1.0 + idx as f64),
                    })
                    .collect(),
                tokens: crate::types::RerankTokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                },
            })
        }
    }

    #[tokio::test]
    async fn adapter_rerank_uses_capability() {
        let model = FakeRerank;
        let resp = RerankModelV3::rerank(
            &model,
            RerankRequest::new(
                "fake".into(),
                "q".into(),
                vec!["a".to_string(), "b".to_string()],
            ),
        )
        .await
        .unwrap();

        assert_eq!(resp.results.len(), 2);
        assert_eq!(resp.results[0].index, 0);
        assert_eq!(resp.results[1].index, 1);
    }

    #[test]
    fn reranking_model_trait_includes_metadata() {
        let model = FakeRerank;

        fn assert_reranking_model<M>(model: &M)
        where
            M: RerankingModel + ?Sized,
        {
            assert_eq!(crate::traits::ModelMetadata::provider_id(model), "fake");
            assert_eq!(crate::traits::ModelMetadata::model_id(model), "fake-rerank");
            assert_eq!(
                crate::traits::ModelMetadata::specification_version(model),
                ModelSpecVersion::V1
            );
        }

        assert_reranking_model(&model);
    }
}
