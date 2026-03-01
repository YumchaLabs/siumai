//! Embedding model family APIs.
//!
//! This is the recommended Rust-first surface for embeddings:
//! - `embed` for a single request
//! - `embed_many` for batch requests

use crate::retry_api::{RetryOptions, retry_with};
use siumai_core::error::LlmError;

pub use siumai_core::embedding::EmbeddingModelV3;
pub use siumai_core::types::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingRequest, EmbeddingResponse,
};

/// Options for `embedding::embed` and `embedding::embed_many`.
#[derive(Debug, Clone, Default)]
pub struct EmbedOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
}

/// Generate embeddings for a single request.
pub async fn embed<M: EmbeddingModelV3 + ?Sized>(
    model: &M,
    request: EmbeddingRequest,
    options: EmbedOptions,
) -> Result<EmbeddingResponse, LlmError> {
    if let Some(retry) = options.retry {
        retry_with(
            || {
                let req = request.clone();
                async move { model.embed(req).await }
            },
            retry,
        )
        .await
    } else {
        model.embed(request).await
    }
}

/// Generate embeddings for a batch of requests.
pub async fn embed_many<M: EmbeddingModelV3 + ?Sized>(
    model: &M,
    requests: BatchEmbeddingRequest,
    options: EmbedOptions,
) -> Result<BatchEmbeddingResponse, LlmError> {
    if let Some(retry) = options.retry {
        retry_with(
            || {
                let req = requests.clone();
                async move { model.embed_many(req).await }
            },
            retry,
        )
        .await
    } else {
        model.embed_many(requests).await
    }
}
