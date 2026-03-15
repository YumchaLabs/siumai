//! Embedding model family APIs.
//!
//! This is the recommended Rust-first surface for embeddings:
//! - `embed` for a single request
//! - `embed_many` for batch requests

use crate::retry_api::{RetryOptions, retry_with};
use siumai_core::error::LlmError;
use siumai_core::types::HttpConfig;
use std::collections::HashMap;
use std::time::Duration;

pub use siumai_core::embedding::{EmbeddingModel, EmbeddingModelV3};
pub use siumai_core::types::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingRequest, EmbeddingResponse,
};

/// Options for `embedding::embed` and `embedding::embed_many`.
#[derive(Debug, Clone, Default)]
pub struct EmbedOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
    /// Optional per-call request timeout.
    ///
    /// This is applied via `EmbeddingRequest.http_config.timeout` (for each request in a batch).
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    ///
    /// These are merged into `EmbeddingRequest.http_config.headers`.
    pub headers: HashMap<String, String>,
}

fn apply_embedding_call_options(
    mut request: EmbeddingRequest,
    timeout: Option<Duration>,
    headers: &HashMap<String, String>,
) -> EmbeddingRequest {
    if timeout.is_some() || !headers.is_empty() {
        let mut http = request.http_config.take().unwrap_or_else(HttpConfig::empty);
        if let Some(t) = timeout {
            http.timeout = Some(t);
        }
        if !headers.is_empty() {
            http.headers.extend(headers.clone());
        }
        request.http_config = Some(http);
    }
    request
}

/// Generate embeddings for a single request.
pub async fn embed<M: EmbeddingModelV3 + ?Sized>(
    model: &M,
    request: EmbeddingRequest,
    options: EmbedOptions,
) -> Result<EmbeddingResponse, LlmError> {
    let request = apply_embedding_call_options(request, options.timeout, &options.headers);
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
    let mut requests = requests;
    if options.timeout.is_some() || !options.headers.is_empty() {
        let timeout = options.timeout;
        let headers = options.headers;
        requests.requests = requests
            .requests
            .into_iter()
            .map(|r| apply_embedding_call_options(r, timeout, &headers))
            .collect();
    }
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
