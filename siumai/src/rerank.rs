//! Rerank model family APIs.
//!
//! This is the recommended Rust-first surface for reranking:
//! - `rerank`

use crate::retry_api::{RetryOptions, retry_with};
use siumai_core::error::LlmError;

pub use siumai_core::rerank::RerankModelV3;
pub use siumai_core::types::{RerankRequest, RerankResponse};

/// Options for `rerank::rerank`.
#[derive(Debug, Clone, Default)]
pub struct RerankOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
}

/// Rerank candidates for a query.
pub async fn rerank<M: RerankModelV3 + ?Sized>(
    model: &M,
    request: RerankRequest,
    options: RerankOptions,
) -> Result<RerankResponse, LlmError> {
    if let Some(retry) = options.retry {
        retry_with(
            || {
                let req = request.clone();
                async move { model.rerank(req).await }
            },
            retry,
        )
        .await
    } else {
        model.rerank(request).await
    }
}
