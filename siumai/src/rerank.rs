//! Rerank model family APIs.
//!
//! This is the recommended Rust-first surface for reranking:
//! - `rerank`

use crate::retry_api::{RetryOptions, retry_with};
use siumai_core::error::LlmError;
use siumai_core::types::HttpConfig;
use std::collections::HashMap;
use std::time::Duration;

pub use siumai_core::rerank::RerankModelV3;
pub use siumai_core::types::{RerankRequest, RerankResponse};

/// Options for `rerank::rerank`.
#[derive(Debug, Clone, Default)]
pub struct RerankOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
    /// Optional per-call request timeout.
    ///
    /// This is applied via `RerankRequest.http_config.timeout`.
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    ///
    /// These are merged into `RerankRequest.http_config.headers`.
    pub headers: HashMap<String, String>,
}

fn apply_rerank_call_options(
    mut request: RerankRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> RerankRequest {
    if timeout.is_some() || !headers.is_empty() {
        let mut http = request.http_config.take().unwrap_or_else(HttpConfig::empty);
        if let Some(t) = timeout {
            http.timeout = Some(t);
        }
        if !headers.is_empty() {
            http.headers.extend(headers);
        }
        request.http_config = Some(http);
    }
    request
}

/// Rerank candidates for a query.
pub async fn rerank<M: RerankModelV3 + ?Sized>(
    model: &M,
    request: RerankRequest,
    options: RerankOptions,
) -> Result<RerankResponse, LlmError> {
    let request = apply_rerank_call_options(request, options.timeout, options.headers);
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
