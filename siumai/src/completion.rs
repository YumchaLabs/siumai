//! Completion model family APIs.
//!
//! This is the recommended Rust-first surface for completion endpoints:
//! - `complete`
//! - `stream`
//! - `stream_with_cancel`

use crate::retry_api::{RetryOptions, retry_with};
use siumai_core::error::LlmError;
use siumai_core::types::HttpConfig;
use std::collections::HashMap;
use std::time::Duration;

pub use siumai_core::completion::{
    CompletionModel, CompletionModelV3, CompletionStream, CompletionStreamHandle,
};
pub use siumai_core::types::{CompletionRequest, CompletionResponse, StreamRequestOptions};

/// Options for `completion::complete`.
#[derive(Debug, Clone, Default)]
pub struct CompleteOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
    /// Optional per-call request timeout.
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    pub headers: HashMap<String, String>,
}

/// Options for `completion::stream`.
#[derive(Debug, Clone, Default)]
pub struct StreamOptions {
    /// Optional retry policy applied when establishing the stream.
    pub retry: Option<RetryOptions>,
    /// Optional per-call request timeout.
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    pub headers: HashMap<String, String>,
    /// Include provider raw chunks on the stream part lane.
    pub include_raw_chunks: bool,
}

fn apply_completion_call_options(
    mut request: CompletionRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> CompletionRequest {
    if timeout.is_some() || !headers.is_empty() {
        let mut http = request.http_config.take().unwrap_or_else(HttpConfig::empty);
        if let Some(timeout) = timeout {
            http.timeout = Some(timeout);
        }
        if !headers.is_empty() {
            http.headers.extend(headers);
        }
        request.http_config = Some(http);
    }

    request
}

/// Execute a non-streaming completion request.
pub async fn complete<M: CompletionModel + ?Sized>(
    model: &M,
    request: CompletionRequest,
    options: CompleteOptions,
) -> Result<CompletionResponse, LlmError> {
    let request = apply_completion_call_options(request, options.timeout, options.headers);
    if let Some(retry) = options.retry {
        retry_with(
            || {
                let request = request.clone();
                async move { model.complete(request).await }
            },
            retry,
        )
        .await
    } else {
        model.complete(request).await
    }
}

/// Execute a streaming completion request.
pub async fn stream<M: CompletionModel + ?Sized>(
    model: &M,
    request: CompletionRequest,
    options: StreamOptions,
) -> Result<CompletionStream, LlmError> {
    let mut request = apply_completion_call_options(request, options.timeout, options.headers);
    if options.include_raw_chunks {
        request = request.with_include_raw_chunks(true);
    }
    if let Some(retry) = options.retry {
        retry_with(
            || {
                let request = request.clone();
                async move { model.stream(request).await }
            },
            retry,
        )
        .await
    } else {
        model.stream(request).await
    }
}

/// Execute a streaming completion request with cancellation support.
pub async fn stream_with_cancel<M: CompletionModel + ?Sized>(
    model: &M,
    request: CompletionRequest,
    options: StreamOptions,
) -> Result<CompletionStreamHandle, LlmError> {
    let mut request = apply_completion_call_options(request, options.timeout, options.headers);
    if options.include_raw_chunks {
        request = request.with_include_raw_chunks(true);
    }
    if let Some(retry) = options.retry {
        retry_with(
            || {
                let request = request.clone();
                async move { model.stream_with_cancel(request).await }
            },
            retry,
        )
        .await
    } else {
        model.stream_with_cancel(request).await
    }
}
