//! Completion model family APIs.
//!
//! This is the recommended Rust-first surface for completion endpoints:
//! - `complete`
//! - `stream`
//! - `stream_with_cancel`

use crate::request_options::{
    EffectiveRequestOptions, link_stream_handle_abort, retry_or_call_with_abort,
    wrap_stream_with_abort,
};
use crate::retry_api::RetryOptions;
use siumai_core::error::LlmError;
use siumai_core::types::{HttpConfig, RequestOptions};
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
    /// AI SDK-style request controls.
    ///
    /// When present, `max_retries` defaults to 2 to match AI SDK. Legacy
    /// `retry`, `timeout`, and `headers` fields override equivalent values here.
    pub request_options: Option<RequestOptions>,
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
    /// AI SDK-style request controls.
    ///
    /// When present, `max_retries` defaults to 2 to match AI SDK. Legacy
    /// `retry`, `timeout`, and `headers` fields override equivalent values here.
    pub request_options: Option<RequestOptions>,
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
    let effective = EffectiveRequestOptions::from_parts(
        options.request_options,
        options.retry,
        options.timeout,
        options.headers,
    );
    let request = apply_completion_call_options(request, effective.timeout(), effective.headers());
    retry_or_call_with_abort(effective.retry(), effective.abort_signal(), || {
        let request = request.clone();
        async move { model.complete(request).await }
    })
    .await
}

/// Execute a streaming completion request.
pub async fn stream<M: CompletionModel + ?Sized>(
    model: &M,
    request: CompletionRequest,
    options: StreamOptions,
) -> Result<CompletionStream, LlmError> {
    let effective = EffectiveRequestOptions::from_parts(
        options.request_options,
        options.retry,
        options.timeout,
        options.headers,
    );
    let mut request =
        apply_completion_call_options(request, effective.timeout(), effective.headers());
    if options.include_raw_chunks {
        request = request.with_include_raw_chunks(true);
    }
    let abort_signal = effective.abort_signal();
    let stream = retry_or_call_with_abort(effective.retry(), abort_signal.clone(), || {
        let request = request.clone();
        async move { model.stream(request).await }
    })
    .await?;
    Ok(wrap_stream_with_abort(stream, abort_signal))
}

/// Execute a streaming completion request with cancellation support.
pub async fn stream_with_cancel<M: CompletionModel + ?Sized>(
    model: &M,
    request: CompletionRequest,
    options: StreamOptions,
) -> Result<CompletionStreamHandle, LlmError> {
    let effective = EffectiveRequestOptions::from_parts(
        options.request_options,
        options.retry,
        options.timeout,
        options.headers,
    );
    let mut request =
        apply_completion_call_options(request, effective.timeout(), effective.headers());
    if options.include_raw_chunks {
        request = request.with_include_raw_chunks(true);
    }
    let abort_signal = effective.abort_signal();
    let handle = retry_or_call_with_abort(effective.retry(), abort_signal.clone(), || {
        let request = request.clone();
        async move { model.stream_with_cancel(request).await }
    })
    .await?;
    Ok(link_stream_handle_abort(handle, abort_signal))
}
