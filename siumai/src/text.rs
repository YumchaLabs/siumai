//! Text model family APIs.
//!
//! This is the recommended Rust-first surface for text generation:
//! - `generate` for non-streaming
//! - `stream` for streaming
//! - `stream_with_cancel` for streaming with cancellation

use crate::retry_api::{RetryOptions, retry_with};
use siumai_core::error::LlmError;

pub use siumai_core::text::{TextModelV3, TextRequest, TextResponse, TextStream, TextStreamHandle};

/// Options for `text::generate`.
#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
}

/// Options for `text::stream`.
#[derive(Debug, Clone, Default)]
pub struct StreamOptions {
    /// Optional retry policy applied when establishing the stream.
    ///
    /// Note: this retries stream *creation* only. It does not retry mid-stream failures.
    pub retry: Option<RetryOptions>,
}

/// Generate a non-streaming text response.
pub async fn generate<M: TextModelV3 + ?Sized>(
    model: &M,
    request: TextRequest,
    options: GenerateOptions,
) -> Result<TextResponse, LlmError> {
    if let Some(retry) = options.retry {
        retry_with(
            || {
                let req = request.clone();
                async move { model.generate(req).await }
            },
            retry,
        )
        .await
    } else {
        model.generate(request).await
    }
}

/// Generate a streaming text response.
pub async fn stream<M: TextModelV3 + ?Sized>(
    model: &M,
    request: TextRequest,
    options: StreamOptions,
) -> Result<TextStream, LlmError> {
    if let Some(retry) = options.retry {
        retry_with(
            || {
                let req = request.clone();
                async move { model.stream(req).await }
            },
            retry,
        )
        .await
    } else {
        model.stream(request).await
    }
}

/// Generate a streaming text response with cancellation support.
pub async fn stream_with_cancel<M: TextModelV3 + ?Sized>(
    model: &M,
    request: TextRequest,
    options: StreamOptions,
) -> Result<TextStreamHandle, LlmError> {
    if let Some(retry) = options.retry {
        retry_with(
            || {
                let req = request.clone();
                async move { model.stream_with_cancel(req).await }
            },
            retry,
        )
        .await
    } else {
        model.stream_with_cancel(request).await
    }
}
