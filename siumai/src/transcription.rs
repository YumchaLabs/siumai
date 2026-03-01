//! Transcription (speech-to-text) model family APIs.
//!
//! This is the recommended Rust-first surface for STT:
//! - `transcribe`

use crate::retry_api::{RetryOptions, retry_with};
use siumai_core::error::LlmError;

pub use siumai_core::transcription::TranscriptionModelV3;
pub use siumai_core::types::{SttRequest, SttResponse};

/// Options for `transcription::transcribe`.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
}

/// Transcribe audio into text.
pub async fn transcribe<M: TranscriptionModelV3 + ?Sized>(
    model: &M,
    request: SttRequest,
    options: TranscribeOptions,
) -> Result<SttResponse, LlmError> {
    if let Some(retry) = options.retry {
        retry_with(
            || {
                let req = request.clone();
                async move { model.transcribe(req).await }
            },
            retry,
        )
        .await
    } else {
        model.transcribe(request).await
    }
}
