//! Speech (text-to-speech) model family APIs.
//!
//! This is the recommended Rust-first surface for TTS:
//! - `synthesize`

use crate::retry_api::{RetryOptions, retry_with};
use siumai_core::error::LlmError;

pub use siumai_core::speech::SpeechModelV3;
pub use siumai_core::types::{TtsRequest, TtsResponse};

/// Options for `speech::synthesize`.
#[derive(Debug, Clone, Default)]
pub struct SynthesizeOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
}

/// Synthesize audio from text.
pub async fn synthesize<M: SpeechModelV3 + ?Sized>(
    model: &M,
    request: TtsRequest,
    options: SynthesizeOptions,
) -> Result<TtsResponse, LlmError> {
    if let Some(retry) = options.retry {
        retry_with(
            || {
                let req = request.clone();
                async move { model.synthesize(req).await }
            },
            retry,
        )
        .await
    } else {
        model.synthesize(request).await
    }
}
