//! Transcription (speech-to-text) model family APIs.
//!
//! This is the recommended Rust-first surface for STT:
//! - `transcribe`

use crate::retry_api::{RetryOptions, retry_with};
use siumai_core::error::LlmError;
use siumai_core::types::HttpConfig;
use std::collections::HashMap;
use std::time::Duration;

pub use siumai_core::transcription::{TranscriptionModel, TranscriptionModelV3};
pub use siumai_core::types::{SttRequest, SttResponse};

/// Options for `transcription::transcribe`.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
    /// Optional per-call request timeout.
    ///
    /// This is applied via `SttRequest.http_config.timeout`.
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    ///
    /// These are merged into `SttRequest.http_config.headers`.
    pub headers: HashMap<String, String>,
}

fn apply_stt_call_options(
    mut request: SttRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> SttRequest {
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

/// Transcribe audio into text.
pub async fn transcribe<M: TranscriptionModelV3 + ?Sized>(
    model: &M,
    request: SttRequest,
    options: TranscribeOptions,
) -> Result<SttResponse, LlmError> {
    let request = apply_stt_call_options(request, options.timeout, options.headers);
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
