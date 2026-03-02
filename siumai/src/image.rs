//! Image generation model family APIs.
//!
//! This is the recommended Rust-first surface for image generation:
//! - `generate` for non-streaming generation

use crate::retry_api::{RetryOptions, retry_with};
use siumai_core::error::LlmError;
use siumai_core::types::HttpConfig;
use std::collections::HashMap;
use std::time::Duration;

pub use siumai_core::image::ImageModelV3;
pub use siumai_core::types::{ImageGenerationRequest, ImageGenerationResponse};

/// Options for `image::generate`.
#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
    /// Optional per-call request timeout.
    ///
    /// This is applied via `ImageGenerationRequest.http_config.timeout`.
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    ///
    /// These are merged into `ImageGenerationRequest.http_config.headers`.
    pub headers: HashMap<String, String>,
}

fn apply_image_call_options(
    mut request: ImageGenerationRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> ImageGenerationRequest {
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

/// Generate images.
pub async fn generate<M: ImageModelV3 + ?Sized>(
    model: &M,
    request: ImageGenerationRequest,
    options: GenerateOptions,
) -> Result<ImageGenerationResponse, LlmError> {
    let request = apply_image_call_options(request, options.timeout, options.headers);
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
