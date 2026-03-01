//! Image generation model family APIs.
//!
//! This is the recommended Rust-first surface for image generation:
//! - `generate` for non-streaming generation

use crate::retry_api::{RetryOptions, retry_with};
use siumai_core::error::LlmError;

pub use siumai_core::image::ImageModelV3;
pub use siumai_core::types::{ImageGenerationRequest, ImageGenerationResponse};

/// Options for `image::generate`.
#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    /// Optional retry policy applied around the model call.
    pub retry: Option<RetryOptions>,
}

/// Generate images.
pub async fn generate<M: ImageModelV3 + ?Sized>(
    model: &M,
    request: ImageGenerationRequest,
    options: GenerateOptions,
) -> Result<ImageGenerationResponse, LlmError> {
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
