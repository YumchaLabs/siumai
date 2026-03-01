//! Image generation model family (V3).
//!
//! This module provides a Rust-first, family-oriented abstraction for image generation.
//! In V3-M2 it is implemented as an adapter over `ImageGenerationCapability`.

use async_trait::async_trait;

use crate::error::LlmError;
use crate::traits::ImageGenerationCapability;
use crate::types::{ImageGenerationRequest, ImageGenerationResponse};

/// V3 interface for image generation models.
#[async_trait]
pub trait ImageModelV3: Send + Sync {
    /// Generate images from a request.
    async fn generate(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError>;
}

/// Adapter: any `ImageGenerationCapability` can be used as an `ImageModelV3`.
#[async_trait]
impl<T> ImageModelV3 for T
where
    T: ImageGenerationCapability + Send + Sync + ?Sized,
{
    async fn generate(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        self.generate_images(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct FakeImage;

    #[async_trait]
    impl ImageGenerationCapability for FakeImage {
        async fn generate_images(
            &self,
            request: ImageGenerationRequest,
        ) -> Result<ImageGenerationResponse, LlmError> {
            Ok(ImageGenerationResponse {
                images: vec![crate::types::GeneratedImage {
                    url: Some(format!("https://example.com/{}.png", request.prompt)),
                    b64_json: None,
                    format: None,
                    width: None,
                    height: None,
                    revised_prompt: None,
                    metadata: HashMap::new(),
                }],
                metadata: HashMap::new(),
                warnings: None,
                response: None,
            })
        }
    }

    #[tokio::test]
    async fn adapter_generate_uses_capability() {
        let model = FakeImage;
        let resp = ImageModelV3::generate(
            &model,
            ImageGenerationRequest {
                prompt: "cat".to_string(),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(resp.images.len(), 1);
        assert_eq!(
            resp.images[0].url.as_deref(),
            Some("https://example.com/cat.png")
        );
    }
}
