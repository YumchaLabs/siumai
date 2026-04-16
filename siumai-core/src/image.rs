//! Image generation model family (V3/V4-compatible).
//!
//! This module provides a Rust-first, family-oriented abstraction for image generation.
//! The stable runtime request/response contract now also exposes AI SDK-style
//! `ImageModelV4` naming while keeping the historical `ImageModelV3` /
//! `ImageModel` aliases for compatibility.
//! In V3-M2 it is implemented as an adapter over `ImageGenerationCapability`.

use async_trait::async_trait;

use crate::error::LlmError;
use crate::traits::{ImageGenerationCapability, ModelMetadata};
use crate::types::{ImageGenerationRequest, ImageGenerationResponse};

/// V3 interface for image generation models.
#[async_trait]
pub trait ImageModelV3: Send + Sync {
    /// Generate images from a request.
    async fn generate(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError>;

    /// Maximum number of images this model can generate in a single call.
    ///
    /// Helpers use this object-safe metadata to batch larger `count` requests in
    /// a way that mirrors AI SDK `maxImagesPerCall`.
    fn max_images_per_call(&self) -> Option<u32> {
        None
    }
}

/// Stable image-model contract aligned with AI SDK `ImageModelV4`.
pub trait ImageModelV4: ImageModelV3 + ModelMetadata + Send + Sync {}

impl<T> ImageModelV4 for T where T: ImageModelV3 + ModelMetadata + Send + Sync + ?Sized {}

/// Short compatibility alias kept for the Rust facade.
pub trait ImageModel: ImageModelV4 {}

impl<T> ImageModel for T where T: ImageModelV4 + ?Sized {}

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

    fn max_images_per_call(&self) -> Option<u32> {
        ImageGenerationCapability::max_images_per_call(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ModelSpecVersion;
    use std::collections::HashMap;

    struct FakeImage;

    impl crate::traits::ModelMetadata for FakeImage {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "fake-image"
        }
    }

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

        fn max_images_per_call(&self) -> Option<u32> {
            Some(3)
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

    #[test]
    fn image_model_trait_includes_metadata() {
        let model = FakeImage;

        fn assert_image_model<M>(model: &M)
        where
            M: ImageModel + ?Sized,
        {
            assert_eq!(crate::traits::ModelMetadata::provider_id(model), "fake");
            assert_eq!(crate::traits::ModelMetadata::model_id(model), "fake-image");
            assert_eq!(
                crate::traits::ModelMetadata::specification_version(model),
                ModelSpecVersion::V1
            );
        }

        assert_image_model(&model);
    }

    #[test]
    fn adapter_preserves_capability_batch_limit() {
        let model = FakeImage;

        assert_eq!(ImageModelV3::max_images_per_call(&model), Some(3));
    }
}
