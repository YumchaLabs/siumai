//! Image generation model family.
//!
//! This module provides a Rust-first, family-oriented abstraction for image generation.
//! It is intentionally implemented as an adapter over the existing
//! `ImageGenerationCapability` while provider construction continues moving toward
//! model-family traits.

use async_trait::async_trait;

use crate::error::LlmError;
use crate::traits::{ImageGenerationCapability, ModelMetadata};
use crate::types::{ImageGenerationRequest, ImageGenerationResponse};

/// Stable Rust interface for image generation models.
#[async_trait]
pub trait ImageModel: ModelMetadata + Send + Sync {
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

/// AI SDK V4 provider-facing image-model marker.
///
/// The current Rust execution surface is carried by `ImageModel`; this marker remains available
/// for code that wants to express intentional alignment with upstream `ImageModelV4` naming.
pub trait ImageModelV4: ImageModel {}

impl<T> ImageModelV4 for T where T: ImageModel + ?Sized {}

/// Adapter: any `ImageGenerationCapability` with metadata can be used as an `ImageModel`.
#[async_trait]
impl<T> ImageModel for T
where
    T: ImageGenerationCapability + ModelMetadata + Send + Sync + ?Sized,
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
        let resp = ImageModel::generate(
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

        assert_eq!(ImageModel::max_images_per_call(&model), Some(3));
    }
}
