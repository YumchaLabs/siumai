//! Image generation capability trait

use crate::error::LlmError;
use crate::types::{
    ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse, ImageVariationRequest,
};
use async_trait::async_trait;

#[async_trait]
pub trait ImageGenerationCapability: Send + Sync {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError>;

    async fn generate_image(
        &self,
        prompt: String,
        size: Option<String>,
        count: Option<u32>,
    ) -> Result<Vec<String>, LlmError> {
        let request = ImageGenerationRequest {
            prompt,
            size,
            count: count.unwrap_or(1),
            ..Default::default()
        };
        let response = self.generate_images(request).await?;
        Ok(response
            .images
            .into_iter()
            .filter_map(|img| img.url)
            .collect())
    }
}

/// Provider-specific image extras (non-unified surface).
///
/// Vercel AI SDK's unified `ImageModel` surface focuses on generation.
/// Image editing/variation and provider-specific metadata are intentionally
/// separated into this extension trait.
#[async_trait]
pub trait ImageExtras: Send + Sync {
    async fn edit_image(
        &self,
        _request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Image editing not supported by this provider".to_string(),
        ))
    }

    async fn create_variation(
        &self,
        _request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Image variations not supported by this provider".to_string(),
        ))
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        Vec::new()
    }

    fn get_supported_formats(&self) -> Vec<String> {
        Vec::new()
    }

    fn supports_image_editing(&self) -> bool {
        false
    }

    fn supports_image_variations(&self) -> bool {
        false
    }
}
