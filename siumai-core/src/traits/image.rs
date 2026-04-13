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

/// Provider-specific image extras that sit outside the stable family contract.
///
/// Vercel AI SDK exposes one shared `ImageModelV4` request/result surface, but
/// providers still vary widely in how image edit/variation routes are actually
/// executed. Siumai keeps those transport/provider-specific behaviors on this
/// extension trait while the stable family contract remains centered on the
/// shared image-generation request/response types.
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
