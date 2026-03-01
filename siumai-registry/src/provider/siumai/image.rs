use super::Siumai;
use crate::error::LlmError;
use crate::traits::ImageGenerationCapability;
use crate::types::{ImageGenerationRequest, ImageGenerationResponse};

#[async_trait::async_trait]
impl ImageGenerationCapability for Siumai {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if let Some(img) = self.client.as_image_generation_capability() {
            img.generate_images(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support image generation.",
                self.client.provider_id()
            )))
        }
    }
}
