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

    fn max_images_per_call(&self) -> Option<u32> {
        self.client
            .as_image_generation_capability()
            .and_then(ImageGenerationCapability::max_images_per_call)
    }
}
