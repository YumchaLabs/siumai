use super::Siumai;
use crate::error::LlmError;
use crate::traits::ImageExtras;
use crate::types::*;

#[async_trait::async_trait]
impl ImageExtras for Siumai {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if let Some(img) = self.client.as_image_extras() {
            img.edit_image(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support image extras.",
                self.client.provider_id()
            )))
        }
    }

    async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if let Some(img) = self.client.as_image_extras() {
            img.create_variation(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support image extras.",
                self.client.provider_id()
            )))
        }
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        self.client
            .as_image_extras()
            .map(|i| i.get_supported_sizes())
            .unwrap_or_default()
    }

    fn get_supported_formats(&self) -> Vec<String> {
        self.client
            .as_image_extras()
            .map(|i| i.get_supported_formats())
            .unwrap_or_default()
    }

    fn supports_image_editing(&self) -> bool {
        self.client
            .as_image_extras()
            .map(|i| i.supports_image_editing())
            .unwrap_or(false)
    }

    fn supports_image_variations(&self) -> bool {
        self.client
            .as_image_extras()
            .map(|i| i.supports_image_variations())
            .unwrap_or(false)
    }
}
