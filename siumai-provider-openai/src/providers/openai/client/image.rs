use super::OpenAiClient;
use crate::error::LlmError;
use crate::traits::{ImageExtras, ImageGenerationCapability};
use crate::types::{
    ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse, ImageVariationRequest,
};
use async_trait::async_trait;

#[async_trait]
impl ImageGenerationCapability for OpenAiClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let mut request = request;
        self.merge_default_provider_options_map(&mut request.provider_options_map);
        let exec = self.build_image_executor(&request);
        use crate::execution::executors::image::ImageExecutor;
        ImageExecutor::execute(&*exec, request).await
    }
}

#[async_trait]
impl ImageExtras for OpenAiClient {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        use crate::execution::executors::image::ImageExecutor;

        let mut request = request;
        self.merge_default_provider_options_map(&mut request.provider_options_map);

        let exec = self.build_image_executor(&ImageGenerationRequest::default());
        ImageExecutor::execute_edit(&*exec, request).await
    }

    async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        use crate::execution::executors::image::ImageExecutor;

        let mut request = request;
        self.merge_default_provider_options_map(&mut request.provider_options_map);

        let exec = self.build_image_executor(&ImageGenerationRequest::default());
        ImageExecutor::execute_variation(&*exec, request).await
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        if self.base_url.contains("siliconflow.cn") {
            vec![
                "1024x1024".to_string(),
                "960x1280".to_string(),
                "768x1024".to_string(),
                "720x1440".to_string(),
                "720x1280".to_string(),
            ]
        } else {
            vec![
                "256x256".to_string(),
                "512x512".to_string(),
                "1024x1024".to_string(),
                "1792x1024".to_string(),
                "1024x1792".to_string(),
                "2048x2048".to_string(),
            ]
        }
    }

    fn get_supported_formats(&self) -> Vec<String> {
        if self.base_url.contains("siliconflow.cn") {
            vec!["url".to_string()]
        } else {
            vec!["url".to_string(), "b64_json".to_string()]
        }
    }

    fn supports_image_editing(&self) -> bool {
        !self.base_url.contains("siliconflow.cn")
    }

    fn supports_image_variations(&self) -> bool {
        !self.base_url.contains("siliconflow.cn")
    }
}
