//! Vision capability trait

use crate::error::LlmError;
use crate::types::{ImageGenRequest, ImageResponse, VisionRequest, VisionResponse};
use async_trait::async_trait;

#[async_trait]
pub trait VisionCapability: Send + Sync {
    async fn analyze_image(&self, request: VisionRequest) -> Result<VisionResponse, LlmError>;
    async fn generate_image(&self, request: ImageGenRequest) -> Result<ImageResponse, LlmError>;

    fn get_supported_input_formats(&self) -> Vec<String> {
        vec!["jpeg".to_string(), "png".to_string(), "webp".to_string()]
    }
    fn get_supported_output_formats(&self) -> Vec<String> {
        vec!["png".to_string(), "jpeg".to_string()]
    }
}
