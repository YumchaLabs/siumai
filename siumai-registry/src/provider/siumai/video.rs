use super::Siumai;
use crate::error::LlmError;
use crate::traits::VideoGenerationCapability;
use crate::types::*;

#[async_trait::async_trait]
impl VideoGenerationCapability for Siumai {
    async fn create_video_task(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResponse, LlmError> {
        if let Some(video) = self.client.as_video_generation_capability() {
            video.create_video_task(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support video generation.",
                self.client.provider_id()
            )))
        }
    }

    async fn query_video_task(&self, task_id: &str) -> Result<VideoTaskStatusResponse, LlmError> {
        if let Some(video) = self.client.as_video_generation_capability() {
            video.query_video_task(task_id).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support video generation.",
                self.client.provider_id()
            )))
        }
    }

    fn get_supported_models(&self) -> Vec<String> {
        self.client
            .as_video_generation_capability()
            .map(|v| v.get_supported_models())
            .unwrap_or_default()
    }

    fn get_supported_resolutions(&self, model: &str) -> Vec<String> {
        self.client
            .as_video_generation_capability()
            .map(|v| v.get_supported_resolutions(model))
            .unwrap_or_default()
    }

    fn get_supported_durations(&self, model: &str) -> Vec<u32> {
        self.client
            .as_video_generation_capability()
            .map(|v| v.get_supported_durations(model))
            .unwrap_or_default()
    }
}
