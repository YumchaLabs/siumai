use super::GeminiClient;
use crate::error::LlmError;
use crate::traits::VideoGenerationCapability;
use async_trait::async_trait;

#[async_trait]
impl VideoGenerationCapability for GeminiClient {
    async fn create_video_task(
        &self,
        request: crate::types::video::VideoGenerationRequest,
    ) -> Result<crate::types::video::VideoGenerationResponse, LlmError> {
        let helper = super::super::video::GeminiVideo::new(
            self.config.clone(),
            self.http_client.clone(),
            self.http_interceptors.clone(),
            self.retry_options.clone(),
            self.config.http_transport.clone(),
        );
        helper.create_video_task(request).await
    }

    async fn query_video_task(
        &self,
        task_id: &str,
    ) -> Result<crate::types::video::VideoTaskStatusResponse, LlmError> {
        let helper = super::super::video::GeminiVideo::new(
            self.config.clone(),
            self.http_client.clone(),
            self.http_interceptors.clone(),
            self.retry_options.clone(),
            self.config.http_transport.clone(),
        );
        helper.query_video_task(task_id).await
    }

    fn get_supported_models(&self) -> Vec<String> {
        super::super::video::get_supported_veo_models()
    }

    fn get_supported_resolutions(&self, model: &str) -> Vec<String> {
        super::super::video::get_supported_veo_resolutions(model)
    }

    fn get_supported_durations(&self, model: &str) -> Vec<u32> {
        super::super::video::get_supported_veo_durations(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::LlmClient;

    #[test]
    fn gemini_client_exposes_video_capability_when_enabled() {
        let client =
            GeminiClient::new(crate::providers::gemini::GeminiConfig::default()).expect("client");
        assert!(client.as_video_generation_capability().is_some());
    }
}
