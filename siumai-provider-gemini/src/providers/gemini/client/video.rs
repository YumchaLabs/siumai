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

    fn polling_options(
        &self,
        request: &crate::types::video::VideoGenerationRequest,
    ) -> Result<siumai_core::video::VideoPollingOptions, LlmError> {
        super::super::video::polling_options(request)
    }

    async fn materialize_video_reference(
        &self,
        provider_reference: &crate::types::ProviderReference,
    ) -> Result<crate::types::MaterializedVideoAsset, LlmError> {
        let file_id = provider_reference
            .preferred_value(&["gemini", "google"])
            .ok_or_else(|| {
                LlmError::InvalidInput(format!(
                    "Gemini video materialization expected a `gemini` or `google` provider reference, got providers {:?}",
                    provider_reference.available_providers()
                ))
            })?;

        let bytes =
            crate::traits::FileManagementCapability::get_file_content(self, file_id.to_string())
                .await?;

        Ok(crate::types::MaterializedVideoAsset::new(bytes))
    }

    fn max_videos_per_call(&self) -> Option<u32> {
        Some(4)
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
