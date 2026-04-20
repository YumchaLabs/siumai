//! Video generation model family (task-oriented, V3/V4-compatible).
//!
//! This module provides a Rust-first, family-oriented abstraction for video generation.
//! Unlike the AI SDK's auto-polling `doGenerate()` shape, the stable Rust contract keeps
//! explicit task submission and task-status querying as first-class operations.
//! The stable naming still exposes `VideoModelV4` for package-surface auditability.

use async_trait::async_trait;

use crate::error::LlmError;
use crate::traits::{ModelMetadata, VideoGenerationCapability};
use crate::types::{VideoGenerationRequest, VideoGenerationResponse, VideoTaskStatusResponse};

/// V3 interface for task-oriented video generation models.
#[async_trait]
pub trait VideoModelV3: Send + Sync {
    /// Submit a new video-generation task.
    async fn create_task(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResponse, LlmError>;

    /// Query an existing video-generation task.
    async fn query_task(&self, task_id: &str) -> Result<VideoTaskStatusResponse, LlmError>;

    /// Maximum number of final videos this model can produce in a single task call.
    ///
    /// Helpers use this object-safe metadata to batch larger `count` requests in
    /// a way that mirrors AI SDK `maxVideosPerCall`.
    fn max_videos_per_call(&self) -> Option<u32> {
        None
    }

    /// List supported model ids when the provider can expose them cheaply.
    fn supported_models(&self) -> Vec<String> {
        Vec::new()
    }

    /// List supported output resolutions for a given model id.
    fn supported_resolutions(&self, _model: &str) -> Vec<String> {
        Vec::new()
    }

    /// List supported output durations for a given model id.
    fn supported_durations(&self, _model: &str) -> Vec<u32> {
        Vec::new()
    }
}

/// Stable metadata-bearing video-model contract.
///
/// The name stays aligned with AI SDK `VideoModelV4`, but the execution semantics remain
/// explicitly task-oriented on the Rust side.
pub trait VideoModelV4: VideoModelV3 + ModelMetadata + Send + Sync {}

impl<T> VideoModelV4 for T where T: VideoModelV3 + ModelMetadata + Send + Sync + ?Sized {}

/// Short compatibility alias kept for the Rust facade.
pub trait VideoModel: VideoModelV4 {}

impl<T> VideoModel for T where T: VideoModelV4 + ?Sized {}

/// Adapter: any `VideoGenerationCapability` can be used as a `VideoModelV3`.
#[async_trait]
impl<T> VideoModelV3 for T
where
    T: VideoGenerationCapability + Send + Sync + ?Sized,
{
    async fn create_task(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResponse, LlmError> {
        self.create_video_task(request).await
    }

    async fn query_task(&self, task_id: &str) -> Result<VideoTaskStatusResponse, LlmError> {
        self.query_video_task(task_id).await
    }

    fn max_videos_per_call(&self) -> Option<u32> {
        VideoGenerationCapability::max_videos_per_call(self)
    }

    fn supported_models(&self) -> Vec<String> {
        VideoGenerationCapability::get_supported_models(self)
    }

    fn supported_resolutions(&self, model: &str) -> Vec<String> {
        VideoGenerationCapability::get_supported_resolutions(self, model)
    }

    fn supported_durations(&self, model: &str) -> Vec<u32> {
        VideoGenerationCapability::get_supported_durations(self, model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ModelSpecVersion;
    use crate::types::VideoTaskStatus;
    use std::collections::HashMap;

    struct FakeVideo;

    impl crate::traits::ModelMetadata for FakeVideo {
        fn provider_id(&self) -> &str {
            "fake"
        }

        fn model_id(&self) -> &str {
            "fake-video"
        }
    }

    #[async_trait]
    impl VideoGenerationCapability for FakeVideo {
        async fn create_video_task(
            &self,
            request: VideoGenerationRequest,
        ) -> Result<VideoGenerationResponse, LlmError> {
            Ok(VideoGenerationResponse {
                task_id: format!("task:{}", request.prompt.unwrap_or_default()),
                base_resp: None,
                metadata: HashMap::new(),
                warnings: None,
                response: None,
            })
        }

        async fn query_video_task(
            &self,
            task_id: &str,
        ) -> Result<VideoTaskStatusResponse, LlmError> {
            Ok(VideoTaskStatusResponse {
                task_id: task_id.to_string(),
                status: VideoTaskStatus::Success,
                file_id: Some("file-123".to_string()),
                video_url: Some("https://example.com/video.mp4".to_string()),
                duration: Some(6.0),
                video_width: Some(1280),
                video_height: Some(720),
                base_resp: None,
                metadata: HashMap::new(),
                response: None,
            })
        }

        fn get_supported_models(&self) -> Vec<String> {
            vec!["fake-video".to_string()]
        }

        fn max_videos_per_call(&self) -> Option<u32> {
            Some(4)
        }

        fn get_supported_resolutions(&self, _model: &str) -> Vec<String> {
            vec!["720p".to_string()]
        }

        fn get_supported_durations(&self, _model: &str) -> Vec<u32> {
            vec![6]
        }
    }

    #[tokio::test]
    async fn adapter_create_task_uses_capability() {
        let model = FakeVideo;
        let response =
            VideoModelV3::create_task(&model, VideoGenerationRequest::new("fake-video", "robot"))
                .await
                .unwrap();

        assert_eq!(response.task_id, "task:robot");
    }

    #[tokio::test]
    async fn adapter_query_task_uses_capability() {
        let model = FakeVideo;
        let response = VideoModelV3::query_task(&model, "task:robot")
            .await
            .unwrap();

        assert_eq!(response.status, VideoTaskStatus::Success);
        assert_eq!(
            response.video_url.as_deref(),
            Some("https://example.com/video.mp4")
        );
    }

    #[test]
    fn video_model_trait_includes_metadata() {
        let model = FakeVideo;

        fn assert_video_model<M>(model: &M)
        where
            M: VideoModel + ?Sized,
        {
            assert_eq!(crate::traits::ModelMetadata::provider_id(model), "fake");
            assert_eq!(crate::traits::ModelMetadata::model_id(model), "fake-video");
            assert_eq!(
                crate::traits::ModelMetadata::specification_version(model),
                ModelSpecVersion::V1
            );
        }

        assert_video_model(&model);
    }

    #[test]
    fn adapter_preserves_supported_metadata_helpers() {
        let model = FakeVideo;

        assert_eq!(VideoModelV3::max_videos_per_call(&model), Some(4));
        assert_eq!(VideoModelV3::supported_models(&model), vec!["fake-video"]);
        assert_eq!(
            VideoModelV3::supported_resolutions(&model, "fake-video"),
            vec!["720p"]
        );
        assert_eq!(
            VideoModelV3::supported_durations(&model, "fake-video"),
            vec![6]
        );
    }
}
