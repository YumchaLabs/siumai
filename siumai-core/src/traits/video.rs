//! Video Generation Capability
//!
//! Trait definition for video generation capabilities.

use crate::LlmError;
use crate::types::video::{
    VideoGenerationRequest, VideoGenerationResponse, VideoTaskStatusResponse,
};
use async_trait::async_trait;

/// Video generation capability trait
///
/// This trait defines the interface for video generation operations.
/// Video generation is typically an asynchronous task-based operation:
/// 1. Submit a video generation task
/// 2. Poll the task status until completion
/// 3. Retrieve the generated video file
#[async_trait]
pub trait VideoGenerationCapability: Send + Sync {
    /// Create a video generation task
    ///
    /// # Arguments
    ///
    /// * `request` - Video generation request with model, prompt, and options
    ///
    /// # Returns
    ///
    /// Returns a `VideoGenerationResponse` containing the task ID for status polling.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use siumai::types::video::VideoGenerationRequest;
    ///
    /// let request = VideoGenerationRequest::new("hailuo-2.3", "A beautiful sunset over the ocean")
    ///     .with_duration(6)
    ///     .with_resolution("1080P");
    ///
    /// let response = client.create_video_task(request).await?;
    /// println!("Task ID: {}", response.task_id);
    /// ```
    async fn create_video_task(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResponse, LlmError>;

    /// Query the status of a video generation task
    ///
    /// # Arguments
    ///
    /// * `task_id` - The task ID returned from `create_video_task`
    ///
    /// # Returns
    ///
    /// Returns a `VideoTaskStatusResponse` with the current task status and video information
    /// (if the task is complete).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let status = client.query_video_task("task_123").await?;
    ///
    /// match status.status {
    ///     VideoTaskStatus::Success => {
    ///         println!("Video ready! File ID: {:?}", status.file_id);
    ///     }
    ///     VideoTaskStatus::Processing => {
    ///         println!("Still processing...");
    ///     }
    ///     VideoTaskStatus::Fail => {
    ///         println!("Task failed");
    ///     }
    ///     _ => {}
    /// }
    /// ```
    async fn query_video_task(&self, task_id: &str) -> Result<VideoTaskStatusResponse, LlmError>;

    /// Get supported video generation models
    ///
    /// # Returns
    ///
    /// Returns a list of supported model names.
    fn get_supported_models(&self) -> Vec<String>;

    /// Get supported resolutions for a specific model
    ///
    /// # Arguments
    ///
    /// * `model` - The model name
    ///
    /// # Returns
    ///
    /// Returns a list of supported resolution strings (e.g., "720P", "1080P").
    fn get_supported_resolutions(&self, model: &str) -> Vec<String>;

    /// Get supported durations for a specific model
    ///
    /// # Arguments
    ///
    /// * `model` - The model name
    ///
    /// # Returns
    ///
    /// Returns a list of supported video durations in seconds.
    fn get_supported_durations(&self, model: &str) -> Vec<u32>;
}
