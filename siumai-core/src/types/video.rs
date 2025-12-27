//! Video Generation Types
//!
//! Type definitions for video generation requests and responses.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Video generation request
///
/// This type is designed to be extensible across different video generation providers.
/// Provider-specific fields (like `prompt_optimizer`, `aigc_watermark`) are optional,
/// and additional parameters can be passed via `extra_params`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoGenerationRequest {
    /// Model name (e.g., "hailuo-2.3", "gen-3-alpha", "sora-2")
    pub model: String,

    /// Text description for video generation
    pub prompt: String,

    /// Video duration in seconds
    ///
    /// Different providers support different durations:
    /// - MiniMaxi Hailuo: 6 or 10 seconds
    /// - Runway Gen-3: 5 or 10 seconds
    /// - Sora: up to 60 seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<u32>,

    /// Video resolution
    ///
    /// Format varies by provider:
    /// - MiniMaxi: "720P", "768P", "1080P"
    /// - Runway: "1280x768", "768x1280", "1280x1280"
    /// - Sora: "1920x1080", "1080x1920", etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,

    /// Seed image for image-to-video generation
    ///
    /// Some providers support generating video from a starting image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed_image: Option<Vec<u8>>,

    /// Seed video for video-to-video transformation
    ///
    /// Some providers support style transfer or video editing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed_video: Option<Vec<u8>>,

    /// Whether to automatically optimize prompt (MiniMaxi-specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_optimizer: Option<bool>,

    /// Whether to shorten prompt optimization time (MiniMaxi-specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fast_pretreatment: Option<bool>,

    /// Callback URL for task status updates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_url: Option<String>,

    /// Whether to add watermark to generated video (MiniMaxi-specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aigc_watermark: Option<bool>,

    /// Additional provider-specific parameters
    ///
    /// Use this for provider-specific features not covered by standard fields.
    /// Examples:
    /// - Runway: `{"watermark": false, "duration": "gen3a_10s"}`
    /// - Sora: `{"aspect_ratio": "16:9", "quality": "high"}`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_params: Option<HashMap<String, serde_json::Value>>,
}

impl VideoGenerationRequest {
    /// Create a new video generation request with minimal required fields
    ///
    /// # Arguments
    ///
    /// * `model` - Model name to use for generation
    /// * `prompt` - Text description of the desired video
    ///
    /// # Example
    ///
    /// ```ignore
    /// let request = VideoGenerationRequest::new("hailuo-2.3", "A cat playing piano");
    /// ```
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            duration: None,
            resolution: None,
            seed_image: None,
            seed_video: None,
            prompt_optimizer: None,
            fast_pretreatment: None,
            callback_url: None,
            aigc_watermark: None,
            extra_params: None,
        }
    }

    /// Set video duration
    pub fn with_duration(mut self, duration: u32) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set video resolution
    pub fn with_resolution(mut self, resolution: impl Into<String>) -> Self {
        self.resolution = Some(resolution.into());
        self
    }

    /// Set prompt optimizer
    pub fn with_prompt_optimizer(mut self, enabled: bool) -> Self {
        self.prompt_optimizer = Some(enabled);
        self
    }

    /// Set fast pretreatment
    pub fn with_fast_pretreatment(mut self, enabled: bool) -> Self {
        self.fast_pretreatment = Some(enabled);
        self
    }

    /// Set callback URL
    pub fn with_callback_url(mut self, url: impl Into<String>) -> Self {
        self.callback_url = Some(url.into());
        self
    }

    /// Set watermark
    pub fn with_watermark(mut self, enabled: bool) -> Self {
        self.aigc_watermark = Some(enabled);
        self
    }

    /// Set seed image for image-to-video generation
    pub fn with_seed_image(mut self, image: Vec<u8>) -> Self {
        self.seed_image = Some(image);
        self
    }

    /// Set seed video for video-to-video transformation
    pub fn with_seed_video(mut self, video: Vec<u8>) -> Self {
        self.seed_video = Some(video);
        self
    }

    /// Add a provider-specific parameter
    pub fn with_extra_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value);
        self
    }
}

/// Video generation response (task creation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoGenerationResponse {
    /// Task ID for querying status
    pub task_id: String,

    /// Base response with status information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_resp: Option<BaseResponse>,
}

/// Base response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseResponse {
    /// Status code (0 for success)
    pub status_code: i32,

    /// Status message
    pub status_msg: String,
}

/// Video generation task status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VideoTaskStatus {
    /// Preparing
    Preparing,

    /// Queueing
    Queueing,

    /// Processing
    Processing,

    /// Success
    Success,

    /// Failed
    Fail,
}

impl std::fmt::Display for VideoTaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VideoTaskStatus::Preparing => write!(f, "Preparing"),
            VideoTaskStatus::Queueing => write!(f, "Queueing"),
            VideoTaskStatus::Processing => write!(f, "Processing"),
            VideoTaskStatus::Success => write!(f, "Success"),
            VideoTaskStatus::Fail => write!(f, "Fail"),
        }
    }
}

/// Video task status query response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoTaskStatusResponse {
    /// Task ID
    pub task_id: String,

    /// Current task status
    pub status: VideoTaskStatus,

    /// File ID (available when status is Success)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,

    /// Video width in pixels (available when status is Success)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_width: Option<u32>,

    /// Video height in pixels (available when status is Success)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_height: Option<u32>,

    /// Base response with status information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_resp: Option<BaseResponse>,
}

impl VideoTaskStatusResponse {
    /// Check if the task is complete (either Success or Fail)
    pub fn is_complete(&self) -> bool {
        matches!(
            self.status,
            VideoTaskStatus::Success | VideoTaskStatus::Fail
        )
    }

    /// Check if the task succeeded
    pub fn is_success(&self) -> bool {
        self.status == VideoTaskStatus::Success
    }

    /// Check if the task failed
    pub fn is_failed(&self) -> bool {
        self.status == VideoTaskStatus::Fail
    }

    /// Check if the task is still in progress
    pub fn is_in_progress(&self) -> bool {
        matches!(
            self.status,
            VideoTaskStatus::Preparing | VideoTaskStatus::Queueing | VideoTaskStatus::Processing
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_generation_request_builder() {
        let req = VideoGenerationRequest::new("hailuo-2.3", "A beautiful sunset")
            .with_duration(6)
            .with_resolution("1080P")
            .with_prompt_optimizer(true);

        assert_eq!(req.model, "hailuo-2.3");
        assert_eq!(req.prompt, "A beautiful sunset");
        assert_eq!(req.duration, Some(6));
        assert_eq!(req.resolution, Some("1080P".to_string()));
        assert_eq!(req.prompt_optimizer, Some(true));
    }

    #[test]
    fn test_task_status_checks() {
        let mut response = VideoTaskStatusResponse {
            task_id: "123".to_string(),
            status: VideoTaskStatus::Processing,
            file_id: None,
            video_width: None,
            video_height: None,
            base_resp: None,
        };

        assert!(response.is_in_progress());
        assert!(!response.is_complete());

        response.status = VideoTaskStatus::Success;
        assert!(response.is_complete());
        assert!(response.is_success());
        assert!(!response.is_failed());

        response.status = VideoTaskStatus::Fail;
        assert!(response.is_complete());
        assert!(response.is_failed());
        assert!(!response.is_success());
    }
}
