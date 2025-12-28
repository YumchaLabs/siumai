//! MiniMaxi video generation helpers (extension API).
//!
//! MiniMaxi video generation uses a task-based API. This module provides a thin,
//! type-safe builder around `VideoGenerationRequest` with MiniMaxi-flavored knobs.

use crate::types::video::VideoGenerationRequest;

/// Type-safe builder for MiniMaxi video generation requests.
#[derive(Debug, Clone)]
pub struct MinimaxiVideoRequestBuilder {
    request: VideoGenerationRequest,
}

impl MinimaxiVideoRequestBuilder {
    /// Create a request builder with required fields.
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            request: VideoGenerationRequest::new(model, prompt),
        }
    }

    /// Set duration in seconds (MiniMaxi models typically support 6 or 10 seconds).
    pub fn duration(mut self, seconds: u32) -> Self {
        self.request = self.request.with_duration(seconds);
        self
    }

    /// Set resolution (e.g. "768P", "1080P").
    pub fn resolution(mut self, resolution: impl Into<String>) -> Self {
        self.request = self.request.with_resolution(resolution);
        self
    }

    /// Enable prompt optimization (MiniMaxi-specific).
    pub fn prompt_optimizer(mut self, enabled: bool) -> Self {
        self.request = self.request.with_prompt_optimizer(enabled);
        self
    }

    /// Enable fast pretreatment for prompt optimization (MiniMaxi-specific).
    pub fn fast_pretreatment(mut self, enabled: bool) -> Self {
        self.request = self.request.with_fast_pretreatment(enabled);
        self
    }

    /// Set callback URL for task status updates.
    pub fn callback_url(mut self, url: impl Into<String>) -> Self {
        self.request = self.request.with_callback_url(url);
        self
    }

    /// Enable watermark (MiniMaxi-specific).
    pub fn watermark(mut self, enabled: bool) -> Self {
        self.request = self.request.with_watermark(enabled);
        self
    }

    /// Attach a seed image (image-to-video).
    pub fn seed_image(mut self, image: Vec<u8>) -> Self {
        self.request = self.request.with_seed_image(image);
        self
    }

    /// Attach a seed video (video-to-video).
    pub fn seed_video(mut self, video: Vec<u8>) -> Self {
        self.request = self.request.with_seed_video(video);
        self
    }

    /// Add a provider-specific parameter.
    pub fn extra_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.request = self.request.with_extra_param(key, value);
        self
    }

    /// Finish building the `VideoGenerationRequest`.
    pub fn build(self) -> VideoGenerationRequest {
        self.request
    }
}
