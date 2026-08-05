//! MiniMax video generation helpers (extension API).
//!
//! MiniMax video generation uses a task-based API. This module provides a thin,
//! type-safe builder around `VideoGenerationRequest` with MiniMax-flavored knobs.

use crate::provider_options::MinimaxVideoOptions;
use crate::types::video::VideoGenerationRequest;

use super::video_options::MinimaxVideoRequestExt;

/// Type-safe builder for MiniMax video generation requests.
#[derive(Debug, Clone)]
pub struct MinimaxVideoRequestBuilder {
    request: VideoGenerationRequest,
}

impl MinimaxVideoRequestBuilder {
    /// Create a request builder with required fields.
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            request: VideoGenerationRequest::new(model, prompt),
        }
    }

    /// Set duration in seconds (MiniMax models typically support 6 or 10 seconds).
    pub fn duration(mut self, seconds: u32) -> Self {
        self.request = self.request.with_duration(seconds);
        self
    }

    /// Set resolution (e.g. "768P", "1080P").
    pub fn resolution(mut self, resolution: impl Into<String>) -> Self {
        self.request = self.request.with_resolution(resolution);
        self
    }

    /// Attach provider-owned MiniMax video options directly.
    pub fn options(mut self, options: MinimaxVideoOptions) -> Self {
        self.request = self.request.with_minimax_video_options(options);
        self
    }

    /// Enable prompt optimization (MiniMax-specific).
    pub fn prompt_optimizer(mut self, enabled: bool) -> Self {
        self.request = self
            .request
            .with_minimax_video_options(MinimaxVideoOptions::new().with_prompt_optimizer(enabled));
        self
    }

    /// Enable fast pretreatment for prompt optimization (MiniMax-specific).
    pub fn fast_pretreatment(mut self, enabled: bool) -> Self {
        self.request = self
            .request
            .with_minimax_video_options(MinimaxVideoOptions::new().with_fast_pretreatment(enabled));
        self
    }

    /// Set callback URL for task status updates.
    pub fn callback_url(mut self, url: impl Into<String>) -> Self {
        self.request = self
            .request
            .with_minimax_video_options(MinimaxVideoOptions::new().with_callback_url(url));
        self
    }

    /// Enable watermark (MiniMax-specific).
    pub fn watermark(mut self, enabled: bool) -> Self {
        self.request = self
            .request
            .with_minimax_video_options(MinimaxVideoOptions::new().with_watermark(enabled));
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

#[cfg(test)]
mod tests {
    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn minimax_video_builder_source_does_not_read_response_metadata() {
        let source = include_str!("video.rs");
        let request_source = source_section(
            source,
            "pub struct MinimaxVideoRequestBuilder",
            "#[cfg(test)]",
        );

        for disallowed in ["provider_metadata", "ProviderMetadata", "ContentPart::"] {
            assert!(
                !request_source.contains(disallowed),
                "MiniMax video request builder must stay request-only"
            );
        }
    }
}
