//! Video Generation Types
//!
//! Type definitions for video generation requests and responses.

use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{
    DataContent, HttpConfig, HttpResponseInfo, InvalidDataContentError, ProviderOptionsMap,
    ProviderReference, Warning,
};

/// Video generation file payload aligned with AI SDK file inputs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum VideoGenerationFileData {
    /// Base64-encoded file data.
    Base64(String),
    /// Binary file data.
    Binary(Vec<u8>),
}

impl VideoGenerationFileData {
    /// Create file data from binary bytes.
    pub fn binary(data: impl Into<Vec<u8>>) -> Self {
        Self::Binary(data.into())
    }

    /// Create file data from a base64 string.
    pub fn base64(data: impl Into<String>) -> Self {
        Self::Base64(data.into())
    }

    /// Convert the file data to a base64 string.
    pub fn as_base64(&self) -> String {
        match self {
            Self::Base64(data) => data.clone(),
            Self::Binary(data) => base64::engine::general_purpose::STANDARD.encode(data),
        }
    }

    /// Convert the file data to bytes, decoding base64 when necessary.
    pub fn as_bytes(&self) -> Result<Vec<u8>, InvalidDataContentError> {
        match self {
            Self::Base64(data) => base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|source| InvalidDataContentError::invalid_base64(data.clone(), source)),
            Self::Binary(data) => Ok(data.clone()),
        }
    }
}

impl From<DataContent> for VideoGenerationFileData {
    fn from(value: DataContent) -> Self {
        match value {
            DataContent::Base64(data) => Self::Base64(data),
            DataContent::Binary(data) => Self::Binary(data),
        }
    }
}

impl From<VideoGenerationFileData> for DataContent {
    fn from(value: VideoGenerationFileData) -> Self {
        match value {
            VideoGenerationFileData::Base64(data) => Self::Base64(data),
            VideoGenerationFileData::Binary(data) => Self::Binary(data),
        }
    }
}

/// Video/image input for video generation or editing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VideoGenerationInput {
    /// File-like input backed by binary or base64 data.
    File {
        /// File payload.
        data: VideoGenerationFileData,
        /// Optional media type such as `image/png` or `video/mp4`.
        #[serde(
            rename = "mediaType",
            alias = "media_type",
            skip_serializing_if = "Option::is_none"
        )]
        media_type: Option<String>,
        /// Optional provider-specific metadata for this file input.
        #[serde(
            default,
            rename = "providerOptions",
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options_map: ProviderOptionsMap,
    },
    /// URL-like input.
    Url {
        /// URL or data URL.
        url: String,
        /// Optional provider-specific metadata for this URL input.
        #[serde(
            default,
            rename = "providerOptions",
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options_map: ProviderOptionsMap,
    },
}

impl VideoGenerationInput {
    /// Create a file input from binary bytes.
    pub fn file(data: impl Into<Vec<u8>>) -> Self {
        Self::File {
            data: VideoGenerationFileData::binary(data),
            media_type: None,
            provider_options_map: ProviderOptionsMap::default(),
        }
    }

    /// Create a file input from binary bytes and media type.
    pub fn file_with_media_type(data: impl Into<Vec<u8>>, media_type: impl Into<String>) -> Self {
        Self::file(data).with_media_type(media_type)
    }

    /// Create a file input from a base64 string.
    pub fn base64(data: impl Into<String>) -> Self {
        Self::File {
            data: VideoGenerationFileData::base64(data),
            media_type: None,
            provider_options_map: ProviderOptionsMap::default(),
        }
    }

    /// Create a file input from a base64 string and media type.
    pub fn base64_with_media_type(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::base64(data).with_media_type(media_type)
    }

    /// Create a file input from shared AI SDK-style data content.
    pub fn from_data_content(data: DataContent) -> Self {
        Self::File {
            data: VideoGenerationFileData::from(data),
            media_type: None,
            provider_options_map: ProviderOptionsMap::default(),
        }
    }

    /// Create a URL input.
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url {
            url: url.into(),
            provider_options_map: ProviderOptionsMap::default(),
        }
    }

    /// Set the media type for file inputs.
    pub fn with_media_type(mut self, media_type: impl Into<String>) -> Self {
        if let Self::File {
            media_type: current,
            ..
        } = &mut self
        {
            *current = Some(media_type.into());
        }
        self
    }

    /// Replace the full provider options map (open JSON map).
    pub fn with_provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        match &mut self {
            Self::File {
                provider_options_map,
                ..
            }
            | Self::Url {
                provider_options_map,
                ..
            } => {
                *provider_options_map = map;
            }
        }
        self
    }

    /// Set provider options for a provider id (open JSON map).
    pub fn with_provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        options: serde_json::Value,
    ) -> Self {
        match &mut self {
            Self::File {
                provider_options_map,
                ..
            }
            | Self::Url {
                provider_options_map,
                ..
            } => {
                provider_options_map.insert(provider_id, options);
            }
        }
        self
    }

    /// Return the media type if available.
    pub fn media_type(&self) -> Option<&str> {
        match self {
            Self::File { media_type, .. } => media_type.as_deref(),
            Self::Url { .. } => None,
        }
    }

    /// Return the URL when this input is URL-backed.
    pub fn as_url(&self) -> Option<&str> {
        match self {
            Self::Url { url, .. } => Some(url.as_str()),
            Self::File { .. } => None,
        }
    }

    /// Return the provider options map attached to this input.
    pub fn provider_options_map(&self) -> &ProviderOptionsMap {
        match self {
            Self::File {
                provider_options_map,
                ..
            }
            | Self::Url {
                provider_options_map,
                ..
            } => provider_options_map,
        }
    }

    /// Return the file payload when this input is file-backed.
    pub fn file_data(&self) -> Option<&VideoGenerationFileData> {
        match self {
            Self::File { data, .. } => Some(data),
            Self::Url { .. } => None,
        }
    }

    /// Check whether the input is file-backed.
    pub fn is_file(&self) -> bool {
        matches!(self, Self::File { .. })
    }

    /// Check whether the input is URL-backed.
    pub fn is_url(&self) -> bool {
        matches!(self, Self::Url { .. })
    }
}

/// High-level video prompt shape aligned with AI SDK `GenerateVideoPrompt`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum VideoGenerationPrompt {
    /// Text-only video prompt.
    Text(String),
    /// Image-to-video prompt with optional text.
    Image {
        /// Optional text description for the generated video.
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        /// Input image for image-to-video generation.
        image: VideoGenerationInput,
    },
}

/// AI SDK-auditable alias for the stable Rust video prompt shape.
pub type GenerateVideoPrompt = VideoGenerationPrompt;

impl VideoGenerationPrompt {
    /// Create a text-only prompt.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Create an image-only prompt.
    pub fn image(image: VideoGenerationInput) -> Self {
        Self::Image { text: None, image }
    }

    /// Create an image prompt with an accompanying text description.
    pub fn image_with_text(image: VideoGenerationInput, text: impl Into<String>) -> Self {
        Self::Image {
            text: Some(text.into()),
            image,
        }
    }

    /// Return the text portion of the prompt when present.
    pub fn text_part(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text.as_str()),
            Self::Image { text, .. } => text.as_deref(),
        }
    }

    /// Return the image portion of the prompt when present.
    pub fn image_part(&self) -> Option<&VideoGenerationInput> {
        match self {
            Self::Text(_) => None,
            Self::Image { image, .. } => Some(image),
        }
    }
}

impl From<String> for VideoGenerationPrompt {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for VideoGenerationPrompt {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<VideoGenerationInput> for VideoGenerationPrompt {
    fn from(value: VideoGenerationInput) -> Self {
        Self::image(value)
    }
}

/// Video generation request
///
/// This type is designed to be extensible across different video generation providers.
/// Provider-owned knobs should flow through `provider_options_map`, while generic overflow fields
/// can still be carried through `extra_params`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoGenerationRequest {
    /// Model name (e.g., "hailuo-2.3", "gen-3-alpha", "sora-2")
    pub model: String,

    /// Optional text description for video generation.
    ///
    /// This is optional to match AI SDK `GenerateVideoPrompt`, where image-to-video requests may
    /// provide only an input image and omit the textual prompt entirely.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,

    /// Number of videos to generate (AI SDK `n`).
    #[serde(skip_serializing_if = "Option::is_none", alias = "n")]
    pub count: Option<u32>,

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

    /// Video aspect ratio (e.g. "16:9", "9:16")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,

    /// Frames per second.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,

    /// Random seed for reproducibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// Seed image for image-to-video generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<VideoGenerationInput>,

    /// Provider-conditional input video for video editing or video-to-video flows.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video: Option<VideoGenerationInput>,

    /// Additional provider-specific parameters
    ///
    /// Use this for provider-specific features not covered by standard fields.
    /// Examples:
    /// - Runway: `{"watermark": false, "duration": "gen3a_10s"}`
    /// - Sora: `{"aspect_ratio": "16:9", "quality": "high"}`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_params: Option<HashMap<String, serde_json::Value>>,

    /// Open provider options map (Vercel-aligned).
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_options_map: ProviderOptionsMap,

    /// Optional per-request HTTP configuration (headers, timeout, etc.).
    #[serde(skip)]
    pub http_config: Option<HttpConfig>,
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
        Self::new_without_prompt(model).with_prompt(prompt)
    }

    /// Create a request from a high-level AI SDK-style video prompt.
    pub fn from_generate_prompt(
        model: impl Into<String>,
        prompt: impl Into<VideoGenerationPrompt>,
    ) -> Self {
        Self::new_without_prompt(model).with_generate_prompt(prompt)
    }

    /// Create a request without a text prompt.
    ///
    /// This is primarily intended for image-to-video providers that support prompt-less requests.
    pub fn new_without_prompt(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: None,
            count: None,
            duration: None,
            resolution: None,
            aspect_ratio: None,
            fps: None,
            seed: None,
            image: None,
            video: None,
            extra_params: None,
            provider_options_map: ProviderOptionsMap::default(),
            http_config: None,
        }
    }

    /// Set or replace the text prompt.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Apply a high-level AI SDK-style video prompt onto the request.
    pub fn with_generate_prompt(mut self, prompt: impl Into<VideoGenerationPrompt>) -> Self {
        match prompt.into() {
            VideoGenerationPrompt::Text(text) => {
                self.prompt = Some(text);
            }
            VideoGenerationPrompt::Image { text, image } => {
                self.prompt = text;
                self.image = Some(image);
            }
        }
        self
    }

    /// Set video duration
    pub fn with_duration(mut self, duration: u32) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set the number of videos to generate (AI SDK `n`).
    pub fn with_count(mut self, count: u32) -> Self {
        self.count = Some(count);
        self
    }

    /// Alias for `with_count`, matching AI SDK naming.
    pub fn with_n(self, count: u32) -> Self {
        self.with_count(count)
    }

    /// Set video resolution
    pub fn with_resolution(mut self, resolution: impl Into<String>) -> Self {
        self.resolution = Some(resolution.into());
        self
    }

    /// Set video aspect ratio
    pub fn with_aspect_ratio(mut self, aspect_ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(aspect_ratio.into());
        self
    }

    /// Set frames per second.
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.fps = Some(fps);
        self
    }

    /// Set random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Attach an image input for image-to-video generation.
    pub fn with_image(mut self, image: VideoGenerationInput) -> Self {
        self.image = Some(image);
        self
    }

    /// Attach a video input for provider-conditional video editing flows.
    pub fn with_video(mut self, video: VideoGenerationInput) -> Self {
        self.video = Some(video);
        self
    }

    /// Add a provider-specific parameter
    pub fn with_extra_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value);
        self
    }

    /// Replace the full provider options map (open JSON map).
    pub fn with_provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        self.provider_options_map = map;
        self
    }

    /// Set provider options for a provider id (open JSON map).
    pub fn with_provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        options: serde_json::Value,
    ) -> Self {
        self.provider_options_map.insert(provider_id, options);
        self
    }

    /// Set per-request HTTP config (headers, proxy, timeout, etc.).
    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = Some(config);
        self
    }

    /// Add a custom header for this request.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let mut config = self.http_config.take().unwrap_or_else(HttpConfig::empty);
        config.headers.insert(key.into(), value.into());
        self.http_config = Some(config);
        self
    }
}

/// Video generation response (task creation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoGenerationResponse {
    /// Task ID for querying status
    #[serde(rename = "taskId", alias = "task_id")]
    pub task_id: String,

    /// Base response with status information
    #[serde(
        rename = "baseResp",
        alias = "base_resp",
        skip_serializing_if = "Option::is_none"
    )]
    pub base_resp: Option<BaseResponse>,

    /// Additional provider-specific metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,

    /// Warnings surfaced while creating the task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<Warning>>,

    /// HTTP response envelope (timestamp, model id, headers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<HttpResponseInfo>,
}

/// Base response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseResponse {
    /// Status code (0 for success)
    pub status_code: i32,

    /// Status message
    pub status_msg: String,
}

/// Provider-owned materialized generated-video asset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MaterializedVideoAsset {
    /// Raw video bytes.
    pub bytes: Vec<u8>,

    /// Optional IANA media type associated with the materialized bytes.
    #[serde(
        rename = "mediaType",
        alias = "media_type",
        skip_serializing_if = "Option::is_none"
    )]
    pub media_type: Option<String>,
}

impl MaterializedVideoAsset {
    /// Create a materialized video asset from bytes.
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            bytes: bytes.into(),
            media_type: None,
        }
    }

    /// Attach a media type to the materialized asset.
    pub fn with_media_type(mut self, media_type: impl Into<String>) -> Self {
        self.media_type = Some(media_type.into());
        self
    }
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
    #[serde(rename = "taskId", alias = "task_id")]
    pub task_id: String,

    /// Current task status
    pub status: VideoTaskStatus,

    /// File ID (available when status is Success)
    #[serde(
        rename = "fileId",
        alias = "file_id",
        skip_serializing_if = "Option::is_none"
    )]
    pub file_id: Option<String>,

    /// Video URL (available when the provider exposes a directly downloadable asset URL).
    #[serde(
        rename = "videoUrl",
        alias = "video_url",
        skip_serializing_if = "Option::is_none"
    )]
    pub video_url: Option<String>,

    /// Canonical provider-owned final video reference.
    ///
    /// This is the preferred stable carrier for provider-managed assets. `file_id` remains as a
    /// legacy compatibility field for providers that still expose only a single opaque id string.
    #[serde(
        rename = "providerReference",
        alias = "provider_reference",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_reference: Option<ProviderReference>,

    /// Video duration in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f32>,

    /// Video width in pixels (available when status is Success)
    #[serde(
        rename = "videoWidth",
        alias = "video_width",
        skip_serializing_if = "Option::is_none"
    )]
    pub video_width: Option<u32>,

    /// Video height in pixels (available when status is Success)
    #[serde(
        rename = "videoHeight",
        alias = "video_height",
        skip_serializing_if = "Option::is_none"
    )]
    pub video_height: Option<u32>,

    /// Base response with status information
    #[serde(
        rename = "baseResp",
        alias = "base_resp",
        skip_serializing_if = "Option::is_none"
    )]
    pub base_resp: Option<BaseResponse>,

    /// Additional provider-specific metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,

    /// HTTP response envelope (timestamp, model id, headers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<HttpResponseInfo>,
}

impl VideoTaskStatusResponse {
    /// Return the canonical provider reference when it is already present.
    pub fn provider_reference(&self) -> Option<&ProviderReference> {
        self.provider_reference.as_ref()
    }

    /// Resolve the effective provider-owned reference, falling back to the legacy `file_id`.
    pub fn effective_provider_reference(
        &self,
        provider_id: impl AsRef<str>,
    ) -> Option<ProviderReference> {
        self.provider_reference.clone().or_else(|| {
            self.file_id
                .as_ref()
                .map(|file_id| ProviderReference::single(provider_id.as_ref(), file_id))
        })
    }

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
    fn test_video_generation_prompt_helpers() {
        let prompt = VideoGenerationPrompt::image_with_text(
            VideoGenerationInput::url("https://example.com/start.png"),
            "animate this scene",
        );

        assert_eq!(prompt.text_part(), Some("animate this scene"));
        assert_eq!(
            prompt.image_part().and_then(VideoGenerationInput::as_url),
            Some("https://example.com/start.png")
        );

        let text_prompt = VideoGenerationPrompt::from("a cat dancing");
        assert_eq!(text_prompt.text_part(), Some("a cat dancing"));
        assert!(text_prompt.image_part().is_none());
    }

    #[test]
    fn test_video_generation_request_builder() {
        let req = VideoGenerationRequest::new("hailuo-2.3", "A beautiful sunset")
            .with_count(2)
            .with_duration(6)
            .with_fps(24)
            .with_seed(7)
            .with_resolution("1080P")
            .with_image(VideoGenerationInput::url("https://example.com/start.png"));

        assert_eq!(req.model, "hailuo-2.3");
        assert_eq!(req.prompt.as_deref(), Some("A beautiful sunset"));
        assert_eq!(req.count, Some(2));
        assert_eq!(req.duration, Some(6));
        assert_eq!(req.fps, Some(24));
        assert_eq!(req.seed, Some(7));
        assert_eq!(req.resolution, Some("1080P".to_string()));
        assert_eq!(req.aspect_ratio, None);
        assert_eq!(
            req.image.as_ref().and_then(VideoGenerationInput::as_url),
            Some("https://example.com/start.png")
        );
    }

    #[test]
    fn test_video_generation_request_can_omit_prompt() {
        let req = VideoGenerationRequest::new_without_prompt("veo-3.1-generate-preview")
            .with_image(VideoGenerationInput::url("https://example.com/start.png"));

        assert_eq!(req.model, "veo-3.1-generate-preview");
        assert!(req.prompt.is_none());
        assert_eq!(
            req.image.as_ref().and_then(VideoGenerationInput::as_url),
            Some("https://example.com/start.png")
        );
    }

    #[test]
    fn test_video_generation_request_from_generate_prompt() {
        let req = VideoGenerationRequest::from_generate_prompt(
            "veo-3.1-generate-preview",
            VideoGenerationPrompt::image(VideoGenerationInput::url(
                "https://example.com/start.png",
            )),
        )
        .with_n(2);

        assert_eq!(req.model, "veo-3.1-generate-preview");
        assert!(req.prompt.is_none());
        assert_eq!(req.count, Some(2));
        assert_eq!(
            req.image.as_ref().and_then(VideoGenerationInput::as_url),
            Some("https://example.com/start.png")
        );
    }

    #[test]
    fn test_video_generation_input_helpers() {
        let file = VideoGenerationInput::file(vec![1, 2, 3])
            .with_media_type("image/png")
            .with_provider_option("openaiCompatible", serde_json::json!({ "detail": "high" }));
        assert!(file.is_file());
        assert!(!file.is_url());
        assert_eq!(file.media_type(), Some("image/png"));
        assert_eq!(
            file.file_data()
                .expect("file data")
                .as_bytes()
                .expect("bytes"),
            vec![1, 2, 3]
        );
        assert_eq!(
            file.provider_options_map()
                .get("openaicompatible")
                .and_then(|value| value.get("detail"))
                .and_then(|value| value.as_str()),
            Some("high")
        );

        let url = VideoGenerationInput::url("https://example.com/video.mp4")
            .with_provider_option("xai", serde_json::json!({ "mode": "fast" }));
        assert!(url.is_url());
        assert!(!url.is_file());
        assert_eq!(url.as_url(), Some("https://example.com/video.mp4"));
        assert!(url.file_data().is_none());
        assert_eq!(
            url.provider_options_map()
                .get("xai")
                .and_then(|value| value.get("mode"))
                .and_then(|value| value.as_str()),
            Some("fast")
        );
    }

    #[test]
    fn test_video_generation_input_provider_options_serde_roundtrip() {
        let value = serde_json::to_value(
            VideoGenerationInput::file(vec![1, 2, 3])
                .with_provider_option("openaiCompatible", serde_json::json!({ "detail": "low" })),
        )
        .expect("serialize video input");

        assert_eq!(
            value.get("type").and_then(|value| value.as_str()),
            Some("file")
        );
        assert!(value.get("providerOptions").is_some());
        assert!(
            value
                .get("providerOptions")
                .and_then(|value| value.get("openaiCompatible"))
                .is_some()
        );
        assert!(
            value
                .get("providerOptions")
                .and_then(|value| value.get("openaicompatible"))
                .is_none()
        );

        let input: VideoGenerationInput = serde_json::from_value(serde_json::json!({
            "type": "url",
            "url": "https://example.com/video.mp4",
            "providerOptions": {
                "XAI": {
                    "mode": "fast"
                }
            }
        }))
        .expect("deserialize video input");

        assert_eq!(input.as_url(), Some("https://example.com/video.mp4"));
        assert_eq!(
            input
                .provider_options_map()
                .get("xai")
                .and_then(|value| value.get("mode"))
                .and_then(|value| value.as_str()),
            Some("fast")
        );
    }

    #[test]
    fn test_video_generation_input_accepts_shared_data_content() {
        let input = VideoGenerationInput::from_data_content(DataContent::binary(vec![4, 5, 6]))
            .with_media_type("image/png");

        assert_eq!(input.media_type(), Some("image/png"));
        assert_eq!(
            input
                .file_data()
                .expect("file data")
                .as_bytes()
                .expect("bytes"),
            vec![4, 5, 6]
        );
    }

    #[test]
    fn test_task_status_checks() {
        let mut response = VideoTaskStatusResponse {
            task_id: "123".to_string(),
            status: VideoTaskStatus::Processing,
            file_id: None,
            video_url: None,
            provider_reference: None,
            duration: None,
            video_width: None,
            video_height: None,
            base_resp: None,
            metadata: HashMap::new(),
            response: None,
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

    #[test]
    fn test_video_task_status_response_serde_prefers_provider_reference_and_camel_case() {
        let response = VideoTaskStatusResponse {
            task_id: "task-123".to_string(),
            status: VideoTaskStatus::Success,
            file_id: Some("file-legacy".to_string()),
            video_url: Some("https://example.com/video.mp4".to_string()),
            provider_reference: Some(ProviderReference::from([
                ("gemini", "files/123"),
                ("google", "files/123"),
            ])),
            duration: Some(4.0),
            video_width: Some(1280),
            video_height: Some(720),
            base_resp: Some(BaseResponse {
                status_code: 0,
                status_msg: "OK".to_string(),
            }),
            metadata: HashMap::new(),
            response: None,
        };

        let value = serde_json::to_value(&response).expect("serialize video task response");
        assert_eq!(value["taskId"], serde_json::json!("task-123"));
        assert_eq!(value["fileId"], serde_json::json!("file-legacy"));
        assert_eq!(
            value["providerReference"]["gemini"],
            serde_json::json!("files/123")
        );
        assert_eq!(
            value["providerReference"]["google"],
            serde_json::json!("files/123")
        );
        assert_eq!(
            value["videoUrl"],
            serde_json::json!("https://example.com/video.mp4")
        );
        assert_eq!(value["videoWidth"], serde_json::json!(1280));
        assert_eq!(value["videoHeight"], serde_json::json!(720));
        assert!(value.get("task_id").is_none());
        assert!(value.get("provider_reference").is_none());

        let parsed: VideoTaskStatusResponse = serde_json::from_value(serde_json::json!({
            "task_id": "task-456",
            "status": "Success",
            "file_id": "file-snake",
            "provider_reference": {
                "minimaxi": "file-789"
            },
            "video_url": "https://example.com/alt.mp4",
            "video_width": 1920,
            "video_height": 1080
        }))
        .expect("deserialize snake_case video task response");

        assert_eq!(parsed.task_id, "task-456");
        assert_eq!(parsed.file_id.as_deref(), Some("file-snake"));
        assert_eq!(
            parsed
                .provider_reference()
                .and_then(|reference| reference.get("minimaxi")),
            Some("file-789")
        );
        assert_eq!(
            parsed.video_url.as_deref(),
            Some("https://example.com/alt.mp4")
        );
        assert_eq!(parsed.video_width, Some(1920));
        assert_eq!(parsed.video_height, Some(1080));
    }

    #[test]
    fn test_video_task_status_response_effective_provider_reference_falls_back_to_file_id() {
        let response = VideoTaskStatusResponse {
            task_id: "task-789".to_string(),
            status: VideoTaskStatus::Success,
            file_id: Some("file-legacy".to_string()),
            video_url: None,
            provider_reference: None,
            duration: None,
            video_width: None,
            video_height: None,
            base_resp: None,
            metadata: HashMap::new(),
            response: None,
        };

        let effective = response
            .effective_provider_reference("gemini")
            .expect("fallback provider reference");
        assert_eq!(effective.get("gemini"), Some("file-legacy"));
    }

    #[test]
    fn test_video_generation_request_provider_options_and_http_config_helpers() {
        let request = VideoGenerationRequest::new("grok-imagine-video", "A neon city")
            .with_aspect_ratio("16:9")
            .with_provider_option(
                "xai",
                serde_json::json!({ "videoUrl": "https://example.com/video.mp4" }),
            )
            .with_header("x-test", "1");

        assert_eq!(request.aspect_ratio.as_deref(), Some("16:9"));
        assert_eq!(
            request
                .provider_options_map
                .get("xai")
                .and_then(|value| value.get("videoUrl"))
                .and_then(|value| value.as_str()),
            Some("https://example.com/video.mp4")
        );
        assert_eq!(
            request
                .http_config
                .as_ref()
                .and_then(|config| config.headers.get("x-test"))
                .map(String::as_str),
            Some("1")
        );
    }
}
