//! Video generation model family APIs.
//!
//! This is the recommended Rust-first surface for task-oriented video generation:
//! - `create_task`
//! - `query_task`
//! - `wait_for_task` for explicit polling to completion
//! - `generate` for a high-level create-and-poll helper
//! - `generate_materialized` for a high-level helper that also downloads/materializes final assets
//!
//! Unlike the AI SDK's current auto-polling helper story, the Rust family keeps explicit
//! task submission and task-status querying as the stable contract.

use crate::retry_api::{RetryOptions, retry_with};
use base64::{Engine, engine::general_purpose::STANDARD};
use reqwest::header::CONTENT_TYPE;
use siumai_core::error::LlmError;
use siumai_core::execution::http::build_http_client_from_config;
use siumai_core::types::{HttpConfig, HttpResponseInfo, ProviderReference, Warning};
use siumai_core::utils::mime::{guess_mime_from_bytes, guess_mime_from_path_or_url};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::{Instant, sleep};

pub use siumai_core::types::{
    GenerateVideoPrompt, VideoGenerationFileData, VideoGenerationInput, VideoGenerationPrompt,
    VideoGenerationRequest, VideoGenerationResponse, VideoModelProviderMetadata,
    VideoModelResponseMetadata, VideoTaskStatus, VideoTaskStatusResponse,
};
pub use siumai_core::video::{VideoModel, VideoModelV3, VideoModelV4};

const DEFAULT_VIDEO_POLL_INTERVAL: Duration = Duration::from_secs(2);
const DEFAULT_VIDEO_POLL_TIMEOUT: Duration = Duration::from_secs(300);

/// Options for `video::create_task`.
#[derive(Debug, Clone, Default)]
pub struct CreateTaskOptions {
    /// Optional retry policy applied around the task submission call.
    pub retry: Option<RetryOptions>,
    /// Optional per-call request timeout.
    ///
    /// This is applied via `VideoGenerationRequest.http_config.timeout`.
    pub timeout: Option<Duration>,
    /// Optional per-call extra headers.
    ///
    /// These are merged into `VideoGenerationRequest.http_config.headers`.
    pub headers: HashMap<String, String>,
}

/// Options for `video::query_task`.
#[derive(Debug, Clone, Default)]
pub struct QueryTaskOptions {
    /// Optional retry policy applied around the task-status query.
    pub retry: Option<RetryOptions>,
}

/// Options for `video::wait_for_task`.
#[derive(Debug, Clone)]
pub struct WaitForTaskOptions {
    /// Optional retry policy applied around each task-status query.
    pub retry: Option<RetryOptions>,
    /// Delay between polling attempts.
    pub poll_interval: Duration,
    /// Optional maximum total polling duration.
    pub poll_timeout: Option<Duration>,
}

impl Default for WaitForTaskOptions {
    fn default() -> Self {
        Self {
            retry: None,
            poll_interval: DEFAULT_VIDEO_POLL_INTERVAL,
            poll_timeout: Some(DEFAULT_VIDEO_POLL_TIMEOUT),
        }
    }
}

/// Options for `video::generate`.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    /// Optional retry policy applied around task submission calls.
    pub create_retry: Option<RetryOptions>,
    /// Optional retry policy applied around polling queries.
    pub query_retry: Option<RetryOptions>,
    /// Maximum number of final videos to request in a single provider task call.
    ///
    /// When omitted, the helper falls back to the model/provider default if one is
    /// exposed, and finally to `1`.
    pub max_videos_per_call: Option<u32>,
    /// Optional per-task submission timeout.
    ///
    /// This is applied via `VideoGenerationRequest.http_config.timeout`.
    pub timeout: Option<Duration>,
    /// Optional per-task submission extra headers.
    ///
    /// These are merged into `VideoGenerationRequest.http_config.headers`.
    pub headers: HashMap<String, String>,
    /// Delay between polling attempts.
    pub poll_interval: Duration,
    /// Optional maximum total polling duration per task.
    pub poll_timeout: Option<Duration>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            create_retry: None,
            query_retry: None,
            max_videos_per_call: None,
            timeout: None,
            headers: HashMap::new(),
            poll_interval: DEFAULT_VIDEO_POLL_INTERVAL,
            poll_timeout: Some(DEFAULT_VIDEO_POLL_TIMEOUT),
        }
    }
}

/// Response metadata collected for one generated video task.
#[derive(Debug, Clone)]
pub struct GenerateVideoResponseMetadata {
    /// Task id returned by task submission.
    pub task_id: String,
    /// HTTP response envelope from task submission, when available.
    pub create_response: Option<HttpResponseInfo>,
    /// HTTP response envelope from the final task-status query, when available.
    pub query_response: Option<HttpResponseInfo>,
    /// Provider-specific metadata for this logical generate call.
    pub provider_metadata: GenerateVideoProviderMetadata,
}

impl GenerateVideoResponseMetadata {
    /// Best-effort AI SDK-style response metadata for the logical video call.
    ///
    /// This prefers the final task-query response when available and otherwise falls back to the
    /// initial task-creation response.
    pub fn response_metadata(&self) -> Option<VideoModelResponseMetadata> {
        self.query_response_metadata()
            .or_else(|| self.create_response_metadata())
    }

    /// Best-effort AI SDK-style metadata for the task-creation response.
    pub fn create_response_metadata(&self) -> Option<VideoModelResponseMetadata> {
        self.create_response
            .as_ref()
            .and_then(|response| self.map_video_model_response_metadata(response))
    }

    /// Best-effort AI SDK-style metadata for the final task-query response.
    pub fn query_response_metadata(&self) -> Option<VideoModelResponseMetadata> {
        self.query_response
            .as_ref()
            .and_then(|response| self.map_video_model_response_metadata(response))
    }

    fn map_video_model_response_metadata(
        &self,
        response: &HttpResponseInfo,
    ) -> Option<VideoModelResponseMetadata> {
        VideoModelResponseMetadata::try_from(response)
            .ok()
            .map(|metadata| metadata.with_provider_metadata(self.provider_metadata.clone()))
    }
}

/// Provider-id keyed metadata map used by `GenerateVideoResult`.
pub type GenerateVideoProviderMetadata = VideoModelProviderMetadata;

/// Provider-specific metadata for one generated video.
pub type GeneratedVideoMetadata = HashMap<String, serde_json::Value>;

/// Options for materializing final generated-video assets into bytes/base64 form.
#[derive(Debug, Clone, Default)]
pub struct MaterializeVideoOptions {
    /// Optional HTTP configuration used when downloading URL-backed final videos.
    ///
    /// When omitted, the helper uses the default `reqwest::Client` configuration.
    pub http_config: Option<HttpConfig>,
}

/// Final generated-video data.
#[derive(Debug, Clone, PartialEq)]
pub enum GeneratedVideoData {
    /// Provider-returned URL/URI for the generated video.
    Url { url: String },
    /// Base64-encoded video data.
    Base64 { data: String },
    /// Raw video bytes.
    Bytes { data: Vec<u8> },
    /// Provider-owned video/file reference.
    ProviderReference {
        provider_reference: ProviderReference,
    },
}

/// Final generated-video asset exposed by `video::generate`.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedVideo {
    /// Task id that produced this final video.
    pub task_id: String,
    /// Resolved media type for the final video.
    pub media_type: String,
    /// Final video data/reference.
    pub data: GeneratedVideoData,
    /// Provider-specific metadata for this video.
    pub metadata: GeneratedVideoMetadata,
}

impl GeneratedVideo {
    /// Provider-returned URL/URI when available.
    pub fn url(&self) -> Option<&str> {
        match &self.data {
            GeneratedVideoData::Url { url } => Some(url.as_str()),
            _ => None,
        }
    }

    /// Provider reference when available.
    pub fn provider_reference(&self) -> Option<&ProviderReference> {
        match &self.data {
            GeneratedVideoData::ProviderReference { provider_reference } => {
                Some(provider_reference)
            }
            _ => None,
        }
    }

    /// Materialize the generated video into a byte/base64-backed file representation.
    ///
    /// URL-backed assets are downloaded eagerly on this path. Provider references remain
    /// intentionally unsupported because the stable Rust video surface does not yet define a
    /// provider-agnostic file-download contract for them.
    pub async fn materialize(
        &self,
        options: MaterializeVideoOptions,
    ) -> Result<MaterializedVideo, LlmError> {
        match &self.data {
            GeneratedVideoData::Url { url } => {
                let (bytes, downloaded_media_type) =
                    download_generated_video_url(url, options.http_config.as_ref()).await?;

                Ok(MaterializedVideo {
                    task_id: self.task_id.clone(),
                    media_type: resolve_materialized_video_media_type(
                        self,
                        Some(&bytes),
                        downloaded_media_type,
                    ),
                    data: MaterializedVideoData::Bytes(bytes),
                    metadata: self.metadata.clone(),
                })
            }
            GeneratedVideoData::Base64 { data } => {
                let decoded = STANDARD.decode(data).map_err(|error| {
                    LlmError::InvalidInput(format!("Invalid generated video base64 data: {error}"))
                })?;

                Ok(MaterializedVideo {
                    task_id: self.task_id.clone(),
                    media_type: resolve_materialized_video_media_type(self, Some(&decoded), None),
                    data: MaterializedVideoData::Base64(data.clone()),
                    metadata: self.metadata.clone(),
                })
            }
            GeneratedVideoData::Bytes { data } => Ok(MaterializedVideo {
                task_id: self.task_id.clone(),
                media_type: resolve_materialized_video_media_type(self, Some(data), None),
                data: MaterializedVideoData::Bytes(data.clone()),
                metadata: self.metadata.clone(),
            }),
            GeneratedVideoData::ProviderReference { provider_reference } => {
                Err(LlmError::UnsupportedOperation(format!(
                    "Generated video task '{}' only exposed a provider reference ({provider_reference:?}); generic materialization is not supported for this asset yet",
                    self.task_id
                )))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum MaterializedVideoData {
    Base64(String),
    Bytes(Vec<u8>),
}

/// Materialized final generated-video file, closer in role to AI SDK `GeneratedFile`.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedVideo {
    /// Task id that produced this final video.
    pub task_id: String,
    /// Resolved IANA media type for the materialized video.
    pub media_type: String,
    data: MaterializedVideoData,
    /// Provider-specific metadata for this video.
    pub metadata: GeneratedVideoMetadata,
}

impl MaterializedVideo {
    /// Return the video as a base64-encoded string.
    pub fn base64(&self) -> String {
        match &self.data {
            MaterializedVideoData::Base64(data) => data.clone(),
            MaterializedVideoData::Bytes(data) => STANDARD.encode(data),
        }
    }

    /// Return the video as raw bytes.
    pub fn bytes(&self) -> Result<Vec<u8>, LlmError> {
        match &self.data {
            MaterializedVideoData::Bytes(data) => Ok(data.clone()),
            MaterializedVideoData::Base64(data) => STANDARD.decode(data).map_err(|error| {
                LlmError::InvalidInput(format!("Invalid generated video base64 data: {error}"))
            }),
        }
    }
}

/// Result returned by `video::generate`.
#[derive(Debug, Clone)]
pub struct GenerateVideoResult {
    /// Final generated videos/assets.
    pub videos: Vec<GeneratedVideo>,
    /// Final completed video-task responses.
    pub tasks: Vec<VideoTaskStatusResponse>,
    /// Non-fatal warnings surfaced during submission or batching.
    pub warnings: Vec<Warning>,
    /// Per-task response envelopes.
    pub responses: Vec<GenerateVideoResponseMetadata>,
    /// Aggregated provider-specific metadata keyed by provider id.
    pub provider_metadata: GenerateVideoProviderMetadata,
}

/// Result returned by `video::generate_materialized`.
#[derive(Debug, Clone)]
pub struct GenerateMaterializedVideoResult {
    /// Final materialized generated videos/files.
    pub videos: Vec<MaterializedVideo>,
    /// Final completed video-task responses.
    pub tasks: Vec<VideoTaskStatusResponse>,
    /// Non-fatal warnings surfaced during submission or batching.
    pub warnings: Vec<Warning>,
    /// Per-task response envelopes.
    pub responses: Vec<GenerateVideoResponseMetadata>,
    /// Aggregated provider-specific metadata keyed by provider id.
    pub provider_metadata: GenerateVideoProviderMetadata,
}

impl GenerateVideoResult {
    /// The first generated video, if any.
    pub fn video(&self) -> Option<&GeneratedVideo> {
        self.videos.first()
    }

    /// Best-effort AI SDK-style response metadata for each logical generate call.
    pub fn video_model_responses(&self) -> Vec<VideoModelResponseMetadata> {
        self.responses
            .iter()
            .filter_map(GenerateVideoResponseMetadata::response_metadata)
            .collect()
    }

    /// The first completed task result, if any.
    pub fn task(&self) -> Option<&VideoTaskStatusResponse> {
        self.tasks.first()
    }

    /// Materialize the first generated video when present.
    pub async fn materialize_video(
        &self,
        options: MaterializeVideoOptions,
    ) -> Result<Option<MaterializedVideo>, LlmError> {
        match self.video() {
            Some(video) => video.materialize(options).await.map(Some),
            None => Ok(None),
        }
    }

    /// Materialize all generated videos.
    pub async fn materialize_videos(
        &self,
        options: MaterializeVideoOptions,
    ) -> Result<Vec<MaterializedVideo>, LlmError> {
        let mut videos = Vec::with_capacity(self.videos.len());
        for video in &self.videos {
            videos.push(video.materialize(options.clone()).await?);
        }
        Ok(videos)
    }

    /// Consume this result and materialize all generated videos.
    pub async fn into_materialized(
        self,
        options: MaterializeVideoOptions,
    ) -> Result<GenerateMaterializedVideoResult, LlmError> {
        let mut videos = Vec::with_capacity(self.videos.len());
        for video in &self.videos {
            videos.push(video.materialize(options.clone()).await?);
        }

        Ok(GenerateMaterializedVideoResult {
            videos,
            tasks: self.tasks,
            warnings: self.warnings,
            responses: self.responses,
            provider_metadata: self.provider_metadata,
        })
    }
}

impl GenerateMaterializedVideoResult {
    /// The first materialized generated video, if any.
    pub fn video(&self) -> Option<&MaterializedVideo> {
        self.videos.first()
    }

    /// Best-effort AI SDK-style response metadata for each logical generate call.
    pub fn video_model_responses(&self) -> Vec<VideoModelResponseMetadata> {
        self.responses
            .iter()
            .filter_map(GenerateVideoResponseMetadata::response_metadata)
            .collect()
    }

    /// The first completed task result, if any.
    pub fn task(&self) -> Option<&VideoTaskStatusResponse> {
        self.tasks.first()
    }
}

fn apply_video_call_options(
    mut request: VideoGenerationRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> VideoGenerationRequest {
    if timeout.is_some() || !headers.is_empty() {
        let mut http = request.http_config.take().unwrap_or_else(HttpConfig::empty);
        if let Some(timeout) = timeout {
            http.timeout = Some(timeout);
        }
        if !headers.is_empty() {
            http.headers.extend(headers);
        }
        request.http_config = Some(http);
    }

    request
}

/// Submit a video-generation task.
pub async fn create_task<M: VideoModelV3 + ?Sized>(
    model: &M,
    request: VideoGenerationRequest,
    options: CreateTaskOptions,
) -> Result<VideoGenerationResponse, LlmError> {
    let request = apply_video_call_options(request, options.timeout, options.headers);
    if let Some(retry) = options.retry {
        retry_with(
            || {
                let request = request.clone();
                async move { model.create_task(request).await }
            },
            retry,
        )
        .await
    } else {
        model.create_task(request).await
    }
}

/// Query a video-generation task.
pub async fn query_task<M: VideoModelV3 + ?Sized>(
    model: &M,
    task_id: &str,
    options: QueryTaskOptions,
) -> Result<VideoTaskStatusResponse, LlmError> {
    if let Some(retry) = options.retry {
        retry_with(
            || {
                let task_id = task_id.to_string();
                async move { model.query_task(&task_id).await }
            },
            retry,
        )
        .await
    } else {
        model.query_task(task_id).await
    }
}

fn validate_poll_interval(poll_interval: Duration) -> Result<(), LlmError> {
    if poll_interval.is_zero() {
        return Err(LlmError::InvalidParameter(
            "video polling interval must be greater than 0".to_string(),
        ));
    }

    Ok(())
}

fn resolve_requested_video_count(request: &VideoGenerationRequest) -> Result<u32, LlmError> {
    let requested_count = request.count.unwrap_or(1);
    if requested_count == 0 {
        return Err(LlmError::InvalidParameter(
            "VideoGenerationRequest.count must be greater than 0".to_string(),
        ));
    }

    Ok(requested_count)
}

fn resolve_effective_max_videos_per_call(
    explicit: Option<u32>,
    model_default: Option<u32>,
) -> Result<u32, LlmError> {
    let limit = explicit.or(model_default).unwrap_or(1);
    if limit == 0 {
        return Err(LlmError::InvalidParameter(
            "GenerateOptions.max_videos_per_call must be greater than 0".to_string(),
        ));
    }

    Ok(limit)
}

fn split_call_video_counts(total_videos: u32, max_videos_per_call: u32) -> Vec<u32> {
    let mut remaining = total_videos;
    let mut counts = Vec::new();
    while remaining > 0 {
        let current = remaining.min(max_videos_per_call);
        counts.push(current);
        remaining -= current;
    }
    counts
}

fn split_generate_requests(
    request: VideoGenerationRequest,
    max_videos_per_call: u32,
) -> Result<Vec<VideoGenerationRequest>, LlmError> {
    let requested_count = resolve_requested_video_count(&request)?;
    let call_counts = split_call_video_counts(requested_count, max_videos_per_call);

    let mut requests = Vec::with_capacity(call_counts.len());
    for count in call_counts {
        let mut split = request.clone();
        split.count = Some(count);
        requests.push(split);
    }

    Ok(requests)
}

fn build_failed_task_error(task_id: &str, response: &VideoTaskStatusResponse) -> LlmError {
    let message = response
        .base_resp
        .as_ref()
        .map(|base| base.status_msg.clone())
        .filter(|message| !message.trim().is_empty())
        .unwrap_or_else(|| format!("Video task '{task_id}' failed"));

    LlmError::ProcessingError(format!("Video task '{task_id}' failed: {message}"))
}

fn generated_video_media_type(
    metadata: &GeneratedVideoMetadata,
    fallback_path_or_url: Option<&str>,
) -> String {
    metadata
        .get("mediaType")
        .or_else(|| metadata.get("mimeType"))
        .and_then(|value| value.as_str())
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| fallback_path_or_url.and_then(guess_mime_from_path_or_url))
        .unwrap_or_else(|| "video/mp4".to_string())
}

fn provider_declared_video_media_type(metadata: &GeneratedVideoMetadata) -> Option<String> {
    metadata
        .get("mediaType")
        .or_else(|| metadata.get("mimeType"))
        .and_then(|value| value.as_str())
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
}

fn usable_video_media_type(value: Option<&str>) -> Option<String> {
    value
        .filter(|value| !value.trim().is_empty() && *value != "application/octet-stream")
        .map(ToOwned::to_owned)
}

fn parse_video_data_url(url: &str) -> Result<(Vec<u8>, Option<String>), LlmError> {
    let Some(payload) = url.strip_prefix("data:") else {
        return Err(LlmError::InvalidParameter(
            "Expected a data URL for generated video materialization".to_string(),
        ));
    };
    let Some((meta, data)) = payload.split_once(',') else {
        return Err(LlmError::InvalidParameter(
            "Invalid generated video data URL".to_string(),
        ));
    };

    let Some(meta) = meta.strip_suffix(";base64") else {
        return Err(LlmError::InvalidParameter(
            "Generated video data URLs must use base64 encoding".to_string(),
        ));
    };

    let bytes = STANDARD.decode(data).map_err(|error| {
        LlmError::InvalidInput(format!(
            "Invalid base64 payload in generated video data URL: {error}"
        ))
    })?;
    let media_type = (!meta.is_empty()).then_some(meta.to_string());
    Ok((bytes, media_type))
}

async fn download_generated_video_url(
    url: &str,
    http_config: Option<&HttpConfig>,
) -> Result<(Vec<u8>, Option<String>), LlmError> {
    if url.starts_with("data:") {
        return parse_video_data_url(url);
    }

    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(LlmError::InvalidParameter(format!(
            "Unsupported generated video URL scheme for materialization: {url}"
        )));
    }

    let client = if let Some(http_config) = http_config {
        build_http_client_from_config(http_config)?
    } else {
        reqwest::Client::new()
    };

    let response = client.get(url).send().await.map_err(|error| {
        LlmError::HttpError(format!("Failed to download generated video: {error}"))
    })?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(LlmError::ApiError {
            code: status.as_u16(),
            message: format!("Failed to download generated video from {url}"),
            details: Some(serde_json::json!({
                "url": url,
                "body": body,
            })),
        });
    }

    let downloaded_media_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let bytes = response.bytes().await.map_err(|error| {
        LlmError::HttpError(format!(
            "Failed to read generated video download bytes: {error}"
        ))
    })?;

    Ok((bytes.to_vec(), downloaded_media_type))
}

fn resolve_materialized_video_media_type(
    video: &GeneratedVideo,
    bytes: Option<&[u8]>,
    downloaded_media_type: Option<String>,
) -> String {
    provider_declared_video_media_type(&video.metadata)
        .or_else(|| usable_video_media_type(downloaded_media_type.as_deref()))
        .or_else(|| bytes.and_then(guess_mime_from_bytes))
        .or_else(|| usable_video_media_type(Some(video.media_type.as_str())))
        .or_else(|| video.url().and_then(guess_mime_from_path_or_url))
        .unwrap_or_else(|| "video/mp4".to_string())
}

fn metadata_from_value(value: &serde_json::Value) -> GeneratedVideoMetadata {
    value
        .as_object()
        .map(|object| {
            object
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

fn metadata_value(metadata: &GeneratedVideoMetadata) -> serde_json::Value {
    serde_json::Value::Object(metadata.clone().into_iter().collect())
}

fn generated_video_from_metadata_item(
    task_id: &str,
    item: &serde_json::Value,
) -> Option<GeneratedVideo> {
    if let Some(url) = item.as_str() {
        return Some(GeneratedVideo {
            task_id: task_id.to_string(),
            media_type: guess_mime_from_path_or_url(url).unwrap_or_else(|| "video/mp4".to_string()),
            data: GeneratedVideoData::Url {
                url: url.to_string(),
            },
            metadata: GeneratedVideoMetadata::new(),
        });
    }

    let metadata = metadata_from_value(item);
    if let Some(data) = item
        .get("bytesBase64Encoded")
        .or_else(|| item.get("base64"))
        .or_else(|| {
            (item.get("type").and_then(|value| value.as_str()) == Some("base64"))
                .then(|| item.get("data"))
                .flatten()
        })
        .and_then(|value| value.as_str())
    {
        return Some(GeneratedVideo {
            task_id: task_id.to_string(),
            media_type: generated_video_media_type(&metadata, None),
            data: GeneratedVideoData::Base64 {
                data: data.to_string(),
            },
            metadata,
        });
    }

    if let Some(url) = item
        .get("url")
        .or_else(|| item.get("uri"))
        .or_else(|| item.get("gcsUri"))
        .and_then(|value| value.as_str())
    {
        return Some(GeneratedVideo {
            task_id: task_id.to_string(),
            media_type: generated_video_media_type(&metadata, Some(url)),
            data: GeneratedVideoData::Url {
                url: url.to_string(),
            },
            metadata,
        });
    }

    let bytes = item
        .get("bytes")
        .and_then(|value| value.as_array())
        .and_then(|values| {
            values
                .iter()
                .map(|value| value.as_u64().map(|value| value as u8))
                .collect::<Option<Vec<u8>>>()
        });
    if let Some(bytes) = bytes {
        return Some(GeneratedVideo {
            task_id: task_id.to_string(),
            media_type: generated_video_media_type(&metadata, None),
            data: GeneratedVideoData::Bytes { data: bytes },
            metadata,
        });
    }

    item.get("providerReference")
        .and_then(|value| serde_json::from_value::<ProviderReference>(value.clone()).ok())
        .map(|provider_reference| GeneratedVideo {
            task_id: task_id.to_string(),
            media_type: generated_video_media_type(&metadata, None),
            data: GeneratedVideoData::ProviderReference { provider_reference },
            metadata,
        })
}

fn generated_video_fallback(
    provider_id: &str,
    response: &VideoTaskStatusResponse,
) -> Option<GeneratedVideo> {
    let mut metadata = GeneratedVideoMetadata::new();
    if let Some(file_id) = response.file_id.as_ref() {
        metadata.insert("fileId".to_string(), serde_json::json!(file_id));
    }
    if let Some(video_url) = response.video_url.as_ref() {
        metadata.insert("videoUrl".to_string(), serde_json::json!(video_url));
    }
    if let Some(duration) = response.duration {
        metadata.insert("duration".to_string(), serde_json::json!(duration));
    }
    if let Some(width) = response.video_width {
        metadata.insert("width".to_string(), serde_json::json!(width));
    }
    if let Some(height) = response.video_height {
        metadata.insert("height".to_string(), serde_json::json!(height));
    }

    if let Some(video_url) = response.video_url.as_ref() {
        return Some(GeneratedVideo {
            task_id: response.task_id.clone(),
            media_type: generated_video_media_type(&metadata, Some(video_url)),
            data: GeneratedVideoData::Url {
                url: video_url.clone(),
            },
            metadata,
        });
    }

    response.file_id.as_ref().map(|file_id| GeneratedVideo {
        task_id: response.task_id.clone(),
        media_type: generated_video_media_type(&metadata, Some(file_id)),
        data: GeneratedVideoData::ProviderReference {
            provider_reference: ProviderReference::single(provider_id, file_id),
        },
        metadata,
    })
}

fn extract_generated_videos(
    provider_id: &str,
    response: &VideoTaskStatusResponse,
) -> Result<Vec<GeneratedVideo>, LlmError> {
    let metadata_videos = response
        .metadata
        .get(provider_id)
        .and_then(|value| value.get("videos"))
        .or_else(|| response.metadata.get("videos"))
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| generated_video_from_metadata_item(&response.task_id, item))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if !metadata_videos.is_empty() {
        return Ok(metadata_videos);
    }

    Ok(generated_video_fallback(provider_id, response)
        .map(|video| vec![video])
        .unwrap_or_default())
}

fn build_call_provider_metadata(
    provider_id: &str,
    task_entry: serde_json::Value,
    videos: &[GeneratedVideo],
    create_metadata: &HashMap<String, serde_json::Value>,
    query_metadata: &HashMap<String, serde_json::Value>,
) -> GenerateVideoProviderMetadata {
    let video_entries = videos
        .iter()
        .map(|video| metadata_value(&video.metadata))
        .collect::<Vec<_>>();

    let mut provider_root = create_metadata
        .get(provider_id)
        .and_then(|value| value.as_object())
        .cloned()
        .unwrap_or_default();
    if let Some(query_provider_root) = query_metadata
        .get(provider_id)
        .and_then(|value| value.as_object())
        .cloned()
    {
        merge_provider_metadata_object(&mut provider_root, query_provider_root);
    }
    provider_root.insert(
        "tasks".to_string(),
        serde_json::Value::Array(vec![task_entry]),
    );
    if !video_entries.is_empty() {
        provider_root.insert(
            "videos".to_string(),
            serde_json::Value::Array(video_entries),
        );
    }

    GenerateVideoProviderMetadata::from([(
        provider_id.to_string(),
        serde_json::Value::Object(provider_root),
    )])
}

fn merge_provider_metadata_object(
    existing: &mut serde_json::Map<String, serde_json::Value>,
    incoming: serde_json::Map<String, serde_json::Value>,
) {
    for (key, value) in incoming {
        match (existing.get_mut(&key), value) {
            (
                Some(serde_json::Value::Array(existing_items)),
                serde_json::Value::Array(mut incoming_items),
            ) if key == "videos" || key == "tasks" => {
                existing_items.append(&mut incoming_items);
            }
            (_, value) => {
                existing.insert(key, value);
            }
        }
    }
}

fn merge_provider_metadata(
    target: &mut GenerateVideoProviderMetadata,
    incoming: GenerateVideoProviderMetadata,
) {
    for (provider_id, value) in incoming {
        match (target.get_mut(&provider_id), value) {
            (
                Some(serde_json::Value::Object(existing)),
                serde_json::Value::Object(incoming_object),
            ) => merge_provider_metadata_object(existing, incoming_object),
            (_, value) => {
                target.insert(provider_id, value);
            }
        }
    }
}

/// Poll a video-generation task until it completes successfully.
///
/// Returns the final successful `VideoTaskStatusResponse`. Failed tasks surface as
/// `LlmError::ProcessingError`, and unfinished tasks past the polling deadline surface as
/// `LlmError::TimeoutError`.
pub async fn wait_for_task<M: VideoModelV3 + ?Sized>(
    model: &M,
    task_id: &str,
    options: WaitForTaskOptions,
) -> Result<VideoTaskStatusResponse, LlmError> {
    validate_poll_interval(options.poll_interval)?;

    let deadline = options.poll_timeout.map(|timeout| Instant::now() + timeout);
    loop {
        let response = query_task(
            model,
            task_id,
            QueryTaskOptions {
                retry: options.retry.clone(),
            },
        )
        .await?;

        if response.is_success() {
            return Ok(response);
        }

        if response.is_failed() {
            return Err(build_failed_task_error(task_id, &response));
        }

        if let Some(deadline) = deadline
            && Instant::now() >= deadline
        {
            return Err(LlmError::TimeoutError(format!(
                "Timed out waiting for video task '{task_id}' after {} ms",
                options.poll_timeout.unwrap_or_default().as_millis()
            )));
        }

        sleep(options.poll_interval).await;
    }
}

/// Submit one or more video-generation tasks and poll them to completion.
///
/// The helper keeps the Rust-first task model honest:
/// - task creation and task polling remain explicit under the hood
/// - larger `count` values are batched using stable `max_videos_per_call` metadata when available
/// - the final result exposes generated video assets plus the underlying completed tasks
///
/// When all tasks complete successfully but no final assets can be recovered, the helper returns
/// `LlmError::NoVideoGenerated` with best-effort final response metadata.
pub async fn generate<M: VideoModelV4 + ?Sized>(
    model: &M,
    request: VideoGenerationRequest,
    options: GenerateOptions,
) -> Result<GenerateVideoResult, LlmError> {
    validate_poll_interval(options.poll_interval)?;

    let max_videos_per_call = resolve_effective_max_videos_per_call(
        options.max_videos_per_call,
        model.max_videos_per_call(),
    )?;
    let requests = split_generate_requests(request, max_videos_per_call)?;
    let mut warnings = Vec::new();
    let mut videos = Vec::new();
    let mut tasks = Vec::with_capacity(requests.len());
    let mut responses = Vec::with_capacity(requests.len());
    let mut provider_metadata = GenerateVideoProviderMetadata::new();

    for request in requests {
        let requested_count = request.count.unwrap_or(1);
        let created = create_task(
            model,
            request,
            CreateTaskOptions {
                retry: options.create_retry.clone(),
                timeout: options.timeout,
                headers: options.headers.clone(),
            },
        )
        .await?;

        if let Some(create_warnings) = created.warnings.as_ref() {
            warnings.extend(create_warnings.iter().cloned());
        }

        let task_id = created.task_id.clone();
        let queried = wait_for_task(
            model,
            &task_id,
            WaitForTaskOptions {
                retry: options.query_retry.clone(),
                poll_interval: options.poll_interval,
                poll_timeout: options.poll_timeout,
            },
        )
        .await?;

        let generated_videos = extract_generated_videos(model.provider_id(), &queried)?;
        let task_entry = serde_json::json!({
            "taskId": task_id.clone(),
            "requestedCount": requested_count,
            "createMetadata": created.metadata.clone(),
            "queryMetadata": queried.metadata.clone(),
        });
        let call_provider_metadata = build_call_provider_metadata(
            model.provider_id(),
            task_entry,
            &generated_videos,
            &created.metadata,
            &queried.metadata,
        );
        merge_provider_metadata(&mut provider_metadata, call_provider_metadata.clone());

        responses.push(GenerateVideoResponseMetadata {
            task_id: task_id.clone(),
            create_response: created.response.clone(),
            query_response: queried.response.clone(),
            provider_metadata: call_provider_metadata,
        });
        tasks.push(queried);
        videos.extend(generated_videos);
    }

    if videos.is_empty() {
        let error_responses = responses
            .iter()
            .filter_map(|response| {
                response
                    .query_response
                    .clone()
                    .or_else(|| response.create_response.clone())
            })
            .collect();
        return Err(LlmError::NoVideoGenerated {
            responses: error_responses,
        });
    }

    Ok(GenerateVideoResult {
        videos,
        tasks,
        warnings,
        responses,
        provider_metadata,
    })
}

/// Submit video-generation tasks, poll them to completion, and materialize final assets.
///
/// This is closer in role to AI SDK `experimental_generateVideo()` than the plain Rust
/// `generate(...)` helper, but keeps the materialization step explicit through
/// `MaterializeVideoOptions`.
pub async fn generate_materialized<M: VideoModelV4 + ?Sized>(
    model: &M,
    request: VideoGenerationRequest,
    generate_options: GenerateOptions,
    materialize_options: MaterializeVideoOptions,
) -> Result<GenerateMaterializedVideoResult, LlmError> {
    generate(model, request, generate_options)
        .await?
        .into_materialized(materialize_options)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{Engine, engine::general_purpose::STANDARD};
    use siumai_core::traits::ModelMetadata;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::Mutex;

    #[derive(Clone)]
    struct FakeGenerateVideoModel {
        next_task: Arc<AtomicUsize>,
        query_counts: Arc<Mutex<HashMap<String, usize>>>,
        task_video_counts: Arc<Mutex<HashMap<String, u32>>>,
        fail_tasks: Arc<Mutex<HashMap<String, String>>>,
        never_finish: bool,
        max_videos_per_call: Option<u32>,
    }

    impl FakeGenerateVideoModel {
        fn new(max_videos_per_call: Option<u32>) -> Self {
            Self {
                next_task: Arc::new(AtomicUsize::new(1)),
                query_counts: Arc::new(Mutex::new(HashMap::new())),
                task_video_counts: Arc::new(Mutex::new(HashMap::new())),
                fail_tasks: Arc::new(Mutex::new(HashMap::new())),
                never_finish: false,
                max_videos_per_call,
            }
        }

        fn failing(message: impl Into<String>) -> Self {
            let mut fail_tasks = HashMap::new();
            fail_tasks.insert("task-1".to_string(), message.into());
            Self {
                next_task: Arc::new(AtomicUsize::new(1)),
                query_counts: Arc::new(Mutex::new(HashMap::new())),
                task_video_counts: Arc::new(Mutex::new(HashMap::new())),
                fail_tasks: Arc::new(Mutex::new(fail_tasks)),
                never_finish: false,
                max_videos_per_call: Some(2),
            }
        }

        fn never_finishing() -> Self {
            Self {
                next_task: Arc::new(AtomicUsize::new(1)),
                query_counts: Arc::new(Mutex::new(HashMap::new())),
                task_video_counts: Arc::new(Mutex::new(HashMap::new())),
                fail_tasks: Arc::new(Mutex::new(HashMap::new())),
                never_finish: true,
                max_videos_per_call: Some(2),
            }
        }
    }

    impl Default for FakeGenerateVideoModel {
        fn default() -> Self {
            Self::new(Some(2))
        }
    }

    #[derive(Clone, Default)]
    struct FakeInlineMaterializedVideoModel;

    #[derive(Clone, Default)]
    struct FakeNoAssetsVideoModel;

    async fn spawn_video_download_server(
        body: Vec<u8>,
        content_type: &str,
        expected_header: Option<(&str, &str)>,
        path: &str,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind local download server");
        let addr = listener.local_addr().expect("download server address");
        let path = path.to_string();
        let response_path = path.clone();
        let content_type = content_type.to_string();
        let expected_header =
            expected_header.map(|(name, value)| (name.to_string(), value.to_string()));

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept download request");
            let mut request = Vec::new();
            let mut buffer = [0u8; 1024];
            loop {
                let read = stream.read(&mut buffer).await.expect("read request bytes");
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }

            let request_text = String::from_utf8_lossy(&request);
            assert!(
                request_text.starts_with(&format!("GET {response_path} HTTP/1.1")),
                "unexpected request line: {request_text}"
            );
            if let Some((name, value)) = expected_header {
                let request_lower = request_text.to_ascii_lowercase();
                let header_lower = format!(
                    "{}: {}",
                    name.to_ascii_lowercase(),
                    value.to_ascii_lowercase()
                );
                assert!(
                    request_lower.contains(&header_lower),
                    "expected request header '{header_lower}' in {request_text}"
                );
            }

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: {}\r\nConnection: close\r\n\r\n",
                body.len(),
                content_type
            );
            stream
                .write_all(response.as_bytes())
                .await
                .expect("write response headers");
            stream.write_all(&body).await.expect("write response body");
        });

        format!("http://{addr}{path}")
    }

    impl ModelMetadata for FakeGenerateVideoModel {
        fn provider_id(&self) -> &str {
            "fake-video"
        }

        fn model_id(&self) -> &str {
            "fake-video-model"
        }
    }

    impl ModelMetadata for FakeInlineMaterializedVideoModel {
        fn provider_id(&self) -> &str {
            "inline-video"
        }

        fn model_id(&self) -> &str {
            "inline-video-model"
        }
    }

    impl ModelMetadata for FakeNoAssetsVideoModel {
        fn provider_id(&self) -> &str {
            "empty-video"
        }

        fn model_id(&self) -> &str {
            "empty-video-model"
        }
    }

    #[async_trait::async_trait]
    impl VideoModelV3 for FakeGenerateVideoModel {
        async fn create_task(
            &self,
            request: VideoGenerationRequest,
        ) -> Result<VideoGenerationResponse, LlmError> {
            let task_number = self.next_task.fetch_add(1, Ordering::SeqCst);
            let task_id = format!("task-{task_number}");
            self.task_video_counts
                .lock()
                .await
                .insert(task_id.clone(), request.count.unwrap_or(1));
            Ok(VideoGenerationResponse {
                task_id,
                base_resp: None,
                metadata: HashMap::from([
                    ("prompt".to_string(), serde_json::json!(request.prompt)),
                    (
                        "requestedCount".to_string(),
                        serde_json::json!(request.count.unwrap_or(1)),
                    ),
                    (
                        "fake-video".to_string(),
                        serde_json::json!({
                            "createId": format!("create-{task_number}"),
                        }),
                    ),
                ]),
                warnings: Some(vec![Warning::compatibility(
                    "fake-create",
                    Some("fake create warning"),
                )]),
                response: Some(HttpResponseInfo {
                    timestamp: chrono::Utc::now(),
                    model_id: Some(request.model),
                    headers: HashMap::from([("x-create".to_string(), "1".to_string())]),
                }),
            })
        }

        async fn query_task(&self, task_id: &str) -> Result<VideoTaskStatusResponse, LlmError> {
            if self.never_finish {
                return Ok(VideoTaskStatusResponse {
                    task_id: task_id.to_string(),
                    status: VideoTaskStatus::Processing,
                    file_id: None,
                    video_url: None,
                    duration: None,
                    video_width: None,
                    video_height: None,
                    base_resp: None,
                    metadata: HashMap::new(),
                    response: None,
                });
            }

            if let Some(message) = self.fail_tasks.lock().await.get(task_id).cloned() {
                return Ok(VideoTaskStatusResponse {
                    task_id: task_id.to_string(),
                    status: VideoTaskStatus::Fail,
                    file_id: None,
                    video_url: None,
                    duration: None,
                    video_width: None,
                    video_height: None,
                    base_resp: Some(siumai_core::types::BaseResponse {
                        status_code: -1,
                        status_msg: message,
                    }),
                    metadata: HashMap::new(),
                    response: None,
                });
            }

            let mut counts = self.query_counts.lock().await;
            let count = counts.entry(task_id.to_string()).or_insert(0);
            *count += 1;

            if *count < 2 {
                return Ok(VideoTaskStatusResponse {
                    task_id: task_id.to_string(),
                    status: VideoTaskStatus::Processing,
                    file_id: None,
                    video_url: None,
                    duration: None,
                    video_width: None,
                    video_height: None,
                    base_resp: None,
                    metadata: HashMap::new(),
                    response: None,
                });
            }

            let requested_count = self
                .task_video_counts
                .lock()
                .await
                .get(task_id)
                .copied()
                .unwrap_or(1);
            let video_entries = (0..requested_count)
                .map(|index| {
                    let url = if requested_count == 1 {
                        format!("https://example.com/{task_id}.mp4")
                    } else {
                        format!("https://example.com/{task_id}-{index}.mp4")
                    };
                    serde_json::json!({
                        "url": url,
                        "mediaType": "video/mp4",
                        "index": index,
                    })
                })
                .collect::<Vec<_>>();

            let primary_video_url = video_entries
                .first()
                .and_then(|entry| entry.get("url"))
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned);

            Ok(VideoTaskStatusResponse {
                task_id: task_id.to_string(),
                status: VideoTaskStatus::Success,
                file_id: Some(format!("file-{task_id}")),
                video_url: primary_video_url,
                duration: Some(6.0),
                video_width: Some(1280),
                video_height: Some(720),
                base_resp: None,
                metadata: HashMap::from([(
                    "fake-video".to_string(),
                    serde_json::json!({
                        "videos": video_entries,
                        "jobType": "predictLongRunning",
                    }),
                )]),
                response: Some(HttpResponseInfo {
                    timestamp: chrono::Utc::now(),
                    model_id: Some("fake-video-model".to_string()),
                    headers: HashMap::from([("x-query".to_string(), "1".to_string())]),
                }),
            })
        }

        fn max_videos_per_call(&self) -> Option<u32> {
            self.max_videos_per_call
        }
    }

    #[async_trait::async_trait]
    impl VideoModelV3 for FakeInlineMaterializedVideoModel {
        async fn create_task(
            &self,
            request: VideoGenerationRequest,
        ) -> Result<VideoGenerationResponse, LlmError> {
            Ok(VideoGenerationResponse {
                task_id: format!("inline-task:{}", request.model),
                base_resp: None,
                metadata: HashMap::new(),
                warnings: None,
                response: None,
            })
        }

        async fn query_task(&self, task_id: &str) -> Result<VideoTaskStatusResponse, LlmError> {
            Ok(VideoTaskStatusResponse {
                task_id: task_id.to_string(),
                status: VideoTaskStatus::Success,
                file_id: None,
                video_url: None,
                duration: Some(4.0),
                video_width: Some(1280),
                video_height: Some(720),
                base_resp: None,
                metadata: HashMap::from([(
                    "inline-video".to_string(),
                    serde_json::json!({
                        "videos": [
                            {
                                "base64": STANDARD.encode([1_u8, 2, 3]),
                                "mediaType": "video/mp4"
                            }
                        ]
                    }),
                )]),
                response: None,
            })
        }

        fn max_videos_per_call(&self) -> Option<u32> {
            Some(1)
        }
    }

    #[async_trait::async_trait]
    impl VideoModelV3 for FakeNoAssetsVideoModel {
        async fn create_task(
            &self,
            request: VideoGenerationRequest,
        ) -> Result<VideoGenerationResponse, LlmError> {
            Ok(VideoGenerationResponse {
                task_id: format!("empty-task:{}", request.model),
                base_resp: None,
                metadata: HashMap::new(),
                warnings: None,
                response: Some(HttpResponseInfo {
                    timestamp: chrono::Utc::now(),
                    model_id: Some(request.model),
                    headers: HashMap::from([("x-create".to_string(), "1".to_string())]),
                }),
            })
        }

        async fn query_task(&self, task_id: &str) -> Result<VideoTaskStatusResponse, LlmError> {
            Ok(VideoTaskStatusResponse {
                task_id: task_id.to_string(),
                status: VideoTaskStatus::Success,
                file_id: None,
                video_url: None,
                duration: Some(4.0),
                video_width: Some(1280),
                video_height: Some(720),
                base_resp: None,
                metadata: HashMap::new(),
                response: Some(HttpResponseInfo {
                    timestamp: chrono::Utc::now(),
                    model_id: Some("empty-video-model".to_string()),
                    headers: HashMap::from([("x-query".to_string(), "1".to_string())]),
                }),
            })
        }

        fn max_videos_per_call(&self) -> Option<u32> {
            Some(1)
        }
    }

    #[tokio::test]
    async fn wait_for_task_polls_until_success() {
        let model = FakeGenerateVideoModel::default();
        let response = wait_for_task(
            &model,
            "task-1",
            WaitForTaskOptions {
                poll_interval: Duration::from_millis(1),
                poll_timeout: Some(Duration::from_millis(50)),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert!(response.is_success());
        assert_eq!(
            response.video_url.as_deref(),
            Some("https://example.com/task-1.mp4")
        );
    }

    #[tokio::test]
    async fn wait_for_task_times_out_when_task_never_completes() {
        let model = FakeGenerateVideoModel::never_finishing();
        let err = wait_for_task(
            &model,
            "task-1",
            WaitForTaskOptions {
                poll_interval: Duration::from_millis(1),
                poll_timeout: Some(Duration::from_millis(10)),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();

        assert!(matches!(err, LlmError::TimeoutError(message) if message.contains("task-1")));
    }

    #[tokio::test]
    async fn wait_for_task_returns_error_when_task_fails() {
        let model = FakeGenerateVideoModel::failing("provider task failed");
        let err = wait_for_task(
            &model,
            "task-1",
            WaitForTaskOptions {
                poll_interval: Duration::from_millis(1),
                poll_timeout: Some(Duration::from_millis(10)),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();

        assert!(
            matches!(err, LlmError::ProcessingError(message) if message.contains("provider task failed"))
        );
    }

    #[tokio::test]
    async fn generate_batches_by_max_videos_per_call_and_collects_generated_assets() {
        let model = FakeGenerateVideoModel::default();
        let result = generate(
            &model,
            VideoGenerationRequest::new("fake-video-model", "animate a robot").with_count(3),
            GenerateOptions {
                poll_interval: Duration::from_millis(1),
                poll_timeout: Some(Duration::from_millis(50)),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(result.videos.len(), 3);
        assert_eq!(result.tasks.len(), 2);
        assert_eq!(result.responses.len(), 2);
        assert_eq!(result.warnings.len(), 2);
        assert_eq!(
            result.video().and_then(GeneratedVideo::url),
            Some("https://example.com/task-1-0.mp4")
        );
        assert_eq!(
            result.task().and_then(|task| task.video_url.as_deref()),
            Some("https://example.com/task-1-0.mp4")
        );

        let provider_metadata = result.provider_metadata.get("fake-video").unwrap();
        assert_eq!(
            provider_metadata
                .get("createId")
                .and_then(|value| value.as_str()),
            Some("create-2")
        );
        assert_eq!(
            provider_metadata
                .get("jobType")
                .and_then(|value| value.as_str()),
            Some("predictLongRunning")
        );
        let task_entries = provider_metadata
            .get("tasks")
            .and_then(|value| value.as_array())
            .unwrap();
        assert_eq!(task_entries.len(), 2);
        let video_entries = provider_metadata
            .get("videos")
            .and_then(|value| value.as_array())
            .unwrap();
        assert_eq!(video_entries.len(), 3);
        assert_eq!(
            task_entries[0]
                .get("createMetadata")
                .and_then(|value| value.get("prompt"))
                .and_then(|value| value.as_str()),
            Some("animate a robot")
        );
        assert_eq!(
            task_entries[0]
                .get("requestedCount")
                .and_then(|value| value.as_u64()),
            Some(2)
        );
        assert_eq!(
            video_entries[2].get("url").and_then(|value| value.as_str()),
            Some("https://example.com/task-2.mp4")
        );
    }

    #[tokio::test]
    async fn generate_response_metadata_projects_to_ai_sdk_video_response_view() {
        let model = FakeGenerateVideoModel::default();
        let result = generate(
            &model,
            VideoGenerationRequest::new("fake-video-model", "animate a robot").with_count(2),
            GenerateOptions {
                poll_interval: Duration::from_millis(1),
                poll_timeout: Some(Duration::from_millis(50)),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let response = result
            .responses
            .first()
            .and_then(GenerateVideoResponseMetadata::response_metadata)
            .expect("ai sdk-style response metadata");
        let create_response = result
            .responses
            .first()
            .and_then(GenerateVideoResponseMetadata::create_response_metadata)
            .expect("create response metadata");
        let query_response = result
            .responses
            .first()
            .and_then(GenerateVideoResponseMetadata::query_response_metadata)
            .expect("query response metadata");

        assert_eq!(create_response.model_id, "fake-video-model");
        assert_eq!(
            create_response
                .headers
                .as_ref()
                .and_then(|headers| headers.get("x-create"))
                .map(String::as_str),
            Some("1")
        );
        assert_eq!(query_response.model_id, "fake-video-model");
        assert_eq!(
            query_response
                .headers
                .as_ref()
                .and_then(|headers| headers.get("x-query"))
                .map(String::as_str),
            Some("1")
        );
        assert_eq!(response.model_id, "fake-video-model");
        assert_eq!(
            response
                .provider_metadata
                .as_ref()
                .and_then(|metadata| metadata.get("fake-video"))
                .and_then(|value| value.get("jobType"))
                .and_then(serde_json::Value::as_str),
            Some("predictLongRunning")
        );
        assert_eq!(result.video_model_responses().len(), 1);
    }

    #[tokio::test]
    async fn generate_returns_no_video_generated_error_with_final_response_metadata() {
        let model = FakeNoAssetsVideoModel;
        let err = generate(
            &model,
            VideoGenerationRequest::new("empty-video-model", "empty result"),
            GenerateOptions {
                poll_interval: Duration::from_millis(1),
                poll_timeout: Some(Duration::from_millis(20)),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();

        match err {
            LlmError::NoVideoGenerated { responses } => {
                assert_eq!(responses.len(), 1);
                assert_eq!(responses[0].model_id.as_deref(), Some("empty-video-model"));
                assert_eq!(
                    responses[0].headers.get("x-query").map(String::as_str),
                    Some("1")
                );
            }
            other => panic!("expected NoVideoGenerated error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn generated_video_materialize_decodes_base64_assets() {
        let base64 = STANDARD.encode([1_u8, 2, 3, 4]);
        let video = GeneratedVideo {
            task_id: "task-1".to_string(),
            media_type: "video/mp4".to_string(),
            data: GeneratedVideoData::Base64 {
                data: base64.clone(),
            },
            metadata: HashMap::new(),
        };

        let materialized = video
            .materialize(MaterializeVideoOptions::default())
            .await
            .expect("materialize base64 video");

        assert_eq!(materialized.media_type, "video/mp4");
        assert_eq!(
            materialized.bytes().expect("decode materialized bytes"),
            vec![1, 2, 3, 4]
        );
        assert_eq!(materialized.base64(), base64);
    }

    #[tokio::test]
    async fn generated_video_materialize_downloads_url_assets_with_http_config() {
        let url = spawn_video_download_server(
            vec![9, 8, 7, 6],
            "video/webm",
            Some(("x-download-token", "secret")),
            "/generated-video",
        )
        .await;
        let video = GeneratedVideo {
            task_id: "task-2".to_string(),
            media_type: "video/mp4".to_string(),
            data: GeneratedVideoData::Url { url },
            metadata: HashMap::new(),
        };

        let materialized = video
            .materialize(MaterializeVideoOptions {
                http_config: Some(
                    HttpConfig::builder()
                        .header("x-download-token", "secret")
                        .build(),
                ),
            })
            .await
            .expect("download generated video");

        assert_eq!(materialized.media_type, "video/webm");
        assert_eq!(
            materialized.bytes().expect("downloaded bytes"),
            vec![9, 8, 7, 6]
        );
    }

    #[tokio::test]
    async fn generated_video_materialize_rejects_provider_reference_assets() {
        let video = GeneratedVideo {
            task_id: "task-3".to_string(),
            media_type: "video/mp4".to_string(),
            data: GeneratedVideoData::ProviderReference {
                provider_reference: ProviderReference::single("fake-video", "file-123"),
            },
            metadata: HashMap::new(),
        };

        let err = video
            .materialize(MaterializeVideoOptions::default())
            .await
            .unwrap_err();

        assert!(
            matches!(err, LlmError::UnsupportedOperation(message) if message.contains("provider reference"))
        );
    }

    #[tokio::test]
    async fn generate_video_result_materialize_videos_materializes_all_assets() {
        let result = GenerateVideoResult {
            videos: vec![
                GeneratedVideo {
                    task_id: "task-1".to_string(),
                    media_type: "video/mp4".to_string(),
                    data: GeneratedVideoData::Base64 {
                        data: STANDARD.encode([1_u8, 2, 3]),
                    },
                    metadata: HashMap::new(),
                },
                GeneratedVideo {
                    task_id: "task-2".to_string(),
                    media_type: "video/webm".to_string(),
                    data: GeneratedVideoData::Bytes {
                        data: vec![4, 5, 6],
                    },
                    metadata: HashMap::new(),
                },
            ],
            tasks: Vec::new(),
            warnings: Vec::new(),
            responses: Vec::new(),
            provider_metadata: HashMap::new(),
        };

        let videos = result
            .materialize_videos(MaterializeVideoOptions::default())
            .await
            .expect("materialize all generated videos");

        assert_eq!(videos.len(), 2);
        assert_eq!(videos[0].bytes().expect("first bytes"), vec![1, 2, 3]);
        assert_eq!(videos[1].base64(), STANDARD.encode([4_u8, 5, 6]));
        assert_eq!(
            result
                .materialize_video(MaterializeVideoOptions::default())
                .await
                .expect("materialize first generated video")
                .expect("first materialized video")
                .media_type,
            "video/mp4"
        );

        let materialized_result = result
            .clone()
            .into_materialized(MaterializeVideoOptions::default())
            .await
            .expect("convert generated result into materialized result");
        assert_eq!(materialized_result.videos.len(), 2);
        assert_eq!(
            materialized_result
                .video()
                .expect("first materialized video")
                .media_type,
            "video/mp4"
        );
    }

    #[tokio::test]
    async fn generate_materialized_combines_generation_and_materialization() {
        let model = FakeInlineMaterializedVideoModel;
        let result = generate_materialized(
            &model,
            VideoGenerationRequest::new("inline-video-model", "animate inline"),
            GenerateOptions {
                poll_interval: Duration::from_millis(1),
                poll_timeout: Some(Duration::from_millis(20)),
                ..Default::default()
            },
            MaterializeVideoOptions::default(),
        )
        .await
        .expect("generate and materialize final videos");

        assert_eq!(result.videos.len(), 1);
        assert_eq!(
            result.video().expect("materialized video").bytes().unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(
            result.task().map(|task| task.task_id.as_str()),
            Some("inline-task:inline-video-model")
        );
        assert!(result.video_model_responses().is_empty());
    }
}
