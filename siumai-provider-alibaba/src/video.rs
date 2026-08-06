use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use http::header::{ACCEPT, HeaderName, HeaderValue};
use http::{Method, StatusCode};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use siumai_core::experimental::{JobId, JobStatus, MediaJob, VideoJobModel};
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, ModelId, ProviderId, ProviderScope,
    PublicDiagnosticText, Warning, WarningKind,
};
use siumai_transport::{
    ProviderTransport, ReplaySafety, RequestBody, RequestBuildError, RequestHeaders, RequestPlan,
    RequestTarget, ResourceDownloadOptions, ResourceDownloader, ResourceUrl, ResourceUrlError,
    TransportResponse,
};
use thiserror::Error as ThisError;

use crate::native_error::{self, response_request_id};

pub const LEGACY_SINGAPORE_VIDEO_BASE_URL: &str = "https://dashscope-intl.aliyuncs.com/api/v1";
pub const VIDEO_API_MODE_ID: &str = "video-job";
pub const VIDEO_PROTOCOL_ID: &str = "alibaba-native";
pub const VIDEO_TEXT_SOURCE: &str =
    "https://www.alibabacloud.com/help/en/model-studio/text-to-video-api-reference";
pub const VIDEO_IMAGE_SOURCE: &str =
    "https://www.alibabacloud.com/help/en/model-studio/image-to-video-general-api-reference";
pub const VIDEO_REFERENCE_SOURCE: &str =
    "https://www.alibabacloud.com/help/en/model-studio/reference-to-video-api-reference";
pub const VIDEO_CANCEL_SOURCE: &str =
    "https://www.alibabacloud.com/help/en/model-studio/cancel-api-task";

pub const WAN_2_7_T2V: &str = "wan2.7-t2v";
pub const WAN_2_7_T2V_SNAPSHOT: &str = "wan2.7-t2v-2026-06-12";
pub const WAN_2_7_I2V: &str = "wan2.7-i2v";
pub const WAN_2_7_I2V_SNAPSHOT: &str = "wan2.7-i2v-2026-04-25";
pub const WAN_2_7_R2V: &str = "wan2.7-r2v";
pub const WAN_2_7_R2V_SNAPSHOT: &str = "wan2.7-r2v-2026-06-12";

const VIDEO_CREATE_TARGET: &str = "services/aigc/video-generation/video-synthesis";
const ASYNC_HEADER: HeaderName = HeaderName::from_static("x-dashscope-async");

const KNOWN_MODELS: &[&str] = &[
    WAN_2_7_T2V,
    WAN_2_7_T2V_SNAPSHOT,
    WAN_2_7_I2V,
    WAN_2_7_I2V_SNAPSHOT,
    WAN_2_7_R2V,
    WAN_2_7_R2V_SNAPSHOT,
    "wan2.6-t2v",
    "wan2.5-t2v-preview",
    "wan2.6-i2v",
    "wan2.6-i2v-flash",
    "wan2.6-r2v",
    "wan2.6-r2v-flash",
];

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AlibabaVideoMediaType {
    FirstFrame,
    LastFrame,
    ReferenceImage,
    ReferenceVideo,
    FirstClip,
    DrivingAudio,
    ReferenceVoice,
    Other(String),
}

impl AlibabaVideoMediaType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::FirstFrame => "first_frame",
            Self::LastFrame => "last_frame",
            Self::ReferenceImage => "reference_image",
            Self::ReferenceVideo => "reference_video",
            Self::FirstClip => "first_clip",
            Self::DrivingAudio => "driving_audio",
            Self::ReferenceVoice => "reference_voice",
            Self::Other(value) => value,
        }
    }
}

impl Serialize for AlibabaVideoMediaType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for AlibabaVideoMediaType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "first_frame" => Self::FirstFrame,
            "last_frame" => Self::LastFrame,
            "reference_image" => Self::ReferenceImage,
            "reference_video" => Self::ReferenceVideo,
            "first_clip" => Self::FirstClip,
            "driving_audio" => Self::DrivingAudio,
            "reference_voice" => Self::ReferenceVoice,
            _ => Self::Other(value),
        })
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct AlibabaVideoMedia {
    #[serde(rename = "type")]
    media_type: AlibabaVideoMediaType,
    url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<f32>,
}

impl AlibabaVideoMedia {
    pub fn new(media_type: AlibabaVideoMediaType, url: impl Into<String>) -> Self {
        Self {
            media_type,
            url: url.into(),
            duration: None,
        }
    }

    pub fn first_frame(url: impl Into<String>) -> Self {
        Self::new(AlibabaVideoMediaType::FirstFrame, url)
    }

    pub fn last_frame(url: impl Into<String>) -> Self {
        Self::new(AlibabaVideoMediaType::LastFrame, url)
    }

    pub fn reference_image(url: impl Into<String>) -> Self {
        Self::new(AlibabaVideoMediaType::ReferenceImage, url)
    }

    pub fn reference_video(url: impl Into<String>) -> Self {
        Self::new(AlibabaVideoMediaType::ReferenceVideo, url)
    }

    pub fn driving_audio(url: impl Into<String>) -> Self {
        Self::new(AlibabaVideoMediaType::DrivingAudio, url)
    }

    pub fn with_duration(mut self, duration: f32) -> Self {
        self.duration = Some(duration);
        self
    }

    pub fn media_type(&self) -> &AlibabaVideoMediaType {
        &self.media_type
    }

    /// Explicit access to a provider-fetched URL. Avoid logging signed URLs.
    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn duration(&self) -> Option<f32> {
        self.duration
    }
}

impl fmt::Debug for AlibabaVideoMedia {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AlibabaVideoMedia")
            .field("media_type", &self.media_type)
            .field("url", &"[REDACTED]")
            .field("duration", &self.duration)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlibabaVideoShotType {
    Single,
    Multi,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlibabaVideoParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_extend: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    watermark: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shot_type: Option<AlibabaVideoShotType>,
}

impl AlibabaVideoParameters {
    pub const fn new() -> Self {
        Self {
            size: None,
            resolution: None,
            ratio: None,
            duration: None,
            prompt_extend: None,
            watermark: None,
            seed: None,
            audio: None,
            shot_type: None,
        }
    }

    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.size = Some(size.into());
        self.resolution = None;
        self.ratio = None;
        self
    }

    pub fn with_resolution(
        mut self,
        resolution: impl Into<String>,
        ratio: impl Into<String>,
    ) -> Self {
        self.size = None;
        self.resolution = Some(resolution.into());
        self.ratio = Some(ratio.into());
        self
    }

    pub const fn with_duration(mut self, duration: u16) -> Self {
        self.duration = Some(duration);
        self
    }

    pub const fn with_prompt_extension(mut self, enabled: bool) -> Self {
        self.prompt_extend = Some(enabled);
        self
    }

    pub const fn with_watermark(mut self, enabled: bool) -> Self {
        self.watermark = Some(enabled);
        self
    }

    pub const fn with_seed(mut self, seed: u32) -> Self {
        self.seed = Some(seed);
        self
    }

    pub const fn with_audio(mut self, enabled: bool) -> Self {
        self.audio = Some(enabled);
        self
    }

    pub const fn with_shot_type(mut self, shot_type: AlibabaVideoShotType) -> Self {
        self.shot_type = Some(shot_type);
        self
    }
}

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AlibabaVideoRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    negative_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    media: Vec<AlibabaVideoMedia>,
    #[serde(rename = "img_url", skip_serializing_if = "Option::is_none")]
    legacy_image_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_url: Option<String>,
    #[serde(default)]
    parameters: AlibabaVideoParameters,
}

impl AlibabaVideoRequest {
    pub const fn new() -> Self {
        Self {
            prompt: None,
            negative_prompt: None,
            media: Vec::new(),
            legacy_image_url: None,
            audio_url: None,
            parameters: AlibabaVideoParameters::new(),
        }
    }

    pub fn text(prompt: impl Into<String>) -> Self {
        Self::new().with_prompt(prompt)
    }

    pub fn image(prompt: impl Into<String>, image_url: impl Into<String>) -> Self {
        Self::text(prompt).with_media(AlibabaVideoMedia::first_frame(image_url))
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    pub fn with_negative_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.negative_prompt = Some(prompt.into());
        self
    }

    pub fn with_media(mut self, media: AlibabaVideoMedia) -> Self {
        self.media.push(media);
        self
    }

    pub fn with_legacy_image_url(mut self, image_url: impl Into<String>) -> Self {
        self.legacy_image_url = Some(image_url.into());
        self
    }

    pub fn with_audio_url(mut self, audio_url: impl Into<String>) -> Self {
        self.audio_url = Some(audio_url.into());
        self
    }

    pub fn with_parameters(mut self, parameters: AlibabaVideoParameters) -> Self {
        self.parameters = parameters;
        self
    }

    pub fn validate(&self) -> Result<(), AlibabaVideoRequestError> {
        if self
            .prompt
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(AlibabaVideoRequestError::EmptyPrompt);
        }
        if self
            .negative_prompt
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(AlibabaVideoRequestError::EmptyNegativePrompt);
        }
        if self.prompt.is_none()
            && self.media.is_empty()
            && self.legacy_image_url.is_none()
            && self.audio_url.is_none()
        {
            return Err(AlibabaVideoRequestError::MissingInput);
        }
        for (index, media) in self.media.iter().enumerate() {
            if media.media_type.as_str().trim().is_empty() {
                return Err(AlibabaVideoRequestError::EmptyMediaType { index });
            }
            if media.url.trim().is_empty() {
                return Err(AlibabaVideoRequestError::EmptyMediaUrl { index });
            }
            if media
                .duration
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
            {
                return Err(AlibabaVideoRequestError::InvalidMediaDuration { index });
            }
        }
        if self
            .legacy_image_url
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(AlibabaVideoRequestError::EmptyLegacyImageUrl);
        }
        if self
            .audio_url
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(AlibabaVideoRequestError::EmptyAudioUrl);
        }
        self.parameters.validate()
    }
}

impl fmt::Debug for AlibabaVideoRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AlibabaVideoRequest")
            .field("has_prompt", &self.prompt.is_some())
            .field("has_negative_prompt", &self.negative_prompt.is_some())
            .field("media_count", &self.media.len())
            .field("has_legacy_image_url", &self.legacy_image_url.is_some())
            .field("has_audio_url", &self.audio_url.is_some())
            .field("parameters", &self.parameters)
            .finish()
    }
}

impl AlibabaVideoParameters {
    fn validate(&self) -> Result<(), AlibabaVideoRequestError> {
        if self
            .size
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(AlibabaVideoRequestError::EmptySize);
        }
        if self
            .resolution
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(AlibabaVideoRequestError::EmptyResolution);
        }
        if self
            .ratio
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(AlibabaVideoRequestError::EmptyRatio);
        }
        if self.size.is_some() && (self.resolution.is_some() || self.ratio.is_some()) {
            return Err(AlibabaVideoRequestError::ConflictingGeometry);
        }
        if self.resolution.is_some() != self.ratio.is_some() {
            return Err(AlibabaVideoRequestError::IncompleteResolution);
        }
        if self.duration == Some(0) {
            return Err(AlibabaVideoRequestError::ZeroDuration);
        }
        if self.seed.is_some_and(|seed| seed > i32::MAX as u32) {
            return Err(AlibabaVideoRequestError::SeedOutOfRange);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, ThisError)]
#[non_exhaustive]
pub enum AlibabaVideoRequestError {
    #[error("Alibaba video request requires prompt, media, image, or audio input")]
    MissingInput,
    #[error("Alibaba video prompt must not be empty")]
    EmptyPrompt,
    #[error("Alibaba video negative prompt must not be empty")]
    EmptyNegativePrompt,
    #[error("Alibaba video media item {index} has an empty type")]
    EmptyMediaType { index: usize },
    #[error("Alibaba video media item {index} has an empty URL")]
    EmptyMediaUrl { index: usize },
    #[error("Alibaba video media item {index} has an invalid duration")]
    InvalidMediaDuration { index: usize },
    #[error("Alibaba legacy image URL must not be empty")]
    EmptyLegacyImageUrl,
    #[error("Alibaba video audio URL must not be empty")]
    EmptyAudioUrl,
    #[error("Alibaba video size must not be empty")]
    EmptySize,
    #[error("Alibaba video resolution must not be empty")]
    EmptyResolution,
    #[error("Alibaba video ratio must not be empty")]
    EmptyRatio,
    #[error("Alibaba video size conflicts with resolution and ratio")]
    ConflictingGeometry,
    #[error("Alibaba video resolution and ratio must be configured together")]
    IncompleteResolution,
    #[error("Alibaba video duration must be greater than zero")]
    ZeroDuration,
    #[error("Alibaba video seed exceeds the provider wire range")]
    SeedOutOfRange,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum AlibabaVideoDownloadPolicy {
    #[default]
    PublicOnly,
    LoopbackExplicit,
    PrivateNetworkExplicit,
    LinkLocalExplicit,
}

impl AlibabaVideoDownloadPolicy {
    fn resource(self, value: &str) -> Result<ResourceUrl, ResourceUrlError> {
        match self {
            Self::PublicOnly => ResourceUrl::public(value),
            Self::LoopbackExplicit => ResourceUrl::local_explicit(value),
            Self::PrivateNetworkExplicit => ResourceUrl::private_network_explicit(value),
            Self::LinkLocalExplicit => ResourceUrl::link_local_explicit(value),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AlibabaVideoUsage {
    pub duration: Option<f32>,
    pub output_video_duration: Option<f32>,
    #[serde(rename = "SR")]
    pub super_resolution: Option<f32>,
    pub size: Option<String>,
}

#[derive(Clone, PartialEq)]
pub struct AlibabaVideoJob {
    id: JobId,
    model: ModelId,
    status: JobStatus,
    request_id: Option<String>,
    video_url: Option<String>,
    provider_code: Option<String>,
    usage: AlibabaVideoUsage,
    warnings: Vec<Warning>,
}

impl AlibabaVideoJob {
    pub fn id(&self) -> &JobId {
        &self.id
    }

    pub fn model_id(&self) -> &ModelId {
        &self.model
    }

    pub fn status(&self) -> &JobStatus {
        &self.status
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    /// Explicit access to the provider-returned download URL. Avoid logging signed URLs.
    pub fn video_url(&self) -> Option<&str> {
        self.video_url.as_deref()
    }

    pub fn provider_code(&self) -> Option<&str> {
        self.provider_code.as_deref()
    }

    pub fn usage(&self) -> &AlibabaVideoUsage {
        &self.usage
    }

    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    fn into_media_job(self) -> Result<MediaJob, Error> {
        let state = serde_json::to_value(AlibabaVideoJobState {
            model: self.model,
            request_id: self.request_id,
            provider_code: self.provider_code,
            usage: self.usage,
            warnings: self.warnings,
        })
        .map_err(job_state_error)?;
        Ok(MediaJob::new(self.id, self.status, state))
    }

    fn from_media_job(job: &MediaJob) -> Result<Self, Error> {
        let state = serde_json::from_value::<AlibabaVideoJobState>(job.state().clone())
            .map_err(job_state_error)?;
        Ok(Self {
            id: job.id().clone(),
            model: state.model,
            status: job.status().clone(),
            request_id: state.request_id,
            video_url: None,
            provider_code: state.provider_code,
            usage: state.usage,
            warnings: state.warnings,
        })
    }
}

impl fmt::Debug for AlibabaVideoJob {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AlibabaVideoJob")
            .field("id", &self.id)
            .field("model", &self.model)
            .field("status", &self.status)
            .field("request_id", &self.request_id)
            .field("video_url", &self.video_url.as_ref().map(|_| "[REDACTED]"))
            .field("provider_code", &self.provider_code)
            .field("usage", &self.usage)
            .field("warnings", &self.warnings)
            .finish()
    }
}

#[derive(Serialize, Deserialize)]
struct AlibabaVideoJobState {
    model: ModelId,
    request_id: Option<String>,
    provider_code: Option<String>,
    usage: AlibabaVideoUsage,
    warnings: Vec<Warning>,
}

pub(crate) struct AlibabaVideoRuntime {
    pub(crate) scope: Arc<ProviderScope>,
    pub(crate) transport: ProviderTransport,
    pub(crate) downloader: ResourceDownloader,
    pub(crate) download_policy: AlibabaVideoDownloadPolicy,
}

impl fmt::Debug for AlibabaVideoRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AlibabaVideoRuntime")
            .field("scope", &self.scope)
            .field("transport", &"shared")
            .field("downloader", &"shared")
            .field("download_policy", &self.download_policy)
            .finish()
    }
}

#[derive(Clone)]
pub struct AlibabaVideoModel {
    runtime: Arc<AlibabaVideoRuntime>,
    model: ModelId,
}

impl AlibabaVideoModel {
    pub(crate) fn new(runtime: Arc<AlibabaVideoRuntime>, model: ModelId) -> Self {
        Self { runtime, model }
    }

    pub fn scope(&self) -> &ProviderScope {
        &self.runtime.scope
    }

    pub fn provider_id(&self) -> &ProviderId {
        self.runtime.scope.provider_id()
    }

    pub fn model_id(&self) -> &ModelId {
        &self.model
    }

    pub async fn create_video(
        &self,
        request: AlibabaVideoRequest,
        options: CallOptions,
    ) -> Result<AlibabaVideoJob, Error> {
        reject_provider_options(&options).map_err(|error| self.contextualize(error))?;
        request
            .validate()
            .map_err(request_validation_error)
            .map_err(|error| self.contextualize(error))?;
        let body = AlibabaVideoWireRequest {
            model: self.model.as_str(),
            input: AlibabaVideoWireInput::from(&request),
            parameters: &request.parameters,
        };
        let response = self
            .runtime
            .transport
            .execute(
                create_plan(&body).map_err(|error| self.contextualize(error))?,
                options,
            )
            .await
            .map_err(|error| self.contextualize(error))?;
        self.decode_job_response(response, None, model_warnings(&self.model))
    }

    pub async fn poll_video(
        &self,
        job: &AlibabaVideoJob,
        options: CallOptions,
    ) -> Result<AlibabaVideoJob, Error> {
        self.validate_job(job)?;
        reject_provider_options(&options).map_err(|error| self.contextualize(error))?;
        let target = task_target(job.id()).map_err(|error| self.contextualize(error))?;
        let response = self
            .runtime
            .transport
            .execute(
                empty_plan(Method::GET, target, ReplaySafety::SemanticallyIdempotent)
                    .map_err(|error| self.contextualize(error))?,
                options,
            )
            .await
            .map_err(|error| self.contextualize(error))?;
        self.decode_job_response(response, Some(job.id()), job.warnings.clone())
    }

    pub async fn cancel_video(
        &self,
        job: &AlibabaVideoJob,
        options: CallOptions,
    ) -> Result<AlibabaVideoJob, Error> {
        self.validate_job(job)?;
        reject_provider_options(&options).map_err(|error| self.contextualize(error))?;
        let target = cancel_target(job.id()).map_err(|error| self.contextualize(error))?;
        let response = self
            .runtime
            .transport
            .execute(
                empty_plan(Method::POST, target, ReplaySafety::Never)
                    .map_err(|error| self.contextualize(error))?,
                options,
            )
            .await
            .map_err(|error| self.contextualize(error))?;
        self.decode_job_response(response, Some(job.id()), job.warnings.clone())
    }

    pub async fn materialize_video(
        &self,
        job: &AlibabaVideoJob,
        options: CallOptions,
    ) -> Result<Vec<u8>, Error> {
        self.validate_job(job)?;
        reject_provider_options(&options).map_err(|error| self.contextualize(error))?;
        if !job.status().is_successful() {
            return Err(self.contextualize(Error::new(
                ErrorKind::InvalidInput,
                "Alibaba video job must be completed before materialization",
            )));
        }
        let video_url = job.video_url().ok_or_else(|| {
            self.contextualize(Error::protocol_violation(
                "completed Alibaba video job has no materialization URL",
            ))
        })?;
        let resource = self
            .runtime
            .download_policy
            .resource(video_url)
            .map_err(resource_url_error)
            .map_err(|error| self.contextualize(error))?;
        let mut download_options =
            ResourceDownloadOptions::default().with_cancellation(options.cancellation().clone());
        if let Some(deadline) = options.deadline() {
            download_options = download_options.with_deadline(deadline);
        }
        let resource = self
            .runtime
            .downloader
            .download(resource, download_options)
            .await
            .map_err(|error| self.contextualize(error))?;
        Ok(resource.data().to_vec())
    }

    fn validate_job(&self, job: &AlibabaVideoJob) -> Result<(), Error> {
        if job.model_id() != self.model_id() {
            return Err(self.contextualize(Error::new(
                ErrorKind::InvalidInput,
                "Alibaba video job belongs to a different model handle",
            )));
        }
        Ok(())
    }

    fn decode_job_response(
        &self,
        response: TransportResponse,
        expected_job: Option<&JobId>,
        warnings: Vec<Warning>,
    ) -> Result<AlibabaVideoJob, Error> {
        if !response.status().is_success() {
            return Err(self.contextualize(provider_status_error(response)));
        }
        let header_request_id = response_request_id(response.headers());
        let decoded = serde_json::from_slice::<AlibabaVideoTaskResponse>(response.body())
            .map_err(video_response_error)
            .map_err(|error| self.contextualize(error))?;
        let output = decoded.output.ok_or_else(|| {
            self.contextualize(Error::protocol_violation(
                "Alibaba video response is missing task output",
            ))
        })?;
        let id = match (output.task_id, expected_job) {
            (Some(id), expected) => {
                let id = JobId::new(id).map_err(|error| {
                    self.contextualize(
                        Error::protocol_violation(
                            "Alibaba video response contains an invalid task identifier",
                        )
                        .with_source(error),
                    )
                })?;
                if expected.is_some_and(|expected| expected != &id) {
                    return Err(self.contextualize(Error::protocol_violation(
                        "Alibaba video response changed the task identifier",
                    )));
                }
                id
            }
            (None, Some(expected)) => expected.clone(),
            (None, None) => {
                return Err(self.contextualize(Error::protocol_violation(
                    "Alibaba video creation response is missing a task identifier",
                )));
            }
        };
        let status = decode_status(output.task_status.as_deref(), expected_job.is_none())
            .map_err(|error| self.contextualize(error))?;
        Ok(AlibabaVideoJob {
            id,
            model: self.model.clone(),
            status,
            request_id: safe_optional_text(decoded.request_id.or(header_request_id)),
            video_url: output.video_url.filter(|value| !value.trim().is_empty()),
            provider_code: safe_optional_text(output.code),
            usage: decoded.usage.unwrap_or_default(),
            warnings,
        })
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: None,
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model.clone()),
        })
    }
}

impl fmt::Debug for AlibabaVideoModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AlibabaVideoModel")
            .field("scope", &self.runtime.scope)
            .field("model", &self.model)
            .finish()
    }
}

#[async_trait]
impl VideoJobModel for AlibabaVideoModel {
    async fn create(&self, request: Value, options: CallOptions) -> Result<MediaJob, Error> {
        let request = serde_json::from_value::<AlibabaVideoRequest>(request).map_err(|source| {
            self.contextualize(
                Error::new(
                    ErrorKind::InvalidInput,
                    "dynamic Alibaba video request is invalid",
                )
                .with_source(source),
            )
        })?;
        self.create_video(request, options).await?.into_media_job()
    }

    async fn poll(&self, job: &MediaJob, options: CallOptions) -> Result<MediaJob, Error> {
        let job =
            AlibabaVideoJob::from_media_job(job).map_err(|error| self.contextualize(error))?;
        self.poll_video(&job, options).await?.into_media_job()
    }

    async fn cancel(&self, job: &MediaJob, options: CallOptions) -> Result<MediaJob, Error> {
        let job =
            AlibabaVideoJob::from_media_job(job).map_err(|error| self.contextualize(error))?;
        self.cancel_video(&job, options).await?.into_media_job()
    }

    async fn materialize(&self, job: &MediaJob, options: CallOptions) -> Result<Vec<u8>, Error> {
        let mut job =
            AlibabaVideoJob::from_media_job(job).map_err(|error| self.contextualize(error))?;
        if job.status().is_successful() {
            job = self.poll_video(&job, options.clone()).await?;
        }
        self.materialize_video(&job, options).await
    }
}

#[derive(Debug, Serialize)]
struct AlibabaVideoWireRequest<'a> {
    model: &'a str,
    input: AlibabaVideoWireInput<'a>,
    parameters: &'a AlibabaVideoParameters,
}

#[derive(Debug, Serialize)]
struct AlibabaVideoWireInput<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<&'a String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    negative_prompt: Option<&'a String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    media: &'a Vec<AlibabaVideoMedia>,
    #[serde(rename = "img_url", skip_serializing_if = "Option::is_none")]
    legacy_image_url: Option<&'a String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_url: Option<&'a String>,
}

impl<'a> From<&'a AlibabaVideoRequest> for AlibabaVideoWireInput<'a> {
    fn from(request: &'a AlibabaVideoRequest) -> Self {
        Self {
            prompt: request.prompt.as_ref(),
            negative_prompt: request.negative_prompt.as_ref(),
            media: &request.media,
            legacy_image_url: request.legacy_image_url.as_ref(),
            audio_url: request.audio_url.as_ref(),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct AlibabaVideoTaskResponse {
    #[serde(default)]
    request_id: Option<String>,
    #[serde(default)]
    output: Option<AlibabaVideoTaskOutput>,
    #[serde(default)]
    usage: Option<AlibabaVideoUsage>,
}

#[derive(Debug, Default, Deserialize)]
struct AlibabaVideoTaskOutput {
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    task_status: Option<String>,
    #[serde(default)]
    video_url: Option<String>,
    #[serde(default)]
    code: Option<String>,
}

fn create_plan(body: &impl Serialize) -> Result<RequestPlan, Error> {
    let headers = RequestHeaders::new()
        .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
        .and_then(|headers| headers.try_insert(ASYNC_HEADER, HeaderValue::from_static("enable")))
        .map_err(request_build_error)?;
    RequestPlan::new(
        Method::POST,
        RequestTarget::new(VIDEO_CREATE_TARGET).map_err(request_build_error)?,
    )
    .with_headers(headers)
    .with_body(RequestBody::json(body).map_err(request_build_error)?)
    .with_replay_safety(ReplaySafety::Never)
    .map_err(request_build_error)
}

fn empty_plan(
    method: Method,
    target: RequestTarget,
    replay_safety: ReplaySafety,
) -> Result<RequestPlan, Error> {
    let headers = RequestHeaders::new()
        .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
        .map_err(request_build_error)?;
    RequestPlan::new(method, target)
        .with_headers(headers)
        .with_replay_safety(replay_safety)
        .map_err(request_build_error)
}

fn task_target(id: &JobId) -> Result<RequestTarget, Error> {
    RequestTarget::new(format!("tasks/{}", id.as_str())).map_err(request_build_error)
}

fn cancel_target(id: &JobId) -> Result<RequestTarget, Error> {
    RequestTarget::new(format!("tasks/{}/cancel", id.as_str())).map_err(request_build_error)
}

fn decode_status(value: Option<&str>, creation: bool) -> Result<JobStatus, Error> {
    match value.map(|value| value.to_ascii_uppercase()) {
        Some(value) if value == "PENDING" || value == "QUEUED" => Ok(JobStatus::Queued),
        Some(value) if value == "RUNNING" || value == "PROCESSING" => Ok(JobStatus::Running),
        Some(value) if value == "SUCCEEDED" || value == "SUCCESS" => Ok(JobStatus::Completed),
        Some(value) if value == "FAILED" || value == "FAIL" => Ok(JobStatus::Failed {
            message: "Alibaba video job failed".to_string(),
        }),
        Some(value) if value == "CANCELED" || value == "CANCELLED" => Ok(JobStatus::Cancelled),
        Some(value) if value == "UNKNOWN" || value == "EXPIRED" => Ok(JobStatus::Expired),
        None if creation => Ok(JobStatus::Queued),
        _ => Err(Error::protocol_violation(
            "Alibaba video response contains an unknown task status",
        )),
    }
}

fn model_warnings(model: &ModelId) -> Vec<Warning> {
    if KNOWN_MODELS.contains(&model.as_str()) {
        Vec::new()
    } else {
        vec![Warning::new(
            WarningKind::UnknownModel,
            "model is absent from the verified Alibaba video advisory catalog",
        )]
    }
}

fn safe_optional_text(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        PublicDiagnosticText::new(value)
            .ok()
            .map(|value| value.as_str().to_owned())
    })
}

fn reject_provider_options(options: &CallOptions) -> Result<(), Error> {
    if options.has_provider_options() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Alibaba video job options must be expressed by AlibabaVideoRequest",
        ));
    }
    Ok(())
}

fn request_validation_error(source: AlibabaVideoRequestError) -> Error {
    Error::new(ErrorKind::InvalidInput, "Alibaba video request is invalid").with_source(source)
}

fn video_response_error(source: serde_json::Error) -> Error {
    Error::new(
        ErrorKind::Protocol,
        "Alibaba returned an invalid native video response",
    )
    .with_source(source)
}

fn job_state_error(source: serde_json::Error) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Alibaba video job state is invalid",
    )
    .with_source(source)
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Alibaba video request violates the transport contract",
    )
    .with_source(source)
}

fn resource_url_error(source: ResourceUrlError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Alibaba video materialization URL violates the configured resource policy",
    )
    .with_source(source)
}

fn provider_status_error(response: TransportResponse) -> Error {
    native_error::provider_status_error(
        response,
        "Alibaba rejected the native video request",
        |status| match status {
            StatusCode::BAD_REQUEST => ErrorKind::InvalidInput,
            StatusCode::UNAUTHORIZED => ErrorKind::Authentication,
            StatusCode::FORBIDDEN => ErrorKind::Authorization,
            StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT => ErrorKind::Timeout,
            StatusCode::TOO_MANY_REQUESTS => ErrorKind::RateLimited,
            _ => ErrorKind::Provider,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::{
        AlibabaVideoMedia, AlibabaVideoMediaType, AlibabaVideoParameters, AlibabaVideoRequest,
        AlibabaVideoRequestError,
    };

    #[test]
    fn request_validation_is_structural_and_future_model_safe() {
        let request = AlibabaVideoRequest::image("hello", "https://example.com/first.png")
            .with_media(AlibabaVideoMedia::new(
                AlibabaVideoMediaType::Other("future_media".to_string()),
                "https://example.com/future.bin",
            ))
            .with_parameters(
                AlibabaVideoParameters::new()
                    .with_resolution("1080P", "16:9")
                    .with_duration(5),
            );
        request.validate().unwrap();
        let debug = format!("{request:?}");
        assert!(!debug.contains("hello"));
        assert!(!debug.contains("first.png"));
    }

    #[test]
    fn request_validation_rejects_ambiguous_or_empty_wire_values() {
        let mut invalid = AlibabaVideoParameters::new().with_size("1280*720");
        invalid.resolution = Some("1080P".to_string());
        invalid.ratio = Some("16:9".to_string());
        assert_eq!(
            AlibabaVideoRequest::text("hello")
                .with_parameters(invalid)
                .validate(),
            Err(AlibabaVideoRequestError::ConflictingGeometry)
        );
        assert!(AlibabaVideoRequest::new().validate().is_err());
    }
}
