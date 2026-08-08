//! Provider-owned ARK video-generation task lifecycle.

use std::collections::BTreeMap;
use std::fmt;

use http::Method;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use siumai_core::{CallOptions, Error, ErrorKind, ModelId};
use siumai_transport::{ReplaySafety, RequestBody};
use url::form_urlencoded;

use crate::native::{SharedArkNativeRuntime, execute_json, target, validate_identifier};

const MAX_CONTENT_PARTS: usize = 64;
const MAX_TEXT_BYTES: usize = 64 * 1024;
const MAX_MEDIA_SOURCE_BYTES: usize = 16 * 1024 * 1024;
const MAX_PAGE_SIZE: u32 = 100;

/// Bounded ARK video task identifier.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArkVideoTaskId(String);

impl ArkVideoTaskId {
    pub fn new(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        validate_identifier(&value, "ARK video task identifier is invalid")?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ArkVideoTaskId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("ArkVideoTaskId")
            .field(&self.0)
            .finish()
    }
}

impl fmt::Display for ArkVideoTaskId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Serialize for ArkVideoTaskId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ArkVideoTaskId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Media source used by ARK task requests and responses.
#[derive(Clone, PartialEq, Eq)]
pub struct ArkMediaSource(String);

impl ArkMediaSource {
    pub fn remote(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        validate_media_source(&value, true)?;
        Ok(Self(value))
    }

    pub fn data_uri(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        validate_media_source(&value, false)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ArkMediaSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArkMediaSource")
            .field("value", &"<redacted>")
            .field("encoded_bytes", &self.0.len())
            .finish()
    }
}

impl Serialize for ArkMediaSource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ArkMediaSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value.trim().is_empty()
            || value != value.trim()
            || value.len() > MAX_MEDIA_SOURCE_BYTES
            || value.chars().any(char::is_control)
        {
            return Err(serde::de::Error::custom("invalid ARK media source"));
        }
        Ok(Self(value))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ArkVideoImageRole {
    FirstFrame,
    LastFrame,
    ReferenceImage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ArkVideoVideoRole {
    ReferenceVideo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ArkVideoAudioRole {
    ReferenceAudio,
}

/// Typed content item accepted by the ARK video task endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ArkVideoContent {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ArkMediaUrlWire,
        #[serde(skip_serializing_if = "Option::is_none")]
        role: Option<ArkVideoImageRole>,
    },
    VideoUrl {
        video_url: ArkMediaUrlWire,
        role: ArkVideoVideoRole,
    },
    AudioUrl {
        audio_url: ArkMediaUrlWire,
        role: ArkVideoAudioRole,
    },
}

impl ArkVideoContent {
    pub fn text(value: impl Into<String>) -> Result<Self, Error> {
        let text = value.into();
        validate_text(&text)?;
        Ok(Self::Text { text })
    }

    pub fn image(source: ArkMediaSource, role: Option<ArkVideoImageRole>) -> Self {
        Self::ImageUrl {
            image_url: ArkMediaUrlWire { url: source },
            role,
        }
    }

    pub fn reference_video(source: ArkMediaSource) -> Self {
        Self::VideoUrl {
            video_url: ArkMediaUrlWire { url: source },
            role: ArkVideoVideoRole::ReferenceVideo,
        }
    }

    pub fn reference_audio(source: ArkMediaSource) -> Self {
        Self::AudioUrl {
            audio_url: ArkMediaUrlWire { url: source },
            role: ArkVideoAudioRole::ReferenceAudio,
        }
    }

    fn validate(&self) -> Result<(), Error> {
        if let Self::Text { text } = self {
            validate_text(text)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ArkMediaUrlWire {
    pub url: ArkMediaSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ArkVideoResolution {
    #[serde(rename = "480p")]
    P480,
    #[serde(rename = "720p")]
    P720,
    #[serde(rename = "1080p")]
    P1080,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ArkVideoServiceTier {
    Default,
    Flex,
}

/// Full provider-owned ARK video task creation request.
#[derive(Clone, Serialize)]
pub struct ArkVideoCreateRequest {
    model: ModelId,
    content: Vec<ArkVideoContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    callback_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_last_frame: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<ArkVideoResolution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generate_audio: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    watermark: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    camera_fixed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<ArkVideoServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    draft: Option<bool>,
}

impl ArkVideoCreateRequest {
    pub fn new(
        model: impl Into<String>,
        content: impl IntoIterator<Item = ArkVideoContent>,
    ) -> Result<Self, Error> {
        let request = Self {
            model: ModelId::new(model.into()).map_err(|source| {
                invalid("ARK video model identifier is invalid").with_source(source)
            })?,
            content: content.into_iter().collect(),
            callback_url: None,
            return_last_frame: None,
            ratio: None,
            duration: None,
            seed: None,
            resolution: None,
            generate_audio: None,
            watermark: None,
            camera_fixed: None,
            service_tier: None,
            draft: None,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn text(model: impl Into<String>, prompt: impl Into<String>) -> Result<Self, Error> {
        Self::new(model, [ArkVideoContent::text(prompt)?])
    }

    pub fn with_callback_url(mut self, value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        validate_callback_url(&value)?;
        self.callback_url = Some(value);
        Ok(self)
    }

    pub const fn with_return_last_frame(mut self, value: bool) -> Self {
        self.return_last_frame = Some(value);
        self
    }

    pub fn with_ratio(mut self, value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        validate_small_string(&value, "ARK video ratio is invalid")?;
        self.ratio = Some(value);
        Ok(self)
    }

    pub fn with_duration(mut self, seconds: u32) -> Result<Self, Error> {
        if !(1..=60).contains(&seconds) {
            return Err(invalid("ARK video duration is invalid"));
        }
        self.duration = Some(seconds);
        Ok(self)
    }

    pub const fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub const fn with_resolution(mut self, resolution: ArkVideoResolution) -> Self {
        self.resolution = Some(resolution);
        self
    }

    pub const fn with_generate_audio(mut self, value: bool) -> Self {
        self.generate_audio = Some(value);
        self
    }

    pub const fn with_watermark(mut self, value: bool) -> Self {
        self.watermark = Some(value);
        self
    }

    pub const fn with_camera_fixed(mut self, value: bool) -> Self {
        self.camera_fixed = Some(value);
        self
    }

    pub const fn with_service_tier(mut self, value: ArkVideoServiceTier) -> Self {
        self.service_tier = Some(value);
        self
    }

    pub const fn with_draft(mut self, value: bool) -> Self {
        self.draft = Some(value);
        self
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn content(&self) -> &[ArkVideoContent] {
        &self.content
    }

    fn validate(&self) -> Result<(), Error> {
        if self.content.is_empty() || self.content.len() > MAX_CONTENT_PARTS {
            return Err(invalid("ARK video content count is invalid"));
        }
        for part in &self.content {
            part.validate()?;
        }
        if let Some(callback_url) = self.callback_url.as_deref() {
            validate_callback_url(callback_url)?;
        }
        if let Some(ratio) = self.ratio.as_deref() {
            validate_small_string(ratio, "ARK video ratio is invalid")?;
        }
        Ok(())
    }
}

impl fmt::Debug for ArkVideoCreateRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArkVideoCreateRequest")
            .field("model", &self.model)
            .field("content", &self.content)
            .field("has_callback_url", &self.callback_url.is_some())
            .field("return_last_frame", &self.return_last_frame)
            .field("ratio", &self.ratio)
            .field("duration", &self.duration)
            .field("seed", &self.seed)
            .field("resolution", &self.resolution)
            .field("generate_audio", &self.generate_audio)
            .field("watermark", &self.watermark)
            .field("camera_fixed", &self.camera_fixed)
            .field("service_tier", &self.service_tier)
            .field("draft", &self.draft)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ArkVideoCreateResult {
    pub id: ArkVideoTaskId,
}

/// Open status carrier that preserves future provider values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ArkVideoTaskStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Expired,
    Cancelled,
    Other(String),
}

impl ArkVideoTaskStatus {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Expired => "expired",
            Self::Cancelled => "cancelled",
            Self::Other(value) => value,
        }
    }

    fn from_wire(value: String) -> Result<Self, Error> {
        validate_small_string(&value, "ARK video task status is invalid")?;
        Ok(match value.as_str() {
            "queued" => Self::Queued,
            "running" => Self::Running,
            "succeeded" => Self::Succeeded,
            "failed" => Self::Failed,
            "expired" => Self::Expired,
            "cancelled" => Self::Cancelled,
            _ => Self::Other(value),
        })
    }
}

impl Serialize for ArkVideoTaskStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ArkVideoTaskStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, PartialEq, Deserialize)]
pub struct ArkVideoUsage {
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl fmt::Debug for ArkVideoUsage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArkVideoUsage")
            .field("completion_tokens", &self.completion_tokens)
            .field("extra_fields", &self.extra.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Generated media URLs. URL contents are redacted from `Debug` by `ArkMediaSource`.
#[derive(Clone, PartialEq, Deserialize)]
pub struct ArkVideoTaskContent {
    #[serde(default)]
    pub video_url: Option<ArkMediaSource>,
    #[serde(default)]
    pub last_frame_url: Option<ArkMediaSource>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl fmt::Debug for ArkVideoTaskContent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArkVideoTaskContent")
            .field("has_video_url", &self.video_url.is_some())
            .field("has_last_frame_url", &self.last_frame_url.is_some())
            .field("extra_fields", &self.extra.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Current ARK video task representation with additive response fields retained.
#[derive(Clone, PartialEq, Deserialize)]
pub struct ArkVideoTask {
    pub id: ArkVideoTaskId,
    #[serde(default)]
    pub model: Option<ModelId>,
    pub status: ArkVideoTaskStatus,
    #[serde(default)]
    pub content: Option<ArkVideoTaskContent>,
    #[serde(default)]
    pub usage: Option<ArkVideoUsage>,
    #[serde(default)]
    pub error: Option<Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl fmt::Debug for ArkVideoTask {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArkVideoTask")
            .field("id", &self.id)
            .field("model", &self.model)
            .field("status", &self.status)
            .field("content", &self.content)
            .field("usage", &self.usage)
            .field("has_error", &self.error.is_some())
            .field("extra_fields", &self.extra.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Bounded query for the ARK video task list endpoint.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArkVideoTaskListQuery {
    task_ids: Vec<ArkVideoTaskId>,
    status: Option<ArkVideoTaskStatus>,
    page_num: Option<u32>,
    page_size: Option<u32>,
}

impl ArkVideoTaskListQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_task_ids(
        mut self,
        task_ids: impl IntoIterator<Item = ArkVideoTaskId>,
    ) -> Result<Self, Error> {
        self.task_ids = task_ids.into_iter().collect();
        if self.task_ids.len() > MAX_PAGE_SIZE as usize {
            return Err(invalid("ARK video task filter is too large"));
        }
        Ok(self)
    }

    pub fn with_status(mut self, status: ArkVideoTaskStatus) -> Result<Self, Error> {
        if matches!(status, ArkVideoTaskStatus::Other(_)) {
            return Err(invalid("ARK video task status filter is unsupported"));
        }
        self.status = Some(status);
        Ok(self)
    }

    pub fn with_page(mut self, page_num: u32, page_size: u32) -> Result<Self, Error> {
        if page_num == 0 || page_size == 0 || page_size > MAX_PAGE_SIZE {
            return Err(invalid("ARK video task pagination is invalid"));
        }
        self.page_num = Some(page_num);
        self.page_size = Some(page_size);
        Ok(self)
    }

    fn target(&self) -> Result<siumai_transport::RequestTarget, Error> {
        let mut query = form_urlencoded::Serializer::new(String::new());
        if !self.task_ids.is_empty() {
            query.append_pair(
                "task_ids",
                &self
                    .task_ids
                    .iter()
                    .map(ArkVideoTaskId::as_str)
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        if let Some(status) = &self.status {
            query.append_pair("status", status.as_str());
        }
        if let Some(page_num) = self.page_num {
            query.append_pair("page_num", &page_num.to_string());
        }
        if let Some(page_size) = self.page_size {
            query.append_pair("page_size", &page_size.to_string());
        }
        let query = query.finish();
        if query.is_empty() {
            target("contents/generations/tasks")
        } else {
            target(format!("contents/generations/tasks?{query}"))
        }
    }
}

#[derive(Clone, PartialEq, Deserialize)]
pub struct ArkVideoTaskList {
    #[serde(default)]
    pub items: Vec<ArkVideoTask>,
    #[serde(default)]
    pub total: Option<u64>,
    #[serde(default)]
    pub page_num: Option<u32>,
    #[serde(default)]
    pub page_size: Option<u32>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl fmt::Debug for ArkVideoTaskList {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArkVideoTaskList")
            .field("items", &self.items)
            .field("total", &self.total)
            .field("page_num", &self.page_num)
            .field("page_size", &self.page_size)
            .field("extra_fields", &self.extra.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArkVideoDeleteResult {
    pub task_id: ArkVideoTaskId,
}

/// Shared ARK video task client.
#[derive(Clone)]
pub struct ArkVideoTasks {
    runtime: SharedArkNativeRuntime,
}

impl ArkVideoTasks {
    pub(crate) fn new(runtime: SharedArkNativeRuntime) -> Self {
        Self { runtime }
    }

    pub async fn create(
        &self,
        request: ArkVideoCreateRequest,
    ) -> Result<ArkVideoCreateResult, Error> {
        self.create_with_options(request, CallOptions::default())
            .await
    }

    pub async fn create_with_options(
        &self,
        request: ArkVideoCreateRequest,
        options: CallOptions,
    ) -> Result<ArkVideoCreateResult, Error> {
        request.validate()?;
        execute_json(
            &self.runtime,
            Method::POST,
            target("contents/generations/tasks")?,
            RequestBody::json(&request).map_err(request_error)?,
            ReplaySafety::Never,
            options,
        )
        .await
    }

    pub async fn retrieve(&self, task_id: &ArkVideoTaskId) -> Result<ArkVideoTask, Error> {
        self.retrieve_with_options(task_id, CallOptions::default())
            .await
    }

    pub async fn retrieve_with_options(
        &self,
        task_id: &ArkVideoTaskId,
        options: CallOptions,
    ) -> Result<ArkVideoTask, Error> {
        execute_json(
            &self.runtime,
            Method::GET,
            target(format!("contents/generations/tasks/{task_id}"))?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            options,
        )
        .await
    }

    pub async fn list(&self, query: &ArkVideoTaskListQuery) -> Result<ArkVideoTaskList, Error> {
        self.list_with_options(query, CallOptions::default()).await
    }

    pub async fn list_with_options(
        &self,
        query: &ArkVideoTaskListQuery,
        options: CallOptions,
    ) -> Result<ArkVideoTaskList, Error> {
        execute_json(
            &self.runtime,
            Method::GET,
            query.target()?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            options,
        )
        .await
    }

    pub async fn delete(&self, task_id: &ArkVideoTaskId) -> Result<ArkVideoDeleteResult, Error> {
        self.delete_with_options(task_id, CallOptions::default())
            .await
    }

    pub async fn delete_with_options(
        &self,
        task_id: &ArkVideoTaskId,
        options: CallOptions,
    ) -> Result<ArkVideoDeleteResult, Error> {
        let _: Value = execute_json(
            &self.runtime,
            Method::DELETE,
            target(format!("contents/generations/tasks/{task_id}"))?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            options,
        )
        .await?;
        Ok(ArkVideoDeleteResult {
            task_id: task_id.clone(),
        })
    }
}

impl fmt::Debug for ArkVideoTasks {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArkVideoTasks")
            .field("runtime", &"shared")
            .finish()
    }
}

fn validate_text(value: &str) -> Result<(), Error> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_TEXT_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(invalid("ARK video text content is invalid"));
    }
    Ok(())
}

fn validate_callback_url(value: &str) -> Result<(), Error> {
    let parsed = url::Url::parse(value)
        .map_err(|source| invalid("ARK video callback URL is invalid").with_source(source))?;
    if parsed.scheme() != "https"
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.fragment().is_some()
        || value.len() > 16 * 1024
    {
        return Err(invalid("ARK video callback URL is invalid"));
    }
    Ok(())
}

fn validate_media_source(value: &str, remote: bool) -> Result<(), Error> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_MEDIA_SOURCE_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(invalid("ARK media source is invalid"));
    }
    let valid = if remote {
        url::Url::parse(value)
            .ok()
            .is_some_and(|url| url.scheme() == "https" && url.host_str().is_some())
    } else {
        value.starts_with("data:") && value.contains(";base64,")
    };
    if !valid {
        return Err(invalid("ARK media source has an unsupported scheme"));
    }
    Ok(())
}

fn validate_small_string(value: &str, message: &'static str) -> Result<(), Error> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > 128
        || value.chars().any(char::is_control)
    {
        return Err(invalid(message));
    }
    Ok(())
}

fn request_error(source: siumai_transport::RequestBuildError) -> Error {
    Error::new(
        ErrorKind::Configuration,
        "ARK video request violates the transport contract",
    )
    .with_source(source)
}

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}
