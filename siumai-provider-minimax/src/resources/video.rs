use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use http::Method;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{CallOptions, Error, ErrorKind};
use siumai_transport::{ReplaySafety, RequestBody};

use super::common::{BaseResponse, NativeResponseEnvelope, NativeRuntime, execute_json, target};

const CREATE_TARGET: &str = "v2/video_generation";
const QUERY_TARGET: &str = "v2/query/video_generation";
const DELETE_TARGET: &str = "v2/video_generation";
const MAX_MODEL_BYTES: usize = 256;
const MAX_PROMPT_CHARACTERS: usize = 7_000;
const MAX_TASK_ID_BYTES: usize = 256;
const MAX_MEDIA_REFERENCE_BYTES: usize = 64 * 1024 * 1024;
const MAX_CALLBACK_BYTES: usize = 8 * 1024;
const MAX_REFERENCE_IMAGES: usize = 9;
const MAX_REFERENCE_VIDEOS: usize = 3;
const MAX_REFERENCE_AUDIO: usize = 3;
const MAX_LIST_TASK_IDS: usize = 100;

/// Valid output resolution for MiniMax H3 video generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxVideoResolution {
    P768,
    K2,
}

impl MinimaxVideoResolution {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::P768 => "768P",
            Self::K2 => "2K",
        }
    }
}

/// Aspect ratio accepted by the MiniMax H3 V2 API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxVideoRatio {
    Adaptive,
    Wide21By9,
    Landscape16By9,
    Landscape4By3,
    Square,
    Portrait3By4,
    Portrait9By16,
}

impl MinimaxVideoRatio {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::Wide21By9 => "21:9",
            Self::Landscape16By9 => "16:9",
            Self::Landscape4By3 => "4:3",
            Self::Square => "1:1",
            Self::Portrait3By4 => "3:4",
            Self::Portrait9By16 => "9:16",
        }
    }
}

/// Provider media reference used by H3 V2 multimodal inputs.
///
/// The value may be a public URL, a supported data URI, or an `mm_file://`
/// reference. Its contents are intentionally omitted from `Debug`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct MinimaxVideoMediaSource(String);

impl MinimaxVideoMediaSource {
    pub fn new(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        validate_bounded_text(
            &value,
            MAX_MEDIA_REFERENCE_BYTES,
            "MiniMax video media reference",
        )?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for MinimaxVideoMediaSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVideoMediaSource")
            .field("bytes", &self.0.len())
            .finish()
    }
}

/// One non-text H3 V2 input.
#[derive(Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MinimaxVideoInput {
    FirstFrame(MinimaxVideoMediaSource),
    LastFrame(MinimaxVideoMediaSource),
    ReferenceImage(MinimaxVideoMediaSource),
    ReferenceVideo(MinimaxVideoMediaSource),
    ReferenceAudio(MinimaxVideoMediaSource),
}

impl fmt::Debug for MinimaxVideoInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::FirstFrame(_) => "MinimaxVideoInput::FirstFrame([REDACTED])",
            Self::LastFrame(_) => "MinimaxVideoInput::LastFrame([REDACTED])",
            Self::ReferenceImage(_) => "MinimaxVideoInput::ReferenceImage([REDACTED])",
            Self::ReferenceVideo(_) => "MinimaxVideoInput::ReferenceVideo([REDACTED])",
            Self::ReferenceAudio(_) => "MinimaxVideoInput::ReferenceAudio([REDACTED])",
        })
    }
}

/// One H3 V2 video-generation request.
#[derive(Clone, PartialEq, Eq)]
pub struct MinimaxVideoRequest {
    model: String,
    prompt: String,
    resolution: MinimaxVideoResolution,
    duration_seconds: u8,
    ratio: Option<MinimaxVideoRatio>,
    inputs: Vec<MinimaxVideoInput>,
    callback_url: Option<String>,
}

impl MinimaxVideoRequest {
    pub fn new(
        model: impl Into<String>,
        prompt: impl Into<String>,
        resolution: MinimaxVideoResolution,
        duration_seconds: u8,
    ) -> Result<Self, Error> {
        let request = Self {
            model: model.into(),
            prompt: prompt.into(),
            resolution,
            duration_seconds,
            ratio: None,
            inputs: Vec::new(),
            callback_url: None,
        };
        request.validate_basic()?;
        Ok(request)
    }

    pub fn with_ratio(mut self, ratio: MinimaxVideoRatio) -> Self {
        self.ratio = Some(ratio);
        self
    }

    pub fn with_input(mut self, input: MinimaxVideoInput) -> Self {
        self.inputs.push(input);
        self
    }

    pub fn with_callback_url(mut self, callback_url: impl Into<String>) -> Result<Self, Error> {
        let callback_url = callback_url.into();
        validate_bounded_text(
            &callback_url,
            MAX_CALLBACK_BYTES,
            "MiniMax video callback URL",
        )?;
        let parsed = url::Url::parse(&callback_url).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "MiniMax video callback URL is invalid",
            )
            .with_source(source)
        })?;
        if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "MiniMax video callback URL must be an absolute HTTP or HTTPS URL",
            ));
        }
        self.callback_url = Some(callback_url);
        Ok(self)
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    pub const fn resolution(&self) -> MinimaxVideoResolution {
        self.resolution
    }

    pub const fn duration_seconds(&self) -> u8 {
        self.duration_seconds
    }

    pub const fn ratio(&self) -> Option<MinimaxVideoRatio> {
        self.ratio
    }

    pub fn inputs(&self) -> &[MinimaxVideoInput] {
        &self.inputs
    }

    pub fn callback_url(&self) -> Option<&str> {
        self.callback_url.as_deref()
    }

    fn validate_basic(&self) -> Result<(), Error> {
        validate_bounded_text(&self.model, MAX_MODEL_BYTES, "MiniMax video model")?;
        if self.prompt.trim().is_empty()
            || self.prompt.chars().count() > MAX_PROMPT_CHARACTERS
            || self.prompt.chars().any(char::is_control)
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "MiniMax video prompt must be non-empty, printable, and at most 7000 characters",
            ));
        }
        if !(4..=15).contains(&self.duration_seconds) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "MiniMax H3 video duration must be between 4 and 15 seconds",
            ));
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), Error> {
        self.validate_basic()?;
        let mut first_frames = 0;
        let mut last_frames = 0;
        let mut reference_images = 0;
        let mut reference_videos = 0;
        let mut reference_audio = 0;
        for input in &self.inputs {
            match input {
                MinimaxVideoInput::FirstFrame(_) => first_frames += 1,
                MinimaxVideoInput::LastFrame(_) => last_frames += 1,
                MinimaxVideoInput::ReferenceImage(_) => reference_images += 1,
                MinimaxVideoInput::ReferenceVideo(_) => reference_videos += 1,
                MinimaxVideoInput::ReferenceAudio(_) => reference_audio += 1,
            }
        }

        if first_frames > 1 || last_frames > 1 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "MiniMax H3 accepts at most one first frame and one last frame",
            ));
        }
        if last_frames > 0 && first_frames == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "MiniMax H3 requires a first frame when a last frame is supplied",
            ));
        }
        if reference_images > MAX_REFERENCE_IMAGES
            || reference_videos > MAX_REFERENCE_VIDEOS
            || reference_audio > MAX_REFERENCE_AUDIO
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "MiniMax H3 reference input count exceeds the provider limit",
            ));
        }

        let uses_frames = first_frames + last_frames > 0;
        let uses_references = reference_images + reference_videos + reference_audio > 0;
        if uses_frames && uses_references {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "MiniMax H3 frame inputs and reference inputs are mutually exclusive",
            ));
        }
        if uses_frames
            && self
                .ratio
                .is_some_and(|ratio| ratio != MinimaxVideoRatio::Adaptive)
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "MiniMax H3 frame-based generation requires the adaptive ratio",
            ));
        }
        if !uses_frames && !uses_references {
            match self.ratio {
                Some(MinimaxVideoRatio::Adaptive) | None => {
                    return Err(Error::new(
                        ErrorKind::InvalidInput,
                        "MiniMax H3 text-to-video requires a concrete aspect ratio",
                    ));
                }
                Some(_) => {}
            }
        }
        Ok(())
    }

    fn wire(&self) -> Result<CreateVideoRequest<'_>, Error> {
        self.validate()?;
        let mut content = Vec::with_capacity(self.inputs.len() + 1);
        content.push(VideoContent::Text { text: &self.prompt });
        for input in &self.inputs {
            content.push(match input {
                MinimaxVideoInput::FirstFrame(source) => VideoContent::ImageUrl {
                    image_url: MediaUrl {
                        url: source.as_str(),
                    },
                    role: "first_frame",
                },
                MinimaxVideoInput::LastFrame(source) => VideoContent::ImageUrl {
                    image_url: MediaUrl {
                        url: source.as_str(),
                    },
                    role: "last_frame",
                },
                MinimaxVideoInput::ReferenceImage(source) => VideoContent::ImageUrl {
                    image_url: MediaUrl {
                        url: source.as_str(),
                    },
                    role: "reference_image",
                },
                MinimaxVideoInput::ReferenceVideo(source) => VideoContent::VideoUrl {
                    video_url: MediaUrl {
                        url: source.as_str(),
                    },
                    role: "reference_video",
                },
                MinimaxVideoInput::ReferenceAudio(source) => VideoContent::AudioUrl {
                    audio_url: MediaUrl {
                        url: source.as_str(),
                    },
                    role: "reference_audio",
                },
            });
        }
        Ok(CreateVideoRequest {
            model: &self.model,
            content,
            resolution: self.resolution.as_str(),
            duration: self.duration_seconds,
            ratio: self.ratio.map(MinimaxVideoRatio::as_str),
            callback_url: self.callback_url.as_deref(),
        })
    }
}

impl fmt::Debug for MinimaxVideoRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVideoRequest")
            .field("model", &self.model)
            .field("prompt_characters", &self.prompt.chars().count())
            .field("resolution", &self.resolution)
            .field("duration_seconds", &self.duration_seconds)
            .field("ratio", &self.ratio)
            .field("input_count", &self.inputs.len())
            .field("callback_url_present", &self.callback_url.is_some())
            .finish()
    }
}

#[derive(Serialize)]
struct CreateVideoRequest<'a> {
    model: &'a str,
    content: Vec<VideoContent<'a>>,
    resolution: &'static str,
    duration: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    ratio: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    callback_url: Option<&'a str>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum VideoContent<'a> {
    Text {
        text: &'a str,
    },
    ImageUrl {
        image_url: MediaUrl<'a>,
        role: &'static str,
    },
    VideoUrl {
        video_url: MediaUrl<'a>,
        role: &'static str,
    },
    AudioUrl {
        audio_url: MediaUrl<'a>,
        role: &'static str,
    },
}

#[derive(Serialize)]
struct MediaUrl<'a> {
    url: &'a str,
}

/// Validated MiniMax H3 task identifier.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MinimaxVideoTaskId(String);

impl MinimaxVideoTaskId {
    pub fn new(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_TASK_ID_BYTES
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "MiniMax video task identifier is invalid",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for MinimaxVideoTaskId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("MinimaxVideoTaskId")
            .field(&self.0)
            .finish()
    }
}

impl fmt::Display for MinimaxVideoTaskId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Open task status returned by MiniMax H3 V2.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxVideoTaskStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Other(String),
}

impl MinimaxVideoTaskStatus {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Other(value) => value,
        }
    }

    fn from_wire(value: String) -> Self {
        match value.as_str() {
            "queued" => Self::Queued,
            "running" => Self::Running,
            "succeeded" => Self::Succeeded,
            "failed" => Self::Failed,
            "cancelled" => Self::Cancelled,
            _ => Self::Other(value),
        }
    }
}

/// Open H3 V2 task type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxVideoTaskType {
    Generation,
    ContextIr,
    Regeneration,
    Other(String),
}

impl MinimaxVideoTaskType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Generation => "generation",
            Self::ContextIr => "h3_context_ir",
            Self::Regeneration => "regeneration",
            Self::Other(value) => value,
        }
    }

    fn from_wire(value: String) -> Self {
        match value.as_str() {
            "generation" => Self::Generation,
            "h3_context_ir" => Self::ContextIr,
            "regeneration" => Self::Regeneration,
            _ => Self::Other(value),
        }
    }
}

/// H3 task output. URLs and generated prompts are omitted from `Debug`.
#[derive(Clone, PartialEq)]
pub struct MinimaxVideoTaskContent {
    url: Option<String>,
    prompt: Option<String>,
    extra: BTreeMap<String, Value>,
}

impl MinimaxVideoTaskContent {
    pub fn url(&self) -> Option<&str> {
        self.url.as_deref()
    }

    pub fn prompt(&self) -> Option<&str> {
        self.prompt.as_deref()
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxVideoTaskContent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVideoTaskContent")
            .field("url_present", &self.url.is_some())
            .field("prompt_present", &self.prompt.is_some())
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Provider-reported task failure. The message is available explicitly but is
/// omitted from `Debug` because it may contain submitted content.
#[derive(Clone, PartialEq, Eq)]
pub struct MinimaxVideoTaskFailure {
    code: Option<String>,
    message: Option<String>,
}

impl MinimaxVideoTaskFailure {
    pub fn code(&self) -> Option<&str> {
        self.code.as_deref()
    }

    pub fn message(&self) -> Option<&str> {
        self.message.as_deref()
    }
}

impl fmt::Debug for MinimaxVideoTaskFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVideoTaskFailure")
            .field("code", &self.code)
            .field("message_present", &self.message.is_some())
            .finish()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct MinimaxVideoUsage {
    #[serde(default)]
    total_seconds: Option<u64>,
    #[serde(default)]
    input_seconds: Option<u64>,
    #[serde(default)]
    output_seconds: Option<u64>,
    #[serde(default)]
    input_image_count: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
}

impl MinimaxVideoUsage {
    pub const fn total_seconds(&self) -> Option<u64> {
        self.total_seconds
    }

    pub const fn input_seconds(&self) -> Option<u64> {
        self.input_seconds
    }

    pub const fn output_seconds(&self) -> Option<u64> {
        self.output_seconds
    }

    pub const fn input_image_count(&self) -> Option<u64> {
        self.input_image_count
    }

    pub const fn total_tokens(&self) -> Option<u64> {
        self.total_tokens
    }

    pub const fn prompt_tokens(&self) -> Option<u64> {
        self.prompt_tokens
    }

    pub const fn completion_tokens(&self) -> Option<u64> {
        self.completion_tokens
    }
}

/// One H3 V2 task returned by query or list.
#[derive(Clone, PartialEq)]
pub struct MinimaxVideoTask {
    id: MinimaxVideoTaskId,
    model: Option<String>,
    status: MinimaxVideoTaskStatus,
    failure: Option<MinimaxVideoTaskFailure>,
    created_at: Option<i64>,
    updated_at: Option<i64>,
    content: Option<MinimaxVideoTaskContent>,
    resolution: Option<String>,
    duration_seconds: Option<u8>,
    usage: Option<MinimaxVideoUsage>,
    ratio: Option<String>,
    task_type: Option<MinimaxVideoTaskType>,
    modality: Option<String>,
    extra: BTreeMap<String, Value>,
}

impl MinimaxVideoTask {
    pub fn id(&self) -> &MinimaxVideoTaskId {
        &self.id
    }

    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    pub fn status(&self) -> &MinimaxVideoTaskStatus {
        &self.status
    }

    pub fn failure(&self) -> Option<&MinimaxVideoTaskFailure> {
        self.failure.as_ref()
    }

    pub const fn created_at_unix_seconds(&self) -> Option<i64> {
        self.created_at
    }

    pub const fn updated_at_unix_seconds(&self) -> Option<i64> {
        self.updated_at
    }

    pub fn content(&self) -> Option<&MinimaxVideoTaskContent> {
        self.content.as_ref()
    }

    pub fn output_url(&self) -> Option<&str> {
        self.content.as_ref().and_then(MinimaxVideoTaskContent::url)
    }

    pub fn resolution(&self) -> Option<&str> {
        self.resolution.as_deref()
    }

    pub const fn duration_seconds(&self) -> Option<u8> {
        self.duration_seconds
    }

    pub const fn usage(&self) -> Option<&MinimaxVideoUsage> {
        self.usage.as_ref()
    }

    pub fn ratio(&self) -> Option<&str> {
        self.ratio.as_deref()
    }

    pub fn task_type(&self) -> Option<&MinimaxVideoTaskType> {
        self.task_type.as_ref()
    }

    pub fn modality(&self) -> Option<&str> {
        self.modality.as_deref()
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxVideoTask {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVideoTask")
            .field("id", &self.id)
            .field("model", &self.model)
            .field("status", &self.status)
            .field("failure", &self.failure)
            .field("created_at", &self.created_at)
            .field("updated_at", &self.updated_at)
            .field("content", &self.content)
            .field("resolution", &self.resolution)
            .field("duration_seconds", &self.duration_seconds)
            .field("usage", &self.usage)
            .field("ratio", &self.ratio)
            .field("task_type", &self.task_type)
            .field("modality", &self.modality)
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Result of creating one H3 V2 task.
#[derive(Clone, PartialEq)]
pub struct MinimaxVideoCreateResult {
    task_id: MinimaxVideoTaskId,
    extra: BTreeMap<String, Value>,
}

impl MinimaxVideoCreateResult {
    pub fn task_id(&self) -> &MinimaxVideoTaskId {
        &self.task_id
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxVideoCreateResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVideoCreateResult")
            .field("task_id", &self.task_id)
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Bounded query for the H3 V2 task list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MinimaxVideoListQuery {
    page: Option<u32>,
    page_size: Option<u32>,
    status: Option<MinimaxVideoTaskStatus>,
    task_ids: Vec<MinimaxVideoTaskId>,
    model: Option<String>,
    task_type: Option<MinimaxVideoTaskType>,
}

impl MinimaxVideoListQuery {
    pub const fn new() -> Self {
        Self {
            page: None,
            page_size: None,
            status: None,
            task_ids: Vec::new(),
            model: None,
            task_type: None,
        }
    }

    pub fn with_page(mut self, page: u32, page_size: u32) -> Result<Self, Error> {
        if page == 0 || page_size == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "MiniMax video list page and page size must be positive",
            ));
        }
        self.page = Some(page);
        self.page_size = Some(page_size);
        Ok(self)
    }

    pub fn with_status(mut self, status: MinimaxVideoTaskStatus) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_task_id(mut self, task_id: MinimaxVideoTaskId) -> Result<Self, Error> {
        if self.task_ids.len() >= MAX_LIST_TASK_IDS {
            return Err(Error::new(
                ErrorKind::LimitExceeded,
                "MiniMax video task filter exceeds the local item limit",
            ));
        }
        self.task_ids.push(task_id);
        Ok(self)
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Result<Self, Error> {
        let model = model.into();
        validate_bounded_text(&model, MAX_MODEL_BYTES, "MiniMax video model filter")?;
        self.model = Some(model);
        Ok(self)
    }

    pub fn with_task_type(mut self, task_type: MinimaxVideoTaskType) -> Self {
        self.task_type = Some(task_type);
        self
    }

    fn target(&self) -> Result<siumai_transport::RequestTarget, Error> {
        let mut query = url::form_urlencoded::Serializer::new(String::new());
        if let Some(page) = self.page {
            query.append_pair("page_num", &page.to_string());
        }
        if let Some(page_size) = self.page_size {
            query.append_pair("page_size", &page_size.to_string());
        }
        if let Some(status) = &self.status {
            query.append_pair("filter.status", status.as_str());
        }
        for task_id in &self.task_ids {
            query.append_pair("filter.task_ids", task_id.as_str());
        }
        if let Some(model) = &self.model {
            query.append_pair("filter.model", model);
        }
        if let Some(task_type) = &self.task_type {
            query.append_pair("filter.task_type", task_type.as_str());
        }
        let query = query.finish();
        if query.is_empty() {
            target(QUERY_TARGET)
        } else {
            target(format!("{QUERY_TARGET}?{query}"))
        }
    }
}

/// One page of H3 V2 tasks.
#[derive(Clone, PartialEq)]
pub struct MinimaxVideoTaskList {
    items: Vec<MinimaxVideoTask>,
    total: Option<u64>,
    extra: BTreeMap<String, Value>,
}

impl MinimaxVideoTaskList {
    pub fn items(&self) -> &[MinimaxVideoTask] {
        &self.items
    }

    pub fn into_items(self) -> Vec<MinimaxVideoTask> {
        self.items
    }

    pub const fn total(&self) -> Option<u64> {
        self.total
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxVideoTaskList {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVideoTaskList")
            .field("item_count", &self.items.len())
            .field("total", &self.total)
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Server-selected cancel or delete action.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxVideoDeleteAction {
    Cancelled,
    Deleted,
    Other(String),
}

impl MinimaxVideoDeleteAction {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Cancelled => "cancelled",
            Self::Deleted => "deleted",
            Self::Other(value) => value,
        }
    }

    fn from_wire(value: String) -> Self {
        match value.as_str() {
            "cancelled" => Self::Cancelled,
            "deleted" => Self::Deleted,
            _ => Self::Other(value),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MinimaxVideoDeleteResult {
    task_id: MinimaxVideoTaskId,
    action: MinimaxVideoDeleteAction,
    status: Option<String>,
    extra: BTreeMap<String, Value>,
}

impl MinimaxVideoDeleteResult {
    pub fn task_id(&self) -> &MinimaxVideoTaskId {
        &self.task_id
    }

    pub fn action(&self) -> &MinimaxVideoDeleteAction {
        &self.action
    }

    pub fn status(&self) -> Option<&str> {
        self.status.as_deref()
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

/// Shared, lightweight handle for MiniMax H3 V2 task operations.
#[derive(Clone)]
pub struct MinimaxVideo {
    runtime: Arc<NativeRuntime>,
}

impl MinimaxVideo {
    pub(crate) fn new(runtime: Arc<NativeRuntime>) -> Self {
        Self { runtime }
    }

    pub async fn create(
        &self,
        request: MinimaxVideoRequest,
    ) -> Result<MinimaxVideoCreateResult, Error> {
        self.create_with_options(request, CallOptions::default())
            .await
    }

    pub async fn create_with_options(
        &self,
        request: MinimaxVideoRequest,
        options: CallOptions,
    ) -> Result<MinimaxVideoCreateResult, Error> {
        let body = RequestBody::json(&request.wire()?).map_err(|source| {
            Error::new(
                ErrorKind::InvalidInput,
                "MiniMax video request could not be encoded",
            )
            .with_source(source)
        })?;
        let response: CreateEnvelope = execute_json(
            &self.runtime,
            Method::POST,
            target(CREATE_TARGET)?,
            body,
            ReplaySafety::Never,
            options,
        )
        .await?;
        response.into_result()
    }

    pub async fn query(&self, task_id: &MinimaxVideoTaskId) -> Result<MinimaxVideoTask, Error> {
        self.query_with_options(task_id, CallOptions::default())
            .await
    }

    pub async fn query_with_options(
        &self,
        task_id: &MinimaxVideoTaskId,
        options: CallOptions,
    ) -> Result<MinimaxVideoTask, Error> {
        let response: QueryEnvelope = execute_json(
            &self.runtime,
            Method::GET,
            task_target(QUERY_TARGET, task_id)?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            options,
        )
        .await?;
        let task = response.task.ok_or_else(|| {
            Error::new(
                ErrorKind::Protocol,
                "MiniMax video query response omitted task",
            )
        })?;
        let task = task.into_task()?;
        if task.id() != task_id {
            return Err(Error::new(
                ErrorKind::ProtocolViolation,
                "MiniMax video query returned a different task identifier",
            ));
        }
        Ok(task)
    }

    pub async fn list(&self, query: &MinimaxVideoListQuery) -> Result<MinimaxVideoTaskList, Error> {
        self.list_with_options(query, CallOptions::default()).await
    }

    pub async fn list_with_options(
        &self,
        query: &MinimaxVideoListQuery,
        options: CallOptions,
    ) -> Result<MinimaxVideoTaskList, Error> {
        let response: ListEnvelope = execute_json(
            &self.runtime,
            Method::GET,
            query.target()?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            options,
        )
        .await?;
        let items = response
            .items
            .into_iter()
            .map(WireTask::into_task)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MinimaxVideoTaskList {
            items,
            total: response.total,
            extra: response.extra,
        })
    }

    /// Cancel a queued task or delete a completed/failed task record.
    ///
    /// The server chooses the action from the current task state. The request
    /// is treated as non-replayable and no hidden query is performed first.
    pub async fn cancel_or_delete(
        &self,
        task_id: &MinimaxVideoTaskId,
    ) -> Result<MinimaxVideoDeleteResult, Error> {
        self.cancel_or_delete_with_options(task_id, CallOptions::default())
            .await
    }

    pub async fn cancel_or_delete_with_options(
        &self,
        task_id: &MinimaxVideoTaskId,
        options: CallOptions,
    ) -> Result<MinimaxVideoDeleteResult, Error> {
        let response: DeleteEnvelope = execute_json(
            &self.runtime,
            Method::DELETE,
            task_target(DELETE_TARGET, task_id)?,
            RequestBody::Empty,
            ReplaySafety::Never,
            options,
        )
        .await?;
        response.into_result(task_id)
    }
}

impl fmt::Debug for MinimaxVideo {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVideo")
            .field("runtime", &"shared")
            .finish()
    }
}

#[derive(Deserialize)]
struct CreateEnvelope {
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl CreateEnvelope {
    fn into_result(mut self) -> Result<MinimaxVideoCreateResult, Error> {
        self.extra.remove("base_resp");
        let task_id = self.task_id.ok_or_else(|| {
            Error::new(
                ErrorKind::Protocol,
                "MiniMax video create response omitted task_id",
            )
        })?;
        Ok(MinimaxVideoCreateResult {
            task_id: MinimaxVideoTaskId::new(task_id).map_err(|error| {
                Error::new(
                    ErrorKind::Protocol,
                    "MiniMax video create response contained an invalid task_id",
                )
                .with_source(error)
            })?,
            extra: self.extra,
        })
    }
}

impl NativeResponseEnvelope for CreateEnvelope {
    const BASE_RESPONSE_REQUIRED: bool = false;

    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

#[derive(Deserialize)]
struct QueryEnvelope {
    #[serde(default)]
    task: Option<WireTask>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    _extra: BTreeMap<String, Value>,
}

impl NativeResponseEnvelope for QueryEnvelope {
    const BASE_RESPONSE_REQUIRED: bool = false;

    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

#[derive(Deserialize)]
struct ListEnvelope {
    #[serde(default)]
    items: Vec<WireTask>,
    #[serde(default)]
    total: Option<u64>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl NativeResponseEnvelope for ListEnvelope {
    const BASE_RESPONSE_REQUIRED: bool = false;

    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

#[derive(Deserialize)]
struct DeleteEnvelope {
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    action: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl DeleteEnvelope {
    fn into_result(
        mut self,
        expected: &MinimaxVideoTaskId,
    ) -> Result<MinimaxVideoDeleteResult, Error> {
        self.extra.remove("base_resp");
        let task_id = self.task_id.ok_or_else(|| {
            Error::new(
                ErrorKind::Protocol,
                "MiniMax video delete response omitted task_id",
            )
        })?;
        let task_id = MinimaxVideoTaskId::new(task_id).map_err(|error| {
            Error::new(
                ErrorKind::Protocol,
                "MiniMax video delete response contained an invalid task_id",
            )
            .with_source(error)
        })?;
        if &task_id != expected {
            return Err(Error::new(
                ErrorKind::ProtocolViolation,
                "MiniMax video delete response returned a different task identifier",
            ));
        }
        let action = self.action.ok_or_else(|| {
            Error::new(
                ErrorKind::Protocol,
                "MiniMax video delete response omitted action",
            )
        })?;
        Ok(MinimaxVideoDeleteResult {
            task_id,
            action: MinimaxVideoDeleteAction::from_wire(action),
            status: self.status,
            extra: self.extra,
        })
    }
}

impl NativeResponseEnvelope for DeleteEnvelope {
    const BASE_RESPONSE_REQUIRED: bool = false;

    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

#[derive(Deserialize)]
struct WireTask {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    error: Option<WireTaskFailure>,
    #[serde(default)]
    created_at: Option<i64>,
    #[serde(default)]
    updated_at: Option<i64>,
    #[serde(default)]
    content: Option<WireTaskContent>,
    #[serde(default)]
    resolution: Option<String>,
    #[serde(default)]
    duration: Option<u8>,
    #[serde(default)]
    usage: Option<MinimaxVideoUsage>,
    #[serde(default)]
    ratio: Option<String>,
    #[serde(default)]
    task_type: Option<String>,
    #[serde(default)]
    modality: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl WireTask {
    fn into_task(self) -> Result<MinimaxVideoTask, Error> {
        let id = self.id.ok_or_else(|| {
            Error::new(
                ErrorKind::Protocol,
                "MiniMax video task omitted its identifier",
            )
        })?;
        let id = MinimaxVideoTaskId::new(id).map_err(|error| {
            Error::new(
                ErrorKind::Protocol,
                "MiniMax video task contained an invalid identifier",
            )
            .with_source(error)
        })?;
        let status = self.status.ok_or_else(|| {
            Error::new(ErrorKind::Protocol, "MiniMax video task omitted its status")
        })?;
        Ok(MinimaxVideoTask {
            id,
            model: self.model,
            status: MinimaxVideoTaskStatus::from_wire(status),
            failure: self.error.map(Into::into),
            created_at: self.created_at,
            updated_at: self.updated_at,
            content: self.content.map(Into::into),
            resolution: self.resolution,
            duration_seconds: self.duration,
            usage: self.usage,
            ratio: self.ratio,
            task_type: self.task_type.map(MinimaxVideoTaskType::from_wire),
            modality: self.modality,
            extra: self.extra,
        })
    }
}

#[derive(Deserialize)]
struct WireTaskFailure {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

impl From<WireTaskFailure> for MinimaxVideoTaskFailure {
    fn from(value: WireTaskFailure) -> Self {
        Self {
            code: value.code,
            message: value.message,
        }
    }
}

#[derive(Deserialize)]
struct WireTaskContent {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl From<WireTaskContent> for MinimaxVideoTaskContent {
    fn from(value: WireTaskContent) -> Self {
        Self {
            url: value.url,
            prompt: value.prompt,
            extra: value.extra,
        }
    }
}

fn task_target(
    base: &str,
    task_id: &MinimaxVideoTaskId,
) -> Result<siumai_transport::RequestTarget, Error> {
    target(format!("{base}/{}", task_id.as_str()))
}

fn validate_bounded_text(value: &str, maximum: usize, _field: &'static str) -> Result<(), Error> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > maximum
        || value.chars().any(char::is_control)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "MiniMax video text field is empty, malformed, or too large",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(value: &str) -> MinimaxVideoMediaSource {
        MinimaxVideoMediaSource::new(value).expect("media source")
    }

    #[test]
    fn request_enforces_current_h3_modes_and_duration() {
        assert!(
            MinimaxVideoRequest::new("MiniMax-H3", "prompt", MinimaxVideoResolution::K2, 3)
                .is_err()
        );

        let text_only =
            MinimaxVideoRequest::new("MiniMax-H3", "prompt", MinimaxVideoResolution::P768, 4)
                .expect("request");
        assert!(text_only.wire().is_err());
        assert!(
            text_only
                .clone()
                .with_ratio(MinimaxVideoRatio::Adaptive)
                .wire()
                .is_err()
        );
        assert!(
            text_only
                .with_ratio(MinimaxVideoRatio::Landscape16By9)
                .wire()
                .is_ok()
        );
    }

    #[test]
    fn frame_and_reference_modes_are_mutually_exclusive() {
        let request =
            MinimaxVideoRequest::new("MiniMax-H3", "prompt", MinimaxVideoResolution::K2, 5)
                .expect("request")
                .with_input(MinimaxVideoInput::FirstFrame(source(
                    "https://example.com/first.png",
                )))
                .with_input(MinimaxVideoInput::ReferenceAudio(source(
                    "https://example.com/reference.mp3",
                )));
        assert!(request.wire().is_err());

        let last_only =
            MinimaxVideoRequest::new("MiniMax-H3", "prompt", MinimaxVideoResolution::K2, 5)
                .expect("request")
                .with_input(MinimaxVideoInput::LastFrame(source(
                    "https://example.com/last.png",
                )));
        assert!(last_only.wire().is_err());
    }

    #[test]
    fn request_wire_uses_v2_multimodal_content() {
        let request =
            MinimaxVideoRequest::new("MiniMax-H3", "prompt", MinimaxVideoResolution::K2, 15)
                .expect("request")
                .with_ratio(MinimaxVideoRatio::Portrait9By16)
                .with_input(MinimaxVideoInput::ReferenceVideo(source("mm_file://123")));
        let value = serde_json::to_value(request.wire().expect("valid request")).expect("json");
        assert_eq!(value["model"], "MiniMax-H3");
        assert_eq!(value["resolution"], "2K");
        assert_eq!(value["duration"], 15);
        assert_eq!(value["ratio"], "9:16");
        assert_eq!(value["content"][0]["type"], "text");
        assert_eq!(value["content"][1]["type"], "video_url");
        assert_eq!(value["content"][1]["role"], "reference_video");
    }

    #[test]
    fn v2_responses_do_not_require_legacy_base_response() {
        let response: CreateEnvelope = serde_json::from_value(serde_json::json!({
            "task_id": "424010985738629"
        }))
        .expect("response");
        assert_eq!(
            response.into_result().expect("result").task_id().as_str(),
            "424010985738629"
        );
    }

    #[test]
    fn task_status_is_open_and_debug_redacts_provider_content() {
        let task: WireTask = serde_json::from_value(serde_json::json!({
            "id": "424010985738629",
            "status": "future_state",
            "content": {"url": "https://signed.example/secret", "prompt": "secret prompt"},
            "error": {"code": "1026", "message": "secret provider detail"}
        }))
        .expect("task");
        let task = task.into_task().expect("task contract");
        assert!(
            matches!(task.status(), MinimaxVideoTaskStatus::Other(value) if value == "future_state")
        );
        let debug = format!("{task:?}");
        assert!(!debug.contains("signed.example"));
        assert!(!debug.contains("secret prompt"));
        assert!(!debug.contains("secret provider detail"));
    }

    #[test]
    fn list_target_encodes_filters_without_raw_interpolation() {
        let query = MinimaxVideoListQuery::new()
            .with_page(1, 20)
            .expect("page")
            .with_model("MiniMax-H3 & future")
            .expect("model")
            .with_task_type(MinimaxVideoTaskType::Generation);
        assert_eq!(
            query.target().expect("target").as_str(),
            "v2/query/video_generation?page_num=1&page_size=20&filter.model=MiniMax-H3+%26+future&filter.task_type=generation"
        );
    }
}
