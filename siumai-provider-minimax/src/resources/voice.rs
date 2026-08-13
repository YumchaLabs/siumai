use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use http::Method;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use siumai_core::{CallOptions, Error, ErrorKind};
use siumai_transport::{ReplaySafety, RequestBody};
use thiserror::Error as ThisError;

use super::common::{BaseResponse, NativeResponseEnvelope, NativeRuntime, execute_json, target};
use super::files::MinimaxFileId;
use super::speech::{MinimaxLanguageBoost, MinimaxSpeechDownloadUrl, MinimaxVoiceId};

const CLONE_TARGET: &str = "v1/voice_clone";
const DESIGN_TARGET: &str = "v1/voice_design";
const LIST_TARGET: &str = "v1/get_voice";
const DELETE_TARGET: &str = "v1/delete_voice";
const MAX_CUSTOM_VOICE_ID_BYTES: usize = 256;
const MIN_CUSTOM_VOICE_ID_BYTES: usize = 8;
const MAX_CLONE_PROMPT_TEXT_CHARACTERS: usize = 2_000;
const MAX_CLONE_PREVIEW_TEXT_CHARACTERS: usize = 1_000;
const MAX_DESIGN_PROMPT_CHARACTERS: usize = 500;
const MAX_DESIGN_PREVIEW_TEXT_CHARACTERS: usize = 500;

pub const VOICE_CLONE_API_SOURCE: &str =
    "https://platform.minimax.io/docs/api-reference/voice-cloning-clone";
pub const VOICE_DESIGN_API_SOURCE: &str =
    "https://platform.minimax.io/docs/api-reference/voice-design-design";
pub const VOICE_LIST_API_SOURCE: &str =
    "https://platform.minimax.io/docs/api-reference/voice-management-get";
pub const VOICE_DELETE_API_SOURCE: &str =
    "https://platform.minimax.io/docs/api-reference/voice-management-delete";
pub const VOICE_API_VERIFIED_ON: &str = "2026-08-09";

/// Validation failures for a caller-selected MiniMax custom voice ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
#[non_exhaustive]
pub enum MinimaxCustomVoiceIdError {
    #[error(
        "MiniMax custom voice ID must contain 8 through 256 ASCII letters, digits, hyphens, or underscores; start with a letter; and end with a letter or digit"
    )]
    Invalid,
}

/// Caller-selected identifier used by voice cloning and voice design.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MinimaxCustomVoiceId(String);

impl MinimaxCustomVoiceId {
    pub fn new(value: impl Into<String>) -> Result<Self, MinimaxCustomVoiceIdError> {
        let value = value.into();
        let bytes = value.as_bytes();
        let first = bytes.first().copied();
        let last = bytes.last().copied();
        if !(MIN_CUSTOM_VOICE_ID_BYTES..=MAX_CUSTOM_VOICE_ID_BYTES).contains(&value.len())
            || !first.is_some_and(|byte| byte.is_ascii_alphabetic())
            || !last.is_some_and(|byte| byte.is_ascii_alphanumeric())
            || !bytes
                .iter()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(MinimaxCustomVoiceIdError::Invalid);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for MinimaxCustomVoiceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxCustomVoiceId")
            .field("bytes", &self.0.len())
            .finish()
    }
}

impl fmt::Display for MinimaxCustomVoiceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Serialize for MinimaxCustomVoiceId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for MinimaxCustomVoiceId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Speech model used only for the optional voice-clone preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxVoicePreviewModel {
    Speech28Hd,
    Speech28Turbo,
    Speech26Hd,
    Speech26Turbo,
    Speech02Hd,
    Speech02Turbo,
    Speech01Hd,
    Speech01Turbo,
}

impl MinimaxVoicePreviewModel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Speech28Hd => "speech-2.8-hd",
            Self::Speech28Turbo => "speech-2.8-turbo",
            Self::Speech26Hd => "speech-2.6-hd",
            Self::Speech26Turbo => "speech-2.6-turbo",
            Self::Speech02Hd => "speech-02-hd",
            Self::Speech02Turbo => "speech-02-turbo",
            Self::Speech01Hd => "speech-01-hd",
            Self::Speech01Turbo => "speech-01-turbo",
        }
    }
}

/// Optional source-audio plus transcript used to improve clone quality.
#[derive(Clone, PartialEq, Eq)]
pub struct MinimaxVoiceClonePrompt {
    file_id: MinimaxFileId,
    prompt_text: String,
}

impl MinimaxVoiceClonePrompt {
    pub fn new(file_id: MinimaxFileId, prompt_text: impl Into<String>) -> Result<Self, Error> {
        let prompt_text = prompt_text.into();
        validate_text(
            &prompt_text,
            MAX_CLONE_PROMPT_TEXT_CHARACTERS,
            "MiniMax voice-clone prompt text is invalid",
        )?;
        Ok(Self {
            file_id,
            prompt_text,
        })
    }

    pub const fn file_id(&self) -> MinimaxFileId {
        self.file_id
    }

    pub fn prompt_text(&self) -> &str {
        &self.prompt_text
    }
}

impl fmt::Debug for MinimaxVoiceClonePrompt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVoiceClonePrompt")
            .field("file_id", &self.file_id)
            .field("prompt_text_characters", &self.prompt_text.chars().count())
            .finish()
    }
}

/// Optional cloned-voice preview. Text and model are inseparable by construction.
#[derive(Clone, PartialEq, Eq)]
pub struct MinimaxVoiceClonePreview {
    text: String,
    model: MinimaxVoicePreviewModel,
}

impl MinimaxVoiceClonePreview {
    pub fn new(text: impl Into<String>, model: MinimaxVoicePreviewModel) -> Result<Self, Error> {
        let text = text.into();
        validate_text(
            &text,
            MAX_CLONE_PREVIEW_TEXT_CHARACTERS,
            "MiniMax voice-clone preview text is invalid",
        )?;
        Ok(Self { text, model })
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub const fn model(&self) -> MinimaxVoicePreviewModel {
        self.model
    }
}

impl fmt::Debug for MinimaxVoiceClonePreview {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVoiceClonePreview")
            .field("text_characters", &self.text.chars().count())
            .field("model", &self.model)
            .finish()
    }
}

/// Typed MiniMax voice-cloning request.
#[derive(Clone)]
pub struct MinimaxVoiceCloneRequest {
    file_id: MinimaxFileId,
    voice_id: MinimaxCustomVoiceId,
    prompt: Option<MinimaxVoiceClonePrompt>,
    preview: Option<MinimaxVoiceClonePreview>,
    language_boost: Option<MinimaxLanguageBoost>,
    need_noise_reduction: Option<bool>,
    need_volume_normalization: Option<bool>,
}

impl MinimaxVoiceCloneRequest {
    pub fn new(file_id: MinimaxFileId, voice_id: MinimaxCustomVoiceId) -> Self {
        Self {
            file_id,
            voice_id,
            prompt: None,
            preview: None,
            language_boost: None,
            need_noise_reduction: None,
            need_volume_normalization: None,
        }
    }

    pub fn with_prompt(mut self, prompt: MinimaxVoiceClonePrompt) -> Self {
        self.prompt = Some(prompt);
        self
    }

    pub fn with_preview(mut self, preview: MinimaxVoiceClonePreview) -> Self {
        self.preview = Some(preview);
        self
    }

    pub fn with_language_boost(mut self, language_boost: MinimaxLanguageBoost) -> Self {
        self.language_boost = Some(language_boost);
        self
    }

    pub const fn with_noise_reduction(mut self, enabled: bool) -> Self {
        self.need_noise_reduction = Some(enabled);
        self
    }

    pub const fn with_volume_normalization(mut self, enabled: bool) -> Self {
        self.need_volume_normalization = Some(enabled);
        self
    }

    pub const fn file_id(&self) -> MinimaxFileId {
        self.file_id
    }

    pub fn voice_id(&self) -> &MinimaxCustomVoiceId {
        &self.voice_id
    }

    fn wire(&self) -> VoiceCloneWire<'_> {
        VoiceCloneWire {
            file_id: self.file_id,
            voice_id: &self.voice_id,
            text: self.preview.as_ref().map(MinimaxVoiceClonePreview::text),
            model: self
                .preview
                .as_ref()
                .map(|preview| preview.model().as_str()),
            clone_prompt: self.prompt.as_ref().map(|prompt| ClonePromptWire {
                prompt_audio: prompt.file_id(),
                prompt_text: prompt.prompt_text(),
            }),
            language_boost: self.language_boost.as_ref(),
            need_noise_reduction: self.need_noise_reduction,
            need_volume_normalization: self.need_volume_normalization,
        }
    }
}

impl fmt::Debug for MinimaxVoiceCloneRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVoiceCloneRequest")
            .field("file_id", &self.file_id)
            .field("voice_id", &self.voice_id)
            .field("prompt", &self.prompt)
            .field("preview", &self.preview)
            .field("has_language_boost", &self.language_boost.is_some())
            .field("need_noise_reduction", &self.need_noise_reduction)
            .field("need_volume_normalization", &self.need_volume_normalization)
            .finish()
    }
}

#[derive(Serialize)]
struct VoiceCloneWire<'a> {
    file_id: MinimaxFileId,
    voice_id: &'a MinimaxCustomVoiceId,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    clone_prompt: Option<ClonePromptWire<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language_boost: Option<&'a MinimaxLanguageBoost>,
    #[serde(skip_serializing_if = "Option::is_none")]
    need_noise_reduction: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    need_volume_normalization: Option<bool>,
}

#[derive(Serialize)]
struct ClonePromptWire<'a> {
    prompt_audio: MinimaxFileId,
    prompt_text: &'a str,
}

/// MiniMax safety classification attached to clone or design input.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MinimaxVoiceInputSafety {
    flagged: Option<bool>,
    type_code: Option<i64>,
}

impl MinimaxVoiceInputSafety {
    pub const fn flagged(self) -> Option<bool> {
        self.flagged
    }

    pub const fn type_code(self) -> Option<i64> {
        self.type_code
    }
}

/// Completed voice-cloning result.
#[derive(Clone)]
pub struct MinimaxVoiceCloneResult {
    voice_id: MinimaxVoiceId,
    demo_audio: Option<MinimaxSpeechDownloadUrl>,
    safety: MinimaxVoiceInputSafety,
    extra: BTreeMap<String, Value>,
}

impl MinimaxVoiceCloneResult {
    pub fn voice_id(&self) -> &MinimaxVoiceId {
        &self.voice_id
    }

    pub fn demo_audio(&self) -> Option<&MinimaxSpeechDownloadUrl> {
        self.demo_audio.as_ref()
    }

    pub const fn safety(&self) -> MinimaxVoiceInputSafety {
        self.safety
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxVoiceCloneResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVoiceCloneResult")
            .field("voice_id", &self.voice_id)
            .field("demo_audio_present", &self.demo_audio.is_some())
            .field("safety", &self.safety)
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Typed MiniMax voice-design request.
#[derive(Clone)]
pub struct MinimaxVoiceDesignRequest {
    prompt: String,
    preview_text: String,
    voice_id: Option<MinimaxCustomVoiceId>,
}

impl MinimaxVoiceDesignRequest {
    pub fn new(prompt: impl Into<String>, preview_text: impl Into<String>) -> Result<Self, Error> {
        let prompt = prompt.into();
        let preview_text = preview_text.into();
        validate_text(
            &prompt,
            MAX_DESIGN_PROMPT_CHARACTERS,
            "MiniMax voice-design prompt is invalid",
        )?;
        validate_text(
            &preview_text,
            MAX_DESIGN_PREVIEW_TEXT_CHARACTERS,
            "MiniMax voice-design preview text is invalid",
        )?;
        Ok(Self {
            prompt,
            preview_text,
            voice_id: None,
        })
    }

    pub fn with_voice_id(mut self, voice_id: MinimaxCustomVoiceId) -> Self {
        self.voice_id = Some(voice_id);
        self
    }

    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    pub fn preview_text(&self) -> &str {
        &self.preview_text
    }
}

impl fmt::Debug for MinimaxVoiceDesignRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVoiceDesignRequest")
            .field("prompt_characters", &self.prompt.chars().count())
            .field(
                "preview_text_characters",
                &self.preview_text.chars().count(),
            )
            .field("voice_id", &self.voice_id)
            .finish()
    }
}

#[derive(Serialize)]
struct VoiceDesignWire<'a> {
    prompt: &'a str,
    preview_text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice_id: Option<&'a MinimaxCustomVoiceId>,
}

/// Completed voice-design result with decoded trial audio.
#[derive(Clone)]
pub struct MinimaxVoiceDesignResult {
    voice_id: MinimaxVoiceId,
    trial_audio: Vec<u8>,
    safety: MinimaxVoiceInputSafety,
    extra: BTreeMap<String, Value>,
}

impl MinimaxVoiceDesignResult {
    pub fn voice_id(&self) -> &MinimaxVoiceId {
        &self.voice_id
    }

    pub fn trial_audio(&self) -> &[u8] {
        &self.trial_audio
    }

    pub const fn safety(&self) -> MinimaxVoiceInputSafety {
        self.safety
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxVoiceDesignResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVoiceDesignResult")
            .field("voice_id", &self.voice_id)
            .field("trial_audio_bytes", &self.trial_audio.len())
            .field("safety", &self.safety)
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Voice categories accepted by MiniMax voice-list queries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MinimaxVoiceListKind {
    System,
    VoiceCloning,
    VoiceGeneration,
    #[default]
    All,
}

/// Caller-created voice categories accepted by deletion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MinimaxCustomVoiceKind {
    VoiceCloning,
    VoiceGeneration,
}

/// One voice returned by MiniMax voice management.
#[derive(Clone, Deserialize)]
pub struct MinimaxVoiceSummary {
    voice_id: MinimaxVoiceId,
    #[serde(default)]
    voice_name: Option<String>,
    #[serde(default)]
    description: Vec<String>,
    #[serde(default)]
    created_time: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl MinimaxVoiceSummary {
    pub fn voice_id(&self) -> &MinimaxVoiceId {
        &self.voice_id
    }

    pub fn voice_name(&self) -> Option<&str> {
        self.voice_name.as_deref()
    }

    pub fn description(&self) -> &[String] {
        &self.description
    }

    pub fn created_time(&self) -> Option<&str> {
        self.created_time.as_deref()
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxVoiceSummary {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVoiceSummary")
            .field("voice_id", &self.voice_id)
            .field("voice_name_present", &self.voice_name.is_some())
            .field("description_count", &self.description.len())
            .field("created_time_present", &self.created_time.is_some())
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Categorized voice-management result.
#[derive(Clone)]
pub struct MinimaxVoiceList {
    system: Vec<MinimaxVoiceSummary>,
    voice_cloning: Vec<MinimaxVoiceSummary>,
    voice_generation: Vec<MinimaxVoiceSummary>,
    extra: BTreeMap<String, Value>,
}

impl MinimaxVoiceList {
    pub fn system(&self) -> &[MinimaxVoiceSummary] {
        &self.system
    }

    pub fn voice_cloning(&self) -> &[MinimaxVoiceSummary] {
        &self.voice_cloning
    }

    pub fn voice_generation(&self) -> &[MinimaxVoiceSummary] {
        &self.voice_generation
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxVoiceList {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVoiceList")
            .field("system_count", &self.system.len())
            .field("voice_cloning_count", &self.voice_cloning.len())
            .field("voice_generation_count", &self.voice_generation.len())
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Successful caller-created voice deletion.
#[derive(Clone, PartialEq)]
pub struct MinimaxVoiceDeleteResult {
    voice_id: MinimaxCustomVoiceId,
    kind: MinimaxCustomVoiceKind,
    created_time: String,
    extra: BTreeMap<String, Value>,
}

impl MinimaxVoiceDeleteResult {
    pub fn voice_id(&self) -> &MinimaxCustomVoiceId {
        &self.voice_id
    }

    pub const fn kind(&self) -> MinimaxCustomVoiceKind {
        self.kind
    }

    pub fn created_time(&self) -> &str {
        &self.created_time
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxVoiceDeleteResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVoiceDeleteResult")
            .field("voice_id", &self.voice_id)
            .field("kind", &self.kind)
            .field("created_time_present", &!self.created_time.is_empty())
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Provider-native MiniMax voice clone, design, list, and delete operations.
#[derive(Clone)]
pub struct MinimaxVoices {
    runtime: Arc<NativeRuntime>,
}

impl MinimaxVoices {
    pub(crate) fn new(runtime: Arc<NativeRuntime>) -> Self {
        Self { runtime }
    }

    pub async fn clone_voice(
        &self,
        request: MinimaxVoiceCloneRequest,
    ) -> Result<MinimaxVoiceCloneResult, Error> {
        self.clone_voice_with_options(request, CallOptions::default())
            .await
    }

    pub async fn clone_voice_with_options(
        &self,
        request: MinimaxVoiceCloneRequest,
        options: CallOptions,
    ) -> Result<MinimaxVoiceCloneResult, Error> {
        let requested_voice_id =
            MinimaxVoiceId::new(request.voice_id().as_str()).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "MiniMax custom voice ID is invalid",
                )
                .with_source(source)
            })?;
        let response: VoiceCloneEnvelope = execute_json(
            &self.runtime,
            Method::POST,
            target(CLONE_TARGET)?,
            request_body(&request.wire(), "MiniMax voice-clone request is invalid")?,
            ReplaySafety::Never,
            options,
        )
        .await?;
        let mut extra = response.extra;
        preserve_unknown_safety(&mut extra, response.input_sensitive.as_ref());
        Ok(MinimaxVoiceCloneResult {
            voice_id: requested_voice_id,
            demo_audio: response
                .demo_audio
                .filter(|value| !value.is_empty())
                .map(MinimaxSpeechDownloadUrl::from_provider)
                .transpose()?,
            safety: safety(response.input_sensitive, response.input_sensitive_type),
            extra,
        })
    }

    pub async fn design_voice(
        &self,
        request: MinimaxVoiceDesignRequest,
    ) -> Result<MinimaxVoiceDesignResult, Error> {
        self.design_voice_with_options(request, CallOptions::default())
            .await
    }

    pub async fn design_voice_with_options(
        &self,
        request: MinimaxVoiceDesignRequest,
        options: CallOptions,
    ) -> Result<MinimaxVoiceDesignResult, Error> {
        let body = VoiceDesignWire {
            prompt: request.prompt(),
            preview_text: request.preview_text(),
            voice_id: request.voice_id.as_ref(),
        };
        let response: VoiceDesignEnvelope = execute_json(
            &self.runtime,
            Method::POST,
            target(DESIGN_TARGET)?,
            request_body(&body, "MiniMax voice-design request is invalid")?,
            ReplaySafety::Never,
            options,
        )
        .await?;
        let mut extra = response.extra;
        preserve_unknown_safety(&mut extra, response.input_sensitive.as_ref());
        Ok(MinimaxVoiceDesignResult {
            voice_id: response.voice_id,
            trial_audio: decode_hex_audio(&response.trial_audio)?,
            safety: safety(response.input_sensitive, response.input_sensitive_type),
            extra,
        })
    }

    pub async fn list(&self, kind: MinimaxVoiceListKind) -> Result<MinimaxVoiceList, Error> {
        self.list_with_options(kind, CallOptions::default()).await
    }

    pub async fn list_with_options(
        &self,
        kind: MinimaxVoiceListKind,
        options: CallOptions,
    ) -> Result<MinimaxVoiceList, Error> {
        let response: VoiceListEnvelope = execute_json(
            &self.runtime,
            Method::POST,
            target(LIST_TARGET)?,
            request_body(
                &VoiceListWire { voice_type: kind },
                "MiniMax voice-list request is invalid",
            )?,
            ReplaySafety::SemanticallyIdempotent,
            options,
        )
        .await?;
        Ok(MinimaxVoiceList {
            system: response.system_voice,
            voice_cloning: response.voice_cloning,
            voice_generation: response.voice_generation,
            extra: response.extra,
        })
    }

    pub async fn delete(
        &self,
        voice_id: MinimaxCustomVoiceId,
        kind: MinimaxCustomVoiceKind,
    ) -> Result<MinimaxVoiceDeleteResult, Error> {
        self.delete_with_options(voice_id, kind, CallOptions::default())
            .await
    }

    pub async fn delete_with_options(
        &self,
        voice_id: MinimaxCustomVoiceId,
        kind: MinimaxCustomVoiceKind,
        options: CallOptions,
    ) -> Result<MinimaxVoiceDeleteResult, Error> {
        let request = VoiceDeleteWire {
            voice_id: &voice_id,
            voice_type: kind,
        };
        let response: VoiceDeleteEnvelope = execute_json(
            &self.runtime,
            Method::POST,
            target(DELETE_TARGET)?,
            request_body(&request, "MiniMax voice-delete request is invalid")?,
            ReplaySafety::Never,
            options,
        )
        .await?;
        if response.voice_id != voice_id {
            return Err(Error::new(
                ErrorKind::Protocol,
                "MiniMax voice-delete response did not match the requested voice",
            ));
        }
        validate_response_text(
            &response.created_time,
            128,
            "MiniMax voice-delete response contained an invalid creation time",
        )?;
        Ok(MinimaxVoiceDeleteResult {
            voice_id,
            kind,
            created_time: response.created_time,
            extra: response.extra,
        })
    }
}

impl fmt::Debug for MinimaxVoices {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVoices")
            .field("runtime", &"shared")
            .finish()
    }
}

#[derive(Serialize)]
struct VoiceListWire {
    voice_type: MinimaxVoiceListKind,
}

#[derive(Serialize)]
struct VoiceDeleteWire<'a> {
    voice_id: &'a MinimaxCustomVoiceId,
    voice_type: MinimaxCustomVoiceKind,
}

#[derive(Deserialize)]
struct VoiceCloneEnvelope {
    #[serde(default)]
    demo_audio: Option<String>,
    #[serde(default)]
    input_sensitive: Option<Value>,
    #[serde(default)]
    input_sensitive_type: Option<i64>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl NativeResponseEnvelope for VoiceCloneEnvelope {
    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

#[derive(Deserialize)]
struct VoiceDesignEnvelope {
    voice_id: MinimaxVoiceId,
    trial_audio: String,
    #[serde(default)]
    input_sensitive: Option<Value>,
    #[serde(default)]
    input_sensitive_type: Option<i64>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl NativeResponseEnvelope for VoiceDesignEnvelope {
    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

#[derive(Deserialize)]
struct VoiceListEnvelope {
    #[serde(default)]
    system_voice: Vec<MinimaxVoiceSummary>,
    #[serde(default)]
    voice_cloning: Vec<MinimaxVoiceSummary>,
    #[serde(default)]
    voice_generation: Vec<MinimaxVoiceSummary>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl NativeResponseEnvelope for VoiceListEnvelope {
    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

#[derive(Deserialize)]
struct VoiceDeleteEnvelope {
    voice_id: MinimaxCustomVoiceId,
    created_time: String,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl NativeResponseEnvelope for VoiceDeleteEnvelope {
    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

fn request_body(value: &impl Serialize, message: &'static str) -> Result<RequestBody, Error> {
    RequestBody::json(value)
        .map_err(|source| Error::new(ErrorKind::InvalidInput, message).with_source(source))
}

fn validate_text(value: &str, maximum: usize, message: &'static str) -> Result<(), Error> {
    let characters = value.chars().count();
    if value.trim().is_empty()
        || characters > maximum
        || value
            .chars()
            .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
    {
        return Err(Error::new(ErrorKind::InvalidInput, message));
    }
    Ok(())
}

fn validate_response_text(
    value: &str,
    maximum_bytes: usize,
    message: &'static str,
) -> Result<(), Error> {
    if value.trim().is_empty() || value.len() > maximum_bytes || value.chars().any(char::is_control)
    {
        return Err(Error::new(ErrorKind::Protocol, message));
    }
    Ok(())
}

fn safety(input_sensitive: Option<Value>, type_code: Option<i64>) -> MinimaxVoiceInputSafety {
    MinimaxVoiceInputSafety {
        flagged: input_sensitive.as_ref().and_then(Value::as_bool),
        type_code,
    }
}

fn preserve_unknown_safety(extra: &mut BTreeMap<String, Value>, value: Option<&Value>) {
    if value.is_some_and(|value| !value.is_boolean())
        && let Some(value) = value
    {
        extra.insert("input_sensitive".to_string(), value.clone());
    }
}

fn decode_hex_audio(value: &str) -> Result<Vec<u8>, Error> {
    if value.is_empty() || !value.len().is_multiple_of(2) {
        return Err(Error::new(
            ErrorKind::Protocol,
            "MiniMax voice response contained invalid hexadecimal audio",
        ));
    }
    let mut decoded = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let high = decode_hex_digit(pair[0]).ok_or_else(|| {
            Error::new(
                ErrorKind::Protocol,
                "MiniMax voice response contained invalid hexadecimal audio",
            )
        })?;
        let low = decode_hex_digit(pair[1]).ok_or_else(|| {
            Error::new(
                ErrorKind::Protocol,
                "MiniMax voice response contained invalid hexadecimal audio",
            )
        })?;
        decoded.push((high << 4) | low);
    }
    Ok(decoded)
}

const fn decode_hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}
