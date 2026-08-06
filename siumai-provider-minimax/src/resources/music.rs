use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use http::Method;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{CallOptions, Error, ErrorKind};
use siumai_transport::{ReplaySafety, RequestBody};

use crate::models::music::{
    MUSIC_2_6, MUSIC_2_6_FREE, MUSIC_3_0, MUSIC_3_0_FREE, MUSIC_COVER, MUSIC_COVER_FREE,
};

use super::common::{BaseResponse, NativeResponseEnvelope, NativeRuntime, execute_json, target};

const MUSIC_GENERATION_TARGET: &str = "v1/music_generation";

/// MiniMax's non-streaming music response representation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxMusicOutputFormat {
    Url,
    #[default]
    Hex,
}

impl MinimaxMusicOutputFormat {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Url => "url",
            Self::Hex => "hex",
        }
    }
}

/// Supported MiniMax music output sample rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxMusicSampleRate {
    Hz16000,
    Hz24000,
    Hz32000,
    Hz44100,
}

impl MinimaxMusicSampleRate {
    pub const fn hertz(self) -> u32 {
        match self {
            Self::Hz16000 => 16_000,
            Self::Hz24000 => 24_000,
            Self::Hz32000 => 32_000,
            Self::Hz44100 => 44_100,
        }
    }
}

/// Supported MiniMax music output bitrates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxMusicBitrate {
    Bps32000,
    Bps64000,
    Bps128000,
    Bps256000,
}

impl MinimaxMusicBitrate {
    pub const fn bits_per_second(self) -> u32 {
        match self {
            Self::Bps32000 => 32_000,
            Self::Bps64000 => 64_000,
            Self::Bps128000 => 128_000,
            Self::Bps256000 => 256_000,
        }
    }
}

/// Supported MiniMax music audio formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxMusicAudioFormat {
    Mp3,
    Wav,
    Pcm,
}

impl MinimaxMusicAudioFormat {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Wav => "wav",
            Self::Pcm => "pcm",
        }
    }
}

/// Optional output encoding settings for MiniMax music generation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MinimaxMusicAudioSetting {
    sample_rate: Option<MinimaxMusicSampleRate>,
    bitrate: Option<MinimaxMusicBitrate>,
    format: Option<MinimaxMusicAudioFormat>,
}

impl MinimaxMusicAudioSetting {
    pub const fn new() -> Self {
        Self {
            sample_rate: None,
            bitrate: None,
            format: None,
        }
    }

    pub const fn with_sample_rate(mut self, sample_rate: MinimaxMusicSampleRate) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    pub const fn with_bitrate(mut self, bitrate: MinimaxMusicBitrate) -> Self {
        self.bitrate = Some(bitrate);
        self
    }

    pub const fn with_format(mut self, format: MinimaxMusicAudioFormat) -> Self {
        self.format = Some(format);
        self
    }

    pub const fn sample_rate(self) -> Option<MinimaxMusicSampleRate> {
        self.sample_rate
    }

    pub const fn bitrate(self) -> Option<MinimaxMusicBitrate> {
        self.bitrate
    }

    pub const fn format(self) -> Option<MinimaxMusicAudioFormat> {
        self.format
    }

    const fn is_empty(self) -> bool {
        self.sample_rate.is_none() && self.bitrate.is_none() && self.format.is_none()
    }
}

/// Direct reference audio for the one-step MiniMax cover workflow.
///
/// Values are deliberately redacted from `Debug` because URLs may be signed
/// and base64 input may contain private audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxMusicCoverSourceKind {
    Url,
    Base64,
}

#[derive(Clone)]
pub struct MinimaxMusicCoverSource {
    kind: MinimaxMusicCoverSourceKind,
    value: String,
}

impl MinimaxMusicCoverSource {
    pub fn url(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        validate_non_empty(&value, "MiniMax cover audio URL must not be empty")?;
        Ok(Self {
            kind: MinimaxMusicCoverSourceKind::Url,
            value,
        })
    }

    pub fn base64(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        validate_non_empty(&value, "MiniMax cover audio base64 must not be empty")?;
        Ok(Self {
            kind: MinimaxMusicCoverSourceKind::Base64,
            value,
        })
    }

    pub const fn kind(&self) -> MinimaxMusicCoverSourceKind {
        self.kind
    }

    pub fn as_url(&self) -> Option<&str> {
        match self.kind {
            MinimaxMusicCoverSourceKind::Url => Some(&self.value),
            MinimaxMusicCoverSourceKind::Base64 => None,
        }
    }

    pub fn as_base64(&self) -> Option<&str> {
        match self.kind {
            MinimaxMusicCoverSourceKind::Url => None,
            MinimaxMusicCoverSourceKind::Base64 => Some(&self.value),
        }
    }
}

impl fmt::Debug for MinimaxMusicCoverSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            MinimaxMusicCoverSourceKind::Url => formatter
                .debug_struct("MinimaxMusicCoverSource::Url")
                .field("value", &"redacted")
                .field("bytes", &self.value.len())
                .finish(),
            MinimaxMusicCoverSourceKind::Base64 => formatter
                .debug_struct("MinimaxMusicCoverSource::Base64")
                .field("value", &"redacted")
                .field("bytes", &self.value.len())
                .finish(),
        }
    }
}

/// The generation workflow represented by a [`MinimaxMusicRequest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxMusicRequestKind {
    Lyrics,
    GeneratedLyrics,
    Instrumental,
    Cover,
}

/// Typed request for MiniMax's provider-native Music Generation API.
///
/// Constructors encode the four supported workflows and keep mutually
/// exclusive cover inputs out of the public request surface.
#[derive(Clone)]
pub struct MinimaxMusicRequest {
    model: String,
    input: MinimaxMusicInput,
    output_format: MinimaxMusicOutputFormat,
    audio_setting: Option<MinimaxMusicAudioSetting>,
}

impl MinimaxMusicRequest {
    /// Create vocal music from caller-provided lyrics.
    pub fn from_lyrics(model: impl Into<String>, lyrics: impl Into<String>) -> Result<Self, Error> {
        let request = Self {
            model: model.into(),
            input: MinimaxMusicInput::Lyrics {
                prompt: None,
                lyrics: lyrics.into(),
            },
            output_format: MinimaxMusicOutputFormat::default(),
            audio_setting: None,
        };
        request.validate()?;
        Ok(request)
    }

    /// Create vocal music whose lyrics are generated from the prompt.
    pub fn generated_lyrics(
        model: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Result<Self, Error> {
        let request = Self {
            model: model.into(),
            input: MinimaxMusicInput::GeneratedLyrics {
                prompt: prompt.into(),
            },
            output_format: MinimaxMusicOutputFormat::default(),
            audio_setting: None,
        };
        request.validate()?;
        Ok(request)
    }

    /// Create instrumental music from a style and scenario prompt.
    pub fn instrumental(
        model: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Result<Self, Error> {
        let request = Self {
            model: model.into(),
            input: MinimaxMusicInput::Instrumental {
                prompt: prompt.into(),
            },
            output_format: MinimaxMusicOutputFormat::default(),
            audio_setting: None,
        };
        request.validate()?;
        Ok(request)
    }

    /// Create a one-step cover request from a URL or base64 reference.
    pub fn cover(
        model: impl Into<String>,
        prompt: impl Into<String>,
        source: MinimaxMusicCoverSource,
    ) -> Result<Self, Error> {
        let request = Self {
            model: model.into(),
            input: MinimaxMusicInput::Cover {
                prompt: prompt.into(),
                reference: MinimaxMusicCoverReference::Direct(source),
                lyrics: None,
            },
            output_format: MinimaxMusicOutputFormat::default(),
            audio_setting: None,
        };
        request.validate()?;
        Ok(request)
    }

    /// Create a two-step cover request from a preprocessed feature ID.
    ///
    /// MiniMax requires caller-provided lyrics when a feature ID is used.
    pub fn cover_from_feature(
        model: impl Into<String>,
        prompt: impl Into<String>,
        feature_id: impl Into<String>,
        lyrics: impl Into<String>,
    ) -> Result<Self, Error> {
        let feature_id = feature_id.into();
        validate_non_empty(
            &feature_id,
            "MiniMax cover feature identifier must not be empty",
        )?;
        let request = Self {
            model: model.into(),
            input: MinimaxMusicInput::Cover {
                prompt: prompt.into(),
                reference: MinimaxMusicCoverReference::FeatureId(feature_id),
                lyrics: Some(lyrics.into()),
            },
            output_format: MinimaxMusicOutputFormat::default(),
            audio_setting: None,
        };
        request.validate()?;
        Ok(request)
    }

    /// Add an optional style prompt to a caller-lyrics request.
    pub fn with_style_prompt(mut self, prompt: impl Into<String>) -> Result<Self, Error> {
        match &mut self.input {
            MinimaxMusicInput::Lyrics {
                prompt: current, ..
            } => *current = Some(prompt.into()),
            _ => {
                return Err(invalid_input(
                    "MiniMax style prompts can be added only to a caller-lyrics request",
                ));
            }
        }
        self.validate()?;
        Ok(self)
    }

    /// Replace or supply lyrics for a cover request.
    pub fn with_cover_lyrics(mut self, lyrics: impl Into<String>) -> Result<Self, Error> {
        match &mut self.input {
            MinimaxMusicInput::Cover {
                lyrics: current, ..
            } => *current = Some(lyrics.into()),
            _ => {
                return Err(invalid_input(
                    "MiniMax cover lyrics can be added only to a cover request",
                ));
            }
        }
        self.validate()?;
        Ok(self)
    }

    pub fn with_output_format(mut self, output_format: MinimaxMusicOutputFormat) -> Self {
        self.output_format = output_format;
        self
    }

    pub fn with_audio_setting(mut self, audio_setting: MinimaxMusicAudioSetting) -> Self {
        self.audio_setting = (!audio_setting.is_empty()).then_some(audio_setting);
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub const fn kind(&self) -> MinimaxMusicRequestKind {
        self.input.kind()
    }

    pub const fn output_format(&self) -> MinimaxMusicOutputFormat {
        self.output_format
    }

    pub const fn audio_setting(&self) -> Option<MinimaxMusicAudioSetting> {
        self.audio_setting
    }

    pub fn prompt(&self) -> Option<&str> {
        match &self.input {
            MinimaxMusicInput::Lyrics { prompt, .. } => prompt.as_deref(),
            MinimaxMusicInput::GeneratedLyrics { prompt }
            | MinimaxMusicInput::Instrumental { prompt }
            | MinimaxMusicInput::Cover { prompt, .. } => Some(prompt),
        }
    }

    pub fn lyrics(&self) -> Option<&str> {
        match &self.input {
            MinimaxMusicInput::Lyrics { lyrics, .. } => Some(lyrics),
            MinimaxMusicInput::Cover { lyrics, .. } => lyrics.as_deref(),
            MinimaxMusicInput::GeneratedLyrics { .. } | MinimaxMusicInput::Instrumental { .. } => {
                None
            }
        }
    }

    pub fn cover_source(&self) -> Option<&MinimaxMusicCoverSource> {
        match &self.input {
            MinimaxMusicInput::Cover {
                reference: MinimaxMusicCoverReference::Direct(source),
                ..
            } => Some(source),
            _ => None,
        }
    }

    pub fn cover_feature_id(&self) -> Option<&str> {
        match &self.input {
            MinimaxMusicInput::Cover {
                reference: MinimaxMusicCoverReference::FeatureId(value),
                ..
            } => Some(value),
            _ => None,
        }
    }

    fn validate(&self) -> Result<(), Error> {
        validate_non_empty(&self.model, "MiniMax music model must not be empty")?;
        validate_model_workflow(&self.model, self.input.kind())?;
        match &self.input {
            MinimaxMusicInput::Lyrics { prompt, lyrics } => {
                validate_character_count(
                    lyrics,
                    1,
                    3_500,
                    "MiniMax music lyrics must contain between 1 and 3500 characters",
                )?;
                if let Some(prompt) = prompt {
                    validate_character_count(
                        prompt,
                        0,
                        2_000,
                        "MiniMax music prompt must contain at most 2000 characters",
                    )?;
                }
            }
            MinimaxMusicInput::GeneratedLyrics { prompt }
            | MinimaxMusicInput::Instrumental { prompt } => validate_character_count(
                prompt,
                1,
                2_000,
                "MiniMax music prompt must contain between 1 and 2000 characters",
            )?,
            MinimaxMusicInput::Cover {
                prompt,
                reference,
                lyrics,
            } => {
                validate_character_count(
                    prompt,
                    10,
                    300,
                    "MiniMax cover prompt must contain between 10 and 300 characters",
                )?;
                match reference {
                    MinimaxMusicCoverReference::Direct(source) => validate_non_empty(
                        &source.value,
                        "MiniMax cover reference must not be empty",
                    )?,
                    MinimaxMusicCoverReference::FeatureId(value) => validate_non_empty(
                        value,
                        "MiniMax cover feature identifier must not be empty",
                    )?,
                }
                if let Some(lyrics) = lyrics {
                    validate_character_count(
                        lyrics,
                        10,
                        1_000,
                        "MiniMax cover lyrics must contain between 10 and 1000 characters",
                    )?;
                } else if matches!(reference, MinimaxMusicCoverReference::FeatureId(_)) {
                    return Err(invalid_input(
                        "MiniMax cover feature requests require caller-provided lyrics",
                    ));
                }
            }
        }
        Ok(())
    }

    fn wire(&self) -> MusicRequestWire<'_> {
        let mut wire = MusicRequestWire {
            model: &self.model,
            prompt: None,
            lyrics: None,
            stream: false,
            output_format: self.output_format.as_str(),
            audio_setting: self.audio_setting.map(MusicAudioSettingWire::from),
            lyrics_optimizer: None,
            is_instrumental: None,
            audio_url: None,
            audio_base64: None,
            cover_feature_id: None,
        };
        match &self.input {
            MinimaxMusicInput::Lyrics { prompt, lyrics } => {
                wire.prompt = prompt.as_deref();
                wire.lyrics = Some(lyrics);
            }
            MinimaxMusicInput::GeneratedLyrics { prompt } => {
                wire.prompt = Some(prompt);
                wire.lyrics_optimizer = Some(true);
            }
            MinimaxMusicInput::Instrumental { prompt } => {
                wire.prompt = Some(prompt);
                wire.is_instrumental = Some(true);
            }
            MinimaxMusicInput::Cover {
                prompt,
                reference,
                lyrics,
            } => {
                wire.prompt = Some(prompt);
                wire.lyrics = lyrics.as_deref();
                match reference {
                    MinimaxMusicCoverReference::Direct(source) => match source.kind {
                        MinimaxMusicCoverSourceKind::Url => {
                            wire.audio_url = Some(&source.value);
                        }
                        MinimaxMusicCoverSourceKind::Base64 => {
                            wire.audio_base64 = Some(&source.value);
                        }
                    },
                    MinimaxMusicCoverReference::FeatureId(value) => {
                        wire.cover_feature_id = Some(value);
                    }
                }
            }
        }
        wire
    }
}

impl fmt::Debug for MinimaxMusicRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxMusicRequest")
            .field("model", &self.model)
            .field("kind", &self.input.kind())
            .field("prompt_characters", &self.input.prompt_characters())
            .field("lyrics_characters", &self.input.lyrics_characters())
            .field("cover_reference", &self.input.cover_reference_kind())
            .field("output_format", &self.output_format)
            .field("audio_setting", &self.audio_setting)
            .finish()
    }
}

#[derive(Clone)]
enum MinimaxMusicInput {
    Lyrics {
        prompt: Option<String>,
        lyrics: String,
    },
    GeneratedLyrics {
        prompt: String,
    },
    Instrumental {
        prompt: String,
    },
    Cover {
        prompt: String,
        reference: MinimaxMusicCoverReference,
        lyrics: Option<String>,
    },
}

impl MinimaxMusicInput {
    const fn kind(&self) -> MinimaxMusicRequestKind {
        match self {
            Self::Lyrics { .. } => MinimaxMusicRequestKind::Lyrics,
            Self::GeneratedLyrics { .. } => MinimaxMusicRequestKind::GeneratedLyrics,
            Self::Instrumental { .. } => MinimaxMusicRequestKind::Instrumental,
            Self::Cover { .. } => MinimaxMusicRequestKind::Cover,
        }
    }

    fn prompt_characters(&self) -> Option<usize> {
        match self {
            Self::Lyrics { prompt, .. } => prompt.as_ref().map(|value| value.chars().count()),
            Self::GeneratedLyrics { prompt }
            | Self::Instrumental { prompt }
            | Self::Cover { prompt, .. } => Some(prompt.chars().count()),
        }
    }

    fn lyrics_characters(&self) -> Option<usize> {
        match self {
            Self::Lyrics { lyrics, .. } => Some(lyrics.chars().count()),
            Self::Cover { lyrics, .. } => lyrics.as_ref().map(|value| value.chars().count()),
            Self::GeneratedLyrics { .. } | Self::Instrumental { .. } => None,
        }
    }

    const fn cover_reference_kind(&self) -> Option<&'static str> {
        match self {
            Self::Cover {
                reference:
                    MinimaxMusicCoverReference::Direct(MinimaxMusicCoverSource {
                        kind: MinimaxMusicCoverSourceKind::Url,
                        ..
                    }),
                ..
            } => Some("url"),
            Self::Cover {
                reference:
                    MinimaxMusicCoverReference::Direct(MinimaxMusicCoverSource {
                        kind: MinimaxMusicCoverSourceKind::Base64,
                        ..
                    }),
                ..
            } => Some("base64"),
            Self::Cover {
                reference: MinimaxMusicCoverReference::FeatureId(_),
                ..
            } => Some("feature_id"),
            _ => None,
        }
    }
}

#[derive(Clone)]
enum MinimaxMusicCoverReference {
    Direct(MinimaxMusicCoverSource),
    FeatureId(String),
}

#[derive(Serialize)]
struct MusicRequestWire<'a> {
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lyrics: Option<&'a str>,
    stream: bool,
    output_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_setting: Option<MusicAudioSettingWire>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lyrics_optimizer: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_instrumental: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_url: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_base64: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cover_feature_id: Option<&'a str>,
}

#[derive(Serialize)]
struct MusicAudioSettingWire {
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bitrate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<&'static str>,
}

impl From<MinimaxMusicAudioSetting> for MusicAudioSettingWire {
    fn from(value: MinimaxMusicAudioSetting) -> Self {
        Self {
            sample_rate: value.sample_rate.map(MinimaxMusicSampleRate::hertz),
            bitrate: value.bitrate.map(MinimaxMusicBitrate::bits_per_second),
            format: value.format.map(MinimaxMusicAudioFormat::as_str),
        }
    }
}

/// Music-generation status returned by MiniMax.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxMusicStatus {
    InProgress,
    Completed,
    Unknown(i64),
}

impl From<i64> for MinimaxMusicStatus {
    fn from(value: i64) -> Self {
        match value {
            1 => Self::InProgress,
            2 => Self::Completed,
            value => Self::Unknown(value),
        }
    }
}

/// Generated audio returned by MiniMax.
#[derive(Clone)]
#[non_exhaustive]
pub enum MinimaxMusicOutput {
    /// A provider URL that may expire after 24 hours.
    Url(String),
    /// Bytes decoded from MiniMax's hexadecimal JSON field.
    Audio(Vec<u8>),
}

impl MinimaxMusicOutput {
    pub fn as_url(&self) -> Option<&str> {
        match self {
            Self::Url(value) => Some(value),
            Self::Audio(_) => None,
        }
    }

    pub fn as_audio(&self) -> Option<&[u8]> {
        match self {
            Self::Url(_) => None,
            Self::Audio(value) => Some(value),
        }
    }

    pub fn into_url(self) -> Option<String> {
        match self {
            Self::Url(value) => Some(value),
            Self::Audio(_) => None,
        }
    }

    pub fn into_audio(self) -> Option<Vec<u8>> {
        match self {
            Self::Url(_) => None,
            Self::Audio(value) => Some(value),
        }
    }
}

impl fmt::Debug for MinimaxMusicOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Url(value) => formatter
                .debug_struct("MinimaxMusicOutput::Url")
                .field("value", &"redacted")
                .field("bytes", &value.len())
                .finish(),
            Self::Audio(value) => formatter
                .debug_struct("MinimaxMusicOutput::Audio")
                .field("bytes", &value.len())
                .finish(),
        }
    }
}

/// Numeric metadata reported by MiniMax for generated music.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MinimaxMusicMetadata {
    duration: Option<u64>,
    sample_rate: Option<u32>,
    channels: Option<u32>,
    bitrate: Option<u32>,
    size: Option<u64>,
}

impl MinimaxMusicMetadata {
    /// Return MiniMax's raw `music_duration` value. The current OpenAPI does
    /// not define its unit.
    pub const fn duration(self) -> Option<u64> {
        self.duration
    }

    pub const fn sample_rate(self) -> Option<u32> {
        self.sample_rate
    }

    pub const fn channels(self) -> Option<u32> {
        self.channels
    }

    pub const fn bitrate(self) -> Option<u32> {
        self.bitrate
    }

    pub const fn size(self) -> Option<u64> {
        self.size
    }
}

/// A MiniMax music-generation response.
#[derive(Clone)]
pub struct MinimaxMusicGeneration {
    status: MinimaxMusicStatus,
    output: Option<MinimaxMusicOutput>,
    trace_id: Option<String>,
    metadata: MinimaxMusicMetadata,
    analysis_info: Option<Value>,
    extra: BTreeMap<String, Value>,
    data_extra: BTreeMap<String, Value>,
}

impl MinimaxMusicGeneration {
    pub const fn status(&self) -> MinimaxMusicStatus {
        self.status
    }

    pub fn output(&self) -> Option<&MinimaxMusicOutput> {
        self.output.as_ref()
    }

    pub fn into_output(self) -> Option<MinimaxMusicOutput> {
        self.output
    }

    pub fn trace_id(&self) -> Option<&str> {
        self.trace_id.as_deref()
    }

    pub const fn metadata(&self) -> MinimaxMusicMetadata {
        self.metadata
    }

    pub fn analysis_info(&self) -> Option<&Value> {
        self.analysis_info.as_ref()
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }

    pub fn data_extra(&self) -> &BTreeMap<String, Value> {
        &self.data_extra
    }
}

impl fmt::Debug for MinimaxMusicGeneration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxMusicGeneration")
            .field("status", &self.status)
            .field("output", &self.output)
            .field("has_trace_id", &self.trace_id.is_some())
            .field("metadata", &self.metadata)
            .field("has_analysis_info", &self.analysis_info.is_some())
            .field("extra_field_count", &self.extra.len())
            .field("data_extra_field_count", &self.data_extra.len())
            .finish()
    }
}

/// Shared, lightweight handle for MiniMax's provider-native Music API.
///
/// This handle implements the non-streaming JSON operation. Streaming music
/// requires a distinct bounded stream decoder and is not silently routed
/// through this method.
#[derive(Clone)]
pub struct MinimaxMusic {
    runtime: Arc<NativeRuntime>,
}

impl MinimaxMusic {
    pub(crate) fn new(runtime: Arc<NativeRuntime>) -> Self {
        Self { runtime }
    }

    pub async fn generate(
        &self,
        request: MinimaxMusicRequest,
    ) -> Result<MinimaxMusicGeneration, Error> {
        self.generate_with_options(request, CallOptions::default())
            .await
    }

    pub async fn generate_with_options(
        &self,
        request: MinimaxMusicRequest,
        options: CallOptions,
    ) -> Result<MinimaxMusicGeneration, Error> {
        request.validate()?;
        let expected_format = request.output_format;
        let body = RequestBody::json(&request.wire()).map_err(|source| {
            Error::new(ErrorKind::InvalidInput, "MiniMax music request is invalid")
                .with_source(source)
        })?;
        let response: MusicResponseEnvelope = execute_json(
            &self.runtime,
            Method::POST,
            target(MUSIC_GENERATION_TARGET)?,
            body,
            ReplaySafety::Never,
            options,
        )
        .await?;
        response.into_generation(expected_format)
    }
}

impl fmt::Debug for MinimaxMusic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxMusic")
            .field("runtime", &"shared")
            .finish()
    }
}

#[derive(Deserialize)]
struct MusicResponseEnvelope {
    #[serde(default)]
    data: Option<MusicDataWire>,
    #[serde(default)]
    trace_id: Option<String>,
    #[serde(default)]
    extra_info: Option<MusicMetadataWire>,
    #[serde(default)]
    analysis_info: Option<Value>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl MusicResponseEnvelope {
    fn into_generation(
        self,
        expected_format: MinimaxMusicOutputFormat,
    ) -> Result<MinimaxMusicGeneration, Error> {
        let data = self.data.ok_or_else(|| {
            Error::new(
                ErrorKind::ProtocolViolation,
                "MiniMax music response omitted generation data",
            )
        })?;
        let status = data.status.map(MinimaxMusicStatus::from).ok_or_else(|| {
            Error::new(
                ErrorKind::ProtocolViolation,
                "MiniMax music response omitted generation status",
            )
        })?;
        let output = data
            .audio
            .map(|audio| decode_music_output(expected_format, audio))
            .transpose()?;
        if status == MinimaxMusicStatus::Completed && output.is_none() {
            return Err(Error::new(
                ErrorKind::ProtocolViolation,
                "MiniMax completed music response omitted generated audio",
            ));
        }
        let metadata = self
            .extra_info
            .map(MinimaxMusicMetadata::from)
            .unwrap_or_default();
        Ok(MinimaxMusicGeneration {
            status,
            output,
            trace_id: self.trace_id,
            metadata,
            analysis_info: self.analysis_info,
            extra: self.extra,
            data_extra: data.extra,
        })
    }
}

impl NativeResponseEnvelope for MusicResponseEnvelope {
    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

#[derive(Deserialize)]
struct MusicDataWire {
    #[serde(default)]
    status: Option<i64>,
    #[serde(default)]
    audio: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct MusicMetadataWire {
    #[serde(default)]
    music_duration: Option<u64>,
    #[serde(default)]
    music_sample_rate: Option<u32>,
    #[serde(default)]
    music_channel: Option<u32>,
    #[serde(default)]
    bitrate: Option<u32>,
    #[serde(default)]
    music_size: Option<u64>,
}

impl From<MusicMetadataWire> for MinimaxMusicMetadata {
    fn from(value: MusicMetadataWire) -> Self {
        Self {
            duration: value.music_duration,
            sample_rate: value.music_sample_rate,
            channels: value.music_channel,
            bitrate: value.bitrate,
            size: value.music_size,
        }
    }
}

fn decode_music_output(
    expected_format: MinimaxMusicOutputFormat,
    value: String,
) -> Result<MinimaxMusicOutput, Error> {
    if value.is_empty() {
        return Err(Error::new(
            ErrorKind::ProtocolViolation,
            "MiniMax music response returned empty generated audio",
        ));
    }
    match expected_format {
        MinimaxMusicOutputFormat::Url => Ok(MinimaxMusicOutput::Url(value)),
        MinimaxMusicOutputFormat::Hex => {
            decode_hex(value.as_bytes()).map(MinimaxMusicOutput::Audio)
        }
    }
}

fn decode_hex(value: &[u8]) -> Result<Vec<u8>, Error> {
    if !value.len().is_multiple_of(2) {
        return Err(invalid_music_hex());
    }
    value
        .chunks_exact(2)
        .map(|pair| {
            let high = decode_hex_digit(pair[0]).ok_or_else(invalid_music_hex)?;
            let low = decode_hex_digit(pair[1]).ok_or_else(invalid_music_hex)?;
            Ok((high << 4) | low)
        })
        .collect()
}

const fn decode_hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn invalid_music_hex() -> Error {
    Error::new(
        ErrorKind::ProtocolViolation,
        "MiniMax music response returned invalid hexadecimal audio",
    )
}

fn validate_model_workflow(model: &str, kind: MinimaxMusicRequestKind) -> Result<(), Error> {
    if is_known_cover_model(model) && kind != MinimaxMusicRequestKind::Cover {
        return Err(invalid_input(
            "MiniMax cover models require a cover-generation request",
        ));
    }
    if is_known_text_music_model(model) && kind == MinimaxMusicRequestKind::Cover {
        return Err(invalid_input(
            "MiniMax text-to-music models do not accept cover inputs",
        ));
    }
    Ok(())
}

fn is_known_text_music_model(model: &str) -> bool {
    matches!(
        model,
        MUSIC_3_0 | MUSIC_3_0_FREE | MUSIC_2_6 | MUSIC_2_6_FREE
    )
}

fn is_known_cover_model(model: &str) -> bool {
    matches!(model, MUSIC_COVER | MUSIC_COVER_FREE)
}

fn validate_character_count(
    value: &str,
    minimum: usize,
    maximum: usize,
    message: &'static str,
) -> Result<(), Error> {
    let count = value.chars().count();
    if !(minimum..=maximum).contains(&count) {
        return Err(invalid_input(message));
    }
    Ok(())
}

fn validate_non_empty(value: &str, message: &'static str) -> Result<(), Error> {
    if value.trim().is_empty() {
        return Err(invalid_input(message));
    }
    Ok(())
}

fn invalid_input(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::common::validate_base_response;

    #[test]
    fn generated_lyrics_request_encodes_the_native_wire_contract() {
        let request = MinimaxMusicRequest::generated_lyrics(
            MUSIC_3_0,
            "private melancholic indie folk prompt",
        )
        .expect("request should be valid")
        .with_output_format(MinimaxMusicOutputFormat::Url)
        .with_audio_setting(
            MinimaxMusicAudioSetting::new()
                .with_sample_rate(MinimaxMusicSampleRate::Hz44100)
                .with_bitrate(MinimaxMusicBitrate::Bps256000)
                .with_format(MinimaxMusicAudioFormat::Mp3),
        );

        assert_eq!(
            serde_json::to_value(request.wire()).expect("request should serialize"),
            serde_json::json!({
                "model": "music-3.0",
                "prompt": "private melancholic indie folk prompt",
                "stream": false,
                "output_format": "url",
                "audio_setting": {
                    "sample_rate": 44100,
                    "bitrate": 256000,
                    "format": "mp3"
                },
                "lyrics_optimizer": true
            })
        );
    }

    #[test]
    fn cover_request_encodes_exactly_one_reference_source() {
        let request = MinimaxMusicRequest::cover(
            MUSIC_COVER,
            "dreamy acoustic cover style",
            MinimaxMusicCoverSource::base64("U1VQRVJfU0VDUkVUX0FVRElP")
                .expect("source should be valid"),
        )
        .expect("cover request should be valid")
        .with_cover_lyrics("[Verse]\nA sufficiently long private lyric")
        .expect("cover lyrics should be valid");

        let wire = serde_json::to_value(request.wire()).expect("request should serialize");
        assert_eq!(
            wire,
            serde_json::json!({
                "model": "music-cover",
                "prompt": "dreamy acoustic cover style",
                "lyrics": "[Verse]\nA sufficiently long private lyric",
                "stream": false,
                "output_format": "hex",
                "audio_base64": "U1VQRVJfU0VDUkVUX0FVRElP"
            })
        );
        assert!(wire.get("audio_url").is_none());
        assert!(wire.get("cover_feature_id").is_none());
        assert!(wire.get("lyrics_optimizer").is_none());
        assert!(wire.get("is_instrumental").is_none());
    }

    #[test]
    fn music_validation_is_operation_and_model_specific() {
        let source = MinimaxMusicCoverSource::url("https://example.invalid/private.mp3")
            .expect("source should be valid");
        assert_eq!(
            MinimaxMusicRequest::cover(MUSIC_3_0, "dreamy acoustic cover style", source)
                .expect_err("text-to-music model must reject cover input")
                .kind(),
            ErrorKind::InvalidInput
        );

        assert_eq!(
            MinimaxMusicRequest::from_lyrics(MUSIC_COVER, "valid original lyrics")
                .expect_err("cover model must reject text-to-music input")
                .kind(),
            ErrorKind::InvalidInput
        );

        assert_eq!(
            MinimaxMusicRequest::cover_from_feature(
                MUSIC_COVER,
                "dreamy acoustic cover style",
                "feature-id",
                "short",
            )
            .expect_err("feature cover lyrics must satisfy the cover limit")
            .kind(),
            ErrorKind::InvalidInput
        );

        MinimaxMusicRequest::instrumental("music-future", "future ambient instrumental")
            .expect("future model ids should remain open for an explicit workflow");
    }

    #[test]
    fn music_response_decodes_hex_and_preserves_unknown_fields() {
        let response: MusicResponseEnvelope = serde_json::from_value(serde_json::json!({
            "data": {
                "status": 2,
                "audio": "00ffA1",
                "future_data": {"nested": true}
            },
            "trace_id": "private-trace-id",
            "extra_info": {
                "music_duration": 25364,
                "music_sample_rate": 44100,
                "music_channel": 2,
                "bitrate": 256000,
                "music_size": 3
            },
            "analysis_info": {"future_analysis": true},
            "future_response": [1, 2, 3],
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }))
        .expect("response should decode");
        validate_base_response(response.base_response()).expect("base response should succeed");

        let generation = response
            .into_generation(MinimaxMusicOutputFormat::Hex)
            .expect("generation should decode");
        assert_eq!(generation.status(), MinimaxMusicStatus::Completed);
        assert_eq!(
            generation.output().and_then(MinimaxMusicOutput::as_audio),
            Some(&[0x00, 0xff, 0xa1][..])
        );
        assert_eq!(generation.trace_id(), Some("private-trace-id"));
        assert_eq!(generation.metadata().duration(), Some(25_364));
        assert_eq!(generation.metadata().sample_rate(), Some(44_100));
        assert_eq!(generation.metadata().channels(), Some(2));
        assert_eq!(generation.metadata().bitrate(), Some(256_000));
        assert_eq!(generation.metadata().size(), Some(3));
        assert!(generation.analysis_info().is_some());
        assert!(generation.extra().contains_key("future_response"));
        assert!(generation.data_extra().contains_key("future_data"));
    }

    #[test]
    fn music_response_rejects_invalid_hex_audio() {
        let response: MusicResponseEnvelope = serde_json::from_value(serde_json::json!({
            "data": {"status": 2, "audio": "not-hex"},
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }))
        .expect("response should decode");

        assert_eq!(
            response
                .into_generation(MinimaxMusicOutputFormat::Hex)
                .expect_err("invalid hex must fail")
                .kind(),
            ErrorKind::ProtocolViolation
        );
    }

    #[test]
    fn music_debug_output_redacts_prompts_lyrics_sources_and_urls() {
        let request = MinimaxMusicRequest::cover(
            MUSIC_COVER,
            "SUPER_SECRET_COVER_PROMPT",
            MinimaxMusicCoverSource::base64("SUPER_SECRET_AUDIO_BASE64")
                .expect("source should be valid"),
        )
        .expect("request should be valid")
        .with_cover_lyrics("SUPER_SECRET_LYRICS_LONG_ENOUGH")
        .expect("lyrics should be valid");
        let request_debug = format!("{request:?}");
        assert!(!request_debug.contains("SUPER_SECRET_COVER_PROMPT"));
        assert!(!request_debug.contains("SUPER_SECRET_AUDIO_BASE64"));
        assert!(!request_debug.contains("SUPER_SECRET_LYRICS_LONG_ENOUGH"));

        let source_debug = format!(
            "{:?}",
            request.cover_source().expect("cover source should exist")
        );
        assert!(!source_debug.contains("SUPER_SECRET_AUDIO_BASE64"));

        let response: MusicResponseEnvelope = serde_json::from_value(serde_json::json!({
            "data": {
                "status": 2,
                "audio": "https://example.invalid/generated.mp3?token=SUPER_SECRET_OUTPUT"
            },
            "trace_id": "SUPER_SECRET_TRACE",
            "analysis_info": {"secret": "SUPER_SECRET_ANALYSIS"},
            "future_secret": "SUPER_SECRET_EXTRA",
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }))
        .expect("response should decode");
        let generation = response
            .into_generation(MinimaxMusicOutputFormat::Url)
            .expect("generation should decode");
        let response_debug = format!("{generation:?}");
        assert!(!response_debug.contains("SUPER_SECRET_OUTPUT"));
        assert!(!response_debug.contains("SUPER_SECRET_TRACE"));
        assert!(!response_debug.contains("SUPER_SECRET_ANALYSIS"));
        assert!(!response_debug.contains("SUPER_SECRET_EXTRA"));
    }
}
