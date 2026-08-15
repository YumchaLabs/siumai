use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use http::{Method, Uri};
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use siumai_core::{CallOptions, Error, ErrorKind, ModelId};
use siumai_transport::{ReplaySafety, RequestBody};
use thiserror::Error as ThisError;

use crate::models::speech::{SPEECH_01_HD, SPEECH_01_TURBO, SPEECH_02_HD, SPEECH_02_TURBO};
#[cfg(test)]
use crate::models::speech::{SPEECH_2_8_HD, SPEECH_2_8_TURBO};

use super::common::{BaseResponse, NativeResponseEnvelope, NativeRuntime, execute_json, target};
use super::files::MinimaxFileId;

const SYNTHESIS_TARGET: &str = "v1/t2a_v2";
const ASYNC_SUBMIT_TARGET: &str = "v1/t2a_async_v2";
const ASYNC_QUERY_TARGET: &str = "v1/query/t2a_async_query_v2";
const SYNCHRONOUS_TEXT_LIMIT: usize = 9_999;
const ASYNC_TEXT_LIMIT: usize = 50_000;
const MAX_PRONUNCIATION_RULES: usize = 1_024;
const MAX_PRONUNCIATION_RULE_BYTES: usize = 4_096;

/// Official MiniMax synchronous Speech API reference used by this module.
pub const SPEECH_HTTP_API_SOURCE: &str =
    "https://platform.minimax.io/docs/api-reference/speech-t2a-http";
/// Official MiniMax asynchronous Speech API reference used by this module.
pub const SPEECH_ASYNC_API_SOURCE: &str =
    "https://platform.minimax.io/docs/api-reference/speech-t2a-async-create";
/// Official MiniMax asynchronous task-query reference used by this module.
pub const SPEECH_ASYNC_QUERY_API_SOURCE: &str =
    "https://platform.minimax.io/docs/api-reference/speech-t2a-async-query";
/// Date on which the Speech API contract in this module was verified.
pub const SPEECH_API_VERIFIED_ON: &str = "2026-08-15";

/// Validation errors for MiniMax speech settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
#[non_exhaustive]
pub enum MinimaxSpeechValueError {
    #[error("MiniMax voice identifier must be non-empty and contain no control characters")]
    InvalidVoiceId,
    #[error("MiniMax speech speed must be finite and in the range 0.5 through 2.0")]
    InvalidSpeed,
    #[error("MiniMax speech volume must be finite and greater than 0 through 10.0")]
    InvalidVolume,
    #[error("MiniMax speech pitch must be in the range -12 through 12")]
    InvalidPitch,
    #[error("MiniMax voice-effect level must be in the range -100 through 100")]
    InvalidVoiceEffectLevel,
    #[error("MiniMax language boost must be non-empty and contain no control characters")]
    InvalidLanguageBoost,
    #[error("MiniMax pronunciation dictionary must contain at least one rule")]
    EmptyPronunciationDictionary,
    #[error("MiniMax pronunciation dictionary contains too many rules")]
    TooManyPronunciationRules,
    #[error("MiniMax pronunciation rule is empty, too large, or contains control characters")]
    InvalidPronunciationRule,
}

/// Open provider-owned voice identifier.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MinimaxVoiceId(String);

impl MinimaxVoiceId {
    pub fn new(value: impl Into<String>) -> Result<Self, MinimaxSpeechValueError> {
        let value = value.into();
        if value.trim().is_empty()
            || value != value.trim()
            || value.len() > 2_048
            || value.chars().any(char::is_control)
        {
            return Err(MinimaxSpeechValueError::InvalidVoiceId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for MinimaxVoiceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVoiceId")
            .field("bytes", &self.0.len())
            .finish()
    }
}

impl fmt::Display for MinimaxVoiceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Serialize for MinimaxVoiceId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for MinimaxVoiceId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

/// Validated MiniMax speech speed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinimaxSpeechSpeed(f32);

impl MinimaxSpeechSpeed {
    pub const NORMAL: Self = Self(1.0);

    pub fn new(value: f32) -> Result<Self, MinimaxSpeechValueError> {
        if !value.is_finite() || !(0.5..=2.0).contains(&value) {
            return Err(MinimaxSpeechValueError::InvalidSpeed);
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> f32 {
        self.0
    }
}

/// Validated MiniMax speech volume.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinimaxSpeechVolume(f32);

impl MinimaxSpeechVolume {
    pub const NORMAL: Self = Self(1.0);

    pub fn new(value: f32) -> Result<Self, MinimaxSpeechValueError> {
        if !value.is_finite() || value <= 0.0 || value > 10.0 {
            return Err(MinimaxSpeechValueError::InvalidVolume);
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> f32 {
        self.0
    }
}

/// Validated MiniMax speech pitch offset in semitones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MinimaxSpeechPitch(i8);

impl MinimaxSpeechPitch {
    pub const ORIGINAL: Self = Self(0);

    pub fn new(value: i8) -> Result<Self, MinimaxSpeechValueError> {
        if !(-12..=12).contains(&value) {
            return Err(MinimaxSpeechValueError::InvalidPitch);
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> i8 {
        self.0
    }
}

/// Explicit MiniMax emotion override.
///
/// An omitted value lets MiniMax select an emotion from the text. No synthetic
/// `neutral` or other undocumented default is sent by this library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MinimaxSpeechEmotion {
    Happy,
    Sad,
    Angry,
    Fearful,
    Disgusted,
    Surprised,
    Calm,
    Fluent,
    Whisper,
}

/// Settings for one MiniMax voice.
#[derive(Clone, PartialEq)]
pub struct MinimaxVoiceSettings {
    voice_id: MinimaxVoiceId,
    speed: Option<MinimaxSpeechSpeed>,
    volume: Option<MinimaxSpeechVolume>,
    pitch: Option<MinimaxSpeechPitch>,
    emotion: Option<MinimaxSpeechEmotion>,
}

impl MinimaxVoiceSettings {
    pub fn new(voice_id: MinimaxVoiceId) -> Self {
        Self {
            voice_id,
            speed: None,
            volume: None,
            pitch: None,
            emotion: None,
        }
    }

    pub fn with_speed(mut self, speed: MinimaxSpeechSpeed) -> Self {
        self.speed = Some(speed);
        self
    }

    pub fn with_volume(mut self, volume: MinimaxSpeechVolume) -> Self {
        self.volume = Some(volume);
        self
    }

    pub fn with_pitch(mut self, pitch: MinimaxSpeechPitch) -> Self {
        self.pitch = Some(pitch);
        self
    }

    pub fn with_emotion(mut self, emotion: MinimaxSpeechEmotion) -> Self {
        self.emotion = Some(emotion);
        self
    }

    pub fn voice_id(&self) -> &MinimaxVoiceId {
        &self.voice_id
    }

    pub const fn speed(&self) -> Option<MinimaxSpeechSpeed> {
        self.speed
    }

    pub const fn volume(&self) -> Option<MinimaxSpeechVolume> {
        self.volume
    }

    pub const fn pitch(&self) -> Option<MinimaxSpeechPitch> {
        self.pitch
    }

    pub const fn emotion(&self) -> Option<MinimaxSpeechEmotion> {
        self.emotion
    }
}

impl fmt::Debug for MinimaxVoiceSettings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxVoiceSettings")
            .field("voice_id", &self.voice_id)
            .field("speed", &self.speed)
            .field("volume", &self.volume)
            .field("pitch", &self.pitch)
            .field("emotion", &self.emotion)
            .finish()
    }
}

/// Open MiniMax language or dialect boost value.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MinimaxLanguageBoost(String);

impl MinimaxLanguageBoost {
    pub fn new(value: impl Into<String>) -> Result<Self, MinimaxSpeechValueError> {
        let value = value.into();
        if value.trim().is_empty()
            || value != value.trim()
            || value.len() > 256
            || value.chars().any(char::is_control)
        {
            return Err(MinimaxSpeechValueError::InvalidLanguageBoost);
        }
        Ok(Self(value))
    }

    pub fn auto() -> Self {
        Self("auto".to_owned())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for MinimaxLanguageBoost {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("MinimaxLanguageBoost")
            .field(&self.0)
            .finish()
    }
}

impl Serialize for MinimaxLanguageBoost {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

/// Operation-bounded custom pronunciation rules.
#[derive(Clone, PartialEq, Eq)]
pub struct MinimaxPronunciationDictionary {
    rules: Vec<String>,
}

impl MinimaxPronunciationDictionary {
    pub fn new<I, S>(rules: I) -> Result<Self, MinimaxSpeechValueError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let rules = rules.into_iter().map(Into::into).collect::<Vec<_>>();
        if rules.is_empty() {
            return Err(MinimaxSpeechValueError::EmptyPronunciationDictionary);
        }
        if rules.len() > MAX_PRONUNCIATION_RULES {
            return Err(MinimaxSpeechValueError::TooManyPronunciationRules);
        }
        if rules.iter().any(|rule| {
            rule.trim().is_empty()
                || rule.len() > MAX_PRONUNCIATION_RULE_BYTES
                || rule.chars().any(char::is_control)
        }) {
            return Err(MinimaxSpeechValueError::InvalidPronunciationRule);
        }
        Ok(Self { rules })
    }

    pub fn rules(&self) -> &[String] {
        &self.rules
    }
}

impl fmt::Debug for MinimaxPronunciationDictionary {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxPronunciationDictionary")
            .field("rule_count", &self.rules.len())
            .finish()
    }
}

impl Serialize for MinimaxPronunciationDictionary {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct Wire<'a> {
            tone: &'a [String],
        }

        Wire { tone: &self.rules }.serialize(serializer)
    }
}

/// Supported MiniMax audio container or codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MinimaxSpeechAudioFormat {
    Mp3,
    Pcm,
    Flac,
    Wav,
    PcmuRaw,
    PcmuWav,
    Opus,
}

/// Supported MiniMax audio sample rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MinimaxSpeechSampleRate {
    Hz8000,
    Hz12000,
    Hz16000,
    Hz22050,
    Hz24000,
    Hz32000,
    Hz44100,
    Hz48000,
}

impl MinimaxSpeechSampleRate {
    pub const fn hertz(self) -> u32 {
        match self {
            Self::Hz8000 => 8_000,
            Self::Hz12000 => 12_000,
            Self::Hz16000 => 16_000,
            Self::Hz22050 => 22_050,
            Self::Hz24000 => 24_000,
            Self::Hz32000 => 32_000,
            Self::Hz44100 => 44_100,
            Self::Hz48000 => 48_000,
        }
    }
}

impl Serialize for MinimaxSpeechSampleRate {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u32(self.hertz())
    }
}

/// Supported MiniMax MP3 bitrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MinimaxSpeechBitrate {
    Kbps32,
    Kbps64,
    Kbps128,
    Kbps256,
}

impl MinimaxSpeechBitrate {
    pub const fn bits_per_second(self) -> u32 {
        match self {
            Self::Kbps32 => 32_000,
            Self::Kbps64 => 64_000,
            Self::Kbps128 => 128_000,
            Self::Kbps256 => 256_000,
        }
    }
}

impl Serialize for MinimaxSpeechBitrate {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u32(self.bits_per_second())
    }
}

/// Requested channel layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MinimaxSpeechChannels {
    Mono,
    Stereo,
}

impl MinimaxSpeechChannels {
    pub const fn count(self) -> u8 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
        }
    }
}

impl Serialize for MinimaxSpeechChannels {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(self.count())
    }
}

/// Audio encoding settings shared by synchronous and asynchronous speech.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinimaxSpeechAudioSettings {
    format: MinimaxSpeechAudioFormat,
    sample_rate: Option<MinimaxSpeechSampleRate>,
    bitrate: Option<MinimaxSpeechBitrate>,
    channels: Option<MinimaxSpeechChannels>,
}

impl MinimaxSpeechAudioSettings {
    pub const fn new(format: MinimaxSpeechAudioFormat) -> Self {
        Self {
            format,
            sample_rate: None,
            bitrate: None,
            channels: None,
        }
    }

    pub const fn with_sample_rate(mut self, sample_rate: MinimaxSpeechSampleRate) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    pub const fn with_bitrate(mut self, bitrate: MinimaxSpeechBitrate) -> Self {
        self.bitrate = Some(bitrate);
        self
    }

    pub const fn with_channels(mut self, channels: MinimaxSpeechChannels) -> Self {
        self.channels = Some(channels);
        self
    }

    pub const fn format(&self) -> MinimaxSpeechAudioFormat {
        self.format
    }

    pub const fn sample_rate(&self) -> Option<MinimaxSpeechSampleRate> {
        self.sample_rate
    }

    pub const fn bitrate(&self) -> Option<MinimaxSpeechBitrate> {
        self.bitrate
    }

    pub const fn channels(&self) -> Option<MinimaxSpeechChannels> {
        self.channels
    }
}

impl Default for MinimaxSpeechAudioSettings {
    fn default() -> Self {
        Self::new(MinimaxSpeechAudioFormat::Mp3)
    }
}

/// Validated MiniMax voice-effect level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MinimaxVoiceEffectLevel(i8);

impl MinimaxVoiceEffectLevel {
    pub const NEUTRAL: Self = Self(0);

    pub fn new(value: i8) -> Result<Self, MinimaxSpeechValueError> {
        if !(-100..=100).contains(&value) {
            return Err(MinimaxSpeechValueError::InvalidVoiceEffectLevel);
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> i8 {
        self.0
    }
}

/// MiniMax provider-native sound effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MinimaxSoundEffect {
    SpaciousEcho,
    AuditoriumEcho,
    LofiTelephone,
    Robotic,
}

/// Optional MiniMax voice post-processing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MinimaxVoiceEffects {
    pitch: Option<MinimaxVoiceEffectLevel>,
    intensity: Option<MinimaxVoiceEffectLevel>,
    timbre: Option<MinimaxVoiceEffectLevel>,
    sound_effect: Option<MinimaxSoundEffect>,
}

impl MinimaxVoiceEffects {
    pub const fn new() -> Self {
        Self {
            pitch: None,
            intensity: None,
            timbre: None,
            sound_effect: None,
        }
    }

    pub const fn with_pitch(mut self, pitch: MinimaxVoiceEffectLevel) -> Self {
        self.pitch = Some(pitch);
        self
    }

    pub const fn with_intensity(mut self, intensity: MinimaxVoiceEffectLevel) -> Self {
        self.intensity = Some(intensity);
        self
    }

    pub const fn with_timbre(mut self, timbre: MinimaxVoiceEffectLevel) -> Self {
        self.timbre = Some(timbre);
        self
    }

    pub const fn with_sound_effect(mut self, sound_effect: MinimaxSoundEffect) -> Self {
        self.sound_effect = Some(sound_effect);
        self
    }

    pub const fn is_empty(&self) -> bool {
        self.pitch.is_none()
            && self.intensity.is_none()
            && self.timbre.is_none()
            && self.sound_effect.is_none()
    }
}

impl Serialize for MinimaxVoiceEffects {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct Wire {
            #[serde(skip_serializing_if = "Option::is_none")]
            pitch: Option<i8>,
            #[serde(skip_serializing_if = "Option::is_none")]
            intensity: Option<i8>,
            #[serde(skip_serializing_if = "Option::is_none")]
            timbre: Option<i8>,
            #[serde(skip_serializing_if = "Option::is_none")]
            sound_effects: Option<MinimaxSoundEffect>,
        }

        Wire {
            pitch: self.pitch.map(MinimaxVoiceEffectLevel::get),
            intensity: self.intensity.map(MinimaxVoiceEffectLevel::get),
            timbre: self.timbre.map(MinimaxVoiceEffectLevel::get),
            sound_effects: self.sound_effect,
        }
        .serialize(serializer)
    }
}

/// Non-streaming response representation for `/v1/t2a_v2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MinimaxSpeechOutputFormat {
    Hex,
    Url,
}

/// Subtitle granularity supported by non-streaming synchronous synthesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MinimaxSubtitleGranularity {
    Sentence,
    Word,
}

/// Buffered synchronous MiniMax speech request.
///
/// This type deliberately represents `stream=false`. HTTP streaming has a
/// different response contract and is not exposed as a boolean escape hatch.
/// Until the provider has a bounded typed stream decoder, callers should use
/// buffered synthesis or the asynchronous task API below.
#[derive(Clone, PartialEq)]
pub struct MinimaxSpeechSynthesisRequest {
    model: ModelId,
    text: String,
    voice: MinimaxVoiceSettings,
    audio: MinimaxSpeechAudioSettings,
    output_format: MinimaxSpeechOutputFormat,
    pronunciation: Option<MinimaxPronunciationDictionary>,
    language_boost: Option<MinimaxLanguageBoost>,
    voice_effects: Option<MinimaxVoiceEffects>,
    subtitles: Option<MinimaxSubtitleGranularity>,
    text_normalization: Option<bool>,
    latex_read: Option<bool>,
}

impl MinimaxSpeechSynthesisRequest {
    pub fn new(
        model: impl Into<String>,
        text: impl Into<String>,
        voice: MinimaxVoiceSettings,
    ) -> Result<Self, Error> {
        let text = text.into();
        validate_text(&text, SYNCHRONOUS_TEXT_LIMIT, "synchronous")?;
        Ok(Self {
            model: validated_model(model)?,
            text,
            voice,
            audio: MinimaxSpeechAudioSettings::default(),
            output_format: MinimaxSpeechOutputFormat::Hex,
            pronunciation: None,
            language_boost: None,
            voice_effects: None,
            subtitles: None,
            text_normalization: None,
            latex_read: None,
        })
    }

    pub fn with_audio(mut self, audio: MinimaxSpeechAudioSettings) -> Self {
        self.audio = audio;
        self
    }

    pub fn with_output_format(mut self, output_format: MinimaxSpeechOutputFormat) -> Self {
        self.output_format = output_format;
        self
    }

    pub fn with_pronunciation(mut self, pronunciation: MinimaxPronunciationDictionary) -> Self {
        self.pronunciation = Some(pronunciation);
        self
    }

    pub fn with_language_boost(mut self, language_boost: MinimaxLanguageBoost) -> Self {
        self.language_boost = Some(language_boost);
        self
    }

    pub fn with_voice_effects(mut self, voice_effects: MinimaxVoiceEffects) -> Self {
        self.voice_effects = (!voice_effects.is_empty()).then_some(voice_effects);
        self
    }

    pub fn with_subtitles(mut self, granularity: MinimaxSubtitleGranularity) -> Self {
        self.subtitles = Some(granularity);
        self
    }

    pub fn with_text_normalization(mut self, enabled: bool) -> Self {
        self.text_normalization = Some(enabled);
        self
    }

    pub fn with_latex_reading(mut self, enabled: bool) -> Self {
        self.latex_read = Some(enabled);
        self
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn voice(&self) -> &MinimaxVoiceSettings {
        &self.voice
    }

    pub const fn audio(&self) -> &MinimaxSpeechAudioSettings {
        &self.audio
    }

    pub const fn output_format(&self) -> MinimaxSpeechOutputFormat {
        self.output_format
    }

    fn validate(&self) -> Result<(), Error> {
        validate_voice_for_model(&self.model, &self.voice)?;
        validate_latex_language(self.latex_read, self.language_boost.as_ref())?;
        validate_audio(
            &self.audio,
            SpeechOperation::Synchronous,
            self.voice_effects.as_ref(),
        )
    }
}

impl fmt::Debug for MinimaxSpeechSynthesisRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxSpeechSynthesisRequest")
            .field("model", &self.model)
            .field("text_chars", &self.text.chars().count())
            .field("voice", &self.voice)
            .field("audio", &self.audio)
            .field("output_format", &self.output_format)
            .field("pronunciation_present", &self.pronunciation.is_some())
            .field("language_boost", &self.language_boost)
            .field("voice_effects", &self.voice_effects)
            .field("subtitles", &self.subtitles)
            .field("text_normalization", &self.text_normalization)
            .field("latex_read", &self.latex_read)
            .finish()
    }
}

/// Input for an asynchronous MiniMax speech task.
#[derive(Clone, PartialEq, Eq)]
pub struct MinimaxAsyncSpeechInput {
    kind: AsyncSpeechInputKind,
}

#[derive(Clone, PartialEq, Eq)]
enum AsyncSpeechInputKind {
    Text(String),
    TextFile(MinimaxFileId),
}

impl MinimaxAsyncSpeechInput {
    pub fn text(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        validate_text(&value, ASYNC_TEXT_LIMIT, "asynchronous")?;
        Ok(Self {
            kind: AsyncSpeechInputKind::Text(value),
        })
    }

    pub fn text_file(file_id: MinimaxFileId) -> Self {
        Self {
            kind: AsyncSpeechInputKind::TextFile(file_id),
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match &self.kind {
            AsyncSpeechInputKind::Text(text) => Some(text),
            AsyncSpeechInputKind::TextFile(_) => None,
        }
    }

    pub fn file_id(&self) -> Option<MinimaxFileId> {
        match &self.kind {
            AsyncSpeechInputKind::Text(_) => None,
            AsyncSpeechInputKind::TextFile(file_id) => Some(*file_id),
        }
    }
}

impl fmt::Debug for MinimaxAsyncSpeechInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            AsyncSpeechInputKind::Text(text) => formatter
                .debug_struct("Text")
                .field("chars", &text.chars().count())
                .finish(),
            AsyncSpeechInputKind::TextFile(file_id) => {
                formatter.debug_tuple("TextFile").field(file_id).finish()
            }
        }
    }
}

/// Request for `/v1/t2a_async_v2`.
#[derive(Clone, PartialEq)]
pub struct MinimaxAsyncSpeechRequest {
    model: ModelId,
    input: MinimaxAsyncSpeechInput,
    voice: MinimaxVoiceSettings,
    audio: MinimaxSpeechAudioSettings,
    pronunciation: Option<MinimaxPronunciationDictionary>,
    language_boost: Option<MinimaxLanguageBoost>,
    voice_effects: Option<MinimaxVoiceEffects>,
    english_normalization: Option<bool>,
    latex_read: Option<bool>,
}

impl MinimaxAsyncSpeechRequest {
    pub fn new(
        model: impl Into<String>,
        input: MinimaxAsyncSpeechInput,
        voice: MinimaxVoiceSettings,
    ) -> Result<Self, Error> {
        Ok(Self {
            model: validated_model(model)?,
            input,
            voice,
            audio: MinimaxSpeechAudioSettings::default(),
            pronunciation: None,
            language_boost: None,
            voice_effects: None,
            english_normalization: None,
            latex_read: None,
        })
    }

    pub fn with_audio(mut self, audio: MinimaxSpeechAudioSettings) -> Self {
        self.audio = audio;
        self
    }

    pub fn with_pronunciation(mut self, pronunciation: MinimaxPronunciationDictionary) -> Self {
        self.pronunciation = Some(pronunciation);
        self
    }

    pub fn with_language_boost(mut self, language_boost: MinimaxLanguageBoost) -> Self {
        self.language_boost = Some(language_boost);
        self
    }

    pub fn with_voice_effects(mut self, voice_effects: MinimaxVoiceEffects) -> Self {
        self.voice_effects = (!voice_effects.is_empty()).then_some(voice_effects);
        self
    }

    pub fn with_english_normalization(mut self, enabled: bool) -> Self {
        self.english_normalization = Some(enabled);
        self
    }

    pub fn with_latex_reading(mut self, enabled: bool) -> Self {
        self.latex_read = Some(enabled);
        self
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn input(&self) -> &MinimaxAsyncSpeechInput {
        &self.input
    }

    pub fn voice(&self) -> &MinimaxVoiceSettings {
        &self.voice
    }

    pub const fn audio(&self) -> &MinimaxSpeechAudioSettings {
        &self.audio
    }

    fn validate(&self) -> Result<(), Error> {
        validate_voice_for_model(&self.model, &self.voice)?;
        validate_latex_language(self.latex_read, self.language_boost.as_ref())?;
        validate_audio(
            &self.audio,
            SpeechOperation::Asynchronous,
            self.voice_effects.as_ref(),
        )
    }
}

impl fmt::Debug for MinimaxAsyncSpeechRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxAsyncSpeechRequest")
            .field("model", &self.model)
            .field("input", &self.input)
            .field("voice", &self.voice)
            .field("audio", &self.audio)
            .field("pronunciation_present", &self.pronunciation.is_some())
            .field("language_boost", &self.language_boost)
            .field("voice_effects", &self.voice_effects)
            .field("english_normalization", &self.english_normalization)
            .field("latex_read", &self.latex_read)
            .finish()
    }
}

/// A validated MiniMax asynchronous speech task identifier.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MinimaxSpeechTaskId(u64);

impl MinimaxSpeechTaskId {
    pub fn new(value: u64) -> Result<Self, MinimaxSpeechTaskIdError> {
        if value == 0 {
            return Err(MinimaxSpeechTaskIdError::Zero);
        }
        if value > i64::MAX as u64 {
            return Err(MinimaxSpeechTaskIdError::OutOfRange);
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Debug for MinimaxSpeechTaskId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("MinimaxSpeechTaskId")
            .field(&self.0)
            .finish()
    }
}

impl fmt::Display for MinimaxSpeechTaskId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for MinimaxSpeechTaskId {
    type Err = MinimaxSpeechTaskIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(MinimaxSpeechTaskIdError::InvalidDecimal);
        }
        let value = value
            .parse::<u64>()
            .map_err(|_| MinimaxSpeechTaskIdError::OutOfRange)?;
        Self::new(value)
    }
}

impl Serialize for MinimaxSpeechTaskId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_i64(self.0 as i64)
    }
}

impl<'de> Deserialize<'de> for MinimaxSpeechTaskId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(MinimaxSpeechTaskIdVisitor)
    }
}

struct MinimaxSpeechTaskIdVisitor;

impl Visitor<'_> for MinimaxSpeechTaskIdVisitor {
    type Value = MinimaxSpeechTaskId;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a positive signed 64-bit MiniMax speech task identifier")
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        MinimaxSpeechTaskId::new(value).map_err(E::custom)
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let value =
            u64::try_from(value).map_err(|_| E::custom(MinimaxSpeechTaskIdError::OutOfRange))?;
        MinimaxSpeechTaskId::new(value).map_err(E::custom)
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        MinimaxSpeechTaskId::from_str(value).map_err(E::custom)
    }
}

/// Validation failure for a MiniMax speech task identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
pub enum MinimaxSpeechTaskIdError {
    #[error("MiniMax speech task identifier must be greater than zero")]
    Zero,
    #[error("MiniMax speech task identifier exceeds the signed 64-bit wire range")]
    OutOfRange,
    #[error("MiniMax speech task identifier must be an unsigned decimal integer")]
    InvalidDecimal,
}

/// Opaque token returned with an asynchronous speech submission.
#[derive(Clone, PartialEq, Eq)]
pub struct MinimaxSpeechTaskToken(String);

impl MinimaxSpeechTaskToken {
    /// Explicitly expose the provider token when a native workflow requires it.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for MinimaxSpeechTaskToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("MinimaxSpeechTaskToken([REDACTED])")
    }
}

/// Receipt returned after an asynchronous speech task is accepted.
#[derive(Clone, PartialEq)]
pub struct MinimaxSpeechSubmission {
    task_id: MinimaxSpeechTaskId,
    file_id: Option<MinimaxFileId>,
    task_token: Option<MinimaxSpeechTaskToken>,
    usage_characters: Option<u64>,
    extra: BTreeMap<String, Value>,
}

impl MinimaxSpeechSubmission {
    pub const fn task_id(&self) -> MinimaxSpeechTaskId {
        self.task_id
    }

    pub const fn file_id(&self) -> Option<MinimaxFileId> {
        self.file_id
    }

    pub fn task_token(&self) -> Option<&MinimaxSpeechTaskToken> {
        self.task_token.as_ref()
    }

    pub const fn usage_characters(&self) -> Option<u64> {
        self.usage_characters
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxSpeechSubmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxSpeechSubmission")
            .field("task_id", &self.task_id)
            .field("file_id", &self.file_id)
            .field("task_token_present", &self.task_token.is_some())
            .field("usage_characters", &self.usage_characters)
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Current state of a MiniMax asynchronous speech task.
#[derive(Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MinimaxSpeechTaskStatus {
    Processing,
    Succeeded,
    Failed,
    Expired,
    Other(String),
}

impl MinimaxSpeechTaskStatus {
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Expired)
    }

    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Succeeded)
    }

    pub fn as_provider_str(&self) -> &str {
        match self {
            Self::Processing => "processing",
            Self::Succeeded => "success",
            Self::Failed => "failed",
            Self::Expired => "expired",
            Self::Other(value) => value,
        }
    }
}

impl fmt::Debug for MinimaxSpeechTaskStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_provider_str())
    }
}

impl<'de> Deserialize<'de> for MinimaxSpeechTaskStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value.is_empty() || value.len() > 128 || value.chars().any(char::is_control) {
            return Err(de::Error::custom(
                "MiniMax speech task status is empty, too large, or contains control characters",
            ));
        }
        let status = if value.eq_ignore_ascii_case("processing") {
            Self::Processing
        } else if value.eq_ignore_ascii_case("success") {
            Self::Succeeded
        } else if value.eq_ignore_ascii_case("failed") {
            Self::Failed
        } else if value.eq_ignore_ascii_case("expired") {
            Self::Expired
        } else {
            Self::Other(value)
        };
        Ok(status)
    }
}

/// Result of querying one asynchronous MiniMax speech task.
#[derive(Clone, PartialEq)]
pub struct MinimaxSpeechTask {
    task_id: MinimaxSpeechTaskId,
    status: MinimaxSpeechTaskStatus,
    file_id: Option<MinimaxFileId>,
    extra: BTreeMap<String, Value>,
}

impl MinimaxSpeechTask {
    pub const fn task_id(&self) -> MinimaxSpeechTaskId {
        self.task_id
    }

    pub const fn status(&self) -> &MinimaxSpeechTaskStatus {
        &self.status
    }

    pub const fn file_id(&self) -> Option<MinimaxFileId> {
        self.file_id
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxSpeechTask {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxSpeechTask")
            .field("task_id", &self.task_id)
            .field("status", &self.status)
            .field("file_id", &self.file_id)
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Provider-issued download URL.
///
/// The value may be signed or short-lived and is therefore omitted from
/// `Debug`. This resource does not follow the URL automatically.
#[derive(Clone, PartialEq, Eq)]
pub struct MinimaxSpeechDownloadUrl(String);

impl MinimaxSpeechDownloadUrl {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub(crate) fn from_provider(value: String) -> Result<Self, Error> {
        if value.len() > 16_384 || value.chars().any(char::is_control) {
            return Err(protocol_error(
                "MiniMax speech response returned an invalid URL",
            ));
        }
        let uri = Uri::from_str(&value).map_err(|source| {
            protocol_error("MiniMax speech response returned an invalid URL").with_source(source)
        })?;
        if !matches!(uri.scheme_str(), Some("https") | Some("http")) || uri.authority().is_none() {
            return Err(protocol_error(
                "MiniMax speech response returned an invalid URL",
            ));
        }
        Ok(Self(value))
    }
}

impl fmt::Debug for MinimaxSpeechDownloadUrl {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("MinimaxSpeechDownloadUrl([REDACTED])")
    }
}

/// Audio returned by buffered synchronous synthesis.
#[derive(Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MinimaxSpeechOutput {
    Audio(Vec<u8>),
    Url(MinimaxSpeechDownloadUrl),
}

impl MinimaxSpeechOutput {
    pub fn audio(&self) -> Option<&[u8]> {
        match self {
            Self::Audio(audio) => Some(audio),
            Self::Url(_) => None,
        }
    }

    pub fn url(&self) -> Option<&MinimaxSpeechDownloadUrl> {
        match self {
            Self::Audio(_) => None,
            Self::Url(url) => Some(url),
        }
    }
}

impl fmt::Debug for MinimaxSpeechOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Audio(audio) => formatter
                .debug_struct("Audio")
                .field("bytes", &audio.len())
                .finish(),
            Self::Url(url) => formatter.debug_tuple("Url").field(url).finish(),
        }
    }
}

/// Provider-reported information for generated audio.
#[derive(Clone, PartialEq, Deserialize)]
pub struct MinimaxSpeechInfo {
    #[serde(default)]
    audio_length: Option<u64>,
    #[serde(default)]
    audio_sample_rate: Option<u64>,
    #[serde(default)]
    audio_size: Option<u64>,
    #[serde(default)]
    bitrate: Option<u64>,
    #[serde(default)]
    word_count: Option<u64>,
    #[serde(default)]
    invisible_character_ratio: Option<f64>,
    #[serde(default)]
    usage_characters: Option<u64>,
    #[serde(default)]
    audio_format: Option<String>,
    #[serde(default)]
    audio_channel: Option<u8>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl MinimaxSpeechInfo {
    pub const fn duration_millis(&self) -> Option<u64> {
        self.audio_length
    }

    pub const fn sample_rate_hertz(&self) -> Option<u64> {
        self.audio_sample_rate
    }

    pub const fn audio_size_bytes(&self) -> Option<u64> {
        self.audio_size
    }

    pub const fn bitrate_bits_per_second(&self) -> Option<u64> {
        self.bitrate
    }

    pub const fn word_count(&self) -> Option<u64> {
        self.word_count
    }

    pub const fn invisible_character_ratio(&self) -> Option<f64> {
        self.invisible_character_ratio
    }

    pub const fn usage_characters(&self) -> Option<u64> {
        self.usage_characters
    }

    pub fn audio_format(&self) -> Option<&str> {
        self.audio_format.as_deref()
    }

    pub const fn channel_count(&self) -> Option<u8> {
        self.audio_channel
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxSpeechInfo {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxSpeechInfo")
            .field("duration_millis", &self.audio_length)
            .field("sample_rate_hertz", &self.audio_sample_rate)
            .field("audio_size_bytes", &self.audio_size)
            .field("bitrate_bits_per_second", &self.bitrate)
            .field("word_count", &self.word_count)
            .field("invisible_character_ratio", &self.invisible_character_ratio)
            .field("usage_characters", &self.usage_characters)
            .field("audio_format_present", &self.audio_format.is_some())
            .field("channel_count", &self.audio_channel)
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Buffered synchronous speech response.
#[derive(Clone, PartialEq)]
pub struct MinimaxSpeechResponse {
    output: MinimaxSpeechOutput,
    subtitle_url: Option<MinimaxSpeechDownloadUrl>,
    trace_id: Option<String>,
    info: Option<MinimaxSpeechInfo>,
    extra: BTreeMap<String, Value>,
}

impl MinimaxSpeechResponse {
    pub const fn output(&self) -> &MinimaxSpeechOutput {
        &self.output
    }

    pub fn subtitle_url(&self) -> Option<&MinimaxSpeechDownloadUrl> {
        self.subtitle_url.as_ref()
    }

    pub fn trace_id(&self) -> Option<&str> {
        self.trace_id.as_deref()
    }

    pub const fn info(&self) -> Option<&MinimaxSpeechInfo> {
        self.info.as_ref()
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for MinimaxSpeechResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxSpeechResponse")
            .field("output", &self.output)
            .field("subtitle_url_present", &self.subtitle_url.is_some())
            .field("trace_id_present", &self.trace_id.is_some())
            .field("info", &self.info)
            .field("extra_field_count", &self.extra.len())
            .finish()
    }
}

/// Shared, lightweight handle for MiniMax's provider-native Speech API.
#[derive(Clone)]
pub struct MinimaxSpeech {
    runtime: Arc<NativeRuntime>,
}

impl MinimaxSpeech {
    pub(crate) fn new(runtime: Arc<NativeRuntime>) -> Self {
        Self { runtime }
    }

    /// Generate one complete, buffered speech response with `stream=false`.
    pub async fn synthesize(
        &self,
        request: MinimaxSpeechSynthesisRequest,
    ) -> Result<MinimaxSpeechResponse, Error> {
        self.synthesize_with_options(request, CallOptions::default())
            .await
    }

    pub async fn synthesize_with_options(
        &self,
        request: MinimaxSpeechSynthesisRequest,
        options: CallOptions,
    ) -> Result<MinimaxSpeechResponse, Error> {
        request.validate()?;
        let output_format = request.output_format;
        let body = request_body(&SynchronousWireRequest::from(&request), "synthesis")?;
        let response: SynchronousEnvelope = execute_json(
            &self.runtime,
            Method::POST,
            target(SYNTHESIS_TARGET)?,
            body,
            ReplaySafety::Never,
            options,
        )
        .await?;
        response.into_response(output_format)
    }

    /// Submit a long-form asynchronous speech task.
    pub async fn submit(
        &self,
        request: MinimaxAsyncSpeechRequest,
    ) -> Result<MinimaxSpeechSubmission, Error> {
        self.submit_with_options(request, CallOptions::default())
            .await
    }

    pub async fn submit_with_options(
        &self,
        request: MinimaxAsyncSpeechRequest,
        options: CallOptions,
    ) -> Result<MinimaxSpeechSubmission, Error> {
        request.validate()?;
        let body = request_body(&AsyncWireRequest::from(&request), "asynchronous synthesis")?;
        let response: AsyncSubmissionEnvelope = execute_json(
            &self.runtime,
            Method::POST,
            target(ASYNC_SUBMIT_TARGET)?,
            body,
            ReplaySafety::Never,
            options,
        )
        .await?;
        response.into_submission()
    }

    /// Query one asynchronous speech task.
    ///
    /// The official endpoint is rate-limited independently by MiniMax. This
    /// method performs exactly one semantically idempotent query and does not
    /// hide polling, sleeps, or provider-specific retry loops.
    pub async fn query(&self, task_id: MinimaxSpeechTaskId) -> Result<MinimaxSpeechTask, Error> {
        self.query_with_options(task_id, CallOptions::default())
            .await
    }

    pub async fn query_with_options(
        &self,
        task_id: MinimaxSpeechTaskId,
        options: CallOptions,
    ) -> Result<MinimaxSpeechTask, Error> {
        let response: AsyncQueryEnvelope = execute_json(
            &self.runtime,
            Method::GET,
            target(format!("{ASYNC_QUERY_TARGET}?task_id={task_id}"))?,
            RequestBody::Empty,
            ReplaySafety::SemanticallyIdempotent,
            options,
        )
        .await?;
        response.into_task(task_id)
    }
}

impl fmt::Debug for MinimaxSpeech {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxSpeech")
            .field("runtime", &"shared")
            .finish()
    }
}

#[derive(Serialize)]
struct SynchronousWireRequest<'a> {
    model: &'a ModelId,
    text: &'a str,
    stream: bool,
    voice_setting: VoiceWireSettings<'a>,
    audio_setting: SynchronousAudioWireSettings,
    output_format: MinimaxSpeechOutputFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    pronunciation_dict: Option<&'a MinimaxPronunciationDictionary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language_boost: Option<&'a MinimaxLanguageBoost>,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice_modify: Option<&'a MinimaxVoiceEffects>,
    subtitle_enable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    subtitle_type: Option<MinimaxSubtitleGranularity>,
}

impl<'a> From<&'a MinimaxSpeechSynthesisRequest> for SynchronousWireRequest<'a> {
    fn from(request: &'a MinimaxSpeechSynthesisRequest) -> Self {
        Self {
            model: &request.model,
            text: &request.text,
            stream: false,
            voice_setting: VoiceWireSettings::synchronous(request),
            audio_setting: SynchronousAudioWireSettings::from(&request.audio),
            output_format: request.output_format,
            pronunciation_dict: request.pronunciation.as_ref(),
            language_boost: request.language_boost.as_ref(),
            voice_modify: request.voice_effects.as_ref(),
            subtitle_enable: request.subtitles.is_some(),
            subtitle_type: request.subtitles,
        }
    }
}

#[derive(Serialize)]
struct AsyncWireRequest<'a> {
    model: &'a ModelId,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text_file_id: Option<MinimaxFileId>,
    voice_setting: VoiceWireSettings<'a>,
    audio_setting: AsyncAudioWireSettings,
    #[serde(skip_serializing_if = "Option::is_none")]
    pronunciation_dict: Option<&'a MinimaxPronunciationDictionary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language_boost: Option<&'a MinimaxLanguageBoost>,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice_modify: Option<&'a MinimaxVoiceEffects>,
}

impl<'a> From<&'a MinimaxAsyncSpeechRequest> for AsyncWireRequest<'a> {
    fn from(request: &'a MinimaxAsyncSpeechRequest) -> Self {
        let (text, text_file_id) = match &request.input.kind {
            AsyncSpeechInputKind::Text(text) => (Some(text.as_str()), None),
            AsyncSpeechInputKind::TextFile(file_id) => (None, Some(*file_id)),
        };
        Self {
            model: &request.model,
            text,
            text_file_id,
            voice_setting: VoiceWireSettings::asynchronous(request),
            audio_setting: AsyncAudioWireSettings::from(&request.audio),
            pronunciation_dict: request.pronunciation.as_ref(),
            language_boost: request.language_boost.as_ref(),
            voice_modify: request.voice_effects.as_ref(),
        }
    }
}

#[derive(Serialize)]
struct VoiceWireSettings<'a> {
    voice_id: &'a MinimaxVoiceId,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vol: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pitch: Option<i8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    emotion: Option<MinimaxSpeechEmotion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text_normalization: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    english_normalization: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    latex_read: Option<bool>,
}

impl<'a> VoiceWireSettings<'a> {
    fn synchronous(request: &'a MinimaxSpeechSynthesisRequest) -> Self {
        Self::new(
            &request.voice,
            request.text_normalization,
            None,
            request.latex_read,
        )
    }

    fn asynchronous(request: &'a MinimaxAsyncSpeechRequest) -> Self {
        Self::new(
            &request.voice,
            None,
            request.english_normalization,
            request.latex_read,
        )
    }

    fn new(
        voice: &'a MinimaxVoiceSettings,
        text_normalization: Option<bool>,
        english_normalization: Option<bool>,
        latex_read: Option<bool>,
    ) -> Self {
        Self {
            voice_id: &voice.voice_id,
            speed: voice.speed.map(MinimaxSpeechSpeed::get),
            vol: voice.volume.map(MinimaxSpeechVolume::get),
            pitch: voice.pitch.map(MinimaxSpeechPitch::get),
            emotion: voice.emotion,
            text_normalization,
            english_normalization,
            latex_read,
        }
    }
}

#[derive(Serialize)]
struct SynchronousAudioWireSettings {
    format: MinimaxSpeechAudioFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_rate: Option<MinimaxSpeechSampleRate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bitrate: Option<MinimaxSpeechBitrate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    channel: Option<MinimaxSpeechChannels>,
}

impl From<&MinimaxSpeechAudioSettings> for SynchronousAudioWireSettings {
    fn from(audio: &MinimaxSpeechAudioSettings) -> Self {
        Self {
            format: audio.format,
            sample_rate: audio.sample_rate,
            bitrate: audio.bitrate,
            channel: audio.channels,
        }
    }
}

#[derive(Serialize)]
struct AsyncAudioWireSettings {
    format: MinimaxSpeechAudioFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_sample_rate: Option<MinimaxSpeechSampleRate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bitrate: Option<MinimaxSpeechBitrate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    channel: Option<MinimaxSpeechChannels>,
}

impl From<&MinimaxSpeechAudioSettings> for AsyncAudioWireSettings {
    fn from(audio: &MinimaxSpeechAudioSettings) -> Self {
        Self {
            format: audio.format,
            audio_sample_rate: audio.sample_rate,
            bitrate: audio.bitrate,
            channel: audio.channels,
        }
    }
}

#[derive(Deserialize)]
struct SynchronousEnvelope {
    #[serde(default)]
    data: Option<SynchronousData>,
    #[serde(default)]
    trace_id: Option<String>,
    #[serde(default)]
    extra_info: Option<MinimaxSpeechInfo>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl SynchronousEnvelope {
    fn into_response(
        self,
        output_format: MinimaxSpeechOutputFormat,
    ) -> Result<MinimaxSpeechResponse, Error> {
        let data = self.data.ok_or_else(|| {
            protocol_error("MiniMax synchronous speech response omitted generated audio")
        })?;
        if data.status.is_some_and(|status| status != 2) {
            return Err(protocol_error(
                "MiniMax non-streaming speech response was not complete",
            ));
        }
        let audio = data.audio.ok_or_else(|| {
            protocol_error("MiniMax synchronous speech response omitted generated audio")
        })?;
        let output = match output_format {
            MinimaxSpeechOutputFormat::Hex => MinimaxSpeechOutput::Audio(decode_hex_audio(&audio)?),
            MinimaxSpeechOutputFormat::Url => {
                MinimaxSpeechOutput::Url(MinimaxSpeechDownloadUrl::from_provider(audio)?)
            }
        };
        let subtitle_url = data
            .subtitle_file
            .map(MinimaxSpeechDownloadUrl::from_provider)
            .transpose()?;
        Ok(MinimaxSpeechResponse {
            output,
            subtitle_url,
            trace_id: self.trace_id,
            info: self.extra_info,
            extra: self.extra,
        })
    }
}

impl NativeResponseEnvelope for SynchronousEnvelope {
    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

#[derive(Deserialize)]
struct SynchronousData {
    #[serde(default)]
    audio: Option<String>,
    #[serde(default)]
    subtitle_file: Option<String>,
    #[serde(default)]
    status: Option<u8>,
}

#[derive(Deserialize)]
struct AsyncSubmissionEnvelope {
    #[serde(default)]
    task_id: Option<MinimaxSpeechTaskId>,
    #[serde(default)]
    file_id: Option<MinimaxFileId>,
    #[serde(default)]
    task_token: Option<String>,
    #[serde(default)]
    usage_characters: Option<u64>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl AsyncSubmissionEnvelope {
    fn into_submission(self) -> Result<MinimaxSpeechSubmission, Error> {
        let task_id = self.task_id.ok_or_else(|| {
            protocol_error("MiniMax asynchronous speech response omitted task_id")
        })?;
        let task_token = self
            .task_token
            .map(|value| {
                if value.is_empty() || value.len() > 16_384 || value.chars().any(char::is_control) {
                    Err(protocol_error(
                        "MiniMax asynchronous speech response returned an invalid task token",
                    ))
                } else {
                    Ok(MinimaxSpeechTaskToken(value))
                }
            })
            .transpose()?;
        Ok(MinimaxSpeechSubmission {
            task_id,
            file_id: self.file_id,
            task_token,
            usage_characters: self.usage_characters,
            extra: self.extra,
        })
    }
}

impl NativeResponseEnvelope for AsyncSubmissionEnvelope {
    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

#[derive(Deserialize)]
struct AsyncQueryEnvelope {
    #[serde(default)]
    task_id: Option<MinimaxSpeechTaskId>,
    #[serde(default)]
    status: Option<MinimaxSpeechTaskStatus>,
    #[serde(default)]
    file_id: Option<MinimaxFileId>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl AsyncQueryEnvelope {
    fn into_task(self, expected: MinimaxSpeechTaskId) -> Result<MinimaxSpeechTask, Error> {
        let task_id = self
            .task_id
            .ok_or_else(|| protocol_error("MiniMax speech query response omitted task_id"))?;
        if task_id != expected {
            return Err(Error::new(
                ErrorKind::ProtocolViolation,
                "MiniMax speech query response returned a different task identifier",
            ));
        }
        let status = self
            .status
            .ok_or_else(|| protocol_error("MiniMax speech query response omitted status"))?;
        if status.is_success() && self.file_id.is_none() {
            return Err(protocol_error(
                "MiniMax completed speech task omitted its output file identifier",
            ));
        }
        Ok(MinimaxSpeechTask {
            task_id,
            status,
            file_id: self.file_id,
            extra: self.extra,
        })
    }
}

impl NativeResponseEnvelope for AsyncQueryEnvelope {
    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpeechOperation {
    Synchronous,
    Asynchronous,
}

fn validated_model(model: impl Into<String>) -> Result<ModelId, Error> {
    ModelId::new(model.into()).map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "MiniMax speech model identifier is invalid",
        )
        .with_source(source)
    })
}

fn validate_text(text: &str, limit: usize, _operation: &'static str) -> Result<(), Error> {
    let count = text.chars().count();
    if count == 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "MiniMax speech input text must not be empty",
        ));
    }
    if count > limit {
        return Err(Error::new(
            ErrorKind::LimitExceeded,
            "MiniMax speech text exceeds the provider character limit",
        ));
    }
    Ok(())
}

fn validate_voice_for_model(model: &ModelId, voice: &MinimaxVoiceSettings) -> Result<(), Error> {
    let restricted_emotion = matches!(
        voice.emotion,
        Some(MinimaxSpeechEmotion::Fluent | MinimaxSpeechEmotion::Whisper)
    );
    if restricted_emotion && is_known_legacy_speech_model(model.as_str()) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "MiniMax fluent and whisper emotion overrides are not supported by legacy speech-01 or speech-02 models",
        ));
    }
    Ok(())
}

fn is_known_legacy_speech_model(model: &str) -> bool {
    matches!(
        model,
        SPEECH_02_HD | SPEECH_02_TURBO | SPEECH_01_HD | SPEECH_01_TURBO
    )
}

fn validate_latex_language(
    latex_read: Option<bool>,
    language_boost: Option<&MinimaxLanguageBoost>,
) -> Result<(), Error> {
    if latex_read == Some(true)
        && language_boost.is_some_and(|language| language.as_str() != "Chinese")
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "MiniMax LaTeX reading requires Chinese language boost when one is explicit",
        ));
    }
    Ok(())
}

fn validate_audio(
    audio: &MinimaxSpeechAudioSettings,
    operation: SpeechOperation,
    effects: Option<&MinimaxVoiceEffects>,
) -> Result<(), Error> {
    if audio.bitrate.is_some() && audio.format != MinimaxSpeechAudioFormat::Mp3 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "MiniMax speech bitrate is only valid for MP3 output",
        ));
    }
    if effects.is_some()
        && !matches!(
            audio.format,
            MinimaxSpeechAudioFormat::Mp3
                | MinimaxSpeechAudioFormat::Wav
                | MinimaxSpeechAudioFormat::Flac
        )
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "MiniMax voice effects require MP3, WAV, or FLAC output",
        ));
    }
    let Some(sample_rate) = audio.sample_rate else {
        return Ok(());
    };
    let supported = match (operation, audio.format) {
        (SpeechOperation::Asynchronous, MinimaxSpeechAudioFormat::Opus) => matches!(
            sample_rate,
            MinimaxSpeechSampleRate::Hz8000
                | MinimaxSpeechSampleRate::Hz12000
                | MinimaxSpeechSampleRate::Hz16000
                | MinimaxSpeechSampleRate::Hz24000
                | MinimaxSpeechSampleRate::Hz48000
        ),
        (SpeechOperation::Synchronous, MinimaxSpeechAudioFormat::Opus) => matches!(
            sample_rate,
            MinimaxSpeechSampleRate::Hz8000
                | MinimaxSpeechSampleRate::Hz16000
                | MinimaxSpeechSampleRate::Hz24000
        ),
        (_, MinimaxSpeechAudioFormat::PcmuRaw | MinimaxSpeechAudioFormat::PcmuWav) => {
            sample_rate == MinimaxSpeechSampleRate::Hz8000
        }
        _ => matches!(
            sample_rate,
            MinimaxSpeechSampleRate::Hz8000
                | MinimaxSpeechSampleRate::Hz16000
                | MinimaxSpeechSampleRate::Hz22050
                | MinimaxSpeechSampleRate::Hz24000
                | MinimaxSpeechSampleRate::Hz32000
                | MinimaxSpeechSampleRate::Hz44100
        ),
    };
    if !supported {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "MiniMax speech sample rate is incompatible with the selected operation and format",
        ));
    }
    Ok(())
}

fn request_body<T>(value: &T, _operation: &'static str) -> Result<RequestBody, Error>
where
    T: Serialize,
{
    RequestBody::json(value).map_err(|source| {
        Error::new(
            ErrorKind::InvalidInput,
            "MiniMax speech request could not be encoded",
        )
        .with_source(source)
    })
}

fn protocol_error(message: &'static str) -> Error {
    Error::new(ErrorKind::Protocol, message)
}

fn decode_hex_audio(value: &str) -> Result<Vec<u8>, Error> {
    if value.is_empty() || !value.len().is_multiple_of(2) {
        return Err(protocol_error(
            "MiniMax speech response contained invalid hexadecimal audio",
        ));
    }
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let high = decode_hex_digit(pair[0]).ok_or_else(|| {
            protocol_error("MiniMax speech response contained invalid hexadecimal audio")
        })?;
        let low = decode_hex_digit(pair[1]).ok_or_else(|| {
            protocol_error("MiniMax speech response contained invalid hexadecimal audio")
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

#[cfg(test)]
mod tests {
    use super::*;

    fn voice() -> MinimaxVoiceSettings {
        MinimaxVoiceSettings::new(
            MinimaxVoiceId::new("English_Insightful_Speaker").expect("valid voice"),
        )
        .with_speed(MinimaxSpeechSpeed::new(1.25).expect("valid speed"))
        .with_volume(MinimaxSpeechVolume::new(2.0).expect("valid volume"))
        .with_pitch(MinimaxSpeechPitch::new(-2).expect("valid pitch"))
    }

    #[test]
    fn synchronous_request_is_explicitly_non_streaming_and_has_no_emotion_default() {
        let request =
            MinimaxSpeechSynthesisRequest::new(SPEECH_2_8_HD, "private narration", voice())
                .expect("valid request")
                .with_audio(
                    MinimaxSpeechAudioSettings::new(MinimaxSpeechAudioFormat::Mp3)
                        .with_sample_rate(MinimaxSpeechSampleRate::Hz32000)
                        .with_bitrate(MinimaxSpeechBitrate::Kbps128)
                        .with_channels(MinimaxSpeechChannels::Mono),
                )
                .with_output_format(MinimaxSpeechOutputFormat::Url)
                .with_language_boost(MinimaxLanguageBoost::auto())
                .with_text_normalization(true)
                .with_subtitles(MinimaxSubtitleGranularity::Word);

        request.validate().expect("request should validate");
        let value = serde_json::to_value(SynchronousWireRequest::from(&request))
            .expect("request should encode");
        assert_eq!(value["model"], SPEECH_2_8_HD);
        assert_eq!(value["stream"], false);
        assert_eq!(value["output_format"], "url");
        assert_eq!(value["voice_setting"]["speed"], 1.25);
        assert_eq!(value["voice_setting"]["vol"], 2.0);
        assert_eq!(value["voice_setting"]["pitch"], -2);
        assert_eq!(value["voice_setting"]["text_normalization"], true);
        assert!(value["voice_setting"].get("emotion").is_none());
        assert_eq!(value["audio_setting"]["sample_rate"], 32_000);
        assert_eq!(value["subtitle_enable"], true);
        assert!(!format!("{request:?}").contains("private narration"));
    }

    #[test]
    fn async_input_encodes_exactly_one_source_and_uses_async_sample_rate_key() {
        let text = MinimaxAsyncSpeechRequest::new(
            SPEECH_2_8_TURBO,
            MinimaxAsyncSpeechInput::text("private long form").expect("valid text"),
            voice(),
        )
        .expect("valid request")
        .with_audio(
            MinimaxSpeechAudioSettings::new(MinimaxSpeechAudioFormat::Opus)
                .with_sample_rate(MinimaxSpeechSampleRate::Hz48000)
                .with_channels(MinimaxSpeechChannels::Stereo),
        )
        .with_english_normalization(true);
        text.validate().expect("async opus should validate");
        let value =
            serde_json::to_value(AsyncWireRequest::from(&text)).expect("request should encode");
        assert_eq!(value["text"], "private long form");
        assert!(value.get("text_file_id").is_none());
        assert_eq!(value["audio_setting"]["audio_sample_rate"], 48_000);
        assert_eq!(value["voice_setting"]["english_normalization"], true);

        let file = MinimaxAsyncSpeechRequest::new(
            SPEECH_2_8_TURBO,
            MinimaxAsyncSpeechInput::text_file(MinimaxFileId::new(42).expect("valid file")),
            voice(),
        )
        .expect("valid request");
        let value =
            serde_json::to_value(AsyncWireRequest::from(&file)).expect("request should encode");
        assert_eq!(value["text_file_id"], 42);
        assert!(value.get("text").is_none());
    }

    #[test]
    fn emotion_support_tracks_legacy_negatives_without_closing_future_models() {
        let whisper =
            MinimaxVoiceSettings::new(MinimaxVoiceId::new("known-voice").expect("valid voice"))
                .with_emotion(MinimaxSpeechEmotion::Whisper);
        let current = MinimaxSpeechSynthesisRequest::new(SPEECH_2_8_HD, "hello", whisper.clone())
            .expect("request construction should succeed");
        current
            .validate()
            .expect("speech-2.8 supports the whisper emotion override");

        let legacy = MinimaxSpeechSynthesisRequest::new(SPEECH_02_HD, "hello", whisper.clone())
            .expect("request construction should succeed");
        assert_eq!(
            legacy
                .validate()
                .expect_err("legacy model restriction must fail")
                .kind(),
            ErrorKind::InvalidInput
        );

        let future = MinimaxSpeechSynthesisRequest::new("speech-future", "hello", whisper)
            .expect("future model IDs stay open");
        future
            .validate()
            .expect("unknown future model must not inherit guessed restrictions");
    }

    #[test]
    fn audio_combinations_are_validated_per_operation() {
        let sync_opus = MinimaxSpeechSynthesisRequest::new(SPEECH_2_8_HD, "hello", voice())
            .expect("valid request")
            .with_audio(
                MinimaxSpeechAudioSettings::new(MinimaxSpeechAudioFormat::Opus)
                    .with_sample_rate(MinimaxSpeechSampleRate::Hz48000),
            );
        assert_eq!(
            sync_opus
                .validate()
                .expect_err("sync documentation does not support 48 kHz")
                .kind(),
            ErrorKind::InvalidInput
        );

        let invalid_bitrate = MinimaxAsyncSpeechRequest::new(
            crate::models::speech::SPEECH_2_6_HD,
            MinimaxAsyncSpeechInput::text("hello").expect("valid text"),
            voice(),
        )
        .expect("valid request")
        .with_audio(
            MinimaxSpeechAudioSettings::new(MinimaxSpeechAudioFormat::Wav)
                .with_bitrate(MinimaxSpeechBitrate::Kbps128),
        );
        assert_eq!(
            invalid_bitrate
                .validate()
                .expect_err("bitrate only applies to MP3")
                .kind(),
            ErrorKind::InvalidInput
        );
    }

    #[test]
    fn synchronous_response_decodes_audio_and_redacts_provider_urls() {
        let hex: SynchronousEnvelope = serde_json::from_value(serde_json::json!({
            "data": {"audio": "00a1FF", "status": 2},
            "extra_info": {"audio_size": 3, "future": true},
            "trace_id": "trace",
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }))
        .expect("response should decode");
        let response = hex
            .into_response(MinimaxSpeechOutputFormat::Hex)
            .expect("hex should decode");
        assert_eq!(response.output().audio(), Some(&[0x00, 0xa1, 0xff][..]));
        assert_eq!(
            response
                .info()
                .and_then(MinimaxSpeechInfo::audio_size_bytes),
            Some(3)
        );

        let url: SynchronousEnvelope = serde_json::from_value(serde_json::json!({
            "data": {
                "audio": "https://example.invalid/audio?token=secret",
                "subtitle_file": "https://example.invalid/subtitles?token=secret",
                "status": 2
            },
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }))
        .expect("response should decode");
        let response = url
            .into_response(MinimaxSpeechOutputFormat::Url)
            .expect("URL should decode");
        let debug = format!("{response:?}");
        assert!(!debug.contains("token=secret"));
        assert!(response.output().url().is_some());
        assert!(response.subtitle_url().is_some());
    }

    #[test]
    fn async_identifiers_accept_documented_schema_and_examples() {
        let numeric: MinimaxSpeechTaskId =
            serde_json::from_value(Value::from(95_157_322_514_444_u64)).expect("numeric task id");
        let string: MinimaxSpeechTaskId =
            serde_json::from_value(Value::from("95157322514444")).expect("string task id");
        assert_eq!(numeric, string);

        let submission: AsyncSubmissionEnvelope = serde_json::from_value(serde_json::json!({
            "task_id": "95157322514444",
            "task_token": "private-task-token",
            "file_id": 95157322514496_u64,
            "usage_characters": 101,
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }))
        .expect("submission should decode");
        let submission = submission.into_submission().expect("valid submission");
        assert_eq!(submission.task_id(), numeric);
        assert_eq!(submission.usage_characters(), Some(101));
        assert!(!format!("{submission:?}").contains("private-task-token"));
    }

    #[test]
    fn query_status_is_case_tolerant_but_unknown_values_remain_visible() {
        let expected = MinimaxSpeechTaskId::new(42).expect("valid task");
        let completed: AsyncQueryEnvelope = serde_json::from_value(serde_json::json!({
            "task_id": 42,
            "status": "Success",
            "file_id": 84,
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }))
        .expect("query should decode");
        let completed = completed.into_task(expected).expect("completed task");
        assert!(completed.status().is_terminal());
        assert!(completed.status().is_success());

        let unknown: MinimaxSpeechTaskStatus =
            serde_json::from_value(Value::from("Queued")).expect("unknown status should decode");
        assert_eq!(unknown, MinimaxSpeechTaskStatus::Other("Queued".to_owned()));
        assert!(!unknown.is_terminal());
    }

    #[test]
    fn text_and_opaque_values_are_bounded_without_leaking_contents() {
        let too_long = "x".repeat(SYNCHRONOUS_TEXT_LIMIT + 1);
        let error = MinimaxSpeechSynthesisRequest::new(SPEECH_2_8_HD, too_long, voice())
            .expect_err("oversized text must fail");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);

        assert_eq!(
            MinimaxSpeechVolume::new(0.0),
            Err(MinimaxSpeechValueError::InvalidVolume)
        );
        assert_eq!(
            MinimaxSpeechSpeed::new(f32::NAN),
            Err(MinimaxSpeechValueError::InvalidSpeed)
        );
        assert_eq!(
            MinimaxVoiceId::new(" voice"),
            Err(MinimaxSpeechValueError::InvalidVoiceId)
        );
    }
}
