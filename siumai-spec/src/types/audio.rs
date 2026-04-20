//! Audio processing types for TTS and STT

use std::collections::HashMap;

use crate::types::{
    HttpConfig, HttpResponseInfo, ProviderMetadataMap, ProviderOptionsMap, Warning,
};
use base64::Engine;

fn should_fill_model_slot(slot: Option<&str>, fallback: &str) -> bool {
    let fallback = fallback.trim();
    if fallback.is_empty() {
        return false;
    }

    match slot {
        Some(value) => value.trim().is_empty(),
        None => true,
    }
}

/// Text-to-speech request
#[derive(Debug, Clone)]
pub struct TtsRequest {
    /// Text to convert to speech
    pub text: String,
    /// Voice to use (provider-specific)
    pub voice: Option<String>,
    /// Audio format (mp3, wav, etc.)
    pub format: Option<String>,
    /// Speech speed (0.25 to 4.0)
    pub speed: Option<f32>,
    /// Audio quality/model
    pub model: Option<String>,
    /// Open provider options map (Vercel-aligned).
    pub provider_options_map: ProviderOptionsMap,
    /// Additional provider-specific parameters
    pub extra_params: HashMap<String, serde_json::Value>,
    /// Optional per-request HTTP configuration (headers, proxy, timeouts, etc.)
    pub http_config: Option<HttpConfig>,
}

impl TtsRequest {
    /// Create a new TTS request with text
    pub fn new(text: String) -> Self {
        Self {
            text,
            voice: None,
            format: None,
            speed: None,
            model: None,
            provider_options_map: ProviderOptionsMap::default(),
            extra_params: HashMap::new(),
            http_config: None,
        }
    }

    /// Set the voice
    pub fn with_voice(mut self, voice: String) -> Self {
        self.voice = Some(voice);
        self
    }

    /// Set the audio format
    pub fn with_format(mut self, format: String) -> Self {
        self.format = Some(format);
        self
    }

    /// Set the speech speed
    pub const fn with_speed(mut self, speed: f32) -> Self {
        self.speed = Some(speed);
        self
    }

    /// Set the model
    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the model only when the request does not already define one.
    pub fn with_model_if_missing(mut self, model: impl Into<String>) -> Self {
        let model = model.into();
        if should_fill_model_slot(self.model.as_deref(), &model) {
            self.model = Some(model);
        }
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

    /// Set per-request HTTP config (headers, proxy, timeouts, etc.)
    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = Some(http_config);
        self
    }
}

/// Text-to-speech response
#[derive(Debug, Clone)]
pub struct TtsResponse {
    /// Generated audio data
    pub audio_data: Vec<u8>,
    /// Audio format
    pub format: String,
    /// Duration in seconds
    pub duration: Option<f32>,
    /// Sample rate
    pub sample_rate: Option<u32>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Provider warnings (e.g. unsupported settings).
    pub warnings: Option<Vec<Warning>>,
    /// Provider-scoped metadata aligned with AI SDK response metadata roots.
    pub provider_metadata: Option<ProviderMetadataMap>,
    /// Best-effort HTTP response envelope (timestamp, model id, headers).
    pub response: Option<HttpResponseInfo>,
}

/// Audio input payload for transcription and audio translation requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioInputData {
    /// Base64-encoded audio data.
    Base64(String),
    /// Binary audio bytes.
    Binary(Vec<u8>),
}

impl AudioInputData {
    /// Create audio input from binary bytes.
    pub fn binary(data: impl Into<Vec<u8>>) -> Self {
        Self::Binary(data.into())
    }

    /// Create audio input from a base64 string.
    pub fn base64(data: impl Into<String>) -> Self {
        Self::Base64(data.into())
    }

    /// Convert the audio input to a base64 string.
    pub fn as_base64(&self) -> String {
        match self {
            Self::Base64(data) => data.clone(),
            Self::Binary(data) => base64::engine::general_purpose::STANDARD.encode(data),
        }
    }

    /// Convert the audio input to bytes, decoding base64 when necessary.
    pub fn as_bytes(&self) -> Result<Vec<u8>, base64::DecodeError> {
        match self {
            Self::Base64(data) => base64::engine::general_purpose::STANDARD.decode(data),
            Self::Binary(data) => Ok(data.clone()),
        }
    }
}

/// Speech-to-text request
#[derive(Debug, Clone)]
pub struct SttRequest {
    /// Canonical audio input aligned with AI SDK transcription call options.
    pub audio: AudioInputData,
    /// Audio format
    pub format: Option<String>,
    /// Required IANA media type for the audio input (e.g., "audio/wav").
    pub media_type: String,
    /// Language code (e.g., "en-US")
    pub language: Option<String>,
    /// Model to use
    pub model: Option<String>,
    /// Enable word-level timestamps
    pub timestamp_granularities: Option<Vec<String>>,
    /// Open provider options map (Vercel-aligned).
    pub provider_options_map: ProviderOptionsMap,
    /// Additional provider-specific parameters
    pub extra_params: HashMap<String, serde_json::Value>,
    /// Optional per-request HTTP configuration (headers, proxy, timeouts, etc.)
    pub http_config: Option<HttpConfig>,
}

impl SttRequest {
    /// Create STT request from audio data
    pub fn from_audio(audio_data: impl Into<Vec<u8>>, media_type: impl Into<String>) -> Self {
        Self {
            audio: AudioInputData::binary(audio_data),
            format: None,
            media_type: media_type.into(),
            language: None,
            model: None,
            timestamp_granularities: None,
            provider_options_map: ProviderOptionsMap::default(),
            extra_params: HashMap::new(),
            http_config: None,
        }
    }

    /// Create STT request from a base64 audio payload.
    pub fn from_base64(audio_data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            audio: AudioInputData::base64(audio_data),
            format: None,
            media_type: media_type.into(),
            language: None,
            model: None,
            timestamp_granularities: None,
            provider_options_map: ProviderOptionsMap::default(),
            extra_params: HashMap::new(),
            http_config: None,
        }
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

    /// Set per-request HTTP config (headers, proxy, timeouts, etc.)
    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = Some(http_config);
        self
    }

    /// Set the model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the model only when the request does not already define one.
    pub fn with_model_if_missing(mut self, model: impl Into<String>) -> Self {
        let model = model.into();
        if should_fill_model_slot(self.model.as_deref(), &model) {
            self.model = Some(model);
        }
        self
    }

    /// Set audio media type (e.g., "audio/mpeg", "audio/wav")
    pub fn with_media_type(mut self, media_type: impl Into<String>) -> Self {
        self.media_type = media_type.into();
        self
    }

    /// Replace the canonical audio input payload.
    pub fn with_audio(mut self, audio: AudioInputData) -> Self {
        self.audio = audio;
        self
    }

    /// Return the audio input as bytes, decoding base64 when necessary.
    pub fn audio_bytes(&self) -> Result<Vec<u8>, base64::DecodeError> {
        self.audio.as_bytes()
    }
}

/// Speech-to-text response
#[derive(Debug, Clone)]
pub struct SttResponse {
    /// Transcribed text
    pub text: String,
    /// Language detected
    pub language: Option<String>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: Option<f32>,
    /// Word-level timestamps
    pub words: Option<Vec<WordTimestamp>>,
    /// Duration of audio in seconds
    pub duration: Option<f32>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Provider warnings (e.g. unsupported settings).
    pub warnings: Option<Vec<Warning>>,
    /// Provider-scoped metadata aligned with AI SDK response metadata roots.
    pub provider_metadata: Option<ProviderMetadataMap>,
    /// Best-effort HTTP response envelope (timestamp, model id, headers).
    pub response: Option<HttpResponseInfo>,
}

/// Word-level timestamp information
#[derive(Debug, Clone)]
pub struct WordTimestamp {
    /// The word
    pub word: String,
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Confidence score for this word
    pub confidence: Option<f32>,
}

/// Audio translation request (speech to English text)
#[derive(Debug, Clone)]
pub struct AudioTranslationRequest {
    /// Canonical audio input aligned with AI SDK transcription call options.
    pub audio: AudioInputData,
    /// Audio format
    pub format: Option<String>,
    /// Required IANA media type for the audio input (e.g., "audio/wav").
    pub media_type: String,
    /// Model to use
    pub model: Option<String>,
    /// Open provider options map (Vercel-aligned).
    pub provider_options_map: ProviderOptionsMap,
    /// Additional provider-specific parameters
    pub extra_params: HashMap<String, serde_json::Value>,
    /// Optional per-request HTTP configuration (headers, proxy, timeouts, etc.)
    pub http_config: Option<HttpConfig>,
}

impl AudioTranslationRequest {
    /// Create translation request from audio data
    pub fn from_audio(audio_data: impl Into<Vec<u8>>, media_type: impl Into<String>) -> Self {
        Self {
            audio: AudioInputData::binary(audio_data),
            format: None,
            media_type: media_type.into(),
            model: None,
            provider_options_map: ProviderOptionsMap::default(),
            extra_params: HashMap::new(),
            http_config: None,
        }
    }

    /// Create translation request from a base64 audio payload.
    pub fn from_base64(audio_data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            audio: AudioInputData::base64(audio_data),
            format: None,
            media_type: media_type.into(),
            model: None,
            provider_options_map: ProviderOptionsMap::default(),
            extra_params: HashMap::new(),
            http_config: None,
        }
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

    /// Set per-request HTTP config (headers, proxy, timeouts, etc.)
    pub fn with_http_config(mut self, http_config: HttpConfig) -> Self {
        self.http_config = Some(http_config);
        self
    }

    /// Set the model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the model only when the request does not already define one.
    pub fn with_model_if_missing(mut self, model: impl Into<String>) -> Self {
        let model = model.into();
        if should_fill_model_slot(self.model.as_deref(), &model) {
            self.model = Some(model);
        }
        self
    }

    /// Set audio media type (e.g., "audio/mpeg", "audio/wav")
    pub fn with_media_type(mut self, media_type: impl Into<String>) -> Self {
        self.media_type = media_type.into();
        self
    }

    /// Replace the canonical audio input payload.
    pub fn with_audio(mut self, audio: AudioInputData) -> Self {
        self.audio = audio;
        self
    }

    /// Return the audio input as bytes, decoding base64 when necessary.
    pub fn audio_bytes(&self) -> Result<Vec<u8>, base64::DecodeError> {
        self.audio.as_bytes()
    }
}

/// Voice information
#[derive(Debug, Clone)]
pub struct VoiceInfo {
    /// Voice ID/name
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Voice description
    pub description: Option<String>,
    /// Language code
    pub language: Option<String>,
    /// Gender (male, female, neutral)
    pub gender: Option<String>,
    /// Voice category (standard, premium, neural, etc.)
    pub category: Option<String>,
}

/// Language information
#[derive(Debug, Clone)]
pub struct LanguageInfo {
    /// Language code (e.g., "en-US")
    pub code: String,
    /// Human-readable name
    pub name: String,
    /// Whether this language supports transcription
    pub supports_transcription: bool,
    /// Whether this language supports translation
    pub supports_translation: bool,
}

/// Audio features that providers can support
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AudioFeature {
    /// Basic text-to-speech conversion
    TextToSpeech,
    /// Streaming text-to-speech conversion
    StreamingTTS,
    /// Basic speech-to-text conversion
    SpeechToText,
    /// Audio translation (speech to English text)
    AudioTranslation,
    /// Real-time audio processing
    RealtimeProcessing,
    /// Speaker diarization (identifying different speakers)
    SpeakerDiarization,
    /// Character-level timing information
    CharacterTiming,
    /// Audio event detection (laughter, applause, etc.)
    AudioEventDetection,
    /// Voice cloning capabilities
    VoiceCloning,
    /// Audio enhancement and noise reduction
    AudioEnhancement,
    /// Multi-modal audio-visual processing
    MultimodalAudio,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_input_data_helpers_support_binary_and_base64() {
        let binary = AudioInputData::binary(vec![1, 2, 3]);
        assert_eq!(binary.as_bytes().expect("bytes"), vec![1, 2, 3]);

        let base64 = AudioInputData::base64("AQID");
        assert_eq!(base64.as_bytes().expect("decode"), vec![1, 2, 3]);
        assert_eq!(base64.as_base64(), "AQID");
    }

    #[test]
    fn stt_request_uses_canonical_audio_input() {
        let request = SttRequest::from_base64("AQID", "audio/mpeg");

        assert_eq!(request.audio_bytes().expect("audio bytes"), vec![1, 2, 3]);
        assert_eq!(request.media_type, "audio/mpeg");
    }

    #[test]
    fn audio_translation_request_uses_canonical_audio_input() {
        let request = AudioTranslationRequest::from_audio(vec![4, 5, 6], "audio/wav");

        assert_eq!(request.audio_bytes().expect("audio bytes"), vec![4, 5, 6]);
        assert_eq!(request.media_type, "audio/wav");
    }
}
