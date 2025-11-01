//! Audio processing types for TTS and STT

use std::collections::HashMap;

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
    /// Additional provider-specific parameters
    pub extra_params: HashMap<String, serde_json::Value>,
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
            extra_params: HashMap::new(),
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
}

/// Speech-to-text request
#[derive(Debug, Clone)]
pub struct SttRequest {
    /// Audio data
    pub audio_data: Option<Vec<u8>>,
    /// File path (alternative to `audio_data`)
    pub file_path: Option<String>,
    /// Audio format
    pub format: Option<String>,
    /// Language code (e.g., "en-US")
    pub language: Option<String>,
    /// Model to use
    pub model: Option<String>,
    /// Enable word-level timestamps
    pub timestamp_granularities: Option<Vec<String>>,
    /// Additional provider-specific parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl SttRequest {
    /// Create STT request from audio data
    pub fn from_audio(audio_data: Vec<u8>) -> Self {
        Self {
            audio_data: Some(audio_data),
            file_path: None,
            format: None,
            language: None,
            model: None,
            timestamp_granularities: None,
            extra_params: HashMap::new(),
        }
    }

    /// Create STT request from file path
    pub fn from_file(file_path: String) -> Self {
        Self {
            audio_data: None,
            file_path: Some(file_path),
            format: None,
            language: None,
            model: None,
            timestamp_granularities: None,
            extra_params: HashMap::new(),
        }
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
    /// Audio data
    pub audio_data: Option<Vec<u8>>,
    /// File path (alternative to `audio_data`)
    pub file_path: Option<String>,
    /// Audio format
    pub format: Option<String>,
    /// Model to use
    pub model: Option<String>,
    /// Additional provider-specific parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl AudioTranslationRequest {
    /// Create translation request from audio data
    pub fn from_audio(audio_data: Vec<u8>) -> Self {
        Self {
            audio_data: Some(audio_data),
            file_path: None,
            format: None,
            model: None,
            extra_params: HashMap::new(),
        }
    }

    /// Create translation request from file path
    pub fn from_file(file_path: String) -> Self {
        Self {
            audio_data: None,
            file_path: Some(file_path),
            format: None,
            model: None,
            extra_params: HashMap::new(),
        }
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
