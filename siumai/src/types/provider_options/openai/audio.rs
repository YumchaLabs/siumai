//! Audio-related types for OpenAI Chat API
//!
//! This module contains types for audio input/output in OpenAI's multimodal chat models.
//! Reference: https://platform.openai.com/docs/guides/audio

use serde::{Deserialize, Serialize};

/// Chat completion modalities
///
/// Specifies the types of content the model can generate.
/// To request audio output, use `["text", "audio"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionModalities {
    /// Text output
    Text,
    /// Audio output
    Audio,
}

/// Audio output configuration for chat completions
///
/// Parameters for audio output. Required when audio output is requested with `modalities: ["audio"]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatCompletionAudio {
    /// The voice the model uses to respond
    pub voice: ChatCompletionAudioVoice,
    /// Specifies the output audio format
    pub format: ChatCompletionAudioFormat,
}

impl ChatCompletionAudio {
    /// Create new audio configuration
    pub fn new(voice: ChatCompletionAudioVoice, format: ChatCompletionAudioFormat) -> Self {
        Self { voice, format }
    }

    /// Create audio config with default format (mp3)
    pub fn with_voice(voice: ChatCompletionAudioVoice) -> Self {
        Self {
            voice,
            format: ChatCompletionAudioFormat::Mp3,
        }
    }
}

/// Voice options for audio output
///
/// Supported voices for audio generation. Recommended voices are `ash`, `ballad`, `coral`, `sage`, and `verse`.
/// Voices `alloy`, `echo`, and `shimmer` are also supported but less expressive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionAudioVoice {
    /// Alloy voice (less expressive)
    Alloy,
    /// Ash voice (recommended)
    Ash,
    /// Ballad voice (recommended)
    Ballad,
    /// Coral voice (recommended)
    Coral,
    /// Echo voice (less expressive)
    Echo,
    /// Sage voice (recommended)
    Sage,
    /// Shimmer voice (less expressive)
    Shimmer,
    /// Verse voice (recommended)
    Verse,
}

/// Audio output format options
///
/// Supported audio formats for output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionAudioFormat {
    /// WAV format
    Wav,
    /// MP3 format (default)
    Mp3,
    /// FLAC format
    Flac,
    /// Opus format
    Opus,
    /// PCM16 format
    Pcm16,
}

/// Input audio format options
///
/// Supported formats for input audio in chat messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputAudioFormat {
    /// WAV format
    Wav,
    /// MP3 format (default)
    Mp3,
}

/// Input audio data
///
/// Audio data for input in chat messages.
/// The audio must be base64-encoded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputAudio {
    /// Base64 encoded audio data
    pub data: String,
    /// The format of the encoded audio data
    pub format: InputAudioFormat,
}

impl InputAudio {
    /// Create new input audio
    pub fn new(data: String, format: InputAudioFormat) -> Self {
        Self { data, format }
    }

    /// Create input audio with MP3 format
    pub fn mp3(data: String) -> Self {
        Self {
            data,
            format: InputAudioFormat::Mp3,
        }
    }

    /// Create input audio with WAV format
    pub fn wav(data: String) -> Self {
        Self {
            data,
            format: InputAudioFormat::Wav,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_config_creation() {
        let audio = ChatCompletionAudio::new(
            ChatCompletionAudioVoice::Ash,
            ChatCompletionAudioFormat::Mp3,
        );
        assert_eq!(audio.voice, ChatCompletionAudioVoice::Ash);
        assert_eq!(audio.format, ChatCompletionAudioFormat::Mp3);
    }

    #[test]
    fn test_audio_config_with_voice() {
        let audio = ChatCompletionAudio::with_voice(ChatCompletionAudioVoice::Ballad);
        assert_eq!(audio.voice, ChatCompletionAudioVoice::Ballad);
        assert_eq!(audio.format, ChatCompletionAudioFormat::Mp3);
    }

    #[test]
    fn test_input_audio_creation() {
        let audio = InputAudio::mp3("base64data".to_string());
        assert_eq!(audio.data, "base64data");
        assert_eq!(audio.format, InputAudioFormat::Mp3);
    }

    #[test]
    fn test_modalities_serialization() {
        let modalities = vec![
            ChatCompletionModalities::Text,
            ChatCompletionModalities::Audio,
        ];
        let json = serde_json::to_string(&modalities).unwrap();
        assert_eq!(json, r#"["text","audio"]"#);
    }
}
