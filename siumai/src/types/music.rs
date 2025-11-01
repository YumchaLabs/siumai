//! Music generation types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Audio settings for music generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicAudioSetting {
    /// Sample rate (e.g., 44100)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,

    /// Bitrate (e.g., 256000)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bitrate: Option<u32>,

    /// Audio format (e.g., "mp3")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

impl Default for MusicAudioSetting {
    fn default() -> Self {
        Self {
            sample_rate: Some(44100),
            bitrate: Some(256000),
            format: Some("mp3".to_string()),
        }
    }
}

/// Music style/genre enumeration
///
/// Common music styles supported across providers. Providers may support additional
/// custom styles via the `extra_params` field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MusicStyle {
    /// Classical music
    Classical,
    /// Jazz music
    Jazz,
    /// Rock music
    Rock,
    /// Pop music
    Pop,
    /// Electronic/EDM
    Electronic,
    /// Hip-hop/Rap
    HipHop,
    /// Country music
    Country,
    /// Folk music
    Folk,
    /// Blues music
    Blues,
    /// Ambient/Atmospheric
    Ambient,
    /// Custom style (provider-specific)
    Custom(String),
}

/// Music generation request
///
/// This type is designed to be extensible across different music generation providers.
/// Required fields are minimal, with most features being optional to accommodate
/// different provider capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicGenerationRequest {
    /// Model name (e.g., "music-2.0", "suno-v3", "stable-audio-2.0")
    pub model: String,

    /// Music description (style, mood, scenario)
    ///
    /// This is the primary input for all providers. Describes the desired music characteristics.
    /// Example: "Indie folk, melancholic, introspective, acoustic guitar"
    pub prompt: String,

    /// Song lyrics with optional structure tags
    ///
    /// Optional field for providers that support lyrics-based generation.
    /// Some providers support structure tags like [Intro], [Verse], [Chorus], [Bridge], [Outro].
    /// Set to `None` for instrumental music.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lyrics: Option<String>,

    /// Desired music duration in seconds
    ///
    /// Not all providers support custom durations. Some may have fixed lengths.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<u32>,

    /// Music style/genre
    ///
    /// Optional style hint for providers that support predefined genres.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<MusicStyle>,

    /// Whether to generate instrumental music only (no vocals)
    ///
    /// Default behavior varies by provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instrumental: Option<bool>,

    /// Audio output settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_setting: Option<MusicAudioSetting>,

    /// Seed audio for continuation/extension
    ///
    /// For providers that support continuing from existing audio.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed_audio: Option<Vec<u8>>,

    /// Additional provider-specific parameters
    ///
    /// Use this for provider-specific features not covered by standard fields.
    /// Examples:
    /// - MiniMaxi: `{"aigc_watermark": true}`
    /// - Suno: `{"make_instrumental": false, "wait_audio": true}`
    /// - Stable Audio: `{"cfg_scale": 7.0, "steps": 100}`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_params: Option<HashMap<String, serde_json::Value>>,
}

impl MusicGenerationRequest {
    /// Create a new music generation request with minimal required fields
    ///
    /// # Arguments
    ///
    /// * `model` - Model name to use for generation
    /// * `prompt` - Text description of the desired music
    ///
    /// # Example
    ///
    /// ```ignore
    /// let request = MusicGenerationRequest::new("music-2.0", "Upbeat jazz with saxophone");
    /// ```
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            lyrics: None,
            duration: None,
            style: None,
            instrumental: None,
            audio_setting: Some(MusicAudioSetting::default()),
            seed_audio: None,
            extra_params: None,
        }
    }

    /// Set lyrics for the music
    ///
    /// Use this for providers that support lyrics-based generation.
    pub fn with_lyrics(mut self, lyrics: impl Into<String>) -> Self {
        self.lyrics = Some(lyrics.into());
        self
    }

    /// Set music duration in seconds
    pub fn with_duration(mut self, duration: u32) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set music style/genre
    pub fn with_style(mut self, style: MusicStyle) -> Self {
        self.style = Some(style);
        self
    }

    /// Set whether to generate instrumental music only
    pub fn with_instrumental(mut self, instrumental: bool) -> Self {
        self.instrumental = Some(instrumental);
        self
    }

    /// Set seed audio for continuation
    pub fn with_seed_audio(mut self, audio: Vec<u8>) -> Self {
        self.seed_audio = Some(audio);
        self
    }

    /// Add a provider-specific parameter
    pub fn with_extra_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value);
        self
    }

    /// Set audio settings
    pub fn with_audio_setting(mut self, setting: MusicAudioSetting) -> Self {
        self.audio_setting = Some(setting);
        self
    }

    /// Set sample rate
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.audio_setting
            .get_or_insert_with(MusicAudioSetting::default)
            .sample_rate = Some(sample_rate);
        self
    }

    /// Set bitrate
    pub fn with_bitrate(mut self, bitrate: u32) -> Self {
        self.audio_setting
            .get_or_insert_with(MusicAudioSetting::default)
            .bitrate = Some(bitrate);
        self
    }

    /// Set audio format
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.audio_setting
            .get_or_insert_with(MusicAudioSetting::default)
            .format = Some(format.into());
        self
    }
}

/// Music metadata from response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicMetadata {
    /// Music duration in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub music_duration: Option<u32>,

    /// Sample rate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub music_sample_rate: Option<u32>,

    /// Number of audio channels
    #[serde(skip_serializing_if = "Option::is_none")]
    pub music_channel: Option<u32>,

    /// Bitrate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bitrate: Option<u32>,

    /// Music file size in bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub music_size: Option<u32>,
}

/// Music generation response
#[derive(Debug, Clone)]
pub struct MusicGenerationResponse {
    /// Generated music audio data
    pub audio_data: Vec<u8>,

    /// Music metadata
    pub metadata: MusicMetadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_music_request_builder() {
        let request = MusicGenerationRequest::new("music-2.0", "Indie folk, melancholic")
            .with_lyrics("[verse]\nTest lyrics")
            .with_duration(30)
            .with_style(MusicStyle::Folk)
            .with_sample_rate(48000)
            .with_bitrate(320000)
            .with_format("wav");

        assert_eq!(request.model, "music-2.0");
        assert_eq!(request.prompt, "Indie folk, melancholic");
        assert_eq!(request.lyrics, Some("[verse]\nTest lyrics".to_string()));
        assert_eq!(request.duration, Some(30));
        assert_eq!(request.style, Some(MusicStyle::Folk));

        let setting = request.audio_setting.unwrap();
        assert_eq!(setting.sample_rate, Some(48000));
        assert_eq!(setting.bitrate, Some(320000));
        assert_eq!(setting.format, Some("wav".to_string()));
    }

    #[test]
    fn test_music_request_minimal() {
        let request = MusicGenerationRequest::new("music-2.0", "Upbeat jazz");

        assert_eq!(request.model, "music-2.0");
        assert_eq!(request.prompt, "Upbeat jazz");
        assert_eq!(request.lyrics, None);
        assert_eq!(request.duration, None);
        assert!(request.audio_setting.is_some());
    }

    #[test]
    fn test_music_request_with_extra_params() {
        let request = MusicGenerationRequest::new("music-2.0", "Classical symphony")
            .with_extra_param("aigc_watermark", serde_json::json!(true))
            .with_extra_param("custom_param", serde_json::json!("value"));

        let extra = request.extra_params.unwrap();
        assert_eq!(extra.get("aigc_watermark"), Some(&serde_json::json!(true)));
        assert_eq!(extra.get("custom_param"), Some(&serde_json::json!("value")));
    }

    #[test]
    fn test_audio_setting_default() {
        let setting = MusicAudioSetting::default();
        assert_eq!(setting.sample_rate, Some(44100));
        assert_eq!(setting.bitrate, Some(256000));
        assert_eq!(setting.format, Some("mp3".to_string()));
    }
}
