//! MiniMaxi music generation helpers (extension API).
//!
//! MiniMaxi Music 2.0 requires a `lyrics` field at the API layer. Siumai’s
//! `MinimaxiClient` will auto-fill a default structure if lyrics are omitted,
//! but this module provides an explicit, type-safe builder for better ergonomics.

use crate::types::music::{MusicAudioSetting, MusicGenerationRequest};

const DEFAULT_LYRICS_TEMPLATE: &str = "[Intro]\n[Main]\n[Outro]";

/// Type-safe builder for MiniMaxi music generation requests.
#[derive(Debug, Clone)]
pub struct MinimaxiMusicRequestBuilder {
    request: MusicGenerationRequest,
}

impl MinimaxiMusicRequestBuilder {
    /// Create a request builder with a prompt and MiniMaxi default model (`music-2.0`).
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            request: MusicGenerationRequest::new("music-2.0", prompt),
        }
    }

    /// Override the model (defaults to `music-2.0`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.request.model = model.into();
        self
    }

    /// Set lyrics explicitly.
    pub fn lyrics(mut self, lyrics: impl Into<String>) -> Self {
        self.request = self.request.with_lyrics(lyrics);
        self
    }

    /// Use MiniMaxi’s recommended minimal lyrics template (useful for instrumental prompts).
    pub fn lyrics_template(mut self) -> Self {
        self.request = self
            .request
            .with_lyrics(DEFAULT_LYRICS_TEMPLATE.to_string());
        self
    }

    /// Set audio settings as a whole.
    pub fn audio_setting(mut self, setting: MusicAudioSetting) -> Self {
        self.request = self.request.with_audio_setting(setting);
        self
    }

    /// Set sample rate (defaults to 44100).
    pub fn sample_rate(mut self, sample_rate: u32) -> Self {
        self.request = self.request.with_sample_rate(sample_rate);
        self
    }

    /// Set bitrate (defaults to 256000).
    pub fn bitrate(mut self, bitrate: u32) -> Self {
        self.request = self.request.with_bitrate(bitrate);
        self
    }

    /// Set output format (e.g. "mp3", "wav").
    pub fn format(mut self, format: impl Into<String>) -> Self {
        self.request = self.request.with_format(format);
        self
    }

    /// Finish building the `MusicGenerationRequest`.
    pub fn build(self) -> MusicGenerationRequest {
        self.request
    }
}
