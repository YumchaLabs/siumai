//! MiniMaxi text-to-speech helpers (extension API).
//!
//! MiniMaxi TTS supports extra vendor parameters (emotion, pitch, bitrate, etc).
//! Since `TtsRequest` is still `extra_params` based, this module provides a
//! type-safe helper builder for MiniMaxi-specific knobs.

use crate::types::TtsRequest;

/// Type-safe builder for MiniMaxi TTS vendor parameters.
#[derive(Debug, Clone)]
pub struct MinimaxiTtsRequestBuilder {
    request: TtsRequest,
}

impl MinimaxiTtsRequestBuilder {
    /// Create a MiniMaxi TTS request builder with required text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            request: TtsRequest::new(text.into()),
        }
    }

    /// Set the MiniMaxi TTS model (e.g. `speech-2.6-hd`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.request = self.request.with_model(model.into());
        self
    }

    /// Set the voice id (e.g. `male-qn-qingse`).
    pub fn voice_id(mut self, voice_id: impl Into<String>) -> Self {
        self.request = self.request.with_voice(voice_id.into());
        self
    }

    /// Set output format (e.g. `mp3`).
    pub fn format(mut self, format: impl Into<String>) -> Self {
        self.request = self.request.with_format(format.into());
        self
    }

    /// Set speed (defaults to 1.0 in MiniMaxi transformer).
    pub fn speed(mut self, speed: f32) -> Self {
        self.request = self.request.with_speed(speed);
        self
    }

    /// Set volume (defaults to 1.0).
    pub fn vol(mut self, vol: f64) -> Self {
        self.request
            .extra_params
            .insert("vol".to_string(), serde_json::json!(vol));
        self
    }

    /// Set pitch (defaults to 0).
    pub fn pitch(mut self, pitch: i64) -> Self {
        self.request
            .extra_params
            .insert("pitch".to_string(), serde_json::json!(pitch));
        self
    }

    /// Set emotion (defaults to `neutral`).
    pub fn emotion(mut self, emotion: impl Into<String>) -> Self {
        self.request.extra_params.insert(
            "emotion".to_string(),
            serde_json::Value::String(emotion.into()),
        );
        self
    }

    /// Set audio sample rate (defaults to 32000).
    pub fn sample_rate(mut self, sample_rate: u32) -> Self {
        self.request.extra_params.insert(
            "sample_rate".to_string(),
            serde_json::Value::Number(serde_json::Number::from(sample_rate)),
        );
        self
    }

    /// Set audio bitrate (defaults to 128000).
    pub fn bitrate(mut self, bitrate: u32) -> Self {
        self.request.extra_params.insert(
            "bitrate".to_string(),
            serde_json::Value::Number(serde_json::Number::from(bitrate)),
        );
        self
    }

    /// Set audio channel count (defaults to 1).
    pub fn channel(mut self, channel: u32) -> Self {
        self.request.extra_params.insert(
            "channel".to_string(),
            serde_json::Value::Number(serde_json::Number::from(channel)),
        );
        self
    }

    /// Provide optional pronunciation dictionary (vendor JSON object).
    pub fn pronunciation_dict(mut self, dict: serde_json::Value) -> Self {
        self.request
            .extra_params
            .insert("pronunciation_dict".to_string(), dict);
        self
    }

    /// Provide optional voice modify config (vendor JSON object).
    pub fn voice_modify(mut self, voice_modify: serde_json::Value) -> Self {
        self.request
            .extra_params
            .insert("voice_modify".to_string(), voice_modify);
        self
    }

    /// Enable subtitle output if the vendor supports it.
    pub fn subtitle_enable(mut self, enabled: bool) -> Self {
        self.request.extra_params.insert(
            "subtitle_enable".to_string(),
            serde_json::Value::Bool(enabled),
        );
        self
    }

    /// Finish building the `TtsRequest`.
    pub fn build(self) -> TtsRequest {
        self.request
    }
}
