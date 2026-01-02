//! MiniMaxi text-to-speech options.
//!
//! These typed option structs are owned by the MiniMaxi provider crate and are serialized into
//! `providerOptions["minimaxi"]` (Vercel-aligned open options map).

use serde::{Deserialize, Serialize};

/// MiniMaxi-specific options for TTS requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MinimaxiTtsOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vol: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pitch: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bitrate: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pronunciation_dict: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_modify: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subtitle_enable: Option<bool>,
}

impl MinimaxiTtsOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.vol.is_none()
            && self.pitch.is_none()
            && self.emotion.is_none()
            && self.sample_rate.is_none()
            && self.bitrate.is_none()
            && self.channel.is_none()
            && self.pronunciation_dict.is_none()
            && self.voice_modify.is_none()
            && self.subtitle_enable.is_none()
    }

    pub fn with_vol(mut self, vol: f64) -> Self {
        self.vol = Some(vol);
        self
    }

    pub fn with_pitch(mut self, pitch: i64) -> Self {
        self.pitch = Some(pitch);
        self
    }

    pub fn with_emotion(mut self, emotion: impl Into<String>) -> Self {
        self.emotion = Some(emotion.into());
        self
    }

    pub fn with_sample_rate(mut self, sample_rate: u64) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    pub fn with_bitrate(mut self, bitrate: u64) -> Self {
        self.bitrate = Some(bitrate);
        self
    }

    pub fn with_channel(mut self, channel: u64) -> Self {
        self.channel = Some(channel);
        self
    }

    pub fn with_pronunciation_dict(mut self, pronunciation_dict: serde_json::Value) -> Self {
        self.pronunciation_dict = Some(pronunciation_dict);
        self
    }

    pub fn with_voice_modify(mut self, voice_modify: serde_json::Value) -> Self {
        self.voice_modify = Some(voice_modify);
        self
    }

    pub fn with_subtitle_enable(mut self, enabled: bool) -> Self {
        self.subtitle_enable = Some(enabled);
        self
    }
}
