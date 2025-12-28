//! MiniMaxi TTS vendor options (non-unified surface)
//!
//! This module builds `ProviderOptions::Custom { provider_id, options }` buckets for MiniMaxi TTS.

use crate::error::LlmError;
use crate::types::{CustomProviderOptions, ProviderOptions};

/// MiniMaxi-specific options for TTS requests.
#[derive(Debug, Clone, Default)]
pub struct MinimaxiTtsOptions {
    pub vol: Option<f64>,
    pub pitch: Option<i64>,
    pub emotion: Option<String>,
    pub sample_rate: Option<u64>,
    pub bitrate: Option<u64>,
    pub channel: Option<u64>,
    pub pronunciation_dict: Option<serde_json::Value>,
    pub voice_modify: Option<serde_json::Value>,
    pub subtitle_enable: Option<bool>,
}

impl MinimaxiTtsOptions {
    pub fn new() -> Self {
        Self::default()
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

    pub fn into_provider_options(self) -> Result<ProviderOptions, LlmError> {
        ProviderOptions::from_custom(self)
    }
}

impl CustomProviderOptions for MinimaxiTtsOptions {
    fn provider_id(&self) -> &str {
        "minimaxi"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        let mut obj = serde_json::Map::new();

        if let Some(v) = self.vol {
            obj.insert("vol".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.pitch {
            obj.insert("pitch".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.emotion.as_deref() {
            obj.insert("emotion".to_string(), serde_json::Value::String(v.to_string()));
        }
        if let Some(v) = self.sample_rate {
            obj.insert("sample_rate".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.bitrate {
            obj.insert("bitrate".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.channel {
            obj.insert("channel".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.pronunciation_dict.as_ref() {
            obj.insert("pronunciation_dict".to_string(), v.clone());
        }
        if let Some(v) = self.voice_modify.as_ref() {
            obj.insert("voice_modify".to_string(), v.clone());
        }
        if let Some(v) = self.subtitle_enable {
            obj.insert("subtitle_enable".to_string(), serde_json::Value::Bool(v));
        }

        Ok(serde_json::Value::Object(obj))
    }
}

