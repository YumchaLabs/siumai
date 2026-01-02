//! MiniMaxi text-to-speech helpers (extension API).
//!
//! MiniMaxi TTS supports extra vendor parameters (emotion, pitch, bitrate, etc).
//! These knobs are carried via the open `providerOptions["minimaxi"]` bucket to keep the unified
//! surface minimal while still supporting provider-specific features.

use crate::provider_options::MinimaxiTtsOptions;
use crate::types::TtsRequest;

/// Type-safe builder for MiniMaxi TTS vendor parameters.
#[derive(Debug, Clone)]
pub struct MinimaxiTtsRequestBuilder {
    request: TtsRequest,
    vendor_options: MinimaxiTtsOptions,
}

impl MinimaxiTtsRequestBuilder {
    /// Create a MiniMaxi TTS request builder with required text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            request: TtsRequest::new(text.into()),
            vendor_options: MinimaxiTtsOptions::new(),
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
        self.vendor_options.vol = Some(vol);
        self
    }

    /// Set pitch (defaults to 0).
    pub fn pitch(mut self, pitch: i64) -> Self {
        self.vendor_options.pitch = Some(pitch);
        self
    }

    /// Set emotion (defaults to `neutral`).
    pub fn emotion(mut self, emotion: impl Into<String>) -> Self {
        self.vendor_options.emotion = Some(emotion.into());
        self
    }

    /// Set audio sample rate (defaults to 32000).
    pub fn sample_rate(mut self, sample_rate: u32) -> Self {
        self.vendor_options.sample_rate = Some(sample_rate as u64);
        self
    }

    /// Set audio bitrate (defaults to 128000).
    pub fn bitrate(mut self, bitrate: u32) -> Self {
        self.vendor_options.bitrate = Some(bitrate as u64);
        self
    }

    /// Set audio channel count (defaults to 1).
    pub fn channel(mut self, channel: u32) -> Self {
        self.vendor_options.channel = Some(channel as u64);
        self
    }

    /// Provide optional pronunciation dictionary (vendor JSON object).
    pub fn pronunciation_dict(mut self, dict: serde_json::Value) -> Self {
        self.vendor_options.pronunciation_dict = Some(dict);
        self
    }

    /// Provide optional voice modify config (vendor JSON object).
    pub fn voice_modify(mut self, voice_modify: serde_json::Value) -> Self {
        self.vendor_options.voice_modify = Some(voice_modify);
        self
    }

    /// Enable subtitle output if the vendor supports it.
    pub fn subtitle_enable(mut self, enabled: bool) -> Self {
        self.vendor_options.subtitle_enable = Some(enabled);
        self
    }

    /// Finish building the `TtsRequest`.
    pub fn build(self) -> TtsRequest {
        let mut request = self.request;
        if !self.vendor_options.is_empty() {
            let value =
                serde_json::to_value(self.vendor_options).unwrap_or(serde_json::Value::Null);
            request.provider_options_map.insert("minimaxi", value);
        }
        request
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_stores_vendor_knobs_in_provider_options_bucket() {
        let req = MinimaxiTtsRequestBuilder::new("hi")
            .voice_id("male-qn-qingse")
            .format("mp3")
            .vol(0.5)
            .pitch(1)
            .emotion("happy")
            .build();

        let obj = req
            .provider_options_map
            .get("minimaxi")
            .and_then(|v| v.as_object())
            .expect("minimaxi provider options bucket");
        assert_eq!(obj.get("vol"), Some(&serde_json::json!(0.5)));
        assert_eq!(obj.get("pitch"), Some(&serde_json::json!(1)));
        assert_eq!(obj.get("emotion"), Some(&serde_json::json!("happy")));
    }
}
