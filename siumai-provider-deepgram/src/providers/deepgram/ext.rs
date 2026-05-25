//! Deepgram request option helpers.

use super::options::{DeepgramSpeechOptions, DeepgramSttOptions};
use crate::types::{CustomProviderOptions, ProviderOptionsMap, SttRequest, TtsRequest};

fn merge_provider_option_object(map: &mut ProviderOptionsMap, value: serde_json::Value) {
    if let serde_json::Value::Object(new_options) = value {
        let mut merged = map
            .get("deepgram")
            .and_then(|value| value.as_object())
            .cloned()
            .unwrap_or_default();

        for (key, value) in new_options {
            merged.insert(key, value);
        }

        map.insert("deepgram", serde_json::Value::Object(merged));
    } else {
        map.insert("deepgram", value);
    }
}

/// Deepgram request option helpers for `TtsRequest`.
pub trait DeepgramTtsRequestExt {
    /// Attach Deepgram-specific speech options to `provider_options_map["deepgram"]`.
    fn with_deepgram_tts_options(self, options: DeepgramSpeechOptions) -> Self;
}

impl DeepgramTtsRequestExt for TtsRequest {
    fn with_deepgram_tts_options(mut self, options: DeepgramSpeechOptions) -> Self {
        let value = options.to_json().expect("serialize DeepgramSpeechOptions");
        merge_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

/// Deepgram request option helpers for `SttRequest`.
pub trait DeepgramSttRequestExt {
    /// Attach Deepgram-specific transcription options to `provider_options_map["deepgram"]`.
    fn with_deepgram_stt_options(self, options: DeepgramSttOptions) -> Self;
}

impl DeepgramSttRequestExt for SttRequest {
    fn with_deepgram_stt_options(mut self, options: DeepgramSttOptions) -> Self {
        let value = options.to_json().expect("serialize DeepgramSttOptions");
        merge_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::deepgram::options::{
        DeepgramSpeechModelOptions, DeepgramTranscriptionModelOptions,
    };

    #[test]
    fn tts_request_ext_merges_existing_deepgram_options() {
        let request = TtsRequest::new("hello".to_string())
            .with_provider_option("deepgram", serde_json::json!({ "existing": true }))
            .with_deepgram_tts_options(
                DeepgramSpeechModelOptions::new()
                    .with_encoding("linear16")
                    .with_sample_rate(24_000),
            );

        assert_eq!(
            request.provider_options_map.get("deepgram"),
            Some(&serde_json::json!({
                "existing": true,
                "encoding": "linear16",
                "sampleRate": 24000
            }))
        );
    }

    #[test]
    fn stt_request_ext_merges_existing_deepgram_options() {
        let request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg")
            .with_provider_option("deepgram", serde_json::json!({ "existing": true }))
            .with_deepgram_stt_options(
                DeepgramTranscriptionModelOptions::new()
                    .with_language("en")
                    .with_diarize(false),
            );

        assert_eq!(
            request.provider_options_map.get("deepgram"),
            Some(&serde_json::json!({
                "existing": true,
                "language": "en",
                "diarize": false
            }))
        );
    }
}
