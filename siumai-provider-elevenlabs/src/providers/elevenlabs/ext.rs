//! ElevenLabs request option helpers.

use super::options::{ElevenLabsSpeechOptions, ElevenLabsSttOptions};
use crate::types::{CustomProviderOptions, ProviderOptionsMap, SttRequest, TtsRequest};

fn merge_provider_option_object(map: &mut ProviderOptionsMap, value: serde_json::Value) {
    if let serde_json::Value::Object(new_options) = value {
        let mut merged = map
            .get("elevenlabs")
            .and_then(|value| value.as_object())
            .cloned()
            .unwrap_or_default();

        for (key, value) in new_options {
            merged.insert(key, value);
        }

        map.insert("elevenlabs", serde_json::Value::Object(merged));
    } else {
        map.insert("elevenlabs", value);
    }
}

/// ElevenLabs request option helpers for `TtsRequest`.
pub trait ElevenLabsTtsRequestExt {
    /// Attach ElevenLabs-specific speech options to `provider_options_map["elevenlabs"]`.
    fn with_elevenlabs_tts_options(self, options: ElevenLabsSpeechOptions) -> Self;
}

impl ElevenLabsTtsRequestExt for TtsRequest {
    fn with_elevenlabs_tts_options(mut self, options: ElevenLabsSpeechOptions) -> Self {
        let value = options
            .to_json()
            .expect("serialize ElevenLabsSpeechOptions");
        merge_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

/// ElevenLabs request option helpers for `SttRequest`.
pub trait ElevenLabsSttRequestExt {
    /// Attach ElevenLabs-specific transcription options to `provider_options_map["elevenlabs"]`.
    fn with_elevenlabs_stt_options(self, options: ElevenLabsSttOptions) -> Self;
}

impl ElevenLabsSttRequestExt for SttRequest {
    fn with_elevenlabs_stt_options(mut self, options: ElevenLabsSttOptions) -> Self {
        let value = options
            .to_json()
            .expect("serialize ElevenLabsTranscriptionOptions");
        merge_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::elevenlabs::options::{
        ElevenLabsSpeechModelOptions, ElevenLabsTranscriptionModelOptions,
        ElevenLabsTranscriptionTimestampsGranularity, ElevenLabsVoiceSettings,
    };

    #[test]
    fn tts_request_ext_merges_existing_elevenlabs_options() {
        let request = TtsRequest::new("hello".to_string())
            .with_provider_option("elevenlabs", serde_json::json!({ "existing": true }))
            .with_elevenlabs_tts_options(
                ElevenLabsSpeechModelOptions::new()
                    .with_voice_settings(ElevenLabsVoiceSettings::new().with_stability(0.5)),
            );

        assert_eq!(
            request.provider_options_map.get("elevenlabs"),
            Some(&serde_json::json!({
                "existing": true,
                "voiceSettings": {
                    "stability": 0.5
                }
            }))
        );
    }

    #[test]
    fn stt_request_ext_merges_existing_elevenlabs_options() {
        let request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg")
            .with_provider_option("elevenlabs", serde_json::json!({ "existing": true }))
            .with_elevenlabs_stt_options(
                ElevenLabsTranscriptionModelOptions::new()
                    .with_language_code("en")
                    .with_timestamps_granularity(
                        ElevenLabsTranscriptionTimestampsGranularity::Word,
                    ),
            );

        assert_eq!(
            request.provider_options_map.get("elevenlabs"),
            Some(&serde_json::json!({
                "existing": true,
                "languageCode": "en",
                "timestampsGranularity": "word"
            }))
        );
    }
}
