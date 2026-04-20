//! Groq audio option helpers (non-unified surface)
//!
//! Groq's audio endpoints are OpenAI-compatible, but the provider id is `"groq"`.
//! These helpers build `(provider_id, json)` entries for `providerOptions`.

use crate::error::LlmError;
use crate::provider_options::GroqTranscriptionModelOptions;
use crate::types::{CustomProviderOptions, SttRequest};
use serde::{Deserialize, Serialize};

/// Groq-specific options for TTS requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GroqTtsOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

impl GroqTtsOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    pub fn into_provider_options_map_entry(self) -> Result<(String, serde_json::Value), LlmError> {
        self.to_provider_options_map_entry()
    }
}

impl CustomProviderOptions for GroqTtsOptions {
    fn provider_id(&self) -> &str {
        "groq"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        let mut obj = serde_json::Map::new();
        if let Some(v) = self.instructions.as_deref() {
            obj.insert(
                "instructions".to_string(),
                serde_json::Value::String(v.to_string()),
            );
        }
        Ok(serde_json::Value::Object(obj))
    }
}

/// Legacy non-unified alias for Groq transcription options.
pub type GroqSttOptions = GroqTranscriptionModelOptions;

fn merge_provider_option_object(
    map: &mut crate::types::ProviderOptionsMap,
    value: serde_json::Value,
) {
    if let serde_json::Value::Object(new_options) = value {
        let mut merged = map
            .get("groq")
            .and_then(|value| value.as_object())
            .cloned()
            .unwrap_or_default();

        for (key, value) in new_options {
            merged.insert(key, value);
        }

        map.insert("groq", serde_json::Value::Object(merged));
    } else {
        map.insert("groq", value);
    }
}

/// Groq request option helpers for `SttRequest`.
pub trait GroqSttRequestExt {
    /// Convenience: attach Groq-specific transcription options to
    /// `provider_options_map["groq"]`.
    fn with_groq_stt_options(self, options: GroqSttOptions) -> Self;
}

impl GroqSttRequestExt for SttRequest {
    fn with_groq_stt_options(mut self, options: GroqSttOptions) -> Self {
        let value = options.to_json().expect("serialize GroqSttOptions");
        merge_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tts_options_serialize_instructions() {
        let value = GroqTtsOptions::new()
            .with_instructions("speak clearly")
            .to_json()
            .expect("tts options serialize");

        assert_eq!(value["instructions"], serde_json::json!("speak clearly"));
    }

    #[test]
    fn stt_options_serialize_ai_sdk_style_keys() {
        let value = GroqSttOptions::new()
            .with_response_format("verbose_json")
            .with_prompt("prompt")
            .with_temperature(0.2)
            .with_language("en")
            .with_timestamp_granularities(vec!["segment".to_string()])
            .to_json()
            .expect("stt options serialize");

        assert_eq!(value["responseFormat"], serde_json::json!("verbose_json"));
        assert_eq!(value["prompt"], serde_json::json!("prompt"));
        assert_eq!(value["temperature"], serde_json::json!(0.2));
        assert_eq!(value["language"], serde_json::json!("en"));
        assert_eq!(
            value["timestampGranularities"],
            serde_json::json!(["segment"])
        );
    }

    #[test]
    fn stt_request_ext_merges_existing_groq_options() {
        let request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg")
            .with_provider_option("groq", serde_json::json!({ "existing": true }))
            .with_groq_stt_options(
                GroqSttOptions::new()
                    .with_language("en")
                    .with_timestamp_granularities(vec!["segment".to_string()]),
            );

        assert_eq!(
            request.provider_options_map.get("groq"),
            Some(&serde_json::json!({
                "existing": true,
                "language": "en",
                "timestampGranularities": ["segment"]
            }))
        );
    }
}
