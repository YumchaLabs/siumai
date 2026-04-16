//! OpenAI audio option helpers (non-unified surface)
//!
//! These helpers build `(provider_id, json)` entries for `providerOptions`.
//! They intentionally keep the unified `TtsRequest`/`SttRequest` surface small while still
//! enabling provider-specific escape hatches.

use crate::error::LlmError;
use crate::types::CustomProviderOptions;
use serde::{Deserialize, Serialize};

/// OpenAI-specific options for TTS requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAiTtsOptions {
    /// Optional TTS instructions.
    pub instructions: Option<String>,
    /// Optional TTS speed override.
    pub speed: Option<f32>,
}

impl OpenAiTtsOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    pub const fn with_speed(mut self, speed: f32) -> Self {
        self.speed = Some(speed);
        self
    }

    pub fn into_provider_options_map_entry(self) -> Result<(String, serde_json::Value), LlmError> {
        self.to_provider_options_map_entry()
    }
}

impl CustomProviderOptions for OpenAiTtsOptions {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        let mut obj = serde_json::Map::new();
        if let Some(v) = self.instructions.as_deref() {
            obj.insert(
                "instructions".to_string(),
                serde_json::Value::String(v.to_string()),
            );
        }
        if let Some(v) = self.speed {
            obj.insert("speed".to_string(), serde_json::json!(v));
        }
        Ok(serde_json::Value::Object(obj))
    }
}

/// OpenAI-specific options for STT requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAiSttOptions {
    pub response_format: Option<String>,
    pub prompt: Option<String>,
    pub temperature: Option<f64>,
    pub language: Option<String>,
    pub timestamp_granularities: Option<Vec<String>>,
    pub chunking_strategy: Option<String>,
    pub include: Option<Vec<String>>,
    pub known_speaker_names: Option<Vec<String>>,
    pub known_speaker_references: Option<Vec<String>>,
    /// Provider-specific streaming flag (not part of the unified surface).
    pub stream: Option<bool>,
}

impl OpenAiSttOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_response_format(mut self, response_format: impl Into<String>) -> Self {
        self.response_format = Some(response_format.into());
        self
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    pub fn with_timestamp_granularities(mut self, granularities: Vec<String>) -> Self {
        self.timestamp_granularities = Some(granularities);
        self
    }

    pub fn with_chunking_strategy(mut self, chunking_strategy: impl Into<String>) -> Self {
        self.chunking_strategy = Some(chunking_strategy.into());
        self
    }

    pub fn with_include(mut self, include: Vec<String>) -> Self {
        self.include = Some(include);
        self
    }

    pub fn with_known_speaker_names(mut self, names: Vec<String>) -> Self {
        self.known_speaker_names = Some(names);
        self
    }

    pub fn with_known_speaker_references(mut self, references: Vec<String>) -> Self {
        self.known_speaker_references = Some(references);
        self
    }

    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn into_provider_options_map_entry(self) -> Result<(String, serde_json::Value), LlmError> {
        self.to_provider_options_map_entry()
    }
}

impl CustomProviderOptions for OpenAiSttOptions {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        let mut obj = serde_json::Map::new();

        if let Some(v) = self.response_format.as_deref() {
            obj.insert(
                "responseFormat".to_string(),
                serde_json::Value::String(v.to_string()),
            );
        }
        if let Some(v) = self.prompt.as_deref() {
            obj.insert(
                "prompt".to_string(),
                serde_json::Value::String(v.to_string()),
            );
        }
        if let Some(v) = self.temperature {
            obj.insert("temperature".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.language.as_deref() {
            obj.insert(
                "language".to_string(),
                serde_json::Value::String(v.to_string()),
            );
        }
        if let Some(v) = self.timestamp_granularities.as_ref() {
            obj.insert("timestampGranularities".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.chunking_strategy.as_deref() {
            obj.insert(
                "chunkingStrategy".to_string(),
                serde_json::Value::String(v.to_string()),
            );
        }
        if let Some(v) = self.include.as_ref() {
            obj.insert("include".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.known_speaker_names.as_ref() {
            obj.insert("knownSpeakerNames".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.known_speaker_references.as_ref() {
            obj.insert("knownSpeakerReferences".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.stream {
            obj.insert("stream".to_string(), serde_json::Value::Bool(v));
        }

        Ok(serde_json::Value::Object(obj))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tts_options_serialize_speed_and_instructions() {
        let value = OpenAiTtsOptions::new()
            .with_instructions("speak slowly")
            .with_speed(1.25)
            .to_json()
            .expect("tts options serialize");

        assert_eq!(value["instructions"], serde_json::json!("speak slowly"));
        assert_eq!(value["speed"], serde_json::json!(1.25));
    }

    #[test]
    fn stt_options_serialize_ai_sdk_style_keys() {
        let value = OpenAiSttOptions::new()
            .with_response_format("verbose_json")
            .with_language("en")
            .with_timestamp_granularities(vec!["word".to_string(), "segment".to_string()])
            .with_chunking_strategy("auto")
            .to_json()
            .expect("stt options serialize");

        assert_eq!(value["responseFormat"], serde_json::json!("verbose_json"));
        assert_eq!(value["language"], serde_json::json!("en"));
        assert_eq!(
            value["timestampGranularities"],
            serde_json::json!(["word", "segment"])
        );
        assert_eq!(value["chunkingStrategy"], serde_json::json!("auto"));
    }
}
