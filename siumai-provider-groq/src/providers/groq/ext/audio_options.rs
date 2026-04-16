//! Groq audio option helpers (non-unified surface)
//!
//! Groq's audio endpoints are OpenAI-compatible, but the provider id is `"groq"`.
//! These helpers build `(provider_id, json)` entries for `providerOptions`.

use crate::error::LlmError;
use crate::types::CustomProviderOptions;
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

/// Groq-specific options for STT requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GroqSttOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_granularities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

impl GroqSttOptions {
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

    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn into_provider_options_map_entry(self) -> Result<(String, serde_json::Value), LlmError> {
        self.to_provider_options_map_entry()
    }
}

impl CustomProviderOptions for GroqSttOptions {
    fn provider_id(&self) -> &str {
        "groq"
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
}
