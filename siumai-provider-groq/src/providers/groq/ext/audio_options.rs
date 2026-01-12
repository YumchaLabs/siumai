//! Groq audio option helpers (non-unified surface)
//!
//! Groq's audio endpoints are OpenAI-compatible, but the provider id is `"groq"`.
//! These helpers build `(provider_id, json)` entries for `providerOptions`.

use crate::error::LlmError;
use crate::types::CustomProviderOptions;

/// Groq-specific options for TTS requests.
#[derive(Debug, Clone, Default)]
pub struct GroqTtsOptions {
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
#[derive(Debug, Clone, Default)]
pub struct GroqSttOptions {
    pub response_format: Option<String>,
    pub prompt: Option<String>,
    pub temperature: Option<f64>,
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
                "response_format".to_string(),
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
        if let Some(v) = self.stream {
            obj.insert("stream".to_string(), serde_json::Value::Bool(v));
        }
        Ok(serde_json::Value::Object(obj))
    }
}
