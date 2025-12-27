//! OpenAI audio option helpers (non-unified surface)
//!
//! These helpers build `ProviderOptions::Custom { provider_id, options }` buckets for audio APIs.
//! They intentionally keep the unified `TtsRequest`/`SttRequest` surface small while still
//! enabling provider-specific escape hatches.

use crate::error::LlmError;
use crate::types::{CustomProviderOptions, ProviderOptions};

/// OpenAI-specific options for TTS requests.
#[derive(Debug, Clone, Default)]
pub struct OpenAiTtsOptions {
    /// Optional TTS instructions.
    pub instructions: Option<String>,
}

impl OpenAiTtsOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    pub fn into_provider_options(self) -> Result<ProviderOptions, LlmError> {
        ProviderOptions::from_custom(self)
    }
}

impl CustomProviderOptions for OpenAiTtsOptions {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        let mut obj = serde_json::Map::new();
        if let Some(v) = self.instructions.as_deref() {
            obj.insert("instructions".to_string(), serde_json::Value::String(v.to_string()));
        }
        Ok(serde_json::Value::Object(obj))
    }
}

/// OpenAI-specific options for STT requests.
#[derive(Debug, Clone, Default)]
pub struct OpenAiSttOptions {
    pub response_format: Option<String>,
    pub prompt: Option<String>,
    pub temperature: Option<f64>,
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

    pub fn into_provider_options(self) -> Result<ProviderOptions, LlmError> {
        ProviderOptions::from_custom(self)
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
                "response_format".to_string(),
                serde_json::Value::String(v.to_string()),
            );
        }
        if let Some(v) = self.prompt.as_deref() {
            obj.insert("prompt".to_string(), serde_json::Value::String(v.to_string()));
        }
        if let Some(v) = self.temperature {
            obj.insert("temperature".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.chunking_strategy.as_deref() {
            obj.insert(
                "chunking_strategy".to_string(),
                serde_json::Value::String(v.to_string()),
            );
        }
        if let Some(v) = self.include.as_ref() {
            obj.insert("include".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.known_speaker_names.as_ref() {
            obj.insert("known_speaker_names".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.known_speaker_references.as_ref() {
            obj.insert(
                "known_speaker_references".to_string(),
                serde_json::json!(v),
            );
        }
        if let Some(v) = self.stream {
            obj.insert("stream".to_string(), serde_json::Value::Bool(v));
        }

        Ok(serde_json::Value::Object(obj))
    }
}

