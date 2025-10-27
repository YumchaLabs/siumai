//! Gemini Parameter Mapping
//!
//! Contains Gemini-specific parameter mapping and validation logic.

use serde::{Deserialize, Serialize};

use crate::types::ProviderType;

// Gemini ParameterMapper removed; use Transformers for mapping/validation.

/// Gemini-specific parameter extensions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeminiParams {
    /// Top-K sampling parameter
    pub top_k: Option<u32>,
    /// Number of candidate responses to generate
    pub candidate_count: Option<u32>,
    /// Safety settings
    pub safety_settings: Option<Vec<SafetySetting>>,
    /// Generation configuration
    pub generation_config: Option<GenerationConfig>,
    /// Whether to stream the response
    pub stream: Option<bool>,
}

impl super::common::ProviderParamsExt for GeminiParams {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Gemini
    }
}

/// Gemini Safety Setting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetySetting {
    pub category: SafetyCategory,
    pub threshold: SafetyThreshold,
}

/// Gemini Safety Categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyCategory {
    #[serde(rename = "HARM_CATEGORY_HARASSMENT")]
    Harassment,
    #[serde(rename = "HARM_CATEGORY_HATE_SPEECH")]
    HateSpeech,
    #[serde(rename = "HARM_CATEGORY_SEXUALLY_EXPLICIT")]
    SexuallyExplicit,
    #[serde(rename = "HARM_CATEGORY_DANGEROUS_CONTENT")]
    DangerousContent,
}

/// Gemini Safety Thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyThreshold {
    #[serde(rename = "BLOCK_NONE")]
    BlockNone,
    #[serde(rename = "BLOCK_LOW_AND_ABOVE")]
    BlockLowAndAbove,
    #[serde(rename = "BLOCK_MEDIUM_AND_ABOVE")]
    BlockMediumAndAbove,
    #[serde(rename = "BLOCK_HIGH_AND_ABOVE")]
    BlockHighAndAbove,
}

/// Gemini Generation Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub candidate_count: Option<u32>,
}

/// Gemini parameter builder for convenient parameter construction
pub struct GeminiParamsBuilder {
    params: GeminiParams,
}

impl GeminiParamsBuilder {
    pub fn new() -> Self {
        Self {
            params: GeminiParams::default(),
        }
    }

    pub const fn top_k(mut self, top_k: u32) -> Self {
        self.params.top_k = Some(top_k);
        self
    }

    pub const fn candidate_count(mut self, count: u32) -> Self {
        self.params.candidate_count = Some(count);
        self
    }

    pub fn safety_settings(mut self, settings: Vec<SafetySetting>) -> Self {
        self.params.safety_settings = Some(settings);
        self
    }

    pub fn add_safety_setting(
        mut self,
        category: SafetyCategory,
        threshold: SafetyThreshold,
    ) -> Self {
        if self.params.safety_settings.is_none() {
            self.params.safety_settings = Some(Vec::new());
        }
        self.params
            .safety_settings
            .as_mut()
            .unwrap()
            .push(SafetySetting {
                category,
                threshold,
            });
        self
    }

    pub fn generation_config(mut self, config: GenerationConfig) -> Self {
        self.params.generation_config = Some(config);
        self
    }

    pub const fn stream(mut self, enabled: bool) -> Self {
        self.params.stream = Some(enabled);
        self
    }

    pub fn build(self) -> GeminiParams {
        self.params
    }
}

impl Default for GeminiParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// tests removed; covered by Transformers
