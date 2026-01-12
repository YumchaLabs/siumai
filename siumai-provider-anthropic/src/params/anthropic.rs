//! Anthropic Parameter Mapping (legacy)
//!
//! Contains Anthropic-specific parameter structs that were historically used as
//! client-level defaults. New code should prefer request-level `AnthropicOptions`
//! via `provider_options_map`.

use crate::LlmError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Anthropic Cache Control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub r#type: String,
}

/// Anthropic-specific parameter extensions (legacy).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicParams {
    /// Cache control
    pub cache_control: Option<CacheControl>,

    /// Thinking budget
    pub thinking_budget: Option<u32>,

    /// System message
    pub system: Option<String>,

    /// Metadata
    pub metadata: Option<HashMap<String, String>>,

    /// Whether to stream the response
    pub stream: Option<bool>,

    /// Beta features
    pub beta_features: Option<Vec<String>>,
}

impl AnthropicParams {
    /// Validate Anthropic-specific parameters.
    pub fn validate_params(&self) -> Result<(), LlmError> {
        // All fields are optional and currently have no range constraints.
        Ok(())
    }

    /// Create a builder for Anthropic parameters.
    pub fn builder() -> AnthropicParamsBuilder {
        AnthropicParamsBuilder::new()
    }
}

/// Anthropic parameter builder for convenient parameter construction.
pub struct AnthropicParamsBuilder {
    params: AnthropicParams,
}

impl AnthropicParamsBuilder {
    pub fn new() -> Self {
        Self {
            params: AnthropicParams::default(),
        }
    }

    pub fn cache_control(mut self, cache_control: CacheControl) -> Self {
        self.params.cache_control = Some(cache_control);
        self
    }

    pub const fn thinking_budget(mut self, budget: u32) -> Self {
        self.params.thinking_budget = Some(budget);
        self
    }

    pub fn system(mut self, system_message: String) -> Self {
        self.params.system = Some(system_message);
        self
    }

    pub fn metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.params.metadata = Some(metadata);
        self
    }

    pub fn add_metadata(mut self, key: String, value: String) -> Self {
        if self.params.metadata.is_none() {
            self.params.metadata = Some(HashMap::new());
        }
        self.params.metadata.as_mut().unwrap().insert(key, value);
        self
    }

    pub const fn stream(mut self, enabled: bool) -> Self {
        self.params.stream = Some(enabled);
        self
    }

    pub fn beta_features(mut self, features: Vec<String>) -> Self {
        self.params.beta_features = Some(features);
        self
    }

    pub fn add_beta_feature(mut self, feature: String) -> Self {
        if self.params.beta_features.is_none() {
            self.params.beta_features = Some(Vec::new());
        }
        self.params.beta_features.as_mut().unwrap().push(feature);
        self
    }

    pub fn build(self) -> AnthropicParams {
        self.params
    }
}

impl Default for AnthropicParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheControl {
    pub fn ephemeral() -> Self {
        Self {
            r#type: "ephemeral".to_string(),
        }
    }
}
