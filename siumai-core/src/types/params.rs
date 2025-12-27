//! Common AI parameter types.
//!
//! This module defines `CommonParams` and its builder, used across providers.

use serde::{Deserialize, Serialize};

/// Common AI parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CommonParams {
    /// Model name
    pub model: String,

    /// Temperature parameter (must be non-negative)
    pub temperature: Option<f32>,

    /// Maximum output tokens (deprecated for o1/o3 models, use max_completion_tokens instead)
    pub max_tokens: Option<u32>,

    /// Maximum completion tokens (for o1/o3 reasoning models)
    /// This is an upper bound for the number of tokens that can be generated for a completion,
    /// including visible output tokens and reasoning tokens.
    pub max_completion_tokens: Option<u32>,

    /// `top_p` parameter
    pub top_p: Option<f32>,

    /// Stop sequences
    pub stop_sequences: Option<Vec<String>>,

    /// Random seed
    pub seed: Option<u64>,
}

impl CommonParams {
    /// Create `CommonParams` with pre-allocated model string capacity
    pub const fn with_model_capacity(model: String, _capacity_hint: usize) -> Self {
        Self {
            model,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            top_p: None,
            stop_sequences: None,
            seed: None,
        }
    }

    /// Check if parameters are effectively empty (for optimization)
    pub const fn is_minimal(&self) -> bool {
        self.model.is_empty()
            && self.temperature.is_none()
            && self.max_tokens.is_none()
            && self.max_completion_tokens.is_none()
            && self.top_p.is_none()
            && self.stop_sequences.is_none()
            && self.seed.is_none()
    }

    /// Estimate memory usage for caching decisions
    pub fn memory_footprint(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();
        size += self.model.capacity();
        if let Some(ref stop_seqs) = self.stop_sequences {
            size += stop_seqs
                .iter()
                .map(std::string::String::capacity)
                .sum::<usize>();
        }
        size
    }

    /// Create a hash for caching (performance optimized)
    pub fn cache_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.model.hash(&mut hasher);
        self.temperature
            .map(|t| (t * 1000.0) as u32)
            .hash(&mut hasher);
        self.max_tokens.hash(&mut hasher);
        self.top_p.map(|t| (t * 1000.0) as u32).hash(&mut hasher);
        hasher.finish()
    }

    /// Validate common parameters
    pub fn validate_params(&self) -> Result<(), crate::error::LlmError> {
        // Validate model name
        if self.model.is_empty() {
            return Err(crate::error::LlmError::InvalidParameter(
                "Model name cannot be empty".to_string(),
            ));
        }

        // Validate temperature (must be non-negative)
        if let Some(temp) = self.temperature
            && temp < 0.0
        {
            return Err(crate::error::LlmError::InvalidParameter(
                "Temperature must be non-negative".to_string(),
            ));
        }

        // Validate top_p (must be between 0.0 and 1.0)
        if let Some(top_p) = self.top_p
            && !(0.0..=1.0).contains(&top_p)
        {
            return Err(crate::error::LlmError::InvalidParameter(
                "top_p must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }

    /// Create a builder for common parameters
    pub fn builder() -> CommonParamsBuilder {
        CommonParamsBuilder::new()
    }
}

/// Builder for CommonParams with validation
#[derive(Debug, Clone, Default)]
pub struct CommonParamsBuilder {
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    stop_sequences: Option<Vec<String>>,
    seed: Option<u64>,
}

impl CommonParamsBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model name
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    /// Set the temperature with validation
    pub fn temperature(mut self, temperature: f32) -> Result<Self, crate::error::LlmError> {
        if !(0.0..=2.0).contains(&temperature) {
            return Err(crate::error::LlmError::InvalidParameter(
                "Temperature must be between 0.0 and 2.0".to_string(),
            ));
        }
        self.temperature = Some(temperature);
        Ok(self)
    }

    /// Set the max tokens
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top_p with validation
    pub fn top_p(mut self, top_p: f32) -> Result<Self, crate::error::LlmError> {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(crate::error::LlmError::InvalidParameter(
                "top_p must be between 0.0 and 1.0".to_string(),
            ));
        }
        self.top_p = Some(top_p);
        Ok(self)
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Set the random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the CommonParams
    pub fn build(self) -> Result<CommonParams, crate::error::LlmError> {
        let params = CommonParams {
            model: self.model,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            max_completion_tokens: None,
            top_p: self.top_p,
            stop_sequences: self.stop_sequences,
            seed: self.seed,
        };

        params.validate_params()?;
        Ok(params)
    }
}
