//! Groq Provider Options
//!
//! This module contains types for Groq-specific features.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Groq-specific options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GroqOptions {
    /// Additional Groq-specific parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl GroqOptions {
    /// Create new Groq options
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a custom parameter
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}
