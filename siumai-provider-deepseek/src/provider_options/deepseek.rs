//! `DeepSeek` provider options.
//!
//! These typed option structs are owned by the DeepSeek provider crate and are serialized into
//! `providerOptions["deepseek"]` (Vercel-aligned open options map).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// DeepSeek-specific options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeepSeekOptions {
    /// Enable reasoning / thinking output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_reasoning: Option<bool>,
    /// Reasoning budget for DeepSeek reasoning-capable models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_budget: Option<i32>,
    /// Additional provider-specific parameters.
    #[serde(flatten)]
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl DeepSeekOptions {
    /// Create new DeepSeek options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable reasoning output.
    pub fn with_reasoning(mut self, enable: bool) -> Self {
        self.enable_reasoning = Some(enable);
        self
    }

    /// Set reasoning budget and implicitly enable reasoning.
    pub fn with_reasoning_budget(mut self, budget: i32) -> Self {
        self.reasoning_budget = Some(budget);
        self.enable_reasoning = Some(true);
        self
    }

    /// Add a custom DeepSeek parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek_options_reasoning_budget_implies_enable_reasoning() {
        let options = DeepSeekOptions::new().with_reasoning_budget(2048);
        let value = serde_json::to_value(options).expect("serialize options");
        assert_eq!(value["enable_reasoning"], serde_json::json!(true));
        assert_eq!(value["reasoning_budget"], serde_json::json!(2048));
    }

    #[test]
    fn deepseek_options_support_extra_params() {
        let options = DeepSeekOptions::new().with_param("foo", serde_json::json!("bar"));
        let value = serde_json::to_value(options).expect("serialize options");
        assert_eq!(value["foo"], serde_json::json!("bar"));
    }
}
