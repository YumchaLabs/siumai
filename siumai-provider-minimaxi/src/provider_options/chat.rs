//! `MiniMaxi` chat provider options.
//!
//! These typed option structs are owned by the MiniMaxi provider crate and are serialized into
//! `providerOptions["minimaxi"]`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// MiniMaxi reasoning / thinking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimaxiThinkingModeConfig {
    /// Whether thinking mode is enabled.
    pub enabled: bool,
    /// Optional thinking budget in tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u32>,
}

impl Default for MinimaxiThinkingModeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            thinking_budget: None,
        }
    }
}

/// MiniMaxi structured output configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MinimaxiResponseFormat {
    /// Plain JSON object output.
    JsonObject,
    /// JSON schema output.
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        strict: bool,
    },
}

/// MiniMaxi-specific chat options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MinimaxiOptions {
    /// Thinking / reasoning mode configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_mode: Option<MinimaxiThinkingModeConfig>,
    /// Structured output configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<MinimaxiResponseFormat>,
    /// Additional provider-specific parameters.
    #[serde(flatten)]
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl MinimaxiOptions {
    /// Create new MiniMaxi options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure thinking mode directly.
    pub fn with_thinking_mode(mut self, config: MinimaxiThinkingModeConfig) -> Self {
        self.thinking_mode = Some(config);
        self
    }

    /// Enable or disable reasoning output.
    pub fn with_reasoning_enabled(mut self, enabled: bool) -> Self {
        let mut config = self.thinking_mode.unwrap_or_default();
        config.enabled = enabled;
        self.thinking_mode = Some(config);
        self
    }

    /// Set a reasoning budget and implicitly enable reasoning output.
    pub fn with_reasoning_budget(mut self, budget: u32) -> Self {
        let mut config = self.thinking_mode.unwrap_or_default();
        config.enabled = true;
        config.thinking_budget = Some(budget);
        self.thinking_mode = Some(config);
        self
    }

    /// Configure structured output as a plain JSON object.
    pub fn with_json_object(mut self) -> Self {
        self.response_format = Some(MinimaxiResponseFormat::JsonObject);
        self
    }

    /// Configure structured output using a JSON schema.
    pub fn with_json_schema(
        mut self,
        name: impl Into<String>,
        schema: serde_json::Value,
        strict: bool,
    ) -> Self {
        self.response_format = Some(MinimaxiResponseFormat::JsonSchema {
            name: name.into(),
            schema,
            strict,
        });
        self
    }

    /// Add a custom MiniMaxi parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimaxi_options_reasoning_budget_implies_enabled() {
        let value = serde_json::to_value(MinimaxiOptions::new().with_reasoning_budget(2048))
            .expect("serialize options");

        assert_eq!(value["thinking_mode"]["enabled"], serde_json::json!(true));
        assert_eq!(
            value["thinking_mode"]["thinking_budget"],
            serde_json::json!(2048)
        );
    }

    #[test]
    fn minimaxi_options_json_schema_and_extra_params_serialize() {
        let value = serde_json::to_value(
            MinimaxiOptions::new()
                .with_json_schema(
                    "response",
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "answer": { "type": "string" }
                        },
                        "required": ["answer"]
                    }),
                    true,
                )
                .with_param("vendor_extra", serde_json::json!(true)),
        )
        .expect("serialize options");

        assert_eq!(
            value["response_format"]["JsonSchema"]["name"],
            serde_json::json!("response")
        );
        assert_eq!(value["vendor_extra"], serde_json::json!(true));
    }

    #[test]
    fn minimaxi_options_serialization_omits_unset_fields() {
        let value = serde_json::to_value(MinimaxiOptions::new()).expect("serialize options");

        assert_eq!(value, serde_json::json!({}));
    }
}
