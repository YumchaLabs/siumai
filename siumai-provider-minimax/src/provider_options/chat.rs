//! `MiniMax` chat provider options.
//!
//! These typed option structs are owned by the MiniMax provider crate and are serialized into
//! `providerOptions["minimax"]`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// MiniMax thinking control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MinimaxThinking {
    /// Let the model decide how much thinking to use.
    Adaptive,
    /// Disable thinking where the selected model permits it.
    Disabled,
}

/// MiniMax request admission tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MinimaxServiceTier {
    Standard,
    Priority,
}

/// MiniMax-specific chat options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MinimaxOptions {
    /// Thinking / reasoning mode configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<MinimaxThinking>,
    /// Request admission tier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<MinimaxServiceTier>,
    /// Additional provider-specific parameters.
    #[serde(flatten)]
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl MinimaxOptions {
    /// Create new MiniMax options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure thinking mode directly.
    pub fn with_thinking(mut self, thinking: MinimaxThinking) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Select the provider request admission tier.
    pub fn with_service_tier(mut self, service_tier: MinimaxServiceTier) -> Self {
        self.service_tier = Some(service_tier);
        self
    }

    /// Add a custom MiniMax parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimax_options_serialize_official_thinking_and_tier_shapes() {
        let value = serde_json::to_value(
            MinimaxOptions::new()
                .with_thinking(MinimaxThinking::Adaptive)
                .with_service_tier(MinimaxServiceTier::Priority),
        )
        .expect("serialize options");

        assert_eq!(value["thinking"], serde_json::json!({ "type": "adaptive" }));
        assert_eq!(value["service_tier"], serde_json::json!("priority"));
    }

    #[test]
    fn minimax_options_extra_params_serialize() {
        let value = serde_json::to_value(
            MinimaxOptions::new().with_param("vendor_extra", serde_json::json!(true)),
        )
        .expect("serialize options");

        assert_eq!(value["vendor_extra"], serde_json::json!(true));
    }

    #[test]
    fn minimax_options_serialization_omits_unset_fields() {
        let value = serde_json::to_value(MinimaxOptions::new()).expect("serialize options");

        assert_eq!(value, serde_json::json!({}));
    }
}
