//! Moonshot AI provider options for the Kimi Chat Completions surface.
//!
//! These typed option structs are owned by this branded provider crate and serialize into the
//! `providerOptions["moonshotai"]` namespace.

use serde::{Deserialize, Serialize};
use siumai_core::{ModelFamily, ProviderOptionError, TypedProviderOptions};
use siumai_protocol_openai::chat_completions::API_MODE_ID;

/// Kimi K3 reasoning effort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KimiReasoningEffort {
    Low,
    High,
    Max,
}

/// Thinking mode supported by current Kimi K2.x models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KimiThinkingMode {
    Enabled,
    Disabled,
}

/// Historical reasoning retention supported by current Kimi K2.x models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KimiThinkingRetention {
    All,
}

/// Current Kimi K2.x thinking configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum KimiThinking {
    Enabled {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        keep: Option<KimiThinkingRetention>,
    },
    Disabled {},
}

impl KimiThinking {
    pub const fn enabled() -> Self {
        Self::Enabled { keep: None }
    }

    pub const fn enabled_with_preserved_history() -> Self {
        Self::Enabled {
            keep: Some(KimiThinkingRetention::All),
        }
    }

    pub const fn disabled() -> Self {
        Self::Disabled {}
    }

    pub const fn mode(&self) -> KimiThinkingMode {
        match self {
            Self::Enabled { .. } => KimiThinkingMode::Enabled,
            Self::Disabled {} => KimiThinkingMode::Disabled,
        }
    }

    pub const fn retention(&self) -> Option<KimiThinkingRetention> {
        match self {
            Self::Enabled { keep } => *keep,
            Self::Disabled {} => None,
        }
    }
}

/// Typed options for the current Kimi Chat Completions profile.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiLanguageOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<KimiReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<KimiThinking>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
}

impl KimiLanguageOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_reasoning_effort(mut self, effort: KimiReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    pub fn with_thinking(mut self, thinking: KimiThinking) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub fn with_prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.prompt_cache_key = Some(key.into());
        self
    }

    pub fn with_safety_identifier(mut self, identifier: impl Into<String>) -> Self {
        self.safety_identifier = Some(identifier.into());
        self
    }
}

impl TypedProviderOptions for KimiLanguageOptions {
    const NAMESPACE: &'static str = "moonshotai";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        for (path, value) in [
            ("prompt_cache_key", self.prompt_cache_key.as_deref()),
            ("safety_identifier", self.safety_identifier.as_deref()),
        ] {
            if value.is_some_and(|value| value.trim().is_empty()) {
                return Err(ProviderOptionError::Rejected {
                    path: path.to_string(),
                    reason: "must not be empty".to_string(),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::ProviderOptions;

    #[test]
    fn current_kimi_options_use_the_moonshotai_namespace_and_wire_names() {
        let options = ProviderOptions::typed(
            &KimiLanguageOptions::new()
                .with_reasoning_effort(KimiReasoningEffort::High)
                .with_thinking(KimiThinking::enabled_with_preserved_history())
                .with_prompt_cache_key("session-42")
                .with_safety_identifier("user-hash"),
        )
        .unwrap();

        assert_eq!(options.namespace().as_str(), "moonshotai");
        assert_eq!(
            options.value(),
            &serde_json::json!({
                "reasoning_effort": "high",
                "thinking": {"type": "enabled", "keep": "all"},
                "prompt_cache_key": "session-42",
                "safety_identifier": "user-hash"
            })
            .as_object()
            .unwrap()
            .clone()
        );
    }

    #[test]
    fn current_kimi_options_reject_empty_cache_and_safety_ids() {
        assert!(
            ProviderOptions::typed(&KimiLanguageOptions::new().with_prompt_cache_key("  "))
                .is_err()
        );
        assert!(
            ProviderOptions::typed(&KimiLanguageOptions::new().with_safety_identifier("")).is_err()
        );
        assert!(
            serde_json::from_value::<KimiLanguageOptions>(serde_json::json!({
                "thinking": {"type": "disabled", "keep": "all"}
            }))
            .is_err()
        );
    }
}
