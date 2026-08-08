use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_anthropic_compatible::MessagesCallOptions;
use siumai_core::{ModelFamily, ProviderOptionError, ProviderOptions, TypedProviderOptions};
use siumai_protocol_anthropic::messages::ThinkingConfig;

const PROVIDER_NAMESPACE: &str = "minimax";
const MESSAGES_API_MODE: &str = "messages";
const CHAT_API_MODE: &str = "chat-completions";
const RESPONSES_API_MODE: &str = "responses";
const MAX_METADATA_ENTRIES: usize = 64;
const MAX_METADATA_KEY_BYTES: usize = 256;
const MAX_METADATA_VALUE_BYTES: usize = 4 * 1024;
const MAX_PROMPT_CACHE_KEY_BYTES: usize = 256;
pub(crate) const MINIMAX_MESSAGES_SERVICE_TIER_OPTION: &str = "minimax_service_tier";

/// Thinking controls supported by the hosted MiniMax M3 APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[non_exhaustive]
pub enum MinimaxThinking {
    Adaptive,
    Disabled,
}

/// Service tiers supported by MiniMax language endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum MinimaxServiceTier {
    Standard,
    Priority,
}

/// Responses reasoning effort supported by MiniMax.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum MinimaxReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MinimaxResponsesReasoning {
    effort: MinimaxReasoningEffort,
}

impl MinimaxResponsesReasoning {
    pub const fn new(effort: MinimaxReasoningEffort) -> Self {
        Self { effort }
    }

    pub const fn effort(self) -> MinimaxReasoningEffort {
        self.effort
    }
}

/// Typed options for the recommended MiniMax Anthropic Messages mode.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MinimaxMessagesOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<MinimaxThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "minimax_service_tier")]
    service_tier: Option<MinimaxServiceTier>,
}

impl MinimaxMessagesOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_thinking(mut self, thinking: MinimaxThinking) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub const fn with_service_tier(mut self, tier: MinimaxServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    pub fn provider_options(&self) -> Result<ProviderOptions, ProviderOptionError> {
        ProviderOptions::typed(self)
    }

    pub(crate) fn to_engine(&self) -> MessagesCallOptions {
        let mut options = MessagesCallOptions::new();
        if let Some(thinking) = self.thinking {
            options = options.with_thinking(match thinking {
                MinimaxThinking::Adaptive => ThinkingConfig::adaptive(),
                MinimaxThinking::Disabled => ThinkingConfig::Disabled,
            });
        }
        if let Some(tier) = self.service_tier {
            options = options.with_extra(BTreeMap::from([(
                MINIMAX_MESSAGES_SERVICE_TIER_OPTION.to_string(),
                Value::String(
                    match tier {
                        MinimaxServiceTier::Standard => "standard",
                        MinimaxServiceTier::Priority => "priority",
                    }
                    .to_string(),
                ),
            )]));
        }
        options
    }
}

impl TypedProviderOptions for MinimaxMessagesOptions {
    const NAMESPACE: &'static str = PROVIDER_NAMESPACE;
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(MESSAGES_API_MODE);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        Ok(())
    }
}

/// Typed options for MiniMax OpenAI Chat Completions compatibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MinimaxChatCompletionsOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<MinimaxThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<MinimaxServiceTier>,
}

impl MinimaxChatCompletionsOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_thinking(mut self, thinking: MinimaxThinking) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub const fn with_service_tier(mut self, tier: MinimaxServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    pub fn provider_options(&self) -> Result<ProviderOptions, ProviderOptionError> {
        ProviderOptions::typed(self)
    }
}

impl TypedProviderOptions for MinimaxChatCompletionsOptions {
    const NAMESPACE: &'static str = PROVIDER_NAMESPACE;
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(CHAT_API_MODE);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        Ok(())
    }
}

/// Typed options for the bounded MiniMax Responses-compatible subset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MinimaxResponsesOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<MinimaxResponsesReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<MinimaxServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    metadata: BTreeMap<String, String>,
}

impl MinimaxResponsesOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_reasoning_effort(mut self, effort: MinimaxReasoningEffort) -> Self {
        self.reasoning = Some(MinimaxResponsesReasoning::new(effort));
        self
    }

    pub const fn with_service_tier(mut self, tier: MinimaxServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    pub fn with_prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.prompt_cache_key = Some(key.into());
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn provider_options(&self) -> Result<ProviderOptions, ProviderOptionError> {
        ProviderOptions::typed(self)
    }
}

impl TypedProviderOptions for MinimaxResponsesOptions {
    const NAMESPACE: &'static str = PROVIDER_NAMESPACE;
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(RESPONSES_API_MODE);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if self.prompt_cache_key.as_ref().is_some_and(|key| {
            key.is_empty()
                || key.len() > MAX_PROMPT_CACHE_KEY_BYTES
                || key.chars().any(char::is_control)
        }) {
            return Err(rejected(
                "prompt_cache_key",
                "must be non-empty, bounded, and contain no control characters",
            ));
        }
        if self.metadata.len() > MAX_METADATA_ENTRIES {
            return Err(rejected("metadata", "contains too many entries"));
        }
        for (key, value) in &self.metadata {
            if key.is_empty()
                || key.len() > MAX_METADATA_KEY_BYTES
                || key.chars().any(char::is_control)
            {
                return Err(rejected("metadata", "contains an invalid key"));
            }
            if value.len() > MAX_METADATA_VALUE_BYTES || value.chars().any(char::is_control) {
                return Err(rejected("metadata", "contains an invalid value"));
            }
        }
        Ok(())
    }
}

fn rejected(path: &str, reason: &str) -> ProviderOptionError {
    ProviderOptionError::Rejected {
        path: path.to_string(),
        reason: reason.to_string(),
    }
}
