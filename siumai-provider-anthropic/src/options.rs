use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_anthropic_compatible::MessagesCallOptions;
use siumai_core::{
    LanguageRequest, Message, MessageRole, ModelFamily, ProviderOptionError, ProviderOptions,
    TypedProviderOptions,
};
use siumai_protocol_anthropic::messages::{
    API_MODE_ID, MessagesMetadata, MessagesRequestOptions, OutputEffort, ServerFallbacks,
    ThinkingConfig, ThinkingDisplay, is_protected_option_field,
};

/// Anthropic extended-thinking policy for one Messages call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[non_exhaustive]
pub enum AnthropicThinking {
    Disabled,
    Enabled {
        budget_tokens: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<ThinkingDisplay>,
    },
    Adaptive {
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<ThinkingDisplay>,
    },
}

impl AnthropicThinking {
    pub const fn enabled(budget_tokens: u64) -> Self {
        Self::Enabled {
            budget_tokens,
            display: None,
        }
    }

    pub const fn adaptive() -> Self {
        Self::Adaptive { display: None }
    }

    pub const fn with_display(self, display: ThinkingDisplay) -> Self {
        match self {
            Self::Disabled => Self::Disabled,
            Self::Enabled { budget_tokens, .. } => Self::Enabled {
                budget_tokens,
                display: Some(display),
            },
            Self::Adaptive { .. } => Self::Adaptive {
                display: Some(display),
            },
        }
    }

    const fn protocol(self) -> ThinkingConfig {
        match self {
            Self::Disabled => ThinkingConfig::Disabled,
            Self::Enabled {
                budget_tokens,
                display,
            } => ThinkingConfig::Enabled {
                budget_tokens,
                display,
            },
            Self::Adaptive { display } => ThinkingConfig::Adaptive { display },
        }
    }
}

/// Typed call options for Anthropic's Messages API.
///
/// Unknown, non-protected request fields can be carried in `extra`. The configured
/// engine validates every erased layer again before encoding, so checked raw and
/// typed values share the same protected-field policy.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "snake_case")]
pub struct AnthropicMessagesOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<MessagesMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_effort: Option<OutputEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fallbacks: Option<ServerFallbacks>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    extra: BTreeMap<String, Value>,
}

impl AnthropicMessagesOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn metadata(&self) -> Option<&MessagesMetadata> {
        self.metadata.as_ref()
    }

    pub const fn thinking(&self) -> Option<AnthropicThinking> {
        self.thinking
    }

    pub const fn output_effort(&self) -> Option<OutputEffort> {
        self.output_effort
    }

    pub fn fallbacks(&self) -> Option<&ServerFallbacks> {
        self.fallbacks.as_ref()
    }

    pub const fn top_k(&self) -> Option<u64> {
        self.top_k
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }

    pub fn with_metadata(mut self, metadata: MessagesMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub const fn with_thinking(mut self, thinking: AnthropicThinking) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub const fn without_thinking(mut self) -> Self {
        self.thinking = Some(AnthropicThinking::Disabled);
        self
    }

    pub const fn with_adaptive_thinking(mut self) -> Self {
        self.thinking = Some(AnthropicThinking::adaptive());
        self
    }

    pub const fn with_enabled_thinking(mut self, budget_tokens: u64) -> Self {
        self.thinking = Some(AnthropicThinking::enabled(budget_tokens));
        self
    }

    pub const fn with_output_effort(mut self, effort: OutputEffort) -> Self {
        self.output_effort = Some(effort);
        self
    }

    pub fn with_fallbacks(mut self, fallbacks: ServerFallbacks) -> Self {
        self.fallbacks = Some(fallbacks);
        self
    }

    pub const fn with_top_k(mut self, top_k: u64) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn try_with_extra(
        mut self,
        name: impl Into<String>,
        value: Value,
    ) -> Result<Self, ProviderOptionError> {
        let name = name.into();
        if name.is_empty() || is_protected_option_field(&name) || is_security_sensitive(&name) {
            return Err(ProviderOptionError::Rejected {
                path: name,
                reason: "field is owned by the provider, protocol codec, or transport".to_string(),
            });
        }
        self.extra.insert(name, value);
        Ok(self)
    }

    /// Erase these options into one call-scoped `anthropic` layer.
    pub fn provider_options(&self) -> Result<ProviderOptions, ProviderOptionError> {
        ProviderOptions::typed(self)
    }

    pub(crate) fn to_engine(&self) -> MessagesCallOptions {
        let mut options = MessagesCallOptions::new().with_extra(self.extra.clone());
        if let Some(metadata) = &self.metadata {
            options = options.with_metadata(metadata.clone());
        }
        if let Some(thinking) = self.thinking {
            options = options.with_thinking(thinking.protocol());
        }
        if let Some(effort) = self.output_effort {
            options = options.with_output_effort(effort);
        }
        if let Some(fallbacks) = &self.fallbacks {
            options = options.with_fallbacks(fallbacks.clone());
        }
        if let Some(top_k) = self.top_k {
            options = options.with_top_k(top_k);
        }
        options
    }

    pub(crate) fn to_protocol(&self, stream: bool) -> MessagesRequestOptions {
        let mut options = MessagesRequestOptions::new(stream).with_extra(self.extra.clone());
        if let Some(metadata) = &self.metadata {
            options = options.with_metadata(metadata.clone());
        }
        if let Some(thinking) = self.thinking {
            options = options.with_thinking(thinking.protocol());
        }
        if let Some(effort) = self.output_effort {
            options = options.with_output_effort(effort);
        }
        if let Some(fallbacks) = &self.fallbacks {
            options = options.with_fallbacks(fallbacks.clone());
        }
        if let Some(top_k) = self.top_k {
            options = options.with_top_k(top_k);
        }
        options
    }
}

impl TypedProviderOptions for AnthropicMessagesOptions {
    const NAMESPACE: &'static str = "anthropic";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if let Some(AnthropicThinking::Enabled { budget_tokens, .. }) = self.thinking
            && budget_tokens < 1_024
        {
            return Err(ProviderOptionError::Rejected {
                path: "thinking.budget_tokens".to_string(),
                reason: "must be at least 1024".to_string(),
            });
        }
        for name in self.extra.keys() {
            if name.is_empty() || is_protected_option_field(name) || is_security_sensitive(name) {
                return Err(ProviderOptionError::Rejected {
                    path: format!("extra.{name}"),
                    reason: "field is owned by the provider, protocol codec, or transport"
                        .to_string(),
                });
            }
        }
        let mut request =
            LanguageRequest::new(vec![Message::text(MessageRole::User, "option validation")]);
        request.generation.max_output_tokens = Some(u64::MAX);
        self.to_protocol(false)
            .validate(&request)
            .map_err(|error| ProviderOptionError::Rejected {
                path: "messages".to_string(),
                reason: error.to_string(),
            })?;
        Ok(())
    }
}

fn is_security_sensitive(name: &str) -> bool {
    let compact = name
        .bytes()
        .filter(|byte| byte.is_ascii_alphanumeric())
        .map(|byte| byte.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let compact = String::from_utf8_lossy(&compact);
    compact.contains("apikey")
        || compact.contains("authorization")
        || compact.contains("credential")
        || compact.contains("endpoint")
        || compact.contains("baseurl")
        || compact.contains("anthropicversion")
        || compact.contains("header")
}
