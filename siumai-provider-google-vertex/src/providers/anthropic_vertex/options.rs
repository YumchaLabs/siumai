use serde::{Deserialize, Serialize};
use siumai_anthropic_compatible::MessagesCallOptions;
use siumai_core::{
    LanguageRequest, Message, MessageRole, ModelFamily, ProviderOptionError, ProviderOptions,
    TypedProviderOptions,
};
use siumai_protocol_anthropic::messages::{
    API_MODE_ID, MessagesMetadata, MessagesRequestOptions, OutputEffort, ThinkingConfig,
};

/// Typed request options for Anthropic Messages calls executed by Vertex AI.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GoogleVertexAnthropicMessagesOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<MessagesMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_effort: Option<OutputEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u64>,
}

impl GoogleVertexAnthropicMessagesOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn metadata(&self) -> Option<&MessagesMetadata> {
        self.metadata.as_ref()
    }

    pub const fn thinking(&self) -> Option<ThinkingConfig> {
        self.thinking
    }

    pub const fn output_effort(&self) -> Option<OutputEffort> {
        self.output_effort
    }

    pub const fn top_k(&self) -> Option<u64> {
        self.top_k
    }

    pub fn with_metadata(mut self, metadata: MessagesMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub const fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub const fn without_thinking(mut self) -> Self {
        self.thinking = Some(ThinkingConfig::Disabled);
        self
    }

    pub const fn with_adaptive_thinking(mut self) -> Self {
        self.thinking = Some(ThinkingConfig::adaptive());
        self
    }

    pub const fn with_enabled_thinking(mut self, budget_tokens: u64) -> Self {
        self.thinking = Some(ThinkingConfig::enabled(budget_tokens));
        self
    }

    pub const fn with_output_effort(mut self, effort: OutputEffort) -> Self {
        self.output_effort = Some(effort);
        self
    }

    pub const fn with_top_k(mut self, top_k: u64) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Erase these options into one call-scoped `google` layer.
    pub fn provider_options(&self) -> Result<ProviderOptions, ProviderOptionError> {
        ProviderOptions::typed(self)
    }

    pub(crate) fn to_engine(&self) -> MessagesCallOptions {
        let mut options = MessagesCallOptions::new();
        if let Some(metadata) = &self.metadata {
            options = options.with_metadata(metadata.clone());
        }
        if let Some(thinking) = self.thinking {
            options = options.with_thinking(thinking);
        }
        if let Some(effort) = self.output_effort {
            options = options.with_output_effort(effort);
        }
        if let Some(top_k) = self.top_k {
            options = options.with_top_k(top_k);
        }
        options
    }

    fn to_protocol(&self) -> MessagesRequestOptions {
        let mut options = MessagesRequestOptions::new(false);
        if let Some(metadata) = &self.metadata {
            options = options.with_metadata(metadata.clone());
        }
        if let Some(thinking) = self.thinking {
            options = options.with_thinking(thinking);
        }
        if let Some(effort) = self.output_effort {
            options = options.with_output_effort(effort);
        }
        if let Some(top_k) = self.top_k {
            options = options.with_top_k(top_k);
        }
        options
    }
}

impl TypedProviderOptions for GoogleVertexAnthropicMessagesOptions {
    const NAMESPACE: &'static str = "google";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        let mut request =
            LanguageRequest::new(vec![Message::text(MessageRole::User, "option validation")]);
        request.generation.max_output_tokens = Some(u64::MAX);
        self.to_protocol()
            .validate(&request)
            .map_err(|error| ProviderOptionError::Rejected {
                path: "messages".to_string(),
                reason: error.to_string(),
            })
    }
}
