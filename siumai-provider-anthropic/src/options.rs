use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_anthropic_compatible::MessagesCallOptions;
use siumai_core::{
    LanguageRequest, Message, MessageRole, ModelFamily, ProviderOptionError, ProviderOptions,
    TypedProviderOptions,
};
use siumai_protocol_anthropic::messages::{
    API_MODE_ID, CacheControl, CacheTtl, ContextManagement, InferenceGeo, InferenceSpeed,
    McpServer, MessagesContainer, MessagesMetadata, MessagesRequestOptions,
    MessagesServiceTierPreference, MessagesTokenCountOptions, OutputEffort, ServerFallbacks,
    ThinkingConfig, ThinkingDisplay, TokenTaskBudget, is_protected_option_field,
};

use crate::annotations::AnthropicCacheTtl;

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
    task_budget: Option<TokenTaskBudget>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fallbacks: Option<ServerFallbacks>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<MessagesServiceTierPreference>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<InferenceSpeed>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inference_geo: Option<InferenceGeo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    container: Option<MessagesContainer>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_management: Option<ContextManagement>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mcp_servers: Option<Vec<McpServer>>,
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

    pub const fn task_budget(&self) -> Option<TokenTaskBudget> {
        self.task_budget
    }

    pub fn fallbacks(&self) -> Option<&ServerFallbacks> {
        self.fallbacks.as_ref()
    }

    pub const fn top_k(&self) -> Option<u64> {
        self.top_k
    }

    pub const fn service_tier(&self) -> Option<MessagesServiceTierPreference> {
        self.service_tier
    }

    pub const fn automatic_cache_ttl(&self) -> Option<AnthropicCacheTtl> {
        match self.cache_control {
            Some(cache_control) => match cache_control.ttl() {
                CacheTtl::FiveMinutes => Some(AnthropicCacheTtl::FiveMinutes),
                CacheTtl::OneHour => Some(AnthropicCacheTtl::OneHour),
            },
            None => None,
        }
    }

    pub const fn speed(&self) -> Option<InferenceSpeed> {
        self.speed
    }

    pub const fn inference_geo(&self) -> Option<InferenceGeo> {
        self.inference_geo
    }

    pub fn container(&self) -> Option<&MessagesContainer> {
        self.container.as_ref()
    }

    pub fn context_management(&self) -> Option<&ContextManagement> {
        self.context_management.as_ref()
    }

    pub fn mcp_servers(&self) -> Option<&[McpServer]> {
        self.mcp_servers.as_deref()
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

    pub const fn with_task_budget(mut self, task_budget: TokenTaskBudget) -> Self {
        self.task_budget = Some(task_budget);
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

    pub const fn with_service_tier(mut self, service_tier: MessagesServiceTierPreference) -> Self {
        self.service_tier = Some(service_tier);
        self
    }

    /// Enable request-level automatic prompt caching.
    ///
    /// Explicit cache annotations remain available on messages, content nodes,
    /// and tools. Anthropic counts the automatic target against the same four-slot
    /// breakpoint budget unless the final cacheable block already has the same TTL.
    pub const fn with_automatic_cache(mut self, ttl: AnthropicCacheTtl) -> Self {
        let ttl = match ttl {
            AnthropicCacheTtl::FiveMinutes => CacheTtl::FiveMinutes,
            AnthropicCacheTtl::OneHour => CacheTtl::OneHour,
        };
        self.cache_control = Some(CacheControl::new(ttl));
        self
    }

    pub const fn with_speed(mut self, speed: InferenceSpeed) -> Self {
        self.speed = Some(speed);
        self
    }

    pub fn with_inference_geo(mut self, inference_geo: InferenceGeo) -> Self {
        self.inference_geo = Some(inference_geo);
        self
    }

    pub fn with_container(mut self, container: MessagesContainer) -> Self {
        self.container = Some(container);
        self
    }

    pub fn with_context_management(mut self, context_management: ContextManagement) -> Self {
        self.context_management = Some(context_management);
        self
    }

    pub fn with_mcp_servers(mut self, servers: impl IntoIterator<Item = McpServer>) -> Self {
        self.mcp_servers = Some(servers.into_iter().collect());
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
        if let Some(task_budget) = self.task_budget {
            options = options.with_task_budget(task_budget);
        }
        if let Some(fallbacks) = &self.fallbacks {
            options = options.with_fallbacks(fallbacks.clone());
        }
        if let Some(top_k) = self.top_k {
            options = options.with_top_k(top_k);
        }
        if let Some(service_tier) = self.service_tier {
            options = options.with_service_tier(service_tier);
        }
        if let Some(cache_control) = self.cache_control {
            options = options.with_cache_control(cache_control);
        }
        if let Some(speed) = self.speed {
            options = options.with_speed(speed);
        }
        if let Some(inference_geo) = self.inference_geo {
            options = options.with_inference_geo(inference_geo);
        }
        if let Some(container) = &self.container {
            options = options.with_container(container.clone());
        }
        if let Some(context_management) = &self.context_management {
            options = options.with_context_management(context_management.clone());
        }
        if let Some(mcp_servers) = &self.mcp_servers {
            options = options.with_mcp_servers(mcp_servers.clone());
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
        if let Some(task_budget) = self.task_budget {
            options = options.with_task_budget(task_budget);
        }
        if let Some(fallbacks) = &self.fallbacks {
            options = options.with_fallbacks(fallbacks.clone());
        }
        if let Some(top_k) = self.top_k {
            options = options.with_top_k(top_k);
        }
        if let Some(service_tier) = self.service_tier {
            options = options.with_service_tier(service_tier);
        }
        if let Some(cache_control) = self.cache_control {
            options = options.with_cache_control(cache_control);
        }
        if let Some(speed) = self.speed {
            options = options.with_speed(speed);
        }
        if let Some(inference_geo) = self.inference_geo {
            options = options.with_inference_geo(inference_geo);
        }
        if let Some(container) = &self.container {
            options = options.with_container(container.clone());
        }
        if let Some(context_management) = &self.context_management {
            options = options.with_context_management(context_management.clone());
        }
        if let Some(mcp_servers) = &self.mcp_servers {
            options = options.with_mcp_servers(mcp_servers.clone());
        }
        options
    }
}

/// Typed options accepted by Anthropic's token-count operation.
///
/// Generation controls and Messages-create-only fields are intentionally absent.
#[derive(Debug, Clone, Default)]
pub struct AnthropicTokenCountOptions {
    cache_control: Option<CacheControl>,
    thinking: Option<AnthropicThinking>,
    output_effort: Option<OutputEffort>,
    task_budget: Option<TokenTaskBudget>,
    speed: Option<InferenceSpeed>,
    context_management: Option<ContextManagement>,
    mcp_servers: Option<Vec<McpServer>>,
}

impl AnthropicTokenCountOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn automatic_cache_ttl(&self) -> Option<AnthropicCacheTtl> {
        match self.cache_control {
            Some(cache_control) => match cache_control.ttl() {
                CacheTtl::FiveMinutes => Some(AnthropicCacheTtl::FiveMinutes),
                CacheTtl::OneHour => Some(AnthropicCacheTtl::OneHour),
            },
            None => None,
        }
    }

    pub const fn thinking(&self) -> Option<AnthropicThinking> {
        self.thinking
    }

    pub const fn output_effort(&self) -> Option<OutputEffort> {
        self.output_effort
    }

    pub const fn task_budget(&self) -> Option<TokenTaskBudget> {
        self.task_budget
    }

    pub const fn speed(&self) -> Option<InferenceSpeed> {
        self.speed
    }

    pub fn context_management(&self) -> Option<&ContextManagement> {
        self.context_management.as_ref()
    }

    pub fn mcp_servers(&self) -> Option<&[McpServer]> {
        self.mcp_servers.as_deref()
    }

    pub const fn with_automatic_cache(mut self, ttl: AnthropicCacheTtl) -> Self {
        let ttl = match ttl {
            AnthropicCacheTtl::FiveMinutes => CacheTtl::FiveMinutes,
            AnthropicCacheTtl::OneHour => CacheTtl::OneHour,
        };
        self.cache_control = Some(CacheControl::new(ttl));
        self
    }

    pub const fn with_thinking(mut self, thinking: AnthropicThinking) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub const fn with_output_effort(mut self, output_effort: OutputEffort) -> Self {
        self.output_effort = Some(output_effort);
        self
    }

    pub const fn with_task_budget(mut self, task_budget: TokenTaskBudget) -> Self {
        self.task_budget = Some(task_budget);
        self
    }

    pub const fn with_speed(mut self, speed: InferenceSpeed) -> Self {
        self.speed = Some(speed);
        self
    }

    pub fn with_context_management(mut self, context_management: ContextManagement) -> Self {
        self.context_management = Some(context_management);
        self
    }

    pub fn with_mcp_servers(mut self, servers: impl IntoIterator<Item = McpServer>) -> Self {
        self.mcp_servers = Some(servers.into_iter().collect());
        self
    }

    pub(crate) fn to_protocol(&self) -> MessagesTokenCountOptions {
        let mut options = MessagesTokenCountOptions::new();
        if let Some(cache_control) = self.cache_control {
            options = options.with_cache_control(cache_control);
        }
        if let Some(thinking) = self.thinking {
            options = options.with_thinking(thinking.protocol());
        }
        if let Some(output_effort) = self.output_effort {
            options = options.with_output_effort(output_effort);
        }
        if let Some(task_budget) = self.task_budget {
            options = options.with_task_budget(task_budget);
        }
        if let Some(speed) = self.speed {
            options = options.with_speed(speed);
        }
        if let Some(context_management) = &self.context_management {
            options = options.with_context_management(context_management.clone());
        }
        if let Some(mcp_servers) = &self.mcp_servers {
            options = options.with_mcp_servers(mcp_servers.clone());
        }
        options
    }

    pub(crate) fn to_engine(&self) -> MessagesCallOptions {
        let mut options = MessagesCallOptions::new();
        if let Some(cache_control) = self.cache_control {
            options = options.with_cache_control(cache_control);
        }
        if let Some(thinking) = self.thinking {
            options = options.with_thinking(thinking.protocol());
        }
        if let Some(output_effort) = self.output_effort {
            options = options.with_output_effort(output_effort);
        }
        if let Some(task_budget) = self.task_budget {
            options = options.with_task_budget(task_budget);
        }
        if let Some(speed) = self.speed {
            options = options.with_speed(speed);
        }
        if let Some(context_management) = &self.context_management {
            options = options.with_context_management(context_management.clone());
        }
        if let Some(mcp_servers) = &self.mcp_servers {
            options = options.with_mcp_servers(mcp_servers.clone());
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
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    matches!(
        compact.as_str(),
        "apikey"
            | "xapikey"
            | "authorization"
            | "auth"
            | "token"
            | "bearer"
            | "credential"
            | "credentials"
            | "endpoint"
            | "baseurl"
            | "url"
            | "host"
            | "header"
            | "headers"
            | "anthropicversion"
            | "anthropicbeta"
            | "proxy"
            | "tls"
            | "audience"
    ) || compact.ends_with("apikey")
        || compact.ends_with("token")
        || compact.ends_with("credential")
        || compact.ends_with("credentials")
        || compact.ends_with("authorization")
        || compact.ends_with("endpoint")
        || compact.ends_with("baseurl")
        || compact.ends_with("headers")
}
