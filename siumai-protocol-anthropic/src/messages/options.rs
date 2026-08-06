use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{LanguageRequest, ModelId, StructuredOutputSpec};

use super::MessagesCodecError;
use super::annotations::{CacheControl, ToolNodeOptions};
use super::request::is_protected_option_field;

const MAX_EXTRA_DEPTH: usize = 16;
const MAX_EXTRA_FIELDS: usize = 1_024;
const MAX_EXTRA_BYTES: usize = 256 * 1024;
const MAX_FALLBACKS: usize = 3;
const MAX_TOOL_LIST: usize = 256;
const MAX_DOMAIN_LIST: usize = 256;
const MAX_DOMAIN_BYTES: usize = 253;
const MAX_TOOL_CONFIGS: usize = 256;
const MAX_TOOL_NAME_BYTES: usize = 128;
const MAX_ALLOWED_CALLERS: usize = 4;

/// Anthropic extended-thinking display mode for one Messages request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ThinkingDisplay {
    /// Return summarized thinking blocks in the response.
    Summarized,
    /// Omit thinking text while retaining a replayable signature.
    Omitted,
}

impl ThinkingDisplay {
    pub(crate) const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Summarized => "summarized",
            Self::Omitted => "omitted",
        }
    }
}

/// Anthropic extended-thinking mode for one Messages request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[non_exhaustive]
pub enum ThinkingConfig {
    Disabled,
    Enabled {
        budget_tokens: u64,
        display: Option<ThinkingDisplay>,
    },
    Adaptive {
        display: Option<ThinkingDisplay>,
    },
}

impl ThinkingConfig {
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

    pub const fn display(self) -> Option<ThinkingDisplay> {
        match self {
            Self::Disabled => None,
            Self::Enabled { display, .. } | Self::Adaptive { display } => display,
        }
    }

    pub const fn budget_tokens(self) -> Option<u64> {
        match self {
            Self::Enabled { budget_tokens, .. } => Some(budget_tokens),
            Self::Disabled | Self::Adaptive { .. } => None,
        }
    }
}

/// Effort level accepted by Anthropic's `output_config.effort` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum OutputEffort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl OutputEffort {
    pub(crate) const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

/// Provider-neutral wire value for the Messages `service_tier` field.
///
/// Compatible providers decide which values they support in their request
/// policy. The protocol codec only owns exact, typed serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MessagesServiceTier {
    Standard,
    Priority,
    Auto,
    StandardOnly,
}

impl MessagesServiceTier {
    pub(crate) const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Priority => "priority",
            Self::Auto => "auto",
            Self::StandardOnly => "standard_only",
        }
    }
}

/// Inference speed override for one server-side fallback attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum InferenceSpeed {
    Standard,
    Fast,
}

impl InferenceSpeed {
    pub(crate) const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Fast => "fast",
        }
    }
}

/// Typed `output_config` override for one fallback attempt.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FallbackOutputConfig {
    effort: Option<OutputEffort>,
    format: Option<StructuredOutputSpec>,
}

impl FallbackOutputConfig {
    pub const fn new() -> Self {
        Self {
            effort: None,
            format: None,
        }
    }

    pub const fn with_effort(mut self, effort: OutputEffort) -> Self {
        self.effort = Some(effort);
        self
    }

    pub fn with_format(mut self, format: StructuredOutputSpec) -> Self {
        self.format = Some(format);
        self
    }

    pub const fn effort(&self) -> Option<OutputEffort> {
        self.effort
    }

    pub const fn format(&self) -> Option<&StructuredOutputSpec> {
        self.format.as_ref()
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        if let Some(format) = &self.format {
            validate_structured_output(format)?;
        }
        Ok(())
    }
}

/// One explicit server-side fallback attempt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerFallback {
    model: String,
    max_tokens: Option<u64>,
    thinking: Option<ThinkingConfig>,
    output_config: Option<FallbackOutputConfig>,
    speed: Option<InferenceSpeed>,
}

impl ServerFallback {
    pub fn new(model: impl Into<String>) -> Result<Self, MessagesCodecError> {
        let value = Self {
            model: model.into(),
            max_tokens: None,
            thinking: None,
            output_config: None,
            speed: None,
        };
        value.validate(None)?;
        Ok(value)
    }

    pub fn model(&self) -> Result<ModelId, MessagesCodecError> {
        ModelId::new(self.model.clone()).map_err(|_| MessagesCodecError::InvalidOption {
            field: "fallbacks[].model",
            reason: "must be a valid model identifier",
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    pub const fn max_tokens(&self) -> Option<u64> {
        self.max_tokens
    }

    pub const fn thinking(&self) -> Option<ThinkingConfig> {
        self.thinking
    }

    pub const fn output_config(&self) -> Option<&FallbackOutputConfig> {
        self.output_config.as_ref()
    }

    pub const fn speed(&self) -> Option<InferenceSpeed> {
        self.speed
    }

    pub const fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub const fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub fn with_output_config(mut self, output_config: FallbackOutputConfig) -> Self {
        self.output_config = Some(output_config);
        self
    }

    pub const fn with_speed(mut self, speed: InferenceSpeed) -> Self {
        self.speed = Some(speed);
        self
    }

    pub(crate) fn validate(
        &self,
        request_max_tokens: Option<u64>,
    ) -> Result<(), MessagesCodecError> {
        self.model()?;
        if self.max_tokens == Some(0) {
            return Err(MessagesCodecError::InvalidOption {
                field: "fallbacks[].max_tokens",
                reason: "must be greater than zero",
            });
        }
        if let Some(output_config) = &self.output_config {
            output_config.validate()?;
        }
        let effective_max_tokens = self.max_tokens.or(request_max_tokens);
        validate_thinking(self.thinking, effective_max_tokens, "fallbacks[].thinking")?;
        Ok(())
    }
}

/// Typed server-side fallback policy.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ServerFallbacks {
    /// Ask Anthropic to choose the fallback chain for the requested model.
    Default,
    /// Try explicit fallback attempts in order.
    Explicit(Vec<ServerFallback>),
}

impl Serialize for ServerFallbacks {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Default => serializer.serialize_str("default"),
            Self::Explicit(fallbacks) => fallbacks.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ServerFallbacks {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(value) if value.eq_ignore_ascii_case("default") => Ok(Self::Default),
            Value::Array(_) => {
                let fallbacks = serde_json::from_value::<Vec<ServerFallback>>(value)
                    .map_err(serde::de::Error::custom)?;
                Ok(Self::Explicit(fallbacks))
            }
            _ => Err(serde::de::Error::custom(
                "fallbacks must be \"default\" or an array of fallback objects",
            )),
        }
    }
}

impl ServerFallbacks {
    pub fn explicit(fallbacks: impl Into<Vec<ServerFallback>>) -> Result<Self, MessagesCodecError> {
        let value = Self::Explicit(fallbacks.into());
        value.validate(None)?;
        Ok(value)
    }

    pub(crate) fn validate(
        &self,
        request_max_tokens: Option<u64>,
    ) -> Result<(), MessagesCodecError> {
        let Self::Explicit(fallbacks) = self else {
            return Ok(());
        };
        if fallbacks.is_empty() {
            return Err(MessagesCodecError::InvalidOption {
                field: "fallbacks",
                reason: "explicit fallback chain must not be empty",
            });
        }
        if fallbacks.len() > MAX_FALLBACKS {
            return Err(MessagesCodecError::InvalidOption {
                field: "fallbacks",
                reason: "explicit fallback chain exceeds three entries",
            });
        }
        let mut models = std::collections::BTreeSet::new();
        for fallback in fallbacks {
            if !models.insert(fallback.model_name()) {
                return Err(MessagesCodecError::InvalidOption {
                    field: "fallbacks",
                    reason: "explicit fallback chain must not repeat a model",
                });
            }
            fallback.validate(request_max_tokens)?;
        }
        Ok(())
    }

    pub(crate) fn validate_primary_model(
        &self,
        primary_model: &ModelId,
    ) -> Result<(), MessagesCodecError> {
        if let Self::Explicit(fallbacks) = self
            && fallbacks
                .iter()
                .any(|fallback| fallback.model_name() == primary_model.as_str())
        {
            return Err(MessagesCodecError::InvalidOption {
                field: "fallbacks",
                reason: "fallback models must differ from the requested model",
            });
        }
        Ok(())
    }
}

/// A caller that is allowed to invoke an Anthropic-defined tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolCaller {
    Direct,
    CodeExecution20250825,
    CodeExecution20260120,
    CodeExecution20260521,
}

impl ToolCaller {
    pub(crate) const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::CodeExecution20250825 => "code_execution_20250825",
            Self::CodeExecution20260120 => "code_execution_20260120",
            Self::CodeExecution20260521 => "code_execution_20260521",
        }
    }
}

/// User location hint for the Anthropic web-search tool.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UserLocation {
    city: Option<String>,
    country: Option<String>,
    region: Option<String>,
    timezone: Option<String>,
}

impl UserLocation {
    pub fn new() -> Self {
        Self {
            city: None,
            country: None,
            region: None,
            timezone: None,
        }
    }

    pub fn with_city(mut self, city: impl Into<String>) -> Self {
        self.city = Some(city.into());
        self
    }

    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    pub fn with_timezone(mut self, timezone: impl Into<String>) -> Self {
        self.timezone = Some(timezone.into());
        self
    }

    pub fn city(&self) -> Option<&str> {
        self.city.as_deref()
    }

    pub fn country(&self) -> Option<&str> {
        self.country.as_deref()
    }

    pub fn region(&self) -> Option<&str> {
        self.region.as_deref()
    }

    pub fn timezone(&self) -> Option<&str> {
        self.timezone.as_deref()
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        for value in [
            self.city.as_deref(),
            self.country.as_deref(),
            self.region.as_deref(),
            self.timezone.as_deref(),
        ] {
            if let Some(value) = value
                && (value.trim().is_empty()
                    || value.len() > 128
                    || value.chars().any(char::is_control))
            {
                return Err(MessagesCodecError::InvalidOption {
                    field: "tools.web_search.user_location",
                    reason: "location fields must be 1..=128 bytes and contain no control characters",
                });
            }
        }
        if let Some(country) = &self.country
            && (country.len() != 2 || !country.bytes().all(|byte| byte.is_ascii_alphabetic()))
        {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.web_search.user_location.country",
                reason: "country must be a two-letter ISO code",
            });
        }
        Ok(())
    }
}

/// Response inclusion policy for Anthropic web tools nested in code execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ResponseInclusion {
    #[default]
    Full,
    Excluded,
}

impl ResponseInclusion {
    pub(crate) const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Excluded => "excluded",
        }
    }
}

/// Typed web-search tool settings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebSearchToolOptions {
    allowed_domains: Option<Vec<String>>,
    blocked_domains: Option<Vec<String>>,
    max_uses: Option<u32>,
    response_inclusion: Option<ResponseInclusion>,
    user_location: Option<UserLocation>,
}

impl WebSearchToolOptions {
    pub fn with_allowed_domains(
        mut self,
        domains: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.allowed_domains = Some(domains.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_blocked_domains(
        mut self,
        domains: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.blocked_domains = Some(domains.into_iter().map(Into::into).collect());
        self
    }

    pub const fn with_max_uses(mut self, max_uses: u32) -> Self {
        self.max_uses = Some(max_uses);
        self
    }

    pub const fn with_response_inclusion(mut self, response_inclusion: ResponseInclusion) -> Self {
        self.response_inclusion = Some(response_inclusion);
        self
    }

    pub fn with_user_location(mut self, user_location: UserLocation) -> Self {
        self.user_location = Some(user_location);
        self
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        validate_domains(&self.allowed_domains, "tools.web_search.allowed_domains")?;
        validate_domains(&self.blocked_domains, "tools.web_search.blocked_domains")?;
        if self.allowed_domains.is_some() && self.blocked_domains.is_some() {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.web_search",
                reason: "allowed_domains and blocked_domains are mutually exclusive",
            });
        }
        if self.max_uses == Some(0) {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.web_search.max_uses",
                reason: "must be greater than zero",
            });
        }
        if let Some(user_location) = &self.user_location {
            user_location.validate()?;
        }
        Ok(())
    }

    pub fn allowed_domains(&self) -> Option<&[String]> {
        self.allowed_domains.as_deref()
    }

    pub fn blocked_domains(&self) -> Option<&[String]> {
        self.blocked_domains.as_deref()
    }

    pub const fn max_uses(&self) -> Option<u32> {
        self.max_uses
    }

    pub const fn response_inclusion(&self) -> Option<ResponseInclusion> {
        self.response_inclusion
    }

    pub fn user_location(&self) -> Option<&UserLocation> {
        self.user_location.as_ref()
    }
}

/// Typed web-fetch tool settings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebFetchToolOptions {
    allowed_domains: Option<Vec<String>>,
    blocked_domains: Option<Vec<String>>,
    citations: Option<bool>,
    max_content_tokens: Option<u32>,
    max_uses: Option<u32>,
    response_inclusion: Option<ResponseInclusion>,
    use_cache: Option<bool>,
}

impl WebFetchToolOptions {
    pub fn with_allowed_domains(
        mut self,
        domains: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.allowed_domains = Some(domains.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_blocked_domains(
        mut self,
        domains: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.blocked_domains = Some(domains.into_iter().map(Into::into).collect());
        self
    }

    pub const fn with_citations(mut self, enabled: bool) -> Self {
        self.citations = Some(enabled);
        self
    }

    pub const fn with_max_content_tokens(mut self, max_content_tokens: u32) -> Self {
        self.max_content_tokens = Some(max_content_tokens);
        self
    }

    pub const fn with_max_uses(mut self, max_uses: u32) -> Self {
        self.max_uses = Some(max_uses);
        self
    }

    pub const fn with_response_inclusion(mut self, response_inclusion: ResponseInclusion) -> Self {
        self.response_inclusion = Some(response_inclusion);
        self
    }

    pub const fn with_use_cache(mut self, use_cache: bool) -> Self {
        self.use_cache = Some(use_cache);
        self
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        validate_domains(&self.allowed_domains, "tools.web_fetch.allowed_domains")?;
        validate_domains(&self.blocked_domains, "tools.web_fetch.blocked_domains")?;
        if self.allowed_domains.is_some() && self.blocked_domains.is_some() {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.web_fetch",
                reason: "allowed_domains and blocked_domains are mutually exclusive",
            });
        }
        if self.max_content_tokens == Some(0) || self.max_uses == Some(0) {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.web_fetch",
                reason: "token and use limits must be greater than zero",
            });
        }
        Ok(())
    }

    pub fn allowed_domains(&self) -> Option<&[String]> {
        self.allowed_domains.as_deref()
    }

    pub fn blocked_domains(&self) -> Option<&[String]> {
        self.blocked_domains.as_deref()
    }

    pub const fn citations(&self) -> Option<bool> {
        self.citations
    }

    pub const fn max_content_tokens(&self) -> Option<u32> {
        self.max_content_tokens
    }

    pub const fn max_uses(&self) -> Option<u32> {
        self.max_uses
    }

    pub const fn response_inclusion(&self) -> Option<ResponseInclusion> {
        self.response_inclusion
    }

    pub const fn use_cache(&self) -> Option<bool> {
        self.use_cache
    }
}

/// Typed advisor tool settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdvisorToolOptions {
    model: String,
    max_tokens: Option<u64>,
    max_uses: Option<u32>,
    caching: Option<CacheControl>,
}

impl AdvisorToolOptions {
    pub fn new(model: impl Into<String>) -> Result<Self, MessagesCodecError> {
        let value = Self {
            model: model.into(),
            max_tokens: None,
            max_uses: None,
            caching: None,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn model(&self) -> Result<ModelId, MessagesCodecError> {
        ModelId::new(self.model.clone()).map_err(|_| MessagesCodecError::InvalidOption {
            field: "tools.advisor.model",
            reason: "must be a valid model identifier",
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    pub const fn with_max_uses(mut self, max_uses: u32) -> Self {
        self.max_uses = Some(max_uses);
        self
    }

    pub const fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub const fn with_caching(mut self, caching: CacheControl) -> Self {
        self.caching = Some(caching);
        self
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        self.model()?;
        if self.max_tokens.is_some_and(|max_tokens| max_tokens < 1_024) || self.max_uses == Some(0)
        {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.advisor",
                reason: "max_tokens must be at least 1024 and max_uses must be greater than zero",
            });
        }
        Ok(())
    }

    pub const fn max_tokens(&self) -> Option<u64> {
        self.max_tokens
    }

    pub const fn max_uses(&self) -> Option<u32> {
        self.max_uses
    }

    pub const fn caching(&self) -> Option<CacheControl> {
        self.caching
    }
}

/// Typed MCP per-tool configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpToolConfig {
    enabled: Option<bool>,
    defer_loading: Option<bool>,
}

impl McpToolConfig {
    pub const fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = Some(enabled);
        self
    }

    pub const fn with_defer_loading(mut self, defer_loading: bool) -> Self {
        self.defer_loading = Some(defer_loading);
        self
    }

    pub const fn enabled(&self) -> Option<bool> {
        self.enabled
    }

    pub const fn defer_loading(&self) -> Option<bool> {
        self.defer_loading
    }
}

/// Typed MCP toolset settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpToolsetOptions {
    anchor_name: String,
    server_name: String,
    configs: BTreeMap<String, McpToolConfig>,
    default_config: Option<McpToolConfig>,
}

impl McpToolsetOptions {
    pub fn new(server_name: impl Into<String>) -> Result<Self, MessagesCodecError> {
        let server_name = server_name.into();
        let value = Self {
            anchor_name: server_name.clone(),
            server_name,
            configs: BTreeMap::new(),
            default_config: None,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn with_anchor(
        anchor_name: impl Into<String>,
        server_name: impl Into<String>,
    ) -> Result<Self, MessagesCodecError> {
        let value = Self {
            anchor_name: anchor_name.into(),
            server_name: server_name.into(),
            configs: BTreeMap::new(),
            default_config: None,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn anchor_name(&self) -> &str {
        &self.anchor_name
    }

    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    pub fn with_config(mut self, tool_name: impl Into<String>, config: McpToolConfig) -> Self {
        self.configs.insert(tool_name.into(), config);
        self
    }

    pub const fn with_default_config(mut self, config: McpToolConfig) -> Self {
        self.default_config = Some(config);
        self
    }

    pub fn configs(&self) -> &BTreeMap<String, McpToolConfig> {
        &self.configs
    }

    pub const fn default_config(&self) -> Option<McpToolConfig> {
        self.default_config
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        validate_anchor_name(&self.anchor_name, "tools.mcp_toolset.anchor_name")?;
        validate_bounded_label(&self.server_name, "tools.mcp_toolset.mcp_server_name")?;
        if self.configs.len() > MAX_TOOL_CONFIGS {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.mcp_toolset.configs",
                reason: "must contain at most 256 tool overrides",
            });
        }
        for name in self.configs.keys() {
            validate_bounded_label(name, "tools.mcp_toolset.configs")?;
        }
        Ok(())
    }
}

/// Typed text-editor tool settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextEditorToolOptions {
    max_characters: Option<u32>,
}

impl TextEditorToolOptions {
    pub const fn new() -> Self {
        Self {
            max_characters: None,
        }
    }

    pub const fn with_max_characters(mut self, max_characters: u32) -> Self {
        self.max_characters = Some(max_characters);
        self
    }

    pub const fn max_characters(&self) -> Option<u32> {
        self.max_characters
    }
}

/// Typed computer-use tool settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComputerToolOptions {
    display_width_px: u32,
    display_height_px: u32,
    display_number: Option<u32>,
    enable_zoom: bool,
}

impl ComputerToolOptions {
    pub const fn new(display_width_px: u32, display_height_px: u32) -> Self {
        Self {
            display_width_px,
            display_height_px,
            display_number: None,
            enable_zoom: false,
        }
    }

    pub const fn with_display_number(mut self, display_number: u32) -> Self {
        self.display_number = Some(display_number);
        self
    }

    pub const fn with_enable_zoom(mut self, enable_zoom: bool) -> Self {
        self.enable_zoom = enable_zoom;
        self
    }

    pub const fn display_width_px(&self) -> u32 {
        self.display_width_px
    }

    pub const fn display_height_px(&self) -> u32 {
        self.display_height_px
    }

    pub const fn display_number(&self) -> Option<u32> {
        self.display_number
    }

    pub const fn enable_zoom(&self) -> bool {
        self.enable_zoom
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        if self.display_width_px == 0 || self.display_height_px == 0 {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.computer",
                reason: "display dimensions must be greater than zero",
            });
        }
        Ok(())
    }
}

impl Default for TextEditorToolOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Anthropic-defined tool families supported by the canonical Messages codec.
///
/// Variants include both provider-executed server tools and provider-defined
/// client tools. Execution ownership is part of each tool contract rather than
/// implied by this enum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AnthropicTool {
    WebSearch20260318(WebSearchToolOptions),
    WebFetch20260318(WebFetchToolOptions),
    CodeExecution20260521,
    Advisor20260301(AdvisorToolOptions),
    ToolSearchRegex20251119,
    ToolSearchBm25V20251119,
    McpToolset(McpToolsetOptions),
    Memory20250818,
    Bash20250124,
    TextEditor20250728(TextEditorToolOptions),
    Computer20251124(ComputerToolOptions),
}

impl AnthropicTool {
    pub const fn web_search_20260318(options: WebSearchToolOptions) -> Self {
        Self::WebSearch20260318(options)
    }

    pub const fn web_fetch_20260318(options: WebFetchToolOptions) -> Self {
        Self::WebFetch20260318(options)
    }

    pub fn advisor_20260301(options: AdvisorToolOptions) -> Self {
        Self::Advisor20260301(options)
    }

    pub fn mcp_toolset(options: McpToolsetOptions) -> Self {
        Self::McpToolset(options)
    }

    pub const fn text_editor_20250728(options: TextEditorToolOptions) -> Self {
        Self::TextEditor20250728(options)
    }

    pub const fn computer_20251124(options: ComputerToolOptions) -> Self {
        Self::Computer20251124(options)
    }

    pub fn canonical_name(&self) -> &str {
        match self {
            Self::WebSearch20260318(_) => "web_search",
            Self::WebFetch20260318(_) => "web_fetch",
            Self::CodeExecution20260521 => "code_execution",
            Self::Advisor20260301(_) => "advisor",
            Self::ToolSearchRegex20251119 => "tool_search_tool_regex",
            Self::ToolSearchBm25V20251119 => "tool_search_tool_bm25",
            Self::McpToolset(options) => options.anchor_name(),
            Self::Memory20250818 => "memory",
            Self::Bash20250124 => "bash",
            Self::TextEditor20250728(_) => "str_replace_based_edit_tool",
            Self::Computer20251124(_) => "computer",
        }
    }

    pub(crate) const fn wire_type(&self) -> &'static str {
        match self {
            Self::WebSearch20260318(_) => "web_search_20260318",
            Self::WebFetch20260318(_) => "web_fetch_20260318",
            Self::CodeExecution20260521 => "code_execution_20260521",
            Self::Advisor20260301(_) => "advisor_20260301",
            Self::ToolSearchRegex20251119 => "tool_search_tool_regex_20251119",
            Self::ToolSearchBm25V20251119 => "tool_search_tool_bm25_20251119",
            Self::McpToolset(_) => "mcp_toolset",
            Self::Memory20250818 => "memory_20250818",
            Self::Bash20250124 => "bash_20250124",
            Self::TextEditor20250728(_) => "text_editor_20250728",
            Self::Computer20251124(_) => "computer_20251124",
        }
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        match self {
            Self::WebSearch20260318(options) => options.validate(),
            Self::WebFetch20260318(options) => options.validate(),
            Self::Advisor20260301(options) => options.validate(),
            Self::McpToolset(options) => options.validate(),
            Self::TextEditor20250728(options) => {
                if options.max_characters() == Some(0) {
                    Err(MessagesCodecError::InvalidOption {
                        field: "tools.text_editor.max_characters",
                        reason: "must be greater than zero",
                    })
                } else {
                    Ok(())
                }
            }
            Self::Computer20251124(options) => options.validate(),
            Self::CodeExecution20260521
            | Self::ToolSearchRegex20251119
            | Self::ToolSearchBm25V20251119
            | Self::Memory20250818
            | Self::Bash20250124 => Ok(()),
        }
    }
}

/// Request metadata accepted by Anthropic Messages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MessagesMetadata {
    user_id: String,
}

impl MessagesMetadata {
    pub fn new(user_id: impl Into<String>) -> Result<Self, MessagesCodecError> {
        let value = Self {
            user_id: user_id.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn user_id(&self) -> &str {
        &self.user_id
    }

    fn validate(&self) -> Result<(), MessagesCodecError> {
        if self.user_id.trim().is_empty()
            || self.user_id.len() > 256
            || self.user_id.chars().any(char::is_control)
        {
            return Err(MessagesCodecError::InvalidOption {
                field: "metadata.user_id",
                reason: "must be 1..=256 bytes and contain no control characters",
            });
        }
        Ok(())
    }
}

/// Protocol-owned shaping for one Anthropic Messages request.
#[derive(Debug, Clone, Default)]
pub struct MessagesRequestOptions {
    pub(crate) stream: bool,
    pub(crate) metadata: Option<MessagesMetadata>,
    pub(crate) thinking: Option<ThinkingConfig>,
    pub(crate) output_effort: Option<OutputEffort>,
    pub(crate) fallbacks: Option<ServerFallbacks>,
    pub(crate) top_k: Option<u64>,
    pub(crate) service_tier: Option<MessagesServiceTier>,
    pub(crate) extra: BTreeMap<String, Value>,
}

impl MessagesRequestOptions {
    pub fn new(stream: bool) -> Self {
        Self {
            stream,
            ..Self::default()
        }
    }

    pub const fn stream(&self) -> bool {
        self.stream
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

    pub fn fallbacks(&self) -> Option<&ServerFallbacks> {
        self.fallbacks.as_ref()
    }

    pub const fn top_k(&self) -> Option<u64> {
        self.top_k
    }

    pub const fn service_tier(&self) -> Option<MessagesServiceTier> {
        self.service_tier
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }

    pub fn with_metadata(mut self, metadata: MessagesMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
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

    pub const fn with_service_tier(mut self, service_tier: MessagesServiceTier) -> Self {
        self.service_tier = Some(service_tier);
        self
    }

    pub fn with_extra(mut self, extra: BTreeMap<String, Value>) -> Self {
        self.extra = extra;
        self
    }

    pub fn validate(&self, request: &LanguageRequest) -> Result<(), MessagesCodecError> {
        if let Some(metadata) = &self.metadata {
            metadata.validate()?;
        }
        validate_thinking(
            self.thinking,
            request.generation.max_output_tokens,
            "thinking",
        )?;
        if let Some(fallbacks) = &self.fallbacks {
            fallbacks.validate(request.generation.max_output_tokens)?;
        }
        if self.top_k == Some(0) {
            return Err(MessagesCodecError::InvalidOption {
                field: "top_k",
                reason: "must be greater than zero",
            });
        }
        if request.tools.len() > MAX_TOOL_LIST {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools",
                reason: "must contain at most 256 tools",
            });
        }
        validate_extra(&self.extra)
    }
}

fn validate_thinking(
    thinking: Option<ThinkingConfig>,
    max_tokens: Option<u64>,
    field: &'static str,
) -> Result<(), MessagesCodecError> {
    if let Some(ThinkingConfig::Enabled { budget_tokens, .. }) = thinking {
        if budget_tokens < 1_024 {
            return Err(MessagesCodecError::InvalidOption {
                field,
                reason: "enabled thinking budget must be at least 1024",
            });
        }
        if max_tokens.is_some_and(|maximum| budget_tokens >= maximum) {
            return Err(MessagesCodecError::InvalidOption {
                field,
                reason: "enabled thinking budget must be less than max_tokens",
            });
        }
    }
    Ok(())
}

fn validate_structured_output(output: &StructuredOutputSpec) -> Result<(), MessagesCodecError> {
    if !output.strict {
        return Err(MessagesCodecError::Unsupported {
            feature: "non-strict structured output",
        });
    }
    if !output.schema.is_object() {
        return Err(MessagesCodecError::Unsupported {
            feature: "boolean structured-output schemas",
        });
    }
    Ok(())
}

fn validate_domains(
    domains: &Option<Vec<String>>,
    field: &'static str,
) -> Result<(), MessagesCodecError> {
    let Some(domains) = domains else {
        return Ok(());
    };
    if domains.len() > MAX_DOMAIN_LIST {
        return Err(MessagesCodecError::InvalidOption {
            field,
            reason: "domain list exceeds 256 entries",
        });
    }
    for domain in domains {
        if domain.trim().is_empty()
            || domain.len() > MAX_DOMAIN_BYTES
            || domain
                .chars()
                .any(|character| character.is_control() || character.is_whitespace())
            || domain
                .chars()
                .any(|character| matches!(character, '/' | ':' | '?' | '#' | '@'))
        {
            return Err(MessagesCodecError::InvalidOption {
                field,
                reason: "domains must be bounded host names without URL delimiters",
            });
        }
    }
    Ok(())
}

fn validate_bounded_label(name: &str, field: &'static str) -> Result<(), MessagesCodecError> {
    if name.trim().is_empty()
        || name.len() > MAX_TOOL_NAME_BYTES
        || name.chars().any(char::is_control)
    {
        return Err(MessagesCodecError::InvalidOption {
            field,
            reason: "names must be 1..=128 bytes and contain no control characters",
        });
    }
    Ok(())
}

fn validate_anchor_name(name: &str, field: &'static str) -> Result<(), MessagesCodecError> {
    if name.is_empty()
        || name.len() > MAX_TOOL_NAME_BYTES
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(MessagesCodecError::InvalidOption {
            field,
            reason: "anchor names must be 1..=128 ASCII letters, digits, '-' or '_'",
        });
    }
    Ok(())
}

pub(crate) fn validate_tool_node_options(
    options: &ToolNodeOptions,
    anthropic_tool: Option<&AnthropicTool>,
) -> Result<(), MessagesCodecError> {
    let callers = options.allowed_callers();
    if callers.len() > MAX_ALLOWED_CALLERS {
        return Err(MessagesCodecError::InvalidOption {
            field: "tools.allowed_callers",
            reason: "must contain at most four callers",
        });
    }
    let mut seen = std::collections::BTreeSet::new();
    for caller in callers {
        if !seen.insert(caller.as_wire_str()) {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.allowed_callers",
                reason: "must not contain duplicate callers",
            });
        }
    }
    if matches!(anthropic_tool, Some(AnthropicTool::McpToolset(_)))
        && (!callers.is_empty() || options.strict().is_some() || options.defer_loading().is_some())
    {
        return Err(MessagesCodecError::Unsupported {
            feature: "allowed_callers, strict, or defer_loading on MCP toolsets",
        });
    }
    if let Some(anthropic_tool) = anthropic_tool {
        anthropic_tool.validate()?;
    }
    Ok(())
}

fn validate_extra(extra: &BTreeMap<String, Value>) -> Result<(), MessagesCodecError> {
    let encoded = serde_json::to_vec(extra).map_err(MessagesCodecError::JsonEncode)?;
    if encoded.len() > MAX_EXTRA_BYTES {
        return Err(MessagesCodecError::InvalidOption {
            field: "extra",
            reason: "encoded value exceeds 256 KiB",
        });
    }

    let mut fields = 0usize;
    for (name, value) in extra {
        if is_protected_option_field(name) {
            return Err(MessagesCodecError::ProtectedOptionField {
                path: safe_path(name),
            });
        }
        validate_extra_value(value, name, 1, &mut fields)?;
    }
    Ok(())
}

fn validate_extra_value(
    value: &Value,
    path: &str,
    depth: usize,
    fields: &mut usize,
) -> Result<(), MessagesCodecError> {
    if depth > MAX_EXTRA_DEPTH {
        return Err(MessagesCodecError::InvalidOption {
            field: "extra",
            reason: "JSON nesting exceeds 16 levels",
        });
    }
    match value {
        Value::Object(object) => {
            for (name, child) in object {
                *fields = fields.saturating_add(1);
                if *fields > MAX_EXTRA_FIELDS {
                    return Err(MessagesCodecError::InvalidOption {
                        field: "extra",
                        reason: "JSON object exceeds 1024 fields",
                    });
                }
                let child_path = format!("{path}.{name}");
                if is_sensitive_nested_field(name) {
                    return Err(MessagesCodecError::ProtectedOptionField {
                        path: safe_path(&child_path),
                    });
                }
                validate_extra_value(child, &child_path, depth.saturating_add(1), fields)?;
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                validate_extra_value(
                    child,
                    &format!("{path}[{index}]"),
                    depth.saturating_add(1),
                    fields,
                )?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn is_sensitive_nested_field(name: &str) -> bool {
    matches!(
        normalize_field(name).as_str(),
        "api_key"
            | "x_api_key"
            | "authorization"
            | "auth"
            | "token"
            | "bearer"
            | "endpoint"
            | "base_url"
            | "host"
            | "headers"
            | "header"
            | "proxy"
            | "tls"
            | "audience"
    )
}

pub(crate) fn normalize_field(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace('-', "_")
}

fn safe_path(path: &str) -> String {
    let mut safe = path
        .chars()
        .take(256)
        .map(|character| {
            if character.is_control() {
                '?'
            } else {
                character
            }
        })
        .collect::<String>();
    if path.chars().count() > 256 {
        safe.push_str("...");
    }
    safe
}
