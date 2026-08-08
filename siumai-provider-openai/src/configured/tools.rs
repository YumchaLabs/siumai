//! Typed provider-native tools for the OpenAI Responses API.
//!
//! The shared language contract intentionally models caller-executed function tools only.
//! This module owns OpenAI-hosted tools and keeps an explicit bounded escape hatch for new
//! provider fields that have not yet earned a stable Rust type.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::Error as DeError;
use serde::ser::Error as SerError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use siumai_core::ProviderOptionError;

pub(crate) const MAX_RAW_TOOL_BYTES: usize = 1024 * 1024;
const MAX_TEXT_CHARS: usize = 16 * 1024;
const MAX_ID_CHARS: usize = 512;
const MAX_DOMAIN_COUNT: usize = 100;
const MAX_TOOL_NAMES: usize = 256;
const MAX_FILTER_DEPTH: usize = 16;
const MAX_FILTER_NODES: usize = 256;

/// A provider-native OpenAI Responses tool.
///
/// The enum is deliberately non-exhaustive. New OpenAI tool kinds can be used immediately
/// through [`OpenAiRawTool`] and can later become typed variants without changing the shared
/// `LanguageModel` contract.
#[non_exhaustive]
#[derive(Clone, PartialEq)]
pub enum OpenAiResponsesTool {
    WebSearch(OpenAiWebSearchTool),
    FileSearch(OpenAiFileSearchTool),
    CodeInterpreter(OpenAiCodeInterpreterTool),
    Computer,
    Mcp(OpenAiMcpTool),
    ImageGeneration(OpenAiImageGenerationTool),
    LocalShell,
    Shell(OpenAiShellTool),
    ApplyPatch,
    ToolSearch(OpenAiToolSearchTool),
    ProgrammaticToolCalling,
    Custom(OpenAiCustomTool),
    /// An unknown or intentionally provider-specific tool object.
    Raw(OpenAiRawTool),
}

impl fmt::Debug for OpenAiResponsesTool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WebSearch(value) => formatter.debug_tuple("WebSearch").field(value).finish(),
            Self::FileSearch(value) => formatter.debug_tuple("FileSearch").field(value).finish(),
            Self::CodeInterpreter(value) => formatter
                .debug_tuple("CodeInterpreter")
                .field(value)
                .finish(),
            Self::Computer => formatter.write_str("Computer"),
            Self::Mcp(value) => formatter.debug_tuple("Mcp").field(value).finish(),
            Self::ImageGeneration(value) => formatter
                .debug_tuple("ImageGeneration")
                .field(value)
                .finish(),
            Self::LocalShell => formatter.write_str("LocalShell"),
            Self::Shell(value) => formatter.debug_tuple("Shell").field(value).finish(),
            Self::ApplyPatch => formatter.write_str("ApplyPatch"),
            Self::ToolSearch(value) => formatter.debug_tuple("ToolSearch").field(value).finish(),
            Self::ProgrammaticToolCalling => formatter.write_str("ProgrammaticToolCalling"),
            Self::Custom(value) => formatter.debug_tuple("Custom").field(value).finish(),
            Self::Raw(value) => formatter.debug_tuple("Raw").field(value).finish(),
        }
    }
}

impl OpenAiResponsesTool {
    /// Construct the default web-search tool.
    pub fn web_search() -> Self {
        Self::WebSearch(OpenAiWebSearchTool::default())
    }

    /// Construct the default file-search tool.
    pub fn file_search(vector_store_ids: Vec<String>) -> Self {
        Self::FileSearch(OpenAiFileSearchTool::new(vector_store_ids))
    }

    /// Construct the default code-interpreter tool.
    pub fn code_interpreter() -> Self {
        Self::CodeInterpreter(OpenAiCodeInterpreterTool::default())
    }

    /// Construct an MCP tool from a validated builder shape.
    pub fn mcp(tool: OpenAiMcpTool) -> Self {
        Self::Mcp(tool)
    }

    /// Construct the default image-generation tool.
    pub fn image_generation() -> Self {
        Self::ImageGeneration(OpenAiImageGenerationTool::default())
    }

    /// Construct the default tool-search tool.
    pub fn tool_search() -> Self {
        Self::ToolSearch(OpenAiToolSearchTool::default())
    }

    /// Construct the Responses programmatic tool-calling tool.
    pub const fn programmatic_tool_calling() -> Self {
        Self::ProgrammaticToolCalling
    }

    /// Construct an OpenAI custom tool.
    pub fn custom(name: impl Into<String>) -> Self {
        Self::Custom(OpenAiCustomTool::new(name))
    }

    /// Construct an unknown tool from its complete JSON object.
    pub fn raw(value: Value) -> Result<Self, ProviderOptionError> {
        Ok(Self::Raw(OpenAiRawTool::new(value)?))
    }

    pub(crate) fn validate(&self) -> Result<(), ProviderOptionError> {
        match self {
            Self::WebSearch(value) => value.validate(),
            Self::FileSearch(value) => value.validate(),
            Self::CodeInterpreter(value) => value.validate(),
            Self::Computer
            | Self::LocalShell
            | Self::ApplyPatch
            | Self::ProgrammaticToolCalling => Ok(()),
            Self::Mcp(value) => value.validate(),
            Self::ImageGeneration(value) => value.validate(),
            Self::Shell(value) => value.validate(),
            Self::ToolSearch(value) => value.validate(),
            Self::Custom(value) => value.validate(),
            Self::Raw(value) => value.validate(),
        }
    }

    pub(crate) fn into_value(self) -> Result<Value, ProviderOptionError> {
        self.validate()?;
        let value = serde_json::to_value(self)
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?;
        let encoded_bytes = serde_json::to_vec(&value)
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?
            .len();
        if encoded_bytes > MAX_RAW_TOOL_BYTES {
            return Err(rejected(
                "tools",
                "OpenAI Responses tools must not exceed 1 MiB when encoded",
            ));
        }
        Ok(value)
    }
}

impl From<OpenAiWebSearchTool> for OpenAiResponsesTool {
    fn from(value: OpenAiWebSearchTool) -> Self {
        Self::WebSearch(value)
    }
}

impl From<OpenAiFileSearchTool> for OpenAiResponsesTool {
    fn from(value: OpenAiFileSearchTool) -> Self {
        Self::FileSearch(value)
    }
}

impl From<OpenAiCodeInterpreterTool> for OpenAiResponsesTool {
    fn from(value: OpenAiCodeInterpreterTool) -> Self {
        Self::CodeInterpreter(value)
    }
}

impl From<OpenAiMcpTool> for OpenAiResponsesTool {
    fn from(value: OpenAiMcpTool) -> Self {
        Self::Mcp(value)
    }
}

impl From<OpenAiImageGenerationTool> for OpenAiResponsesTool {
    fn from(value: OpenAiImageGenerationTool) -> Self {
        Self::ImageGeneration(value)
    }
}

impl From<OpenAiShellTool> for OpenAiResponsesTool {
    fn from(value: OpenAiShellTool) -> Self {
        Self::Shell(value)
    }
}

impl From<OpenAiToolSearchTool> for OpenAiResponsesTool {
    fn from(value: OpenAiToolSearchTool) -> Self {
        Self::ToolSearch(value)
    }
}

impl From<OpenAiCustomTool> for OpenAiResponsesTool {
    fn from(value: OpenAiCustomTool) -> Self {
        Self::Custom(value)
    }
}

impl Serialize for OpenAiResponsesTool {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let value = match self {
            Self::WebSearch(tool) => tagged("web_search", tool),
            Self::FileSearch(tool) => tagged("file_search", tool),
            Self::CodeInterpreter(tool) => tagged("code_interpreter", tool),
            Self::Computer => Ok(unit_tagged("computer")),
            Self::Mcp(tool) => tagged("mcp", tool),
            Self::ImageGeneration(tool) => tagged("image_generation", tool),
            Self::LocalShell => Ok(unit_tagged("local_shell")),
            Self::Shell(tool) => tagged("shell", tool),
            Self::ApplyPatch => Ok(unit_tagged("apply_patch")),
            Self::ToolSearch(tool) => tagged("tool_search", tool),
            Self::ProgrammaticToolCalling => Ok(unit_tagged("programmatic_tool_calling")),
            Self::Custom(tool) => tagged("custom", tool),
            Self::Raw(tool) => Ok(tool.value.clone()),
        }
        .map_err(S::Error::custom)?;
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for OpenAiResponsesTool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| D::Error::custom("OpenAI Responses tool requires a string type"))?;
        let decoded = match kind {
            "web_search" => Self::WebSearch(decode_tagged(value)?),
            "file_search" => Self::FileSearch(decode_tagged(value)?),
            "code_interpreter" => Self::CodeInterpreter(decode_tagged(value)?),
            "computer" => {
                decode_unit::<D::Error>(&value)?;
                Self::Computer
            }
            "mcp" => Self::Mcp(decode_tagged(value)?),
            "image_generation" => Self::ImageGeneration(decode_tagged(value)?),
            "local_shell" => {
                decode_unit::<D::Error>(&value)?;
                Self::LocalShell
            }
            "shell" => Self::Shell(decode_tagged(value)?),
            "apply_patch" => {
                decode_unit::<D::Error>(&value)?;
                Self::ApplyPatch
            }
            "tool_search" => Self::ToolSearch(decode_tagged(value)?),
            "programmatic_tool_calling" => {
                decode_unit::<D::Error>(&value)?;
                Self::ProgrammaticToolCalling
            }
            "custom" => Self::Custom(decode_tagged(value)?),
            _ => Self::Raw(OpenAiRawTool::from_value(value).map_err(D::Error::custom)?),
        };
        decoded.validate().map_err(D::Error::custom)?;
        Ok(decoded)
    }
}

/// A bounded raw OpenAI tool escape hatch.
#[derive(Clone, PartialEq)]
pub struct OpenAiRawTool {
    value: Value,
    encoded_bytes: usize,
}

impl fmt::Debug for OpenAiRawTool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiRawTool")
            .field("type", &self.value.get("type").and_then(Value::as_str))
            .field("encoded_bytes", &self.encoded_bytes)
            .field("value", &"<redacted>")
            .finish()
    }
}

impl Serialize for OpenAiRawTool {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for OpenAiRawTool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_value(Value::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

impl OpenAiRawTool {
    pub fn new(value: Value) -> Result<Self, ProviderOptionError> {
        Self::from_value(value)
    }

    pub fn as_value(&self) -> &Value {
        &self.value
    }

    pub fn encoded_bytes(&self) -> usize {
        self.encoded_bytes
    }

    fn from_value(value: Value) -> Result<Self, ProviderOptionError> {
        let Some(object) = value.as_object() else {
            return Err(rejected(
                "tools",
                "OpenAI Responses tools must be JSON objects",
            ));
        };
        let Some(kind) = object.get("type").and_then(Value::as_str) else {
            return Err(rejected(
                "tools.type",
                "OpenAI Responses tools require a string type",
            ));
        };
        validate_text("tools.type", kind, 128)?;
        let encoded_bytes = serde_json::to_vec(&value)
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?
            .len();
        if encoded_bytes > MAX_RAW_TOOL_BYTES {
            return Err(rejected(
                "tools",
                "raw OpenAI Responses tools must not exceed 1 MiB when encoded",
            ));
        }
        Ok(Self {
            value,
            encoded_bytes,
        })
    }

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if self.encoded_bytes > MAX_RAW_TOOL_BYTES {
            return Err(rejected("tools", "raw OpenAI Responses tool exceeds 1 MiB"));
        }
        Ok(())
    }
}

/// Web-search hosted tool controls.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiWebSearchTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_web_access: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<OpenAiWebSearchFilters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<OpenAiWebSearchContextSize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_token_budget: Option<OpenAiWebSearchReturnTokenBudget>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<OpenAiApproximateLocation>,
}

impl OpenAiWebSearchTool {
    pub fn with_search_context_size(mut self, value: OpenAiWebSearchContextSize) -> Self {
        self.search_context_size = Some(value);
        self
    }

    pub fn with_return_token_budget(mut self, value: OpenAiWebSearchReturnTokenBudget) -> Self {
        self.return_token_budget = Some(value);
        self
    }

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if let Some(filters) = &self.filters {
            filters.validate()?;
        }
        if let Some(location) = &self.user_location {
            location.validate()?;
        }
        Ok(())
    }
}

/// Web-search context size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiWebSearchContextSize {
    Low,
    Medium,
    High,
}

/// Web-search returned-token budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiWebSearchReturnTokenBudget {
    Default,
    Unlimited,
}

/// Domain filters for web search.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiWebSearchFilters {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_domains: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocked_domains: Vec<String>,
}

impl OpenAiWebSearchFilters {
    fn validate(&self) -> Result<(), ProviderOptionError> {
        validate_domains("tools.filters.allowed_domains", &self.allowed_domains)?;
        validate_domains("tools.filters.blocked_domains", &self.blocked_domains)
    }
}

/// Approximate user location used for geographically relevant web search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiApproximateLocation {
    #[serde(default = "approximate_location_type")]
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

impl Default for OpenAiApproximateLocation {
    fn default() -> Self {
        Self {
            r#type: approximate_location_type(),
            country: None,
            city: None,
            region: None,
            timezone: None,
        }
    }
}

impl OpenAiApproximateLocation {
    pub fn new() -> Self {
        Self {
            r#type: "approximate".to_string(),
            ..Self::default()
        }
    }

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if self.r#type != "approximate" {
            return Err(rejected(
                "tools.user_location.type",
                "OpenAI web-search location type must be approximate",
            ));
        }
        for (path, value) in [
            ("country", self.country.as_deref()),
            ("city", self.city.as_deref()),
            ("region", self.region.as_deref()),
            ("timezone", self.timezone.as_deref()),
        ] {
            if let Some(value) = value {
                validate_text(format!("tools.user_location.{path}"), value, 256)?;
            }
        }
        Ok(())
    }
}

/// File-search tool filter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum OpenAiFileSearchFilter {
    Eq { key: String, value: Value },
    Ne { key: String, value: Value },
    Gt { key: String, value: Value },
    Gte { key: String, value: Value },
    Lt { key: String, value: Value },
    Lte { key: String, value: Value },
    In { key: String, value: Vec<String> },
    Nin { key: String, value: Vec<String> },
    And { filters: Vec<Self> },
    Or { filters: Vec<Self> },
}

/// File-search ranking options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiFileSearchRankingOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f64>,
}

/// File-search hosted tool controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiFileSearchTool {
    pub vector_store_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_num_results: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranking_options: Option<OpenAiFileSearchRankingOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<OpenAiFileSearchFilter>,
}

impl OpenAiFileSearchTool {
    pub fn new(vector_store_ids: Vec<String>) -> Self {
        Self {
            vector_store_ids,
            max_num_results: None,
            ranking_options: None,
            filters: None,
        }
    }

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if self.vector_store_ids.is_empty() {
            return Err(rejected(
                "tools.vector_store_ids",
                "file search requires at least one vector store id",
            ));
        }
        for id in &self.vector_store_ids {
            validate_text("tools.vector_store_ids", id, MAX_ID_CHARS)?;
        }
        if self.max_num_results == Some(0) {
            return Err(rejected(
                "tools.max_num_results",
                "file-search max_num_results must be greater than zero",
            ));
        }
        if let Some(ranking) = &self.ranking_options {
            if ranking
                .score_threshold
                .is_some_and(|value| !(0.0..=1.0).contains(&value))
            {
                return Err(rejected(
                    "tools.ranking_options.score_threshold",
                    "file-search score_threshold must be between 0 and 1",
                ));
            }
            if let Some(ranker) = &ranking.ranker {
                validate_text("tools.ranking_options.ranker", ranker, 256)?;
            }
        }
        if let Some(filter) = &self.filters {
            validate_file_filter(filter, 0, &mut 0)?;
        }
        Ok(())
    }
}

/// Code-interpreter container selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAiCodeInterpreterContainer {
    Id(String),
    Auto(OpenAiCodeInterpreterAutoContainer),
}

/// Automatic code-interpreter container settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiCodeInterpreterAutoContainer {
    #[serde(rename = "type", default = "auto_container_type")]
    pub r#type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub file_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_limit: Option<OpenAiContainerMemoryLimit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network_policy: Option<OpenAiContainerNetworkPolicy>,
}

impl Default for OpenAiCodeInterpreterAutoContainer {
    fn default() -> Self {
        Self {
            r#type: auto_container_type(),
            file_ids: Vec::new(),
            memory_limit: None,
            network_policy: None,
        }
    }
}

/// Container memory limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiContainerMemoryLimit {
    #[serde(rename = "1g")]
    OneGb,
    #[serde(rename = "4g")]
    FourGb,
    #[serde(rename = "16g")]
    SixteenGb,
    #[serde(rename = "64g")]
    SixtyFourGb,
}

/// Container network policy.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAiContainerNetworkPolicy {
    Disabled,
    Allowlist {
        allowed_domains: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        domain_secrets: Vec<OpenAiDomainSecret>,
    },
}

impl fmt::Debug for OpenAiContainerNetworkPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => formatter.write_str("Disabled"),
            Self::Allowlist {
                allowed_domains,
                domain_secrets,
            } => formatter
                .debug_struct("Allowlist")
                .field("allowed_domains", allowed_domains)
                .field(
                    "domain_secrets",
                    &format_args!("<{} redacted>", domain_secrets.len()),
                )
                .finish(),
        }
    }
}

/// A secret injected for one shell/network domain.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiDomainSecret {
    pub domain: String,
    pub name: String,
    pub value: String,
}

impl fmt::Debug for OpenAiDomainSecret {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiDomainSecret")
            .field("domain", &self.domain)
            .field("name", &self.name)
            .field("value", &"<redacted>")
            .finish()
    }
}

/// Code-interpreter hosted tool controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiCodeInterpreterTool {
    pub container: OpenAiCodeInterpreterContainer,
}

impl Default for OpenAiCodeInterpreterTool {
    fn default() -> Self {
        Self {
            container: OpenAiCodeInterpreterContainer::Auto(
                OpenAiCodeInterpreterAutoContainer::default(),
            ),
        }
    }
}

impl OpenAiCodeInterpreterTool {
    fn validate(&self) -> Result<(), ProviderOptionError> {
        match &self.container {
            OpenAiCodeInterpreterContainer::Id(id) => {
                validate_text("tools.container", id, MAX_ID_CHARS)?;
            }
            OpenAiCodeInterpreterContainer::Auto(container) => {
                if container.r#type != "auto" {
                    return Err(rejected(
                        "tools.container.type",
                        "code-interpreter automatic containers must use type=auto",
                    ));
                }
                for id in &container.file_ids {
                    validate_text("tools.container.file_ids", id, MAX_ID_CHARS)?;
                }
                if let Some(policy) = &container.network_policy {
                    policy.validate()?;
                }
            }
        }
        Ok(())
    }
}

/// Image-generation hosted tool controls.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiImageGenerationTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<OpenAiImageBackground>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_fidelity: Option<OpenAiImageInputFidelity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_image_mask: Option<OpenAiImageInputMask>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moderation: Option<OpenAiImageModeration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_compression: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OpenAiImageOutputFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_images: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<OpenAiImageQuality>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<OpenAiImageSize>,
}

impl OpenAiImageGenerationTool {
    fn validate(&self) -> Result<(), ProviderOptionError> {
        if self.output_compression.is_some_and(|value| value > 100) {
            return Err(rejected(
                "tools.output_compression",
                "image output compression must be between 0 and 100",
            ));
        }
        if self.partial_images.is_some_and(|value| value > 3) {
            return Err(rejected(
                "tools.partial_images",
                "image partial_images must be between 0 and 3",
            ));
        }
        if let Some(model) = &self.model {
            validate_text("tools.model", model, MAX_ID_CHARS)?;
        }
        if let Some(mask) = &self.input_image_mask
            && mask.file_id.is_none()
            && mask.image_url.is_none()
        {
            return Err(rejected(
                "tools.input_image_mask",
                "image input_image_mask requires file_id or image_url",
            ));
        }
        Ok(())
    }
}

/// Image generation background mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiImageBackground {
    Auto,
    Opaque,
    Transparent,
}

/// Image input fidelity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiImageInputFidelity {
    Low,
    High,
}

/// Image input mask.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiImageInputMask {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

/// Image moderation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiImageModeration {
    Auto,
}

/// Image output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiImageOutputFormat {
    Png,
    Jpeg,
    Webp,
}

/// Image generation quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiImageQuality {
    Auto,
    Low,
    Medium,
    High,
}

/// Image output size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenAiImageSize {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "1024x1024")]
    Square,
    #[serde(rename = "1024x1536")]
    Portrait,
    #[serde(rename = "1536x1024")]
    Landscape,
}

/// MCP server tool controls.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiMcpTool {
    pub server_label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<OpenAiMcpAllowedTools>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authorization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connector_id: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_approval: Option<OpenAiMcpApproval>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tunnel_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
}

impl fmt::Debug for OpenAiMcpTool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiMcpTool")
            .field("server_label", &self.server_label)
            .field("allowed_tools", &self.allowed_tools)
            .field(
                "authorization",
                &self.authorization.as_ref().map(|_| "<redacted>"),
            )
            .field("connector_id", &self.connector_id)
            .field(
                "headers",
                &self.headers.keys().map(String::as_str).collect::<Vec<_>>(),
            )
            .field("require_approval", &self.require_approval)
            .field("server_description", &self.server_description)
            .field("server_url", &self.server_url)
            .field("tunnel_id", &self.tunnel_id)
            .field("defer_loading", &self.defer_loading)
            .finish()
    }
}

impl OpenAiMcpTool {
    pub fn new(server_label: impl Into<String>) -> Self {
        Self {
            server_label: server_label.into(),
            allowed_tools: None,
            authorization: None,
            connector_id: None,
            headers: BTreeMap::new(),
            require_approval: None,
            server_description: None,
            server_url: None,
            tunnel_id: None,
            defer_loading: None,
        }
    }

    pub fn with_server_url(mut self, server_url: impl Into<String>) -> Self {
        self.server_url = Some(server_url.into());
        self.connector_id = None;
        self.tunnel_id = None;
        self
    }

    pub fn with_connector(mut self, connector_id: impl Into<String>) -> Self {
        self.connector_id = Some(connector_id.into());
        self.server_url = None;
        self.tunnel_id = None;
        self
    }

    pub fn with_tunnel(mut self, tunnel_id: impl Into<String>) -> Self {
        self.tunnel_id = Some(tunnel_id.into());
        self.server_url = None;
        self.connector_id = None;
        self
    }

    pub fn with_authorization(mut self, authorization: impl Into<String>) -> Self {
        self.authorization = Some(authorization.into());
        self
    }

    pub fn with_approval(mut self, approval: OpenAiMcpApproval) -> Self {
        self.require_approval = Some(approval);
        self
    }

    fn validate(&self) -> Result<(), ProviderOptionError> {
        validate_text("tools.server_label", &self.server_label, 256)?;
        let connection_count = usize::from(self.server_url.is_some())
            + usize::from(self.connector_id.is_some())
            + usize::from(self.tunnel_id.is_some());
        if connection_count != 1 {
            return Err(rejected(
                "tools.mcp",
                "MCP tools require exactly one server_url, connector_id, or tunnel_id",
            ));
        }
        if let Some(url) = &self.server_url {
            validate_text("tools.server_url", url, 2048)?;
        }
        if let Some(id) = &self.connector_id {
            validate_text("tools.connector_id", id, MAX_ID_CHARS)?;
        }
        if let Some(id) = &self.tunnel_id {
            validate_text("tools.tunnel_id", id, MAX_ID_CHARS)?;
        }
        if let Some(description) = &self.server_description {
            validate_text("tools.server_description", description, MAX_TEXT_CHARS)?;
        }
        if let Some(authorization) = &self.authorization {
            validate_text("tools.authorization", authorization, 16 * 1024)?;
        }
        if let Some(allowed) = &self.allowed_tools {
            allowed.validate()?;
        }
        if self.headers.len() > 64 {
            return Err(rejected(
                "tools.headers",
                "MCP headers must not exceed 64 entries",
            ));
        }
        for (name, value) in &self.headers {
            validate_text("tools.headers.name", name, 256)?;
            validate_control_free("tools.headers.value", value, 8192)?;
        }
        if let Some(approval) = &self.require_approval {
            approval.validate()?;
        }
        Ok(())
    }
}

/// MCP allowed-tool selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAiMcpAllowedTools {
    Names(Vec<String>),
    Filter {
        #[serde(skip_serializing_if = "Option::is_none")]
        read_only: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_names: Option<Vec<String>>,
    },
}

impl OpenAiMcpAllowedTools {
    fn validate(&self) -> Result<(), ProviderOptionError> {
        match self {
            Self::Names(names) => validate_tool_names(names),
            Self::Filter { tool_names, .. } => {
                if let Some(names) = tool_names {
                    validate_tool_names(names)?;
                }
                Ok(())
            }
        }
    }
}

/// MCP approval policy.
#[derive(Debug, Clone, PartialEq)]
pub enum OpenAiMcpApproval {
    Always,
    Never,
    NeverFilter { never: NeverApprovalFilter },
}

impl Serialize for OpenAiMcpApproval {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Always => serializer.serialize_str("always"),
            Self::Never => serializer.serialize_str("never"),
            Self::NeverFilter { never } => {
                let mut object = Map::new();
                object.insert(
                    "never".to_string(),
                    serde_json::to_value(never).map_err(serde::ser::Error::custom)?,
                );
                Value::Object(object).serialize(serializer)
            }
        }
    }
}

impl<'de> Deserialize<'de> for OpenAiMcpApproval {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(value) if value == "always" => Ok(Self::Always),
            Value::String(value) if value == "never" => Ok(Self::Never),
            Value::Object(mut object) if object.len() == 1 => {
                let never = object
                    .remove("never")
                    .ok_or_else(|| D::Error::custom("expected never approval filter"))?;
                Ok(Self::NeverFilter {
                    never: serde_json::from_value(never).map_err(D::Error::custom)?,
                })
            }
            _ => Err(D::Error::custom(
                "MCP approval must be always, never, or a never filter",
            )),
        }
    }
}

/// Tool-name filter for the MCP `never` approval policy.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeverApprovalFilter {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_names: Vec<String>,
}

impl OpenAiMcpApproval {
    fn validate(&self) -> Result<(), ProviderOptionError> {
        if let Self::NeverFilter { never } = self {
            validate_tool_names(&never.tool_names)?;
        }
        Ok(())
    }
}

/// Tool-search controls.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiToolSearchTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution: Option<OpenAiToolSearchExecution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

/// Tool-search execution owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiToolSearchExecution {
    Server,
    Client,
}

impl OpenAiToolSearchTool {
    fn validate(&self) -> Result<(), ProviderOptionError> {
        if let Some(description) = &self.description {
            validate_text("tools.description", description, MAX_TEXT_CHARS)?;
        }
        if self
            .parameters
            .as_ref()
            .is_some_and(|value| !value.is_object())
        {
            return Err(rejected(
                "tools.parameters",
                "tool-search parameters must be a JSON Schema object",
            ));
        }
        Ok(())
    }
}

/// Custom caller-executed Responses tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiCustomTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<OpenAiCustomToolFormat>,
}

impl OpenAiCustomTool {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            format: None,
        }
    }

    fn validate(&self) -> Result<(), ProviderOptionError> {
        validate_text("tools.name", &self.name, 128)?;
        if let Some(description) = &self.description {
            validate_text("tools.description", description, MAX_TEXT_CHARS)?;
        }
        if let Some(OpenAiCustomToolFormat::Grammar { definition, .. }) = &self.format {
            validate_text("tools.format.definition", definition, MAX_TEXT_CHARS)?;
        }
        Ok(())
    }
}

/// Output format for a custom tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum OpenAiCustomToolFormat {
    Text,
    Grammar {
        syntax: OpenAiGrammarSyntax,
        definition: String,
    },
}

/// Grammar syntax accepted by a custom tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiGrammarSyntax {
    Regex,
    Lark,
}

/// Hosted shell tool controls.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiShellTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<OpenAiShellEnvironment>,
}

impl fmt::Debug for OpenAiShellTool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiShellTool")
            .field("environment", &self.environment)
            .finish()
    }
}

/// Shell execution environment.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAiShellEnvironment {
    ContainerAuto {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        file_ids: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        memory_limit: Option<OpenAiContainerMemoryLimit>,
        #[serde(skip_serializing_if = "Option::is_none")]
        network_policy: Option<OpenAiContainerNetworkPolicy>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        skills: Vec<OpenAiShellSkill>,
    },
    ContainerReference {
        container_id: String,
    },
    Local {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        skills: Vec<OpenAiLocalShellSkill>,
    },
}

impl fmt::Debug for OpenAiShellEnvironment {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContainerAuto {
                file_ids,
                memory_limit,
                network_policy,
                skills,
            } => formatter
                .debug_struct("ContainerAuto")
                .field("file_ids", file_ids)
                .field("memory_limit", memory_limit)
                .field("network_policy", network_policy)
                .field("skills", &format_args!("<{} entries>", skills.len()))
                .finish(),
            Self::ContainerReference { container_id } => formatter
                .debug_struct("ContainerReference")
                .field("container_id", container_id)
                .finish(),
            Self::Local { skills } => formatter
                .debug_struct("Local")
                .field("skills", &format_args!("<{} entries>", skills.len()))
                .finish(),
        }
    }
}

/// A shell skill reference or inline skill package.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAiShellSkill {
    SkillReference {
        skill_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        version: Option<String>,
    },
    Inline {
        name: String,
        description: String,
        source: OpenAiInlineSkillSource,
    },
}

/// A host-local shell skill mounted from a caller-controlled path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiLocalShellSkill {
    pub name: String,
    pub description: String,
    pub path: String,
}

impl fmt::Debug for OpenAiShellSkill {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SkillReference { skill_id, version } => formatter
                .debug_struct("SkillReference")
                .field("skill_id", skill_id)
                .field("version", version)
                .finish(),
            Self::Inline {
                name,
                description,
                source,
            } => formatter
                .debug_struct("Inline")
                .field("name", name)
                .field("description", description)
                .field("source", source)
                .finish(),
        }
    }
}

/// Inline shell skill source.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiInlineSkillSource {
    #[serde(rename = "type")]
    pub r#type: String,
    pub media_type: String,
    pub data: String,
}

impl fmt::Debug for OpenAiInlineSkillSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiInlineSkillSource")
            .field("type", &self.r#type)
            .field("media_type", &self.media_type)
            .field(
                "data",
                &format_args!("<{} bytes redacted>", self.data.len()),
            )
            .finish()
    }
}

impl OpenAiShellTool {
    fn validate(&self) -> Result<(), ProviderOptionError> {
        if let Some(environment) = &self.environment {
            validate_shell_environment(environment)?;
        }
        Ok(())
    }
}

fn tagged<T: Serialize>(kind: &str, value: &T) -> Result<Value, serde_json::Error> {
    let mut object = match serde_json::to_value(value)? {
        Value::Object(object) => object,
        _ => {
            return Err(serde_json::Error::io(std::io::Error::other(
                "OpenAI tool options must serialize as objects",
            )));
        }
    };
    object.insert("type".to_string(), Value::String(kind.to_string()));
    Ok(Value::Object(object))
}

fn unit_tagged(kind: &str) -> Value {
    let mut object = Map::new();
    object.insert("type".to_string(), Value::String(kind.to_string()));
    Value::Object(object)
}

fn decode_tagged<T: for<'de> Deserialize<'de>, D: DeError>(mut value: Value) -> Result<T, D> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| D::custom("OpenAI tool must be a JSON object"))?;
    object.remove("type");
    serde_json::from_value(value).map_err(D::custom)
}

fn decode_unit<E: DeError>(value: &Value) -> Result<(), E> {
    if value.as_object().is_some_and(|object| object.len() == 1) {
        Ok(())
    } else {
        Err(E::custom(
            "unit OpenAI tool must contain only its type field",
        ))
    }
}

fn validate_file_filter(
    filter: &OpenAiFileSearchFilter,
    depth: usize,
    nodes: &mut usize,
) -> Result<(), ProviderOptionError> {
    *nodes = nodes.saturating_add(1);
    if *nodes > MAX_FILTER_NODES || depth > MAX_FILTER_DEPTH {
        return Err(rejected(
            "tools.filters",
            "file-search filters are too deep or large",
        ));
    }
    match filter {
        OpenAiFileSearchFilter::Eq { key, value }
        | OpenAiFileSearchFilter::Ne { key, value }
        | OpenAiFileSearchFilter::Gt { key, value }
        | OpenAiFileSearchFilter::Gte { key, value }
        | OpenAiFileSearchFilter::Lt { key, value }
        | OpenAiFileSearchFilter::Lte { key, value } => {
            validate_text("tools.filters.key", key, 256)?;
            if !matches!(value, Value::String(_) | Value::Number(_) | Value::Bool(_)) {
                return Err(rejected(
                    "tools.filters.value",
                    "comparison filter values must be strings, numbers, or booleans",
                ));
            }
        }
        OpenAiFileSearchFilter::In { key, value } | OpenAiFileSearchFilter::Nin { key, value } => {
            validate_text("tools.filters.key", key, 256)?;
            if value.is_empty() {
                return Err(rejected(
                    "tools.filters.value",
                    "filter value list cannot be empty",
                ));
            }
            for item in value {
                validate_text("tools.filters.value", item, 1024)?;
            }
        }
        OpenAiFileSearchFilter::And { filters } | OpenAiFileSearchFilter::Or { filters } => {
            if filters.is_empty() {
                return Err(rejected(
                    "tools.filters",
                    "compound filters cannot be empty",
                ));
            }
            for child in filters {
                validate_file_filter(child, depth + 1, nodes)?;
            }
        }
    }
    Ok(())
}

fn validate_shell_environment(
    environment: &OpenAiShellEnvironment,
) -> Result<(), ProviderOptionError> {
    match environment {
        OpenAiShellEnvironment::ContainerAuto {
            file_ids,
            network_policy,
            skills,
            ..
        } => {
            for id in file_ids {
                validate_text("tools.environment.file_ids", id, MAX_ID_CHARS)?;
            }
            if let Some(policy) = network_policy {
                policy.validate()?;
            }
            for skill in skills {
                validate_skill(skill)?;
            }
        }
        OpenAiShellEnvironment::ContainerReference { container_id } => {
            validate_text("tools.environment.container_id", container_id, MAX_ID_CHARS)?;
        }
        OpenAiShellEnvironment::Local { skills } => {
            for skill in skills {
                validate_text("tools.skills.name", &skill.name, 256)?;
                validate_text(
                    "tools.skills.description",
                    &skill.description,
                    MAX_TEXT_CHARS,
                )?;
                validate_text("tools.skills.path", &skill.path, 4096)?;
            }
        }
    }
    Ok(())
}

impl OpenAiContainerNetworkPolicy {
    fn validate(&self) -> Result<(), ProviderOptionError> {
        if let Self::Allowlist {
            allowed_domains,
            domain_secrets,
        } = self
        {
            validate_domains("tools.network_policy.allowed_domains", allowed_domains)?;
            for secret in domain_secrets {
                validate_text("tools.network_policy.secret.domain", &secret.domain, 253)?;
                validate_text("tools.network_policy.secret.name", &secret.name, 256)?;
                validate_text(
                    "tools.network_policy.secret.value",
                    &secret.value,
                    16 * 1024,
                )?;
            }
        }
        Ok(())
    }
}

fn validate_skill(skill: &OpenAiShellSkill) -> Result<(), ProviderOptionError> {
    match skill {
        OpenAiShellSkill::SkillReference { skill_id, version } => {
            validate_text("tools.skills.skill_id", skill_id, MAX_ID_CHARS)?;
            if let Some(version) = version {
                validate_text("tools.skills.version", version, 128)?;
            }
        }
        OpenAiShellSkill::Inline {
            name,
            description,
            source,
        } => {
            validate_text("tools.skills.name", name, 256)?;
            validate_text("tools.skills.description", description, MAX_TEXT_CHARS)?;
            validate_text("tools.skills.source.type", &source.r#type, 32)?;
            validate_text("tools.skills.source.media_type", &source.media_type, 128)?;
            if source.r#type != "base64" || source.media_type != "application/zip" {
                return Err(rejected(
                    "tools.skills.source",
                    "inline shell skills must use base64 application/zip data",
                ));
            }
            validate_text("tools.skills.source.data", &source.data, MAX_RAW_TOOL_BYTES)?;
        }
    }
    Ok(())
}

fn validate_domains(path: &str, domains: &[String]) -> Result<(), ProviderOptionError> {
    if domains.len() > MAX_DOMAIN_COUNT {
        return Err(rejected(path, "domain filters must not exceed 100 entries"));
    }
    for domain in domains {
        validate_text(path, domain, 253)?;
        if domain.contains('/') || domain.contains(':') {
            return Err(rejected(path, "domains must omit an URL scheme and path"));
        }
    }
    Ok(())
}

fn validate_tool_names(names: &[String]) -> Result<(), ProviderOptionError> {
    if names.len() > MAX_TOOL_NAMES {
        return Err(rejected(
            "tools.allowed_tools",
            "tool name filters are too large",
        ));
    }
    for name in names {
        validate_text("tools.allowed_tools", name, 256)?;
    }
    Ok(())
}

fn validate_text(
    path: impl Into<String>,
    value: &str,
    max_chars: usize,
) -> Result<(), ProviderOptionError> {
    if value.is_empty() || value != value.trim() || value.chars().any(char::is_control) {
        return Err(rejected(
            path,
            "value must be non-empty, trimmed, and control-free",
        ));
    }
    if value.chars().count() > max_chars {
        return Err(rejected(
            path,
            format!("value must not exceed {max_chars} characters"),
        ));
    }
    Ok(())
}

fn validate_control_free(
    path: impl Into<String>,
    value: &str,
    max_chars: usize,
) -> Result<(), ProviderOptionError> {
    if value.chars().any(char::is_control) || value.chars().count() > max_chars {
        return Err(rejected(
            path,
            format!("value must be control-free and not exceed {max_chars} characters"),
        ));
    }
    Ok(())
}

fn rejected(path: impl Into<String>, reason: impl Into<String>) -> ProviderOptionError {
    ProviderOptionError::Rejected {
        path: path.into(),
        reason: reason.into(),
    }
}

fn approximate_location_type() -> String {
    "approximate".to_string()
}

fn auto_container_type() -> String {
    "auto".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn typed_tools_round_trip_current_wire_shapes() {
        let tool = OpenAiResponsesTool::WebSearch(OpenAiWebSearchTool {
            external_web_access: Some(true),
            filters: Some(OpenAiWebSearchFilters {
                allowed_domains: vec!["example.com".to_string()],
                blocked_domains: Vec::new(),
            }),
            search_context_size: Some(OpenAiWebSearchContextSize::High),
            return_token_budget: Some(OpenAiWebSearchReturnTokenBudget::Unlimited),
            user_location: Some(OpenAiApproximateLocation::new()),
        });
        let value = serde_json::to_value(&tool).unwrap();
        assert_eq!(value["type"], "web_search");
        assert_eq!(value["filters"]["allowed_domains"][0], "example.com");
        assert_eq!(value["return_token_budget"], "unlimited");
        assert_eq!(
            serde_json::from_value::<OpenAiResponsesTool>(value).unwrap(),
            tool
        );
    }

    #[test]
    fn unknown_tool_round_trips_through_bounded_raw_escape_hatch() {
        let value = json!({
            "type": "future_tool",
            "new_option": {"nested": true},
        });
        let tool: OpenAiResponsesTool = serde_json::from_value(value.clone()).unwrap();
        assert!(matches!(tool, OpenAiResponsesTool::Raw(_)));
        assert_eq!(serde_json::to_value(tool).unwrap(), value);
    }

    #[test]
    fn secret_tool_debug_is_redacted() {
        let mcp = OpenAiMcpTool::new("calendar")
            .with_server_url("https://mcp.example.test")
            .with_authorization("oauth-secret")
            .with_approval(OpenAiMcpApproval::Never);
        let debug = format!("{mcp:?}");
        assert!(!debug.contains("oauth-secret"));
        assert!(debug.contains("redacted"));

        let shell = OpenAiShellTool {
            environment: Some(OpenAiShellEnvironment::ContainerAuto {
                file_ids: Vec::new(),
                memory_limit: None,
                network_policy: Some(OpenAiContainerNetworkPolicy::Allowlist {
                    allowed_domains: vec!["api.example.test".to_string()],
                    domain_secrets: vec![OpenAiDomainSecret {
                        domain: "api.example.test".to_string(),
                        name: "TOKEN".to_string(),
                        value: "shell-secret".to_string(),
                    }],
                }),
                skills: Vec::new(),
            }),
        };
        let debug = format!("{shell:?}");
        assert!(!debug.contains("shell-secret"));
        assert!(debug.contains("redacted"));
    }

    #[test]
    fn unit_tools_reject_silent_extra_fields() {
        let error = serde_json::from_value::<OpenAiResponsesTool>(json!({
            "type": "computer",
            "future": true,
        }))
        .unwrap_err();
        assert!(error.to_string().contains("only its type"));
    }

    #[test]
    fn raw_tool_rejects_oversized_payload() {
        let error = OpenAiResponsesTool::raw(json!({
            "type": "future_tool",
            "payload": "x".repeat(MAX_RAW_TOOL_BYTES),
        }))
        .unwrap_err();
        assert!(error.to_string().contains("1 MiB"));
    }
}
