//! Typed provider-native tools for the OpenAI Responses API.
//!
//! The shared language contract intentionally models caller-executed function tools only.
//! This module owns OpenAI-hosted tools and keeps an explicit bounded escape hatch for new
//! provider fields that have not yet earned a stable Rust type.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::Error as DeError;
use serde::ser::{Error as SerError, SerializeMap};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Number, Value};
use siumai_core::ProviderOptionError;

pub(crate) const MAX_RAW_TOOL_BYTES: usize = 1024 * 1024;
const MAX_TEXT_CHARS: usize = 16 * 1024;
const MAX_ID_CHARS: usize = 512;
const MAX_DOMAIN_COUNT: usize = 100;
const MAX_TOOL_NAMES: usize = 256;
const MAX_FILTER_DEPTH: usize = 16;
const MAX_FILTER_NODES: usize = 256;

/// OpenAI execution paths allowed to invoke a caller-owned tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiToolCaller {
    Direct,
    Programmatic,
}

/// A provider-native OpenAI Responses tool.
///
/// The enum is deliberately non-exhaustive. New OpenAI tool kinds can be used immediately
/// through [`OpenAiRawTool`] and can later become typed variants without changing the shared
/// `LanguageModel` contract.
#[non_exhaustive]
#[derive(Clone, PartialEq)]
pub enum OpenAiResponsesTool {
    WebSearch(OpenAiWebSearchTool),
    WebSearchPreview(OpenAiWebSearchPreviewTool),
    FileSearch(OpenAiFileSearchTool),
    CodeInterpreter(OpenAiCodeInterpreterTool),
    Computer,
    ComputerUsePreview(OpenAiComputerUsePreviewTool),
    Mcp(OpenAiMcpTool),
    ImageGeneration(OpenAiImageGenerationTool),
    LocalShell,
    Shell(OpenAiShellTool),
    ApplyPatch(OpenAiApplyPatchTool),
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
            Self::WebSearchPreview(value) => formatter
                .debug_tuple("WebSearchPreview")
                .field(value)
                .finish(),
            Self::FileSearch(value) => formatter.debug_tuple("FileSearch").field(value).finish(),
            Self::CodeInterpreter(value) => formatter
                .debug_tuple("CodeInterpreter")
                .field(value)
                .finish(),
            Self::Computer => formatter.write_str("Computer"),
            Self::ComputerUsePreview(value) => formatter
                .debug_tuple("ComputerUsePreview")
                .field(value)
                .finish(),
            Self::Mcp(value) => formatter.debug_tuple("Mcp").field(value).finish(),
            Self::ImageGeneration(value) => formatter
                .debug_tuple("ImageGeneration")
                .field(value)
                .finish(),
            Self::LocalShell => formatter.write_str("LocalShell"),
            Self::Shell(value) => formatter.debug_tuple("Shell").field(value).finish(),
            Self::ApplyPatch(value) => formatter.debug_tuple("ApplyPatch").field(value).finish(),
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

    /// Construct the unversioned preview web-search tool.
    pub fn web_search_preview() -> Self {
        Self::WebSearchPreview(OpenAiWebSearchPreviewTool::default())
    }

    /// Construct the `web_search_preview_2025_03_11` tool.
    pub fn web_search_preview_2025_03_11() -> Self {
        Self::WebSearchPreview(OpenAiWebSearchPreviewTool::versioned_2025_03_11())
    }

    /// Construct the default file-search tool.
    pub fn file_search(vector_store_ids: Vec<String>) -> Self {
        Self::FileSearch(OpenAiFileSearchTool::new(vector_store_ids))
    }

    /// Construct the default code-interpreter tool.
    pub fn code_interpreter() -> Self {
        Self::CodeInterpreter(OpenAiCodeInterpreterTool::default())
    }

    /// Construct the preview computer-use tool with an explicit display configuration.
    pub fn computer_use_preview(
        display_width: u32,
        display_height: u32,
        environment: OpenAiComputerEnvironment,
    ) -> Self {
        Self::ComputerUsePreview(OpenAiComputerUsePreviewTool::new(
            display_width,
            display_height,
            environment,
        ))
    }

    /// Construct the default apply-patch tool.
    pub fn apply_patch() -> Self {
        Self::ApplyPatch(OpenAiApplyPatchTool::default())
    }

    /// Construct an MCP tool from a validated builder shape.
    pub fn mcp(tool: OpenAiMcpTool) -> Self {
        Self::Mcp(tool)
    }

    /// Construct the default image-generation tool.
    pub fn image_generation() -> Self {
        Self::ImageGeneration(OpenAiImageGenerationTool::default())
    }

    /// Construct the provider-native local-shell tool.
    pub const fn local_shell() -> Self {
        Self::LocalShell
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

    /// Construct a current or future tool from its complete JSON object.
    ///
    /// This escape hatch also preserves newly added fields on a known tool kind
    /// before the typed variant adopts them.
    pub fn raw(value: Value) -> Result<Self, ProviderOptionError> {
        Ok(Self::Raw(OpenAiRawTool::new(value)?))
    }

    pub(crate) fn validate(&self) -> Result<(), ProviderOptionError> {
        match self {
            Self::WebSearch(value) => value.validate(),
            Self::WebSearchPreview(value) => value.validate(),
            Self::FileSearch(value) => value.validate(),
            Self::CodeInterpreter(value) => value.validate(),
            Self::Computer | Self::LocalShell | Self::ProgrammaticToolCalling => Ok(()),
            Self::ComputerUsePreview(_) => Ok(()),
            Self::ApplyPatch(value) => value.validate(),
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

impl From<OpenAiWebSearchPreviewTool> for OpenAiResponsesTool {
    fn from(value: OpenAiWebSearchPreviewTool) -> Self {
        Self::WebSearchPreview(value)
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

impl From<OpenAiComputerUsePreviewTool> for OpenAiResponsesTool {
    fn from(value: OpenAiComputerUsePreviewTool) -> Self {
        Self::ComputerUsePreview(value)
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

impl From<OpenAiApplyPatchTool> for OpenAiResponsesTool {
    fn from(value: OpenAiApplyPatchTool) -> Self {
        Self::ApplyPatch(value)
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
            Self::WebSearchPreview(tool) => tagged(tool.version.as_str(), tool),
            Self::FileSearch(tool) => tagged("file_search", tool),
            Self::CodeInterpreter(tool) => tagged("code_interpreter", tool),
            Self::Computer => Ok(unit_tagged("computer")),
            Self::ComputerUsePreview(tool) => tagged("computer_use_preview", tool),
            Self::Mcp(tool) => tagged("mcp", tool),
            Self::ImageGeneration(tool) => tagged("image_generation", tool),
            Self::LocalShell => Ok(unit_tagged("local_shell")),
            Self::Shell(tool) => tagged("shell", tool),
            Self::ApplyPatch(tool) => tagged("apply_patch", tool),
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
            "web_search" => {
                decode_known_or_raw(value, |value| Ok(Self::WebSearch(decode_tagged(value)?)))?
            }
            "web_search_preview" => decode_known_or_raw(value, |value| {
                Ok(Self::WebSearchPreview(
                    decode_tagged::<OpenAiWebSearchPreviewToolWire, D::Error>(value)?
                        .into_tool(OpenAiWebSearchPreviewVersion::Unversioned),
                ))
            })?,
            "web_search_preview_2025_03_11" => decode_known_or_raw(value, |value| {
                Ok(Self::WebSearchPreview(
                    decode_tagged::<OpenAiWebSearchPreviewToolWire, D::Error>(value)?
                        .into_tool(OpenAiWebSearchPreviewVersion::V20250311),
                ))
            })?,
            "file_search" => {
                decode_known_or_raw(value, |value| Ok(Self::FileSearch(decode_tagged(value)?)))?
            }
            "code_interpreter" => decode_known_or_raw(value, |value| {
                Ok(Self::CodeInterpreter(decode_tagged(value)?))
            })?,
            "computer" => decode_known_or_raw(value, |value| {
                decode_unit::<D::Error>(&value)?;
                Ok(Self::Computer)
            })?,
            "computer_use_preview" => decode_known_or_raw(value, |value| {
                Ok(Self::ComputerUsePreview(decode_tagged(value)?))
            })?,
            "mcp" => decode_known_or_raw(value, |value| Ok(Self::Mcp(decode_tagged(value)?)))?,
            "image_generation" => decode_known_or_raw(value, |value| {
                Ok(Self::ImageGeneration(decode_tagged(value)?))
            })?,
            "local_shell" => decode_known_or_raw(value, |value| {
                decode_unit::<D::Error>(&value)?;
                Ok(Self::LocalShell)
            })?,
            "shell" => decode_known_or_raw(value, |value| Ok(Self::Shell(decode_tagged(value)?)))?,
            "apply_patch" => {
                decode_known_or_raw(value, |value| Ok(Self::ApplyPatch(decode_tagged(value)?)))?
            }
            "tool_search" => {
                decode_known_or_raw(value, |value| Ok(Self::ToolSearch(decode_tagged(value)?)))?
            }
            "programmatic_tool_calling" => decode_known_or_raw(value, |value| {
                decode_unit::<D::Error>(&value)?;
                Ok(Self::ProgrammaticToolCalling)
            })?,
            "custom" => {
                decode_known_or_raw(value, |value| Ok(Self::Custom(decode_tagged(value)?)))?
            }
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
pub struct OpenAiWebSearchTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_web_access: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<OpenAiWebSearchFilters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<OpenAiWebSearchContextSize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_token_budget: Option<OpenAiWebSearchReturnTokenBudget>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub search_content_types: Vec<OpenAiWebSearchContentType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_settings: Option<OpenAiWebSearchImageSettings>,
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
        let mut seen = std::collections::BTreeSet::new();
        for content_type in &self.search_content_types {
            if !seen.insert(*content_type) {
                return Err(rejected(
                    "tools.search_content_types",
                    "search content types must be unique",
                ));
            }
        }
        if let Some(settings) = &self.image_settings
            && settings.max_results == Some(0)
        {
            return Err(rejected(
                "tools.image_settings.max_results",
                "image search max_results must be greater than zero",
            ));
        }
        Ok(())
    }
}

/// Wire version selected for the preview web-search tool.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OpenAiWebSearchPreviewVersion {
    #[default]
    Unversioned,
    V20250311,
}

impl OpenAiWebSearchPreviewVersion {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Unversioned => "web_search_preview",
            Self::V20250311 => "web_search_preview_2025_03_11",
        }
    }
}

/// Preview web-search hosted tool controls.
///
/// This is an options payload rather than a standalone wire object. Serialize and deserialize the
/// enclosing [`OpenAiResponsesTool`] when the versioned `type` discriminator must be preserved.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpenAiWebSearchPreviewTool {
    #[serde(skip)]
    version: OpenAiWebSearchPreviewVersion,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub search_content_types: Vec<OpenAiWebSearchContentType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<OpenAiWebSearchContextSize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<OpenAiApproximateLocation>,
}

#[derive(Deserialize)]
struct OpenAiWebSearchPreviewToolWire {
    #[serde(default)]
    search_content_types: Vec<OpenAiWebSearchContentType>,
    #[serde(default)]
    search_context_size: Option<OpenAiWebSearchContextSize>,
    #[serde(default)]
    user_location: Option<OpenAiApproximateLocation>,
}

impl OpenAiWebSearchPreviewToolWire {
    fn into_tool(self, version: OpenAiWebSearchPreviewVersion) -> OpenAiWebSearchPreviewTool {
        OpenAiWebSearchPreviewTool {
            version,
            search_content_types: self.search_content_types,
            search_context_size: self.search_context_size,
            user_location: self.user_location,
        }
    }
}

impl Default for OpenAiWebSearchPreviewTool {
    fn default() -> Self {
        Self {
            version: OpenAiWebSearchPreviewVersion::Unversioned,
            search_content_types: Vec::new(),
            search_context_size: None,
            user_location: None,
        }
    }
}

impl OpenAiWebSearchPreviewTool {
    pub fn versioned_2025_03_11() -> Self {
        Self {
            version: OpenAiWebSearchPreviewVersion::V20250311,
            ..Self::default()
        }
    }

    pub const fn version(&self) -> OpenAiWebSearchPreviewVersion {
        self.version
    }

    pub fn with_search_context_size(mut self, value: OpenAiWebSearchContextSize) -> Self {
        self.search_context_size = Some(value);
        self
    }

    pub fn with_user_location(mut self, value: OpenAiApproximateLocation) -> Self {
        self.user_location = Some(value);
        self
    }

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if let Some(location) = &self.user_location {
            location.validate()?;
        }
        Ok(())
    }
}

/// Content kinds returned by web search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiWebSearchContentType {
    Text,
    Image,
}

/// Image-result controls for web search.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiWebSearchImageSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_results: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caption: Option<bool>,
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
pub struct OpenAiApproximateLocation {
    #[serde(default = "approximate_location_type")]
    r#type: String,
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

/// Preview computer-use display and environment controls.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiComputerUsePreviewTool {
    pub display_height: u32,
    pub display_width: u32,
    pub environment: OpenAiComputerEnvironment,
}

impl OpenAiComputerUsePreviewTool {
    pub const fn new(
        display_width: u32,
        display_height: u32,
        environment: OpenAiComputerEnvironment,
    ) -> Self {
        Self {
            display_height,
            display_width,
            environment,
        }
    }
}

/// Computer environment understood by the preview computer-use tool.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiComputerEnvironment {
    Windows,
    Mac,
    Linux,
    Ubuntu,
    Browser,
}

/// File-search tool filter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum OpenAiFileSearchFilter {
    Eq {
        key: String,
        value: OpenAiFileSearchFilterScalar,
    },
    Ne {
        key: String,
        value: OpenAiFileSearchFilterScalar,
    },
    Gt {
        key: String,
        value: OpenAiFileSearchFilterScalar,
    },
    Gte {
        key: String,
        value: OpenAiFileSearchFilterScalar,
    },
    Lt {
        key: String,
        value: OpenAiFileSearchFilterScalar,
    },
    Lte {
        key: String,
        value: OpenAiFileSearchFilterScalar,
    },
    In {
        key: String,
        value: OpenAiFileSearchFilterList,
    },
    Nin {
        key: String,
        value: OpenAiFileSearchFilterList,
    },
    And {
        filters: Vec<Self>,
    },
    Or {
        filters: Vec<Self>,
    },
}

/// Scalar accepted by a file-search comparison filter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAiFileSearchFilterScalar {
    String(String),
    Number(Number),
    Boolean(bool),
}

/// Homogeneous values accepted by an `in` or `nin` file-search filter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAiFileSearchFilterList {
    Strings(Vec<String>),
    Numbers(Vec<Number>),
}

/// File-search ranking options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenAiFileSearchRankingOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hybrid_search: Option<OpenAiFileSearchHybridSearch>,
}

/// Hybrid ranking weights for file search.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OpenAiFileSearchHybridSearch {
    pub embedding_weight: f64,
    pub text_weight: f64,
}

/// File-search hosted tool controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        if self
            .max_num_results
            .is_some_and(|value| !(1..=50).contains(&value))
        {
            return Err(rejected(
                "tools.max_num_results",
                "file-search max_num_results must be between 1 and 50",
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
            if let Some(hybrid) = ranking.hybrid_search
                && (!hybrid.embedding_weight.is_finite()
                    || !hybrid.text_weight.is_finite()
                    || hybrid.embedding_weight < 0.0
                    || hybrid.text_weight < 0.0
                    || (hybrid.embedding_weight == 0.0 && hybrid.text_weight == 0.0))
            {
                return Err(rejected(
                    "tools.ranking_options.hybrid_search",
                    "hybrid search weights must be finite, non-negative, and not both zero",
                ));
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
pub struct OpenAiCodeInterpreterAutoContainer {
    #[serde(rename = "type", default = "auto_container_type")]
    r#type: String,
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
pub struct OpenAiCodeInterpreterTool {
    pub container: OpenAiCodeInterpreterContainer,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_callers: Vec<OpenAiToolCaller>,
}

impl Default for OpenAiCodeInterpreterTool {
    fn default() -> Self {
        Self {
            container: OpenAiCodeInterpreterContainer::Auto(
                OpenAiCodeInterpreterAutoContainer::default(),
            ),
            allowed_callers: Vec::new(),
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
        validate_callers(&self.allowed_callers)?;
        Ok(())
    }
}

/// Image-generation hosted tool controls.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenAiImageGenerationTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<OpenAiImageAction>,
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
        if let Some(mask) = &self.input_image_mask {
            match (&mask.file_id, &mask.image_url) {
                (Some(file_id), None) => {
                    validate_text("tools.input_image_mask.file_id", file_id, MAX_ID_CHARS)?;
                }
                (None, Some(image_url)) => {
                    validate_text(
                        "tools.input_image_mask.image_url",
                        image_url,
                        MAX_RAW_TOOL_BYTES,
                    )?;
                }
                _ => {
                    return Err(rejected(
                        "tools.input_image_mask",
                        "image input_image_mask requires exactly one file_id or image_url",
                    ));
                }
            }
        }
        if let Some(size) = &self.size {
            OpenAiImageSize::new(size.as_str())?;
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

/// Image-generation operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiImageAction {
    Generate,
    Edit,
    Auto,
}

/// Image input fidelity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiImageInputFidelity {
    Low,
    High,
}

/// Image input mask.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenAiImageInputMask {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

impl fmt::Debug for OpenAiImageInputMask {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiImageInputMask")
            .field("file_id", &self.file_id)
            .field("image_url", &self.image_url.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

/// Image moderation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiImageModeration {
    Auto,
    Low,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OpenAiImageSize(String);

impl OpenAiImageSize {
    pub const AUTO: &str = "auto";
    pub const SQUARE: &str = "1024x1024";
    pub const PORTRAIT: &str = "1024x1536";
    pub const LANDSCAPE: &str = "1536x1024";

    pub fn new(value: impl Into<String>) -> Result<Self, ProviderOptionError> {
        let value = value.into();
        validate_text("tools.size", &value, 64)?;
        if !value.eq_ignore_ascii_case(Self::AUTO) && !value.contains('x') {
            return Err(rejected(
                "tools.size",
                "image size must be auto or a provider-supported dimension",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn auto() -> Self {
        Self(Self::AUTO.to_string())
    }

    pub fn square() -> Self {
        Self(Self::SQUARE.to_string())
    }

    pub fn portrait() -> Self {
        Self(Self::PORTRAIT.to_string())
    }

    pub fn landscape() -> Self {
        Self(Self::LANDSCAPE.to_string())
    }
}

/// MCP endpoint selected for one hosted MCP tool.
#[derive(Clone, PartialEq)]
pub enum OpenAiMcpEndpoint {
    ServerUrl { server_url: String },
    Connector { connector_id: String },
    Tunnel { tunnel_id: String },
}

impl fmt::Debug for OpenAiMcpEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ServerUrl { .. } => formatter.write_str("ServerUrl(<redacted>)"),
            Self::Connector { .. } => formatter.write_str("Connector(<redacted>)"),
            Self::Tunnel { .. } => formatter.write_str("Tunnel(<redacted>)"),
        }
    }
}

/// MCP server tool controls.
#[derive(Clone, PartialEq)]
pub struct OpenAiMcpTool {
    pub server_label: String,
    pub endpoint: OpenAiMcpEndpoint,
    pub allowed_tools: Option<OpenAiMcpAllowedTools>,
    pub allowed_callers: Vec<OpenAiToolCaller>,
    pub authorization: Option<String>,
    pub headers: BTreeMap<String, String>,
    pub require_approval: Option<OpenAiMcpApproval>,
    pub server_description: Option<String>,
    pub defer_loading: Option<bool>,
}

impl Serialize for OpenAiMcpTool {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("server_label", &self.server_label)?;
        match &self.endpoint {
            OpenAiMcpEndpoint::ServerUrl { server_url } => {
                map.serialize_entry("server_url", server_url)?;
            }
            OpenAiMcpEndpoint::Connector { connector_id } => {
                map.serialize_entry("connector_id", connector_id)?;
            }
            OpenAiMcpEndpoint::Tunnel { tunnel_id } => {
                map.serialize_entry("tunnel_id", tunnel_id)?;
            }
        }
        if let Some(value) = &self.allowed_tools {
            map.serialize_entry("allowed_tools", value)?;
        }
        if !self.allowed_callers.is_empty() {
            map.serialize_entry("allowed_callers", &self.allowed_callers)?;
        }
        if let Some(value) = &self.authorization {
            map.serialize_entry("authorization", value)?;
        }
        if !self.headers.is_empty() {
            map.serialize_entry("headers", &self.headers)?;
        }
        if let Some(value) = &self.require_approval {
            map.serialize_entry("require_approval", value)?;
        }
        if let Some(value) = &self.server_description {
            map.serialize_entry("server_description", value)?;
        }
        if let Some(value) = self.defer_loading {
            map.serialize_entry("defer_loading", &value)?;
        }
        map.end()
    }
}

#[derive(Deserialize)]
struct OpenAiMcpToolWire {
    server_label: String,
    allowed_tools: Option<OpenAiMcpAllowedTools>,
    #[serde(default)]
    allowed_callers: Vec<OpenAiToolCaller>,
    authorization: Option<String>,
    connector_id: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    require_approval: Option<OpenAiMcpApproval>,
    server_description: Option<String>,
    server_url: Option<String>,
    tunnel_id: Option<String>,
    defer_loading: Option<bool>,
}

impl<'de> Deserialize<'de> for OpenAiMcpTool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = OpenAiMcpToolWire::deserialize(deserializer)?;
        let endpoint = match (wire.server_url, wire.connector_id, wire.tunnel_id) {
            (Some(server_url), None, None) => OpenAiMcpEndpoint::ServerUrl { server_url },
            (None, Some(connector_id), None) => OpenAiMcpEndpoint::Connector { connector_id },
            (None, None, Some(tunnel_id)) => OpenAiMcpEndpoint::Tunnel { tunnel_id },
            _ => {
                return Err(D::Error::custom(
                    "MCP tools require exactly one server_url, connector_id, or tunnel_id",
                ));
            }
        };
        let value = Self {
            server_label: wire.server_label,
            endpoint,
            allowed_tools: wire.allowed_tools,
            allowed_callers: wire.allowed_callers,
            authorization: wire.authorization,
            headers: wire.headers,
            require_approval: wire.require_approval,
            server_description: wire.server_description,
            defer_loading: wire.defer_loading,
        };
        value.validate().map_err(D::Error::custom)?;
        Ok(value)
    }
}

impl fmt::Debug for OpenAiMcpTool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let endpoint_kind = match self.endpoint {
            OpenAiMcpEndpoint::ServerUrl { .. } => "server_url",
            OpenAiMcpEndpoint::Connector { .. } => "connector",
            OpenAiMcpEndpoint::Tunnel { .. } => "tunnel",
        };
        let (allowed_tools_kind, allowed_tool_count) = match &self.allowed_tools {
            None => (None, 0),
            Some(OpenAiMcpAllowedTools::Names(names)) => (Some("names"), names.len()),
            Some(OpenAiMcpAllowedTools::Filter { tool_names, .. }) => {
                (Some("filter"), tool_names.as_ref().map_or(0, Vec::len))
            }
        };
        formatter
            .debug_struct("OpenAiMcpTool")
            .field("server_label_bytes", &self.server_label.len())
            .field("endpoint_kind", &endpoint_kind)
            .field("allowed_tools_kind", &allowed_tools_kind)
            .field("allowed_tool_count", &allowed_tool_count)
            .field("allowed_caller_count", &self.allowed_callers.len())
            .field("authorization_present", &self.authorization.is_some())
            .field("header_count", &self.headers.len())
            .field("approval_present", &self.require_approval.is_some())
            .field(
                "server_description_present",
                &self.server_description.is_some(),
            )
            .field("defer_loading", &self.defer_loading)
            .field("data", &"<redacted>")
            .finish()
    }
}

impl OpenAiMcpTool {
    pub fn new(server_label: impl Into<String>, endpoint: OpenAiMcpEndpoint) -> Self {
        Self {
            server_label: server_label.into(),
            endpoint,
            allowed_tools: None,
            allowed_callers: Vec::new(),
            authorization: None,
            headers: BTreeMap::new(),
            require_approval: None,
            server_description: None,
            defer_loading: None,
        }
    }

    pub fn server(server_label: impl Into<String>, server_url: impl Into<String>) -> Self {
        Self::new(
            server_label,
            OpenAiMcpEndpoint::ServerUrl {
                server_url: server_url.into(),
            },
        )
    }

    pub fn connector(server_label: impl Into<String>, connector_id: impl Into<String>) -> Self {
        Self::new(
            server_label,
            OpenAiMcpEndpoint::Connector {
                connector_id: connector_id.into(),
            },
        )
    }

    pub fn tunnel(server_label: impl Into<String>, tunnel_id: impl Into<String>) -> Self {
        Self::new(
            server_label,
            OpenAiMcpEndpoint::Tunnel {
                tunnel_id: tunnel_id.into(),
            },
        )
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
        match &self.endpoint {
            OpenAiMcpEndpoint::ServerUrl { server_url } => {
                validate_text("tools.server_url", server_url, 2048)?;
            }
            OpenAiMcpEndpoint::Connector { connector_id } => {
                validate_text("tools.connector_id", connector_id, MAX_ID_CHARS)?;
            }
            OpenAiMcpEndpoint::Tunnel { tunnel_id } => {
                validate_text("tools.tunnel_id", tunnel_id, MAX_ID_CHARS)?;
            }
        }
        validate_callers(&self.allowed_callers)?;
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
    Filter {
        always: Option<OpenAiMcpApprovalFilter>,
        never: Option<OpenAiMcpApprovalFilter>,
    },
}

impl Serialize for OpenAiMcpApproval {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Always => serializer.serialize_str("always"),
            Self::Never => serializer.serialize_str("never"),
            Self::Filter { always, never } => {
                let mut object = Map::new();
                if let Some(always) = always {
                    object.insert(
                        "always".to_string(),
                        serde_json::to_value(always).map_err(serde::ser::Error::custom)?,
                    );
                }
                if let Some(never) = never {
                    object.insert(
                        "never".to_string(),
                        serde_json::to_value(never).map_err(serde::ser::Error::custom)?,
                    );
                }
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
            Value::Object(mut object)
                if object.contains_key("always") || object.contains_key("never") =>
            {
                let always = object
                    .remove("always")
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(D::Error::custom)?;
                let never = object
                    .remove("never")
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(D::Error::custom)?;
                Ok(Self::Filter { always, never })
            }
            _ => Err(D::Error::custom(
                "MCP approval must be always, never, or an always/never filter object",
            )),
        }
    }
}

/// Tool filter for an MCP approval policy branch.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenAiMcpApprovalFilter {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_only: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_names: Vec<String>,
}

impl OpenAiMcpApproval {
    fn validate(&self) -> Result<(), ProviderOptionError> {
        if let Self::Filter { always, never } = self {
            if always.is_none() && never.is_none() {
                return Err(rejected(
                    "tools.require_approval",
                    "MCP approval filter requires always or never",
                ));
            }
            for filter in [always.as_ref(), never.as_ref()].into_iter().flatten() {
                validate_tool_names(&filter.tool_names)?;
            }
        }
        Ok(())
    }
}

/// OpenAI namespace metadata attached to an existing caller-owned function tool.
///
/// The Responses request encoder groups functions with identical namespace metadata into one
/// provider-native `namespace` object. Function definitions remain owned by the shared `ToolSpec`
/// contract instead of being duplicated here.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiToolNamespace {
    pub name: String,
    pub description: String,
}

impl fmt::Debug for OpenAiToolNamespace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiToolNamespace")
            .field("name_bytes", &self.name.len())
            .field("description_bytes", &self.description.len())
            .field("data", &"<redacted>")
            .finish()
    }
}

impl OpenAiToolNamespace {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), ProviderOptionError> {
        if self.name.is_empty() {
            return Err(rejected(
                "function_tool_options.namespace.name",
                "namespace name must be non-empty",
            ));
        }
        validate_control_free(
            "function_tool_options.namespace.name",
            &self.name,
            MAX_ID_CHARS,
        )?;
        validate_control_free(
            "function_tool_options.namespace.description",
            &self.description,
            MAX_TEXT_CHARS,
        )
    }
}

/// Tool-search controls.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
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
    pub fn server() -> Self {
        Self {
            execution: Some(OpenAiToolSearchExecution::Server),
            description: None,
            parameters: None,
        }
    }

    pub fn client(description: impl Into<String>, parameters: Value) -> Self {
        Self {
            execution: Some(OpenAiToolSearchExecution::Client),
            description: Some(description.into()),
            parameters: Some(parameters),
        }
    }

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if let Some(description) = &self.description {
            validate_text("tools.description", description, MAX_TEXT_CHARS)?;
        }
        match self.execution {
            Some(OpenAiToolSearchExecution::Client) => {
                if self.description.is_none() || self.parameters.is_none() {
                    return Err(rejected(
                        "tools.tool_search",
                        "client tool search requires description and parameters",
                    ));
                }
            }
            Some(OpenAiToolSearchExecution::Server) | None => {
                if self.description.is_some() || self.parameters.is_some() {
                    return Err(rejected(
                        "tools.tool_search",
                        "description and parameters are only valid for client tool search",
                    ));
                }
            }
        }
        if let Some(parameters) = &self.parameters
            && !parameters.is_object()
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
pub struct OpenAiCustomTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<OpenAiCustomToolFormat>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_callers: Vec<OpenAiToolCaller>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
}

impl OpenAiCustomTool {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            format: None,
            allowed_callers: Vec::new(),
            defer_loading: None,
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
        validate_callers(&self.allowed_callers)?;
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
pub struct OpenAiShellTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<OpenAiShellEnvironment>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_callers: Vec<OpenAiToolCaller>,
}

impl fmt::Debug for OpenAiShellTool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiShellTool")
            .field("environment", &self.environment)
            .field("allowed_callers", &self.allowed_callers)
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
                .field("file_ids", &format_args!("<{} entries>", file_ids.len()))
                .field("memory_limit", memory_limit)
                .field("network_policy", network_policy)
                .field("skills", &format_args!("<{} entries>", skills.len()))
                .finish(),
            Self::ContainerReference { .. } => formatter
                .debug_struct("ContainerReference")
                .field("container_id", &"<redacted>")
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
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiLocalShellSkill {
    pub name: String,
    pub description: String,
    pub path: String,
}

impl fmt::Debug for OpenAiLocalShellSkill {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiLocalShellSkill")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("path", &"<redacted>")
            .finish()
    }
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
pub struct OpenAiInlineSkillSource {
    #[serde(rename = "type")]
    r#type: String,
    media_type: String,
    data: String,
}

impl OpenAiInlineSkillSource {
    pub fn zip_base64(data: impl Into<String>) -> Self {
        Self {
            r#type: "base64".to_string(),
            media_type: "application/zip".to_string(),
            data: data.into(),
        }
    }

    pub fn data(&self) -> &str {
        &self.data
    }
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
        validate_callers(&self.allowed_callers)?;
        Ok(())
    }
}

/// Apply-patch hosted tool controls.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenAiApplyPatchTool {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_callers: Vec<OpenAiToolCaller>,
}

impl OpenAiApplyPatchTool {
    fn validate(&self) -> Result<(), ProviderOptionError> {
        validate_callers(&self.allowed_callers)
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

fn decode_known_or_raw<E: DeError>(
    value: Value,
    decode: impl Fn(Value) -> Result<OpenAiResponsesTool, E>,
) -> Result<OpenAiResponsesTool, E> {
    let known = decode(value.clone())?;
    known.validate().map_err(E::custom)?;
    let known_wire = serde_json::to_value(&known).map_err(E::custom)?;
    if known_wire == value {
        Ok(known)
    } else {
        OpenAiResponsesTool::raw(value).map_err(E::custom)
    }
}

fn decode_unit<E: DeError>(value: &Value) -> Result<(), E> {
    if value.as_object().is_some() {
        Ok(())
    } else {
        Err(E::custom("unit OpenAI tool must be a JSON object"))
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
            match value {
                OpenAiFileSearchFilterScalar::String(value) => {
                    validate_text("tools.filters.value", value, 1024)?;
                }
                OpenAiFileSearchFilterScalar::Number(_)
                | OpenAiFileSearchFilterScalar::Boolean(_) => {}
            }
        }
        OpenAiFileSearchFilter::In { key, value } | OpenAiFileSearchFilter::Nin { key, value } => {
            validate_text("tools.filters.key", key, 256)?;
            match value {
                OpenAiFileSearchFilterList::Strings(values) => {
                    if values.is_empty() {
                        return Err(rejected(
                            "tools.filters.value",
                            "filter value list cannot be empty",
                        ));
                    }
                    for item in values {
                        validate_text("tools.filters.value", item, 1024)?;
                    }
                }
                OpenAiFileSearchFilterList::Numbers(values) if values.is_empty() => {
                    return Err(rejected(
                        "tools.filters.value",
                        "filter value list cannot be empty",
                    ));
                }
                OpenAiFileSearchFilterList::Numbers(_) => {}
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

fn validate_callers(callers: &[OpenAiToolCaller]) -> Result<(), ProviderOptionError> {
    let mut seen = std::collections::BTreeSet::new();
    for caller in callers {
        if !seen.insert(*caller) {
            return Err(rejected(
                "tools.allowed_callers",
                "allowed callers must be unique",
            ));
        }
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
            search_content_types: vec![
                OpenAiWebSearchContentType::Text,
                OpenAiWebSearchContentType::Image,
            ],
            image_settings: Some(OpenAiWebSearchImageSettings {
                max_results: Some(4),
                caption: Some(true),
            }),
            user_location: Some(OpenAiApproximateLocation::new()),
        });
        let value = serde_json::to_value(&tool).unwrap();
        assert_eq!(value["type"], "web_search");
        assert_eq!(value["filters"]["allowed_domains"][0], "example.com");
        assert_eq!(value["return_token_budget"], "unlimited");
        assert_eq!(value["image_settings"]["max_results"], 4);
        assert_eq!(
            serde_json::from_value::<OpenAiResponsesTool>(value).unwrap(),
            tool
        );
    }

    #[test]
    fn bounded_hosted_tool_increment_has_exact_wire_shapes() {
        let computer = OpenAiResponsesTool::computer_use_preview(
            1440,
            900,
            OpenAiComputerEnvironment::Browser,
        );
        let computer_wire = computer.clone().into_value().unwrap();
        assert_eq!(
            computer_wire,
            json!({
                "type": "computer_use_preview",
                "display_height": 900,
                "display_width": 1440,
                "environment": "browser",
            })
        );
        assert_eq!(
            serde_json::from_value::<OpenAiResponsesTool>(computer_wire).unwrap(),
            computer
        );

        let local_shell = OpenAiResponsesTool::local_shell();
        let local_shell_wire = local_shell.clone().into_value().unwrap();
        assert_eq!(local_shell_wire, json!({"type": "local_shell"}));
        assert_eq!(
            serde_json::from_value::<OpenAiResponsesTool>(local_shell_wire).unwrap(),
            local_shell
        );

        let namespace = OpenAiToolNamespace::new("crm", "Customer relationship tools");
        namespace.validate().unwrap();
        assert_eq!(
            serde_json::to_value(&namespace).unwrap(),
            json!({
                "name": "crm",
                "description": "Customer relationship tools",
            })
        );
        assert_eq!(
            serde_json::from_value::<OpenAiToolNamespace>(
                serde_json::to_value(&namespace).unwrap()
            )
            .unwrap(),
            namespace
        );

        let mut preview_options = OpenAiWebSearchPreviewTool::versioned_2025_03_11()
            .with_search_context_size(OpenAiWebSearchContextSize::Low)
            .with_user_location(OpenAiApproximateLocation {
                r#type: approximate_location_type(),
                country: Some("US".to_string()),
                city: None,
                region: None,
                timezone: None,
            });
        preview_options.search_content_types = vec![OpenAiWebSearchContentType::Image];
        let preview = OpenAiResponsesTool::WebSearchPreview(preview_options);
        let preview_wire = preview.clone().into_value().unwrap();
        assert_eq!(
            preview_wire,
            json!({
                "type": "web_search_preview_2025_03_11",
                "search_content_types": ["image"],
                "search_context_size": "low",
                "user_location": {"type": "approximate", "country": "US"},
            })
        );
        assert_eq!(
            serde_json::from_value::<OpenAiResponsesTool>(preview_wire).unwrap(),
            preview
        );

        assert_eq!(
            OpenAiResponsesTool::web_search_preview()
                .into_value()
                .unwrap(),
            json!({"type": "web_search_preview"})
        );
    }

    #[test]
    fn unknown_tool_round_trips_through_bounded_raw_escape_hatch() {
        for value in [
            json!({
                "type": "future_tool",
                "new_option": {"nested": true},
            }),
            json!({
                "type": "local_shell",
                "future_option": true,
            }),
        ] {
            let tool = OpenAiResponsesTool::raw(value.clone()).unwrap();
            assert!(matches!(tool, OpenAiResponsesTool::Raw(_)));
            assert_eq!(serde_json::to_value(tool).unwrap(), value);
        }
    }

    #[test]
    fn known_tool_with_additive_field_deserializes_as_bounded_raw() {
        let value = json!({
            "type": "web_search",
            "future_option": {"mode": "next"},
        });

        let tool = serde_json::from_value::<OpenAiResponsesTool>(value.clone()).unwrap();

        assert!(matches!(tool, OpenAiResponsesTool::Raw(_)));
        assert_eq!(serde_json::to_value(tool).unwrap(), value);
    }

    #[test]
    fn known_tool_with_nested_additive_field_deserializes_as_bounded_raw() {
        let value = json!({
            "type": "file_search",
            "vector_store_ids": ["vs_123"],
            "ranking_options": {
                "ranker": "auto",
                "score_threshold": 0.5,
                "future": {"mode": "next"},
            },
        });

        let tool = serde_json::from_value::<OpenAiResponsesTool>(value.clone()).unwrap();

        assert!(matches!(tool, OpenAiResponsesTool::Raw(_)));
        assert_eq!(serde_json::to_value(tool).unwrap(), value);
    }

    #[test]
    fn known_tool_additive_fields_do_not_bypass_typed_validation() {
        let invalid_cases = [
            (
                json!({
                    "type": "file_search",
                    "vector_store_ids": ["vs_123"],
                    "max_num_results": 51,
                    "future_option": true,
                }),
                "between 1 and 50",
            ),
            (
                json!({
                    "type": "file_search",
                    "vector_store_ids": ["vs_123"],
                    "ranking_options": {
                        "score_threshold": 2.0,
                        "future_option": true,
                    },
                }),
                "between 0 and 1",
            ),
            (
                json!({
                    "type": "mcp",
                    "server_label": "",
                    "server_url": "https://mcp.example.test",
                    "future_option": true,
                }),
                "tools.server_label",
            ),
        ];

        for (value, expected) in invalid_cases {
            let error = serde_json::from_value::<OpenAiResponsesTool>(value).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn known_tool_structure_errors_stay_invalid_with_additive_fields() {
        for value in [
            json!({
                "type": "file_search",
                "future_option": true,
            }),
            json!({
                "type": "file_search",
                "vector_store_ids": ["vs_123"],
                "ranking_options": [],
                "future_option": true,
            }),
            json!({
                "type": "file_search",
                "vector_store_ids": ["vs_123"],
                "filters": {
                    "type": "future_comparison",
                    "future_option": true,
                },
                "future_option": true,
            }),
        ] {
            assert!(serde_json::from_value::<OpenAiResponsesTool>(value).is_err());
        }
    }

    #[test]
    fn secret_tool_debug_is_redacted() {
        let mut mcp = OpenAiMcpTool::server(
            "server-label-secret",
            "https://mcp.example.test/connect?signature=url-secret",
        )
        .with_authorization("oauth-secret")
        .with_approval(OpenAiMcpApproval::Never);
        mcp.allowed_tools = Some(OpenAiMcpAllowedTools::Names(vec![
            "tenant-tool-name-secret".to_string(),
        ]));
        mcp.headers.insert(
            "tenant-header-name-secret".to_string(),
            "nested-header-secret".to_string(),
        );
        let debug = format!("{mcp:?}");
        for secret in [
            "server-label-secret",
            "tenant-tool-name-secret",
            "tenant-header-name-secret",
            "oauth-secret",
            "url-secret",
            "nested-header-secret",
        ] {
            assert!(!debug.contains(secret));
        }
        assert!(debug.contains("redacted"));
        let wire = serde_json::to_value(&mcp).unwrap();
        assert_eq!(
            wire["server_url"],
            "https://mcp.example.test/connect?signature=url-secret"
        );
        assert_eq!(serde_json::from_value::<OpenAiMcpTool>(wire).unwrap(), mcp);

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
            allowed_callers: Vec::new(),
        };
        let debug = format!("{shell:?}");
        assert!(!debug.contains("shell-secret"));
        assert!(debug.contains("redacted"));

        let mask = OpenAiImageInputMask {
            file_id: None,
            image_url: Some("data:image/png;base64,mask-secret".to_string()),
        };
        assert!(!format!("{mask:?}").contains("mask-secret"));

        let skill = OpenAiLocalShellSkill {
            name: "local".to_string(),
            description: "Local skill".to_string(),
            path: "/private/tenant/path-secret".to_string(),
        };
        assert!(!format!("{skill:?}").contains("path-secret"));

        let namespace =
            OpenAiToolNamespace::new("namespace-name-secret", "namespace-description-secret");
        let debug = format!("{namespace:?}");
        assert!(!debug.contains("namespace-name-secret"));
        assert!(!debug.contains("namespace-description-secret"));
        assert!(debug.contains("redacted"));

        let invalid = OpenAiToolNamespace::new("private\nnamespace", "description-secret");
        let error = invalid.validate().unwrap_err().to_string();
        assert!(error.contains("function_tool_options.namespace.name"));
        assert!(!error.contains("private"));
        assert!(!error.contains("description-secret"));
    }

    #[test]
    fn provider_native_hosted_tools_do_not_use_the_portable_function_slot() {
        for tool in [
            OpenAiResponsesTool::computer_use_preview(1024, 768, OpenAiComputerEnvironment::Linux),
            OpenAiResponsesTool::local_shell(),
            OpenAiResponsesTool::web_search_preview_2025_03_11(),
        ] {
            let wire = tool.into_value().unwrap();
            assert_ne!(wire["type"], "function");
        }
    }

    #[test]
    fn current_hosted_tool_contracts_are_typed_and_fail_closed() {
        let file_search = OpenAiResponsesTool::FileSearch(OpenAiFileSearchTool {
            vector_store_ids: vec!["vs_123".to_string()],
            max_num_results: Some(50),
            ranking_options: Some(OpenAiFileSearchRankingOptions {
                ranker: Some("auto".to_string()),
                score_threshold: Some(0.25),
                hybrid_search: Some(OpenAiFileSearchHybridSearch {
                    embedding_weight: 0.7,
                    text_weight: 0.3,
                }),
            }),
            filters: Some(OpenAiFileSearchFilter::In {
                key: "year".to_string(),
                value: OpenAiFileSearchFilterList::Numbers(vec![Number::from(2026)]),
            }),
        })
        .into_value()
        .unwrap();
        assert_eq!(file_search["max_num_results"], 50);
        assert_eq!(
            file_search["ranking_options"]["hybrid_search"]["text_weight"],
            0.3
        );

        let image = OpenAiResponsesTool::ImageGeneration(OpenAiImageGenerationTool {
            action: Some(OpenAiImageAction::Edit),
            moderation: Some(OpenAiImageModeration::Low),
            size: Some(OpenAiImageSize::new("2048x1024").unwrap()),
            ..OpenAiImageGenerationTool::default()
        })
        .into_value()
        .unwrap();
        assert_eq!(image["action"], "edit");
        assert_eq!(image["size"], "2048x1024");

        let combined = [
            OpenAiResponsesTool::ApplyPatch(OpenAiApplyPatchTool {
                allowed_callers: vec![OpenAiToolCaller::Programmatic],
            }),
            OpenAiResponsesTool::ToolSearch(OpenAiToolSearchTool::client(
                "Search tools",
                json!({"type": "object"}),
            )),
            OpenAiResponsesTool::Custom(OpenAiCustomTool {
                name: "command".to_string(),
                description: None,
                format: Some(OpenAiCustomToolFormat::Text),
                allowed_callers: vec![OpenAiToolCaller::Direct],
                defer_loading: Some(true),
            }),
        ];
        for tool in combined {
            tool.into_value().unwrap();
        }

        let error = OpenAiResponsesTool::FileSearch(OpenAiFileSearchTool {
            vector_store_ids: vec!["vs_123".to_string()],
            max_num_results: Some(51),
            ranking_options: None,
            filters: None,
        })
        .into_value()
        .unwrap_err();
        assert!(error.to_string().contains("between 1 and 50"));
    }

    #[test]
    fn unit_tools_with_additive_fields_use_bounded_raw_fidelity() {
        let value = json!({
            "type": "computer",
            "future": true,
        });
        let tool = serde_json::from_value::<OpenAiResponsesTool>(value.clone()).unwrap();

        assert!(matches!(tool, OpenAiResponsesTool::Raw(_)));
        assert_eq!(serde_json::to_value(tool).unwrap(), value);
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
