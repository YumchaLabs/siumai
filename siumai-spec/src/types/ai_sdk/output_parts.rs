use crate::types::ToolResultOutput;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{GeneratedFile, JSONValue, ProviderMetadata};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum TextOutputMarker {
    #[default]
    Text,
}

impl Serialize for TextOutputMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("text")
    }
}

impl<'de> Deserialize<'de> for TextOutputMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "text" {
            Ok(Self::Text)
        } else {
            Err(serde::de::Error::custom("expected text type marker"))
        }
    }
}

/// AI SDK-style text output content part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextOutput {
    #[serde(rename = "type", default)]
    marker: TextOutputMarker,
    /// Generated text.
    pub text: String,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl TextOutput {
    /// Create a text output content part.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            marker: TextOutputMarker::Text,
            text: text.into(),
            provider_metadata: None,
        }
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "text"
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum CustomOutputMarker {
    #[default]
    Custom,
}

impl Serialize for CustomOutputMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("custom")
    }
}

impl<'de> Deserialize<'de> for CustomOutputMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "custom" {
            Ok(Self::Custom)
        } else {
            Err(serde::de::Error::custom("expected custom type marker"))
        }
    }
}

/// AI SDK-style custom output content part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CustomOutput {
    #[serde(rename = "type", default)]
    marker: CustomOutputMarker,
    /// Provider-specific custom output kind, e.g. `openai.compaction`.
    pub kind: String,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl CustomOutput {
    /// Create a custom output content part.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            marker: CustomOutputMarker::Custom,
            kind: kind.into(),
            provider_metadata: None,
        }
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "custom"
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum FileOutputMarker {
    #[default]
    File,
}

impl Serialize for FileOutputMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("file")
    }
}

impl<'de> Deserialize<'de> for FileOutputMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "file" {
            Ok(Self::File)
        } else {
            Err(serde::de::Error::custom("expected file type marker"))
        }
    }
}

/// AI SDK-style generated file output content part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileOutput {
    #[serde(rename = "type", default)]
    marker: FileOutputMarker,
    /// Generated file payload.
    pub file: GeneratedFile,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl FileOutput {
    /// Create a generated file output content part.
    pub fn new(file: GeneratedFile) -> Self {
        Self {
            marker: FileOutputMarker::File,
            file,
            provider_metadata: None,
        }
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "file"
    }

    /// Attach provider metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ReasoningOutputMarker {
    #[default]
    Reasoning,
}

impl Serialize for ReasoningOutputMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("reasoning")
    }
}

impl<'de> Deserialize<'de> for ReasoningOutputMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "reasoning" {
            Ok(Self::Reasoning)
        } else {
            Err(serde::de::Error::custom("expected reasoning type marker"))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ReasoningFileOutputMarker {
    #[default]
    ReasoningFile,
}

impl Serialize for ReasoningFileOutputMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("reasoning-file")
    }
}

impl<'de> Deserialize<'de> for ReasoningFileOutputMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "reasoning-file" {
            Ok(Self::ReasoningFile)
        } else {
            Err(serde::de::Error::custom(
                "expected reasoning-file type marker",
            ))
        }
    }
}

/// AI SDK-style reasoning output part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReasoningOutput {
    #[serde(rename = "type", default)]
    marker: ReasoningOutputMarker,
    /// Reasoning text.
    pub text: String,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl ReasoningOutput {
    /// Create a reasoning output part.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            marker: ReasoningOutputMarker::Reasoning,
            text: text.into(),
            provider_metadata: None,
        }
    }

    /// Attach provider-specific response metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "reasoning"
    }
}

/// AI SDK-style reasoning-file output part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReasoningFileOutput {
    #[serde(rename = "type", default)]
    marker: ReasoningFileOutputMarker,
    /// Generated file attached to model reasoning.
    pub file: GeneratedFile,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl ReasoningFileOutput {
    /// Create a reasoning-file output part.
    pub fn new(file: GeneratedFile) -> Self {
        Self {
            marker: ReasoningFileOutputMarker::ReasoningFile,
            file,
            provider_metadata: None,
        }
    }

    /// Attach provider-specific response metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "reasoning-file"
    }
}

/// AI SDK-style typed tool call view returned by higher-level text helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ToolCallMarker {
    #[default]
    ToolCall,
}

impl Serialize for ToolCallMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("tool-call")
    }
}

impl<'de> Deserialize<'de> for ToolCallMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "tool-call" {
            Ok(Self::ToolCall)
        } else {
            Err(serde::de::Error::custom("expected tool-call type marker"))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall<NAME = String, INPUT = JSONValue> {
    #[serde(rename = "type", default)]
    marker: ToolCallMarker,
    /// ID of the tool call. This ID is used to match the tool call with the tool result.
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    /// Name of the tool being called.
    #[serde(rename = "toolName")]
    pub tool_name: NAME,
    /// Tool arguments.
    pub input: INPUT,
    /// Whether the tool call will be executed by the provider.
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
    /// Whether the tool is dynamic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    /// Whether the tool call is invalid, e.g. caused by invalid input or an unknown tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invalid: Option<bool>,
    /// Error payload explaining why the tool call is invalid when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<JSONValue>,
    /// Human-readable title for the tool call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
}

impl<NAME, INPUT> ToolCall<NAME, INPUT> {
    /// Create a typed tool call.
    pub fn new(tool_call_id: impl Into<String>, tool_name: NAME, input: INPUT) -> Self {
        Self {
            marker: ToolCallMarker::ToolCall,
            tool_call_id: tool_call_id.into(),
            tool_name,
            input,
            provider_executed: None,
            dynamic: None,
            invalid: None,
            error: None,
            title: None,
            provider_metadata: None,
        }
    }

    /// Mark whether the tool call is provider-executed.
    pub const fn with_provider_executed(mut self, provider_executed: bool) -> Self {
        self.provider_executed = Some(provider_executed);
        self
    }

    /// Mark whether the tool call is dynamic.
    pub const fn with_dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = Some(dynamic);
        self
    }

    /// Mark whether the tool call is invalid.
    pub const fn with_invalid(mut self, invalid: bool) -> Self {
        self.invalid = Some(invalid);
        self
    }

    /// Attach an invalid-tool-call error payload.
    pub fn with_error(mut self, error: impl Into<JSONValue>) -> Self {
        self.error = Some(error.into());
        self
    }

    /// Attach a human-readable title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Attach provider-specific response metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-call"
    }
}

/// Static AI SDK tool-call view. Rust keeps the same carrier and uses `dynamic` as data.
pub type StaticToolCall<NAME = String, INPUT = JSONValue> = ToolCall<NAME, INPUT>;

/// Dynamic AI SDK tool-call view. Rust keeps the same carrier and uses `dynamic` as data.
pub type DynamicToolCall<INPUT = JSONValue> = ToolCall<String, INPUT>;

/// Typed AI SDK tool-call view (`StaticToolCall | DynamicToolCall`).
pub type TypedToolCall<NAME = String, INPUT = JSONValue> = ToolCall<NAME, INPUT>;

/// AI SDK-style typed tool result view returned by higher-level text helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ToolResultMarker {
    #[default]
    ToolResult,
}

impl Serialize for ToolResultMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("tool-result")
    }
}

impl<'de> Deserialize<'de> for ToolResultMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "tool-result" {
            Ok(Self::ToolResult)
        } else {
            Err(serde::de::Error::custom("expected tool-result type marker"))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResult<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> {
    #[serde(rename = "type", default)]
    marker: ToolResultMarker,
    /// ID of the tool call. This ID is used to match the tool call with the tool result.
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    /// Name of the tool that was called.
    #[serde(rename = "toolName")]
    pub tool_name: NAME,
    /// Tool input.
    pub input: INPUT,
    /// Tool output.
    pub output: OUTPUT,
    /// Whether the tool result was executed by the provider.
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
    /// Whether the tool is dynamic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Whether the result is preliminary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preliminary: Option<bool>,
    /// Human-readable title for the tool result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl<NAME, INPUT, OUTPUT> ToolResult<NAME, INPUT, OUTPUT> {
    /// Create a typed tool result.
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: NAME,
        input: INPUT,
        output: OUTPUT,
    ) -> Self {
        Self {
            marker: ToolResultMarker::ToolResult,
            tool_call_id: tool_call_id.into(),
            tool_name,
            input,
            output,
            provider_executed: None,
            dynamic: None,
            provider_metadata: None,
            preliminary: None,
            title: None,
        }
    }

    /// Mark whether the tool result is provider-executed.
    pub const fn with_provider_executed(mut self, provider_executed: bool) -> Self {
        self.provider_executed = Some(provider_executed);
        self
    }

    /// Mark whether the tool is dynamic.
    pub const fn with_dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = Some(dynamic);
        self
    }

    /// Attach provider-specific response metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Mark whether the tool result is preliminary.
    pub const fn with_preliminary(mut self, preliminary: bool) -> Self {
        self.preliminary = Some(preliminary);
        self
    }

    /// Attach a human-readable title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-result"
    }
}

/// Static AI SDK tool-result view. Rust keeps the same carrier and uses `dynamic` as data.
pub type StaticToolResult<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    ToolResult<NAME, INPUT, OUTPUT>;

/// Dynamic AI SDK tool-result view. Rust keeps the same carrier and uses `dynamic` as data.
pub type DynamicToolResult<INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    ToolResult<String, INPUT, OUTPUT>;

/// Typed AI SDK tool-result view (`StaticToolResult | DynamicToolResult`).
pub type TypedToolResult<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> =
    ToolResult<NAME, INPUT, OUTPUT>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ToolErrorMarker {
    #[default]
    ToolError,
}

impl Serialize for ToolErrorMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("tool-error")
    }
}

impl<'de> Deserialize<'de> for ToolErrorMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "tool-error" {
            Ok(Self::ToolError)
        } else {
            Err(serde::de::Error::custom("expected tool-error type marker"))
        }
    }
}

/// AI SDK-style `tool-error` output part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolError<NAME = String, INPUT = JSONValue> {
    #[serde(rename = "type", default)]
    marker: ToolErrorMarker,
    /// ID of the tool call.
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    /// Name of the tool that failed.
    #[serde(rename = "toolName")]
    pub tool_name: NAME,
    /// Tool input.
    pub input: INPUT,
    /// Error payload.
    pub error: JSONValue,
    /// Whether the tool call was provider-executed.
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
    /// Additional provider-specific metadata.
    #[serde(
        rename = "providerMetadata",
        alias = "provider_metadata",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
    /// Whether the tool is dynamic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
    /// Human-readable title for the tool error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl<NAME, INPUT> ToolError<NAME, INPUT> {
    /// Create a tool error output part.
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: NAME,
        input: INPUT,
        error: impl Into<JSONValue>,
    ) -> Self {
        Self {
            marker: ToolErrorMarker::ToolError,
            tool_call_id: tool_call_id.into(),
            tool_name,
            input,
            error: error.into(),
            provider_executed: None,
            provider_metadata: None,
            dynamic: None,
            title: None,
        }
    }

    /// Mark whether the tool call was provider-executed.
    pub const fn with_provider_executed(mut self, provider_executed: bool) -> Self {
        self.provider_executed = Some(provider_executed);
        self
    }

    /// Attach provider-specific response metadata.
    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = Some(provider_metadata);
        self
    }

    /// Mark whether the tool is dynamic.
    pub const fn with_dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = Some(dynamic);
        self
    }

    /// Attach a human-readable title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-error"
    }
}

/// Static AI SDK tool-error view. Rust keeps the same carrier and uses `dynamic` as data.
pub type StaticToolError<NAME = String, INPUT = JSONValue> = ToolError<NAME, INPUT>;

/// Dynamic AI SDK tool-error view. Rust keeps the same carrier and uses `dynamic` as data.
pub type DynamicToolError<INPUT = JSONValue> = ToolError<String, INPUT>;

/// Typed AI SDK tool-error view (`StaticToolError | DynamicToolError`).
pub type TypedToolError<NAME = String, INPUT = JSONValue> = ToolError<NAME, INPUT>;

/// AI SDK-style tool execution output union.
///
/// This mirrors `generate-text/tool-output.ts`: successful tool executions carry
/// `tool-result`, while failed executions carry `tool-error`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ToolOutput<NAME = String, INPUT = JSONValue, OUTPUT = ToolResultOutput> {
    /// Successful tool output.
    Result(ToolResult<NAME, INPUT, OUTPUT>),
    /// Failed tool output.
    Error(ToolError<NAME, INPUT>),
}

impl<NAME, INPUT, OUTPUT> ToolOutput<NAME, INPUT, OUTPUT> {
    /// Return the AI SDK output discriminator.
    pub fn r#type(&self) -> &'static str {
        match self {
            Self::Result(output) => output.r#type(),
            Self::Error(output) => output.r#type(),
        }
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolResult<NAME, INPUT, OUTPUT>>
    for ToolOutput<NAME, INPUT, OUTPUT>
{
    fn from(value: ToolResult<NAME, INPUT, OUTPUT>) -> Self {
        Self::Result(value)
    }
}

impl<NAME, INPUT, OUTPUT> From<ToolError<NAME, INPUT>> for ToolOutput<NAME, INPUT, OUTPUT> {
    fn from(value: ToolError<NAME, INPUT>) -> Self {
        Self::Error(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ToolOutputDeniedMarker {
    #[default]
    ToolOutputDenied,
}

impl Serialize for ToolOutputDeniedMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("tool-output-denied")
    }
}

impl<'de> Deserialize<'de> for ToolOutputDeniedMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "tool-output-denied" {
            Ok(Self::ToolOutputDenied)
        } else {
            Err(serde::de::Error::custom(
                "expected tool-output-denied type marker",
            ))
        }
    }
}

/// AI SDK-style `tool-output-denied` output part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolOutputDenied<NAME = String> {
    #[serde(rename = "type", default)]
    marker: ToolOutputDeniedMarker,
    /// ID of the tool call.
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    /// Name of the tool whose output was denied.
    #[serde(rename = "toolName")]
    pub tool_name: NAME,
    /// Whether the tool call was provider-executed.
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
    /// Whether the tool is dynamic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic: Option<bool>,
}

impl<NAME> ToolOutputDenied<NAME> {
    /// Create a tool-output-denied output part.
    pub fn new(tool_call_id: impl Into<String>, tool_name: NAME) -> Self {
        Self {
            marker: ToolOutputDeniedMarker::ToolOutputDenied,
            tool_call_id: tool_call_id.into(),
            tool_name,
            provider_executed: None,
            dynamic: None,
        }
    }

    /// Mark whether the tool call was provider-executed.
    pub const fn with_provider_executed(mut self, provider_executed: bool) -> Self {
        self.provider_executed = Some(provider_executed);
        self
    }

    /// Mark whether the tool is dynamic.
    pub const fn with_dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = Some(dynamic);
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-output-denied"
    }
}

/// Static tool-output-denied view. Rust keeps the same runtime carrier as the typed view.
pub type StaticToolOutputDenied<NAME = String> = ToolOutputDenied<NAME>;

/// Typed tool-output-denied view. Rust keeps the same runtime carrier as the static view.
pub type TypedToolOutputDenied<NAME = String> = ToolOutputDenied<NAME>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ToolApprovalRequestOutputMarker {
    #[default]
    ToolApprovalRequest,
}

impl Serialize for ToolApprovalRequestOutputMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("tool-approval-request")
    }
}

impl<'de> Deserialize<'de> for ToolApprovalRequestOutputMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "tool-approval-request" {
            Ok(Self::ToolApprovalRequest)
        } else {
            Err(serde::de::Error::custom(
                "expected tool-approval-request type marker",
            ))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ToolApprovalResponseOutputMarker {
    #[default]
    ToolApprovalResponse,
}

impl Serialize for ToolApprovalResponseOutputMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("tool-approval-response")
    }
}

impl<'de> Deserialize<'de> for ToolApprovalResponseOutputMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value == "tool-approval-response" {
            Ok(Self::ToolApprovalResponse)
        } else {
            Err(serde::de::Error::custom(
                "expected tool-approval-response type marker",
            ))
        }
    }
}

/// AI SDK-style `tool-approval-request` output part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolApprovalRequestOutput<NAME = String, INPUT = JSONValue> {
    #[serde(rename = "type", default)]
    marker: ToolApprovalRequestOutputMarker,
    /// ID of the tool approval request.
    #[serde(rename = "approvalId")]
    pub approval_id: String,
    /// Tool call that the approval request is for.
    #[serde(rename = "toolCall")]
    pub tool_call: ToolCall<NAME, INPUT>,
    /// Whether the tool was automatically approved or denied.
    #[serde(
        rename = "isAutomatic",
        alias = "is_automatic",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub is_automatic: Option<bool>,
}

impl<NAME, INPUT> ToolApprovalRequestOutput<NAME, INPUT> {
    /// Create a tool approval request output part.
    pub fn new(approval_id: impl Into<String>, tool_call: ToolCall<NAME, INPUT>) -> Self {
        Self {
            marker: ToolApprovalRequestOutputMarker::ToolApprovalRequest,
            approval_id: approval_id.into(),
            tool_call,
            is_automatic: None,
        }
    }

    /// Mark whether the request was generated automatically.
    pub const fn with_is_automatic(mut self, is_automatic: bool) -> Self {
        self.is_automatic = Some(is_automatic);
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-approval-request"
    }
}

/// AI SDK-style `tool-approval-response` output part returned by text helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolApprovalResponseOutput<NAME = String, INPUT = JSONValue> {
    #[serde(rename = "type", default)]
    marker: ToolApprovalResponseOutputMarker,
    /// ID of the tool approval.
    #[serde(rename = "approvalId")]
    pub approval_id: String,
    /// Tool call that the approval response is for.
    #[serde(rename = "toolCall")]
    pub tool_call: ToolCall<NAME, INPUT>,
    /// Whether the approval was granted or denied.
    pub approved: bool,
    /// Optional reason for the approval or denial.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Whether the tool call is provider-executed.
    #[serde(
        rename = "providerExecuted",
        alias = "provider_executed",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_executed: Option<bool>,
}

impl<NAME, INPUT> ToolApprovalResponseOutput<NAME, INPUT> {
    /// Create a tool approval response output part.
    pub fn new(
        approval_id: impl Into<String>,
        tool_call: ToolCall<NAME, INPUT>,
        approved: bool,
    ) -> Self {
        Self {
            marker: ToolApprovalResponseOutputMarker::ToolApprovalResponse,
            approval_id: approval_id.into(),
            tool_call,
            approved,
            reason: None,
            provider_executed: None,
        }
    }

    /// Attach an optional human-readable approval reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Mark whether the approval response refers to a provider-executed tool call.
    pub const fn with_provider_executed(mut self, provider_executed: bool) -> Self {
        self.provider_executed = Some(provider_executed);
        self
    }

    /// Return the AI SDK output part discriminator.
    pub const fn r#type(&self) -> &'static str {
        "tool-approval-response"
    }
}
