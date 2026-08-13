use std::fmt;

use serde_json::{Map, Value};
use siumai_core::OpaqueProviderItem;

use super::{MessagesCodecError, OPAQUE_CONTENT_BLOCK_KIND, PROTOCOL_ID};

const MAX_NATIVE_IDENTIFIER_BYTES: usize = 512;

/// Maintained Anthropic server-tool result block kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AnthropicHostedToolResultKind {
    WebSearch,
    WebFetch,
    CodeExecution,
    BashCodeExecution,
    TextEditorCodeExecution,
    ToolSearch,
    Advisor,
    Mcp,
}

impl AnthropicHostedToolResultKind {
    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::WebSearch => "web_search_tool_result",
            Self::WebFetch => "web_fetch_tool_result",
            Self::CodeExecution => "code_execution_tool_result",
            Self::BashCodeExecution => "bash_code_execution_tool_result",
            Self::TextEditorCodeExecution => "text_editor_code_execution_tool_result",
            Self::ToolSearch => "tool_search_tool_result",
            Self::Advisor => "advisor_tool_result",
            Self::Mcp => "mcp_tool_result",
        }
    }

    fn from_wire_str(value: &str) -> Option<Self> {
        match value {
            "web_search_tool_result" => Some(Self::WebSearch),
            "web_fetch_tool_result" => Some(Self::WebFetch),
            "code_execution_tool_result" => Some(Self::CodeExecution),
            "bash_code_execution_tool_result" => Some(Self::BashCodeExecution),
            "text_editor_code_execution_tool_result" => Some(Self::TextEditorCodeExecution),
            "tool_search_tool_result" => Some(Self::ToolSearch),
            "advisor_tool_result" => Some(Self::Advisor),
            "mcp_tool_result" => Some(Self::Mcp),
            _ => None,
        }
    }
}

/// Borrowed inspection of one Anthropic-managed server-tool invocation.
#[derive(Clone, Copy)]
pub struct AnthropicServerToolUseRef<'a> {
    raw: &'a Value,
    id: &'a str,
    name: &'a str,
    input: &'a Value,
    caller: Option<&'a Value>,
}

impl<'a> AnthropicServerToolUseRef<'a> {
    pub fn id(self) -> &'a str {
        self.id
    }

    pub fn name(self) -> &'a str {
        self.name
    }

    pub fn input(self) -> &'a Value {
        self.input
    }

    pub fn caller(self) -> Option<&'a Value> {
        self.caller
    }

    pub fn raw(self) -> &'a Value {
        self.raw
    }
}

impl fmt::Debug for AnthropicServerToolUseRef<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicServerToolUseRef")
            .field("id_bytes", &self.id.len())
            .field("name_bytes", &self.name.len())
            .field("input_kind", &value_kind(self.input))
            .field("has_caller", &self.caller.is_some())
            .finish()
    }
}

/// Borrowed inspection of one Anthropic-managed MCP invocation.
#[derive(Clone, Copy)]
pub struct AnthropicMcpToolUseRef<'a> {
    raw: &'a Value,
    id: &'a str,
    name: &'a str,
    server_name: &'a str,
    input: &'a Value,
    caller: Option<&'a Value>,
}

impl<'a> AnthropicMcpToolUseRef<'a> {
    pub fn id(self) -> &'a str {
        self.id
    }

    pub fn name(self) -> &'a str {
        self.name
    }

    pub fn server_name(self) -> &'a str {
        self.server_name
    }

    pub fn input(self) -> &'a Value {
        self.input
    }

    pub fn caller(self) -> Option<&'a Value> {
        self.caller
    }

    pub fn raw(self) -> &'a Value {
        self.raw
    }
}

impl fmt::Debug for AnthropicMcpToolUseRef<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicMcpToolUseRef")
            .field("id_bytes", &self.id.len())
            .field("name_bytes", &self.name.len())
            .field("server_name_bytes", &self.server_name.len())
            .field("input_kind", &value_kind(self.input))
            .field("has_caller", &self.caller.is_some())
            .finish()
    }
}

/// Borrowed inspection of one Anthropic-managed server-tool result.
#[derive(Clone, Copy)]
pub struct AnthropicHostedToolResultRef<'a> {
    raw: &'a Value,
    kind: AnthropicHostedToolResultKind,
    tool_use_id: &'a str,
    content: &'a Value,
    is_error: Option<bool>,
    caller: Option<&'a Value>,
}

impl<'a> AnthropicHostedToolResultRef<'a> {
    pub const fn kind(self) -> AnthropicHostedToolResultKind {
        self.kind
    }

    pub fn tool_use_id(self) -> &'a str {
        self.tool_use_id
    }

    pub fn content(self) -> &'a Value {
        self.content
    }

    pub const fn is_error(self) -> Option<bool> {
        self.is_error
    }

    pub fn caller(self) -> Option<&'a Value> {
        self.caller
    }

    pub fn raw(self) -> &'a Value {
        self.raw
    }
}

impl fmt::Debug for AnthropicHostedToolResultRef<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicHostedToolResultRef")
            .field("kind", &self.kind)
            .field("tool_use_id_bytes", &self.tool_use_id.len())
            .field("content_kind", &value_kind(self.content))
            .field("is_error", &self.is_error)
            .field("has_caller", &self.caller.is_some())
            .finish()
    }
}

/// Borrowed, provider-owned view of maintained Anthropic hosted-tool content.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub enum AnthropicHostedToolBlockRef<'a> {
    ServerToolUse(AnthropicServerToolUseRef<'a>),
    McpToolUse(AnthropicMcpToolUseRef<'a>),
    Result(AnthropicHostedToolResultRef<'a>),
}

impl<'a> AnthropicHostedToolBlockRef<'a> {
    pub fn item_id(self) -> Option<&'a str> {
        match self {
            Self::ServerToolUse(value) => Some(value.id()),
            Self::McpToolUse(value) => Some(value.id()),
            Self::Result(_) => None,
        }
    }

    pub fn related_tool_use_id(self) -> Option<&'a str> {
        match self {
            Self::Result(value) => Some(value.tool_use_id()),
            Self::ServerToolUse(_) | Self::McpToolUse(_) => None,
        }
    }

    pub fn caller_tool_id(self) -> Option<&'a str> {
        let caller = self.caller()?;
        caller.as_object()?.get("tool_id")?.as_str()
    }

    pub fn caller(self) -> Option<&'a Value> {
        match self {
            Self::ServerToolUse(value) => value.caller(),
            Self::McpToolUse(value) => value.caller(),
            Self::Result(value) => value.caller(),
        }
    }

    pub fn raw(self) -> &'a Value {
        match self {
            Self::ServerToolUse(value) => value.raw(),
            Self::McpToolUse(value) => value.raw(),
            Self::Result(value) => value.raw(),
        }
    }
}

impl fmt::Debug for AnthropicHostedToolBlockRef<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ServerToolUse(value) => value.fmt(formatter),
            Self::McpToolUse(value) => value.fmt(formatter),
            Self::Result(value) => value.fmt(formatter),
        }
    }
}

/// Anthropic-specific inspection for bounded provider-native content blocks.
pub trait AnthropicOpaqueContentExt {
    fn anthropic_hosted_tool(
        &self,
    ) -> Result<Option<AnthropicHostedToolBlockRef<'_>>, MessagesCodecError>;
}

impl AnthropicOpaqueContentExt for OpaqueProviderItem {
    fn anthropic_hosted_tool(
        &self,
    ) -> Result<Option<AnthropicHostedToolBlockRef<'_>>, MessagesCodecError> {
        if self.kind() != OPAQUE_CONTENT_BLOCK_KIND
            || self.provenance().protocol().as_str() != PROTOCOL_ID
        {
            return Ok(None);
        }
        inspect_hosted_tool_value(self.data())
    }
}

pub(crate) fn inspect_hosted_tool_value(
    value: &Value,
) -> Result<Option<AnthropicHostedToolBlockRef<'_>>, MessagesCodecError> {
    let Some(object) = value.as_object() else {
        return Err(MessagesCodecError::ProtocolViolation {
            reason: "native hosted-tool content block was not an object",
        });
    };
    let Some(kind) = object.get("type").and_then(Value::as_str) else {
        return Err(MessagesCodecError::ProtocolViolation {
            reason: "native hosted-tool content block omitted its type",
        });
    };
    match kind {
        "server_tool_use" => {
            let id = required_identifier(object, "id")?;
            let name = required_identifier(object, "name")?;
            let input = required_value(object, "input")?;
            let caller = validate_caller(object)?;
            Ok(Some(AnthropicHostedToolBlockRef::ServerToolUse(
                AnthropicServerToolUseRef {
                    raw: value,
                    id,
                    name,
                    input,
                    caller,
                },
            )))
        }
        "mcp_tool_use" => {
            let id = required_identifier(object, "id")?;
            let name = required_identifier(object, "name")?;
            let server_name = required_identifier(object, "server_name")?;
            let input = required_value(object, "input")?;
            let caller = validate_caller(object)?;
            Ok(Some(AnthropicHostedToolBlockRef::McpToolUse(
                AnthropicMcpToolUseRef {
                    raw: value,
                    id,
                    name,
                    server_name,
                    input,
                    caller,
                },
            )))
        }
        kind if AnthropicHostedToolResultKind::from_wire_str(kind).is_some() => {
            let kind = AnthropicHostedToolResultKind::from_wire_str(kind)
                .expect("matched hosted-tool result kind");
            let tool_use_id = required_identifier(object, "tool_use_id")?;
            let content = object
                .get("content")
                .filter(|value| !value.is_null())
                .ok_or(MessagesCodecError::ProtocolViolation {
                    reason: "hosted-tool result block omitted its content",
                })?;
            if kind == AnthropicHostedToolResultKind::Mcp
                && !matches!(content, Value::String(_) | Value::Array(_))
            {
                return Err(MessagesCodecError::ProtocolViolation {
                    reason: "MCP tool result content was neither text nor a content array",
                });
            }
            let is_error = match object.get("is_error") {
                None => None,
                Some(Value::Bool(value)) => Some(*value),
                Some(_) => {
                    return Err(MessagesCodecError::ProtocolViolation {
                        reason: "hosted-tool result is_error was not a boolean",
                    });
                }
            };
            let caller = validate_caller(object)?;
            Ok(Some(AnthropicHostedToolBlockRef::Result(
                AnthropicHostedToolResultRef {
                    raw: value,
                    kind,
                    tool_use_id,
                    content,
                    is_error,
                    caller,
                },
            )))
        }
        _ => Ok(None),
    }
}

pub(crate) fn validate_caller(
    object: &Map<String, Value>,
) -> Result<Option<&Value>, MessagesCodecError> {
    let Some(caller) = object.get("caller") else {
        return Ok(None);
    };
    if caller.is_null() {
        return Ok(None);
    }
    let caller_object = caller
        .as_object()
        .ok_or(MessagesCodecError::ProtocolViolation {
            reason: "native tool caller was not an object",
        })?;
    let caller_type = required_identifier(caller_object, "type")?;
    match caller_type {
        "direct" if caller_object.contains_key("tool_id") => {
            return Err(MessagesCodecError::ProtocolViolation {
                reason: "direct native tool caller unexpectedly contained a tool ID",
            });
        }
        "code_execution_20250825" | "code_execution_20260120" => {
            required_identifier(caller_object, "tool_id")?;
        }
        _ if caller_object.contains_key("tool_id") => {
            required_identifier(caller_object, "tool_id")?;
        }
        _ => {}
    }
    Ok(Some(caller))
}

pub(crate) fn caller_is_replayable(caller: Option<&Value>) -> bool {
    caller.is_none_or(|caller| {
        matches!(
            caller
                .as_object()
                .and_then(|object| object.get("type"))
                .and_then(Value::as_str),
            Some("direct" | "code_execution_20250825" | "code_execution_20260120")
        )
    })
}

pub(crate) fn is_hosted_tool_use_kind(kind: &str) -> bool {
    matches!(kind, "server_tool_use" | "mcp_tool_use")
}

pub(crate) fn is_maintained_hosted_kind(kind: &str) -> bool {
    is_hosted_tool_use_kind(kind) || AnthropicHostedToolResultKind::from_wire_str(kind).is_some()
}

fn required_identifier<'a>(
    object: &'a Map<String, Value>,
    field: &'static str,
) -> Result<&'a str, MessagesCodecError> {
    let value =
        object
            .get(field)
            .and_then(Value::as_str)
            .ok_or(MessagesCodecError::ProtocolViolation {
                reason: "native hosted-tool content block omitted a required identifier",
            })?;
    if value.is_empty()
        || value.len() > MAX_NATIVE_IDENTIFIER_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(MessagesCodecError::ProtocolViolation {
            reason: "native hosted-tool content block contained an invalid identifier",
        });
    }
    Ok(value)
}

fn required_value<'a>(
    object: &'a Map<String, Value>,
    field: &'static str,
) -> Result<&'a Value, MessagesCodecError> {
    object
        .get(field)
        .ok_or(MessagesCodecError::ProtocolViolation {
            reason: "native hosted-tool content block omitted a required value",
        })
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}
