use std::fmt;

use serde::{Deserialize, Serialize};
use siumai_core::{ContentAnnotations, MessageAnnotations, ProviderScope, ToolAnnotations};

use super::MessagesCodecError;
use super::options::{AnthropicTool, ToolCaller};

/// Lifetime of one Anthropic-protocol prompt-cache entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheTtl {
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

impl CacheTtl {
    pub(crate) const fn as_wire_str(self) -> &'static str {
        match self {
            Self::FiveMinutes => "5m",
            Self::OneHour => "1h",
        }
    }
}

/// One explicit Anthropic-protocol prompt-cache breakpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheControl {
    ttl: CacheTtl,
}

const MAX_TOOL_REFERENCE_COMPONENT_BYTES: usize = 256;
const MAX_FILE_ID_BYTES: usize = 512;

/// A typed reference to one tool declared in the request's stable tool set.
///
/// Mid-conversation changes reference existing definitions; they never carry a
/// second tool schema. This preserves the prompt-cache prefix and prevents a
/// content block from smuggling a conflicting definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AnthropicToolReference {
    kind: AnthropicToolReferenceKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum AnthropicToolReferenceKind {
    #[serde(rename = "tool_reference")]
    Tool { name: String },
    #[serde(rename = "mcp_tool_reference")]
    McpTool { server_name: String, name: String },
    #[serde(rename = "mcp_toolset_reference")]
    McpToolset { server_name: String },
}

impl AnthropicToolReference {
    pub fn tool(name: impl Into<String>) -> Result<Self, MessagesCodecError> {
        let name = name.into();
        validate_reference_component(&name, "messages.tool_change.tool.name")?;
        Ok(Self {
            kind: AnthropicToolReferenceKind::Tool { name },
        })
    }

    pub fn mcp_tool(
        server_name: impl Into<String>,
        name: impl Into<String>,
    ) -> Result<Self, MessagesCodecError> {
        let server_name = server_name.into();
        let name = name.into();
        validate_reference_component(&server_name, "messages.tool_change.tool.server_name")?;
        validate_reference_component(&name, "messages.tool_change.tool.name")?;
        Ok(Self {
            kind: AnthropicToolReferenceKind::McpTool { server_name, name },
        })
    }

    pub fn mcp_toolset(server_name: impl Into<String>) -> Result<Self, MessagesCodecError> {
        let server_name = server_name.into();
        validate_reference_component(&server_name, "messages.tool_change.tool.server_name")?;
        Ok(Self {
            kind: AnthropicToolReferenceKind::McpToolset { server_name },
        })
    }

    pub(crate) fn kind(&self) -> &AnthropicToolReferenceKind {
        &self.kind
    }

    pub fn validate(&self) -> Result<(), MessagesCodecError> {
        match &self.kind {
            AnthropicToolReferenceKind::Tool { name } => {
                validate_reference_component(name, "messages.tool_change.tool.name")
            }
            AnthropicToolReferenceKind::McpTool { server_name, name } => {
                validate_reference_component(server_name, "messages.tool_change.tool.server_name")?;
                validate_reference_component(name, "messages.tool_change.tool.name")
            }
            AnthropicToolReferenceKind::McpToolset { server_name } => {
                validate_reference_component(server_name, "messages.tool_change.tool.server_name")
            }
        }
    }
}

/// One typed addition or removal inside a mid-conversation system message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MidConversationToolChange {
    kind: MidConversationToolChangeKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum MidConversationToolChangeKind {
    ToolAddition { tool: AnthropicToolReference },
    ToolRemoval { tool: AnthropicToolReference },
}

impl MidConversationToolChange {
    pub fn add(tool: AnthropicToolReference) -> Self {
        Self {
            kind: MidConversationToolChangeKind::ToolAddition { tool },
        }
    }

    pub fn remove(tool: AnthropicToolReference) -> Self {
        Self {
            kind: MidConversationToolChangeKind::ToolRemoval { tool },
        }
    }

    pub(crate) fn kind(&self) -> &MidConversationToolChangeKind {
        &self.kind
    }

    pub fn reference(&self) -> &AnthropicToolReference {
        match &self.kind {
            MidConversationToolChangeKind::ToolAddition { tool }
            | MidConversationToolChangeKind::ToolRemoval { tool } => tool,
        }
    }

    pub fn validate(&self) -> Result<(), MessagesCodecError> {
        self.reference().validate()
    }
}

fn validate_reference_component(
    value: &str,
    field: &'static str,
) -> Result<(), MessagesCodecError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(MessagesCodecError::InvalidOption {
            field,
            reason: "must be a non-empty printable string",
        });
    }
    if value.len() > MAX_TOOL_REFERENCE_COMPONENT_BYTES {
        return Err(MessagesCodecError::InvalidOption {
            field,
            reason: "exceeds the 256-byte limit",
        });
    }
    Ok(())
}

impl CacheControl {
    pub const fn new(ttl: CacheTtl) -> Self {
        Self { ttl }
    }

    pub const fn ttl(self) -> CacheTtl {
        self.ttl
    }
}

/// Scope-bound Anthropic file reference projected by a branded provider.
#[derive(Clone, PartialEq, Eq)]
pub struct MessagesFileReference {
    file_id: String,
    scope: ProviderScope,
}

impl MessagesFileReference {
    pub fn new(
        file_id: impl Into<String>,
        scope: ProviderScope,
    ) -> Result<Self, MessagesCodecError> {
        let reference = Self {
            file_id: file_id.into(),
            scope,
        };
        reference.validate()?;
        Ok(reference)
    }

    pub fn file_id(&self) -> &str {
        &self.file_id
    }

    pub fn scope(&self) -> &ProviderScope {
        &self.scope
    }

    fn validate(&self) -> Result<(), MessagesCodecError> {
        if self.file_id.is_empty()
            || self.file_id.len() > MAX_FILE_ID_BYTES
            || self.file_id.chars().any(char::is_control)
        {
            return Err(MessagesCodecError::InvalidOption {
                field: "messages.file.file_id",
                reason: "must be non-empty, bounded, and free of control characters",
            });
        }
        if !file_scope_is_complete(&self.scope) {
            return Err(MessagesCodecError::InvalidOption {
                field: "messages.file.scope",
                reason: "must include platform, protocol, API mode, replay domain, and caller scope",
            });
        }
        Ok(())
    }
}

pub(crate) fn file_scope_is_complete(scope: &ProviderScope) -> bool {
    scope.platform().is_some()
        && scope.protocol().is_some()
        && scope.api_mode().is_some()
        && scope
            .replay_domain()
            .and_then(|domain| domain.caller_scope())
            .is_some()
}

impl fmt::Debug for MessagesFileReference {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MessagesFileReference")
            .field("file_id_bytes", &self.file_id.len())
            .field("scope", &"bound")
            .finish()
    }
}

/// One Anthropic Files API reference projected into a Messages content block.
#[derive(Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MessagesFileBlock {
    Image(MessagesFileReference),
    Document {
        reference: MessagesFileReference,
        title: Option<String>,
        context: Option<String>,
        citations: Option<bool>,
    },
    ContainerUpload(MessagesFileReference),
}

impl fmt::Debug for MessagesFileBlock {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Image(reference) => formatter
                .debug_tuple("MessagesFileBlock::Image")
                .field(reference)
                .finish(),
            Self::Document {
                reference,
                title,
                context,
                citations,
            } => formatter
                .debug_struct("MessagesFileBlock::Document")
                .field("reference", reference)
                .field("title_bytes", &title.as_ref().map(String::len))
                .field("context_bytes", &context.as_ref().map(String::len))
                .field("citations", citations)
                .finish(),
            Self::ContainerUpload(reference) => formatter
                .debug_tuple("MessagesFileBlock::ContainerUpload")
                .field(reference)
                .finish(),
        }
    }
}

impl MessagesFileBlock {
    pub fn reference(&self) -> &MessagesFileReference {
        match self {
            Self::Image(reference)
            | Self::Document { reference, .. }
            | Self::ContainerUpload(reference) => reference,
        }
    }

    pub const fn is_container_upload(&self) -> bool {
        matches!(self, Self::ContainerUpload(_))
    }
}

/// Wire controls resolved for one complete message node.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MessageNodeOptions {
    cache_control: Option<CacheControl>,
}

impl MessageNodeOptions {
    pub const fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }

    pub const fn cache_control(&self) -> Option<CacheControl> {
        self.cache_control
    }
}

/// Wire controls resolved for one content node.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContentNodeOptions {
    cache_control: Option<CacheControl>,
    tool_change: Option<MidConversationToolChange>,
    file: Option<MessagesFileBlock>,
}

impl ContentNodeOptions {
    pub const fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }

    pub const fn cache_control(&self) -> Option<CacheControl> {
        self.cache_control
    }

    pub fn with_tool_change(mut self, tool_change: MidConversationToolChange) -> Self {
        self.tool_change = Some(tool_change);
        self
    }

    pub fn tool_change(&self) -> Option<&MidConversationToolChange> {
        self.tool_change.as_ref()
    }

    pub fn with_file(mut self, file: MessagesFileBlock) -> Self {
        self.file = Some(file);
        self
    }

    pub fn file(&self) -> Option<&MessagesFileBlock> {
        self.file.as_ref()
    }
}

/// Wire controls resolved for one model-visible tool node.
///
/// Function tools remain ordinary [`siumai_core::ToolSpec`] values. A branded
/// provider may instead resolve one typed annotation to a bounded
/// Anthropic-defined tool projection. Some variants execute at Anthropic while
/// others are client-executed; the codec only owns their wire definition. It
/// validates that projection against a canonical anchor before replacing the
/// placeholder tool definition.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolNodeOptions {
    cache_control: Option<CacheControl>,
    anthropic_tool: Option<AnthropicTool>,
    allowed_callers: Vec<ToolCaller>,
    strict: Option<bool>,
    defer_loading: Option<bool>,
}

impl ToolNodeOptions {
    pub const fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }

    pub const fn cache_control(&self) -> Option<CacheControl> {
        self.cache_control
    }

    pub fn with_anthropic_tool(mut self, anthropic_tool: AnthropicTool) -> Self {
        self.anthropic_tool = Some(anthropic_tool);
        self
    }

    pub fn anthropic_tool(&self) -> Option<&AnthropicTool> {
        self.anthropic_tool.as_ref()
    }

    pub fn with_allowed_callers(mut self, callers: impl IntoIterator<Item = ToolCaller>) -> Self {
        self.allowed_callers = callers.into_iter().collect();
        self
    }

    pub fn allowed_callers(&self) -> &[ToolCaller] {
        &self.allowed_callers
    }

    pub const fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    pub const fn strict(&self) -> Option<bool> {
        self.strict
    }

    pub const fn with_defer_loading(mut self, defer_loading: bool) -> Self {
        self.defer_loading = Some(defer_loading);
        self
    }

    pub const fn defer_loading(&self) -> Option<bool> {
        self.defer_loading
    }
}

/// Resolve provider-owned durable annotations into bounded Messages wire controls.
///
/// Protocol code never selects a provider namespace. A branded provider or a
/// verified compatibility profile owns its annotation schemas, strictly decodes
/// only its namespace, and returns these limited wire projections. Foreign
/// annotations remain inert because the resolver never interprets them.
pub trait MessagesAnnotationResolver: Send + Sync {
    fn resolve_message(
        &self,
        _annotations: &MessageAnnotations,
    ) -> Result<MessageNodeOptions, MessagesCodecError> {
        Ok(MessageNodeOptions::default())
    }

    fn resolve_content(
        &self,
        _annotations: &ContentAnnotations,
    ) -> Result<ContentNodeOptions, MessagesCodecError> {
        Ok(ContentNodeOptions::default())
    }

    fn resolve_tool(
        &self,
        _annotations: &ToolAnnotations,
    ) -> Result<ToolNodeOptions, MessagesCodecError> {
        Ok(ToolNodeOptions::default())
    }
}

/// Resolver used by the provider-neutral encoding entry point.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoMessagesAnnotations;

impl MessagesAnnotationResolver for NoMessagesAnnotations {}
