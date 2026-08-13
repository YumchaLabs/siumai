use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use siumai_core::{
    ApiModeId, ContentAnnotationTarget, ContentAnnotations, InvalidToolSpec,
    MessageAnnotationTarget, MessageAnnotations, MessagePart, PlatformId, ProtocolId,
    ProviderAnnotationError, ProviderId, ProviderScope, ReplayDomain, ReplayDomainId,
    ToolAnnotationTarget, ToolAnnotations, ToolSpec, TypedProviderAnnotation,
};
use siumai_protocol_anthropic::messages::{
    API_MODE_ID, AnthropicTool, CacheControl, CacheTtl, ContentNodeOptions, MessageNodeOptions,
    MessagesAnnotationResolver, MessagesCodecError, MessagesFileBlock, MessagesFileReference,
    MidConversationToolChange, ToolCaller, ToolNodeOptions, anthropic_tool_anchor_schema,
};
use thiserror::Error;

/// Anthropic prompt-cache lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicCacheTtl {
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

impl AnthropicCacheTtl {
    const fn protocol(self) -> CacheTtl {
        match self {
            Self::FiveMinutes => CacheTtl::FiveMinutes,
            Self::OneHour => CacheTtl::OneHour,
        }
    }
}

macro_rules! cache_annotation {
    ($name:ident, $target:ty) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct $name {
            ttl: AnthropicCacheTtl,
        }

        impl $name {
            pub const fn new(ttl: AnthropicCacheTtl) -> Self {
                Self { ttl }
            }

            pub const fn five_minutes() -> Self {
                Self::new(AnthropicCacheTtl::FiveMinutes)
            }

            pub const fn one_hour() -> Self {
                Self::new(AnthropicCacheTtl::OneHour)
            }

            pub const fn ttl(self) -> AnthropicCacheTtl {
                self.ttl
            }
        }

        impl TypedProviderAnnotation for $name {
            type Target = $target;

            const NAMESPACE: &'static str = "anthropic";
            const API_MODE: Option<&'static str> = Some(API_MODE_ID);
        }
    };
}

cache_annotation!(AnthropicMessageCache, MessageAnnotationTarget);

const MAX_FILE_ID_BYTES: usize = 512;

/// A file ID bound to the exact durable Anthropic replay scope that created it.
#[derive(Clone, PartialEq, Eq)]
pub struct AnthropicFileReference {
    file_id: String,
    scope: ProviderScope,
}

impl AnthropicFileReference {
    pub(crate) fn bind(
        file_id: impl Into<String>,
        scope: ProviderScope,
    ) -> Result<Self, ProviderAnnotationError> {
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

    fn validate(&self) -> Result<(), ProviderAnnotationError> {
        if self.file_id.is_empty()
            || self.file_id.len() > MAX_FILE_ID_BYTES
            || self.file_id.chars().any(char::is_control)
        {
            return Err(rejected(
                "content.file.file_id",
                "file identifiers must be non-empty, bounded, and free of control characters",
            ));
        }
        if self.scope.platform().is_none()
            || self.scope.protocol().is_none()
            || self.scope.api_mode().is_none()
        {
            return Err(rejected(
                "content.file.scope",
                "file references require a complete provider execution scope",
            ));
        }
        let replay_domain = self.scope.replay_domain().ok_or_else(|| {
            rejected(
                "content.file.scope",
                "file references require a replay domain",
            )
        })?;
        if replay_domain.caller_scope().is_none() {
            return Err(rejected(
                "content.file.scope",
                "file references require an explicit caller scope",
            ));
        }
        Ok(())
    }
}

impl fmt::Debug for AnthropicFileReference {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicFileReference")
            .field("file_id_bytes", &self.file_id.len())
            .field("scope", &"bound")
            .finish()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnthropicFileReferenceWire {
    file_id: String,
    provider_id: String,
    platform_id: String,
    protocol_id: String,
    api_mode_id: String,
    replay_kind: AnthropicReplayKind,
    replay_domain_id: String,
    caller_scope_id: String,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AnthropicReplayKind {
    Official,
    Custom,
}

impl Serialize for AnthropicFileReference {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let replay_domain = self
            .scope
            .replay_domain()
            .ok_or_else(|| serde::ser::Error::custom("missing replay domain"))?;
        let caller_scope = replay_domain
            .caller_scope()
            .ok_or_else(|| serde::ser::Error::custom("missing caller scope"))?;
        AnthropicFileReferenceWire {
            file_id: self.file_id.clone(),
            provider_id: self.scope.provider_id().to_string(),
            platform_id: required_scope_id(self.scope.platform(), "platform")
                .map_err(serde::ser::Error::custom)?,
            protocol_id: required_scope_id(self.scope.protocol(), "protocol")
                .map_err(serde::ser::Error::custom)?,
            api_mode_id: required_scope_id(self.scope.api_mode(), "API mode")
                .map_err(serde::ser::Error::custom)?,
            replay_kind: if replay_domain.audience().is_official() {
                AnthropicReplayKind::Official
            } else {
                AnthropicReplayKind::Custom
            },
            replay_domain_id: replay_domain.audience().id().to_string(),
            caller_scope_id: caller_scope.to_string(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AnthropicFileReference {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = AnthropicFileReferenceWire::deserialize(deserializer)?;
        let replay_id =
            ReplayDomainId::new(wire.replay_domain_id).map_err(serde::de::Error::custom)?;
        let replay_domain = match wire.replay_kind {
            AnthropicReplayKind::Official => ReplayDomain::official(replay_id),
            AnthropicReplayKind::Custom => ReplayDomain::custom(replay_id),
        }
        .with_caller_scope(
            ReplayDomainId::new(wire.caller_scope_id).map_err(serde::de::Error::custom)?,
        );
        let scope = ProviderScope::new(
            ProviderId::new(wire.provider_id).map_err(serde::de::Error::custom)?,
        )
        .with_platform(PlatformId::new(wire.platform_id).map_err(serde::de::Error::custom)?)
        .with_protocol(ProtocolId::new(wire.protocol_id).map_err(serde::de::Error::custom)?)
        .with_api_mode(ApiModeId::new(wire.api_mode_id).map_err(serde::de::Error::custom)?)
        .with_replay_domain(replay_domain);
        Self::bind(wire.file_id, scope).map_err(serde::de::Error::custom)
    }
}

fn required_scope_id<T: ToString>(value: Option<&T>, name: &str) -> Result<String, String> {
    value
        .map(ToString::to_string)
        .ok_or_else(|| format!("missing {name}"))
}

/// One Anthropic file reference projected into a Messages content block.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum AnthropicMessageFile {
    Image {
        reference: AnthropicFileReference,
    },
    Document {
        reference: AnthropicFileReference,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<bool>,
    },
    ContainerUpload {
        reference: AnthropicFileReference,
    },
}

impl AnthropicMessageFile {
    pub fn image(reference: AnthropicFileReference) -> Self {
        Self::Image { reference }
    }

    pub fn document(reference: AnthropicFileReference) -> Self {
        Self::Document {
            reference,
            title: None,
            context: None,
            citations: None,
        }
    }

    pub fn container_upload(reference: AnthropicFileReference) -> Self {
        Self::ContainerUpload { reference }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Result<Self, ProviderAnnotationError> {
        let Self::Document { title: field, .. } = &mut self else {
            return Err(rejected(
                "content.file.title",
                "is only valid for an Anthropic document file block",
            ));
        };
        *field = Some(title.into());
        Ok(self)
    }

    pub fn with_context(
        mut self,
        context: impl Into<String>,
    ) -> Result<Self, ProviderAnnotationError> {
        let Self::Document { context: field, .. } = &mut self else {
            return Err(rejected(
                "content.file.context",
                "is only valid for an Anthropic document file block",
            ));
        };
        *field = Some(context.into());
        Ok(self)
    }

    pub fn with_citations(mut self, enabled: bool) -> Result<Self, ProviderAnnotationError> {
        let Self::Document { citations, .. } = &mut self else {
            return Err(rejected(
                "content.file.citations",
                "is only valid for an Anthropic document file block",
            ));
        };
        *citations = Some(enabled);
        Ok(self)
    }

    pub fn reference(&self) -> &AnthropicFileReference {
        match self {
            Self::Image { reference }
            | Self::Document { reference, .. }
            | Self::ContainerUpload { reference } => reference,
        }
    }

    fn protocol(self) -> Result<MessagesFileBlock, MessagesCodecError> {
        Ok(match self {
            Self::Image { reference } => MessagesFileBlock::Image(reference.protocol()?),
            Self::Document {
                reference,
                title,
                context,
                citations,
            } => MessagesFileBlock::Document {
                reference: reference.protocol()?,
                title,
                context,
                citations,
            },
            Self::ContainerUpload { reference } => {
                MessagesFileBlock::ContainerUpload(reference.protocol()?)
            }
        })
    }
}

impl fmt::Debug for AnthropicMessageFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Image { reference } => formatter
                .debug_struct("AnthropicMessageFile::Image")
                .field("reference", reference)
                .finish(),
            Self::Document {
                reference,
                title,
                context,
                citations,
            } => formatter
                .debug_struct("AnthropicMessageFile::Document")
                .field("reference", reference)
                .field("title_bytes", &title.as_ref().map(String::len))
                .field("context_bytes", &context.as_ref().map(String::len))
                .field("citations", citations)
                .finish(),
            Self::ContainerUpload { reference } => formatter
                .debug_struct("AnthropicMessageFile::ContainerUpload")
                .field("reference", reference)
                .finish(),
        }
    }
}

impl AnthropicFileReference {
    fn protocol(self) -> Result<MessagesFileReference, MessagesCodecError> {
        MessagesFileReference::new(self.file_id, self.scope)
    }
}

fn rejected(path: &str, reason: &str) -> ProviderAnnotationError {
    ProviderAnnotationError::Rejected {
        path: path.to_string(),
        reason: reason.to_string(),
    }
}

/// Anthropic controls attached to one canonical message-content node.
///
/// Provider-native file references and tool changes use an empty text part as
/// a fail-closed anchor. Prompt caching may decorate file blocks, while tool
/// changes remain mutually exclusive with every other block replacement.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicContentOptions {
    cache_ttl: Option<AnthropicCacheTtl>,
    tool_change: Option<MidConversationToolChange>,
    file: Option<AnthropicMessageFile>,
}

impl AnthropicContentOptions {
    pub const fn new() -> Self {
        Self {
            cache_ttl: None,
            tool_change: None,
            file: None,
        }
    }

    pub const fn five_minutes() -> Self {
        Self::new().with_cache_ttl(AnthropicCacheTtl::FiveMinutes)
    }

    pub const fn one_hour() -> Self {
        Self::new().with_cache_ttl(AnthropicCacheTtl::OneHour)
    }

    pub fn for_tool_change(tool_change: MidConversationToolChange) -> Self {
        Self::new().with_tool_change(tool_change)
    }

    pub const fn with_cache_ttl(mut self, ttl: AnthropicCacheTtl) -> Self {
        self.cache_ttl = Some(ttl);
        self
    }

    pub fn with_tool_change(mut self, tool_change: MidConversationToolChange) -> Self {
        self.tool_change = Some(tool_change);
        self
    }

    pub fn with_file(mut self, file: AnthropicMessageFile) -> Self {
        self.file = Some(file);
        self
    }

    pub const fn cache_ttl(&self) -> Option<AnthropicCacheTtl> {
        self.cache_ttl
    }

    pub fn tool_change(&self) -> Option<&MidConversationToolChange> {
        self.tool_change.as_ref()
    }

    pub fn file(&self) -> Option<&AnthropicMessageFile> {
        self.file.as_ref()
    }

    /// Build the canonical empty-text anchor for one tool-change block.
    pub fn tool_change_part(
        tool_change: MidConversationToolChange,
    ) -> Result<MessagePart, ProviderAnnotationError> {
        MessagePart::text("").with_provider_annotation(&Self::for_tool_change(tool_change))
    }

    /// Build the canonical empty-text anchor for one scope-bound file block.
    pub fn file_part(file: AnthropicMessageFile) -> Result<MessagePart, ProviderAnnotationError> {
        MessagePart::text("").with_provider_annotation(&Self::new().with_file(file))
    }
}

impl TypedProviderAnnotation for AnthropicContentOptions {
    type Target = ContentAnnotationTarget;

    const NAMESPACE: &'static str = "anthropic";
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderAnnotationError> {
        if self.tool_change.is_some() && (self.cache_ttl.is_some() || self.file.is_some()) {
            return Err(ProviderAnnotationError::Rejected {
                path: "content".to_string(),
                reason: "a tool change cannot share one content block with caching or a file"
                    .to_string(),
            });
        }
        if let Some(tool_change) = &self.tool_change {
            tool_change
                .validate()
                .map_err(|_| ProviderAnnotationError::Rejected {
                    path: "content.tool_change".to_string(),
                    reason: "tool references must use bounded printable names".to_string(),
                })?;
        }
        if let Some(file) = &self.file {
            file.reference().validate()?;
        }
        Ok(())
    }
}

/// Anthropic controls attached to one canonical [`ToolSpec`].
///
/// Prompt caching, Anthropic-defined tool projection, and advanced tool-use
/// controls share one provider namespace on the tool node. Keeping them in one
/// annotation avoids numeric selectors and prevents duplicate `anthropic`
/// envelopes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicToolOptions {
    cache_ttl: Option<AnthropicCacheTtl>,
    anthropic_tool: Option<AnthropicTool>,
    allowed_callers: Vec<ToolCaller>,
    strict: Option<bool>,
    defer_loading: Option<bool>,
}

impl AnthropicToolOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn for_tool(anthropic_tool: AnthropicTool) -> Self {
        Self::new().with_anthropic_tool(anthropic_tool)
    }

    pub const fn with_cache_ttl(mut self, ttl: AnthropicCacheTtl) -> Self {
        self.cache_ttl = Some(ttl);
        self
    }

    pub fn with_anthropic_tool(mut self, anthropic_tool: AnthropicTool) -> Self {
        self.anthropic_tool = Some(anthropic_tool);
        self
    }

    pub fn with_allowed_callers(mut self, callers: impl IntoIterator<Item = ToolCaller>) -> Self {
        self.allowed_callers = callers.into_iter().collect();
        self
    }

    pub const fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    pub const fn with_defer_loading(mut self, defer_loading: bool) -> Self {
        self.defer_loading = Some(defer_loading);
        self
    }

    pub const fn cache_ttl(&self) -> Option<AnthropicCacheTtl> {
        self.cache_ttl
    }

    pub fn anthropic_tool(&self) -> Option<&AnthropicTool> {
        self.anthropic_tool.as_ref()
    }

    pub fn allowed_callers(&self) -> &[ToolCaller] {
        &self.allowed_callers
    }

    pub const fn strict(&self) -> Option<bool> {
        self.strict
    }

    pub const fn defer_loading(&self) -> Option<bool> {
        self.defer_loading
    }

    /// Build the canonical unified-tool anchor for this Anthropic-defined tool.
    pub fn into_tool_spec(self) -> Result<ToolSpec, AnthropicToolSpecError> {
        let name = self
            .anthropic_tool
            .as_ref()
            .ok_or(AnthropicToolSpecError::MissingAnthropicTool)?
            .canonical_name()
            .to_string();
        ToolSpec::new(name, None, anthropic_tool_anchor_schema())?
            .with_provider_annotation(&self)
            .map_err(AnthropicToolSpecError::Annotation)
    }
}

impl TypedProviderAnnotation for AnthropicToolOptions {
    type Target = ToolAnnotationTarget;

    const NAMESPACE: &'static str = "anthropic";
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AnthropicToolSpecError {
    #[error("Anthropic tool options require an Anthropic-defined tool")]
    MissingAnthropicTool,
    #[error("invalid Anthropic-defined tool anchor: {0}")]
    Tool(#[from] InvalidToolSpec),
    #[error("invalid Anthropic tool annotation: {0}")]
    Annotation(ProviderAnnotationError),
}

/// Strict projection from provider-owned annotations to bounded Messages controls.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnthropicAnnotationResolver;

impl MessagesAnnotationResolver for AnthropicAnnotationResolver {
    fn resolve_message(
        &self,
        annotations: &MessageAnnotations,
    ) -> Result<MessageNodeOptions, MessagesCodecError> {
        resolve::<AnthropicMessageCache, _>(annotations, "message")
            .map(|value| value.map_or_else(MessageNodeOptions::default, message_options))
    }

    fn resolve_content(
        &self,
        annotations: &ContentAnnotations,
    ) -> Result<ContentNodeOptions, MessagesCodecError> {
        match resolve::<AnthropicContentOptions, _>(annotations, "content")? {
            Some(value) => content_options(value),
            None => Ok(ContentNodeOptions::default()),
        }
    }

    fn resolve_tool(
        &self,
        annotations: &ToolAnnotations,
    ) -> Result<ToolNodeOptions, MessagesCodecError> {
        resolve::<AnthropicToolOptions, _>(annotations, "tool")
            .map(|value| value.map_or_else(ToolNodeOptions::default, tool_options))
    }
}

fn resolve<T, Target>(
    annotations: &siumai_core::ProviderAnnotations<Target>,
    node: &'static str,
) -> Result<Option<T>, MessagesCodecError>
where
    T: serde::de::DeserializeOwned + TypedProviderAnnotation<Target = Target>,
    Target: siumai_core::ProviderAnnotationTarget,
{
    annotations
        .decode::<T>()
        .map_err(|source| invalid_annotation(node, source))
}

fn invalid_annotation(node: &'static str, source: ProviderAnnotationError) -> MessagesCodecError {
    MessagesCodecError::InvalidAnnotation { node, source }
}

fn cache_control(ttl: AnthropicCacheTtl) -> CacheControl {
    CacheControl::new(ttl.protocol())
}

fn message_options(annotation: AnthropicMessageCache) -> MessageNodeOptions {
    MessageNodeOptions::default().with_cache_control(cache_control(annotation.ttl()))
}

fn content_options(
    annotation: AnthropicContentOptions,
) -> Result<ContentNodeOptions, MessagesCodecError> {
    let AnthropicContentOptions {
        cache_ttl,
        tool_change,
        file,
    } = annotation;
    let mut options = ContentNodeOptions::default();
    if let Some(ttl) = cache_ttl {
        options = options.with_cache_control(cache_control(ttl));
    }
    if let Some(tool_change) = tool_change {
        options = options.with_tool_change(tool_change);
    }
    if let Some(file) = file {
        options = options.with_file(file.protocol()?);
    }
    Ok(options)
}

fn tool_options(annotation: AnthropicToolOptions) -> ToolNodeOptions {
    let AnthropicToolOptions {
        cache_ttl,
        anthropic_tool,
        allowed_callers,
        strict,
        defer_loading,
    } = annotation;
    let mut options = ToolNodeOptions::default().with_allowed_callers(allowed_callers);
    if let Some(strict) = strict {
        options = options.with_strict(strict);
    }
    if let Some(defer_loading) = defer_loading {
        options = options.with_defer_loading(defer_loading);
    }
    if let Some(anthropic_tool) = anthropic_tool {
        options = options.with_anthropic_tool(anthropic_tool);
    }
    if let Some(ttl) = cache_ttl {
        options = options.with_cache_control(cache_control(ttl));
    }
    options
}
