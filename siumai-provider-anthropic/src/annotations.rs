use serde::{Deserialize, Serialize};
use siumai_core::{
    ContentAnnotationTarget, ContentAnnotations, InvalidToolSpec, MessageAnnotationTarget,
    MessageAnnotations, MessagePart, ProviderAnnotationError, ToolAnnotationTarget,
    ToolAnnotations, ToolSpec, TypedProviderAnnotation,
};
use siumai_protocol_anthropic::messages::{
    API_MODE_ID, AnthropicTool, CacheControl, CacheTtl, ContentNodeOptions, MessageNodeOptions,
    MessagesAnnotationResolver, MessagesCodecError, MidConversationToolChange, ToolCaller,
    ToolNodeOptions, anthropic_tool_anchor_schema,
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

/// Anthropic controls attached to one canonical message-content node.
///
/// A tool change uses an empty text part as a fail-closed anchor and is replaced
/// by the Messages codec with a typed `tool_addition` or `tool_removal` block.
/// Prompt caching and tool changes are mutually exclusive on the same block.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicContentOptions {
    cache_ttl: Option<AnthropicCacheTtl>,
    tool_change: Option<MidConversationToolChange>,
}

impl AnthropicContentOptions {
    pub const fn new() -> Self {
        Self {
            cache_ttl: None,
            tool_change: None,
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

    pub const fn cache_ttl(&self) -> Option<AnthropicCacheTtl> {
        self.cache_ttl
    }

    pub fn tool_change(&self) -> Option<&MidConversationToolChange> {
        self.tool_change.as_ref()
    }

    /// Build the canonical empty-text anchor for one tool-change block.
    pub fn tool_change_part(
        tool_change: MidConversationToolChange,
    ) -> Result<MessagePart, ProviderAnnotationError> {
        MessagePart::text("").with_provider_annotation(&Self::for_tool_change(tool_change))
    }
}

impl TypedProviderAnnotation for AnthropicContentOptions {
    type Target = ContentAnnotationTarget;

    const NAMESPACE: &'static str = "anthropic";
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderAnnotationError> {
        if self.cache_ttl.is_some() && self.tool_change.is_some() {
            return Err(ProviderAnnotationError::Rejected {
                path: "content".to_string(),
                reason: "prompt caching and a tool change cannot share one content block"
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
        resolve::<AnthropicContentOptions, _>(annotations, "content")
            .map(|value| value.map_or_else(ContentNodeOptions::default, content_options))
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

fn content_options(annotation: AnthropicContentOptions) -> ContentNodeOptions {
    let AnthropicContentOptions {
        cache_ttl,
        tool_change,
    } = annotation;
    let mut options = ContentNodeOptions::default();
    if let Some(ttl) = cache_ttl {
        options = options.with_cache_control(cache_control(ttl));
    }
    if let Some(tool_change) = tool_change {
        options = options.with_tool_change(tool_change);
    }
    options
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
