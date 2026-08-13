use serde::{Deserialize, Serialize};
use siumai_core::{
    ContentAnnotationTarget, ContentAnnotations, InvalidToolSpec, MessageAnnotationTarget,
    MessageAnnotations, ProviderAnnotationError, ToolAnnotationTarget, ToolAnnotations, ToolSpec,
    TypedProviderAnnotation,
};
use siumai_protocol_anthropic::messages::{
    API_MODE_ID, AnthropicTool, CacheControl, CacheTtl, ComputerToolOptions, ContentNodeOptions,
    MessageNodeOptions, MessagesAnnotationResolver, MessagesCodecError, TextEditorToolOptions,
    ToolNodeOptions, WebSearchToolOptions, anthropic_tool_anchor_schema,
};
use thiserror::Error;

/// Prompt-cache lifetime supported by Claude on Vertex AI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoogleVertexAnthropicCacheTtl {
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

impl GoogleVertexAnthropicCacheTtl {
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
            ttl: GoogleVertexAnthropicCacheTtl,
        }

        impl $name {
            pub const fn new(ttl: GoogleVertexAnthropicCacheTtl) -> Self {
                Self { ttl }
            }

            pub const fn five_minutes() -> Self {
                Self::new(GoogleVertexAnthropicCacheTtl::FiveMinutes)
            }

            pub const fn one_hour() -> Self {
                Self::new(GoogleVertexAnthropicCacheTtl::OneHour)
            }

            pub const fn ttl(self) -> GoogleVertexAnthropicCacheTtl {
                self.ttl
            }
        }

        impl TypedProviderAnnotation for $name {
            type Target = $target;

            const NAMESPACE: &'static str = "google";
            const API_MODE: Option<&'static str> = Some(API_MODE_ID);
        }
    };
}

cache_annotation!(GoogleVertexAnthropicMessageCache, MessageAnnotationTarget);
cache_annotation!(GoogleVertexAnthropicContentCache, ContentAnnotationTarget);

/// Anthropic-defined tool contracts verified for Google Cloud.
///
/// The enum intentionally omits code execution, web fetch, advisor, and MCP
/// connector contracts, which are not supported on Google Cloud.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "options")]
#[non_exhaustive]
pub enum GoogleVertexAnthropicTool {
    #[serde(rename = "web_search_20260318")]
    WebSearch20260318(WebSearchToolOptions),
    #[serde(rename = "tool_search_regex_20251119")]
    ToolSearchRegex20251119,
    #[serde(rename = "tool_search_bm25_20251119")]
    ToolSearchBm25V20251119,
    #[serde(rename = "memory_20250818")]
    Memory20250818,
    #[serde(rename = "bash_20250124")]
    Bash20250124,
    #[serde(rename = "text_editor_20250728")]
    TextEditor20250728(TextEditorToolOptions),
    #[serde(rename = "computer_20251124")]
    Computer20251124(ComputerToolOptions),
}

impl GoogleVertexAnthropicTool {
    pub const fn web_search_20260318(options: WebSearchToolOptions) -> Self {
        Self::WebSearch20260318(options)
    }

    pub const fn tool_search_regex_20251119() -> Self {
        Self::ToolSearchRegex20251119
    }

    pub const fn tool_search_bm25_20251119() -> Self {
        Self::ToolSearchBm25V20251119
    }

    pub const fn memory_20250818() -> Self {
        Self::Memory20250818
    }

    pub const fn bash_20250124() -> Self {
        Self::Bash20250124
    }

    pub const fn text_editor_20250728(options: TextEditorToolOptions) -> Self {
        Self::TextEditor20250728(options)
    }

    pub const fn computer_20251124(options: ComputerToolOptions) -> Self {
        Self::Computer20251124(options)
    }

    fn canonical_name(&self) -> &'static str {
        match self {
            Self::WebSearch20260318(_) => "web_search",
            Self::ToolSearchRegex20251119 => "tool_search_tool_regex",
            Self::ToolSearchBm25V20251119 => "tool_search_tool_bm25",
            Self::Memory20250818 => "memory",
            Self::Bash20250124 => "bash",
            Self::TextEditor20250728(_) => "str_replace_based_edit_tool",
            Self::Computer20251124(_) => "computer",
        }
    }

    fn protocol(&self) -> AnthropicTool {
        match self {
            Self::WebSearch20260318(options) => AnthropicTool::web_search_20260318(options.clone()),
            Self::ToolSearchRegex20251119 => AnthropicTool::ToolSearchRegex20251119,
            Self::ToolSearchBm25V20251119 => AnthropicTool::ToolSearchBm25V20251119,
            Self::Memory20250818 => AnthropicTool::Memory20250818,
            Self::Bash20250124 => AnthropicTool::Bash20250124,
            Self::TextEditor20250728(options) => AnthropicTool::text_editor_20250728(*options),
            Self::Computer20251124(options) => AnthropicTool::computer_20251124(*options),
        }
    }
}

/// Vertex-owned controls attached to one function or client-tool definition.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GoogleVertexAnthropicToolOptions {
    cache_ttl: Option<GoogleVertexAnthropicCacheTtl>,
    anthropic_tool: Option<GoogleVertexAnthropicTool>,
    strict: Option<bool>,
}

impl GoogleVertexAnthropicToolOptions {
    pub const fn new() -> Self {
        Self {
            cache_ttl: None,
            anthropic_tool: None,
            strict: None,
        }
    }

    pub fn for_tool(anthropic_tool: GoogleVertexAnthropicTool) -> Self {
        Self::new().with_anthropic_tool(anthropic_tool)
    }

    pub const fn with_cache_ttl(mut self, ttl: GoogleVertexAnthropicCacheTtl) -> Self {
        self.cache_ttl = Some(ttl);
        self
    }

    pub fn with_anthropic_tool(mut self, anthropic_tool: GoogleVertexAnthropicTool) -> Self {
        self.anthropic_tool = Some(anthropic_tool);
        self
    }

    pub const fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    pub const fn cache_ttl(&self) -> Option<GoogleVertexAnthropicCacheTtl> {
        self.cache_ttl
    }

    pub fn anthropic_tool(&self) -> Option<&GoogleVertexAnthropicTool> {
        self.anthropic_tool.as_ref()
    }

    pub const fn strict(&self) -> Option<bool> {
        self.strict
    }

    /// Build the canonical anchor for one supported Anthropic-defined tool.
    pub fn into_tool_spec(self) -> Result<ToolSpec, GoogleVertexAnthropicToolSpecError> {
        let name = self
            .anthropic_tool
            .as_ref()
            .ok_or(GoogleVertexAnthropicToolSpecError::MissingTool)?
            .canonical_name();
        ToolSpec::new(name, None, anthropic_tool_anchor_schema())?
            .with_provider_annotation(&self)
            .map_err(GoogleVertexAnthropicToolSpecError::Annotation)
    }
}

impl TypedProviderAnnotation for GoogleVertexAnthropicToolOptions {
    type Target = ToolAnnotationTarget;

    const NAMESPACE: &'static str = "google";
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GoogleVertexAnthropicToolSpecError {
    #[error("Google Vertex Anthropic tool options require a supported Anthropic tool")]
    MissingTool,
    #[error("invalid Google Vertex Anthropic tool anchor: {0}")]
    Tool(#[from] InvalidToolSpec),
    #[error("invalid Google Vertex Anthropic tool annotation: {0}")]
    Annotation(ProviderAnnotationError),
}

/// Strict projection from Google-owned annotations to bounded Messages controls.
#[derive(Debug, Clone, Copy, Default)]
pub struct GoogleVertexAnthropicAnnotationResolver;

impl MessagesAnnotationResolver for GoogleVertexAnthropicAnnotationResolver {
    fn resolve_message(
        &self,
        annotations: &MessageAnnotations,
    ) -> Result<MessageNodeOptions, MessagesCodecError> {
        resolve::<GoogleVertexAnthropicMessageCache, _>(annotations, "message")
            .map(|value| value.map_or_else(MessageNodeOptions::default, message_options))
    }

    fn resolve_content(
        &self,
        annotations: &ContentAnnotations,
    ) -> Result<ContentNodeOptions, MessagesCodecError> {
        resolve::<GoogleVertexAnthropicContentCache, _>(annotations, "content")
            .map(|value| value.map_or_else(ContentNodeOptions::default, content_options))
    }

    fn resolve_tool(
        &self,
        annotations: &ToolAnnotations,
    ) -> Result<ToolNodeOptions, MessagesCodecError> {
        resolve::<GoogleVertexAnthropicToolOptions, _>(annotations, "tool")
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
        .map_err(|source| MessagesCodecError::InvalidAnnotation { node, source })
}

fn cache_control(ttl: GoogleVertexAnthropicCacheTtl) -> CacheControl {
    CacheControl::new(ttl.protocol())
}

fn message_options(annotation: GoogleVertexAnthropicMessageCache) -> MessageNodeOptions {
    MessageNodeOptions::default().with_cache_control(cache_control(annotation.ttl()))
}

fn content_options(annotation: GoogleVertexAnthropicContentCache) -> ContentNodeOptions {
    ContentNodeOptions::default().with_cache_control(cache_control(annotation.ttl()))
}

fn tool_options(annotation: GoogleVertexAnthropicToolOptions) -> ToolNodeOptions {
    let mut options = ToolNodeOptions::default();
    if let Some(anthropic_tool) = annotation.anthropic_tool() {
        options = options.with_anthropic_tool(anthropic_tool.protocol());
    }
    if let Some(strict) = annotation.strict() {
        options = options.with_strict(strict);
    }
    if let Some(ttl) = annotation.cache_ttl() {
        options = options.with_cache_control(cache_control(ttl));
    }
    options
}
