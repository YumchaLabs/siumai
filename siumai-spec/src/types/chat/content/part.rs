use serde::{Deserialize, Serialize};

use crate::types::{ProviderMetadataMap, ProviderOptionsMap};

use super::super::metadata::{ToolCallInfo, ToolResultInfo};
use super::{
    FilePartSource, ImageDetail, MediaSource, ProviderReference, ToolResultContentPart,
    ToolResultOutput,
};

/// Source citation payload aligned with the AI SDK URL/document union.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "sourceType", rename_all = "lowercase")]
pub enum SourcePart {
    /// URL-backed source.
    Url {
        /// Source URL.
        url: String,
        /// Optional human-readable title.
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    /// Document/file-backed source.
    Document {
        /// IANA media type for the document source.
        #[serde(rename = "mediaType", alias = "media_type")]
        media_type: String,
        /// Human-readable document title.
        title: String,
        /// Optional filename.
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

impl SourcePart {
    /// Return the Vercel-aligned source type discriminator.
    pub fn source_type(&self) -> &'static str {
        match self {
            Self::Url { .. } => "url",
            Self::Document { .. } => "document",
        }
    }

    /// Return the title if one exists for this source.
    pub fn title(&self) -> Option<&str> {
        match self {
            Self::Url { title, .. } => title.as_deref(),
            Self::Document { title, .. } => Some(title.as_str()),
        }
    }

    /// Return the URL when this is a URL-backed source.
    pub fn url(&self) -> Option<&str> {
        match self {
            Self::Url { url, .. } => Some(url.as_str()),
            Self::Document { .. } => None,
        }
    }

    /// Return the media type when this is a document source.
    pub fn media_type(&self) -> Option<&str> {
        match self {
            Self::Url { .. } => None,
            Self::Document { media_type, .. } => Some(media_type.as_str()),
        }
    }

    /// Return the filename when this is a document source and a filename exists.
    pub fn filename(&self) -> Option<&str> {
        match self {
            Self::Url { .. } => None,
            Self::Document { filename, .. } => filename.as_deref(),
        }
    }
}

/// Content part - provider-agnostic multimodal content
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ContentPart {
    /// Text content
    Text {
        text: String,

        /// Provider-specific request options (Vercel-aligned).
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadataMap>,
    },

    /// Image content - supports URL, base64, binary data, or provider references
    Image {
        /// Image data source
        #[serde(flatten)]
        source: FilePartSource,
        /// Optional detail level (for providers that support it, e.g., OpenAI)
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,

        /// Provider-specific request options (Vercel-aligned).
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadataMap>,
    },

    /// Audio content - supports URL, base64, or binary data
    Audio {
        /// Audio data source
        #[serde(flatten)]
        source: MediaSource,
        /// Media type (e.g., "audio/wav", "audio/mp3", "audio/mpeg")
        #[serde(
            rename = "mediaType",
            alias = "media_type",
            skip_serializing_if = "Option::is_none"
        )]
        media_type: Option<String>,

        /// Provider-specific request options (Vercel-aligned).
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadataMap>,
    },

    /// File content (PDF, documents, etc.)
    File {
        /// File data source
        #[serde(flatten)]
        source: FilePartSource,
        /// Media type (e.g., "application/pdf", "text/plain")
        #[serde(rename = "mediaType", alias = "media_type")]
        media_type: String,
        /// Optional filename (for providers that support it)
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,

        /// Provider-specific request options (Vercel-aligned).
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadataMap>,
    },

    /// File content generated as part of model reasoning.
    #[serde(rename = "reasoning-file")]
    ReasoningFile {
        /// File data source
        #[serde(flatten)]
        source: MediaSource,
        /// IANA media type of the file.
        #[serde(rename = "mediaType", alias = "media_type")]
        media_type: String,

        /// Provider-specific request options (Vercel-aligned).
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadataMap>,
    },

    /// Provider-specific content with no standardized payload beyond kind + provider fields.
    Custom {
        /// Provider-specific custom content kind, e.g. `openai.compaction`.
        kind: String,

        /// Provider-specific request options (Vercel-aligned).
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadataMap>,
    },

    /// Source citation (Vercel-aligned).
    ///
    /// Providers may surface citations/attributions as a list of sources.
    /// This content part represents a single source entry in the unified content stream.
    #[serde(rename = "source")]
    Source {
        /// Source id (Vercel-aligned, e.g. "id-0").
        id: String,
        /// Strict URL/document source union.
        #[serde(flatten)]
        source: SourcePart,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadataMap>,
    },

    /// Tool call (function call request from AI)
    ///
    /// This represents a request from the AI model to call a tool/function.
    /// The tool can be either user-defined (executed by your code) or
    /// provider-defined (executed by the provider, e.g., web search).
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    /// use serde_json::json;
    ///
    /// let tool_call = ContentPart::tool_call(
    ///     "call_123",
    ///     "search",
    ///     json!({"query": "rust programming"}),
    ///     None, // user-defined function
    /// );
    /// ```
    #[serde(rename = "tool-call")]
    ToolCall {
        /// Tool call ID (used to match with tool result)
        #[serde(rename = "toolCallId")]
        tool_call_id: String,

        /// Tool/function name
        #[serde(rename = "toolName")]
        tool_name: String,

        /// Arguments as JSON value
        #[serde(rename = "input")]
        arguments: serde_json::Value,

        /// Whether this tool will be executed by the provider
        ///
        /// - `Some(true)`: Provider-defined tool (e.g., web search, code execution)
        /// - `None` or `Some(false)`: User-defined function
        #[serde(
            rename = "providerExecuted",
            alias = "provider_executed",
            skip_serializing_if = "Option::is_none"
        )]
        provider_executed: Option<bool>,

        /// Whether the tool is dynamic/runtime-defined instead of declared statically.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,

        /// Whether the tool call is invalid (e.g. unparsable or unknown tool).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        invalid: Option<bool>,

        /// Error payload explaining why the tool call is invalid when available.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<serde_json::Value>,

        /// Optional human-readable title for the tool call.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,

        /// Provider-specific request options (Vercel-aligned).
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadataMap>,
    },

    /// Tool approval response (for MCP approval workflows)
    ///
    /// This represents a user/system approval decision for a pending tool call,
    /// aligned with Vercel AI SDK's `tool-approval-response` part.
    #[serde(rename = "tool-approval-response")]
    ToolApprovalResponse {
        /// The approval request ID (MCP approval request item id)
        #[serde(rename = "approvalId")]
        approval_id: String,
        /// Whether the tool call was approved
        approved: bool,
        /// Optional approval/denial reason.
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        /// Whether the approved tool call is provider-executed.
        #[serde(
            rename = "providerExecuted",
            alias = "provider_executed",
            skip_serializing_if = "Option::is_none"
        )]
        provider_executed: Option<bool>,
        /// Provider-specific request options (Vercel-aligned).
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,
    },

    /// Tool approval request (for MCP approval workflows)
    ///
    /// This represents a model request to obtain user/system approval for a pending tool call,
    /// aligned with Vercel AI SDK's `tool-approval-request` part.
    #[serde(rename = "tool-approval-request")]
    ToolApprovalRequest {
        /// The approval request ID (MCP approval request item id)
        #[serde(rename = "approvalId")]
        approval_id: String,

        /// The tool call ID associated with this approval request.
        #[serde(rename = "toolCallId")]
        tool_call_id: String,

        /// Provider-specific request options (forward-compatible).
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadataMap>,
    },

    /// Tool result (function execution result)
    ///
    /// This represents the result of executing a tool/function.
    /// It matches a previous tool call by ID.
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ContentPart, ToolResultOutput};
    /// use serde_json::json;
    ///
    /// // Success result
    /// let result = ContentPart::tool_result_json(
    ///     "call_123",
    ///     "search",
    ///     json!({"results": ["..."]}),
    /// );
    ///
    /// // Error result
    /// let error = ContentPart::tool_error(
    ///     "call_123",
    ///     "search",
    ///     "API timeout",
    /// );
    /// ```
    #[serde(rename = "tool-result")]
    ToolResult {
        /// Tool call ID (matches the tool call)
        #[serde(rename = "toolCallId")]
        tool_call_id: String,

        /// Tool/function name
        #[serde(rename = "toolName")]
        tool_name: String,

        /// Structured output
        #[serde(rename = "output")]
        output: ToolResultOutput,

        /// Optional original tool input, aligned with AI SDK `tool-result.input`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        input: Option<serde_json::Value>,

        /// Whether this result was generated by the provider
        #[serde(
            rename = "providerExecuted",
            alias = "provider_executed",
            skip_serializing_if = "Option::is_none"
        )]
        provider_executed: Option<bool>,

        /// Whether the tool is dynamic/runtime-defined instead of declared statically.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dynamic: Option<bool>,

        /// Whether the result is preliminary.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        preliminary: Option<bool>,

        /// Optional human-readable title for the tool result.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,

        /// Provider-specific request options (Vercel-aligned).
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadataMap>,
    },

    /// Reasoning/thinking content
    ///
    /// This represents the model's reasoning or thinking process.
    /// Supported by models like:
    /// - OpenAI o1, o3
    /// - DeepSeek-R1
    /// - Gemini (thought summary)
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    ///
    /// let reasoning = ContentPart::reasoning(
    ///     "Let me think about this step by step..."
    /// );
    /// ```
    Reasoning {
        text: String,

        /// Provider-specific request options (Vercel-aligned).
        #[serde(
            rename = "providerOptions",
            alias = "provider_options",
            default,
            skip_serializing_if = "ProviderOptionsMap::is_empty"
        )]
        provider_options: ProviderOptionsMap,

        /// Provider-specific metadata (Vercel-aligned).
        #[serde(
            rename = "providerMetadata",
            alias = "provider_metadata",
            skip_serializing_if = "Option::is_none"
        )]
        provider_metadata: Option<ProviderMetadataMap>,
    },
}

impl ContentPart {
    /// Create a text content part
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create an image content part from URL
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            source: FilePartSource::url(url),
            detail: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create an image content part from URL with detail level
    pub fn image_url_with_detail(url: impl Into<String>, detail: ImageDetail) -> Self {
        Self::Image {
            source: FilePartSource::url(url),
            detail: Some(detail),
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create an image content part from base64 data
    pub fn image_base64(data: impl Into<String>) -> Self {
        Self::Image {
            source: FilePartSource::base64(data),
            detail: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create an image content part from binary data
    pub fn image_binary(data: Vec<u8>) -> Self {
        Self::Image {
            source: FilePartSource::binary(data),
            detail: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create an image content part from provider-managed file references.
    pub fn image_provider_reference(provider_reference: impl Into<ProviderReference>) -> Self {
        Self::Image {
            source: FilePartSource::provider_reference(provider_reference),
            detail: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create an audio content part from URL
    pub fn audio_url(url: impl Into<String>, media_type: Option<String>) -> Self {
        Self::Audio {
            source: MediaSource::url(url),
            media_type,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create an audio content part from base64 data
    pub fn audio_base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::Audio {
            source: MediaSource::base64(data),
            media_type: Some(media_type.into()),
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create an audio content part from binary data
    pub fn audio_binary(data: Vec<u8>, media_type: impl Into<String>) -> Self {
        Self::Audio {
            source: MediaSource::binary(data),
            media_type: Some(media_type.into()),
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a file content part from URL
    pub fn file_url(url: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::File {
            source: FilePartSource::url(url),
            media_type: media_type.into(),
            filename: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a file content part from base64 data
    pub fn file_base64(
        data: impl Into<String>,
        media_type: impl Into<String>,
        filename: Option<String>,
    ) -> Self {
        Self::File {
            source: FilePartSource::base64(data),
            media_type: media_type.into(),
            filename,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a file content part from binary data
    pub fn file_binary(
        data: Vec<u8>,
        media_type: impl Into<String>,
        filename: Option<String>,
    ) -> Self {
        Self::File {
            source: FilePartSource::binary(data),
            media_type: media_type.into(),
            filename,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a file content part from provider-managed file references.
    pub fn file_provider_reference(
        provider_reference: impl Into<ProviderReference>,
        media_type: impl Into<String>,
        filename: Option<String>,
    ) -> Self {
        Self::File {
            source: FilePartSource::provider_reference(provider_reference),
            media_type: media_type.into(),
            filename,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a reasoning file content part from URL.
    pub fn reasoning_file_url(url: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::ReasoningFile {
            source: MediaSource::url(url),
            media_type: media_type.into(),
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a reasoning file content part from base64 data.
    pub fn reasoning_file_base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::ReasoningFile {
            source: MediaSource::base64(data),
            media_type: media_type.into(),
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a reasoning file content part from binary data.
    pub fn reasoning_file_binary(data: Vec<u8>, media_type: impl Into<String>) -> Self {
        Self::ReasoningFile {
            source: MediaSource::binary(data),
            media_type: media_type.into(),
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a custom provider-specific content part.
    pub fn custom(kind: impl Into<String>) -> Self {
        Self::Custom {
            kind: kind.into(),
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a tool call content part
    ///
    /// # Arguments
    ///
    /// * `tool_call_id` - Unique ID for this tool call
    /// * `tool_name` - Name of the tool/function to call
    /// * `arguments` - JSON value of arguments
    /// * `provider_executed` - Whether this tool is executed by the provider
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    /// use serde_json::json;
    ///
    /// // User-defined function
    /// let call = ContentPart::tool_call(
    ///     "call_123",
    ///     "get_weather",
    ///     json!({"location": "Tokyo"}),
    ///     None,
    /// );
    ///
    /// // Provider-defined tool
    /// let search = ContentPart::tool_call(
    ///     "call_456",
    ///     "web_search",
    ///     json!({"query": "rust"}),
    ///     Some(true),
    /// );
    /// ```
    pub fn tool_call(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        arguments: serde_json::Value,
        provider_executed: Option<bool>,
    ) -> Self {
        Self::ToolCall {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            arguments,
            provider_executed,
            dynamic: None,
            invalid: None,
            error: None,
            title: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a source content part (compatibility helper).
    ///
    /// Prefer `source_url` or `source_document` for new code.
    pub fn source(
        id: impl Into<String>,
        source_type: impl Into<String>,
        url: impl Into<String>,
        title: impl Into<String>,
    ) -> Self {
        let id = id.into();
        let source_type = source_type.into();
        let url = url.into();
        let title = title.into();

        if source_type.eq_ignore_ascii_case("document") {
            return Self::source_document(id, "application/octet-stream", title, Some(url));
        }

        Self::source_url(id, url, title)
    }

    /// Create a URL-backed source content part.
    pub fn source_url(
        id: impl Into<String>,
        url: impl Into<String>,
        title: impl Into<String>,
    ) -> Self {
        Self::Source {
            id: id.into(),
            source: SourcePart::Url {
                url: url.into(),
                title: Some(title.into()),
            },
            provider_metadata: None,
        }
    }

    /// Create a document-backed source content part.
    pub fn source_document(
        id: impl Into<String>,
        media_type: impl Into<String>,
        title: impl Into<String>,
        filename: Option<String>,
    ) -> Self {
        Self::Source {
            id: id.into(),
            source: SourcePart::Document {
                media_type: media_type.into(),
                title: title.into(),
                filename,
            },
            provider_metadata: None,
        }
    }

    /// Create a tool approval response content part.
    pub fn tool_approval_response(approval_id: impl Into<String>, approved: bool) -> Self {
        Self::ToolApprovalResponse {
            approval_id: approval_id.into(),
            approved,
            reason: None,
            provider_executed: None,
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a tool approval response content part with a reason.
    pub fn tool_approval_response_with_reason(
        approval_id: impl Into<String>,
        approved: bool,
        reason: Option<String>,
    ) -> Self {
        Self::ToolApprovalResponse {
            approval_id: approval_id.into(),
            approved,
            reason,
            provider_executed: None,
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Create a tool approval request content part.
    pub fn tool_approval_request(
        approval_id: impl Into<String>,
        tool_call_id: impl Into<String>,
    ) -> Self {
        Self::ToolApprovalRequest {
            approval_id: approval_id.into(),
            tool_call_id: tool_call_id.into(),
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a tool approval request content part with provider metadata.
    pub fn tool_approval_request_with_metadata(
        approval_id: impl Into<String>,
        tool_call_id: impl Into<String>,
        provider_metadata: ProviderMetadataMap,
    ) -> Self {
        Self::ToolApprovalRequest {
            approval_id: approval_id.into(),
            tool_call_id: tool_call_id.into(),
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: Some(provider_metadata),
        }
    }

    /// Create a tool result content part with text output
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    ///
    /// let result = ContentPart::tool_result_text(
    ///     "call_123",
    ///     "get_weather",
    ///     "Temperature is 18掳C, sunny",
    /// );
    /// ```
    pub fn tool_result_text(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        result: impl Into<String>,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            output: ToolResultOutput::text(result),
            input: None,
            provider_executed: None,
            dynamic: None,
            preliminary: None,
            title: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a tool result content part with JSON output
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    /// use serde_json::json;
    ///
    /// let result = ContentPart::tool_result_json(
    ///     "call_123",
    ///     "get_weather",
    ///     json!({"temperature": 18, "condition": "sunny"}),
    /// );
    /// ```
    pub fn tool_result_json(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        result: serde_json::Value,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            output: ToolResultOutput::json(result),
            input: None,
            provider_executed: None,
            dynamic: None,
            preliminary: None,
            title: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a tool error content part with text error
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    ///
    /// let error = ContentPart::tool_error(
    ///     "call_123",
    ///     "get_weather",
    ///     "API timeout",
    /// );
    /// ```
    pub fn tool_error(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            output: ToolResultOutput::error_text(error),
            input: None,
            provider_executed: None,
            dynamic: None,
            preliminary: None,
            title: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a tool error content part with JSON error
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    /// use serde_json::json;
    ///
    /// let error = ContentPart::tool_error_json(
    ///     "call_123",
    ///     "get_weather",
    ///     json!({"error": "API timeout", "code": 504}),
    /// );
    /// ```
    pub fn tool_error_json(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        error: serde_json::Value,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            output: ToolResultOutput::error_json(error),
            input: None,
            provider_executed: None,
            dynamic: None,
            preliminary: None,
            title: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a tool result with execution denied
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    ///
    /// let denied = ContentPart::tool_execution_denied(
    ///     "call_123",
    ///     "delete_file",
    ///     Some("User rejected the operation".to_string()),
    /// );
    /// ```
    pub fn tool_execution_denied(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        reason: Option<String>,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            output: ToolResultOutput::execution_denied(reason),
            input: None,
            provider_executed: None,
            dynamic: None,
            preliminary: None,
            title: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a tool result with multimodal content
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ContentPart, ToolResultContentPart};
    ///
    /// let result = ContentPart::tool_result_content(
    ///     "call_123",
    ///     "generate_image",
    ///     vec![
    ///         ToolResultContentPart::text("Generated image:"),
    ///         ToolResultContentPart::image_url("https://example.com/image.png"),
    ///     ],
    /// );
    /// ```
    pub fn tool_result_content(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: Vec<ToolResultContentPart>,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            output: ToolResultOutput::content(content),
            input: None,
            provider_executed: None,
            dynamic: None,
            preliminary: None,
            title: None,
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Create a reasoning content part
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::ContentPart;
    ///
    /// let reasoning = ContentPart::reasoning(
    ///     "Let me think about this step by step..."
    /// );
    /// ```
    pub fn reasoning(text: impl Into<String>) -> Self {
        Self::Reasoning {
            text: text.into(),
            provider_options: ProviderOptionsMap::default(),
            provider_metadata: None,
        }
    }

    /// Mark a tool call/result as dynamic.
    pub fn with_tool_dynamic(mut self, dynamic: bool) -> Self {
        match &mut self {
            Self::ToolCall { dynamic: slot, .. } | Self::ToolResult { dynamic: slot, .. } => {
                *slot = Some(dynamic)
            }
            _ => {}
        }
        self
    }

    /// Attach a human-readable title to a tool call/result.
    pub fn with_tool_title(mut self, title: impl Into<String>) -> Self {
        match &mut self {
            Self::ToolCall { title: slot, .. } | Self::ToolResult { title: slot, .. } => {
                *slot = Some(title.into());
            }
            _ => {}
        }
        self
    }

    /// Mark a tool call as invalid.
    pub fn with_tool_invalid(mut self, invalid: bool) -> Self {
        if let Self::ToolCall { invalid: slot, .. } = &mut self {
            *slot = Some(invalid);
        }
        self
    }

    /// Attach an invalidity/error payload to a tool call.
    pub fn with_tool_error_value(mut self, error: serde_json::Value) -> Self {
        if let Self::ToolCall { error: slot, .. } = &mut self {
            *slot = Some(error);
        }
        self
    }

    /// Attach the original tool input to a tool result.
    pub fn with_tool_result_input(mut self, input: serde_json::Value) -> Self {
        if let Self::ToolResult { input: slot, .. } = &mut self {
            *slot = Some(input);
        }
        self
    }

    /// Mark a tool result as preliminary.
    pub fn with_tool_preliminary(mut self, preliminary: bool) -> Self {
        if let Self::ToolResult {
            preliminary: slot, ..
        } = &mut self
        {
            *slot = Some(preliminary);
        }
        self
    }

    /// Check if this is a text part
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text { .. })
    }

    /// Check if this is an image part
    pub fn is_image(&self) -> bool {
        matches!(self, Self::Image { .. })
    }

    /// Check if this is an audio part
    pub fn is_audio(&self) -> bool {
        matches!(self, Self::Audio { .. })
    }

    /// Check if this is a file part
    pub fn is_file(&self) -> bool {
        matches!(self, Self::File { .. })
    }

    /// Check if this is a reasoning file part.
    pub fn is_reasoning_file(&self) -> bool {
        matches!(self, Self::ReasoningFile { .. })
    }

    /// Check if this is a tool call
    pub fn is_tool_call(&self) -> bool {
        matches!(self, Self::ToolCall { .. })
    }

    /// Check if this is a tool result
    pub fn is_tool_result(&self) -> bool {
        matches!(self, Self::ToolResult { .. })
    }

    /// Check if this is a source
    pub fn is_source(&self) -> bool {
        matches!(self, Self::Source { .. })
    }

    /// Check if this is reasoning
    pub fn is_reasoning(&self) -> bool {
        matches!(self, Self::Reasoning { .. } | Self::ReasoningFile { .. })
    }

    /// Check if this is a custom provider-specific part.
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::Custom { .. })
    }

    /// Get the strict source payload when this is a source part.
    pub fn as_source(&self) -> Option<(&str, &SourcePart)> {
        match self {
            Self::Source { id, source, .. } => Some((id.as_str(), source)),
            _ => None,
        }
    }

    /// Get the text content if this is a text part
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text, .. } => Some(text),
            _ => None,
        }
    }

    /// Get the tool call ID if this is a tool call
    pub fn as_tool_call_id(&self) -> Option<&str> {
        match self {
            Self::ToolCall { tool_call_id, .. } => Some(tool_call_id),
            _ => None,
        }
    }

    /// Get the tool name if this is a tool call or tool result
    pub fn as_tool_name(&self) -> Option<&str> {
        match self {
            Self::ToolCall { tool_name, .. } | Self::ToolResult { tool_name, .. } => {
                Some(tool_name)
            }
            _ => None,
        }
    }

    /// Get tool call information if this is a tool call
    ///
    /// Returns a structured view of the tool call fields, making it easier to
    /// work with tool calls without manual pattern matching.
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ChatResponse, ContentPart};
    ///
    /// # let response = ChatResponse::empty();
    /// for tool_call in response.tool_calls() {
    ///     if let Some(info) = tool_call.as_tool_call() {
    ///         println!("Tool: {}", info.tool_name);
    ///         println!("ID: {}", info.tool_call_id);
    ///         println!("Args: {}", info.arguments);
    ///     }
    /// }
    /// ```
    pub fn as_tool_call(&self) -> Option<ToolCallInfo<'_>> {
        match self {
            Self::ToolCall {
                tool_call_id,
                tool_name,
                arguments,
                provider_executed,
                dynamic,
                invalid,
                error,
                title,
                ..
            } => Some(ToolCallInfo {
                tool_call_id,
                tool_name,
                input: arguments,
                arguments,
                provider_executed: provider_executed.as_ref(),
                dynamic: dynamic.as_ref(),
                invalid: invalid.as_ref(),
                error: error.as_ref(),
                title: title.as_deref(),
            }),
            _ => None,
        }
    }

    /// Get tool result information if this is a tool result
    ///
    /// Returns a structured view of the tool result fields.
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::{ChatMessage, ContentPart};
    ///
    /// # let msg = ChatMessage::user("test").build();
    /// for part in msg.content.as_multimodal().unwrap_or(&vec![]) {
    ///     if let Some(info) = part.as_tool_result() {
    ///         println!("Tool: {}", info.tool_name);
    ///         println!("ID: {}", info.tool_call_id);
    ///     }
    /// }
    /// ```
    pub fn as_tool_result(&self) -> Option<ToolResultInfo<'_>> {
        match self {
            Self::ToolResult {
                tool_call_id,
                tool_name,
                input,
                output,
                provider_executed,
                dynamic,
                preliminary,
                title,
                ..
            } => Some(ToolResultInfo {
                tool_call_id,
                tool_name,
                input: input.as_ref(),
                output,
                provider_executed: provider_executed.as_ref(),
                dynamic: dynamic.as_ref(),
                preliminary: preliminary.as_ref(),
                title: title.as_deref(),
            }),
            _ => None,
        }
    }

    /// Get provider options when this part supports request-side configuration.
    pub fn provider_options(&self) -> Option<&ProviderOptionsMap> {
        match self {
            Self::Text {
                provider_options, ..
            }
            | Self::Image {
                provider_options, ..
            }
            | Self::Audio {
                provider_options, ..
            }
            | Self::File {
                provider_options, ..
            }
            | Self::ReasoningFile {
                provider_options, ..
            }
            | Self::Custom {
                provider_options, ..
            }
            | Self::ToolCall {
                provider_options, ..
            }
            | Self::ToolApprovalResponse {
                provider_options, ..
            }
            | Self::ToolApprovalRequest {
                provider_options, ..
            }
            | Self::ToolResult {
                provider_options, ..
            }
            | Self::Reasoning {
                provider_options, ..
            } => Some(provider_options),
            Self::Source { .. } => None,
        }
    }

    /// Get mutable provider options when this part supports request-side configuration.
    pub fn provider_options_mut(&mut self) -> Option<&mut ProviderOptionsMap> {
        match self {
            Self::Text {
                provider_options, ..
            }
            | Self::Image {
                provider_options, ..
            }
            | Self::Audio {
                provider_options, ..
            }
            | Self::File {
                provider_options, ..
            }
            | Self::ReasoningFile {
                provider_options, ..
            }
            | Self::Custom {
                provider_options, ..
            }
            | Self::ToolCall {
                provider_options, ..
            }
            | Self::ToolApprovalResponse {
                provider_options, ..
            }
            | Self::ToolApprovalRequest {
                provider_options, ..
            }
            | Self::ToolResult {
                provider_options, ..
            }
            | Self::Reasoning {
                provider_options, ..
            } => Some(provider_options),
            Self::Source { .. } => None,
        }
    }

    /// Attach provider-specific request options to this content part.
    pub fn with_provider_option(
        mut self,
        provider_id: impl AsRef<str>,
        value: serde_json::Value,
    ) -> Self {
        if let Some(provider_options) = self.provider_options_mut() {
            provider_options.insert(provider_id, value);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::ContentPart;
    use super::SourcePart;
    use crate::types::ProviderReference;

    #[test]
    fn source_url_serializes_strict_url_shape() {
        let part = ContentPart::source_url("url:0", "https://example.com", "Example");

        let value = serde_json::to_value(&part).expect("serialize source");
        assert_eq!(value["type"], serde_json::json!("source"));
        assert_eq!(value["sourceType"], serde_json::json!("url"));
        assert_eq!(value["url"], serde_json::json!("https://example.com"));
        assert_eq!(value["title"], serde_json::json!("Example"));
        assert!(value.get("mediaType").is_none());
        assert!(value.get("filename").is_none());
    }

    #[test]
    fn source_document_serializes_document_fields() {
        let mut part = ContentPart::source_document(
            "doc:0",
            "application/pdf",
            "Reference PDF",
            Some("ref.pdf".to_string()),
        );

        if let ContentPart::Source {
            source: _,
            provider_metadata,
            ..
        } = &mut part
        {
            *provider_metadata = Some(std::collections::HashMap::from([(
                "anthropic".to_string(),
                serde_json::json!({
                    "startPageNumber": 1,
                    "endPageNumber": 2
                }),
            )]));
        }

        let value = serde_json::to_value(&part).expect("serialize source");
        assert_eq!(value["type"], serde_json::json!("source"));
        assert_eq!(value["sourceType"], serde_json::json!("document"));
        assert_eq!(value["mediaType"], serde_json::json!("application/pdf"));
        assert_eq!(value["title"], serde_json::json!("Reference PDF"));
        assert_eq!(value["filename"], serde_json::json!("ref.pdf"));
        assert_eq!(
            value["providerMetadata"]["anthropic"]["startPageNumber"],
            serde_json::json!(1)
        );

        let Some((id, source)) = part.as_source() else {
            panic!("expected source part");
        };
        assert_eq!(id, "doc:0");
        assert_eq!(
            source,
            &SourcePart::Document {
                media_type: "application/pdf".to_string(),
                title: "Reference PDF".to_string(),
                filename: Some("ref.pdf".to_string()),
            }
        );
    }

    #[test]
    fn legacy_source_json_deserializes_into_strict_source_union() {
        let value = serde_json::json!({
            "type": "source",
            "id": "doc:1",
            "sourceType": "document",
            "mediaType": "text/plain",
            "title": "Notes",
            "filename": "notes.txt"
        });

        let part = serde_json::from_value::<ContentPart>(value).expect("deserialize source");
        let Some((id, source)) = part.as_source() else {
            panic!("expected source part");
        };
        assert_eq!(id, "doc:1");
        assert_eq!(
            source,
            &SourcePart::Document {
                media_type: "text/plain".to_string(),
                title: "Notes".to_string(),
                filename: Some("notes.txt".to_string()),
            }
        );
    }

    #[test]
    fn tool_approval_parts_serialize_reason_and_provider_metadata() {
        let response = ContentPart::tool_approval_response_with_reason(
            "apr_1",
            false,
            Some("need manual review".to_string()),
        );
        let response_value = serde_json::to_value(&response).expect("serialize response part");
        assert_eq!(
            response_value["type"],
            serde_json::json!("tool-approval-response")
        );
        assert_eq!(
            response_value["reason"],
            serde_json::json!("need manual review")
        );

        let request = ContentPart::tool_approval_request_with_metadata(
            "apr_1",
            "call_1",
            std::collections::HashMap::from([(
                "openai".to_string(),
                serde_json::json!({ "itemId": "out_123" }),
            )]),
        );
        let request_value = serde_json::to_value(&request).expect("serialize request part");
        assert_eq!(
            request_value["type"],
            serde_json::json!("tool-approval-request")
        );
        assert_eq!(
            request_value["providerMetadata"]["openai"]["itemId"],
            serde_json::json!("out_123")
        );
    }

    #[test]
    fn reasoning_file_serializes_vercel_aligned_shape() {
        let mut part = ContentPart::reasoning_file_base64("aGVsbG8=", "image/png")
            .with_provider_option("google", serde_json::json!({ "thoughtSignature": "sig_1" }));

        if let ContentPart::ReasoningFile {
            provider_metadata, ..
        } = &mut part
        {
            *provider_metadata = Some(std::collections::HashMap::from([(
                "google".to_string(),
                serde_json::json!({ "thoughtSignature": "sig_1" }),
            )]));
        }

        let value = serde_json::to_value(&part).expect("serialize reasoning file");
        assert_eq!(value["type"], serde_json::json!("reasoning-file"));
        assert_eq!(value["mediaType"], serde_json::json!("image/png"));
        assert_eq!(
            value["providerOptions"]["google"]["thoughtSignature"],
            serde_json::json!("sig_1")
        );
        assert_eq!(
            value["providerMetadata"]["google"]["thoughtSignature"],
            serde_json::json!("sig_1")
        );
    }

    #[test]
    fn custom_part_serializes_kind_and_provider_options() {
        let part = ContentPart::custom("openai.compaction").with_provider_option(
            "openai",
            serde_json::json!({
                "type": "compaction",
                "itemId": "cmp_1",
            }),
        );

        let value = serde_json::to_value(&part).expect("serialize custom part");
        assert_eq!(value["type"], serde_json::json!("custom"));
        assert_eq!(value["kind"], serde_json::json!("openai.compaction"));
        assert_eq!(
            value["providerOptions"]["openai"]["type"],
            serde_json::json!("compaction")
        );
    }

    #[test]
    fn image_provider_reference_serializes_vercel_shape() {
        let part = ContentPart::image_provider_reference(ProviderReference::from([
            ("openai", "file-openai"),
            ("anthropic", "file-anthropic"),
        ]));

        let value = serde_json::to_value(&part).expect("serialize image provider reference");
        assert_eq!(value["type"], serde_json::json!("image"));
        assert_eq!(
            value["providerReference"],
            serde_json::json!({
                "anthropic": "file-anthropic",
                "openai": "file-openai"
            })
        );
        assert!(value.get("url").is_none());
        assert!(value.get("data").is_none());
    }

    #[test]
    fn file_provider_reference_accepts_alias() {
        let value = serde_json::json!({
            "type": "file",
            "provider_reference": {
                "openai": "file-openai"
            },
            "mediaType": "application/pdf",
            "filename": "doc.pdf"
        });

        let part = serde_json::from_value::<ContentPart>(value).expect("deserialize file");
        let ContentPart::File {
            source,
            media_type,
            filename,
            ..
        } = part
        else {
            panic!("expected file part");
        };

        assert_eq!(media_type, "application/pdf");
        assert_eq!(filename.as_deref(), Some("doc.pdf"));
        assert_eq!(
            source
                .as_provider_reference()
                .and_then(|provider_reference| provider_reference.get("openai")),
            Some("file-openai")
        );
    }
}
