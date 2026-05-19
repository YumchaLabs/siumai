//! Google Interactions response-side legacy content compatibility constructors.
//!
//! `ChatResponse.content` still uses the stable legacy `ContentPart` compatibility carrier.
//! This provider-owned parser adapter keeps request-side `provider_options` empty while preserving
//! Google Interactions response metadata on content parts.

use crate::types::{
    ContentPart, FilePartSource, MessageContent, ProviderMetadataMap, ProviderOptionsMap,
    SourcePart, ToolResultOutput,
};

pub(super) fn text(
    text: impl Into<String>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::Text {
        text: text.into(),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata,
    }
}

pub(super) fn file_base64(
    data: impl Into<String>,
    media_type: impl Into<String>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::File {
        source: FilePartSource::base64(data.into()),
        media_type: media_type.into(),
        filename: None,
        provider_options: ProviderOptionsMap::default(),
        provider_metadata,
    }
}

pub(super) fn file_url(
    url: impl Into<String>,
    media_type: impl Into<String>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::File {
        source: FilePartSource::url(url.into()),
        media_type: media_type.into(),
        filename: None,
        provider_options: ProviderOptionsMap::default(),
        provider_metadata,
    }
}

pub(super) fn reasoning(
    text: impl Into<String>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::Reasoning {
        text: text.into(),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata,
    }
}

pub(super) fn tool_call(
    tool_call_id: impl Into<String>,
    tool_name: impl Into<String>,
    arguments: serde_json::Value,
    provider_executed: Option<bool>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::ToolCall {
        tool_call_id: tool_call_id.into(),
        tool_name: tool_name.into(),
        arguments,
        provider_executed,
        dynamic: None,
        invalid: None,
        error: None,
        title: None,
        provider_options: ProviderOptionsMap::default(),
        provider_metadata,
    }
}

pub(super) fn tool_result(
    tool_call_id: impl Into<String>,
    tool_name: impl Into<String>,
    output: ToolResultOutput,
    input: Option<serde_json::Value>,
    provider_executed: Option<bool>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::ToolResult {
        tool_call_id: tool_call_id.into(),
        tool_name: tool_name.into(),
        output,
        input,
        provider_executed,
        dynamic: None,
        preliminary: None,
        title: None,
        provider_options: ProviderOptionsMap::default(),
        provider_metadata,
    }
}

pub(super) fn source_url(
    id: impl Into<String>,
    url: impl Into<String>,
    title: Option<String>,
) -> ContentPart {
    ContentPart::Source {
        id: id.into(),
        source: SourcePart::Url {
            url: url.into(),
            title,
        },
        provider_metadata: None,
    }
}

pub(super) fn source_document(
    id: impl Into<String>,
    media_type: impl Into<String>,
    title: impl Into<String>,
    filename: Option<String>,
) -> ContentPart {
    ContentPart::Source {
        id: id.into(),
        source: SourcePart::Document {
            media_type: media_type.into(),
            title: title.into(),
            filename,
        },
        provider_metadata: None,
    }
}

pub(super) fn message_content_from_parts(parts: Vec<ContentPart>) -> MessageContent {
    MessageContent::MultiModal(parts)
}
