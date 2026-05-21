//! OpenAI-compatible chat response-side legacy content compatibility constructors.
//!
//! The compatibility chat endpoint still materializes `ChatResponse.content` with the stable
//! legacy carrier. Keep response-owned metadata and empty request option defaults here so the broad
//! transformer does not scatter `ContentPart` construction details.

use crate::types::{
    ContentPart, MessageContent, ProviderMetadataMap, ProviderOptionsMap, SourcePart,
};

pub(super) fn text(text: impl Into<String>) -> ContentPart {
    ContentPart::Text {
        text: text.into(),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: None,
    }
}

pub(super) fn tool_call(
    tool_call_id: impl Into<String>,
    tool_name: impl Into<String>,
    arguments: serde_json::Value,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::ToolCall {
        tool_call_id: tool_call_id.into(),
        tool_name: tool_name.into(),
        arguments,
        provider_executed: None,
        dynamic: None,
        invalid: None,
        error: None,
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

pub(super) fn reasoning(text: impl Into<String>) -> ContentPart {
    ContentPart::Reasoning {
        text: text.into(),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: None,
    }
}

pub(super) fn parts_from_message_content(content: MessageContent) -> Vec<ContentPart> {
    match content {
        MessageContent::Text(value) if !value.is_empty() => vec![text(value)],
        MessageContent::MultiModal(parts) => parts,
        _ => Vec::new(),
    }
}

pub(super) fn message_content_from_parts(parts: Vec<ContentPart>) -> MessageContent {
    if parts.len() == 1 && parts[0].is_text() {
        MessageContent::Text(parts[0].as_text().unwrap_or_default().to_string())
    } else if !parts.is_empty() {
        MessageContent::MultiModal(parts)
    } else {
        MessageContent::Text(String::new())
    }
}
