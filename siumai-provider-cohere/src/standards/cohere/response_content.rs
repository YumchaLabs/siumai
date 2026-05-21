//! Cohere response-side legacy content compatibility constructors.
//!
//! `ChatResponse.content` remains the stable compatibility carrier. Keep response parsing metadata
//! and empty request option defaults local to this adapter instead of the broader chat transformer.

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

pub(super) fn reasoning(text: impl Into<String>) -> ContentPart {
    ContentPart::Reasoning {
        text: text.into(),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: None,
    }
}

pub(super) fn source_document(
    id: impl Into<String>,
    media_type: impl Into<String>,
    title: impl Into<String>,
    filename: Option<String>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::Source {
        id: id.into(),
        source: SourcePart::Document {
            media_type: media_type.into(),
            title: title.into(),
            filename,
        },
        provider_metadata,
    }
}

pub(super) fn tool_call(
    tool_call_id: impl Into<String>,
    tool_name: impl Into<String>,
    arguments: serde_json::Value,
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
        provider_metadata: None,
    }
}

pub(super) fn message_content_from_parts(parts: Vec<ContentPart>) -> MessageContent {
    if parts.is_empty() {
        MessageContent::Text(String::new())
    } else if parts.len() == 1 {
        match &parts[0] {
            ContentPart::Text { text, .. } => MessageContent::Text(text.clone()),
            _ => MessageContent::MultiModal(parts),
        }
    } else {
        MessageContent::MultiModal(parts)
    }
}
