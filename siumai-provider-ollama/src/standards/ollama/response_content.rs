//! Ollama response-side legacy content compatibility constructors.
//!
//! Ollama response conversion still materializes `ChatMessage.content` through the stable
//! compatibility carrier. Keep those legacy defaults in one response adapter.

use crate::types::{ContentPart, FilePartSource, MessageContent, ProviderOptionsMap};

pub(super) fn text(text: impl Into<String>) -> ContentPart {
    ContentPart::Text {
        text: text.into(),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: None,
    }
}

pub(super) fn image_base64(data: impl Into<String>) -> ContentPart {
    ContentPart::Image {
        source: FilePartSource::base64(data.into()),
        media_type: None,
        detail: None,
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: None,
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

pub(super) fn reasoning(text: impl Into<String>) -> ContentPart {
    ContentPart::Reasoning {
        text: text.into(),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: None,
    }
}

pub(super) fn message_content_from_parts(
    parts: Vec<ContentPart>,
    text_fallback: String,
) -> MessageContent {
    if parts.is_empty() {
        MessageContent::Text(String::new())
    } else if parts.len() == 1 && parts[0].is_text() {
        MessageContent::Text(text_fallback)
    } else {
        MessageContent::MultiModal(parts)
    }
}
