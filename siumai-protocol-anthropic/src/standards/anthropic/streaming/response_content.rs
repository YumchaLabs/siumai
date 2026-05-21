//! Anthropic streaming response-side legacy content compatibility constructors.
//!
//! Streaming still has to finish with `ChatResponse.content` in the stable compatibility carrier.
//! Keep those legacy defaults in this parser-local adapter instead of scattering them through the
//! event state machine.

use crate::types::{ContentPart, MessageContent, ProviderOptionsMap};

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

pub(super) fn message_content_from_parts(
    parts: Vec<ContentPart>,
    text_buffer: String,
) -> MessageContent {
    if parts.len() == 1 && parts[0].is_text() {
        MessageContent::Text(text_buffer)
    } else if !parts.is_empty() {
        MessageContent::MultiModal(parts)
    } else if !text_buffer.is_empty() {
        MessageContent::Text(text_buffer)
    } else {
        MessageContent::Text(String::new())
    }
}
