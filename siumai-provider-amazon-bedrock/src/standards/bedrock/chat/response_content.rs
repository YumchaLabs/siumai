//! Bedrock response-side legacy content compatibility constructors.
//!
//! `ChatResponse.content` still uses the stable legacy `ContentPart` compatibility carrier.
//! This Bedrock-owned response adapter keeps request-side `provider_options` empty while
//! preserving Bedrock response metadata on content parts.

use crate::types::{ContentPart, MessageContent, ProviderMetadataMap, ProviderOptionsMap};

pub(super) fn text(text: impl Into<String>) -> ContentPart {
    ContentPart::text(text.into())
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
) -> ContentPart {
    ContentPart::tool_call(tool_call_id, tool_name, arguments, None)
}

pub(super) fn message_content_from_parts(parts: Vec<ContentPart>) -> MessageContent {
    MessageContent::MultiModal(parts)
}
