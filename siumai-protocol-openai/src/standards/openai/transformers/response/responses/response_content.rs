//! OpenAI Responses response-side legacy content compatibility constructors.
//!
//! `ChatResponse.content` still uses the stable legacy `ContentPart` compatibility carrier.
//! This module is the parser-local response adapter for that shape: it preserves response
//! `provider_metadata`, keeps request-side `provider_options` empty, and prevents the broad
//! response parser from owning the legacy field defaults directly.

use crate::types::{
    ContentPart, MessageContent, ProviderMetadataMap, ProviderOptionsMap, SourcePart,
    ToolResultOutput,
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

pub(super) fn custom(
    kind: impl Into<String>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::Custom {
        kind: kind.into(),
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
) -> ContentPart {
    tool_call_with_metadata(tool_call_id, tool_name, arguments, provider_executed, None)
}

pub(super) fn tool_call_with_metadata(
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
    provider_metadata: Option<ProviderMetadataMap>,
    output: ToolResultOutput,
) -> ContentPart {
    tool_result_with_provider_executed(tool_call_id, tool_name, provider_metadata, output, None)
}

pub(super) fn tool_result_with_provider_executed(
    tool_call_id: impl Into<String>,
    tool_name: impl Into<String>,
    provider_metadata: Option<ProviderMetadataMap>,
    output: ToolResultOutput,
    provider_executed: Option<bool>,
) -> ContentPart {
    ContentPart::ToolResult {
        tool_call_id: tool_call_id.into(),
        tool_name: tool_name.into(),
        output,
        input: None,
        provider_executed,
        dynamic: None,
        preliminary: None,
        title: None,
        provider_options: ProviderOptionsMap::default(),
        provider_metadata,
    }
}

pub(super) fn tool_approval_request(
    approval_id: impl Into<String>,
    tool_call_id: impl Into<String>,
) -> ContentPart {
    ContentPart::ToolApprovalRequest {
        approval_id: approval_id.into(),
        tool_call_id: tool_call_id.into(),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata: None,
    }
}

pub(super) fn source_url(
    id: impl Into<String>,
    url: impl Into<String>,
    title: impl Into<String>,
) -> ContentPart {
    ContentPart::Source {
        id: id.into(),
        source: SourcePart::Url {
            url: url.into(),
            title: Some(title.into()),
        },
        provider_metadata: None,
    }
}

pub(super) fn message_content_from_parts(
    content_parts: Vec<ContentPart>,
    plain_text_fallback: String,
) -> MessageContent {
    if content_parts.is_empty() {
        return MessageContent::Text(String::new());
    }

    if content_parts.len() == 1
        && let ContentPart::Text {
            provider_options,
            provider_metadata,
            ..
        } = &content_parts[0]
        && provider_options.is_empty()
        && provider_metadata.is_none()
    {
        return MessageContent::Text(plain_text_fallback);
    }

    MessageContent::MultiModal(content_parts)
}
