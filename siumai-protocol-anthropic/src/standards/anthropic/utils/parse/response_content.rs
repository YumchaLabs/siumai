use crate::types::{
    ContentPart, MessageContent, ProviderMetadataMap, ProviderOptionsMap, SourcePart,
    ToolResultContentPart, ToolResultOutput,
};
use std::collections::HashMap;

pub(super) fn anthropic_provider_metadata(value: serde_json::Value) -> Option<ProviderMetadataMap> {
    Some(HashMap::from([("anthropic".to_string(), value)]))
}

pub(super) fn text_part_provider_metadata(
    citations: Option<&Vec<serde_json::Value>>,
) -> Option<ProviderMetadataMap> {
    let citations = citations.filter(|citations| !citations.is_empty())?;
    anthropic_provider_metadata(serde_json::json!({
        "citations": citations
    }))
}

pub(super) fn reasoning_part_provider_metadata(
    signature: Option<&str>,
    redacted_data: Option<&str>,
) -> Option<ProviderMetadataMap> {
    let mut anthropic = serde_json::Map::new();

    if let Some(signature) = signature {
        anthropic.insert("signature".to_string(), serde_json::json!(signature));
    }
    if let Some(redacted_data) = redacted_data {
        anthropic.insert("redactedData".to_string(), serde_json::json!(redacted_data));
    }

    (!anthropic.is_empty())
        .then(|| serde_json::Value::Object(anthropic))
        .and_then(anthropic_provider_metadata)
}

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

pub(super) fn source_url(
    id: impl Into<String>,
    url: impl Into<String>,
    title: Option<String>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::Source {
        id: id.into(),
        source: SourcePart::Url {
            url: url.into(),
            title,
        },
        provider_metadata,
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
    provider_executed: Option<bool>,
    dynamic: Option<bool>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::ToolCall {
        tool_call_id: tool_call_id.into(),
        tool_name: tool_name.into(),
        arguments,
        provider_executed,
        dynamic,
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
    dynamic: Option<bool>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::ToolResult {
        tool_call_id: tool_call_id.into(),
        tool_name: tool_name.into(),
        output,
        input,
        provider_executed,
        dynamic,
        preliminary: None,
        title: None,
        provider_options: ProviderOptionsMap::default(),
        provider_metadata,
    }
}

pub(super) fn tool_result_text_part(text: &str) -> ToolResultContentPart {
    ToolResultContentPart::Text {
        text: text.to_string(),
        provider_options: ProviderOptionsMap::default(),
    }
}

pub(super) fn message_content_from_parts(parts: Vec<ContentPart>) -> MessageContent {
    if parts.is_empty() {
        return MessageContent::Text(String::new());
    }

    if let [
        ContentPart::Text {
            text,
            provider_options,
            provider_metadata: None,
        },
    ] = parts.as_slice()
        && provider_options.is_empty()
    {
        return MessageContent::Text(text.clone());
    }

    MessageContent::MultiModal(parts)
}
