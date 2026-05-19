use crate::types::{
    ContentPart, FilePartSource, MediaSource, MessageContent, ProviderMetadataMap,
    ProviderOptionsMap, ToolResultOutput,
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

pub(super) fn reasoning_file_base64(
    data: impl Into<String>,
    media_type: impl Into<String>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::ReasoningFile {
        source: MediaSource::Base64 { data: data.into() },
        media_type: media_type.into(),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata,
    }
}

pub(super) fn reasoning_file_url(
    url: impl Into<String>,
    media_type: impl Into<String>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::ReasoningFile {
        source: MediaSource::Url { url: url.into() },
        media_type: media_type.into(),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata,
    }
}

pub(super) fn image_base64(
    data: impl Into<String>,
    media_type: impl Into<String>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::Image {
        source: FilePartSource::base64(data.into()),
        media_type: Some(media_type.into()),
        detail: None,
        provider_options: ProviderOptionsMap::default(),
        provider_metadata,
    }
}

pub(super) fn image_url(
    url: impl Into<String>,
    media_type: impl Into<String>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::Image {
        source: FilePartSource::url(url.into()),
        media_type: Some(media_type.into()),
        detail: None,
        provider_options: ProviderOptionsMap::default(),
        provider_metadata,
    }
}

pub(super) fn audio_base64(
    data: impl Into<String>,
    media_type: impl Into<String>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::Audio {
        source: MediaSource::Base64 { data: data.into() },
        media_type: Some(media_type.into()),
        provider_options: ProviderOptionsMap::default(),
        provider_metadata,
    }
}

pub(super) fn audio_url(
    url: impl Into<String>,
    media_type: impl Into<String>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::Audio {
        source: MediaSource::Url { url: url.into() },
        media_type: Some(media_type.into()),
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

pub(super) fn is_client_tool_call(part: &ContentPart) -> bool {
    matches!(
        part,
        ContentPart::ToolCall {
            provider_executed,
            ..
        } if provider_executed != &Some(true)
    )
}

pub(super) fn message_content_from_parts(
    content_parts: Vec<ContentPart>,
    text_content: String,
) -> MessageContent {
    if content_parts.is_empty() {
        return MessageContent::Text(text_content);
    }

    if content_parts.iter().all(ContentPart::is_text) {
        MessageContent::Text(text_content)
    } else {
        MessageContent::MultiModal(content_parts)
    }
}
