use crate::compat::content::ContentPart;
use crate::streaming::processor::{AccumulatedStreamRecord, StreamProcessor, ToolCallBuilder};
use crate::types::{
    ChatResponse, ChatStreamFileData, FinishReason, MessageContent, ProviderMetadataMap,
    ResponseMetadata, ToolExecutionOwner, merge_provider_metadata, provider_metadata_from_object,
    provider_metadata_without_private_diagnostics,
};
use std::collections::HashMap;

fn public_provider_metadata(metadata: Option<ProviderMetadataMap>) -> Option<ProviderMetadataMap> {
    metadata
        .as_ref()
        .map(provider_metadata_without_private_diagnostics)
        .filter(|metadata| !metadata.is_empty())
}

impl StreamProcessor {
    /// Build the final response
    pub fn build_final_response(&self) -> ChatResponse {
        self.build_final_response_with_finish_reason(None)
    }

    /// Build the final response with finish reason
    pub fn build_final_response_with_finish_reason(
        &self,
        finish_reason: Option<FinishReason>,
    ) -> ChatResponse {
        let terminal_response = self.terminal_response.as_ref();
        let accumulated = self.accumulated_stream_record();
        let mut stream_metadata = HashMap::new();

        if !accumulated.reasoning.is_empty() {
            stream_metadata.insert(
                "thinking".to_string(),
                serde_json::Value::String(accumulated.reasoning.clone()),
            );
        }

        let content = build_final_content(&accumulated, terminal_response);

        // Convert to nested provider_metadata structure
        let mut provider_metadata = self.final_provider_metadata.clone().unwrap_or_default();
        if !stream_metadata.is_empty() {
            merge_provider_metadata(
                &mut provider_metadata,
                provider_metadata_from_object("stream", stream_metadata),
            );
        }
        let provider_metadata = if provider_metadata.is_empty() {
            None
        } else {
            Some(provider_metadata)
        };

        ChatResponse {
            id: terminal_response
                .and_then(|response| response.id.clone())
                .or_else(|| {
                    self.start_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.id.clone())
                }),
            content,
            model: terminal_response
                .and_then(|response| response.model.clone())
                .or_else(|| {
                    self.start_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.model.clone())
                }),
            usage: self
                .usage_ledger
                .latest_or_else(terminal_response.and_then(|response| response.usage.as_ref())),
            finish_reason: finish_reason
                .or_else(|| self.stream_finish_reason.clone())
                .or_else(|| terminal_response.and_then(|response| response.finish_reason.clone())),
            raw_finish_reason: terminal_response
                .and_then(|response| response.raw_finish_reason.clone())
                .or_else(|| self.stream_raw_finish_reason.clone()),
            audio: terminal_response.and_then(|response| response.audio.clone()),
            system_fingerprint: terminal_response
                .and_then(|response| response.system_fingerprint.clone()),
            service_tier: terminal_response.and_then(|response| response.service_tier.clone()),
            warnings: terminal_response
                .and_then(|response| response.warnings.clone())
                .or_else(|| {
                    (!self.stream_warnings.is_empty()).then(|| self.stream_warnings.clone())
                }),
            request: terminal_response.and_then(|response| response.request.clone()),
            response: final_http_response_info(
                terminal_response.and_then(|response| response.response.clone()),
                self.start_metadata.as_ref(),
            ),
            provider_metadata,
        }
    }
}

fn build_final_content(
    accumulated: &AccumulatedStreamRecord,
    terminal_response: Option<&ChatResponse>,
) -> MessageContent {
    if !accumulated.has_accumulated_content() {
        return terminal_response
            .map(|response| response.content.clone())
            .unwrap_or_else(|| MessageContent::Text(String::new()));
    }

    #[cfg(feature = "structured-messages")]
    if matches!(
        terminal_response.map(|response| &response.content),
        Some(MessageContent::Json(_))
    ) {
        return terminal_response
            .map(|response| response.content.clone())
            .unwrap_or_else(|| MessageContent::Text(String::new()));
    }

    let mut parts = if !accumulated.text.is_empty() {
        vec![build_text_part(&accumulated.text, terminal_response)]
    } else {
        terminal_response
            .map(|response| extract_terminal_text_parts(&response.content))
            .unwrap_or_default()
    };

    if accumulated.has_tool_call_builders() {
        parts.extend(build_accumulated_tool_call_parts(
            accumulated,
            terminal_response,
        ));
    } else if let Some(response) = terminal_response {
        parts.extend(extract_terminal_tool_call_parts(&response.content));
    }

    if !accumulated.reasoning.is_empty() {
        parts.push(build_reasoning_part(
            &accumulated.reasoning,
            terminal_response,
        ));
    } else if let Some(response) = terminal_response {
        parts.extend(extract_terminal_reasoning_parts(&response.content));
    }

    if let Some(response) = terminal_response {
        parts.extend(extract_terminal_extra_parts(&response.content));
    }

    parts.extend(extract_stream_tool_call_parts(&accumulated.stream_parts));
    parts.extend(extract_stream_reasoning_extra_parts(
        &accumulated.stream_parts,
    ));
    parts.extend(extract_stream_extra_parts(&accumulated.stream_parts));

    message_content_from_parts(parts)
}

fn build_accumulated_tool_call_parts(
    accumulated: &AccumulatedStreamRecord,
    terminal_response: Option<&ChatResponse>,
) -> Vec<ContentPart> {
    let mut parts = Vec::new();

    for (tool_index, builder) in accumulated.tool_calls.iter().enumerate() {
        if accumulated.stream_parts.iter().any(
            |part| matches!(part, ContentPart::ToolCall { tool_call_id, .. } if tool_call_id == &builder.id),
        ) {
            continue;
        }

        if builder.name.is_empty() {
            continue;
        }

        let arguments = serde_json::from_str(&builder.arguments)
            .unwrap_or_else(|_| serde_json::Value::String(builder.arguments.clone()));

        let terminal_match = terminal_response.and_then(|response| {
            find_terminal_tool_call_part(&response.content, builder, tool_index)
        });

        parts.push(build_tool_call_part(builder, arguments, terminal_match));
    }

    parts
}

pub(super) fn stream_file_part_to_content_part(
    file: &crate::types::ChatStreamFilePart,
    reasoning: bool,
) -> ContentPart {
    let source = match &file.data {
        ChatStreamFileData::Base64(data) => crate::types::MediaSource::base64(data.clone()),
        ChatStreamFileData::Bytes(data) => crate::types::MediaSource::binary(data.clone()),
    };

    if reasoning {
        ContentPart::ReasoningFile {
            source,
            media_type: file.media_type.clone(),
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: public_provider_metadata(file.provider_metadata.clone()),
        }
    } else {
        ContentPart::File {
            source: crate::types::FilePartSource::from(source),
            media_type: file.media_type.clone(),
            filename: None,
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: public_provider_metadata(file.provider_metadata.clone()),
        }
    }
}

fn response_text_part(
    text: impl Into<String>,
    provider_metadata: Option<ProviderMetadataMap>,
) -> ContentPart {
    ContentPart::Text {
        text: text.into(),
        provider_options: crate::types::ProviderOptionsMap::default(),
        provider_metadata: public_provider_metadata(provider_metadata),
    }
}

fn build_text_part(text: &str, terminal_response: Option<&ChatResponse>) -> ContentPart {
    let provider_metadata =
        terminal_response.and_then(|response| first_terminal_text_metadata(&response.content));

    response_text_part(text, provider_metadata)
}

fn build_reasoning_part(text: &str, terminal_response: Option<&ChatResponse>) -> ContentPart {
    let provider_metadata =
        terminal_response.and_then(|response| first_terminal_reasoning_metadata(&response.content));

    ContentPart::Reasoning {
        text: text.to_string(),
        provider_options: crate::types::ProviderOptionsMap::default(),
        provider_metadata: public_provider_metadata(provider_metadata),
    }
}

fn build_tool_call_part(
    builder: &ToolCallBuilder,
    arguments: serde_json::Value,
    terminal_part: Option<&ContentPart>,
) -> ContentPart {
    let (provider_executed, dynamic, title, provider_metadata) = match terminal_part {
        Some(ContentPart::ToolCall {
            provider_executed,
            dynamic,
            title,
            provider_metadata,
            ..
        }) => (
            *provider_executed,
            *dynamic,
            title.clone(),
            provider_metadata.clone(),
        ),
        _ => (None, None, None, None),
    };

    ContentPart::ToolCall {
        tool_call_id: builder.id.clone(),
        tool_name: builder.name.clone(),
        arguments,
        provider_executed: ToolExecutionOwner::merge_provider_executed_flags(
            builder.provider_executed,
            provider_executed,
        ),
        dynamic: builder.dynamic.or(dynamic),
        invalid: None,
        error: None,
        title: builder.title.clone().or(title),
        provider_options: crate::types::ProviderOptionsMap::default(),
        provider_metadata: public_provider_metadata(
            builder.provider_metadata.clone().or(provider_metadata),
        ),
    }
}

pub(super) fn tool_input_from_builder(builder: &ToolCallBuilder) -> serde_json::Value {
    serde_json::from_str(&builder.arguments)
        .unwrap_or_else(|_| serde_json::Value::String(builder.arguments.clone()))
}

fn find_terminal_tool_call_part<'a>(
    content: &'a MessageContent,
    builder: &ToolCallBuilder,
    tool_index: usize,
) -> Option<&'a ContentPart> {
    let terminal_parts = match content {
        MessageContent::MultiModal(parts) => parts,
        _ => return None,
    };

    terminal_parts
        .iter()
        .find(|part| matches!(part, ContentPart::ToolCall { tool_call_id, .. } if tool_call_id == &builder.id))
        .or_else(|| {
            terminal_parts
                .iter()
                .filter(|part| part.is_tool_call())
                .nth(tool_index)
        })
}

fn first_terminal_text_metadata(content: &MessageContent) -> Option<ProviderMetadataMap> {
    match content {
        MessageContent::MultiModal(parts) => parts.iter().find_map(|part| match part {
            ContentPart::Text {
                provider_metadata, ..
            } => provider_metadata.clone(),
            _ => None,
        }),
        _ => None,
    }
}

fn first_terminal_reasoning_metadata(content: &MessageContent) -> Option<ProviderMetadataMap> {
    match content {
        MessageContent::MultiModal(parts) => parts.iter().find_map(|part| match part {
            ContentPart::Reasoning {
                provider_metadata, ..
            } => provider_metadata.clone(),
            _ => None,
        }),
        _ => None,
    }
}

pub(super) fn extract_terminal_text_parts(content: &MessageContent) -> Vec<ContentPart> {
    match content {
        MessageContent::Text(text) if !text.is_empty() => {
            vec![response_text_part(text.as_str(), None)]
        }
        MessageContent::MultiModal(parts) => parts
            .iter()
            .filter(|part| part.is_text())
            .cloned()
            .collect(),
        _ => Vec::new(),
    }
}

fn extract_terminal_tool_call_parts(content: &MessageContent) -> Vec<ContentPart> {
    match content {
        MessageContent::MultiModal(parts) => parts
            .iter()
            .filter(|part| part.is_tool_call())
            .cloned()
            .collect(),
        _ => Vec::new(),
    }
}

fn extract_terminal_reasoning_parts(content: &MessageContent) -> Vec<ContentPart> {
    match content {
        MessageContent::MultiModal(parts) => parts
            .iter()
            .filter(|part| part.is_reasoning())
            .cloned()
            .collect(),
        _ => Vec::new(),
    }
}

fn extract_terminal_extra_parts(content: &MessageContent) -> Vec<ContentPart> {
    match content {
        MessageContent::MultiModal(parts) => parts
            .iter()
            .filter(|part| !part.is_text() && !part.is_tool_call() && !part.is_reasoning())
            .cloned()
            .collect(),
        _ => Vec::new(),
    }
}

fn extract_stream_tool_call_parts(parts: &[ContentPart]) -> Vec<ContentPart> {
    parts
        .iter()
        .filter(|part| matches!(part, ContentPart::ToolCall { .. }))
        .cloned()
        .collect()
}

fn extract_stream_reasoning_extra_parts(parts: &[ContentPart]) -> Vec<ContentPart> {
    parts
        .iter()
        .filter(|part| matches!(part, ContentPart::ReasoningFile { .. }))
        .cloned()
        .collect()
}

fn extract_stream_extra_parts(parts: &[ContentPart]) -> Vec<ContentPart> {
    parts
        .iter()
        .filter(|part| !part.is_text() && !part.is_reasoning() && !part.is_tool_call())
        .cloned()
        .collect()
}

fn message_content_from_parts(parts: Vec<ContentPart>) -> MessageContent {
    match parts.as_slice() {
        [
            ContentPart::Text {
                text,
                provider_options,
                provider_metadata: None,
            },
        ] if provider_options.is_empty() => MessageContent::Text(text.clone()),
        [] => MessageContent::Text(String::new()),
        _ => MessageContent::MultiModal(parts),
    }
}

fn final_http_response_info(
    terminal: Option<crate::types::HttpResponseInfo>,
    start_metadata: Option<&ResponseMetadata>,
) -> Option<crate::types::HttpResponseInfo> {
    let Some(metadata) = start_metadata else {
        return terminal;
    };

    let headers = metadata
        .headers
        .as_ref()
        .filter(|headers| !headers.is_empty());
    if headers.is_none() && metadata.body.is_none() {
        return terminal;
    }

    match terminal {
        Some(mut response) => {
            if response.headers.is_empty()
                && let Some(headers) = headers
            {
                response.headers = headers.clone();
            }
            if response.body.is_none() {
                response.body = metadata.body.clone();
            }
            Some(response)
        }
        None => Some(crate::types::HttpResponseInfo {
            timestamp: metadata.created.unwrap_or_else(chrono::Utc::now),
            model_id: metadata.model.clone(),
            headers: headers.cloned().unwrap_or_default(),
            body: metadata.body.clone(),
        }),
    }
}
