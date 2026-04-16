use crate::core::ProviderContext;
use crate::error::LlmError;
use crate::types::{
    ChatResponse, ContentPart, FilePartSource, FinishReason, MediaSource, MessageContent,
    ProviderOptionsMap, ResponseMetadata, Usage,
};
use base64::{Engine, engine::general_purpose::STANDARD};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::de::DeserializeOwned;
use std::collections::HashMap;

pub fn build_headers(ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
    let api_key = ctx
        .api_key
        .as_deref()
        .ok_or_else(|| LlmError::ConfigurationError("Cohere API key is required".to_string()))?;

    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|error| {
            LlmError::ConfigurationError(format!("Invalid Cohere API key: {error}"))
        })?,
    );
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    for (key, value) in &ctx.http_extra_headers {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(key.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            headers.insert(name, value);
        }
    }

    Ok(headers)
}

pub fn cohere_provider_options<T: DeserializeOwned>(
    provider_options_map: &ProviderOptionsMap,
) -> Result<Option<T>, LlmError> {
    match provider_options_map.get("cohere") {
        Some(value) => serde_json::from_value(value.clone())
            .map(Some)
            .map_err(|error| {
                LlmError::InvalidParameter(format!("Invalid Cohere provider options: {error}"))
            }),
        None => Ok(None),
    }
}

pub fn map_finish_reason(raw: Option<&str>) -> FinishReason {
    match raw {
        Some("COMPLETE") | Some("STOP_SEQUENCE") => FinishReason::Stop,
        Some("MAX_TOKENS") => FinishReason::Length,
        Some("ERROR") => FinishReason::Error,
        Some("TOOL_CALL") => FinishReason::ToolCalls,
        Some(other) if !other.trim().is_empty() => FinishReason::Other(other.to_string()),
        _ => FinishReason::Unknown,
    }
}

pub fn build_usage(
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    raw: Option<serde_json::Value>,
) -> Usage {
    let mut builder = Usage::builder();

    if let Some(input_tokens) = input_tokens {
        builder = builder
            .with_input_total_tokens(input_tokens)
            .with_input_no_cache_tokens(input_tokens)
            .prompt_tokens(input_tokens);
    }

    if let Some(output_tokens) = output_tokens {
        builder = builder
            .with_output_total_tokens(output_tokens)
            .with_output_text_tokens(output_tokens)
            .completion_tokens(output_tokens);
    }

    if let Some(raw) = raw {
        builder = builder.with_raw_usage_value(raw);
    }

    builder.build()
}

pub fn response_metadata(
    provider_id: &str,
    id: Option<String>,
    model: Option<String>,
) -> ResponseMetadata {
    ResponseMetadata {
        id,
        model,
        created: None,
        provider: provider_id.to_string(),
        request_id: None,
    }
}

pub fn message_content_from_parts(parts: Vec<ContentPart>) -> MessageContent {
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

pub fn build_stream_end_response(
    id: Option<String>,
    model: Option<String>,
    usage: Option<Usage>,
    finish_reason: FinishReason,
) -> ChatResponse {
    let mut response = ChatResponse::empty_with_finish_reason(finish_reason);
    response.id = id;
    response.model = model;
    response.usage = usage;
    response
}

pub fn provider_metadata_entry(
    provider_id: &str,
    value: serde_json::Value,
) -> Option<HashMap<String, serde_json::Value>> {
    if value.is_null() {
        None
    } else {
        Some(HashMap::from([(provider_id.to_string(), value)]))
    }
}

pub fn text_document_media_type_supported(media_type: &str) -> bool {
    media_type.starts_with("text/") || media_type.eq_ignore_ascii_case("application/json")
}

pub fn decode_text_media(source: &FilePartSource, media_type: &str) -> Result<String, LlmError> {
    if !text_document_media_type_supported(media_type) {
        return Err(LlmError::UnsupportedOperation(format!(
            "Cohere only supports file parts with text/* or application/json media types, got {media_type}"
        )));
    }

    match source {
        FilePartSource::ProviderReference { .. } => Err(LlmError::UnsupportedOperation(
            "Cohere provider-managed file references are not supported; send inline text content instead".to_string(),
        )),
        FilePartSource::Media(MediaSource::Url { .. }) => Err(LlmError::UnsupportedOperation(
            "Cohere file URL parts are not supported; download the file and send inline text content instead".to_string(),
        )),
        FilePartSource::Media(MediaSource::Base64 { data }) => {
            let bytes = STANDARD.decode(data).map_err(|error| {
                LlmError::InvalidInput(format!("Failed to decode Cohere file part as base64: {error}"))
            })?;
            String::from_utf8(bytes).map_err(|error| {
                LlmError::InvalidInput(format!("Cohere file part is not valid UTF-8 text: {error}"))
            })
        }
        FilePartSource::Media(MediaSource::Binary { data }) => {
            String::from_utf8(data.clone()).map_err(|error| {
            LlmError::InvalidInput(format!("Cohere file part is not valid UTF-8 text: {error}"))
        })
        }
    }
}
