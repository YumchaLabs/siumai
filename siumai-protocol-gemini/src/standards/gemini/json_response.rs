//!
//! English-only comments in code as requested.

use crate::encoding::{JsonEncodeOptions, JsonResponseConverter};
use crate::error::LlmError;
use crate::types::{
    ChatResponse, ContentPart as UnifiedContentPart, FinishReason, MessageContent,
    ProviderMetadataMap, SourcePart, ToolResultContentPart, ToolResultOutput, Usage,
};
use base64::Engine;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use super::types::{
    Blob, Candidate, CodeExecutionOutcome, CodeExecutionResult, CodeLanguage, Content,
    ExecutableCode, FileData, FinishReason as GeminiFinishReason, FunctionCall, FunctionResponse,
    GenerateContentResponse, GroundingMetadata, LogprobsResult, Part, PromptFeedback, SafetyRating,
    UrlContextMetadata, UsageMetadata,
};

fn gemini_finish_reason(reason: Option<&FinishReason>) -> Option<GeminiFinishReason> {
    match reason? {
        FinishReason::Stop | FinishReason::StopSequence => Some(GeminiFinishReason::Stop),
        FinishReason::Length => Some(GeminiFinishReason::MaxTokens),
        FinishReason::ContentFilter => Some(GeminiFinishReason::Safety),
        FinishReason::ToolCalls => Some(GeminiFinishReason::Stop),
        FinishReason::Error => Some(GeminiFinishReason::Stop),
        FinishReason::Unknown | FinishReason::Other(_) => None,
    }
}

fn raw_finish_reason(response: &ChatResponse) -> Option<&str> {
    response
        .raw_finish_reason
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn usage_json(usage: &Usage) -> UsageMetadata {
    let normalized_input = usage.normalized_input_tokens();
    let normalized_output = usage.normalized_output_tokens();
    let prompt_total = normalized_input
        .total
        .or_else(|| usage.prompt_tokens_value());
    let output_total = normalized_output
        .total
        .or_else(|| usage.completion_tokens_value());
    let total_tokens = usage.total_tokens_value().or_else(|| {
        prompt_total
            .zip(output_total)
            .map(|(prompt, completion)| prompt.saturating_add(completion))
    });
    let mut metadata = usage
        .raw_usage_value()
        .and_then(|value| serde_json::from_value::<UsageMetadata>(value).ok())
        .unwrap_or(UsageMetadata {
            prompt_token_count: None,
            total_token_count: None,
            cached_content_token_count: None,
            candidates_token_count: None,
            thoughts_token_count: None,
            traffic_type: None,
            prompt_tokens_details: None,
            candidates_tokens_details: None,
        });

    if let Some(prompt_total) = prompt_total {
        metadata.prompt_token_count = Some(i32::try_from(prompt_total).unwrap_or(i32::MAX));
    }
    if let Some(total_tokens) = total_tokens {
        metadata.total_token_count = Some(i32::try_from(total_tokens).unwrap_or(i32::MAX));
    }
    if let Some(cached_tokens) = normalized_input.cache_read {
        metadata.cached_content_token_count =
            Some(i32::try_from(cached_tokens).unwrap_or(i32::MAX));
    }
    if let Some(text_tokens) = normalized_output.text {
        metadata.candidates_token_count = Some(i32::try_from(text_tokens).unwrap_or(i32::MAX));
    }
    if let Some(reasoning_tokens) = normalized_output.reasoning {
        metadata.thoughts_token_count = Some(i32::try_from(reasoning_tokens).unwrap_or(i32::MAX));
    }

    metadata
}

fn google_response_metadata(response: &ChatResponse) -> Option<&serde_json::Map<String, Value>> {
    let provider_metadata = response.provider_metadata.as_ref()?;
    crate::types::provider_metadata::provider_metadata_object(provider_metadata, "google").or_else(
        || crate::types::provider_metadata::provider_metadata_object(provider_metadata, "vertex"),
    )
}

fn google_response_metadata_raw_value(response: &ChatResponse, key: &str) -> Option<Value> {
    google_response_metadata(response)?.get(key).cloned()
}

fn google_response_metadata_value<T: DeserializeOwned>(
    response: &ChatResponse,
    key: &str,
) -> Option<T> {
    let value = google_response_metadata_raw_value(response, key)?;
    serde_json::from_value(value).ok()
}

fn source_grounding_chunk(part: &UnifiedContentPart) -> Option<Value> {
    let UnifiedContentPart::Source { source, .. } = part else {
        return None;
    };

    match source {
        SourcePart::Url { url, title } => Some(json!({
            "web": {
                "uri": url,
                "title": title.clone().unwrap_or_else(|| url.clone()),
            }
        })),
        SourcePart::Document {
            media_type,
            title,
            filename,
        } => {
            let uri = filename.clone().unwrap_or_else(|| media_type.clone());
            Some(json!({
                "retrievedContext": {
                    "uri": uri,
                    "title": title,
                }
            }))
        }
    }
}

fn response_source_grounding_chunks(response: &ChatResponse) -> Vec<Value> {
    let Some(parts) = response.content.as_multimodal() else {
        return Vec::new();
    };

    let mut chunks = Vec::new();
    for part in parts {
        let Some(chunk) = source_grounding_chunk(part) else {
            continue;
        };
        if !chunks.iter().any(|existing| existing == &chunk) {
            chunks.push(chunk);
        }
    }
    chunks
}

fn merge_grounding_metadata(base: Option<Value>, extra_chunks: Vec<Value>) -> Option<Value> {
    if extra_chunks.is_empty() {
        return base;
    }

    match base {
        Some(mut value) => {
            if let Some(obj) = value.as_object_mut() {
                match obj.get_mut("groundingChunks").and_then(Value::as_array_mut) {
                    Some(chunks) => {
                        for chunk in extra_chunks {
                            if !chunks.iter().any(|existing| existing == &chunk) {
                                chunks.push(chunk);
                            }
                        }
                    }
                    None => {
                        obj.insert("groundingChunks".to_string(), Value::Array(extra_chunks));
                    }
                }
                Some(value)
            } else {
                Some(json!({
                    "groundingChunks": extra_chunks,
                }))
            }
        }
        None => Some(json!({
            "groundingChunks": extra_chunks,
        })),
    }
}

fn merged_grounding_metadata(response: &ChatResponse) -> Option<GroundingMetadata> {
    let merged = merge_grounding_metadata(
        google_response_metadata_raw_value(response, "groundingMetadata"),
        response_source_grounding_chunks(response),
    )?;
    serde_json::from_value(merged).ok()
}

fn response_content(response: &ChatResponse) -> Result<Content, LlmError> {
    match &response.content {
        MessageContent::Text(text) if text.trim().is_empty() => Ok(Content {
            role: Some("model".to_string()),
            parts: Vec::new(),
        }),
        MessageContent::MultiModal(parts) if parts.is_empty() => Ok(Content {
            role: Some("model".to_string()),
            parts: Vec::new(),
        }),
        MessageContent::Text(text) => Ok(Content {
            role: Some("model".to_string()),
            parts: vec![Part::Text {
                text: text.clone(),
                thought: None,
                thought_signature: None,
            }],
        }),
        MessageContent::MultiModal(content_parts) => {
            let mut parts = Vec::new();
            for part in content_parts {
                push_response_part(&mut parts, part)?;
            }
            if parts.is_empty() {
                return Err(LlmError::InvalidInput("Message has no content".to_string()));
            }
            Ok(Content {
                role: Some("model".to_string()),
                parts,
            })
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(value) => {
            let text = serde_json::to_string(value).unwrap_or_default();
            if text.trim().is_empty() {
                Ok(Content {
                    role: Some("model".to_string()),
                    parts: Vec::new(),
                })
            } else {
                Ok(Content {
                    role: Some("model".to_string()),
                    parts: vec![Part::Text {
                        text,
                        thought: None,
                        thought_signature: None,
                    }],
                })
            }
        }
    }
}

fn response_thought_signature(provider_metadata: Option<&ProviderMetadataMap>) -> Option<String> {
    let provider_metadata = provider_metadata?;

    for preferred in ["google", "vertex"] {
        if let Some(sig) = provider_metadata
            .get(preferred)
            .and_then(|value| value.get("thoughtSignature"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|sig| !sig.is_empty())
        {
            return Some(sig.to_string());
        }
    }

    provider_metadata.values().find_map(|value| {
        value
            .get("thoughtSignature")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|sig| !sig.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn parse_data_url(data_url: &str) -> Option<(String, String)> {
    let (header, data) = data_url.strip_prefix("data:")?.split_once(',')?;
    let mime_type = header
        .split_once(';')
        .map(|(mime_type, _)| mime_type)
        .unwrap_or(header);
    Some((mime_type.to_string(), data.to_string()))
}

fn guess_mime_type(url: &str) -> String {
    crate::utils::guess_mime_from_path_or_url(url)
        .unwrap_or_else(|| "application/octet-stream".to_string())
}

fn push_media_response_part(
    parts: &mut Vec<Part>,
    source: &crate::types::chat::MediaSource,
    media_type: Option<&str>,
    thought: Option<bool>,
    thought_signature: Option<String>,
) {
    match source {
        crate::types::chat::MediaSource::Url { url } if url.starts_with("data:") => {
            if let Some((mime_type, data)) = parse_data_url(url) {
                parts.push(Part::InlineData {
                    inline_data: Blob { mime_type, data },
                    thought,
                    thought_signature,
                });
            }
        }
        crate::types::chat::MediaSource::Url { url } => {
            parts.push(Part::FileData {
                file_data: FileData {
                    file_uri: url.clone(),
                    mime_type: Some(
                        media_type
                            .map(ToOwned::to_owned)
                            .unwrap_or_else(|| guess_mime_type(url)),
                    ),
                },
                thought,
                thought_signature,
            });
        }
        crate::types::chat::MediaSource::Base64 { data } => {
            parts.push(Part::InlineData {
                inline_data: Blob {
                    mime_type: media_type.unwrap_or("application/octet-stream").to_string(),
                    data: data.clone(),
                },
                thought,
                thought_signature,
            });
        }
        crate::types::chat::MediaSource::Binary { data } => {
            parts.push(Part::InlineData {
                inline_data: Blob {
                    mime_type: media_type.unwrap_or("application/octet-stream").to_string(),
                    data: base64::engine::general_purpose::STANDARD.encode(data),
                },
                thought,
                thought_signature,
            });
        }
    }
}

fn json_value_to_option_string(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        other => Some(other.to_string()),
    }
}

fn parse_code_execution_outcome(value: Option<&str>) -> CodeExecutionOutcome {
    match value {
        Some("OUTCOME_OK") => CodeExecutionOutcome::Ok,
        Some("OUTCOME_FAILED") => CodeExecutionOutcome::Failed,
        Some("OUTCOME_DEADLINE_EXCEEDED") => CodeExecutionOutcome::DeadlineExceeded,
        _ => CodeExecutionOutcome::Unspecified,
    }
}

fn code_execution_result_from_output(output: &ToolResultOutput) -> CodeExecutionResult {
    match output {
        ToolResultOutput::Json { value, .. } | ToolResultOutput::ErrorJson { value, .. } => {
            let obj = value.as_object();
            CodeExecutionResult {
                outcome: parse_code_execution_outcome(
                    obj.and_then(|inner| inner.get("outcome"))
                        .and_then(Value::as_str),
                ),
                output: obj
                    .and_then(|inner| inner.get("output"))
                    .and_then(json_value_to_option_string),
            }
        }
        ToolResultOutput::Text { value, .. } | ToolResultOutput::ErrorText { value, .. } => {
            CodeExecutionResult {
                outcome: CodeExecutionOutcome::Unspecified,
                output: Some(value.clone()),
            }
        }
        ToolResultOutput::Content { value, .. } => CodeExecutionResult {
            outcome: CodeExecutionOutcome::Unspecified,
            output: Some(format!("Multimodal content with {} parts", value.len())),
        },
        ToolResultOutput::ExecutionDenied { reason, .. } => CodeExecutionResult {
            outcome: CodeExecutionOutcome::Failed,
            output: reason.clone(),
        },
    }
}

fn push_function_response_part(
    parts: &mut Vec<Part>,
    tool_name: &str,
    content: Value,
    thought_signature: Option<String>,
) {
    parts.push(Part::FunctionResponse {
        function_response: FunctionResponse {
            name: tool_name.to_string(),
            response: json!({
                "name": tool_name,
                "content": content
            }),
        },
        thought_signature,
    });
}

fn push_tool_result_content_parts(
    parts: &mut Vec<Part>,
    tool_name: &str,
    value: &[ToolResultContentPart],
    thought_signature: Option<String>,
) {
    for content_part in value {
        match content_part {
            ToolResultContentPart::Text { text, .. } => {
                push_function_response_part(
                    parts,
                    tool_name,
                    Value::String(text.clone()),
                    thought_signature.clone(),
                );
            }
            ToolResultContentPart::ImageData {
                data, media_type, ..
            }
            | ToolResultContentPart::FileData {
                data, media_type, ..
            } => {
                parts.push(Part::InlineData {
                    inline_data: Blob {
                        mime_type: media_type.clone(),
                        data: data.clone(),
                    },
                    thought: None,
                    thought_signature: thought_signature.clone(),
                });
                parts.push(Part::Text {
                    text: "Tool executed successfully and returned this file as a response"
                        .to_string(),
                    thought: None,
                    thought_signature: thought_signature.clone(),
                });
            }
            content_part => {
                let text = serde_json::to_string(content_part).unwrap_or_default();
                if !text.is_empty() {
                    parts.push(Part::Text {
                        text,
                        thought: None,
                        thought_signature: thought_signature.clone(),
                    });
                }
            }
        }
    }
}

fn push_response_tool_result_part(
    parts: &mut Vec<Part>,
    tool_name: &str,
    output: &ToolResultOutput,
    thought_signature: Option<String>,
) {
    if tool_name == "code_execution" {
        parts.push(Part::CodeExecutionResult {
            code_execution_result: code_execution_result_from_output(output),
            thought_signature,
        });
        return;
    }

    match output {
        ToolResultOutput::Text { value, .. } | ToolResultOutput::ErrorText { value, .. } => {
            push_function_response_part(
                parts,
                tool_name,
                Value::String(value.clone()),
                thought_signature,
            );
        }
        ToolResultOutput::Json { value, .. } | ToolResultOutput::ErrorJson { value, .. } => {
            push_function_response_part(parts, tool_name, value.clone(), thought_signature);
        }
        ToolResultOutput::ExecutionDenied { reason, .. } => {
            push_function_response_part(
                parts,
                tool_name,
                Value::String(
                    reason
                        .clone()
                        .unwrap_or_else(|| "Tool call execution denied.".to_string()),
                ),
                thought_signature,
            );
        }
        ToolResultOutput::Content { value, .. } => {
            push_tool_result_content_parts(parts, tool_name, value, thought_signature);
        }
    }
}

fn push_response_part(parts: &mut Vec<Part>, part: &UnifiedContentPart) -> Result<(), LlmError> {
    match part {
        UnifiedContentPart::Text {
            text,
            provider_metadata,
            ..
        } => {
            if !text.is_empty() {
                parts.push(Part::Text {
                    text: text.clone(),
                    thought: None,
                    thought_signature: response_thought_signature(provider_metadata.as_ref()),
                });
            }
        }
        UnifiedContentPart::Reasoning {
            text,
            provider_metadata,
            ..
        } => {
            if !text.trim().is_empty() {
                parts.push(Part::Text {
                    text: text.clone(),
                    thought: Some(true),
                    thought_signature: response_thought_signature(provider_metadata.as_ref()),
                });
            }
        }
        UnifiedContentPart::Image {
            source,
            media_type,
            provider_metadata,
            ..
        } => {
            let Some(source) = source.as_media_source() else {
                return Err(LlmError::InvalidParameter(
                    "Gemini response parts do not support provider-managed file references"
                        .to_string(),
                ));
            };
            push_media_response_part(
                parts,
                source,
                media_type.as_deref().or(Some("image/jpeg")),
                None,
                response_thought_signature(provider_metadata.as_ref()),
            );
        }
        UnifiedContentPart::File {
            source,
            media_type,
            provider_metadata,
            ..
        } => {
            let Some(source) = source.as_media_source() else {
                return Err(LlmError::InvalidParameter(
                    "Gemini response parts do not support provider-managed file references"
                        .to_string(),
                ));
            };
            push_media_response_part(
                parts,
                source,
                Some(media_type.as_str()),
                None,
                response_thought_signature(provider_metadata.as_ref()),
            );
        }
        UnifiedContentPart::Audio {
            source,
            media_type,
            provider_metadata,
            ..
        } => {
            push_media_response_part(
                parts,
                source,
                media_type.as_deref().or(Some("audio/wav")),
                None,
                response_thought_signature(provider_metadata.as_ref()),
            );
        }
        UnifiedContentPart::ReasoningFile {
            source,
            media_type,
            provider_metadata,
            ..
        } => {
            push_media_response_part(
                parts,
                source,
                Some(media_type.as_str()),
                Some(true),
                response_thought_signature(provider_metadata.as_ref()),
            );
        }
        UnifiedContentPart::ToolCall {
            tool_name,
            arguments,
            provider_executed,
            provider_metadata,
            ..
        } => {
            let thought_signature = response_thought_signature(provider_metadata.as_ref());
            if *provider_executed == Some(true) && tool_name == "code_execution" {
                let language = match arguments.get("language").and_then(Value::as_str) {
                    Some("PYTHON") => CodeLanguage::Python,
                    _ => CodeLanguage::Unspecified,
                };
                let code = arguments
                    .get("code")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                parts.push(Part::ExecutableCode {
                    executable_code: ExecutableCode { language, code },
                    thought_signature,
                });
            } else {
                parts.push(Part::FunctionCall {
                    function_call: FunctionCall {
                        name: tool_name.clone(),
                        args: Some(arguments.clone()),
                    },
                    thought_signature,
                });
            }
        }
        UnifiedContentPart::ToolResult {
            tool_name,
            output,
            provider_metadata,
            ..
        } => {
            push_response_tool_result_part(
                parts,
                tool_name,
                output,
                response_thought_signature(provider_metadata.as_ref()),
            );
        }
        UnifiedContentPart::Source { .. }
        | UnifiedContentPart::Custom { .. }
        | UnifiedContentPart::ToolApprovalRequest { .. }
        | UnifiedContentPart::ToolApprovalResponse { .. } => {}
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub struct GeminiGenerateContentJsonResponseConverter;

impl GeminiGenerateContentJsonResponseConverter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GeminiGenerateContentJsonResponseConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonResponseConverter for GeminiGenerateContentJsonResponseConverter {
    fn serialize_response(
        &self,
        response: &ChatResponse,
        out: &mut Vec<u8>,
        opts: JsonEncodeOptions,
    ) -> Result<(), LlmError> {
        let safety_ratings: Vec<SafetyRating> =
            google_response_metadata_value(response, "safetyRatings").unwrap_or_default();

        let body = GenerateContentResponse {
            candidates: vec![Candidate {
                content: Some(response_content(response)?),
                finish_reason: gemini_finish_reason(response.finish_reason.as_ref()),
                safety_ratings,
                citation_metadata: None,
                token_count: None,
                grounding_metadata: merged_grounding_metadata(response),
                url_context_metadata: google_response_metadata_value::<UrlContextMetadata>(
                    response,
                    "urlContextMetadata",
                ),
                index: None,
                avg_logprobs: google_response_metadata(response)
                    .and_then(|meta| meta.get("avgLogprobs"))
                    .and_then(Value::as_f64),
                logprobs_result: google_response_metadata_value::<LogprobsResult>(
                    response,
                    "logprobsResult",
                ),
            }],
            prompt_feedback: google_response_metadata_value::<PromptFeedback>(
                response,
                "promptFeedback",
            ),
            usage_metadata: response.usage.as_ref().map(usage_json),
            model_version: response.model.clone(),
            response_id: response.id.clone(),
            service_tier: response.service_tier.clone().or_else(|| {
                google_response_metadata(response)
                    .and_then(|meta| meta.get("serviceTier"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            }),
        };
        let mut body = serde_json::to_value(body).map_err(|error| {
            LlmError::JsonError(format!(
                "Failed to serialize Gemini GenerateContent response shape: {error}"
            ))
        })?;

        if let Some(raw_finish_reason) = raw_finish_reason(response)
            && let Some(candidate) = body
                .get_mut("candidates")
                .and_then(Value::as_array_mut)
                .and_then(|candidates| candidates.first_mut())
                .and_then(Value::as_object_mut)
        {
            candidate.insert(
                "finishReason".to_string(),
                Value::String(raw_finish_reason.to_string()),
            );
        }

        if let Some(finish_message) = google_response_metadata(response)
            .and_then(|meta| meta.get("finishMessage"))
            .and_then(Value::as_str)
            && let Some(object) = body.as_object_mut()
        {
            object.insert(
                "finishMessage".to_string(),
                Value::String(finish_message.to_string()),
            );
        }

        if opts.pretty {
            serde_json::to_writer_pretty(out, &body).map_err(|error| {
                LlmError::JsonError(format!(
                    "Failed to serialize Gemini GenerateContent JSON response: {error}"
                ))
            })?;
        } else {
            serde_json::to_writer(out, &body).map_err(|error| {
                LlmError::JsonError(format!(
                    "Failed to serialize Gemini GenerateContent JSON response: {error}"
                ))
            })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_json_response_encoder_source_does_not_read_request_provider_options() {
        let source = include_str!("json_response.rs");
        let production_source = source
            .split("#[cfg(test)]")
            .next()
            .expect("production source");

        for forbidden in ["provider_options", ".provider_options", "providerOptions"] {
            assert!(
                !production_source.contains(forbidden),
                "Gemini JSON response encoder must not read request-side provider options fragment `{forbidden}`"
            );
        }
    }

    #[test]
    fn gemini_encoder_serializes_normalized_usage_and_preserves_raw_metadata() {
        let mut response = ChatResponse::new(MessageContent::Text("hello".to_string()));
        response.model = Some("gemini-2.5-pro".to_string());
        response.usage = Some(
            Usage::builder()
                .prompt_tokens(12)
                .completion_tokens(9)
                .total_tokens(21)
                .with_input_total_tokens(12)
                .with_input_no_cache_tokens(7)
                .with_input_cache_read_tokens(5)
                .with_output_total_tokens(9)
                .with_output_text_tokens(6)
                .with_output_reasoning_tokens(3)
                .with_raw_usage_value(serde_json::json!({
                    "trafficType": "ON_DEMAND",
                    "promptTokensDetails": [
                        { "modality": "TEXT", "tokenCount": 10 },
                        { "modality": "IMAGE", "tokenCount": 2 }
                    ],
                    "candidatesTokensDetails": [
                        { "modality": "TEXT", "tokenCount": 6 }
                    ]
                }))
                .build(),
        );

        let mut out = Vec::new();
        GeminiGenerateContentJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["usageMetadata"]["promptTokenCount"],
            serde_json::json!(12)
        );
        assert_eq!(
            value["usageMetadata"]["cachedContentTokenCount"],
            serde_json::json!(5)
        );
        assert_eq!(
            value["usageMetadata"]["candidatesTokenCount"],
            serde_json::json!(6)
        );
        assert_eq!(
            value["usageMetadata"]["thoughtsTokenCount"],
            serde_json::json!(3)
        );
        assert_eq!(
            value["usageMetadata"]["trafficType"],
            serde_json::json!("ON_DEMAND")
        );
        assert_eq!(
            value["usageMetadata"]["promptTokensDetails"],
            serde_json::json!([
                { "modality": "TEXT", "tokenCount": 10 },
                { "modality": "IMAGE", "tokenCount": 2 }
            ])
        );
        assert_eq!(
            value["usageMetadata"]["candidatesTokensDetails"],
            serde_json::json!([{ "modality": "TEXT", "tokenCount": 6 }])
        );
    }

    #[test]
    fn gemini_encoder_omits_unknown_usage_totals_in_usage_metadata() {
        let mut response = ChatResponse::new(MessageContent::Text("hello".to_string()));
        response.model = Some("gemini-2.5-pro".to_string());
        response.usage = Some(
            Usage::builder()
                .with_raw_usage_value(serde_json::json!({
                    "trafficType": "ON_DEMAND"
                }))
                .build(),
        );

        let mut out = Vec::new();
        GeminiGenerateContentJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        let usage_metadata = value["usageMetadata"]
            .as_object()
            .expect("usageMetadata object");

        assert!(!usage_metadata.contains_key("promptTokenCount"));
        assert!(!usage_metadata.contains_key("totalTokenCount"));
        assert!(!usage_metadata.contains_key("candidatesTokenCount"));
        assert!(!usage_metadata.contains_key("thoughtsTokenCount"));
        assert_eq!(
            value["usageMetadata"]["trafficType"],
            serde_json::json!("ON_DEMAND")
        );
    }

    #[test]
    fn gemini_encoder_prefers_raw_finish_reason() {
        let mut response = ChatResponse::new(MessageContent::Text("blocked".to_string()));
        response.finish_reason = Some(FinishReason::ContentFilter);
        response.raw_finish_reason = Some("PROHIBITED_CONTENT".to_string());

        let mut out = Vec::new();
        GeminiGenerateContentJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["candidates"][0]["finishReason"],
            serde_json::json!("PROHIBITED_CONTENT")
        );
    }

    #[test]
    fn gemini_encoder_preserves_finish_message_and_service_tier_metadata() {
        let mut response = ChatResponse::new(MessageContent::Text("done".to_string()));
        response.provider_metadata = Some(
            crate::types::provider_metadata::provider_metadata_from_object(
                "google",
                serde_json::Map::from_iter([
                    (
                        "finishMessage".to_string(),
                        serde_json::json!("natural stop"),
                    ),
                    ("serviceTier".to_string(), serde_json::json!("flex")),
                ]),
            ),
        );

        let mut out = Vec::new();
        GeminiGenerateContentJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(value["finishMessage"], serde_json::json!("natural stop"));
        assert_eq!(value["serviceTier"], serde_json::json!("flex"));
    }

    #[test]
    fn gemini_encoder_serializes_document_sources_into_grounding_chunks() {
        let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
            UnifiedContentPart::text("See attached design document."),
            UnifiedContentPart::source_document(
                "doc_1",
                "application/pdf",
                "Design Doc",
                Some("design.pdf".to_string()),
            ),
        ]));
        response.model = Some("gemini-2.5-pro".to_string());

        let mut out = Vec::new();
        GeminiGenerateContentJsonResponseConverter::new()
            .serialize_response(&response, &mut out, JsonEncodeOptions::default())
            .expect("serialize");

        let value: serde_json::Value = serde_json::from_slice(&out).expect("json");
        assert_eq!(
            value["candidates"][0]["groundingMetadata"]["groundingChunks"][0]["retrievedContext"]["uri"],
            serde_json::json!("design.pdf")
        );
        assert_eq!(
            value["candidates"][0]["groundingMetadata"]["groundingChunks"][0]["retrievedContext"]["title"],
            serde_json::json!("Design Doc")
        );
    }
}
