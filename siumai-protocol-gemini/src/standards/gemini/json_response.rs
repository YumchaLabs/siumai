//!
//! English-only comments in code as requested.

use crate::encoding::{JsonEncodeOptions, JsonResponseConverter};
use crate::error::LlmError;
use crate::types::{
    ChatResponse, ContentPart as UnifiedContentPart, FinishReason, MessageContent, SourcePart,
    Usage,
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use super::convert::convert_message_to_content;
use super::types::{
    Candidate, Content, FinishReason as GeminiFinishReason, GenerateContentResponse,
    GroundingMetadata, LogprobsResult, PromptFeedback, SafetyRating, UrlContextMetadata,
    UsageMetadata,
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
        .unwrap_or_else(|| UsageMetadata {
            prompt_token_count: None,
            total_token_count: None,
            cached_content_token_count: None,
            candidates_token_count: None,
            thoughts_token_count: None,
            traffic_type: None,
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
    let assistant_message = response.to_assistant_message();

    match &assistant_message.content {
        MessageContent::Text(text) if text.trim().is_empty() => Ok(Content {
            role: Some("model".to_string()),
            parts: Vec::new(),
        }),
        MessageContent::MultiModal(parts) if parts.is_empty() => Ok(Content {
            role: Some("model".to_string()),
            parts: Vec::new(),
        }),
        _ => convert_message_to_content(&assistant_message, None),
    }
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
        };

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
                    "trafficType": "ON_DEMAND"
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
