//!
//! English-only comments in code as requested.

use crate::encoding::{JsonEncodeOptions, JsonResponseConverter};
use crate::error::LlmError;
use crate::types::{
    ChatResponse, ContentPart as UnifiedContentPart, FinishReason, MessageContent, Usage,
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
    let cached = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens)
        .or({
            #[allow(deprecated)]
            {
                usage.cached_tokens
            }
        });
    let thoughts = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|details| details.reasoning_tokens)
        .or({
            #[allow(deprecated)]
            {
                usage.reasoning_tokens
            }
        });

    UsageMetadata {
        prompt_token_count: Some(usage.prompt_tokens as i32),
        total_token_count: Some(usage.total_tokens as i32),
        cached_content_token_count: cached.map(|value| value as i32),
        candidates_token_count: Some(usage.completion_tokens as i32),
        thoughts_token_count: thoughts.map(|value| value as i32),
    }
}

fn google_response_metadata(
    response: &ChatResponse,
) -> Option<&std::collections::HashMap<String, Value>> {
    let provider_metadata = response.provider_metadata.as_ref()?;
    provider_metadata
        .get("google")
        .or_else(|| provider_metadata.get("vertex"))
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
    let UnifiedContentPart::Source {
        source_type,
        url,
        title,
        ..
    } = part
    else {
        return None;
    };

    match source_type.as_str() {
        "url" => Some(json!({
            "web": {
                "uri": url,
                "title": title,
            }
        })),
        _ => Some(json!({
            "retrievedContext": {
                "uri": url,
                "title": title,
            }
        })),
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
        _ => convert_message_to_content(&assistant_message),
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
