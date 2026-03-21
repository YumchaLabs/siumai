//! Gemini GenerateContent non-streaming JSON response encoder (gateway/proxy).
//!
//! English-only comments in code as requested.

use crate::encoding::{JsonEncodeOptions, JsonResponseConverter};
use crate::error::LlmError;
use crate::types::{ChatResponse, ContentPart, FinishReason, Usage};
use serde::Serialize;
use serde_json::{Value, json};

fn gemini_finish_reason(reason: Option<&FinishReason>) -> Option<&'static str> {
    match reason? {
        FinishReason::Stop | FinishReason::StopSequence => Some("STOP"),
        FinishReason::Length => Some("MAX_TOKENS"),
        FinishReason::ContentFilter => Some("SAFETY"),
        FinishReason::ToolCalls => Some("STOP"),
        FinishReason::Error => Some("STOP"),
        FinishReason::Unknown => None,
        FinishReason::Other(_) => None,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiGenerateContentResponse {
    pub candidates: Vec<GeminiCandidate>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "promptFeedback")]
    pub prompt_feedback: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "usageMetadata")]
    pub usage_metadata: Option<GeminiUsageMetadata>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "modelVersion")]
    pub model_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseId")]
    pub response_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiCandidate {
    pub content: GeminiContent,
    #[serde(skip_serializing_if = "Option::is_none", rename = "finishReason")]
    pub finish_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "groundingMetadata")]
    pub grounding_metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "urlContextMetadata")]
    pub url_context_metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "safetyRatings")]
    pub safety_ratings: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "avgLogprobs")]
    pub avg_logprobs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "logprobsResult")]
    pub logprobs_result: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiContent {
    pub role: &'static str,
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum GeminiPart {
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiFunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: u32,
    #[serde(rename = "totalTokenCount")]
    pub total_token_count: u32,
    #[serde(
        rename = "cachedContentTokenCount",
        skip_serializing_if = "Option::is_none"
    )]
    pub cached_content_token_count: Option<u32>,
    #[serde(rename = "thoughtsTokenCount", skip_serializing_if = "Option::is_none")]
    pub thoughts_token_count: Option<u32>,
}

fn usage_json(u: &Usage) -> GeminiUsageMetadata {
    let cached = u
        .prompt_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens)
        .or({
            #[allow(deprecated)]
            {
                u.cached_tokens
            }
        });
    let thoughts = u
        .completion_tokens_details
        .as_ref()
        .and_then(|d| d.reasoning_tokens)
        .or({
            #[allow(deprecated)]
            {
                u.reasoning_tokens
            }
        });

    GeminiUsageMetadata {
        prompt_token_count: u.prompt_tokens,
        candidates_token_count: u.completion_tokens,
        total_token_count: u.total_tokens,
        cached_content_token_count: cached,
        thoughts_token_count: thoughts,
    }
}

fn google_response_metadata(
    response: &ChatResponse,
) -> Option<&std::collections::HashMap<String, Value>> {
    response.provider_metadata.as_ref()?.get("google")
}

fn google_response_metadata_value(response: &ChatResponse, key: &str) -> Option<Value> {
    google_response_metadata(response)?.get(key).cloned()
}

fn source_grounding_chunk(part: &ContentPart) -> Option<Value> {
    let ContentPart::Source {
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
        let text = response.content_text().unwrap_or_default().to_string();
        let source_chunks = response_source_grounding_chunks(response);

        let mut parts = Vec::new();
        if !text.trim().is_empty() {
            parts.push(GeminiPart::Text { text });
        }

        for p in response.tool_calls() {
            if let ContentPart::ToolCall {
                tool_name,
                arguments,
                ..
            } = p
            {
                parts.push(GeminiPart::FunctionCall {
                    function_call: GeminiFunctionCall {
                        name: tool_name.clone(),
                        args: arguments.clone(),
                    },
                });
            }
        }

        let body = GeminiGenerateContentResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiContent {
                    role: "model",
                    parts,
                },
                finish_reason: gemini_finish_reason(response.finish_reason.as_ref()),
                grounding_metadata: merge_grounding_metadata(
                    google_response_metadata_value(response, "groundingMetadata"),
                    source_chunks,
                ),
                url_context_metadata: google_response_metadata_value(
                    response,
                    "urlContextMetadata",
                ),
                safety_ratings: google_response_metadata_value(response, "safetyRatings"),
                avg_logprobs: google_response_metadata(response)
                    .and_then(|meta| meta.get("avgLogprobs"))
                    .and_then(Value::as_f64),
                logprobs_result: google_response_metadata_value(response, "logprobsResult"),
            }],
            prompt_feedback: google_response_metadata_value(response, "promptFeedback"),
            usage_metadata: response.usage.as_ref().map(usage_json),
            model_version: response.model.clone(),
            response_id: response.id.clone(),
        };

        if opts.pretty {
            serde_json::to_writer_pretty(out, &body).map_err(|e| {
                LlmError::JsonError(format!(
                    "Failed to serialize Gemini GenerateContent JSON response: {e}"
                ))
            })?;
        } else {
            serde_json::to_writer(out, &body).map_err(|e| {
                LlmError::JsonError(format!(
                    "Failed to serialize Gemini GenerateContent JSON response: {e}"
                ))
            })?;
        }
        Ok(())
    }
}
