//! Gemini GenerateContent non-streaming JSON response encoder (gateway/proxy).
//!
//! English-only comments in code as requested.

use crate::encoding::{JsonEncodeOptions, JsonResponseConverter};
use crate::error::LlmError;
use crate::types::{ChatResponse, ContentPart, FinishReason, Usage};
use serde::Serialize;

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
    #[serde(skip_serializing_if = "Option::is_none", rename = "usageMetadata")]
    pub usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiCandidate {
    pub content: GeminiContent,
    #[serde(skip_serializing_if = "Option::is_none", rename = "finishReason")]
    pub finish_reason: Option<&'static str>,
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
    #[serde(rename = "thoughtsTokenCount", skip_serializing_if = "Option::is_none")]
    pub thoughts_token_count: Option<u32>,
}

fn usage_json(u: &Usage) -> GeminiUsageMetadata {
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
        thoughts_token_count: thoughts,
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
            }],
            usage_metadata: response.usage.as_ref().map(usage_json),
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
